"""Contract tests: each tool returns the right shape and clear errors."""

import asyncio

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from screening_mcp.server import build_server


@pytest.fixture
def acme():
    return build_server("acme")


async def test_lists_the_expected_tools(acme):
    async with Client(acme) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == {"find_candidate", "get_check_status", "list_outstanding_checks", "submit_check"}


async def test_read_tools_are_marked_read_only(acme):
    async with Client(acme) as c:
        tools = {t.name: t for t in await c.list_tools()}
    for name in ("find_candidate", "get_check_status", "list_outstanding_checks"):
        assert tools[name].annotations.read_only_hint is True
    assert tools["submit_check"].annotations.read_only_hint is False


async def test_find_candidate_by_partial_name(acme):
    async with Client(acme) as c:
        r = await c.call_tool("find_candidate", {"name": "sara"})
    assert r.data == [{"ref": "AC-1002", "name": "Sara Ahmed", "role": "Nurse"}]


async def test_get_check_status_shape(acme):
    async with Client(acme) as c:
        r = await c.call_tool("get_check_status", {"candidate_ref": "ac-1001"})
    assert r.data["ref"] == "AC-1001"
    assert {k["type"] for k in r.data["checks"]} == {"identity", "right_to_work", "criminal_record"}


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("get_check_status", {"candidate_ref": "AC-9999"}),
        ("submit_check", {"candidate_ref": "AC-9999", "check_type": "credit", "confirmed": False}),
        ("submit_check", {"candidate_ref": "AC-9999", "check_type": "credit", "confirmed": True}),
    ],
)
async def test_unknown_reference_tells_the_model_to_check_with_the_user(acme, tool, args):
    async with Client(acme) as c:
        with pytest.raises(ToolError) as e:
            await c.call_tool(tool, args)
    msg = str(e.value)
    assert "Ask the user to check the reference" in msg
    assert "find_candidate" in msg


async def test_outstanding_excludes_finished_checks(acme):
    async with Client(acme) as c:
        r = await c.call_tool("list_outstanding_checks", {})
    assert {row["status"] for row in r.data} <= {"not_started", "in_progress", "awaiting_candidate"}
    assert len(r.data) == 2


async def test_submit_needs_confirmation_first(acme):
    async with Client(acme) as c:
        r = await c.call_tool("submit_check", {"candidate_ref": "AC-1002", "check_type": "credit"})
        assert r.data["status"] == "needs_confirmation"
        cid = r.data["confirmation_id"]
        assert await _credit_checks(c, "AC-1002") == 0  # nothing created yet

        r = await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        assert r.data["status"] == "submitted"
        assert await _credit_checks(c, "AC-1002") == 1


# --- idempotency: one confirmation_id starts at most one paid check ---


def _confirm(ref, check_type, cid):
    return {"candidate_ref": ref, "check_type": check_type, "confirmed": True, "confirmation_id": cid}


async def _ask(c, ref="AC-1002", check_type="credit"):
    r = await c.call_tool("submit_check", {"candidate_ref": ref, "check_type": check_type})
    return r.data["confirmation_id"]


async def _credit_checks(c, ref):
    status = await c.call_tool("get_check_status", {"candidate_ref": ref})
    return sum(k["type"] == "credit" for k in status.data["checks"])


async def test_retry_after_timeout_returns_first_result_and_does_not_pay_twice(acme):
    async with Client(acme) as c:
        cid = await _ask(c)
        first = await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        retry = await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))  # client timed out, tries again
        assert retry.data == first.data  # same check_id
        assert await _credit_checks(c, "AC-1002") == 1
    assert [e["outcome"] for e in acme.audit_log if e["tool"] == "submit_check"] == [
        "awaiting_confirmation", "submitted", "replayed",
    ]


async def test_parallel_retries_start_only_one_check(acme):
    async with Client(acme) as c:
        cid = await _ask(c)
        results = await asyncio.gather(*[c.call_tool("submit_check", _confirm("AC-1002", "credit", cid)) for _ in range(5)])
        assert len({r.data["check_id"] for r in results}) == 1
        assert await _credit_checks(c, "AC-1002") == 1


async def test_two_separate_confirmations_are_two_checks(acme):
    # Idempotency is per confirmation, not per candidate: asking twice on purpose is allowed.
    async with Client(acme) as c:
        for _ in range(2):
            await c.call_tool("submit_check", _confirm("AC-1002", "credit", await _ask(c)))
        assert await _credit_checks(c, "AC-1002") == 2


@pytest.mark.parametrize("cid", [None, "", "made-up-id"])
async def test_confirming_without_a_real_confirmation_id_submits_nothing(acme, cid):
    async with Client(acme) as c:
        with pytest.raises(ToolError, match="Unknown or missing confirmation_id"):
            await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        assert await _credit_checks(c, "AC-1002") == 0
    assert _outcomes(acme)[-1] == "no_confirmation"


def _outcomes(server):
    return [e["outcome"] for e in server.audit_log if e["tool"] == "submit_check"]


# --- confirmation ids expire: 15 minutes to confirm, 24 hours to retry ---


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, minutes=0, hours=0):
        self.now += minutes * 60 + hours * 3600


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def acme_clocked(clock):
    return build_server("acme", clock=clock)


async def test_confirming_within_15_minutes_works(acme_clocked, clock):
    async with Client(acme_clocked) as c:
        cid = await _ask(c)
        clock.advance(minutes=14)
        r = await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        assert r.data["status"] == "submitted"


async def test_confirming_after_15_minutes_is_refused(acme_clocked, clock):
    async with Client(acme_clocked) as c:
        cid = await _ask(c)
        clock.advance(minutes=16)
        with pytest.raises(ToolError, match="This confirmation has expired"):
            await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        assert _outcomes(acme_clocked)[-1] == "confirmation_expired"  # not mixed up with a made-up id
        assert await _credit_checks(c, "AC-1002") == 0
        # asking again gives a fresh id that works
        r = await c.call_tool("submit_check", _confirm("AC-1002", "credit", await _ask(c)))
        assert r.data["status"] == "submitted"


async def test_unconfirmed_ids_are_forgotten_after_24_hours(acme_clocked, clock):
    async with Client(acme_clocked) as c:
        cid = await _ask(c)
        clock.advance(hours=25)
        with pytest.raises(ToolError, match="Unknown or missing confirmation_id"):
            await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        assert await _credit_checks(c, "AC-1002") == 0


async def test_retry_within_24_hours_still_returns_first_result(acme_clocked, clock):
    async with Client(acme_clocked) as c:
        cid = await _ask(c)
        first = await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        clock.advance(hours=23)  # well past the 15-minute limit for confirming
        retry = await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        assert retry.data == first.data
        assert await _credit_checks(c, "AC-1002") == 1


async def test_retry_after_24_hours_is_refused_and_never_pays_twice(acme_clocked, clock):
    async with Client(acme_clocked) as c:
        cid = await _ask(c)
        await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        clock.advance(hours=25)
        with pytest.raises(ToolError, match="Unknown or missing confirmation_id"):
            await c.call_tool("submit_check", _confirm("AC-1002", "credit", cid))
        assert await _credit_checks(c, "AC-1002") == 1


@pytest.mark.parametrize(("ref", "check_type"), [("AC-1001", "credit"), ("AC-1002", "education")])
async def test_confirmation_id_only_works_for_the_check_it_was_given_for(acme, ref, check_type):
    async with Client(acme) as c:
        cid = await _ask(c, "AC-1002", "credit")
        with pytest.raises(ToolError, match="was for a credit check on AC-1002"):
            await c.call_tool("submit_check", _confirm(ref, check_type, cid))
        r = await c.call_tool("get_check_status", {"candidate_ref": ref})
        assert len(r.data["checks"]) == {"AC-1001": 3, "AC-1002": 2}[ref]  # unchanged


async def test_invalid_check_type_is_rejected(acme):
    async with Client(acme) as c:
        with pytest.raises(ToolError):
            await c.call_tool("submit_check", {"candidate_ref": "AC-1002", "check_type": "astrology", "confirmed": True})


async def test_every_call_is_audited(acme):
    async with Client(acme) as c:
        await c.call_tool("find_candidate", {"name": "ali"})
        await c.call_tool("list_outstanding_checks", {})
    assert [e["tool"] for e in acme.audit_log] == ["find_candidate", "list_outstanding_checks"]
    assert all(e["tenant"] == "acme" for e in acme.audit_log)
