"""Access tests: one client account can never reach another account's data.

This is the test that matters most for a multi-client product.
"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from screening_mcp.fake_api import FakeScreeningAPI
from screening_mcp.server import build_server


@pytest.fixture
def shared_api():
    return FakeScreeningAPI()  # both accounts share one backend, like a real product


async def test_search_only_returns_own_candidates(shared_api):
    # "Ali Khan" exists in both accounts on purpose.
    async with Client(build_server("acme", shared_api)) as c:
        r = await c.call_tool("find_candidate", {"name": "ali khan"})
    assert [row["ref"] for row in r.data] == ["AC-1001"]


async def test_cannot_read_another_accounts_candidate(shared_api):
    async with Client(build_server("acme", shared_api)) as c:
        with pytest.raises(ToolError) as e:
            await c.call_tool("get_check_status", {"candidate_ref": "GX-2002"})
    assert "Maria" not in str(e.value)  # the error must not leak the other account's data


@pytest.mark.parametrize(
    ("tenant", "foreign_ref", "foreign_name", "missing_ref"),
    [
        ("acme", "GX-2002", "Maria", "GX-9999"),
        ("acme", "gx-2002", "Maria", "gx-9999"),  # lowercase: the API uppercases refs
        ("globex", "AC-1003", "James", "AC-9999"),
    ],
)
async def test_foreign_ref_looks_exactly_like_a_missing_one(shared_api, tenant, foreign_ref, foreign_name, missing_ref):
    # Another account's ref must give the same error as a ref that doesn't exist,
    # otherwise the error itself tells you which refs other clients have.
    async with Client(build_server(tenant, shared_api)) as c:
        with pytest.raises(ToolError) as foreign:
            await c.call_tool("get_check_status", {"candidate_ref": foreign_ref})
        with pytest.raises(ToolError) as missing:
            await c.call_tool("get_check_status", {"candidate_ref": missing_ref})
    assert foreign_name not in str(foreign.value)
    assert str(foreign.value).replace(foreign_ref, "REF") == str(missing.value).replace(missing_ref, "REF")


async def test_cannot_submit_for_another_accounts_candidate(shared_api):
    async with Client(build_server("acme", shared_api)) as c:
        with pytest.raises(ToolError):
            await c.call_tool("submit_check", {"candidate_ref": "GX-2001", "check_type": "credit", "confirmed": True})
    gx = shared_api.get_candidate("globex", "GX-2001")
    assert [k.check_type for k in gx.checks] == ["credit"]  # unchanged


async def test_confirmation_id_from_another_account_is_useless(shared_api):
    # globex asks to start a check; acme gets hold of the id and tries to confirm it.
    async with Client(build_server("globex", shared_api)) as gx:
        r = await gx.call_tool("submit_check", {"candidate_ref": "GX-2002", "check_type": "credit"})
        cid = r.data["confirmation_id"]
    async with Client(build_server("acme", shared_api)) as ac:
        with pytest.raises(ToolError):
            await ac.call_tool("submit_check", {"candidate_ref": "AC-1002", "check_type": "credit",
                                                "confirmed": True, "confirmation_id": cid})
    assert "credit" not in {k.check_type for k in shared_api.get_candidate("acme", "AC-1002").checks}
    assert "credit" not in {k.check_type for k in shared_api.get_candidate("globex", "GX-2002").checks}


async def test_outstanding_is_scoped_to_the_account(shared_api):
    async with Client(build_server("globex", shared_api)) as c:
        r = await c.call_tool("list_outstanding_checks", {})
    assert all(row["ref"].startswith("GX-") for row in r.data)
