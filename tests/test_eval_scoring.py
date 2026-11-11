"""The eval scorer itself needs tests, or a broken scorer looks like a good model."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

from fastmcp import Client

from screening_mcp.server import build_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from run_eval import run_case, score, score_then, tools_for_anthropic  # noqa: E402

CASE = {"expected_tool": "submit_check", "expected_args": {"candidate_ref": "AC-1002", "confirmed": False}}


def test_right_tool_and_args_pass():
    assert score(CASE, "submit_check", {"candidate_ref": "ac-1002", "check_type": "credit"})[0]


def test_wrong_tool_fails():
    assert not score(CASE, "get_check_status", {"candidate_ref": "AC-1002"})[0]


def test_confirming_too_early_fails():
    assert not score(CASE, "submit_check", {"candidate_ref": "AC-1002", "confirmed": True})[0]


def test_no_tool_expected():
    assert score({"expected_tool": None, "expected_args": {}}, None, {})[0]


async def test_tool_list_converts_for_the_model():
    async with Client(build_server("acme")) as c:
        tools = tools_for_anthropic(await c.list_tools())
    assert {t["name"] for t in tools} >= {"find_candidate", "submit_check"}
    assert all(t["input_schema"]["type"] == "object" and t["description"] for t in tools)


# --- the "then" step: what the model does after seeing the tool result ---

ASK_USER = {"expected_tool": None, "reply_mentions": ["AC-9999"]}


def test_then_passes_when_model_asks_the_user():
    assert score_then(ASK_USER, None, "I couldn't find ac-9999. Could you check the reference?")[0]


def test_then_fails_when_model_guesses_another_tool_call():
    assert not score_then(ASK_USER, "get_check_status", "")[0]


def test_then_fails_when_reply_misses_a_keyword():
    assert not score_then(ASK_USER, None, "Something went wrong.")[0]


class FakeClaude:
    """Stands in for the Anthropic client: returns canned replies and keeps what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []
        self.messages = self

    async def create(self, **kw):
        self.sent.append(kw["messages"])
        return self.replies.pop(0)


def _text(s):
    return NS(type="text", text=s)


def _call(name, tool_id="tu1", **args):
    return NS(type="tool_use", name=name, input=args, id=tool_id)


TWO_STEP = {
    "request": "What's the status of the checks for AC-9999?",
    "expected_tool": "get_check_status",
    "expected_args": {"candidate_ref": "AC-9999"},
    "then": ASK_USER,
}


async def test_two_step_case_sends_the_real_server_error_back():
    claude = FakeClaude(
        NS(content=[_call("get_check_status", candidate_ref="AC-9999")]),
        NS(content=[_text("I couldn't find AC-9999. Could you check it?")]),
    )
    assert await run_case(claude, TWO_STEP) == (True, "ok (two steps)")
    tool_result = claude.sent[1][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "Ask the user to check the reference" in tool_result["content"]


async def test_two_step_case_fails_when_model_guesses():
    claude = FakeClaude(
        NS(content=[_call("get_check_status", candidate_ref="AC-9999")]),
        NS(content=[_call("get_check_status", tool_id="tu2", candidate_ref="AC-1001")]),
    )
    ok, _ = await run_case(claude, TWO_STEP)
    assert not ok
