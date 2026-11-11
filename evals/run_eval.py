"""Tool-use evaluation: does the model pick the right tool with the right inputs?

For each case in a JSONL file, send the request to a model together with the
server's real tool list (names, descriptions, schemas), take the FIRST tool
call it makes, and score it against what you expected.

A case can also have a "then" step. The first tool call is then actually run
on the server (account "acme", fresh data per case), its result or error is
sent back to the model, and the model's NEXT move is scored too:

    "then": {"expected_tool": null, "reply_mentions": ["AC-9999"]}

expected_tool null means "reply to the user, don't call another tool".
reply_mentions lists words the reply must contain (case-insensitive).

    uv sync --extra dev              # installs anthropic with the other dev extras
    $env:ANTHROPIC_API_KEY = "..."   # PowerShell; in bash: export ANTHROPIC_API_KEY=...
    uv run python evals/run_eval.py evals/cases.example.jsonl

Re-run it whenever you change a tool description or an error message: that's
the regression gate.

Limits:
- Scores at most two steps: the first tool call and the move after it.
- Argument matching is a simple case-insensitive subset match.
- "Did it ask the user?" is checked with keywords, not by reading the reply.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastmcp import Client  # noqa: E402

from screening_mcp.server import build_server  # noqa: E402

MODEL = os.environ.get("EVAL_MODEL", "claude-sonnet-5-5")
TENANT = "acme"
SYSTEM = "You help staff at a client company check on background screening. Use the tools when they help."


def _norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def score(case: dict, tool: str | None, args: dict) -> tuple[bool, str]:
    if case["expected_tool"] != tool:
        return False, f"expected tool {case['expected_tool']}, got {tool}"
    for k, v in case.get("expected_args", {}).items():
        if _norm(args.get(k, False if v is False else None)) != _norm(v):
            return False, f"arg {k}: expected {v!r}, got {args.get(k)!r}"
    return True, "ok"


def score_then(then: dict, tool: str | None, reply: str) -> tuple[bool, str]:
    expected = then.get("expected_tool")
    if tool != expected:
        if expected is None:
            return False, f"after the result, expected a reply to the user, but it called {tool}"
        return False, f"after the result, expected tool {expected}, got {tool}"
    for word in then.get("reply_mentions", []):
        if word.lower() not in reply.lower():
            return False, f"reply doesn't mention {word!r}: {reply[:100]!r}"
    return True, "ok (two steps)"


def tools_for_anthropic(mcp_tools) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in mcp_tools]


async def ask(claude, tools: list[dict], messages: list[dict]):
    return await claude.messages.create(model=MODEL, max_tokens=512, system=SYSTEM, tools=tools, messages=messages)


def first_tool_use(msg):
    return next((b for b in msg.content if b.type == "tool_use"), None)


def reply_text(msg) -> str:
    return " ".join(b.text for b in msg.content if b.type == "text")


async def run_case(claude, case: dict) -> tuple[bool, str]:
    async with Client(build_server(TENANT)) as mcp:  # fresh data for every case
        tools = tools_for_anthropic(await mcp.list_tools())
        messages = [{"role": "user", "content": case["request"]}]

        msg = await ask(claude, tools, messages)
        call = first_tool_use(msg)
        ok, why = score(case, call.name if call else None, dict(call.input) if call else {})
        if not ok or "then" not in case or call is None:
            return ok, why

        result = await mcp.call_tool(call.name, dict(call.input), raise_on_error=False)
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        messages += [
            {"role": "assistant", "content": msg.content},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": result.is_error},
            ]},
        ]

        follow = await ask(claude, tools, messages)
        nxt = first_tool_use(follow)
        return score_then(case["then"], nxt.name if nxt else None, reply_text(follow))


async def main(path: str) -> int:
    import anthropic

    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    claude = anthropic.AsyncAnthropic()
    passed = 0
    for case in cases:
        ok, why = await run_case(claude, case)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<24} {why}")
    print(f"\n{passed}/{len(cases)} passed ({passed / len(cases):.0%}) with {MODEL}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "evals/cases.example.jsonl")))
