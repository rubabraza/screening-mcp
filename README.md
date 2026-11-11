# screening-mcp

A practice MCP server for a background-screening product. It wraps a fake screening API (invented candidates and checks) and shows the parts that matter in a real connector:

- a small set of task-shaped tools, with read-only tools marked read-only
- a confirmation step before anything that changes data
- strict separation between client accounts (tenants)
- an audit log of every call
- contract tests, access tests, and a tool-use evaluation that checks the model picks the right tool

All data is made up. Nothing here connects to a real system.

## Layout

```
src/screening_mcp/fake_api.py   the fake product API (two client accounts: acme, globex)
src/screening_mcp/server.py     the MCP server: four tools, built per account
tests/test_tools.py             contract tests: shapes, errors, confirmation, audit
tests/test_access.py            access tests: one account can never reach another's data
tests/test_eval_scoring.py      tests for the eval scorer itself
evals/cases.example.jsonl       example tool-use evaluation cases
evals/run_eval.py               runs the cases against a model and scores its tool calls
```

## Set up (Windows, PowerShell)

Needs [uv](https://docs.astral.sh/uv/). uv downloads a matching Python itself if you don't have one, so you don't need to install Python separately.

```powershell
cd "D:\Projects Data\AI Engineer\projects\screening-mcp"
uv sync --extra dev
uv run pytest
```

`uv sync` creates `.venv`, installs the project with its dev extras, and writes `uv.lock`. You should see all tests pass.

`uv run <command>` runs a command inside `.venv` without activating it. If you'd rather activate it once per terminal, run `.venv\Scripts\Activate.ps1`. After that, `pytest`, `python` and `screening-mcp` work without the `uv run` prefix.

## Try it by hand: MCP Inspector

Needs Node.js (for `npx`).

```powershell
npx @modelcontextprotocol/inspector .venv\Scripts\screening-mcp.exe
```

The Inspector opens in your browser. Connect, open **Tools**, and try each one: find "Ali", get the status of AC-1001, list outstanding checks, then submit a check with and without `confirmed`.

Switch account by setting `SCREENING_TENANT=globex` in the Inspector's environment settings, then try reading AC-1001 again. It should fail.

## Use it from Claude Code

```powershell
claude mcp add screening -e SCREENING_TENANT=acme -- "D:\Projects Data\AI Engineer\projects\screening-mcp\.venv\Scripts\screening-mcp.exe"
```

Then ask Claude Code things like "who are we still waiting on?" and watch which tools it calls.

## Run over HTTP (how a hosted connector would run)

```powershell
uv run screening-mcp --http
```

Serves on `http://127.0.0.1:8000/mcp`. In the Inspector, choose the Streamable HTTP transport and that URL.

## Run the tool-use evaluation

```powershell
$env:ANTHROPIC_API_KEY = "..."
uv run python evals\run_eval.py evals\cases.example.jsonl
```

Change a tool description or an error message, run it again, and compare. That's the regression gate.

Most cases score only the first tool call. A case with a `then` step also runs that call on the server, sends the result back to the model, and scores what it does next. For example, given a reference that doesn't exist, it should ask the user to check it, not guess another one.

## Design decisions

- **Why these four tools and not more.** Each tool matches something a client's staff actually asks: find a person, see where their checks are, see what's still pending, start a new check. Wrapping every API endpoint as its own tool would give the model more to choose between and more ways to choose wrong. The model picks tools from their descriptions alone, so fewer, clearer tools mean better choices.
- **What the confirmation step protects against.** Starting a check costs money and affects a real person, and it can't be undone. `submit_check` returns a summary on the first call and only submits when called again with `confirmed=true`. So a misread request, the wrong one of two people with the same name, or the model acting on its own can't start a check before a person has said yes to that exact summary.
- **Why a retry never pays twice.** The first call also returns a `confirmation_id`, and the confirming call must send it back. Each id can start one check at most. If the client times out and retries with the same id, it gets the first result back (same `check_id`) instead of starting and paying for a second check. An id also only works for the exact candidate and check type it was given for, and only in the account that asked for it. Asking twice on purpose gives two ids, so two checks are still possible when someone really wants them. Ids expire: an unconfirmed one after 15 minutes (a yes should follow the summary closely, otherwise the user sees a fresh one), and a used one after 24 hours (long enough for any realistic retry). An expired id is refused, so even a very late retry can't pay twice.
- **How the account is chosen.** Here, one server instance serves one account, set by `SCREENING_TENANT` or by `build_server()` in tests. Every lookup is scoped to that account, and another account's reference gets the same error as one that doesn't exist, so the error itself reveals nothing. With real sign-in, the account would come from the signed-in user's OAuth token on each request, never from anything the model or user types.
- **What I'd add before real clients used it.** OAuth sign-in with per-user permissions (who may start checks, not just read them), an audit log in durable storage instead of memory, rate limits and spending limits on `submit_check`, confirmation ids stored in a database (here they're in memory and lost on restart), a warning when the same check is already in progress for that candidate, and a larger evaluation set run on every change.
