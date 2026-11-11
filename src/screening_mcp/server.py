"""Practice MCP server for a background-screening product.

One server instance = one client account (tenant). In a real hosted server the
tenant would come from the signed-in user's OAuth token; here it comes from the
SCREENING_TENANT environment variable (see main) or from build_server() in tests.

Design choices worth being able to explain:
- Few tools, each matching a real task, not one tool per API endpoint.
- Read-only tools are marked read-only; the one write tool asks for confirmation.
- Errors are written for the model, so it can recover and ask the user.
- Results carry only what the request needs.
- Every call is written to an audit log.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from .fake_api import FakeScreeningAPI, NotFound

log = logging.getLogger("screening_mcp.audit")

CheckType = Literal["identity", "right_to_work", "criminal_record", "credit", "employment_history", "education"]

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

# Added to every "no such candidate" error. The error goes to the model, so it
# says what to do next: check with the user rather than guess another ref.
CHECK_REF_HINT = "Ask the user to check the reference is correct, or search by name with find_candidate."

# How long confirmation ids live, in seconds.
PENDING_TTL = 15 * 60  # asked but not confirmed: a yes should follow soon after the summary was shown
DONE_TTL = 24 * 60 * 60  # already submitted: long enough for any realistic client retry
FORGET_AFTER = DONE_TTL  # unconfirmed ids are kept this long (but refused after PENDING_TTL), so an
                         # expired id gets "expired" rather than looking like one that never existed


def build_server(
    tenant_id: str,
    api: FakeScreeningAPI | None = None,
    clock: Callable[[], float] = time.monotonic,  # tests pass a fake clock to move time forward
) -> FastMCP:
    api = api or FakeScreeningAPI()
    mcp = FastMCP(
        name="screening",
        instructions=(
            "Tools for one client account of a background-screening service. "
            "Use them to look up candidates and the status of their checks. "
            "Never submit a check without the user's explicit confirmation."
        ),
    )
    audit: list[dict] = []
    mcp.audit_log = audit  # exposed for tests; a real server would write to durable storage

    def record(tool: str, **detail) -> None:
        entry = {"at": datetime.now(timezone.utc).isoformat(), "tenant": tenant_id, "tool": tool, **detail}
        audit.append(entry)
        log.info("audit %s", entry)

    @mcp.tool(annotations=READ_ONLY)
    def find_candidate(name: str) -> list[dict]:
        """Find candidates in this account by full or partial name.
        Use this first when the user mentions a person by name, to get their
        candidate reference (like AC-1001). Returns reference, name and role.
        If several people match, ask the user which one they mean.
        """
        matches = api.find_candidates(tenant_id, name)
        record("find_candidate", query=name, results=len(matches))
        return [{"ref": c.ref, "name": c.name, "role": c.role} for c in matches]

    @mcp.tool(annotations=READ_ONLY)
    def get_check_status(candidate_ref: str) -> dict:
        """Get the current status of every screening check for one candidate.

        Needs a candidate reference (like AC-1001); use find_candidate first if
        you only have a name. Does not return check results in detail.
        """
        try:
            c = api.get_candidate(tenant_id, candidate_ref)
        except NotFound as e:
            record("get_check_status", ref=candidate_ref, outcome="not_found")
            raise ToolError(f"{e} {CHECK_REF_HINT}") from None
        record("get_check_status", ref=c.ref, outcome="ok")
        return {
            "ref": c.ref,
            "name": c.name,
            "checks": [{"type": k.check_type, "status": k.status, "updated": k.updated.isoformat()} for k in c.checks],
        }

    @mcp.tool(annotations=READ_ONLY)
    def list_outstanding_checks() -> list[dict]:
        """List every check in this account that is not finished yet.

        Covers checks not started, in progress, or waiting on the candidate.
        Use for questions like "what's still pending?" or "who are we waiting on?".
        """
        rows = api.list_outstanding(tenant_id)
        record("list_outstanding_checks", results=len(rows))
        return [{"ref": c.ref, "name": c.name, "type": k.check_type, "status": k.status} for c, k in rows]

    # Idempotency for the paid write. Each confirmation_id can start at most one
    # check: a retry after a timeout gets the first result back instead of paying
    # twice. A real server keeps these in a database table with a TTL; here they
    # live in memory, per server, so one account can't use another's ids.
    pending: dict[str, tuple[str, str, float]] = {}  # id -> (ref, check_type, asked_at), not confirmed yet
    done: dict[str, tuple[dict, float]] = {}  # id -> (result of the submit it started, submitted_at)
    submit_lock = threading.Lock()  # two retries arriving at once must not both submit

    def drop_expired(now: float) -> None:
        """Forget old ids so memory doesn't grow forever. Call with submit_lock held."""
        for cid in [k for k, (*_, at) in pending.items() if now - at > FORGET_AFTER]:
            del pending[cid]
        for cid in [k for k, (_, at) in done.items() if now - at > DONE_TTL]:
            del done[cid]

    def replay(cid: str) -> dict:
        record("submit_check", confirmation_id=cid, outcome="replayed")
        return done[cid][0]

    @mcp.tool(annotations=WRITE)
    def submit_check(
        candidate_ref: str,
        check_type: CheckType,
        confirmed: bool = False,
        confirmation_id: str | None = None,
    ) -> dict:
        """Start a new screening check for a candidate. This costs money and affects a real person.

        Always call it first with confirmed=false: you get a summary to show the
        user and a confirmation_id. Only after the user has explicitly said yes
        to that summary, call again with confirmed=true and the same
        candidate_ref, check_type and confirmation_id.

        A confirmation_id expires 15 minutes after it was given: if the user
        takes longer to say yes, ask again with confirmed=false.

        If a confirmed call times out or fails, retry it with the same
        confirmation_id: it never starts a second check, it returns the result
        of the first. Never make up a confirmation_id or reuse one for a
        different check.
        """
        if confirmed:
            with submit_lock:
                drop_expired(clock())
                if confirmation_id in done:  # a retry: return the first result, start nothing
                    return replay(confirmation_id)

        try:
            c = api.get_candidate(tenant_id, candidate_ref)
        except NotFound as e:
            record("submit_check", ref=candidate_ref, check_type=check_type, outcome="not_found")
            raise ToolError(f"{e} Nothing was submitted. {CHECK_REF_HINT}") from None

        if not confirmed:
            cid = uuid4().hex
            with submit_lock:
                now = clock()
                drop_expired(now)
                pending[cid] = (c.ref, check_type, now)
            record("submit_check", ref=c.ref, check_type=check_type, outcome="awaiting_confirmation", confirmation_id=cid)
            return {
                "status": "needs_confirmation",
                "confirmation_id": cid,
                "summary": f"Start a {check_type.replace('_', ' ')} check for {c.name} ({c.ref}).",
                "next_step": (
                    "Show this summary to the user. If they say yes, call submit_check again with "
                    "confirmed=true and this confirmation_id."
                ),
            }

        with submit_lock:
            now = clock()
            drop_expired(now)
            if confirmation_id in done:  # a parallel retry finished while we looked up the candidate
                return replay(confirmation_id)
            # Guard 1: no id, or one this server never gave out (or forgot long ago).
            if confirmation_id not in pending:
                record("submit_check", ref=c.ref, check_type=check_type, outcome="no_confirmation")
                raise ToolError(
                    "Unknown or missing confirmation_id. Nothing was submitted. "
                    "Call submit_check with confirmed=false first and show the user the summary."
                )
            asked_ref, asked_type, asked_at = pending[confirmation_id]
            # Guard 2: approved, but too long ago. The user has to approve again.
            if now - asked_at > PENDING_TTL:
                del pending[confirmation_id]
                record("submit_check", ref=c.ref, check_type=check_type, outcome="confirmation_expired",
                       confirmation_id=confirmation_id)
                raise ToolError(
                    "This confirmation has expired. Nothing was submitted. Ask the user again: call "
                    "submit_check with confirmed=false to show them a fresh summary."
                )
            # Guard 3: the id proves consent to one exact check, not to "something".
            if (asked_ref, asked_type) != (c.ref, check_type):
                record("submit_check", ref=c.ref, check_type=check_type, outcome="confirmation_mismatch")
                raise ToolError(
                    f"This confirmation_id was for a {asked_type.replace('_', ' ')} check on {asked_ref}, "
                    "not this request. Nothing was submitted. Ask again with confirmed=false."
                )
            check = api.submit_check(tenant_id, c.ref, check_type)
            result = {"status": "submitted", "check_id": check.check_id, "ref": c.ref, "type": check_type}
            done[confirmation_id] = (result, now)
            del pending[confirmation_id]

        record("submit_check", ref=c.ref, check_type=check_type, outcome="submitted",
               check_id=check.check_id, confirmation_id=confirmation_id)
        return result

    return mcp


def main() -> None:
    """Run the server.

    stdio (default):  screening-mcp
    HTTP:             screening-mcp --http   (serves on http://127.0.0.1:8000/mcp)
    Choose the account with SCREENING_TENANT=acme or globex.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # never log to stdout with stdio
    tenant = os.environ.get("SCREENING_TENANT", "acme")
    server = build_server(tenant)
    if "--http" in sys.argv:
        server.run(transport="http", host="127.0.0.1", port=8000)
    else:
        server.run()


if __name__ == "__main__":
    main()
