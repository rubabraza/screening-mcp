"""A fake background-screening API.

Stands in for a real product API (submit checks, read status) so the MCP
server has something to wrap. All data is invented. Nothing here talks to a
real system.

Every method takes a tenant_id: one client company. A tenant can only ever
see its own candidates. That rule lives here AND is tested from the MCP side
in tests/test_access.py.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date

CHECK_TYPES = ("identity", "right_to_work", "criminal_record", "credit", "employment_history", "education")
STATUSES = ("not_started", "in_progress", "awaiting_candidate", "complete_clear", "complete_flagged")
OUTSTANDING = {"not_started", "in_progress", "awaiting_candidate"}


class NotFound(Exception):
    """Raised when a candidate doesn't exist FOR THIS TENANT.

    The message never says whether the candidate exists for another tenant.
    """


@dataclass
class Check:
    check_id: str
    check_type: str
    status: str
    updated: date


@dataclass
class Candidate:
    ref: str
    name: str
    role: str
    checks: list[Check] = field(default_factory=list)


def _seed() -> dict[str, dict[str, Candidate]]:
    d = date(2026, 11, 2)
    return {
        "acme": {
            "AC-1001": Candidate("AC-1001", "Ali Khan", "Care assistant", [
                Check("CHK-1", "identity", "complete_clear", d),
                Check("CHK-2", "right_to_work", "complete_clear", d),
                Check("CHK-3", "criminal_record", "in_progress", d),
            ]),
            "AC-1002": Candidate("AC-1002", "Sara Ahmed", "Nurse", [
                Check("CHK-4", "identity", "complete_clear", d),
                Check("CHK-5", "employment_history", "awaiting_candidate", d),
            ]),
            "AC-1003": Candidate("AC-1003", "James Wright", "Driver", [
                Check("CHK-6", "identity", "complete_flagged", d),
            ]),
        },
        "globex": {
            "GX-2001": Candidate("GX-2001", "Ali Khan", "Analyst", [  # same name, different tenant: on purpose
                Check("CHK-7", "credit", "not_started", d),
            ]),
            "GX-2002": Candidate("GX-2002", "Maria Lopez", "Teacher", [
                Check("CHK-8", "criminal_record", "complete_clear", d),
                Check("CHK-9", "education", "in_progress", d),
            ]),
        },
    }


class FakeScreeningAPI:
    def __init__(self) -> None:
        self._data = _seed()
        self._ids = itertools.count(100)

    def _tenant(self, tenant_id: str) -> dict[str, Candidate]:
        if tenant_id not in self._data:
            raise NotFound("Unknown client account.")
        return self._data[tenant_id]

    def find_candidates(self, tenant_id: str, name: str) -> list[Candidate]:
        q = name.strip().lower()
        return [c for c in self._tenant(tenant_id).values() if q in c.name.lower()]

    def get_candidate(self, tenant_id: str, ref: str) -> Candidate:
        cand = self._tenant(tenant_id).get(ref.strip().upper())
        if cand is None:
            raise NotFound(f"No candidate with reference {ref} in this account.")
        return cand

    def list_outstanding(self, tenant_id: str) -> list[tuple[Candidate, Check]]:
        return [(c, k) for c in self._tenant(tenant_id).values() for k in c.checks if k.status in OUTSTANDING]

    def submit_check(self, tenant_id: str, ref: str, check_type: str) -> Check:
        if check_type not in CHECK_TYPES:
            raise ValueError(f"Unknown check type {check_type}.")
        cand = self.get_candidate(tenant_id, ref)
        check = Check(f"CHK-{next(self._ids)}", check_type, "not_started", date.today())
        cand.checks.append(check)
        return check
