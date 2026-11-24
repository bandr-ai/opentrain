"""The budget ledger. Set in the Trainfile, enforced here; no run can raise it.

Spend is reserved before an action (a sampling batch, a training job) and settled after it with
the real figure. A reservation that would take the period's total over the budget is refused
before anything is launched. The ledger is an append-only file, so it survives restarts and a
crashed run's open reservations still count until they are settled or released.

Model tokens under a subscription have no dollar figure. They are recorded as turns and tokens
with ``usd=0`` and the vendor's own limits apply (plan: "How the limits are enforced").
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .trainfile import Budget


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, state: Path, budget: Budget):
        self.path = state / "runs" / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget = budget

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def _append(self, entry: dict) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _window_start(self, now: datetime) -> datetime | None:
        if self.budget.period_days is None:
            return None
        return now - timedelta(days=self.budget.period_days)

    def committed(self, run_id: str | None = None, now: datetime | None = None) -> float:
        """Settled spend plus open reservations, inside the current window (or the run, for a per-run budget)."""
        now = now or datetime.now(timezone.utc)
        start = self._window_start(now)
        state: dict[str, dict] = {}
        for e in self._entries():
            at = datetime.fromisoformat(e["at"])
            if e["op"] == "reserve":
                if start is not None and at < start:
                    continue
                # A per-run budget counts only this run's reservations; settlements follow them by id.
                if self.budget.period_days is None and run_id is not None and e.get("run_id") != run_id:
                    continue
            if e["op"] == "reserve":
                state[e["id"]] = {"amount": e["usd"], "open": True}
            elif e["id"] in state:
                if e["op"] == "settle":
                    state[e["id"]] = {"amount": e["usd"], "open": False}
                elif e["op"] == "release":
                    state[e["id"]] = {"amount": 0.0, "open": False}
        return round(sum(s["amount"] for s in state.values()), 6)

    def remaining(self, run_id: str | None = None) -> float:
        return round(self.budget.dollars - self.committed(run_id), 6)

    def reserve(self, usd: float, what: str, run_id: str | None = None) -> str:
        if usd < 0:
            raise ValueError("a reservation cannot be negative")
        left = self.remaining(run_id)
        if usd > left + 1e-9:
            raise BudgetExceeded(
                f"{what} needs ${usd:.2f}; ${left:.2f} of {self.budget.describe()} is left. "
                "Raise the budget in the Trainfile to allow it."
            )
        rid = uuid.uuid4().hex[:12]
        self._append({"op": "reserve", "id": rid, "usd": usd, "what": what, "run_id": run_id,
                      "at": datetime.now(timezone.utc).isoformat()})
        return rid

    def settle(self, rid: str, usd: float) -> None:
        self._append({"op": "settle", "id": rid, "usd": usd, "at": datetime.now(timezone.utc).isoformat()})

    def release(self, rid: str) -> None:
        self._append({"op": "release", "id": rid, "usd": 0, "at": datetime.now(timezone.utc).isoformat()})
