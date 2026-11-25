"""The promotion gate. The trainer proposes; the gate decides. A run cannot change it.

Default rule (plan, "Limits the engine cannot change"): the candidate is better on the goal's
objective, no held-out task regresses, and every verifier still passes its controls. "Better" is
measured on the held-out set only, never on training scores.

Objectives:
    speed  median seconds on tasks both versions pass, with pass rate not lower
    cost   median tokens on tasks both pass, with pass rate not lower
    fix    the flagged tasks now pass, pass rate not lower
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Score:
    task_id: str
    passed: bool
    seconds: float | None = None
    tokens: int | None = None


@dataclass
class Decision:
    promote: bool
    reasons: list[str]
    regressions: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        head = "promoted" if self.promote else "rejected"
        return f"{head}: " + "; ".join(self.reasons)


def _by_id(scores: list[Score]) -> dict[str, Score]:
    return {s.task_id: s for s in scores}


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def decide(
    current: list[Score] | None,
    candidate: list[Score],
    goal: str = "speed",
    *,
    allow_regressions: bool = False,
    broken_verifiers: list[str] | None = None,
    flagged: list[str] | None = None,
    min_gain: float = 0.05,
) -> Decision:
    """``current`` None means there is no live version yet; the base model's scores stand in."""
    cand = _by_id(candidate)
    if broken_verifiers:
        return Decision(False, [f"{len(broken_verifiers)} verifiers failed their controls"],
                        summary={"broken_verifiers": broken_verifiers})
    if not cand:
        return Decision(False, ["no held-out scores for the candidate"])

    cur = _by_id(current or [])
    missing = sorted(set(cur) - set(cand))
    if missing:
        return Decision(False, [f"candidate was not scored on {len(missing)} held-out tasks the current version was"])
    unseen = sorted(set(cand) - set(cur))
    if cur and unseen:
        # The baseline predates these held-out tasks, so nothing says the current version would
        # fail them. Comparing only the overlap would let a candidate fail every new task.
        return Decision(False, [f"the baseline was scored before {len(unseen)} of these held-out tasks existed; "
                                "re-score the current version first"], summary={"unscored_in_baseline": unseen})

    shared = sorted(set(cur) & set(cand))
    regressions = [t for t in shared if cur[t].passed and not cand[t].passed]
    fixed = [t for t in shared if not cur[t].passed and cand[t].passed]
    cur_rate = sum(cur[t].passed for t in shared) / len(shared) if shared else None
    cand_rate = sum(s.passed for s in cand.values()) / len(cand)
    cand_rate_shared = sum(cand[t].passed for t in shared) / len(shared) if shared else cand_rate

    summary: dict[str, Any] = {
        "tasks": len(cand),
        "pass_rate": round(cand_rate, 4),
        "current_pass_rate": round(cur_rate, 4) if cur_rate is not None else None,
        "fixed": len(fixed),
        "regressed": len(regressions),
    }
    reasons: list[str] = []
    ok = True

    if regressions and not allow_regressions:
        ok = False
        reasons.append(f"{len(regressions)} held-out tasks regressed")
    if cur_rate is not None and cand_rate_shared < cur_rate:
        ok = False
        reasons.append(f"pass rate fell from {cur_rate:.0%} to {cand_rate_shared:.0%}")

    if not cur:
        reasons.append(f"first version: {cand_rate:.0%} pass on {len(cand)} held-out tasks")
        return Decision(ok and cand_rate > 0, reasons or ["no task passes"], regressions, fixed, summary)

    both = [t for t in shared if cur[t].passed and cand[t].passed]
    if goal in ("speed", "cost"):
        attr = "seconds" if goal == "speed" else "tokens"
        def measured(s: Score) -> float | None:
            v = getattr(s, attr)
            return None if v is None or (attr == "tokens" and v <= 0) else v  # 0 tokens = not reported

        # Only tasks measured on both sides, so the two medians describe the same work.
        pairs = [(measured(cur[t]), measured(cand[t])) for t in both]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        a = _median([x for x, _ in pairs])
        b = _median([y for _, y in pairs])
        summary[f"median_{attr}"] = b
        summary[f"current_median_{attr}"] = a
        if a is None or b is None:
            gained = len(fixed) > 0
            reasons.append(f"no {attr} on shared passes; " + ("more tasks pass" if gained else "nothing gained"))
        else:
            gain = (a - b) / a if a else 0.0
            summary["gain"] = round(gain, 4)
            gained = gain >= min_gain or (len(fixed) > 0 and gain >= 0)
            reasons.append(f"median {attr} {a:.4g} -> {b:.4g} ({gain:+.0%})")
        if not gained:
            ok = False
            reasons.append(f"not better on {goal}")
    elif goal == "fix":
        targets = flagged or []
        still = [t for t in targets if t not in cand or not cand[t].passed]  # unscored counts as failing
        if not targets:
            ok = False
            reasons.append("fix goal with no flagged tasks")
        elif still:
            ok = False
            reasons.append(f"{len(still)} of {len(targets)} flagged tasks still fail")
        else:
            reasons.append(f"all {len(targets)} flagged tasks pass")
    else:
        ok = False
        reasons.append(f"unknown goal {goal!r}")

    return Decision(ok, reasons, regressions, fixed, summary)
