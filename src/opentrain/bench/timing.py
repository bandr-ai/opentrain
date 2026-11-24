"""Where a task's time goes: the model, the tools, or the person.

The speed claim rests on this. A tuned model can only shrink model time; if tools dominate, a
5x speedup is out of reach whatever the model does (plan risk "Speed gain falls short"). So the
number that matters is the model's share of *working* time (model + tool), not of wall-clock.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

from ..sessions.model import Session

_SEP = re.compile(r"&&|\|\||;|\n|\|")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def command_head(command: str) -> str:
    """The program a shell command really runs: past ``cd``, ``export`` and ``VAR=`` prefixes."""
    for part in _SEP.split(command):
        words = [w for w in part.split() if not _ASSIGN.match(w)]
        if words and words[0] not in ("cd", "export", "set", "source", ".", "time", "env"):
            return words[0].rsplit("/", 1)[-1][:30]
    return "cd"


def _pct(a: float, b: float) -> float | None:
    return round(100 * a / b, 1) if b else None


def ceiling_speedup(model_share: float, model_speedup: float) -> float:
    """Amdahl: the end-to-end speedup if only model time gets ``model_speedup`` times faster."""
    return 1 / ((1 - model_share) + model_share / model_speedup)


def breakdown(sessions: list[Session]) -> dict[str, Any]:
    totals = {k: sum(s.seconds(k) for s in sessions) for k in ("model", "tool", "human", "idle")}
    working = totals["model"] + totals["tool"]
    shares = [
        s.seconds("model") / (s.seconds("model") + s.seconds("tool"))
        for s in sessions
        if s.seconds("model") + s.seconds("tool") > 0
    ]
    model_share = totals["model"] / working if working else 0.0

    tool_time: Counter[str] = Counter()
    for s in sessions:
        for c in s.tool_calls:
            if c.seconds is not None and not c.sidechain:
                label = f"{c.name}: {command_head(c.command)}" if c.command else c.name
                tool_time[label] += c.seconds

    reported = [
        (s.reported_api_seconds, s.reported_tool_seconds)
        for s in sessions
        if s.reported_api_seconds is not None and s.reported_tool_seconds is not None
    ]
    rep_api = sum(a for a, _ in reported)
    rep_tool = sum(t for _, t in reported)

    return {
        "sessions": len(sessions),
        "hours": {k: round(v / 3600, 2) for k, v in totals.items()},
        "model_share_of_working_time_pct": _pct(totals["model"], working),
        "median_session_model_share_pct": round(100 * statistics.median(shares), 1) if shares else None,
        "ceiling_speedup_if_model_10x": round(ceiling_speedup(model_share, 10), 2) if working else None,
        "ceiling_speedup_if_model_infinite": round(1 / (1 - model_share), 2) if 0 < model_share < 1 else None,
        "top_tool_time_hours": {k: round(v / 3600, 2) for k, v in tool_time.most_common(12)},
        "agent_reported": {
            "sessions": len(reported),
            "model_share_pct": _pct(rep_api, rep_api + rep_tool),
            "note": "Claude Code's own cost-state totals, a cross-check on the timeline split",
        },
        "untimed_tool_calls": sum(s.untimed for s in sessions),
    }
