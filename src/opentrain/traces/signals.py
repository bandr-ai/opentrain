"""Where a trace went wrong: the evidence the agent reads to name what to fix (skill ``data/diagnose``).

These are cheap, local checks over a trace, not judgements. Each one points at a moment the
person or the tools pushed back: the person corrected the agent or stopped it, a check kept failing,
the agent edited the same file over and over, a tool call errored. The train agent reads the
moments in batches, names the recurring themes in the person's terms ("keeps using the old auth
API"), and the person picks which ones to fix. Nothing here decides what a theme is.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from . import EDIT_KINDS, Trace

CORRECTION = re.compile(
    r"^\s*(no\b|nope\b|wrong\b|stop\b|wait\b|undo\b|revert\b)"
    r"|\b(that'?s not (it|what|right)|not what i (asked|wanted|meant)|i (said|told you|asked you)|"
    r"you (didn'?t|did not|forgot|broke|missed|ignored)|why did you|don'?t (do|change|touch|use)|"
    r"instead of|still (fails|failing|broken|not working|doesn'?t work)|it'?s (still )?broken|"
    r"doesn'?t work|didn'?t work|try again|wrong (file|place|approach)|roll ?back)\b",
    re.IGNORECASE,
)
INTERRUPTED = re.compile(r"^\[Request interrupted", re.IGNORECASE)
_TEST = re.compile(r"(?<![\w-])(pytest|py\.test|unittest|tox|(npm|pnpm|yarn|bun)( run)? test|vitest|jest|go test|"
                   r"cargo test|make (test|check)|rspec|phpunit|mvn test|gradlew? test|dotnet test|tsc\b|ruff|eslint|"
                   r"mypy|pyright)")


@dataclass
class Moment:
    trace: str
    agent: str
    request: int
    kind: str
    """``corrected``, ``interrupted``, ``failing_check``, ``churn`` or ``tool_errors``."""
    detail: str
    prompt: str
    """The request the moment belongs to (clipped)."""
    cwd: str | None
    at: str | None

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _clip(text: str, n: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def moments(trace: Trace) -> list[Moment]:
    out: list[Moment] = []
    req = -1
    prompt = ""
    checks: Counter = Counter()
    edits: Counter = Counter()
    errors = 0

    def flush() -> None:
        nonlocal errors
        base = dict(trace=trace.trace_id, agent=trace.agent, request=req, prompt=_clip(prompt), cwd=trace.cwd,
                    at=trace.started_at.isoformat() if trace.started_at else None)
        for cmd, n in checks.items():
            if n >= 3:
                out.append(Moment(kind="failing_check", detail=f"`{_clip(cmd, 120)}` failed {n} times", **base))
        for path, n in edits.items():
            if n >= 5:
                out.append(Moment(kind="churn", detail=f"edited {path} {n} times", **base))
        if errors >= 4:
            out.append(Moment(kind="tool_errors", detail=f"{errors} tool calls errored", **base))
        checks.clear()
        edits.clear()
        errors = 0

    for s in trace.steps:
        if s.source == "user" and s.text.strip():
            text = s.text.strip()
            if req >= 0:
                flush()
                base = dict(trace=trace.trace_id, agent=trace.agent, request=req, prompt=_clip(prompt),
                            cwd=trace.cwd, at=trace.started_at.isoformat() if trace.started_at else None)
                if INTERRUPTED.search(text):
                    out.append(Moment(kind="interrupted", detail="the person stopped the agent", **base))
                elif CORRECTION.search(text[:300]):
                    out.append(Moment(kind="corrected", detail=_clip(text, 300), **base))
            if not INTERRUPTED.search(text):
                req += 1
                prompt = text
            continue
        if s.source != "agent" or s.sidechain:
            continue
        for t in s.tools:
            if t.is_error:
                errors += 1
            if t.kind in EDIT_KINDS and isinstance(t.args.get("path"), str):
                edits[t.args["path"]] += 1
            cmd = t.args.get("command")
            cmd = " ".join(map(str, cmd)) if isinstance(cmd, list) else cmd
            failed = t.is_error or (t.exit_code not in (None, 0))
            if t.kind == "shell" and isinstance(cmd, str) and _TEST.search(cmd) and failed:
                checks[cmd.strip()[:200]] += 1
    if req >= 0:
        flush()
    return out


def summary(traces: list[Trace]) -> dict[str, Any]:
    """Counts of each kind of moment, and how many traces have any."""
    kinds: Counter = Counter()
    with_any = 0
    for t in traces:
        ms = moments(t)
        with_any += bool(ms)
        kinds.update(m.kind for m in ms)
    return {"traces": len(traces), "traces_with_moments": with_any, "moments": dict(kinds.most_common())}
