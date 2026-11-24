"""The session model every bench measure and data recipe reads.

A session is one coding-agent conversation as the agent recorded it. Readers for each agent
(Claude Code first) produce this shape; nothing downstream opens a raw log again.

Times are measured, never synthesised: an interval is only recorded when the log holds a
timestamp at both ends. Where it does not, the interval is left out and counted in
``Session.untimed``, so a time breakdown never rests on invented numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

TestOutcome = Literal["pass", "fail", "unknown"]


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    started_at: datetime
    ended_at: datetime | None
    """None when the result never arrived or carried no timestamp."""
    is_error: bool = False
    command: str | None = None
    """The shell command, for a shell tool. Kept for test detection, never exported raw."""
    sidechain: bool = False
    """Run by a subagent rather than the main loop."""

    @property
    def seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


@dataclass(frozen=True)
class TestRun:
    """One run of a test command found in the session, and what it reported."""

    __test__ = False  # not a pytest class

    call_id: str
    command: str
    at: datetime
    outcome: TestOutcome
    cwd: str | None = None
    """The shell's folder when the command started. Claude Code's shell keeps its folder between
    commands, so a bare ``pnpm test`` may have run wherever an earlier ``cd`` left it."""


@dataclass(frozen=True)
class Interval:
    kind: Literal["model", "tool", "human", "idle"]
    """model: waiting on the model. tool: a tool is running. human: the agent finished and waits for
    the person. idle: a model or tool gap so long it is the machine asleep, not work (see ``IDLE_CAP``)."""
    started_at: datetime
    ended_at: datetime

    @property
    def seconds(self) -> float:
        return max(0.0, (self.ended_at - self.started_at).total_seconds())


@dataclass
class Session:
    session_id: str
    agent: str
    path: str
    cwd: str | None
    git_branch: str | None
    started_at: datetime
    ended_at: datetime
    instruction: str | None
    """The first thing the person typed. None when the log records no human prompt."""
    human_turns: int
    prompts: list[str] = field(default_factory=list)
    """Every prompt the person typed, in order (``instruction`` is the first)."""
    tool_calls: list[ToolCall] = field(default_factory=list)
    intervals: list[Interval] = field(default_factory=list)
    test_runs: list[TestRun] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    cost_usd: float | None = None
    reported_api_seconds: float | None = None
    """The agent's own count of model time, when it records one (Claude Code's cost-state)."""
    reported_tool_seconds: float | None = None
    untimed: int = 0
    """Intervals the log could not time at both ends."""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def edited(self) -> bool:
        return bool(self.files_edited)

    def seconds(self, kind: str) -> float:
        return sum(i.seconds for i in self.intervals if i.kind == kind)

    def summary(self) -> dict[str, Any]:
        """A JSON-safe digest with no session content beyond counts, paths and the instruction length."""
        return {
            "session_id": self.session_id,
            "agent": self.agent,
            "cwd": self.cwd,
            "git_branch": self.git_branch,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "human_turns": self.human_turns,
            "instruction_chars": len(self.instruction or ""),
            "tool_calls": len(self.tool_calls),
            "test_runs": [
                {"at": t.at.isoformat(), "outcome": t.outcome} for t in self.test_runs
            ],
            "files_edited": len(self.files_edited),
            "models": self.models,
            "cost_usd": self.cost_usd,
            "seconds": {k: round(self.seconds(k), 1) for k in ("model", "tool", "human", "idle")},
            "reported_api_seconds": self.reported_api_seconds,
            "reported_tool_seconds": self.reported_tool_seconds,
            "untimed": self.untimed,
        }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
