"""Race: one task, several models at once, side by side.

The demo of the product's claim: the frontier (headless ``claude -p``) and the specialist (direct
runtime) start the same task at the same moment, each in its own fresh worktree, and each is judged
by the task's verifier after it finishes. Times are measured, never estimated. Both lanes share the
machine, so a CPU-heavy test suite slows both; the comparison stays fair, not the absolute numbers.

By default the task is the newest held-out task, since a race on a training task proves nothing.
Only numbers are shown, never the task's content or test output.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .plugs import AgentRunner
from .sample import attempt_task
from .tasks import HeldoutStore, Task


@dataclass
class Lane:
    name: str
    model: str
    passed: bool
    seconds: float
    model_seconds: float
    steps: int
    tool_calls: int
    tokens: int
    cost_usd: float
    error: str | None = None


@dataclass
class RaceResult:
    task_id: str
    lanes: dict[str, Lane] = field(default_factory=dict)

    @property
    def winner(self) -> str | None:
        """The fastest lane that passed, or None if none did."""
        passing = [lane for lane in self.lanes.values() if lane.passed]
        return min(passing, key=lambda lane: lane.seconds).name if passing else None

    def render(self) -> str:
        names = list(self.lanes)
        rows = [
            ("result", lambda x: "PASS" if x.passed else "fail"),
            ("time", lambda x: f"{x.seconds:.1f}s"),
            ("model time", lambda x: f"{x.model_seconds:.1f}s"),
            ("steps", lambda x: str(x.steps)),
            ("tool calls", lambda x: str(x.tool_calls)),
            ("tokens", lambda x: f"{x.tokens:,}"),
            ("cost", lambda x: f"${x.cost_usd:.4f}" if x.cost_usd else "-"),
        ]
        width = max(12, *(len(n) for n in names)) + 2
        lines = [f"race on {self.task_id}", "".ljust(12) + "".join(n.ljust(width) for n in names)]
        for label, fn in rows:
            lines.append(label.ljust(12) + "".join(fn(self.lanes[n]).ljust(width) for n in names))
        for n in names:
            if self.lanes[n].error:
                lines.append(f"  {n}: {self.lanes[n].error[:200]}")
        win = self.winner
        if win:
            others = [lane for lane in self.lanes.values() if lane.name != win and lane.passed]
            tail = ""
            if others:
                runner_up = min(others, key=lambda lane: lane.seconds)
                if self.lanes[win].seconds > 0:
                    tail = f", {runner_up.seconds / self.lanes[win].seconds:.1f}x faster than {runner_up.name}"
            lines.append(f"winner: {win}{tail}")
        else:
            lines.append("no lane passed")
        return "\n".join(lines).rstrip()


def newest_heldout(state: Path) -> Task:
    tasks = list(HeldoutStore(state))
    if not tasks:
        raise ValueError("there are no held-out tasks to race on")
    return max(tasks, key=lambda t: (datetime.fromisoformat(t.created_at.replace("Z", "+00:00")).timestamp()
                                     if t.created_at else float("-inf"), t.task_id))


def race(task: Task, runners: dict[str, AgentRunner], *, timeout: float = 900,
         test_timeout: float = 900) -> RaceResult:
    """Run every runner on ``task`` concurrently, verify each, and line the numbers up."""
    def lane(item: tuple[str, AgentRunner]) -> Lane:
        name, runner = item
        att = attempt_task(task, runner, timeout=timeout, test_timeout=test_timeout)
        return Lane(name=name, model=att.model, passed=att.passed is True, seconds=att.seconds,
                    model_seconds=att.model_seconds, steps=att.steps,
                    tool_calls=att.tool_calls, tokens=att.tokens_in + att.tokens_out, cost_usd=att.cost_usd,
                    error=att.error)

    with ThreadPoolExecutor(max_workers=max(1, len(runners))) as pool:
        lanes = list(pool.map(lane, runners.items()))
    return RaceResult(task.task_id, {lane.name: lane for lane in lanes})
