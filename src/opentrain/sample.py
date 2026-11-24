"""Sampling: recipes ``cl/rollout``, ``cl/filter`` and ``data/shortest``.

``sample`` runs each training task ``n`` times, each in a fresh worktree, then runs the task's
verifier on that same worktree after the runner has finished. The runner never verifies, so no
model grades itself. Every attempt is reserved against the budget before it starts and settled
with its real cost after; a reservation the budget cannot cover stops the sampling before that
attempt launches.

Attempts are written to ``.opentrain/attempts/<task>/<n>.json``. Held-out tasks are refused:
their transcripts must never exist on disk where the trainer can read them.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import workspace as ws_mod
from .budget import BudgetExceeded, Ledger
from .plugs import AgentRunner, Attempt
from .record import Run
from .tasks import HeldoutStore, Task


def fresh_checkout(task: Task):
    """Kept for callers; ``workspace.checkout`` serialises git's worktree calls itself."""
    return ws_mod.checkout(task)


def attempt_task(task: Task, runner: AgentRunner, *, timeout: float = 900, test_timeout: float = 900) -> Attempt:
    """One attempt in a fresh worktree, then the task's verifier on that worktree. Sets ``passed``."""
    with fresh_checkout(task) as ws:
        try:
            att = runner.attempt(task.instruction, ws, task_id=task.task_id, timeout=timeout)
        except Exception as exc:  # a crashing runner is a failed attempt, not a crashed run
            return Attempt(task_id=task.task_id, model=getattr(runner, "name", "?"), passed=False, seconds=0.0,
                           error=f"runner crashed: {type(exc).__name__}: {exc}")
        att.passed = ws_mod.run_tests(task, ws, test_timeout).passed
    return att


def attempts_dir(state: Path) -> Path:
    return state / "attempts"


def _next_index(folder: Path) -> int:
    nums = [int(p.stem) for p in folder.glob("*.json") if p.stem.isdigit()]
    return max(nums, default=-1) + 1


def save_attempt(state: Path, att: Attempt, index: int) -> Path:
    path = attempts_dir(state) / att.task_id / f"{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(att), indent=1, default=str))
    return path


def load_attempts(state: Path, task_ids: list[str] | None = None) -> list[Attempt]:
    root = attempts_dir(state)
    if not root.is_dir():
        return []
    out = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if task_ids is not None and folder.name not in task_ids:
            continue
        for p in sorted(folder.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
            out.append(Attempt(**json.loads(p.read_text())))
    return out


def sample(tasks: list[Task], runner: AgentRunner, n: int, state: Path, ledger: Ledger, run: Run | None = None,
           est_usd_per_attempt: float = 0.0, *, workers: int = 1, timeout: float = 900,
           test_timeout: float = 900) -> list[Attempt]:
    """Recipe ``cl/rollout``: every task ``n`` times, verified afterwards, budgeted, recorded, saved.

    Raises ``BudgetExceeded`` once the budget refuses a reservation; attempts finished before then are
    already saved (``load_attempts``) and in the record.
    """
    held = HeldoutStore(state)
    leaked = [t.task_id for t in tasks if t.task_id in held]
    if leaked:
        raise ValueError(f"refusing to sample held-out tasks: {', '.join(leaked)}")
    if n < 1:
        raise ValueError("n must be at least 1")

    lock = threading.Lock()  # the ledger and the run record are read-then-append files
    stop = threading.Event()
    refused: list[str] = []
    jobs: list[tuple[Task, int]] = []
    for task in tasks:
        first = _next_index(attempts_dir(state) / task.task_id)
        jobs += [(task, first + i) for i in range(n)]
    run_id = run.run_id if run else None

    def one(job: tuple[Task, int]) -> Attempt | None:
        task, index = job
        if stop.is_set():
            return None
        with lock:
            try:
                rid = ledger.reserve(est_usd_per_attempt, f"attempt {task.task_id}#{index} ({runner.name})", run_id)
            except BudgetExceeded as exc:
                stop.set()
                refused.append(str(exc))
                return None
        started = time.monotonic()
        try:
            att = attempt_task(task, runner, timeout=timeout, test_timeout=test_timeout)
        except BaseException:
            with lock:
                ledger.release(rid)
            raise
        with lock:
            ledger.settle(rid, att.cost_usd)
            if run:
                run.add("spend", what="sample", task_id=task.task_id, index=index, model=att.model,
                        usd=att.cost_usd, tokens_in=att.tokens_in, tokens_out=att.tokens_out,
                        seconds=round(time.monotonic() - started, 3), passed=att.passed)
        save_attempt(state, att, index)
        return att

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, jobs))
    else:
        results = [one(j) for j in jobs]
    done = [a for a in results if a is not None]
    if refused:
        if run:
            run.add("note", text=f"sampling stopped by the budget after {len(done)} of {len(jobs)} attempts")
        raise BudgetExceeded(f"{refused[0]} ({len(done)} of {len(jobs)} attempts ran and are saved)")
    return done


def shortest(attempts: list[Attempt]) -> dict[str, Attempt]:
    """Recipe ``data/shortest``: per task, the verified pass with the fewest steps.

    Ties go to fewer tool calls, then fewer seconds. Unverified and failed attempts never qualify.
    """
    best: dict[str, Attempt] = {}
    for a in attempts:
        if a.passed is not True:
            continue
        cur = best.get(a.task_id)
        if cur is None or (a.steps, a.tool_calls, a.seconds) < (cur.steps, cur.tool_calls, cur.seconds):
            best[a.task_id] = a
    return best


@dataclass
class FilterReport:
    """Recipe ``cl/filter``: tasks by how many of their ``n`` attempts passed."""

    none_passed: list[str] = field(default_factory=list)
    all_passed: list[str] = field(default_factory=list)
    mixed: list[str] = field(default_factory=list)
    rates: dict[str, tuple[int, int]] = field(default_factory=dict)


def filter_tasks(attempts: list[Attempt]) -> FilterReport:
    """Flag tasks that passed 0/n or n/n.

    RL (GRPO-style) needs variance within a task's group: when every attempt passes, or none does,
    the advantages are all zero and the task teaches nothing, so RL filtering drops both. For the
    speed goal with SFT on the shortest pass this differs: an n/n task is fine (it has passes to
    imitate, and ``shortest`` picks the quickest), while a 0/n task has nothing to imitate and is a
    candidate for the fix goal instead.
    """
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for a in attempts:
        tally[a.task_id][0] += a.passed is True
        tally[a.task_id][1] += 1
    rep = FilterReport(rates={t: (p, n) for t, (p, n) in tally.items()})
    for t, (p, n) in sorted(tally.items()):
        (rep.none_passed if p == 0 else rep.all_passed if p == n else rep.mixed).append(t)
    return rep
