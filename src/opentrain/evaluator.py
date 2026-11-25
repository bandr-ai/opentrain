"""The evaluator role: the only reader of the held-out set.

The trainer asks for an evaluation and gets back numbers per held-out task: passed, seconds,
tokens. Never the instruction, the transcript, the patch or the test output, because anything more
would let the trainer fit the held-out set. Each attempt runs in a fresh worktree and is judged by
the task's own verifier afterwards, never by the runner.

Requests are capped per run (``eval cap`` in the Trainfile). The count is the number of ``eval``
events already in the run record, so it survives restarts and cannot be reset by the engine.

Baselines (the base model, the frontier, each served endpoint) are stored under
``.opentrain/heldout/scores/<name>.json``, inside the folder only the evaluator reads.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .gate import Score
from .plugs import AgentRunner
from .record import Run
from .sample import attempt_task
from .state import write_json
from .tasks import HeldoutStore


class EvalBroken(RuntimeError):
    """Attempts errored (CLI not logged in, endpoint down, rate limit). Their failures say nothing
    about the model, so they are never kept as scores."""


class EvalCapReached(RuntimeError):
    pass


def summarise(scores: list[Score]) -> dict[str, Any]:
    """Pass rate, and median seconds over passing tasks (a fast failure is not speed)."""
    passed = [s for s in scores if s.passed]
    secs = [s.seconds for s in passed if s.seconds is not None]
    return {"tasks": len(scores), "passed": len(passed),
            "pass_rate": round(len(passed) / len(scores), 4) if scores else None,
            "median_seconds": round(statistics.median(secs), 3) if secs else None}


class Evaluator:
    def __init__(self, state: Path, cap: int, *, timeout: float = 900, test_timeout: float = 900):
        self.state = state
        self.cap = cap
        self.heldout = HeldoutStore(state)
        self.timeout, self.test_timeout = timeout, test_timeout

    def _score_all(self, runner: AgentRunner) -> list[Score]:
        scores, errors = [], 0
        for task in self.heldout:
            att = attempt_task(task, runner, timeout=self.timeout, test_timeout=self.test_timeout)
            errors += bool(att.error) and not att.passed
            # The attempt (transcript, patch) is dropped here: only the numbers leave the evaluator.
            tokens = att.tokens_in + att.tokens_out
            scores.append(Score(task.task_id, att.passed is True, round(att.seconds, 3), tokens or None))
        if scores and errors > len(scores) // 2:
            raise EvalBroken(f"{errors} of {len(scores)} attempts by {runner.name} errored before finishing; "
                             "check the model is reachable (logged in, endpoint up) and try again")
        return scores

    def requests_used(self, run: Run) -> int:
        return sum(1 for e in run.events() if e["kind"] == "eval")

    def request(self, runner: AgentRunner, model_name: str, run: Run) -> list[Score]:
        """Score ``runner`` on every held-out task. Counts against this run's eval cap."""
        used = self.requests_used(run)
        if used >= self.cap:
            raise EvalCapReached(f"run {run.run_id} has used its {self.cap} eval requests (eval cap in the Trainfile)")
        if not self.heldout.ids():
            raise ValueError("there are no held-out tasks; run the split first")
        scores = self._score_all(runner)
        run.add("eval", model=model_name, runner=runner.name, request=used + 1, cap=self.cap,
                scores=[asdict(s) for s in scores], **summarise(scores))
        return scores

    # ------------------------------------------------------------ baselines

    def _scores_path(self, name: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._@+-]", "_", name).lstrip(".")
        if not safe:
            raise ValueError(f"bad baseline name {name!r}")
        return self.heldout.root / "scores" / f"{safe}.json"

    def baseline(self, runner: AgentRunner, name: str) -> list[Score]:
        """Score a reference model (base, frontier, an endpoint) and keep it for the gate to compare against.

        Outside any run, so it does not use a run's eval cap.
        """
        scores = self._score_all(runner)
        self.save_scores(name, scores)
        return scores

    def save_scores(self, name: str, scores: list[Score]) -> Path:
        """Keep a version's held-out scores under its own name, for later gates and ``opentrain diff``."""
        return write_json(self._scores_path(name), {"name": name, **summarise(scores),
                                                    "scores": [asdict(s) for s in scores]})

    def scores_for(self, name: str) -> list[Score] | None:
        path = self._scores_path(name)
        if not path.exists():
            return None
        return [Score(**s) for s in json.loads(path.read_text())["scores"]]
