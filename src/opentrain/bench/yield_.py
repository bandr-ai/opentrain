"""Task yield: of the sessions we have, how many could become a valid, verified task, and why the rest cannot.

This is the first risk in the plan ("Task yield is low") and the first step of v0.1 bench. Each
session goes down a funnel; it stops at the first check it fails, and that check is its skip
reason. The checks are the cheap, local ones: they say a task *could* be built. Building it and
proving its verifier (reference passes, doing nothing fails) is the next step, and its own count.

    has an instruction  -> the person asked for something
    changed code        -> there is work to check, not just a question answered
    repo state          -> the starting code can be recovered from git
    tests in repo       -> something can judge the work
    checks passed       -> the session's last test run passed (it ran tests and ended green)
    reference           -> the finished code can be recovered from git

Sessions that clear every step are "verified candidates". Sessions that clear all but the two
session-side checks (tests run green, reference committed) are "repairable": the engine may
still build a task from them by running the tests itself, at a higher cost.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..sessions.model import Session
from .gitstate import RepoState, repo_state

STEPS = (
    ("instruction", "no instruction"),
    ("edits", "no code changed"),
    ("repo_state", "no repo state"),
    ("tests_in_repo", "no tests in repo"),
    ("checks_passed", "checks not run or failed"),
    ("reference", "no committed reference"),
)


@dataclass
class Verdict:
    session: Session
    repo: RepoState
    passed: list[str]
    skip: str | None
    """The label of the first check failed, or None for a verified candidate."""
    detail: str | None = None

    @property
    def verified(self) -> bool:
        return self.skip is None

    @property
    def repairable(self) -> bool:
        return self.skip in ("checks not run or failed", "no committed reference")

    def row(self) -> dict[str, Any]:
        return {
            **self.session.summary(),
            "repo_root": self.repo.root,
            "base_commit": self.repo.base_commit,
            "reference_commit": self.repo.reference_commit,
            "tests_in_repo": self.repo.tests_in_repo,
            "repo_resolved_from": self.repo.resolved_from,
            "passed": self.passed,
            "skip": self.skip,
            "detail": self.detail,
            "verified": self.verified,
            "repairable": self.repairable,
        }


def _repo_detail(s: Session, repo: RepoState) -> str:
    if not s.cwd:
        return "no folder recorded"
    if repo.root:
        return "no commit before the session"
    return "folder gone" if not Path(s.cwd).exists() else "not a git repo"


def judge(s: Session, known_repos: list[str] | None = None) -> Verdict:
    repo = repo_state(
        s.cwd, s.git_branch, s.started_at, s.ended_at, s.files_edited, known_repos=known_repos
    )
    last_test = s.test_runs[-1] if s.test_runs else None
    checks = {
        "instruction": (s.instruction is not None, None),
        "edits": (s.edited, None),
        "repo_state": (
            repo.base_commit is not None,
            _repo_detail(s, repo),
        ),
        "tests_in_repo": (repo.tests_in_repo, None),
        "checks_passed": (
            last_test is not None and last_test.outcome == "pass",
            "never ran tests" if last_test is None else f"last test run: {last_test.outcome}",
        ),
        "reference": (repo.reference_commit is not None, "edits never committed"),
    }
    passed: list[str] = []
    for key, label in STEPS:
        ok, why = checks[key]
        if not ok:
            return Verdict(s, repo, passed, label, why)
        passed.append(key)
    return Verdict(s, repo, passed, None)


def funnel(verdicts: list[Verdict]) -> list[dict[str, Any]]:
    """How many sessions remain after each step, and how many each step drops."""
    out = []
    remaining = len(verdicts)
    for key, label in STEPS:
        dropped = sum(1 for v in verdicts if v.skip == label)
        out.append({"step": key, "label": label, "dropped": dropped, "remaining": remaining - dropped})
        remaining -= dropped
    return out


def summarise(verdicts: list[Verdict]) -> dict[str, Any]:
    total = len(verdicts)
    verified = sum(v.verified for v in verdicts)
    repairable = sum(v.repairable for v in verdicts)
    details = Counter(f"{v.skip}: {v.detail}" for v in verdicts if v.skip and v.detail)
    return {
        "sessions": total,
        "verified_candidates": verified,
        "repairable": repairable,
        "yield_per_100": round(100 * verified / total, 1) if total else 0.0,
        "coding_sessions": sum(1 for v in verdicts if v.session.edited),
        "yield_per_coding_session_pct": round(
            100 * verified / max(1, sum(1 for v in verdicts if v.session.edited)), 1
        ),
        "yield_with_repair_per_100": round(100 * (verified + repairable) / total, 1) if total else 0.0,
        "funnel": funnel(verdicts),
        "skip_details": dict(details.most_common()),
        "multi_prompt_candidates": sum(1 for v in verdicts if v.verified and v.session.human_turns > 3),
    }
