"""What git can still tell us about a session's repository, after the fact.

A task needs three things from the repo: the code as it stood when the session began (the
environment), tests that can judge the work (the verifier), and the code as it stood when the
work was done (the reference). This module answers each question from the repo's history and
never changes the repo: every call is a read.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

TEST_FILE = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.py$|_test\.(py|go)$|\.(test|spec)\.[jt]sx?$"
    r"|(^|/)[^/]+Test\.(java|kt)$|_spec\.rb$"
)


def _git(cwd: str, *args: str, timeout: float = 20) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@lru_cache(maxsize=None)
def repo_root(cwd: str) -> str | None:
    if not Path(cwd).is_dir():
        return None
    return _git(cwd, "rev-parse", "--show-toplevel")


@lru_cache(maxsize=None)
def has_tests(root: str, commit: str) -> bool:
    files = _git(root, "ls-tree", "-r", "--name-only", commit, timeout=60)
    return bool(files) and any(TEST_FILE.search(f) for f in files.splitlines())


@dataclass(frozen=True)
class RepoState:
    root: str | None
    base_commit: str | None
    """The last commit at or before the session began: the task's starting environment."""
    tests_in_repo: bool
    reference_commit: str | None
    """The first commit after the session began that touches a file the session edited."""
    resolved_from: str | None = None
    """Set when the session's folder was gone and ``root`` was matched from its path instead."""


def find_repos(search: list[Path]) -> list[str]:
    """Git repos directly inside each search folder (and the folders themselves, if repos)."""
    found: list[str] = []
    for base in search:
        if not base.is_dir():
            continue
        for p in [base, *sorted(c for c in base.iterdir() if c.is_dir())]:
            if (p / ".git").exists() and str(p) not in found:
                found.append(str(p))
    return found


def resolve_vanished(cwd: str, known_repos: list[str]) -> str | None:
    """The repo a deleted worktree belonged to.

    Workspace tools (Conductor, T3 worktrees, ``git worktree``) delete a session's folder once its
    branch merges, but the history stays in the main clone. The match is by name: a known repo
    whose folder name appears as a component of the vanished path. Longest name wins, so
    ``acme-platform`` beats ``acme``.
    """
    comps = Path(cwd).parts
    matches = [
        r for r in known_repos
        if any(c == Path(r).name or c.startswith(Path(r).name + "-") for c in comps)
    ]
    return max(matches, key=lambda r: len(Path(r).name)) if matches else None


def repo_of_edits(files_edited: list[str]) -> str | None:
    """The repo most of a session's edits landed in, for a session started outside any repo."""
    counts: dict[str, int] = {}
    for f in files_edited:
        parent = Path(f).parent
        while not parent.is_dir() and parent != parent.parent:
            parent = parent.parent
        root = repo_root(str(parent)) if parent.is_dir() else None
        if root:
            counts[root] = counts.get(root, 0) + 1
    return max(counts, key=counts.__getitem__) if counts else None


def _relative(files: list[str], *bases: str) -> list[str]:
    rel: list[str] = []
    for f in files:
        for base in bases:
            try:
                r = str(Path(f).relative_to(base))
            except ValueError:
                continue
            if r not in rel:
                rel.append(r)
            break
    return rel


def repo_state(
    cwd: str | None,
    branch: str | None,
    started: datetime,
    ended: datetime,
    files_edited: list[str],
    reference_window: timedelta = timedelta(days=2),
    known_repos: list[str] | None = None,
) -> RepoState:
    if not cwd:
        return RepoState(None, None, False, None)
    root = repo_root(cwd)
    vanished = root is None and not Path(cwd).exists()
    if vanished and known_repos:
        root = resolve_vanished(cwd, known_repos)
    if root is None:
        root = repo_of_edits(files_edited)
        vanished = vanished or root is not None
    if root is None:
        return RepoState(None, None, False, None)
    refs = [branch] if branch and branch != "HEAD" and _git(root, "rev-parse", "--verify", "-q", branch) else []
    refs = refs or ["--all"]
    base = _git(root, "rev-list", "-1", f"--before={started.isoformat()}", *refs) or None
    tests = has_tests(root, base) if base else False

    reference = None
    # A vanished worktree's paths are relative to the session's folder, which was its root.
    rel = _relative(files_edited, root, str(Path(root).resolve()), cwd)
    if base and rel:
        until = (ended + reference_window).isoformat()
        log = _git(
            root, "log", "--format=%H", "--reverse", f"--since={started.isoformat()}", f"--until={until}",
            *refs, "--", *rel,
        )
        if log:
            reference = log.splitlines()[0]
    return RepoState(root, base, tests, reference, resolved_from=cwd if vanished else None)
