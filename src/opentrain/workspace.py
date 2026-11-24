"""Workspaces and the verifier: the one place a task's code is checked out and judged.

A workspace is a detached ``git worktree`` of the task's repo at its base commit, in a temp
folder. Nothing touches the user's own checkout. The verifier overlays the task's test files (as
they stood at the reference commit) and runs the test command. The verdict comes from the exit
code alone, so no model, and no runner, grades its own work.

Locally this is a light sandbox: the code runs as the user. Harbor/Docker execution of the same
task folder is the isolated path (``opentrain lint --harbor``).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .tasks import Task


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Verdict:
    passed: bool
    exit_code: int | None
    seconds: float
    output: str
    """The tail of the test output. Never shown to the trainer for held-out tasks."""
    timed_out: bool = False


def _git(repo: str | Path, *args: str, check: bool = True, input: str | None = None) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, input=input)
    if check and out.returncode != 0:
        raise WorkspaceError(f"git {' '.join(args[:3])}: {out.stderr.strip()[:500]}")
    return out.stdout


@contextlib.contextmanager
def checkout(task: Task, keep: bool = False) -> Iterator[Path]:
    """A throwaway worktree of the task's repo at ``base_commit``."""
    if not Path(task.repo).is_dir():
        raise WorkspaceError(f"repo {task.repo} is not on this machine")
    tmp = Path(tempfile.mkdtemp(prefix=f"ot-{task.task_id[:12]}-"))
    ws = tmp / "repo"
    _git(task.repo, "worktree", "add", "--detach", "-q", str(ws), task.base_commit)
    try:
        yield ws
    finally:
        if not keep:
            _git(task.repo, "worktree", "remove", "--force", str(ws), check=False)
            shutil.rmtree(tmp, ignore_errors=True)
            _git(task.repo, "worktree", "prune", check=False)


def diff(workspace: Path, exclude: tuple[str, ...] = ()) -> str:
    """Everything changed in the workspace since its base commit, new files included."""
    _git(workspace, "add", "-A", "-N")
    args = ["diff", "--binary", "HEAD", "--", "."]
    args += [f":(exclude){p}" for p in exclude]
    return _git(workspace, *args)


def apply_patch(workspace: Path, patch: str) -> None:
    if not patch.strip():
        return
    out = subprocess.run(["git", "-C", str(workspace), "apply", "--whitespace=nowarn", "-"],
                         input=patch, capture_output=True, text=True)
    if out.returncode != 0:
        raise WorkspaceError(f"patch does not apply: {out.stderr.strip()[:500]}")


def reference_patch(task: Task) -> str:
    """The reference answer: base..reference, test files left out (the verifier brings its own)."""
    args = ["diff", "--binary", task.base_commit, task.reference_commit, "--", "."]
    args += [f":(exclude){p}" for p in task.test_files]
    return _git(task.repo, *args)


def overlay_tests(task: Task, workspace: Path) -> None:
    """Put the task's test files in place, exactly as they stood at the reference commit."""
    for rel in task.test_files:
        dest = workspace / rel
        content = subprocess.run(["git", "-C", task.repo, "show", f"{task.reference_commit}:{rel}"],
                                 capture_output=True)
        if content.returncode != 0:
            if dest.exists():
                dest.unlink()  # deleted at the reference
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content.stdout)


def run_tests(task: Task, workspace: Path, timeout: float = 900) -> Verdict:
    overlay_tests(task, workspace)
    start = time.monotonic()
    env = {**os.environ, "CI": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen(task.test_command, shell=True, cwd=workspace, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        timed_out = True
    secs = time.monotonic() - start
    code = None if timed_out else proc.returncode
    return Verdict(code == 0, code, round(secs, 2), (out or "")[-4000:], timed_out)


def verify(task: Task, patch: str, timeout: float = 900) -> Verdict:
    """Apply ``patch`` to a fresh checkout at the base commit and run the verifier on it."""
    with checkout(task) as ws:
        try:
            apply_patch(ws, patch)
        except WorkspaceError as exc:
            return Verdict(False, None, 0.0, str(exc))
        return run_tests(task, ws, timeout)
