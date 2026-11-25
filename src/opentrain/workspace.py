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
import hashlib
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .tasks import Task


class WorkspaceError(RuntimeError):
    pass


ENV_DIRS = (".venv", "venv", "node_modules", ".tox", "vendor/bundle")
"""Untracked dependency folders a task's tests need but a fresh worktree lacks. They are linked
from the main clone, read-only in spirit: an attempt that changes dependencies changes them for
the user too, so the agent's prompt says not to install packages."""

_git_lock = threading.Lock()
"""``git worktree add/remove/prune`` on one repo from several threads at once races on the
repo's shared metadata. Worktrees are independent once they exist, so only those calls lock."""


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
    with _git_lock:
        _git(task.repo, "worktree", "add", "--detach", "-q", str(ws), task.base_commit)
    try:
        link_env(Path(task.repo), ws)
        yield ws
    finally:
        if not keep:
            with _git_lock:
                _git(task.repo, "worktree", "remove", "--force", str(ws), check=False)
                shutil.rmtree(tmp, ignore_errors=True)
                _git(task.repo, "worktree", "prune", check=False)


def env_dirs(repo: Path, depth: int = 3) -> list[str]:
    """Dependency folders anywhere in the top few levels (monorepos keep ``web/node_modules``,
    ``api/.venv``), never looking inside one, nor inside ``.git``."""
    names = {Path(r).name for r in ENV_DIRS}
    found: list[str] = [r for r in ENV_DIRS if "/" in r and (repo / r).is_dir()]

    def walk(d: Path, level: int) -> None:
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir() and not c.is_symlink())
        except OSError:
            return
        for c in children:
            if c.name in names:
                found.append(str(c.relative_to(repo)))
            elif level < depth and c.name != ".git" and not c.name.startswith("."):
                walk(c, level + 1)

    walk(repo, 1)
    return found


def link_env(repo: Path, ws: Path) -> list[str]:
    """Link the main clone's dependency folders into the worktree, where git does not track them."""
    linked = []
    for rel in env_dirs(repo):
        src, dest = repo / rel, ws / rel
        if src.is_dir() and not dest.exists() and dest.parent.is_dir():
            tracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
                                     capture_output=True).returncode == 0
            if not tracked:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(env_cache(src), target_is_directory=True)
                linked.append(rel)
    return linked


ENV_CACHE = Path(tempfile.gettempdir()) / "opentrain-envcache"


def _clone_tree(src: Path, dest: Path) -> bool:
    """Copy-on-write clone (APFS clonefile, or reflink/hardlinks on Linux): fast, and a write or
    delete in the clone never reaches the source."""
    import platform

    cmds = ([["cp", "-c", "-R", str(src), str(dest)]] if platform.system() == "Darwin"
            else [["cp", "-a", "--reflink=always", str(src), str(dest)], ["cp", "-al", str(src), str(dest)]])
    for cmd in cmds:
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            return True
        shutil.rmtree(dest, ignore_errors=True)
    return False


def env_cache(src: Path) -> Path:
    """A private clone of a dependency folder, made once and refreshed when the source changes.

    Attempts get this clone, so ``rm -rf node_modules`` or a broken install in an attempt damages
    the cache, never the user's own folder. Console scripts in a Python venv carry absolute
    shebangs to the original interpreter, so ``.venv/bin/pip install`` still installs into the
    user's venv: the agents' prompts say not to install packages. If no clone can be made the
    source is linked directly, as before.
    """
    key = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:16]
    dest = ENV_CACHE / key / src.name
    stamp = ENV_CACHE / key / ".source-mtime"
    mtime = str(src.stat().st_mtime_ns)
    with _git_lock:
        if dest.is_dir() and stamp.exists() and stamp.read_text() == mtime:
            return dest
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not _clone_tree(src, dest):
            return src
        stamp.write_text(mtime)
    return dest


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


TEST_CONFIG = re.compile(
    r"(^|/)(conftest\.py|pytest\.ini|tox\.ini|setup\.cfg|\.coveragerc|noxfile\.py|\.rspec|phpunit\.xml(\.dist)?"
    r"|(jest|vitest|playwright|karma|cypress)\.config\.[cm]?[jt]s|vitest\.workspace\.[jt]s|\.mocharc\.[a-z]+)$"
)
"""Files that decide how tests run. An attempt that edits one could make every test pass without
doing the work, so they are always put back as they stood at the reference before grading.
Known gap: test settings inside ``pyproject.toml`` or ``package.json`` are not restored, because
real solutions legitimately change those files."""


def _changed_paths(workspace: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(workspace), "status", "--porcelain", "-uall", "--no-renames"],
                         capture_output=True, text=True)
    return [line[3:].strip().strip('"') for line in out.stdout.splitlines() if len(line) > 3]


def overlay_tests(task: Task, workspace: Path) -> None:
    """Put the task's test files in place, exactly as they stood at the reference commit, and undo
    any change the attempt made to other test files or to test configuration."""
    from .bench.gitstate import TEST_FILE

    protected = [p for p in _changed_paths(workspace)
                 if (TEST_FILE.search(p) or TEST_CONFIG.search(p)) and p not in task.test_files]
    for rel in [*task.test_files, *protected]:
        dest = workspace / rel
        content = subprocess.run(["git", "-C", task.repo, "show", f"{task.reference_commit}:{rel}"],
                                 capture_output=True)
        if content.returncode != 0 or not content.stdout.strip():  # deleted, or emptied, at the reference
            if dest.is_file() or dest.is_symlink():
                dest.unlink()  # deleted (or never present) at the reference
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content.stdout)


def run_tests(task: Task, workspace: Path, timeout: float = 900) -> Verdict:
    overlay_tests(task, workspace)
    start = time.monotonic()
    env = {**os.environ, "CI": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    # A linked .venv may hold an editable install pointing at the main clone. Putting the
    # worktree's own source first makes the tests import the code under test, not the user's copy.
    own = [str(p) for p in (workspace / "src", workspace) if p.is_dir()]
    env["PYTHONPATH"] = os.pathsep.join(own + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
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
