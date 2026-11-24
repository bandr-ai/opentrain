"""Mine tasks from sessions (recipe ``cl/mine``): from a judged session to an accepted task.

A session the yield judge passed (or could repair) becomes a Harbor task folder (layout in
:mod:`opentrain.tasks`): the person's instruction verbatim, the repo at the base commit as the
environment, the test files changed by the reference as the verifier, and the rest of the
reference change as the solution. The folder is then certified (:mod:`opentrain.certify`) and
added to the store only if its controls hold. Anything that cannot be built is reported with a
reason; nothing is accepted on trust.

The test command is the one the person's session used, cleaned so it runs from the repo root:
``cd <dir> &&`` prefixes become a ``cd <sub> &&`` relative to the root (or vanish), pipes and
redirections that would hide the exit code are cut, and environment noise is dropped. When the
session never ran tests, a command is inferred from the repo. When the command does not detect
the change (doing nothing passes, usually because the session ran tests that do not touch it),
the command is retargeted at the changed test files before the session is given up on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .bench.gitstate import TEST_FILE
from .bench.yield_ import Verdict as YieldVerdict
from .certify import run_controls
from .sessions.claude_code import TEST_COMMAND
from .tasks import Controls, HeldoutStore, Store, Task, TaskRejected, read_task, task_toml
from .workspace import reference_patch


class MineError(ValueError):
    """A session that cannot become a task. ``reason`` is the short category for reports."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


# Variables that change how output looks, not what the tests check.
_NOISE_VARS = {"CI", "FORCE_COLOR", "NO_COLOR", "TERM", "COLUMNS", "LINES", "PAGER", "GIT_PAGER",
               "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PY_COLORS"}
_TOOL_DIRS = {".venv", "venv", "env", ".tox", ".nox", "node_modules", "target"}
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(\S*)\s+")
_WRAPPER = re.compile(r"^(timeout\s+(-\S+\s+)*\S+|time|env)\s+")
_REDIRECT = re.compile(r"\s+(\d?>&\d|&>>?\s*\S+|\d?>>?\s*\S+|<\s*\S+)")
_PYTEST = re.compile(r"^(.*?(?<![\w-])(?:pytest|py\.test))(?=\s|$)")


def _git(repo: str, *args: str) -> str:
    out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise MineError("git error", out.stderr.strip()[:300])
    return out.stdout


def _split(cmd: str, seps: tuple[str, ...]) -> list[str]:
    """Split a shell command on top-level separators (outside quotes). Good enough for the
    one-liners agents run; anything stranger is left whole."""
    parts, buf, quote, i = [], [], None, 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            buf.append(c)
            if c == quote:
                quote = None
            elif c == "\\" and quote == '"' and i + 1 < len(cmd):
                buf.append(cmd[i + 1])
                i += 1
            i += 1
            continue
        if c in "'\"":
            quote = c
        sep = next((s for s in seps if cmd.startswith(s, i)), None)
        if sep and not (sep == "|" and cmd.startswith("||", i)):
            parts.append("".join(buf).strip())
            buf = []
            i += len(sep)
            continue
        buf.append(c)
        i += 1
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _root_commits(path: str) -> set[str]:
    out = subprocess.run(["git", "-C", path, "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True)
    return set(out.stdout.split()) if out.returncode == 0 else set()


def other_clone_root(here: str, repo_root: str) -> str | None:
    """The root of another clone of the same repo that ``here`` sits in, if it is one.

    People keep several clones (a main checkout, workspace worktrees, a scratch clone) and a
    session often edits one and runs the tests in another. A folder that still exists is matched
    by its root commit; one that is gone, by the repo's folder name in its path.
    """
    if os.path.isdir(here):
        out = subprocess.run(["git", "-C", here, "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        top = out.stdout.strip() if out.returncode == 0 else ""
        if top and _root_commits(top) & _root_commits(repo_root):
            return top
        return None
    name = os.path.basename(repo_root.rstrip("/"))
    parts = here.split(os.sep)
    if name in parts:
        i = len(parts) - 1 - parts[::-1].index(name)
        return os.sep.join(parts[: i + 1])
    return None


def normalize_command(raw: str, cwd: str | None, bases: list[str]) -> str | None:
    """The session's test command, made to run from the repo root.

    ``bases`` are the absolute paths that stand for the repo root (the root, its resolved path,
    and a vanished worktree's path). Returns None when no test command is found in ``raw``.
    """
    here = cwd or bases[0]
    test = None
    for seg in _split(raw, ("&&", "||", ";", "\n")):
        seg = seg.strip().lstrip("(").rstrip(")").strip()
        words = seg.split(maxsplit=1)
        if words and words[0] == "cd" and test is None:
            target = os.path.expanduser(words[1].strip("'\"")) if len(words) > 1 else "~"
            here = os.path.normpath(os.path.join(here, os.path.expanduser(target)))
        elif TEST_COMMAND.search(seg):
            test = seg
            break
    if test is None:
        return None
    test = next((p for p in _split(test, ("|",)) if TEST_COMMAND.search(p)), test)
    test = _REDIRECT.sub("", test).strip()
    kept: list[str] = []
    while True:  # leading noise: VAR=x for display-only vars, `timeout N`, `time`, `env`
        m = _ASSIGN.match(test)
        w = _WRAPPER.match(test)
        if m:
            if m.group(1) not in _NOISE_VARS:
                kept.append(m.group(0).strip())
            test = test[m.end():]
        elif w:
            test = test[w.end():]
        else:
            break
    test = " ".join([*kept, test])

    bases = list(bases)
    if not any(here == b or here.startswith(b.rstrip("/") + "/") for b in bases):
        other = other_clone_root(here, bases[0])
        if other:
            bases.append(other)
    sub = ""
    for base in bases:
        if here == base or here.startswith(base.rstrip("/") + "/"):
            sub = os.path.relpath(here, base)
            break
    sub = "" if sub == "." else sub
    def rel(m: re.Match) -> str:
        inside = (m.group(1) or "").lstrip("/")
        if inside.split("/", 1)[0] in _TOOL_DIRS:
            return m.group(0)  # an untracked toolchain (a venv): not in a fresh worktree, keep absolute
        return os.path.relpath(inside or ".", sub or ".")

    for base in sorted(bases, key=len, reverse=True):  # absolute paths into the repo -> relative
        test = re.sub(re.escape(base.rstrip("/")) + r"(/[^\s'\"]*)?", rel, test)
    return f"cd {shlex.quote(sub)} && {test}" if sub else test


def infer_command(repo: str, commit: str, test_files: tuple[str, ...] = ()) -> str | None:
    """A test command for a repo whose session never ran one, by the repo's own conventions."""
    files = set(_git(repo, "ls-tree", "-r", "--name-only", commit).splitlines())
    tests = [f for f in files if TEST_FILE.search(f)]
    exts = Counter(Path(f).suffix for f in (list(test_files) or tests))
    if exts[".py"] and exts[".py"] == max(exts.values()):
        return "python3 -m pytest -q"
    if "package.json" in files:
        pkg = json.loads(_git(repo, "show", f"{commit}:package.json") or "{}")
        if (pkg.get("scripts") or {}).get("test"):
            return "npm test"
    if "go.mod" in files:
        return "go test ./..."
    if "Cargo.toml" in files:
        return "cargo test"
    if exts[".py"]:
        return "python3 -m pytest -q"
    return None


def narrow_command(command: str, test_files: tuple[str, ...]) -> str | None:
    """Retarget ``command`` at just the changed test files, or None when it cannot be done.

    Used when doing nothing passes: the session's command ran tests that never touch the change.
    """
    sub, cmd = "", command
    m = re.match(r"^cd (\S+) && (.*)$", command)
    if m:
        sub, cmd = shlex.split(m.group(1))[0], m.group(2)
    rel = [os.path.relpath(f, sub) for f in test_files if not sub or f.startswith(sub.rstrip("/") + "/")]
    if not rel:
        return None
    quoted = " ".join(shlex.quote(f) for f in rel)
    out = None
    if (pm := _PYTEST.match(cmd)) or re.search(r"-m unittest", cmd):
        py = [f for f in rel if f.endswith(".py")]
        if py:
            head = pm.group(1) if pm else cmd.split("-m unittest")[0] + "-m pytest"
            out = f"{head} -q {' '.join(shlex.quote(f) for f in py)}"
    elif re.search(r"(?<![\w-])go test", cmd):
        pkgs = sorted({"./" + str(Path(f).parent) if str(Path(f).parent) != "." else "."
                       for f in rel if f.endswith("_test.go")})
        out = f"go test {' '.join(pkgs)}" if pkgs else None
    elif re.search(r"(?<![\w-])(jest|vitest)", cmd):
        out = f"{cmd.split(' -- ')[0]} {quoted}"
    elif re.search(r"(?<![\w-])(npm|pnpm|yarn|bun)( run)? test\b", cmd):
        out = f"{cmd.split(' -- ')[0]} -- {quoted}"
    elif re.search(r"cargo test", cmd):
        names = [Path(f).stem for f in rel if f.startswith("tests/") and f.endswith(".rs")]
        out = "cargo test " + " ".join(f"--test {shlex.quote(n)}" for n in names) if names else None
    if out is None:
        return None
    return f"cd {shlex.quote(sub)} && {out}" if sub else out


def _bases(v: YieldVerdict) -> list[str]:
    root = v.repo.root or ""
    bases = [root, os.path.realpath(root)]
    if v.repo.resolved_from:
        bases.append(v.repo.resolved_from)
    return list(dict.fromkeys(b for b in bases if b))


def _command(v: YieldVerdict, test_files: tuple[str, ...]) -> tuple[str, str]:
    """(command, where it came from): the last passing run, else the last run, else inferred."""
    runs = sorted(v.session.test_runs, key=lambda r: r.at)
    for r in [x for x in runs if x.outcome == "pass"][::-1] + runs[::-1]:
        cmd = normalize_command(r.command, r.cwd or v.session.cwd, _bases(v))
        if cmd:
            return cmd, "session" if r.outcome == "pass" else f"session ({r.outcome} run)"
    cmd = infer_command(v.repo.root or "", v.repo.base_commit or "", test_files)
    if cmd is None:
        raise MineError("no test command", "the session ran none and none can be inferred from the repo")
    return cmd, "inferred"


MAX_PROMPTS = 12


def compose_instruction(prompts: list[str], first: str) -> tuple[str, str]:
    """The task as the person asked for it, over the whole session.

    A first prompt is often an opening ("look into this, it seems broken"), and the spec arrives in
    later turns. The verifier checks the finished work, so a model handed only the opening could not
    know what to build. Every prompt the person typed goes in, in order and verbatim. Nothing comes
    from the agent's replies or the diff, so the instruction cannot leak the reference answer.
    """
    asks = [p for p in prompts if p.strip()] or [first]
    if len(asks) == 1:
        return asks[0], "first prompt"
    if len(asks) > MAX_PROMPTS:
        asks = asks[: MAX_PROMPTS // 2] + asks[-MAX_PROMPTS // 2:]
    body = "\n\n".join(f"{i}. {a.strip()}" for i, a in enumerate(asks, 1))
    head = "Over one working session the user asked for the following, in this order. Do all of it.\n\n"
    return head + body, f"{len(asks)} prompts"


def task_id(v: YieldVerdict) -> str:
    """Stable across re-mining: repo name, short session id, digest of what the task is built from."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(v.repo.root or "repo").name).strip("-.") or "repo"
    sid = re.sub(r"[^A-Za-z0-9]", "", v.session.session_id)[:8] or "session"
    key = f"{v.repo.root}|{v.session.session_id}|{v.repo.base_commit}|{v.repo.reference_commit}"
    return f"{name}-{sid}-{hashlib.sha256(key.encode()).hexdigest()[:6]}"


def _family(changed: list[str]) -> str | None:
    tops = Counter(f.split("/", 1)[0] if "/" in f else "root" for f in changed)
    return tops.most_common(1)[0][0] if tops else None


_DOCKER_BASE = {"python3 -m pytest": "python:3.12-slim", "pytest": "python:3.12-slim",
                "npm": "node:20-slim", "go test": "golang:1.22", "cargo": "rust:1"}


def _dockerfile(task: Task) -> str:
    image = next((img for k, img in _DOCKER_BASE.items() if k in task.test_command), "ubuntu:24.04")
    return f"""\
# Environment for Harbor: the repo exactly as it stood at the base commit. Nothing is cloned.
# Build context: environment/repo/ must hold that snapshot, e.g.
#   mkdir -p environment/repo && git -C <repo> archive {task.base_commit} | tar -x -C environment/repo
# Locally OpenTrain does not build this image: the environment is a git worktree of the repo at
# the base commit (opentrain.workspace.checkout), using the machine's own toolchain.
FROM {image}
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY repo/ /app/
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi; \\
    if [ -f pyproject.toml ]; then pip install -e . pytest; fi; \\
    if [ -f package.json ]; then npm install; fi
"""


def _test_sh(task: Task, deleted: list[str]) -> str:
    removes = "".join(f"rm -f {shlex.quote(f)}\n" for f in deleted)
    return f"""\
#!/usr/bin/env bash
# Verifier: put the test files in place as they stood at the reference commit, run the test
# command, and score by its exit code alone.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${{APP_DIR:-/app}}"
if [ -d "$HERE/files" ]; then cp -R "$HERE/files/." .; fi
{removes}mkdir -p /logs/verifier
if ( {task.test_command} ); then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


_SOLVE_SH = """\
#!/usr/bin/env bash
# Reference solution: the recorded change, test files left out (the verifier brings its own).
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "${APP_DIR:-/app}"
git apply --whitespace=nowarn "$HERE/solution.patch"
"""


def mine(verdict: YieldVerdict, out_dir: Path, test_command: str | None = None) -> Path:
    """Build the task folder for one judged session under ``out_dir``; return the folder.

    Raises :class:`MineError` when the session cannot become a task. ``test_command`` overrides
    the command taken from the session (the miner passes a narrowed one).
    """
    s, rs = verdict.session, verdict.repo
    if not s.instruction:
        raise MineError("no instruction")
    if not rs.root or not rs.base_commit:
        raise MineError("no repo state", verdict.detail or "")
    if not Path(rs.root).is_dir():
        raise MineError("no repo state", f"repo {rs.root} is not on this machine")
    if not rs.reference_commit:
        raise MineError("no reference", "the edits were never committed")
    changed = _git(rs.root, "diff", "--name-only", "--no-renames", rs.base_commit, rs.reference_commit).splitlines()
    test_files = tuple(f for f in changed if TEST_FILE.search(f))
    command, source = (test_command, "narrowed") if test_command else _command(verdict, test_files)
    task = Task(
        task_id=task_id(verdict),
        instruction=compose_instruction(s.prompts, s.instruction)[0],
        repo=rs.root,
        base_commit=rs.base_commit,
        reference_commit=rs.reference_commit,
        test_command=command,
        test_files=test_files,
        family=_family([f for f in changed if f not in test_files]),
        source_session=s.session_id,
        created_at=s.started_at.isoformat(),
    )
    patch = reference_patch(task)
    if not patch.strip():
        raise MineError("patch empty", "the reference changes only test files")

    folder = Path(out_dir) / task.task_id
    if folder.exists():
        shutil.rmtree(folder)
    for d in ("environment", "tests/files", "solution"):
        (folder / d).mkdir(parents=True)
    (folder / "task.toml").write_text(task_toml(task))
    (folder / "instruction.md").write_text(compose_instruction(s.prompts, s.instruction)[0])
    (folder / "environment" / "Dockerfile").write_text(_dockerfile(task))
    deleted = []
    for rel in test_files:
        blob = subprocess.run(["git", "-C", rs.root, "show", f"{rs.reference_commit}:{rel}"], capture_output=True)
        if blob.returncode != 0:
            deleted.append(rel)
            continue
        dest = folder / "tests" / "files" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob.stdout)
    (folder / "tests" / "test.sh").write_text(_test_sh(task, deleted))
    (folder / "solution" / "solution.patch").write_text(patch)
    (folder / "solution" / "solve.sh").write_text(_SOLVE_SH)
    for sh in (folder / "tests" / "test.sh", folder / "solution" / "solve.sh"):
        sh.chmod(0o755)
    meta = {"test_command_source": source, "instruction_source": compose_instruction(s.prompts, s.instruction)[1],
            "session_path": s.path, "agent": s.agent,
            "yield": "verified" if verdict.verified else f"repairable ({verdict.skip})"}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return folder


def _patch(folder: Path) -> str:
    """Certify the exact bytes the task ships, not a fresh diff."""
    return (folder / "solution" / "solution.patch").read_text()


def _rejection(c: Controls) -> tuple[str, str]:
    if c.detail.startswith("environment"):
        return "environment", c.detail
    if not c.reference:
        return "controls failed: reference fails", c.detail
    if c.noop:
        return "controls failed: noop passes", c.detail
    return "controls failed: wrong passes", c.detail


def mine_one(v: YieldVerdict, state: Path, timeout: float = 900) -> dict[str, Any]:
    """Build, certify (narrowing once if doing nothing passes) and store one session's task."""
    tid = task_id(v)
    if tid in HeldoutStore(state) or tid in Store(state):
        return {"outcome": "kept", "task_id": tid, "reason": "already in the store"}
    with tempfile.TemporaryDirectory(prefix="ot-mine-") as tmp:
        folder = mine(v, Path(tmp))
        task = read_task(folder)
        controls, runs = run_controls(task, _patch(folder), timeout)
        narrowed_from = None
        noop_passed = controls.reference and runs.get("noop") and runs["noop"].kind == "pass"
        # A reference that fails on tests may only be failing tests the change never touched
        # (a suite needing a database, a known-flaky test). The changed test files alone may hold.
        ref_failed = not controls.reference and runs.get("reference") and runs["reference"].kind == "tests"
        if noop_passed or ref_failed:
            narrowed = narrow_command(task.test_command, task.test_files)
            if narrowed and narrowed != task.test_command:
                folder = mine(v, Path(tmp), test_command=narrowed)
                meta = json.loads((folder / "meta.json").read_text())
                meta["narrowed_from"] = narrowed_from = task.test_command
                (folder / "meta.json").write_text(json.dumps(meta, indent=2))
                controls = run_controls(read_task(folder), _patch(folder), timeout)[0]
        row: dict[str, Any] = {"task_id": tid, "controls": controls.detail, "seconds": controls.seconds}
        if narrowed_from:
            row["narrowed_from"] = narrowed_from
        try:
            Store(state).add(folder, controls)
        except TaskRejected:
            reason, detail = _rejection(controls)
            return {**row, "outcome": "rejected", "reason": reason, "detail": detail}
        return {**row, "outcome": "accepted", "reason": "controls passed"}


def mine_all(verdicts: list[YieldVerdict], state: Path, include_repairable: bool = False,
             timeout: float = 900) -> dict[str, Any]:
    """Mine every eligible session into the store; one row per session with its outcome and reason.

    Sessions the yield judge skipped are recorded with its reason. A session that errors is
    recorded as an error and the rest carry on: one odd repo must not cost the whole batch.
    """
    rows: list[dict[str, Any]] = []
    for v in verdicts:
        base = {"session_id": v.session.session_id}
        if not (v.verified or (include_repairable and v.repairable)):
            rows.append({**base, "outcome": "skipped", "reason": v.skip, "detail": v.detail})
            continue
        try:
            rows.append({**base, **mine_one(v, state, timeout)})
        except MineError as exc:
            rows.append({**base, "outcome": "rejected", "reason": exc.reason, "detail": exc.detail})
        except Exception as exc:  # report, never crash the batch
            rows.append({**base, "outcome": "rejected", "reason": "error",
                         "detail": f"{type(exc).__name__}: {exc}"[:500]})
    outcomes = Counter(r["outcome"] for r in rows)
    return {
        "sessions": len(rows),
        "accepted": outcomes["accepted"],
        "kept": outcomes["kept"],
        "rejected": outcomes["rejected"],
        "skipped": outcomes["skipped"],
        "reasons": dict(Counter(r["reason"] for r in rows if r["outcome"] in ("rejected", "skipped")).most_common()),
        "rows": rows,
    }
