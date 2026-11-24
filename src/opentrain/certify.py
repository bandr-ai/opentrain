"""Verifier controls (recipe ``bandits/certify``): prove a task's verifier before anyone trusts it.

A verifier is only worth something if it tells right from wrong. Each task is run three ways
against answers whose outcome is known in advance:

    reference   the recorded answer                          must pass
    noop        no change at all                             must fail
    wrong       the reference with its biggest part withheld must fail (when there is one to try)

"Fail" has to mean *the tests failed*. A command that cannot start (interpreter or module
missing, command not found, nothing collected, timeout) fails for every answer, so it says
nothing about the answer. Such a run is classed ``environment`` and never counts as a good
noop or wrong failure; a reference that fails that way is reported with a detail starting
``environment:``, so the miner can say the machine, not the task, is the problem.

The known-wrong answer is chosen mechanically, from the reference patch alone:

* if the patch touches more than one file, withhold the largest code file's change (docs and
  lockfiles are never chosen: dropping them should not change a test outcome);
* else, if that one file has several hunks, withhold its largest hunk;
* else there is no answer distinct from the noop, and ``wrong`` stays None.

If withholding the biggest part of the answer still passes, the tests do not check the answer.

``lint`` reruns the controls on every stored task. Environments drift (a dependency upgrade, a
deleted clone), so the evaluator reruns it before every evaluation rather than trusting
controls recorded weeks ago.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import Controls, HeldoutStore, Store, Task, read_task
from .workspace import Verdict, reference_patch, verify

DOC_FILE = re.compile(r"(^|/)(docs?|doc)/|\.(md|rst|txt|adoc)$|(^|/)(CHANGELOG|LICENSE|README)[^/]*$", re.I)
LOCK_FILE = re.compile(r"(\.lock|-lock\.(json|yaml)|\.lockb)$|(^|/)go\.sum$")

_NO_MODULE = re.compile(r"(?:ModuleNotFoundError|ImportError): No module named '([\w.]+)'")
_NODE_MODULE = re.compile(r"Cannot find module '([^']+)'")
_ENV_PATTERNS = (
    # Only messages from the shell or the test runner itself: a test may legitimately fail with
    # "No such file or directory" or "Permission denied" from the code under test.
    (re.compile(r"command not found|^(/bin/)?(ba|z|da)?sh: .*(No such file or directory|not found)$"
                r"|can't open file .*No such file or directory", re.M), "command not found"),
    (re.compile(r"npm (ERR!|error) Missing script"), "npm script missing"),
    (re.compile(r"ERROR: file or directory not found|ERROR: usage:"), "test command usage error"),
    (re.compile(r"\bno tests ran\b"), "no tests ran"),
    (re.compile(r"go: cannot find main module|go: go\.mod file not found|no required module provides"),
     "go module missing"),
    (re.compile(r"error: no such (sub)?command|could not find `Cargo\.toml`"), "cargo command missing"),
)


@dataclass(frozen=True)
class Outcome:
    """What one control run showed."""

    kind: str
    """``pass``, ``tests`` (the tests ran and failed), ``environment`` or ``patch`` (did not apply)."""
    why: str
    seconds: float


def _repo_names(task: Task) -> set[str]:
    """Every path component (``.py`` dropped) in the reference tree: what the repo itself provides."""
    out = subprocess.run(["git", "-C", task.repo, "ls-tree", "-r", "--name-only", task.reference_commit],
                         capture_output=True, text=True)
    names: set[str] = set()
    for line in out.stdout.splitlines():
        for part in line.split("/"):
            names.add(part.removesuffix(".py"))
    return names


def classify(v: Verdict, repo_names: set[str]) -> Outcome:
    """Tell "the tests failed" from "the tests could not run".

    A missing module counts as the environment's fault only when the repo does not provide it:
    a test importing a module the reference adds fails at base for the right reason.
    """
    if v.passed:
        return Outcome("pass", "passed", v.seconds)
    if v.timed_out:
        return Outcome("environment", "timed out", v.seconds)
    if v.exit_code is None:
        return Outcome("patch", v.output.strip().splitlines()[0] if v.output.strip() else "patch failed", v.seconds)
    if v.exit_code in (126, 127):
        return Outcome("environment", f"exit {v.exit_code}: command not found or not executable", v.seconds)
    for m in _NO_MODULE.finditer(v.output):
        if m.group(1).split(".")[0] not in repo_names:
            return Outcome("environment", f"missing module {m.group(1)!r}", v.seconds)
    for m in _NODE_MODULE.finditer(v.output):
        if not m.group(1).startswith("."):
            return Outcome("environment", f"missing node module {m.group(1)!r}", v.seconds)
    for pattern, why in _ENV_PATTERNS:
        if pattern.search(v.output):
            return Outcome("environment", why, v.seconds)
    return Outcome("tests", f"tests failed (exit {v.exit_code})", v.seconds)


@dataclass
class _Section:
    path: str
    header: list[str]
    hunks: list[list[str]]
    binary: bool

    def text(self, drop: int | None = None) -> str:
        lines = self.header + [ln for i, h in enumerate(self.hunks) if i != drop for ln in h]
        return "".join(lines)

    def size(self, hunk: list[str] | None = None) -> int:
        lines = hunk if hunk is not None else [ln for h in self.hunks for ln in h]
        return sum(1 for ln in lines if ln[:1] in "+-" and not ln.startswith(("+++", "---")))


def _sections(patch: str) -> list[_Section]:
    out: list[_Section] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            path = line.rstrip("\n").split(" b/", 1)[-1]
            out.append(_Section(path, [line], [], False))
        elif not out:
            continue
        elif line.startswith("@@"):
            out[-1].hunks.append([line])
        elif out[-1].hunks:
            out[-1].hunks[-1].append(line)
        else:
            out[-1].header.append(line)
            if line.startswith(("GIT binary patch", "Binary files")):
                out[-1].binary = True
    return out


def known_wrong(patch: str) -> tuple[str, str] | None:
    """A partial answer that should fail, and what was withheld; None when there is none distinct
    from doing nothing. See the module docstring for the rule."""
    secs = _sections(patch)
    code = [s for s in secs if not s.binary and s.hunks and not DOC_FILE.search(s.path)
            and not LOCK_FILE.search(s.path)]
    if not code:
        return None
    biggest = max(code, key=lambda s: s.size())
    if len(secs) > 1:
        return "".join(s.text() for s in secs if s is not biggest), f"withheld {biggest.path}"
    if len(biggest.hunks) > 1:
        i = max(range(len(biggest.hunks)), key=lambda k: biggest.size(biggest.hunks[k]))
        return biggest.text(drop=i), f"withheld hunk {i + 1} of {biggest.path}"
    return None


def run_controls(task: Task, patch: str | None = None, timeout: float = 900) -> tuple[Controls, dict[str, Outcome]]:
    """Run the controls; return them with each run's outcome (the miner uses the outcomes to
    decide whether narrowing the test command is worth a try)."""
    start = time.monotonic()
    patch = reference_patch(task) if patch is None else patch
    names = _repo_names(task)
    runs: dict[str, Outcome] = {}

    runs["reference"] = ref = classify(verify(task, patch, timeout), names)
    if ref.kind != "pass":
        prefix = "environment: " if ref.kind == "environment" else ""
        detail = f"{prefix}reference {ref.why}; noop and wrong not run"
        return Controls(False, None, None, round(time.monotonic() - start, 2), detail), runs

    runs["noop"] = noop = classify(verify(task, "", timeout), names)
    parts = [f"reference passed ({ref.seconds}s)"]
    if noop.kind == "environment":
        # Inconclusive: a failure that every answer would get proves nothing about the verifier.
        detail = f"environment: noop {noop.why} while the reference passed; wrong not run"
        return Controls(True, None, None, round(time.monotonic() - start, 2), detail), runs
    parts.append(f"noop {noop.why} ({noop.seconds}s)")
    if noop.kind == "pass":
        return Controls(True, True, None, round(time.monotonic() - start, 2), "; ".join(parts)), runs

    wrong_flag: bool | None = None
    alt = known_wrong(patch)
    if alt is None:
        parts.append("wrong: none distinct from noop")
    else:
        runs["wrong"] = w = classify(verify(task, alt[0], timeout), names)
        if w.kind in ("pass", "tests"):
            wrong_flag = w.kind == "pass"
            parts.append(f"wrong ({alt[1]}) {w.why} ({w.seconds}s)")
        else:
            parts.append(f"wrong ({alt[1]}) inconclusive: {w.why}")
    return Controls(True, False, wrong_flag, round(time.monotonic() - start, 2), "; ".join(parts)), runs


def _load(task_or_folder: Task | Path | str) -> tuple[Task, str | None]:
    if isinstance(task_or_folder, Task):
        return task_or_folder, None
    folder = Path(task_or_folder)
    sol = folder / "solution" / "solution.patch"
    return read_task(folder), sol.read_text() if sol.is_file() else None


def certify(task_or_folder: Task | Path | str, timeout: float = 900) -> Controls:
    """Run the verifier controls on a task (or a task folder, using the exact patch it ships)."""
    task, patch = _load(task_or_folder)
    return run_controls(task, patch, timeout)[0]


def lint(state: Path, timeout: float = 900, *, only_heldout: bool = False) -> dict[str, Any]:
    """Re-run the controls on every stored task and list the verifiers that no longer hold.

    Nothing is moved or rewritten: a broken verifier is reported, and fixing it (or dropping the
    task) is a decision. For held-out tasks only the id and the reason are reported, never test
    output, so the report is safe to show the trainer.
    """
    start = time.monotonic()
    rows: list[dict[str, Any]] = []
    stores = (HeldoutStore(state),) if only_heldout else (Store(state), HeldoutStore(state))
    for store in stores:
        for tid in store.ids():
            try:
                c = certify(store.folder(tid), timeout)
                ok, reason, detail = c.valid, c.reason(), c.detail
                if detail.startswith("environment"):
                    reason = "environment"
            except Exception as exc:  # a broken task must not stop the lint of the rest
                ok, reason, detail = False, "error", f"{type(exc).__name__}: {exc}"
            row = {"task_id": tid, "store": store.subdir, "ok": ok, "reason": reason}
            if store.subdir != HeldoutStore.subdir:
                row["detail"] = detail
            rows.append(row)
    broken = [r for r in rows if not r["ok"]]
    return {"checked": len(rows), "ok": len(rows) - len(broken), "broken": broken,
            "seconds": round(time.monotonic() - start, 2)}
