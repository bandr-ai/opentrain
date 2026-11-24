"""The task store: the part of OpenTrain that compounds.

A task is a Harbor-format folder. For a coding task mined from a session:

    <id>/task.toml        Harbor metadata, plus an [opentrain] table (repo, commits, source session)
    <id>/instruction.md   what the person asked, in their words
    <id>/environment/     Dockerfile for Harbor; locally the environment is the repo at base_commit
    <id>/tests/test.sh    the verifier: overlay the reference's test files, run the test command
    <id>/tests/files/     those test files, as they stood at the reference commit
    <id>/solution/solve.sh + solution.patch   the reference answer (non-test changes only)
    <id>/controls.json    the verifier controls: reference passes, doing nothing fails

Training tasks live in ``.opentrain/tasks``; held-out tasks in ``.opentrain/heldout``, which only
the evaluator reads. ``Store.add`` refuses a task whose controls have not passed: a verifier that
passes an empty answer, or fails the reference, would teach the wrong thing and grade wrong.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


class TaskRejected(ValueError):
    pass


@dataclass(frozen=True)
class Controls:
    """The verifier checked against answers whose outcome is known in advance."""

    reference: bool
    """The reference answer passes."""
    noop: bool | None
    """Doing nothing passes. Must be False. None: not run, or inconclusive (the environment broke,
    so the failure proves nothing about the verifier)."""
    wrong: bool | None = None
    """A known-wrong answer passes. Must be False when checked; None when there was none to try."""
    seconds: float | None = None
    detail: str = ""

    @property
    def valid(self) -> bool:
        return self.reference and self.noop is False and self.wrong is not True

    def reason(self) -> str:
        if not self.reference:
            return "the reference answer fails its own verifier"
        if self.noop is None:
            return "doing nothing was inconclusive: " + (self.detail or "not run")
        if self.noop:
            return "doing nothing passes the verifier"
        if self.wrong:
            return "a known-wrong answer passes the verifier"
        return "controls passed"


@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    repo: str
    base_commit: str
    reference_commit: str
    test_command: str
    test_files: tuple[str, ...] = ()
    kind: str = "coding"
    family: str | None = None
    """Kind of work (for the capability map), e.g. the top-level folder touched."""
    source_session: str | None = None
    created_at: str | None = None
    """When the work happened: the split by date uses this, not when the task was mined."""
    meta: dict[str, Any] = field(default_factory=dict)
    timeout: float = 900.0
    """Seconds the verifier (and an attempt) may take, from task.toml."""

    def digest(self) -> str:
        body = json.dumps({k: v for k, v in asdict(self).items() if k not in ("meta", "timeout")}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]


def _toml_str(v: str) -> str:
    return json.dumps(v)  # a JSON string is a valid TOML basic string


def task_toml(t: Task, timeout: float | None = None) -> str:
    timeout = t.timeout if timeout is None else timeout
    lines = [
        'version = "1.0"',
        "",
        "[metadata]",
        f"difficulty = {_toml_str('unknown')}",
        f"category = {_toml_str(t.kind)}",
        f"tags = [{', '.join(_toml_str(x) for x in ['opentrain', t.kind] + ([t.family] if t.family else []))}]",
        "",
        "[verifier]",
        f"timeout_sec = {float(timeout)}",
        "",
        "[agent]",
        f"timeout_sec = {float(timeout)}",
        "",
        "[opentrain]",
        f"task_id = {_toml_str(t.task_id)}",
        f"repo = {_toml_str(t.repo)}",
        f"base_commit = {_toml_str(t.base_commit)}",
        f"reference_commit = {_toml_str(t.reference_commit)}",
        f"test_command = {_toml_str(t.test_command)}",
        f"test_files = [{', '.join(_toml_str(x) for x in t.test_files)}]",
        f"family = {_toml_str(t.family or '')}",
        f"source_session = {_toml_str(t.source_session or '')}",
        f"created_at = {_toml_str(t.created_at or '')}",
        f"digest = {_toml_str(t.digest())}",
    ]
    return "\n".join(lines) + "\n"


def read_task(folder: Path) -> Task:
    doc = tomllib.loads((folder / "task.toml").read_text())
    data = doc["opentrain"]
    meta_path = folder / "meta.json"
    return Task(
        task_id=data["task_id"],
        instruction=(folder / "instruction.md").read_text(),
        repo=data["repo"],
        base_commit=data["base_commit"],
        reference_commit=data["reference_commit"],
        test_command=data["test_command"],
        test_files=tuple(data.get("test_files", [])),
        kind=doc.get("metadata", {}).get("category", "coding"),
        timeout=float(doc.get("verifier", {}).get("timeout_sec", 900.0)),
        family=data.get("family") or None,
        source_session=data.get("source_session") or None,
        created_at=data.get("created_at") or None,
        meta=json.loads(meta_path.read_text()) if meta_path.exists() else {},
    )


class Store:
    """Training tasks. Held-out tasks are in :class:`HeldoutStore`, which the trainer never gets."""

    subdir = "tasks"

    def __init__(self, state: Path):
        self.root = state / self.subdir
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, task_id: str) -> Path:
        if "/" in task_id or task_id.startswith("."):
            raise ValueError(f"bad task id {task_id!r}")
        return self.root / task_id

    def ids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "task.toml").is_file())

    def __iter__(self) -> Iterator[Task]:
        for i in self.ids():
            yield self.get(i)

    def __contains__(self, task_id: str) -> bool:
        return (self.folder(task_id) / "task.toml").is_file()

    def get(self, task_id: str) -> Task:
        return read_task(self.folder(task_id))

    def controls(self, task_id: str) -> Controls | None:
        p = self.folder(task_id) / "controls.json"
        return Controls(**json.loads(p.read_text())) if p.exists() else None

    def add(self, folder: Path, controls: Controls) -> Task:
        """Accept a built task folder, only if its verifier passed its controls."""
        if not controls.valid:
            raise TaskRejected(controls.reason())
        task = read_task(folder)
        if self.subdir == "tasks" and task.task_id in HeldoutStore(self.root.parent):
            raise TaskRejected(f"{task.task_id} is held out; it can never become a training task")
        dest = self.folder(task.task_id)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(folder, dest)
        (dest / "controls.json").write_text(json.dumps(asdict(controls), indent=2))
        return task

    def move_to(self, task_id: str, other: "Store") -> None:
        src = self.folder(task_id)
        dest = other.folder(task_id)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))


class HeldoutStore(Store):
    subdir = "heldout"

    def __init__(self, state: Path):
        super().__init__(state)
        self.root.chmod(0o700)


def split_store(state: Path, fraction: float, cutoff: str | None = None) -> dict[str, Any]:
    """Move the latest ``fraction`` of tasks (by when the work happened) into the held-out store.

    Tasks already held out stay held out: a task the evaluator has seen is never handed to the
    trainer, even if the split moves. New tasks after the cutoff keep the held-out set fresh.
    """
    from datetime import datetime

    from .bench.split import split_by_date

    train, held = Store(state), HeldoutStore(state)
    items = []
    for t in list(train) + list(held):
        when = datetime.fromisoformat(t.created_at) if t.created_at else datetime.min.replace(tzinfo=None)
        if when.tzinfo is None:
            from datetime import timezone

            when = when.replace(tzinfo=timezone.utc)
        items.append((t.task_id, when))
    sp = split_by_date(items, fraction, datetime.fromisoformat(cutoff) if cutoff else None)
    moved = 0
    for tid in sp.heldout:
        if tid in train:
            train.move_to(tid, held)
            moved += 1
    return {"train": len(train.ids()), "heldout": len(held.ids()), "moved": moved, "cutoff": str(sp.cutoff)}
