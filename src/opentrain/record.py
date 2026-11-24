"""The run record: append-only, hash-chained, one folder per run.

Every score must be traceable to the data, recipe and checkpoint that produced it, across many
context windows and many runs. The engine reads the record and adds to it; it never rewrites it.
Each event carries the hash of the one before it, so an edit anywhere breaks ``verify()``.

    .opentrain/runs/<run_id>/events.jsonl
    .opentrain/runs/index.jsonl     one line per run opened, for listing
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64

# Event kinds, roughly in the order a run writes them (the fields of the plan's run record).
KINDS = (
    "opened",        # trigger, policy snapshot
    "data",          # data snapshot: task ids and hashes
    "recipe",        # which data and training recipe ran, and the choices made
    "spend",         # tokens or compute, dollars
    "job",           # a training job: id, backend, state
    "checkpoint",    # resulting weights
    "eval",          # pass/fail and time per held-out task
    "gate",          # promoted or not, and why
    "note",          # anything the engine wants the next run to know
    "closed",
)


class RecordError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(prev: str, body: dict[str, Any]) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + payload).encode()).hexdigest()


@dataclass
class Run:
    run_id: str
    folder: Path

    @property
    def events_path(self) -> Path:
        return self.folder / "events.jsonl"

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines() if line.strip()]

    def _last_hash(self) -> str:
        ev = self.events()
        return ev[-1]["hash"] if ev else GENESIS

    def add(self, kind: str, **data: Any) -> dict[str, Any]:
        if kind not in KINDS:
            raise RecordError(f"unknown event kind {kind!r}")
        ev = self.events()
        if ev and ev[-1]["kind"] == "closed":
            raise RecordError(f"run {self.run_id} is closed")
        body = {"seq": len(ev), "at": _now(), "kind": kind, "data": data}
        entry = {**body, "prev": ev[-1]["hash"] if ev else GENESIS}
        entry["hash"] = _hash(entry["prev"], body)
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return entry

    def verify(self) -> bool:
        prev = GENESIS
        for i, e in enumerate(self.events()):
            body = {k: e[k] for k in ("seq", "at", "kind", "data")}
            if e["seq"] != i or e["prev"] != prev or _hash(prev, body) != e["hash"]:
                return False
            prev = e["hash"]
        return True

    def last(self, kind: str) -> dict[str, Any] | None:
        for e in reversed(self.events()):
            if e["kind"] == kind:
                return e
        return None

    def spent(self) -> float:
        return sum(float(e["data"].get("usd", 0)) for e in self.events() if e["kind"] == "spend")

    @property
    def closed(self) -> bool:
        last = self.events()[-1:] or [{}]
        return last[0].get("kind") == "closed"


class RunRecord:
    def __init__(self, state: Path):
        self.root = state / "runs"
        self.root.mkdir(parents=True, exist_ok=True)

    def open(self, trigger: str, policy: dict[str, Any], **extra: Any) -> Run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{uuid.uuid4().hex[:6]}"
        run = Run(run_id, self.root / run_id)
        run.folder.mkdir(parents=True)
        run.add("opened", trigger=trigger, policy=policy, **extra)
        with (self.root / "index.jsonl").open("a") as fh:
            fh.write(json.dumps({"run_id": run_id, "at": _now(), "trigger": trigger}) + "\n")
        return run

    def get(self, run_id: str) -> Run:
        folder = self.root / run_id
        if not folder.is_dir():
            matches = [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith(run_id)]
            if len(matches) != 1:
                raise RecordError(f"no single run matches {run_id!r}")
            folder = matches[0]
        return Run(folder.name, folder)

    def runs(self) -> Iterator[Run]:
        for p in sorted((p for p in self.root.iterdir() if p.is_dir()), reverse=True):
            yield Run(p.name, p)

    def latest(self) -> Run | None:
        return next(self.runs(), None)
