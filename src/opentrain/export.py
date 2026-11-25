"""``opentrain export``: the task store and training data in open formats, for any other tool.

    <out>/tasks/<id>/      each accepted training task, a Harbor folder copied as stored
    <out>/train.jsonl      the training data, if it has been built (.opentrain/data/train.jsonl)
    <out>/evals.jsonl      held-out tasks (id, instruction, test command): only with include_heldout
    <out>/heldout/<id>/    their Harbor folders: only with include_heldout
    <out>/manifest.json    what is in the export, with digests, counts and the opentrain version

Held-out tasks are left out by default. The held-out set is only a fair test while no model has
trained on it, and an export is made to leave this machine and be fed to other tools, which
may well train on whatever they are given. Include it only for an evaluator you control.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .tasks import HeldoutStore, Store

_OURS = ("tasks", "heldout", "train.jsonl", "evals.jsonl", "manifest.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _folder_digest(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(folder)).encode() + b"\0" + _sha256(p).encode() + b"\n")
    return h.hexdigest()


def _prepare(out: Path) -> None:
    """Refuse to write into a folder that holds anything but an earlier export."""
    if out.exists() and any(out.iterdir()):
        if not (out / "manifest.json").is_file():
            raise ValueError(f"{out} is not empty and is not an opentrain export; choose an empty folder")
        for name in _OURS:
            p = out / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)


def export(state: Path, out: Path, include_heldout: bool = False) -> dict[str, Any]:
    """Write the export under ``out`` and return its manifest. ``state`` is the .opentrain folder."""
    out = Path(out)
    _prepare(out)
    train, held = Store(state), HeldoutStore(state)

    def copy(store: Store, dest: str) -> list[dict[str, Any]]:
        rows = []
        for tid in store.ids():
            target = out / dest / tid
            shutil.copytree(store.folder(tid), target)
            t = store.get(tid)
            rows.append({"task_id": tid, "digest": t.digest(), "folder_sha256": _folder_digest(target),
                         "family": t.family, "created_at": t.created_at})
        return rows

    manifest: dict[str, Any] = {
        "opentrain": __version__,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "include_heldout": include_heldout,
        "tasks": copy(train, "tasks"),
        "files": {},
    }
    counts = {"tasks": len(manifest["tasks"]), "heldout_withheld": 0, "heldout_exported": 0, "train_records": 0}

    if include_heldout:
        manifest["heldout"] = copy(held, "heldout")
        with (out / "evals.jsonl").open("w") as fh:
            for t in held:
                fh.write(json.dumps({"task_id": t.task_id, "instruction": t.instruction,
                                     "test_command": t.test_command}) + "\n")
        counts["heldout_exported"] = len(manifest["heldout"])
    else:
        counts["heldout_withheld"] = len(held.ids())

    src = state / "data" / "train.jsonl"
    mpath = state / "data" / "manifest.json"
    built_on = {t["task_id"] for t in json.loads(mpath.read_text()).get("tasks", [])} if mpath.is_file() else None
    # A set built before the split moved, or with no manifest to check, may contain held-out work.
    clean = built_on is not None and not built_on & set(HeldoutStore(state).ids())
    if src.is_file() and not clean:
        counts["train_skipped"] = "stale: rebuild with `opentrain data build`"
    if src.is_file() and clean:
        shutil.copyfile(src, out / "train.jsonl")
        with (out / "train.jsonl").open("rb") as fh:
            counts["train_records"] = sum(1 for line in fh if line.strip())

    for name in ("train.jsonl", "evals.jsonl"):
        if (out / name).is_file():
            manifest["files"][name] = _sha256(out / name)
    manifest["counts"] = counts
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
