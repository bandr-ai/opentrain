"""Integrity checks the gate runs before a version can go live.

Training agents left to themselves have trained on the test set, swapped in someone else's
checkpoint and used keys they found (PostTrainBench). The held-out store, the sandbox and the
eval cap make those hard; this module checks, from the checkpoint backwards, that none of them
happened anyway. A flagged version is rejected and scored as the baseline, never promoted.

Checks:

- **Enough held-out tasks** to decide at all (``MIN_HELDOUT``, default 5).
- **Records intact**: the run record's hash chain and the project's event log verify.
- **Base model**: the version's base model is the Trainfile's.
- **No held-out task in the training set**, by id and by digest.
- **No contamination**: the training text does not reproduce a held-out task's instruction or
  reference solution (13-gram overlap of 50% or more, as in Athanor's decontamination).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

N = 13
OVERLAP = 0.5


def min_heldout() -> int:
    return int(os.environ.get("OPENTRAIN_MIN_HELDOUT", "5"))


_WORD = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]|[^\W\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]+")


def _words(text: str) -> list[str]:
    """Words in any script; Chinese and Japanese characters count one each (no spaces to split on)."""
    return _WORD.findall(text.lower())


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def contaminated(state: Path, training_text: str) -> list[dict[str, Any]]:
    """Held-out tasks whose instruction or reference solution appears in the training text."""
    from ..tasks import HeldoutStore

    train = ngrams(training_text)
    out = []
    if not train:
        return out
    held = HeldoutStore(state)
    for tid in held.ids():
        folder = held.folder(tid)
        for label, f in (("instruction", folder / "instruction.md"), ("solution", folder / "solution" / "solution.patch")):
            if not f.exists():
                continue
            grams = ngrams(f.read_text(errors="replace"))
            if len(grams) < 3:
                continue
            share = len(grams & train) / len(grams)
            if share >= OVERLAP:
                out.append({"task_id": tid, "what": label, "overlap": round(share, 2)})
    return out


def provenance(project_state: Path, version: str, base_model: str, event_store: Any = None) -> list[str]:
    """Reasons ``version`` must not be promoted; empty when it is clean."""
    from ..record import RunRecord
    from ..registry import Registry
    from ..tasks import HeldoutStore

    flags: list[str] = []
    reg = Registry(project_state)
    v = reg.get(version)
    held_ids = set(HeldoutStore(project_state).ids())
    if len(held_ids) < min_heldout():
        flags.append(f"only {len(held_ids)} held-out tasks; at least {min_heldout()} are needed to decide")
    if v.base_model != base_model:
        flags.append(f"base model {v.base_model!r} is not the Trainfile's {base_model!r}")
    if v.run_id:
        try:
            run = RunRecord(project_state).get(v.run_id)
            if not run.verify():
                flags.append(f"run record {v.run_id} fails its hash chain")
        except Exception as exc:
            flags.append(f"run record {v.run_id} unreadable: {exc}")
    if event_store is not None:
        ok, where = event_store.verify()
        if not ok:
            flags.append(f"event log tampered: {where}")
    manifest = project_state / "data" / "manifest.json"
    data = project_state / "data" / "train.jsonl"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        used = {t["task_id"] for t in m.get("tasks", [])}
        leaked = sorted(used & held_ids)
        if leaked:
            flags.append(f"held-out tasks in the training set: {', '.join(leaked[:5])}")
        held_digests = {HeldoutStore(project_state).get(t).digest() for t in held_ids}
        dup = [t["task_id"] for t in m.get("tasks", []) if t.get("task_digest") in held_digests]
        if dup:
            flags.append(f"training tasks identical to held-out tasks: {', '.join(dup[:5])}")
    if data.exists():
        hits = contaminated(project_state, data.read_text(errors="replace"))
        if hits:
            ids = ", ".join(f"{h['task_id']} ({h['what']} {int(h['overlap'] * 100)}%)" for h in hits[:5])
            flags.append(f"training data reproduces held-out tasks: {ids}")
    return flags
