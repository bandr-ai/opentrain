"""The ``.opentrain/`` state folder that sits beside the Trainfile.

    .opentrain/
      sessions/index.jsonl   one summary per imported session (counts and paths, no content)
      bench/                 v0.1 bench results, one JSON per run plus report.md
      tasks/                 the task store; train.json lists the training split
      heldout/               held-out task ids. Only the evaluator reads this.
      runs/                  the run record (append-only)
      models/                the model registry

The held-out folder is made private to the owner here. That is not a wall against a trainer
process running as the same user; the plan's answer to that (a separate account or container
for the evaluator) arrives with the evaluator in v0.2.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAYOUT = ("sessions", "bench", "tasks", "heldout", "runs", "models")


def state_dir(root: str | Path | None = None) -> Path:
    base = Path(root) if root else Path.cwd()
    return base / ".opentrain" if base.name != ".opentrain" else base


def ensure(root: str | Path | None = None) -> Path:
    d = state_dir(root)
    for name in LAYOUT:
        (d / name).mkdir(parents=True, exist_ok=True)
    os.chmod(d / "heldout", 0o700)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("# Session-derived data stays local until you approve an upload.\n*\n")
    return d


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)
    return path
