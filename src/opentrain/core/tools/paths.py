"""Path rules shared by the file tools: what a path means, and what no tool may touch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..tool import ToolError

# Held-out tasks and the event log are never reachable through the general tools, whatever an
# agent's rules say. The evaluator reads held-out tasks through its own process (integrity).
SEALED = ("heldout", "opentrain.db", "opentrain.db-wal", "opentrain.db-shm", "evaluator")


def resolve(project: Any, path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = project.directory / p
    return p.resolve()


def rel(project: Any, p: Path) -> str:
    try:
        return str(p.relative_to(project.directory))
    except ValueError:
        return str(p)


def sealed(project: Any, p: Path) -> bool:
    try:
        inside = p.resolve().relative_to(project.state.resolve())
    except ValueError:
        return False
    return bool(inside.parts) and inside.parts[0] in SEALED


def guard(project: Any, p: Path) -> None:
    if sealed(project, p):
        raise ToolError(f"{rel(project, p)} is sealed: held-out tasks and the run record are only reachable "
                        "through the server's tools")
