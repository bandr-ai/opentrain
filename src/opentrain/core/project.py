"""A project: one directory, its ``.opentrain/`` state, and the services the server runs for it.

The server holds one Project per directory a client has asked about (opencode's "location").
Everything a request touches is reached through it, so two repos never share a session, a
budget or a held-out set.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .store import Store


def _now() -> float:
    return time.time()


class Project:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.state = self.directory / ".opentrain"
        self.store = Store(self.state, location=str(self.directory))
        self.services: dict[str, Any] = {}  # filled by the server: runner, permissions, jobs, ...

    @property
    def name(self) -> str:
        return self.directory.name

    def info(self) -> dict[str, Any]:
        from .. import trainfile

        tf_path = trainfile.find(self.directory)
        out: dict[str, Any] = {"directory": str(self.directory), "name": self.name, "state": str(self.state),
                               "trainfile": str(tf_path) if tf_path else None,
                               "git": (self.directory / ".git").exists()}
        try:
            tf = trainfile.load(tf_path)
            out["policy"] = {"goal": tf.goal, "budget": tf.budget.describe(), "trainer": tf.trainer,
                             "base_model": tf.base_model, "runtime": tf.runtime, "heldout": tf.heldout_fraction,
                             "eval_cap": tf.eval_cap}
        except Exception as exc:  # a broken Trainfile is shown, not fatal
            out["policy_error"] = str(exc)
        return out

    # ------------------------------------------------------------------ sessions

    def create_session(self, *, title: str | None = None, agent: str = "train", model: dict | None = None,
                       parent_id: str | None = None, metadata: dict | None = None) -> dict[str, Any]:
        now = _now()
        body = {"title": title or "New session", "agent": agent, "model": model, "parent_id": parent_id,
                "directory": str(self.directory), "status": "idle", "time": {"created": now, "updated": now},
                "cost": 0.0, "tokens": {"input": 0, "output": 0}, "metadata": metadata or {}}
        return self.store.create("session", {k: v for k, v in body.items() if v is not None})

    def sessions(self, *, include_children: bool = False) -> list[dict[str, Any]]:
        rows = self.store.find("session", newest_first=True)
        return rows if include_children else [s for s in rows if not s.get("parent_id")]

    def session(self, sid: str) -> dict[str, Any]:
        s = self.store.get("session", sid)
        if s is None:
            raise KeyError(f"no session {sid}")
        return s

    def messages(self, sid: str) -> list[dict[str, Any]]:
        """Messages in order, each with its parts under ``parts``."""
        msgs = self.store.find("message", session_id=sid)
        parts: dict[str, list[dict]] = {}
        for p in self.store.find("part", session_id=sid):
            parts.setdefault(p["message_id"], []).append(p)
        return [{**m, "parts": parts.get(m["id"], [])} for m in msgs]


def default_directory() -> Path:
    return Path(os.environ.get("OPENTRAIN_DIRECTORY") or os.getcwd()).resolve()
