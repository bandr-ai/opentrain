"""Event-sourced state for one project: every change is an event; tables are projections of them.

``<project>/.opentrain/opentrain.db`` (SQLite, WAL) holds:

- ``event``: the append-only log. Each row is hash-chained to the one before it, so any edit to
  history is detectable (``verify``). This is the run record for everything the harness does.
- ``object``: the projection. Sessions, messages, parts, inbox items, jobs, todos and saved
  approvals are rows keyed by ``(kind, id)`` with their JSON body.

Events named ``<kind>.created`` / ``<kind>.updated`` / ``<kind>.deleted`` are projected
generically: create inserts the body, update deep-merges into it, delete removes it. Every other
durable event is log-only. Ephemeral events (``durable=False``: text deltas, status) are only
broadcast, never stored.

Writes come from the server's event loop and from tool threads, so one connection is shared under
a lock. Subscribers are told after the transaction commits.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from . import ids

GENESIS = "0" * 64
PROJECTED = ("session", "message", "part", "inbox", "job", "todo", "approval", "question", "permission")

SCHEMA = """
CREATE TABLE IF NOT EXISTS event (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  id TEXT UNIQUE NOT NULL,
  type TEXT NOT NULL,
  aggregate TEXT,
  time REAL NOT NULL,
  data TEXT NOT NULL,
  prev TEXT NOT NULL,
  hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS event_aggregate ON event(aggregate, seq);
CREATE TABLE IF NOT EXISTS object (
  kind TEXT NOT NULL,
  id TEXT NOT NULL,
  session_id TEXT,
  parent_id TEXT,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  data TEXT NOT NULL,
  PRIMARY KEY (kind, id)
);
CREATE INDEX IF NOT EXISTS object_session ON object(kind, session_id, created);
CREATE INDEX IF NOT EXISTS object_parent ON object(kind, parent_id, created);
"""

Listener = Callable[[dict[str, Any]], None]


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _chain(prev: str, body: dict[str, Any]) -> str:
    return hashlib.sha256((prev + _canon(body)).encode()).hexdigest()


def merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Deep merge; a ``None`` in the patch deletes the key."""
    out = dict(base)
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


class StoreError(RuntimeError):
    pass


class Store:
    """The state of one project. ``location`` is the project directory, used to tag events."""

    def __init__(self, state: Path, location: str | None = None):
        self.state = Path(state)
        self.state.mkdir(parents=True, exist_ok=True)
        self.location = location or str(self.state.parent)
        self.path = self.state / "opentrain.db"
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(SCHEMA)
        self._listeners: list[Listener] = []

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ publish

    def subscribe(self, fn: Listener) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    def _notify(self, event: dict[str, Any]) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # a broken subscriber never breaks a write
                pass

    def publish(self, type: str, data: dict[str, Any], *, aggregate: str | None = None,
                durable: bool = True) -> dict[str, Any]:
        """Record (if durable), project, then broadcast one event. Returns the event."""
        event = {"id": ids.new("event"), "type": type, "aggregate": aggregate or data.get("session_id") or data.get("id"),
                 "time": time.time(), "data": data, "location": self.location, "durable": durable}
        if durable:
            with self._lock:
                cur = self._db.cursor()
                cur.execute("BEGIN IMMEDIATE")
                try:
                    row = cur.execute("SELECT hash FROM event ORDER BY seq DESC LIMIT 1").fetchone()
                    prev = row["hash"] if row else GENESIS
                    body = {k: event[k] for k in ("id", "type", "aggregate", "time", "data")}
                    h = _chain(prev, body)
                    cur.execute("INSERT INTO event (id, type, aggregate, time, data, prev, hash) VALUES (?,?,?,?,?,?,?)",
                                (event["id"], type, event["aggregate"], event["time"], _canon(data), prev, h))
                    event["seq"] = cur.lastrowid
                    self._project(cur, type, data, event["time"])
                    cur.execute("COMMIT")
                except Exception:
                    cur.execute("ROLLBACK")
                    raise
        self._notify(event)
        return event

    def _project(self, cur: sqlite3.Cursor, type: str, data: dict[str, Any], now: float) -> None:
        kind, _, verb = type.partition(".")
        if kind not in PROJECTED or verb not in ("created", "updated", "deleted"):
            return
        oid = data.get("id")
        if not oid:
            raise StoreError(f"{type} needs an id")
        if verb == "created":
            cur.execute("INSERT OR REPLACE INTO object (kind, id, session_id, parent_id, created, updated, data) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (kind, oid, data.get("session_id"), data.get("parent_id") or data.get("message_id"),
                         now, now, _canon(data)))
        elif verb == "updated":
            row = cur.execute("SELECT data FROM object WHERE kind=? AND id=?", (kind, oid)).fetchone()
            if row is None:
                raise StoreError(f"{type}: no {kind} {oid}")
            body = merge(json.loads(row["data"]), data)
            cur.execute("UPDATE object SET data=?, updated=? WHERE kind=? AND id=?", (_canon(body), now, kind, oid))
        else:
            cur.execute("DELETE FROM object WHERE kind=? AND id=?", (kind, oid))
            if kind == "session":
                cur.execute("DELETE FROM object WHERE session_id=?", (oid,))

    # ------------------------------------------------------------------ read

    def get(self, kind: str, oid: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT data FROM object WHERE kind=? AND id=?", (kind, oid)).fetchone()
        return json.loads(row["data"]) if row else None

    def find(self, kind: str, *, session_id: str | None = None, parent_id: str | None = None,
             newest_first: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        sql, args = "SELECT data FROM object WHERE kind=?", [kind]
        if session_id is not None:
            sql += " AND session_id=?"
            args.append(session_id)
        if parent_id is not None:
            sql += " AND parent_id=?"
            args.append(parent_id)
        sql += f" ORDER BY created {'DESC' if newest_first else 'ASC'}, id {'DESC' if newest_first else 'ASC'}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def events(self, *, after: int = 0, aggregate: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM event WHERE seq > ?", [after]
        if aggregate:
            sql += " AND aggregate=?"
            args.append(aggregate)
        sql += " ORDER BY seq LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [{"seq": r["seq"], "id": r["id"], "type": r["type"], "aggregate": r["aggregate"], "time": r["time"],
                 "data": json.loads(r["data"]), "hash": r["hash"]} for r in rows]

    def iter_events(self, aggregate: str | None = None) -> Iterator[dict[str, Any]]:
        after = 0
        while batch := self.events(after=after, aggregate=aggregate):
            yield from batch
            after = batch[-1]["seq"]

    def verify(self) -> tuple[bool, str]:
        """Re-walk the hash chain. (True, "") or (False, where it breaks)."""
        prev = GENESIS
        with self._lock:
            rows = self._db.execute("SELECT * FROM event ORDER BY seq").fetchall()
        for r in rows:
            body = {"id": r["id"], "type": r["type"], "aggregate": r["aggregate"], "time": r["time"],
                    "data": json.loads(r["data"])}
            if r["prev"] != prev or _chain(prev, body) != r["hash"]:
                return False, f"event {r['seq']} ({r['type']}) does not match the chain"
            prev = r["hash"]
        return True, ""

    # ------------------------------------------------------------------ helpers

    def create(self, kind: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        body = {"id": body.get("id") or ids.new(kind), **body}
        self.publish(f"{kind}.created", body, **kw)
        return body

    def update(self, kind: str, oid: str, patch: dict[str, Any], **kw: Any) -> dict[str, Any]:
        cur = self.get(kind, oid)
        if cur is None:
            raise StoreError(f"no {kind} {oid}")
        patch = {**patch, "id": oid}
        for key in ("session_id", "message_id", "parent_id"):
            if key in cur and key not in patch:
                patch[key] = cur[key]
        self.publish(f"{kind}.updated", patch, **kw)
        return merge(cur, patch)

    def delete(self, kind: str, oid: str) -> None:
        self.publish(f"{kind}.deleted", {"id": oid})
