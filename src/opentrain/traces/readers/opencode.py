"""opencode: ``$XDG_DATA_HOME/opencode/opencode.db`` (default ``~/.local/share/opencode``;
``$OPENCODE_DIR`` overrides) and the older ``storage/{session,message,part}/**.json`` layout. Kilo
Code's CLI keeps the same format under ``~/.local/share/kilo`` and is read here too.

SQLite (opened read-only), schema inspected at run time:

- v1: ``session(id, project_id, title, directory?, time_created, ...)``, ``message(id, session_id,
  time_created, data)``, ``part(id, message_id, session_id?, time_created, data)``; ``data`` is
  JSON (message: ``role``, ``modelID``, ``time``; part: ``type`` text / reasoning / tool / ...).
  A tool part is ``{tool, callID, state: {status, input, output, error, metadata: {exit}, time}}``.
- v2 (beta): ``session_v2`` and ``session_message(session_id, type, seq, data)`` where an
  assistant message's ``data.content`` holds text / reasoning / tool items and ``shell`` messages
  are commands the person ran.

Locations are ``<db>#<session id>`` for SQLite and the session JSON path for the file layout.
Times are epoch milliseconds. Parts flagged ``synthetic`` (injected by opencode) become ``system``.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from .. import Step, ToolUse, Trace
from ._util import (
    Source,
    Unreadable,
    columns,
    connect_ro,
    existing,
    finish,
    load_json,
    maybe_json,
    parse_ts,
    split_loc,
    tables,
    text_of,
    tool_args,
    xdg_data,
)

NAME = "opencode"


def roots() -> list[Path]:
    custom = os.environ.get("OPENCODE_DIR")
    base = [Path(custom).expanduser()] if custom else [xdg_data() / "opencode"]
    return existing([*base, xdg_data() / "kilo"])


def _dbs() -> list[Path]:
    return [p for r in roots() for p in sorted(r.glob("*.db")) if p.is_file()]


def _session_ids(con: sqlite3.Connection) -> list[str]:
    names = tables(con)
    ids: list[str] = []
    if "session_v2" in names:
        ids += [r[0] for r in con.execute("select id from session_v2 order by time_created")]
    if "session" in names and "message" in names:
        seen = set(ids)
        ids += [r[0] for r in con.execute("select id from session order by time_created") if r[0] not in seen]
    return ids


def discover() -> list[str]:
    out: list[str] = []
    for db in _dbs():
        try:
            con = connect_ro(db)
            try:
                out += [f"{db}#{sid}" for sid in _session_ids(con)]
            finally:
                con.close()
        except sqlite3.Error:
            out.append(f"{db}#?")  # counted; read() reports why it cannot be opened
    for r in roots():
        sess = r / "storage" / "session"
        if sess.is_dir():
            out += [str(p) for p in sorted(sess.rglob("*.json"))]
    return out


# ---------------------------------------------------------------- shared conversion


def _tool(part: dict, at) -> ToolUse:
    name = str(part.get("tool") or part.get("name") or "tool")
    state = part.get("state") if isinstance(part.get("state"), dict) else {}
    kind, args = tool_args(name, state.get("input") if state.get("input") is not None else part.get("input"))
    meta = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    output = state.get("output")
    if output is None and isinstance(state.get("content"), list):
        output = text_of(state["content"])
    err = state.get("error")
    if isinstance(err, dict):
        err = err.get("message") or str(err)
    times = state.get("time") if isinstance(state.get("time"), dict) else part.get("time") or {}
    t = ToolUse(
        str(part.get("callID") or part.get("id") or ""),
        name,
        kind,
        args,
        output=output if isinstance(output, str) else (text_of(output) if output is not None else err),
        is_error=state.get("status") == "error" or bool(err),
        started_at=parse_ts(times.get("start") or times.get("created")) or at,
        ended_at=parse_ts(times.get("end") or times.get("completed")),
    )
    exit_code = meta.get("exit", meta.get("exitCode"))
    if isinstance(exit_code, int):
        t.exit_code = exit_code
        t.is_error = t.is_error or exit_code != 0
    return t


def _add_message(trace: Trace, msg: dict, parts: list[dict]) -> None:
    role = msg.get("role")
    times = msg.get("time") if isinstance(msg.get("time"), dict) else {}
    at = parse_ts(times.get("created")) or parse_ts(msg.get("time_created"))
    model = msg.get("modelID")
    if not model and isinstance(msg.get("model"), dict):
        model = msg["model"].get("modelID")
    if role == "user":
        for p in parts:
            if p.get("type") == "text" and isinstance(p.get("text"), str) and p["text"].strip():
                source = "system" if p.get("synthetic") else "user"
                trace.steps.append(Step(source, p["text"], at=at))
        return
    texts, tools = [], []
    for p in parts:
        ptype = p.get("type")
        if ptype == "text" and isinstance(p.get("text"), str) and not p.get("synthetic"):
            texts.append(p["text"])
        elif ptype == "tool":
            tools.append(_tool(p, at))
    if texts or tools:
        trace.steps.append(Step("agent", "\n\n".join(texts), tools=tools, at=at, model=model))


def _row_dict(cols: list[str], row: tuple) -> dict:
    d = dict(zip(cols, row))
    data = maybe_json(d.pop("data", None))
    if isinstance(data, dict):
        d = {**d, **data}
    return d


# ---------------------------------------------------------------- SQLite


def _read_v1(con: sqlite3.Connection, sid: str, db: Path) -> Trace:
    scols = columns(con, "session")
    row = con.execute("select * from session where id = ?", (sid,)).fetchone()
    if row is None:
        raise Unreadable("session not found")
    sess = _row_dict(scols, row)
    cwd = sess.get("directory")
    if not cwd and sess.get("project_id") and "project" in tables(con):
        pcols = columns(con, "project")
        prow = con.execute("select * from project where id = ?", (sess["project_id"],)).fetchone()
        if prow:
            cwd = _row_dict(pcols, prow).get("worktree")
    trace = Trace(sid, NAME, f"{db}#{sid}", cwd, None, None)
    if sess.get("title"):
        trace.extra["title"] = sess["title"]
    if sess.get("parent_id") or sess.get("parentID"):
        trace.extra["parent_session"] = sess.get("parent_id") or sess.get("parentID")
    mcols = columns(con, "message")
    order = "time_created, id" if "time_created" in mcols else "id"
    msgs = [
        _row_dict(mcols, r) for r in con.execute(f"select * from message where session_id = ? order by {order}", (sid,))
    ]
    pcols = columns(con, "part")
    porder = "time_created, id" if "time_created" in pcols else "id"
    if "session_id" in pcols:
        prows = con.execute(f"select * from part where session_id = ? order by {porder}", (sid,))
    else:
        ids = [m["id"] for m in msgs]
        marks = ",".join("?" * len(ids)) or "''"
        prows = con.execute(f"select * from part where message_id in ({marks}) order by {porder}", ids)
    parts: dict[str, list[dict]] = {}
    for r in prows:
        p = _row_dict(pcols, r)
        parts.setdefault(str(p.get("message_id") or p.get("messageID")), []).append(p)
    for m in msgs:
        _add_message(trace, m, parts.get(str(m.get("id")), []))
    return trace


def _read_v2(con: sqlite3.Connection, sid: str, db: Path) -> Trace:
    scols = columns(con, "session_v2")
    sess = _row_dict(scols, con.execute("select * from session_v2 where id = ?", (sid,)).fetchone())
    trace = Trace(sid, NAME, f"{db}#{sid}", sess.get("directory"), None, None)
    if sess.get("title"):
        trace.extra["title"] = sess["title"]
    mcols = columns(con, "session_message")
    order = "seq" if "seq" in mcols else "time_created"
    for r in con.execute(f"select * from session_message where session_id = ? order by {order}", (sid,)):
        m = _row_dict(mcols, r)
        mtype = m.get("type")
        at = parse_ts((m.get("time") or {}).get("created") if isinstance(m.get("time"), dict) else None) or parse_ts(
            m.get("time_created")
        )
        if mtype == "user":
            text = m.get("text") if isinstance(m.get("text"), str) else ""
            if text.strip():
                trace.steps.append(Step("user", text, at=at))
        elif mtype == "assistant":
            model = m.get("model") if isinstance(m.get("model"), dict) else {}
            texts, tools = [], []
            for c in m.get("content") or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text" and c.get("text"):
                    texts.append(c["text"])
                elif c.get("type") == "tool":
                    tools.append(_tool(c, at))
            if texts or tools:
                trace.steps.append(Step("agent", "\n\n".join(texts), tools=tools, at=at, model=model.get("id")))
        elif mtype == "shell":  # a command the person ran with "!"
            out = m.get("output") if isinstance(m.get("output"), str) else ""
            trace.steps.append(Step("system", f"$ {m.get('command') or ''}\n{out}".rstrip(), at=at))
        elif mtype in ("system", "synthetic", "skill", "compaction"):
            text = m.get("summary") or m.get("text")
            if isinstance(text, str) and text.strip():
                trace.steps.append(Step("system", text, at=at))
    return trace


def _read_db(loc: str) -> Trace | None:
    db, sid = split_loc(loc)
    if sid == "?":
        raise Unreadable("cannot open database")
    con = connect_ro(db)
    try:
        names = tables(con)
        if "session_v2" in names and con.execute("select 1 from session_v2 where id = ?", (sid,)).fetchone():
            trace = _read_v2(con, sid, db)
        elif "session" in names:
            trace = _read_v1(con, sid, db)
        else:
            raise Unreadable("no session table")
    finally:
        con.close()
    return finish(trace)


# ---------------------------------------------------------------- JSON storage


def _read_storage(path: Path) -> Trace | None:
    sess = load_json(path)
    if not isinstance(sess, dict):
        raise Unreadable("session file is not an object")
    storage = path
    while storage.name != "storage" and storage.parent != storage:
        storage = storage.parent
    sid = str(sess.get("id") or path.stem)
    trace = Trace(sid, NAME, str(path), sess.get("directory"), None, None)
    if sess.get("title"):
        trace.extra["title"] = sess["title"]
    if not trace.cwd and sess.get("projectID"):
        pj = storage / "project" / f"{sess['projectID']}.json"
        if pj.is_file():
            proj = load_json(pj)
            trace.cwd = proj.get("worktree") if isinstance(proj, dict) else None
    msg_dir = storage / "message" / sid
    msgs = (
        [m for m in (load_json(p) for p in sorted(msg_dir.glob("*.json"))) if isinstance(m, dict)]
        if msg_dir.is_dir()
        else []
    )
    msgs.sort(key=lambda m: ((m.get("time") or {}).get("created") or 0, str(m.get("id"))))
    for m in msgs:
        pdir = storage / "part" / str(m.get("id"))
        parts = (
            [p for p in (load_json(f) for f in sorted(pdir.glob("*.json"))) if isinstance(p, dict)]
            if pdir.is_dir()
            else []
        )
        _add_message(trace, m, parts)
    return finish(trace)


def read(loc: str) -> Trace | None:
    if loc.endswith(".json") and Path(loc).is_file():
        return _read_storage(Path(loc))
    return _read_db(loc)


SOURCE = Source(NAME, "opencode", roots, discover, read)
