"""Cursor: the ``cursor-agent`` CLI and the Cursor IDE. Best effort; formats change often.

- CLI transcripts: ``~/.cursor/projects/<encoded path>/agent-transcripts/**/<id>.jsonl`` (lines
  ``{"role", "message": {"content": [...]}}`` with Anthropic-style blocks, no timestamps; the
  person's words sit in ``<user_query>``) or the older ``.txt`` (``user:`` / ``assistant:`` blocks,
  ``[Tool call] Name`` with indented ``key=value`` lines, ``[Tool result]``). Logs under
  ``subagents/`` are read as their own traces, flagged ``extra["subagent"]``.
- CLI chat stores: ``~/.cursor/chats/<workspace>/<agent>/store.db``. These hold protobuf blobs;
  only blobs that are plain JSON messages are read, so most are reported unreadable (counted).
- IDE: ``<Cursor>/User/globalStorage/state.vscdb``, table ``cursorDiskKV``: ``composerData:<id>``
  lists the conversation's bubbles (``fullConversationHeadersOnly`` or an inline
  ``conversation``), each stored as ``bubbleId:<composer>:<bubble>`` with ``type`` 1 (person) or 2
  (AI), ``text``, ``createdAt`` and ``toolFormerData`` (``name``, ``rawArgs``/``params``,
  ``result``, ``status``). Locations are ``<state.vscdb>#<composer id>``. Older per-workspace
  chats (``ItemTable`` key ``workbench.panel.aichat.view.aichat.chatdata``) are located as
  ``<workspace state.vscdb>#chat:<tab id>``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .. import Step, ToolUse, Trace
from ._util import (
    Source,
    Unreadable,
    connect_ro,
    env_dir,
    existing,
    exit_code_in,
    finish,
    home,
    iter_jsonl,
    maybe_json,
    parse_ts,
    split_loc,
    tables,
    text_of,
    tool_args,
    vscode_user_dirs,
)

NAME = "cursor"
CHAT_KEY = "workbench.panel.aichat.view.aichat.chatdata"


def _cursor_home() -> Path:
    return env_dir("CURSOR_CONFIG_DIR", home() / ".cursor")


def roots() -> list[Path]:
    ch = _cursor_home()
    return existing([ch / "projects", ch / "chats", *vscode_user_dirs(("Cursor",))])


def _transcripts() -> list[Path]:
    base = _cursor_home() / "projects"
    if not base.is_dir():
        return []
    out: list[Path] = []
    for d in base.glob("*/agent-transcripts"):
        files = sorted(d.rglob("*.jsonl"))
        stems = {f.with_suffix("") for f in files}
        files += [t for t in sorted(d.rglob("*.txt")) if t.with_suffix("") not in stems]
        out += files
    return out


def _kv(con: sqlite3.Connection, key: str) -> Any:
    for table in ("cursorDiskKV", "ItemTable"):
        try:
            row = con.execute(f"select value from {table} where key = ?", (key,)).fetchone()
        except sqlite3.Error:
            continue
        if row is not None:
            v = row[0]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            return maybe_json(v)
    return None


def discover() -> list[str]:
    out = [str(p) for p in _transcripts()]
    chats = _cursor_home() / "chats"
    if chats.is_dir():
        out += [str(p) for p in sorted(chats.glob("*/*/store.db"))]
    for user in vscode_user_dirs(("Cursor",)):
        g = user / "globalStorage" / "state.vscdb"
        if g.is_file():
            try:
                con = connect_ro(g)
                try:
                    if "cursorDiskKV" in tables(con):
                        rows = con.execute("select key from cursorDiskKV where key like 'composerData:%'")
                        out += [f"{g}#{k.split(':', 1)[1]}" for (k,) in rows]
                finally:
                    con.close()
            except sqlite3.Error:
                out.append(f"{g}#?")
        for w in sorted((user / "workspaceStorage").glob("*/state.vscdb")):
            try:
                con = connect_ro(w)
                try:
                    data = _kv(con, CHAT_KEY)
                finally:
                    con.close()
            except sqlite3.Error:
                continue
            for tab in (data or {}).get("tabs", []) if isinstance(data, dict) else []:
                if isinstance(tab, dict) and tab.get("bubbles"):
                    out.append(f"{w}#chat:{tab.get('tabId')}")
    return out


# ---------------------------------------------------------------- tools

_EXTRA_KINDS = {"Shell": "shell", "ReadFile": "read", "EditFile": "edit", "ApplyPatch": "patch", "Subagent": "task"}


def _tool(tid: str, name: str, raw: Any) -> ToolUse:
    kind, args = tool_args(name, raw, _EXTRA_KINDS.get(name))  # type: ignore[arg-type]
    return ToolUse(tid, name, kind, args)


def _result_text(result: Any) -> tuple[str | None, int | None]:
    r = maybe_json(result)
    if r is None:
        return None, None
    code = None
    if isinstance(r, dict):
        for k in ("exitCode", "exit_code", "exitCodeV2"):
            if isinstance(r.get(k), int):
                code = r[k]
        for k in ("output", "contents", "content", "result", "stdout", "text"):
            if isinstance(r.get(k), str):
                return r[k], code
        return json.dumps(r), code
    text = r if isinstance(r, str) else json.dumps(r)
    return text, code if code is not None else exit_code_in(text)


# ---------------------------------------------------------------- CLI transcripts

_QUERY = re.compile(r"<user_query>(.*?)</user_query>", re.S)


def _read_jsonl(path: Path, trace: Trace) -> None:
    by_id: dict[str, ToolUse] = {}
    for r in iter_jsonl(path):
        role = r.get("role") or (r.get("message") or {}).get("role")
        msg = r.get("message") if isinstance(r.get("message"), dict) else r
        content = msg.get("content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
        texts, tools, results = [], [], []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                texts.append(b["text"])
            elif b.get("type") == "tool_use":
                t = _tool(str(b.get("id") or ""), str(b.get("name") or "tool"), b.get("input"))
                by_id[t.id] = t
                tools.append(t)
            elif b.get("type") == "tool_result":
                results.append(b)
        for b in results:
            t = by_id.get(str(b.get("tool_use_id")))
            if t is not None:
                t.output = text_of(b.get("content"))
                t.is_error = bool(b.get("is_error"))
                if t.kind == "shell":
                    t.exit_code = exit_code_in(t.output)
        text = "\n".join(texts).strip()
        if role == "user":
            if not text:
                continue
            q = _QUERY.findall(text)
            if q:
                trace.steps.append(Step("user", "\n".join(x.strip() for x in q)))
            else:
                trace.steps.append(Step("system", text))
        elif role == "assistant" and (text or tools):
            trace.steps.append(Step("agent", text, tools=tools))


def _read_txt(path: Path, trace: Trace) -> None:
    role, buf = None, []
    blocks: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^(user|assistant):\s*(.*)$", line)
        if m:
            if role:
                blocks.append((role, "\n".join(buf)))
            role, buf = m.group(1), [m.group(2)]
        else:
            buf.append(line)
    if role:
        blocks.append((role, "\n".join(buf)))
    n = 0
    for role, body in blocks:
        if role == "user":
            q = _QUERY.findall(body)
            text = "\n".join(x.strip() for x in q) if q else body.strip()
            if text:
                trace.steps.append(Step("user", text))
            continue
        texts: list[str] = []
        tools: list[ToolUse] = []
        for seg in re.split(r"^(?=\[(?:Tool call|Tool result|Thinking)\])", body, flags=re.M):
            if seg.startswith("[Tool call]"):
                head, _, rest = seg.partition("\n")
                name = head[len("[Tool call]") :].strip() or "tool"
                args: dict[str, Any] = {}
                lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
                if lines and all("=" in ln for ln in lines):
                    for ln in lines:
                        k, _, v = ln.partition("=")
                        args[k.strip()] = v
                elif lines:
                    j = maybe_json("\n".join(lines))
                    args = j if isinstance(j, dict) else {"input": "\n".join(lines)}
                if name == "ApplyPatch" and "input" in args:
                    args = {"patch": args["input"]}
                tools.append(_tool(f"t{n}", name, args))
                n += 1
            elif seg.startswith("[Tool result]"):
                if tools and tools[-1].output is None:
                    tools[-1].output = "\n".join(ln.strip() for ln in seg.splitlines()[1:]).strip()
            elif seg.startswith("[Thinking]"):
                continue
            elif seg.strip():
                texts.append(seg.strip())
        if texts or tools:
            trace.steps.append(Step("agent", "\n\n".join(texts), tools=tools))


# ---------------------------------------------------------------- IDE


def _bubble_step(b: dict) -> Step | None:
    at = parse_ts(b.get("createdAt") or (b.get("timingInfo") or {}).get("clientStartTime"))
    text = b.get("text") if isinstance(b.get("text"), str) else ""
    if b.get("type") == 1:
        return Step("user", text, at=at) if text.strip() else None
    tools = []
    tf = b.get("toolFormerData")
    if isinstance(tf, dict) and tf.get("name"):
        raw = tf.get("rawArgs") or tf.get("params")
        t = _tool(str(tf.get("toolCallId") or b.get("bubbleId") or ""), str(tf["name"]), raw)
        t.output, t.exit_code = _result_text(tf.get("result"))
        status = str(tf.get("status") or "")
        t.is_error = status in ("error", "cancelled") or (t.exit_code not in (None, 0))
        t.started_at = at
        tools.append(t)
    model = (b.get("modelInfo") or {}).get("modelName") if isinstance(b.get("modelInfo"), dict) else None
    if not text.strip() and not tools:
        return None
    return Step("agent", text, tools=tools, at=at, model=model)


def _read_composer(db: Path, cid: str) -> Trace | None:
    con = connect_ro(db)
    try:
        data = _kv(con, f"composerData:{cid}")
        if not isinstance(data, dict):
            raise Unreadable("composer data missing or not JSON")
        trace = Trace(
            cid, NAME, f"{db}#{cid}", None, parse_ts(data.get("createdAt")), parse_ts(data.get("lastUpdatedAt"))
        )
        trace.extra["app"] = "cursor-ide"
        if data.get("name"):
            trace.extra["title"] = data["name"]
        ws = data.get("workspaceIdentifier") if isinstance(data.get("workspaceIdentifier"), dict) else {}
        uri = ws.get("uri") if isinstance(ws.get("uri"), dict) else {}
        trace.cwd = uri.get("fsPath")
        if not trace.cwd and isinstance(data.get("trackedGitRepos"), list) and data["trackedGitRepos"]:
            trace.cwd = (data["trackedGitRepos"][0] or {}).get("repoPath")
        bubbles: list[dict] = []
        if isinstance(data.get("conversation"), list) and data["conversation"]:
            bubbles = [b for b in data["conversation"] if isinstance(b, dict)]
        else:
            headers = data.get("fullConversationHeadersOnly") or []
            if not headers and isinstance(data.get("headerLog"), dict):
                for i in range(int(data["headerLog"].get("chunkCount") or 0)):
                    chunk = _kv(con, f"composerConvHeaders:{cid}:{i:06d}")
                    headers += chunk if isinstance(chunk, list) else []
            for h in headers:
                if isinstance(h, dict) and h.get("bubbleId"):
                    b = _kv(con, f"bubbleId:{cid}:{h['bubbleId']}")
                    if isinstance(b, dict):
                        b.setdefault("type", h.get("type"))
                        bubbles.append(b)
    finally:
        con.close()
    for b in bubbles:
        s = _bubble_step(b)
        if s is not None:
            trace.steps.append(s)
    return finish(trace)


def _read_old_chat(db: Path, tab_id: str) -> Trace | None:
    con = connect_ro(db)
    try:
        data = _kv(con, CHAT_KEY)
    finally:
        con.close()
    tabs = data.get("tabs", []) if isinstance(data, dict) else []
    tab = next((t for t in tabs if isinstance(t, dict) and str(t.get("tabId")) == tab_id), None)
    if tab is None:
        raise Unreadable("chat tab not found")
    trace = Trace(tab_id, NAME, f"{db}#chat:{tab_id}", None, parse_ts(tab.get("lastSendTime")), None)
    trace.extra["app"] = "cursor-ide"
    ws = db.parent / "workspace.json"
    if ws.is_file():
        folder = maybe_json(ws.read_text(errors="replace"))
        if isinstance(folder, dict) and isinstance(folder.get("folder"), str):
            trace.cwd = folder["folder"].removeprefix("file://")
    for b in tab.get("bubbles") or []:
        if not isinstance(b, dict):
            continue
        text = b.get("text") or b.get("rawText") or ""
        if b.get("type") == "user" and text.strip():
            trace.steps.append(Step("user", text))
        elif b.get("type") == "ai" and text.strip():
            trace.steps.append(Step("agent", text, model=b.get("modelType")))
    return finish(trace)


def _read_store(path: Path) -> Trace | None:
    con = connect_ro(path)
    try:
        rows = con.execute("select data from blobs").fetchall() if "blobs" in tables(con) else []
    finally:
        con.close()
    trace = Trace(path.parent.name, NAME, str(path), None, None, None, extra={"app": "cursor-agent"})
    for (data,) in rows:
        if not isinstance(data, (bytes, str)):
            continue
        try:
            raw = data.decode("utf-8") if isinstance(data, bytes) else data
        except UnicodeDecodeError:
            continue  # protobuf
        msg = maybe_json(raw)
        if not isinstance(msg, dict) or msg.get("role") not in ("user", "assistant"):
            continue
        text = text_of(msg.get("content"))
        if msg["role"] == "user":
            q = _QUERY.findall(text)
            trace.steps.append(Step("user" if q else "system", "\n".join(q) if q else text))
        elif text:
            trace.steps.append(Step("agent", text))
    if not trace.steps:
        raise Unreadable("store.db holds protobuf blobs this reader does not decode")
    return finish(trace)


def read(loc: str) -> Trace | None:
    if "#" in loc and not Path(loc).exists():
        db, key = split_loc(loc)
        if key == "?":
            raise Unreadable("cannot open state.vscdb")
        if key.startswith("chat:"):
            return _read_old_chat(db, key[5:])
        return _read_composer(db, key)
    path = Path(loc)
    if path.name == "store.db":
        return _read_store(path)
    trace = Trace(path.stem, NAME, str(path), None, None, None, extra={"app": "cursor-agent"})
    if "subagents" in path.parts:
        trace.extra["subagent"] = True
    proj = next((p for p in path.parents if p.parent.name == "projects"), None)
    if proj is not None:
        trace.extra["project"] = proj.name  # "/" and "." replaced by "-": not reversible
    if path.suffix == ".jsonl":
        _read_jsonl(path, trace)
    else:
        _read_txt(path, trace)
    return finish(trace)


SOURCE = Source(NAME, "Cursor", roots, discover, read)
