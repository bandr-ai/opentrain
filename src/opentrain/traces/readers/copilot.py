"""GitHub Copilot: the Copilot CLI and VS Code's Copilot Chat. Best effort.

- Copilot CLI: ``$COPILOT_HOME/session-state/<id>/events.jsonl`` or ``session-state/<id>.jsonl``
  (default ``~/.copilot``). Lines are ``{type, data, timestamp}``: ``session.start``
  (``context.cwd``/``branch``), ``user.message`` (``content``), ``assistant.message`` (``content``,
  ``toolRequests[{toolCallId, name, arguments}]``), ``tool.execution_complete`` (``toolCallId``,
  ``success``, ``result``), ``session.model_change``. The older
  ``history-session-state/*.json`` (OpenAI-style ``chatMessages``) is read when present.
- VS Code Copilot Chat: ``<editor>/User/workspaceStorage/<hash>/chatSessions/*.json(l)`` and
  ``globalStorage/{emptyWindowChatSessions,transferredChatSessions}/*.json(l)``. ``.json`` is the
  session document; ``.jsonl`` is a log of operations (``kind`` 0 snapshot, 1 set, 2 push/splice,
  3 delete at path ``k``) replayed into the same document. Each request is a person's
  ``message.text`` and a ``response`` list of parts: markdown (``value``) becomes text,
  ``toolInvocationSerialized`` a ToolUse (VS Code does not store the tool's arguments; the
  terminal command and the past-tense message are what is kept), ``textEditGroup`` an edit with
  the file path and new text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import Step, ToolKind, ToolUse, Trace
from ._util import (
    Source,
    Unreadable,
    canonical,
    env_dir,
    existing,
    exit_code_in,
    finish,
    home,
    iter_jsonl,
    kind_for,
    load_json,
    maybe_json,
    parse_ts,
    text_of,
    tool_args,
    vscode_user_dirs,
)

NAME = "copilot"
_VSCODE = ("Code", "Code - Insiders", "VSCodium")


def _home() -> Path:
    return env_dir("COPILOT_HOME", home() / ".copilot")


def roots() -> list[Path]:
    h = _home()
    paths = [h / "session-state", h / "history-session-state"]
    for u in vscode_user_dirs(_VSCODE):
        paths += [u / "workspaceStorage", u / "globalStorage" / "emptyWindowChatSessions"]
        paths += [u / "globalStorage" / "transferredChatSessions"]
    return existing(paths)


def discover() -> list[str]:
    out: list[str] = []
    h = _home()
    ss = h / "session-state"
    if ss.is_dir():
        for p in sorted(ss.iterdir()):
            if p.is_dir() and (p / "events.jsonl").is_file():
                out.append(str(p / "events.jsonl"))
            elif p.suffix == ".jsonl":
                out.append(str(p))
    hs = h / "history-session-state"
    if hs.is_dir():
        out += [str(p) for p in sorted(hs.glob("*.json"))]
    for u in vscode_user_dirs(_VSCODE):
        dirs = sorted((u / "workspaceStorage").glob("*/chatSessions"))
        dirs += [u / "globalStorage" / "emptyWindowChatSessions", u / "globalStorage" / "transferredChatSessions"]
        for d in dirs:
            if d.is_dir():
                files = sorted([*d.glob("*.json"), *d.glob("*.jsonl")])
                stems = {f.stem for f in files if f.suffix == ".jsonl"}
                out += [str(f) for f in files if f.suffix == ".jsonl" or f.stem not in stems]
    return out


# ---------------------------------------------------------------- Copilot CLI

_CLI_KINDS: dict[str, ToolKind] = {"create": "write", "view": "read", "str_replace": "edit", "bash": "shell"}


def _cli_tool(tid: str, name: str, raw: Any) -> ToolUse:
    kind, args = tool_args(name, raw, _CLI_KINDS.get(name))
    return ToolUse(tid, name, kind, args)


def _read_events(path: Path) -> Trace | None:
    trace = Trace(path.parent.name if path.name == "events.jsonl" else path.stem, NAME, str(path), None, None, None)
    trace.extra["app"] = "copilot-cli"
    by_id: dict[str, ToolUse] = {}
    model = None
    for r in iter_jsonl(path):
        t = r.get("type")
        d = r.get("data") if isinstance(r.get("data"), dict) else {}
        at = parse_ts(r.get("timestamp"))
        if t == "session.start":
            trace.trace_id = str(d.get("sessionId") or trace.trace_id)
            ctx = d.get("context") if isinstance(d.get("context"), dict) else {}
            trace.cwd = ctx.get("cwd") or trace.cwd
            trace.git_branch = ctx.get("branch") or trace.git_branch
        elif t == "session.model_change":
            model = d.get("newModel") or model
        elif t == "user.message":
            text = text_of(d.get("content"))
            if text.strip():
                source = "system" if d.get("source") not in (None, "user") else "user"
                trace.steps.append(Step(source, text, at=at))
        elif t == "assistant.message":
            tools = []
            for q in d.get("toolRequests") or []:
                if isinstance(q, dict):
                    tu = _cli_tool(str(q.get("toolCallId") or ""), str(q.get("name") or "tool"), q.get("arguments"))
                    tu.started_at = at
                    by_id[tu.id] = tu
                    tools.append(tu)
            text = text_of(d.get("content"))
            if text or tools:
                trace.steps.append(Step("agent", text, tools=tools, at=at, model=d.get("model") or model))
        elif t == "tool.execution_complete":
            tu = by_id.get(str(d.get("toolCallId")))
            if tu is not None:
                res = d.get("result")
                tu.output = res if isinstance(res, str) else text_of(res) if res is not None else None
                tu.is_error = d.get("success") is False
                tu.ended_at = at
                if tu.kind == "shell":
                    tu.exit_code = exit_code_in(tu.output)
    return finish(trace)


def _read_history(path: Path) -> Trace | None:
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise Unreadable("not a session document")
    trace = Trace(str(doc.get("sessionId") or path.stem), NAME, str(path), None, parse_ts(doc.get("startTime")), None)
    trace.extra["app"] = "copilot-cli"
    by_id: dict[str, ToolUse] = {}
    for m in doc.get("chatMessages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = text_of(m.get("content"))
        if role == "user" and text.strip():
            trace.steps.append(Step("user", text))
        elif role == "assistant":
            tools = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function") if isinstance(c, dict) and isinstance(c.get("function"), dict) else {}
                tu = _cli_tool(str(c.get("id") or ""), str(fn.get("name") or "tool"), fn.get("arguments"))
                by_id[tu.id] = tu
                tools.append(tu)
            if text or tools:
                trace.steps.append(Step("agent", text, tools=tools))
        elif role == "tool":
            tu = by_id.get(str(m.get("tool_call_id")))
            if tu is not None:
                tu.output = text
                if tu.kind == "shell":
                    tu.exit_code = exit_code_in(text)
    return finish(trace)


# ---------------------------------------------------------------- VS Code chat


def _replay(path: Path) -> dict:
    doc: Any = {}
    for op in iter_jsonl(path):
        kind, keys, v = op.get("kind"), op.get("k") or [], op.get("v")
        if kind == 0:
            doc = v
            continue
        if not keys:
            continue
        parent = doc
        try:
            for k in keys[:-1]:
                parent = parent[k]
            last = keys[-1]
            if kind == 1:
                parent[last] = v
            elif kind == 2:
                arr = parent.setdefault(last, []) if isinstance(parent, dict) else parent[last]
                items = v if isinstance(v, list) else [v]
                i = op.get("i")
                if isinstance(i, int):
                    arr[i : i + len(items)] = items
                else:
                    arr.extend(items)
            elif kind == 3:
                del parent[last]
        except (KeyError, IndexError, TypeError):
            continue
    if not isinstance(doc, dict):
        raise Unreadable("chat session log did not replay to a document")
    return doc


_VS_KINDS: dict[str, ToolKind] = {
    "copilot_readFile": "read", "copilot_replaceString": "edit", "copilot_multiReplaceString": "edit",
    "copilot_insertEdit": "edit", "copilot_applyPatch": "patch", "copilot_createFile": "write",
    "copilot_runInTerminal": "shell", "run_in_terminal": "shell", "copilot_searchFiles": "search",
    "copilot_findFiles": "search", "copilot_findTextInFiles": "search", "copilot_listDir": "search",
    "copilot_searchCodebase": "search", "copilot_fetchWebPage": "web", "copilot_githubRepo": "web",
    "runSubagent": "task", "copilot_runSubagent": "task",
}  # fmt: skip


def _msg(v: Any) -> str:
    return v.get("value", "") if isinstance(v, dict) else v if isinstance(v, str) else ""


def _uri_path(u: Any) -> str | None:
    if isinstance(u, dict):
        return u.get("fsPath") or u.get("path")
    return u if isinstance(u, str) else None


def _vscode_tool(part: dict) -> ToolUse:
    tool_id = str(part.get("toolId") or "tool")
    kind = _VS_KINDS.get(tool_id) or kind_for(tool_id.removeprefix("copilot_"))
    data = part.get("toolSpecificData") if isinstance(part.get("toolSpecificData"), dict) else {}
    args: dict[str, Any] = {"message": _msg(part.get("pastTenseMessage")) or _msg(part.get("invocationMessage"))}
    cl = data.get("commandLine")
    cmd = (cl.get("toolEdited") or cl.get("original")) if isinstance(cl, dict) else data.get("command")
    if isinstance(cmd, str):
        args["command"] = cmd
        kind = "shell"
    for k in ("uri", "file", "path"):
        p = _uri_path(data.get(k))
        if p:
            args["path"] = p
    t = ToolUse(str(part.get("toolCallId") or ""), tool_id, kind, canonical(kind, args))
    details = part.get("resultDetails")
    if isinstance(details, dict) and details.get("output") is not None:
        t.output = text_of(details.get("output"))
    t.is_error = bool(part.get("isError")) or (isinstance(details, dict) and bool(details.get("isError")))
    if kind == "shell":
        t.exit_code = exit_code_in(t.output)
    return t


def _read_vscode(path: Path) -> Trace | None:
    doc = _replay(path) if path.suffix == ".jsonl" else load_json(path)
    if not isinstance(doc, dict):
        raise Unreadable("not a chat session document")
    trace = Trace(
        str(doc.get("sessionId") or path.stem), NAME, str(path), None, parse_ts(doc.get("creationDate")), None
    )
    trace.extra["app"] = "vscode-copilot-chat"
    if doc.get("customTitle"):
        trace.extra["title"] = doc["customTitle"]
    ws = path.parent.parent / "workspace.json"
    if path.parent.name == "chatSessions" and ws.is_file():
        w = maybe_json(ws.read_text(errors="replace"))
        if isinstance(w, dict) and isinstance(w.get("folder"), str):
            trace.cwd = w["folder"].removeprefix("file://")
    for req in doc.get("requests") or []:
        if not isinstance(req, dict):
            continue
        at = parse_ts(req.get("timestamp"))
        msg = req.get("message") if isinstance(req.get("message"), dict) else {}
        if isinstance(msg.get("text"), str) and msg["text"].strip():
            trace.steps.append(Step("user", msg["text"], at=at))
        texts: list[str] = []
        tools: list[ToolUse] = []
        for part in req.get("response") or []:
            if not isinstance(part, dict):
                continue
            k = part.get("kind")
            if k in (None, "markdownContent") and isinstance(_msg(part.get("value", part.get("content"))), str):
                v = _msg(part.get("value", part.get("content")))
                if v:
                    texts.append(v)
            elif k == "toolInvocationSerialized":
                if part.get("presentation") != "hidden":
                    tools.append(_vscode_tool(part))
            elif k == "textEditGroup":
                p = _uri_path(part.get("uri"))
                edits = part.get("edits") or []
                new = "\n".join(
                    e.get("text", "")
                    for group in edits
                    for e in (group if isinstance(group, list) else [group])
                    if isinstance(e, dict)
                )
                target = next(
                    (t for t in reversed(tools) if t.kind in ("edit", "write") and not t.args.get("path")), None
                )
                if target is not None and p:
                    target.args["path"] = p
                    target.args.setdefault("new", new)
                elif p:
                    tools.append(ToolUse(f"edit:{p}", "textEditGroup", "edit", {"path": p, "old": "", "new": new}))
        result = req.get("result") if isinstance(req.get("result"), dict) else {}
        meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        model = meta.get("resolvedModel") or req.get("modelId")
        if texts or tools:
            trace.steps.append(Step("agent", "".join(texts), tools=tools, at=at, model=model))
    return finish(trace)


def read(loc: str) -> Trace | None:
    path = Path(loc)
    if "session-state" in path.parts and path.suffix == ".jsonl":
        return _read_events(path)
    if "history-session-state" in path.parts:
        return _read_history(path)
    return _read_vscode(path)


SOURCE = Source(NAME, "GitHub Copilot", roots, discover, read)
