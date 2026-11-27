"""Cline and its forks (Roo Code, Kilo Code), as VS Code extensions and as the Cline CLI.

Extension tasks live in ``<editor>/User/globalStorage/<extension id>/tasks/<task id>/``
(``saoudrizwan.claude-dev``, ``rooveterinaryinc.roo-cline``, ``kilocode.kilo-code``) for VS Code,
Insiders, VSCodium, Cursor and Windsurf, and in ``$CLINE_DIR/data/tasks/<task id>/`` (default
``~/.cline``). A task folder holds:

- ``api_conversation_history.json``: the Anthropic-style messages sent to the model. Older
  versions write tool calls as XML in the assistant text (``<execute_command><command>ls</command>
  </execute_command>``) and results as user text (``[execute_command for 'ls'] Result: ...``);
  newer ones use ``tool_use`` / ``tool_result`` blocks. Both are read. The person's words are the
  ``<task>``, ``<feedback>``, ``<answer>`` and ``<user_message>`` sections of user turns;
  ``<environment_details>`` is dropped.
- ``ui_messages.json``: what the panel showed (``say`` / ``ask`` entries with ``ts``). Used for
  timestamps, and as the whole source when the API history is missing.

Cline CLI sessions live in ``$CLINE_DIR/data/sessions/<id>/`` as ``<id>.json`` (cwd, model, times)
plus ``<id>.messages.json`` (``messages`` with ``tool_use`` / ``tool_result`` blocks).

Location: the task or session folder.
"""

from __future__ import annotations

import re
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
    kind_for,
    load_json,
    maybe_json,
    parse_ts,
    text_of,
    vscode_user_dirs,
)

NAME = "cline"
EXTENSIONS = {
    "saoudrizwan.claude-dev": "cline",
    "rooveterinaryinc.roo-cline": "roo-code",
    "kilocode.kilo-code": "kilo-code",
}


def _cline_home() -> Path:
    return env_dir("CLINE_DIR", home() / ".cline")


def roots() -> list[Path]:
    paths = [u / "globalStorage" / ext for u in vscode_user_dirs() for ext in EXTENSIONS]
    paths.append(_cline_home() / "data")
    return existing(paths)


def discover() -> list[str]:
    out: list[str] = []
    for root in roots():
        tasks = root / "tasks"
        if tasks.is_dir():
            for d in sorted(p for p in tasks.iterdir() if p.is_dir()):
                if (d / "api_conversation_history.json").is_file() or (d / "ui_messages.json").is_file():
                    out.append(str(d))
        sessions = root / "sessions"
        if sessions.is_dir():
            for d in sorted(p for p in sessions.iterdir() if p.is_dir()):
                if (d / f"{d.name}.messages.json").is_file() or (d / f"{d.name}.json").is_file():
                    out.append(str(d))
    return out


# ---------------------------------------------------------------- tools

_KINDS: dict[str, ToolKind] = {
    "execute_command": "shell", "read_file": "read", "read_files": "read", "write_to_file": "write",
    "replace_in_file": "edit", "apply_diff": "edit", "search_and_replace": "edit", "insert_content": "edit",
    "edit_file": "edit", "search_files": "search", "list_files": "search", "list_code_definition_names": "search",
    "codebase_search": "search", "browser_action": "web", "web_fetch": "web", "new_task": "task",
}  # fmt: skip
XML_TOOLS = (*_KINDS, "use_mcp_tool", "access_mcp_resource", "ask_followup_question", "attempt_completion",
             "plan_mode_respond", "load_mcp_documentation", "switch_mode", "update_todo_list", "fetch_instructions")  # fmt: skip
_XML_TOOL = re.compile(r"<(" + "|".join(XML_TOOLS) + r")>(.*?)</\1>", re.S)
_XML_PARAM = re.compile(r"<([a-z_]+)>\n?(.*?)\n?</\1>", re.S)
_SR = re.compile(r"(?:<{3,7}|-{3,7}) SEARCH\n(.*?)\n?={3,7}\n(.*?)\n?(?:>{3,7}|\+{3,7}) REPLACE", re.S)
_RESULT = re.compile(r"^\[(\w+)(?: for [^\]]*)?\] Result:\s*", re.S)


def make_tool(tid: str, name: str, raw: Any) -> ToolUse:
    kind = _KINDS.get(name) or ("other" if name in XML_TOOLS else None)
    args = maybe_json(raw)
    args = dict(args) if isinstance(args, dict) else {"input": args}
    if name == "execute_command" and isinstance(args.get("commands"), list):
        args["command"] = "\n".join(map(str, args["commands"]))
    if name == "read_files" and isinstance(args.get("files"), list) and args["files"]:
        f = args["files"][0]
        args.setdefault("path", f.get("path") if isinstance(f, dict) else f)
    if kind is None:
        kind = kind_for(name)
    canon = canonical(kind, args)
    if kind == "edit" and isinstance(args.get("diff"), str):
        m = _SR.search(args["diff"])
        if m:
            canon["old"], canon["new"] = m.group(1), m.group(2)
    return ToolUse(tid, name, kind, canon)


def _xml_tools(text: str, prefix: str) -> tuple[str, list[ToolUse]]:
    tools = []
    for i, m in enumerate(_XML_TOOL.finditer(text)):
        params = {p.group(1): p.group(2) for p in _XML_PARAM.finditer(m.group(2))}
        tools.append(make_tool(f"{prefix}.{i}", m.group(1), params))
    rest = _XML_TOOL.sub("", text)
    rest = re.sub(r"</?thinking>", "", rest).strip()
    return rest, tools


# ---------------------------------------------------------------- API history

_ENV = re.compile(r"<environment_details>.*?(</environment_details>|\Z)", re.S)
_PERSON = re.compile(r"<(task|feedback|answer|user_message|user_input)\b[^>]*>(.*?)</\1>", re.S)
_CWD = re.compile(r"Current (?:Working|Workspace) Directory \(([^)]+)\)")


def _blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _from_messages(trace: Trace, messages: list[dict], times: list) -> None:
    pending_xml: list[ToolUse] = []
    by_id: dict[str, ToolUse] = {}
    for n, msg in enumerate(messages):
        role = msg.get("role")
        at = parse_ts(msg.get("ts")) or (times[n] if n < len(times) else None)
        blocks = _blocks(msg.get("content"))
        if role == "assistant":
            texts, tools = [], []
            for b in blocks:
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    rest, xml = _xml_tools(b["text"], f"{n}")
                    if rest:
                        texts.append(rest)
                    tools.extend(xml)
                elif b.get("type") == "tool_use":
                    t = make_tool(str(b.get("id") or f"{n}"), str(b.get("name") or "tool"), b.get("input"))
                    by_id[t.id] = t
                    tools.append(t)
            for t in tools:
                t.started_at = at
            pending_xml = [t for t in tools if t.id not in by_id]
            model = (msg.get("modelInfo") or {}).get("id") if isinstance(msg.get("modelInfo"), dict) else None
            if texts or tools:
                trace.steps.append(Step("agent", "\n\n".join(texts), tools=tools, at=at, model=model))
            continue
        person: list[str] = []
        other: list[str] = []
        for b in blocks:
            if b.get("type") == "tool_result":
                t = by_id.get(str(b.get("tool_use_id")))
                if t is not None:
                    t.output = text_of(b.get("content"))
                    t.is_error = bool(b.get("is_error"))
                    t.ended_at = at
                    if t.kind == "shell":
                        t.exit_code = exit_code_in(t.output)
                continue
            if b.get("type") != "text" or not isinstance(b.get("text"), str):
                continue
            text = b["text"]
            if not trace.cwd:
                m = _CWD.search(text)
                if m:
                    trace.cwd = m.group(1).strip()
            text = _ENV.sub("", text).strip()
            r = _RESULT.match(text)
            if r and pending_xml:
                t = pending_xml.pop(0)
                t.output = text[r.end() :]
                t.ended_at = at
                t.is_error = t.output.lstrip().startswith(("The tool execution failed", "Error"))
                if t.kind == "shell":
                    t.exit_code = exit_code_in(t.output)
                person += [m.group(2).strip() for m in _PERSON.finditer(t.output)]
                continue
            found = [m.group(2).strip() for m in _PERSON.finditer(text)]
            if found:
                person += found
            elif text:
                other.append(text)
        for p in person:
            if p:
                trace.steps.append(Step("user", p, at=at))
        if other and n == 0 and not person:
            trace.steps.append(Step("user", "\n".join(other), at=at))
        elif other:
            trace.steps.append(Step("system", "\n".join(other), at=at))


# ---------------------------------------------------------------- UI messages only


def _from_ui(trace: Trace, ui: list[dict]) -> None:
    last_shell: ToolUse | None = None
    first = True
    for i, m in enumerate(ui):
        if m.get("partial"):
            continue
        at = parse_ts(m.get("ts"))
        text = m.get("text") if isinstance(m.get("text"), str) else ""
        say, ask = m.get("say"), m.get("ask")
        if say == "text" or say == "task":
            trace.steps.append(Step("user" if first else "agent", text, at=at))
            first = False
        elif say in ("user_feedback", "user_feedback_diff"):
            trace.steps.append(Step("user", text, at=at))
        elif say == "completion_result":
            trace.steps.append(Step("agent", text, at=at))
        elif ask == "command" or say == "command":
            last_shell = make_tool(str(i), "execute_command", {"command": text})
            last_shell.started_at = at
            trace.steps.append(Step("agent", "", tools=[last_shell], at=at))
        elif say == "command_output" and last_shell is not None:
            last_shell.output = (last_shell.output or "") + text
            last_shell.exit_code = exit_code_in(last_shell.output)
        elif ask == "tool" or say == "tool":
            data = maybe_json(text)
            if isinstance(data, dict):
                name = str(data.get("tool") or "tool")
                ui_kinds = {
                    "editedExistingFile": "replace_in_file", "appliedDiff": "apply_diff", "newFileCreated": "write_to_file",
                    "readFile": "read_file", "listFilesTopLevel": "list_files", "listFilesRecursive": "list_files",
                    "searchFiles": "search_files", "codebaseSearch": "codebase_search", "newTask": "new_task",
                }  # fmt: skip
                tool = make_tool(str(i), ui_kinds.get(name, name), data)
                tool.name = name
                if name == "readFile" and isinstance(data.get("content"), str):
                    tool.output = data["content"]
                trace.steps.append(Step("agent", "", tools=[tool], at=at))
        elif say == "error" and text:
            trace.steps.append(Step("system", text, at=at))


# ---------------------------------------------------------------- entry


def _ui(dir_: Path) -> list[dict]:
    f = dir_ / "ui_messages.json"
    if not f.is_file():
        return []
    data = load_json(f)
    return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []


def _app(dir_: Path) -> str:
    for part in dir_.parts:
        if part in EXTENSIONS:
            return EXTENSIONS[part]
    return "cline-cli" if dir_.parent.name == "sessions" else "cline"


def read(loc: str) -> Trace | None:
    d = Path(loc)
    if not d.is_dir():
        raise Unreadable("task folder missing")
    trace = Trace(d.name, NAME, str(d), None, None, None, extra={"app": _app(d)})
    if d.parent.name == "sessions":
        meta_f = d / f"{d.name}.json"
        meta = load_json(meta_f) if meta_f.is_file() else {}
        meta = meta if isinstance(meta, dict) else {}
        trace.cwd = meta.get("cwd") or meta.get("workspace_root")
        trace.started_at = parse_ts(meta.get("started_at"))
        trace.ended_at = parse_ts(meta.get("ended_at"))
        if meta.get("model"):
            trace.models.append(str(meta["model"]))
        msgs_f = d / f"{d.name}.messages.json"
        data = load_json(msgs_f) if msgs_f.is_file() else {}
        messages = data.get("messages") if isinstance(data, dict) else data
        if not isinstance(messages, list):
            raise Unreadable("no messages list")
        _from_messages(trace, [m for m in messages if isinstance(m, dict)], [])
        return finish(trace)

    ui = _ui(d)
    api_f = d / "api_conversation_history.json"
    if api_f.is_file():
        history = load_json(api_f)
        if not isinstance(history, list):
            raise Unreadable("api_conversation_history.json is not a list")
        # UI timestamps: the first one starts the task; api_req_started marks each model call.
        req_times = [parse_ts(m.get("ts")) for m in ui if m.get("say") == "api_req_started"]
        times: list = []
        r = 0
        for msg in history:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                times.append(req_times[r] if r < len(req_times) else None)
                r += 1
            else:
                times.append(times[-1] if times else None)
        _from_messages(trace, [m for m in history if isinstance(m, dict)], times)
    else:
        _from_ui(trace, ui)
    if ui:
        stamps = [t for t in (parse_ts(m.get("ts")) for m in ui) if t]
        if stamps:
            trace.started_at, trace.ended_at = min(stamps), max(stamps)
    return finish(trace)


SOURCE = Source(NAME, "Cline / Roo Code / Kilo Code", roots, discover, read)
