"""Helpers shared by the per-agent readers: where things live, timestamps, JSON lines, read-only
SQLite, and mapping an agent's tool call to a :data:`~opentrain.traces.ToolKind` with canonical
arguments."""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import ToolKind, Trace


class Unreadable(Exception):
    """A log that exists but cannot be turned into a trace; the message is the reason."""


@dataclass
class Source:
    name: str
    label: str
    roots: Callable[[], list[Path]]
    """Directories or files where this agent keeps its logs (only those that exist)."""
    discover: Callable[[], list[str]]
    """Every log location: a file path, or ``<db path>#<session id>`` for a SQLite store."""
    read: Callable[[str], Trace | None]
    """One log as a :class:`Trace`; None when it holds no conversation. May raise; the registry
    turns that into an unreadable entry."""


# ---------------------------------------------------------------- locations


def home() -> Path:
    return Path.home()


def env_dir(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


def xdg_data() -> Path:
    return env_dir("XDG_DATA_HOME", home() / ".local" / "share")


def xdg_config() -> Path:
    return env_dir("XDG_CONFIG_HOME", home() / ".config")


VSCODE_FLAVOURS = ("Code", "Code - Insiders", "VSCodium", "Cursor", "Windsurf")


def vscode_user_dirs(flavours: tuple[str, ...] = VSCODE_FLAVOURS) -> list[Path]:
    """``<editor>/User`` folders of VS Code and its forks, macOS and Linux."""
    bases = [home() / "Library" / "Application Support"] if sys.platform == "darwin" else []
    bases.append(xdg_config())
    out = []
    for base in bases:
        for f in flavours:
            p = base / f / "User"
            if p.is_dir() and p not in out:
                out.append(p)
    return out


def existing(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.exists() and p not in out:
            out.append(p)
    return out


# ---------------------------------------------------------------- values


def parse_ts(value: Any) -> datetime | None:
    """ISO 8601 strings, or epoch seconds / milliseconds / microseconds as numbers."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            return None
        while v > 1e11:  # ms, us, ns
            v /= 1000
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if re.fullmatch(r"\d+(\.\d+)?", s):
            return parse_ts(float(s))
        try:
            ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return None


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Every JSON object line; bad lines are skipped."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise Unreadable(f"invalid JSON: {exc.msg} at line {exc.lineno}") from None


def maybe_json(value: Any) -> Any:
    """A JSON string decoded, anything else unchanged."""
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return value
    return value


def text_of(value: Any) -> str:
    """Text from the common content shapes: a string, a list of blocks with ``text``, a dict."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(t for t in (text_of(v) for v in value) if t)
    if isinstance(value, dict):
        for k in ("text", "content", "output", "value"):
            if k in value and value[k] is not None:
                return text_of(value[k])
        return ""
    return str(value)


_EXIT = re.compile(
    r"(?:Process exited with code|Exit code:?|exited with code|Exit status:?)\s*(-?\d+)",
    re.I,
)


def exit_code_in(text: str | None) -> int | None:
    if not text:
        return None
    m = _EXIT.search(text[:4000])
    return int(m.group(1)) if m else None


def connect_ro(path: Path) -> sqlite3.Connection:
    """Open a SQLite file read-only. Falls back to ``immutable`` when the WAL files are not
    writable (the app may hold them)."""
    uri = path.resolve().as_uri()
    try:
        con = sqlite3.connect(f"{uri}?mode=ro", uri=True, timeout=2)
        con.execute("select 1 from sqlite_master limit 1").fetchall()
        return con
    except sqlite3.Error:
        return sqlite3.connect(f"{uri}?mode=ro&immutable=1", uri=True, timeout=2)


def columns(con: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [r[1] for r in con.execute(f'pragma table_info("{table}")')]
    except sqlite3.Error:
        return []


def tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("select name from sqlite_master where type in ('table','view')")}


def pick(cols: list[str], *names: str) -> str | None:
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def split_loc(loc: str) -> tuple[Path, str]:
    db, _, key = loc.rpartition("#")
    if not db:
        raise Unreadable(f"not a <db>#<id> location: {loc}")
    return Path(db), key


# ---------------------------------------------------------------- tools

SHELL_NAMES = {
    "bash", "shell", "exec_command", "run_terminal_cmd", "run_shell_command", "execute_command",
    "local_shell", "run_in_terminal", "runinterminal", "terminal", "run_command", "command",
    "bashtool", "shell_command", "run_terminal_command",
}  # fmt: skip
READ_NAMES = {
    "read", "read_file", "view", "readfile", "read_many_files", "cat", "view_file", "open_file",
    "view_image", "readnotebook", "notebookread", "read_notebook",
}  # fmt: skip
EDIT_NAMES = {
    "edit", "edit_file", "str_replace", "str_replace_editor", "replace", "replace_in_file",
    "search_replace", "multiedit", "multi_edit", "notebookedit", "edit_notebook", "replace_string_in_file",
    "insert_edit_into_file", "strreplace", "update_file", "editfile", "apply_diff",
}  # fmt: skip
WRITE_NAMES = {"write", "write_file", "create_file", "write_to_file", "createfile", "writefile", "new_file"}
PATCH_NAMES = {"apply_patch", "patch", "applypatch"}
SEARCH_NAMES = {
    "grep", "glob", "ls", "list", "list_dir", "list_directory", "list_files", "search", "find",
    "search_files", "codebase_search", "grep_search", "file_search", "search_file_content",
    "find_files", "file_glob_search", "semantic_search", "list_code_definition_names", "ripgrep",
    "list_dir_tool", "toolsearch",
}  # fmt: skip
WEB_NAMES = {
    "webfetch", "websearch", "web_search", "web_fetch", "fetch", "google_web_search", "fetch_webpage",
    "browser_action", "web", "search_web", "web_run", "_web_run", "web__run", "read_url",
}  # fmt: skip
TASK_NAMES = {"task", "agent", "spawn_agent", "subagent", "new_task", "delegate", "dispatch_agent", "runsubagent"}


def kind_for(name: str) -> ToolKind:
    n = (name or "").strip()
    low = n.lower()
    if low.startswith("mcp__") or low.startswith("mcp_"):
        return "other"
    if low in SHELL_NAMES:
        return "shell"
    if low in PATCH_NAMES:
        return "patch"
    if low in WRITE_NAMES:
        return "write"
    if low in EDIT_NAMES:
        return "edit"
    if low in READ_NAMES:
        return "read"
    if low in SEARCH_NAMES:
        return "search"
    if low in WEB_NAMES or low.endswith("web_run"):
        return "web"
    if low in TASK_NAMES:
        return "task"
    return "other"


def _first(args: dict, *keys: str) -> Any:
    for k in keys:
        v = args.get(k)
        if v is not None and v != "":
            return v
    return None


def shell_text(command: Any) -> str | None:
    """A shell command as one string. ``["bash", "-lc", "script"]`` becomes ``script``."""
    if isinstance(command, str):
        return command
    if isinstance(command, list) and command and all(isinstance(c, str) for c in command):
        if (
            len(command) >= 3
            and command[-2] in ("-c", "-lc", "-ic", "/c", "-Command")
            and os.path.basename(command[0]).lower()
            in ("bash", "sh", "zsh", "fish", "cmd", "cmd.exe", "powershell", "pwsh", "dash")
        ):
            return command[-1]
        return shlex.join(command)
    return None


def canonical(kind: ToolKind, args: Any) -> dict[str, Any]:
    """The agent's own arguments with the canonical ones for ``kind`` merged over them."""
    a: dict[str, Any] = dict(args) if isinstance(args, dict) else ({"input": args} if args not in (None, "") else {})
    out = dict(a)
    if kind == "shell":
        cmd = shell_text(_first(a, "command", "cmd", "commandLine", "script", "input"))
        if cmd is not None:
            out["command"] = cmd
    elif kind in ("read", "edit", "write"):
        p = _first(
            a,
            "path",
            "file_path",
            "filePath",
            "absolute_path",
            "target_file",
            "targetFile",
            "notebook_path",
            "filename",
            "file",
            "uri",
            "relative_workspace_path",
        )
        if isinstance(p, dict):
            p = p.get("path") or p.get("fsPath")
        if isinstance(p, str):
            out["path"] = p
        if kind == "edit":
            old = _first(a, "old", "old_string", "oldString", "old_str", "search", "oldText", "old_text")
            new = _first(
                a,
                "new",
                "new_string",
                "newString",
                "new_str",
                "replace",
                "newText",
                "new_text",
                "new_source",
                "code_edit",
                "replacement",
                "diff",
            )
            out["old"] = old if isinstance(old, str) else ""
            if isinstance(new, str):
                out["new"] = new
        if kind == "write":
            c = _first(a, "content", "contents", "file_text", "text", "fileContent", "code")
            if isinstance(c, str):
                out["content"] = c
    elif kind == "patch":
        p = _first(a, "patch", "input", "diff", "patchText", "content")
        if isinstance(p, str):
            out["patch"] = p
    elif kind == "search":
        q = _first(a, "pattern", "query", "regex", "glob", "glob_pattern", "search", "q")
        if isinstance(q, str):
            out.setdefault("query", q)
    elif kind == "web":
        u = _first(a, "url", "uri")
        q = _first(a, "query", "q", "search_term")
        if isinstance(u, str):
            out.setdefault("url", u)
        if isinstance(q, str):
            out.setdefault("query", q)
    elif kind == "task":
        p = _first(a, "prompt", "message", "description", "task", "instructions")
        if isinstance(p, str):
            out.setdefault("prompt", p)
    return out


def tool_args(name: str, raw: Any, kind: ToolKind | None = None) -> tuple[ToolKind, dict[str, Any]]:
    k = kind or kind_for(name)
    args = maybe_json(raw)
    return k, canonical(k, args)


def finish(trace: Trace) -> Trace | None:
    """Fill start/end from steps; None when the trace has no conversation at all."""
    if not any(s.text.strip() or s.tools for s in trace.steps):
        return None
    times = [s.at for s in trace.steps if s.at]
    if times:
        trace.started_at = trace.started_at or min(times)
        trace.ended_at = trace.ended_at or max(times)
    for s in trace.steps:
        if s.model and s.model not in trace.models:
            trace.models.append(s.model)
    return trace


_LEADING_INSTRUCTIONS = re.compile(r"\A\s*((?:<system_instruction>.*?</system_instruction>\s*)+)(.*)\Z", re.S)


def split_instructions(text: str) -> tuple[str, str]:
    """(harness instructions, the person's words) for a prompt that a wrapper such as Conductor
    prefixed with ``<system_instruction>`` blocks; ``("", text)`` otherwise."""
    m = _LEADING_INSTRUCTIONS.match(text)
    if not m:
        return "", text
    return m.group(1).strip(), m.group(2).strip()
