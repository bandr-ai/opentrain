"""Codex CLI: ``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl`` and ``$CODEX_HOME/archived_sessions``
(default ``~/.codex``).

Handles the current rollout format (``{"type": "response_item", "payload": {...}}`` with
``session_meta``/``turn_context``/``event_msg`` lines) and the older one where response items sit at
the top level and the first line is a bare ``{"id", "timestamp", "instructions"}`` header.

- ``message`` items: role ``user`` is the person's prompt unless it is harness context (environment
  context, AGENTS.md / user instructions, handoffs), which becomes ``system``; ``assistant`` is an
  agent step; ``developer``/``system`` messages (the system prompt, repeated per turn) are skipped.
- ``event_msg`` ``user_message`` and ``item_completed`` UserMessage repeat the prompt in newer
  versions; they are used only when the log has no ``response_item`` user prompts, and a prompt
  that repeats the one just recorded is dropped.
- ``function_call``/``custom_tool_call``/``local_shell_call`` are paired with their outputs by
  ``call_id``. A ``custom_tool_call`` named ``exec`` holds JavaScript that calls the real tools
  (``tools.exec_command({cmd: "..."})``, ``tools.apply_patch("*** Begin Patch ...")``); each such
  call becomes its own ToolUse (``<call_id>#<n>``) with the JS literal decoded. Code that cannot be
  decoded stays one ``other`` ToolUse holding the raw source under ``code``.
- Exit codes come from the output (JSON ``exit_code``, ``Process exited with code N``, older
  ``metadata.exit_code``) or from ``item_completed`` CommandExecution events.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...sessions.claude_code import NOT_HUMAN
from .. import Step, ToolKind, ToolUse, Trace
from ._util import (
    Source,
    canonical,
    env_dir,
    existing,
    exit_code_in,
    finish,
    home,
    iter_jsonl,
    kind_for,
    maybe_json,
    parse_ts,
    shell_text,
    split_instructions,
    text_of,
)

NAME = "codex"


def roots() -> list[Path]:
    base = env_dir("CODEX_HOME", home() / ".codex")
    return existing([base / "sessions", base / "archived_sessions"])


def discover() -> list[str]:
    out: list[str] = []
    for root in roots():
        out.extend(str(p) for p in sorted(root.rglob("*.jsonl")))
    return out


# ---------------------------------------------------------------- JS literal decoding


class _JSError(ValueError):
    pass


class Raw(str):
    """A JavaScript expression that is not a literal, kept as its source text."""


_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_NUM = re.compile(r"-?(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)(?![\w$])")
_OPEN, _CLOSE = "([{", ")]}"


class _JS:
    """A small reader for the JavaScript literals Codex's ``exec`` tool passes to its nested tools:
    strings in all three quote styles, numbers, booleans, null, arrays and object literals (bare
    or quoted keys). A template's ``${expr}`` and any non-literal value (a variable, a call, a
    shorthand property) are kept as source text: a template stays a string with ``${expr}`` in it,
    ``"a/" + name`` becomes ``"a/${name}"``, and a lone expression becomes a :class:`Raw`."""

    def __init__(self, src: str, pos: int = 0):
        self.s = src
        self.i = pos

    def ws(self) -> None:
        s = self.s
        while self.i < len(s):
            c = s[self.i]
            if c in " \t\r\n":
                self.i += 1
            elif s.startswith("//", self.i):
                j = s.find("\n", self.i)
                self.i = len(s) if j < 0 else j
            elif s.startswith("/*", self.i):
                j = s.find("*/", self.i + 2)
                self.i = len(s) if j < 0 else j + 2
            else:
                break

    def peek(self) -> str:
        self.ws()
        return self.s[self.i] if self.i < len(self.s) else ""

    def expect(self, c: str) -> None:
        if self.peek() != c:
            raise _JSError(f"expected {c!r} at {self.i}")
        self.i += 1

    def skip_expr(self) -> Raw:
        """Source of the expression starting here, up to a top-level ``,`` or closing bracket."""
        s, start, depth = self.s, self.i, 0
        while self.i < len(s):
            c = s[self.i]
            if c in "\"'`":
                self.skip_string()
                continue
            if c in _OPEN:
                depth += 1
            elif c in _CLOSE:
                if depth == 0:
                    break
                depth -= 1
            elif c == "," and depth == 0:
                break
            self.i += 1
        if self.i >= len(s):
            raise _JSError("unterminated expression")
        return Raw(s[start : self.i].strip())

    def skip_string(self) -> None:
        s, q = self.s, self.s[self.i]
        self.i += 1
        depth = 0
        while self.i < len(s):
            c = s[self.i]
            if c == "\\":
                self.i += 2
                continue
            if q == "`" and s.startswith("${", self.i):
                depth += 1
                self.i += 2
                continue
            if q == "`" and c == "}" and depth:
                depth -= 1
            elif c == q and not depth:
                self.i += 1
                return
            self.i += 1
        raise _JSError("unterminated string")

    def value(self) -> Any:
        parts = [self.atom()]
        while self.peek() == "+":
            self.i += 1
            parts.append(self.atom())
        if len(parts) == 1:
            return parts[0]
        if not any(isinstance(p, str) and not isinstance(p, Raw) for p in parts):
            return Raw(" + ".join(map(str, parts)))
        return "".join(f"${{{p}}}" if isinstance(p, Raw) else str(p) for p in parts)

    def atom(self) -> Any:
        c = self.peek()
        if c in "\"'`":
            return self.string()
        start = self.i
        try:
            if c == "{":
                return self.obj()
            if c == "[":
                return self.arr()
            m = _NUM.match(self.s, self.i)
            if m:
                self.i = m.end()
                t = m.group(0)
                if t.lower().lstrip("-").startswith("0x"):
                    return int(t, 16)
                return float(t) if any(x in t for x in ".eE") else int(t)
            m = _IDENT.match(self.s, self.i)
            if m and m.group(0) in ("true", "false", "null", "undefined"):
                end = m.end()
                nxt = _JS(self.s, end).peek()
                if nxt in (",", ")", "]", "}", "+", ""):
                    self.i = end
                    return {"true": True, "false": False}.get(m.group(0))
        except _JSError:
            pass
        # Not a literal: keep its source up to the next operand boundary.
        self.i = start
        return self.skip_operand()

    def skip_operand(self) -> Raw:
        s, start, depth = self.s, self.i, 0
        while self.i < len(s):
            c = s[self.i]
            if c in "\"'`":
                self.skip_string()
                continue
            if c in _OPEN:
                depth += 1
            elif c in _CLOSE:
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and (c == "," or (c == "+" and self.i > start)):
                break
            self.i += 1
        if self.i >= len(s):
            raise _JSError("unterminated expression")
        text = s[start : self.i].strip()
        if not text:
            raise _JSError(f"empty expression at {start}")
        return Raw(text)

    def string(self) -> str:
        s = self.s
        q = s[self.i]
        self.i += 1
        out: list[str] = []
        while self.i < len(s):
            c = s[self.i]
            if c == q:
                self.i += 1
                return "".join(out)
            if q == "`" and s.startswith("${", self.i):
                self.i += 2
                expr = self.skip_expr()
                self.expect("}")
                out.append(f"${{{expr}}}")
                continue
            if c == "\\":
                self.i += 1
                e = s[self.i] if self.i < len(s) else ""
                self.i += 1
                if e in _ESC and not (e == "0" and self.i < len(s) and s[self.i].isdigit()):
                    out.append(_ESC[e])
                elif e == "x":
                    out.append(chr(int(s[self.i : self.i + 2], 16)))
                    self.i += 2
                elif e == "u":
                    if s.startswith("{", self.i):
                        j = s.index("}", self.i)
                        out.append(chr(int(s[self.i + 1 : j], 16)))
                        self.i = j + 1
                    else:
                        code = int(s[self.i : self.i + 4], 16)
                        self.i += 4
                        if 0xD800 <= code < 0xDC00 and s.startswith("\\u", self.i):
                            low = int(s[self.i + 2 : self.i + 6], 16)
                            if 0xDC00 <= low < 0xE000:
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                                self.i += 6
                        out.append(chr(code))
                elif e == "\r":
                    if s.startswith("\n", self.i):
                        self.i += 1
                elif e in ("\n", "\u2028", "\u2029"):
                    pass  # line continuation
                else:
                    out.append(e)
                continue
            if c == "\n" and q != "`":
                raise _JSError("newline in string")
            out.append(c)
            self.i += 1
        raise _JSError("unterminated string")

    def arr(self) -> list:
        self.expect("[")
        out = []
        while self.peek() != "]":
            if self.peek() == "":
                raise _JSError("unterminated array")
            out.append(self.value())
            if self.peek() == ",":
                self.i += 1
            elif self.peek() != "]":
                raise _JSError(f"expected , or ] at {self.i}")
        self.i += 1
        return out

    def obj(self) -> dict:
        self.expect("{")
        out: dict[str, Any] = {}
        while self.peek() != "}":
            c = self.peek()
            if c == "":
                raise _JSError("unterminated object")
            if c in "\"'":
                key = self.string()
            elif self.s.startswith("...", self.i):
                self.i += 3
                out[f"...{self.skip_expr()}"] = None
                if self.peek() == ",":
                    self.i += 1
                continue
            else:
                m = _IDENT.match(self.s, self.i) or _NUM.match(self.s, self.i)
                if not m:
                    raise _JSError(f"bad key at {self.i}")
                key = m.group(0)
                self.i = m.end()
            if self.peek() in (",", "}"):  # shorthand {cmd}
                out[key] = Raw(key)
            else:
                self.expect(":")
                out[key] = self.value()
            if self.peek() == ",":
                self.i += 1
            elif self.peek() != "}":
                raise _JSError(f"expected , or }} at {self.i}")
        self.i += 1
        return out


_CALL = re.compile(r"\btools\.([A-Za-z_$][\w$]*)\s*\(")


def js_tool_calls(code: str) -> list[tuple[str, Any, str]]:
    """``(tool name, first argument, call source)`` for each ``tools.<name>(...)`` call in
    ``code``. The argument is decoded as far as it is a literal (see :class:`_JS`); when even
    that fails it is the :class:`Raw` source."""
    out: list[tuple[str, Any, str]] = []
    for m in _CALL.finditer(code):
        p = _JS(code, m.end())
        try:
            arg: Any = {} if p.peek() == ")" else p.value()
        except (_JSError, ValueError, IndexError):
            p = _JS(code, m.end())
            try:
                arg = p.skip_expr()
            except _JSError:
                arg = Raw(code[m.end() : m.end() + 2000])
        out.append((m.group(1), arg, code[m.start() : max(p.i + 1, m.end())]))
    return out


# ---------------------------------------------------------------- tools


_KINDS: dict[str, ToolKind] = {
    "exec_command": "shell", "shell": "shell", "local_shell": "shell", "shell_command": "shell",
    "container.exec": "shell", "apply_patch": "patch", "view_image": "read", "web_search": "web",
}  # fmt: skip


def _kind(name: str, namespace: str | None) -> ToolKind:
    if name in _KINDS:
        return _KINDS[name]
    if namespace and namespace.startswith("mcp__"):
        return "web" if "web_run" in name or "search_service" in namespace else "other"
    if name.startswith("mcp__"):
        return "web" if "web_run" in name else "other"
    if name in ("write_stdin", "update_plan", "wait", "request_user_input", "request_user_input_async"):
        return "other"
    return kind_for(name)


def _make(call_id: str, name: str, namespace: str | None, raw: Any) -> ToolUse:
    kind = _kind(name, namespace)
    args = maybe_json(raw)
    if kind == "patch" and isinstance(args, str):
        args = {"patch": args}
    if kind == "shell" and isinstance(args, dict) and isinstance(args.get("command"), list):
        args = {**args, "command": shell_text(args["command"])}
    if namespace and not name.startswith(namespace):
        full = f"{namespace}__{name}" if namespace.startswith("mcp__") else f"{namespace}.{name}"
    else:
        full = name
    return ToolUse(call_id, full, kind, canonical(kind, args))


def _plain(v: Any) -> Any:
    """Raw expressions as ``${expr}`` strings, recursively."""
    if isinstance(v, Raw):
        return f"${{{v}}}"
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def _exec_tools(call_id: str, code: str) -> list[ToolUse]:
    """The nested tool calls of an ``exec`` cell. A call whose key argument is not a literal
    (``{cmd}`` from a loop variable, ``apply_patch(text)``) is kept as ``other`` with its
    source under ``code``."""
    calls = js_tool_calls(code)
    if not calls:
        return [ToolUse(call_id, "exec", "other", {"code": code})]
    out = []
    for i, (name, arg, src) in enumerate(calls):
        tid = f"{call_id}#{i}"
        kind = _kind(name, None)
        key = None
        if kind == "shell":
            key = arg.get("cmd", arg.get("command")) if isinstance(arg, dict) else arg
        elif kind == "patch":
            key = arg.get("patch", arg.get("input")) if isinstance(arg, dict) else arg
        if isinstance(arg, Raw) or (kind in ("shell", "patch") and (not isinstance(key, str) or isinstance(key, Raw))):
            out.append(ToolUse(tid, name, "other", {"code": src}))
        else:
            out.append(_make(tid, name, None, _plain(arg)))
    return out


_DECODER = json.JSONDecoder()
_CHUNK_EXIT = re.compile(r'"exit_code"\s*:\s*(-?\d+)')


def _chunks(text: str) -> list[dict]:
    """The ``{"chunk_id": ..., "exit_code": ..., "output": ...}`` results an ``exec`` cell prints,
    wherever they sit in the text. A chunk cut short by truncation keeps its exit code."""
    out: list[dict] = []
    starts = [m.start() for m in re.finditer(r'\{"chunk_id"', text)]
    if not starts and text.lstrip().startswith("{"):
        starts = [text.index("{")]
    for n, i in enumerate(starts):
        try:
            value, _ = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            seg = text[i : starts[n + 1] if n + 1 < len(starts) else len(text)]
            m = _CHUNK_EXIT.search(seg)
            if m:
                out.append({"exit_code": int(m.group(1)), "output": seg})
            continue
        if isinstance(value, dict) and "output" in value:
            out.append(value)
    return out


def _output(raw: Any) -> tuple[str, list[dict], int | None]:
    """(text, JSON chunks with ``exit_code``, exit code) from a function/custom tool output."""
    chunks: list[dict] = []
    exit_code: int | None = None
    if isinstance(raw, dict) and "content" in raw:  # {"content": ..., "success": bool}
        raw = raw.get("content")
    if isinstance(raw, list):
        parts = []
        for item in raw:
            t = text_of(item) if isinstance(item, dict) else str(item)
            found = _chunks(t)
            if not found:
                parts.append(t)
                continue
            head = t[: t.find('{"chunk_id"')] if '{"chunk_id"' in t else ""
            if head.strip():
                parts.append(head)
            chunks.extend(found)
            parts.extend(str(c.get("output") or "") for c in found)
        text = "".join(parts)
    else:
        text = raw if isinstance(raw, str) else text_of(raw)
        j = maybe_json(text) if text.lstrip().startswith("{") else None
        if isinstance(j, dict) and "output" in j:
            meta = j.get("metadata") if isinstance(j.get("metadata"), dict) else {}
            ec = j.get("exit_code", meta.get("exit_code"))
            if isinstance(ec, int):
                exit_code = ec
            text = str(j.get("output") or "")
    if exit_code is None and chunks and isinstance(chunks[-1].get("exit_code"), int):
        exit_code = chunks[-1]["exit_code"]
    if exit_code is None:
        exit_code = exit_code_in(text)
    return text, chunks, exit_code


def _apply_output(tools: list[ToolUse], raw: Any, at) -> None:
    text, chunks, code = _output(raw)
    takers = [t for t in tools if t.kind == "shell" or t.name == "write_stdin"]
    if len(tools) > 1 and chunks and len(chunks) == len(takers):
        for t, c in zip(takers, chunks):
            t.output = str(c.get("output") or "")
            t.exit_code = c.get("exit_code") if isinstance(c.get("exit_code"), int) else exit_code_in(t.output)
        for t in tools:
            if t not in takers:
                t.output = text
    else:
        for t in tools:
            t.output = text
            if t.kind == "shell" or len(tools) == 1:
                t.exit_code = code
    failed = text.startswith(("Script failed", "Error:", "error:")) or "\nScript failed" in text[:200]
    for t in tools:
        t.ended_at = at
        t.is_error = failed or (t.exit_code not in (None, 0))


# ---------------------------------------------------------------- prompts

_SYSTEM_PREFIXES = (
    "<environment_context>", "<user_instructions>", "# AGENTS.md instructions", "<permissions instructions>",
    "<system_instruction>", "[Historical user;", "Context handoff (", "Delegated task node:", "<turn_aborted>",
    "<user_shell_command>", "<collaboration_mode>", "<skills_instructions>", "<INSTRUCTIONS>", "<subagent_notification>",
)  # fmt: skip


def _is_system(text: str) -> bool:
    t = text.lstrip()
    return t.startswith(_SYSTEM_PREFIXES) or bool(NOT_HUMAN.match(t))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        if b.get("type") in ("input_image", "image"):
            parts.append("[image]")
        elif isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(p for p in parts if p)


def _items(path: Path):
    """(timestamp, kind, payload) for every line, flattening both formats."""
    for r in iter_jsonl(path):
        at = parse_ts(r.get("timestamp"))
        t = r.get("type")
        p = r.get("payload")
        if t in ("response_item", "event_msg", "session_meta", "turn_context", "compacted") and isinstance(p, dict):
            yield at, t, p
        elif t is None and "id" in r and ("instructions" in r or "timestamp" in r) and "record_type" not in r:
            yield at, "session_meta", r
        elif isinstance(t, str) and t not in ("token_usage_record", "world_state") and "record_type" not in r:
            yield at, "response_item", r


def read(loc: str) -> Trace | None:
    path = Path(loc)
    items = list(_items(path))
    has_ri_prompt = any(
        k == "response_item"
        and p.get("type") == "message"
        and p.get("role") == "user"
        and not _is_system(_message_text(p.get("content")))
        for _, k, p in items
    )
    has_ri_agent = any(
        k == "response_item" and p.get("type") == "message" and p.get("role") == "assistant" for _, k, p in items
    )

    trace = Trace(trace_id=path.stem, agent=NAME, path=str(path), cwd=None, started_at=None, ended_at=None)
    pending: dict[str, list[ToolUse]] = {}
    unexited: list[ToolUse] = []
    model: str | None = None
    agent_step: Step | None = None

    def agent(at) -> Step:
        nonlocal agent_step
        if agent_step is None:
            agent_step = Step("agent", at=at, model=model)
            trace.steps.append(agent_step)
        return agent_step

    def add_user(text: str, at, system: bool, split: bool = True) -> None:
        nonlocal agent_step
        instructions, rest = split_instructions(text) if split else ("", text)
        if instructions:
            add_user(instructions, at, True, split=False)
            if not rest:
                return
            text, system = rest, _is_system(rest)
        source = "system" if system else "user"
        # The same prompt arrives as a response item and as an event; keep one.
        for s in reversed(trace.steps):
            if s.source == "agent":
                break
            if s.source == source and s.text.strip() == text.strip():
                return
        trace.steps.append(Step(source, text, at=at))
        agent_step = None

    for at, kind, p in items:
        if kind == "session_meta":
            trace.trace_id = str(p.get("id") or p.get("session_id") or trace.trace_id)
            trace.cwd = trace.cwd or p.get("cwd")
            git = p.get("git") if isinstance(p.get("git"), dict) else {}
            trace.git_branch = trace.git_branch or git.get("branch")
            for k in ("cli_version", "originator", "source", "model_provider"):
                if p.get(k) and k not in trace.extra:
                    trace.extra[k] = p[k]
            continue
        if kind == "turn_context":
            trace.cwd = trace.cwd or p.get("cwd")
            if isinstance(p.get("model"), str):
                model = p["model"]
            continue
        if kind == "compacted":
            add_user(
                "[context compacted]" + (f"\n{p['message']}" if isinstance(p.get("message"), str) else ""), at, True
            )
            continue
        if kind == "event_msg":
            et = p.get("type")
            if et == "user_message" and not has_ri_prompt:
                text = p.get("message") if isinstance(p.get("message"), str) else text_of(p.get("message"))
                if text.strip():
                    add_user(text, at, _is_system(text))
            elif et == "item_completed" and isinstance(p.get("item"), dict):
                item = p["item"]
                it = item.get("type")
                if it == "UserMessage" and not has_ri_prompt:
                    text = _message_text(item.get("content"))
                    if text.strip():
                        add_user(text, at, _is_system(text))
                elif it == "AgentMessage" and not has_ri_agent:
                    text = _message_text(item.get("content"))
                    if text:
                        step = agent(at)
                        step.text = "\n\n".join(x for x in (step.text, text) if x)
                elif it == "CommandExecution" and isinstance(item.get("exit_code"), int):
                    cmd = shell_text(item.get("command"))
                    for t in unexited:
                        if t.args.get("command") == cmd:
                            t.exit_code = item["exit_code"]
                            t.is_error = t.is_error or t.exit_code != 0
                            unexited.remove(t)
                            break
            elif et == "agent_message" and not has_ri_agent and isinstance(p.get("message"), str):
                step = agent(at)
                step.text = "\n\n".join(x for x in (step.text, p["message"]) if x)
            continue

        pt = p.get("type")
        if pt == "message":
            role = p.get("role")
            text = _message_text(p.get("content"))
            if role == "user":
                if text.strip():
                    add_user(text, at, _is_system(text))
            elif role == "assistant" and text:
                step = agent(at)
                step.text = "\n\n".join(x for x in (step.text, text) if x)
            continue
        if pt in ("function_call", "custom_tool_call", "local_shell_call", "web_search_call"):
            call_id = str(p.get("call_id") or p.get("id") or "")
            name = str(p.get("name") or "")
            if pt == "custom_tool_call" and name == "exec":
                tools = _exec_tools(call_id, str(p.get("input") or ""))
            elif pt == "custom_tool_call":
                tools = [_make(call_id, name, p.get("namespace"), p.get("input"))]
            elif pt == "local_shell_call":
                action = p.get("action") if isinstance(p.get("action"), dict) else {}
                args = {"command": shell_text(action.get("command")) or "", "workdir": action.get("working_directory")}
                tools = [ToolUse(call_id, "local_shell", "shell", canonical("shell", args))]
            elif pt == "web_search_call":
                action = p.get("action") if isinstance(p.get("action"), dict) else {}
                tools = [ToolUse(call_id, "web_search", "web", canonical("web", action))]
            else:
                tools = [_make(call_id, name, p.get("namespace"), p.get("arguments"))]
            for t in tools:
                t.started_at = at
                if t.kind == "shell":
                    unexited.append(t)
            agent(at).tools.extend(tools)
            if pt != "web_search_call":
                pending[call_id] = tools
            continue
        if pt in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
            tools = pending.pop(str(p.get("call_id") or ""), None)
            if tools:
                _apply_output(tools, p.get("output"), at)
                for t in tools:
                    if t.exit_code is not None and t in unexited:
                        unexited.remove(t)
            continue
    return finish(trace)


SOURCE = Source(NAME, "Codex", roots, discover, read)
