"""Gemini CLI: ``$GEMINI_CLI_HOME/.gemini/tmp/<project>/chats/session-*.json`` or ``.jsonl``
(``$GEMINI_CLI_HOME`` defaults to ``~``; ``<project>`` is a hash of the project path in older
versions and a short name in newer ones).

- ``.json``: one document ``{sessionId, projectHash, startTime, lastUpdated, messages: [...]}``.
- ``.jsonl``: a header line (``sessionId``, ...), then message records; a later record with the
  same ``id`` replaces the earlier one, ``{"$set": {...}}`` lines patch the header.

Messages have ``type`` ``user`` / ``gemini`` (kept) or ``info`` / ``error`` / ``warning`` (kept as
``system``), ``content`` (a string or parts), ``model``, ``thoughts`` (dropped) and ``toolCalls``
(``id``, ``name``, ``args``, ``status``, ``timestamp``, ``result`` as functionResponse parts).

``logs.json`` in each project folder holds only the person's prompts (no replies or tools). It is
read as a prompts-only trace when the folder has no chat files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import Step, ToolUse, Trace
from ._util import (
    Source,
    Unreadable,
    env_dir,
    existing,
    exit_code_in,
    finish,
    home,
    iter_jsonl,
    load_json,
    parse_ts,
    tool_args,
)

NAME = "gemini-cli"


def roots() -> list[Path]:
    return existing([env_dir("GEMINI_CLI_HOME", home()) / ".gemini" / "tmp"])


def discover() -> list[str]:
    out: list[str] = []
    for root in roots():
        for project in sorted(p for p in root.iterdir() if p.is_dir()):
            chats = project / "chats"
            files = sorted([*chats.glob("*.json"), *chats.glob("*.jsonl")]) if chats.is_dir() else []
            if files:
                out.extend(str(f) for f in files)
            elif (project / "logs.json").is_file():
                out.append(str(project / "logs.json"))
    return out


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (_text(c) for c in content) if t)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        fr = content.get("functionResponse")
        if isinstance(fr, dict):
            resp = fr.get("response")
            if isinstance(resp, dict):
                for k in ("output", "error", "content"):
                    if resp.get(k) is not None:
                        return resp[k] if isinstance(resp[k], str) else json.dumps(resp[k])
                return json.dumps(resp)
            return _text(resp)
    return ""


_GEMINI_KINDS = {
    "run_shell_command": "shell", "read_file": "read", "read_many_files": "read", "write_file": "write",
    "replace": "edit", "edit": "edit", "edit_file": "edit", "smart_edit": "edit", "list_directory": "search",
    "glob": "search", "search_file_content": "search", "grep": "search", "web_fetch": "web",
    "google_web_search": "web",
}  # fmt: skip


def _load(path: Path) -> tuple[dict, list[dict]]:
    if path.suffix == ".json":
        doc = load_json(path)
        if isinstance(doc, list):  # logs.json: [{sessionId, messageId, type, message, timestamp}]
            return {}, doc
        if not isinstance(doc, dict):
            raise Unreadable("not a Gemini chat document")
        msgs = doc.get("messages") if isinstance(doc.get("messages"), list) else []
        return doc, [m for m in msgs if isinstance(m, dict)]
    header: dict = {}
    order: list[str] = []
    by_id: dict[str, dict] = {}
    for i, r in enumerate(iter_jsonl(path)):
        if "$set" in r and isinstance(r["$set"], dict):
            header.update(r["$set"])
            continue
        if not header and "sessionId" in r and "type" not in r:
            header = dict(r)
            continue
        key = str(r.get("id") or f"#{i}")
        if key not in by_id:
            order.append(key)
        by_id[key] = r
    return header, [by_id[k] for k in order]


def read(loc: str) -> Trace | None:
    path = Path(loc)
    header, msgs = _load(path)
    trace = Trace(
        trace_id=str(header.get("sessionId") or path.stem),
        agent=NAME,
        path=str(path),
        cwd=header.get("projectRoot") or header.get("cwd"),
        started_at=parse_ts(header.get("startTime")),
        ended_at=parse_ts(header.get("lastUpdated")),
    )
    if header.get("projectHash"):
        trace.extra["project_hash"] = header["projectHash"]
    if header.get("summary"):
        trace.extra["title"] = header["summary"]
    for m in msgs:
        mtype = m.get("type") or m.get("role")
        at = parse_ts(m.get("timestamp"))
        if path.name == "logs.json":
            trace.trace_id = str(m.get("sessionId") or trace.trace_id)
            text = m.get("message") if isinstance(m.get("message"), str) else ""
            if mtype == "user" and text.strip():
                trace.steps.append(Step("user", text, at=at))
            continue
        text = _text(m.get("content"))
        if mtype == "user":
            if text.strip():
                trace.steps.append(Step("user", text, at=at))
        elif mtype in ("gemini", "model", "assistant"):
            tools = []
            for c in m.get("toolCalls") or []:
                if not isinstance(c, dict):
                    continue
                name = str(c.get("name") or "tool")
                kind, args = tool_args(name, c.get("args"), _GEMINI_KINDS.get(name))  # type: ignore[arg-type]
                status = str(c.get("status") or "")
                out = _text(c.get("result")) if c.get("result") is not None else c.get("resultDisplay")
                tu = ToolUse(
                    str(c.get("id") or ""), name, kind, args, output=out if isinstance(out, str) else None,
                    is_error=status in ("error", "cancelled"), started_at=parse_ts(c.get("timestamp")),
                )  # fmt: skip
                if kind == "shell" and tu.output:
                    tu.exit_code = exit_code_in(tu.output)
                tools.append(tu)
            if text or tools:
                trace.steps.append(Step("agent", text, tools=tools, at=at, model=m.get("model")))
        elif mtype in ("info", "error", "warning") and text.strip():
            trace.steps.append(Step("system", text, at=at))
    return finish(trace)


SOURCE = Source(NAME, "Gemini CLI", roots, discover, read)
