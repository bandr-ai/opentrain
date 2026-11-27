"""pi coding agent: ``$PI_CODING_AGENT_DIR/sessions/<project>/*.jsonl`` (default ``~/.pi/agent``;
``$PI_CODING_AGENT_SESSION_DIR`` overrides the sessions folder).

Records carry ``type``: ``session`` (``id``, ``cwd``, ``timestamp``), ``message`` (``message.role``
``user`` / ``assistant`` / ``toolResult``), ``model_change``, ``compaction`` (``summary``) and others.
Assistant content blocks are ``text``, ``thinking`` (dropped) and ``toolCall`` (``id``, ``name``,
``arguments``); a ``toolResult`` message pairs by ``toolCallId``. Logs of subagent runs nested
deeper under a session folder are read as their own traces. A branched session (several children
of one ``parentId``) is read in file order, not as a tree.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import Step, ToolUse, Trace
from ._util import Source, env_dir, existing, exit_code_in, finish, home, iter_jsonl, parse_ts, text_of, tool_args

NAME = "pi"


def roots() -> list[Path]:
    custom = os.environ.get("PI_CODING_AGENT_SESSION_DIR")
    if custom:
        return existing([Path(custom).expanduser()])
    return existing([env_dir("PI_CODING_AGENT_DIR", home() / ".pi" / "agent") / "sessions"])


def discover() -> list[str]:
    return [str(p) for r in roots() for p in sorted(r.rglob("*.jsonl"))]


def _blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def read(loc: str) -> Trace | None:
    path = Path(loc)
    recs = list(iter_jsonl(path))
    results: dict[str, dict] = {}
    for r in recs:
        m = r.get("message") if isinstance(r.get("message"), dict) else {}
        if r.get("type") == "message" and m.get("role") == "toolResult" and m.get("toolCallId"):
            results[str(m["toolCallId"])] = {**m, "_at": parse_ts(r.get("timestamp")) or parse_ts(m.get("timestamp"))}

    trace = Trace(trace_id=path.stem, agent=NAME, path=str(path), cwd=None, started_at=None, ended_at=None)
    model: str | None = None
    for r in recs:
        t = r.get("type")
        at = parse_ts(r.get("timestamp"))
        if t == "session":
            trace.trace_id = str(r.get("id") or trace.trace_id)
            trace.cwd = r.get("cwd") or trace.cwd
            continue
        if t == "model_change":
            model = r.get("modelId") or model
            continue
        if t == "compaction" and r.get("summary"):
            trace.steps.append(Step("system", f"[context compacted]\n{r['summary']}", at=at))
            continue
        if t == "session_info" and r.get("name"):
            trace.extra["title"] = r["name"]
            continue
        if t != "message":
            continue
        m = r.get("message") if isinstance(r.get("message"), dict) else {}
        role = m.get("role")
        at = at or parse_ts(m.get("timestamp"))
        if role == "user":
            text = "\n".join(b.get("text", "") for b in _blocks(m.get("content")) if b.get("type") == "text").strip()
            if text:
                trace.steps.append(Step("user", text, at=at))
        elif role == "assistant":
            texts, tools = [], []
            for b in _blocks(m.get("content")):
                if b.get("type") == "text" and b.get("text"):
                    texts.append(b["text"])
                elif b.get("type") == "toolCall":
                    name = str(b.get("name") or "tool")
                    kind, args = tool_args(name, b.get("arguments"))
                    if kind == "edit" and isinstance(args.get("edits"), list) and args["edits"]:
                        first = args["edits"][0] if isinstance(args["edits"][0], dict) else {}
                        args.setdefault("old", first.get("oldText", ""))
                        args["old"] = args["old"] or first.get("oldText", "")
                        args.setdefault("new", first.get("newText", ""))
                    tu = ToolUse(str(b.get("id") or ""), name, kind, args, started_at=at)
                    res = results.get(tu.id)
                    if res is not None:
                        tu.output = text_of(res.get("content"))
                        tu.is_error = bool(res.get("isError"))
                        tu.ended_at = res.get("_at")
                        if kind == "shell":
                            tu.exit_code = exit_code_in(tu.output) if tu.is_error else 0
                    tools.append(tu)
            if texts or tools:
                trace.steps.append(Step("agent", "\n\n".join(texts), tools=tools, at=at, model=m.get("model") or model))
    return finish(trace)


SOURCE = Source(NAME, "pi", roots, discover, read)
