"""Claude Code: ``$CLAUDE_CONFIG_DIR/projects/<project>/<session>.jsonl`` (default ``~/.claude``).

One log per session. Subagent transcripts that newer versions write beside it
(``<session>/subagents/agent-*.jsonl``) are merged in as sidechain steps, as are sidechain records
inside the main log (``isSidechain``).

- A user record a person typed becomes a ``user`` step (cleaned with the summary reader's
  :func:`clean_prompt`). Other user-role text (task notifications, PR-watch updates, compaction
  summaries, ``isMeta`` skill bodies, interrupt markers, a subagent's own prompt) becomes ``system``.
- Assistant records that share a ``message.id`` (Claude Code writes one record per content block)
  are merged into one ``agent`` step. Thinking blocks are dropped.
- Each ``tool_use`` is paired with its ``tool_result`` (text, ``is_error``; an exit code when the
  text carries one).
- ``MultiEdit`` becomes one ``edit`` ToolUse per edit (ids ``<id>#<n>``), each carrying the
  call's output, so ``old``/``new`` stay one replacement each.
"""

from __future__ import annotations

from pathlib import Path

from ...sessions.claude_code import _content, _human_prompt, _result_text, _ts, clean_prompt
from .. import Step, ToolKind, ToolUse, Trace
from ._util import Source, canonical, env_dir, existing, exit_code_in, finish, home, iter_jsonl, split_instructions

NAME = "claude-code"

_KINDS: dict[str, ToolKind] = {
    "Bash": "shell", "BashOutput": "other", "Read": "read", "Edit": "edit", "MultiEdit": "edit",
    "NotebookEdit": "edit", "Write": "write", "Grep": "search", "Glob": "search", "LS": "search",
    "WebFetch": "web", "WebSearch": "web", "Task": "task", "Agent": "task",
}  # fmt: skip


def roots() -> list[Path]:
    base = env_dir("CLAUDE_CONFIG_DIR", home() / ".claude")
    return existing([base / "projects"])


def discover() -> list[str]:
    out: list[str] = []
    for root in roots():
        for project in sorted(p for p in root.iterdir() if p.is_dir()):
            out.extend(str(p) for p in sorted(project.glob("*.jsonl")))
    return out


def tool_uses(block: dict) -> list[ToolUse]:
    """A ``tool_use`` block as ToolUses (several for MultiEdit), without results."""
    name = block.get("name") or "tool"
    raw = block.get("input") if isinstance(block.get("input"), dict) else {}
    tid = str(block.get("id") or "")
    kind: ToolKind = "other" if name.startswith("mcp__") else _KINDS.get(name, "other")
    if name == "MultiEdit" and isinstance(raw.get("edits"), list) and raw["edits"]:
        out = []
        for i, e in enumerate(raw["edits"]):
            e = e if isinstance(e, dict) else {}
            args = canonical("edit", {"file_path": raw.get("file_path"), **e})
            out.append(ToolUse(f"{tid}#{i}", name, "edit", args))
        return out
    if name == "NotebookEdit":
        args = canonical("edit", {**raw, "old": ""})
        args["new"] = raw.get("new_source") or ""
        return [ToolUse(tid, name, "edit", args)]
    return [ToolUse(tid, name, kind, canonical(kind, raw))]


def _records(path: Path) -> list[tuple[dict, bool]]:
    """(record, from a subagent file) for the main log and any subagent logs, in time order."""
    main = [(r, False) for r in iter_jsonl(path)]
    sub_dir = path.with_suffix("") / "subagents"
    subs = sorted(sub_dir.glob("*.jsonl")) if sub_dir.is_dir() else []
    if not subs:
        return main
    merged = list(main)
    for f in subs:
        merged.extend((r, True) for r in iter_jsonl(f))
    # Stable sort by time; records without a timestamp keep the time of the one before them.
    keyed = []
    last = None
    for i, (r, side) in enumerate(merged):
        t = _ts(r) or last
        last = t
        keyed.append((t.timestamp() if t else 0.0, i, r, side))
    keyed.sort(key=lambda k: (k[0], k[1]))
    return [(r, side) for _, _, r, side in keyed]


def read(loc: str) -> Trace | None:
    path = Path(loc)
    recs = _records(path)
    results: dict[str, tuple[dict, object]] = {}
    for r, _ in recs:
        if r.get("type") == "user":
            for b in _content(r):
                if b.get("type") == "tool_result" and isinstance(b.get("tool_use_id"), str):
                    results[b["tool_use_id"]] = (b, _ts(r))

    trace = Trace(trace_id=path.stem, agent=NAME, path=str(path), cwd=None, started_at=None, ended_at=None)
    last_msg_id: str | None = None
    for r, from_sub in recs:
        rtype = r.get("type")
        if rtype in ("custom-title", "ai-title") and isinstance(r.get("title") or r.get("customTitle"), str):
            trace.extra.setdefault("title", r.get("customTitle") or r.get("title"))
            continue
        if rtype not in ("user", "assistant"):
            continue
        if not from_sub:
            trace.cwd = trace.cwd or r.get("cwd")
            trace.git_branch = trace.git_branch or r.get("gitBranch")
            if r.get("sessionId"):
                trace.trace_id = r["sessionId"] if trace.trace_id == path.stem else trace.trace_id
        side = bool(r.get("isSidechain")) or from_sub
        at = _ts(r)

        if rtype == "user":
            blocks = _content(r)
            if any(b.get("type") == "tool_result" for b in blocks):
                continue
            last_msg_id = None
            prompt = None if side else _human_prompt(r)
            if prompt is not None and not prompt.startswith("[Request interrupted"):
                instructions, prompt = split_instructions(prompt)
                if instructions:
                    trace.steps.append(Step("system", instructions, at=at))
                if prompt:
                    trace.steps.append(Step("user", prompt, at=at))
                continue
            text = clean_prompt("\n".join(b.get("text", "") for b in blocks if b.get("type") == "text"))
            if text:
                trace.steps.append(Step("system", text, at=at, sidechain=side))
            continue

        message = r.get("message") if isinstance(r.get("message"), dict) else {}
        model = message.get("model")
        model = model if isinstance(model, str) and model != "<synthetic>" else None
        texts: list[str] = []
        tools: list[ToolUse] = []
        for b in _content(r):
            if b.get("type") == "text" and b.get("text"):
                texts.append(b["text"])
            elif b.get("type") == "tool_use":
                res, res_at = results.get(str(b.get("id")), (None, None))
                for t in tool_uses(b):
                    t.started_at = at
                    if res is not None:
                        t.output = _result_text(res)
                        t.is_error = bool(res.get("is_error"))
                        t.ended_at = res_at  # type: ignore[assignment]
                        if t.kind == "shell":
                            t.exit_code = exit_code_in(t.output) if t.is_error else 0
                    tools.append(t)
        if not texts and not tools:
            continue
        msg_id = message.get("id")
        prev = trace.steps[-1] if trace.steps else None
        if msg_id and msg_id == last_msg_id and prev is not None and prev.source == "agent" and prev.sidechain == side:
            prev.text = "\n\n".join(x for x in (prev.text, *texts) if x)
            prev.tools.extend(tools)
            continue
        last_msg_id = msg_id
        trace.steps.append(Step("agent", "\n\n".join(texts), tools=tools, at=at, model=model, sidechain=side))
    return finish(trace)


SOURCE = Source(NAME, "Claude Code", roots, discover, read)
