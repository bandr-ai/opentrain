"""Public coding-agent trajectories from Hugging Face as traces (recipe ``data/find`` for coding).

When the person's own traces are few, open datasets of solved software tasks fill in: SWE-smith's
Claude trajectories, SWE-Gym's and R2E-Gym's OpenHands trajectories. Each is read into a
:class:`~opentrain.traces.Trace` and then becomes training examples exactly like the person's own
traces (``opentrain.traces.sft``), so the trained model sees one format whatever the source.

Two shapes exist: OpenAI-style ``tool_calls`` (SWE-smith's ``tool`` split) and function calls
written in the text, ``<function=name><parameter=key>value</parameter></function>``, answered by a
user turn ("EXECUTION RESULT of [name]:"). Their editors (``str_replace_editor``,
``file_editor``) map to read/write/edit; ``bash``/``execute_bash`` to shell; ``search`` to search;
``finish``/``submit`` end the work; anything else is kept as ``other`` (and so not learned from).
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import Step, ToolUse, Trace

# Datasets known to convert, with the split and the filter that keeps solved trajectories.
KNOWN: dict[str, dict[str, Any]] = {
    "SWE-bench/SWE-smith-trajectories": {"config": "default", "split": "tool", "resolved": "resolved",
                                         "license": "MIT", "note": "Claude 3.7 Sonnet solving SWE-smith tasks"},
    "SWE-Gym/OpenHands-SFT-Trajectories": {"config": "default", "split": "train.success.oss", "license": "MIT",
                                           "note": "OpenHands, successful SWE-Gym trajectories"},
    "R2E-Gym/R2EGym-SFT-Trajectories": {"config": "default", "split": "train", "license": "MIT",
                                        "note": "R2E-Gym agent, verified trajectories"},
}

_FN = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.DOTALL)
_PARAM = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.DOTALL)
_RESULT = re.compile(r"^(?:EXECUTION RESULT of \[[\w.-]+\]:|Execution output of \[[\w.-]+\]:|OBSERVATION:)\s*\n?")
_ROOT = re.compile(r"(/testbed|/workspace/[\w.@-]+)")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return "" if content is None else str(content)


def tool_use(call_id: str, name: str, args: dict[str, Any], output: str | None) -> ToolUse | None:
    """One public-dataset tool call in OpenTrain's kinds; None for calls that are not work (think)."""
    n = name.lower()
    if n in ("think",):
        return None
    if n in ("bash", "execute_bash", "run_bash", "shell"):
        return ToolUse(call_id, name, "shell", {**args, "command": args.get("command") or args.get("cmd") or ""},
                       output)
    if n in ("str_replace_editor", "file_editor", "str_replace_based_edit_tool", "edit_file"):
        cmd, path = args.get("command"), args.get("path")
        if cmd == "view":
            rng = args.get("view_range")
            extra: dict[str, Any] = {}
            if isinstance(rng, str):
                try:
                    rng = json.loads(rng)
                except ValueError:
                    rng = None
            if isinstance(rng, list) and len(rng) == 2 and all(isinstance(x, int) for x in rng):
                extra = {"offset": rng[0], **({"limit": rng[1] - rng[0] + 1} if rng[1] > 0 else {})}
            return ToolUse(call_id, name, "read", {**args, "path": path, **extra}, output)
        if cmd == "create":
            return ToolUse(call_id, name, "write", {**args, "path": path, "content": args.get("file_text") or ""},
                           output)
        if cmd == "str_replace":
            return ToolUse(call_id, name, "edit", {**args, "path": path, "old": args.get("old_str"),
                                                   "new": args.get("new_str") or ""}, output)
        return ToolUse(call_id, name, "other", args, output)  # insert, undo_edit
    if n == "search":
        return ToolUse(call_id, name, "search", {**args, "pattern": args.get("search_term"),
                                                 "path": args.get("path") or "."}, output)
    if n in ("finish", "submit"):
        return ToolUse(call_id, name, "other", {**args, "finish": True}, output)
    return ToolUse(call_id, name, "other", args, output)


def trace_from_messages(messages: list[dict[str, Any]], trace_id: str, agent: str) -> Trace:
    first_user = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    root = _ROOT.search(first_user)
    t = Trace(trace_id=trace_id, agent=agent, path=trace_id, cwd=root.group(1) if root else "/testbed",
              started_at=None, ended_at=None)
    pending: dict[str, ToolUse] = {}
    last_calls: list[ToolUse] = []
    seen_prompt = False
    for m in messages:
        role, text = m.get("role"), _text(m.get("content"))
        if role == "system":
            continue
        if role == "user" and not seen_prompt:
            t.steps.append(Step("user", text))
            seen_prompt = True
        elif role == "assistant":
            uses: list[ToolUse] = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                u = tool_use(str(c.get("id") or len(pending)), fn.get("name") or "?", args, None)
                if u:
                    pending[u.id] = u
                    uses.append(u)
            body = text
            if not uses:
                for i, f in enumerate(_FN.finditer(text)):
                    args = {k: v.strip("\n") for k, v in _PARAM.findall(f.group(2))}
                    u = tool_use(f"{trace_id}-{len(t.steps)}-{i}", f.group(1), args, None)
                    if u:
                        uses.append(u)
                body = _FN.split(text)[0] if uses else text
            last_calls = uses
            t.steps.append(Step("agent", body.strip(), uses))
        elif role == "tool":
            out = _RESULT.sub("", text)
            ids = m.get("tool_call_ids") or ([m["tool_call_id"]] if m.get("tool_call_id") else [])
            for i in ids:
                if i in pending:
                    pending[i].output = out
        elif role == "user":
            out = _RESULT.sub("", text)
            if text != out or last_calls:  # the answer to text-format calls
                for u in last_calls:
                    u.output = out
                last_calls = []
    return t


def finished(t: Trace) -> Trace:
    """Drop the closing finish/submit call (the example adds its own ``finish``), and make the
    dataset's checkout folder (``/testbed``) the repository root, as it is where the model will work."""
    root = (t.cwd or "").rstrip("/")

    def local(text: str) -> str:
        return text.replace(root + "/", "").replace(root, ".") if root else text

    for s in t.steps:
        s.tools = [u for u in s.tools if not u.args.get("finish")]
        for u in s.tools:
            if u.kind == "read" and (u.output or "").lstrip().startswith("Here's the files and directories"):
                # the editor's view of a folder: the shell command that lists the same thing
                u.kind, u.args = "shell", {"command": f"find {u.args.get('path') or '.'} -maxdepth 2 -not -path '*/.*'"}
            if isinstance(u.args.get("command"), str):
                u.args["command"] = local(u.args["command"])
            if u.output:
                u.output = local(u.output)
    return t


def rows_to_traces(dataset: str, rows: list[dict[str, Any]]) -> list[Trace]:
    spec = KNOWN.get(dataset, {})
    out = []
    for i, r in enumerate(rows):
        if spec.get("resolved") and not r.get(spec["resolved"]):
            continue
        msgs = r.get("messages") or r.get("trajectory")
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except ValueError:
                continue
        if not isinstance(msgs, list):
            continue
        tid = str(r.get("traj_id") or r.get("trajectory_id") or r.get("instance_id") or f"{dataset}#{i}")
        out.append(finished(trace_from_messages(msgs, tid, f"hf:{dataset}")))
    return out
