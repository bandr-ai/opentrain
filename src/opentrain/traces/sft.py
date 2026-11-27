"""Any agent's trace as training examples for the trained model's own tools (recipe ``data/traces``).

The trained model works through the direct runtime's five tools (``bash``, ``read_file``,
``write_file``, ``edit_file``, ``finish``), so a trace from Claude Code, Codex or any other agent is
translated into them: a shell call stays a shell call, Read/Edit/Write map one to one, a search
becomes the ``rg`` command that finds the same thing, and an apply_patch becomes the edits it makes.
A step that cannot be expressed in those tools (a web search, a subagent, an MCP tool) cannot be
learned from, so the request it belongs to is left out, with the reason, rather than trained with a
hole in it.

One example per request: the person's prompt, then the agent's work until the next prompt, ended
with ``finish``. Paths become relative to the repository, outputs are clipped as the runtime clips
them, and timing is dropped, so the example is what the model will see when it is served.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any

from ..runtime.direct import SYSTEM_PROMPT, TOOLS, clip
from . import Step, ToolUse, Trace

Example = dict[str, Any]


class Skip(ValueError):
    """A request that cannot become an example; the message is the reason."""


# Harness bookkeeping, not work: finding tools, waiting on background jobs, plans. Left out of the
# example (call and result) instead of costing the whole request.
BOOKKEEPING = re.compile(r"^(ToolSearch|TodoWrite|TodoRead|TaskStop|TaskOutput|TaskList|TaskGet|TaskCreate|TaskUpdate|"
                         r"BashOutput|KillShell|KillBash|Monitor|ScheduleWakeup|update_plan|sleep|wait|"
                         r"mcp__.*(background_wait|task_status|thread_wait))$")


def work_root(trace: Trace) -> str | None:
    """The repository the trace worked in: its folder's repo, else the repo of the files it edited
    (sessions often start in a scratch folder and edit a repo elsewhere)."""
    from collections import Counter

    from ..bench.gitstate import repo_root

    def root_of(path: str | None) -> str | None:
        if not path or not os.path.isabs(path):
            return None
        d = path if os.path.isdir(path) else os.path.dirname(path)
        while d and not os.path.isdir(d):
            d = os.path.dirname(d)
        try:
            return repo_root(d) if d else None
        except Exception:  # noqa: BLE001 - not a repo, or git missing
            return None

    here = root_of(trace.cwd)
    if here:
        return here
    roots = Counter(r for r in (root_of(p) for p in trace.edited_files) if r)
    return roots.most_common(1)[0][0] if roots else None


def _rel(path: str, cwd: str | None) -> str:
    if cwd and os.path.isabs(path):
        try:
            rel = os.path.relpath(path, cwd)
        except ValueError:
            raise Skip(f"edits a file on another drive ({path})") from None
        if rel.startswith(".."):
            raise Skip(f"works on a file outside the repository ({path})")
        return rel
    return path


HOME = os.path.expanduser("~")


def relative_command(cmd: str, root: str | None) -> str:
    """A shell command as the trained model should write it: from the repository root. A leading
    ``cd <repo> &&`` goes, paths under the repo become relative, and any other path under the home
    folder starts with ``~``, so the model never learns this machine's folders."""
    if root:
        r = re.escape(root.rstrip("/"))
        cmd = re.sub(rf"^\s*cd\s+(['\"]?){r}/?\1\s*(&&|;)\s*", "", cmd)
        cmd = re.sub(rf"(['\"]?){r}/", r"\1", cmd)
        cmd = re.sub(rf"(?<![\w./-]){r}(?=$|[\s'\";|&)])", ".", cmd)
    return _home(cmd)


def _home(text: str) -> str:
    """The home folder written as ``~`` (``/Users/me/x`` -> ``~/x``, ``/Users/me`` -> ``~``)."""
    if not HOME or HOME == "/":
        return text
    return re.sub(re.escape(HOME) + r"(?![\w.-])", "~", text)


def _search_command(t: ToolUse) -> str:
    a = t.args
    pattern = a.get("pattern") or a.get("query") or a.get("regex")
    where = a.get("path") or "."
    if t.name.lower() in ("glob", "find", "list_files", "ls") or (pattern is None and a.get("glob")):
        glob = a.get("pattern") or a.get("glob") or "*"
        return f"rg --files {shlex.quote(where)} -g {shlex.quote(glob)}"
    if pattern is None:
        return f"ls {shlex.quote(where)}"
    flags = " -i" if a.get("-i") or a.get("ignore_case") else ""
    glob = f" -g {shlex.quote(a['glob'])}" if a.get("glob") else ""
    mode = " -l" if a.get("output_mode") == "files_with_matches" else " -n"
    return f"rg{mode}{flags}{glob} {shlex.quote(str(pattern))} {shlex.quote(where)}"


_HUNK = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")


def patch_calls(patch: str) -> list[tuple[str, dict[str, Any]]]:
    """An apply_patch body (``*** Begin Patch`` format) as write_file / edit_file / bash calls."""
    calls: list[tuple[str, dict[str, Any]]] = []
    cur: tuple[str, str] | None = None
    body: list[str] = []

    def flush() -> None:
        if cur is None:
            return
        kind, path = cur
        if kind == "Add":
            calls.append(("write_file", {"path": path, "content": "".join(ln[1:] + "\n" for ln in body
                                                                       if ln.startswith("+"))}))
        elif kind == "Delete":
            calls.append(("bash", {"command": f"rm {shlex.quote(path)}"}))
        else:
            chunks: list[list[str]] = [[]]
            for ln in body:
                if ln.startswith("@@"):
                    chunks.append([])
                elif ln.startswith("*** Move to:"):
                    raise Skip("a patch that renames a file")
                elif ln.startswith(("+", "-", " ")) or ln == "":
                    chunks[-1].append(ln)
            for ch in chunks:
                if not ch:
                    continue
                old = "".join((ln[1:] if ln else "") + "\n" for ln in ch if not ln.startswith("+"))
                new = "".join((ln[1:] if ln else "") + "\n" for ln in ch if not ln.startswith("-"))
                if old == new:
                    continue
                if not old.strip():
                    raise Skip("a patch hunk with no context to anchor it")
                calls.append(("edit_file", {"path": path, "old_string": old.rstrip("\n"),
                                            "new_string": new.rstrip("\n")}))

    for ln in patch.splitlines():
        m = _HUNK.match(ln)
        if m:
            flush()
            cur, body = (m.group(1), m.group(2).strip()), []
        elif ln.startswith("*** End Patch") or ln.startswith("*** Begin Patch"):
            continue
        elif cur is not None:
            body.append(ln)
    flush()
    if not calls:
        raise Skip("a patch with no changes this runtime can make")
    return calls


def translate(t: ToolUse, cwd: str | None) -> list[tuple[str, dict[str, Any], str]]:
    """(tool, arguments, result) for the runtime, for one tool use of any agent."""
    out = _home(clip(t.output or ""))
    if t.kind == "shell":
        cmd = t.args.get("command")
        cmd = " ".join(map(str, cmd)) if isinstance(cmd, list) else cmd
        if not isinstance(cmd, str) or not cmd.strip():
            raise Skip("a shell call with no command")
        code = t.exit_code if t.exit_code is not None else (1 if t.is_error else 0)
        return [("bash", {"command": relative_command(cmd, cwd)}, f"exit code {code}\n{out}")]
    if t.kind == "read":
        args = {"path": _rel(str(t.args.get("path") or ""), cwd)}
        for k in ("offset", "limit"):
            if isinstance(t.args.get(k), int):
                args[k] = t.args[k]
        return [("read_file", args, out)]
    if t.kind == "write":
        return [("write_file", {"path": _rel(str(t.args.get("path") or ""), cwd),
                                "content": str(t.args.get("content") or "")}, out or "written")]
    if t.kind == "edit":
        old, new = t.args.get("old"), t.args.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            raise Skip(f"an edit ({t.name}) without old and new text")
        return [("edit_file", {"path": _rel(str(t.args.get("path") or ""), cwd), "old_string": old,
                               "new_string": new}, out or "edited")]
    if t.kind == "patch":
        calls = patch_calls(str(t.args.get("patch") or ""))
        res = out or "done"
        return [(n, {**a, "path": _rel(a["path"], cwd)} if "path" in a else a, res) for n, a in calls]
    if t.kind == "search":
        return [("bash", {"command": relative_command(_search_command(t), cwd)},
                 f"exit code {0 if not t.is_error else 1}\n{out}")]
    raise Skip(f"uses {t.name} ({t.kind}), which the trained model will not have")


def requests(trace: Trace) -> list[tuple[str, list[Step]]]:
    """Each prompt the person typed, with the agent's main-loop steps until the next one."""
    out: list[tuple[str, list[Step]]] = []
    from .signals import INTERRUPTED

    for s in trace.steps:
        if s.source == "user" and s.text.strip():
            if INTERRUPTED.search(s.text.strip()):
                continue  # "[Request interrupted by user]" ends a request; it is not one
            out.append((s.text.strip(), []))
        elif s.source == "agent" and not s.sidechain and out:
            out[-1][1].append(s)
    return out


def example(prompt: str, steps: list[Step], cwd: str | None) -> Example:
    """One request as a chat example in the runtime's format, or ``Skip`` with the reason."""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": _home(prompt)}]
    n = 0
    last_text = ""
    for s in steps:
        if not s.tools:
            if s.text.strip():
                last_text = s.text.strip()
            continue
        calls, results = [], []
        for t in s.tools:
            if BOOKKEEPING.match(t.name):
                continue
            for name, args, result in translate(t, cwd):
                n += 1
                cid = f"call_{n}"
                calls.append({"id": cid, "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
                results.append({"role": "tool", "tool_call_id": cid, "content": result})
        if not calls:
            if s.text.strip():
                last_text = s.text.strip()
            continue
        msgs.append({"role": "assistant", "content": _home(s.text.strip()), "tool_calls": calls})
        msgs.extend(results)
    if n == 0:
        raise Skip("a reply with no tool use (a question or a chat)")
    summary = _home((last_text.splitlines() or ["done"])[0][:200])
    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{n + 1}", "type": "function",
                 "function": {"name": "finish", "arguments": json.dumps({"summary": summary})}}]})
    return {"messages": msgs, "tools": TOOLS}


def examples(trace: Trace, *, edits_only: bool = True) -> tuple[list[Example], list[dict[str, str]]]:
    """Every usable request of a trace as an example, and why each other one was left out."""
    good: list[Example] = []
    skipped: list[dict[str, str]] = []
    from . import changes_files
    from .signals import outcomes

    root = work_root(trace) or trace.cwd
    ends = outcomes(trace)
    for i, (prompt, steps) in enumerate(requests(trace)):
        where = {"trace": trace.trace_id, "agent": trace.agent, "request": str(i)}
        if edits_only and not any(changes_files(t) for s in steps for t in s.tools):
            skipped.append({**where, "reason": "changed no code"})
            continue
        try:
            ex = example(prompt, steps, root)
        except Skip as exc:
            skipped.append({**where, "reason": str(exc)})
            continue
        ex["meta"] = {**where, "cwd": root or "", "at": trace.started_at.isoformat() if trace.started_at else "",
                      "outcome": ends[i] if i < len(ends) else {}}
        good.append(ex)
    return good, skipped
