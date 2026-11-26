"""General tools: shell, read, write, edit, glob, grep, todo, question, skill, subagent, webfetch."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ..proposals import GRANTABLE, propose_tool
from ..tool import Tool, ToolContext, ToolError, ToolResult, schema
from .paths import guard, rel, resolve, sealed

MAX_READ_LINES = 2000
MAX_LINE = 2000


# ---------------------------------------------------------------------- shell


async def _shell(ctx: ToolContext, command: str, timeout: float = 120, description: str = "",
                 workdir: str | None = None) -> ToolResult:
    cwd = resolve(ctx.project, workdir) if workdir else ctx.project.directory
    if not cwd.is_dir():
        raise ToolError(f"no such directory: {workdir}")
    if any(f"{ctx.project.state.name}/{s}" in command for s in ("heldout", "opentrain.db")):
        raise ToolError("held-out tasks and the run record are sealed; use the train tools")
    timeout = max(1.0, min(float(timeout), 3600.0))
    from ..sandbox import wrap

    proc = await asyncio.create_subprocess_exec(*wrap(["bash", "-lc", command]), cwd=str(cwd),
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
                                                env={**os.environ, "OPENTRAIN_AGENT": ctx.agent.name})
    chunks: list[bytes] = []
    last = 0.0
    timed_out = False

    async def pump() -> None:
        nonlocal last
        assert proc.stdout is not None
        async for line in proc.stdout:
            chunks.append(line)
            now = time.monotonic()
            if now - last > 0.5:
                last = now
                ctx.progress(output=b"".join(chunks)[-4000:].decode(errors="replace"))

    try:
        await asyncio.wait_for(pump(), timeout)
        await proc.wait()
    except asyncio.TimeoutError:
        timed_out = True
    except asyncio.CancelledError:
        _kill(proc)
        raise
    finally:
        if proc.returncode is None:
            _kill(proc)
            await proc.wait()
    out = b"".join(chunks).decode(errors="replace")
    if timed_out:
        out += f"\n[timed out after {timeout:.0f}s and was killed]"
    code = proc.returncode
    return ToolResult(out + (f"\n[exit code {code}]" if code else ""), title=description or command[:80],
                      metadata={"exit": code, "timed_out": timed_out})


def _kill(proc: Any) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------------- files


def _read(ctx: ToolContext, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> ToolResult:
    p = resolve(ctx.project, path)
    guard(ctx.project, p)
    if p.is_dir():
        entries = sorted(os.listdir(p))
        return ToolResult("\n".join(e + ("/" if (p / e).is_dir() else "") for e in entries[:500]),
                          title=rel(ctx.project, p))
    if not p.exists():
        raise ToolError(f"no such file: {path}")
    data = p.read_bytes()
    if b"\0" in data[:8000]:
        raise ToolError(f"{path} is a binary file ({len(data)} bytes)")
    lines = data.decode(errors="replace").splitlines()
    offset = max(1, int(offset))
    chunk = lines[offset - 1: offset - 1 + int(limit)]
    body = "\n".join(f"{i:>6}\t{(ln if len(ln) <= MAX_LINE else ln[:MAX_LINE] + '…')}"
                     for i, ln in enumerate(chunk, start=offset))
    more = len(lines) - (offset - 1 + len(chunk))
    if more > 0:
        body += f"\n\n[{more} more lines; read again with offset={offset + len(chunk)}]"
    ctx.project.services.setdefault("read_files", {}).setdefault(ctx.session_id, set()).add(str(p))
    return ToolResult(body or "(empty file)", title=rel(ctx.project, p), metadata={"lines": len(lines)})


def _must_have_read(ctx: ToolContext, p: Path) -> None:
    seen = ctx.project.services.get("read_files", {}).get(ctx.session_id, set())
    if p.exists() and str(p) not in seen:
        raise ToolError(f"read {rel(ctx.project, p)} before changing it")


def _write(ctx: ToolContext, path: str, content: str) -> ToolResult:
    p = resolve(ctx.project, path)
    guard(ctx.project, p)
    _must_have_read(ctx, p)
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    ctx.project.services.setdefault("read_files", {}).setdefault(ctx.session_id, set()).add(str(p))
    return ToolResult(f"{'Overwrote' if existed else 'Created'} {rel(ctx.project, p)} ({len(content)} chars)",
                      title=rel(ctx.project, p))


def _edit(ctx: ToolContext, path: str, old: str, new: str, replace_all: bool = False) -> ToolResult:
    p = resolve(ctx.project, path)
    guard(ctx.project, p)
    if not p.exists():
        raise ToolError(f"no such file: {path}")
    _must_have_read(ctx, p)
    text = p.read_text()
    n = text.count(old)
    if old == new:
        raise ToolError("old and new are the same")
    if n == 0:
        raise ToolError("old text not found; copy it exactly from a fresh read")
    if n > 1 and not replace_all:
        raise ToolError(f"old text occurs {n} times; add surrounding lines or set replace_all")
    p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1))
    return ToolResult(f"Edited {rel(ctx.project, p)} ({n if replace_all else 1} replacement)", title=rel(ctx.project, p))


SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".opentrain", ".mypy_cache", ".pytest_cache",
             "dist", "build", ".next", "target"}


def _walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            yield Path(dirpath) / f


def _glob(ctx: ToolContext, pattern: str, path: str | None = None) -> ToolResult:
    root = resolve(ctx.project, path) if path else ctx.project.directory
    out = []
    for f in _walk(root):
        r = str(f.relative_to(root))
        if fnmatch.fnmatch(r, pattern) or fnmatch.fnmatch(f.name, pattern):
            out.append(f)
    out.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    lines = [rel(ctx.project, f) for f in out[:300]]
    return ToolResult("\n".join(lines) + (f"\n[{len(out) - 300} more]" if len(out) > 300 else "") or "no matches",
                      title=pattern, metadata={"count": len(out)})


def _grep(ctx: ToolContext, pattern: str, path: str | None = None, include: str | None = None) -> ToolResult:
    root = resolve(ctx.project, path) if path else ctx.project.directory
    guard(ctx.project, root)
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "-n", "--no-heading", "--max-columns", "300", "--glob", "!.opentrain", pattern, str(root)]
        if include:
            cmd[1:1] = ["--glob", include]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode not in (0, 1):
            raise ToolError(r.stderr.strip() or "grep failed")
        lines = r.stdout.splitlines()
    else:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"bad pattern: {exc}") from None
        lines = []
        for f in _walk(root):
            if include and not fnmatch.fnmatch(f.name, include):
                continue
            try:
                for i, ln in enumerate(f.read_text(errors="strict").splitlines(), 1):
                    if rx.search(ln):
                        lines.append(f"{f}:{i}:{ln[:300]}")
            except (UnicodeDecodeError, OSError):
                continue
    base = str(ctx.project.directory) + os.sep
    lines = [ln.replace(base, "", 1) for ln in lines if not sealed(ctx.project, Path(ln.split(":", 1)[0]))]
    return ToolResult("\n".join(lines[:400]) + (f"\n[{len(lines) - 400} more]" if len(lines) > 400 else "")
                      or "no matches", title=pattern, metadata={"count": len(lines)})


# ---------------------------------------------------------------------- todo, question


def _todo(ctx: ToolContext, todos: list[dict[str, Any]]) -> ToolResult:
    clean = []
    for t in todos:
        if not isinstance(t, dict) or not t.get("content"):
            raise ToolError("each todo needs content and status")
        status = t.get("status", "pending")
        if status not in ("pending", "in_progress", "completed", "cancelled"):
            raise ToolError(f"bad status {status!r}")
        clean.append({"content": str(t["content"]), "status": status})
    store = ctx.project.store
    tid = f"todo_{ctx.session_id}"
    if store.get("todo", tid):
        store.publish("todo.updated", {"id": tid, "session_id": ctx.session_id, "todos": clean})
    else:
        store.create("todo", {"id": tid, "session_id": ctx.session_id, "todos": clean})
    left = sum(t["status"] in ("pending", "in_progress") for t in clean)
    return ToolResult("\n".join(f"[{'x' if t['status'] == 'completed' else ' '}] {t['content']}" for t in clean),
                      title=f"{left} to do", metadata={"todos": clean})


async def _question(ctx: ToolContext, questions: list[dict[str, Any]]) -> ToolResult:
    qs = ctx.project.services["questions"]
    answers = await qs.ask(ctx, questions)
    lines = []
    for q, a in zip(questions, answers):
        lines.append(f"Q: {q.get('question')}\nA: {', '.join(a) if a else '(no answer)'}")
    return ToolResult("\n\n".join(lines), title="asked the user", metadata={"answers": answers})


# ---------------------------------------------------------------------- skill, subagent, webfetch


def _skill(ctx: ToolContext, name: str) -> ToolResult:
    from ..skills import Skills

    sk = Skills(ctx.project.directory).get(name)
    return ToolResult(f"# Skill {sk.name}\n(from {sk.path})\n\n{sk.body}", title=sk.name)


async def _subagent(ctx: ToolContext, agent: str, prompt: str, description: str = "",
                    session_id: str | None = None) -> ToolResult:
    agents = ctx.project.services["agents"]
    a = agents.find(agent)
    if a is None or not a.subagent:
        names = ", ".join(x.name for x in agents.subagents())
        raise ToolError(f"no subagent {agent!r}; available: {names}")
    if ctx.agent.name in ("data", "monitor") or ctx.project.session(ctx.session_id).get("parent_id"):
        raise ToolError("subagents cannot start subagents")
    runner = ctx.project.services["runner"]
    parent = ctx.project.session(ctx.session_id)
    if session_id:
        child = ctx.project.session(session_id)
        if child.get("parent_id") != ctx.session_id:
            raise ToolError("that session is not a subagent of this one")
    else:
        child = ctx.project.create_session(title=description or f"{agent}: {prompt[:40]}", agent=agent,
                                           model=parent.get("model") if not a.model else None,
                                           parent_id=ctx.session_id)
    ctx.progress(session_id=child["id"], agent=agent)
    runner.prompt(child["id"], prompt)
    await asyncio.sleep(0)
    try:
        await runner.wait_idle(child["id"])
    except asyncio.CancelledError:
        await runner.interrupt(child["id"])
        raise
    msgs = [m for m in ctx.project.messages(child["id"]) if m["role"] == "assistant"]
    text = "\n".join(p.get("text", "") for m in msgs[-1:] for p in m["parts"] if p["type"] == "text").strip()
    errs = [m.get("error") for m in msgs if m.get("error")]
    return ToolResult((text or "(the subagent returned no text)") + (f"\n\nErrors: {errs[-1]}" if errs else "")
                      + f"\n\n[subagent session {child['id']}]", title=description or agent,
                      metadata={"session_id": child["id"], "agent": agent})


async def _webfetch(ctx: ToolContext, url: str) -> ToolResult:
    import html as htmlmod

    import aiohttp

    if not url.startswith(("http://", "https://")):
        raise ToolError("url must start with http:// or https://")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
        async with s.get(url, headers={"User-Agent": "opentrain"}) as r:
            if r.status >= 400:
                raise ToolError(f"HTTP {r.status}")
            text = await r.text(errors="replace")
            ctype = r.headers.get("Content-Type", "")
    if "html" in ctype:
        text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = htmlmod.unescape(re.sub(r"[ \t]+", " ", text))
        text = re.sub(r"\n\s*\n+", "\n\n", text)
    return ToolResult(text.strip()[:60_000], title=url)


def _path_perm(action: str):
    def perm(args: dict[str, Any]) -> list[tuple[str, str]]:
        return [(action, args.get("path") or args.get("pattern") or "*")]
    return perm


def tools() -> list[Tool]:
    return [
        Tool("shell", "Run a bash command in the project directory. Use for git, tests, inspecting repos and logs. "
             "Not for anything that spends money or touches evals: use the train tools.",
             schema({"command": {"type": "string"}, "timeout": {"type": "number", "description": "seconds, default 120"},
                     "description": {"type": "string", "description": "5-10 words on what it does"},
                     "workdir": {"type": "string"}}, ["command"]),
             _shell, permission=lambda a: [("shell", a["command"])], title=lambda a: a.get("description") or a["command"][:80]),
        Tool("read", "Read a file (line-numbered) or list a directory. Paths are relative to the project.",
             schema({"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
             _read, permission=_path_perm("read"), title=lambda a: a["path"]),
        Tool("write", "Create or overwrite a file. Read an existing file first.",
             schema({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
             _write, permission=_path_perm("write"), title=lambda a: a["path"]),
        Tool("edit", "Replace exact text in a file. old must match exactly once unless replace_all.",
             schema({"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
                     "replace_all": {"type": "boolean"}}, ["path", "old", "new"]),
             _edit, permission=_path_perm("edit"), title=lambda a: a["path"]),
        Tool("glob", "Find files by name pattern (e.g. **/*.py), newest first.",
             schema({"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
             _glob, permission=lambda a: [("glob", a.get("path") or "*")], title=lambda a: a["pattern"]),
        Tool("grep", "Search file contents with a regular expression.",
             schema({"pattern": {"type": "string"}, "path": {"type": "string"}, "include": {"type": "string"}}, ["pattern"]),
             _grep, permission=lambda a: [("grep", a.get("path") or "*")], title=lambda a: a["pattern"]),
        Tool("todo", "Keep a short checklist for multi-step work. Send the whole list each time.",
             schema({"todos": {"type": "array", "items": {"type": "object", "properties": {
                 "content": {"type": "string"}, "status": {"type": "string",
                                                           "enum": ["pending", "in_progress", "completed", "cancelled"]}},
                 "required": ["content", "status"]}}}, ["todos"]),
             _todo),
        Tool("question", "Ask the user one or more multiple-choice questions and wait for the answers. Use for real "
             "decisions only.",
             schema({"questions": {"type": "array", "items": {"type": "object", "properties": {
                 "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}},
                 "multiple": {"type": "boolean"}}, "required": ["question", "options"]}}}, ["questions"]),
             _question),
        Tool("propose", "Propose the next step and wait for the user to approve, change or skip it. Use this for "
             "anything that costs money, sends data out, trains or changes what is live: one step at a time, with "
             "what it is, why, the cost and how long. Approving grants the gated actions listed in needs for this step "
             "only, so the user is not asked again for each of them.",
             schema({"title": {"type": "string", "description": "the step, imperative and short: 'Evaluate v2 on "
                                                                   "the 24 held-out rows'"},
                     "why": {"type": "string", "description": "one or two sentences: why this is the next step"},
                     "cost_usd": {"type": "number", "description": "estimated spend in dollars (0 if free)"},
                     "minutes": {"type": "number", "description": "rough wall time"},
                     "needs": {"type": "array", "items": {"type": "string", "enum": list(GRANTABLE)},
                               "description": "gated actions this step will take"},
                     "stage": {"type": "string", "enum": ["goal", "data", "grade", "baseline", "train", "prove",
                                                          "ship"]},
                     "dataset": {"type": "string"}}, ["title"]),
             propose_tool, title=lambda a: a["title"]),
        Tool("skill", "Load a recipe (skill) by name, e.g. goal/speed, data/shortest, train/sft, cl/mine, "
             "bandits/certify. Read it before running that recipe.",
             schema({"name": {"type": "string"}}, ["name"]), _skill, title=lambda a: a["name"]),
        Tool("subagent", "Hand a self-contained job to a subagent (data: mine and repair tasks; monitor: watch jobs). "
             "It runs in its own session and returns its final report. Pass session_id to follow up.",
             schema({"agent": {"type": "string"}, "prompt": {"type": "string"}, "description": {"type": "string"},
                     "session_id": {"type": "string"}}, ["agent", "prompt"]),
             _subagent, title=lambda a: a.get("description") or a["agent"]),
        Tool("webfetch", "Fetch a web page as text (docs, model cards, papers).",
             schema({"url": {"type": "string"}}, ["url"]), _webfetch,
             permission=lambda a: [("webfetch", a["url"])], title=lambda a: a["url"]),
    ]
