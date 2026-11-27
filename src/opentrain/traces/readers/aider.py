"""aider: ``.aider.chat.history.md`` inside each repository aider ran in.

aider keeps no central store, so discovery looks in a bounded set of folders: the working
directories of other agents' sessions (Claude Code, Codex), ``~``, and the children of
``~/conductor/repos``, ``~/code``, ``~/src``, ``~/projects``, ``~/dev``, ``~/workspace``, ``~/git``
(one level deep), plus ``$AIDER_CHAT_HISTORY_FILE``. It never crawls the disk.

One history file holds every session run in that repo, each opened by ``# aider chat started at
<time>``; each becomes its own trace, located as ``<file>#<n>``. In a session, ``#### `` lines are
what the person typed, ``> `` lines are aider's own output (commands, "Applied edit to ...",
test output), everything else is the model's reply. SEARCH/REPLACE blocks in a reply become
``edit`` ToolUses (name ``search_replace``); aider's "Applied edit to <file>" lines are attached as
their output. Whole-file and udiff edit formats are left as text.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .. import Step, ToolUse, Trace
from ._util import Source, canonical, env_dir, existing, finish, home

NAME = "aider"
FILE = ".aider.chat.history.md"
_START = re.compile(r"^# aider chat started at (.+?)\s*$", re.M)
_PARENTS = ("conductor/repos", "code", "src", "projects", "dev", "workspace", "git", "repos")


def _agent_cwds() -> list[Path]:
    """Working directories named in Claude Code and Codex logs (first lines only)."""
    out: set[str] = set()
    claude = env_dir("CLAUDE_CONFIG_DIR", home() / ".claude") / "projects"
    codex = env_dir("CODEX_HOME", home() / ".codex")
    files: list[Path] = []
    if claude.is_dir():
        files += [p for d in claude.iterdir() if d.is_dir() for p in d.glob("*.jsonl")]
    for sub in ("sessions", "archived_sessions"):
        if (codex / sub).is_dir():
            files += list((codex / sub).rglob("*.jsonl"))
    for f in files:
        try:
            with f.open(encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i > 30:
                        break
                    if '"cwd"' not in line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cwd = r.get("cwd") or (r.get("payload") or {}).get("cwd")
                    if isinstance(cwd, str):
                        out.add(cwd)
                        break
        except OSError:
            continue
    return [Path(c) for c in sorted(out)]


def roots() -> list[Path]:
    h = home()
    return existing([h / p for p in _PARENTS])


def _candidates() -> list[Path]:
    dirs: list[Path] = [home()]
    for parent in roots():
        try:
            dirs += [c for c in parent.iterdir() if c.is_dir()]
        except OSError:
            continue
    dirs += _agent_cwds()
    files = [d / FILE for d in dirs]
    extra = os.environ.get("AIDER_CHAT_HISTORY_FILE")
    if extra:
        files.append(Path(extra).expanduser())
    seen: list[Path] = []
    for f in files:
        if f.is_file() and f not in seen:
            seen.append(f)
    return seen


def discover() -> list[str]:
    out: list[str] = []
    for f in _candidates():
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n = len(_START.findall(text)) or 1
        out += [f"{f}#{i}" for i in range(n)]
    return out


def _sessions(text: str) -> list[tuple[datetime | None, str]]:
    marks = list(_START.finditer(text))
    if not marks:
        return [(None, text)]
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        try:
            at = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").astimezone(timezone.utc)
        except ValueError:
            at = None
        out.append((at, text[m.end() : end]))
    return out


_SR = re.compile(
    r"(?:^|\n)(?P<path>[^\n`]+?)\n(?:```[^\n]*\n)?<{5,9} SEARCH[^\n]*\n(?P<old>.*?)\n?={5,9}\n(?P<new>.*?)\n?>{5,9} REPLACE",
    re.S,
)
_APPLIED = re.compile(r"^Applied edit to (.+)$", re.M)


def _edits(reply: str, step_no: int) -> list[ToolUse]:
    out = []
    for i, m in enumerate(_SR.finditer(reply)):
        path = m.group("path").strip().strip("`*").strip()
        args = canonical("edit", {"path": path, "old": m.group("old"), "new": m.group("new")})
        out.append(ToolUse(f"{step_no}.{i}", "search_replace", "edit", args))
    return out


def read(loc: str) -> Trace | None:
    file, _, idx = loc.rpartition("#")
    if not file or not idx.isdigit():
        file, idx = loc, "0"
    path = Path(file)
    sessions = _sessions(path.read_text(encoding="utf-8", errors="replace"))
    at, body = sessions[int(idx)] if int(idx) < len(sessions) else (None, "")
    trace = Trace(
        trace_id=f"{path.parent.name}-{idx}", agent=NAME, path=loc, cwd=str(path.parent), started_at=at, ended_at=None
    )
    # Group consecutive lines by kind: user (####), aider output (>), model reply (rest).
    blocks: list[tuple[str, list[str]]] = []
    for line in body.splitlines():
        if line.startswith("#### ") or line == "####":
            kind, content = "user", line[5:]
        elif line.startswith("> ") or line == ">":
            kind, content = "system", line[2:]
        else:
            kind, content = "agent", line
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(content)
        else:
            blocks.append((kind, [content]))
    last_agent: Step | None = None
    for kind, lines in blocks:
        text = "\n".join(lines).strip()
        if not text:
            continue
        if kind == "user":
            trace.steps.append(Step("user", text, at=at))
        elif kind == "agent":
            last_agent = Step("agent", text, tools=_edits(text, len(trace.steps)), at=at)
            trace.steps.append(last_agent)
        else:
            m = re.search(r"^(?:Main )?[Mm]odel: (\S+)", text, re.M)
            if m and m.group(1) not in trace.models:
                trace.models.append(m.group(1))
            trace.steps.append(Step("system", text, at=at))
            if last_agent is not None and last_agent.tools:
                for t in last_agent.tools:
                    for applied in _APPLIED.findall(text):
                        if applied.strip() == t.args.get("path"):
                            t.output = f"Applied edit to {applied.strip()}"
    if at is not None:
        trace.ended_at = at
    return finish(trace)


SOURCE = Source(NAME, "aider", roots, discover, read)
