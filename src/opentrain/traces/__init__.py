"""Coding-agent traces from every agent on the machine, in one shape.

Every coding agent keeps its conversations somewhere on disk (Claude Code under
``~/.claude/projects``, Codex under ``~/.codex/sessions``, opencode in a SQLite file, ...). A
reader per agent turns one log into a :class:`Trace`: the person's prompts, the agent's messages,
its tool calls with their arguments and results, in order, with timestamps where the log has them.
The shape follows Harbor's agent trajectory format (ATIF) closely enough to export to it.

Tool calls also carry a ``kind`` so recipes need not know each agent's tool names: ``shell``
(``command``), ``read`` (``path``), ``edit`` (``path``, ``old``, ``new``), ``write`` (``path``,
``content``), ``patch`` (``patch``: an apply_patch or unified diff body), ``search``, ``web``,
``task`` (a subagent), ``other``. ``args`` keeps the agent's own arguments untouched.

Everything here runs locally. Nothing a reader returns leaves the machine unless the person
trains on a cloud trainer, and then only after ``opentrain.traces.scrub`` removes credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

ToolKind = Literal["shell", "read", "edit", "write", "patch", "search", "web", "task", "other"]
EDIT_KINDS = ("edit", "write", "patch")

# A shell command that writes files: how agents (and people) often edit without an edit tool.
SHELL_EDIT = re.compile(r"(\bsed\s+-i|\bperl\s+-p?i|\bcat\s*>|\btee\s+(-a\s+)?[\w./-]|\bpython3?\s+-\s*<<|"
                        r"\bapply_patch\b|\bgit\s+apply\b|\bpatch\s+-p|>\s*[\w./-]+\.\w{1,5}\b)")


def changes_files(t: "ToolUse") -> bool:
    """An edit tool, or a shell command that writes files."""
    if t.kind in EDIT_KINDS:
        return True
    cmd = t.args.get("command")
    cmd = " ".join(map(str, cmd)) if isinstance(cmd, list) else cmd
    return t.kind == "shell" and isinstance(cmd, str) and bool(SHELL_EDIT.search(cmd))


@dataclass
class ToolUse:
    id: str
    name: str
    """The agent's own tool name (``Bash``, ``exec_command``, ``apply_patch``, ...)."""
    kind: ToolKind
    args: dict[str, Any]
    """Canonical arguments for ``kind`` (see the module docstring) merged over the agent's own."""
    output: str | None = None
    """The result as text; None when the log never recorded one."""
    is_error: bool = False
    exit_code: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


@dataclass
class Step:
    source: Literal["user", "agent", "system"]
    """user: something the person typed. agent: a model turn (text and/or tool calls). system: a
    harness notice injected as a user turn (task notifications, compaction summaries)."""
    text: str = ""
    tools: list[ToolUse] = field(default_factory=list)
    at: datetime | None = None
    model: str | None = None
    sidechain: bool = False
    """Run by a subagent rather than the main loop."""


@dataclass
class Trace:
    trace_id: str
    agent: str
    """Reader name: ``claude-code``, ``codex``, ``opencode``, ``gemini-cli``, ``cursor``, ..."""
    path: str
    cwd: str | None
    started_at: datetime | None
    ended_at: datetime | None
    steps: list[Step] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    git_branch: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def prompts(self) -> list[str]:
        return [s.text for s in self.steps if s.source == "user" and s.text.strip()]

    def tool_uses(self, *, main_only: bool = True) -> list[ToolUse]:
        return [t for s in self.steps if s.source == "agent" and not (main_only and s.sidechain) for t in s.tools]

    @property
    def one_shot(self) -> bool:
        """A single prompt and no tool use: a script's ``claude -p`` call, a quick question, not a work session."""
        return len(self.prompts) <= 1 and not self.tool_uses(main_only=False)

    @property
    def changed_code(self) -> bool:
        return any(changes_files(t) for t in self.tool_uses())

    @property
    def edited_files(self) -> list[str]:
        out: list[str] = []
        for t in self.tool_uses():
            if t.kind in EDIT_KINDS:
                for p in _paths(t):
                    if p not in out:
                        out.append(p)
        return out


_PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$|^\+\+\+ b/(.+)$", re.MULTILINE)


def _paths(t: ToolUse) -> list[str]:
    if t.kind == "patch":
        return [a.strip() or b.strip() for a, b in _PATCH_FILE.findall(t.args.get("patch") or "")]
    p = t.args.get("path")
    return [p] if isinstance(p, str) and p else []
