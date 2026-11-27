"""Slash commands: prompt templates in Markdown.

Bundled ones live in ``opentrain/slash/``; users add or override them in
``~/.config/opentrain/commands/`` and ``<project>/.opentrain/commands/`` (nested folders give
``/team/review``), or under ``commands`` in ``opentrain.jsonc``. Front matter may set
``description``, ``agent`` and ``model``. In the body, ``$ARGUMENTS`` is everything after the
command and ``$1``, ``$2``... are its words; arguments are appended when the body names none.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import parse_markdown

BUILTIN = Path(__file__).resolve().parent.parent / "slash"


@dataclass
class Command:
    name: str
    description: str
    template: str
    agent: str | None = None
    model: str | None = None
    source: str = "built-in"

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "agent": self.agent, "model": self.model,
                "source": self.source}

    def expand(self, arguments: str) -> str:
        try:
            words = shlex.split(arguments)
        except ValueError:
            words = arguments.split()
        body = self.template
        named = "$ARGUMENTS" in body or any(f"${i}" in body for i in range(1, 10))
        for i in range(9, 0, -1):
            body = body.replace(f"${i}", words[i - 1] if i <= len(words) else "")
        body = body.replace("$ARGUMENTS", arguments)
        if not named and arguments:
            body = f"{body.rstrip()}\n\n{arguments}"
        return body.strip()


def load(directory: Path | None) -> dict[str, Command]:
    from . import config

    out: dict[str, Command] = {}
    dirs = [(BUILTIN, "built-in"), (config.global_dir() / "commands", "global")]
    if directory:
        dirs.append((directory / ".opentrain" / "commands", "project"))
    for d, kind in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.md")):
            name = str(f.relative_to(d).with_suffix(""))
            try:
                meta, body = parse_markdown(f.read_text(), str(f))
            except Exception:
                continue
            out[name] = Command(name, meta.get("description") or "", body, meta.get("agent"), meta.get("model"),
                                "built-in" if kind == "built-in" else str(f))
    try:
        cfg = config.load(directory)
    except ValueError:
        cfg = {}
    for name, spec in (cfg.get("commands") or {}).items():
        if isinstance(spec, dict) and spec.get("template"):
            out[name] = Command(name, spec.get("description", ""), spec["template"], spec.get("agent"),
                                spec.get("model"), "opentrain.jsonc")
    return out
