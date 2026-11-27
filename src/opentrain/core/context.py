"""What the engine's model is told: the agent's instructions, the environment, the policy, and
the project's own instructions (AGENTS.md / CLAUDE.md)."""

from __future__ import annotations

import platform
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md")


def _git(directory: Path) -> str:
    try:
        branch = subprocess.run(["git", "-C", str(directory), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        return f"git repo, branch {branch}" if branch else "not a git repo"
    except (OSError, subprocess.SubprocessError):
        return "not a git repo"


def instructions(directory: Path) -> list[tuple[str, str]]:
    """Project instruction files, nearest first; AGENTS.md wins over CLAUDE.md in the same folder."""
    out = []
    for d in [directory, *directory.parents][:4]:
        for name in INSTRUCTION_FILES:
            f = d / name
            if f.is_file():
                out.append((str(f), f.read_text(errors="replace")[:20_000]))
                break
        if (d / ".git").exists():
            break
    return out


def environment(project: Any) -> str:
    info = project.info()
    pol = info.get("policy") or {}
    lines = [
        f"Working directory: {project.directory}",
        f"Project state: {project.state} (only the server's tools write here)",
        f"Repository: {_git(project.directory)}",
        f"Platform: {platform.system()} {platform.machine()}",
        f"Today: {date.today().isoformat()}",
    ]
    if pol:
        lines.append("Trainfile policy (enforced by the server; you cannot change it): "
                     f"goal {pol['goal']}, budget {pol['budget']}, trainer {pol['trainer']}, base model "
                     f"{pol['base_model']}, held-out {int(pol['heldout'] * 100)}%, eval cap {pol['eval_cap']} per run.")
    elif info.get("policy_error"):
        lines.append(f"Trainfile problem: {info['policy_error']}")
    else:
        lines.append("No Trainfile yet: the defaults apply until the user writes one (the `init` tool can).")
    return "\n".join(lines)


def configured_instructions(directory: Path) -> list[tuple[str, str]]:
    """Extra instruction files named under ``instructions`` in opentrain.jsonc (paths or globs)."""
    from . import config

    try:
        names = config.load(directory).get("instructions") or []
    except ValueError:
        return []
    out = []
    for n in names:
        for f in sorted(directory.glob(n)) if not Path(n).expanduser().is_absolute() else [Path(n).expanduser()]:
            if f.is_file():
                out.append((str(f), f.read_text(errors="replace")[:20_000]))
    return out


def system_prompt(project: Any, agent: Any, extra: list[str] | None = None) -> str:
    from .skills import Skills

    parts = [agent.prompt.strip(), "# Environment\n" + environment(project)]
    index = Skills(project.directory).index()
    if index:
        parts.append("# Skills (load one with the skill tool before running that recipe)\n" + index)
    for path, text in instructions(project.directory) + configured_instructions(project.directory):
        parts.append(f"# Instructions from {path}\n{text.strip()}")
    parts.extend(extra or [])
    return "\n\n".join(p for p in parts if p)
