"""The tools the engine's model can call: general ones (shell, files, search, todo, question,
skill, subagent, webfetch) and the train tools (tasks, sampling, data, jobs, evals, versions)."""

from __future__ import annotations

from typing import Any


def register_all(project: Any) -> None:
    from . import general, train

    reg = project.services["tools"]
    reg.add(*general.tools())
    reg.add(*train.tools())
