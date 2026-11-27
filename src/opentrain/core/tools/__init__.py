"""The tools the engine's model can call: general ones (shell, files, search, todo, question,
skill, subagent, webfetch), the dataset tools (any data, graders, evals, training, the gate) and
the coding-session tools (mining tasks from agent sessions, sampling, jobs, versions)."""

from __future__ import annotations

from typing import Any


def register_all(project: Any) -> None:
    from . import artifacts, coding, data, general, train

    reg = project.services["tools"]
    reg.add(*general.tools())
    reg.add(*data.tools())
    reg.add(*train.tools())
    reg.add(*coding.tools())
    reg.add(*artifacts.tools())
