"""Agents: who the engine is in a session — its instructions, model, step limit and permissions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Agent:
    name: str
    description: str
    prompt: str
    mode: str = "primary"  # "primary" | "subagent" | "all"
    model: str | None = None
    steps: int = 200
    permissions: list[dict[str, Any]] = field(default_factory=list)
    hidden: bool = False
    color: str | None = None
    source: str = "built-in"

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "mode": self.mode, "model": self.model,
                "steps": self.steps, "hidden": self.hidden, "color": self.color, "source": self.source,
                "permissions": self.permissions}


TRAIN = Agent("train", "Post-trains models for this project, within the Trainfile's budget and gate.",
              "You are OpenTrain's train agent. You post-train models for the user.")


class Agents:
    def __init__(self, project: Any = None):
        self.project = project
        self.agents: dict[str, Agent] = {"train": TRAIN}

    def get(self, name: str | None) -> Agent:
        return self.agents.get(name or "train") or self.agents["train"]

    def list(self) -> list[Agent]:
        return list(self.agents.values())

    def default(self) -> str:
        return "train"
