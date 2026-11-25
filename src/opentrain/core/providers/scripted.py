"""A provider that plays back a script: for tests and demos, never for real work.

``ScriptedProvider.register(name, turns)`` makes ``scripted/<name>`` available. Each turn is a list
of events: strings become text, ``{"tool": name, "input": {...}}`` becomes a tool call. When the
script runs out it answers "done".
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from ..llm import Delta, Event, Finish, Request, ToolCall, Usage

_SCRIPTS: dict[str, "ScriptedProvider"] = {}


class ScriptedProvider:
    kind = "step"

    def __init__(self, name: str, turns: list[list[Any]], delay: float = 0.0):
        self.id, self.name, self.turns, self.delay = "scripted", name, list(turns), delay
        self.requests: list[Request] = []

    @classmethod
    def register(cls, name: str, turns: list[list[Any]], delay: float = 0.0) -> "ScriptedProvider":
        _SCRIPTS[name] = cls(name, turns, delay)
        return _SCRIPTS[name]

    @classmethod
    def named(cls, name: str) -> "ScriptedProvider":
        if name not in _SCRIPTS:
            _SCRIPTS[name] = cls(name, [])
        return _SCRIPTS[name]

    async def stream(self, req: Request) -> AsyncIterator[Event]:
        self.requests.append(req)
        turn = self.turns.pop(0) if self.turns else ["done"]
        calls = 0
        for i, item in enumerate(turn):
            if self.delay:
                await asyncio.sleep(self.delay)
            if isinstance(item, str):
                for word in item.split(" "):
                    yield Delta(word + " ")
            elif isinstance(item, dict) and "tool" in item:
                calls += 1
                yield ToolCall(item.get("id") or f"call_{len(self.requests)}_{i}", item["tool"], item.get("input", {}))
        yield Usage(input=100, output=20, cost=0.0)
        yield Finish("tool_calls" if calls else "stop")
