"""Tools: what the engine's model can call, and how a call is run.

A tool is a name, a description, a JSON schema and an ``execute`` function (sync functions run in
a worker thread). ``permission`` maps an input to the (action, resource) pairs the call needs; the
permission service decides allow / ask / deny for each before ``execute`` runs. Outputs are
clipped, head and tail kept, so one noisy command cannot flood the context.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .project import Project

MAX_OUTPUT = 30_000


class ToolError(Exception):
    """A refusal or failure the model should read and adapt to."""


@dataclass
class ToolResult:
    output: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    project: "Project"
    session_id: str
    message_id: str
    call_id: str
    agent: Any  # agent.Agent
    part_id: str = ""
    abort: asyncio.Event = field(default_factory=asyncio.Event)
    loop: asyncio.AbstractEventLoop | None = None

    def progress(self, **metadata: Any) -> None:
        """Update this call's live metadata (shown by clients while it runs)."""
        if self.part_id:
            self.project.store.update("part", self.part_id, {"tool": {"metadata": metadata}})

    async def ask(self, action: str, resources: list[str], *, metadata: dict | None = None,
                  always: list[str] | None = None) -> None:
        """Check permission; raises PermissionRejected. Pauses for the user when the answer is ask."""
        perms = self.project.services.get("permissions")
        if perms is not None:
            await perms.assert_allowed(self, action, resources, metadata=metadata or {}, always=always)

    def ask_sync(self, action: str, resources: list[str], **kw: Any) -> None:
        """``ask`` from a tool running in a worker thread."""
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.ask(action, resources, **kw), self.loop).result()


Execute = Callable[..., Any]


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    execute: Execute  # (ctx, **input) -> ToolResult | str | dict
    permission: Callable[[dict[str, Any]], list[tuple[str, str]]] | None = None
    title: Callable[[dict[str, Any]], str] | None = None
    group: str = "general"  # "general" | "train" | "mcp"

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "schema": self.schema, "group": self.group}


def schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-limit // 3:]
    return f"{head}\n\n... [{len(text) - len(head) - len(tail)} characters cut] ...\n\n{tail}"


def _normalize(result: Any) -> ToolResult:
    if isinstance(result, ToolResult):
        return ToolResult(clip(result.output), result.title, result.metadata)
    if isinstance(result, str):
        return ToolResult(clip(result))
    return ToolResult(clip(json.dumps(result, indent=1, default=str)))


def validate(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Light schema check: required keys present, no unknown keys when additionalProperties is false."""
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    if "_invalid_json" in args:
        raise ToolError("the arguments were not valid JSON; send them again")
    props = tool.schema.get("properties", {})
    missing = [k for k in tool.schema.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    if tool.schema.get("additionalProperties") is False:
        extra = [k for k in args if k not in props]
        if extra:
            raise ToolError(f"unknown argument(s): {', '.join(extra)}; expected {', '.join(props) or 'none'}")
    return args


async def run(tool: Tool, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    args = validate(tool, args)
    for action, resource in (tool.permission(args) if tool.permission else [(tool.name, "*")]):
        await ctx.ask(action, [resource], metadata={"tool": tool.name, "input": args})
    if inspect.iscoroutinefunction(tool.execute):
        result = await tool.execute(ctx, **args)
    else:
        ctx.loop = asyncio.get_running_loop()
        result = await asyncio.to_thread(tool.execute, ctx, **args)
    return _normalize(result)


class Registry:
    """The tools available in one project: built-ins, train tools, and MCP servers' tools."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def add(self, *tools: Tool) -> None:
        for t in tools:
            self.tools[t.name] = t

    def remove(self, name: str) -> None:
        self.tools.pop(name, None)

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ToolError(f"unknown tool {name!r}")
        return self.tools[name]

    def for_agent(self, agent: Any, perms: Any = None) -> list[Tool]:
        """Tools this agent may see: any whose whole action is denied to it is hidden."""
        out = []
        for t in self.tools.values():
            if perms is not None and perms.hidden(agent, t.name):
                continue
            out.append(t)
        return sorted(out, key=lambda t: (t.group != "general", t.name))
