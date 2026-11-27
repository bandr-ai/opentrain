"""Provider-neutral model calls: one request shape in, one stream of events out.

The engine never sees a vendor format. A step sends ``Request`` (system prompt, neutral messages,
tool specs) to a provider and reads a stream of ``Delta`` / ``Reasoning`` / ``ToolCall`` /
``Usage`` / ``Finish`` events. Providers translate both ways.

Neutral messages::

    {"role": "user", "content": [{"type": "text", "text": ...}]}
    {"role": "assistant", "content": [{"type": "text", "text": ...},
                                      {"type": "tool_call", "id": ..., "name": ..., "input": {...}}]}
    {"role": "tool", "content": [{"type": "tool_result", "id": ..., "output": "...", "is_error": False}]}

Some providers are agents themselves (Claude Code, Codex): they run their own loop and call our
tools back through the MCP bridge. Those implement ``AgentProvider`` instead of ``Provider``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, Union


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]


@dataclass
class Request:
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int = 8192
    tool_choice: str = "auto"  # "auto" | "none"


@dataclass
class Delta:
    text: str


@dataclass
class Reasoning:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float | None = None  # dollars, when the provider reports or we can price it


@dataclass
class Finish:
    reason: str  # "stop" | "tool_calls" | "length" | "error"
    error: str | None = None


@dataclass
class Native:
    """Provider-specific replay data for this assistant turn (e.g. Anthropic content blocks)."""

    data: dict[str, Any]


Event = Union[Delta, Reasoning, ToolCall, Usage, Native, Finish]


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable, self.status = retryable, status


class Provider(Protocol):
    id: str
    kind: str  # "step"

    def stream(self, req: Request) -> AsyncIterator[Event]: ...


@dataclass
class Model:
    """One entry of the model catalogue."""

    provider: str
    id: str
    name: str
    context: int = 200_000
    output: int = 8192
    input_price: float = 0.0  # $ per million tokens
    output_price: float = 0.0
    reasoning: bool = False

    @property
    def ref(self) -> str:
        return f"{self.provider}/{self.id}"

    def price(self, u: Usage) -> float:
        return (u.input * self.input_price + u.output * self.output_price
                + u.cache_read * self.input_price * 0.1 + u.cache_write * self.input_price * 1.25) / 1e6


def parse_ref(ref: str) -> tuple[str, str]:
    """``provider/model`` -> (provider, model). The model part may itself contain slashes."""
    provider, sep, model = ref.partition("/")
    if not sep or not provider or not model:
        raise ValueError(f"model must be provider/model, got {ref!r}")
    return provider, model


def tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)
