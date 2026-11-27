"""The Anthropic Messages API through the official SDK, streamed.

Assistant turns are replayed with their native content blocks (thinking included, unchanged), so
preserved thinking and prompt caching keep working across steps.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..llm import Delta, Event, Finish, Native, ProviderError, Reasoning, Request, ToolCall, Usage


def _messages(req: Request) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in req.messages:
        if m["role"] == "assistant":
            native = m.get("native") or {}
            if native.get("provider") == "anthropic" and native.get("content"):
                content = native["content"]
            else:
                content = []
                for c in m["content"]:
                    if c["type"] == "text" and c["text"]:
                        content.append({"type": "text", "text": c["text"]})
                    elif c["type"] == "tool_call":
                        content.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]})
            if content:
                out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c["id"], "content": c["output"] or "(no output)",
                 "is_error": bool(c.get("is_error"))} for c in m["content"]]})
        else:
            out.append({"role": "user", "content": [{"type": "text", "text": c["text"]} for c in m["content"]
                                                    if c.get("text")] or [{"type": "text", "text": "(empty)"}]})
    # Consecutive user turns (a tool result then a steering message) are merged: the API wants alternation.
    merged: list[dict[str, Any]] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] == "user":
            merged[-1] = {"role": "user", "content": list(merged[-1]["content"]) + list(m["content"])}
        else:
            merged.append(m)
    return merged


class AnthropicProvider:
    id = "anthropic"
    kind = "step"

    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError:
            raise ProviderError("the anthropic package is not installed: pip install 'opentrain-cli[anthropic]'") from None
        self._anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def stream(self, req: Request) -> AsyncIterator[Event]:
        a = self._anthropic
        tools = [{"name": t.name, "description": t.description, "input_schema": t.schema,
                  "eager_input_streaming": True} for t in req.tools]
        kwargs: dict[str, Any] = {"model": req.model, "max_tokens": max(req.max_tokens, 16_000),
                                  "system": [{"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}],
                                  "messages": _messages(req),
                                  "thinking": {"type": "adaptive", "display": "summarized"}}
        if tools and req.tool_choice != "none":
            kwargs["tools"] = tools
        elif tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "none"}
        try:
            async with self.client.messages.stream(**kwargs) as stream:
                async for ev in stream:
                    if ev.type == "text":
                        yield Delta(ev.text)
                    elif ev.type == "thinking":
                        yield Reasoning(ev.thinking)
                final = await stream.get_final_message()
        except a.RateLimitError as exc:
            raise ProviderError(str(exc), retryable=True, status=429) from None
        except a.APIStatusError as exc:
            raise ProviderError(str(exc), retryable=exc.status_code >= 500, status=exc.status_code) from None
        except a.APIConnectionError as exc:
            raise ProviderError(str(exc), retryable=True) from None
        if final.stop_reason == "refusal":
            yield Finish("error", "the model declined this request")
            return
        for b in final.content:
            if b.type == "tool_use":
                if final.stop_reason == "max_tokens":
                    continue  # a cut-off tool call is never run
                yield ToolCall(b.id, b.name, dict(b.input or {}))
        u = final.usage
        yield Usage(input=u.input_tokens or 0, output=u.output_tokens or 0,
                    cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                    cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0)
        yield Native({"provider": "anthropic", "content": [b.model_dump(exclude_none=True) for b in final.content]})
        yield Finish({"tool_use": "tool_calls", "max_tokens": "length"}.get(final.stop_reason or "", "stop"),
                     None if final.stop_reason != "max_tokens" else "output limit reached")
