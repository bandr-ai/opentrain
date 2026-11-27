"""Any OpenAI-compatible chat-completions endpoint, streamed: OpenAI, OpenRouter, Together, Groq,
Fireworks, and local servers (Ollama, LM Studio, vLLM) — including a model OpenTrain trained and
serves (``opentrain model serve``)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..llm import Delta, Event, Finish, ProviderError, Reasoning, Request, ToolCall, Usage

RETRY = {408, 409, 429, 500, 502, 503, 504, 529}


def _messages(req: Request) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": req.system}]
    for m in req.messages:
        if m["role"] == "assistant":
            text = "".join(c["text"] for c in m["content"] if c["type"] == "text")
            calls = [{"id": c["id"], "type": "function",
                      "function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
                     for c in m["content"] if c["type"] == "tool_call"]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif m["role"] == "tool":
            for c in m["content"]:
                out.append({"role": "tool", "tool_call_id": c["id"], "content": c["output"] or "(no output)"})
        else:
            out.append({"role": "user", "content": "\n".join(c.get("text", "") for c in m["content"])})
    return out


class OpenAICompatProvider:
    kind = "step"

    def __init__(self, id: str, base_url: str, api_key: str | None, headers: dict[str, str] | None = None):
        self.id, self.base_url, self.api_key = id, base_url.rstrip("/"), api_key
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def stream(self, req: Request) -> AsyncIterator[Event]:
        import aiohttp

        body: dict[str, Any] = {"model": req.model, "messages": _messages(req), "stream": True,
                                "max_tokens": req.max_tokens, "stream_options": {"include_usage": True}}
        if req.tools:
            body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                               "parameters": t.schema}} for t in req.tools]
            body["tool_choice"] = "none" if req.tool_choice == "none" else "auto"
        calls: dict[int, dict[str, Any]] = {}
        finish, usage = "stop", Usage()
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(f"{self.base_url}/chat/completions", json=body, headers=self.headers) as r:
                    if r.status >= 400:
                        text = (await r.text())[:500]
                        raise ProviderError(f"{self.id} HTTP {r.status}: {text}", retryable=r.status in RETRY,
                                            status=r.status)
                    async for raw in r.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except ValueError:
                            continue
                        if chunk.get("usage"):
                            u = chunk["usage"]
                            usage = Usage(input=u.get("prompt_tokens", 0) or 0, output=u.get("completion_tokens", 0) or 0,
                                          cost=u.get("cost"))
                        for ch in chunk.get("choices") or []:
                            d = ch.get("delta") or {}
                            if d.get("reasoning_content") or d.get("reasoning"):
                                yield Reasoning(d.get("reasoning_content") or d.get("reasoning"))
                            if d.get("content"):
                                yield Delta(d["content"])
                            for tc in d.get("tool_calls") or []:
                                slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                                slot["id"] = tc.get("id") or slot["id"]
                                fn = tc.get("function") or {}
                                slot["name"] = fn.get("name") or slot["name"]
                                slot["args"] += fn.get("arguments") or ""
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
        except aiohttp.ClientError as exc:
            raise ProviderError(f"{self.id}: {exc}", retryable=True) from None
        for i in sorted(calls):
            c = calls[i]
            try:
                args = json.loads(c["args"] or "{}")
            except ValueError:
                args = {"_invalid_json": c["args"]}
            yield ToolCall(c["id"] or f"call_{i}", c["name"], args if isinstance(args, dict) else {"value": args})
        yield usage
        yield Finish({"tool_calls": "tool_calls", "length": "length"}.get(finish, "stop"))

    async def list_models(self) -> list[str]:
        import aiohttp

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(f"{self.base_url}/models", headers=self.headers) as r:
                if r.status >= 400:
                    return []
                data = await r.json()
        return sorted(m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id"))
