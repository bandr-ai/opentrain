"""One completion from any model OpenTrain can reach: what evals, labelling and the playground use.

A *target* is either

- an engine model ref, ``provider/model`` (``anthropic/claude-opus-5-5``, ``claude-code/haiku``,
  ``openrouter/qwen/qwen3-8b``): the same providers the agent runs on, or
- a model version from the registry, ``v3`` (or ``version:v3``): its OpenAI-compatible endpoint
  (Tinker, Modal, Fireworks, a local vLLM/MLX server), or
- ``modal/<hugging face id>`` or ``runpod/<hugging face id>`` (``modal/Qwen/Qwen3-4B-Instruct-2507``): an
  open base model served on the user's own Modal or RunPod account, so any open model can be
  measured before training it.

Calls are synchronous and meant for worker threads (jobs). Claude Code runs ``claude -p`` with no
tools and none of the user's settings, so it only answers.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = re.compile(r"^(?:version:)?(v\d+)$")


class CompletionError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    seconds: float
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    billing: str = "api"


def is_version(target: str) -> bool:
    return bool(VERSION.match(target or ""))


def complete(target: str, messages: list[dict[str, str]], *, state: Path | None = None, max_tokens: int = 2048,
             timeout: float = 300.0) -> Completion:
    """``messages`` is plain chat (system/user/assistant with string content)."""
    m = VERSION.match(target or "")
    t0 = time.time()
    if m:
        if state is None:
            raise CompletionError("a model version needs the project state")
        return _version(state, m.group(1), messages, max_tokens, timeout, t0)
    if "/" not in (target or ""):
        raise CompletionError(f"{target!r} is not a model: use provider/model (e.g. claude-code/haiku) or a version (v3)")
    provider = target.split("/", 1)[0]
    if provider == "claude-code":
        return _claude_code(target.split("/", 1)[1], messages, timeout, t0)
    if provider in HOSTS:
        return _hosted(provider, target.split("/", 1)[1], messages, max_tokens, timeout, t0)
    return _provider(target, messages, max_tokens, t0)


def _version(state: Path, version: str, messages, max_tokens, timeout, t0) -> Completion:
    from ..registry import Registry, RegistryError
    from ..serving import ServingError, endpoint_for

    try:
        v = Registry(state).get(version)
    except RegistryError as exc:
        raise CompletionError(str(exc)) from None
    if v.checkpoint.startswith("fireworks-sampler://"):
        return _fireworks(v, messages, max_tokens, timeout, t0)
    try:
        ep = endpoint_for(v)
    except ServingError as exc:
        raise CompletionError(str(exc)) from None
    host = next((h for h in HOSTS if v.backend == h or v.checkpoint.startswith(f"{h}://")), None)
    return _openai(ep, version, messages, max_tokens, timeout, t0, host=(host, v.base_model) if host else None)


def _fireworks(v, messages, max_tokens, timeout, t0) -> Completion:
    """A version trained on Fireworks serverless: answered from its sampler snapshot."""
    from ..backends import fireworks

    try:
        text, p, c, cost = fireworks.sample(v.checkpoint, v.base_model, messages, max_tokens=max_tokens,
                                            timeout=timeout)
    except fireworks.FireworksError as exc:
        raise CompletionError(str(exc)) from None
    return Completion(text=THINK.sub("", text), seconds=time.time() - t0, cost=cost, input_tokens=p,
                      output_tokens=c)


HOSTS = ("modal", "runpod")
"""GPU clouds OpenTrain serves open models on itself: ``<host>/<hugging face id>`` targets and versions
trained there. Each module has ``endpoint``, ``wait_ready`` and ``answering``."""


def _host(name: str):
    if name == "modal":
        from ..backends import modal_backend

        return modal_backend
    from ..backends import runpod_backend

    return runpod_backend


def _hosted(host: str, base: str, messages, max_tokens, timeout, t0) -> Completion:
    """``modal/<id>`` or ``runpod/<id>``: an open model served on your own GPU account (started on first use)."""
    try:
        ep = _host(host).endpoint(base)
    except RuntimeError as exc:
        raise CompletionError(str(exc)) from None
    return _openai(ep, f"{host}/{base}", messages, max_tokens, timeout, t0, host=(host, base))


THINK = re.compile(r"^\s*(?:<think>)?.*?</think>\s*", re.S)  # the opening tag may be in the prompt (Qwen3.5)


def _openai(ep, label: str, messages, max_tokens, timeout, t0, host: tuple[str, str] | None = None) -> Completion:
    headers = {"Content-Type": "application/json"}
    if ep.api_key_env:
        key = os.environ.get(ep.api_key_env)
        if not key:
            raise CompletionError(f"{label} is served by an endpoint that needs {ep.api_key_env}; /connect it")
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": ep.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    mod = _host(host[0]) if host else None
    if host:
        try:
            mod.wait_ready(ep)  # a scaled-to-zero endpoint starts on the first request
        except RuntimeError as exc:
            raise CompletionError(str(exc)) from None
        t0 = time.time()  # the answer's time, not the cold start's
        if "qwen3" in host[1].lower() and "instruct" not in host[1].lower():
            body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(f"{ep.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    cost_of = None
    try:
        if host:
            with mod.answering(host[1]) as cost_of_answer:
                with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
                    out = json.loads(r.read().decode())
                cost_of = cost_of_answer()
        else:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
                out = json.loads(r.read().decode())
    except Exception as exc:  # urllib raises several unrelated types
        raise CompletionError(f"{label} did not answer: {exc}") from None
    u = out.get("usage") or {}
    text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return Completion(text=THINK.sub("", text), seconds=time.time() - t0, cost=cost_of or 0.0,
                      input_tokens=u.get("prompt_tokens", 0) or 0, output_tokens=u.get("completion_tokens", 0) or 0,
                      billing=f"{host[0]}-gpu" if host else "api")


def _flatten(messages: list[dict[str, str]]) -> tuple[str, str]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    if len(turns) == 1:
        return system, turns[0]["content"]
    convo = "\n\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in turns)
    return system, f"Continue this conversation as the assistant. Reply with the next assistant turn only.\n\n{convo}"


def _claude_code(model: str, messages, timeout, t0) -> Completion:
    exe = shutil.which("claude")
    if not exe:
        raise CompletionError("Claude Code is not installed")
    system, prompt = _flatten(messages)
    argv = [exe, "-p", "--output-format", "stream-json", "--verbose", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--system-prompt", system or "You are a helpful assistant. Answer directly."]
    if model not in ("default", ""):
        argv += ["--model", model]
    try:
        p = subprocess.run(argv, input=prompt, capture_output=True, text=True, timeout=timeout,
                           cwd=os.path.expanduser("~"))
    except subprocess.TimeoutExpired:
        raise CompletionError(f"claude did not answer within {timeout:.0f}s") from None
    out: dict[str, Any] = {}
    billing = "api"
    for line in p.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            billing = "plan" if ev.get("apiKeySource") == "none" else "api"
        elif ev.get("type") == "result":
            out = ev
    if not out:
        raise CompletionError(f"claude failed (exit {p.returncode}): {(p.stderr or p.stdout)[-400:]}")
    if out.get("is_error"):
        raise CompletionError(f"claude: {out.get('result') or out.get('subtype')}")
    u = out.get("usage") or {}
    # Under a subscription the reported cost is notional: nothing is billed, so nothing is drawn from the budget.
    cost = float(out.get("total_cost_usd") or 0.0) if billing == "api" else 0.0
    return Completion(text=out.get("result") or "", seconds=time.time() - t0, cost=cost,
                      input_tokens=(u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0),
                      output_tokens=u.get("output_tokens") or 0, billing=billing)


def _provider(ref: str, messages, max_tokens, t0) -> Completion:
    from . import llm
    from .providers import ProviderMissing, get, model_info

    try:
        provider = get(ref)
    except ProviderMissing as exc:
        raise CompletionError(str(exc)) from None
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages if m["role"] != "system"]
    req = llm.Request(model=ref.split("/", 1)[1], system=system, messages=msgs, max_tokens=max_tokens,
                      tool_choice="none")

    async def go() -> tuple[str, Any, str | None]:
        text, usage, err = "", llm.Usage(), None
        async for ev in provider.stream(req):
            if isinstance(ev, llm.Delta):
                text += ev.text
            elif isinstance(ev, llm.Usage):
                usage = ev
            elif isinstance(ev, llm.Finish) and ev.reason == "error":
                err = ev.error or "error"
        return text, usage, err

    try:
        text, usage, err = asyncio.run(go())
    except llm.ProviderError as exc:
        raise CompletionError(str(exc)) from None
    if err:
        raise CompletionError(err)
    cost = usage.cost if usage.cost is not None else model_info(ref).price(usage)
    return Completion(text=text, seconds=time.time() - t0, cost=cost or 0.0, input_tokens=usage.input,
                      output_tokens=usage.output)
