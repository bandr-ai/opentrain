"""Model providers for the engine, and the catalogue of models it can pick from.

A model is named ``provider/model`` (``anthropic/claude-opus-5-5``, ``openrouter/qwen/qwen3-coder``,
``claude-code/opus``). Three kinds of provider:

- ``anthropic``: the Messages API through the official SDK (``pip install opentrain-cli[anthropic]``).
- OpenAI-compatible: ``openai``, ``openrouter``, ``together``, ``groq``, ``ollama``, ``lmstudio``,
  ``vllm`` and any ``providers`` entry in ``opentrain.jsonc`` with a ``base_url``.
- Agent providers: ``claude-code`` (and later ``codex``). They run the user's own CLI under their
  own login; that CLI runs its own loop and calls OpenTrain's tools back through the MCP bridge,
  so permissions, budget and the event log apply exactly as with a direct provider.

Credentials come from the auth store (``opentrain`` ``/connect``) first, then the environment.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

from ..llm import Model, parse_ref

# provider id -> (display name, env var for the key, base url or None, kind)
OPENAI_COMPATIBLE: dict[str, tuple[str, str | None, str]] = {
    "openai": ("OpenAI", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "together": ("Together", "TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    "groq": ("Groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "fireworks": ("Fireworks", "FIREWORKS_API_KEY", "https://api.fireworks.ai/inference/v1"),
    "ollama": ("Ollama (local)", None, "http://127.0.0.1:11434/v1"),
    "lmstudio": ("LM Studio (local)", None, "http://127.0.0.1:1234/v1"),
    "vllm": ("vLLM (local)", None, "http://127.0.0.1:8000/v1"),
}

# A small built-in catalogue; providers that list models (OpenAI-compatible /v1/models) add more.
CATALOGUE: list[Model] = [
    Model("anthropic", "claude-opus-5-5", "Claude Opus 5.5", 1_000_000, 128_000, 4.0, 20.0, True),
    Model("anthropic", "claude-sonnet-5-5", "Claude Sonnet 5.5", 1_000_000, 128_000, 2.0, 10.0, True),
    Model("anthropic", "claude-haiku-5-5", "Claude Haiku 5.5", 1_000_000, 128_000, 0.10, 0.50, True),
    Model("anthropic", "claude-fable-5-2", "Claude Fable 5.2", 1_000_000, 128_000, 8.0, 40.0, True),
    Model("claude-code", "default", "Claude Code (your login, its default model)", 1_000_000, 128_000),
    Model("claude-code", "opus", "Claude Code · Opus (your login)", 1_000_000, 128_000),
    Model("claude-code", "sonnet", "Claude Code · Sonnet (your login)", 1_000_000, 128_000),
    Model("claude-code", "haiku", "Claude Code · Haiku (your login)", 1_000_000, 128_000),
    Model("openai", "gpt-6.1", "GPT 6.1", 400_000, 128_000, 2.0, 10.0, True),
    Model("openrouter", "qwen/qwen3-coder", "Qwen3 Coder (OpenRouter)", 262_000, 32_000, 0.3, 1.2),
]


@dataclass
class ProviderInfo:
    id: str
    name: str
    kind: str  # "anthropic" | "openai" | "agent"
    base_url: str | None = None
    env: str | None = None
    connected: bool = False
    how: str = ""  # where the credential came from
    models: list[Model] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind, "base_url": self.base_url, "env": self.env,
                "connected": self.connected, "how": self.how,
                "models": [{"id": m.id, "ref": m.ref, "name": m.name, "context": m.context,
                            "input_price": m.input_price, "output_price": m.output_price} for m in self.models]}


def _credential(provider: str) -> tuple[str | None, str]:
    """(key, where it came from). The auth store wins over the environment."""
    from .. import credentials

    stored = credentials.get(provider)
    if stored and stored.get("key"):
        return stored["key"], "connected"
    env = {"anthropic": "ANTHROPIC_API_KEY", **{k: v[1] for k, v in OPENAI_COMPATIBLE.items()}}.get(provider)
    if env and os.environ.get(env):
        return os.environ[env], f"${env}"
    return None, ""


def custom_providers() -> dict[str, dict[str, Any]]:
    from .. import config

    return config.load_global().get("providers", {}) or {}


def providers() -> list[ProviderInfo]:
    out: list[ProviderInfo] = []
    key, how = _credential("anthropic")
    out.append(ProviderInfo("anthropic", "Anthropic", "anthropic", env="ANTHROPIC_API_KEY", connected=bool(key),
                            how=how, models=[m for m in CATALOGUE if m.provider == "anthropic"]))
    claude = shutil.which("claude")
    out.append(ProviderInfo("claude-code", "Claude Code (your own login)", "agent", connected=bool(claude),
                            how=claude or "install Claude Code", models=[m for m in CATALOGUE if m.provider == "claude-code"]))
    for pid, (name, env, base) in OPENAI_COMPATIBLE.items():
        key, how = _credential(pid)
        local = env is None
        out.append(ProviderInfo(pid, name, "openai", base_url=base, env=env, connected=bool(key) or local,
                                how=how or ("local" if local else ""),
                                models=[m for m in CATALOGUE if m.provider == pid]))
    for pid, spec in custom_providers().items():
        key, how = _credential(pid)
        models = [Model(pid, m if isinstance(m, str) else m["id"], m if isinstance(m, str) else m.get("name", m["id"]))
                  for m in spec.get("models", [])]
        out.append(ProviderInfo(pid, spec.get("name", pid), "openai", base_url=spec.get("base_url"),
                                env=spec.get("env"), connected=bool(key or spec.get("env") and os.environ.get(spec["env"])
                                                                    or not spec.get("env")),
                                how=how or "opentrain.jsonc", models=models))
    return out


def model_info(ref: str) -> Model:
    provider, mid = parse_ref(ref)
    for m in CATALOGUE:
        if m.provider == provider and m.id == mid:
            return m
    return Model(provider, mid, mid)


def default_model() -> str:
    """The first model that can run here: a stored or env key, else the user's Claude Code login."""
    from .. import config

    chosen = config.load_global().get("model")
    if chosen:
        return chosen
    if _credential("anthropic")[0]:
        return "anthropic/claude-opus-5-5"
    if shutil.which("claude"):
        return "claude-code/default"
    for pid in ("openrouter", "openai", "together", "groq"):
        if _credential(pid)[0]:
            return next((m.ref for m in CATALOGUE if m.provider == pid), f"{pid}/default")
    return "claude-code/default"


class ProviderMissing(RuntimeError):
    pass


def get(ref: str):
    """A provider instance for ``ref``. Raises ProviderMissing with what to do about it."""
    provider, mid = parse_ref(ref)
    if provider == "scripted":
        from .scripted import ScriptedProvider

        return ScriptedProvider.named(mid)
    if provider == "claude-code":
        from .claude_code import ClaudeCodeProvider

        if not shutil.which("claude"):
            raise ProviderMissing("Claude Code is not installed: https://claude.com/claude-code")
        return ClaudeCodeProvider(mid)
    if provider == "anthropic":
        key, _ = _credential("anthropic")
        if not key:
            raise ProviderMissing("no Anthropic key: /connect anthropic, or set ANTHROPIC_API_KEY")
        from .anthropic import AnthropicProvider

        return AnthropicProvider(key)
    spec = custom_providers().get(provider)
    if provider in OPENAI_COMPATIBLE or spec:
        name, env, base = OPENAI_COMPATIBLE.get(provider, (provider, None, None))
        base = (spec or {}).get("base_url") or base
        key, _ = _credential(provider)
        if not key and spec and spec.get("env"):
            key = os.environ.get(spec["env"])
        if not key and env:
            raise ProviderMissing(f"no {name} key: /connect {provider}, or set {env}")
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(provider, base, key, headers=(spec or {}).get("headers"))
    raise ProviderMissing(f"unknown provider {provider!r}; /connect lists the ones OpenTrain knows")
