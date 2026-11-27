"""Plugs: what connects to OpenTrain, and ``/connect``.

Five kinds plug in. Each has one place to look things up and one way to add credentials:

- **model providers** power the engine (``core.providers``),
- **trainers** run fine-tuning jobs (``opentrain.backends``; third parties register more through
  the ``opentrain.trainers`` entry point group),
- **base models**, offered per trainer,
- **data sources**: where sessions come from,
- **serving**: where a trained version answers (``opentrain model serve``).

``/connect <id>`` stores a key in the auth store (0600) and exports it into the server's
environment, so trainer workers and evaluator processes started later inherit it. Keys are
never written into a project or sent to a client.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import credentials

# id -> (name, kind, env var, how to get a key)
INTEGRATIONS: dict[str, tuple[str, str, str, str]] = {
    "anthropic": ("Anthropic", "provider", "ANTHROPIC_API_KEY", "https://platform.claude.com/settings/keys"),
    "openai": ("OpenAI", "provider", "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    "openrouter": ("OpenRouter", "provider", "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "together": ("Together", "provider", "TOGETHER_API_KEY", "https://api.together.ai/settings/api-keys"),
    "groq": ("Groq", "provider", "GROQ_API_KEY", "https://console.groq.com/keys"),
    "fireworks": ("Fireworks", "provider", "FIREWORKS_API_KEY", "https://fireworks.ai/account/api-keys"),
    "tinker": ("Tinker (Thinking Machines)", "trainer", "TINKER_API_KEY", "https://tinker-console.thinkingmachines.ai"),
    "modal": ("Modal (GPUs, your account)", "trainer", "MODAL_TOKEN_ID", "https://modal.com/settings/tokens"),
    "runpod": ("RunPod (GPU pods, your account)", "trainer", "RUNPOD_API_KEY", "https://console.runpod.io/user/settings"),
    "huggingface": ("Hugging Face", "trainer", "HF_TOKEN", "https://huggingface.co/settings/tokens"),
    "kaggle": ("Kaggle (datasets)", "data", "KAGGLE_KEY", "https://www.kaggle.com/settings"),
    "prime": ("Prime Intellect", "trainer", "PRIME_API_KEY", "https://app.primeintellect.ai"),
}

# Open models OpenTrain trains and serves on rented GPUs (Modal, RunPod): text models vLLM serves
# with LoRA. "opd" names the same-family teacher on-policy distillation uses (recipes.TEACHERS).
GPU_MODELS: list[dict[str, Any]] = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "params": "0.5B", "note": "tiny: cheapest to train and serve; opd teacher "
                                                                     "Qwen2.5-7B"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "params": "1.5B", "note": "opd teacher Qwen2.5-7B"},
    {"id": "Qwen/Qwen3-1.7B", "params": "1.7B", "note": "served without thinking; opd teacher Qwen3-8B"},
    {"id": "Qwen/Qwen3-4B-Instruct-2507", "params": "4B", "note": "default: strong for its size, no thinking"},
    {"id": "Qwen/Qwen3-8B", "params": "8B", "note": "served without thinking"},
    {"id": "Qwen/Qwen3.5-2B", "params": "2B", "note": "newest Qwen; opd teacher Qwen3.5-9B"},
    {"id": "Qwen/Qwen3.5-4B", "params": "4B", "note": "opd teacher Qwen3.5-9B"},
    {"id": "Qwen/Qwen3.5-9B", "params": "9B", "note": ""},
    {"id": "Qwen/Qwen3.5-27B", "params": "27B", "note": "H200 to train"},
    {"id": "google/gemma-3-1b-it", "params": "1B", "note": "gated: /connect huggingface; opd teacher gemma-3-4b"},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "params": "3B", "note": "gated: /connect huggingface"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "params": "8B", "note": "gated: /connect huggingface"},
    {"id": "Qwen/Qwen3-14B", "params": "14B", "note": "H100 to train"},
]

# Base models each trainer can fine-tune (LoRA). Tinker's list follows its documentation.
from ..models import CATALOG as _CAT  # noqa: E402


def _from_catalog(trainer: str) -> list[dict[str, Any]]:
    """Students for coding work first (``opentrain.models``), each with what it needs."""
    from ..models import CATALOG

    return [{"id": m.id, "params": m.params(), "note": m.notes, "tier": m.tier, "coding": m.coding}
            for m in CATALOG if trainer in m.trainers]


BASE_MODELS: dict[str, list[dict[str, Any]]] = {
    "tinker": [
        *_from_catalog("tinker"),
        {"id": "Qwen/Qwen3-8B", "params": "8B", "note": "fast to train and to serve"},
        {"id": "Qwen/Qwen3-32B", "params": "32B", "note": "stronger, slower"},
        {"id": "Qwen/Qwen3-30B-A3B", "params": "30B MoE (3B active)", "note": "fast inference"},
        {"id": "Qwen/Qwen3-235B-A22B-Instruct-2507", "params": "235B MoE", "note": "largest"},
        {"id": "meta-llama/Llama-3.1-8B-Instruct", "params": "8B", "note": ""},
        {"id": "meta-llama/Llama-3.3-70B-Instruct", "params": "70B", "note": ""},
        {"id": "openai/gpt-oss-20b", "params": "20B MoE", "note": ""},
    ],
    "modal": [*_from_catalog("modal"), *[m for m in GPU_MODELS if m["id"] not in {c.id for c in _CAT}]],
    "runpod": [*_from_catalog("runpod"), *[m for m in GPU_MODELS if m["id"] not in {c.id for c in _CAT}]],
    "local": [
        {"id": "Qwen/Qwen3-0.6B", "params": "0.6B", "note": "runs on a laptop (MPS or CPU)"},
        {"id": "Qwen/Qwen3-1.7B", "params": "1.7B", "note": "laptop with 16GB+"},
        {"id": "Qwen/Qwen3-4B", "params": "4B", "note": "a GPU with 24GB+"},
        {"id": "Qwen/Qwen3-8B", "params": "8B", "note": "a GPU with 40GB+"},
    ],
    "fake": [{"id": "fake/model", "params": "-", "note": "for tests and demos; trains nothing"}],
}


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def found_elsewhere(iid: str) -> tuple[str, str | None] | None:
    """A login another tool already keeps on this machine: (where it is, the key if OpenTrain must
    export it itself). Nothing to paste when the person has already signed in with that tool."""
    home = Path.home()
    if iid == "huggingface":
        hf_home = Path(os.environ.get("HF_HOME") or home / ".cache" / "huggingface")
        for p in (Path(os.environ["HF_TOKEN_PATH"]) if os.environ.get("HF_TOKEN_PATH") else None,
                  hf_home / "token", home / ".huggingface" / "token"):
            if p and (key := _read(p)):
                return f"{p} (huggingface-cli login)", key
    elif iid == "kaggle":
        cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR") or home / ".kaggle")
        if (cfg / "kaggle.json").is_file() or (cfg / "access_token").is_file():
            return f"{cfg} (kaggle CLI)", None
    elif iid == "modal":
        p = Path(os.environ.get("MODAL_CONFIG_PATH") or home / ".modal.toml")
        if p.is_file() and "token_id" in (_read(p) or ""):
            return f"{p} (modal setup)", None
    elif iid == "runpod":
        for p in (home / ".runpod" / "config.toml", home / ".runpod.yaml"):
            if p.is_file() and "apikey" in (_read(p) or "").lower().replace("_", ""):
                return f"{p} (runpodctl)", None
    elif iid == "fireworks":
        p = home / ".fireworks" / "auth.ini"
        if p.is_file() and "api_key" in (_read(p) or ""):
            return f"{p} (firectl)", None
    return None


def load_into_env() -> list[str]:
    """Export stored keys, and logins other tools keep, into this process's environment (never
    overriding what is already set). Workers and Modal secrets started later inherit them."""
    done = []
    for iid, (_, _, var, _) in INTEGRATIONS.items():
        if os.environ.get(var):
            continue
        stored = credentials.get(iid)
        if stored and stored.get("key"):
            _export(iid, stored["key"])
            done.append(iid)
        elif (found := found_elsewhere(iid)) and found[1]:
            os.environ[var] = found[1]
            done.append(iid)
    return done


def connect(iid: str, key: str) -> dict[str, Any]:
    if iid not in INTEGRATIONS:
        from .providers import custom_providers

        if iid not in custom_providers():
            raise KeyError(f"no integration {iid!r}; known: {', '.join(INTEGRATIONS)}")
        credentials.set_key(iid, key.strip())
        return status(iid)
    key = key.strip()
    if not key:
        raise ValueError("the key is empty")
    if iid == "modal" and not re.fullmatch(r"ak-\S+[:\s]+as-\S+", key):
        raise ValueError("paste the Modal token as token_id:token_secret (ak-...:as-...)")
    credentials.set_key(iid, key)
    _export(iid, key)
    return status(iid)


def _export(iid: str, key: str) -> None:
    if iid == "modal":  # one stored value, two variables
        tid, sec = re.split(r"[:\s]+", key, maxsplit=1)
        os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"] = tid, sec
    else:
        os.environ[INTEGRATIONS[iid][2]] = key


def disconnect(iid: str) -> bool:
    removed = credentials.remove(iid)
    if iid in INTEGRATIONS and removed:
        os.environ.pop(INTEGRATIONS[iid][2], None)
        if iid == "modal":
            os.environ.pop("MODAL_TOKEN_SECRET", None)
    return removed


def status(iid: str) -> dict[str, Any]:
    name, kind, var, url = INTEGRATIONS.get(iid, (iid, "provider", "", ""))
    stored = credentials.listing().get(iid)
    elsewhere = None if stored else found_elsewhere(iid)
    env = bool(var and os.environ.get(var))
    return {"id": iid, "name": name, "kind": kind, "env": var, "url": url,
            "connected": bool(stored) or env or bool(elsewhere),
            "via": ("connected" if stored else f"found at {elsewhere[0]}" if elsewhere
                    else f"${var}" if env else None),
            "hint": stored["hint"] if stored else None}


def listing() -> list[dict[str, Any]]:
    out = [status(i) for i in INTEGRATIONS]
    claude = shutil.which("claude")
    out.insert(0, {"id": "claude-code", "name": "Claude Code (your own login)", "kind": "provider", "env": None,
                   "url": "https://claude.com/claude-code", "connected": bool(claude),
                   "via": claude, "hint": None})
    return out


def trainers(state: Path) -> list[dict[str, Any]]:
    from ..backends import list_backends

    needs = {"tinker": "tinker", "fireworks": "fireworks", "modal": "modal", "runpod": "runpod"}
    return [{"name": n, "available": ok, "detail": why, "credential": needs.get(n),
             "base_models": BASE_MODELS.get(n, [])} for n, ok, why in list_backends(state)]


def data_sources() -> list[dict[str, Any]]:
    from ..sessions import claude_code

    home = Path.home()
    out = []
    root = claude_code.DEFAULT_ROOT
    n = sum(1 for _ in root.glob("*/*.jsonl")) if root.is_dir() else 0
    out.append({"id": "claude-code", "name": "Claude Code sessions", "path": str(root), "found": n,
                "supported": True})
    for sid, name, path in (("codex", "Codex sessions", home / ".codex" / "sessions"),
                            ("opencode", "opencode sessions", home / ".local" / "share" / "opencode"),
                            ("pi", "pi sessions", home / ".pi" / "agent" / "sessions")):
        out.append({"id": sid, "name": name, "path": str(path), "found": path.exists(), "supported": False})
    return out


def load_plugins() -> list[str]:
    """Register trainer backends published by other packages (entry point group ``opentrain.trainers``)."""
    from importlib.metadata import entry_points

    from ..backends import BACKENDS

    added = []
    for ep in entry_points(group="opentrain.trainers"):
        if ep.name not in BACKENDS:
            BACKENDS[ep.name] = ep.load()
            added.append(ep.name)
    return added
