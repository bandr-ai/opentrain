"""The models OpenTrain knows how to train and serve, and what each needs.

Picking a student model should pick a working setup: which trainers can tune it, the vLLM flags
that make its tool calls come back as tool calls (a coding agent is useless without them), how long
a context to serve, and roughly what it costs. Facts here come from each model's card and vLLM
recipe (read 2026-11-27; links per entry). Benchmarks are the vendors' own, on their own harnesses,
so they say which models are worth trying, not which one wins on your work: only your eval says that.

A model not listed still works where its family is known (``family_of``); otherwise it serves
without tool calling and the agent is told so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Model:
    id: str
    """Hugging Face id."""
    label: str
    tier: str
    """``small`` (cheap to run, up to ~10B active), ``mid`` (~25-35B), ``large`` (MoE above 100B total)."""
    total_b: float
    active_b: float
    tool_parser: str | None
    reasoning_parser: str | None = None
    min_vllm: str = "0.21.0"
    trainers: tuple[str, ...] = ("modal", "runpod")
    """Where OpenTrain can tune it. ``tinker`` uses Tinker's hosted LoRA; ``modal``/``runpod`` LoRA on
    one GPU with PEFT and serve with vLLM."""
    context: int = 32768
    """What OpenTrain serves and trains it at for coding work (traces are long)."""
    license: str = "apache-2.0"
    coding: str = ""
    """Vendor-reported coding numbers, quoted with the harness where known."""
    notes: str = ""
    source: str = ""
    tinker_id: str | None = None
    serve_args: tuple[str, ...] = field(default_factory=tuple)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "tier": self.tier, "params": self.params(),
                "trainers": list(self.trainers), "tool_calling": bool(self.tool_parser), "context": self.context,
                "license": self.license, "coding": self.coding, "notes": self.notes, "source": self.source}

    def params(self) -> str:
        if self.active_b < self.total_b:
            return f"{self.total_b:g}B total, {self.active_b:g}B active"
        return f"{self.total_b:g}B"


CATALOG: tuple[Model, ...] = (
    # ---- small: cheap to serve, the default students
    Model("thinkingmachines/Inkling-2-Mini", "Inkling 2 Mini", "small", 26.4, 3.4, "inkling", "inkling",
          min_vllm="0.35.0", trainers=("tinker",), context=65536,
          coding="SWE-bench Verified 79.2% (bash only), Terminal-Bench 2.1 71.6% (vendor)",
          notes="Looped MoE (middle layers run three times). Tinker: $0.75/M train tokens, free tier up to 64K "
                "context. Serving it yourself needs a vLLM nightly, so OpenTrain trains and serves it on Tinker.",
          source="https://thinkingmachines.ai/model-card/inkling-2-mini/", tinker_id="thinkingmachines/Inkling-2-Mini"),
    Model("Qwen/Qwen3.5-4B", "Qwen3.5 4B", "small", 4, 4, "qwen3_coder", "qwen3",
          trainers=("modal", "runpod", "tinker"), context=32768,
          notes="Cheapest self-hosted student; trains on one L40S.",
          source="https://recipes.vllm.ai/Qwen/Qwen3.5-4B.md", tinker_id="Qwen/Qwen3.5-4B"),
    Model("Qwen/Qwen3.5-9B", "Qwen3.5 9B", "small", 9, 9, "qwen3_coder", "qwen3",
          trainers=("modal", "runpod", "tinker"), context=32768,
          notes="More headroom on real repos than 4B at about twice the cost.",
          source="https://recipes.vllm.ai/Qwen/Qwen3.5-9B.md", tinker_id="Qwen/Qwen3.5-9B"),
    Model("google/gemma-4-E4B-it", "Gemma 4 E4B", "small", 8, 4.5, "gemma4", "gemma4", min_vllm="0.22.0",
          trainers=("modal", "runpod"), context=32768,
          notes="Not on Tinker. Its tool parser needs a newer vLLM than OpenTrain pins today.",
          source="https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html"),
    # ---- mid: when small models fall short on your repos
    Model("Qwen/Qwen3.8-27B", "Qwen3.8 27B", "mid", 27, 27, "qwen3_coder", "qwen3", min_vllm="0.17.0",
          trainers=("tinker", "modal", "runpod"), context=65536,
          coding="SWE-bench Pro 61.7, Terminal-Bench 2.1 73.0 (vendor, Claude Code harness)",
          notes="Dense, vision. Tinker $4.10/M train tokens; on Modal LoRA fits one H200.",
          source="https://recipes.vllm.ai/Qwen/Qwen3.8-27B", tinker_id="Qwen/Qwen3.8-27B"),
    Model("Qwen/Qwen3.6-35B-A3B", "Qwen3.6 35B-A3B", "mid", 35, 3, "qwen3_coder", "qwen3",
          trainers=("tinker", "modal", "runpod"), context=65536,
          coding="SWE-bench Verified 73.4% (secondary source)",
          notes="MoE with 3B active: fast and cheap to serve for its size. Tinker $1.18/M train tokens.",
          source="https://huggingface.co/Qwen/Qwen3.6-35B-A3B", tinker_id="Qwen/Qwen3.6-35B-A3B"),
    Model("google/gemma-4-31B-it", "Gemma 4 31B", "mid", 31, 31, "gemma4", "gemma4", min_vllm="0.22.0",
          trainers=("modal", "runpod"), context=32768, notes="Not on Tinker; needs an 80 GB GPU to serve.",
          source="https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html"),
    # ---- large MoE: Tinker only (serving them yourself takes several big GPUs)
    Model("thinkingmachines/Inkling-Small", "Inkling Small", "large", 276, 12, "inkling", "inkling",
          min_vllm="0.26.0", trainers=("tinker",), context=65536,
          coding="SWE-bench Verified 80.2% (bash only, vendor)",
          notes="Tinker $1.73/M train tokens. Serving it yourself needs 180 GB+ of GPU memory.",
          source="https://thinkingmachines.ai/news/inkling-small/", tinker_id="thinkingmachines/Inkling-Small"),
    Model("thinkingmachines/Inkling-2-Small", "Inkling 2 Small", "large", 594, 17, "inkling", "inkling",
          min_vllm="0.26.0", trainers=("tinker",), context=65536, notes="Tinker $1.50/M train tokens.",
          source="https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/",
          tinker_id="thinkingmachines/Inkling-2-Small"),
)

BY_ID = {m.id: m for m in CATALOG}

# Families for models not in the catalog: (pattern on the HF id, tool parser, reasoning parser).
_FAMILIES = (
    (r"qwen3\.[5-9]|qwen3-coder|qwen3\.\d+-\d", "qwen3_coder", "qwen3"),
    (r"qwen3", "hermes", "qwen3"),
    (r"qwen2\.5", "hermes", None),
    (r"gemma-?4", "gemma4", "gemma4"),
    (r"inkling", "inkling", "inkling"),
    (r"llama-?3", "llama3_json", None),
    (r"mistral|devstral", "mistral", None),
    (r"gpt-oss", "openai", "openai_gptoss"),
)


def get(model_id: str) -> Model | None:
    return BY_ID.get(model_id)


def family_of(model_id: str) -> tuple[str | None, str | None]:
    """(tool parser, reasoning parser) for a model, from the catalog or its family."""
    m = BY_ID.get(model_id)
    if m:
        return m.tool_parser, m.reasoning_parser
    low = model_id.lower()
    for pat, tool, reason in _FAMILIES:
        if re.search(pat, low):
            return tool, reason
    return None, None


def vllm_tool_args(model_id: str) -> list[str]:
    """The ``vllm serve`` flags that turn the model's tool calls into OpenAI tool calls."""
    tool, reason = family_of(model_id)
    out = ["--enable-auto-tool-choice", "--tool-call-parser", tool] if tool else []
    if reason:
        out += ["--reasoning-parser", reason]
    return out


def serve_context(model_id: str) -> int:
    m = BY_ID.get(model_id)
    return m.context if m else 32768


def trainers_for(model_id: str) -> tuple[str, ...]:
    m = BY_ID.get(model_id)
    return m.trainers if m else ("modal", "runpod")


def catalog(tier: str | None = None) -> list[dict[str, Any]]:
    return [m.public() for m in CATALOG if tier is None or m.tier == tier]
