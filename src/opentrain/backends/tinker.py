"""Tinker (Thinking Machines): LoRA SFT through the Tinker training API. The launch trainer.

Tinker runs the GPU work remotely; the training loop itself runs on this machine. So ``launch``
starts a detached worker (``python -m opentrain.backends.tinker_worker <job.json>``) that keeps
going after the CLI exits, writes progress and accrued cost into the job file after every step,
stops at the next step when the cancel flag appears, and finishes by saving sampler weights. The
sampler path (``tinker://<run_id>/sampler_weights/<name>``) is the checkpoint: Tinker serves it
directly over an OpenAI-compatible API and an Anthropic-compatible API, so there is no deploy step.

The API key is read from ``TINKER_API_KEY`` by the SDK itself and reaches the worker through the
inherited environment. It is never written anywhere by OpenTrain.

Docs relied on (read 2026-11-25), and the SDK source of ``tinker`` 0.40.0 / ``tinker-cookbook``
0.7.1 from PyPI:

* SDK cheatsheet: https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/
  (``ServiceClient()`` reads ``TINKER_API_KEY``; ``create_lora_training_client(base_model, rank)``;
  ``forward_backward(data, "cross_entropy")``; ``optim_step(AdamParams(learning_rate=...))``;
  ``save_weights_for_sampler(name)``. Every call returns a future; ``.result()`` in sync code.)
* First SFT tutorial: https://tinker-docs.thinkingmachines.ai/tutorials/basics/first-sft/
  (``get_renderer``, ``conversation_to_datum``, ``TrainOnWhat``)
* Rendering: https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/rendering/
* Checkpoints: https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/
  (sampler path ``tinker://<run_id>/sampler_weights/<name>``; kept until deleted unless a TTL is set)
* Prices: https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/ and its
  machine-readable ``models.json``. "Train" is USD per million tokens through forward+backward.
* Serving: https://tinker-docs.thinkingmachines.ai/inference/openai/overview/ ,
  https://tinker-docs.thinkingmachines.ai/inference/anthropic/ ,
  https://tinker-docs.thinkingmachines.ai/inference/opencode/

Not run end to end here: no ``TINKER_API_KEY`` was available while this was written. The calls
match the SDK source signatures; the data conversion and the loop's bookkeeping are unit-tested.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..plugs import JobStatus, TrainSpec
from . import JobStore, missing_modules

WORKER = "opentrain.backends.tinker_worker"

OPENAI_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
"""Tinker's OpenAI-compatible endpoint (beta). The model name is the ``tinker://`` sampler path."""
ANTHROPIC_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api"
"""Tinker's Anthropic Messages-compatible endpoint; clients append ``/v1/messages``."""
KEY_ENV = "TINKER_API_KEY"

PRICES_AS_OF = "2026-11-25"
TRAIN_USD_PER_MTOK: dict[str, float] = {
    # From https://tinker-docs.thinkingmachines.ai/tinker/models.json ("train" field), as of PRICES_AS_OF.
    # A model not listed here needs spec.extra["usd_per_mtok"]; we never guess a price.
    "Qwen/Qwen3-8B": 0.44,
    "Qwen/Qwen3.5-4B": 0.737,
    "Qwen/Qwen3.5-9B": 1.463,
    "Qwen/Qwen3.6-35B-A3B": 1.177,
    "Qwen/Qwen3.8-27B": 4.103,
    "Qwen/Qwen3.5-397B-A17B": 6.60,
    "openai/gpt-oss-20b": 0.396,
    "openai/gpt-oss-120b": 0.737,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": 0.44,
    "moonshotai/Kimi-K2.6": 4.84,
}

CHARS_PER_TOKEN = 3.0
"""For the pre-launch estimate only (no tokenizer needed). Code and JSON tokenize at roughly 3 to 4
characters per token; using the low end over-reserves rather than under-reserves the budget. The
worker reports cost from the real token counts."""


def price_per_mtok(spec: TrainSpec) -> float:
    if "usd_per_mtok" in spec.extra:
        return float(spec.extra["usd_per_mtok"])
    try:
        return TRAIN_USD_PER_MTOK[spec.base_model]
    except KeyError:
        raise ValueError(
            f"no Tinker training price on file for {spec.base_model!r}. Look it up at "
            "https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/ and pass it as "
            "extra usd_per_mtok, so the budget can be reserved."
        ) from None


# ---------------------------------------------------------------- data conversion (pure; tested)


def read_examples(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} has no training examples")
    return rows


def split_example(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """An OpenAI chat fine-tuning row as (system prompt, tool specs, the other messages).

    Tool specs come back in the renderer's ``ToolSpec`` shape (``name``, ``description``,
    ``parameters``), unwrapped from OpenAI's ``{"type": "function", "function": {...}}``.
    """
    msgs = list(row["messages"])
    system = ""
    if msgs and msgs[0].get("role") == "system":
        system = msgs.pop(0).get("content") or ""
    specs = []
    for t in row.get("tools") or []:
        fn = t.get("function", t)
        specs.append({"name": fn["name"], "description": fn.get("description", ""),
                      "parameters": fn.get("parameters", {"type": "object", "properties": {}})})
    return system, specs, msgs


def tool_names_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for m in messages:
        for c in m.get("tool_calls") or []:
            if c.get("id"):
                out[c["id"]] = c["function"]["name"]
    return out


def hf_supervised_tokens(tokenizer: Any, messages: list[dict[str, Any]],
                         tools: list[dict[str, Any]] | None) -> tuple[list[int], list[float]]:
    """Fallback when tinker-cookbook is absent: tokens and per-token weights via the HF chat template.

    The whole conversation is rendered once; each assistant message's span is found by rendering
    the conversation up to it (with the generation prompt) and through it. Tokens whose text
    starts inside an assistant span get weight 1. This needs the template to be prefix-stable
    (rendering fewer messages gives a prefix of rendering more); templates that rewrite earlier
    turns (Qwen3 drops old ``<think>`` blocks, for one) are not, and are refused loudly rather than
    trained with misplaced weights. Install tinker-cookbook to use its renderer for those.
    """
    kw = {"tools": tools} if tools else {}
    full = tokenizer.apply_chat_template(messages, tokenize=False, **kw)
    spans = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        before = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True, **kw)
        through = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False, **kw)
        if not (full.startswith(before) and full.startswith(through) and len(through) > len(before)):
            raise ValueError("the chat template is not prefix-stable for this conversation; "
                             "install tinker-cookbook so its renderer computes the training weights")
        spans.append((len(before), len(through)))
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    tokens = list(enc["input_ids"])
    weights = [1.0 if any(a <= start < b for a, b in spans) else 0.0 for start, _ in enc["offset_mapping"]]
    return tokens, weights


def hf_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI messages as HF chat templates expect them: tool-call arguments as dicts, not JSON text."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"].get("arguments")
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                calls.append({**c, "function": {**c["function"], "arguments": args}})
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def datum_parts(tokens: list[int], weights: list[float], max_tokens: int) -> tuple[list[int], list[int], list[float]]:
    """(input tokens, target tokens, weights) for next-token cross-entropy; weights mean-normalised.

    Normalising per example matches tinker-cookbook's default ``reduction="mean"``, so a long
    transcript does not outweigh a short one.
    """
    tokens, weights = tokens[:max_tokens], weights[:max_tokens]
    target_w = weights[1:]
    total = sum(target_w)
    if total <= 0:
        raise ValueError("example has no assistant tokens to train on (after truncation)")
    return tokens[:-1], tokens[1:], [w / total for w in target_w]


# ---------------------------------------------------------------- backend


class TinkerBackend:
    name = "tinker"

    def __init__(self, state: Path):
        self.store = JobStore(state)

    def available(self) -> tuple[bool, str]:
        if missing_modules("tinker"):
            return False, "the tinker SDK is not installed: pip install 'opentrain[tinker]'"
        if not os.environ.get(KEY_ENV):
            return False, f"set {KEY_ENV} (create a key at https://tinker.thinkingmachines.ai/)"
        if missing_modules("tinker_cookbook"):
            return True, ("ready; tinker-cookbook not installed, so chat templates come from the HF "
                          "tokenizer (pip install 'opentrain[tinker]' for the cookbook renderers)")
        return True, "ready (tinker SDK, tinker-cookbook renderers, key from TINKER_API_KEY)"

    def estimate(self, spec: TrainSpec) -> float:
        price = price_per_mtok(spec)
        tokens = 0
        for line in Path(spec.data).read_text().splitlines():
            if line.strip():
                tokens += min(int(len(line) / CHARS_PER_TOKEN) + 1, spec.max_tokens)
        return round(tokens * spec.epochs * price / 1e6, 6)

    def launch(self, spec: TrainSpec) -> str:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"tinker backend unavailable: {why}")
        if not Path(spec.data).is_file():
            raise FileNotFoundError(f"training data {spec.data} does not exist")
        price = price_per_mtok(spec)
        job = self.store.new(self.name, spec, usd_per_mtok=price,
                             detail=f"starting worker; cost from tokens x ${price}/Mtok (prices {PRICES_AS_OF})")
        self.store.spawn(WORKER, job["job_id"])
        return job["job_id"]

    def status(self, job_id: str) -> JobStatus:
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)
