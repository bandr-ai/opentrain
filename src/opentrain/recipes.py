"""Recipes: tested defaults for fine-tuning a small model on one narrow task, so nobody guesses.

A recipe is picked from three things OpenTrain already knows: the kind of task (read from how the
dataset is graded), how many training rows there are, and the base model's size. It says what to
do before training (distil when answers are missing), the LoRA settings, and what to try next if
the first run does not beat the baseline.

The numbers follow common LoRA practice for instruction-tuned models (rank 16-32 on all linear
layers, a learning rate near 2e-4 for small models and lower for larger ones, more passes over
small datasets and fewer over large ones). They are starting points measured on our own demo
tasks (``docs/RECIPES.md``), not guarantees: the held-out score decides, every time.
"""

from __future__ import annotations

import re
from typing import Any

KINDS = {
    "label": "short answers from a fixed set (classification, routing, yes/no)",
    "extract": "structured output (JSON, fields, numbers)",
    "answer": "free-form answers graded by a judge or a check",
}


def task_kind(grader: dict[str, Any] | None, references: list[str] | None = None) -> str:
    g = (grader or {}).get("type")
    if g in ("json", "number", "regex"):
        return "extract"
    if g in ("exact", "contains"):
        refs = [r for r in (references or []) if r]
        if refs and len(set(refs)) <= max(12, len(refs) // 5) and max(len(r) for r in refs) <= 40:
            return "label"
        return "extract"
    return "answer"


def _params_b(model: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model.split("/")[-1])
    return float(m.group(1)) if m else 8.0


def recipe(kind: str, rows: int, base_model: str, *, labelled: int | None = None) -> dict[str, Any]:
    """The settings for one training run, and why."""
    p = _params_b(base_model)
    epochs = 4 if rows < 100 else 3 if rows < 500 else 2 if rows < 5000 else 1
    if kind == "label":
        rank, lr = 16, 2e-4
    elif kind == "extract":
        rank, lr = 32, 2e-4
    else:
        rank, lr = 32, 1e-4
    if p > 10:
        lr /= 2
    before = []
    if labelled is not None and labelled < rows:
        before.append(f"distil: {rows - labelled} training rows have no answer; label them with a strong model first "
                      "(dataset_label), then check a few by hand")
    if rows < 30:
        before.append(f"only {rows} training rows: add more (aim for 100+) or expect a small gain")
    if_not = ["more and more varied rows beat more epochs",
              "try a larger base model before tuning settings",
              f"one sweep: learning rate {lr / 2:g} and {lr * 2:g}, keep the best held-out score"]
    if kind == "answer":
        if_not.insert(0, "distil from the best baseline's answers: a strong teacher's outputs are often better "
                         "targets than hand-written ones")
    return {"kind": kind, "about": KINDS.get(kind, kind), "epochs": epochs, "lora_rank": rank, "learning_rate": lr,
            "batch_size": 8, "before": before, "if_it_does_not_beat_the_baseline": if_not,
            "why": f"{kind} task, {rows} rows, {p:g}B base: rank {rank}, lr {lr:g}, {epochs} epochs"}
