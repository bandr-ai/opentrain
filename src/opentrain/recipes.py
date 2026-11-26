"""Recipes: tested defaults for training a small model on one narrow task, so nobody guesses.

Four methods (run by ``opentrain.backends.gpu_train``): ``sft`` on reference answers, ``rejection``
sampling (train on the model's own graded-correct answers), on-policy distillation (``opd``: a
same-family teacher scores the student's own tokens) and ``rl`` (GRPO on the grader's score).

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
    "reason": "a checkable final answer after working (math, logic): long references, a number/regex/exact check",
    "answer": "free-form answers graded by a judge or a check",
}

METHODS = {
    "sft": "learn the reference answers (cross-entropy on the answer tokens)",
    "rejection": "sample several answers per prompt, keep the ones the grader passes, train on those",
    "opd": "on-policy distillation: a bigger model of the same family scores the student's own answers, token "
           "by token",
    "rl": "GRPO: sample a group per prompt, reward = grader score, push toward the better-than-average answers",
}
GRADED_HERE = ("exact", "contains", "regex", "json", "number", "python")
"""Graders that run inside the training loop (rejection sampling and RL need a score per sample)."""

# Student -> same-tokenizer teacher for on-policy distillation (a larger model of the same family,
# small enough to share one GPU with the student and its sampler).
TEACHERS = [
    (r"^Qwen/Qwen3\.5-(0\.8|2|4)B", "Qwen/Qwen3.5-9B"),
    (r"^Qwen/Qwen3-(0\.6|1\.7|4)B", "Qwen/Qwen3-8B"),
    (r"^Qwen/Qwen3-8B$", "Qwen/Qwen3-14B"),  # 8B + 14B fit one H200
    (r"^Qwen/Qwen2\.5-(0\.5|1\.5|3)B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
    (r"^meta-llama/Llama-3\.2-(1|3)B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    (r"^google/gemma-3-(270m|1b)-it", "google/gemma-3-4b-it"),
]


def teacher_for(base_model: str) -> str | None:
    return next((t for pat, t in TEACHERS if re.match(pat, base_model)), None)


def task_kind(grader: dict[str, Any] | None, references: list[str] | None = None) -> str:
    g = (grader or {}).get("type")
    refs = [r for r in (references or []) if r]
    median = sorted(len(r) for r in refs)[len(refs) // 2] if refs else 0
    # Working before a checked final answer: a number/regex check on long references (GSM8K-like),
    # or an exact/contains check on very long ones.
    if (g in ("number", "regex") and median > 40) or (g in ("exact", "contains") and median > 120):
        return "reason"
    if g in ("json", "number", "regex"):
        return "extract"
    if g in ("exact", "contains"):
        if refs and len(set(refs)) <= max(12, len(refs) // 5) and max(len(r) for r in refs) <= 40:
            return "label"
        return "extract"
    return "answer"


def _params_b(model: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model.split("/")[-1])
    return float(m.group(1)) if m else 8.0


def recipe(kind: str, rows: int, base_model: str, *, labelled: int | None = None, method: str | None = None,
           grader: dict[str, Any] | None = None) -> dict[str, Any]:
    """The settings for one training run, and why. ``method`` defaults to the plan's first step."""
    p = _params_b(base_model)
    epochs = 4 if rows < 100 else 3 if rows < 500 else 2 if rows < 5000 else 1
    if kind == "label":
        rank, lr = 16, 2e-4
    elif kind in ("extract", "reason"):
        rank, lr = 32, 2e-4
    else:
        rank, lr = 32, 1e-4
    if p > 10:
        lr /= 2
    graded_here = (grader or {}).get("type") in GRADED_HERE
    teacher = teacher_for(base_model)
    plan = _plan(kind, graded_here, teacher)
    method = method or plan[0]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {', '.join(METHODS)}")
    gen = 512 if kind == "reason" else 384 if kind == "answer" else 96
    steps = max(20, min(100, rows // 4))
    extra: dict[str, Any] = {}
    batch = 8
    if method == "rejection":
        lr, batch = lr / 2, 16
        extra = {"rounds": 2, "samples": 8, "keep": 2, "temperature": 1.0, "max_new_tokens": gen}
    elif method == "rl":
        lr, rank = (4e-5 if p <= 10 else 2e-5), min(rank, 32)
        extra = {"steps": steps, "prompts_per_step": 16, "samples": 8, "temperature": 1.0, "max_new_tokens": gen}
    elif method == "opd":
        lr = 1e-4 if p <= 10 else 5e-5
        extra = {"steps": steps, "prompts_per_step": 16, "samples": 2, "temperature": 1.0, "max_new_tokens": gen,
                 "teacher": teacher}
    if method in ("rejection", "rl"):
        if not graded_here:
            raise ValueError(f"{method} needs a grader that runs in the training loop ({', '.join(GRADED_HERE)}); "
                             f"this dataset's is {(grader or {}).get('type') or 'not set'}. Use sft or opd.")
        extra["grader"] = grader
    if method == "opd" and not teacher:
        raise ValueError(f"no same-family teacher known for {base_model}; pass extra.teacher, or use sft")
    before = []
    if labelled is not None and labelled < rows and method in ("sft", "rejection", "rl"):
        before.append(f"distil: {rows - labelled} training rows have no answer; label them with a strong model first "
                      "(dataset_label), then check a few by hand")
    if rows < 30:
        before.append(f"only {rows} training rows: add more (aim for 100+) or expect a small gain")
    if_not = ["more and more varied rows beat more epochs",
              "try a larger base model before tuning settings",
              f"one sweep: learning rate {lr / 2:g} and {lr * 2:g}, keep the best held-out score"]
    nxt = [m for m in plan if m != method]
    if nxt:
        if_not.insert(0, f"next method: {nxt[0]} ({METHODS[nxt[0]]})")
    if kind == "answer":
        if_not.insert(0, "distil from the best baseline's answers: a strong teacher's outputs are often better "
                         "targets than hand-written ones")
    shown = ", ".join(f"{k} {v}" for k, v in extra.items() if k not in ("grader", "temperature"))
    return {"kind": kind, "about": KINDS.get(kind, kind), "method": method, "method_about": METHODS[method],
            "plan": plan, "epochs": epochs, "lora_rank": rank, "learning_rate": lr, "batch_size": batch,
            "extra": extra, "before": before, "if_it_does_not_beat_the_baseline": if_not,
            "why": f"{kind} task, {rows} rows, {p:g}B base, {method}: rank {rank}, lr {lr:g}"
                   + (f", {epochs} epochs" if method == "sft" else f", {shown}")}


def _plan(kind: str, graded_here: bool, teacher: str | None) -> list[str]:
    """Methods worth trying, in order. SFT first wherever there are references (cheapest, most
    stable); RL and rejection sampling only with a grader that runs in the loop; distillation when a
    same-family teacher exists."""
    if kind == "reason":
        plan = ["sft", "rl", "rejection"] if graded_here else ["sft"]
    elif kind == "answer":
        plan = ["sft"]
    else:
        plan = ["sft", "rejection"] if graded_here else ["sft"]
    if teacher:
        plan.insert(1 if kind != "answer" else 0, "opd")
    return plan
