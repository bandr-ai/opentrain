"""Graders: how an answer is checked. Each returns a score in [0, 1] and a short reason.

A dataset names its grader in ``meta.json``::

    {"type": "exact"}                          same text as the reference (case and spacing ignored)
    {"type": "contains"}                       the reference appears in the answer
    {"type": "regex", "pattern": "..."}        the answer matches the pattern (or the reference, if none)
    {"type": "json"}                           the answer parses as JSON equal to the reference's JSON
    {"type": "number", "tolerance": 0.01}      the last number in the answer equals the reference's
    {"type": "judge", "rubric": "...",         a model grades against the rubric and the reference,
     "model": "claude-code/haiku"}             0-10, scaled to [0, 1]
    {"type": "command", "command": "..."}      a script: JSON {prompt, answer, reference} on stdin;
                                               exit 0 passes, or it prints a score on its last line

``passed`` is ``score >= threshold`` (default 1.0 for exact kinds, 0.7 for the judge and scripts).
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KINDS = ("exact", "contains", "regex", "json", "number", "judge", "command")
JUDGE_PROMPT = """You are grading one answer. Be strict and consistent.

Rubric:
{rubric}

The task given to the model:
<task>
{prompt}
</task>

{reference_block}The model's answer:
<answer>
{answer}
</answer>

Reply with one line: SCORE: <0-10> | <one short reason>"""
DEFAULT_RUBRIC = ("Score 10 when the answer fully and correctly does what the task asks (as well as the reference "
                  "answer or better), 0 when it is wrong or off-task. Partial credit only for answers that are "
                  "mostly right with small flaws.")


class GraderError(ValueError):
    pass


@dataclass
class Grade:
    score: float
    passed: bool
    reason: str = ""
    cost: float = 0.0


def check(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate a grader spec; returns it normalized."""
    if not isinstance(spec, dict) or spec.get("type") not in KINDS:
        raise GraderError(f"grader type must be one of {', '.join(KINDS)}")
    if spec["type"] == "regex" and spec.get("pattern"):
        try:
            re.compile(spec["pattern"])
        except re.error as exc:
            raise GraderError(f"bad pattern: {exc}") from None
    if spec["type"] == "command" and not spec.get("command"):
        raise GraderError("a command grader needs a command")
    return spec


def needs_reference(spec: dict[str, Any]) -> bool:
    return spec.get("type") in ("exact", "contains", "json", "number") or (
        spec.get("type") == "regex" and not spec.get("pattern"))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _strip_fence(s: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", s or "", re.S)
    return m.group(1) if m else (s or "")


def _last_number(s: str) -> float | None:
    found = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", s or "")
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def grade(spec: dict[str, Any], prompt: str, answer: str, reference: str | None, *,
          directory: Path | None = None, default_model: str | None = None) -> Grade:
    kind = spec.get("type")
    thr = float(spec.get("threshold", 0.7 if kind in ("judge", "command") else 1.0))
    if needs_reference(spec) and reference is None:
        raise GraderError(f"the {kind} grader needs a reference answer; this row has none")
    if kind == "exact":
        s = float(_norm(answer) == _norm(reference or ""))
        return Grade(s, s >= thr, "" if s else "differs from the reference")
    if kind == "contains":
        s = float(_norm(reference or "") in _norm(answer))
        return Grade(s, s >= thr, "" if s else "reference not found in the answer")
    if kind == "regex":
        pat = spec.get("pattern") or re.escape((reference or "").strip())
        s = float(re.search(pat, answer or "", re.S | (re.I if spec.get("ignore_case", True) else 0)) is not None)
        return Grade(s, s >= thr, "" if s else "no match")
    if kind == "json":
        try:
            a, r = json.loads(_strip_fence(answer)), json.loads(_strip_fence(reference or ""))
        except ValueError as exc:
            return Grade(0.0, False, f"not JSON: {exc}")
        s = float(a == r)
        return Grade(s, s >= thr, "" if s else "JSON differs")
    if kind == "number":
        a, r = _last_number(answer), _last_number(reference or "")
        if a is None or r is None:
            return Grade(0.0, False, "no number found")
        tol = float(spec.get("tolerance", 1e-6))
        s = float(abs(a - r) <= tol * max(1.0, abs(r)))
        return Grade(s, s >= thr, "" if s else f"{a:g} != {r:g}")
    if kind == "judge":
        return _judge(spec, prompt, answer, reference, thr, default_model)
    if kind == "command":
        return _command(spec, prompt, answer, reference, thr, directory)
    raise GraderError(f"unknown grader {kind!r}")


def _judge(spec, prompt, answer, reference, thr, default_model) -> Grade:
    from .core.complete import CompletionError, complete

    model = spec.get("model") or default_model or "claude-code/haiku"
    ref_block = f"A reference answer (a good answer; others may also be right):\n<reference>\n{reference}\n</reference>\n\n" \
        if reference else ""
    text = JUDGE_PROMPT.format(rubric=spec.get("rubric") or DEFAULT_RUBRIC, prompt=prompt[:20000],
                               reference_block=ref_block, answer=(answer or "")[:20000])
    try:
        c = complete(model, [{"role": "user", "content": text}], max_tokens=200)
    except CompletionError as exc:
        raise GraderError(f"the judge ({model}) failed: {exc}") from None
    m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)\s*\|?\s*(.*)", c.text)
    if not m:
        return Grade(0.0, False, f"judge gave no score: {c.text[:120]}", c.cost)
    s = max(0.0, min(10.0, float(m.group(1)))) / 10
    return Grade(s, s >= thr, m.group(2).strip()[:200], c.cost)


def _command(spec, prompt, answer, reference, thr, directory) -> Grade:
    from .core.sandbox import wrap

    argv = wrap(["/bin/sh", "-c", spec["command"]] if isinstance(spec["command"], str) else list(spec["command"]))
    try:
        p = subprocess.run(argv, input=json.dumps({"prompt": prompt, "answer": answer, "reference": reference}),
                           capture_output=True, text=True, timeout=float(spec.get("timeout", 60)),
                           cwd=str(directory) if directory else None)
    except subprocess.TimeoutExpired:
        return Grade(0.0, False, "grader timed out")
    last = (p.stdout.strip().splitlines() or [""])[-1]
    try:
        s = max(0.0, min(1.0, float(last)))
    except ValueError:
        s = 1.0 if p.returncode == 0 else 0.0
    return Grade(s, s >= thr, (p.stderr.strip() or last)[-200:] if s < thr else "")


def describe(spec: dict[str, Any]) -> str:
    k = spec.get("type")
    if k == "judge":
        return f"judged by {spec.get('model') or 'the default model'}" + (" with a rubric" if spec.get("rubric") else "")
    if k == "command":
        return f"script: {spec.get('command') if isinstance(spec.get('command'), str) else shlex.join(spec['command'])}"
    if k == "regex" and spec.get("pattern"):
        return f"matches /{spec['pattern']}/"
    return {"exact": "exact match", "contains": "contains the reference", "regex": "matches the reference",
            "json": "same JSON as the reference", "number": "same number as the reference"}.get(k, str(k))
