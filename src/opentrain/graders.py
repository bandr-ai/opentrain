"""Graders: how an answer is checked. Each returns a score in [0, 1] and a short reason.

A dataset names its grader in ``meta.json``::

    {"type": "exact"}                          same text as the reference (case and spacing ignored)
    {"type": "contains"}                       the reference appears in the answer
    {"type": "regex", "pattern": "..."}        the answer matches the pattern (or the reference, if none);
                                               with a capture group, what it captures must equal the
                                               reference (or what the pattern captures in it)
    {"type": "json"}                           the answer parses as JSON equal to the reference's JSON
    {"type": "number", "tolerance": 0.01}      the last number in the answer equals the reference's
    {"type": "judge", "rubric": "...",         a model grades against the rubric and the reference,
     "model": "claude-code/haiku"}             0-10, scaled to [0, 1]
    {"type": "command", "command": "..."}      a script: JSON {prompt, answer, reference} on stdin;
                                               exit 0 passes, or it prints a score on its last line
    {"type": "python", "path": "reward.py"}    your environment's reward: a file in the project with
                                               ``def reward(prompt, answer, reference)`` returning a
                                               0-1 score (or a bool, or ``(score, reason)``). Unlike a
                                               command it also runs inside the RL and rejection loops:
                                               its source travels with the job to the GPU.

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

KINDS = ("exact", "contains", "regex", "json", "number", "judge", "command", "python")
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
    if spec["type"] == "python":
        if not spec.get("path") and not spec.get("source"):
            raise GraderError("a python grader needs path: a .py file defining reward(prompt, answer, reference)")
        try:
            compile(spec.get("source") or "", spec.get("path") or "reward.py", "exec")
        except SyntaxError as exc:
            raise GraderError(f"the reward file does not parse: {exc}") from None
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
    thr = float(spec.get("threshold", 0.7 if kind in ("judge", "command", "python") else 1.0))
    if needs_reference(spec) and reference is None:
        raise GraderError(f"the {kind} grader needs a reference answer; this row has none")
    if kind == "exact":
        s = float(_norm(answer) == _norm(reference or ""))
        return Grade(s, s >= thr, "" if s else "differs from the reference")
    if kind == "contains":
        s = float(_norm(reference or "") in _norm(answer))
        return Grade(s, s >= thr, "" if s else "reference not found in the answer")
    if kind == "regex":
        flags = re.S | (re.I if spec.get("ignore_case", True) else 0)
        pat = spec.get("pattern") or re.escape((reference or "").strip())
        m = re.search(pat, answer or "", flags)
        if m and m.groups() and reference is not None:
            # a capture group extracts the answer (a letter, a number): it must equal the reference's
            got = _norm(next((g for g in m.groups() if g is not None), ""))
            rm = re.search(pat, reference, flags)
            want = _norm(next((g for g in rm.groups() if g is not None), "")) if rm and rm.groups() else \
                _norm(reference)
            s = float(got == want)
            return Grade(s, s >= thr, "" if s else f"extracted {got!r}, expected {want!r}")
        s = float(m is not None)
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
    if kind == "python":
        return _python(spec, prompt, answer, reference, thr, directory)
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


def reward_source(spec: dict[str, Any], directory: Path | None) -> str:
    """The python grader's code: carried in the spec (on a GPU), else read from the project."""
    if spec.get("source"):
        return spec["source"]
    path = Path(spec["path"]).expanduser()
    path = path if path.is_absolute() or directory is None else Path(directory) / path
    try:
        return path.read_text()
    except OSError as exc:
        raise GraderError(f"cannot read the reward file {path}: {exc}") from None


def load_reward(source: str, function: str = "reward"):
    """``reward(prompt, answer, reference) -> (score in [0, 1], reason)``, from the file's source."""
    ns: dict[str, Any] = {"__name__": "opentrain_reward"}
    exec(compile(source, "reward.py", "exec"), ns)  # noqa: S102 - the user's own reward file
    fn = ns.get(function)
    if not callable(fn):
        raise GraderError(f"the reward file defines no function {function}(prompt, answer, reference)")

    def call(prompt: str, answer: str, reference: str | None) -> tuple[float, str]:
        out = fn(prompt, answer, reference)
        reason = ""
        if isinstance(out, (tuple, list)):
            out, reason = out[0], str(out[1]) if len(out) > 1 else ""
        return max(0.0, min(1.0, float(out))), reason

    return call


_RUNNER = """
import json, sys
sys.path.insert(0, %r)
from opentrain.graders import load_reward
job = json.load(sys.stdin)
score, reason = load_reward(job["source"], job["function"])(job["prompt"], job["answer"], job["reference"])
print(json.dumps({"score": score, "reason": reason}))
"""


def _python(spec, prompt, answer, reference, thr, directory) -> Grade:
    """Here (evals), in a child process so a bad reward cannot take the server down; on the GPU,
    ``gpu_train`` calls ``load_reward`` in process."""
    import sys

    from .core.sandbox import wrap

    src = reward_source(spec, directory)
    pkg = str(Path(__file__).resolve().parent.parent)
    try:
        p = subprocess.run(wrap([sys.executable, "-c", _RUNNER % pkg]),
                           input=json.dumps({"source": src, "function": spec.get("function", "reward"),
                                             "prompt": prompt, "answer": answer, "reference": reference}),
                           capture_output=True, text=True, timeout=float(spec.get("timeout", 60)),
                           cwd=str(directory) if directory else None)
    except subprocess.TimeoutExpired:
        return Grade(0.0, False, "reward timed out")
    if p.returncode != 0:
        return Grade(0.0, False, "reward failed: " + (p.stderr.strip().splitlines() or ["?"])[-1][:200])
    out = json.loads(p.stdout.strip().splitlines()[-1])
    s = float(out["score"])
    return Grade(s, s >= thr, out.get("reason", "")[:200] if s < thr else "")


def describe(spec: dict[str, Any]) -> str:
    k = spec.get("type")
    if k == "judge":
        return f"judged by {spec.get('model') or 'the default model'}" + (" with a rubric" if spec.get("rubric") else "")
    if k == "command":
        return f"script: {spec.get('command') if isinstance(spec.get('command'), str) else shlex.join(spec['command'])}"
    if k == "python":
        return f"reward function in {spec.get('path') or 'the job'}"
    if k == "regex" and spec.get("pattern"):
        if re.compile(spec["pattern"]).groups:
            return f"extracts /{spec['pattern']}/ and compares it with the reference"
        return f"matches /{spec['pattern']}/"
    return {"exact": "exact match", "contains": "contains the reference", "regex": "matches the reference",
            "json": "same JSON as the reference", "number": "same number as the reference"}.get(k, str(k))
