"""Evaluate any model on a dataset's held-out rows, and decide promotion from the results.

An eval runs a target (an engine model like ``claude-code/haiku`` or a trained version like
``v3``) on held-out rows, grades each answer with the dataset's grader, and stores:

    .opentrain/evals/<eval_id>.json              summary + per-row id, score, pass, time (no content)
    .opentrain/heldout/evals/<eval_id>.jsonl     the answers themselves (sealed, like the rows)

Only ids and numbers leave the sealed area, so the training agent can read results without ever
seeing a held-out prompt or answer. The same code runs on training rows (``split="train"``) for a
quick look at a model's outputs, which may be shown.

The gate (``decide``) compares a candidate with the baseline on the same held-out rows: it must
score higher, must not lose rows the baseline passed (unless the Trainfile allows regressions),
needs enough rows to mean anything, and its training data must not contain held-out text.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .core.complete import Completion, CompletionError, complete
from .datasets import DatasetError, Datasets, prompt_of, prompt_text, reference_of, with_system
from .graders import Grade, GraderError, grade

WORKERS = 4


def _dir(state: Path) -> Path:
    return state / "evals"


def _sealed(state: Path) -> Path:
    return state / "heldout" / "evals"


def run(state: Path, directory: Path, dataset: str, target: str, *, split: str = "heldout", limit: int | None = None,
        judge_model: str | None = None, budget_check: Callable[[float], None] | None = None,
        progress: Callable[[int, int], None] | None = None, cancelled: Callable[[], bool] | None = None,
        ids: list[str] | None = None) -> dict[str, Any]:
    ds = Datasets(state)
    meta = ds.meta(dataset)
    rows = ds.heldout_rows(dataset) if split == "heldout" else ds.rows(dataset)
    if ids:
        want = set(ids)
        rows = [r for r in rows if r["id"] in want]
    if limit and split == "train" and not ids and int(limit) < len(rows):
        step = len(rows) / int(limit)  # a look at training rows: spread across the set, not its head
        rows = [rows[int(i * step)] for i in range(int(limit))]
    elif limit:
        rows = rows[: int(limit)]  # held-out: the same first rows for every model, so they compare
    if not rows:
        raise DatasetError(f"{dataset} has no {split} rows to evaluate")
    spec = meta.get("grader") or {"type": "judge"}
    eid = "eval_" + time.strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
    done = [0]
    spent = [0.0]

    def one(row: dict[str, Any]) -> dict[str, Any]:
        if cancelled and cancelled():
            return {"id": row["id"], "score": 0.0, "passed": False, "error": "cancelled"}
        if budget_check:
            budget_check(spent[0])
        out: dict[str, Any] = {"id": row["id"]}
        try:
            c: Completion = complete(target, prompt_of(with_system(row, meta.get("system"))), state=state)
            out.update(seconds=round(c.seconds, 3), tokens=c.output_tokens, cost=c.cost, answer=c.text)
            g: Grade = grade(spec, prompt_text(row), c.text, reference_of(row), directory=directory,
                             default_model=judge_model)
            out.update(score=round(g.score, 4), passed=g.passed, reason=g.reason, cost=c.cost + g.cost)
        except (CompletionError, GraderError) as exc:
            out.update(score=0.0, passed=False, error=str(exc)[:300])
        spent[0] += out.get("cost", 0.0)
        done[0] += 1
        if progress:
            progress(done[0], len(rows))
        return out

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, rows))
    errors = [r for r in results if r.get("error")]
    if errors and len(errors) == len(results):
        raise CompletionError(f"every row failed; first error: {errors[0]['error']}")
    secs = [r["seconds"] for r in results if "seconds" in r]
    summary = {"rows": len(results), "passed": sum(1 for r in results if r["passed"]),
               "mean_score": round(statistics.fmean(r["score"] for r in results), 4),
               "errors": len(errors), "median_seconds": round(statistics.median(secs), 3) if secs else None,
               "cost_usd": round(sum(r.get("cost", 0.0) for r in results), 4), "wall_seconds": round(time.time() - t0, 1)}
    summary["pass_rate"] = round(summary["passed"] / summary["rows"], 4)
    record = {"id": eid, "dataset": dataset, "target": target, "split": split, "grader": spec, "time": time.time(),
              "summary": summary,
              "rows": [{k: r.get(k) for k in ("id", "score", "passed", "seconds", "tokens", "error")}
                       | ({"reason": r.get("reason")} if split != "heldout" else {}) for r in results]}
    _dir(state).mkdir(parents=True, exist_ok=True)
    (_dir(state) / f"{eid}.json").write_text(json.dumps(record, indent=1))
    answers_dir = _sealed(state) if split == "heldout" else _dir(state) / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    with open(answers_dir / f"{eid}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps({"id": r["id"], "answer": r.get("answer"), "reason": r.get("reason"),
                                "error": r.get("error")}, ensure_ascii=False) + "\n")
    return record


def answers(state: Path, eval_id: str, *, sealed_ok: bool = False) -> list[dict[str, Any]]:
    rec = get(state, eval_id)
    if rec["split"] == "heldout" and not sealed_ok:
        raise DatasetError("answers on held-out rows are sealed")
    p = (_sealed(state) if rec["split"] == "heldout" else _dir(state) / "answers") / f"{eval_id}.jsonl"
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def get(state: Path, eval_id: str) -> dict[str, Any]:
    p = _dir(state) / f"{eval_id}.json"
    if not p.exists() or "/" in eval_id:
        raise DatasetError(f"no eval {eval_id}")
    return json.loads(p.read_text())


def list_evals(state: Path, dataset: str | None = None, target: str | None = None) -> list[dict[str, Any]]:
    d = _dir(state)
    out = []
    for p in sorted(d.glob("eval_*.json"), reverse=True) if d.exists() else []:
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            continue
        if (dataset and rec["dataset"] != dataset) or (target and rec["target"] != target):
            continue
        out.append({k: rec[k] for k in ("id", "dataset", "target", "split", "time", "summary")})
    return out


def latest(state: Path, dataset: str, target: str) -> dict[str, Any] | None:
    """The held-out eval that speaks for ``target``: the newest of those covering the most rows, so a
    quick eval on a few rows never replaces a full one at the gate."""
    try:
        grader = Datasets(state).meta(dataset).get("grader") or {"type": "judge"}
    except DatasetError:
        grader = None
    recs = [get(state, e["id"]) for e in list_evals(state, dataset, target) if e["split"] == "heldout"]
    # only evals graded the way the dataset grades now: a changed grader makes older scores incomparable
    held = [r for r in recs if grader is None or _same_grader(r.get("grader"), grader)]
    if not held:
        return None
    most = max(r["summary"]["rows"] for r in held)
    return next(r for r in held if r["summary"]["rows"] == most)


def _same_grader(a: dict[str, Any] | None, b: dict[str, Any]) -> bool:
    strip = lambda g: {k: v for k, v in (g or {"type": "judge"}).items() if k != "source"}  # noqa: E731
    return strip(a) == strip(b)


def contamination(state: Path, dataset: str, training_file: Path) -> list[dict[str, Any]]:
    """Held-out rows whose prompt or reference appears (13-gram overlap >= 50%) in the training file."""
    from .core.integrity import OVERLAP, ngrams

    if not training_file.exists():
        return []
    train = ngrams(training_file.read_text(errors="replace"))
    if not train:
        return []
    out = []
    for r in Datasets(state).heldout_rows(dataset):
        for label, text in (("prompt", prompt_text(r)), ("reference", reference_of(r) or "")):
            g = ngrams(text)
            if len(g) >= 3 and len(g & train) / len(g) >= OVERLAP:
                out.append({"id": r["id"], "what": label})
    return out


def decide(state: Path, dataset: str, candidate: str, baseline: str, *, allow_regressions: bool = False,
           min_rows: int | None = None) -> dict[str, Any]:
    from .core.integrity import min_heldout

    need = min_rows if min_rows is not None else min_heldout()
    cand, base = latest(state, dataset, candidate), latest(state, dataset, baseline)
    reasons: list[str] = []
    if cand is None:
        return {"promote": False, "reasons": [f"{candidate} has no held-out eval on {dataset} under its current "
                                              "grader yet"]}
    if base is None:
        return {"promote": False, "reasons": [f"the baseline {baseline} has no held-out eval on {dataset} under its "
                                              "current grader yet (re-evaluate it after a grader change)"]}
    c_rows = {r["id"]: r for r in cand["rows"]}
    b_rows = {r["id"]: r for r in base["rows"]}
    common = sorted(set(c_rows) & set(b_rows))
    if len(common) < need:
        reasons.append(f"only {len(common)} held-out rows in common (need {need})")
    c_mean = statistics.fmean(c_rows[i]["score"] for i in common) if common else 0.0
    b_mean = statistics.fmean(b_rows[i]["score"] for i in common) if common else 0.0
    regressed = [i for i in common if b_rows[i]["passed"] and not c_rows[i]["passed"]]
    fixed = [i for i in common if c_rows[i]["passed"] and not b_rows[i]["passed"]]
    if c_mean <= b_mean:
        reasons.append(f"{candidate} scores {c_mean:.3f}, not above {baseline} at {b_mean:.3f}")
    if regressed and not allow_regressions:
        reasons.append(f"{len(regressed)} rows regressed")
    return {"promote": not reasons, "reasons": reasons, "rows": len(common), "candidate": candidate,
            "baseline": baseline, "candidate_score": round(c_mean, 4), "baseline_score": round(b_mean, 4),
            "fixed": fixed, "regressed": regressed, "candidate_eval": cand["id"], "baseline_eval": base["id"]}
