"""The board: one project's loop, computed for any client (the web app, the TUI, the agent).

The loop is goal → data → grade → baseline → train → prove → ship. ``board()`` reads the
Trainfile, the datasets, the evals, the registry and the jobs, and says for each stage whether it is
done, running, next or waiting, with a one-line summary. It also returns the evidence: the
scoreboard (the latest held-out eval of every model on the focus dataset, on the same rows), the
training runs with their loss curve, and the gate between the newest version and the best baseline.

``grid()`` is the row-by-model view behind the scoreboard. Held-out rows are sealed from the agent;
the user may read them here (``reveal``) because the browser, unlike the agent's sandbox, is theirs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .datasets import DatasetError, Datasets, prompt_text, reference_of
from .evals import answers, decide, get, list_evals
from .registry import Registry

STAGES = ("goal", "data", "grade", "baseline", "train", "prove", "ship")
LABELS = {"goal": "goal", "data": "data", "grade": "grade", "baseline": "baseline", "train": "train",
          "prove": "prove", "ship": "ship"}


def _is_version(target: str) -> bool:
    from .core.complete import is_version

    return is_version(target)


def focus_dataset(state: Path, wanted: str | None = None) -> str | None:
    ds = Datasets(state)
    names = ds.names()
    if wanted and wanted in names:
        return wanted
    if not names:
        return None
    # the dataset touched last: newest eval, else newest meta
    evs = list_evals(state)
    for e in evs:
        if e["dataset"] in names:
            return e["dataset"]
    return max(names, key=lambda n: ds.meta(n).get("updated") or ds.meta(n).get("created") or 0)


def scoreboard(state: Path, dataset: str, live: str | None) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for e in list_evals(state, dataset):  # newest first
        if e["split"] != "heldout" or e["target"] in seen:
            continue
        s = e["summary"]
        rows = max(1, s.get("rows") or 0)
        seen[e["target"]] = {
            "target": e["target"], "kind": "version" if _is_version(e["target"]) else "model", "eval": e["id"],
            "score": s.get("mean_score", 0.0), "pass_rate": s.get("pass_rate", 0.0), "passed": s.get("passed", 0),
            "rows": s.get("rows", 0), "errors": s.get("errors", 0), "median_seconds": s.get("median_seconds"),
            "cost_usd": s.get("cost_usd", 0.0), "cost_per_1k": round((s.get("cost_usd") or 0.0) / rows * 1000, 4),
            "live": e["target"] == live, "time": e["time"]}
    return sorted(seen.values(), key=lambda r: (-r["score"], r["target"]))


def board(state: Path, directory: Path, *, dataset: str | None = None, jobs: list[dict[str, Any]] | None = None,
          proposals: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from . import trainfile
    from .budget import Ledger
    from .core.tools.data import summarize

    jobs = jobs or []
    tf_path = directory / "Trainfile"
    tf = trainfile.for_project(directory)
    reg = Registry(state)
    cur = reg.current()
    live = cur.version if cur else None
    name = focus_dataset(state, dataset)
    ds = Datasets(state)
    summaries = [summarize(state, n) for n in ds.names()]
    focus = next((s for s in summaries if s["name"] == name), None)

    ledger = Ledger(state, tf.budget)
    held = ledger.reserved()
    budget = {"spent_usd": round(ledger.committed() - held, 4), "reserved_usd": round(held, 4),
              "remaining_usd": round(ledger.remaining(), 4),
              "limit_usd": tf.budget.dollars, "text": tf.budget.describe()}

    versions = [v for v in reg.versions() if name and v.dataset == name]
    board_rows = scoreboard(state, name, live) if name else []
    baselines = [r for r in board_rows if r["kind"] == "model"]
    proven = {r["target"]: r for r in board_rows if r["kind"] == "version"}
    running = [j for j in jobs if j.get("state") == "running"]
    train_jobs = [j for j in jobs if j.get("kind") == "train" and (not name or j.get("dataset") in (None, name))]
    eval_running = [j for j in running if j.get("kind") == "eval" and j.get("dataset") == name]

    stages: dict[str, dict[str, Any]] = {}
    objective = (tf.objective or "").strip()
    default_goal = objective.startswith("better than the current model") or not objective
    stages["goal"] = {"state": "todo" if default_goal else "done",
                      "summary": "say what the model should get better at" if default_goal else objective}
    if not focus:
        stages["data"] = {"state": "todo", "summary": "no examples yet"}
    else:
        c = focus["counts"] or {}
        total = (c.get("train") or 0) + (c.get("heldout") or 0)
        ok = (c.get("heldout") or 0) > 0 and (c.get("labelled") or 0) > 0
        stages["data"] = {"state": "done" if ok else "todo",
                          "summary": f"{total} rows · {c.get('heldout', 0)} sealed" if ok else
                          (f"{total} rows, no answers yet" if total else "empty")}
    stages["grade"] = {"state": "done" if focus else "todo", "summary": focus["grader"] if focus else "—"}
    if eval_running and any(not _is_version(j.get("target") or "") for j in eval_running):
        stages["baseline"] = {"state": "running", "summary": "measuring…"}
    elif baselines:
        best = baselines[0]
        stages["baseline"] = {"state": "done", "summary": f"{len(baselines)} model{'s' * (len(baselines) > 1)} · "
                                                          f"best {round(best['score'] * 100)}%"}
    else:
        stages["baseline"] = {"state": "todo", "summary": "no scores yet"}
    tr = next((j for j in train_jobs if j.get("state") == "running"), None)
    if tr:
        pct = f" · {round((tr.get('progress') or 0) * 100)}%" if tr.get("progress") is not None else ""
        stages["train"] = {"state": "running", "summary": f"training{pct}"}
    elif versions:
        stages["train"] = {"state": "done", "summary": f"{versions[-1].version} trained"
                                                       + (f" · {len(versions)} versions" if len(versions) > 1 else "")}
    else:
        stages["train"] = {"state": "todo", "summary": "nothing trained"}
    newest = versions[-1].version if versions else None
    gate = None
    if newest and newest in proven and baselines:
        gate = decide(state, name, newest, baselines[0]["target"], allow_regressions=False)
    if eval_running and any(_is_version(j.get("target") or "") for j in eval_running):
        stages["prove"] = {"state": "running", "summary": "evaluating…"}
    elif newest and newest in proven:
        stages["prove"] = {"state": "done", "summary": f"{newest} {round(proven[newest]['score'] * 100)}%"
                                                       + (" · beats baseline" if gate and gate["promote"] else "")}
    else:
        stages["prove"] = {"state": "todo", "summary": "—" if not newest else f"{newest} not scored"}
    live_here = cur is not None and cur.dataset == name and name is not None
    stages["ship"] = {"state": "done" if live_here else "todo",
                      "summary": f"{live} live" if live_here else (f"{live} live (other data)" if live else "—")}

    current = next((k for k in STAGES if stages[k]["state"] == "running"), None) or \
        next((k for k in STAGES if stages[k]["state"] != "done"), "ship")
    for k in STAGES:
        stages[k]["key"] = k
        stages[k]["label"] = LABELS[k]
        stages[k]["current"] = k == current

    curves = []
    for j in sorted(train_jobs, key=lambda j: (j.get("time") or {}).get("created") or 0, reverse=True)[:3]:
        curves.append({k: j.get(k) for k in ("id", "title", "state", "progress", "detail", "cost_usd", "estimate_usd",
                                              "loss", "curve", "version", "error", "time", "backend")})

    return {
        "project": {"name": directory.name, "directory": str(directory)},
        "goal": objective, "trainfile": tf_path.exists(),
        "policy": {"trainer": tf.trainer, "base_model": tf.base_model},
        "budget": budget,
        "datasets": summaries, "dataset": name, "focus": focus,
        "stages": [stages[k] for k in STAGES], "current": current,
        "scoreboard": board_rows,
        "versions": [{"version": v.version, "base_model": v.base_model, "backend": v.backend, "decision": v.decision,
                      "created_at": v.created_at, "live": v.version == live} for v in versions],
        "live": {"version": live, "dataset": cur.dataset if cur else None},
        "training": curves, "gate": gate,
        "running": [{k: j.get(k) for k in ("id", "kind", "title", "progress", "detail", "dataset", "target")}
                    for j in running],
        "proposals": proposals or [],
        "time": time.time(),
    }


def grid(state: Path, dataset: str, *, reveal: bool = False, targets: list[str] | None = None) -> dict[str, Any]:
    """Every held-out row against every model's latest held-out eval: pass, score, and (revealed) the text."""
    ds = Datasets(state)
    if dataset not in ds.names():
        raise DatasetError(f"no dataset {dataset}")
    live = Registry(state).current()
    cols = scoreboard(state, dataset, live.version if live else None)
    if targets:
        cols = [c for c in cols if c["target"] in targets]
    evals = {c["target"]: get(state, c["eval"]) for c in cols}
    texts: dict[str, dict[str, Any]] = {}
    answer_text: dict[str, dict[str, dict[str, Any]]] = {}
    if reveal:
        for r in ds.heldout_rows(dataset):
            texts[r["id"]] = {"prompt": prompt_text(r), "reference": reference_of(r)}
        for t, rec in evals.items():
            answer_text[t] = {a["id"]: a for a in answers(state, rec["id"], sealed_ok=True)}
    ids: list[str] = []
    for rec in evals.values():
        for r in rec["rows"]:
            if r["id"] not in ids:
                ids.append(r["id"])
    if reveal:
        ids += [i for i in texts if i not in ids]
    rows = []
    for i in ids:
        cells = {}
        for t, rec in evals.items():
            r = next((x for x in rec["rows"] if x["id"] == i), None)
            if r is None:
                continue
            cell = {"passed": r.get("passed"), "score": r.get("score"), "error": r.get("error")}
            if reveal:
                a = answer_text.get(t, {}).get(i) or {}
                cell.update(answer=a.get("answer"), reason=a.get("reason") or r.get("reason"))
            cells[t] = cell
        passes = [c["passed"] for c in cells.values()]
        row = {"id": i, "cells": cells, "disagree": len(set(passes)) > 1}
        if reveal:
            row.update(texts.get(i) or {})
        rows.append(row)
    return {"dataset": dataset, "columns": cols, "rows": rows, "revealed": reveal}
