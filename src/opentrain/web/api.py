"""JSON views of a ``.opentrain`` state folder, one function per screen of the web UI.

Every function takes the state folder and returns plain JSON-ready data. Nothing here writes to
the state except :func:`revert` and :func:`save_policy`, which the server exposes as POSTs.

Files written by other parts of OpenTrain are read defensively: any of them may be absent, empty
or half-written, and a view degrades to "nothing yet" rather than failing.

Held-out tasks: the UI may show how many there are and which ids passed or failed for a version,
but never what they ask or how they are tested. Nothing in this module opens a file under
``heldout/`` other than ``ids.json`` and ``scores/``, and per-task score rows are reduced to
id, passed, seconds and tokens before they leave.
"""

from __future__ import annotations

import json
import re
import statistics
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from .. import trainfile as trainfile_mod
from ..budget import Ledger
from ..record import Run, RunRecord
from ..registry import Registry, Version
from ..tasks import Store

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SCORE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}$")
STALE_AFTER_S = 2 * 3600
PREVIEW_CHARS = 320
SCRUB_KEYS = {"instruction", "instructions", "instruction_md", "prompt", "messages", "transcript",
              "tests", "test_files", "solution", "patch"}


class NotFound(LookupError):
    pass


# ---------------------------------------------------------------- helpers


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line being written right now
        if isinstance(row, dict):
            out.append(row)
    return out


def _parse_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            d = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 3) if xs else None


def _scrub(obj: Any) -> Any:
    """Drop fields that could carry task content (for files whose shape we do not own)."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if str(k).lower() not in SCRUB_KEYS}
    if isinstance(obj, list):
        return [_scrub(x) for x in obj]
    return obj


def trainfile_path(state: Path) -> Path:
    """The Trainfile sits beside ``.opentrain/``."""
    return state.parent / "Trainfile"


def _trainfile(state: Path) -> tuple[trainfile_mod.Trainfile, str | None]:
    p = trainfile_path(state)
    try:
        if p.is_file():
            return trainfile_mod.parse(p.read_text(), str(p)), None
    except trainfile_mod.TrainfileError as e:
        return trainfile_mod.parse(trainfile_mod.DEFAULT), str(e)
    return trainfile_mod.parse(trainfile_mod.DEFAULT), None


def _registry(state: Path) -> Registry | None:
    return Registry(state) if (state / "models").is_dir() else None


def _versions(state: Path) -> list[Version]:
    reg = _registry(state)
    if reg is None:
        return []
    try:
        return reg.versions()
    except (ValueError, TypeError, OSError):
        return []


def _current_name(state: Path) -> str | None:
    data = _read_json(state / "models" / "registry.json")
    return data.get("current") if isinstance(data, dict) else None


def _record(state: Path) -> RunRecord | None:
    return RunRecord(state) if (state / "runs").is_dir() else None


def _count_tasks(folder: Path) -> int:
    """Count task folders without reading them (used for the held-out store)."""
    try:
        return sum(1 for p in folder.iterdir() if (p / "task.toml").is_file())
    except OSError:
        return 0


def _training_store(state: Path) -> Store | None:
    return Store(state) if (state / "tasks").is_dir() else None


# ---------------------------------------------------------------- runs


def _note_text(data: dict[str, Any]) -> str:
    for k in ("text", "note", "message", "summary"):
        if isinstance(data.get(k), str):
            return data[k]
    return json.dumps(data, default=str)[:400]


def _gate_info(data: dict[str, Any]) -> dict[str, Any]:
    decision = data.get("decision")
    if not isinstance(decision, str):
        promote = data.get("promote", data.get("promoted"))
        decision = "promoted" if promote is True else "rejected" if promote is False else "unknown"
    reasons = data.get("reasons")
    if isinstance(reasons, str):
        reasons = [reasons]
    return {"decision": decision, "reasons": reasons or [], "version": data.get("version"),
            "summary": data.get("summary") if isinstance(data.get("summary"), dict) else {}}


def _job_file(state: Path, job_id: Any) -> dict[str, Any] | None:
    if not isinstance(job_id, str) or not RUN_ID.match(job_id):
        return None
    data = _read_json(state / "jobs" / f"{job_id}.json")
    if not isinstance(data, dict):
        return None
    logs = data.get("logs")
    if not isinstance(logs, list):
        try:
            logs = (state / "jobs" / f"{job_id}.log").read_text(errors="replace").splitlines()
        except OSError:
            logs = None
    return {
        "job_id": data.get("job_id", job_id),
        "state": data.get("state"),
        "backend": data.get("backend"),
        "cost_usd": _num(data.get("cost_usd")),
        "progress": _num(data.get("progress")),
        "detail": data.get("detail", ""),
        "checkpoint": data.get("checkpoint"),
        "logs": [str(x) for x in logs[-200:]] if isinstance(logs, list) else [],
        "updated_at": data.get("updated_at"),
    }


def _run_summary(state: Path, run: Run, events: list[dict[str, Any]],
                 by_run: dict[str, list[str]]) -> dict[str, Any]:
    opened = events[0] if events else {}
    last = events[-1] if events else {}
    od = opened.get("data", {}) if isinstance(opened.get("data"), dict) else {}
    closed = last.get("kind") == "closed"
    start, end = _parse_at(opened.get("at")), _parse_at(last.get("at"))
    now = datetime.now(timezone.utc)
    if closed:
        cd = last.get("data") or {}
        status = cd.get("status") or ("failed" if cd.get("error") else "done")
    elif not events:
        status = "empty"
    else:
        status = "live" if end and (now - end).total_seconds() < STALE_AFTER_S else "stalled"
    spend = 0.0
    tokens = 0
    gate = note = job_id = None
    for e in events:
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        k = e.get("kind")
        if k == "spend":
            spend += _num(d.get("usd")) or 0.0
            tokens += int(_num(d.get("tokens")) or 0) + int(_num(d.get("tokens_in")) or 0) \
                + int(_num(d.get("tokens_out")) or 0)
        elif k == "gate":
            gate = _gate_info(d)
        elif k == "note":
            note = _note_text(d)
        elif k == "job":
            job_id = d.get("job_id") or d.get("id") or job_id
    job = _job_file(state, job_id) if job_id else None
    return {
        "run_id": run.run_id,
        "trigger": od.get("trigger"),
        "opened_at": opened.get("at"),
        "last_at": last.get("at"),
        "seconds": round(((now if status == "live" else end) - start).total_seconds(), 1)
        if start and end else None,
        "status": status,
        "events": len(events),
        "spend_usd": round(spend, 4),
        "tokens": tokens,
        "job": job,
        "gate": gate,
        "note": note,
        "versions": by_run.get(run.run_id, []),
    }


def _by_run(state: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for v in _versions(state):
        if v.run_id:
            out.setdefault(v.run_id, []).append(v.version)
    return out


def _safe_events(run: Run) -> list[dict[str, Any]]:
    return [e for e in _read_jsonl(run.events_path) if "kind" in e]


def runs(state: Path) -> dict[str, Any]:
    rec = _record(state)
    if rec is None:
        return {"runs": []}
    by_run = _by_run(state)
    out = []
    for r in rec.runs():
        out.append(_run_summary(state, r, _safe_events(r), by_run))
    return {"runs": out, "live": sum(1 for r in out if r["status"] == "live")}


def get_run(state: Path, run_id: str) -> Run:
    if not RUN_ID.match(run_id or ""):
        raise NotFound(f"bad run id {run_id!r}")
    rec = _record(state)
    if rec is None:
        raise NotFound("no runs yet")
    try:
        return rec.get(run_id)
    except Exception as e:  # RecordError, or a race with a folder being created
        raise NotFound(str(e)) from e


def run(state: Path, run_id: str) -> dict[str, Any]:
    r = get_run(state, run_id)
    events = _safe_events(r)
    try:
        verified = r.verify()
    except (KeyError, ValueError, OSError):
        verified = False
    return {**_run_summary(state, r, events, _by_run(state)), "verified": verified,
            "timeline": [_scrub_event(e) for e in events]}


def _scrub_event(e: dict[str, Any]) -> dict[str, Any]:
    return {"seq": e.get("seq"), "at": e.get("at"), "kind": e.get("kind"), "data": _scrub(e.get("data")),
            "hash": (e.get("hash") or "")[:12]}


def events_since(state: Path, run_id: str, after: int = -1) -> tuple[list[dict[str, Any]], bool]:
    """New events with ``seq > after``, and whether the run is closed. Used by the SSE stream."""
    r = get_run(state, run_id)
    events = _safe_events(r)
    new = [_scrub_event(e) for e in events if isinstance(e.get("seq"), int) and e["seq"] > after]
    return new, bool(events) and events[-1].get("kind") == "closed"


# ---------------------------------------------------------------- scores


def _score_row(task_id: str, row: Any) -> dict[str, Any] | None:
    if isinstance(row, bool):
        return {"task_id": task_id, "passed": row, "seconds": None, "tokens": None}
    if not isinstance(row, dict):
        return None
    passed = row.get("passed", row.get("pass", row.get("ok")))
    return {"task_id": task_id, "passed": passed if isinstance(passed, bool) else None,
            "seconds": _num(row.get("seconds")), "tokens": _num(row.get("tokens"))}


def normalize_scores(obj: Any) -> dict[str, dict[str, Any]]:
    """Per-task scores from any of the shapes OpenTrain writes, reduced to id/passed/seconds/tokens."""
    out: dict[str, dict[str, Any]] = {}
    if isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict) and isinstance(row.get("task_id"), str):
                r = _score_row(row["task_id"], row)
                if r:
                    out[r["task_id"]] = r
        return out
    if not isinstance(obj, dict):
        return out
    for key in ("scores", "tasks", "per_task", "results"):
        if key in obj and isinstance(obj[key], (list, dict)):
            return normalize_scores(obj[key])
    for k, v in obj.items():
        if isinstance(v, (dict, bool)):
            r = _score_row(str(k), v)
            if r and (r["passed"] is not None or r["seconds"] is not None):
                out[str(k)] = r
    return out


def _score_file_names(state: Path) -> list[str]:
    d = state / "heldout" / "scores"
    try:
        return sorted(p.stem for p in d.glob("*.json"))
    except OSError:
        return []


def version_scores(state: Path, name: str, versions: list[Version] | None = None) -> dict[str, dict[str, Any]]:
    """Held-out scores for a version (or ``base``): the evaluator's score file first, then the
    registry entry, then the eval events of the run that made it."""
    if not SCORE_NAME.match(name or ""):
        return {}
    got = normalize_scores(_read_json(state / "heldout" / "scores" / f"{name}.json"))
    if got:
        return got
    versions = versions if versions is not None else _versions(state)
    v = next((x for x in versions if x.version == name), None)
    if v is None:
        return {}
    got = normalize_scores(v.scores)
    if got or not v.run_id:
        return got
    try:
        r = get_run(state, v.run_id)
    except NotFound:
        return {}
    for e in _safe_events(r):
        if e.get("kind") != "eval" or not isinstance(e.get("data"), dict):
            continue
        d = e["data"]
        if d.get("version") not in (None, name):
            continue
        model = str(d.get("model") or "")
        if model.startswith("opentrain/") and model.split("/", 1)[1] not in (name, "current"):
            continue
        if isinstance(d.get("task_id"), str):
            row = _score_row(d["task_id"], d)
            if row:
                got[row["task_id"]] = row
        else:
            got.update(normalize_scores(d))
    return got


def score_summary(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(scores.values())
    judged = [r for r in rows if r["passed"] is not None]
    passed = [r for r in judged if r["passed"]]
    return {
        "tasks": len(rows),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(judged), 4) if judged else None,
        "median_seconds": _median([r["seconds"] for r in passed if r["seconds"] is not None]),
    }


# ---------------------------------------------------------------- evals


def evals(state: Path, a: str | None = None, b: str | None = None) -> dict[str, Any]:
    versions = _versions(state)
    names = [v.version for v in versions]
    names += [n for n in _score_file_names(state) if n not in names]
    scored = {n: version_scores(state, n, versions) for n in names}
    choices = [n for n in names if scored[n]]
    if "base" in choices:
        choices = ["base"] + [n for n in choices if n != "base"]
    cur = _current_name(state)
    if b not in choices:
        b = cur if cur in choices else (choices[-1] if choices else None)
    if a not in choices or a == b:
        i = choices.index(b) if b in choices else 0
        a = choices[i - 1] if i > 0 else None
    sa, sb = scored.get(a, {}) if a else {}, scored.get(b, {}) if b else {}
    rows = []
    for tid in sorted(set(sa) | set(sb)):
        ra, rb = sa.get(tid), sb.get(tid)
        pa, pb = (ra or {}).get("passed"), (rb or {}).get("passed")
        if ra is None:
            change = "new"
        elif rb is None:
            change = "dropped"
        elif pa and pb is False:
            change = "regressed"
        elif pa is False and pb:
            change = "fixed"
        elif pa and pb:
            change = "pass"
        else:
            change = "fail"
        ta, tb = (ra or {}).get("seconds"), (rb or {}).get("seconds")
        delta = round(tb - ta, 2) if ta is not None and tb is not None else None
        rows.append({"task_id": tid, "a_passed": pa, "b_passed": pb, "a_seconds": ta, "b_seconds": tb,
                     "delta_seconds": delta,
                     "delta_pct": round(100 * delta / ta, 1) if delta is not None and ta else None,
                     "change": change})
    order = {"regressed": 0, "fixed": 1, "new": 2, "dropped": 3, "pass": 4, "fail": 5}
    rows.sort(key=lambda r: (order[r["change"]], r["task_id"]))
    both = [r for r in rows if r["change"] == "pass" and r["delta_seconds"] is not None]
    return {
        "choices": choices,
        "a": a, "b": b,
        "a_summary": score_summary(sa) if a else None,
        "b_summary": score_summary(sb) if b else None,
        "counts": {k: sum(1 for r in rows if r["change"] == k) for k in order},
        "median_seconds_both_pass": {
            "a": _median([r["a_seconds"] for r in both]), "b": _median([r["b_seconds"] for r in both]),
            "tasks": len(both)},
        "rows": rows,
        "heldout_tasks": _count_tasks(state / "heldout"),
    }


# ---------------------------------------------------------------- versions


def _run_training_data(state: Path, run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {}
    try:
        r = get_run(state, run_id)
    except NotFound:
        return {}
    out: dict[str, Any] = {}
    for e in _safe_events(r):
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        if e.get("kind") == "data":
            ids = d.get("task_ids") or d.get("ids") or d.get("tasks")
            out["tasks"] = len(ids) if isinstance(ids, (list, dict)) else _num(d.get("count"))
            out["data_hash"] = d.get("sha256") or d.get("hash") or d.get("digest")
            out["examples"] = _num(d.get("examples"))
        elif e.get("kind") == "recipe":
            out["recipe"] = {k: v for k, v in _scrub(d).items() if isinstance(v, (str, int, float, bool))}
        elif e.get("kind") == "gate":
            out["gate"] = _gate_info(d)
        elif e.get("kind") == "spend":
            out["spend_usd"] = round(out.get("spend_usd", 0.0) + (_num(d.get("usd")) or 0.0), 4)
    return out


def versions(state: Path) -> dict[str, Any]:
    vs = _versions(state)
    cur = _current_name(state)
    reg = _read_json(state / "models" / "registry.json")
    history = reg.get("history", []) if isinstance(reg, dict) else []
    out = []
    for v in vs:
        per_task = version_scores(state, v.version, vs)
        summ = score_summary(per_task)
        sc = v.scores if isinstance(v.scores, dict) else {}
        pass_rate = _num(sc.get("pass_rate"))
        med = _num(sc.get("median_seconds"))
        out.append({
            "version": v.version,
            "model_name": v.model_name,
            "run_id": v.run_id,
            "base_model": v.base_model,
            "backend": v.backend,
            "checkpoint": v.checkpoint,
            "decision": v.decision,
            "created_at": v.created_at,
            "current": v.version == cur,
            "pass_rate": pass_rate if pass_rate is not None else summ["pass_rate"],
            "median_seconds": med if med is not None else summ["median_seconds"],
            "heldout_scored": summ["tasks"],
            "trained_on": _run_training_data(state, v.run_id),
        })
    can_revert = cur is not None
    previous = None
    if cur:
        live = {v.version: v.decision for v in vs}
        for h in reversed(history):
            if h.get("to") and h["to"] != cur and live.get(h["to"]) == "promoted":
                previous = h["to"]
                break
    base = version_scores(state, "base", vs)
    return {"versions": out, "current": cur, "history": history[-50:], "can_revert": can_revert,
            "revert_to": previous, "base": score_summary(base) if base else None}


def revert(state: Path, why: str = "rolled back from the web UI") -> dict[str, Any]:
    reg = Registry(state)
    before = _current_name(state)
    now = reg.revert(why)
    return {"reverted": before, "current": now.version if now else None}


# ---------------------------------------------------------------- home


def _serving_share(state: Path, days: int = 7) -> dict[str, Any]:
    rows = _read_jsonl(state / "serving" / "requests.jsonl")
    now = datetime.now(timezone.utc)

    def ours(r: dict[str, Any]) -> bool:
        if r.get("fallback") is True or r.get("handoff") is True:
            return False
        if isinstance(r.get("ours"), bool):
            return r["ours"]
        model = str(r.get("model") or r.get("served_by") or "")
        return model.startswith("opentrain") or model == "local"

    # Requests that failed outright did no work for anyone; handoffs count as the frontier's work.
    rows = [r for r in rows if not (isinstance(r.get("status"), int) and r["status"] >= 400)]
    recent = [r for r in rows if (t := _parse_at(r.get("at") or r.get("ts"))) and (now - t).days < days]
    pool = recent if recent else rows
    n_ours = sum(1 for r in pool if ours(r))
    by_version: dict[str, int] = {}
    for r in pool:
        if ours(r):
            key = str(r.get("version") or r.get("model") or "?")
            by_version[key] = by_version.get(key, 0) + 1
    return {
        "requests": len(pool),
        "ours": n_ours,
        "share": round(n_ours / len(pool), 4) if pool else None,
        "window_days": days if recent else None,
        "all_time": len(rows),
        "by_version": by_version,
    }


def home(state: Path) -> dict[str, Any]:
    tf, tf_error = _trainfile(state)
    rec = _record(state)
    latest = None
    last_note = None
    if rec is not None:
        by_run = _by_run(state)
        for r in rec.runs():
            ev = _safe_events(r)
            s = _run_summary(state, r, ev, by_run)
            if latest is None:
                latest = s
            if s["note"]:
                last_note = {"run_id": s["run_id"], "at": s["last_at"], "text": s["note"]}
                break
    committed = None
    if (state / "runs").is_dir():
        try:
            per_run = tf.budget.period_days is None
            committed = Ledger(state, tf.budget).committed(
                run_id=latest["run_id"] if per_run and latest else None)
        except (ValueError, KeyError, TypeError, OSError):
            committed = None
    vs = versions(state)
    cur = next((v for v in vs["versions"] if v["current"]), None)
    store = _training_store(state)
    bench = _latest_bench(state)
    return {
        "state": str(state),
        "opentrain": __version__,
        "exists": state.is_dir(),
        "share": _serving_share(state),
        "current": cur,
        "versions": len(vs["versions"]),
        "budget": {"dollars": tf.budget.dollars, "period_days": tf.budget.period_days,
                   "describe": tf.budget.describe(), "committed_usd": committed,
                   "remaining_usd": round(tf.budget.dollars - committed, 4) if committed is not None else None},
        "goal": tf.goal,
        "trainfile_error": tf_error,
        "latest_run": latest,
        "last_note": last_note,
        "tasks": {"training": len(store.ids()) if store else 0, "heldout": _count_tasks(state / "heldout")},
        "bench": ({"at": bench.get("at"), "yield": (bench.get("yield") or {}).get("verified_candidates"),
                   "sessions": (bench.get("yield") or {}).get("sessions"),
                   "model_share_pct": (bench.get("time") or {}).get("model_share_of_working_time_pct")}
                  if bench else None),
        "trend": [{"version": v["version"], "pass_rate": v["pass_rate"], "median_seconds": v["median_seconds"]}
                  for v in vs["versions"]],
    }


# ---------------------------------------------------------------- data


def _latest_bench(state: Path) -> dict[str, Any] | None:
    files = sorted((state / "bench").glob("bench-*.json")) if (state / "bench").is_dir() else []
    for f in reversed(files):
        data = _read_json(f)
        if isinstance(data, dict):
            return {**data, "file": f.name}
    return None


def _attempts(state: Path, task_id: str) -> dict[str, Any] | None:
    d = state / "attempts" / task_id
    if not d.is_dir():
        return None
    rows = [x for x in (_read_json(p) for p in sorted(d.glob("*.json"))) if isinstance(x, dict)]
    if not rows:
        return None
    passed = [r for r in rows if r.get("passed") is True]
    secs = [s for r in passed if (s := _num(r.get("seconds"))) is not None]
    return {"count": len(rows), "passed": len(passed), "best_seconds": min(secs) if secs else None,
            "median_seconds": _median(secs)}


def _task_row(store: Store, state: Path, tid: str) -> dict[str, Any]:
    folder = store.folder(tid)
    row: dict[str, Any] = {"task_id": tid}
    try:
        meta = tomllib.loads((folder / "task.toml").read_text())
        ot = meta.get("opentrain", {})
        row.update({"repo": ot.get("repo"), "family": ot.get("family") or None,
                    "created_at": ot.get("created_at") or None, "test_command": ot.get("test_command"),
                    "base_commit": (ot.get("base_commit") or "")[:10]})
    except (OSError, ValueError):
        row["error"] = "task.toml unreadable"
    try:
        text = (folder / "instruction.md").read_text()
        row["preview"] = text[:PREVIEW_CHARS]
        row["truncated"] = len(text) > PREVIEW_CHARS
    except OSError:
        row["preview"] = ""
    ctl = _read_json(folder / "controls.json")
    if isinstance(ctl, dict):
        row["controls"] = {"reference": ctl.get("reference"), "noop": ctl.get("noop"),
                           "wrong": ctl.get("wrong"), "seconds": _num(ctl.get("seconds"))}
    row["attempts"] = _attempts(state, tid)
    return row


def data(state: Path) -> dict[str, Any]:
    bench = _latest_bench(state)
    store = _training_store(state)
    tasks = [_task_row(store, state, t) for t in store.ids()] if store else []
    tasks.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    heldout_ids = _read_json(state / "heldout" / "ids.json")
    train_ids = _read_json(state / "tasks" / "train.json")
    y = (bench or {}).get("yield") or {}
    return {
        "bench": {
            "file": bench.get("file"), "at": bench.get("at"), "source": bench.get("source"),
            "yield": {k: y.get(k) for k in ("sessions", "verified_candidates", "repairable", "yield_per_100",
                                            "coding_sessions", "yield_per_coding_session_pct",
                                            "yield_with_repair_per_100", "multi_prompt_candidates")},
            "funnel": y.get("funnel") or [],
            "skip_details": y.get("skip_details") or {},
            "time": bench.get("time"),
            "split": bench.get("split"),
        } if bench else None,
        "built": {
            "training": len(tasks),
            "heldout": _count_tasks(state / "heldout"),
            "split_train_ids": len(train_ids.get("ids", [])) if isinstance(train_ids, dict) else None,
            "split_heldout_ids": len(heldout_ids.get("ids", [])) if isinstance(heldout_ids, dict) else None,
        },
        "tasks": tasks,
    }


def task(state: Path, task_id: str) -> dict[str, Any]:
    """One TRAINING task, in full. Held-out tasks are never served."""
    store = _training_store(state)
    if store is None or not RUN_ID.match(task_id or ""):
        raise NotFound(task_id)
    try:
        if task_id not in store:
            raise NotFound(task_id)
        folder = store.folder(task_id)
    except ValueError as e:
        raise NotFound(task_id) from e
    row = _task_row(store, state, task_id)
    row.pop("preview", None)
    row["instruction"] = (folder / "instruction.md").read_text() if (folder / "instruction.md").exists() else ""
    return row


# ---------------------------------------------------------------- race


def race(state: Path) -> dict[str, Any]:
    d = state / "races"
    files = sorted(d.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name)) if d.is_dir() else []
    for f in reversed(files):
        body = _read_json(f)
        if body is not None:
            return {"file": f.name, "race": _scrub(body), "count": len(files)}
    return {"file": None, "race": None, "count": 0}


# ---------------------------------------------------------------- skills


_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)


def _skill(path: Path, source: str) -> dict[str, Any] | None:
    try:
        text = path.read_text()[:200_000]
    except OSError:
        return None
    meta: dict[str, str] = {}
    body = text
    m = _FRONT.match(text)
    if m:
        body = text[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                meta[k.strip().lower()] = v.strip().strip("\"'")
    return {"name": meta.get("name") or path.parent.name, "description": meta.get("description", ""),
            "source": source, "path": str(path), "body": body, "meta": meta}


def skill_roots(state: Path) -> list[tuple[Path, str]]:
    roots = [(state.parent / "skills", "repo"), (state / "skills", "installed")]
    bundled = Path(__file__).resolve().parents[3] / "skills"
    if bundled.is_dir():
        roots.append((bundled, "opentrain"))
    return roots


def skills(state: Path) -> dict[str, Any]:
    seen: set[Path] = set()
    out = []
    for root, source in skill_roots(state):
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*/SKILL.md")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            s = _skill(p, source)
            if s:
                out.append(s)
    return {"skills": out, "roots": [str(r) for r, _ in skill_roots(state)]}


# ---------------------------------------------------------------- policy


def _describe(tf: trainfile_mod.Trainfile) -> dict[str, Any]:
    return {
        "goal": tf.goal,
        "budget": tf.budget.describe(),
        "promote": tf.promote,
        "allow_regressions": tf.allow_regressions,
        "runtime": tf.runtime,
        "trainer": tf.trainer,
        "base_model": tf.base_model,
        "data": tf.data,
        "uploads_need_approval": tf.uploads_need_approval,
        "heldout": f"{tf.heldout_fraction:.0%}",
        "eval_cap": tf.eval_cap,
        "watch": tf.watch.__dict__ if tf.watch else None,
        "serve": tf.serve,
        "provider": tf.provider,
    }


def policy(state: Path) -> dict[str, Any]:
    p = trainfile_path(state)
    exists = p.is_file()
    text = p.read_text() if exists else trainfile_mod.DEFAULT
    try:
        parsed, error = _describe(trainfile_mod.parse(text, str(p))), None
    except trainfile_mod.TrainfileError as e:
        parsed, error = None, str(e)
    return {"path": str(p), "exists": exists, "text": text, "parsed": parsed, "error": error,
            "keys": list(trainfile_mod.KEYS)}


def save_policy(state: Path, text: str) -> dict[str, Any]:
    """Validate with the real parser, then write. Raises ``TrainfileError`` and writes nothing if invalid."""
    if not isinstance(text, str):
        raise trainfile_mod.TrainfileError("the Trainfile must be text")
    if len(text) > 64_000:
        raise trainfile_mod.TrainfileError("the Trainfile is too long")
    p = trainfile_path(state)
    tf = trainfile_mod.parse(text, str(p))
    if not text.endswith("\n"):
        text += "\n"
    tmp = p.with_name(".Trainfile.tmp")
    tmp.write_text(text)
    tmp.replace(p)
    return {"saved": str(p), "parsed": _describe(tf)}
