"""``opentrain report``: what happened in a project, as counts only, to send to whoever helps you.

Beta users paste this into an issue or a message so we can see where the pipeline worked and where
it broke on their machine, without seeing their work. It holds no code, no prompts, no file contents,
no repo or file names and no theme names (unless ``names=True``): only what was found, built,
measured and spent, and the first line of each error with the home folder shortened and secrets
scrubbed.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def _clean(text: str) -> str:
    from .traces.scrub import scrub

    text = scrub(str(text).splitlines()[0] if text else "")
    text = text.replace(str(Path.home()), "~")
    text = re.sub(r"(/[\w.@-]+){3,}", "<path>", text)  # paths can name customers' repos
    return text[:160]


def collect(directory: Path, *, names: bool = False) -> dict[str, Any]:
    from . import __version__, trainfile
    from .traces.readers import scan

    state = directory / ".opentrain"
    out: dict[str, Any] = {"opentrain": __version__, "python": sys.version.split()[0],
                           "os": f"{platform.system()} {platform.machine()}"}
    try:
        tf = trainfile.for_project(directory)
        out["trainer"], out["base_model"], out["budget"] = tf.trainer, tf.base_model, tf.budget.describe()
    except Exception as exc:  # noqa: BLE001 - a broken Trainfile is itself worth reporting
        out["trainfile_error"] = _clean(str(exc))

    out["agents_found"] = {r["label"]: r["count"] for r in scan() if r.get("count")}
    idx = _jsonl(state / "traces" / "index.jsonl")
    if idx:
        coding = [r for r in idx if r.get("edits")]
        moments: Counter = Counter()
        for r in idx:
            moments.update(r.get("moments") or {})
        out["imported"] = {"traces": len(idx), "by_agent": dict(Counter(r["agent"] for r in idx)),
                           "changed_code": len(coding), "repos": len({r["repo"] for r in coding if r.get("repo")}),
                           "training_examples": sum(r.get("examples", 0) for r in idx),
                           "moments": dict(moments)}
    themes_path = state / "traces" / "themes.json"
    if themes_path.exists():
        themes = json.loads(themes_path.read_text())
        out["themes"] = {"named": len(themes), "chosen": sum(1 for t in themes if t.get("chosen"))}
        if names:
            out["themes"]["names"] = [t["name"] for t in themes]
    public = {p.stem.removeprefix("public-").replace("__", "/"): len(_jsonl(p))
              for p in (state / "traces").glob("public-*.jsonl")}
    if public:
        out["public_examples"] = public

    tasks_dir, held_dir = state / "tasks", state / "heldout"
    n_train = sum(1 for p in tasks_dir.glob("*/task.toml")) if tasks_dir.is_dir() else 0
    n_held = sum(1 for p in held_dir.glob("*/task.toml")) if held_dir.is_dir() else 0
    if n_train or n_held:
        out["tasks"] = {"training": n_train, "heldout": n_held}
    reports = sorted((state / "data").glob("mine-*.json")) if (state / "data").is_dir() else []
    if reports:
        try:
            r = json.loads(reports[-1].read_text())
            out["last_mine"] = {k: r.get(k) for k in ("sessions", "accepted", "rejected", "skipped", "reasons")}
        except (OSError, ValueError):
            pass
    manifest = state / "data" / "manifest.json"
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text())
            out["training_set"] = {"examples": m.get("examples"), "from_traces": m.get("from_traces", 0)}
        except (OSError, ValueError):
            pass

    scores = {}
    for p in sorted((held_dir / "scores").glob("*.json")) if (held_dir / "scores").is_dir() else []:
        try:
            s = json.loads(p.read_text())
            scores[s.get("name", p.stem)] = {k: s.get(k) for k in ("tasks", "passed", "pass_rate", "median_seconds")}
        except (OSError, ValueError):
            continue
    if scores:
        out["coding_scores"] = scores
    reg = state / "registry.json"
    if reg.exists():
        try:
            data = json.loads(reg.read_text())
            out["versions"] = [{k: v.get(k) for k in ("version", "base_model", "backend", "decision")}
                               for v in data.get("versions", [])][-10:]
        except (OSError, ValueError):
            pass

    ledger = _jsonl(state / "runs" / "ledger.jsonl")
    if ledger:
        spent: dict[str, float] = {}
        for e in ledger:  # a reservation counts at what it settled for; an open one at its estimate
            if e.get("op") in ("reserve", "settle"):
                spent[e["id"]] = float(e.get("usd") or 0)
            elif e.get("op") == "release":
                spent[e["id"]] = 0.0
        out["spent_usd"] = round(sum(spent.values()), 2)

    db = state / "opentrain.db"
    if db.exists():
        try:
            from .core.store import Store

            jobs = Store(state).find("job")
            out["jobs"] = dict(Counter(f"{j.get('kind')}:{j.get('state')}" for j in jobs))
            failed = [f"{j.get('kind')}: {_clean(j.get('error') or '')}" for j in jobs if j.get("state") == "failed"]
            if failed:
                out["errors"] = failed[-8:]
        except Exception:  # noqa: BLE001 - the report must never fail on a damaged store
            pass
    return out


def render(r: dict[str, Any]) -> str:
    lines = ["## opentrain report", "", "Counts only: no code, prompts, file or repo names.", "", "```json",
             json.dumps(r, indent=1, default=str), "```"]
    return "\n".join(lines)
