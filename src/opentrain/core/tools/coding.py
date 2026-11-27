"""Coding-agent traces as data: find every agent's logs on the machine, import them, see where they
went wrong, and turn them (and the repos they touched) into tasks and training examples.

Imported traces live under ``.opentrain/traces/``: ``index.jsonl`` (one line per trace: agent, repo,
when, how many requests and edits, its moments), ``examples.jsonl`` (one training example per usable
request, in the trained model's tool format, credentials scrubbed) and ``moments.jsonl`` (where a
trace went wrong). ``themes.json`` holds what the agent named from those moments and which ones the
person chose to fix.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..jobs import JobContext
from ..tool import Tool, ToolError, ToolResult, schema
from .train import _jobs, _server, _started


def _dir(ctx) -> Path:
    d = ctx.project.state / "traces"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def _repo_of(cwd: str | None) -> str | None:
    if not cwd:
        return None
    from ...bench import gitstate

    try:
        return gitstate.repo_root(cwd)
    except Exception:  # noqa: BLE001 - a folder that is gone or not a repo
        return None


# ---------------------------------------------------------------------- find and import


def traces_scan(ctx) -> Any:
    """Every coding agent's logs on this machine: where they are and how many (nothing is read yet)."""
    from ...traces.readers import scan

    rows = scan()
    imported = _read_jsonl(_dir(ctx) / "index.jsonl")
    by_agent = Counter(r["agent"] for r in imported)
    out = [{**r, "imported": by_agent.get(r["name"], 0)} for r in rows]
    found = [r for r in out if r.get("count")]
    return {"sources": out, "found": {r["label"]: r["count"] for r in found},
            "note": "Nothing leaves this machine: traces_import reads them here and scrubs credentials."
            if found else "No coding-agent logs found on this machine."}


def import_traces(state: Path | None, sources: list[str] | None = None, since: str | None = None,
                  progress=lambda frac, text: None, check=lambda: None) -> dict[str, Any]:
    """Read every agent's logs on this machine into ``<state>/traces`` (scrubbed); the import summary.
    With ``state=None`` nothing is written: the summary alone (``opentrain traces`` before an import)."""
    from ...traces import EDIT_KINDS
    from ...traces.readers import read_all
    from ...traces.scrub import scrub_trace
    from ...traces.sft import examples, work_root
    from ...traces.signals import moments

    progress(None, "reading every agent's logs on this machine")
    traces, unreadable = read_all(sources, since=since)
    index, exs, skipped, moms = [], [], Counter(), []
    for i, t in enumerate(traces):
        if i % 50 == 0:
            progress(i / max(1, len(traces)), f"{i} of {len(traces)} traces")
        check()
        t = scrub_trace(t)
        repo = work_root(t)
        good, bad = examples(t)
        for ex in good:
            ex["meta"]["repo"] = repo or ""
        exs += good
        skipped.update(b["reason"].split(" (")[0] for b in bad)
        ms = [m.public() for m in moments(t)]
        moms += ms
        index.append({"trace": t.trace_id, "agent": t.agent, "path": t.path, "cwd": t.cwd, "repo": repo,
                      "started_at": t.started_at.isoformat() if t.started_at else None,
                      "requests": len(t.prompts), "edits": sum(1 for u in t.tool_uses() if u.kind in EDIT_KINDS),
                      "files_edited": len(t.edited_files), "examples": len(good),
                      "moments": dict(Counter(m["kind"] for m in ms)), "models": t.models[:3]})
    if state is not None:
        d = state / "traces"
        d.mkdir(parents=True, exist_ok=True)
        _write_jsonl(d / "index.jsonl", index)
        _write_jsonl(d / "examples.jsonl", exs)
        _write_jsonl(d / "moments.jsonl", moms)
    coding = [r for r in index if r["edits"]]
    return {"traces": len(index), "unreadable": len(unreadable),
            "by_agent": dict(Counter(r["agent"] for r in index)),
            "coding_by_agent": dict(Counter(r["agent"] for r in coding)),
            "coding_traces": len(coding), "repos": len({r["repo"] for r in coding if r["repo"]}),
            "examples": len(exs), "skipped_requests": dict(skipped.most_common(8)),
            "moments": dict(Counter(m["kind"] for m in moms))}


def traces_import(ctx, sources: list[str] | None = None, since: str | None = None) -> ToolResult:
    state = ctx.project.state

    def work(job: JobContext) -> Any:
        return import_traces(state, sources, since, progress=job.progress, check=job.check)

    def summary(r: Any) -> str:
        agents = ", ".join(f"{n} {a}" for a, n in r["by_agent"].items())
        return (f"Imported {r['traces']} traces ({agents}); {r['coding_traces']} changed code in {r['repos']} repos. "
                f"{r['examples']} training examples; {sum(r['moments'].values())} moments where something went wrong. "
                "Next: moments (what went wrong), tasks_mine and tasks_from_history (verified tasks).")

    return _started(_jobs(ctx).start_local(ctx.session_id, "import", "import coding-agent traces", work,
                                           summarize=summary), "importing traces")


def traces_list(ctx, agent: str | None = None, repo: str | None = None, coding_only: bool = True,
                limit: int = 50) -> Any:
    rows = _read_jsonl(_dir(ctx) / "index.jsonl")
    if not rows:
        raise ToolError("no traces imported yet; call traces_import")
    rows = [r for r in rows if (not agent or r["agent"] == agent) and (not repo or repo in (r["repo"] or ""))
            and (not coding_only or r["edits"])]
    rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    repos = Counter(r["repo"] for r in rows if r["repo"])
    return {"traces": rows[: max(1, min(int(limit), 200))], "total": len(rows),
            "repos": [{"repo": k, "traces": v} for k, v in repos.most_common(20)]}


def trace_show(ctx, trace: str, request: int | None = None) -> Any:
    """One trace's requests: the prompt, then each tool step in a line (outputs clipped)."""
    from ...traces.readers import SOURCES

    row = next((r for r in _read_jsonl(_dir(ctx) / "index.jsonl") if r["trace"] == trace), None)
    if row is None:
        raise ToolError(f"no imported trace {trace}")
    src = next((s for s in SOURCES if s.name == row["agent"]), None)
    found = src.read(row["path"]) if src else None
    if found is None:
        raise ToolError(f"{row['path']} could not be read again")
    from ...traces.scrub import scrub_trace
    from ...traces.sft import requests

    reqs = requests(scrub_trace(found))
    out = []
    for i, (prompt, steps) in enumerate(reqs):
        if request is not None and i != int(request):
            continue
        lines = []
        for s in steps:
            if s.text.strip():
                lines.append("agent: " + " ".join(s.text.split())[:300])
            for t in s.tools:
                arg = t.args.get("command") or t.args.get("path") or t.args.get("pattern") or ""
                res = " ".join((t.output or "").split())[:160]
                flag = " [error]" if t.is_error or t.exit_code not in (None, 0) else ""
                lines.append(f"{t.kind} {str(arg)[:160]}{flag} → {res}")
        out.append({"request": i, "prompt": prompt[:2000], "steps": lines[:80], "more_steps": max(0, len(lines) - 80)})
    return {"trace": trace, "agent": row["agent"], "repo": row["repo"], "requests": out}


# ---------------------------------------------------------------------- what went wrong


def moments_list(ctx, kind: str | None = None, offset: int = 0, limit: int = 40) -> Any:
    rows = _read_jsonl(_dir(ctx) / "moments.jsonl")
    if not rows and not (_dir(ctx) / "index.jsonl").exists():
        raise ToolError("no traces imported yet; call traces_import")
    rows = [r for r in rows if not kind or r["kind"] == kind]
    rows.sort(key=lambda r: r.get("at") or "", reverse=True)
    page = rows[int(offset): int(offset) + max(1, min(int(limit), 100))]
    return {"moments": page, "total": len(rows), "offset": int(offset),
            "kinds": dict(Counter(r["kind"] for r in _read_jsonl(_dir(ctx) / "moments.jsonl")))}


def themes_set(ctx, themes: list[dict[str, Any]]) -> Any:
    """Save the themes named from the moments; keeps the person's choices for themes already there."""
    path = _dir(ctx) / "themes.json"
    old = {t["name"]: t for t in (json.loads(path.read_text()) if path.exists() else [])}
    out = []
    for t in themes:
        name = str(t.get("name") or "").strip()
        if not name:
            raise ToolError("every theme needs a name")
        ev = [str(e) for e in (t.get("evidence") or [])][:50]
        out.append({"name": name[:80], "description": str(t.get("description") or "")[:600], "evidence": ev,
                    "count": int(t.get("count") or len(ev)), "fix": str(t.get("fix") or "")[:400],
                    "chosen": old.get(name, {}).get("chosen", bool(t.get("chosen", False)))})
    path.write_text(json.dumps(out, indent=1))
    return {"themes": out, "note": "The person picks which to fix (web app, or ask with question)."}


def themes(ctx) -> Any:
    path = _dir(ctx) / "themes.json"
    return {"themes": json.loads(path.read_text()) if path.exists() else []}


def themes_choose(ctx, names: list[str]) -> Any:
    path = _dir(ctx) / "themes.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    unknown = set(names) - {t["name"] for t in rows}
    if unknown:
        raise ToolError(f"no theme named {', '.join(sorted(unknown))}")
    for t in rows:
        t["chosen"] = t["name"] in names
    path.write_text(json.dumps(rows, indent=1))
    return {"chosen": names}


# ---------------------------------------------------------------------- tasks from repo history


def tasks_from_history(ctx, repos: list[str] | None = None, limit: int = 30, author: str | None = None,
                       since: str | None = None, rewrite: bool = True) -> ToolResult:
    srv = _server(ctx)
    state, tf, run = ctx.project.state, srv.tf, srv.run
    if not repos:
        idx = _read_jsonl(_dir(ctx) / "index.jsonl")
        repos = [r for r, _ in Counter(x["repo"] for x in idx if x.get("repo") and x.get("edits")).most_common(8)]
    if not repos:
        raise ToolError("name the repos (repos=[...]) or import traces first so their repos are known")

    def work(job: JobContext) -> Any:
        from ...history import mine_history
        from ...tasks import HeldoutStore, Store, split_store

        r = mine_history(state, list(repos), author=author, since=since, limit=max(1, min(int(limit), 200)),
                         rewrite_with="claude-code/sonnet" if rewrite else None, say=job.log)
        split = split_store(state, tf.heldout_fraction)
        run.add("data", source="history", repos=list(repos), accepted=r["accepted"], rejected=r["rejected"],
                reasons=r["reasons"], split=split)
        return {**{k: v for k, v in r.items() if k != "rows"}, "rows": r["rows"][:60],
                "train": len(Store(state).ids()), "heldout": len(HeldoutStore(state).ids())}

    def summary(r: Any) -> str:
        top = ", ".join(f"{v} {k}" for k, v in list(r["reasons"].items())[:3])
        return (f"{r['accepted']} new verified tasks from {r['tried']} commits (rejected: {top or 'none'}). Store: "
                f"{r['train']} training, {r['heldout']} held out.")

    return _started(_jobs(ctx).start_local(ctx.session_id, "mine", f"tasks from {len(repos)} repos' history", work,
                                           run_id=run.run_id, summarize=summary), "mining repo history")


# ---------------------------------------------------------------------- public coding trajectories


def coding_data_import(ctx, dataset: str, limit: int = 500) -> ToolResult:
    """Job: a public trajectory dataset as training examples in the trained model's tool format."""
    from ...traces.public import KNOWN

    if dataset not in KNOWN:
        raise ToolError(f"{dataset} is not a trajectory dataset OpenTrain converts; known: {', '.join(KNOWN)}")
    d = _dir(ctx)
    spec = KNOWN[dataset]

    def work(job: JobContext) -> Any:
        from ...datasets import read_hf
        from ...traces.public import rows_to_traces
        from ...traces.sft import examples

        job.progress(None, f"downloading {dataset} ({spec['split']})")
        rows, _ = read_hf(dataset, config=spec["config"], split=spec["split"], limit=max(1, min(int(limit), 20000)))
        exs, skipped = [], Counter()
        for t in rows_to_traces(dataset, rows):
            good, bad = examples(t)
            for ex in good:
                ex["meta"]["public"] = dataset
            exs += good
            skipped.update(b["reason"].split(" (")[0] for b in bad)
        slug = dataset.replace("/", "__")
        _write_jsonl(d / f"public-{slug}.jsonl", exs)
        return {"dataset": dataset, "license": spec["license"], "rows": len(rows), "examples": len(exs),
                "skipped": dict(skipped.most_common(6))}

    def summary(r: Any) -> str:
        return (f"{r['examples']} training examples from {r['rows']} trajectories of {r['dataset']} "
                f"(license {r['license']}). data_build includes them.")

    return _started(_jobs(ctx).start_local(ctx.session_id, "import", f"import {dataset}", work, summarize=summary),
                    f"importing {dataset}")


def public_examples(state: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((state / "traces").glob("public-*.jsonl")):
        out += [{k: v for k, v in r.items() if k != "meta"} for r in _read_jsonl(p)]
    return out


# ---------------------------------------------------------------------- training data from traces


def heldout_cutoff(state: Path) -> datetime | None:
    """The oldest held-out task's date: traces at or after it may be the very work the eval checks."""
    from ...tasks import HeldoutStore

    dates = []
    for t in HeldoutStore(state):
        if t.created_at:
            d = datetime.fromisoformat(t.created_at)
            dates.append(d if d.tzinfo else d.replace(tzinfo=timezone.utc))
    return min(dates) if dates else None


def trace_examples(state: Path, *, themes_only: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Imported trace examples safe to train on: none from on or after the held-out cutoff, none from a
    session a held-out task came from. With ``themes_only``, only requests a chosen theme points at."""
    from ...tasks import HeldoutStore

    rows = _read_jsonl(state / "traces" / "examples.jsonl")
    cut = heldout_cutoff(state)
    held_sessions = {t.source_session for t in HeldoutStore(state) if t.source_session}
    keep, dropped = [], Counter()
    chosen: set[str] = set()
    if themes_only:
        tp = state / "traces" / "themes.json"
        for t in (json.loads(tp.read_text()) if tp.exists() else []):
            if t.get("chosen"):
                chosen.update(t.get("evidence") or [])
    for ex in rows:
        m = ex.get("meta") or {}
        at = m.get("at")
        when = datetime.fromisoformat(at) if at else None
        if when and not when.tzinfo:
            when = when.replace(tzinfo=timezone.utc)
        if cut and (when is None or when >= cut):
            dropped["on or after the held-out cutoff"] += 1
        elif m.get("trace") in held_sessions:
            dropped["same session as a held-out task"] += 1
        elif themes_only and f"{m.get('trace')}:{m.get('request')}" not in chosen and m.get("trace") not in chosen:
            dropped["not in a chosen theme"] += 1
        else:
            keep.append({k: v for k, v in ex.items() if k != "meta"})
    return keep, dict(dropped)


PINNED_VLLM = "0.21.0"  # what OpenTrain's Modal and RunPod images serve with


def _ver(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def students(ctx, tier: str | None = None) -> Any:
    """The student models OpenTrain trains for coding, with where each trains and whether that trainer is
    connected here."""
    from ...backends import list_backends
    from ...models import CATALOG

    ready = {n: ok for n, ok, _ in list_backends(ctx.project.state)}
    out = []
    for m in CATALOG:
        if tier and m.tier != tier:
            continue
        ok = [t for t in m.trainers if ready.get(t) and (t == "tinker" or _ver(m.min_vllm) <= _ver(PINNED_VLLM))]
        out.append({**m.public(), "ready_on": ok})
    return {"models": out, "note": "Coding scores are the vendors' own; only the held-out eval says which wins on "
                                   "this user's work. A model with empty ready_on needs a trainer connected."}


def tools() -> list[Tool]:
    S = schema
    strs = {"type": "array", "items": {"type": "string"}}
    T = lambda *a, **k: Tool(*a, group="train", **k)  # noqa: E731
    return [
        T("students", "Student models for coding work (small, mid, large): parameters, where each trains (tinker, modal, "
          "runpod) and which of those are connected here, context, license, vendor coding scores, notes.",
          S({"tier": {"type": "string", "enum": ["small", "mid", "large"]}}), students),
        T("traces_scan", "Find every coding agent's logs on this machine (Claude Code, Codex, Cursor, opencode, Gemini "
          "CLI, Cline, Aider, pi, Copilot): where they are and how many. Free, local, reads nothing yet.", S(),
          traces_scan),
        T("traces_import", "Job: read the coding-agent logs on this machine (all sources, or `sources`), scrub "
          "credentials, and index them: which repos, which requests changed code, training examples in the trained "
          "model's tool format, and moments where something went wrong. Local; nothing is uploaded.",
          S({"sources": strs, "since": {"type": "string", "description": "ISO date"}}), traces_import,
          permission=lambda a: [("traces.read", ",".join(a.get("sources") or []) or "all agents")]),
        T("traces", "Imported traces (newest first) and the repos they worked in.",
          S({"agent": {"type": "string"}, "repo": {"type": "string"}, "coding_only": {"type": "boolean"},
             "limit": {"type": "integer"}}), traces_list),
        T("trace_show", "One imported trace, request by request: the prompt and each tool step in a line.",
          S({"trace": {"type": "string"}, "request": {"type": "integer"}}, ["trace"]), trace_show),
        T("moments", "Where imported traces went wrong (the person corrected or stopped the agent, a check kept "
          "failing, the same file edited over and over, tool errors), newest first, paged. Read them in batches to "
          "name themes (skill data/diagnose).",
          S({"kind": {"type": "string", "enum": ["corrected", "interrupted", "failing_check", "churn", "tool_errors"]},
             "offset": {"type": "integer"}, "limit": {"type": "integer"}}), moments_list),
        T("themes_set", "Save the themes you named from the moments: name (the person's terms), description, count, "
          "evidence (trace ids or trace:request), and the fix (what training data would teach it).",
          S({"themes": {"type": "array", "items": {"type": "object"}}}, ["themes"]), themes_set),
        T("themes", "The saved themes and which ones the person chose to fix.", S(), themes),
        T("themes_choose", "Record which themes the person chose to fix (after asking them).",
          S({"names": strs}, ["names"]), themes_choose),
        T("coding_data_import", "Job: a public coding-agent trajectory dataset from Hugging Face (SWE-bench/SWE-smith-"
          "trajectories, SWE-Gym/OpenHands-SFT-Trajectories, R2E-Gym/R2EGym-SFT-Trajectories) as training examples in "
          "the trained model's tool format. Fills in when the user's own traces are few; data_build includes them.",
          S({"dataset": {"type": "string"}, "limit": {"type": "integer"}}, ["dataset"]), coding_data_import,
          permission=lambda a: [("download", a.get("dataset") or "")]),
        T("tasks_from_history", "Job: verified tasks from the repos' own commits that changed code and tests (the "
          "repo at the parent commit; the commit's tests decide). Each commit is rewritten into the request a person "
          "would have typed, by claude-code/sonnet (free on the user's Claude login) unless rewrite=false. Then the "
          "held-out split is refreshed. Default repos: the ones imported traces worked in.",
          S({"repos": strs, "limit": {"type": "integer"}, "author": {"type": "string"},
             "since": {"type": "string"}, "rewrite": {"type": "boolean"}}), tasks_from_history),
    ]
