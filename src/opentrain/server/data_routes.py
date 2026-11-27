"""Data routes: datasets, evals, the playground and versions, for the web app and any client.

Held-out rows stay sealed here too: a client can count them and read per-row scores by id, never
their content or a model's answers to them.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web

from .app import ApiError, body_of, ok, project_of

routes = web.RouteTableDef()
MAX_UPLOAD = 30 * 1024 * 1024


def _guard(fn, *a, **kw):
    from ..core.complete import CompletionError
    from ..core.tool import ToolError
    from ..datasets import DatasetError
    from ..graders import GraderError
    from ..registry import RegistryError

    try:
        return fn(*a, **kw)
    except (DatasetError, GraderError, CompletionError, RegistryError, ToolError) as exc:
        raise ApiError(400, str(exc)) from None


async def _run(fn, *a, **kw):
    return await asyncio.to_thread(_guard, fn, *a, **kw)


def _datasets(request):
    from ..datasets import Datasets

    return Datasets(project_of(request).state)


def _heldout(request) -> float:
    from .. import trainfile

    return trainfile.for_project(project_of(request).directory).heldout_fraction


@routes.get("/api/dataset")
async def list_datasets(request: web.Request):
    from ..core.tools.data import summarize

    p = project_of(request)
    ds = _datasets(request)
    return ok([{**summarize(p.state, n), "updated": ds.meta(n).get("updated")} for n in ds.names()])


@routes.get("/api/dataset/{name}")
async def get_dataset(request: web.Request):
    from ..core.tools.data import summarize
    from ..evals import list_evals

    p, name = project_of(request), request.match_info["name"]
    ds = _datasets(request)
    meta = await _run(ds.meta, name)
    rows = await _run(ds.rows, name)
    q = request.query.get("q", "").lower().strip()
    if q:
        rows = [r for r in rows if q in r["id"].lower() or any(q in m["content"].lower() for m in r["messages"])]
    offset = max(0, int(request.query.get("offset", 0)))
    limit = max(1, min(int(request.query.get("limit", 50)), 500))
    return ok({**summarize(p.state, name), "grader_spec": meta.get("grader"), "system": meta.get("system") or "",
               "sources": meta.get("sources", []), "total": len(rows), "offset": offset,
               "rows": rows[offset: offset + limit], "evals": list_evals(p.state, name)[:50]})


@routes.post("/api/dataset/{name}/upload")
async def upload(request: web.Request):
    """Body: {filename, content (text), input_key?, output_key?, description?, system?}."""
    from ..datasets import parse_bytes

    b = await body_of(request)
    content = b.get("content") or ""
    if not content or len(content) > MAX_UPLOAD:
        raise ApiError(400, "content is empty or larger than 30 MB")
    raws = await _run(parse_bytes, content.encode(), str(b.get("filename") or "upload.jsonl"))
    r = await _run(_datasets(request).add, request.match_info["name"], raws, description=b.get("description"),
                   source={"upload": b.get("filename") or "upload"}, input_key=b.get("input_key") or None,
                   output_key=b.get("output_key") or None, system=b.get("system") or None,
                   heldout=_heldout(request))
    return ok(r, 201)


@routes.post("/api/dataset/{name}/hf")
async def import_hf(request: web.Request):
    from ..datasets import read_hf

    b = await body_of(request)
    if not b.get("hf"):
        raise ApiError(400, "hf (a dataset id like openai/gsm8k) is required")
    limit = max(1, min(int(b.get("limit") or 2000), 50_000))
    raws, src = await _run(read_hf, b["hf"], split=b.get("split") or None, config=b.get("config") or None, limit=limit)
    r = await _run(_datasets(request).add, request.match_info["name"], raws, description=b.get("description"),
                   source=src, input_key=b.get("input_key") or None, output_key=b.get("output_key") or None,
                   system=b.get("system") or None, heldout=_heldout(request))
    return ok({**r, "columns_seen": sorted(raws[0].keys())[:30] if raws else []}, 201)


@routes.post("/api/dataset/{name}/rows")
async def add_rows(request: web.Request):
    b = await body_of(request)
    rows = b.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ApiError(400, "rows must be a non-empty list")
    return ok(await _run(_datasets(request).add, request.match_info["name"], rows, description=b.get("description"),
                         source={"web": "added by hand"}, heldout=_heldout(request)), 201)


@routes.patch("/api/dataset/{name}")
async def patch_dataset(request: web.Request):
    from ..graders import check

    b = await body_of(request)
    ds, name = _datasets(request), request.match_info["name"]
    if b.get("grader") is not None:
        b["grader"] = {k: v for k, v in b["grader"].items() if k != "source"} if isinstance(b["grader"], dict) \
            else b["grader"]
        await _run(check, b["grader"], project_of(request).directory)
    await _run(ds.update_meta, name, description=b.get("description"), system=b.get("system"), grader=b.get("grader"))
    for rid, answer in (b.get("label") or {}).items():
        await _run(ds.label, name, rid, answer)
    return ok(await _run(ds.meta, name))


@routes.post("/api/dataset/{name}/rows/delete")
async def delete_rows(request: web.Request):
    b = await body_of(request)
    return ok({"removed": await _run(_datasets(request).remove_rows, request.match_info["name"], b.get("ids") or [])})


@routes.delete("/api/dataset/{name}")
async def delete_dataset(request: web.Request):
    await _run(_datasets(request).delete, request.match_info["name"])
    return ok({"deleted": request.match_info["name"]})


@routes.post("/api/dataset/{name}/eval")
async def start_eval(request: web.Request):
    """Start an eval job: {model, split?, limit?}. Free with Claude Code; metered otherwise."""
    from .. import trainfile
    from ..budget import Ledger
    from ..core.providers import default_model
    from ..evals import run

    b = await body_of(request)
    p, name = project_of(request), request.match_info["name"]
    model = (b.get("model") or "").strip()
    if not model:
        raise ApiError(400, "model is required (provider/model, or a version like v3)")
    split = b.get("split") or "heldout"
    if split not in ("heldout", "train"):
        raise ApiError(400, "split is heldout or train")
    await _run(_datasets(request).meta, name)
    tf = trainfile.for_project(p.directory)
    ledger = Ledger(p.state, tf.budget)
    left = ledger.remaining()

    def check(spent: float) -> None:
        if spent > left:
            raise RuntimeError(f"stopped: the eval would exceed the budget (${left:.2f} left)")

    def work(job):
        job.progress(0.0, f"running {model} on {split} rows")
        rec = run(p.state, p.directory, name, model, split=split, limit=b.get("limit"), judge_model=default_model(),
                  budget_check=check, progress=lambda d, n: job.progress(d / n, f"{d}/{n} rows"),
                  cancelled=lambda: job.cancel_event.is_set())
        cost = rec["summary"]["cost_usd"]
        if cost > 0:
            try:
                ledger.settle(ledger.reserve(cost, f"eval {model} on {name}"), cost)
            except Exception:  # noqa: BLE001 - already spent; the ledger refuses only new reservations
                pass
        return {"eval": rec["id"], "model": model, "dataset": name, "split": split, **rec["summary"]}

    job = p.services["jobs"].start_local(b.get("session_id"), "eval", f"eval {model} on {name}", work)
    job = p.store.update("job", job["id"], {"dataset": name, "target": model, "split": split}) or job
    return ok(job, 202)


@routes.get("/api/eval")
async def list_all_evals(request: web.Request):
    from ..evals import list_evals

    return ok(list_evals(project_of(request).state, request.query.get("dataset") or None))


@routes.get("/api/eval/{eid}")
async def get_eval(request: web.Request):
    from ..datasets import Datasets, prompt_text, reference_of
    from ..evals import answers, get

    p = project_of(request)
    rec = await _run(get, p.state, request.match_info["eid"])
    if rec["split"] == "train":  # training rows and answers to them may be shown
        ans = {a["id"]: a for a in await _run(answers, p.state, rec["id"])}
        rows = {r["id"]: r for r in Datasets(p.state).rows(rec["dataset"])}
        for r in rec["rows"]:
            src = rows.get(r["id"])
            r["prompt"] = prompt_text(src)[:4000] if src else None
            r["reference"] = (reference_of(src) or "")[:4000] if src else None
            r["answer"] = (ans.get(r["id"], {}).get("answer") or "")[:4000]
            r["reason"] = ans.get(r["id"], {}).get("reason")
    return ok(rec)


@routes.get("/api/gate")
async def gate_preview(request: web.Request):
    from .. import trainfile
    from ..evals import decide

    p = project_of(request)
    q = request.query
    if not (q.get("dataset") and q.get("candidate") and q.get("baseline")):
        raise ApiError(400, "dataset, candidate and baseline are required")
    tf = trainfile.for_project(p.directory)
    return ok(await _run(decide, p.state, q["dataset"], q["candidate"], q["baseline"],
                         allow_regressions=tf.allow_regressions))


@routes.post("/api/try")
async def try_model(request: web.Request):
    """One prompt, one answer: {model, prompt, system?}."""
    from ..core.complete import complete

    b = await body_of(request)
    if not b.get("model") or not b.get("prompt"):
        raise ApiError(400, "model and prompt are required")
    msgs: list[dict[str, Any]] = ([{"role": "system", "content": b["system"]}] if b.get("system") else [])
    msgs.append({"role": "user", "content": b["prompt"]})
    c = await _run(complete, b["model"], msgs, state=project_of(request).state)
    return ok({"model": b["model"], "answer": c.text, "seconds": round(c.seconds, 2), "cost_usd": round(c.cost, 5),
               "tokens": c.output_tokens})


@routes.get("/api/version")
async def versions(request: web.Request):
    from ..registry import Registry
    from ..serving import live_version

    p = project_of(request)
    reg = Registry(p.state)
    cur = reg.current()
    return ok({"current": cur.version if cur else None, "pinned": live_version(p.state),
               "versions": [v.__dict__ for v in reversed(reg.versions())]})


@routes.post("/api/version/{v}/promote")
async def promote(request: web.Request):
    """Run the dataset gate for a version: {baseline}. Only a passing version goes live."""
    from .. import trainfile
    from ..evals import decide
    from ..registry import Registry

    b = await body_of(request)
    p, version = project_of(request), request.match_info["v"]
    reg = Registry(p.state)
    v = await _run(reg.get, version)
    if not v.dataset:
        raise ApiError(400, f"{version} was trained on coding tasks; promote it from a session (version_propose)")
    cur = reg.current()
    base = b.get("baseline") or (cur.version if cur and cur.version != version else None)
    if not base:
        raise ApiError(400, "name a baseline model you evaluated on the same dataset")
    tf = trainfile.for_project(p.directory)
    d = await _run(decide, p.state, v.dataset, version, base, allow_regressions=tf.allow_regressions)
    reg.record_gate(version, "promoted" if d["promote"] else "rejected: " + "; ".join(d["reasons"]), d)
    if d["promote"]:
        reg.promote(version, f"beat {base} on {v.dataset} (web)")
    return ok(d)


# ---------------------------------------------------------------------- the board


@routes.get("/api/board")
async def board_view(request: web.Request):
    from ..board import board

    p = project_of(request)
    jobs = p.services["jobs"].list() if "jobs" in p.services else []
    props = p.services["proposals"].pending() if "proposals" in p.services else []
    return ok(await asyncio.to_thread(board, p.state, p.directory, dataset=request.query.get("dataset"), jobs=jobs,
                                      proposals=props))


@routes.get("/api/board/grid")
async def board_grid(request: web.Request):
    from ..board import grid
    from ..datasets import DatasetError

    p = project_of(request)
    name = request.query.get("dataset")
    if not name:
        raise ApiError(400, "dataset is required")
    try:
        return ok(await asyncio.to_thread(grid, p.state, name, reveal=request.query.get("reveal") == "1"))
    except DatasetError as exc:
        raise ApiError(404, str(exc)) from None


# ------------------------------------------------------------------ coding-agent traces


def _trace_ctx(request, session_id: str | None = None):
    from types import SimpleNamespace

    return SimpleNamespace(project=project_of(request), session_id=session_id)


@routes.get("/api/traces")
async def traces_view(request: web.Request):
    """What the web app's traces panel shows: every agent's logs found here, what was imported, themes."""
    from ..core.tools import coding

    ctx = _trace_ctx(request)
    scan = await _run(coding.traces_scan, ctx)
    d = ctx.project.state / "traces"
    index = coding._read_jsonl(d / "index.jsonl")
    coding_rows = [r for r in index if r.get("edits")]
    repos: dict[str, int] = {}
    for r in coding_rows:
        if r.get("repo"):
            repos[r["repo"]] = repos.get(r["repo"], 0) + 1
    moments: dict[str, int] = {}
    for r in index:
        for k, v in (r.get("moments") or {}).items():
            moments[k] = moments.get(k, 0) + v
    return ok({"sources": scan["sources"], "imported": len(index), "coding": len(coding_rows),
               "examples": sum(r.get("examples", 0) for r in index),
               "repos": sorted(({"repo": k, "traces": v} for k, v in repos.items()), key=lambda x: -x["traces"])[:12],
               "moments": moments, "themes": coding.themes(ctx)["themes"]})


@routes.post("/api/traces/import")
async def traces_import_route(request: web.Request):
    from ..core.tools import coding

    b = await body_of(request)
    res = await _run(coding.traces_import, _trace_ctx(request, b.get("session_id")), b.get("sources"), b.get("since"))
    return ok({"job_id": res.metadata.get("job_id")})


@routes.post("/api/traces/themes")
async def traces_themes_route(request: web.Request):
    from ..core.tools import coding

    b = await body_of(request)
    return ok(await _run(coding.themes_choose, _trace_ctx(request), [str(n) for n in b.get("chosen") or []]))
