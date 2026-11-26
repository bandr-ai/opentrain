"""Dataset tools: bring any data, say how answers are checked, evaluate any model, train, gate.

This is the general path. The coding-session pipeline (``sessions_scan``, ``tasks_mine``, ...) is
one specialised source next to it, used when the user asks for it.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from ..jobs import JobContext
from ..tool import Tool, ToolContext, ToolError, ToolResult, schema


def _ds(ctx: ToolContext):
    from ...datasets import Datasets

    return Datasets(ctx.project.state)


def _err(fn):
    from ...datasets import DatasetError
    from ...graders import GraderError

    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except (DatasetError, GraderError) as exc:
            raise ToolError(str(exc)) from None
    return wrapped


def _tf(ctx: ToolContext):
    from ... import trainfile

    return trainfile.for_project(ctx.project.directory)


def _engine_model(ctx: ToolContext) -> str:
    s = ctx.project.session(ctx.session_id)
    from ..providers import default_model

    return s.get("model") or default_model()


def summarize(state: Path, name: str) -> dict[str, Any]:
    from ...datasets import Datasets
    from ...evals import list_evals
    from ...graders import describe

    m = Datasets(state).meta(name)
    evs = [e for e in list_evals(state, name) if e["split"] == "heldout"]
    best: dict[str, Any] = {}
    for e in evs:  # newest first: keep the latest per target
        sm = e["summary"]
        best.setdefault(e["target"], {"eval": e["id"], "score": sm["mean_score"], "pass_rate": sm["pass_rate"],
                                      "rows": sm["rows"], "median_seconds": sm.get("median_seconds"),
                                      "cost_per_1k_usd": round((sm.get("cost_usd") or 0.0) / max(1, sm["rows"]) * 1000,
                                                               4)})
    return {"name": name, "description": m.get("description"), "counts": m.get("counts"),
            "grader": describe(m.get("grader") or {"type": "judge"}), "system": (m.get("system") or "")[:300],
            "sources": [{k: v for k, v in s.items() if k != "time"} for s in m.get("sources", [])][-5:],
            "latest_scores": best}


# ---------------------------------------------------------------------- reading


def datasets_list(ctx: ToolContext) -> Any:
    ds = _ds(ctx)
    return [summarize(ctx.project.state, n) for n in ds.names()] or "no datasets yet: dataset_import or dataset_add"


def dataset_show(ctx: ToolContext, name: str, offset: int = 0, limit: int = 10) -> Any:
    ds = _ds(ctx)
    rows = _err(ds.rows)(name)
    limit = max(1, min(int(limit), 50))
    page = rows[int(offset): int(offset) + limit]
    for r in page:
        for m in r["messages"]:
            if len(m["content"]) > 3000:
                m["content"] = m["content"][:3000] + "…"
    return {**summarize(ctx.project.state, name), "showing": f"{offset}-{offset + len(page)} of {len(rows)} training rows",
            "rows": page, "note": "held-out rows are sealed; only evals read them"}


# ---------------------------------------------------------------------- building


async def dataset_import(ctx: ToolContext, name: str, path: str | None = None, hf: str | None = None,
                         split: str | None = None, config: str | None = None, limit: int | None = None,
                         input_key: str | None = None, output_key: str | None = None, description: str | None = None,
                         system: str | None = None, kaggle: str | None = None, file: str | None = None,
                         input_template: str | None = None, output_template: str | None = None,
                         heldout: float | None = None) -> Any:
    from ...datasets import DatasetError, apply_template, read_file, read_hf, read_kaggle

    if sum(map(bool, (path, hf, kaggle))) != 1:
        raise ToolError("give exactly one of path (a JSONL/JSON/CSV file), hf (a Hugging Face dataset id) or kaggle "
                        "(a Kaggle dataset id, owner/name)")
    tf = _tf(ctx)
    remote_limit = max(1, min(int(limit or 2000), 50_000))
    if heldout is not None and not 0 <= float(heldout) <= 1:
        raise ToolError("heldout is a fraction from 0 to 1 (1: every row is for evaluation only)")
    if path:
        p = Path(path).expanduser()
        p = p if p.is_absolute() else ctx.project.directory / p
        from .paths import guard

        guard(ctx.project, p)
        raws = await asyncio.to_thread(_err(read_file), p)
        source = {"file": str(p)}
    elif kaggle:
        await ctx.ask("download", [f"kaggle.com/datasets/{kaggle}"], metadata={"dataset": kaggle})
        raws, source = await asyncio.to_thread(_err(read_kaggle), kaggle, file=file, limit=remote_limit)
    else:
        await ctx.ask("download", [f"huggingface.co/datasets/{hf}"], metadata={"dataset": hf})
        raws, source = await asyncio.to_thread(_err(read_hf), hf, split=split, config=config, limit=remote_limit)
    total = len(raws)
    if limit and len(raws) > int(limit):
        raws = raws[: int(limit)]
    if input_template or output_template:
        try:
            raws = [apply_template(r, input_template, output_template) for r in raws]
        except DatasetError as exc:
            raise ToolError(str(exc)) from None
        input_key = "prompt" if input_template else input_key
        output_key = "completion" if output_template else output_key
    r = await asyncio.to_thread(_err(_ds(ctx).add), name, raws, description=description, source=source,
                                input_key=input_key, output_key=output_key, system=system,
                                heldout=tf.heldout_fraction if heldout is None else float(heldout))
    sample_cols = sorted(raws[0].keys())[:20] if raws and isinstance(raws[0], dict) else []
    out = {**r, "columns_seen": sample_cols}
    if (hf or kaggle) and total >= remote_limit:
        out["note"] = f"took the first {remote_limit} rows of the source; pass limit to take more (up to 50,000)"
    elif path and limit and total > int(limit):
        out["note"] = f"took {limit} of the file's {total} rows"
    grader = _ds(ctx).meta(name).get("grader") or {}
    if grader.get("type", "judge") == "judge" and len(grader) <= 1:  # still the default judge
        hint = suggest_grader(_ds(ctx).rows(name))
        if hint:
            out["suggested_grader"] = hint
    out["next"] = ("dataset_show to check the rows, dataset_edit with a grader to say how answers are checked "
                   "(see suggested_grader), then dataset_eval for a baseline")
    return out


def suggest_grader(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A check that fits the references, when one plainly does: final numbers (GSM8K's ``#### 72``)."""
    from ...datasets import reference_of
    from ...graders import _last_number

    refs = [x for x in (reference_of(r) for r in rows[:200]) if x]
    if len(refs) >= 5 and all(re.fullmatch(r"\s*\(?[A-Ja-j][).:]?\s*", x) for x in refs):
        return {"type": "regex", "pattern": r"^\W*([A-J])\b",
                "why": "every reference is an option letter: the first letter of the answer must match it (ask "
                       "for the letter only in the system prompt)"}
    if len(refs) >= 5 and sum(_last_number(x) is not None for x in refs) >= 0.95 * len(refs):
        tail = sum(bool(re.search(r"(####|\\boxed\{|answer is)\s*-?[\d$]", x, re.I)) for x in refs)
        if tail >= 0.8 * len(refs) or max(map(len, refs)) <= 20:
            return {"type": "number", "why": "every reference ends in a final number"}
    return None


def dataset_add(ctx: ToolContext, name: str, rows: list[dict[str, Any]], description: str | None = None,
                system: str | None = None, heldout: float | None = None) -> Any:
    if not rows:
        raise ToolError("rows is empty")
    if heldout is not None and not 0 <= float(heldout) <= 1:
        raise ToolError("heldout is a fraction from 0 to 1 (1: evaluation rows; 0: training rows)")
    return _err(_ds(ctx).add)(name, rows, description=description, source={"agent": ctx.agent.name},
                              system=system, heldout=_tf(ctx).heldout_fraction if heldout is None else float(heldout))


def dataset_edit(ctx: ToolContext, name: str, description: str | None = None, system: str | None = None,
                 grader: dict[str, Any] | None = None, remove_ids: list[str] | None = None,
                 label: dict[str, str] | None = None) -> Any:
    from ...graders import check, describe

    ds = _ds(ctx)
    out: dict[str, Any] = {}
    if grader is not None:
        _err(check)(grader)
        _err(ds.update_meta)(name, grader=grader)
        out["grader"] = describe(grader)
    if description is not None or system is not None:
        _err(ds.update_meta)(name, description=description, system=system)
    if remove_ids:
        out["removed"] = _err(ds.remove_rows)(name, remove_ids)
    for rid, answer in (label or {}).items():
        _err(ds.label)(name, rid, answer)
    out["counts"] = ds.meta(name)["counts"]
    return out


def dataset_label(ctx: ToolContext, name: str, teacher: str | None = None, only_missing: bool = True,
                  limit: int = 500) -> ToolResult:
    """Fill reference answers on training rows with a teacher model (distillation)."""
    from ...core.complete import CompletionError, complete
    from ...datasets import prompt_of, reference_of, with_system

    ds = _ds(ctx)
    rows = _err(ds.rows)(name)
    todo = [r for r in rows if not only_missing or reference_of(r) is None][: max(1, int(limit))]
    if not todo:
        raise ToolError(f"every training row in {name} already has a reference answer")
    model = teacher or _engine_model(ctx)
    state = ctx.project.state

    def work(job: JobContext) -> Any:
        done, failed, cost = 0, 0, 0.0
        for r in todo:
            job.check()
            try:
                c = complete(model, prompt_of(with_system(r, ds.meta(name).get("system"))), state=state)
                ds.label(name, r["id"], c.text)
                cost += c.cost
                done += 1
            except CompletionError as exc:
                failed += 1
                job.log(f"{r['id']}: {exc}")
            job.progress((done + failed) / len(todo), f"{done} labelled, {failed} failed")
        return {"labelled": done, "failed": failed, "teacher": model, "cost_usd": round(cost, 4)}

    job = ctx.project.services["jobs"].start_local(
        ctx.session_id, "label", f"label {name} with {model}", work,
        summarize=lambda r: f"{r['labelled']} rows labelled by {r['teacher']}, {r['failed']} failed, "
                            f"${r['cost_usd']:.2f}.")
    return ToolResult(f"Started job {job['id']} (labelling {len(todo)} rows of {name} with {model}). You will get "
                      "a system reminder when it ends.", title=f"label {name}", metadata={"job_id": job["id"]})


# ---------------------------------------------------------------------- evals


def _budget_guard(ctx: ToolContext):
    from ...budget import Ledger

    tf = _tf(ctx)
    left = Ledger(ctx.project.state, tf.budget).remaining()

    def check(spent: float) -> None:
        if spent > left:
            raise ToolError(f"the eval stopped: it would exceed the budget (${left:.2f} left)")
    return check, left


def _charge(ctx: ToolContext, usd: float, what: str) -> None:
    from ...budget import BudgetExceeded, Ledger

    if usd <= 0:
        return
    ledger = Ledger(ctx.project.state, _tf(ctx).budget)
    try:
        rid = ledger.reserve(usd, what)
    except BudgetExceeded:
        rid = None
    if rid:
        ledger.settle(rid, usd)


def dataset_eval(ctx: ToolContext, name: str, model: str, limit: int | None = None,
                 split: str = "heldout", ids: list[str] | None = None) -> ToolResult:
    from ...evals import run

    if split not in ("heldout", "train"):
        raise ToolError("split is heldout or train")
    if ids and split != "train":
        raise ToolError("ids pick training rows (split=train); held-out rows are not yours to pick")
    _err(_ds(ctx).meta)(name)
    check, _ = _budget_guard(ctx)
    state, directory, judge = ctx.project.state, ctx.project.directory, _engine_model(ctx)

    def work(job: JobContext) -> Any:
        job.progress(0.0, f"running {model} on {split} rows of {name}")
        rec = run(state, directory, name, model, split=split, limit=limit, ids=ids, judge_model=judge,
                  budget_check=check,
                  progress=lambda d, n: job.progress(d / n, f"{d}/{n} rows"),
                  cancelled=lambda: job.cancel_event.is_set())
        _charge(ctx, rec["summary"]["cost_usd"], f"eval {model} on {name}")
        return {"eval": rec["id"], "model": model, "dataset": name, "split": split, **rec["summary"],
                "failed_ids": [r["id"] for r in rec["rows"] if not r["passed"]][:40]}

    def summary(r: Any) -> str:
        return (f"{r['model']} on {r['dataset']} ({r['split']}): mean score {r['mean_score']:.3f}, "
                f"{r['passed']}/{r['rows']} passed, {r['errors']} errors, ${r['cost_usd']:.2f}. Eval id {r['eval']}.")

    job = ctx.project.services["jobs"].start_local(ctx.session_id, "eval", f"eval {model} on {name}", work,
                                                   summarize=summary)
    ctx.project.store.update("job", job["id"], {"dataset": name, "target": model, "split": split})
    return ToolResult(f"Started job {job['id']} (evaluating {model} on {name}, {split}). You will get a system "
                      "reminder with the score when it ends. Do not poll.", title=f"eval {model}",
                      metadata={"job_id": job["id"]})


def evals_list(ctx: ToolContext, dataset: str | None = None, eval_id: str | None = None) -> Any:
    from ...evals import get, list_evals

    if eval_id:
        from ...evals import answers

        rec = _err(get)(ctx.project.state, eval_id)
        out = {k: rec[k] for k in ("id", "dataset", "target", "split", "summary", "rows")}
        if rec["split"] == "train":  # training rows are yours to read: show each answer next to its score
            said = {a["id"]: a.get("answer") for a in answers(ctx.project.state, eval_id)}
            out["rows"] = [{**r, "answer": (said.get(r["id"]) or "")[:1500]} for r in rec["rows"]]
        return out
    return list_evals(ctx.project.state, dataset)[:30]


def dataset_grade(ctx: ToolContext, name: str, answer: str, row_id: str | None = None, prompt: str | None = None,
                  reference: str | None = None) -> Any:
    """Grade an answer you give with the dataset's grader: check a grader or reward on controls (the
    reference, an empty answer, a stuffed one) before anything spends money."""
    from ...datasets import prompt_text, reference_of
    from ...graders import GraderError, grade

    ds = _ds(ctx)
    spec = _err(ds.meta)(name).get("grader") or {"type": "judge"}
    if row_id:
        row = next((r for r in ds.rows(name) if r["id"] == row_id), None)
        if row is None:
            raise ToolError(f"no training row {row_id} in {name} (held-out rows cannot be graded here)")
        prompt, reference = prompt_text(row), reference if reference is not None else reference_of(row)
    try:
        g = grade(spec, prompt or "", answer, reference, directory=ctx.project.directory,
                  default_model=_engine_model(ctx))
    except GraderError as exc:
        raise ToolError(str(exc)) from None
    _charge(ctx, g.cost, f"grade on {name}")
    return {"score": g.score, "passed": g.passed, "reason": g.reason, "grader": spec.get("type")}


def model_try(ctx: ToolContext, model: str, prompt: str, system: str | None = None) -> Any:
    from ...core.complete import CompletionError, complete

    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        c = complete(model, msgs, state=ctx.project.state)
    except CompletionError as exc:
        raise ToolError(str(exc)) from None
    _charge(ctx, c.cost, f"try {model}")
    return {"model": model, "answer": c.text, "seconds": round(c.seconds, 2), "cost_usd": round(c.cost, 5)}


# ---------------------------------------------------------------------- training and the gate


def build_training_set(ctx: ToolContext, name: str) -> dict[str, Any]:
    from ...evals import contamination

    state = ctx.project.state
    out = state / "data" / "train.jsonl"
    manifest = _err(_ds(ctx).training_file)(name, out)
    leaks = contamination(state, name, out)
    if leaks:
        out.unlink(missing_ok=True)
        raise ToolError(f"{len(leaks)} held-out rows overlap the training rows (ids: "
                        f"{', '.join(x['id'] for x in leaks[:10])}); remove the duplicates with dataset_edit")
    (state / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


async def dataset_train(ctx: ToolContext, name: str, epochs: int | None = None, lora_rank: int | None = None,
                        learning_rate: float | None = None, method: str | None = None,
                        settings: dict[str, Any] | None = None) -> ToolResult:
    from .train import job_start

    manifest = await asyncio.to_thread(build_training_set, ctx, name)
    r = recipe_for(ctx, name, method)
    epochs = epochs or r["epochs"]
    lora_rank = lora_rank or r["lora_rank"]
    learning_rate = learning_rate or r["learning_rate"]
    extra = {**r["extra"], **(settings or {})}
    res = await job_start(ctx, epochs, lora_rank, learning_rate, r["method"], extra, r["batch_size"])
    jid = res.metadata.get("job_id")
    if jid:
        ctx.project.store.update("job", jid, {"dataset": name, "title": f"{r['method']} on {name}",
                                              "method": r["method"],
                                              "recipe": {**r, "epochs": epochs, "lora_rank": lora_rank,
                                                         "learning_rate": learning_rate, "extra": extra}})
    res.output = (f"{manifest['examples']} examples from {name}; recipe: {r['why']} (used: rank {lora_rank}, "
                  f"lr {learning_rate:g}, {epochs} epochs). " + res.output)
    return res


def recipe_for(ctx: ToolContext, name: str, method: str | None = None) -> dict[str, Any]:
    from ...datasets import reference_of
    from ...recipes import recipe, task_kind

    ds = _ds(ctx)
    rows = _err(ds.rows)(name)
    refs = [reference_of(r) for r in rows]
    grader = ds.meta(name).get("grader")
    if (grader or {}).get("type") == "python":  # the reward's code goes to the GPU with the job
        from ...graders import GraderError, reward_source

        try:
            grader = {**grader, "source": reward_source(grader, ctx.project.directory)}
        except GraderError as exc:
            raise ToolError(str(exc)) from None
    kind = task_kind(grader, [x for x in refs if x])
    try:
        return recipe(kind, len(rows), _tf(ctx).base_model, labelled=sum(1 for x in refs if x), method=method,
                      grader=grader)
    except ValueError as exc:
        raise ToolError(str(exc)) from None


async def dataset_search(ctx: ToolContext, query: str, source: str = "both", limit: int = 8) -> Any:
    from ...datasets import search_hf, search_kaggle

    limit = max(1, min(int(limit), 20))
    out: dict[str, Any] = {"query": query}
    for name, fn in (("hf", search_hf), ("kaggle", search_kaggle)):
        if source in (name, "both"):
            try:
                out[name] = await asyncio.to_thread(fn, query, limit)
            except Exception as exc:  # noqa: BLE001 - one source down must not hide the other
                out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    out["next"] = "dataset_peek the promising ones (hf), then dataset_import a small slice (limit 200) to look at"
    return out


async def dataset_peek(ctx: ToolContext, hf: str) -> Any:
    from ...datasets import peek_hf

    return await asyncio.to_thread(_err(peek_hf), hf)


def dataset_recipe(ctx: ToolContext, name: str, method: str | None = None, epochs: int | None = None,
                   lora_rank: int | None = None, learning_rate: float | None = None,
                   settings: dict[str, Any] | None = None) -> Any:
    """The recipe, with any overrides you plan to pass to dataset_train applied, priced as it would run."""
    r = recipe_for(ctx, name, method)
    over = {k: v for k, v in (("epochs", epochs), ("lora_rank", lora_rank), ("learning_rate", learning_rate))
            if v is not None}
    if over or settings:
        r = {**r, **over, "extra": {**r["extra"], **(settings or {})}, "overridden": sorted(over) + sorted(settings or {})}
    return {**r, "cost": recipe_cost(ctx, name, r)}


def recipe_cost(ctx: ToolContext, name: str, r: dict[str, Any]) -> dict[str, Any]:
    """What training with this recipe would cost on the Trainfile's trainer (nothing is uploaded)."""
    import tempfile

    from ...backends import get_backend
    from ...plugs import TrainSpec

    tf = _tf(ctx)
    out: dict[str, Any] = {"trainer": tf.trainer, "base_model": tf.base_model}
    backend = get_backend(tf.trainer, ctx.project.state)
    ok, why = backend.available()
    if not ok:
        return {**out, "available": False, "detail": why}
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "train.jsonl"
        try:
            _err(_ds(ctx).training_file)(name, data)
            spec = TrainSpec(base_model=tf.base_model, data=data, method=r["method"], lora_rank=r["lora_rank"],
                             epochs=r["epochs"], learning_rate=r["learning_rate"], batch_size=r["batch_size"],
                             name="estimate", extra=r["extra"])
            out["usd"] = round(backend.estimate(spec), 2)
            if tf.trainer in ("modal", "runpod"):
                from ...backends.modal_backend import _rows, gpu_seconds, loop_gpu

                out["gpu"] = loop_gpu(spec.base_model, spec.method, spec.extra.get("teacher"))[0]
                out["minutes"] = round(gpu_seconds(spec, _rows(data)) / 60 + (5 if tf.trainer == "runpod" else 0))
        except (ToolError, ValueError, RuntimeError) as exc:
            return {**out, "error": str(exc)}
    return out


def dataset_promote(ctx: ToolContext, version: str, baseline: str | None = None) -> Any:
    from ...evals import decide
    from ...registry import Registry, RegistryError

    reg = Registry(ctx.project.state)
    try:
        v = reg.get(version)
    except RegistryError as exc:
        raise ToolError(str(exc)) from None
    if not v.dataset:
        raise ToolError(f"{version} was not trained on a dataset; use version_propose")
    cur = reg.current()
    base = baseline or (cur.version if cur and cur.version != version else None)
    if not base:
        raise ToolError("nothing is live to compare with: name a baseline model you evaluated on the same dataset "
                        "(for example the base model's endpoint or the engine model)")
    tf = _tf(ctx)
    d = decide(ctx.project.state, v.dataset, version, base, allow_regressions=tf.allow_regressions)
    reg.record_gate(version, "promoted" if d["promote"] else "rejected: " + "; ".join(d["reasons"]), d)
    if d["promote"]:
        reg.promote(version, f"beat {base} on {v.dataset}: {d['candidate_score']} vs {d['baseline_score']}")
    return d


def tools() -> list[Tool]:
    S = schema
    T = lambda *a, **k: Tool(*a, group="data", **k)  # noqa: E731
    obj = {"type": "object"}
    return [
        T("dataset_search", "Find public datasets for a use case on Hugging Face and Kaggle: id, downloads, size, "
          "license, tasks. Search several phrasings (the task, the domain, benchmark names).",
          S({"query": {"type": "string"}, "source": {"type": "string", "enum": ["both", "hf", "kaggle"]},
             "limit": {"type": "integer"}}, ["query"]), dataset_search, title=lambda a: a["query"]),
        T("dataset_peek", "A Hugging Face dataset before importing it: configs, splits with row counts, columns "
          "and three example rows.", S({"hf": {"type": "string"}}, ["hf"]), dataset_peek, title=lambda a: a["hf"]),
        T("dataset_recipe", "The tested training recipe for a dataset, with its cost on the Trainfile's trainer: task kind, the plan (which methods suit it, in "
          "order: sft, opd = on-policy distillation, rl = GRPO, rejection = rejection sampling), the settings for "
          "one method (default: the plan's first), what to do first and what to try if the run does not beat the "
          "baseline. dataset_train applies it; read it before proposing a training run.",
          S({"name": {"type": "string"}, "method": {"type": "string", "enum": ["sft", "rejection", "opd", "rl"]},
             "epochs": {"type": "integer"}, "lora_rank": {"type": "integer"}, "learning_rate": {"type": "number"},
             "settings": {"type": "object", "description": "the same overrides you will pass to dataset_train, "
                          "so the price matches the run"}},
            ["name"]), dataset_recipe),
        T("datasets", "Every dataset: rows (training, held out, labelled), how answers are graded, latest scores per "
          "model.", S(), datasets_list),
        T("dataset_show", "Training rows of a dataset (held-out rows are sealed).",
          S({"name": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["name"]),
          dataset_show),
        T("dataset_import", "Create or extend a dataset from a file (JSONL, JSON, CSV: chat 'messages', or "
          "prompt/completion-like columns), a public Hugging Face dataset, or a Kaggle dataset. Rows are split "
          "train/held-out by the Trainfile.",
          S({"name": {"type": "string", "description": "lowercase, e.g. support-replies"},
             "path": {"type": "string"}, "hf": {"type": "string", "description": "e.g. openai/gsm8k"},
             "kaggle": {"type": "string", "description": "owner/name, e.g. uciml/sms-spam-collection-dataset"},
             "file": {"type": "string", "description": "Kaggle: which file in the dataset (default: the largest "
                                                       "CSV/JSON/JSONL)"},
             "split": {"type": "string"}, "config": {"type": "string"}, "limit": {"type": "integer"},
             "input_key": {"type": "string"}, "output_key": {"type": "string"},
             "input_template": {"type": "string", "description": "build the prompt from columns, e.g. "
                                "\"{question}\\n\\n{options_lettered}\" (a list/dict column as A. B. C. lines)"},
             "output_template": {"type": "string", "description": "build the answer, e.g. \"{answer_idx}\" or "
                                 "\"{cop_letter}\" (an index column as a letter)"},
             "heldout": {"type": "number", "description": "share of new rows sealed for evaluation (default: the "
                         "Trainfile's); 1 makes an eval-only dataset, e.g. a benchmark's test split"},
             "description": {"type": "string"}, "system": {"type": "string"}}, ["name"]),
          dataset_import, permission=lambda a: [("import", a["path"])] if a.get("path") else []),
        T("dataset_add", "Add rows you wrote or generated: objects with 'messages' (chat) or prompt and completion. "
          "Use this to build a dataset from the user's description, from documents, or from examples they paste.",
          S({"name": {"type": "string"}, "rows": {"type": "array", "items": obj},
             "description": {"type": "string"}, "system": {"type": "string"},
             "heldout": {"type": "number", "description": "share sealed for evaluation (default: the Trainfile's); "
                         "1 for evaluation rows you wrote first, 0 for training rows written after them"}},
            ["name", "rows"]), dataset_add),
        T("dataset_edit", "Change a dataset: description, system prompt, grader ({type: exact|contains|regex|json|"
          "number|judge|command|python, rubric, pattern, model, command, path, threshold}; python: a .py file "
          "with reward(prompt, answer, reference) -> 0..1, which also runs inside RL), remove rows by id, or set reference "
          "answers ({row_id: answer}).",
          S({"name": {"type": "string"}, "description": {"type": "string"}, "system": {"type": "string"},
             "grader": obj, "remove_ids": {"type": "array", "items": {"type": "string"}},
             "label": {"type": "object", "additionalProperties": {"type": "string"}}}, ["name"]),
          dataset_edit, permission=lambda a: []),
        T("dataset_label", "Job: write reference answers for training rows with a teacher model (distillation). "
          "Costs model tokens.",
          S({"name": {"type": "string"}, "teacher": {"type": "string", "description": "provider/model"},
             "only_missing": {"type": "boolean"}, "limit": {"type": "integer"}}, ["name"]),
          dataset_label, permission=lambda a: [] if str(a.get("teacher") or "").startswith("claude-code/")
          else [("spend", f"labelling {a['name']}")]),
        T("dataset_eval", "Job: run a model (provider/model, or a trained version like v3) on a dataset's held-out "
          "rows and grade every answer. Returns scores per row id, never held-out content. Use it for the "
          "baseline before training and for every new version; compare models on full held-out evals (the gate "
          "uses each model's eval covering the most rows). split=train is a quick look you may read: limit "
          "samples across the set, ids picks rows (e.g. a batch you just added), and evals eval_id= shows "
          "each answer.",
          S({"name": {"type": "string"}, "model": {"type": "string"}, "limit": {"type": "integer"},
             "split": {"type": "string", "enum": ["heldout", "train"]},
             "ids": {"type": "array", "items": {"type": "string"}}}, ["name", "model"]),
          dataset_eval, permission=lambda a: [] if str(a.get("model") or "").startswith("claude-code/")
          else [("spend", f"eval {a['model']}")]),
        T("evals", "Past evals (a dataset's, or one eval's per-row scores; on split=train also each answer).",
          S({"dataset": {"type": "string"}, "eval_id": {"type": "string"}}), evals_list),
        T("dataset_grade", "Grade an answer you give with the dataset's grader, against a training row "
          "(row_id) or a prompt and reference. Check a grader or a reward on controls before spending: the "
          "reference should pass; an empty, off-task or stuffed answer should not.",
          S({"name": {"type": "string"}, "answer": {"type": "string"}, "row_id": {"type": "string"},
             "prompt": {"type": "string"}, "reference": {"type": "string"}}, ["name", "answer"]), dataset_grade),
        T("model_try", "Ask one model one prompt and see its answer (a version like v3, or provider/model).",
          S({"model": {"type": "string"}, "prompt": {"type": "string"}, "system": {"type": "string"}},
            ["model", "prompt"]), model_try),
        T("dataset_train", "Train the Trainfile's base model on a dataset's training rows with one method (sft, "
          "rejection, opd, rl; default: the recipe's first) on the Trainfile's trainer, with the recipe's settings. "
          "`settings` overrides the method's own (steps, samples, rounds, teacher, max_new_tokens...). Checks "
          "held-out contamination, then asks before uploading and spending.",
          S({"name": {"type": "string"}, "method": {"type": "string", "enum": ["sft", "rejection", "opd", "rl"]},
             "epochs": {"type": "integer"}, "lora_rank": {"type": "integer"}, "learning_rate": {"type": "number"},
             "settings": obj}, ["name"]), dataset_train, permission=lambda a: []),
        T("dataset_promote", "The gate for a version trained on a dataset: it must beat the baseline (what is live, or "
          "the model you name) on the same held-out rows, with no regressions unless the Trainfile allows them.",
          S({"version": {"type": "string"}, "baseline": {"type": "string"}}, ["version"]), dataset_promote,
          permission=lambda a: [("promote", a["version"])]),
    ]
