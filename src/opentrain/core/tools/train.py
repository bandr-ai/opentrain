"""Train tools: the typed actions that cost money, touch the evals, or change what is live.

Each one goes through the server's enforcement in ``opentrain.mcp.Server`` (budget ledger, eval
cap, held-out isolation, the gate, the registry) under the session's own run in the run record.
Work that takes minutes or hours (mining, sampling, scoring, training) starts a job and returns at
once; the job reports back into the session when it ends.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from ..jobs import JobContext
from ..tool import Tool, ToolContext, ToolError, ToolResult, schema


def _server(ctx: ToolContext, role: str = "trainer"):
    """The enforcement server for this session's run (opened on first use)."""
    from ... import trainfile
    from ...mcp import Server

    s = ctx.project.session(ctx.session_id)
    root = ctx.project.session(s["parent_id"]) if s.get("parent_id") else s  # subagents share the parent's run
    run_id = (root.get("metadata") or {}).get("run_id")
    tf = trainfile.for_project(ctx.project.directory)
    ctx.project.state.mkdir(parents=True, exist_ok=True)
    srv = Server(ctx.project.state, role, run_id, tf)
    if not run_id:
        ctx.project.store.update("session", root["id"], {"metadata": {"run_id": srv.run.run_id}})
    return srv


def _refuse(fn):
    """Turn the enforcement server's refusals into ToolErrors the model reads."""
    from ...mcp import ToolError as McpToolError

    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except McpToolError as exc:
            raise ToolError(str(exc)) from None
    return wrapped


def _jobs(ctx: ToolContext):
    return ctx.project.services["jobs"]


def _started(job: dict[str, Any], what: str) -> ToolResult:
    return ToolResult(f"Started job {job['id']} ({what}). It runs on the server; you will get a system reminder "
                      "when it ends. Do not poll; continue with other work or end your turn.",
                      title=what, metadata={"job_id": job["id"]})


# ---------------------------------------------------------------------- status


def status(ctx: ToolContext) -> Any:
    return project_status(ctx.project)


def project_status(project: Any) -> dict[str, Any]:
    from ... import trainfile
    from ...budget import Ledger
    from ...registry import Registry
    from ...serving import live_version
    from ...tasks import HeldoutStore, Store

    state, d = project.state, project.directory
    tf_path = trainfile.find(d)
    tf = trainfile.for_project(d)
    reg = Registry(state)
    cur = reg.current()
    ledger = Ledger(state, tf.budget)
    jobs = project.services["jobs"].list(active=True)
    from ...datasets import Datasets
    from .data import summarize

    return {
        "trainfile": str(tf_path) if tf_path else "none (defaults apply; the init tool writes one)",
        "policy": {"goal": tf.objective or tf.goal, "budget": tf.budget.describe(), "promote": tf.promote, "trainer": tf.trainer,
                   "base_model": tf.base_model, "runtime": tf.runtime, "heldout": tf.heldout_fraction,
                   "eval_cap": tf.eval_cap, "provider": tf.provider},
        "budget": {"committed_usd": round(ledger.committed(), 4), "reserved_usd": round(ledger.reserved(), 4),
                   "remaining_usd": round(ledger.remaining(), 4)},
        "datasets": [summarize(state, n) for n in Datasets(state).names()],
        "coding_tasks": {"training": len(Store(state).ids()), "heldout": len(HeldoutStore(state).ids())},
        "live": {"version": cur.version if cur else None, "pinned": live_version(state),
                 "base_model": cur.base_model if cur else tf.base_model},
        "versions": [{"version": v.version, "decision": v.decision, "base_model": v.base_model}
                     for v in reg.versions()[-5:]],
        "sandbox": _sandbox(),
        "jobs_running": [{"id": j["id"], "kind": j["kind"], "title": j["title"], "progress": j.get("progress"),
                          "detail": j.get("detail")} for j in jobs],
    }


def _sandbox() -> str:
    from ..sandbox import kind

    return {"seatbelt": "on (macOS sandbox-exec)", "bwrap": "on (bubblewrap)",
            "off": "OFF (OPENTRAIN_SANDBOX=off)", "none": "unavailable on this machine"}[kind()]


def init(ctx: ToolContext, goal: str = "better than the current model on my held-out examples", budget: str = "$20 a week", trainer: str | None = None,
         base_model: str = "Qwen/Qwen3-8B", heldout: str = "30%") -> ToolResult:
    from ... import trainfile

    trainer = trainer or trainfile.ready_trainer()

    path = ctx.project.directory / "Trainfile"
    if path.exists():
        raise ToolError("a Trainfile already exists; edit it instead")
    text = trainfile.default_text()
    for key, val in (("goal", goal), ("budget", f"{budget}, model tokens and compute together"),
                     ("trainer", trainer), ("base model", base_model), ("heldout", heldout)):
        lines = text.splitlines()
        if any(ln.split("  ")[0].strip() == key for ln in lines):
            text = "\n".join(f"{key:<12} {val}" if ln.split("  ")[0].strip() == key else ln for ln in lines) + "\n"
        else:
            text = text.rstrip("\n") + f"\n{key:<12} {val}\n"
    trainfile.parse(text)  # refuse anything the server would not accept
    path.write_text(text)
    return ToolResult(f"Wrote {path}:\n\n{text}", title="Trainfile")


def goal_set(ctx: ToolContext, goal: str) -> ToolResult:
    """Only the goal line: the user's words for what the model should get better at. Budget and gate stay."""
    from ... import trainfile

    goal = " ".join(goal.split())
    if not goal:
        raise ToolError("say the goal in words")
    path = ctx.project.directory / "Trainfile"
    text = path.read_text() if path.exists() else trainfile.default_text()
    lines = text.splitlines()
    if any(ln.split("  ")[0].strip() == "goal" for ln in lines):
        text = "\n".join(f"{'goal':<12} {goal}" if ln.split("  ")[0].strip() == "goal" else ln for ln in lines) + "\n"
    else:
        text = f"{'goal':<12} {goal}\n" + text
    trainfile.parse(text)
    path.write_text(text)
    return ToolResult(f"Goal set: {goal}", title="goal")


def trainer_set(ctx: ToolContext, trainer: str | None = None, base_model: str | None = None) -> ToolResult:
    """Only the trainer and base model lines: where to train and what to train. Budget and gate stay."""
    from ... import trainfile
    from ...backends import BACKENDS, get_backend

    if not trainer and not base_model:
        raise ToolError("give trainer, base_model or both")
    said = []
    if trainer:
        if trainer not in BACKENDS or trainer == "fake":
            raise ToolError(f"no trainer {trainer!r}; known: {', '.join(b for b in BACKENDS if b != 'fake')}")
        trainfile.set_value(ctx.project.directory, "trainer", trainer)
        ok, why = get_backend(trainer, ctx.project.state).available()
        said.append(f"trainer {trainer}" + ("" if ok else f" (not ready here: {why})"))
    if base_model:
        trainfile.set_value(ctx.project.directory, "base model", " ".join(base_model.split()))
        said.append(f"base model {base_model}")
    return ToolResult("Set " + " and ".join(said) + ". dataset_recipe shows the new cost.", title="trainer")


def backends(ctx: ToolContext) -> Any:
    from ...backends import list_backends

    return [{"name": n, "available": ok, "detail": why} for n, ok, why in list_backends(ctx.project.state)]


def record_query(ctx: ToolContext, run_id: str | None = None, last: int = 5) -> Any:
    return _refuse(_server(ctx).t_record_query)(run_id, last)


# ---------------------------------------------------------------------- data


def sessions_scan(ctx: ToolContext, since: str | None = None) -> Any:
    from datetime import datetime, timezone

    from ...bench import timing, yield_
    from ...cli import known_repos
    from ...sessions import claude_code

    sessions, unreadable = claude_code.load_all(claude_code.DEFAULT_ROOT)
    if since:
        cut = datetime.fromisoformat(since)
        cut = cut if cut.tzinfo else cut.replace(tzinfo=timezone.utc)
        sessions = [s for s in sessions if s.started_at >= cut]
    if not sessions:
        return {"sessions": 0, "note": f"no Claude Code sessions under {claude_code.DEFAULT_ROOT}"}
    verdicts = [yield_.judge(s, known_repos(sessions, [])) for s in sessions]
    y = yield_.summarise(verdicts)
    t = timing.breakdown(sessions)
    return {"sessions": len(sessions), "unreadable": len(unreadable),
            "yield": {k: y[k] for k in ("verified_candidates", "repairable", "coding_sessions", "yield_per_100",
                                        "yield_with_repair_per_100")},
            "skipped": {f["label"]: f["dropped"] for f in y["funnel"] if f["dropped"]},
            "time_hours": t["hours"], "model_share_of_working_time_pct": t["model_share_of_working_time_pct"]}


def tasks_mine(ctx: ToolContext, include_repairable: bool = False) -> ToolResult:
    srv = _server(ctx)

    def work(job: JobContext) -> Any:
        job.progress(None, "reading sessions and certifying verifiers")
        return _refuse(srv.t_tasks_mine)(include_repairable)

    def summary(r: Any) -> str:
        top = ", ".join(f"{v} {k}" for k, v in list((r.get("reasons") or {}).items())[:4])
        return f"{r['train']} training tasks, {r['heldout']} held out. Skipped: {top or 'none'}."

    return _started(_jobs(ctx).start_local(ctx.session_id, "mine", "mine sessions into tasks", work,
                                           run_id=srv.run.run_id, summarize=summary), "mining tasks")


def tasks_list(ctx: ToolContext, family: str | None = None) -> Any:
    return _refuse(_server(ctx).t_tasks_list)(family)


def task_show(ctx: ToolContext, task_id: str) -> Any:
    from ...tasks import HeldoutStore, Store

    if task_id in HeldoutStore(ctx.project.state):
        raise ToolError(f"{task_id} is held out; only the evaluator reads it")
    store = Store(ctx.project.state)
    if task_id not in store:
        raise ToolError(f"no training task {task_id}")
    t = store.get(task_id)
    c = store.controls(task_id)
    return {"task_id": t.task_id, "family": t.family, "repo": t.repo, "base_commit": t.base_commit,
            "test_command": t.test_command, "test_files": list(t.test_files), "created_at": t.created_at,
            "instruction": t.instruction[:6000], "controls": c.detail if c else None,
            "folder": str(store.folder(task_id))}


def task_add(ctx: ToolContext, folder: str) -> Any:
    from ...certify import run_controls
    from ...tasks import Store, TaskRejected, read_task

    p = Path(folder).expanduser()
    p = p if p.is_absolute() else ctx.project.directory / p
    try:
        task = read_task(p)
    except Exception as exc:
        raise ToolError(f"not a task folder ({exc}); it needs task.toml, instruction.md, tests/, solution/") from None
    patch_file = p / "solution" / "solution.patch"
    controls, _ = run_controls(task, patch_file.read_text() if patch_file.exists() else None)
    try:
        Store(ctx.project.state).add(p, controls)
    except TaskRejected as exc:
        raise ToolError(f"rejected: {exc}") from None
    return {"added": task.task_id, "controls": controls.detail}


def tasks_lint(ctx: ToolContext) -> Any:
    from ...certify import lint

    r = lint(ctx.project.state)
    return {"checked": r.get("checked"), "broken": [{"task_id": b["task_id"], "reason": b.get("reason")}
                                                   for b in r.get("broken", [])]}


def sample_run(ctx: ToolContext, task_ids: list[str] | None = None, n: int = 4, teacher: str = "frontier") -> ToolResult:
    srv = _server(ctx)
    if not (1 <= int(n) <= 16):
        raise ToolError("n must be between 1 and 16")

    def work(job: JobContext) -> Any:
        job.progress(None, f"{n} attempts per task with the {teacher} model")
        return _refuse(srv.t_sample_run)(task_ids, int(n), teacher)

    def summary(r: Any) -> str:
        return (f"{r['passed']} of {r['attempts']} attempts passed ({r['teacher']}). {len(r['never_passed'])} tasks "
                f"never passed. Next: data_build.")

    return _started(_jobs(ctx).start_local(ctx.session_id, "sample", f"sample {teacher} ×{n}", work,
                                           run_id=srv.run.run_id, summarize=summary), "sampling")


def data_build(ctx: ToolContext, task_ids: list[str] | None = None) -> Any:
    return _refuse(_server(ctx).t_data_build)(task_ids)


def train_estimate(ctx: ToolContext, epochs: int | None = None, lora_rank: int | None = None,
                   method: str | None = None, extra: dict[str, Any] | None = None) -> Any:
    from ...backends import get_backend
    from ...budget import Ledger
    from ...plugs import TrainSpec

    srv = _server(ctx)
    tf = srv.tf
    data = ctx.project.state / "data" / "train.jsonl"
    examples = sum(1 for ln in data.read_text().splitlines() if ln.strip()) if data.exists() else 0
    backend = get_backend(tf.trainer, ctx.project.state)
    ok, why = backend.available()
    out: dict[str, Any] = {"trainer": tf.trainer, "base_model": tf.base_model, "examples": examples,
                           "backend_available": ok, "backend_detail": why,
                           "budget_remaining_usd": round(Ledger(ctx.project.state, tf.budget).remaining(), 2),
                           "uploads": f"{examples} redacted training examples" if tf.trainer not in ("local", "fake")
                           else "nothing (trains on this machine)"}
    if examples and ok:
        kw = {k: v for k, v in {"epochs": epochs, "lora_rank": lora_rank, "method": method,
                                "extra": extra}.items() if v is not None}
        out["estimate_usd"] = round(backend.estimate(TrainSpec(base_model=tf.base_model, data=data, name="estimate",
                                                               **kw)), 2)
    elif not examples:
        out["note"] = "no training set yet: sample_run, then data_build"
    return out


# ---------------------------------------------------------------------- jobs


async def job_start(ctx: ToolContext, epochs: int | None = None, lora_rank: int | None = None,
                    learning_rate: float | None = None, method: str | None = None,
                    extra: dict[str, Any] | None = None, batch_size: int | None = None) -> ToolResult:
    try:
        est = await asyncio.to_thread(train_estimate, ctx, epochs, lora_rank, method, extra)
    except (ValueError, RuntimeError) as exc:  # a backend that cannot run this method says why
        raise ToolError(str(exc)) from None
    if not est.get("backend_available"):
        raise ToolError(f"trainer {est['trainer']} is not available here: {est['backend_detail']}")
    if not est.get("examples"):
        raise ToolError("no training set; run sample_run, then data_build")
    remote = est["trainer"] not in ("local", "fake")
    if remote:
        await ctx.ask("upload", [est["trainer"]], metadata={"what": est["uploads"], "trainer": est["trainer"]},
                      always=[est["trainer"]])
    await ctx.ask("spend", [f"${est.get('estimate_usd', 0):.2f} on {est['trainer']}"],
                  metadata={"usd": est.get("estimate_usd"), "trainer": est["trainer"],
                            "budget_remaining_usd": est["budget_remaining_usd"]}, always=[])
    await ctx.ask("job.start", [est["trainer"]], metadata={"base_model": est["base_model"]})
    srv = await asyncio.to_thread(_server, ctx)
    r = await asyncio.to_thread(_refuse(srv.t_train_launch), epochs, lora_rank, learning_rate, method, extra,
                                batch_size)
    job = _jobs(ctx).track_backend(ctx.session_id, r["job_id"], est["trainer"],
                                   f"{est['base_model']} on {est['trainer']}", srv.run.run_id, r.get("estimate_usd"))
    return ToolResult(f"Training job {job['id']} launched on {est['trainer']} (backend id {r['job_id']}, estimated "
                      f"${r['estimate_usd']:.2f}, reserved against the budget). You will get a system reminder when it "
                      "ends; end your turn now unless there is other work.", title="training launched",
                      metadata={"job_id": job["id"], "estimate_usd": r["estimate_usd"]})


def jobs_list(ctx: ToolContext, active_only: bool = False) -> Any:
    return [{k: j.get(k) for k in ("id", "kind", "title", "state", "progress", "detail", "cost_usd", "version",
                                   "error")} for j in _jobs(ctx).list(active=active_only)[:20]]


def job_status(ctx: ToolContext, job_id: str) -> Any:
    j = _jobs(ctx).get(job_id)
    if j.get("backend_job") and j["state"] not in ("succeeded", "failed", "cancelled"):
        _jobs(ctx).refresh(job_id)
        j = _jobs(ctx).get(job_id)
    return {k: j.get(k) for k in ("id", "kind", "title", "state", "progress", "detail", "cost_usd", "estimate_usd",
                                  "version", "error", "result", "time")}


async def job_wait(ctx: ToolContext, job_id: str, timeout: int = 600) -> Any:
    """Block until a job ends or ``timeout`` seconds pass (at most 15 minutes), then report it.
    For agents outside OpenTrain, which get no reminder when a job ends."""
    end = time.time() + max(5, min(int(timeout), 900))
    while True:
        j = await asyncio.to_thread(job_status, ctx, job_id)
        if j["state"] in ("succeeded", "failed", "cancelled") or time.time() >= end or ctx.abort.is_set():
            return {**j, "waited": "finished" if j["state"] != "running" else "timed out; call job_wait again"}
        await asyncio.sleep(3)


def job_logs(ctx: ToolContext, job_id: str, tail: int = 80) -> ToolResult:
    lines = _jobs(ctx).logs(job_id, max(1, min(int(tail), 1000)))
    return ToolResult("\n".join(lines) or "(no log lines yet)", title=f"logs {job_id}")


def job_stop(ctx: ToolContext, job_id: str) -> Any:
    j = _jobs(ctx).cancel(job_id)
    return {"id": j["id"], "state": j["state"], "detail": j.get("detail")}


# ---------------------------------------------------------------------- evals and versions


def evaluate_in_process(state: Path, version: str, run_id: str | None, job: JobContext | None = None) -> Any:
    """Run the evaluator as its own process; only scores come back."""
    import json
    import subprocess
    import sys

    argv = [sys.executable, "-m", "opentrain.core.evaluate", "--state", str(state), "--version", version]
    if run_id:
        argv += ["--run", run_id]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    while True:
        try:
            out, err = proc.communicate(timeout=2)
            break
        except subprocess.TimeoutExpired:
            if job is not None and job.cancel_event.is_set():
                proc.kill()
                proc.communicate()
                job.check()
    if job is not None and err:
        for line in err.splitlines()[-200:]:
            job.log(line)
    try:
        result = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
    except ValueError:
        result = {}
    if proc.returncode == 2 and result.get("error"):
        raise ToolError(result["error"])
    if proc.returncode != 0 or not result:
        raise ToolError(f"the evaluator failed (exit {proc.returncode}): {(err or '').strip()[-400:]}")
    return result


def eval_request(ctx: ToolContext, version: str) -> ToolResult:
    srv = _server(ctx)
    state, run_id = ctx.project.state, srv.run.run_id

    def work(job: JobContext) -> Any:
        job.progress(None, f"scoring {version} on held-out tasks (evaluator process)")
        return evaluate_in_process(state, version, run_id, job)

    def summary(r: Any) -> str:
        failed = [t["task_id"] for t in r.get("per_task", []) if not t["passed"]]
        return (f"{version}: {r.get('passed')}/{r.get('tasks')} held-out tasks passed, median "
                f"{r.get('median_seconds')}s. Failed ids: {', '.join(failed[:20]) or 'none'}. "
                f"Next: version_propose {version}.")

    return _started(_jobs(ctx).start_local(ctx.session_id, "eval", f"score {version} on held-out", work,
                                           run_id=srv.run.run_id, summarize=summary), f"scoring {version}")


def version_propose(ctx: ToolContext, version: str) -> Any:
    from ...registry import Registry
    from ..integrity import provenance

    srv = _server(ctx)
    flags = provenance(ctx.project.state, version, srv.tf.base_model, ctx.project.store)
    if flags:
        Registry(ctx.project.state).record_gate(version, "rejected: " + "; ".join(flags), {"provenance": flags})
        srv.run.add("gate", promote=False, version=version, reasons=flags, regressions=[], fixed=[],
                    summary={"provenance": flags})
        return {"promoted": False, "reasons": flags, "note": "integrity checks failed; the version is scored as "
                                                             "the baseline and cannot be promoted"}
    return _refuse(srv.t_version_propose)(version)


def versions(ctx: ToolContext) -> Any:
    from ...registry import Registry
    from ...serving import live_version

    reg = Registry(ctx.project.state)
    cur = reg.current()
    return {"current": cur.version if cur else None, "pinned": live_version(ctx.project.state),
            "versions": [{"version": v.version, "decision": v.decision, "base_model": v.base_model,
                          "backend": v.backend, "run_id": v.run_id} for v in reg.versions()]}


def version_use(ctx: ToolContext, version: str) -> Any:
    from ...registry import Registry
    from ...serving import use_version

    if version == "previous":
        v = Registry(ctx.project.state).revert("reverted by the train agent")
        return {"current": v.version if v else None}
    use_version(ctx.project.state, None if version == "current" else version, why="set by the train agent")
    return {"serving": version}


def export(ctx: ToolContext, out: str = "opentrain-export") -> Any:
    from ...export import export as do_export

    p = Path(out).expanduser()
    p = p if p.is_absolute() else ctx.project.directory / p
    m = do_export(ctx.project.state, p, include_heldout=False)
    return {"out": str(p), **{k: m[k] for k in m if k in ("tasks", "examples", "files")}}


def tools() -> list[Tool]:
    S = schema
    ids = {"type": "array", "items": {"type": "string"}}
    T = lambda *a, **k: Tool(*a, group="train", **k)  # noqa: E731
    return [
        T("status", "Policy (Trainfile), budget left, task counts, what is live, recent versions and running jobs. "
          "Call this first.", S(), status),
        T("init", "Write a Trainfile for this project (only when none exists).",
          S({"goal": {"type": "string", "description": "what the model should get better at, in plain words "
                                                      "(or speed/fix/cost for coding-session work)"}, "budget": {"type": "string"},
             "trainer": {"type": "string"}, "base_model": {"type": "string"}, "heldout": {"type": "string"}}),
          init, permission=lambda a: [("write", "Trainfile")]),
        T("goal_set", "Set the Trainfile's goal: what the model should get better at, in the user's words. Changes "
          "only the goal line; creates the Trainfile with defaults when there is none.",
          S({"goal": {"type": "string"}}, ["goal"]), goal_set, title=lambda a: a["goal"][:60]),
        T("trainer_set", "Set where to train (trainer: modal, runpod, fireworks, tinker, local) and the base model "
          "(a Hugging Face id, e.g. Qwen/Qwen3.5-4B). Changes only those Trainfile lines; budget and gate stay.",
          S({"trainer": {"type": "string"}, "base_model": {"type": "string"}}), trainer_set,
          title=lambda a: " ".join(x for x in (a.get("trainer"), a.get("base_model")) if x)),
        T("backends", "Trainer backends and whether each can run here.", S(), backends),
        T("record_query", "Past runs: triggers, data, recipes, spend, scores, gate decisions.",
          S({"run_id": {"type": "string"}, "last": {"type": "integer"}}), record_query),
        T("sessions_scan", "Scan the user's Claude Code sessions: how many can become tasks and where their time goes.",
          S({"since": {"type": "string", "description": "ISO date"}}), sessions_scan),
        T("tasks_mine", "Job: build verified tasks from the user's sessions (verifiers certified by controls), then "
          "split held-out by date.", S({"include_repairable": {"type": "boolean"}}), tasks_mine),
        T("tasks_list", "Training tasks (never held-out ones).", S({"family": {"type": "string"}}), tasks_list),
        T("task_show", "One training task: instruction, test command, controls, folder.",
          S({"task_id": {"type": "string"}}, ["task_id"]), task_show),
        T("task_add", "Register a task folder you built or repaired (Harbor layout). The server runs its controls "
          "and refuses it unless the reference passes and doing nothing fails.",
          S({"folder": {"type": "string"}}, ["folder"]), task_add),
        T("tasks_lint", "Re-run every stored verifier's controls; ids and reasons of broken ones.", S(), tasks_lint),
        T("sample_run", "Job: run the teacher (frontier) or base model n times on training tasks; each attempt is "
          "verified. Metered against the budget.",
          S({"task_ids": ids, "n": {"type": "integer"}, "teacher": {"type": "string", "enum": ["frontier", "base"]}}),
          sample_run, permission=lambda a: [("spend", f"sampling ×{a.get('n', 4)}")]),
        T("data_build", "Build the training set: the shortest verified pass per task (data/shortest).",
          S({"task_ids": ids}), data_build),
        T("train_estimate", "Estimated cost of training on the current data, what would be uploaded, budget left.",
          S({"epochs": {"type": "integer"}, "lora_rank": {"type": "integer"}}), train_estimate),
        T("job_start", "Launch training on the Trainfile's trainer and base model. Asks the user before uploading and "
          "spending. Returns at once; the job reports back.",
          S({"epochs": {"type": "integer"}, "lora_rank": {"type": "integer"}, "learning_rate": {"type": "number"}}),
          job_start, permission=lambda a: []),
        T("jobs", "List jobs (mining, sampling, scoring, training).", S({"active_only": {"type": "boolean"}}), jobs_list),
        T("job_status", "One job's state, progress, cost and result.", S({"job_id": {"type": "string"}}, ["job_id"]),
          job_status),
        T("job_wait", "Wait until a job ends (or up to `timeout` seconds, default 600) and return its result. Use it "
          "instead of polling job_status.", S({"job_id": {"type": "string"}, "timeout": {"type": "integer"}},
                                              ["job_id"]), job_wait),
        T("job_logs", "The last lines of a job's log.", S({"job_id": {"type": "string"}, "tail": {"type": "integer"}},
                                                         ["job_id"]), job_logs),
        T("job_stop", "Cancel a job.", S({"job_id": {"type": "string"}}, ["job_id"]), job_stop,
          permission=lambda a: [("job.stop", a["job_id"])]),
        T("eval_request", "Job: the evaluator scores a version on held-out tasks (pass/fail and time per task id, "
          "never content). Capped per run.", S({"version": {"type": "string"}}, ["version"]), eval_request),
        T("version_propose", "Submit a scored version to the promotion gate. The gate decides and deploys.",
          S({"version": {"type": "string"}}, ["version"]), version_propose,
          permission=lambda a: [("promote", a["version"])]),
        T("versions", "Every model version, its gate decision, and what is live.", S(), versions),
        T("version_use", "Change what opentrain serve answers with: a version, 'current' (follow the gate) or "
          "'previous' (roll back).", S({"version": {"type": "string"}}, ["version"]), version_use,
          permission=lambda a: [("serve", a["version"])]),
        T("export", "Write tasks and the training set in open formats (Harbor, JSONL) for any trainer.",
          S({"out": {"type": "string"}}), export, permission=lambda a: [("export", a.get("out") or "*")]),
    ]
