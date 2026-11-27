"""The engine: one run of a goal, end to end, with every limit enforced by the server.

``train()`` is goal/speed (and goal/cost) as a fixed recipe pairing, the smallest engine that
repeats v0.1 bench from one command (plan: "What we do first", step 7):

    1. data      sessions -> verified tasks (cl/mine + bandits/certify) -> split by date (data/heldout)
    2. baseline  the live version (or the base model) on the held-out tasks, by the evaluator
    3. sample    a teacher attempts each training task n times; keep the shortest verified pass
                 (cl/rollout + data/shortest)
    4. approve   show the estimate and what would be uploaded; ask (attended) or check the policy
    5. train     the Trainfile's backend fine-tunes a LoRA on those passes (train/sft)
    6. eval      the evaluator scores the checkpoint on held-out tasks (eval/heldout)
    7. gate      better on the goal, no held-out task regresses -> promote, else keep the old one

``agent()`` is the open-ended engine: the user's own ``claude -p`` (or ``codex exec``) runs the
trainer role with OpenTrain's train tools served over MCP, choosing recipes itself; the server
still holds the budget, the held-out store and the gate.

Every step writes to the run record, so ``opentrain log`` explains any result afterwards.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import gate as gate_mod
from .backends import get_backend
from .budget import BudgetExceeded, Ledger
from .evaluator import EvalBroken, Evaluator, summarise
from .plugs import AgentRunner, Endpoint, TrainSpec
from .record import Run, RunRecord
from .registry import Registry, Version
from .trainfile import Trainfile

Say = Callable[[str], None]
Ask = Callable[[str], bool]


class EngineStop(RuntimeError):
    """A run stopped on purpose (declined, nothing to learn, budget). Not a crash."""


@dataclass
class RunResult:
    run_id: str
    promoted: bool
    version: str | None
    reasons: list[str]
    summary: dict[str, Any] = field(default_factory=dict)


def parse_provider(text: str | None) -> tuple[str, str | None] | None:
    """``provider  https://host/v1 key ENV_NAME price 0.2 0.6`` -> (base_url, key env).

    The key itself is never stored, only the variable that holds it. ``price`` (USD per million
    input and output tokens) is read by ``provider_prices``.
    """
    if not text:
        return None
    m = re.match(r"\s*(https?://\S+)(?:\s+key\s+([A-Za-z_][A-Za-z0-9_]*))?(?:\s+price\s+[\d.]+\s+[\d.]+)?\s*$", text)
    if not m:
        raise ValueError("provider must be 'https://host/v1', then optionally 'key ENV_NAME' and "
                         f"'price <in> <out>' (USD per million tokens), not {text!r}")
    return m.group(1).rstrip("/"), m.group(2)


def provider_prices(text: str | None) -> tuple[float, float]:
    m = re.search(r"\bprice\s+([\d.]+)\s+([\d.]+)", text or "")
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def teacher_runner(tf: Trainfile, *, max_steps: int = 60) -> tuple[AgentRunner, str]:
    """Who attempts the training tasks.

    With a ``provider`` line, the base model itself (the plan's recipe: sample the base model,
    keep its shortest verified pass, so training sharpens what it can already do). Without one,
    the Trainfile's runtime (the user's frontier CLI) is the teacher, and the specialist learns
    from the frontier's shortest verified passes instead.
    """
    from .runtime import DirectRunner, HeadlessRunner
    from .runtime.headless import TEACHER_TOOLS

    prov = parse_provider(tf.provider)
    if prov:
        p_in, p_out = provider_prices(tf.provider)
        return DirectRunner(Endpoint(prov[0], tf.base_model, prov[1]), max_steps=max_steps,
                            name=f"base:{tf.base_model}", usd_per_mtok_in=p_in, usd_per_mtok_out=p_out), "base model"
    kind = tf.runtime if tf.runtime in ("claude", "codex") else "claude"
    return (HeadlessRunner(kind, name=f"teacher:{kind}", allowed_tools=TEACHER_TOOLS, tools=TEACHER_TOOLS),
            f"frontier ({kind}, headless)")


def _baseline_runner(tf: Trainfile) -> AgentRunner:
    from .runtime import HeadlessRunner

    prov = parse_provider(tf.provider)
    if prov:
        return teacher_runner(tf)[0]
    kind = tf.runtime if tf.runtime in ("claude", "codex") else "claude"
    return HeadlessRunner(kind, name=f"frontier:{kind}")


def specialist_runner(version: Version, *, max_steps: int = 60) -> AgentRunner:
    from .runtime import DirectRunner
    from .serving import endpoint_for

    ep = endpoint_for(version)
    if version.backend == "modal":  # scaled to zero: wait out the cold start once, not inside every attempt
        from .backends import modal_backend

        modal_backend.wait_ready(ep)
    return DirectRunner(ep, max_steps=max_steps, name=version.model_name)


def baseline_name(tf: Trainfile, registry: Registry) -> str:
    """The scores a new version must beat: the live version's, else the base model's (when it can be
    run through a provider), else the frontier teacher's."""
    cur = registry.current()
    if cur:
        return cur.version
    if parse_provider(tf.provider):
        return f"base:{tf.base_model}"
    return f"frontier:{tf.runtime}"


def _scores_from(raw: list[dict] | None) -> list[gate_mod.Score] | None:
    return [gate_mod.Score(**s) for s in raw] if raw else None


def _policy(tf: Trainfile) -> dict[str, Any]:
    d = {k: v for k, v in asdict(tf).items() if k not in ("raw",)}
    d["budget"] = tf.budget.describe()
    return d


def mine_step(state: Path, sessions_root: Path, tf: Trainfile, run: Run, say: Say,
              include_repairable: bool = False, since: str | None = None) -> dict[str, Any]:
    from datetime import datetime, timezone

    from .bench import yield_
    from .cli import known_repos
    from .mine import mine_all
    from .sessions import claude_code
    from .tasks import HeldoutStore, Store, split_store

    sessions, _ = claude_code.load_all(sessions_root)
    if since:
        cut = datetime.fromisoformat(since)
        cut = cut if cut.tzinfo else cut.replace(tzinfo=timezone.utc)
        sessions = [s for s in sessions if s.started_at >= cut]
    repos = {s.cwd for s in sessions if s.cwd}
    say(f"Found       {len(sessions)} Claude Code sessions in {len(repos)} folders")
    say("Trainer is building tasks...")
    verdicts = [yield_.judge(s, known_repos(sessions, [])) for s in sessions]
    report = mine_all(verdicts, state, include_repairable=include_repairable)
    split = split_store(state, tf.heldout_fraction)
    train_ids, held_ids = Store(state).ids(), HeldoutStore(state).ids()
    skipped = {k: v for k, v in report["reasons"].items()}
    say(f"  {len(train_ids) + len(held_ids)} verified tasks, {len(held_ids)} held out for the evaluator")
    if skipped:
        top = ", ".join(f"{v} {k}" for k, v in list(skipped.items())[:4])
        say(f"  {report['skipped'] + report['rejected']} skipped ({top})")
    digests = {t.task_id: t.digest() for t in Store(state)}
    run.add("data", sessions=len(sessions), accepted=report["accepted"], rejected=report["rejected"],
            skipped=report["skipped"], reasons=skipped, train=train_ids, heldout_count=len(held_ids),
            digests=digests, split=split)
    return {"train": train_ids, "heldout": held_ids, "report": report}


def _await_job(backend, job_id: str, run: Run, ledger: Ledger, rid: str, say: Say,
               poll: float, timeout: float, reserved: float = 0.0) -> Any:
    start = time.monotonic()
    last_progress = -1.0
    cancelled = False
    while True:
        st = backend.status(job_id)
        if st.progress is not None and st.progress - last_progress >= 0.1:
            last_progress = st.progress
            say(f"  training  {st.progress:.0%}  ${st.cost_usd:.2f} so far")
        if st.done:
            ledger.settle(rid, st.cost_usd)
            run.add("spend", usd=st.cost_usd, what=f"training job {job_id}", estimate=st.detail)
            run.add("job", job_id=job_id, backend=backend.name, state=st.state, detail=st.detail)
            return st
        # The job may spend its own reservation plus whatever the budget has left, no more.
        if st.cost_usd > reserved + ledger.remaining(run.run_id) and not cancelled:
            backend.cancel(job_id)
            cancelled = True
            say(f"  training  cancelled: ${st.cost_usd:.2f} would exceed {ledger.budget.describe()}")
            run.add("note", text=f"job {job_id} cancelled at ${st.cost_usd:.2f}: over budget")
        if time.monotonic() - start > timeout:
            backend.cancel(job_id)
            raise EngineStop(f"training job {job_id} did not finish in {timeout / 3600:.1f}h; cancelled")
        time.sleep(poll)


def train(
    state: Path,
    tf: Trainfile,
    *,
    sessions_root: Path,
    trigger: str = "manual",
    say: Say = print,
    ask: Ask | None = None,
    n_samples: int = 4,
    workers: int = 2,
    include_repairable: bool = False,
    skip_mine: bool = False,
    poll: float = 15.0,
    job_timeout: float = 12 * 3600,
    teacher: AgentRunner | None = None,
    max_steps: int = 60,
) -> RunResult:
    """One goal run. ``ask`` None means unattended: the Trainfile's data line decides uploads."""
    from .data.sft import build_sft
    from .sample import filter_tasks, sample, shortest
    from .tasks import HeldoutStore, Store

    if tf.goal == "fix":
        raise EngineStop("the fix goal runs through `opentrain fix`, which starts from a flagged session")
    record = RunRecord(state)
    run = record.open(trigger, _policy(tf))
    ledger = Ledger(state, tf.budget)
    registry = Registry(state)
    evaluator = Evaluator(state, tf.eval_cap)
    try:
        # 1. data
        if not skip_mine:
            mine_step(state, sessions_root, tf, run, say, include_repairable)
        train_tasks, held = list(Store(state)), HeldoutStore(state).ids()
        if not held:
            raise EngineStop("no held-out tasks, so nothing can grade a new version. Mine more sessions first.")
        if not train_tasks:
            raise EngineStop("no training tasks after the split. Mine more sessions first.")

        # 2. baseline: what a new version has to beat
        current = registry.current()
        cur_name = baseline_name(tf, registry)
        cur_scores = evaluator.scores_for(cur_name)
        if cur_scores is not None and {s.task_id for s in cur_scores} != set(held):
            say(f"Evaluator   held-out set changed since {cur_name} was scored; scoring it again")
            cur_scores = None
        if cur_scores is None:
            # Without a provider the base model cannot be run, so the bar is the frontier's own score.
            runner = specialist_runner(current, max_steps=max_steps) if current else _baseline_runner(tf)
            say(f"Evaluator   scoring {cur_name} on {len(held)} held-out tasks...")
            cur_scores = evaluator.baseline(runner, cur_name)
        base = summarise(cur_scores)
        med = f"{_dur(base['median_seconds'])} median" if base["median_seconds"] is not None else "no passes"
        say(f"Evaluator   {cur_name}: {base['pass_rate']:.0%} pass, {med}")
        run.add("note", text=f"baseline {cur_name}", **base)

        # 3. sample + shortest
        runner, who = (teacher, getattr(teacher, "name", "custom")) if teacher else teacher_runner(tf, max_steps=max_steps)
        say(f"Sampling    {who}: {n_samples} attempts on each of {len(train_tasks)} training tasks")
        run.add("recipe", data="data/shortest", training="train/sft", goal=f"goal/{tf.goal}", teacher=who,
                n=n_samples)
        try:
            attempts = sample(train_tasks, runner, n_samples, state, ledger, run, workers=workers,
                              timeout=max(t.timeout for t in train_tasks),
                              test_timeout=max(t.timeout for t in train_tasks))
        except BudgetExceeded as exc:
            raise EngineStop(f"budget: {exc}") from None
        fr = filter_tasks(attempts)
        chosen = shortest(attempts)
        say(f"  {len(chosen)} of {len(train_tasks)} tasks have a verified pass"
            f" ({len(fr.none_passed)} never passed)")
        if not chosen:
            raise EngineStop("no verified passes to train on; the teacher could not solve any training task")
        manifest = build_sft(chosen, state, run=run)

        # 4. approve
        backend = get_backend(tf.trainer, state)
        ok, why = backend.available()
        if not ok:
            raise EngineStop(f"trainer backend {tf.trainer!r} cannot run here: {why}")
        if not manifest["examples"]:
            reasons = {s["reason"] for s in manifest["skipped"]}
            raise EngineStop(f"no usable training examples ({'; '.join(sorted(reasons))})")
        spec = TrainSpec(base_model=tf.base_model, data=Path(manifest["file"]), name=f"opentrain-{run.run_id}")
        est = backend.estimate(spec)
        uploads = 0 if tf.trainer == "local" else manifest.get("examples", len(chosen))
        say(f"\nProposed    goal/{tf.goal}: tune {tf.base_model} on shortest verified passes")
        say(f"            est. ${est:.2f} on {tf.trainer}, uploads {uploads} redacted tasks;"
            f" ${ledger.remaining(run.run_id):.2f} of {tf.budget.describe()} left")
        if ask is not None:
            if not ask("Proceed?"):
                raise EngineStop("declined at the approval step")
        elif uploads and tf.uploads_need_approval and not _approved(state, tf.trainer):
            raise EngineStop("unattended run would upload data, and the Trainfile says 'until I approve'. "
                             "Run `opentrain train` once attended, or `opentrain approve " + tf.trainer + "`.")
        if ask is not None and uploads:
            _approve(state, tf.trainer)

        # 5. train
        try:
            rid = ledger.reserve(est, f"training on {tf.trainer}", run.run_id)
        except BudgetExceeded as exc:
            raise EngineStop(f"budget: {exc}") from None
        job_id = backend.launch(spec)
        run.add("job", job_id=job_id, backend=tf.trainer, state="launched", estimate_usd=est, spec=str(spec))
        say(f"Training    job {job_id} on {tf.trainer}")
        st = _await_job(backend, job_id, run, ledger, rid, say, poll, job_timeout, reserved=est)
        if st.state != "succeeded" or not st.checkpoint:
            raise EngineStop(f"training job {job_id} {st.state}: {st.detail}")
        version = registry.add(run_id=run.run_id, base_model=tf.base_model, backend=tf.trainer,
                               checkpoint=st.checkpoint)
        run.add("checkpoint", version=version.version, checkpoint=st.checkpoint)
        if st.checkpoint.startswith("fake://"):
            say("  (fake backend: simulated, no weights were trained)")

        # 6. eval
        say(f"Evaluator   scoring {version.version} on {len(held)} held-out tasks...")
        cand_scores = evaluator.request(specialist_runner(version, max_steps=max_steps), version.model_name, run)
        evaluator.save_scores(version.version, cand_scores)

        # 7. gate
        from .certify import lint

        # Verifier controls rerun at every evaluation: a held-out verifier that stopped holding
        # (a dependency moved, a flaky test) must not decide a promotion.
        broken = [r["task_id"] for r in lint(state, only_heldout=True)["broken"]]
        decision = gate_mod.decide(cur_scores, cand_scores, tf.goal, allow_regressions=tf.allow_regressions,
                                   broken_verifiers=broken)
        cand = summarise(cand_scores)
        registry.record_gate(version.version, decision.explain(),
                             {**cand, "per_task": [asdict(s) for s in cand_scores], "against": cur_name})
        run.add("gate", promote=decision.promote, version=version.version, reasons=decision.reasons,
                regressions=decision.regressions, fixed=decision.fixed, summary=decision.summary)
        if decision.promote:
            registry.promote(version.version, decision.explain())
        _report(say, version.version, decision, base, cand)
        run.add("closed", promoted=decision.promote)
        return RunResult(run.run_id, decision.promote, version.version, decision.reasons, decision.summary)
    except EvalBroken as exc:
        run.add("note", text=f"stopped: {exc}")
        run.add("closed", promoted=False, stopped=str(exc))
        say(f"\nStopped     {exc}")
        return RunResult(run.run_id, False, None, [str(exc)])
    except EngineStop as stop:
        run.add("note", text=f"stopped: {stop}")
        run.add("closed", promoted=False, stopped=str(stop))
        say(f"\nStopped     {stop}")
        return RunResult(run.run_id, False, None, [str(stop)])
    except BaseException as exc:
        if not run.closed:
            run.add("note", text=f"crashed: {type(exc).__name__}: {exc}")
            run.add("closed", promoted=False, crashed=True)
        raise


def _report(say: Say, version: str, d: gate_mod.Decision, base: dict, cand: dict) -> None:
    head = f"{version} passed the gate" if d.promote else f"{version} did not pass the gate"
    say(f"\n{head}")
    if base.get("median_seconds") is not None and cand.get("median_seconds") is not None:
        say(f"  Median task   {_dur(base['median_seconds'])} -> {_dur(cand['median_seconds'])}")
    if base.get("pass_rate") is not None and cand.get("pass_rate") is not None:
        say(f"  Pass rate     {base['pass_rate']:.0%} -> {cand['pass_rate']:.0%}"
            f"    {len(d.fixed)} fixed, {len(d.regressions)} regressed (opentrain diff)")
    if not d.promote:
        for r in d.reasons:
            say(f"  {r}")
    if d.promote:
        say(f"  Live as opentrain/{version} and opentrain/current (opentrain serve; opentrain connect pi)")
        say("  Next: opentrain race")


def _dur(s: float) -> str:
    if s >= 60:
        return f"{int(s // 60)}m{int(s % 60):02d}"
    return f"{s:.0f}s" if s >= 10 else f"{s:.1f}s"


def _approvals(state: Path) -> Path:
    return state / "approvals.json"


def _approved(state: Path, trainer: str) -> bool:
    p = _approvals(state)
    return p.exists() and trainer in json.loads(p.read_text()).get("uploads", [])


def _approve(state: Path, trainer: str) -> None:
    p = _approvals(state)
    data = json.loads(p.read_text()) if p.exists() else {"uploads": []}
    if trainer not in data["uploads"]:
        data["uploads"].append(trainer)
    p.write_text(json.dumps(data, indent=2))


# --------------------------------------------------------------------------- agent mode

ROLE_DIR = Path(__file__).parent / "roles"
SKILLS_DIR = (Path(__file__).resolve().parents[2] / "skills" if (Path(__file__).resolve().parents[2] / "skills").is_dir()
              else Path(__file__).parent / "skills")


def agent(state: Path, tf: Trainfile, instruction: str, *, role: str = "trainer", say: Say = print,
          timeout: float = 4 * 3600, model: str | None = None) -> dict[str, Any]:
    """Run the trainer role in the user's own CLI with OpenTrain's train tools (headless runtime).

    The CLI does the reasoning; OpenTrain decides which role runs, with which tools and limits.
    The role's tools reach the server through ``opentrain mcp --role <role> --run <id>``, so the
    budget, the eval cap, the held-out store and the gate stay on this side.
    """
    import sys

    from .runtime.headless import headless_command, run_headless

    if tf.runtime not in ("claude", "codex"):
        raise EngineStop("agent mode needs the headless runtime (runtime claude -p or codex exec)")
    if shutil.which(tf.runtime) is None:
        raise EngineStop(f"`{tf.runtime}` is not on PATH; install it or switch runtime in the Trainfile")
    run = RunRecord(state).open(f"agent: {instruction[:80]}", _policy(tf), role=role)
    mcp = state / "runs" / run.run_id / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"opentrain": {
        "command": sys.executable,
        "args": ["-m", "opentrain.mcp", "--state", str(state), "--role", role, "--run", run.run_id],
    }}}, indent=2))
    prompt_file = ROLE_DIR / f"{role}.md"
    from .mcp import TOOLS_BY_ROLE

    # The trainer reasons with OpenTrain's tools and may read code and skills, but gets no shell and
    # no file writes: with either it could read the held-out store or rewrite the record, ledger or
    # registry. Reads under .opentrain are denied outright.
    builtin = ["Read", "Grep", "Glob"]
    allowed = builtin + [f"mcp__opentrain__{t}" for t in TOOLS_BY_ROLE[role]]
    denied = [f"{t}({state}/**)" for t in builtin] + ["Bash", "Write", "Edit", "NotebookEdit", "WebFetch"]
    argv = headless_command(tf.runtime, instruction, system_prompt_file=prompt_file, mcp_config=mcp,
                            allowed_tools=allowed, cwd=state.parent, model=model, tools=builtin)
    if tf.runtime == "claude":
        argv += ["--disallowedTools", ",".join(denied), "--add-dir", str(SKILLS_DIR)]
    say(f"Engine      {tf.runtime} headless, your own login; role {role}; run {run.run_id}")
    res = run_headless(argv, state.parent, timeout)
    run.add("spend", usd=res.cost_usd or 0.0, what=f"{tf.runtime} engine turns", tokens=res.tokens,
            turns=res.turns, notional_usd=getattr(res, "notional_usd", None))
    if not run.closed:
        run.add("closed", text=(res.text or "")[-2000:], error=res.error)
    say(res.text or res.error or "")
    return {"run_id": run.run_id, "text": res.text, "cost_usd": res.cost_usd, "error": res.error}
