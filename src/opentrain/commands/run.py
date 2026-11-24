"""Commands that run models on tasks: ``sample``, ``data build``, ``eval``, ``race``, ``bench-models``.

Which model runs is one flag, ``--model``:

    claude, claude:<model>, codex, codex:<model>   the user's own CLI, headless, under their login
    v3                                             a registry version, via its serving endpoint
    <name> --endpoint <base url>                   any OpenAI-compatible endpoint, direct runtime

Spend is reserved against the Trainfile's budget before each attempt and settled after it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import trainfile
from ..budget import BudgetExceeded, Ledger
from ..data.sft import build_sft
from ..evaluator import EvalCapReached, Evaluator, summarise
from ..plugs import AgentRunner, Endpoint
from ..race import newest_heldout, race
from ..record import Run, RunRecord
from ..registry import Registry, RegistryError
from ..runtime import DirectRunner, HeadlessRunner
from ..sample import filter_tasks, load_attempts, sample, shortest
from ..state import ensure
from ..tasks import HeldoutStore, Store


def _trainfile(args) -> trainfile.Trainfile:
    return trainfile.load(trainfile.find(Path(args.state or ".").resolve()))


def make_runner(model: str, *, state: Path, endpoint: str | None = None, api_key_env: str | None = None,
                max_steps: int = 60) -> AgentRunner:
    """A runner from a ``--model`` value (see the module docstring)."""
    kind, _, sub = model.partition(":")
    if endpoint:
        return DirectRunner(Endpoint(endpoint, model, api_key_env), max_steps=max_steps)
    if kind in ("claude", "codex"):
        return HeadlessRunner(kind, sub or None)  # type: ignore[arg-type]
    try:
        version = Registry(state).get(model)
    except RegistryError:
        raise SystemExit(f"--model {model!r}: not claude/codex, not a registry version; "
                         "give --endpoint for an OpenAI-compatible server") from None
    serving = version.serving
    if not serving.get("base_url"):
        raise SystemExit(f"{model} has no serving endpoint yet; deploy it first")
    ep = Endpoint(serving["base_url"], serving.get("model", version.model_name), serving.get("api_key_env"))
    return DirectRunner(ep, max_steps=max_steps, name=f"{model}:{ep.model}")


def _runner(args, state: Path) -> AgentRunner:
    return make_runner(args.model, state=state, endpoint=args.endpoint, api_key_env=args.api_key_env,
                       max_steps=args.max_steps)


def _run(args, state: Path, trigger: str) -> Run:
    """The run to record into: ``--run`` when given, else a new one."""
    record = RunRecord(state)
    if getattr(args, "run", None):
        run = record.get(args.run)
        if run.closed:
            raise SystemExit(f"run {run.run_id} is closed")
        return run
    tf = _trainfile(args)
    return record.open(trigger, {"budget": tf.budget.describe(), "eval_cap": tf.eval_cap, "goal": tf.goal})


def cmd_sample(args) -> int:
    state = ensure(args.state)
    tf = _trainfile(args)
    store = Store(state)
    tasks = [store.get(t) for t in args.task] if args.task else list(store)
    if not tasks:
        print("no training tasks; mine and split some first", file=sys.stderr)
        return 1
    runner = _runner(args, state)
    run = _run(args, state, f"sample {runner.name}")
    try:
        attempts = sample(tasks, runner, args.n, state, Ledger(state, tf.budget), run, args.est_usd,
                          workers=args.workers, timeout=args.timeout)
    except BudgetExceeded as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2
    rep = filter_tasks(attempts)
    passed = sum(a.passed is True for a in attempts)
    print(f"run         {run.run_id}")
    print(f"attempts    {len(attempts)} on {len(tasks)} tasks, {passed} verified passes ({runner.name})")
    print(f"tasks       {len(rep.all_passed)} passed {args.n}/{args.n}, {len(rep.mixed)} mixed, "
          f"{len(rep.none_passed)} passed 0/{args.n}")
    return 0


def cmd_data_build(args) -> int:
    state = ensure(args.state)
    attempts = load_attempts(state)
    chosen = shortest(attempts)
    run = RunRecord(state).get(args.run) if args.run else None
    manifest = build_sft(chosen, state, Path(args.out) if args.out else None, run=run)
    print(f"examples    {manifest['examples']} (shortest verified pass per task, of {len(attempts)} attempts)")
    for s in manifest["skipped"]:
        print(f"  skipped {s['task_id']}: {s['reason']}")
    print(f"wrote       {manifest['file']}  sha256 {manifest['sha256'][:16]}")
    return 0


def cmd_eval(args) -> int:
    state = ensure(args.state)
    tf = _trainfile(args)
    runner = _runner(args, state)
    run = _run(args, state, f"eval {runner.name}")
    try:
        scores = Evaluator(state, tf.eval_cap, timeout=args.timeout).request(runner, args.model, run)
    except EvalCapReached as exc:
        print(str(exc), file=sys.stderr)
        return 2
    s = summarise(scores)
    med = f"{s['median_seconds']:.1f}s" if s["median_seconds"] is not None else "-"
    print(f"{runner.name}: {s['passed']}/{s['tasks']} held-out tasks pass ({s['pass_rate']:.0%}), "
          f"median time on passes {med}  [run {run.run_id}]")
    return 0


def cmd_race(args) -> int:
    state = ensure(args.state)
    if args.task:
        held, train = HeldoutStore(state), Store(state)
        task = held.get(args.task) if args.task in held else train.get(args.task)
    else:
        task = newest_heldout(state)
    runners = {"frontier": make_runner(args.frontier, state=state),
               "specialist": _runner(args, state)}
    print(race(task, runners, timeout=args.timeout).render())
    return 0


def cmd_bench_models(args) -> int:
    state = ensure(args.state)
    tf = _trainfile(args)
    runners: list[AgentRunner] = [make_runner(spec, state=state) for spec in args.headless]
    runners += [DirectRunner(Endpoint(url, model, args.api_key_env), max_steps=args.max_steps)
                for url, model in args.direct]
    if not runners:
        print("give at least one --headless or --direct model", file=sys.stderr)
        return 1
    ev = Evaluator(state, tf.eval_cap, timeout=args.timeout)
    print(f"{'model':32}  {'pass':>9}  {'median time':>11}")
    for r in runners:
        s = summarise(ev.baseline(r, r.name))
        med = f"{s['median_seconds']:.1f}s" if s["median_seconds"] is not None else "-"
        print(f"{r.name:32}  {s['passed']:>3}/{s['tasks']:<3} {s['pass_rate']:>4.0%}  {med:>11}")
    return 0


def _model_flags(sp: argparse.ArgumentParser, required: bool = True) -> None:
    sp.add_argument("--model", required=required,
                    help="claude[:model], codex[:model], a registry version (v3), or a model name with --endpoint")
    sp.add_argument("--endpoint", help="OpenAI-compatible base URL (direct runtime), e.g. http://localhost:8000/v1")
    sp.add_argument("--api-key-env", help="name of the environment variable holding the endpoint's key")
    sp.add_argument("--max-steps", type=int, default=60, help="direct runtime: most model turns per attempt")


def _common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--state", default=None, help="folder that holds .opentrain/ (default: here)")
    sp.add_argument("--timeout", type=float, default=900, help="seconds per attempt (default 900)")


def register(sub) -> None:
    sp = sub.add_parser("sample", help="run a model n times on each training task, verify, keep the attempts")
    sp.add_argument("--n", type=int, default=4, help="attempts per task (default 4)")
    sp.add_argument("--task", action="append", default=[], help="only this training task (repeatable)")
    sp.add_argument("--workers", type=int, default=1, help="attempts run at once (default 1)")
    sp.add_argument("--est-usd", type=float, default=0.5,
                    help="dollars reserved per attempt before it starts; settled with the real cost (default 0.5)")
    sp.add_argument("--run", help="record into this run (default: open a new one)")
    _model_flags(sp)
    _common(sp)
    sp.set_defaults(fn=cmd_sample)

    data = sub.add_parser("data", help="training data from verified attempts")
    dsub = data.add_subparsers(dest="sub", required=True)
    db = dsub.add_parser("build", help="SFT JSONL from the shortest verified pass per task")
    db.add_argument("--out", help="output file (default .opentrain/data/train.jsonl)")
    db.add_argument("--run", help="record the data snapshot in this run")
    db.add_argument("--state", default=None, help="folder that holds .opentrain/ (default: here)")
    db.set_defaults(fn=cmd_data_build)

    ev = sub.add_parser("eval", help="score a model on the held-out tasks (counts against the eval cap)")
    ev.add_argument("--run", help="the run this request counts against (default: open a new one)")
    _model_flags(ev)
    _common(ev)
    ev.set_defaults(fn=cmd_eval)

    rc = sub.add_parser("race", help="frontier vs specialist on one task, side by side")
    rc.add_argument("--task", help="task id (default: the newest held-out task)")
    rc.add_argument("--frontier", default="claude", help="the frontier lane: claude[:model] or codex[:model]")
    _model_flags(rc)
    _common(rc)
    rc.set_defaults(fn=cmd_race)

    bm = sub.add_parser("bench-models", help="baseline several models on the held-out tasks")
    bm.add_argument("--headless", action="append", default=[], metavar="SPEC",
                    help="a frontier via the user's CLI: claude[:model] or codex[:model] (repeatable)")
    bm.add_argument("--direct", action="append", default=[], nargs=2, metavar=("BASE_URL", "MODEL"),
                    help="an OpenAI-compatible endpoint and model name (repeatable)")
    bm.add_argument("--api-key-env", help="name of the environment variable holding the endpoints' key")
    bm.add_argument("--max-steps", type=int, default=60, help="direct runtime: most model turns per attempt")
    _common(bm)
    bm.set_defaults(fn=cmd_bench_models)
