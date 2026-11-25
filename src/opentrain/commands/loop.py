"""The loop's commands: ``init``, ``train``, ``run``, ``status``, ``watch``, ``fix``, ``diff``, ``log``,
``revert``, ``approve``, ``skill``, ``mcp``.

These are the verbs a user types day to day. Each one is a thin shell over the engine, the run
record and the registry, so the web UI and the train tools see exactly the same state.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import engine, trainfile
from ..evaluator import Evaluator, summarise
from ..record import RecordError, RunRecord
from ..registry import Registry, RegistryError
from ..sessions import claude_code
from ..state import ensure
from ..tasks import HeldoutStore, Store

PACKAGED_SKILLS = Path(__file__).resolve().parents[1] / "skills"  # installed wheel
REPO_SKILLS = Path(__file__).resolve().parents[3] / "skills"  # a source checkout


def _tf(args) -> trainfile.Trainfile:
    try:
        return trainfile.load(trainfile.find(Path(args.state).resolve() if args.state else None))
    except trainfile.TrainfileError as exc:
        raise SystemExit(f"Trainfile: {exc}") from None


def _ask(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/n] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _state_arg(p) -> None:
    p.add_argument("--state", default=None, help="folder holding .opentrain/ (default: here)")


# ------------------------------------------------------------------ init / status


def cmd_init(args) -> int:
    root = Path(args.state or ".").resolve()
    path = root / "Trainfile"
    if path.exists() and not args.force:
        print(f"{path} exists; edit it, or pass --force to start over")
        return 1
    path.write_text(trainfile.DEFAULT)
    ensure(root)
    print(f"Wrote       {path}")
    print("            goal speed, $20 a week, promote only if no held-out task regresses")
    print("Next        opentrain bench    (how many of your sessions can become tasks)")
    print("            opentrain train    (the first run)")
    return 0


def cmd_status(args) -> int:
    state = ensure(args.state)
    tf = _tf(args)
    from ..budget import Ledger

    reg = Registry(state)
    cur = reg.current()
    led = Ledger(state, tf.budget)
    last = RunRecord(state).latest()
    print(f"Trainfile   goal/{tf.goal} · {tf.budget.describe()} · runtime {tf.runtime} · trainer {tf.trainer}"
          f" · base {tf.base_model}")
    print(f"Tasks       {len(Store(state).ids())} training · {len(HeldoutStore(state).ids())} held out")
    if cur:
        s = cur.scores or {}
        rate = f"{s['pass_rate']:.0%} pass" if s.get("pass_rate") is not None else "unscored"
        med = f", {s['median_seconds']:.0f}s median" if s.get("median_seconds") else ""
        print(f"Live        opentrain/{cur.version} ({cur.backend}) {rate}{med}")
    else:
        print("Live        nothing yet")
    print(f"Spend       ${led.committed():.2f} of {tf.budget.describe()}")
    if last:
        ev = last.events()
        gate = next((e["data"] for e in reversed(ev) if e["kind"] == "gate"), None)
        note = next((e["data"].get("text") for e in reversed(ev) if e["kind"] == "note"), None)
        state_word = "closed" if last.closed else "running"
        print(f"Last run    {last.run_id} ({state_word})"
              + (f": {'promoted ' + gate['version'] if gate['promote'] else 'not promoted'}" if gate else ""))
        if note:
            print(f"            {note}")
    return 0


def cmd_home(args) -> int:
    """``opentrain`` with no command: the first run if nothing exists yet, else the status."""
    root = Path(args.state or ".").resolve()
    if not (root / ".opentrain").exists() and not trainfile.find(root):
        print("OpenTrain   the open harness for training models\n")
        if shutil.which("claude"):
            print("Engine      claude -p, your own login")
        sessions = list(claude_code.discover())
        print(f"Found       {len(sessions)} Claude Code session logs\n")
        print("Start with  opentrain init     write a Trainfile (goal, budget, gate)")
        print("            opentrain bench    see how many sessions can become verified tasks")
        print("            opentrain train    mine tasks, train, evaluate, gate")
        return 0
    return cmd_status(args)


# ------------------------------------------------------------------ train / run


def cmd_train(args) -> int:
    state = ensure(args.state)
    tf = _tf(args)
    if args.goal:
        import dataclasses

        tf = dataclasses.replace(tf, goal=args.goal)
    print(f"$ opentrain train   goal/{tf.goal}\n")
    if shutil.which(tf.runtime):
        print(f"Engine      {tf.runtime}, your own login")
    res = engine.train(
        state, tf, sessions_root=Path(args.sessions).expanduser(), trigger=args.trigger,
        ask=None if args.yes or args.unattended else _ask, n_samples=args.n, workers=args.workers,
        include_repairable=args.include_repairable, skip_mine=args.skip_mine,
    )
    if res.promoted:
        return 0
    return 2 if res.version else 1  # 2: trained but the gate said no; 1: stopped before training


def cmd_run(args) -> int:
    state = ensure(args.state)
    tf = _tf(args)
    try:
        out = engine.agent(state, tf, " ".join(args.instruction), role=args.role, model=args.model)
    except engine.EngineStop as exc:
        print(f"Stopped     {exc}", file=sys.stderr)
        return 1
    return 1 if out.get("error") else 0


def cmd_approve(args) -> int:
    state = ensure(args.state)
    engine._approve(state, args.trainer)
    print(f"Approved    unattended runs may upload redacted tasks to {args.trainer}")
    return 0


# ------------------------------------------------------------------ watch


def _new_sessions_since(when: datetime | None) -> int:
    n = 0
    for p in claude_code.discover():
        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        if when is None or mtime > when:
            n += 1
    return n


def _last_run_at(state: Path) -> datetime | None:
    last = RunRecord(state).latest()
    if not last:
        return None
    ev = last.events()
    return datetime.fromisoformat(ev[0]["at"]) if ev else None


def live_health(state: Path, version: str, hours: float = 24) -> dict:
    """How the live version is doing for real: errors and handoffs per request, from the serving log."""
    path = state / "serving" / "requests.jsonl"
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    total = bad = handoffs = 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("version") != version or datetime.fromisoformat(e["at"]) < since:
                continue
            total += 1
            bad += int(e.get("status", 200) >= 500)
            handoffs += int(bool(e.get("handoff")))
    return {"requests": total, "error_rate": bad / total if total else 0.0,
            "handoff_rate": handoffs / total if total else 0.0}


def check_rollback(state: Path, min_requests: int = 20, max_error: float = 0.2, max_handoff: float = 0.5) -> str | None:
    """Revert the live version when live results drop (plan v0.3: automatic rollback). Returns why, or None."""
    reg = Registry(state)
    cur = reg.current()
    if cur is None:
        return None
    h = live_health(state, cur.version)
    if h["requests"] < min_requests:
        return None
    why = None
    if h["error_rate"] > max_error:
        why = f"{cur.version}: {h['error_rate']:.0%} of {h['requests']} live requests failed"
    elif h["handoff_rate"] > max_handoff:
        why = f"{cur.version}: {h['handoff_rate']:.0%} of {h['requests']} live requests needed a handoff"
    if why:
        reg.revert(why)  # also clears any serving pin, so opentrain/current follows the rollback
    return why


def watch_once(state: Path, tf: trainfile.Trainfile, sessions_root: Path, say=print, now: datetime | None = None) -> str:
    """One tick of the unattended loop. Returns what it did."""
    now = now or datetime.now(timezone.utc)
    why = check_rollback(state)
    if why:
        say(f"Reverted    {why}")
        return "reverted"
    w = tf.watch or trainfile.Watch(new_sessions=30, nightly=True)
    last = _last_run_at(state)
    new = _new_sessions_since(last)
    flags = state / "flags.jsonl"
    flagged = flags.exists() and last is not None and datetime.fromtimestamp(flags.stat().st_mtime, timezone.utc) > last
    trigger = None
    if w.new_sessions and new >= w.new_sessions:
        trigger = f"{new} new sessions"
    elif w.flagged and flagged:
        trigger = "a flagged session"
    elif w.nightly and (last is None or now - last >= timedelta(hours=23)) and 1 <= now.astimezone().hour <= 5:
        trigger = "nightly"
    if trigger is None:
        return "idle"
    say(f"Trigger     {trigger}")
    res = engine.train(state, tf, sessions_root=sessions_root, trigger=trigger, say=say, ask=None)
    return "promoted" if res.promoted else "ran"


def cmd_watch(args) -> int:
    state = ensure(args.state)
    tf = _tf(args)
    sessions = Path(args.sessions).expanduser()
    if args.once:
        print(watch_once(state, tf, sessions))
        return 0
    print(f"Watching    every {args.every}s; triggers from the Trainfile's watch line. Ctrl-C stops.")
    while True:
        try:
            watch_once(state, tf, sessions)
            time.sleep(args.every)
        except KeyboardInterrupt:
            return 0


# ------------------------------------------------------------------ fix


def cmd_fix(args) -> int:
    """File a bug against the model: flag the session, then run goal/fix through the engine."""
    state = ensure(args.state)
    tf = _tf(args)
    paths = sorted(claude_code.discover(), key=lambda p: p.stat().st_mtime)
    if not paths:
        print("no sessions found", file=sys.stderr)
        return 1
    if args.session == "latest":
        path = paths[-1]
    else:
        matches = [p for p in paths if p.stem.startswith(args.session)]
        if len(matches) != 1:
            print(f"no single session matches {args.session!r}", file=sys.stderr)
            return 1
        path = matches[0]
    message = args.message or (input("> ") if sys.stdin.isatty() else "")
    flag = {"session": path.stem, "path": str(path), "message": message,
            "at": datetime.now(timezone.utc).isoformat()}
    with (state / "flags.jsonl").open("a") as fh:
        fh.write(json.dumps(flag) + "\n")
    print(f"Flagged     session {path.stem[:8]}: {message or '(no message)'}")
    if args.flag_only:
        return 0
    instruction = (
        f"goal/fix. The user filed a bug against their model on session {path.stem}: {message!r}. "
        "Follow skills/goal-fix/SKILL.md: reproduce the failure from that session, look for the same "
        "pattern in earlier sessions (tasks_list), build related training tasks (data/expand), train, "
        "eval, and propose. The flagged failure must pass and nothing else may regress."
    )
    try:
        engine.agent(state, tf, instruction)
    except engine.EngineStop as exc:
        print(f"Stopped     {exc}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------ diff / log / revert


def cmd_diff(args) -> int:
    state = ensure(args.state)
    reg = Registry(state)
    ev = Evaluator(state, 1)
    versions = [v.version for v in reg.versions()]
    b = args.b or (versions[-1] if versions else None)
    a = args.a or (versions[-2] if len(versions) > 1 else engine.baseline_name(_tf(args), reg))
    sa, sb = ev.scores_for(a), ev.scores_for(b) if b else None
    if not sa or not sb:
        print(f"need held-out scores for both {a} and {b}; run opentrain eval", file=sys.stderr)
        return 1
    da, db = {s.task_id: s for s in sa}, {s.task_id: s for s in sb}
    flipped = []
    for t in sorted(set(da) & set(db)):
        if da[t].passed != db[t].passed:
            flipped.append((t, "fixed" if db[t].passed else "REGRESSED", da[t].seconds, db[t].seconds))
    xa, xb = summarise(sa), summarise(sb)
    print(f"{a} -> {b}   pass {xa['pass_rate']:.0%} -> {xb['pass_rate']:.0%}   "
          f"median {xa['median_seconds']}s -> {xb['median_seconds']}s")
    for t, what, s1, s2 in flipped:
        print(f"  {what:9}  {t}  ({s1}s -> {s2}s)")
    if not flipped:
        print("  no task flipped")
    return 0


def cmd_log(args) -> int:
    state = ensure(args.state)
    rr = RunRecord(state)
    if args.run:
        try:
            run = rr.get(args.run)
        except RecordError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"run {run.run_id}  record {'intact' if run.verify() else 'TAMPERED'}")
        for e in run.events():
            d = e["data"]
            brief = ", ".join(f"{k}={v}" for k, v in d.items() if not isinstance(v, (list, dict)))[:150]
            print(f"  {e['at'][:19]}  {e['kind']:10} {brief}")
        return 0
    for i, run in enumerate(rr.runs()):
        if i >= args.n:
            break
        ev = run.events()
        gate = next((e["data"] for e in reversed(ev) if e["kind"] == "gate"), None)
        outcome = ("promoted " + gate["version"]) if gate and gate.get("promote") else (
            "rejected" if gate else ("running" if not run.closed else "stopped"))
        trig = ev[0]["data"].get("trigger", "") if ev else ""
        print(f"{run.run_id}  ${run.spent():6.2f}  {outcome:14}  {trig}")
    return 0


def cmd_revert(args) -> int:
    state = ensure(args.state)
    try:
        prev = Registry(state).revert(args.why)
    except RegistryError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Reverted    live is now {('opentrain/' + prev.version) if prev else 'nothing (base model)'}")
    return 0


# ------------------------------------------------------------------ skills


def skill_dirs(state: Path) -> list[Path]:
    shipped = REPO_SKILLS if REPO_SKILLS.is_dir() else PACKAGED_SKILLS
    return [d for d in (shipped, state / "skills") if d.is_dir()]


def cmd_skill_ls(args) -> int:
    state = ensure(args.state)
    for base in skill_dirs(state):
        for f in sorted(base.glob("*/SKILL.md")):
            name, desc = f.parent.name, ""
            for line in f.read_text().splitlines()[1:8]:
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip()
                if line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip().strip('"')
            print(f"{name:18} {desc[:90]}")
    return 0


def cmd_skill_add(args) -> int:
    state = ensure(args.state)
    dest_root = state / "skills"
    dest_root.mkdir(exist_ok=True)
    src = args.source
    if src.startswith(("https://", "git@")):
        tmp = dest_root / ".clone"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", src, str(tmp)], check=True)
        src_path = tmp
    else:
        src_path = Path(src).expanduser()
    found = [src_path] if (src_path / "SKILL.md").is_file() else [p.parent for p in src_path.glob("*/SKILL.md")]
    if not found:
        print(f"no SKILL.md in {src}", file=sys.stderr)
        return 1
    for f in found:
        shutil.copytree(f, dest_root / f.name, dirs_exist_ok=True)
        print(f"Installed   {f.name}  (read it: {dest_root / f.name / 'SKILL.md'})")
    shutil.rmtree(dest_root / ".clone", ignore_errors=True)
    return 0


def cmd_mcp(args) -> int:
    from ..mcp import Server

    Server(ensure(args.state), args.role, args.run).serve()
    return 0


# ------------------------------------------------------------------ register


def register(sub) -> None:
    p = sub.add_parser("init", help="write a Trainfile here (goal, budget, gate)")
    p.add_argument("--force", action="store_true")
    _state_arg(p)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("status", help="what is live, what it cost, how the last run went")
    _state_arg(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("train", help="run the goal: mine tasks, sample, train, evaluate, gate")
    p.add_argument("--goal", choices=trainfile.GOALS, help="override the Trainfile's goal for this run")
    p.add_argument("--n", type=int, default=4, help="attempts per training task (default 4)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--yes", "-y", action="store_true", help="approve the estimate without asking")
    p.add_argument("--unattended", action="store_true", help="never ask; the Trainfile decides")
    p.add_argument("--include-repairable", action="store_true")
    p.add_argument("--skip-mine", action="store_true", help="use the task store as it is")
    p.add_argument("--trigger", default="manual")
    p.add_argument("--sessions", default=str(claude_code.DEFAULT_ROOT))
    _state_arg(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("run", help='one instruction to the engine, e.g. opentrain run "make it faster on migrations"')
    p.add_argument("instruction", nargs="+")
    p.add_argument("--role", choices=["trainer", "evaluator"], default="trainer")
    p.add_argument("--model", help="model for the engine's CLI (default: its own default)")
    _state_arg(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("approve", help="allow unattended runs to upload redacted tasks to a trainer")
    p.add_argument("trainer")
    _state_arg(p)
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("watch", help="the unattended loop: new sessions, flags or nightly trigger a run")
    p.add_argument("--once", action="store_true", help="one tick (for cron or launchd)")
    p.add_argument("--every", type=int, default=600, help="seconds between ticks")
    p.add_argument("--sessions", default=str(claude_code.DEFAULT_ROOT))
    _state_arg(p)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("fix", help="file a bug against the model from a session that went badly")
    p.add_argument("--session", default="latest")
    p.add_argument("-m", "--message")
    p.add_argument("--flag-only", action="store_true", help="record the flag; the next run picks it up")
    _state_arg(p)
    p.set_defaults(fn=cmd_fix)

    p = sub.add_parser("diff", help="which held-out tasks flipped between two versions")
    p.add_argument("a", nargs="?")
    p.add_argument("b", nargs="?")
    _state_arg(p)
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("log", help="the run record")
    p.add_argument("run", nargs="?")
    p.add_argument("-n", type=int, default=20)
    _state_arg(p)
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("revert", help="put the previous version back live")
    p.add_argument("--why", default="manual revert")
    _state_arg(p)
    p.set_defaults(fn=cmd_revert)

    sk = sub.add_parser("skill", help="recipes: every recipe is a skill")
    ssub = sk.add_subparsers(dest="skill_cmd", required=True)
    p = ssub.add_parser("ls", help="installed recipes")
    _state_arg(p)
    p.set_defaults(fn=cmd_skill_ls)
    p = ssub.add_parser("add", help="install a recipe from a folder or git URL")
    p.add_argument("source")
    _state_arg(p)
    p.set_defaults(fn=cmd_skill_add)

    p = sub.add_parser("mcp", help="serve the train tools over MCP stdio (used by the engine)")
    p.add_argument("--role", choices=["trainer", "evaluator"], default="trainer")
    p.add_argument("--run")
    _state_arg(p)
    p.set_defaults(fn=cmd_mcp)
