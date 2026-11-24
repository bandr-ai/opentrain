"""Data commands: ``mine``, ``lint``, ``export``, ``task ls`` and ``task show``.

``mine`` is the whole data path in one step: judge each session (v0.1 yield), build and certify
a task from each one that qualifies, then refresh the held-out split by date with the
Trainfile's held-out share. Every session ends up counted: accepted, or skipped with a reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..state import ensure, stamp, write_json


def _common(p, sessions: bool = False) -> None:
    p.add_argument("--state", default=None, help="folder that holds .opentrain/ (default: here)")
    p.add_argument("--json", action="store_true", help="print machine-readable output")
    if sessions:
        from ..sessions import claude_code

        p.add_argument("--sessions", default=str(claude_code.DEFAULT_ROOT),
                       help="Claude Code projects folder (default: ~/.claude/projects)")
        p.add_argument("--since", help="only sessions that started on or after this date (YYYY-MM-DD)")
        p.add_argument("--project", help="only sessions whose folder path contains this text")
        p.add_argument("--repos", action="append", default=[],
                       help="a folder of git clones, to find the repo of a deleted worktree (repeatable)")


def cmd_mine(args) -> int:
    from .. import trainfile
    from ..bench import yield_
    from ..cli import known_repos
    from ..cli import load_sessions as _load
    from ..mine import mine_all
    from ..tasks import split_store

    d = ensure(args.state)
    sessions = _load(args)
    if not sessions:
        print("No sessions found. Point --sessions at a Claude Code projects folder.", file=sys.stderr)
        return 1
    tf = trainfile.load(trainfile.find(Path(args.state) if args.state else None))
    repos = known_repos(sessions, args.repos)
    verdicts = [yield_.judge(s, repos) for s in sessions]
    report = mine_all(verdicts, d, include_repairable=args.include_repairable, timeout=args.timeout)
    report["split"] = split_store(d, tf.heldout_fraction)
    out = write_json(d / "data" / f"mine-{stamp()}.json", report)
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, default=str))
        return 0
    sp = report["split"]
    not_taken = report["skipped"] + report["rejected"]
    why = ", ".join(f"{n} {r}" for r, n in report["reasons"].items())
    print(f"{sp['train'] + sp['heldout']} verified tasks, {sp['heldout']} held out for the evaluator"
          f" / {not_taken} skipped" + (f" ({why})" if why else ""))
    print(f"            {report['accepted']} new this run, {report['kept']} already in the store")
    print(f"Saved       {out}")
    return 0


def cmd_lint(args) -> int:
    from ..certify import lint

    report = lint(ensure(args.state), timeout=args.timeout)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Checked     {report['checked']} tasks in {report['seconds']}s: {report['ok']} verifiers hold")
        for r in report["broken"]:
            extra = f"  ({r['detail']})" if r.get("detail") else ""
            print(f"  broken    {r['task_id']} [{r['store']}] {r['reason']}{extra}")
    return 1 if report["broken"] else 0


def cmd_export(args) -> int:
    from ..export import export

    m = export(ensure(args.state), Path(args.out), include_heldout=args.include_heldout)
    if args.json:
        print(json.dumps(m, indent=2))
        return 0
    c = m["counts"]
    print(f"Exported    {c['tasks']} tasks, {c['train_records']} training records to {args.out}")
    if args.include_heldout:
        print(f"            {c['heldout_exported']} held-out tasks included: keep them away from any trainer")
    elif c["heldout_withheld"]:
        print(f"            {c['heldout_withheld']} held-out tasks left out (--include-heldout to add them)")
    return 0


def cmd_task_ls(args) -> int:
    from ..tasks import HeldoutStore, Store

    d = ensure(args.state)
    store = Store(d)
    rows = []
    for t in store:
        c = store.controls(t.task_id)
        rows.append({"task_id": t.task_id, "family": t.family, "created_at": t.created_at,
                     "controls": c.reason() if c else "not certified"})
    held = len(HeldoutStore(d).ids())
    if args.json:
        print(json.dumps({"tasks": rows, "heldout": held}, indent=2))
        return 0
    for r in rows:
        print(f"{r['task_id']:<40}  {r['family'] or '-':<16}  {(r['created_at'] or '-')[:10]}  {r['controls']}")
    print(f"\n{len(rows)} training tasks; {held} held out (only the evaluator reads those)")
    return 0


def cmd_task_show(args) -> int:
    from ..tasks import HeldoutStore, Store

    d = ensure(args.state)
    store = Store(d)
    if args.task_id in HeldoutStore(d):
        print(f"{args.task_id} is held out: only the evaluator reads it.", file=sys.stderr)
        return 1
    if args.task_id not in store:
        print(f"No task {args.task_id}. See `opentrain task ls`.", file=sys.stderr)
        return 1
    t = store.get(args.task_id)
    c = store.controls(t.task_id)
    if args.json:
        from dataclasses import asdict

        print(json.dumps({"task": asdict(t), "controls": asdict(c) if c else None}, indent=2, default=str))
        return 0
    print(f"Task        {t.task_id}  ({store.folder(t.task_id)})")
    print(f"Repo        {t.repo} {t.base_commit[:10]}..{t.reference_commit[:10]}")
    print(f"Family      {t.family or '-'}   created {t.created_at or '-'}   session {t.source_session or '-'}")
    print(f"Tests       {t.test_command}")
    for f in t.test_files:
        print(f"            {f}")
    if c:
        print(f"Controls    {c.reason()}  ({c.seconds}s)  {c.detail}")
    print(f"\n{t.instruction}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("mine", help="build verified tasks from sessions (cl/mine), then split held-out")
    p.add_argument("--include-repairable", action="store_true",
                   help="also mine sessions whose tests did not end green (the miner runs them itself)")
    p.add_argument("--timeout", type=float, default=900, help="seconds per verifier run (default 900)")
    _common(p, sessions=True)
    p.set_defaults(fn=cmd_mine)

    p = sub.add_parser("lint", help="re-run every stored task's verifier controls; list the broken ones")
    p.add_argument("--timeout", type=float, default=900, help="seconds per verifier run (default 900)")
    _common(p)
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("export", help="write tasks and training data in open formats (Harbor, JSONL)")
    p.add_argument("out", help="folder to write (empty, or an earlier export)")
    p.add_argument("--include-heldout", action="store_true",
                   help="also export held-out tasks (only for an evaluator you control; see docs)")
    _common(p)
    p.set_defaults(fn=cmd_export)

    task = sub.add_parser("task", help="the task store")
    tsub = task.add_subparsers(dest="sub", required=True)
    p = tsub.add_parser("ls", help="list training tasks")
    _common(p)
    p.set_defaults(fn=cmd_task_ls)
    p = tsub.add_parser("show", help="show one training task")
    p.add_argument("task_id")
    _common(p)
    p.set_defaults(fn=cmd_task_show)
