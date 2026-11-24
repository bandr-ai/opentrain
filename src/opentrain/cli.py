"""The ``opentrain`` command (alias ``ot``).

v0.1 bench is what runs today: ``session import`` and ``bench``. Everything reads local session
logs and writes only under ``.opentrain/``; nothing leaves the machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .bench import gitstate, timing, yield_
from .bench import split as split_mod
from .sessions import claude_code
from .sessions.model import Session
from .state import ensure, stamp, write_json


def _parse_date(value: str) -> datetime:
    d = datetime.fromisoformat(value)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _load(args) -> list[Session]:
    sessions, unreadable = claude_code.load_all(Path(args.sessions).expanduser())
    if args.since:
        sessions = [s for s in sessions if s.started_at >= _parse_date(args.since)]
    if args.project:
        sessions = [s for s in sessions if s.cwd and args.project in s.cwd]
    if unreadable and not args.json:
        print(f"  ({len(unreadable)} logs had no timestamped conversation and were left out)", file=sys.stderr)
    return sessions


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def cmd_import(args) -> int:
    d = ensure(args.state)
    sessions = _load(args)
    out = d / "sessions" / "index.jsonl"
    with out.open("w") as fh:
        for s in sorted(sessions, key=lambda s: s.started_at):
            fh.write(json.dumps({**s.summary(), "path": s.path}) + "\n")
    repos = {s.cwd for s in sessions if s.cwd}
    print(f"Found       {len(sessions)} Claude Code sessions in {len(repos)} folders")
    print(f"Wrote       {out}")
    return 0


def known_repos(sessions: list[Session], extra: list[str]) -> list[str]:
    """Where to look for the repo of a session whose worktree was deleted."""
    search = [Path(p).expanduser() for p in extra] + [Path.home() / "conductor" / "repos"]
    for s in sessions:
        root = gitstate.repo_root(s.cwd) if s.cwd else None
        if root:
            search.append(Path(root).parent)
    return gitstate.find_repos(list(dict.fromkeys(search)))


def _yield(sessions: list[Session], extra_repos: list[str] | None = None):
    repos = known_repos(sessions, extra_repos or [])
    verdicts = [yield_.judge(s, repos) for s in sessions]
    return verdicts, yield_.summarise(verdicts)


def _print_yield(summary: dict) -> None:
    n = summary["sessions"]
    print(f"\nTask yield  {summary['verified_candidates']} verified candidates of {n} sessions"
          f"  ({summary['yield_per_100']} per 100)")
    print(f"            {summary['yield_per_coding_session_pct']}% of the {summary['coding_sessions']}"
          " sessions that changed code")
    print(f"            +{summary['repairable']} repairable if the engine runs the tests itself"
          f"  ({summary['yield_with_repair_per_100']} per 100)\n")
    rows = [[f["label"], f["dropped"], f["remaining"]] for f in summary["funnel"]]
    print(_table(rows, ["Skipped because", "Dropped", "Left"]))
    if summary["skip_details"]:
        print("\nDetail")
        for k, v in list(summary["skip_details"].items())[:8]:
            print(f"  {v:>4}  {k}")


def _print_time(b: dict) -> None:
    h = b["hours"]
    print(f"\nTime        model {h['model']}h · tools {h['tool']}h · waiting on you {h['human']}h · idle {h['idle']}h")
    print(f"            model share of working time: {b['model_share_of_working_time_pct']}%"
          f"  (median session {b['median_session_model_share_pct']}%)")
    rep = b["agent_reported"]
    if rep["sessions"]:
        print(f"            Claude Code's own totals agree? model share {rep['model_share_pct']}%"
              f" over {rep['sessions']} sessions")
    print(f"Ceiling     a 10x faster model gives {b['ceiling_speedup_if_model_10x']}x end to end;"
          f" an instant one {b['ceiling_speedup_if_model_infinite']}x")
    if b["top_tool_time_hours"]:
        print("\nTool time by tool (hours, parallel calls each counted)")
        for k, v in list(b["top_tool_time_hours"].items())[:8]:
            print(f"  {v:>6}  {k}")


def cmd_bench(args) -> int:
    d = ensure(args.state)
    sessions = _load(args)
    if not sessions:
        print("No sessions found. Point --sessions at a Claude Code projects folder.", file=sys.stderr)
        return 1
    result: dict = {"at": stamp(), "opentrain": __version__, "source": str(args.sessions)}
    what = args.what

    verdicts = None
    if what in ("all", "yield", "split"):
        verdicts, summary = _yield(sessions, args.repos)
        result["yield"] = summary
        write_json(d / "bench" / f"yield-{result['at']}.json", [v.row() for v in verdicts])
    if what in ("all", "time"):
        result["time"] = timing.breakdown(sessions)
    if what in ("all", "split"):
        chosen = [v for v in verdicts if v.verified or (args.include_repairable and v.repairable)]
        sp = split_mod.split_by_date(
            [(v.session.session_id, v.session.started_at) for v in chosen],
            heldout_fraction=args.heldout,
            cutoff=_parse_date(args.cutoff) if args.cutoff else None,
        )
        write_json(d / "tasks" / "train.json", {"cutoff": sp.cutoff, "ids": sp.train})
        write_json(d / "heldout" / "ids.json", {"cutoff": sp.cutoff, "ids": sp.heldout})
        result["split"] = {"cutoff": sp.cutoff, "train": len(sp.train), "heldout": len(sp.heldout)}

    out = write_json(d / "bench" / f"bench-{result['at']}.json", result)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    print(f"Sessions    {len(sessions)} from {args.sessions}")
    if "yield" in result:
        _print_yield(result["yield"])
    if "time" in result:
        _print_time(result["time"])
    if "split" in result:
        s = result["split"]
        print(f"\nSplit       {s['train']} train · {s['heldout']} held out for the evaluator"
              f" (cutoff {s['cutoff']})")
    print(f"\nSaved       {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opentrain", description="The open harness for training models.")
    p.add_argument("--version", action="version", version=f"opentrain {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--sessions", default=str(claude_code.DEFAULT_ROOT),
                        help="Claude Code projects folder (default: ~/.claude/projects)")
        sp.add_argument("--state", default=None, help="folder that holds .opentrain/ (default: here)")
        sp.add_argument("--since", help="only sessions that started on or after this date (YYYY-MM-DD)")
        sp.add_argument("--project", help="only sessions whose folder path contains this text")
        sp.add_argument("--json", action="store_true", help="print machine-readable output")
        sp.add_argument("--repos", action="append", default=[],
                        help="a folder of git clones, to find the repo of a deleted worktree (repeatable)")

    sess = sub.add_parser("session", help="coding-agent sessions")
    ssub = sess.add_subparsers(dest="sub", required=True)
    imp = ssub.add_parser("import", help="find and index local Claude Code sessions")
    common(imp)
    imp.set_defaults(fn=cmd_import)

    bench = sub.add_parser("bench", help="v0.1 bench: task yield, time breakdown, held-out split")
    bench.add_argument("what", nargs="?", default="all", choices=["all", "yield", "time", "split"])
    bench.add_argument("--heldout", type=float, default=0.3, help="share of candidates held out (default 0.3)")
    bench.add_argument("--cutoff", help="hold out every candidate that started on or after this date")
    bench.add_argument("--include-repairable", action="store_true",
                       help="put repairable sessions in the split too")
    common(bench)
    bench.set_defaults(fn=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
