"""The evaluator in its own process: ``python -m opentrain.core.evaluate --state S --version V``.

The server starts this for every ``eval_request``. It is the one process that reads held-out
tasks; it scores the version (each attempt runs sandboxed in a fresh worktree), counts the
request against the run's eval cap, saves the scores, and prints only pass/fail and time per task
id as JSON on stdout. Nothing else crosses back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opentrain-evaluate")
    ap.add_argument("--state", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--run")
    a = ap.parse_args(argv)
    from ..mcp import Server, ToolError

    state = Path(a.state)
    try:
        srv = Server(state, "evaluator", a.run)
        out = srv.t_eval_request(a.version)
    except ToolError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    keep = {k: out[k] for k in out if k != "per_task"}
    keep["per_task"] = [{"task_id": t["task_id"], "passed": t["passed"], "seconds": t["seconds"]}
                        for t in out.get("per_task", [])]
    print(json.dumps(keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
