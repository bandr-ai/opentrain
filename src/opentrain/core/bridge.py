"""The MCP bridge: serves a session's OpenTrain tools to an agent provider (Claude Code, Codex).

Started by the provider as ``python -m opentrain.core.bridge`` with the server url, password,
session and directory in its environment. ``tools/list`` asks the server which tools the
session's agent may see; ``tools/call`` posts the call to the server, which checks permission,
runs it, records it in the session and returns the output. The bridge holds no state and makes
no decisions.

Protocol: JSON-RPC 2.0, one message per line (MCP stdio).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from typing import Any

from .. import __version__
from ..client import Client, ClientError

PROTOCOL = "2025-06-18"


INSTRUCTIONS = """OpenTrain post-trains models: it measures models on a dataset's sealed held-out rows,
trains a small open model when that is worth it, and ships a version only if it beats the baseline.
Do only what the user's request needs: an eval question ends at the eval.

What the tools are for:
- data: datasets, dataset_show; dataset_search / dataset_peek (Hugging Face, Kaggle);
  dataset_import (a file, hf= or kaggle=; templates reshape columns; heldout=1 for an eval-only
  benchmark); dataset_add (rows you write: heldout=1 for eval rows you write first, 0 for
  training rows after them); dataset_label (distil answers from a teacher).
- grading: dataset_edit grader= (exact, contains, regex, json, number, judge, command, or python:
  the user's reward(prompt, answer, reference), which also drives RL); dataset_grade checks it on
  answers you write.
- measuring: dataset_eval (held-out: the score that counts; split=train: a quick look you may read,
  with evals eval_id= for the answers); model_try for one prompt. modal/<hf id> and runpod/<hf id>
  run open models on the user's own GPU account.
- training: trainer_set (where and what), dataset_recipe (task kind, a plan of methods sft / opd /
  rl / rejection, settings, and the price), dataset_train, then dataset_eval the new version and
  dataset_promote (the gate decides).
- coding work (a small model that does the user's own coding): traces_scan / traces_import (every
  coding agent's logs on this machine, read locally), moments + themes_set / themes_choose (what
  went wrong, which the user picks), tasks_mine and tasks_from_history (verified tasks from sessions
  and the repos' commits), sample_run, data_build, job_start, eval_request (a version, or base),
  version_propose. Playbook: goal/coding.
- skills (the skill tool) are playbooks to adapt: goal/coding, data/diagnose, data/history, data/find (a use case, no data), data/tasks
  (the user's own data or documents), data/triage (is this data good enough), train/env (a reward
  for RL).

Jobs run on the OpenTrain server and return at once; job_wait waits up to its timeout (call it
again if the job is still running), jobs / job_status / job_logs look. The user watches the same
project in the web app (opentrain web). Ask the user before any call that spends money, uploads
their data, trains or changes what is live; say the price (dataset_recipe gives it for training).
The server enforces the budget and never lets anyone read held-out rows."""


class Bridge:
    def __init__(self, outside: bool = False) -> None:
        env = os.environ
        self.outside = outside
        if outside:  # an agent outside OpenTrain, in this directory, against the user's server
            from ..client import connect

            self.client = connect(env.get("OPENTRAIN_DIRECTORY") or os.getcwd())
            self.session = ""
        else:
            self.client = Client(env["OPENTRAIN_SERVER"], env["OPENTRAIN_SERVER_PASSWORD"], env["OPENTRAIN_DIRECTORY"])
            self.session = env["OPENTRAIN_SESSION"]

    def tools(self) -> list[dict[str, Any]]:
        specs = self.client.get("/api/outside/tools" if self.outside else f"/api/session/{self.session}/tools")
        return [{"name": t["name"], "description": t["description"], "inputSchema": t["schema"]} for t in specs]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            path = "/api/outside/call" if self.outside else f"/api/session/{self.session}/tool-call"
            r = self.client.request("POST", path,
                                    {"name": name, "input": args or {}, "call_id": f"mcp_{uuid.uuid4().hex[:16]}"},
                                    timeout=7 * 24 * 3600)
        except ClientError as exc:
            return {"content": [{"type": "text", "text": f"error: {exc.message}"}], "isError": True}
        return {"content": [{"type": "text", "text": r.get("output") or "(no output)"}], "isError": bool(r.get("is_error"))}

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            return None
        try:
            if method == "initialize":
                result: Any = {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                               "capabilities": {"tools": {}}, "serverInfo": {"name": "opentrain", "version": __version__},
                               **({"instructions": INSTRUCTIONS} if self.outside else {})}
            elif method == "tools/list":
                result = {"tools": self.tools()}
            elif method == "tools/call":
                p = msg.get("params", {})
                result = self.call(p.get("name", ""), p.get("arguments") or {})
            elif method == "ping":
                result = {}
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}
        except Exception as exc:  # the agent sees the error; the bridge stays up
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self) -> None:
        lock = threading.Lock()

        def reply(msg: dict[str, Any]) -> None:
            out = self.handle(msg)
            if out is not None:
                with lock:
                    sys.stdout.write(json.dumps(out) + "\n")
                    sys.stdout.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("method") == "tools/call":  # calls may wait on a permission prompt: run them side by side
                threading.Thread(target=reply, args=(msg,), daemon=True).start()
            else:
                reply(msg)


if __name__ == "__main__":
    Bridge().serve()
