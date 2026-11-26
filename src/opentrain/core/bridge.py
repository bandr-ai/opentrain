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
trains a small open model when that is worth it, and ships it only if it beats the baseline.

Typical path: datasets / dataset_import (a file, hf= or kaggle=) -> dataset_edit (grader; the import
suggests one) -> dataset_eval a strong model and a small one (modal/<hf id> runs open models on the
user's Modal account) -> trainer_set (where and what to train) -> dataset_recipe (settings, the plan
of methods sft / rejection / opd / rl, and the cost) -> dataset_label (distil missing answers) ->
dataset_train -> dataset_eval the new version -> dataset_promote. Jobs run on the OpenTrain server and return at once: wait for one with job_wait
(it blocks until the job ends), or look with jobs / job_status. The user watches the same project in the OpenTrain web app (opentrain web).
Ask the user before any call that spends money, uploads data, trains or changes what is live.
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
