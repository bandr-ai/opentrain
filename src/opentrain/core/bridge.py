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


class Bridge:
    def __init__(self) -> None:
        env = os.environ
        self.client = Client(env["OPENTRAIN_SERVER"], env["OPENTRAIN_SERVER_PASSWORD"], env["OPENTRAIN_DIRECTORY"])
        self.session = env["OPENTRAIN_SESSION"]

    def tools(self) -> list[dict[str, Any]]:
        specs = self.client.get(f"/api/session/{self.session}/tools")
        return [{"name": t["name"], "description": t["description"], "inputSchema": t["schema"]} for t in specs]

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self.client.request("POST", f"/api/session/{self.session}/tool-call",
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
                               "capabilities": {"tools": {}}, "serverInfo": {"name": "opentrain", "version": __version__}}
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
