"""External MCP servers: their tools join the registry as ``<server>_<tool>``.

``opentrain.jsonc``::

    "mcp": {
      "github": {"type": "local", "command": ["github-mcp-server", "stdio"],
                 "env": {"GITHUB_TOKEN": "{env:GITHUB_TOKEN}"}},
      "docs":   {"type": "local", "command": ["npx", "-y", "some-mcp"], "disabled": true}
    }

Local (stdio) servers are supported. A server's tools are gated by the permission action named
after the server, so ``"github": "ask"`` in permissions makes every GitHub call ask.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from typing import Any

from .tool import Tool, ToolError, ToolResult

log = logging.getLogger("opentrain.mcp")
PROTOCOL = "2025-06-18"


class McpServer:
    def __init__(self, name: str, command: list[str], env: dict[str, str] | None = None, cwd: str | None = None,
                 timeout: float = 120.0):
        self.name, self.command, self.timeout = name, command, timeout
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, cwd=cwd,
                                     env={**os.environ, **(env or {})}, bufsize=1)
        self._lock = threading.Lock()
        self._id = 0
        self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                    "clientInfo": {"name": "opentrain", "version": "0"}})
        self.notify("notifications/initialized")

    def notify(self, method: str, params: dict | None = None) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None) -> Any:
        with self._lock:
            self._id += 1
            mid = self._id
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": mid, "method": method,
                                              "params": params or {}}) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise ToolError(f"MCP server {self.name} exited")
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("id") != mid:
                    continue  # a notification or a stale reply
                if "error" in msg:
                    raise ToolError(f"{self.name}: {msg['error'].get('message')}")
                return msg.get("result")

    def tools(self) -> list[Tool]:
        listed = self.request("tools/list").get("tools", [])
        out = []
        for t in listed:
            remote = t["name"]

            def call(ctx: Any, _remote: str = remote, **args: Any) -> ToolResult:
                res = self.request("tools/call", {"name": _remote, "arguments": args})
                text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
                if res.get("isError"):
                    raise ToolError(text or "the MCP tool failed")
                return ToolResult(text or json.dumps(res)[:20000])

            schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            out.append(Tool(f"{self.name}_{remote}"[:64], t.get("description") or remote, schema, call,
                            permission=lambda a, _n=self.name, _r=remote: [(_n, _r)], group="mcp"))
        return out

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(5)
        except Exception:
            self.proc.kill()


class McpServers:
    def __init__(self, project: Any):
        self.project = project
        self.servers: dict[str, McpServer] = {}
        self.status: dict[str, str] = {}

    def start(self) -> None:
        from . import config

        spec = (config.load(self.project.directory).get("mcp") or {})
        for name, s in spec.items():
            if not isinstance(s, dict) or s.get("disabled"):
                self.status[name] = "disabled"
                continue
            if s.get("type", "local") != "local" or not s.get("command"):
                self.status[name] = "unsupported: only local (stdio) servers with a command"
                continue
            cmd = s["command"] if isinstance(s["command"], list) else [s["command"]]
            try:
                srv = McpServer(name, cmd, s.get("env"), str(self.project.directory))
                tools = srv.tools()
            except Exception as exc:
                self.status[name] = f"failed: {exc}"
                log.warning("MCP server %s failed: %s", name, exc)
                continue
            self.servers[name] = srv
            self.project.services["tools"].add(*tools)
            self.status[name] = f"connected ({len(tools)} tools)"

    def close(self) -> None:
        for s in self.servers.values():
            s.close()
