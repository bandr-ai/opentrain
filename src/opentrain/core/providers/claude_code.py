"""Claude Code as the engine, under the user's own login (``claude-code/<model>``).

Each turn runs ``claude -p`` once with: OpenTrain's system prompt for the agent, no built-in
tools (``--tools ""``), and OpenTrain's tools served to it by the MCP bridge
(``python -m opentrain.core.bridge``). Every tool call therefore comes back to this server, where
it is permission-checked, budgeted and recorded as a part of the current message, exactly as with
a direct API provider. The conversation continues across turns with ``--resume``.

Text and thinking stream from ``--include-partial-messages``. Steering input waits for the end of
the ``claude -p`` turn. User settings, hooks and plugins are not loaded (``--setting-sources ""``),
so a user's Claude Code configuration cannot change what the engine may do.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import uuid
from typing import TYPE_CHECKING, Any

from .. import llm
from ..context import system_prompt

if TYPE_CHECKING:
    from ..runner import Runner, Turn

SERVER_NAME = "opentrain"
INIT_NOTE = ("Your tools are served by OpenTrain over MCP (names start with mcp__opentrain__). Use them for "
             "everything: there is no other shell or file access.")


def _billing(init: dict[str, Any]) -> str:
    return "subscription" if init.get("apiKeySource") == "none" else "api"


class ClaudeCodeProvider:
    id = "claude-code"
    kind = "agent"

    def __init__(self, model: str):
        self.model = model

    def _mcp_config(self, runner: "Runner", sid: str, agent: Any) -> str:
        hub = runner.project.services["hub"]
        cfg = {"mcpServers": {SERVER_NAME: {
            "command": sys.executable, "args": ["-m", "opentrain.core.bridge"],
            "env": {"OPENTRAIN_SERVER": hub.url, "OPENTRAIN_SERVER_PASSWORD": hub.password,
                    "OPENTRAIN_SESSION": sid, "OPENTRAIN_DIRECTORY": str(runner.project.directory),
                    "OPENTRAIN_AGENT": agent.name, "PYTHONPATH": os.pathsep.join(sys.path)}}}}
        d = runner.project.state / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"mcp-{sid}.json"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f)
        return str(path)

    def _prompt(self, runner: "Runner", sid: str, seen: set[str]) -> tuple[str, list[str]]:
        parts, ids = [], []
        for m in runner.project.messages(sid):
            if m["role"] in ("user", "synthetic") and m["id"] not in seen:
                text = "\n".join(p.get("text", "") for p in m["parts"] if p["type"] == "text")
                parts.append(f"<system-reminder>\n{text}\n</system-reminder>" if m["role"] == "synthetic" else text)
                ids.append(m["id"])
        return "\n\n".join(parts) or "Continue.", ids

    async def run_turn(self, runner: "Runner", turn: "Turn", agent: Any, ref: str) -> None:
        store, sid = runner.store, turn.session_id
        session = runner.project.session(sid)
        meta = session.get("metadata") or {}
        claude_session = meta.get("claude_session")
        seen = set(meta.get("claude_seen") or [])
        prompt, ids = self._prompt(runner, sid, seen)
        tools = runner.tools_for(agent)
        allowed = ",".join(f"mcp__{SERVER_NAME}__{t.name}" for t in tools)
        argv = [shutil.which("claude") or "claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
                "--include-partial-messages", "--tools", "", "--mcp-config", self._mcp_config(runner, sid, agent),
                "--strict-mcp-config", "--setting-sources", "",
                "--system-prompt", system_prompt(runner.project, agent, [INIT_NOTE])]
        if allowed:
            argv += ["--allowedTools", allowed]
        if self.model not in ("default", ""):
            argv += ["--model", self.model]
        if claude_session:
            argv += ["--resume", claude_session]
        else:
            claude_session = str(uuid.uuid4())
            argv += ["--session-id", claude_session]
        msg = store.create("message", {"session_id": sid, "role": "assistant", "agent": agent.name, "model": ref,
                                       "time": {"created": runner_now()}, "tokens": {"input": 0, "output": 0},
                                       "cost": 0.0})
        turn.message_id = msg["id"]
        env = {**os.environ, "MCP_TOOL_TIMEOUT": str(24 * 3600 * 1000), "MCP_TIMEOUT": "60000"}
        proc = await asyncio.create_subprocess_exec(*argv, cwd=str(runner.project.directory), env=env,
                                                    stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                    limit=64 * 1024 * 1024)
        turn.process = proc
        usage, cost, billing, finish, error = llm.Usage(), 0.0, "api", "stop", None
        try:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                try:
                    ev = json.loads(raw)
                except ValueError:
                    continue
                t = ev.get("type")
                if t == "system" and ev.get("subtype") == "init":
                    billing = _billing(ev)
                    if ev.get("session_id"):
                        claude_session = ev["session_id"]
                elif t == "stream_event":
                    e = ev.get("event", {})
                    if e.get("type") == "content_block_delta":
                        d = e.get("delta", {})
                        if d.get("type") == "text_delta":
                            runner._text(turn, d.get("text", ""))
                        elif d.get("type") == "thinking_delta" and d.get("thinking"):
                            runner._reason(turn, d["thinking"])
                    elif e.get("type") == "message_start" and turn.text_part:
                        runner._flush(turn)  # a new model message starts a new text part
                elif t == "result":
                    u = ev.get("usage") or {}
                    usage = llm.Usage(input=u.get("input_tokens", 0) or 0, output=u.get("output_tokens", 0) or 0,
                                      cache_read=u.get("cache_read_input_tokens", 0) or 0,
                                      cache_write=u.get("cache_creation_input_tokens", 0) or 0)
                    cost = float(ev.get("total_cost_usd") or 0.0)
                    if ev.get("is_error") or ev.get("subtype") not in (None, "success"):
                        finish, error = "error", ev.get("result") or ev.get("subtype")
            await proc.wait()
            if proc.returncode not in (0, None) and finish != "error":
                err = (await proc.stderr.read()).decode(errors="replace")[-2000:] if proc.stderr else ""
                finish, error = "error", f"claude exited with {proc.returncode}: {err.strip() or 'no output'}"
        finally:
            if proc.returncode is None:
                proc.kill()
            runner._flush(turn)
            turn.process = None
            store.update("session", sid, {"metadata": {"claude_session": claude_session,
                                                       "claude_seen": sorted(seen | set(ids)),
                                                       "billing": billing}})
        store.update("message", msg["id"], {"finish": finish, "error": error, "cost": cost, "billing": billing,
                                            "tokens": {"input": usage.input, "output": usage.output,
                                                       "cache_read": usage.cache_read,
                                                       "cache_write": usage.cache_write},
                                            "time": {"completed": runner_now()}})
        usage.cost = cost
        runner._account(sid, msg["id"], usage, cost if billing == "api" else 0.0)


def runner_now() -> float:
    import time

    return time.time()
