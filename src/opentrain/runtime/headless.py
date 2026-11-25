"""The headless runtime: the user's own ``claude -p`` or ``codex exec``, under their own login.

Two uses. The engine runs a ROLE turn (trainer, evaluator) with ``headless_command`` +
``run_headless``, serving OpenTrain's train tools to the CLI as a local MCP server. And
``HeadlessRunner`` attempts a task with a frontier model, so it can be raced and benchmarked
against the specialist on exactly the same terms (fresh worktree, same verifier, run afterwards).

What is measured comes from the CLI's own report: Claude Code's ``stream-json`` result event
(``total_cost_usd``, ``usage``, ``duration_api_ms``, ``num_turns``) and Codex's ``--json`` events.
Under a subscription login there is no dollar bill: Claude Code still prints a list-price figure,
which is kept as ``notional_usd`` while ``cost_usd`` is None, so the budget ledger is not charged
money that was never spent. Credentials are never read: the CLI uses whatever login the user has,
and credential-looking fields are dropped from the events kept.

Every attempt runs with ``--no-session-persistence``: a frontier attempt on a held-out task must not
land in ``~/.claude/projects``, where the miner would turn it into a training task.

Codex support follows its documented flags (``exec --json``, ``--sandbox``, ``-C``, ``-m``,
``--output-schema``, ``-c`` overrides) and is untested against a real install.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .. import workspace as ws_mod
from ..plugs import Attempt

Kind = Literal["claude", "codex"]

ATTEMPT_NOTE = ("You are working non-interactively: no one will answer questions. Make the requested change "
                "in the current repository's working tree. Do not commit, do not install or remove "
                "dependencies, and do not edit test configuration.")
CLAUDE_ATTEMPT_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep")
TEACHER_TOOLS = ("Bash", "Read", "Write", "Edit")
"""A teacher's transcript becomes the specialist's training data, so the teacher gets exactly the
tools the direct runtime has (bash, read_file, write_file, edit_file). Anything else would make a
pass untrainable: the specialist cannot call a tool it was never given."""


@dataclass
class HeadlessResult:
    text: str
    """The CLI's final answer."""
    cost_usd: float | None
    """Dollars billed. None when nothing is billed per token (a subscription login) or not reported."""
    tokens: int
    turns: int | None
    events: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    model_seconds: float | None = None
    """Time spent waiting on the model API, as the CLI reports it."""
    seconds: float = 0.0
    """Wall clock of the whole process, measured here."""
    exit_code: int | None = None
    error: str | None = None
    notional_usd: float | None = None
    """The list-price figure a subscription run reports; never charged to the budget."""
    billing: Literal["api", "subscription", "unknown"] = "unknown"


# ---------------------------------------------------------------- argv


def _safe_prompt(prompt: str) -> str:
    # A prompt starting with "-" would be read as a flag.
    return " " + prompt if prompt.startswith("-") else prompt


def _codex_mcp_overrides(mcp_config: Path) -> list[str]:
    """Codex has no --mcp-config; the same servers go in as ``-c mcp_servers.<name>.<key>=<toml>`` overrides."""
    servers = json.loads(Path(mcp_config).read_text()).get("mcpServers", {})
    args: list[str] = []
    for name, spec in servers.items():
        if "command" in spec:
            args += ["-c", f"mcp_servers.{name}.command={json.dumps(spec['command'])}"]
        if spec.get("args"):
            args += ["-c", f"mcp_servers.{name}.args={json.dumps(list(spec['args']))}"]
        if spec.get("env"):
            table = ", ".join(f"{k} = {json.dumps(str(v))}" for k, v in spec["env"].items())
            args += ["-c", f"mcp_servers.{name}.env={{{table}}}"]
    return args


def headless_command(kind: Kind, prompt: str, *, system_prompt_file: Path | None = None,
                     mcp_config: Path | None = None, allowed_tools: list[str] | tuple[str, ...] | None = None,
                     cwd: Path | None = None, model: str | None = None, permission_mode: str = "acceptEdits",
                     output_schema: Path | None = None, sandbox: str = "workspace-write",
                     tools: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """The argv for one non-interactive turn. Run it with ``run_headless(argv, cwd, timeout)``.

    claude: the prompt follows ``-p`` directly, because ``--allowedTools`` and ``--mcp-config``
    take several values and would swallow a trailing prompt. ``--strict-mcp-config`` means only the
    servers given here load, never the user's own. ``cwd`` is not a flag; run_headless sets it.
    ``output_schema`` and ``sandbox`` are codex-only.

    codex: there is no system-prompt flag, so the file's text goes before the prompt; there is no
    per-tool allowlist, so ``allowed_tools`` is ignored and ``--sandbox`` is the confinement.
    """
    if kind == "claude":
        argv = ["claude", "-p", _safe_prompt(prompt), "--output-format", "stream-json", "--verbose",
                "--permission-mode", permission_mode, "--no-session-persistence", "--strict-mcp-config"]
        if tools is not None:
            argv += ["--tools", ",".join(tools)]  # the built-in tools that exist at all
        if allowed_tools:
            argv += ["--allowedTools", ",".join(allowed_tools)]
        if system_prompt_file:
            argv += ["--append-system-prompt-file", str(system_prompt_file)]
        if mcp_config:
            argv += ["--mcp-config", str(mcp_config)]
        if model:
            argv += ["--model", model]
        return argv
    if kind == "codex":
        argv = ["codex", "exec", "--json", "--sandbox", sandbox, "--skip-git-repo-check"]
        if cwd:
            argv += ["-C", str(cwd)]
        if model:
            argv += ["-m", model]
        if output_schema:
            argv += ["--output-schema", str(output_schema)]
        if mcp_config:
            argv += _codex_mcp_overrides(mcp_config)
        if system_prompt_file:
            prompt = Path(system_prompt_file).read_text().strip() + "\n\n" + prompt
        return argv + [_safe_prompt(prompt)]
    raise ValueError(f"unknown headless runtime {kind!r}; use claude or codex")


# ---------------------------------------------------------------- running and parsing


def _scrub(event: dict[str, Any]) -> dict[str, Any]:
    """Drop credential-looking fields (Claude Code's result carries ``credential_expires_at``)."""
    def secretish(key: str) -> bool:
        low = key.lower().replace("_", "")
        return "credential" in low or "apikey" in low

    return {k: v for k, v in event.items() if not secretish(k)}


def _billing(init: dict[str, Any]) -> str:
    # Claude Code reports where the key came from; "none" means an OAuth (subscription) login.
    return "subscription" if init.get("apiKeySource") == "none" else "api"


def parse_events(lines: list[str]) -> HeadlessResult:
    """Read Claude Code stream-json or Codex --json event lines into one result (events kept, scrubbed)."""
    events: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)

    res = HeadlessResult(text="", cost_usd=None, tokens=0, turns=None)
    codex_turns = 0
    for ev in events:
        kind = ev.get("type")
        if kind == "system" and ev.get("subtype") == "init":
            res.billing = _billing(ev)  # type: ignore[assignment]
        elif kind == "result":  # claude
            usage = ev.get("usage") or {}
            res.tokens_in = int(usage.get("input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0) \
                + int(usage.get("cache_read_input_tokens") or 0)
            res.tokens_out = int(usage.get("output_tokens") or 0)
            res.turns = ev.get("num_turns")
            res.text = ev.get("result") or ""
            if ev.get("duration_api_ms") is not None:
                res.model_seconds = round(ev["duration_api_ms"] / 1000, 3)
            cost = ev.get("total_cost_usd")
            if res.billing == "subscription":
                res.notional_usd = cost
            else:
                res.cost_usd = cost
            if ev.get("is_error") or ev.get("subtype") not in (None, "success"):
                res.error = f"{ev.get('subtype') or 'error'}: {str(ev.get('result') or '')[:300]}"
        elif kind == "turn.completed":  # codex
            usage = ev.get("usage") or {}
            res.tokens_in += int(usage.get("input_tokens") or 0)
            res.tokens_out += int(usage.get("output_tokens") or 0)
            codex_turns += 1
        elif kind == "item.completed" and (ev.get("item") or {}).get("type") == "agent_message":
            res.text = ev["item"].get("text") or ""
        elif kind in ("turn.failed", "error"):
            err = ev.get("error") or {}
            res.error = str(err.get("message") if isinstance(err, dict) else err or ev.get("message"))[:300]
    if codex_turns:
        res.turns = codex_turns
    res.tokens = res.tokens_in + res.tokens_out
    res.events = [_scrub(e) for e in events]
    return res


def run_headless(argv: list[str], cwd: Path, timeout: float) -> HeadlessResult:
    """Run one headless turn to completion (or kill its whole process group at ``timeout``)."""
    start = time.monotonic()
    try:
        proc = subprocess.Popen(argv, cwd=cwd, text=True, errors="replace", stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except FileNotFoundError:
        return HeadlessResult("", None, 0, None, error=f"{argv[0]} is not installed or not on PATH",
                              seconds=round(time.monotonic() - start, 3))
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        timed_out = True
    res = parse_events((out or "").splitlines())
    res.seconds = round(time.monotonic() - start, 3)
    res.exit_code = None if timed_out else proc.returncode
    if timed_out:
        res.error = f"timed out after {timeout:.0f}s"
    elif proc.returncode != 0 and not res.error:
        res.error = f"exit code {proc.returncode}: {(err or out or '').strip()[-300:]}"
    return res


# ---------------------------------------------------------------- transcripts in the direct runtime's shape


def _rel(path: Any, cwd: Path) -> str | None:
    if not isinstance(path, str):
        return None
    p = Path(path)
    if not p.is_absolute():
        return path
    try:
        return str(p.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return None


def _map_claude_tool(name: str, args: dict[str, Any], cwd: Path) -> tuple[str, dict[str, Any]]:
    """Claude Code's Bash/Read/Write/Edit as the direct runtime's tools, so a frontier pass can become an
    SFT example under ``SYSTEM_PROMPT``/``TOOLS``. Anything without an exact equivalent keeps its own name,
    and ``build_sft`` leaves such transcripts out rather than train on tools the specialist does not have."""
    rel = _rel(args.get("file_path"), cwd)
    if name == "Bash" and "command" in args:
        return "bash", {"command": args["command"]}
    if name == "Read" and rel:
        return "read_file", {"path": rel, **{k: args[k] for k in ("offset", "limit") if k in args}}
    if name == "Write" and rel and "content" in args:
        return "write_file", {"path": rel, "content": args["content"]}
    if name == "Edit" and rel and not args.get("replace_all"):
        return "edit_file", {"path": rel, "old_string": args.get("old_string", ""),
                             "new_string": args.get("new_string", "")}
    return name, args


def _tool_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return json.dumps(content)


def transcript(events: list[dict[str, Any]], instruction: str, cwd: Path) -> list[dict[str, Any]]:
    """The run as OpenAI chat messages: user instruction, assistant turns with tool_calls, tool results."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    current: dict[str, Any] | None = None
    current_id = None
    for ev in events:
        kind = ev.get("type")
        if kind == "assistant":  # claude: one event per content block; blocks of one API message share its id
            msg = ev.get("message") or {}
            if current is None or msg.get("id") != current_id:
                current = {"role": "assistant", "content": ""}
                current_id = msg.get("id")
                messages.append(current)
            for block in msg.get("content") or []:
                if block.get("type") == "text":
                    current["content"] = (current["content"] + "\n" + block["text"]).strip()
                elif block.get("type") == "tool_use":
                    name, args = _map_claude_tool(block.get("name", ""), block.get("input") or {}, cwd)
                    current.setdefault("tool_calls", []).append(
                        {"id": block.get("id"), "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}})
        elif kind == "user":
            current = None
            for block in (ev.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                                     "content": _tool_text(block.get("content"))})
        elif kind == "item.completed":  # codex
            item = ev.get("item") or {}
            if item.get("type") == "agent_message":
                messages.append({"role": "assistant", "content": item.get("text") or ""})
            elif item.get("type") == "command_execution":
                cid = item.get("id") or f"call_{len(messages)}"
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"id": cid, "type": "function",
                     "function": {"name": "bash", "arguments": json.dumps({"command": item.get("command", "")})}}]})
                messages.append({"role": "tool", "tool_call_id": cid,
                                 "content": f"exit code {item.get('exit_code')}\n{item.get('aggregated_output', '')}"})
            elif item.get("type") == "file_change":
                cid = item.get("id") or f"call_{len(messages)}"
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"id": cid, "type": "function",
                     "function": {"name": "apply_patch", "arguments": json.dumps({"changes": item.get("changes")})}}]})
                messages.append({"role": "tool", "tool_call_id": cid, "content": str(item.get("status", ""))})
    return messages


class HeadlessRunner:
    """A frontier model attempting a task through the user's own CLI. Does not verify (see ``AgentRunner``)."""

    def __init__(self, kind: Kind = "claude", model: str | None = None, *, name: str | None = None,
                 allowed_tools: tuple[str, ...] = CLAUDE_ATTEMPT_TOOLS, permission_mode: str = "acceptEdits",
                 tools: tuple[str, ...] | None = None):
        if kind not in ("claude", "codex"):
            raise ValueError(f"unknown headless runtime {kind!r}; use claude or codex")
        self.kind, self.model = kind, model
        self.name = name or (f"{kind}:{model}" if model else kind)
        self.allowed_tools, self.permission_mode, self.tools = allowed_tools, permission_mode, tools

    def attempt(self, instruction: str, workspace: Path, *, task_id: str, timeout: float) -> Attempt:
        start = time.monotonic()
        # The note lives outside the worktree, so it never shows up in the attempt's diff.
        with tempfile.NamedTemporaryFile("w", suffix=".md", prefix="ot-note-") as note:
            note.write(ATTEMPT_NOTE)
            note.flush()
            argv = headless_command(self.kind, instruction, system_prompt_file=Path(note.name),
                                    allowed_tools=self.allowed_tools, cwd=workspace, model=self.model,
                                    permission_mode=self.permission_mode, tools=self.tools)
            res = run_headless(argv, workspace, timeout)
        messages = transcript(res.events, instruction, workspace)
        out = Attempt(task_id=task_id, model=self.name, passed=None, seconds=0.0,
                      model_seconds=res.model_seconds or 0.0,
                      tool_calls=sum(len(m.get("tool_calls", [])) for m in messages),
                      tokens_in=res.tokens_in, tokens_out=res.tokens_out, cost_usd=res.cost_usd or 0.0,
                      messages=messages, error=res.error)
        try:
            out.patch = ws_mod.diff(workspace)
        except ws_mod.WorkspaceError as exc:
            out.error = (out.error + "; " if out.error else "") + f"could not read the diff: {exc}"
        out.seconds = round(time.monotonic() - start, 3)
        return out
