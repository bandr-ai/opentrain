"""The direct runtime: OpenTrain's own coding-agent loop over an OpenAI-compatible chat API.

This is how the trained specialist attempts tasks, and the transcripts it writes are what the
specialist is trained on. So the prompt and tools here are a contract: ``SYSTEM_PROMPT`` and
``TOOLS`` are what every SFT example carries (``opentrain.data.sft``), and changing either means
the next model is trained under something other than what it is served under. Keep them stable.

Deliberately small (in the spirit of mini-swe-agent): five tools, plain files, the model decides
every step. Measurements are taken, never inferred: each assistant message carries the seconds of
the model call that produced it, each tool message the seconds the tool took (``"seconds"`` keys,
stripped before anything is sent to a model or written as training data).

The bash tool runs as the user inside the task's worktree; it is the same light sandbox as the
verifier (see ``opentrain.workspace``). File tools are confined to the workspace.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .. import workspace as ws_mod
from ..plugs import Attempt, Endpoint

SYSTEM_PROMPT = """\
You are a coding agent working in a git repository, which is your current directory.
Make the change the user asks for by editing files in the working tree.

Tools: bash runs a shell command in the repository; read_file, write_file and edit_file work on
files by path relative to the repository root. Use them to look before you change anything, make
the smallest change that does the job, and run the relevant tests when the repository has them.
Do not commit, do not install or remove dependencies, and do not edit test configuration. Do not ask
questions: no one will answer. When the change is done, call finish with a one-line summary."""

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the repository root. Returns the exit code and output "
                       "(long output is shortened).",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command, run with bash -c."},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file, with line numbers.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "offset": {"type": "integer", "description": "First line to show, from 1 (default 1)."},
            "limit": {"type": "integer", "description": "Most lines to show (default 2000)."},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace one exact occurrence of old_string with new_string. old_string must "
                       "appear exactly once in the file; include surrounding lines to make it unique.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root."},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        }, "required": ["path", "old_string", "new_string"]},
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Say the change is done. Ends the attempt.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "One line: what was changed."},
        }, "required": ["summary"]},
    }},
]

TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOLS)
MAX_OUTPUT = 10_000
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class ToolError(ValueError):
    """A tool call the agent got wrong. The message goes back to the model as the tool result."""


class ModelError(RuntimeError):
    pass


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    """Keep the head and tail of long output and say how much was dropped."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - 2 * half} characters omitted] ...\n{text[-half:]}"


def resolve(workspace: Path, rel: str) -> Path:
    """A path inside the workspace. Absolute paths and anything that escapes (``..``, symlinks) are refused."""
    if not isinstance(rel, str) or not rel.strip():
        raise ToolError("path is required")
    if Path(rel).is_absolute():
        raise ToolError(f"{rel}: use a path relative to the repository root")
    root = workspace.resolve()
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise ToolError(f"{rel}: path is outside the repository")
    if ".git" in path.relative_to(root).parts:
        raise ToolError(f"{rel}: the .git folder is off limits")
    return path


def strip_timing(message: dict[str, Any]) -> dict[str, Any]:
    """The message as the chat API (and the training file) expects it: timing keys removed."""
    return {k: v for k, v in message.items() if k != "seconds"}


class DirectRunner:
    """Runs one model through the tool loop on one task. Does not verify (see ``AgentRunner``)."""

    def __init__(self, endpoint: Endpoint, max_steps: int = 60, *, name: str | None = None,
                 temperature: float | None = 0.0, max_tokens: int | None = 4096, bash_timeout: float = 120,
                 request_timeout: float = 300, retries: int = 5, backoff: float = 1.0,
                 usd_per_mtok_in: float = 0.0, usd_per_mtok_out: float = 0.0):
        self.endpoint = endpoint
        self.max_steps = max_steps
        self.name = name or f"direct:{endpoint.model}"
        self.temperature, self.max_tokens = temperature, max_tokens
        self.bash_timeout, self.request_timeout = bash_timeout, request_timeout
        self.retries, self.backoff = retries, backoff
        self.usd_per_mtok_in, self.usd_per_mtok_out = usd_per_mtok_in, usd_per_mtok_out

    # ------------------------------------------------------------ the model call

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.endpoint.api_key_env) if self.endpoint.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(self, messages: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
        """One /chat/completions call, retried with backoff on rate limits, 5xx and dropped connections."""
        body: dict[str, Any] = {"model": self.endpoint.model, "messages": [strip_timing(m) for m in messages],
                                "tools": TOOLS, "tool_choice": "auto"}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        data = json.dumps(body).encode()
        last = ""
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
            wait = self.backoff * 2 ** attempt
            try:
                with urllib.request.urlopen(req, timeout=max(1.0, min(self.request_timeout, timeout))) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                last = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRY_STATUS:
                    raise ModelError(last) from exc
                after = exc.headers.get("Retry-After") if exc.headers else None
                if after and after.replace(".", "", 1).isdigit():
                    wait = min(float(after), 60.0)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            except json.JSONDecodeError as exc:
                last = f"the endpoint returned something that is not JSON: {exc}"
            if attempt < self.retries:
                time.sleep(wait)
        raise ModelError(f"gave up after {self.retries + 1} tries: {last}")

    # ------------------------------------------------------------ tools

    def _bash(self, workspace: Path, command: str, timeout: float) -> str:
        env = dict(os.environ)
        if self.endpoint.api_key_env:
            env.pop(self.endpoint.api_key_env, None)  # the agent's shell never sees the model key
        proc = subprocess.Popen(["bash", "-c", command], cwd=workspace, env=env, text=True, errors="replace",
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
            return f"timed out after {timeout:.0f}s and was stopped\n{clip(out or '')}"
        return f"exit code {proc.returncode}\n{clip(out or '')}"

    def run_tool(self, workspace: Path, name: str, args: dict[str, Any], timeout: float) -> str:
        if name == "bash":
            command = args.get("command")
            if not isinstance(command, str) or not command.strip():
                raise ToolError("command is required")
            return self._bash(workspace, command, max(1.0, min(self.bash_timeout, timeout)))
        if name == "read_file":
            path = resolve(workspace, args.get("path"))
            if not path.is_file():
                raise ToolError(f"{args.get('path')}: no such file")
            lines = path.read_text(errors="replace").splitlines()
            start = max(1, int(args.get("offset") or 1))
            count = max(1, int(args.get("limit") or 2000))
            shown = lines[start - 1:start - 1 + count]
            text = "\n".join(f"{n:>6}\t{line}" for n, line in enumerate(shown, start))
            more = len(lines) - (start - 1 + len(shown))
            return clip(text + (f"\n... {more} more lines" if more > 0 else "")) or "(empty file)"
        if name == "write_file":
            path = resolve(workspace, args.get("path"))
            content = args.get("content")
            if not isinstance(content, str):
                raise ToolError("content must be a string")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"wrote {len(content)} characters to {args['path']}"
        if name == "edit_file":
            path = resolve(workspace, args.get("path"))
            old, new = args.get("old_string"), args.get("new_string")
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                raise ToolError("old_string and new_string are required, and old_string must not be empty")
            if not path.is_file():
                raise ToolError(f"{args.get('path')}: no such file")
            text = path.read_text()
            n = text.count(old)
            if n != 1:
                raise ToolError(f"old_string appears {n} times in {args['path']}; it must appear exactly once")
            path.write_text(text.replace(old, new, 1))
            return f"edited {args['path']}"
        if name == "finish":
            return "finished"
        raise ToolError(f"unknown tool {name!r}; the tools are {', '.join(sorted(TOOL_NAMES))}")

    # ------------------------------------------------------------ the loop

    def attempt(self, instruction: str, workspace: Path, *, task_id: str, timeout: float) -> Attempt:
        start = time.monotonic()
        deadline = start + timeout
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": instruction}]
        out = Attempt(task_id=task_id, model=self.name, passed=None, seconds=0.0, messages=messages)
        finished = False
        try:
            for _ in range(self.max_steps):
                left = deadline - time.monotonic()
                if left <= 0:
                    out.error = f"timed out after {timeout:.0f}s"
                    break
                t0 = time.monotonic()
                resp = self.complete(messages, left)
                model_secs = time.monotonic() - t0
                out.model_seconds += model_secs
                usage = resp.get("usage") or {}
                out.tokens_in += int(usage.get("prompt_tokens") or 0)
                out.tokens_out += int(usage.get("completion_tokens") or 0)
                choices = resp.get("choices") or []
                if not choices:
                    raise ModelError(f"no choices in the response: {json.dumps(resp)[:300]}")
                msg = choices[0].get("message") or {}
                calls = [{"id": c.get("id") or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                          "function": {"name": (c.get("function") or {}).get("name", ""),
                                       "arguments": (c.get("function") or {}).get("arguments") or "{}"}}
                         for c in msg.get("tool_calls") or []]
                assistant: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
                if calls:
                    assistant["tool_calls"] = calls
                assistant["seconds"] = round(model_secs, 3)
                messages.append(assistant)
                if not calls:  # the model stopped talking to tools: it considers itself done
                    finished = True
                    break
                for call in calls:
                    fn = call["function"]
                    t1 = time.monotonic()
                    try:
                        args = json.loads(fn["arguments"])
                        if not isinstance(args, dict):
                            raise ToolError("arguments must be a JSON object")
                        result = self.run_tool(workspace, fn["name"], args, deadline - time.monotonic())
                    except json.JSONDecodeError as exc:
                        result = f"error: arguments are not valid JSON ({exc})"
                    except (ToolError, OSError, ValueError, TypeError) as exc:
                        result = f"error: {exc}"
                    out.tool_calls += 1
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result,
                                     "seconds": round(time.monotonic() - t1, 3)})
                    finished = finished or fn["name"] == "finish"
                if finished:
                    break
            else:
                out.error = f"stopped after {self.max_steps} steps without finishing"
        except ModelError as exc:
            out.error = f"model call failed: {exc}"
        out.model_seconds = round(out.model_seconds, 3)
        out.cost_usd = round(out.tokens_in * self.usd_per_mtok_in / 1e6 + out.tokens_out * self.usd_per_mtok_out / 1e6,
                             6)
        try:
            out.patch = ws_mod.diff(workspace)
        except ws_mod.WorkspaceError as exc:
            out.error = (out.error + "; " if out.error else "") + f"could not read the diff: {exc}"
        out.seconds = round(time.monotonic() - start, 3)
        return out
