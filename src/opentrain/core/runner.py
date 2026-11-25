"""The engine: the agent loop that runs a session.

A prompt is an inbox item. ``steer`` items (the default) are delivered at the next step boundary,
so the user can redirect a turn while it runs; ``queue`` items wait until the turn ends. One drain
task per busy session; prompts arriving while it runs only add to the inbox.

A turn is steps until the model stops calling tools:

1. Deliver pending steering input as user messages.
2. Build the request: the agent's system prompt, the history (from the last compaction on),
   the tools the agent may see. Compact first when the history nears the model's window.
3. Stream the model. Text and reasoning are parts updated live (ephemeral deltas, then one
   durable update); every tool call becomes a tool part, run concurrently, each checked by the
   permission service before it executes.
4. Record usage and cost on the message and the session; the budget hook can stop the turn.
5. Stop on no tool calls, an error, an interrupt, or the agent's step limit (the last step gets
   no tools, so the model must answer).

Agent providers (Claude Code) run their own loop instead; their tool calls come back through
``external_tool_call`` and land in the same message as parts, so clients cannot tell the
difference.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import llm
from .context import system_prompt
from .tool import ToolContext, ToolError, ToolResult, run

if TYPE_CHECKING:
    from .project import Project

log = logging.getLogger("opentrain.runner")

MAX_RETRIES = 4
DOOM_REPEATS = 3
COMPACT_AT = 0.75  # of the model's context window
KEEP_RECENT = 6  # messages kept verbatim after a compaction
MAX_STEPS_NOTE = ("You have reached the step limit for this turn. Do not call tools. Summarise what you did, "
                  "what is left, and what the user should decide next.")


class PermissionRejected(ToolError):
    pass


class BudgetStop(Exception):
    pass


@dataclass
class Turn:
    session_id: str
    message_id: str | None = None
    text_part: str | None = None
    text: str = ""
    reasoning_part: str | None = None
    reasoning: str = ""
    recent: list[str] = field(default_factory=list)  # doom-loop hashes
    process: Any = None  # an agent provider's subprocess


def _now() -> float:
    return time.time()


class Runner:
    def __init__(self, project: "Project", loop: asyncio.AbstractEventLoop | None = None):
        self.project = project
        self.store = project.store
        self.loop = loop
        self.tasks: dict[str, asyncio.Task] = {}
        self.turns: dict[str, Turn] = {}
        self.on_usage: list[Any] = []  # callables(session, message, usage, cost) -> None; may raise BudgetStop
        self._settle_crashed()

    # ------------------------------------------------------------------ admission

    def _settle_crashed(self) -> None:
        """Parts and messages left running by a process that died are closed as interrupted."""
        for p in self.store.find("part"):
            if p.get("type") == "tool" and p.get("tool", {}).get("state") in ("pending", "running"):
                self.store.update("part", p["id"], {"tool": {"state": "error", "error": "the server restarted"}})
        for s in self.store.find("session"):
            if s.get("status") not in (None, "idle"):
                self.store.update("session", s["id"], {"status": "idle"})

    def prompt(self, sid: str, text: str, *, delivery: str = "steer", agent: str | None = None,
               model: str | None = None, synthetic: bool = False, files: list[str] | None = None) -> dict[str, Any]:
        self.project.session(sid)
        if delivery not in ("steer", "queue"):
            raise ValueError("delivery must be steer or queue")
        patch = {k: v for k, v in {"agent": agent, "model": model}.items() if v}
        if patch:
            self.store.update("session", sid, patch)
        item = self.store.create("inbox", {"session_id": sid, "text": text, "delivery": delivery, "status": "pending",
                                           "synthetic": synthetic, "files": files or [], "time": {"created": _now()}})
        self.wake(sid)
        return item

    def wake(self, sid: str) -> None:
        loop = self.loop or asyncio.get_event_loop()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._start(sid)
        else:
            loop.call_soon_threadsafe(self._start, sid)

    def _start(self, sid: str) -> None:
        t = self.tasks.get(sid)
        if t and not t.done():
            return
        self.tasks[sid] = asyncio.get_running_loop().create_task(self._drain(sid), name=f"drain {sid}")

    def busy(self, sid: str) -> bool:
        t = self.tasks.get(sid)
        return bool(t and not t.done())

    async def interrupt(self, sid: str) -> bool:
        t = self.tasks.get(sid)
        if not t or t.done():
            return False
        turn = self.turns.get(sid)
        if turn and turn.process is not None:
            try:
                turn.process.kill()
            except ProcessLookupError:
                pass
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
        return True

    async def wait_idle(self, sid: str, timeout: float | None = None) -> None:
        t = self.tasks.get(sid)
        if t and not t.done():
            await asyncio.wait_for(asyncio.shield(t), timeout)

    # ------------------------------------------------------------------ the loop

    def _status(self, sid: str, status: str, **extra: Any) -> None:
        self.store.update("session", sid, {"status": status, "time": {"updated": _now()}})
        self.store.publish("session.status", {"session_id": sid, "status": status, **extra}, durable=False)

    def _pending(self, sid: str, delivery: str | None = None) -> list[dict[str, Any]]:
        items = [i for i in self.store.find("inbox", session_id=sid) if i["status"] == "pending"]
        return [i for i in items if delivery is None or i["delivery"] == delivery]

    def _deliver(self, items: list[dict[str, Any]]) -> None:
        for it in items:
            role = "synthetic" if it.get("synthetic") else "user"
            msg = self.store.create("message", {"session_id": it["session_id"], "role": role,
                                                "time": {"created": _now()}, "inbox_id": it["id"]})
            self.store.create("part", {"session_id": it["session_id"], "message_id": msg["id"], "type": "text",
                                       "text": it["text"], "time": {"created": _now()}})
            self.store.update("inbox", it["id"], {"status": "delivered", "message_id": msg["id"]})

    async def _drain(self, sid: str) -> None:
        self._status(sid, "busy")
        try:
            while True:
                items = self._pending(sid)
                if not items:
                    break
                self._deliver(items)
                await self._turn(sid)
        except asyncio.CancelledError:
            self._close_turn(sid, "interrupted", "interrupted by the user")
        except Exception as exc:  # the loop never dies silently
            log.exception("turn failed")
            self._close_turn(sid, "error", f"{type(exc).__name__}: {exc}")
        finally:
            self.turns.pop(sid, None)
            self._status(sid, "idle")
            self._title(sid)

    def _close_turn(self, sid: str, finish: str, error: str | None) -> None:
        turn = self.turns.get(sid)
        if turn is None:
            return
        self._flush(turn)
        for p in self.store.find("part", session_id=sid):
            if p.get("type") == "tool" and p["tool"].get("state") in ("pending", "running"):
                self.store.update("part", p["id"], {"tool": {"state": "error", "error": error or finish,
                                                             "time": {"end": _now()}}})
        if turn.message_id:
            msg = self.store.get("message", turn.message_id) or {}
            self.store.update("message", turn.message_id, {"finish": finish, "error": error,
                                                           "time": {"completed": msg.get("time", {}).get("completed")
                                                                    or _now()}})
        elif error:
            self._note(sid, f"Error: {error}", kind="error")

    def _note(self, sid: str, text: str, kind: str = "info") -> None:
        msg = self.store.create("message", {"session_id": sid, "role": "assistant", "agent": "system",
                                            "finish": "error" if kind == "error" else "stop", "error": text if kind == "error" else None,
                                            "time": {"created": _now(), "completed": _now()}})
        self.store.create("part", {"session_id": sid, "message_id": msg["id"], "type": "text", "text": text})

    def resolve_model(self, session: dict[str, Any], agent: Any) -> str:
        from . import config
        from .providers import default_model

        m = session.get("model")
        if isinstance(m, dict) and m.get("ref"):
            return m["ref"]
        if isinstance(m, str) and m:
            return m
        return agent.model or config.load(self.project.directory).get("model") or default_model()

    async def _turn(self, sid: str) -> None:
        from .providers import get as get_provider

        session = self.project.session(sid)
        agents = self.project.services["agents"]
        agent = agents.get(session.get("agent"))
        ref = self.resolve_model(session, agent)
        provider = get_provider(ref)
        turn = Turn(sid)
        self.turns[sid] = turn
        if getattr(provider, "kind", "step") == "agent":
            await provider.run_turn(self, turn, agent, ref)
            return
        await self._step_loop(turn, provider, agent, ref)

    async def _step_loop(self, turn: Turn, provider: Any, agent: Any, ref: str) -> None:
        sid = turn.session_id
        model = _model_info(ref)
        tools = self.tools_for(agent)
        steps = 0
        while True:
            steps += 1
            if steps > 1:
                steer = self._pending(sid, "steer")
                if steer:
                    self._deliver(steer)
            last = steps >= agent.steps
            history = self.history(sid)
            if _estimate(history) > model.context * COMPACT_AT:
                await self.compact(sid, provider, ref)
                history = self.history(sid)
            extra = [MAX_STEPS_NOTE] if last else []
            req = llm.Request(model=model.id, system=system_prompt(self.project, agent, extra), messages=history,
                              tools=[llm.ToolSpec(t.name, t.description, t.schema) for t in tools],
                              tool_choice="none" if last else "auto", max_tokens=min(model.output, 32_000))
            calls, finish = await self._one_step(turn, provider, req, agent, ref, model)
            if calls:
                await self._run_calls(turn, agent, calls)
            if not calls or finish in ("error", "length") or last:
                break
            if self._doomed(turn):
                self._note(sid, "Stopped: the same tool call returned the same result several times in a row.",
                           kind="error")
                break

    async def _one_step(self, turn: Turn, provider: Any, req: llm.Request, agent: Any, ref: str,
                        model: llm.Model) -> tuple[list[llm.ToolCall], str]:
        sid = turn.session_id
        msg = self.store.create("message", {"session_id": sid, "role": "assistant", "agent": agent.name,
                                            "model": ref, "time": {"created": _now()},
                                            "tokens": {"input": 0, "output": 0}, "cost": 0.0})
        turn.message_id, turn.text_part, turn.text, turn.reasoning_part, turn.reasoning = msg["id"], None, "", None, ""
        calls: list[llm.ToolCall] = []
        usage, finish, error, native = llm.Usage(), "stop", None, None
        attempt = 0
        while True:
            try:
                first = True
                async for ev in provider.stream(req):
                    if first:
                        first = False
                        self.store.update("message", msg["id"], {"time": {"first_token": _now()}})
                    if isinstance(ev, llm.Delta):
                        self._text(turn, ev.text)
                    elif isinstance(ev, llm.Reasoning):
                        self._reason(turn, ev.text)
                    elif isinstance(ev, llm.ToolCall):
                        calls.append(ev)
                    elif isinstance(ev, llm.Usage):
                        usage = ev
                    elif isinstance(ev, llm.Native):
                        native = ev.data
                    elif isinstance(ev, llm.Finish):
                        finish, error = ev.reason, ev.error
                break
            except llm.ProviderError as exc:
                attempt += 1
                if not exc.retryable or attempt > MAX_RETRIES or turn.text or calls:
                    finish, error = "error", str(exc)
                    break
                wait = min(30.0, 2 ** attempt) * (0.75 + 0.5 * (hash(msg["id"]) % 100) / 100)
                self.store.publish("session.status", {"session_id": sid, "status": "retry", "attempt": attempt,
                                                      "wait": wait, "error": str(exc)}, durable=False)
                await asyncio.sleep(wait)
        self._flush(turn)
        cost = usage.cost if usage.cost is not None else model.price(usage)
        self.store.update("message", msg["id"], {
            "finish": finish, "error": error, "native": native, "cost": cost,
            "tokens": {"input": usage.input, "output": usage.output, "cache_read": usage.cache_read,
                       "cache_write": usage.cache_write},
            "time": {"completed": _now()}})
        self._account(sid, msg["id"], usage, cost)
        if error and not calls:
            log.info("step ended with error: %s", error)
        return calls, finish

    def _account(self, sid: str, mid: str, usage: llm.Usage, cost: float) -> None:
        s = self.project.session(sid)
        tok = s.get("tokens") or {}
        self.store.update("session", sid, {"cost": round((s.get("cost") or 0.0) + cost, 6),
                                           "tokens": {"input": tok.get("input", 0) + usage.input,
                                                      "output": tok.get("output", 0) + usage.output}})
        for hook in self.on_usage:
            hook(s, mid, usage, cost)

    # ------------------------------------------------------------------ parts

    def _text(self, turn: Turn, delta: str) -> None:
        if turn.text_part is None:
            p = self.store.create("part", {"session_id": turn.session_id, "message_id": turn.message_id,
                                           "type": "text", "text": "", "time": {"created": _now()}})
            turn.text_part, turn.text = p["id"], ""
        turn.text += delta
        self.store.publish("part.delta", {"session_id": turn.session_id, "message_id": turn.message_id,
                                          "part_id": turn.text_part, "field": "text", "delta": delta}, durable=False)

    def _reason(self, turn: Turn, delta: str) -> None:
        if turn.reasoning_part is None:
            p = self.store.create("part", {"session_id": turn.session_id, "message_id": turn.message_id,
                                           "type": "reasoning", "text": "", "time": {"created": _now()}})
            turn.reasoning_part, turn.reasoning = p["id"], ""
        turn.reasoning += delta
        self.store.publish("part.delta", {"session_id": turn.session_id, "message_id": turn.message_id,
                                          "part_id": turn.reasoning_part, "field": "text", "delta": delta},
                           durable=False)

    def _flush(self, turn: Turn) -> None:
        """Make streamed text durable and close the open text parts (a tool call splits text)."""
        if turn.text_part:
            self.store.update("part", turn.text_part, {"text": turn.text, "time": {"end": _now()}})
        if turn.reasoning_part:
            self.store.update("part", turn.reasoning_part, {"text": turn.reasoning, "time": {"end": _now()}})
        turn.text_part, turn.text, turn.reasoning_part, turn.reasoning = None, "", None, ""

    # ------------------------------------------------------------------ tools

    def tools_for(self, agent: Any) -> list[Any]:
        return self.project.services["tools"].for_agent(agent, self.project.services.get("permissions"))

    async def _run_calls(self, turn: Turn, agent: Any, calls: list[llm.ToolCall]) -> None:
        parts = [self._tool_part(turn, c.name, c.input, c.id) for c in calls]
        await asyncio.gather(*(self._execute(turn, agent, c.name, c.input, c.id, p) for c, p in zip(calls, parts)))

    def _tool_part(self, turn: Turn, name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
        self._flush(turn)
        return self.store.create("part", {"session_id": turn.session_id, "message_id": turn.message_id,
                                          "type": "tool", "tool": {"name": name, "call_id": call_id, "input": args,
                                                                   "state": "running", "time": {"start": _now()}}})

    async def _execute(self, turn: Turn, agent: Any, name: str, args: dict[str, Any], call_id: str,
                       part: dict[str, Any]) -> ToolResult:
        ctx = ToolContext(self.project, turn.session_id, turn.message_id or "", call_id, agent, part_id=part["id"],
                          loop=asyncio.get_running_loop())
        try:
            tool = self.project.services["tools"].get(name)
            perms = self.project.services.get("permissions")
            if perms is not None and perms.hidden(agent, name):
                raise ToolError(f"the {agent.name} agent may not use {name}")
            title = tool.title(args) if tool.title else None
            if title:
                self.store.update("part", part["id"], {"tool": {"title": title}})
            result = await run(tool, ctx, args)
            self.store.update("part", part["id"], {"tool": {"state": "completed", "output": result.output,
                                                            "title": result.title or title,
                                                            "metadata": result.metadata, "time": {"end": _now()}}})
        except asyncio.CancelledError:
            self.store.update("part", part["id"], {"tool": {"state": "error", "error": "interrupted",
                                                            "time": {"end": _now()}}})
            raise
        except Exception as exc:
            msg = str(exc) if isinstance(exc, ToolError) else f"{type(exc).__name__}: {exc}"
            self.store.update("part", part["id"], {"tool": {"state": "error", "error": msg, "time": {"end": _now()}}})
            result = ToolResult(msg, metadata={"error": True})
        h = hashlib.sha256(json.dumps([name, args, result.output], sort_keys=True, default=str).encode()).hexdigest()
        turn.recent.append(h)
        return result

    def _doomed(self, turn: Turn) -> bool:
        r = turn.recent
        return len(r) >= DOOM_REPEATS and len(set(r[-DOOM_REPEATS:])) == 1

    async def external_tool_call(self, sid: str, name: str, args: dict[str, Any], call_id: str) -> ToolResult:
        """A tool call from an agent provider (through the MCP bridge), recorded in the current turn."""
        turn = self.turns.get(sid)
        if turn is None or turn.message_id is None:
            raise ToolError("no turn is running in this session")
        agent = self.project.services["agents"].get(self.project.session(sid).get("agent"))
        part = self._tool_part(turn, name, args, call_id)
        result = await self._execute(turn, agent, name, args, call_id, part)
        return result

    # ------------------------------------------------------------------ history

    def window(self, sid: str) -> list[dict[str, Any]]:
        """Store messages the model sees: the last compaction summary, the messages it kept, and
        everything after it."""
        msgs = self.project.messages(sid)
        idx = max((i for i, m in enumerate(msgs) if m["role"] == "compaction" and m.get("finish") == "stop"),
                  default=None)
        if idx is None:
            return msgs
        kept = msgs[idx]["kept_ids"]
        return [msgs[idx]] + [m for m in msgs[:idx] if m["id"] in kept] + msgs[idx + 1:]

    def history(self, sid: str) -> list[dict[str, Any]]:
        return neutral(self.window(sid))

    async def compact(self, sid: str, provider: Any, ref: str) -> None:
        """Summarise all but the most recent messages into one compaction message."""
        msgs = self.window(sid)
        if len(msgs) <= KEEP_RECENT + 1:
            return
        old, recent = msgs[:-KEEP_RECENT], msgs[-KEEP_RECENT:]
        transcript = json.dumps(neutral(old), default=str)[-400_000:]
        self.store.publish("session.status", {"session_id": sid, "status": "compacting"}, durable=False)
        req = llm.Request(model=ref.split("/", 1)[1], system=(
            "Summarise this OpenTrain session for its own continuation. Keep: the user's goal and constraints, "
            "decisions made, tasks/runs/jobs/versions with their ids and results, money spent, open problems, "
            "and the next step. Be specific and terse."), messages=[{"role": "user", "content": [
                {"type": "text", "text": transcript}]}], max_tokens=4000)
        text = ""
        async for ev in provider.stream(req):
            if isinstance(ev, llm.Delta):
                text += ev.text
        msg = self.store.create("message", {"session_id": sid, "role": "compaction", "time": {"created": _now()}})
        self.store.create("part", {"session_id": sid, "message_id": msg["id"], "type": "text", "text": text})
        self.store.update("message", msg["id"], {"finish": "stop", "time": {"completed": _now()},
                                                 "kept_ids": [m["id"] for m in recent]})

    def _title(self, sid: str) -> None:
        s = self.store.get("session", sid)
        if not s or s.get("title") not in (None, "", "New session"):
            return
        for m in self.project.messages(sid):
            if m["role"] == "user":
                text = " ".join(p.get("text", "") for p in m["parts"] if p["type"] == "text").strip()
                if text:
                    first = text.splitlines()[0]
                    self.store.update("session", sid, {"title": first[:60] + ("…" if len(first) > 60 else "")})
                return


def neutral(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Store messages (with parts) as provider-neutral messages."""
    out: list[dict[str, Any]] = []
    for m in msgs:
        texts = [p["text"] for p in m["parts"] if p["type"] == "text" and p.get("text")]
        if m["role"] == "compaction":
            out.append({"role": "user", "content": [{"type": "text", "text":
                        "Summary of the conversation so far (older messages were compacted):\n\n" + "\n".join(texts)}]})
        elif m["role"] == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": "\n".join(texts) or "(empty)"}]})
        elif m["role"] == "synthetic":
            out.append({"role": "user", "content": [{"type": "text", "text":
                        "<system-reminder>\n" + "\n".join(texts) + "\n</system-reminder>"}]})
        elif m["role"] == "assistant" and m.get("agent") != "system":
            content: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []
            for p in m["parts"]:
                if p["type"] == "text" and p.get("text"):
                    content.append({"type": "text", "text": p["text"]})
                elif p["type"] == "tool":
                    t = p["tool"]
                    content.append({"type": "tool_call", "id": t["call_id"], "name": t["name"], "input": t["input"]})
                    state = t.get("state")
                    output = t.get("output") if state == "completed" else f"Error: {t.get('error') or state}"
                    results.append({"type": "tool_result", "id": t["call_id"], "output": output or "",
                                    "is_error": state != "completed"})
            if not content:
                continue
            native = m.get("native")
            out.append({"role": "assistant", "content": content, **({"native": native} if native else {})})
            if results:
                out.append({"role": "tool", "content": results})
    return out


def _model_info(ref: str) -> llm.Model:
    from .providers import model_info

    return model_info(ref)


def _estimate(history: list[dict[str, Any]]) -> int:
    return len(json.dumps(history, default=str)) // 4
