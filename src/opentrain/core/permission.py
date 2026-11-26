"""Permissions: what the engine may do without asking. About money, data and what is live.

A rule is ``{action, resource, effect}`` with ``*`` / ``?`` wildcards; effect is allow, ask or
deny. Rules are read in layers, last match wins:

1. the base: everything allowed, except reading ``.env`` files (ask)
2. the agent's own rules (its Markdown front matter)
3. ``permissions`` in ``opentrain.jsonc`` (global, then project)
4. the session's rules

An ``ask`` that the user once answered with "always" is settled by the saved approval. In an
unattended session (the watch loop, ``opentrain run --yes`` excluded) nobody can answer, so ask
becomes deny unless the run was started with ``auto-approve`` for that action.

Actions the harness uses: tool names (``shell``, ``read``, ``edit``, ``write``, ``webfetch``,
...) plus ``spend`` (any cost: sampling, training), ``upload`` (data leaving the machine, the
resource is the trainer), ``job.start``, ``job.stop``, ``promote``, ``serve``, ``export`` and
``heldout.read``. The budget itself is enforced by the ledger, never by a rule: an allowed
``spend`` still cannot exceed the Trainfile.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from typing import Any

from .tool import ToolContext, ToolError

BASE: list[dict[str, Any]] = [
    {"action": "*", "resource": "*", "effect": "allow"},
    {"action": "read", "resource": "*.env", "effect": "ask"},
    {"action": "read", "resource": "*.env.*", "effect": "ask"},
    {"action": "read", "resource": "*.env.example", "effect": "allow"},
]
REPLIES = ("once", "always", "reject")


class PermissionRejected(ToolError):
    pass


def matches(rule: dict[str, Any], action: str, resource: str) -> bool:
    return fnmatch.fnmatchcase(action, rule.get("action", "*")) and _res(rule.get("resource", "*"), resource)


def _res(pattern: str, resource: str) -> bool:
    if pattern in ("*", "**"):
        return True
    if fnmatch.fnmatchcase(resource, pattern):
        return True
    # "dir/**" also covers "dir" itself; a bare file pattern also matches by basename
    if pattern.endswith("/**") and resource == pattern[:-3]:
        return True
    return "/" not in pattern and fnmatch.fnmatchcase(resource.rsplit("/", 1)[-1], pattern)


def evaluate(rules: list[dict[str, Any]], action: str, resource: str) -> str:
    effect = "ask"
    for r in rules:
        if matches(r, action, resource):
            effect = r["effect"]
    return effect


class Permissions:
    def __init__(self, project: Any):
        self.project = project
        self.store = project.store
        self.waiting: dict[str, asyncio.Future] = {}
        for p in self.store.find("permission"):  # requests orphaned by a restart
            if p.get("status") == "pending":
                self.store.update("permission", p["id"], {"status": "cancelled"})

    # ------------------------------------------------------------------ rules

    def rules(self, agent: Any, session: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from . import config

        cfg = config.load(self.project.directory)
        out = list(BASE) + list(getattr(agent, "permissions", []) or [])
        out += _rules(cfg.get("permissions"))
        if session:
            out += list(session.get("permissions") or [])
        return out

    def approvals(self) -> list[dict[str, Any]]:
        return self.store.find("approval")

    def effect(self, agent: Any, session: dict[str, Any] | None, action: str, resource: str) -> str:
        e = evaluate(self.rules(agent, session), action, resource)
        if e == "ask" and any(matches(a, action, resource) for a in self.approvals()):
            return "allow"
        if e == "ask" and session and (session.get("metadata") or {}).get("unattended"):
            auto = (session.get("metadata") or {}).get("auto_approve") or []
            return "allow" if any(fnmatch.fnmatchcase(action, a) for a in auto) else "deny"
        return e

    def hidden(self, agent: Any, tool: str) -> bool:
        """A tool is hidden from an agent when its whole action is denied and no narrower rule opens it."""
        rules = self.rules(agent)
        if evaluate(rules, tool, "*") != "deny":
            return False
        return not any(fnmatch.fnmatchcase(tool, r["action"]) and r.get("resource", "*") not in ("*", "**")
                       and r["effect"] != "deny" for r in rules)

    # ------------------------------------------------------------------ asking

    async def assert_allowed(self, ctx: ToolContext, action: str, resources: list[str], *,
                             metadata: dict[str, Any], always: list[str] | None = None) -> None:
        session = self.project.session(ctx.session_id)
        resources = [_relative(self.project, r) if action in ("read", "write", "edit", "glob", "grep") else r
                     for r in resources]
        need = []
        for r in resources:
            e = self.effect(ctx.agent, session, action, r)
            if e == "deny":
                raise PermissionRejected(f"{action} {r!s} is not allowed for the {ctx.agent.name} agent")
            if e == "ask":
                need.append(r)
        if not need:
            return
        props = self.project.services.get("proposals")
        if props is not None and props.granted(ctx.session_id, action, metadata):
            return  # the user approved the step this belongs to
        req = self.store.create("permission", {
            "session_id": ctx.session_id, "message_id": ctx.message_id, "call_id": ctx.call_id,
            "agent": ctx.agent.name, "action": action, "resources": need, "metadata": metadata,
            "always": always if always is not None else need, "status": "pending", "time": {"created": time.time()}})
        fut = asyncio.get_running_loop().create_future()
        self.waiting[req["id"]] = fut
        self.store.publish("session.status", {"session_id": ctx.session_id, "status": "waiting",
                                              "permission_id": req["id"]}, durable=False)
        try:
            reply, message = await fut
        except asyncio.CancelledError:
            self.store.update("permission", req["id"], {"status": "cancelled"})
            raise
        finally:
            self.waiting.pop(req["id"], None)
            self.store.publish("session.status", {"session_id": ctx.session_id, "status": "busy"}, durable=False)
        if reply == "reject":
            raise PermissionRejected("the user rejected this" + (f": {message}" if message else
                                                                 "; ask what they want instead"))

    def reply(self, pid: str, reply: str, message: str | None = None) -> dict[str, Any]:
        if reply not in REPLIES:
            raise ValueError(f"reply must be one of {', '.join(REPLIES)}")
        req = self.store.get("permission", pid)
        if req is None or req.get("status") != "pending" or pid not in self.waiting:
            raise KeyError(f"no pending permission request {pid}")
        if reply == "always":
            for r in req.get("always") or req["resources"]:
                self.store.create("approval", {"action": req["action"], "resource": r, "effect": "allow",
                                               "time": {"created": time.time()}, "from": pid})
        self.store.update("permission", pid, {"status": reply, "message": message, "time": {"replied": time.time()}})
        self.waiting[pid].set_result((reply, message))
        return self.store.get("permission", pid)

    def pending(self, sid: str | None = None) -> list[dict[str, Any]]:
        return [p for p in self.store.find("permission", session_id=sid)
                if p.get("status") == "pending" and p["id"] in self.waiting]

    def revoke(self, approval_id: str) -> None:
        if self.store.get("approval", approval_id) is None:
            raise KeyError(f"no approval {approval_id}")
        self.store.delete("approval", approval_id)


def _rules(spec: Any) -> list[dict[str, Any]]:
    from .agent import rules_from

    if not spec:
        return []
    if isinstance(spec, list):
        return [r for r in spec if isinstance(r, dict) and r.get("effect") in ("allow", "ask", "deny")]
    return rules_from(spec, "opentrain.jsonc")


def _relative(project: Any, path: str) -> str:
    from pathlib import Path

    if path in ("*", "**"):
        return path
    p = Path(path).expanduser()
    if not p.is_absolute():
        return path
    try:
        return str(p.resolve().relative_to(project.directory))
    except ValueError:
        return str(p)


class BudgetHook:
    """Engine tokens paid by the provider key count against the Trainfile budget like any spend.

    Plan logins (Claude Code under a subscription) report cost 0 here: their tokens are tracked on
    the session, but no dollars are drawn from the budget.
    """

    def __init__(self, project: Any):
        self.project = project

    def __call__(self, session: dict[str, Any], mid: str, usage: Any, cost: float) -> None:
        from .. import trainfile
        from ..budget import BudgetExceeded, Ledger
        from .runner import BudgetStop

        if not cost or cost <= 0:
            return
        tf = trainfile.for_project(self.project.directory)
        ledger = Ledger(self.project.state, tf.budget)
        run_id = (session.get("metadata") or {}).get("run_id")
        try:
            rid = ledger.reserve(cost, f"engine tokens ({session.get('agent')})", run_id)
        except BudgetExceeded as exc:
            raise BudgetStop(f"the Trainfile budget is used up ({exc}); the turn was stopped") from None
        ledger.settle(rid, cost)
