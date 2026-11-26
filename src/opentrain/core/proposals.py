"""Next steps: the agent proposes one step at a time and the user approves, changes or skips it.

A proposal is how the engine asks for anything that costs money, sends data out, trains or changes
what is live. It says what the step is, why, what it costs and how long it takes, and which gated
actions it needs (``spend``, ``upload``, ``job.start``, ``promote``, ``serve``, ``webfetch``, ``download``).
Approving it grants exactly those actions to the session for that step, so the user answers once
instead of once per permission. The Trainfile budget is still enforced by the ledger, and a grant
never covers ``shell``, ``edit`` or ``write``.

The ``propose`` tool blocks on the answer like ``question`` does; clients answer through
``POST /api/session/{sid}/proposal/{pid}/reply``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .tool import ToolContext, ToolError

GRANTABLE = ("spend", "upload", "job.start", "promote", "serve", "webfetch", "download")
REPLIES = ("approve", "change", "skip")
GRANT_SECONDS = 2 * 3600
SPEND_SLACK = 1.5  # a granted step may spend up to 1.5x its estimate (+5 cents) before asking again


class Proposals:
    def __init__(self, project: Any):
        self.project = project
        self.store = project.store
        self.waiting: dict[str, asyncio.Future] = {}
        self.grants: dict[str, dict[str, Any]] = {}  # session id -> the approved step's grant
        for p in self.store.find("proposal"):  # proposals orphaned by a restart
            if p.get("status") == "pending":
                self.store.update("proposal", p["id"], {"status": "cancelled"})

    async def propose(self, ctx: ToolContext, step: dict[str, Any]) -> tuple[str, str | None]:
        needs = [n for n in step.get("needs") or [] if n in GRANTABLE]
        # one open proposal per session: a new one replaces the old
        for old in self.pending(ctx.session_id):
            self._settle(old["id"], "superseded", None)
        p = self.store.create("proposal", {
            "session_id": ctx.session_id, "message_id": ctx.message_id, "call_id": ctx.call_id,
            "agent": ctx.agent.name, "title": step["title"], "why": step.get("why") or "",
            "cost_usd": float(step.get("cost_usd") or 0.0), "minutes": step.get("minutes"),
            "needs": needs, "stage": step.get("stage"), "dataset": step.get("dataset"),
            "status": "pending", "time": {"created": time.time()}})
        fut = asyncio.get_running_loop().create_future()
        self.waiting[p["id"]] = fut
        self.store.publish("session.status", {"session_id": ctx.session_id, "status": "waiting",
                                              "proposal_id": p["id"]}, durable=False)
        try:
            reply, message = await fut
        except asyncio.CancelledError:
            self.store.update("proposal", p["id"], {"status": "cancelled"})
            raise
        finally:
            self.waiting.pop(p["id"], None)
            self.store.publish("session.status", {"session_id": ctx.session_id, "status": "busy"}, durable=False)
        if reply == "approve":
            self.grants[ctx.session_id] = {"proposal_id": p["id"], "needs": needs, "cap_usd": p["cost_usd"],
                                           "spent_usd": 0.0, "until": time.time() + GRANT_SECONDS}
        return reply, message

    def reply(self, pid: str, reply: str, message: str | None = None) -> dict[str, Any]:
        if reply not in REPLIES:
            raise ValueError(f"reply must be one of {', '.join(REPLIES)}")
        p = self.store.get("proposal", pid)
        if p is None or p.get("status") != "pending" or pid not in self.waiting:
            raise KeyError(f"no pending proposal {pid}")
        if reply == "change" and not (message or "").strip():
            raise ValueError("say what to change")
        return self._settle(pid, {"approve": "approved", "change": "changed", "skip": "skipped"}[reply], message,
                            reply)

    def _settle(self, pid: str, status: str, message: str | None, reply: str | None = None) -> dict[str, Any]:
        self.store.update("proposal", pid, {"status": status, "message": message, "time": {"replied": time.time()}})
        fut = self.waiting.get(pid)
        if fut is not None and not fut.done():
            fut.set_result((reply or "skip", message))
        return self.store.get("proposal", pid)

    def pending(self, sid: str | None = None) -> list[dict[str, Any]]:
        return [p for p in self.store.find("proposal", session_id=sid)
                if p.get("status") == "pending" and p["id"] in self.waiting]

    # ------------------------------------------------------------------ grants

    def granted(self, sid: str, action: str, metadata: dict[str, Any]) -> bool:
        """True when the approved step covers this action; spends are counted against its estimate."""
        g = self.grants.get(sid)
        if not g or time.time() > g["until"] or action not in g["needs"]:
            return False
        if action == "spend":
            usd = metadata.get("usd")
            if isinstance(usd, (int, float)):
                if g["spent_usd"] + usd > g["cap_usd"] * SPEND_SLACK + 0.05:
                    return False
                g["spent_usd"] += usd
        return True

    def end_step(self, sid: str) -> None:
        self.grants.pop(sid, None)


async def propose_tool(ctx: ToolContext, title: str, why: str = "", cost_usd: float = 0.0,
                       minutes: float | None = None, needs: list[str] | None = None, stage: str | None = None,
                       dataset: str | None = None) -> str:
    ps: Proposals | None = ctx.project.services.get("proposals")
    if ps is None:
        raise ToolError("proposals are not available here; ask with the question tool")
    bad = [n for n in needs or [] if n not in GRANTABLE]
    if bad:
        raise ToolError(f"needs may only list {', '.join(GRANTABLE)}; {', '.join(bad)} still ask on their own")
    reply, message = await ps.propose(ctx, {"title": title, "why": why, "cost_usd": cost_usd, "minutes": minutes,
                                            "needs": needs or [], "stage": stage, "dataset": dataset})
    if reply == "approve":
        covered = ", ".join(needs or []) or "nothing gated"
        return (f"Approved: {title}. Do it now. This step's approval covers {covered}"
                + (f" up to about ${cost_usd:.2f}" if cost_usd else "") + "; anything beyond that still asks. "
                "When it is done, report the result in a few lines and propose the next step.")
    if reply == "change":
        return (f"The user wants this step changed: {message}\nAdjust the plan and propose again (or just do it if "
                "the change makes it free and harmless).")
    raise ToolError("the user skipped this step. Do not do it. Say in one line what you will do instead, or ask "
                    "what they want.")
