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
import fnmatch
import time
from typing import Any

from .questions import _after_wait
from .tool import ToolContext, ToolError

GRANTABLE = ("spend", "upload", "job.start", "promote", "serve", "webfetch", "download")
REPLIES = ("approve", "change", "skip")
FREE = "free and gated by nothing, so it went ahead without asking"
GRANT_SECONDS = 2 * 3600
PLAN_GRANT_SECONDS = 12 * 3600  # an approved plan covers its whole run: baseline, training, scoring
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
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.waiting[p["id"]] = fut
        auto = self._unattended(ctx.session_id, needs)
        if auto is None and self.harmless(p):  # nothing for the person to decide: never shown as a question
            auto = ("approved", FREE, "approve")
        if auto is not None:
            self._settle(p["id"], *auto)
        else:
            self.store.publish("session.status", {"session_id": ctx.session_id, "status": "waiting",
                                                  "proposal_id": p["id"]}, durable=False)
        try:
            reply, message = await fut
        except asyncio.CancelledError:
            self.store.update("proposal", p["id"], {"status": "cancelled"})
            raise
        finally:
            self.waiting.pop(p["id"], None)
            _after_wait(self.project, ctx.session_id)
        if reply == "approve" and needs:
            self.grant(ctx.session_id, needs, p["cost_usd"], source=p["id"])
        return reply, message

    def grant(self, sid: str, needs: list[str], cap_usd: float, *, source: str,
              seconds: float = GRANT_SECONDS) -> None:
        """The person approved a step (a proposal, or a plan page's go-ahead): its gated actions need no
        further asking, and its spending up to about ``cap_usd``. A newer approval replaces an older one."""
        self.grants[sid] = {"source": source, "needs": [n for n in needs if n in GRANTABLE],
                            "cap_usd": float(cap_usd or 0.0), "spent_usd": 0.0, "until": time.time() + seconds}

    @staticmethod
    def harmless(p: dict[str, Any]) -> bool:
        """A step that may go ahead by itself: it costs nothing and needs no gated action (no spend,
        upload, training, serving, promotion, fetch or download). Everything else waits for the person."""
        return not p.get("needs") and not float(p.get("cost_usd") or 0)

    def _unattended(self, sid: str, needs: list[str]) -> tuple[str, str, str] | None:
        """``opentrain run --yes`` (auto_approve) approves; an unattended run without it skips."""
        meta = (self.project.session(sid) or {}).get("metadata") or {}
        if not meta.get("unattended"):
            return None
        auto = meta.get("auto_approve") or []
        if not needs or (auto and all(any(fnmatch.fnmatchcase(n, a) for a in auto) for n in needs)):
            return "approved", "approved automatically (opentrain run --yes)", "approve"
        return ("skipped", "unattended run: nobody can approve this step. Stop here and say what you would do "
                "next; the user can rerun with --yes or continue in the app.", "skip")

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

    def answer_with_message(self, pid: str, text: str) -> None:
        """The person wrote a message instead of choosing: the step waits, the agent answers them."""
        if self.store.get("proposal", pid) is not None and pid in self.waiting:
            self._settle(pid, "answered", text, "message")

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
        if message == FREE:
            return (f"Not shown to the person: {title} is {FREE}. Do it now. Free steps never need propose: just "
                    "do them and say what you did in a line.")
        covered = ", ".join(needs or []) or "nothing gated"
        return (f"Approved: {title}. Do it now. This step's approval covers {covered}"
                + (f" up to about ${cost_usd:.2f}" if cost_usd else "") + "; anything beyond that still asks. "
                "When it is done, report the result in a line or two and go on.")
    if reply == "message":
        return (f"Not approved yet: instead of choosing, the person wrote: {message}\nAnswer that first, briefly. "
                "Do not run this step unless they asked you to; propose it again afterwards if it still makes sense.")
    if reply == "change":
        return (f"The user wants this step changed: {message}\nAdjust the plan and propose again (or just do it if "
                "the change makes it free and harmless).")
    if message:  # skipped for a reason (an unattended run)
        raise ToolError(f"not approved: {message}")
    raise ToolError("the user skipped this step. Do not do it. Say in one line what you will do instead, or ask "
                    "what they want.")
