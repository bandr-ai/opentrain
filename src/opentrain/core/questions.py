"""Questions from the engine to the user: the ``question`` tool waits here for an answer.

A question is a durable object (so a client that connects later still sees it) and an awaitable
the tool blocks on. Clients answer through ``POST /api/session/{sid}/question/{qid}/reply``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .tool import ToolContext, ToolError


class Questions:
    def __init__(self, project: Any):
        self.project = project
        self.waiting: dict[str, asyncio.Future] = {}
        for q in project.store.find("question"):  # questions orphaned by a restart
            if q.get("status") == "pending":
                project.store.update("question", q["id"], {"status": "cancelled"})

    async def ask(self, ctx: ToolContext, questions: list[dict[str, Any]]) -> list[list[str]]:
        if not questions:
            raise ToolError("ask at least one question")
        for q in questions:
            if not q.get("question") or not isinstance(q.get("options"), list) or not q["options"]:
                raise ToolError("each question needs text and at least one option")
        meta = (self.project.session(ctx.session_id) or {}).get("metadata") or {}
        if meta.get("unattended"):  # nobody can answer: say so at once instead of waiting forever
            raise ToolError("unattended run: nobody can answer questions. If one option is clearly safe and "
                            "cheap, take it and say which and why; otherwise stop and say what you need.")
        q = self.project.store.create("question", {
            "session_id": ctx.session_id, "message_id": ctx.message_id, "call_id": ctx.call_id,
            "questions": questions, "status": "pending", "time": {"created": time.time()}})
        fut = asyncio.get_running_loop().create_future()
        self.waiting[q["id"]] = fut
        self.project.store.publish("session.status", {"session_id": ctx.session_id, "status": "waiting",
                                                      "question_id": q["id"]}, durable=False)
        try:
            answers = await fut
        except asyncio.CancelledError:
            self.project.store.update("question", q["id"], {"status": "cancelled"})
            raise
        finally:
            self.waiting.pop(q["id"], None)
            self.project.store.publish("session.status", {"session_id": ctx.session_id, "status": "busy"},
                                       durable=False)
        return answers

    def reply(self, qid: str, answers: list[list[str]] | None, rejected: bool = False) -> dict[str, Any]:
        q = self.project.store.get("question", qid)
        if q is None or q.get("status") != "pending" or qid not in self.waiting:
            raise KeyError(f"no pending question {qid}")
        if rejected:
            self.project.store.update("question", qid, {"status": "rejected"})
            self.waiting[qid].set_exception(ToolError("the user dismissed the question; decide sensibly or ask "
                                                      "in your reply"))
        else:
            answers = [list(map(str, a)) for a in (answers or [])]
            self.project.store.update("question", qid, {"status": "answered", "answers": answers})
            self.waiting[qid].set_result(answers)
        return self.project.store.get("question", qid)

    def pending(self, sid: str | None = None) -> list[dict[str, Any]]:
        return [q for q in self.project.store.find("question", session_id=sid)
                if q.get("status") == "pending" and q["id"] in self.waiting]
