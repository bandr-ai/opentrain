"""Questions from the engine to the user: the ``question`` tool waits here for an answer.

A question is a durable object (so a client that connects later still sees it) and an awaitable
the tool blocks on. Clients answer through ``POST /api/session/{sid}/question/{qid}/reply``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from .tool import ToolContext, ToolError

AUTO_SECONDS = 30  # a question with a default for every part takes the defaults when nobody answers in time


def _after_wait(project: Any, sid: str) -> None:
    runner = project.services.get("runner")
    if runner is not None:
        runner.status_after_wait(sid)
    else:
        project.store.publish("session.status", {"session_id": sid, "status": "busy"}, durable=False)


class Questions:
    def __init__(self, project: Any):
        self.project = project
        self.waiting: dict[str, asyncio.Future] = {}
        for q in project.store.find("question"):  # questions orphaned by a restart
            if q.get("status") == "pending":
                project.store.update("question", q["id"], {"status": "cancelled"})

    async def ask(self, ctx: ToolContext, questions: list[dict[str, Any]]) -> tuple[list[list[str]], bool]:
        """The answers, and whether they are the defaults taken because nobody answered in time."""
        if not questions:
            raise ToolError("ask at least one question")
        for q in questions:
            if not q.get("question") or not isinstance(q.get("options"), list) or not q["options"]:
                raise ToolError("each question needs text and at least one option")
            if q.get("default") is not None and q["default"] not in q["options"]:
                raise ToolError(f"default {q['default']!r} is not one of the options")
        defaults = [[q["default"]] for q in questions] if all(q.get("default") for q in questions) else None
        meta = (self.project.session(ctx.session_id) or {}).get("metadata") or {}
        if meta.get("unattended"):  # nobody can answer: say so at once instead of waiting forever
            if defaults:
                return defaults, True
            raise ToolError("unattended run: nobody can answer questions. If one option is clearly safe and "
                            "cheap, take it and say which and why; otherwise stop and say what you need.")
        q = self.project.store.create("question", {
            "session_id": ctx.session_id, "message_id": ctx.message_id, "call_id": ctx.call_id,
            "questions": questions, "status": "pending", "time": {"created": time.time()},
            "auto_at": time.time() + AUTO_SECONDS if defaults else None})
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.waiting[q["id"]] = fut
        timer = loop.call_later(AUTO_SECONDS, self._auto, q["id"]) if defaults else None
        self.project.store.publish("session.status", {"session_id": ctx.session_id, "status": "waiting",
                                                      "question_id": q["id"]}, durable=False)
        try:
            answers = await fut
        except asyncio.CancelledError:
            self.project.store.update("question", q["id"], {"status": "cancelled"})
            raise
        finally:
            if timer:
                timer.cancel()
            self.waiting.pop(q["id"], None)
            _after_wait(self.project, ctx.session_id)
        return answers, bool((self.project.store.get("question", q["id"]) or {}).get("auto"))

    def _auto(self, qid: str) -> None:
        q = self.project.store.get("question", qid)
        fut = self.waiting.get(qid)
        if q is None or fut is None or fut.done() or q.get("status") != "pending" or not q.get("auto_at"):
            return
        answers = [[x["default"]] for x in q["questions"]]
        self.project.store.update("question", qid, {"status": "answered", "answers": answers, "auto": True})
        fut.set_result(answers)

    def hold(self, qid: str) -> dict[str, Any]:
        """The person is looking at it: stop the countdown and wait for their answer."""
        q = self.project.store.get("question", qid)
        if q is None or q.get("status") != "pending" or qid not in self.waiting:
            raise KeyError(f"no pending question {qid}")
        self.project.store.update("question", qid, {"auto_at": None})
        return self.project.store.get("question", qid)

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

    def answer_with_message(self, qid: str, text: str) -> None:
        """The person wrote a message instead of picking an option: it is their answer, in their words."""
        q = self.project.store.get("question", qid)
        fut = self.waiting.get(qid)
        if q is None or fut is None or fut.done():
            return
        self.project.store.update("question", qid, {"status": "answered", "answers": [[text]], "free_text": True})
        fut.set_exception(ToolError(f"Instead of picking an option, the person wrote: {text}\nTake that as their "
                                    "answer (it may be a question for you: answer it first) and continue."))

    def pending(self, sid: str | None = None) -> list[dict[str, Any]]:
        return [q for q in self.project.store.find("question", session_id=sid)
                if q.get("status") == "pending" and q["id"] in self.waiting]
