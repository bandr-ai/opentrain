"""Jobs belong to the server, not to the agent's turn.

Anything that takes minutes or hours — mining tasks, sampling a model, scoring held-out tasks,
a training run on a remote backend — runs here as a job. The tool that starts one returns at
once with the job id. The job's state is a projected object (``job.created`` / ``job.updated``)
that every client sees live, and when it finishes the session that started it gets a system
reminder queued, which wakes the agent to act on the result. Jobs survive the agent's turn and
the client; training jobs on a backend are re-attached when the server restarts.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("opentrain.jobs")

TERMINAL = ("succeeded", "failed", "cancelled")
POLL = 15.0


def _now() -> float:
    return time.time()


class JobCancelled(Exception):
    pass


class JobContext:
    """Handed to a local job's function: report progress, write log lines, notice cancellation."""

    def __init__(self, jobs: "Jobs", job_id: str):
        self.jobs, self.job_id = jobs, job_id
        self.cancel_event = threading.Event()

    def progress(self, fraction: float | None = None, detail: str | None = None) -> None:
        patch: dict[str, Any] = {}
        if fraction is not None:
            patch["progress"] = max(0.0, min(1.0, float(fraction)))
        if detail is not None:
            patch["detail"] = detail
        if patch:
            self.jobs.store.update("job", self.job_id, patch)

    def log(self, line: str) -> None:
        self.jobs.append_log(self.job_id, line)

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()


class Jobs:
    def __init__(self, project: Any, loop: asyncio.AbstractEventLoop | None = None, poll: float = POLL):
        self.project, self.store, self.loop, self.poll = project, project.store, loop, poll
        self.contexts: dict[str, JobContext] = {}
        self.pollers: dict[str, asyncio.Task] = {}
        self.dir = project.state / "jobs"
        self._closed = False
        for j in self.store.find("job"):  # resume: local jobs died with the process; backend jobs go on
            if j["state"] in TERMINAL:
                continue
            if j.get("backend_job"):
                self._schedule_poll(j["id"])
            else:
                self.store.update("job", j["id"], {"state": "failed", "error": "the server restarted during the job",
                                                   "time": {"ended": _now()}})

    # ------------------------------------------------------------------ reading

    def get(self, job_id: str) -> dict[str, Any]:
        j = self.store.get("job", job_id)
        if j is None:
            raise KeyError(f"no job {job_id}")
        return j

    def list(self, session_id: str | None = None, active: bool = False) -> list[dict[str, Any]]:
        out = self.store.find("job", session_id=session_id, newest_first=True)
        return [j for j in out if not active or j["state"] not in TERMINAL]

    def append_log(self, job_id: str, line: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{job_id}.log", "a") as f:
            f.write(line.rstrip("\n") + "\n")

    def logs(self, job_id: str, tail: int = 200) -> list[str]:
        j = self.get(job_id)
        if j.get("backend_job"):
            from ..backends import JobStore

            return JobStore(self.project.state).logs(j["backend_job"], tail)
        p = self.dir / f"{job_id}.log"
        return p.read_text().splitlines()[-tail:] if p.exists() else []

    # ------------------------------------------------------------------ local jobs

    def start_local(self, session_id: str | None, kind: str, title: str, fn: Callable[[JobContext], Any], *,
                    run_id: str | None = None, summarize: Callable[[Any], str] | None = None) -> dict[str, Any]:
        job = self.store.create("job", {"session_id": session_id, "kind": kind, "title": title, "state": "running",
                                        "progress": None, "detail": "", "run_id": run_id,
                                        "time": {"created": _now(), "started": _now()}})
        ctx = JobContext(self, job["id"])
        self.contexts[job["id"]] = ctx

        def work() -> None:
            try:
                result = fn(ctx)
                state, err = "succeeded", None
            except JobCancelled:
                result, state, err = None, "cancelled", "cancelled"
            except Exception as exc:
                log.exception("job %s failed", job["id"])
                self.append_log(job["id"], traceback.format_exc())
                result, state, err = None, "failed", f"{type(exc).__name__}: {exc}"
            finally:
                self.contexts.pop(job["id"], None)
            self.store.update("job", job["id"], {"state": state, "result": _jsonable(result), "error": err,
                                                 "progress": 1.0 if state == "succeeded" else None,
                                                 "time": {"ended": _now()}})
            summary = summarize(result) if (summarize and state == "succeeded") else None
            self._notify(job["id"], summary)

        threading.Thread(target=work, name=f"job {job['id']}", daemon=True).start()
        return job

    # ------------------------------------------------------------------ backend jobs

    def track_backend(self, session_id: str | None, backend_job: str, backend: str, title: str,
                      run_id: str | None, estimate: float | None = None) -> dict[str, Any]:
        job = self.store.create("job", {"session_id": session_id, "kind": "train", "title": title, "state": "running",
                                        "backend": backend, "backend_job": backend_job, "run_id": run_id,
                                        "estimate_usd": estimate, "cost_usd": 0.0, "progress": None, "detail": "",
                                        "time": {"created": _now(), "started": _now()}})
        self._schedule_poll(job["id"])
        return job

    def _schedule_poll(self, job_id: str) -> None:
        if self.loop is None or self.loop.is_closed():
            return

        def start() -> None:
            if job_id not in self.pollers or self.pollers[job_id].done():
                self.pollers[job_id] = asyncio.get_running_loop().create_task(self._poll(job_id))

        try:
            if asyncio.get_running_loop() is self.loop:
                start()
                return
        except RuntimeError:
            pass
        self.loop.call_soon_threadsafe(start)

    async def _poll(self, job_id: str) -> None:
        while not self._closed:
            try:
                done = await asyncio.to_thread(self.refresh, job_id)
            except Exception as exc:
                log.warning("polling %s failed: %s", job_id, exc)
                done = False
            if done:
                return
            await asyncio.sleep(self.poll)

    def refresh(self, job_id: str) -> bool:
        """Read a backend job's status once; settle and register the checkpoint when it ends."""
        from ..mcp import Server

        j = self.get(job_id)
        if j["state"] in TERMINAL:
            return True
        srv = Server(self.project.state, "trainer", j.get("run_id"))
        st = srv.t_train_status(j["backend_job"])
        patch: dict[str, Any] = {"progress": st.get("progress"), "detail": st.get("detail") or "",
                                 "cost_usd": st.get("cost_usd", 0.0)}
        if st["state"] in TERMINAL:
            patch.update(state=st["state"], version=st.get("version"), time={"ended": _now()},
                         error=None if st["state"] == "succeeded" else (st.get("detail") or st["state"]))
            if st.get("cancelled"):
                patch["error"] = f"cancelled: {st['cancelled']}"
        self.store.update("job", job_id, patch)
        if st["state"] in TERMINAL:
            v = st.get("version")
            self._notify(job_id, f"Checkpoint registered as {v} (not live). Next: eval_request {v}, then "
                                 f"version_propose {v}." if v else None)
            return True
        return False

    # ------------------------------------------------------------------ control

    def cancel(self, job_id: str) -> dict[str, Any]:
        j = self.get(job_id)
        if j["state"] in TERMINAL:
            return j
        if j.get("backend_job"):
            from ..backends import get_backend

            get_backend(j["backend"], self.project.state).cancel(j["backend_job"])
            self.store.update("job", job_id, {"detail": "cancelling"})
        elif job_id in self.contexts:
            self.contexts[job_id].cancel_event.set()
            self.store.update("job", job_id, {"detail": "cancelling"})
        return self.get(job_id)

    def _notify(self, job_id: str, summary: str | None) -> None:
        j = self.get(job_id)
        sid = j.get("session_id")
        if not sid or self.store.get("session", sid) is None:
            return
        lines = [f"Job {job_id} ({j['kind']}: {j['title']}) {j['state']}."]
        if j.get("error") and j["state"] != "succeeded":
            lines.append(f"Error: {j['error']}")
        if summary:
            lines.append(summary)
        if j.get("cost_usd"):
            lines.append(f"Cost: ${j['cost_usd']:.2f}")
        try:
            self.project.services["runner"].prompt(sid, "\n".join(lines), delivery="queue", synthetic=True)
        except Exception as exc:  # the job result is recorded either way
            log.warning("could not notify session %s: %s", sid, exc)

    def close(self) -> None:
        self._closed = True
        for t in self.pollers.values():
            t.cancel()
        for ctx in self.contexts.values():
            ctx.cancel_event.set()


def _jsonable(x: Any) -> Any:
    import json

    try:
        json.dumps(x, default=str)
        return json.loads(json.dumps(x, default=str))
    except (TypeError, ValueError):
        return str(x)


def job_dir(state: Path) -> Path:
    return state / "jobs"
