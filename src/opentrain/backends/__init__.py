"""Trainer backends and the job store they share.

A training job outlives the CLI call that launched it: ``opentrain train`` returns, and later
``opentrain job ps`` (a different process) must still answer. So every backend keeps its job state
in one place on disk, not in memory:

    .opentrain/jobs/<job_id>.json     the job: spec, state, progress, cost so far, checkpoint
    .opentrain/jobs/<job_id>.log      what the worker printed
    .opentrain/jobs/<job_id>.cancel   present once someone asked to cancel

Only the worker process writes the job file while it runs; anyone else asks it to stop by
creating the cancel flag, so there is never a second writer. When the worker dies without
reporting (machine slept, OOM kill), ``status`` notices the dead pid and says so instead of
showing "running" forever.

Backends are registered by name in ``BACKENDS``; each factory imports its module lazily so
listing backends never imports torch or the tinker SDK.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..plugs import JobStatus, TrainerBackend, TrainSpec

_DONE = ("succeeded", "failed", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def missing_modules(*names: str) -> list[str]:
    """Which of these importable top-level packages are not installed (without importing them)."""
    out = []
    for n in names:
        try:
            found = importlib.util.find_spec(n) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(n)
    return out


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def spec_to_dict(spec: TrainSpec) -> dict[str, Any]:
    d = asdict(spec)
    d["data"] = str(spec.data)
    return d


def spec_from_dict(d: dict[str, Any]) -> TrainSpec:
    return TrainSpec(**{**d, "data": Path(d["data"])})


class JobStore:
    """The on-disk job folder. Safe to use from the CLI and from a detached worker at once."""

    def __init__(self, state: Path):
        self.root = Path(state) / "jobs"

    def path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def log_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.log"

    def cancel_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.cancel"

    def new(self, backend: str, spec: TrainSpec, **fields: Any) -> dict[str, Any]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        job = {
            "job_id": f"{backend}-{stamp}-{uuid.uuid4().hex[:6]}",
            "backend": backend,
            "state": "queued",
            "spec": spec_to_dict(spec),
            "created_at": _now(),
            "updated_at": _now(),
            "pid": None,
            "progress": 0.0,
            "cost_usd": 0.0,
            "checkpoint": None,
            "detail": "",
            **fields,
        }
        self.save(job)
        return job

    def save(self, job: dict[str, Any]) -> None:
        job["updated_at"] = _now()
        p = self.path(job["job_id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(job, indent=2, default=str))
        tmp.replace(p)

    def load(self, job_id: str) -> dict[str, Any]:
        p = self.path(job_id)
        if not p.exists():
            matches = [q for q in self.root.glob(f"{job_id}*.json")]
            if len(matches) != 1:
                raise KeyError(f"no single job matches {job_id!r}")
            p = matches[0]
        return json.loads(p.read_text())

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        job = self.load(job_id)
        job.update(fields)
        self.save(job)
        return job

    def jobs(self) -> list[dict[str, Any]]:
        out = [json.loads(p.read_text()) for p in self.root.glob("*.json")]
        return sorted(out, key=lambda j: j["created_at"], reverse=True)

    def log(self, job_id: str, line: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_path(job_id).open("a") as fh:
            fh.write(f"{_now()}  {line}\n")

    def logs(self, job_id: str, tail: int | None = 200) -> list[str]:
        p = self.log_path(self.load(job_id)["job_id"])
        if not p.exists():
            return []
        lines = p.read_text().splitlines()
        return lines[-tail:] if tail else lines

    def request_cancel(self, job_id: str) -> None:
        job = self.load(job_id)
        if job["state"] in _DONE:
            return
        self.cancel_path(job["job_id"]).write_text(_now())
        self.log(job["job_id"], "cancel requested")

    def cancel_requested(self, job_id: str) -> bool:
        return self.cancel_path(job_id).exists()

    def status(self, job_id: str) -> JobStatus:
        """The job as last reported, corrected for a worker that is gone."""
        job = self.load(job_id)
        jid = job["job_id"]
        if job["state"] not in _DONE and not pid_alive(job.get("pid")):
            if self.cancel_requested(jid):
                job = self.update(jid, state="cancelled", detail=(job.get("detail") or "") + " cancelled")
            elif job.get("pid"):
                job = self.update(jid, state="failed",
                                  detail=f"worker (pid {job['pid']}) exited without reporting; see job logs {jid}")
        return JobStatus(job_id=jid, state=job["state"], cost_usd=float(job.get("cost_usd") or 0.0),
                         checkpoint=job.get("checkpoint"), progress=job.get("progress"),
                         detail=job.get("detail") or "", logs=self.logs(jid, tail=20),
                         loss=job.get("loss") if isinstance(job.get("loss"), (int, float)) else None)

    def spawn(self, module: str, job_id: str) -> int:
        """Start ``python -m <module> <job.json>`` detached, so the job survives this process exiting.

        The child inherits the environment (that is how ``TINKER_API_KEY`` reaches it); keys are
        never written to the job file.
        """
        log = self.log_path(job_id).open("a")
        proc = subprocess.Popen([sys.executable, "-m", module, str(self.path(job_id))],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
        log.close()
        self.update(job_id, pid=proc.pid)
        return proc.pid


def worker_main(argv: list[str], run: Callable[[JobStore, dict[str, Any]], None]) -> int:
    """Shared entry for detached workers: load the job, run it, record any crash as ``failed``."""
    if len(argv) != 1:
        print("usage: python -m <worker> <.opentrain/jobs/<job_id>.json>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    store = JobStore(path.parent.parent)
    job = json.loads(path.read_text())
    jid = job["job_id"]
    store.update(jid, state="running", pid=os.getpid())
    try:
        run(store, store.load(jid))
    except Exception as e:  # noqa: BLE001 - every failure must land in the job file, not just stderr
        store.log(jid, f"failed: {type(e).__name__}: {e}")
        store.update(jid, state="failed", detail=f"{type(e).__name__}: {e}")
        raise
    return 0


# ---------------------------------------------------------------- registry


def _tinker(state: Path) -> TrainerBackend:
    from .tinker import TinkerBackend

    return TinkerBackend(state)


def _local(state: Path) -> TrainerBackend:
    from .local import LocalBackend

    return LocalBackend(state)


def _fireworks(state: Path) -> TrainerBackend:
    from .fireworks import FireworksBackend

    return FireworksBackend(state)


def _fake(state: Path) -> TrainerBackend:
    from .fake import FakeBackend

    return FakeBackend(state)


BACKENDS: dict[str, Callable[[Path], TrainerBackend]] = {
    "tinker": _tinker,
    "local": _local,
    "fireworks": _fireworks,
    "fake": _fake,
}


def get_backend(name: str, state: Path) -> TrainerBackend:
    try:
        factory = BACKENDS[name]
    except KeyError:
        raise KeyError(f"no trainer backend {name!r}; known: {', '.join(BACKENDS)}") from None
    return factory(Path(state))


def list_backends(state: Path | None = None) -> list[tuple[str, bool, str]]:
    """Every backend with whether it can run here and, if not, what to install or set."""
    from ..state import state_dir

    st = Path(state) if state else state_dir()
    out = []
    for name in BACKENDS:
        ok, why = get_backend(name, st).available()
        out.append((name, ok, why))
    return out
