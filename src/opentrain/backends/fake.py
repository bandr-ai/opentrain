"""A trainer that trains nothing, for tests and demos only.

It exists so the loop (launch, status, registry, gate, serve) can be exercised end to end without
keys, GPUs or money. Because a fake checkpoint must never be mistaken for a trained model it is
guarded twice: it refuses to run unless ``OPENTRAIN_FAKE_BACKEND=1`` is set, and every status it
reports says the weights were simulated. Its checkpoints are ``fake://`` URIs that no serving
backend will accept.

``spec.extra["hold"] = True`` leaves the job running until cancelled, to exercise cancellation.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..plugs import JobStatus, TrainSpec
from . import JobStore

ENV = "OPENTRAIN_FAKE_BACKEND"
SIMULATED = "simulated — no weights were trained"


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


class FakeBackend:
    name = "fake"

    def __init__(self, state: Path):
        self.store = JobStore(state)

    def available(self) -> tuple[bool, str]:
        if not enabled():
            return False, f"test/demo only: set {ENV}=1 to enable ({SIMULATED})"
        return True, f"enabled by {ENV}=1 ({SIMULATED})"

    def _guard(self) -> None:
        if not enabled():
            raise RuntimeError(f"the fake backend trains nothing and is off; set {ENV}=1 for tests or demos")

    def estimate(self, spec: TrainSpec) -> float:
        self._guard()
        return 0.0

    def launch(self, spec: TrainSpec) -> str:
        self._guard()
        job = self.store.new(self.name, spec, detail=SIMULATED)
        jid = job["job_id"]
        self.store.log(jid, f"{SIMULATED}: base {spec.base_model}, data {spec.data}")
        if spec.extra.get("hold"):
            self.store.update(jid, state="running", detail=f"{SIMULATED}; holding until cancelled")
        else:
            self.store.update(jid, state="succeeded", progress=1.0, checkpoint=f"fake://{jid}",
                              detail=SIMULATED)
        return jid

    def status(self, job_id: str) -> JobStatus:
        st = self.store.status(job_id)
        if SIMULATED not in st.detail:
            st = JobStatus(**{**st.__dict__, "detail": f"{st.detail} ({SIMULATED})".strip()})
        return st

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)
