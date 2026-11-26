"""A Modal training job survives the laptop losing its connection: the app runs detached and the
worker follows the job's event log until it ends."""

from __future__ import annotations

import contextlib
import socket
import sys
import types

import pytest

from opentrain.backends import JobStore, modal_backend
from opentrain.plugs import TrainSpec

EVENTS = [{"event": "started", "gpu": "NVIDIA L40S", "versions": {}},
          {"event": "step", "step": 1, "total": 3, "loss": 0.5, "seconds": 1},
          {"event": "step", "step": 2, "total": 3, "loss": 0.4, "seconds": 2},
          {"event": "step", "step": 3, "total": 3, "loss": 0.3, "seconds": 3},
          {"event": "done", "adapter": "/vol/adapters/x", "seconds": 4}]


def _fake(monkeypatch, tmp_path, drop_after: int | None, final=None):
    events = list(EVENTS[:-1]) + [final or EVENTS[-1]]
    log = {f"j/{i}": ev for i, ev in enumerate(events)} | {"j/n": len(events)}
    stopped = []

    class Dict(dict):
        @staticmethod
        def from_name(name, create_if_missing=False):
            return Dict(log)

    def remote_gen(jid, spec, rows):
        for i, ev in enumerate(events):
            if drop_after is not None and i == drop_after:
                raise socket.gaierror(8, "nodename nor servname provided, or not known")
            yield ev

    app = types.SimpleNamespace(app_id="ap-test", run=lambda detach=False: contextlib.nullcontext())
    mod = types.SimpleNamespace(app=app, train=types.SimpleNamespace(remote_gen=remote_gen),
                                train_loop=types.SimpleNamespace(remote_gen=remote_gen))
    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Dict=Dict, enable_output=contextlib.nullcontext))
    monkeypatch.setattr(modal_backend, "load_app", lambda **env: mod)
    monkeypatch.setattr(modal_backend, "stop_app", lambda a: stopped.append(a))
    monkeypatch.setattr(modal_backend.time, "sleep", lambda s: None)
    data = tmp_path / "d.jsonl"
    data.write_text('{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}\n')
    store = JobStore(tmp_path)
    job = store.new("modal", TrainSpec(base_model="Qwen/Qwen2.5-0.5B-Instruct", data=data))
    job["job_id"] = "j"
    store.save(job)
    return store, job, stopped


def test_a_dropped_connection_does_not_lose_the_job(monkeypatch, tmp_path):
    store, job, stopped = _fake(monkeypatch, tmp_path, drop_after=2)
    modal_backend.run(store, job)
    got = store.load("j")
    assert got["state"] == "succeeded" and got["checkpoint"] == "modal://j" and got["progress"] == 1.0
    assert "lost the stream from Modal" in store.log_path("j").read_text()
    assert stopped and stopped[-1] == "ap-test"


def test_a_job_that_fails_after_the_drop_still_reports_why(monkeypatch, tmp_path):
    store, job, stopped = _fake(monkeypatch, tmp_path, drop_after=1, final={"event": "error", "error": "OOM"})
    with pytest.raises(modal_backend.ModalError, match="OOM"):
        modal_backend.run(store, job)
    assert stopped


def test_a_clean_run_stops_its_app(monkeypatch, tmp_path):
    store, job, stopped = _fake(monkeypatch, tmp_path, drop_after=None)
    modal_backend.run(store, job)
    assert store.load("j")["state"] == "succeeded" and stopped == ["ap-test"]


def test_over_budget_a_job_ends_early_and_keeps_its_adapter(tmp_path, monkeypatch):
    """Out of budget mid-run: first ask the job to stop at its next step (adapter kept); cancel only if
    it runs on well past the limit."""
    from opentrain import mcp, trainfile
    from opentrain.backends import JobStatus

    state = tmp_path / ".opentrain"
    state.mkdir()
    srv = mcp.Server(state, "trainer", None, trainfile.parse("trainer  modal\nbudget  $2 a week"))
    calls = []
    cost = {"usd": 1.3}

    class B:
        def status(self, jid):
            return JobStatus(job_id=jid, state="running", progress=0.5, cost_usd=cost["usd"])

        def stop_early(self, jid):
            calls.append("stop")
            return True

        def cancel(self, jid):
            calls.append("cancel")

    monkeypatch.setattr(srv, "_job_event", lambda jid: {"backend": "modal", "estimate_usd": 1.0, "reservation": "r"})
    monkeypatch.setattr("opentrain.backends.get_backend", lambda name, st: B())
    monkeypatch.setattr(srv.ledger, "remaining", lambda run_id=None: 0.1)
    out = srv.t_train_status("j")
    assert calls == ["stop"] and "keeping the adapter" in out["stopping"]
    srv.t_train_status("j")
    assert calls == ["stop"]  # asked once; give it its next step
    cost["usd"] = 1.3 * 1.25 + 0.5
    out = srv.t_train_status("j")
    assert calls == ["stop", "cancel"] and out["cancelled"] == "over budget"


def test_the_trainer_stops_when_asked():
    from opentrain.backends.gpu_train import _stopper

    said, flag = [], {"v": False}
    stop = _stopper(lambda: flag["v"], said.append)
    assert not stop()
    flag["v"] = True
    assert not stop()  # checked at most every 10 s
    stop_now = _stopper(lambda: True, said.append)
    assert stop_now() and stop_now.fired and "saving the adapter" in said[0]
    assert not _stopper(lambda: 1 / 0, said.append)()  # a failed lookup is not a stop


def test_the_reason_a_job_was_stopped_reaches_the_agent(tmp_path, monkeypatch):
    from opentrain import mcp, trainfile
    from opentrain.backends import JobStatus

    state = tmp_path / ".opentrain"
    state.mkdir()
    srv = mcp.Server(state, "trainer", None, trainfile.parse("trainer  modal\nbudget  $2 a week"))
    st = {"state": "running", "cost": 3.0}

    class B:
        def status(self, jid):
            return JobStatus(job_id=jid, state=st["state"], progress=0.5, cost_usd=st["cost"])

        def cancel(self, jid):
            pass

    monkeypatch.setattr(srv, "_job_event", lambda jid: {"backend": "x", "estimate_usd": 1.0, "reservation": "r"})
    monkeypatch.setattr("opentrain.backends.get_backend", lambda name, s: B())
    monkeypatch.setattr(srv.ledger, "remaining", lambda run_id=None: 0.0)
    monkeypatch.setattr(srv.ledger, "settle", lambda rid, usd: None)
    assert srv.t_train_status("j")["cancelled"] == "over budget"
    st["state"] = "cancelled"
    out = srv.t_train_status("j")
    assert "budget had left" in out["why"] and "$3.00" in out["why"]
