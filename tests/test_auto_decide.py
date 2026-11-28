"""Asks that cannot strand the person, and small choices that do not wait for them.

- A turn that ends while a proposal or question is open (the agent's CLI died, say) withdraws it:
  nothing is left to click into a turn that no longer exists, and the session reads idle.
- A free step with no gated needs goes ahead after AUTO_SECONDS unless the person holds it; a step
  that spends or needs anything never does.
- A question or artifact decision with a default takes it after AUTO_SECONDS, unless held.
"""

import asyncio
import time

import pytest

from opentrain import artifacts as art
from opentrain.core import proposals, questions
from opentrain.core.providers.scripted import ScriptedProvider


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(proposals, "AUTO_SECONDS", 0.3)
    monkeypatch.setattr(questions, "AUTO_SECONDS", 0.3)


def _start(live, name, turns):
    c = live.client(live.proj)
    ScriptedProvider.register(name, turns)
    s = c.post("/api/session", {"model": f"scripted/{name}", "agent": "train"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    return c, s


def _until(fn, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("timed out")


def _tools(c, sid):
    c.post(f"/api/session/{sid}/wait", {"timeout": 20})
    return [p["tool"] for m in c.get(f"/api/session/{sid}/message") for p in m["parts"] if p["type"] == "tool"]


def test_a_free_step_goes_ahead_by_itself(live):
    c, s = _start(live, "a1", [[{"tool": "propose", "input": {"title": "Look at the data"}}], ["done"]])
    p = _until(lambda: c.get("/api/proposal"))[0]
    assert p["auto_at"]
    t = _tools(c, s["id"])[0]
    assert t["state"] == "completed" and "went ahead" in t["output"]


def test_a_step_that_spends_never_goes_ahead_by_itself(live):
    c, s = _start(live, "a2", [[{"tool": "propose", "input": {"title": "Evaluate the base", "needs": ["spend"],
                                                               "cost_usd": 0.15}}], ["done"]])
    p = _until(lambda: c.get("/api/proposal"))[0]
    assert not p.get("auto_at")
    time.sleep(0.8)
    assert c.get("/api/proposal")[0]["status"] == "pending"
    c.post(f"/api/session/{s['id']}/proposal/{p['id']}/reply", {"reply": "skip"})
    _tools(c, s["id"])


def test_holding_a_free_step_waits_for_the_person(live):
    c, s = _start(live, "a3", [[{"tool": "propose", "input": {"title": "Look at the data"}}], ["done"]])
    p = _until(lambda: c.get("/api/proposal"))[0]
    held = c.post(f"/api/session/{s['id']}/proposal/{p['id']}/reply", {"reply": "hold"})
    assert held["status"] == "pending" and not held.get("auto_at")
    time.sleep(0.8)
    assert c.get("/api/proposal")[0]["status"] == "pending"
    c.post(f"/api/session/{s['id']}/proposal/{p['id']}/reply", {"reply": "approve"})
    assert "went ahead" not in _tools(c, s["id"])[0]["output"]


def test_a_question_with_defaults_takes_them(live):
    c, s = _start(live, "a4", [[{"tool": "question", "input": {"questions": [
        {"question": "How many examples to show?", "options": ["3", "10"], "default": "3"}]}}], ["ok"]])
    t = _tools(c, s["id"])[0]
    assert t["state"] == "completed" and "A: 3" in t["output"] and "defaults were taken" in t["output"]


def test_a_question_without_defaults_waits(live):
    c, s = _start(live, "a5", [[{"tool": "question", "input": {"questions": [
        {"question": "Which Modal account?", "options": ["mine", "shared"]}]}}], ["ok"]])
    q = _until(lambda: c.get("/api/question"))[0]
    assert not q.get("auto_at")
    time.sleep(0.8)
    assert c.get("/api/question")
    c.post(f"/api/session/{s['id']}/question/{q['id']}/reply", {"answers": [["mine"]]})
    assert "A: mine" in _tools(c, s["id"])[0]["output"]


def test_holding_a_question_waits_for_the_person(live):
    c, s = _start(live, "a6", [[{"tool": "question", "input": {"questions": [
        {"question": "How many?", "options": ["3", "10"], "default": "3"}]}}], ["ok"]])
    q = _until(lambda: c.get("/api/question"))[0]
    c.post(f"/api/session/{s['id']}/question/{q['id']}/reply", {"hold": True})
    time.sleep(0.8)
    assert c.get("/api/question")
    c.post(f"/api/session/{s['id']}/question/{q['id']}/reply", {"answers": [["10"]]})
    assert "A: 10" in _tools(c, s["id"])[0]["output"]


def test_a_default_must_be_an_option(live):
    c, s = _start(live, "a7", [[{"tool": "question", "input": {"questions": [
        {"question": "How many?", "options": ["3", "10"], "default": "7"}]}}], ["ok"]])
    t = _tools(c, s["id"])[0]
    assert t["state"] == "error" and "not one of the options" in t["error"]


class _DiesWhileWaiting:
    """An agent provider whose CLI dies while its proposal is still open (as on a lost connection)."""
    kind = "agent"
    seen: list = []

    async def run_turn(self, runner, turn, agent, ref):
        msg = runner.store.create("message", {"session_id": turn.session_id, "role": "assistant", "agent": agent.name,
                                              "model": ref, "time": {"created": time.time()},
                                              "tokens": {"input": 0, "output": 0}, "cost": 0.0})
        turn.message_id = msg["id"]
        call = asyncio.get_running_loop().create_task(runner.external_tool_call(
            turn.session_id, "propose", {"title": "Evaluate the base", "needs": ["spend"], "cost_usd": 0.15}, "c1"))
        call.add_done_callback(lambda f: self.seen.append(f.result()))
        while not runner.project.services["proposals"].pending(turn.session_id):
            await asyncio.sleep(0.02)
        raise RuntimeError("API Error: Unable to connect to API")


def test_a_turn_that_ends_withdraws_its_open_step(live, monkeypatch):
    from opentrain.core import providers

    real = providers.get
    monkeypatch.setattr(providers, "get", lambda ref: _DiesWhileWaiting() if ref == "fake/dies" else real(ref))
    c = live.client(live.proj)
    s = c.post("/api/session", {"model": "fake/dies", "agent": "train"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    assert c.get("/api/proposal") == []
    assert _until(lambda: _DiesWhileWaiting.seen)[0].metadata.get("error")
    assert c.get(f"/api/session/{s['id']}")["status"] == "idle"


def test_artifact_default_and_hold(tmp_path):
    blocks = [{"type": "text", "text": "x"}]
    d = {"question": "Show more samples?", "options": ["no", "yes"], "default": "no"}
    a = art.save(tmp_path, kind="sample_page", title="Samples", blocks=blocks, decision=d)
    assert a["decision"]["auto_at"]
    assert art.decide_by_default(tmp_path, a["id"], a["version"])["decision"]["chosen"] == "no"
    assert art.decide_by_default(tmp_path, a["id"], a["version"]) is None  # already decided

    b = art.save(tmp_path, kind="sample_page", title="Samples", blocks=blocks, decision=d)
    art.hold(tmp_path, b["id"])
    assert art.decide_by_default(tmp_path, b["id"], b["version"]) is None
    assert not art.get(tmp_path, b["id"])["decision"].get("chosen")

    c = art.save(tmp_path, kind="sample_page", title="Samples", blocks=blocks, decision=d)
    art.save(tmp_path, kind="sample_page", title="Samples v2", blocks=blocks, decision=d, id=c["id"])
    assert art.decide_by_default(tmp_path, c["id"], c["version"]) is None  # a newer version replaced it

    with pytest.raises(art.ArtifactError):
        art.save(tmp_path, kind="sample_page", title="x", blocks=blocks,
                 decision={"question": "q", "options": ["a", "b"], "default": "c"})


def test_the_app_learns_when_it_runs_old_code(live, monkeypatch):
    from opentrain.server import dashboard

    c = live.client(live.proj)
    assert c.get("/api/build")["restart"] is False
    monkeypatch.setattr(dashboard, "STARTED", dashboard.STARTED - 3600)  # the code changed after start
    assert c.get("/api/build")["restart"] is True
