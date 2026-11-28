"""The person is asked only what is theirs to decide, once; everything else just happens.

- A turn that ends while a proposal or question is open (the agent's CLI died, say) withdraws it:
  nothing is left to click into a turn that no longer exists, and the session reads idle.
- A free step with no gated needs is never shown: it is approved at once. A step that spends waits.
- A question with a default is decided by it at once and never shown; only the rest is asked.
- An artifact never carries a default. A plan page's go-ahead (``approves``) grants the whole run to the
  session that made it, so its steps do not ask again.
"""

import asyncio
import time

import pytest

from opentrain import artifacts as art
from opentrain.core.providers.scripted import ScriptedProvider


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


def test_a_free_step_is_never_shown(live):
    c, s = _start(live, "a1", [[{"tool": "propose", "input": {"title": "Look at the data"}}], ["done"]])
    t = _tools(c, s["id"])[0]
    assert t["state"] == "completed" and "Not shown to the person" in t["output"]
    assert c.get("/api/proposal") == []


def test_a_step_that_spends_never_goes_ahead_by_itself(live):
    c, s = _start(live, "a2", [[{"tool": "propose", "input": {"title": "Evaluate the base", "needs": ["spend"],
                                                               "cost_usd": 0.15}}], ["done"]])
    p = _until(lambda: c.get("/api/proposal"))[0]
    time.sleep(0.5)
    assert c.get("/api/proposal")[0]["status"] == "pending"
    c.post(f"/api/session/{s['id']}/proposal/{p['id']}/reply", {"reply": "skip"})
    _tools(c, s["id"])


def test_a_question_with_a_default_is_decided_without_asking(live):
    c, s = _start(live, "a4", [[{"tool": "question", "input": {"questions": [
        {"question": "How many examples to show?", "options": ["3", "10"], "default": "3"}]}}], ["ok"]])
    t = _tools(c, s["id"])[0]
    assert t["state"] == "completed" and "A: 3" in t["output"] and "decided without asking" in t["output"]
    assert c.get("/api/question") == []


def test_only_the_questions_without_a_default_are_asked(live):
    c, s = _start(live, "a8", [[{"tool": "question", "input": {"questions": [
        {"question": "How many?", "options": ["3", "10"], "default": "3"},
        {"question": "Which Modal account?", "options": ["mine", "shared"]}]}}], ["ok"]])
    q = _until(lambda: c.get("/api/question"))[0]
    assert [x["question"] for x in q["questions"]] == ["Which Modal account?"]
    c.post(f"/api/session/{s['id']}/question/{q['id']}/reply", {"answers": [["shared"]]})
    out = _tools(c, s["id"])[0]["output"]
    assert "A: 3 (your default" in out and "A: shared" in out


def test_a_question_without_defaults_waits(live):
    c, s = _start(live, "a5", [[{"tool": "question", "input": {"questions": [
        {"question": "Which Modal account?", "options": ["mine", "shared"]}]}}], ["ok"]])
    q = _until(lambda: c.get("/api/question"))[0]
    time.sleep(0.5)
    assert c.get("/api/question")
    c.post(f"/api/session/{s['id']}/question/{q['id']}/reply", {"answers": [["mine"]]})
    assert "A: mine" in _tools(c, s["id"])[0]["output"]


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


def test_an_artifact_asks_no_default(tmp_path):
    blocks = [{"type": "text", "text": "x"}]
    with pytest.raises(art.ArtifactError, match="decide it yourself"):
        art.save(tmp_path, kind="sample_page", title="Samples", blocks=blocks,
                 decision={"question": "Show more?", "options": ["no", "yes"], "default": "no"})


def test_approves_must_name_gated_actions(tmp_path):
    blocks = [{"type": "text", "text": "x"}]
    for bad in ({"needs": ["shell"], "cost_usd": 1}, {"needs": [], "cost_usd": 1}, {"needs": ["spend"], "cost_usd": -1}):
        with pytest.raises(art.ArtifactError, match="approves"):
            art.save(tmp_path, kind="recipe_plan", title="Plan", blocks=blocks,
                     decision={"question": "Run this plan?", "options": ["Run it", "Not now"], "approves": bad})


def test_a_plans_go_ahead_grants_the_whole_run(live):
    c = live.client(live.proj)
    s = c.post("/api/session", {"model": "scripted/none", "agent": "train"})
    project = live.hub.project(live.proj)
    props = project.services["proposals"]
    d = {"question": "Run this plan?", "options": ["Run it", "Not now"],
         "approves": {"needs": ["spend", "upload", "job.start"], "cost_usd": 0.65}}
    a = art.save(project.state, kind="recipe_plan", title="Plan", blocks=[{"type": "text", "text": "x"}],
                 decision=d, session_id=s["id"])
    b = art.save(project.state, kind="recipe_plan", title="Plan", blocks=[{"type": "text", "text": "x"}],
                 decision=d, session_id=s["id"])

    c.post(f"/api/artifacts/{b['id']}/decide", {"option": "Not now"})
    assert not props.granted(s["id"], "job.start", {})

    c.post(f"/api/artifacts/{a['id']}/decide", {"option": "Run it"})
    assert props.granted(s["id"], "job.start", {}) and props.granted(s["id"], "upload", {})
    assert props.granted(s["id"], "spend", {"usd": 0.5})
    assert not props.granted(s["id"], "spend", {"usd": 0.6})  # past about the plan's total: asks again
    assert not props.granted(s["id"], "promote", {})  # not part of the plan
    with pytest.raises(Exception):
        c.post(f"/api/artifacts/{a['id']}/decide", {"option": "Run it"})  # decided once


def test_the_app_learns_when_it_runs_old_code(live, monkeypatch):
    from opentrain.server import dashboard

    c = live.client(live.proj)
    assert c.get("/api/build")["restart"] is False
    monkeypatch.setattr(dashboard, "STARTED", dashboard.STARTED - 3600)  # the code changed after start
    assert c.get("/api/build")["restart"] is True


def test_a_decision_is_not_made_on_an_older_version(tmp_path):
    blocks = [{"type": "text", "text": "x"}]
    d = {"question": "Run this plan?", "options": ["Run it", "Not now"]}
    a = art.save(tmp_path, kind="recipe_plan", title="Plan", blocks=blocks, decision=d)
    art.save(tmp_path, kind="recipe_plan", title="Plan v2", blocks=blocks, decision=d, id=a["id"])
    with pytest.raises(art.ArtifactError):
        art.decide(tmp_path, a["id"], "Run it", version=a["version"])
    assert not art.get(tmp_path, a["id"])["decision"].get("chosen")
