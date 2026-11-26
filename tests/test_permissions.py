import time

import pytest

from opentrain.core.permission import evaluate, matches
from opentrain.core.providers.scripted import ScriptedProvider


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_rule_semantics():
    rules = [{"action": "*", "resource": "*", "effect": "allow"},
             {"action": "edit", "resource": "src/**", "effect": "ask"},
             {"action": "edit", "resource": "src/gen/**", "effect": "deny"},
             {"action": "job.*", "resource": "*", "effect": "ask"}]
    assert evaluate(rules, "read", "x") == "allow"
    assert evaluate(rules, "edit", "src/a.py") == "ask"
    assert evaluate(rules, "edit", "src/gen/b.py") == "deny"
    assert evaluate(rules, "job.start", "tinker") == "ask"
    assert matches({"action": "read", "resource": "*.env"}, "read", "config/.env")
    assert matches({"action": "write", "resource": ".opentrain/plans/**"}, "write", ".opentrain/plans/a.md")


def _start(live, name, turns, agent="train", **body):
    c = live.client(live.proj)
    ScriptedProvider.register(name, turns)
    s = c.post("/api/session", {"model": f"scripted/{name}", "agent": agent, **body})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    return c, s


def _pending(c, n=1):
    for _ in range(100):
        p = c.get("/api/permission")
        if len(p) >= n:
            return p
        time.sleep(0.05)
    raise AssertionError("no permission request appeared")


def _tools(c, sid):
    c.post(f"/api/session/{sid}/wait", {"timeout": 20})
    return [p["tool"] for m in c.get(f"/api/session/{sid}/message") for p in m["parts"] if p["type"] == "tool"]


def test_ask_once_reject_and_always(live):
    c, s = _start(live, "p1", [[{"tool": "shell", "input": {"command": "echo hi"}}],
                               [{"tool": "shell", "input": {"command": "echo again"}}],
                               [{"tool": "shell", "input": {"command": "rm -rf build"}}], ["done"]])
    req = _pending(c)[0]
    assert req["action"] == "shell" and req["resources"] == ["echo hi"] and req["agent"] == "train"
    c.post(f"/api/session/{s['id']}/permission/{req['id']}/reply", {"reply": "always"})
    req2 = _pending(c)[0]  # "always" saved "echo hi" only; "echo again" still asks
    assert req2["resources"] == ["echo again"]
    c.post(f"/api/session/{s['id']}/permission/{req2['id']}/reply", {"reply": "once"})
    req3 = _pending(c)[0]
    c.post(f"/api/session/{s['id']}/permission/{req3['id']}/reply", {"reply": "reject", "message": "not that"})
    tools = _tools(c, s["id"])
    assert [t["state"] for t in tools] == ["completed", "completed", "error"]
    assert "not that" in tools[2]["error"]
    assert [a["resource"] for a in c.get("/api/approval")] == ["echo hi"]
    # The saved approval now settles the same ask without a prompt.
    c2, s2 = _start(live, "p1b", [[{"tool": "shell", "input": {"command": "echo hi"}}], ["done"]])
    assert _tools(c2, s2["id"])[0]["state"] == "completed"


def _proposal(c, n=1):
    for _ in range(100):
        p = c.get("/api/proposal")
        if len(p) >= n:
            return p
        time.sleep(0.05)
    raise AssertionError("no proposal appeared")


def _dataset(c, name="pp"):
    import json

    content = "".join(json.dumps({"question": f"q{i}", "answer": "done"}) + "\n" for i in range(20))
    c.post(f"/api/dataset/{name}/upload", {"filename": "x.jsonl", "content": content})
    c.patch(f"/api/dataset/{name}", {"grader": {"type": "exact"}})


def test_an_approved_step_covers_its_gated_actions(live):
    c = live.client(live.proj)
    _dataset(c)
    ScriptedProvider.register("pp-eval", [])
    step = {"title": "Evaluate pp-eval on pp", "why": "baseline", "cost_usd": 0.01, "minutes": 1,
            "needs": ["spend"], "stage": "baseline", "dataset": "pp"}
    c, s = _start(live, "pp1", [[{"tool": "propose", "input": step}],
                                [{"tool": "dataset_eval", "input": {"name": "pp", "model": "scripted/pp-eval"}}],
                                ["done"]])
    prop = _proposal(c)[0]
    assert prop["title"] == step["title"] and prop["needs"] == ["spend"] and prop["cost_usd"] == 0.01
    assert c.get(f"/api/session/{s['id']}")["status"] in ("waiting", "busy")
    board = c.get("/api/board")
    assert board["proposals"][0]["id"] == prop["id"] and board["dataset"] == "pp"
    with pytest.raises(Exception):
        c.post(f"/api/session/{s['id']}/proposal/{prop['id']}/reply", {"reply": "change"})  # needs a message
    c.post(f"/api/session/{s['id']}/proposal/{prop['id']}/reply", {"reply": "approve"})
    tools = _tools(c, s["id"])
    assert [t["state"] for t in tools] == ["completed", "completed"]
    assert "Approved" in tools[0]["output"]
    assert c.get("/api/permission") == []  # the spend did not ask again


def test_a_skipped_step_is_not_done_and_ungranted_actions_still_ask(live):
    c = live.client(live.proj)
    _dataset(c, "pq")
    ScriptedProvider.register("pq-eval", [])
    c, s = _start(live, "pq1", [[{"tool": "propose", "input": {"title": "Train on pq", "needs": ["job.start"]}}],
                                ["ok, skipped"]])
    prop = _proposal(c)[0]
    c.post(f"/api/session/{s['id']}/proposal/{prop['id']}/reply", {"reply": "skip"})
    tools = _tools(c, s["id"])
    assert tools[0]["state"] == "error" and "skipped" in tools[0]["error"]
    # approving a step that needs only job.start does not cover spend
    c, s = _start(live, "pq2", [[{"tool": "propose", "input": {"title": "x", "needs": ["job.start"]}}],
                                [{"tool": "dataset_eval", "input": {"name": "pq", "model": "scripted/pq-eval"}}],
                                ["done"]])
    prop = _proposal(c)[0]
    c.post(f"/api/session/{s['id']}/proposal/{prop['id']}/reply", {"reply": "approve"})
    req = _pending(c)[0]
    assert req["action"] == "spend"
    c.post(f"/api/session/{s['id']}/permission/{req['id']}/reply", {"reply": "reject"})
    _tools(c, s["id"])


def test_propose_refuses_ungrantable_needs(live):
    c, s = _start(live, "pr1", [[{"tool": "propose", "input": {"title": "x", "needs": ["shell"]}}], ["done"]])
    t = _tools(c, s["id"])[0]
    assert t["state"] == "error" and "still ask on their own" in t["error"]


def test_unattended_sessions_never_wait(live):
    c, s = _start(live, "p3", [[{"tool": "shell", "input": {"command": "echo hi"}}], ["done"]],
                  metadata={"unattended": True})
    t = _tools(c, s["id"])[0]
    assert t["state"] == "error" and "not allowed" in t["error"]
    assert c.get("/api/permission") == []


def test_interrupt_cancels_a_pending_ask(live):
    c, s = _start(live, "p4", [[{"tool": "shell", "input": {"command": "echo hi"}}]])
    _pending(c)
    c.post(f"/api/session/{s['id']}/interrupt")
    assert c.get("/api/permission") == []
    assert _tools(c, s["id"])[0]["state"] == "error"


def test_engine_tokens_count_against_the_budget(live):
    (live.proj / "Trainfile").write_text("goal speed\nbudget $1 a week\n")
    c = live.client(live.proj)
    ScriptedProvider.register("p5", [["a"], ["b"], ["c"]], cost=0.6)
    s = c.post("/api/session", {"model": "scripted/p5"})
    for text in ("one", "two"):
        c.post(f"/api/session/{s['id']}/prompt", {"text": text})
        c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    msgs = c.get(f"/api/session/{s['id']}/message")
    assert "budget is used up" in (msgs[-1].get("error") or "")
    assert c.get(f"/api/session/{s['id']}")["cost"] == pytest.approx(1.2)


def test_unattended_proposals_never_wait(live):
    """`opentrain run --yes` approves proposed steps; an unattended run without it skips them, with a reason."""
    c, s = _start(live, "p6", [[{"tool": "propose", "input": {"title": "Import x", "needs": ["download"]}}], ["done"]],
                  metadata={"unattended": True, "auto_approve": ["*"]})
    t = _tools(c, s["id"])[0]
    assert t["state"] == "completed" and "Approved" in t["output"]
    c, s = _start(live, "p7", [[{"tool": "propose", "input": {"title": "Train", "needs": ["job.start"]}}], ["done"]],
                  metadata={"unattended": True})
    t = _tools(c, s["id"])[0]
    assert t["state"] == "error" and "nobody can approve" in t["error"]
    assert c.get("/api/proposal") == []


def test_unattended_questions_never_wait(live):
    c, s = _start(live, "p8", [[{"tool": "question", "input": {"questions": [
        {"question": "Which?", "options": ["a", "b"]}]}}], ["done"]], metadata={"unattended": True})
    t = _tools(c, s["id"])[0]
    assert t["state"] == "error" and "nobody can answer" in t["error"]
