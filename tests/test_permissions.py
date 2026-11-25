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


def test_plan_agent_cannot_spend_or_write_outside_plans(live):
    p = live.hub.project(live.proj)
    plan = p.services["agents"].get("plan")
    perms = p.services["permissions"]
    visible = {t.name for t in p.services["runner"].tools_for(plan)}
    assert {"status", "train_estimate", "read", "write", "question"} <= visible
    assert not ({"job_start", "sample_run", "tasks_mine", "version_propose", "edit_x"} & visible)
    assert perms.effect(plan, None, "write", ".opentrain/plans/a.md") == "allow"
    assert perms.effect(plan, None, "write", "src/a.py") == "deny"
    c, s = _start(live, "p2", [[{"tool": "write", "input": {"path": "src/a.py", "content": "x"}}],
                               [{"tool": "job_start", "input": {}}], ["done"]], agent="plan")
    tools = _tools(c, s["id"])
    assert tools[0]["state"] == "error" and "not allowed for the plan agent" in tools[0]["error"]
    assert tools[1]["state"] == "error" and "may not use job_start" in tools[1]["error"]
    assert not (live.proj / "src" / "a.py").exists()


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
