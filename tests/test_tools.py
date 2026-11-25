import json
import threading
import time

import pytest

from opentrain.core.providers.scripted import ScriptedProvider


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


ALLOW = [{"action": "*", "resource": "*", "effect": "allow"}]


def call(live, name, tool, args, then=None, model=None, agent=None):
    """Run one scripted turn that calls ``tool`` and return its tool part."""
    c = live.client(live.proj)
    ScriptedProvider.register(name, [[{"tool": tool, "input": args}], then or ["ok"]])
    s = c.post("/api/session", {"model": f"scripted/{name}", "permissions": ALLOW,
                                **({"agent": agent} if agent else {})})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 30})
    msgs = c.get(f"/api/session/{s['id']}/message")
    return [p for m in msgs for p in m["parts"] if p["type"] == "tool"][0]["tool"], s["id"]


def test_read_write_edit_cycle(live):
    (live.proj / "a.txt").write_text("one\ntwo\n")
    t, _ = call(live, "rw1", "edit", {"path": "a.txt", "old": "two", "new": "2"})
    assert t["state"] == "error" and "read a.txt before" in t["error"]
    c = live.client(live.proj)
    ScriptedProvider.register("rw2", [[{"tool": "read", "input": {"path": "a.txt"}}],
                                      [{"tool": "edit", "input": {"path": "a.txt", "old": "two", "new": "2"}}],
                                      [{"tool": "write", "input": {"path": "sub/b.txt", "content": "new"}}], ["done"]])
    s = c.post("/api/session", {"model": "scripted/rw2", "permissions": ALLOW})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 30})
    tools = [p["tool"] for m in c.get(f"/api/session/{s['id']}/message") for p in m["parts"] if p["type"] == "tool"]
    assert [t["state"] for t in tools] == ["completed"] * 3, tools
    assert "     1\tone" in tools[0]["output"]
    assert (live.proj / "a.txt").read_text() == "one\n2\n" and (live.proj / "sub/b.txt").read_text() == "new"


def test_sealed_paths_are_unreachable(live):
    held = live.proj / ".opentrain" / "heldout" / "t1"
    held.mkdir(parents=True)
    (held / "instruction.md").write_text("secret")
    t, _ = call(live, "seal1", "read", {"path": ".opentrain/heldout/t1/instruction.md"})
    assert t["state"] == "error" and "sealed" in t["error"]
    t, _ = call(live, "seal2", "shell", {"command": "cat .opentrain/heldout/t1/instruction.md"})
    assert t["state"] == "error" and "sealed" in t["error"]
    t, _ = call(live, "seal3", "grep", {"pattern": "secret"})
    assert "heldout" not in (t.get("output") or "")


def test_shell_runs_in_the_project_and_reports_exit(live):
    t, _ = call(live, "sh1", "shell", {"command": "pwd; exit 3", "description": "where am I"})
    assert t["state"] == "completed" and str(live.proj.resolve()) in t["output"] and "[exit code 3]" in t["output"]
    assert t["title"] == "where am I"
    t, _ = call(live, "sh2", "shell", {"command": "sleep 5", "timeout": 1})
    assert "timed out" in t["output"]


def test_glob_grep_todo_skill(live):
    (live.proj / "x.py").write_text("def hello():\n    pass\n")
    t, _ = call(live, "g1", "glob", {"pattern": "*.py"})
    assert t["output"].strip() == "x.py"
    t, _ = call(live, "g2", "grep", {"pattern": "def hello"})
    assert "x.py:1:" in t["output"]
    t, sid = call(live, "g3", "todo", {"todos": [{"content": "mine", "status": "completed"},
                                                 {"content": "train", "status": "pending"}]})
    assert t["title"] == "1 to do"
    assert live.client(live.proj).get(f"/api/session/{sid}/todo")[1]["content"] == "train"
    t, _ = call(live, "g4", "skill", {"name": "goal/speed"})
    assert t["state"] == "completed" and "goal/speed" in t["output"]
    names = [s["name"] for s in live.client(live.proj).get("/api/skill")]
    assert "data/shortest" in names and "train/sft" in names


def test_question_waits_for_the_user(live):
    c = live.client(live.proj)
    ScriptedProvider.register("q1", [[{"tool": "question", "input": {"questions": [
        {"question": "Which base model?", "options": ["Qwen3-8B", "Qwen3-32B"]}]}}], ["thanks"]])
    s = c.post("/api/session", {"model": "scripted/q1"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    for _ in range(50):
        pending = c.get("/api/question")
        if pending:
            break
        time.sleep(0.05)
    assert pending[0]["questions"][0]["question"] == "Which base model?"
    c.post(f"/api/session/{s['id']}/question/{pending[0]['id']}/reply", {"answers": [["Qwen3-32B"]]})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    t = [p["tool"] for m in c.get(f"/api/session/{s['id']}/message") for p in m["parts"] if p["type"] == "tool"][0]
    assert "A: Qwen3-32B" in t["output"]


def test_subagent_runs_in_a_child_session(live):
    c = live.client(live.proj)
    ScriptedProvider.register("sub", [[{"tool": "subagent", "input": {"agent": "monitor", "prompt": "check jobs",
                                                                      "description": "watch"}}],
                                      ["No jobs are running."], ["parent done"]])
    s = c.post("/api/session", {"model": "scripted/sub"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    t = [p["tool"] for m in c.get(f"/api/session/{s['id']}/message") for p in m["parts"] if p["type"] == "tool"][0]
    assert t["state"] == "completed" and "No jobs are running." in t["output"]
    children = [x for x in c.get("/api/session", children="1") if x.get("parent_id") == s["id"]]
    assert len(children) == 1 and children[0]["agent"] == "monitor"
    t, _ = call(live, "sub2", "subagent", {"agent": "plan", "prompt": "x"})
    assert t["state"] == "error" and "no subagent 'plan'" in t["error"]


def test_status_init_and_versions(live):
    t, _ = call(live, "st1", "status", {})
    st = json.loads(t["output"])
    assert st["trainfile"].startswith("none") and st["tasks"] == {"training": 0, "heldout": 0}
    t, _ = call(live, "st2", "init", {"goal": "cost", "budget": "$5 a week", "heldout": "25%"})
    assert t["state"] == "completed"
    text = (live.proj / "Trainfile").read_text()
    assert "goal         cost" in text and "$5 a week" in text and "heldout      25%" in text
    st = json.loads(call(live, "st3", "status", {})[0]["output"])
    assert st["policy"]["goal"] == "cost" and st["policy"]["heldout"] == 0.25
    t, _ = call(live, "st4", "init", {})
    assert t["state"] == "error"
    assert json.loads(call(live, "st5", "versions", {})[0]["output"])["versions"] == []


def test_task_show_refuses_heldout(live):
    held = live.proj / ".opentrain" / "heldout" / "secret-task"
    held.mkdir(parents=True)
    (held / "task.toml").write_text("")
    t, _ = call(live, "ts1", "task_show", {"task_id": "secret-task"})
    assert t["state"] == "error" and "held out" in t["error"]


def test_a_local_job_reports_back_into_the_session(live):
    p = live.hub.project(live.proj)
    c = live.client(live.proj)
    ScriptedProvider.register("job1", [["waiting for the job"], ["saw the result"]])
    s = c.post("/api/session", {"model": "scripted/job1"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    gate = threading.Event()

    def work(job):
        job.progress(0.5, "halfway")
        gate.wait(5)
        return {"n": 3}

    j = p.services["jobs"].start_local(s["id"], "mine", "test job", work, summarize=lambda r: f"made {r['n']} tasks")
    time.sleep(0.2)
    assert c.get(f"/api/job/{j['id']}")["detail"] == "halfway"
    gate.set()
    for _ in range(100):
        if c.get(f"/api/job/{j['id']}")["state"] == "succeeded" and not p.services["runner"].busy(s["id"]) \
                and len(c.get(f"/api/session/{s['id']}/message")) >= 4:
            break
        time.sleep(0.05)
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    msgs = c.get(f"/api/session/{s['id']}/message")
    syn = [m for m in msgs if m["role"] == "synthetic"]
    assert syn and "made 3 tasks" in syn[0]["parts"][0]["text"]
    assert msgs[-1]["parts"][0]["text"].strip() == "saw the result"
