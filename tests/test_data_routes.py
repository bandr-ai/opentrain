"""The data API the web app uses: datasets, uploads, evals, the playground, versions and the gate."""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from opentrain.client import ClientError
from opentrain.core.providers.scripted import ScriptedProvider
from opentrain.registry import Registry


def _wait_job(c, kind="eval", timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = [j for j in c.get("/api/job") if j["kind"] == kind]
        if jobs and jobs[0]["state"] != "running":
            return jobs[0]
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_upload_browse_grade_eval(live):
    c = live.client(live.proj)
    content = "".join(json.dumps({"question": f"q{i}", "answer": "done"}) + "\n" for i in range(40))
    r = c.post("/api/dataset/qa/upload", {"filename": "qa.jsonl", "content": content, "description": "toy"})
    assert r["added_train"] + r["added_heldout"] == 40 and r["added_heldout"] > 0
    lst = c.get("/api/dataset")
    assert lst[0]["name"] == "qa" and lst[0]["description"] == "toy"
    d = c.get("/api/dataset/qa", limit=5)
    assert len(d["rows"]) == 5 and d["total"] == r["added_train"]
    assert c.get("/api/dataset/qa", q="q1")["total"] >= 1
    with pytest.raises(ClientError):
        c.patch("/api/dataset/qa", {"grader": {"type": "nope"}})
    assert c.patch("/api/dataset/qa", {"grader": {"type": "exact"}, "system": "Answer with one word."})["grader"] == \
        {"type": "exact"}

    ScriptedProvider.register("web-good", [])
    c.post("/api/dataset/qa/eval", {"model": "scripted/web-good"})
    job = _wait_job(c)
    assert job["state"] == "succeeded" and job["result"]["passed"] == job["result"]["rows"]
    ev = c.get(f"/api/eval/{job['result']['eval']}")
    assert ev["split"] == "heldout" and all("answer" not in r and "prompt" not in r for r in ev["rows"])

    ScriptedProvider.register("web-train", [])
    c.post("/api/dataset/qa/eval", {"model": "scripted/web-train", "split": "train", "limit": 3})
    time.sleep(0.05)
    job = _wait_job(c)
    tr = c.get(f"/api/eval/{job['result']['eval']}")
    assert tr["split"] == "train" and tr["rows"][0]["prompt"].startswith("q") and tr["rows"][0]["answer"].strip()
    assert ScriptedProvider.named("web-train").requests[0].system == "Answer with one word."

    evs = c.get("/api/eval", dataset="qa")
    assert {e["target"] for e in evs} == {"scripted/web-good", "scripted/web-train"}

    removed = c.post("/api/dataset/qa/rows/delete", {"ids": [d["rows"][0]["id"]]})["removed"]
    assert removed == 1
    c.delete("/api/dataset/qa")
    assert c.get("/api/dataset") == []


def test_hf_import_needs_a_dataset(live):
    c = live.client(live.proj)
    with pytest.raises(ClientError) as e:
        c.post("/api/dataset/x/hf", {})
    assert "required" in str(e.value)


def test_try_and_versions_and_gate(live):
    c = live.client(live.proj)
    ScriptedProvider.register("web-try", [["hello there"]])
    r = c.post("/api/try", {"model": "scripted/web-try", "prompt": "hi"})
    assert r["answer"].strip() == "hello there"
    with pytest.raises(ClientError):
        c.post("/api/try", {"model": "nonsense", "prompt": "hi"})

    content = "".join(json.dumps({"prompt": f"p{i}", "completion": "done"}) + "\n" for i in range(40))
    c.post("/api/dataset/g/upload", {"filename": "g.jsonl", "content": content})
    c.patch("/api/dataset/g", {"grader": {"type": "exact"}})
    state = live.proj / ".opentrain"
    reg = Registry(state)
    v = reg.add(run_id=None, base_model="b", backend="fake", checkpoint="fake://x")
    reg.update(v.version, dataset="g")
    assert c.get("/api/version")["versions"][0]["dataset"] == "g"
    with pytest.raises(ClientError):  # no evals yet
        c.post(f"/api/version/{v.version}/promote", {})

    # a candidate that answers right and a baseline that does not, on the same held-out rows
    from opentrain import evals

    n = c.get("/api/dataset/g")["counts"]["heldout"]
    ScriptedProvider.register("cand", [])
    ScriptedProvider.register("base", [["wrong"]] * n)
    rec = evals.run(state, live.proj, "g", "scripted/cand")
    (state / "evals" / f"{rec['id']}.json").write_text(json.dumps({**rec, "target": v.version}))
    evals.run(state, live.proj, "g", "scripted/base")
    gate = c.get("/api/gate", dataset="g", candidate=v.version, baseline="scripted/base")
    assert gate["promote"], gate
    d = c.post(f"/api/version/{v.version}/promote", {"baseline": "scripted/base"})
    assert d["promote"] and c.get("/api/version")["current"] == v.version


def test_web_app_is_served(live):
    with urllib.request.urlopen(live.url + "/") as r:
        html = r.read().decode()
    assert r.status == 200 and '<div id="root">' in html
    asset = html.split('src="')[1].split('"')[0]
    with urllib.request.urlopen(live.url + asset) as r:  # assets need no sign-in; they hold no data
        assert r.status == 200 and "javascript" in r.headers["Content-Type"]
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(live.url + "/assets/../../server/app.py")
    assert e.value.code in (403, 404)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(live.url + "/api/dataset")
    assert e.value.code == 401


def test_board_walks_the_loop(live):
    c = live.client(live.proj)
    b = c.get("/api/board")
    assert b["current"] == "goal" and b["dataset"] is None and [s["key"] for s in b["stages"]] == \
        ["goal", "data", "grade", "baseline", "train", "prove", "ship"]
    (live.proj / "Trainfile").write_text("goal  sort tickets into four labels\n")
    content = "".join(json.dumps({"question": f"q{i}", "answer": "done"}) + "\n" for i in range(30))
    c.post("/api/dataset/bd/upload", {"filename": "bd.jsonl", "content": content})
    c.patch("/api/dataset/bd", {"grader": {"type": "exact"}})
    b = c.get("/api/board")
    st = {s["key"]: s for s in b["stages"]}
    assert st["goal"]["state"] == "done" and st["goal"]["summary"] == "sort tickets into four labels"
    assert st["data"]["state"] == "done" and b["current"] == "baseline" and b["budget"]["limit_usd"] > 0
    ScriptedProvider.register("bd-good", [])
    c.post("/api/dataset/bd/eval", {"model": "scripted/bd-good"})
    _wait_job(c)
    b = c.get("/api/board")
    assert b["scoreboard"][0]["target"] == "scripted/bd-good" and b["scoreboard"][0]["score"] == 1.0
    assert b["current"] == "train"
    g = c.get("/api/board/grid", dataset="bd")
    assert g["rows"] and "prompt" not in g["rows"][0] and g["columns"][0]["target"] == "scripted/bd-good"
    g = c.get("/api/board/grid", dataset="bd", reveal="1")
    r = g["rows"][0]
    assert r["prompt"].startswith("q") and r["reference"] == "done" and r["cells"]["scripted/bd-good"]["answer"]


def test_an_outside_agent_gets_the_tools_and_the_rules(live):
    """`opentrain mcp`: Claude Code calls OpenTrain's tools; held-out stays sealed, calls show in a session."""
    c = live.client(live.proj)
    names = {t["name"] for t in c.get("/api/outside/tools")}
    assert {"datasets", "dataset_eval", "dataset_train", "dataset_recipe", "goal_set"} <= names
    assert not names & {"shell", "edit", "write", "read", "propose", "question", "subagent"}, names & {"shell", "edit", "write", "read", "propose", "question", "subagent"}
    content = "".join(json.dumps({"question": f"q{i}", "answer": "yes"}) + "\n" for i in range(30))
    c.post("/api/dataset/oa/upload", {"filename": "oa.jsonl", "content": content})
    r = c.post("/api/outside/call", {"name": "dataset_recipe", "input": {"name": "oa"}})
    assert not r["is_error"] and "epochs" in r["output"]
    r = c.post("/api/outside/call", {"name": "goal_set", "input": {"goal": "answer yes reliably"}})
    assert not r["is_error"]
    assert "answer yes reliably" in (live.proj / "Trainfile").read_text()
    s = [s for s in c.get("/api/session") if (s.get("metadata") or {}).get("outside")]
    assert len(s) == 1 and s[0]["agent"] == "outside"
    parts = [p for m in c.get(f"/api/session/{s[0]['id']}/message") for p in m["parts"] if p["type"] == "tool"]
    assert [p["tool"]["name"] for p in parts] == ["dataset_recipe", "goal_set"]
    assert all(p["tool"]["state"] == "completed" for p in parts)


def test_a_job_an_outside_agent_starts_wakes_no_model(live):
    c = live.client(live.proj)
    content = "".join(json.dumps({"question": f"q{i}", "answer": "done"}) + "\n" for i in range(30))
    c.post("/api/dataset/ow/upload", {"filename": "ow.jsonl", "content": content})
    c.patch("/api/dataset/ow", {"grader": {"type": "exact"}})  # no model judge: the test must not need Claude Code
    ScriptedProvider.register("ow-good", [])
    r = c.post("/api/outside/call", {"name": "dataset_eval", "input": {"name": "ow", "model": "scripted/ow-good"}})
    assert "job_wait" in r["output"] and "system reminder" not in r["output"]
    jid = r["output"].split("Started job ")[1].split(" ")[0]
    w = c.post("/api/outside/call", {"name": "job_wait", "input": {"job_id": jid, "timeout": 30}})
    assert '"state": "succeeded"' in w["output"]
    sid = next(s["id"] for s in c.get("/api/session") if (s.get("metadata") or {}).get("outside"))
    time.sleep(0.3)
    msgs = c.get(f"/api/session/{sid}/message")
    assert any(m["role"] == "synthetic" for m in msgs)
    assert all(m["role"] != "assistant" or m.get("agent") == "outside" for m in msgs)
    assert not any(p["type"] == "text" for m in msgs if m["role"] == "assistant" for p in m["parts"])


def test_an_outside_agent_imports_a_hugging_face_dataset(live, monkeypatch):
    """Claude Code over MCP can pull GSM8K: a dataset download is not the general webfetch tool."""
    import opentrain.datasets as ds

    rows = [{"question": f"What is {i} + {i}?", "answer": f"{i} plus {i} is {2 * i}.\n#### {2 * i}"} for i in range(40)]
    monkeypatch.setattr(ds, "read_hf", lambda *a, **k: (rows, {"hf": "openai/gsm8k", "rows": len(rows)}))
    c = live.client(live.proj)
    r = c.post("/api/outside/call", {"name": "dataset_import", "input": {
        "name": "gsm8k", "hf": "openai/gsm8k", "input_key": "question", "output_key": "answer"}})
    assert not r["is_error"], r["output"]
    out = json.loads(r["output"])
    assert out["added_train"] + out["added_heldout"] == 40
    assert out["suggested_grader"]["type"] == "number" and "dataset_edit" in out["next"]
    assert "webfetch" not in {t["name"] for t in c.get("/api/outside/tools")}


def test_an_outside_agent_picks_the_trainer_and_model(live):
    c = live.client(live.proj)
    r = c.post("/api/outside/call", {"name": "trainer_set", "input": {"trainer": "modal", "base_model": "Qwen/Qwen3.5-4B"}})
    assert not r["is_error"], r["output"]
    text = (live.proj / "Trainfile").read_text()
    assert "trainer      modal" in text and "base model   Qwen/Qwen3.5-4B" in text and "budget" in text
    r = c.post("/api/outside/call", {"name": "trainer_set", "input": {"trainer": "fake"}})
    assert r["is_error"]
