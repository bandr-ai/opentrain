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
