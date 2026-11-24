from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from opentrain import trainfile
from opentrain.record import RunRecord
from opentrain.registry import Registry
from opentrain.state import ensure, write_json
from opentrain.tasks import Controls, HeldoutStore, Store, Task, task_toml
from opentrain.web import api
from opentrain.web.server import make_server

SECRET = "SENTINEL-heldout-instruction-7f3a91"


def _task_folder(tmp: Path, task_id: str, instruction: str) -> Path:
    t = Task(task_id=task_id, instruction=instruction, repo="/src/widget", base_commit="a" * 40,
             reference_commit="b" * 40, test_command="pytest -q", test_files=("tests/test_app.py",),
             family="app", created_at="2026-11-01T10:00:00+00:00")
    f = tmp / "build" / task_id
    f.mkdir(parents=True)
    (f / "task.toml").write_text(task_toml(t))
    (f / "instruction.md").write_text(instruction)
    return f


@pytest.fixture
def state(tmp_path: Path) -> Path:
    s = ensure(tmp_path)
    (tmp_path / "Trainfile").write_text(trainfile.DEFAULT)
    ok = Controls(reference=True, noop=False, wrong=False, seconds=3.2)

    store = Store(s)
    store.add(_task_folder(tmp_path, "t-train-1", "Make f return 2 and update the docs."), ok)
    store.add(_task_folder(tmp_path, "t-train-2", "Add a --verbose flag."), ok)
    held = HeldoutStore(s)
    held.add(_task_folder(tmp_path, "h1", f"{SECRET} fix the parser"), ok)
    held.add(_task_folder(tmp_path, "h2", f"{SECRET} speed up the loader"), ok)

    rec = RunRecord(s)
    r1 = rec.open("manual", {"goal": "speed"})
    r1.add("data", task_ids=["t-train-1", "t-train-2"], hash="abc123")
    r1.add("recipe", method="sft", epochs=2, lora_rank=32)
    r1.add("spend", usd=1.25, what="training job", tokens=12000)
    r1.add("job", job_id="job-1", backend="tinker", state="succeeded")
    r1.add("checkpoint", checkpoint="tinker://ckpt/1")
    r1.add("eval", version="v1", task_id="h1", passed=True, seconds=40.0)
    r1.add("eval", version="v1", task_id="h2", passed=False, seconds=90.0)
    r1.add("gate", promote=True, reasons=["first version: 50% pass on 2 held-out tasks"], version="v1")
    r1.add("note", text="h2 times out on the loader; try more loader sessions")
    r1.add("closed", status="done")
    reg = Registry(s)
    reg.add(run_id=r1.run_id, base_model="Qwen/Qwen3-8B", backend="tinker", checkpoint="tinker://ckpt/1")
    reg.promote("v1", "first version")

    r2 = rec.get(rec.open("watch", {"goal": "speed"}).run_id)
    r2.add("spend", usd=0.5, what="sampling")
    reg.add(run_id=r2.run_id, base_model="Qwen/Qwen3-8B", backend="tinker", checkpoint="tinker://ckpt/2",
            scores={"pass_rate": 0.5, "median_seconds": 20.0})
    reg.promote("v2", "beats v1")
    write_json(s / "heldout" / "scores" / "v2.json",
               [{"task_id": "h1", "passed": False, "seconds": 50.0, "instruction": SECRET},
                {"task_id": "h2", "passed": True, "seconds": 20.0}])
    write_json(s / "jobs" / "job-1.json", {"state": "succeeded", "cost_usd": 1.2, "progress": 1.0,
                                           "logs": ["step 1 loss 1.2", "step 2 loss 0.9"]})
    (s / "serving").mkdir()
    rows = [{"at": "2026-11-24T10:00:00+00:00", "model": "opentrain/v2"}] * 3 + [
        {"at": "2026-11-24T10:00:00+00:00", "model": "claude-opus", "fallback": True}]
    (s / "serving" / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "{torn")
    write_json(s / "bench" / "bench-20261120T000000Z.json", {
        "at": "20261120T000000Z", "yield": {
            "sessions": 10, "verified_candidates": 4, "repairable": 2, "yield_per_100": 40.0,
            "funnel": [{"step": "instruction", "label": "no instruction", "dropped": 2, "remaining": 8},
                       {"step": "edits", "label": "no code changed", "dropped": 4, "remaining": 4}],
            "skip_details": {"no code changed: x": 4}},
        "time": {"sessions": 10, "hours": {"model": 1.0, "tool": 2.0, "human": 3.0, "idle": 0.5},
                 "model_share_of_working_time_pct": 33.3}})
    write_json(s / "races" / "race-1.json", {"winner": "v2", "contenders": [
        {"name": "v1", "pass_rate": 0.5, "tasks": [{"task_id": "h1", "passed": True, "instruction": SECRET}]},
        {"name": "v2", "pass_rate": 0.5, "tasks": [{"task_id": "h1", "passed": False}]}]})
    sk = tmp_path / "skills" / "triage"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: triage\ndescription: Sort failing tasks\n---\n# Triage\nBody.\n")
    return s


# ---------------------------------------------------------------- views


def test_home(state):
    h = api.home(state)
    assert h["share"]["requests"] == 4 and h["share"]["ours"] == 3 and h["share"]["share"] == 0.75
    assert h["current"]["version"] == "v2"
    assert h["budget"]["dollars"] == 20.0 and h["budget"]["committed_usd"] == 0.0
    assert h["last_note"]["text"].startswith("h2 times out")
    assert h["tasks"] == {"training": 2, "heldout": 2}


def test_runs_and_run(state):
    rs = api.runs(state)["runs"]
    assert len(rs) == 2
    done = next(r for r in rs if r["status"] == "done")
    assert done["spend_usd"] == 1.25 and done["gate"]["decision"] == "promoted"
    assert done["job"]["state"] == "succeeded" and done["job"]["logs"][-1] == "step 2 loss 0.9"
    assert done["versions"] == ["v1"]
    live = next(r for r in rs if r["status"] == "live")
    one = api.run(state, live["run_id"])
    assert one["verified"] and [e["kind"] for e in one["timeline"]] == ["opened", "spend"]
    new, closed = api.events_since(state, live["run_id"], 0)
    assert [e["seq"] for e in new] == [1] and not closed
    with pytest.raises(api.NotFound):
        api.run(state, "../models")


def test_evals_flips(state):
    e = api.evals(state)
    assert (e["a"], e["b"]) == ("v1", "v2")
    by = {r["task_id"]: r for r in e["rows"]}
    assert by["h1"]["change"] == "regressed" and by["h2"]["change"] == "fixed"
    assert by["h1"]["delta_seconds"] == 10.0
    assert e["counts"]["regressed"] == 1 and e["counts"]["fixed"] == 1
    assert set(by["h1"]) >= {"a_passed", "b_passed"} and "instruction" not in by["h1"]


def test_versions(state):
    v = api.versions(state)
    assert [x["version"] for x in v["versions"]] == ["v1", "v2"]
    assert v["current"] == "v2" and v["revert_to"] == "v1"
    v1 = v["versions"][0]
    assert v1["pass_rate"] == 0.5 and v1["trained_on"]["tasks"] == 2
    assert v1["trained_on"]["recipe"]["method"] == "sft"
    assert v["versions"][1]["median_seconds"] == 20.0


def test_data_skills_policy_race(state):
    d = api.data(state)
    assert d["bench"]["yield"]["sessions"] == 10 and len(d["bench"]["funnel"]) == 2
    assert {t["task_id"] for t in d["tasks"]} == {"t-train-1", "t-train-2"}
    assert d["built"]["heldout"] == 2
    assert api.task(state, "t-train-1")["instruction"].startswith("Make f return 2")
    with pytest.raises(api.NotFound):
        api.task(state, "h1")
    sk = api.skills(state)["skills"]
    assert sk[0]["name"] == "triage" and sk[0]["description"] == "Sort failing tasks"
    p = api.policy(state)
    assert p["exists"] and p["parsed"]["goal"] == "speed" and p["error"] is None
    race = api.race(state)
    assert race["file"] == "race-1.json" and race["race"]["winner"] == "v2"


def test_empty_state_views(tmp_path):
    s = tmp_path / ".opentrain"
    assert api.home(s)["current"] is None
    assert api.runs(s) == {"runs": []}
    assert api.evals(s)["choices"] == []
    assert api.versions(s)["versions"] == []
    assert api.data(s)["bench"] is None
    assert api.race(s)["race"] is None
    assert not s.exists(), "reading views must not create the state folder"


# ---------------------------------------------------------------- server


@pytest.fixture
def server(state):
    srv = make_server(state, 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    h = {"Host": f"127.0.0.1:{srv.server_address[1]}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    out = r.status, r.getheader("Content-Type") or "", r.read().decode()
    c.close()
    return out


def _post(srv, path, body, token=True):
    return _req(srv, "POST", path, body, {"X-OpenTrain-Token": srv.token} if token else {})


GETS = ["/api/home", "/api/runs", "/api/evals", "/api/evals?a=v1&b=v2", "/api/versions", "/api/race",
        "/api/data", "/api/skills", "/api/policy", "/api/tasks/t-train-1"]


def test_get_routes(server, state):
    run_id = api.runs(state)["runs"][0]["run_id"]
    for path in GETS + [f"/api/runs/{run_id}"]:
        status, ctype, body = _req(server, "GET", path)
        assert status == 200, (path, body)
        assert ctype.startswith("application/json")
        json.loads(body)
    status, ctype, html = _req(server, "GET", "/")
    assert status == 200 and "text/html" in ctype
    assert server.token in html and "__OPENTRAIN_TOKEN__" not in html
    for asset in ("/app.js", "/app.css"):
        assert _req(server, "GET", asset)[0] == 200
    assert _req(server, "GET", "/api/runs/nope")[0] == 404
    assert _req(server, "GET", "/api/tasks/h1")[0] == 404
    assert _req(server, "GET", "/api/home", headers={"Host": "evil.example:80"})[0] == 403


def test_post_requires_token(server, state):
    assert _post(server, "/api/versions/revert", {}, token=False)[0] == 403
    assert _req(server, "POST", "/api/versions/revert", {}, {"X-OpenTrain-Token": "wrong"})[0] == 403
    assert _req(server, "POST", "/api/policy", {"text": "goal  fix\n"},
                {"X-OpenTrain-Token": server.token, "Origin": "http://evil.example"})[0] == 403
    assert Registry(state).current().version == "v2"


def test_revert(server, state):
    status, _, body = _post(server, "/api/versions/revert", {"why": "test"})
    assert status == 200, body
    assert json.loads(body) == {"reverted": "v2", "current": "v1"}
    assert Registry(state).current().version == "v1"
    assert Registry(state).get("v2").decision == "reverted"
    status, _, body = _post(server, "/api/versions/revert", {})
    assert status == 200 and json.loads(body)["current"] is None
    assert _post(server, "/api/versions/revert", {})[0] == 409  # nothing is live


def test_policy_post(server, state):
    path = state.parent / "Trainfile"
    before = path.read_text()
    status, _, body = _post(server, "/api/policy", {"text": "goal  fastest\n"})
    assert status == 400 and "goal must be one of" in json.loads(body)["error"]
    status, _, body = _post(server, "/api/policy", {"text": "bogus key  1\n"})
    assert status == 400 and json.loads(body)["invalid"]
    assert path.read_text() == before
    new = before.replace("goal         speed", "goal         fix")
    status, _, body = _post(server, "/api/policy", {"text": new})
    assert status == 200, body
    assert json.loads(body)["parsed"]["goal"] == "fix"
    assert trainfile.load(path).goal == "fix"


def test_heldout_content_never_served(server, state):
    run_ids = [r["run_id"] for r in api.runs(state)["runs"]]
    paths = ["/", *GETS, *(f"/api/runs/{r}" for r in run_ids), "/api/tasks/h1", "/api/tasks/h2",
             "/api/evals?a=v2&b=v1", "/api/evals?a=base&b=v2"]
    for p in paths:
        _, _, body = _req(server, "GET", p)
        assert SECRET not in body, p
    # the pure views too, including the summarised Data and Race screens
    for view in (api.home, api.runs, api.versions, api.data, api.race, api.skills, api.policy, api.evals):
        assert SECRET not in json.dumps(view(state), default=str)


def test_sse_streams_new_events(server, state):
    live = next(r for r in api.runs(state)["runs"] if r["status"] == "live")
    run = RunRecord(state).get(live["run_id"])
    c = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    c.request("GET", f"/api/runs/{run.run_id}/events?after=1",
              headers={"Host": f"127.0.0.1:{server.server_address[1]}"})
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type").startswith("text/event-stream")
    run.add("note", text="streamed")
    run.add("closed", status="done")
    text = r.read().decode()  # the server ends the stream once the run closes
    c.close()
    assert '"streamed"' in text and "event: end" in text
    assert "id: 2" in text and "id: 3" in text and "id: 1\n" not in text


def test_version_scores_shapes():
    assert api.normalize_scores({"h1": True, "h2": {"passed": False, "seconds": 3}})["h2"]["seconds"] == 3.0
    assert api.normalize_scores({"scores": [{"task_id": "x", "passed": True}]})["x"]["passed"] is True
    assert api.normalize_scores({"pass_rate": 0.5}) == {}



def test_real_writer_shapes(state):
    """The shapes evaluator.py, serving.py, backends/ and race.py write."""
    write_json(state / "heldout" / "scores" / "claude@frontier.json", {
        "name": "claude@frontier", "runner": "claude", "tasks": 2, "passed": 2, "pass_rate": 1.0,
        "median_seconds": 30.0, "scores": [{"task_id": "h1", "passed": True, "seconds": 30.0, "tokens": 9},
                                           {"task_id": "h2", "passed": True, "seconds": 31.0, "tokens": 9}]})
    e = api.evals(state, "claude@frontier", "v2")
    assert e["a"] == "claude@frontier" and e["counts"]["regressed"] == 1
    with (state / "serving" / "requests.jsonl").open("a") as fh:
        fh.write("\n" + json.dumps({"at": "2026-11-24T11:00:00+00:00", "model": "opentrain/current",
                                    "version": "v2", "handoff": True, "status": 200}) + "\n")
        fh.write(json.dumps({"at": "2026-11-24T11:00:00+00:00", "model": "opentrain/current",
                             "status": 502}) + "\n")
    share = api.home(state)["share"]
    assert share["requests"] == 5 and share["ours"] == 3
    write_json(state / "jobs" / "tinker-1.json", {"job_id": "tinker-1", "state": "running", "progress": 0.4})
    (state / "jobs" / "tinker-1.log").write_text("a\nb\n")
    run = RunRecord(state).open("manual", {})
    run.add("job", job_id="tinker-1", backend="tinker")
    run.add("eval", model="opentrain/v9", scores=[{"task_id": "h1", "passed": True}])
    assert api.run(state, run.run_id)["job"]["logs"] == ["a", "b"]
    write_json(state / "races" / "race-2.json", {"task_id": "h1", "lanes": {
        "frontier": {"name": "frontier", "model": "claude", "passed": True, "seconds": 90.0}}})
    import os
    os.utime(state / "races" / "race-2.json", (2e9, 2e9))
    assert api.race(state)["race"]["lanes"]["frontier"]["seconds"] == 90.0
