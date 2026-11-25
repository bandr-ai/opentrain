import json
import sys

import pytest

from opentrain.core import integrity, sandbox
from opentrain.core.providers.scripted import ScriptedProvider
from opentrain.registry import Registry
from opentrain.tasks import Task, task_toml

INSTR = ("Rename the fetch_user helper to load_account across the billing service and update every caller so "
         "that the invoices page and the nightly reconciliation job keep working after the change lands")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def held_task(state, tid="h1", instruction=INSTR):
    t = Task(task_id=tid, instruction=instruction, repo="/r", base_commit="a", reference_commit="b",
             test_command="pytest")
    f = state / "heldout" / tid
    (f / "solution").mkdir(parents=True)
    (f / "task.toml").write_text(task_toml(t))
    (f / "instruction.md").write_text(instruction)
    (f / "solution" / "solution.patch").write_text("diff --git a/x b/x\n")
    return t


def test_contamination_is_found(tmp_path):
    state = tmp_path / ".opentrain"
    held_task(state)
    clean = integrity.contaminated(state, "an unrelated training example about pagination limits " * 20)
    assert clean == []
    hits = integrity.contaminated(state, json.dumps({"messages": [{"role": "user", "content": INSTR}]}))
    assert hits and hits[0]["task_id"] == "h1" and hits[0]["what"] == "instruction"


def test_provenance_flags(tmp_path, monkeypatch):
    state = tmp_path / ".opentrain"
    held_task(state)
    v = Registry(state).add(run_id=None, base_model="other/model", backend="fake", checkpoint="ck")
    data = state / "data"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text(json.dumps({"tasks": [{"task_id": "h1"}]}))
    (data / "train.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": INSTR}]}) + "\n")
    flags = integrity.provenance(state, v.version, "Qwen/Qwen3-8B")
    text = " | ".join(flags)
    assert "only 1 held-out tasks" in text and "base model" in text
    assert "held-out tasks in the training set: h1" in text and "reproduces held-out tasks" in text
    monkeypatch.setenv("OPENTRAIN_MIN_HELDOUT", "1")
    (data / "manifest.json").write_text(json.dumps({"tasks": []}))
    (data / "train.jsonl").write_text("{}\n")
    v2 = Registry(state).add(run_id=None, base_model="Qwen/Qwen3-8B", backend="fake", checkpoint="ck2")
    assert integrity.provenance(state, v2.version, "Qwen/Qwen3-8B") == []


def test_propose_refuses_a_flagged_version(live):
    state = live.proj / ".opentrain"
    held_task(state)
    v = Registry(state).add(run_id=None, base_model="someone/else", backend="fake", checkpoint="ck")
    c = live.client(live.proj)
    ScriptedProvider.register("prov", [[{"tool": "version_propose", "input": {"version": v.version}}], ["ok"]])
    s = c.post("/api/session", {"model": "scripted/prov",
                                "permissions": [{"action": "*", "resource": "*", "effect": "allow"}]})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    t = [p["tool"] for m in c.get(f"/api/session/{s['id']}/message") for p in m["parts"] if p["type"] == "tool"][0]
    out = json.loads(t["output"])
    assert out["promoted"] is False and any("base model" in r for r in out["reasons"])
    assert Registry(state).get(v.version).decision.startswith("rejected: ")
    assert Registry(state).current() is None


@pytest.mark.skipif(sandbox.kind() not in ("seatbelt", "bwrap"), reason="no OS sandbox here")
def test_the_shell_cannot_read_heldout_even_indirectly(live):
    state = live.proj / ".opentrain"
    held_task(state)
    live.hub.project(live.proj)
    # The path is assembled at run time, so no check on the command text can see it: only the OS wall can.
    code = f"import os; print(open(os.path.join({str(state)!r}, 'held' + 'out', 'h1', 'instruction.md')).read())"
    c = live.client(live.proj)
    ScriptedProvider.register("sbx", [[{"tool": "shell", "input": {"command": f"{sys.executable} -c \"{code}\""}}],
                                      ["ok"]])
    s = c.post("/api/session", {"model": "scripted/sbx",
                                "permissions": [{"action": "*", "resource": "*", "effect": "allow"}]})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 30})
    t = [p["tool"] for m in c.get(f"/api/session/{s['id']}/message") for p in m["parts"] if p["type"] == "tool"][0]
    assert "billing service" not in t["output"] and ("not permitted" in t["output"] or "Permission" in t["output"])


def test_the_evaluator_process_reports_refusals(tmp_path):
    from opentrain.core.tools.train import evaluate_in_process
    from opentrain.core.tool import ToolError

    state = tmp_path / ".opentrain"
    state.mkdir()
    with pytest.raises(ToolError):
        evaluate_in_process(state, "v99", None)
