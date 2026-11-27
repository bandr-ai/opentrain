import os

import pytest

from opentrain import trainfile


def test_connect_stores_and_exports_a_key(live):
    c = live.client(live.proj)
    rows = {r["id"]: r for r in c.get("/api/integration")}
    assert rows["tinker"]["connected"] is False and rows["tinker"]["env"] == "TINKER_API_KEY"
    out = c.post("/api/integration/tinker/connect", {"key": "tk-secret-1234"})
    assert out["connected"] and out["hint"] == "1234" and "key" not in out
    assert os.environ["TINKER_API_KEY"] == "tk-secret-1234"
    from opentrain.core import credentials

    assert oct(credentials.path().stat().st_mode & 0o777) == "0o600"
    assert "tk-secret" not in str(c.get("/api/integration"))
    assert c.delete("/api/integration/tinker")["removed"] is True
    assert "TINKER_API_KEY" not in os.environ
    with pytest.raises(Exception):
        c.post("/api/integration/tinker/connect", {"key": "  "})


def test_models_and_default(live):
    c = live.client(live.proj)
    m = c.get("/api/model")
    refs = {r["ref"] for r in m["models"]}
    assert {"anthropic/claude-opus-5-5", "claude-code/default"} <= refs
    c.post("/api/model/default", {"ref": "openrouter/qwen/qwen3-coder"})
    assert c.get("/api/model")["default"] == "openrouter/qwen/qwen3-coder"
    s = c.post("/api/session", {})
    assert c.get("/api/model", session=s["id"])["session"] == "openrouter/qwen/qwen3-coder"


def test_trainer_and_base_model_edit_the_trainfile(live):
    c = live.client(live.proj)
    t = c.get("/api/trainer")
    names = [x["name"] for x in t["trainers"]]
    assert "tinker" in names and "local" in names
    assert any(b["id"] == "Qwen/Qwen3-8B" for b in next(x for x in t["trainers"] if x["name"] == "tinker")["base_models"])
    out = c.post("/api/trainer", {"trainer": "local", "base_model": "Qwen/Qwen3-0.6B"})
    assert out["policy"]["trainer"] == "local" and out["policy"]["base_model"] == "Qwen/Qwen3-0.6B"
    text = (live.proj / "Trainfile").read_text()
    assert text.startswith("# Trainfile") and "trainer      local" in text
    with pytest.raises(Exception):
        c.post("/api/trainer", {"trainer": "nope"})


def test_set_value_keeps_comments(tmp_path):
    (tmp_path / "Trainfile").write_text("# mine\ngoal  speed   # why\nbudget $5 a week\n")
    trainfile.set_value(tmp_path, "goal", "cost")
    trainfile.set_value(tmp_path, "heldout", "20%")
    text = (tmp_path / "Trainfile").read_text()
    assert text.splitlines()[0] == "# mine" and "goal         cost" in text and "heldout      20%" in text
    tf = trainfile.for_project(tmp_path)
    assert tf.goal == "cost" and tf.heldout_fraction == 0.2
    with pytest.raises(trainfile.TrainfileError):
        trainfile.set_value(tmp_path, "heldout", "500%")


def test_datasources(live):
    ids = [d["id"] for d in live.client(live.proj).get("/api/datasource")]
    assert ids[0] == "claude-code" and "codex" in ids
