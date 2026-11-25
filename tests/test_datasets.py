"""Datasets, graders, evals and the dataset gate: the general (non-coding) path."""

from __future__ import annotations

import json
import time

import pytest

from opentrain import datasets as dsm
from opentrain import evals, graders
from opentrain.core.providers.scripted import ScriptedProvider
from opentrain.registry import Registry


def test_normalize_shapes():
    a = dsm.normalize({"question": "2+2?", "answer": "4"})
    assert [m["role"] for m in a["messages"]] == ["user", "assistant"] and a["messages"][1]["content"] == "4"
    b = dsm.normalize({"messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "hello"}]})
    assert dsm.reference_of(b) == "hello" and dsm.prompt_text(b) == "hi"
    c = dsm.normalize({"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]})
    assert dsm.reference_of(c) == "a"
    d = dsm.normalize({"body": "x", "reply": "y"}, input_key="body", output_key="reply", system="sys")
    assert d["messages"][0] == {"role": "system", "content": "sys"} and dsm.reference_of(d) == "y"
    e = dsm.normalize({"prompt": "unlabelled"})
    assert dsm.reference_of(e) is None
    with pytest.raises(dsm.DatasetError):
        dsm.normalize({"nothing": 1})


def test_parse_csv_json_jsonl():
    assert dsm.parse_bytes(b"prompt,completion\na,b\n", "x.csv") == [{"prompt": "a", "completion": "b"}]
    assert dsm.parse_bytes(b'[{"prompt":"a"}]', "x.json") == [{"prompt": "a"}]
    assert dsm.parse_bytes(b'{"prompt":"a"}\n{"prompt":"b"}\n', "x.jsonl") == [{"prompt": "a"}, {"prompt": "b"}]
    assert dsm.parse_bytes(b'{"data":[{"prompt":"a"}]}', "x.json") == [{"prompt": "a"}]


def _rows(n):
    return [{"prompt": f"what is {i} plus {i}?", "completion": str(2 * i)} for i in range(n)]


def test_split_is_deterministic_and_sealed(tmp_path):
    ds = dsm.Datasets(tmp_path)
    r = ds.add("math", _rows(100), heldout=0.3)
    assert r["added_train"] + r["added_heldout"] == 100 and 15 < r["added_heldout"] < 45
    again = ds.add("math", _rows(100))
    assert again["duplicates"] == 100 and again["added_train"] == 0
    train_ids = {x["id"] for x in ds.rows("math")}
    assert not train_ids & set(ds.heldout_ids("math"))
    assert (tmp_path / "heldout" / "datasets" / "math.jsonl").exists()
    assert "heldout" not in str(tmp_path / "datasets" / "math" / "train.jsonl").split("datasets")[1]
    other = dsm.Datasets(tmp_path / "b")
    other.add("math", _rows(100), heldout=0.3)
    assert set(other.heldout_ids("math")) == set(ds.heldout_ids("math"))


def test_label_remove_and_training_file(tmp_path):
    ds = dsm.Datasets(tmp_path)
    ds.add("q", [{"prompt": f"p{i}"} for i in range(10)], heldout=0.0)
    with pytest.raises(dsm.DatasetError, match="no labelled"):
        ds.training_file("q", tmp_path / "t.jsonl")
    first = ds.rows("q")[0]["id"]
    ds.label("q", first, "answer")
    m = ds.training_file("q", tmp_path / "t.jsonl")
    assert m["examples"] == 1 and json.loads((tmp_path / "t.jsonl").read_text())["messages"][-1]["content"] == "answer"
    assert ds.remove_rows("q", [first]) == 1 and ds.meta("q")["counts"]["train"] == 9


def test_bad_names(tmp_path):
    with pytest.raises(dsm.DatasetError):
        dsm.Datasets(tmp_path).add("../x", _rows(2))


def test_read_hf_pages():
    calls = []

    def fetch(url):
        calls.append(url)
        if "/splits?" in url:
            return {"splits": [{"config": "main", "split": "test"}, {"config": "main", "split": "train"}]}
        off = int(url.split("offset=")[1].split("&")[0])
        n = int(url.split("length=")[1])
        return {"rows": [{"row": {"question": f"q{off + i}", "answer": "a"}} for i in range(min(n, 250 - off))],
                "num_rows_total": 250}

    rows, src = dsm.read_hf("org/set", limit=1000, fetch=fetch)
    assert len(rows) == 250 and src == {"hf": "org/set", "config": "main", "split": "train"}
    with pytest.raises(dsm.DatasetError, match="available"):
        dsm.read_hf("org/set", split="nope", fetch=fetch)


@pytest.mark.parametrize("spec,answer,ref,ok", [
    ({"type": "exact"}, "  Paris ", "paris", True),
    ({"type": "exact"}, "Lyon", "Paris", False),
    ({"type": "contains"}, "The answer is Paris.", "paris", True),
    ({"type": "regex", "pattern": r"\byes\b"}, "Yes, it is", None, True),
    ({"type": "json"}, '```json\n{"a": 1, "b": [2]}\n```', '{"b":[2],"a":1}', True),
    ({"type": "json"}, "{bad", '{"a":1}', False),
    ({"type": "number"}, "so the total is 1,234.", "#### 1234", True),
    ({"type": "number"}, "42", "41", False),
])
def test_graders(spec, answer, ref, ok):
    assert graders.grade(graders.check(spec), "p", answer, ref).passed is ok


def test_grader_needs_reference_and_validates():
    with pytest.raises(graders.GraderError):
        graders.grade({"type": "exact"}, "p", "a", None)
    with pytest.raises(graders.GraderError):
        graders.check({"type": "vibes"})
    with pytest.raises(graders.GraderError):
        graders.check({"type": "regex", "pattern": "("})


def test_command_grader(tmp_path):
    spec = {"type": "command", "command": "python3 -c \"import json,sys; d=json.load(sys.stdin); "
                                         "print(1.0 if d['answer'].strip()==d['reference'] else 0.0)\""}
    assert graders.grade(spec, "p", "ok ", "ok", directory=tmp_path).passed
    assert not graders.grade(spec, "p", "no", "ok", directory=tmp_path).passed


def test_judge_grader_uses_model():
    ScriptedProvider.register("judge", [["SCORE: 8 | close enough"], ["nonsense"]])
    g = graders.grade({"type": "judge", "model": "scripted/judge"}, "p", "a", "b")
    assert g.passed and g.score == pytest.approx(0.8) and "close" in g.reason
    assert not graders.grade({"type": "judge", "model": "scripted/judge"}, "p", "a", "b").passed


def test_eval_run_and_gate(tmp_path):
    ds = dsm.Datasets(tmp_path)
    ds.add("echo", [{"prompt": f"say {i}", "completion": "done"} for i in range(40)], heldout=0.5)
    ds.update_meta("echo", grader={"type": "exact"})
    n = len(ds.heldout_ids("echo"))
    ScriptedProvider.register("good", [])        # answers "done" to everything
    ScriptedProvider.register("bad", [["nope"]] * n)
    good = evals.run(tmp_path, tmp_path, "echo", "scripted/good")
    bad = evals.run(tmp_path, tmp_path, "echo", "scripted/bad")
    assert good["summary"]["passed"] == n and bad["summary"]["passed"] == 0
    assert all("answer" not in r for r in good["rows"])  # content stays sealed
    with pytest.raises(dsm.DatasetError, match="sealed"):
        evals.answers(tmp_path, good["id"])
    assert evals.answers(tmp_path, good["id"], sealed_ok=True)[0]["answer"].strip() == "done"
    d = evals.decide(tmp_path, "echo", "scripted/good", "scripted/bad", min_rows=5)
    assert d["promote"] and len(d["fixed"]) == n
    d = evals.decide(tmp_path, "echo", "scripted/bad", "scripted/good", min_rows=5)
    assert not d["promote"] and any("regressed" in r for r in d["reasons"])
    assert not evals.decide(tmp_path, "echo", "scripted/good", "scripted/bad", min_rows=10_000)["promote"]


def test_contamination(tmp_path):
    ds = dsm.Datasets(tmp_path)
    ds.add("c", [{"prompt": " ".join(f"word{i}_{j}" for j in range(40)), "completion": "x"} for i in range(20)],
           heldout=0.5)
    held = ds.heldout_rows("c")[0]
    f = tmp_path / "train.jsonl"
    f.write_text(json.dumps({"messages": held["messages"]}))
    assert evals.contamination(tmp_path, "c", f)[0]["id"] == held["id"]


def test_registry_dataset_field_is_backward_compatible(tmp_path):
    reg = Registry(tmp_path)
    v = reg.add(run_id=None, base_model="b", backend="fake", checkpoint="fake://1")
    assert v.dataset is None
    assert reg.update(v.version, dataset="echo").dataset == "echo"


# ---------------------------------------------------------------------- through the engine


def _tool_call(live, name, tool, args):
    from test_tools import call

    return call(live, name, tool, args)


def test_dataset_tools_end_to_end(live, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(live.proj / "xdg"))
    (live.proj / "qa.jsonl").write_text("".join(json.dumps({"q": f"capital {i}", "a": "done"}) + "\n" for i in range(30)))
    t, _ = _tool_call(live, "dt1", "dataset_import", {"name": "qa", "path": "qa.jsonl", "input_key": "q",
                                                      "output_key": "a"})
    assert t["state"] == "completed", t
    out = json.loads(t["output"])
    assert out["counts"]["train"] + out["counts"]["heldout"] == 30
    t, _ = _tool_call(live, "dt2", "dataset_edit", {"name": "qa", "grader": {"type": "exact"}})
    assert json.loads(t["output"])["grader"] == "exact match"
    t, _ = _tool_call(live, "dt3", "dataset_show", {"name": "qa", "limit": 2})
    shown = json.loads(t["output"])
    assert len(shown["rows"]) == 2 and "sealed" in shown["note"]
    ScriptedProvider.register("answerer", [])
    t, sid = _tool_call(live, "dt4", "dataset_eval", {"name": "qa", "model": "scripted/answerer"})
    assert t["state"] == "completed" and "Started job" in t["output"]
    c = live.client(live.proj)
    for _ in range(100):
        jobs = [j for j in c.get("/api/job") if j["kind"] == "eval"]
        if jobs and jobs[0]["state"] != "running":
            break
        time.sleep(0.1)
    assert jobs[0]["state"] == "succeeded", jobs[0]
    assert jobs[0]["result"]["passed"] == jobs[0]["result"]["rows"]
    t, _ = _tool_call(live, "dt5", "read", {"path": ".opentrain/heldout/datasets/qa.jsonl"})
    assert t["state"] == "error" and "sealed" in t["error"]
