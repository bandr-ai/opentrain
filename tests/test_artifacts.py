"""Artifacts: versioned pages of blocks; a few ask the person one decision; two built from the project's data."""

from __future__ import annotations

import json

import pytest

from opentrain import artifacts as A


def test_an_artifact_is_versioned_and_decided(tmp_path):
    a = A.save(tmp_path, kind="recipe_plan", title="Plan", blocks=[{"type": "text", "text": "SFT then RL"}],
               decision={"question": "Run it?", "options": ["Run", "Change it"]})
    assert a["id"] == "recipe-plan-1" and a["version"] == 1
    b = A.save(tmp_path, kind="recipe_plan", id=a["id"], title="Plan v2", blocks=[{"type": "text", "text": "SFT only"}],
               decision={"question": "Run it?", "options": ["Run", "Change it"]})
    assert b["version"] == 2 and A.get(tmp_path, a["id"], 1)["title"] == "Plan"
    d = A.decide(tmp_path, a["id"], "Run", "go")
    assert d["decision"]["chosen"] == "Run" and A.get(tmp_path, a["id"])["decision"]["note"] == "go"
    with pytest.raises(A.ArtifactError, match="choose one of"):
        A.decide(tmp_path, a["id"], "Maybe")
    assert [x["id"] for x in A.list_all(tmp_path)] == ["recipe-plan-1"]


def test_any_new_kind_works_and_bad_input_says_why(tmp_path):
    a = A.save(tmp_path, kind="data_fix", title="Drop the AWS sessions?",
               blocks=[{"type": "list", "items": ["12 sessions"]}])
    assert a["kind"] == "data_fix"
    for blocks, msg in (([], "non-empty"), ([{"type": "pie"}], "type must be"), ([{"type": "table"}], "needs")):
        with pytest.raises(A.ArtifactError, match=msg):
            A.save(tmp_path, kind="x", title="t", blocks=blocks)
    with pytest.raises(A.ArtifactError, match="no artifact"):
        A.save(tmp_path, kind="x", title="t", blocks=[{"type": "text", "text": "x"}], links=["nope"])
    with pytest.raises(A.ArtifactError, match="is a data_fix"):
        A.save(tmp_path, kind="other", id=a["id"], title="t", blocks=[{"type": "text", "text": "x"}])


def _traces(state):
    d = state / "traces"
    d.mkdir(parents=True)
    idx = [{"trace": f"t{i}", "agent": "codex" if i % 2 else "claude-code", "repo": "/r/app" if i < 3 else None,
            "edits": 1 if i < 4 else 0} for i in range(6)]
    call = {"function": {"name": "bash", "arguments": json.dumps({"command": "pytest -q"})}}
    ex = [{"messages": [{"role": "user", "content": f"fix {i}"}, {"role": "assistant", "tool_calls": [call]}],
           "meta": {"outcome": {"ended_well": i != 0, "pushback": i == 0}}} for i in range(4)]
    for name, rows in (("index", idx), ("examples", ex), ("moments", [{"kind": "pushback"}])):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_data_card_shows_what_is_trained_and_why_some_is_left_out(tmp_path):
    with pytest.raises(A.ArtifactError, match="no traces imported"):
        A.data_card(tmp_path)
    _traces(tmp_path)
    a = A.data_card(tmp_path)
    stats = {i["label"]: i["value"] for i in a["blocks"][0]["items"]}
    assert stats["work sessions"] == 6 and stats["changed code"] == 4 and stats["requests to train on"] == 3
    left_out = next(b for b in a["blocks"] if b.get("title") == "Requests left out, and why")
    assert left_out["items"] == [{"label": "pushback after", "value": 1, "max": 4}]
    assert a["decision"] is None  # it shows; the agent carries on to building the training set
    assert A.data_card(tmp_path)["version"] == 2  # refreshing keeps one card, versioned


def test_the_eval_comparison_puts_regressions_first(tmp_path):
    scores = tmp_path / "heldout" / "scores"
    scores.mkdir(parents=True)
    for name, passed in (("base", {"a": True, "b": False, "c": False}), ("v1", {"a": False, "b": True, "c": True})):
        (scores / f"{name}.json").write_text(json.dumps({"name": name, "scores": [
            {"task_id": t, "passed": p, "seconds": 10} for t, p in passed.items()]}))
    a = A.eval_comparison(tmp_path, after="v1")
    table = next(b for b in a["blocks"] if b["type"] == "table")
    assert table["rows"][0][:3] == ["a", "pass", "fail"]  # the regression leads
    assert "2 fixed, 1 regressed" in a["summary"] and a["title"].endswith("mixed")
    with pytest.raises(A.ArtifactError, match="no held-out scores for 'v9'"):
        A.eval_comparison(tmp_path, after="v9")


def test_the_shared_page_is_self_contained_and_escaped(tmp_path):
    a = A.save(tmp_path, kind="note", title="<script>x</script>",
               blocks=[{"type": "samples", "items": [{"left_label": "l", "left": "<b>", "right_label": "r", "right": "x"}]}])
    page = A.to_html(a)
    assert "<script>x" not in page and "&lt;script&gt;" in page and "<script" not in page.split("<style>")[0]
    assert "http" not in page.split("<main>")[0]  # no outside files


def test_the_agent_has_the_artifact_tools():
    from opentrain.core.tools import artifacts

    names = {t.name for t in artifacts.tools()}
    assert names == {"data_card", "eval_comparison", "artifact_make", "artifacts", "artifact_show"}


def test_the_data_card_sorts_themes_into_training_and_rules(tmp_path):
    _traces(tmp_path)
    (tmp_path / "traces" / "themes.json").write_text(json.dumps([
        {"name": "does more than asked", "count": 7, "trainable": True, "fix": "corrected continuations",
         "evidence": ["t1:0"]},
        {"name": "adds a co-author trailer", "count": 3, "trainable": False, "fix": "a commit rule"}]))
    t = next(b for b in A.data_card(tmp_path)["blocks"] if b.get("title", "").startswith("What went wrong"))
    assert [r[2] for r in t["rows"]] == ["yes", "no: a rule"]


def test_themes_keep_their_classification_and_count_sessions_not_moments(tmp_path):
    import types

    from opentrain.core.tools import coding

    ctx = types.SimpleNamespace(project=types.SimpleNamespace(state=tmp_path))
    coding.themes_set(ctx, [{"name": "trailer", "trainable": False, "evidence": ["t1:0", "t1:2", "t1:5"]},
                            {"name": "scope", "evidence": ["t1:0", "t2:1"]}])
    saved = {t["name"]: t for t in json.loads((tmp_path / "traces" / "themes.json").read_text())}
    assert saved["trailer"]["trainable"] is False and saved["trailer"]["sessions"] == 1
    assert saved["scope"]["trainable"] is True and saved["scope"]["sessions"] == 2
