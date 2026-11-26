"""Tools a general data workflow leans on: templates, eval-only imports, checks on chosen rows,
grading by hand, and a gate a quick eval cannot fool."""

from __future__ import annotations

import json

import pytest

from opentrain.core.providers.scripted import ScriptedProvider
from opentrain.datasets import DatasetError, apply_template


def test_templates_build_multiple_choice():
    medqa = {"question": "Which drug?", "options": {"A": "x", "B": "y"}, "answer_idx": "B"}
    r = apply_template(medqa, "{question}\n\n{options_lettered}", "{answer_idx}")
    assert r["prompt"] == "Which drug?\n\nA. x\nB. y" and r["completion"] == "B"
    medmcqa = {"q": "Q?", "choices": ["p", "q", "r"], "cop": 2}
    r = apply_template(medmcqa, "{q}\n{choices_lettered}", "{cop_letter}")
    assert r["prompt"].endswith("C. r") and r["completion"] == "C"
    assert apply_template({"c": ["a", "b"]}, "{c[1]}", None)["prompt"] == "b"
    with pytest.raises(DatasetError):
        apply_template(medqa, "{question.__class__}", None)
    with pytest.raises(DatasetError, match="no column"):
        apply_template(medqa, "{nope}", None)


def _upload(c, name, n=30, answer="done"):
    content = "".join(json.dumps({"question": f"q{i}", "answer": answer}) + "\n" for i in range(n))
    c.post(f"/api/dataset/{name}/upload", {"filename": f"{name}.jsonl", "content": content})


def _call(c, tool, **inp):
    r = c.post("/api/outside/call", {"name": tool, "input": inp})
    return r, (json.loads(r["output"]) if not r["is_error"] and r["output"].lstrip().startswith(("{", "[")) else None)


def test_import_a_file_with_templates_as_eval_only(live):
    rows = [{"question": f"Q{i}?", "choices": ["a", "b", "c", "d"], "cop": i % 4} for i in range(40)]
    (live.proj / "mcq.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    c = live.client(live.proj)
    r, out = _call(c, "dataset_import", name="bench", path="mcq.jsonl", heldout=1,
                   input_template="{question}\n{choices_lettered}", output_template="{cop_letter}")
    assert not r["is_error"], r["output"]
    assert out["added_train"] == 0 and out["added_heldout"] == 40
    r, _ = _call(c, "dataset_import", name="bad", path="mcq.jsonl", heldout=2)
    assert r["is_error"]


def test_train_checks_pick_rows_and_show_answers_and_grade_by_hand(live):
    c = live.client(live.proj)
    _upload(c, "chk", 60)
    c.patch("/api/dataset/chk", {"grader": {"type": "exact"}})
    ids = [r["id"] for r in c.get("/api/dataset/chk", limit=100)["rows"]]
    ScriptedProvider.register("chk-m", [])
    r, _ = _call(c, "dataset_eval", name="chk", model="scripted/chk-m", split="train", ids=ids[5:8])
    assert not r["is_error"], r["output"]
    jid = r["output"].split("job ")[1].split(" ")[0]
    job = c.post("/api/outside/call", {"name": "job_wait", "input": {"job_id": jid, "timeout": 30}})
    eid = json.loads(job["output"])["result"]["eval"]
    _, ev = _call(c, "evals", eval_id=eid)
    assert sorted(x["id"] for x in ev["rows"]) == sorted(ids[5:8]) and all("answer" in x for x in ev["rows"])
    r, _ = _call(c, "dataset_eval", name="chk", model="scripted/chk-m", ids=ids[:2])
    assert r["is_error"] and "held-out" in r["output"]
    _, g = _call(c, "dataset_grade", name="chk", answer="done", row_id=ids[0])
    assert g["passed"] and g["score"] == 1.0
    _, g = _call(c, "dataset_grade", name="chk", answer="", row_id=ids[0])
    assert not g["passed"]


def test_a_quick_eval_does_not_replace_a_full_one_at_the_gate(live):
    from opentrain.evals import latest, run

    c = live.client(live.proj)
    _upload(c, "gate", 60)
    c.patch("/api/dataset/gate", {"grader": {"type": "exact"}})
    state = live.proj / ".opentrain"
    ScriptedProvider.register("gate-m", [])
    full = run(state, live.proj, "gate", "scripted/gate-m")
    run(state, live.proj, "gate", "scripted/gate-m", limit=2)
    assert latest(state, "gate", "scripted/gate-m")["id"] == full["id"]


def test_overlap_check_reads_any_script():
    from opentrain.core.integrity import ngrams

    ru = "Это длинный вопрос на русском языке о том как работает модель машинного обучения и почему она ошибается"
    ja = "機械学習モデルがなぜ間違えるのかについての長い日本語の質問です"
    assert ngrams(ru) and ngrams(ja)
    assert ngrams(ru + " и ещё") & ngrams(ru)


def test_a_serving_key_is_made_once_under_a_crowd(tmp_path, monkeypatch):
    import threading

    from opentrain.core import credentials

    monkeypatch.setenv("OPENTRAIN_AUTH_FILE", str(tmp_path / "auth.json"))
    made, got = [], []

    def make():
        made.append(1)
        return f"k{len(made)}"

    ts = [threading.Thread(target=lambda: got.append(credentials.get_or_create("svc", make))) for _ in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(made) == 1 and set(got) == {"k1"}


def test_regex_with_a_capture_compares_and_letters_are_suggested():
    from opentrain.core.tools.data import suggest_grader
    from opentrain.graders import describe, grade

    sp = {"type": "regex", "pattern": r"^\W*([A-D])\b"}
    assert grade(sp, "", "B. because", "B").passed
    assert not grade(sp, "", "A", "B").passed  # matching the shape is not enough
    assert grade({"type": "regex", "pattern": "yes"}, "", "oh yes", None).passed
    assert "compares" in describe(sp)
    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": x}]} for x in "ABCDA"]
    assert suggest_grader(rows)["type"] == "regex"


def test_reasoning_is_stripped_before_grading():
    from opentrain.core.complete import THINK

    assert THINK.sub("", "A\n</think>\n\nA") == "A"
    assert THINK.sub("", "<think>hm</think> B") == "B"
    assert THINK.sub("", "plain") == "plain"


def test_recipe_prices_the_overrides_you_will_run(live, monkeypatch):
    monkeypatch.setenv("OPENTRAIN_FAKE_BACKEND", "1")
    (live.proj / "Trainfile").write_text("trainer      fake\nbase model   Qwen/Qwen2.5-0.5B-Instruct\n")
    c = live.client(live.proj)
    _upload(c, "ov", 40)
    _, base = _call(c, "dataset_recipe", name="ov")
    _, one = _call(c, "dataset_recipe", name="ov", epochs=1, learning_rate=5e-5)
    assert one["epochs"] == 1 and one["learning_rate"] == 5e-5 and one["overridden"] == ["epochs", "learning_rate"]
    assert "usd" in base["cost"] and one["cost"]["usd"] <= base["cost"]["usd"]


def test_a_rollout_404_is_retried_not_scored(monkeypatch):
    import io
    import urllib.error

    from opentrain.core import complete

    calls = []

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b"{}"))
        return R(b'{"choices": [{"message": {"content": "B"}}]}')

    monkeypatch.setattr(complete.urllib.request, "urlopen", fake)
    monkeypatch.setattr(complete.time, "sleep", lambda s: None)
    assert complete._rollout_safe(object(), 5)["choices"][0]["message"]["content"] == "B" and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(complete, "ROLLOUT_SECONDS", -1)
    with pytest.raises(urllib.error.HTTPError):
        complete._rollout_safe(object(), 5)


def test_rows_you_write_go_where_you_say(live):
    c = live.client(live.proj)
    rows = [{"prompt": f"p{i}", "completion": "c"} for i in range(10)]
    _, out = _call(c, "dataset_add", name="w", rows=rows, heldout=1)
    assert out["added_heldout"] == 10 and out["added_train"] == 0
    _, out = _call(c, "dataset_add", name="w", rows=[{"prompt": f"t{i}", "completion": "c"} for i in range(10)], heldout=0)
    assert out["added_train"] == 10 and out["added_heldout"] == 0


def test_spent_and_held_are_told_apart(tmp_path):
    from opentrain.budget import Ledger
    from opentrain.trainfile import Budget

    led = Ledger(tmp_path, Budget(2.0, 7, "$2 a week"))
    a = led.reserve(1.5, "train")
    b = led.reserve(0.1, "eval")
    led.settle(b, 0.11)
    assert led.reserved() == 1.5 and led.committed() == 1.61 and led.remaining() == 0.39
    led.settle(a, 0.6)
    assert led.reserved() == 0 and led.committed() == 0.71
