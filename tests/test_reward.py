"""A user's environment as a reward function: graded here in a child process, and in process on the GPU."""

from __future__ import annotations

import json

import pytest

from opentrain import graders
from opentrain.backends.gpu_train import grader_fn

REWARD = '''
def reward(prompt, answer, reference):
    if "boom" in answer:
        raise RuntimeError("env crashed")
    return (1.0, "solved") if answer.strip() == reference else (0.25, "partly")
'''


def test_the_reward_grades_here(tmp_path):
    (tmp_path / "reward.py").write_text(REWARD)
    spec = graders.check({"type": "python", "path": "reward.py"})
    g = graders.grade(spec, "q", "42", "42", directory=tmp_path)
    assert g.score == 1.0 and g.passed
    g = graders.grade(spec, "q", "41", "42", directory=tmp_path)
    assert g.score == 0.25 and not g.passed and g.reason == "partly"
    g = graders.grade(spec, "q", "boom", "42", directory=tmp_path)
    assert not g.passed and "env crashed" in g.reason
    assert "reward.py" in graders.describe(spec)


def test_a_broken_reward_is_refused():
    with pytest.raises(graders.GraderError):
        graders.check({"type": "python"})
    with pytest.raises(graders.GraderError):
        graders.check({"type": "python", "path": "r.py", "source": "def reward(:"})
    with pytest.raises(graders.GraderError):
        graders.load_reward("x = 1")


def test_the_reward_runs_in_the_training_loop():
    fn = grader_fn({"type": "python", "path": "reward.py", "source": REWARD})
    assert fn("q", "7", "7") == 1.0 and fn("q", "8", "7") == 0.25
    assert fn("q", "7", "7") == 1.0  # bools and tuples both become scores
    assert grader_fn({"type": "python", "source": "def reward(p, a, r):\n    return a == r\n"})("q", "x", "x") == 1.0


def test_rl_on_a_dataset_ships_the_reward(live):
    c = live.client(live.proj)
    (live.proj / "reward.py").write_text(REWARD)
    content = "".join(json.dumps({"question": f"what is {i}*2?", "answer": str(2 * i)}) + "\n" for i in range(40))
    c.post("/api/dataset/env/upload", {"filename": "env.jsonl", "content": content})
    c.patch("/api/dataset/env", {"grader": {"type": "python", "path": "reward.py"}})
    r = c.post("/api/outside/call", {"name": "dataset_recipe", "input": {"name": "env", "method": "rl"}})
    assert not r["is_error"], r["output"]
    out = json.loads(r["output"])
    assert out["method"] == "rl" and out["extra"]["grader"]["source"] == REWARD


def test_dataset_search_and_peek_shapes():
    from opentrain.datasets import peek_hf, search_hf, search_kaggle

    hf = [{"id": "a/b", "downloads": 9, "likes": 2, "tags": ["license:mit", "size_categories:1K<n<10K",
                                                            "task_categories:question-answering"],
           "description": "  many   words "}, {"id": "p/q", "private": True}]
    r = search_hf("qa", 5, fetch=lambda url: hf)
    assert r == [{"source": "hf", "id": "a/b", "downloads": 9, "likes": 2, "gated": False, "size": "1K<n<10K",
                  "license": "mit", "tasks": ["question-answering"], "languages": [], "about": "many words"}]
    kg = [{"ref": "o/n", "titleNullable": "T", "downloadCount": 5, "voteCount": 1, "totalBytesNullable": 10,
           "licenseNameNullable": "CC0", "usabilityRatingNullable": 0.88235, "subtitleNullable": "s"}]
    k = search_kaggle("x", 5, fetch=lambda url: kg)[0]
    assert k["id"] == "o/n" and k["title"] == "T" and k["usability"] == 0.88 and k["license"] == "CC0"

    def fetch(url):
        if "/info?" in url:
            return {"dataset_info": {"main": {"splits": {"train": {"num_examples": 3}},
                                              "features": {"q": {"dtype": "string"}}}}}
        return {"rows": [{"row": {"q": "hi" * 300}}]}
    p = peek_hf("a/b", fetch=fetch)
    assert p["configs"]["main"] == {"splits": {"train": 3}, "columns": {"q": "string"}}
    assert len(p["sample"]["rows"][0]["q"]) == 400


def test_the_entry_skills_ship_and_name_real_tools(live):
    import re

    from opentrain.core.skills import Skills

    c = live.client(live.proj)
    tools = {t["name"] for t in c.get("/api/outside/tools")} | {"question", "todo", "propose", "goal_set"}
    sk = Skills().all()
    for name in ("data/find", "data/triage", "data/tasks", "train/env"):
        body = sk[name].body
        used = set(re.findall(r"`(dataset_[a-z]+|model_try|goal_set|evals|question|todo)\b", body))
        assert used <= tools, (name, used - tools)
    r = c.post("/api/outside/call", {"name": "skill", "input": {"name": "data/find"}})
    assert not r["is_error"] and "dataset_search" in r["output"]


def test_vision_language_bases_load_with_their_own_class(monkeypatch):
    """Qwen3.5 / Gemma 3 checkpoints carry a vision config; the causal-LM class cannot read it."""
    import sys
    import types

    from opentrain.backends.gpu_train import load_lm

    class VLConfig:
        vision_config = {"x": 1}
        text_config = {"y": 2}

    class TextConfig:
        vocab_size = 10
        vision_config = None

    cfgs = {"Qwen/Qwen3.5-0.8B": VLConfig(), "Qwen/Qwen3-1.7B": TextConfig()}
    used = []
    fake = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda m: cfgs[m]),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda m, dtype=None: used.append(("lm", m))),
        AutoModelForImageTextToText=types.SimpleNamespace(
            from_pretrained=lambda m, dtype=None: used.append(("vl", m))))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    load_lm("Qwen/Qwen3.5-0.8B", None)
    load_lm("Qwen/Qwen3-1.7B", None)
    assert used == [("vl", "Qwen/Qwen3.5-0.8B"), ("lm", "Qwen/Qwen3-1.7B")]
