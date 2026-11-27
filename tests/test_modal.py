"""The Modal backend and recipes, without the network: deploys are faked, the trainer is not run."""

import json

import pytest

from opentrain.backends import get_backend
from opentrain.backends import modal_backend as mb
from opentrain.plugs import TrainSpec
from opentrain.recipes import recipe, task_kind


def test_sizes_pick_gpus():
    assert mb.params_b("Qwen/Qwen2.5-0.5B-Instruct") == 0.5
    assert mb.params_b("Qwen/Qwen3-30B-A3B") == 30
    assert mb.train_gpu("Qwen/Qwen3-4B-Instruct-2507")[0] == "L40S"
    assert mb.train_gpu("Qwen/Qwen3-14B")[0] == "H100"
    assert mb.device_gpu("NVIDIA A100-SXM4-80GB") == "A100-80GB" and mb.device_gpu("NVIDIA L40S") == "L40S"
    assert mb.serve_gpu("Qwen/Qwen2.5-1.5B-Instruct")[0] == "L4"
    with pytest.raises(mb.ModalError):
        mb.train_gpu("meta-llama/Llama-3.3-70B-Instruct")


def test_examples_become_prompt_completion_pairs():
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "billing"}]}
    pc = mb.prompt_completion(row)
    assert [m["role"] for m in pc["prompt"]] == ["system", "user"]
    assert pc["completion"] == [{"role": "assistant", "content": "billing"}]
    with pytest.raises(ValueError):
        mb.prompt_completion({"messages": [{"role": "user", "content": "u"}]})


def test_estimate_is_cents_for_a_small_job(tmp_path, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-y")
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": "hi " * 20},
                                                       {"role": "assistant", "content": "ok"}]})
                              for _ in range(50)))
    b = get_backend("modal", tmp_path)
    est = b.estimate(TrainSpec(base_model="Qwen/Qwen2.5-0.5B-Instruct", data=data, epochs=3))
    assert 0.05 < est < 0.5


def test_endpoint_deploys_once_and_again_for_a_new_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTRAIN_MODAL_STATE", str(tmp_path / "modal.json"))
    monkeypatch.setenv("OPENTRAIN_AUTH_FILE", str(tmp_path / "auth.json"))
    monkeypatch.delenv(mb.KEY_ENV, raising=False)
    calls = []
    monkeypatch.setattr(mb, "deploy", lambda base, adapters: calls.append(dict(adapters)) or "https://x--y.modal.run")
    base = mb.endpoint("Qwen/Qwen2.5-0.5B-Instruct")
    assert base.model == "base" and base.base_url == "https://x--y.modal.run/v1" and base.api_key_env == mb.KEY_ENV
    mb.endpoint("Qwen/Qwen2.5-0.5B-Instruct")
    assert calls == [{}]  # already deployed
    v = mb.endpoint("Qwen/Qwen2.5-0.5B-Instruct", "modal-job-1", 16)
    assert v.model == "modal-job-1" and calls == [{}, {"modal-job-1": 16}]
    key = json.loads((tmp_path / "auth.json").read_text())[mb.KEY_ID]["key"]
    assert key.startswith("ot-")  # generated once, kept in the auth store


def test_recipes():
    assert task_kind({"type": "exact"}, ["billing", "bug", "account", "feature"] * 10) == "label"
    assert task_kind({"type": "json"}) == "extract"
    assert task_kind({"type": "judge"}) == "answer"
    r = recipe("label", 54, "Qwen/Qwen2.5-0.5B-Instruct", labelled=40)
    assert (r["lora_rank"], r["epochs"]) == (16, 4)
    assert any("distil" in b for b in r["before"])
    big = recipe("answer", 20000, "Qwen/Qwen3-32B")
    assert big["epochs"] == 1 and big["learning_rate"] == 5e-5
