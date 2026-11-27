"""Bugs a first real run would hit, kept fixed: Hugging Face reads, dataset shapes, Fireworks limits."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from opentrain import datasets as d


def test_alpaca_rows_keep_the_instruction():
    row = d.normalize({"instruction": "Identify the odd one out.", "input": "Twitter, Instagram, Telegram",
                       "output": "Telegram"})
    assert row["messages"][0] == {"role": "user", "content": "Identify the odd one out.\n\nTwitter, Instagram, Telegram"}
    assert d.normalize({"instruction": "Say hi", "input": "", "output": "hi"})["messages"][0]["content"] == "Say hi"


def test_a_preference_dataset_is_refused_not_mangled():
    with pytest.raises(d.DatasetError, match="preference dataset"):
        d.normalize({"chosen": "\n\nHuman: hi\n\nAssistant: hello", "rejected": "\n\nHuman: hi\n\nAssistant: no"})


def _http(code):
    def urlopen(req, timeout=None):
        _http.seen = req
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b""))
    return urlopen


def test_a_missing_or_gated_dataset_says_what_to_do(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(d.urllib.request, "urlopen", _http(401))
    with pytest.raises(d.DatasetError, match="connect Hugging Face"):
        d.read_hf(" nobody/nothing ")
    assert _http.seen.full_url.endswith("dataset=nobody/nothing")  # the stray spaces are gone


def test_a_connected_token_signs_hub_reads_only(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    seen = []

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("Authorization")))
        return R(json.dumps({"datasets": []}).encode())

    monkeypatch.setattr(d.urllib.request, "urlopen", urlopen)
    d._get_json("https://datasets-server.huggingface.co/splits?dataset=a/b")
    d._get_json("https://www.kaggle.com/api/v1/datasets/list?search=x")
    assert seen[0][1] == "Bearer hf_test" and seen[1][1] is None


def test_fireworks_lists_its_base_models():
    from opentrain.core.integrations import BASE_MODELS

    assert any(m["id"] == "Qwen/Qwen3.6-35B-A3B" for m in BASE_MODELS["fireworks"])


def test_fireworks_refuses_coding_data_before_billing(tmp_path):
    from opentrain.backends.fireworks import FireworksBackend
    from opentrain.plugs import TrainSpec

    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                                                            "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
        {"role": "assistant", "content": "done"}]}) + "\n")
    spec = TrainSpec(base_model="Qwen/Qwen3.6-35B-A3B", data=data)
    with pytest.raises(ValueError, match="cannot train coding agents"):
        FireworksBackend.__new__(FireworksBackend).estimate(spec)


def test_preference_rows_are_told_apart_from_prompt_answer_rows():
    # a prompt and a plain chosen answer is an ordinary example (rejected is ignored), any key case
    assert d.normalize({"Prompt": "Q", "chosen": "A", "rejected": "B"})["messages"][-1]["content"] == "A"
    with pytest.raises(d.DatasetError, match="preference dataset"):  # transcripts, even beside a prompt
        d.normalize({"prompt": "Q", "chosen": "\n\nHuman: Q\n\nAssistant: A", "rejected": "\n\nHuman: Q\n\nAssistant: B"})


def test_a_conversation_folded_into_the_prompt_counts_as_a_trace():
    from opentrain.backends.gpu_train import multi_turn

    plain = {"prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
             "completion": [{"role": "assistant", "content": "a"}]}
    assert not multi_turn(plain)
    assert multi_turn({**plain, "prompt": [*plain["prompt"], {"role": "assistant", "content": "x"},
                                           {"role": "user", "content": "y"}]})
