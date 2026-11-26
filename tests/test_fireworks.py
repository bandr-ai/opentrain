"""The Fireworks backend, end to end against a fake SDK: nothing here talks to Fireworks or spends."""

from __future__ import annotations

import json
import sys
import textwrap
import types

import pytest

from opentrain.backends import JobStore, fireworks, fireworks_worker
from opentrain.plugs import TrainSpec

ROWS = [
    {"messages": [{"role": "system", "content": "Label it."}, {"role": "user", "content": f"ticket {i}"},
                  {"role": "assistant", "content": "billing" if i % 2 else "bug"}]}
    for i in range(5)
]


@pytest.fixture
def data(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    return p


class CharTokenizer:
    """A tokenizer whose tokens are characters, with a ChatML-ish template that is prefix-stable."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, toks, skip_special_tokens=False):
        return "".join(chr(t) for t in toks)


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def fake_sdk(calls: list):
    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input, self.loss_fn_inputs = model_input, loss_fn_inputs

    class ModelInput:
        @staticmethod
        def from_ints(ints):
            return list(ints)

    class AdamParams:
        def __init__(self, learning_rate):
            self.learning_rate = learning_rate

    class Training:
        def forward_backward(self, batch, loss):
            calls.append(("fb", len(batch), loss))
            return Future(types.SimpleNamespace(metrics={"loss:sum": 0.5}))

        def optim_step(self, adam):
            calls.append(("opt", adam.learning_rate))
            return Future(None)

        def save_weights_for_sampler(self, name):
            calls.append(("save", name))
            return Future(types.SimpleNamespace(path=f"acct/run-1/{name}"))

    class Service:
        def __init__(self, api_key, base_url):
            calls.append(("service", base_url))

        def create_lora_training_client(self, base_model, rank):
            calls.append(("client", base_model, rank))
            return Training()

        def delete_training_session(self):
            calls.append(("delete",))

        def close(self):
            calls.append(("close",))

    sdk = types.SimpleNamespace(Datum=Datum, ModelInput=ModelInput, AdamParams=AdamParams,
                                FiretitanServiceClient=Service)
    return sdk


@pytest.fixture
def sdk(monkeypatch):
    calls: list = []
    mod = fake_sdk(calls)
    training = types.ModuleType("fireworks.training")
    training.sdk = mod
    fw = types.ModuleType("fireworks")
    fw.training = training
    monkeypatch.setitem(sys.modules, "fireworks", fw)
    monkeypatch.setitem(sys.modules, "fireworks.training", training)
    monkeypatch.setitem(sys.modules, "fireworks.training.sdk", mod)
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: CharTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test")
    return calls


def test_model_names_resolve_and_small_models_are_refused():
    assert fireworks.model_id("Qwen/Qwen3.6-35B-A3B") == "qwen3p6-35b-a3b"
    assert fireworks.model_id("accounts/fireworks/models/qwen3p6-35b-a3b") == "qwen3p6-35b-a3b"
    assert fireworks.model_id("fireworks/muse-glimmer-30b") == "muse-glimmer-30b"
    with pytest.raises(fireworks.FireworksError, match="trainer modal"):
        fireworks.model_id("Qwen/Qwen2.5-0.5B-Instruct")


def test_estimate_uses_the_training_price(tmp_path, data):
    spec = TrainSpec(base_model="Qwen/Qwen3.6-35B-A3B", data=data, epochs=2)
    est = fireworks.FireworksBackend(tmp_path).estimate(spec)
    assert 0 < est < 0.01


def test_worker_trains_saves_and_closes_the_session(tmp_path, data, sdk):
    store = JobStore(tmp_path)
    spec = TrainSpec(base_model="qwen3p6-35b-a3b", data=data, epochs=2, batch_size=2, lora_rank=8, learning_rate=2e-4)
    job = store.new("fireworks", spec, usd_per_mtok=2.14, fireworks_model="qwen3p6-35b-a3b",
                    tokenizer="Qwen/Qwen3.6-35B-A3B")
    fireworks_worker.run(store, store.load(job["job_id"]))
    done = store.load(job["job_id"])
    assert done["state"] == "succeeded" and done["progress"] == 1.0
    assert done["checkpoint"].startswith("fireworks-sampler://acct/run-1/ot-")
    assert len(done["checkpoint"].rsplit("/", 1)[-1]) <= 17
    assert done["cost_usd"] > 0 and done["tokens"] > 0
    assert ("client", "accounts/fireworks/models/qwen3p6-35b-a3b", 8) in sdk
    assert sum(1 for c in sdk if c[0] == "fb") == 6  # 3 steps x 2 epochs
    assert sdk[-2:] == [("delete",), ("close",)]


def test_cancel_stops_at_the_next_step_and_still_closes(tmp_path, data, sdk):
    store = JobStore(tmp_path)
    spec = TrainSpec(base_model="qwen3p6-35b-a3b", data=data, epochs=1, batch_size=2)
    job = store.new("fireworks", spec, usd_per_mtok=2.14, fireworks_model="qwen3p6-35b-a3b", tokenizer="t")
    store.request_cancel(job["job_id"])
    fireworks_worker.run(store, store.load(job["job_id"]))
    assert store.load(job["job_id"])["state"] == "cancelled"
    assert not any(c[0] == "save" for c in sdk) and sdk[-2:] == [("delete",), ("close",)]


def test_sampler_protocol_round_trip(tmp_path, monkeypatch):
    """The parent side of the sampler, against a stand-in process speaking the same JSON lines."""
    mod = tmp_path / "pkg" / "fake_sampler.py"
    mod.parent.mkdir()
    mod.write_text(textwrap.dedent('''
        import json, sys
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            req = json.loads(line)
            print(json.dumps({"text": "billing", "prompt_tokens": 40, "completion_tokens": 2}), flush=True)
    '''))
    monkeypatch.setattr(fireworks, "SAMPLER", "fake_sampler")
    monkeypatch.setattr(fireworks, "runner", lambda: ([sys.executable], {"PYTHONPATH": str(mod.parent)}))
    monkeypatch.setattr(fireworks, "_samplers", {})
    text, p, c, cost = fireworks.sample("fireworks-sampler://acct/run-1/ot-x", "qwen3p6-35b-a3b",
                                        [{"role": "user", "content": "hi"}])
    assert (text, p, c) == ("billing", 40, 2) and cost == round(42 * 1.335 / 1e6, 6)
    for s in fireworks._samplers.values():
        s.close()


def test_a_fireworks_version_is_answered_by_the_sampler(tmp_path, monkeypatch):
    from opentrain.core import complete as cm

    v = types.SimpleNamespace(checkpoint="fireworks-sampler://acct/run-1/ot-x", base_model="qwen3p6-35b-a3b",
                              backend="fireworks")
    monkeypatch.setattr(fireworks, "sample", lambda ck, base, msgs, **kw: ("<think>\n</think>\n\nbug", 30, 3, 0.0001))
    c = cm._fireworks(v, [{"role": "user", "content": "x"}], 16, 30, 0.0)
    assert c.text == "bug" and c.cost == 0.0001 and c.output_tokens == 3


def test_rl_samples_grades_and_trains_with_importance_sampling(sdk, data, tmp_path, monkeypatch):
    """GRPO on the Training API: a snapshot per step, samples graded here, advantages around the group
    mean, the sampler's logprobs as the anchor, one optimizer step per batch."""
    mod = sys.modules["fireworks.training.sdk"]

    class Tok(CharTokenizer):
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    sys.modules["transformers"].AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: Tok())
    answers = iter(["billing", "bug"] * 1000)

    class Sampler:
        def sample(self, prompt, num_samples, sampling_params):
            seqs = [types.SimpleNamespace(tokens=[ord(c) for c in next(answers)], logprobs=None)
                    for _ in range(num_samples)]
            return Future(types.SimpleNamespace(sequences=[
                types.SimpleNamespace(tokens=q.tokens, logprobs=[-0.5] * len(q.tokens)) for q in seqs]))

    service = mod.FiretitanServiceClient
    service.create_sampling_client = lambda self, model_path, base_model: sdk.append(("sampler", model_path)) or Sampler()
    training_cls = type(service("k", "u").create_lora_training_client("b", 8))
    training_cls.optim_step = lambda self, adam, grad_accumulation_normalization=None: (
        sdk.append(("opt", adam.learning_rate, grad_accumulation_normalization)) or Future(None))
    mod.SamplingParams = lambda max_tokens, temperature: (max_tokens, temperature)
    mod.GradAccNormalization = types.SimpleNamespace(NUM_LOSS_TOKENS="num_loss_tokens")
    store = JobStore(tmp_path)
    spec = TrainSpec(base_model="qwen3p6-35b-a3b", data=data, method="rl", lora_rank=8, learning_rate=4e-5,
                     extra={"steps": 2, "prompts_per_step": 2, "samples": 2, "max_new_tokens": 8,
                            "grader": {"type": "exact"}})
    job = store.new("fireworks", spec, usd_per_mtok=2.14, usd_per_mtok_sample=1.335, fireworks_model="qwen3p6-35b-a3b",
                    tokenizer="t")
    fireworks_worker.run(store, store.load(job["job_id"]))
    j = store.load(job["job_id"])
    assert j["state"] == "succeeded" and j["checkpoint"].startswith("fireworks-sampler://")
    fb = [c for c in sdk if c[0] == "fb"]
    assert fb and all(c[2] == "importance_sampling" for c in fb)
    assert ("opt", 4e-5, "num_loss_tokens") in sdk
    assert sum(1 for c in sdk if c[0] == "sampler") == 2  # a fresh snapshot per step
    assert 0 <= j["reward"] <= 1 and j["cost_usd"] > 0 and sdk[-2:] == [("delete",), ("close",)]


def test_opd_is_refused_on_fireworks(data):
    with pytest.raises(ValueError, match="modal or runpod"):
        fireworks.FireworksBackend.estimate(None, TrainSpec(base_model="qwen3p6-35b-a3b", data=data, method="opd"))
