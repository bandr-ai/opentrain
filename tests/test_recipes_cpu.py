"""The training recipes for real, on a CPU and a tiny model: every method runs end to end and saves
an adapter, and an agent trace trains on its assistant turns only.

Skipped without torch (CI installs no ML stack). To run them:
``uv run --with torch --with transformers --with peft pytest tests/test_recipes_cpu.py``
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from opentrain.backends import gpu_train  # noqa: E402
from opentrain.backends.modal_backend import sft_row  # noqa: E402
from opentrain.runtime.direct import SYSTEM_PROMPT, TOOLS  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def _rows(n=6):
    return [{"prompt": [{"role": "user", "content": f"{i} + {i}?"}],
             "completion": [{"role": "assistant", "content": str(2 * i)}]} for i in range(n)]


def _spec(method):
    return {"method": method, "base_model": TINY, "lora_rank": 4, "learning_rate": 1e-3, "epochs": 1,
            "batch_size": 3, "max_tokens": 512,
            "extra": {"rounds": 1, "samples": 2, "steps": 2, "prompts_per_step": 2, "max_new_tokens": 4,
                      "teacher": TINY}}


@pytest.mark.parametrize("method", gpu_train.METHODS)
def test_every_method_trains_and_saves_an_adapter(method, tmp_path):
    events = []
    grade = (lambda p, a, r: 1.0) if method in ("rejection", "rl") else None
    summary = gpu_train.run(_spec(method), _rows(), str(tmp_path), events.append, grade)
    assert summary["method"] == method and summary["steps"] >= 1
    assert (tmp_path / "adapter_config.json").exists()
    assert any(e["event"] == "step" for e in events)


def _trace(n):
    def call(i, name, args):
        return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"make {n}.txt"},
        {"role": "assistant", "content": "", "tool_calls": [call(1, "write_file", {"path": f"{n}.txt", "content": "x"})]},
        {"role": "tool", "tool_call_id": "c1", "content": "TOOL-OUTPUT-NOT-TRAINED"},
        {"role": "assistant", "content": "", "tool_calls": [call(2, "finish", {"summary": "done"})]}],
        "tools": TOOLS}


def test_a_trace_trains_on_every_assistant_turn_and_nothing_else(tmp_path):
    from transformers import AutoTokenizer

    row = sft_row(_trace("a"))  # what Modal and RunPod send: the trace kept whole
    assert "messages" in row
    tok = AutoTokenizer.from_pretrained(TINY)
    labelled = tok.decode([t for ex in gpu_train.trace_tokens(tok, row, 8192) for t in ex["labels"] if t != -100])
    assert "write_file" in labelled and "finish" in labelled
    assert "make a.txt" not in labelled and "TOOL-OUTPUT-NOT-TRAINED" not in labelled

    events = []
    rows = [sft_row(_trace(n)) for n in "abc"] + _rows(2)  # traces and plain pairs in one batch
    summary = gpu_train.run(_spec("sft"), rows, str(tmp_path), events.append)
    assert summary["steps"] == 2 and (tmp_path / "adapter_config.json").exists()


def test_a_sampling_method_refuses_traces_before_loading_a_model(tmp_path):
    with pytest.raises(ValueError, match="train with sft"):
        gpu_train.run({**_spec("rl"), "base_model": "no/such-model"}, [sft_row(_trace("a"))], str(tmp_path),
                      lambda e: None, lambda p, a, r: 1.0)
