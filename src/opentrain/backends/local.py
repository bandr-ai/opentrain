"""Local: LoRA SFT on your own GPU (CUDA), Apple silicon (MPS) or, slowly, CPU, with TRL and PEFT.

For people who would rather spend electricity than dollars, or whose data may not leave the
machine. Like the Tinker backend it trains in a detached worker
(``python -m opentrain.backends.local <job.json>``) that reports into the shared job file, so the
job survives the CLI exiting and ``opentrain job ps`` works from any shell. The checkpoint is a
PEFT adapter folder, ``.opentrain/models/<job_id>/adapter``, which ``opentrain serve`` hands to
``vllm serve --enable-lora``.

Docs relied on (read 2026-11-25): TRL 1.18 SFTTrainer, including tool-calling datasets
(``messages`` with ``tool_calls`` plus a ``tools`` column), ``assistant_only_loss`` and PEFT:
https://huggingface.co/docs/trl/sft_trainer and https://huggingface.co/docs/trl/dataset_formats
(``Dataset.from_list(rows, on_mixed_types="use_json")`` needs datasets>=4.7 for the tool schemas).

Not run end to end here (no GPU, and the extras are not installed in this environment).
Unverified: ``assistant_only_loss=True`` needs a chat template with generation markers; TRL
patches known families such as Qwen3 and errors otherwise (set ``extra assistant_only_loss=false``
to train on the whole sequence). bf16 is used only on CUDA devices that report support; MPS and
CPU train in fp32.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..plugs import JobStatus, TrainSpec
from . import JobStore, missing_modules, spec_from_dict, worker_main

DEPS = ("torch", "transformers", "peft", "trl", "datasets")


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalBackend:
    name = "local"

    def __init__(self, state: Path):
        self.state = Path(state)
        self.store = JobStore(state)

    def available(self) -> tuple[bool, str]:
        missing = missing_modules(*DEPS)
        if missing:
            return False, f"missing {', '.join(missing)}: pip install 'opentrain-cli[local]'"
        dev = device()
        if dev == "cpu":
            return True, "ready on CPU only: expect many hours for a 7B+ model; a GPU or tinker is faster"
        return True, f"ready on {dev}"

    def estimate(self, spec: TrainSpec) -> float:
        """Your own hardware: no dollars are reserved against the budget."""
        return 0.0

    def launch(self, spec: TrainSpec) -> str:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"local backend unavailable: {why}")
        if not Path(spec.data).is_file():
            raise FileNotFoundError(f"training data {spec.data} does not exist")
        job = self.store.new(self.name, spec, detail="starting worker")
        out = self.state / "models" / job["job_id"]
        self.store.update(job["job_id"], output_dir=str(out))
        self.store.spawn("opentrain.backends.local", job["job_id"])
        return job["job_id"]

    def status(self, job_id: str) -> JobStatus:
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)


def run(store: JobStore, job: dict[str, Any]) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from .tinker import hf_messages, read_examples

    jid = job["job_id"]
    spec = spec_from_dict(job["spec"])
    out = Path(job["output_dir"])
    dev = device()
    store.log(jid, f"device {dev}; base {spec.base_model}; LoRA rank {spec.lora_rank}")

    rows = [{"messages": hf_messages(r["messages"]), "tools": r.get("tools") or []}
            for r in read_examples(spec.data)]
    dataset = Dataset.from_list(rows, on_mixed_types="use_json")

    import torch

    bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    config = SFTConfig(
        output_dir=str(out / "trainer"),
        num_train_epochs=spec.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=spec.batch_size,
        learning_rate=spec.learning_rate,
        max_length=spec.max_tokens,
        assistant_only_loss=bool(spec.extra.get("assistant_only_loss", True)),
        bf16=bf16,
        use_cpu=dev == "cpu",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=int(spec.extra.get("seed", 0)),
    )
    lora = LoraConfig(r=spec.lora_rank, lora_alpha=2 * spec.lora_rank, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    class Report(TrainerCallback):
        """Mirror the trainer's progress into the job file; stop at the next step when asked to."""

        cancelled = False

        def on_log(self, args, state, control, logs=None, **kwargs):
            total = state.max_steps or 1
            store.update(jid, progress=round(state.global_step / total, 4), loss=(logs or {}).get("loss"),
                         detail=f"step {state.global_step}/{total} on {dev}")
            store.log(jid, f"step {state.global_step}/{total} {logs}")

        def on_step_end(self, args, state, control, **kwargs):
            if store.cancel_requested(jid):
                Report.cancelled = True
                control.should_training_stop = True
            return control

    trainer = SFTTrainer(model=spec.base_model, args=config, train_dataset=dataset, peft_config=lora,
                         callbacks=[Report()])
    trainer.train()
    if Report.cancelled:
        store.log(jid, "cancelled")
        store.update(jid, state="cancelled", detail="cancelled; no adapter saved")
        return
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    store.log(jid, f"adapter saved: {adapter}")
    store.update(jid, state="succeeded", progress=1.0, checkpoint=str(adapter), cost_usd=0.0,
                 detail=f"done on {dev}; adapter at {adapter}")


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:], run))
