"""The Modal app OpenTrain runs: LoRA training on a Modal GPU, and vLLM serving that scales to zero.

This file is shipped to Modal as it stands, so it imports only the standard library and ``modal``
at the top; the training and serving libraries are imported inside the functions, in the image.
It is configured by environment variables that ``modal_backend`` sets before importing it (and,
for serving, bakes into the image so the container defines the same function):

* training (the default mode): ``OT_TRAIN_GPU`` picks the GPU. ``train`` is a generator: it yields
  ``started``, then one ``step`` event per logged step (with the loss), then ``done`` with the
  adapter's path on the ``opentrain`` volume, or ``error``. The caller runs it inside
  ``app.run(detach=True)`` and stops the app when it is done or cancelled; if the caller loses its
  connection, the job goes on and the caller follows it through ``EventLog``.
* serving (``OT_MODAL_MODE=serve``): one app per base model, ``opentrain-serve-<base>``, serving
  the base model as ``base`` and every adapter in ``OT_SERVE_ADAPTERS`` (``{name: rank}``) under its
  name, behind ``--api-key`` (the key comes in a secret). It scales to zero after
  ``OT_SERVE_SCALEDOWN`` idle seconds.

Docs relied on (read 2026-11-26): Modal's vLLM example (vllm 0.21.0, transformers 5.8.1, CUDA
12.9 devel base image, HF cache volume), https://modal.com/docs/examples/vllm_inference;
``Function.remote_gen`` and ``App.run``, https://modal.com/docs/reference/modal.Function; TRL's
prompt-completion format, where the loss is on the completion only,
https://huggingface.co/docs/trl/dataset_formats.
"""

from __future__ import annotations

import json
import os
import re

import modal

MODE = os.environ.get("OT_MODAL_MODE", "train")
VOLUME = "opentrain"
HF_CACHE = "opentrain-hf-cache"
ADAPTERS = "/vol/adapters"

vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
hf = modal.Volume.from_name(HF_CACHE, create_if_missing=True)
mounts = {"/vol": vol, "/root/.cache/huggingface": hf}


def _secrets() -> list[modal.Secret]:
    env = {k: os.environ[k] for k in ("HF_TOKEN", "OPENTRAIN_SERVE_KEY") if os.environ.get(k)}
    return [modal.Secret.from_dict(env)] if env else []


def gpus(env: str, default: str):
    """A GPU, or a JSON list of GPUs Modal tries in order (whichever has capacity first)."""
    v = os.environ.get(env) or default
    return json.loads(v) if v.startswith("[") else v


EVENTS = "opentrain-events"


class EventLog:
    """Every event a training function yields is also kept in a Modal Dict, so a client that lost its
    connection (a laptop's network blip) can follow the job again by polling: the app runs detached."""

    def __init__(self, job_id: str):
        self.job_id, self.n = job_id, 0
        self.d = modal.Dict.from_name(EVENTS, create_if_missing=True)

    def __call__(self, ev: dict) -> dict:
        try:
            self.d[f"{self.job_id}/{self.n}"] = ev
            self.d[f"{self.job_id}/n"] = self.n + 1
        except Exception:  # noqa: BLE001 - the stream to a connected client still carries it
            pass
        self.n += 1
        return ev

    def stop_requested(self) -> bool:
        """OpenTrain asks a job to end early (its budget ran out) by setting ``<job>/stop``."""
        return bool(self.d.get(f"{self.job_id}/stop", False))


def slug(base: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:40]


if MODE == "train":
    app = modal.App("opentrain-train")
    train_image = (
        modal.Image.debian_slim(python_version="3.12")
        # torchvision and pillow: vision-language bases (Qwen3.5, Gemma 3) load an image processor
        .uv_pip_install("torch", "torchvision", "pillow", "transformers==5.8.1", "peft", "trl", "datasets",
                        "accelerate")
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
        .add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_train.py"),
                        "/root/gpu_train.py")
    )

    @app.function(image=train_image, gpu=gpus("OT_TRAIN_GPU", "L40S"), timeout=6 * 3600,
                  volumes=mounts, secrets=_secrets())
    def train(job_id: str, spec: dict, rows: list[dict]):
        import queue
        import threading
        import time
        from importlib.metadata import version as pkg_version

        events: queue.Queue = queue.Queue()
        t0 = time.time()
        keep = EventLog(job_id)
        import torch

        yield keep({"event": "started", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "versions": {p: pkg_version(p) for p in ("torch", "transformers", "peft", "trl")}})

        def work() -> None:
            try:
                from datasets import Dataset
                from peft import LoraConfig
                from transformers import TrainerCallback
                from trl import SFTConfig, SFTTrainer

                data = Dataset.from_list(rows, on_mixed_types="use_json")
                extra = spec.get("extra") or {}
                config = SFTConfig(
                    output_dir="/tmp/out", num_train_epochs=spec["epochs"],
                    per_device_train_batch_size=int(extra.get("device_batch", 4)),
                    gradient_accumulation_steps=max(1, spec["batch_size"] // int(extra.get("device_batch", 4))),
                    learning_rate=spec["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=0.05,
                    max_length=spec["max_tokens"], bf16=True, logging_steps=1, save_strategy="no",
                    report_to="none", seed=int(extra.get("seed", 0)), gradient_checkpointing=True,
                )
                lora = LoraConfig(r=spec["lora_rank"], lora_alpha=2 * spec["lora_rank"], lora_dropout=0.0,
                                  target_modules="all-linear", task_type="CAUSAL_LM")

                class Report(TrainerCallback):
                    checked = [0.0]

                    def on_step_end(self, args, state, control, **kw):
                        # asked to end early (the budget ran out): stop here and save what was trained
                        if time.time() - self.checked[0] > 10:
                            self.checked[0] = time.time()
                            if keep.stop_requested():
                                control.should_training_stop = True
                                events.put({"event": "log", "note": "asked to stop early: saving the adapter"})

                    def on_log(self, args, state, control, logs=None, **kw):
                        logs = logs or {}
                        if "loss" in logs:
                            events.put({"event": "step", "step": state.global_step, "total": state.max_steps,
                                        "loss": logs["loss"], "seconds": round(time.time() - t0, 1)})

                if any("messages" in r for r in rows):
                    # Agent traces: one example per trace, loss on every assistant turn (not only the last).
                    import sys

                    from transformers import AutoTokenizer

                    sys.path.insert(0, "/root")
                    from gpu_train import trace_tokens

                    tok = AutoTokenizer.from_pretrained(spec["base_model"])
                    data = Dataset.from_list([ex for r in rows for ex in trace_tokens(tok, r, spec["max_tokens"])])
                trainer = SFTTrainer(model=spec["base_model"], args=config, train_dataset=data, peft_config=lora,
                                     callbacks=[Report()])
                trainer.train()
                out = f"{ADAPTERS}/{job_id}"
                trainer.save_model(out)
                vol.commit()
                events.put({"event": "done", "adapter": out, "seconds": round(time.time() - t0, 1)})
            except Exception as e:  # noqa: BLE001 - the caller records it
                events.put({"event": "error", "error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = events.get()
            yield keep(ev)
            if ev["event"] in ("done", "error"):
                return

    # Rejection sampling, on-policy distillation and RL: ``gpu_train`` (shipped with ``graders``),
    # sampling with vLLM in the same container, on the serving image's CUDA stack.
    HERE = os.path.dirname(os.path.abspath(__file__))
    loop_image = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install("vllm==0.21.0", "transformers==5.8.1", "peft")
        .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
        .add_local_file(os.path.join(HERE, "gpu_train.py"), "/root/gpu_train.py")
        .add_local_file(os.path.join(os.path.dirname(HERE), "graders.py"), "/root/graders.py")
    )

    @app.function(image=loop_image, gpu=gpus("OT_LOOP_GPU", "L40S"), timeout=12 * 3600,
                  volumes=mounts, secrets=_secrets())
    def train_loop(job_id: str, spec: dict, rows: list[dict]):
        import queue
        import sys
        import threading
        import time
        from importlib.metadata import version as pkg_version

        sys.path.insert(0, "/root")
        import gpu_train
        import torch

        events: queue.Queue = queue.Queue()
        t0 = time.time()
        keep = EventLog(job_id)
        yield keep({"event": "started", "gpu": torch.cuda.get_device_name(0),
                    "versions": {p: pkg_version(p) for p in ("torch", "transformers", "peft", "vllm")}})

        def work() -> None:
            try:
                out = f"{ADAPTERS}/{job_id}"
                summary = gpu_train.run(spec, rows, out, events.put,
                                        gpu_train.grader_fn((spec.get("extra") or {}).get("grader")),
                                        should_stop=keep.stop_requested)
                vol.commit()
                events.put({"event": "done", "adapter": out, **summary, "seconds": round(time.time() - t0, 1)})
            except Exception as e:  # noqa: BLE001 - the caller records it
                import traceback

                events.put({"event": "error", "error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-2000:]})

        threading.Thread(target=work, daemon=True).start()
        while True:
            ev = events.get()
            yield keep(ev)
            if ev["event"] in ("done", "error"):
                return

if MODE == "serve":
    BASE = os.environ["OT_SERVE_BASE"]
    ADAPTER_RANKS: dict[str, int] = json.loads(os.environ.get("OT_SERVE_ADAPTERS") or "{}")
    app = modal.App(f"opentrain-serve-{slug(BASE)}")
    serve_image = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
        .env({"HF_XET_HIGH_PERFORMANCE": "1", "OT_MODAL_MODE": "serve", "OT_SERVE_BASE": BASE,
              "OT_SERVE_GPU": os.environ.get("OT_SERVE_GPU", "L4"),
              "OT_SERVE_ADAPTERS": json.dumps(ADAPTER_RANKS),
              "OT_SERVE_ARGS": os.environ.get("OT_SERVE_ARGS", "[]"),
              "OT_SERVE_MAX_LEN": os.environ.get("OT_SERVE_MAX_LEN", "8192")})
    )

    @app.function(image=serve_image, gpu=gpus("OT_SERVE_GPU", "L4"), timeout=24 * 3600,
                  scaledown_window=int(os.environ.get("OT_SERVE_SCALEDOWN", "180")), volumes=mounts,
                  secrets=_secrets())
    @modal.concurrent(max_inputs=64)
    @modal.web_server(port=8000, startup_timeout=20 * 60)
    def serve():
        import subprocess

        cmd = ["vllm", "serve", BASE, "--served-model-name", "base", "--host", "0.0.0.0", "--port", "8000",
               "--max-model-len", os.environ.get("OT_SERVE_MAX_LEN", "8192"), "--enforce-eager",
               "--generation-config", "vllm"]
        if os.environ.get("OPENTRAIN_SERVE_KEY"):
            cmd += ["--api-key", os.environ["OPENTRAIN_SERVE_KEY"]]
        cmd += json.loads(os.environ.get("OT_SERVE_ARGS") or "[]")  # tool-call and reasoning parsers
        if ADAPTER_RANKS:
            rank = next(r for r in (8, 16, 32, 64, 128, 256, 512) if r >= max(ADAPTER_RANKS.values()))
            cmd += ["--enable-lora", "--max-loras", str(min(8, len(ADAPTER_RANKS))),
                    "--max-lora-rank", str(rank), "--lora-modules"]
            cmd += [f"{name}={ADAPTERS}/{name}" for name in ADAPTER_RANKS]
        subprocess.Popen(cmd)
