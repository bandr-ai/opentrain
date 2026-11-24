"""Fireworks: a clearly partial backend, off until its new Training API is ported.

What the docs say (read 2026-11-25):

* Managed supervised fine-tuning ("SFT jobs", ``firectl sftj``) no longer accepts new jobs; only
  jobs created before 2026-11-09 can finish and be deployed.
  https://docs.fireworks.ai/fine-tuning/fine-tuning-models
  https://docs.fireworks.ai/fine-tuning/migrate-from-managed-training.md
* Its replacement, the Training API, is a client-driven loop like Tinker's: ``pip install
  "fireworks-ai[training]"``, ``from fireworks.training import sdk``,
  ``sdk.FiretitanServiceClient(api_key=..., base_url="https://api.fireworks.ai/training/v1/serverless")``,
  ``create_lora_training_client(base_model=..., rank=...)``, ``forward_backward(batch,
  loss_fn="cross_entropy")``, ``optim_step``, ``save_weights_for_sampler(name).result().path``.
  Its ``Datum``/``ModelInput``/``TensorData`` are its own classes, not Tinker's.
  https://docs.fireworks.ai/fine-tuning/training-api/introduction.md
  https://docs.fireworks.ai/fine-tuning/training-api/migrate-from-tinker.md
* Serverless training is billed per token, dedicated per GPU hour; the docs pages read give no
  rates, so ``estimate`` could not honour the budget.
* A trained LoRA is served from an on-demand deployment (``firectl deployment create
  accounts/<ACCOUNT_ID>/models/<MODEL_ID>``); serverless per-token serving of your own adapter is
  not available. The final checkpoint's promote call is not named in the pages read.

So the loop could mirror ``tinker_worker`` with Fireworks' native types, but the price, the
optimizer-step arguments, the promote step and the deployment's OpenAI model name are unverified.
Rather than guess and bill someone, ``available()`` is False and every action says why.
"""

from __future__ import annotations

from pathlib import Path

from ..plugs import JobStatus, TrainSpec
from . import JobStore

WHY = ("not implemented: Fireworks closed managed SFT to new jobs (Nov 2026) and its replacement "
       "Training API (fireworks-ai[training]) has no documented per-token price or promote step we "
       "could verify; use trainer tinker or local")


class FireworksBackend:
    name = "fireworks"

    def __init__(self, state: Path):
        self.store = JobStore(state)

    def available(self) -> tuple[bool, str]:
        return False, WHY

    def estimate(self, spec: TrainSpec) -> float:
        raise NotImplementedError(WHY)

    def launch(self, spec: TrainSpec) -> str:
        raise NotImplementedError(WHY)

    def status(self, job_id: str) -> JobStatus:
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError(WHY)
