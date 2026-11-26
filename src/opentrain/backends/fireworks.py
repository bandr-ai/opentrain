"""Fireworks: LoRA SFT on Fireworks' serverless Training API, and answers from the trained snapshot.

Fireworks closed managed SFT jobs to new work in Nov 2026; its replacement, the Training API, is a
client-driven loop like Tinker's (``fireworks-ai[training]``, ``FiretitanServiceClient``,
``create_lora_training_client``, ``forward_backward``, ``optim_step``, ``save_weights_for_sampler``).
https://docs.fireworks.ai/fine-tuning/training-api/introduction.md (read 2026-11-26)

Checked against the live API on 2026-11-26 (key from the user's account):

* Serverless training takes only the models Fireworks marks serverless; ``qwen3-4b`` and
  ``qwen3-8b`` are refused ("not available for serverless training"). ``qwen3p6-35b-a3b`` and
  ``muse-glimmer-30b`` are accepted. Small models need dedicated training (GPU hours, from
  $8/h), which this backend does not do.
* The server does not resolve tokenizers: we tokenize with the model's Hugging Face tokenizer.
* ``forward_backward(batch, "cross_entropy")`` with ``Datum(ModelInput.from_ints(...),
  {"target_tokens", "weights"})`` trains; ``metrics["loss:sum"]`` is the loss.

The SDK needs Python <= 3.14 (its pydantic-core does not build on 3.15), and OpenTrain may run on a
newer one, so the worker and the sampler run under ``uv run --python 3.12 --with fireworks-ai...``,
importing only OpenTrain's standard-library modules through a one-link path shim. Nothing is
installed into OpenTrain's own environment.

Answers. A trained version is the sampler snapshot (``fireworks-sampler://<path>``). Evals ask a
long-lived sampler process (``fireworks_sampler``) that holds one serverless session per snapshot
and exits after five idle minutes. Serving it to other apps needs a dedicated deployment ($8/h
and up, scale to zero after an hour idle), which this backend does not create on its own.

Prices: https://fireworks.ai/pricing (read 2026-11-26), USD per million tokens.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from ..plugs import JobStatus, TrainSpec
from . import JobStore

WORKER = "opentrain.backends.fireworks_worker"
SAMPLER = "opentrain.backends.fireworks_sampler"
KEY_ENV = "FIREWORKS_API_KEY"
TRAIN_URL = "https://api.fireworks.ai/training/v1/serverless"
SDK = "fireworks-ai[training]==1.2.39"
PYTHON = "3.12"
CHECKPOINT = "fireworks-sampler://"

PRICES_AS_OF = "2026-11-26"
# Fireworks model id -> (Hugging Face tokenizer, train $/Mtok, sample $/Mtok). Only models whose
# serverless training we saw accepted and whose tokenizer is documented.
MODELS: dict[str, tuple[str, float, float]] = {
    "qwen3p6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B", 2.14, 1.335),
    "muse-glimmer-30b": ("meta-models/Muse-Glimmer-30B", 5.86, 4.88),
}
ALIASES = {tok.lower(): fid for fid, (tok, _, _) in MODELS.items()}

CHARS_PER_TOKEN = 3.0
"""Pre-launch estimate only; the worker reports cost from real token counts. Low end, so the budget
is over- rather than under-reserved."""


class FireworksError(RuntimeError):
    pass


def model_id(base: str) -> str:
    """``qwen3p6-35b-a3b``, ``fireworks/qwen3p6-35b-a3b``, ``accounts/fireworks/models/...`` or the
    Hugging Face id ``Qwen/Qwen3.6-35B-A3B`` -> the short Fireworks id."""
    b = base.strip()
    for prefix in ("accounts/fireworks/models/", "fireworks/"):
        if b.startswith(prefix):
            b = b[len(prefix):]
    b = ALIASES.get(b.lower(), b)
    if b not in MODELS:
        raise FireworksError(
            f"{base!r} cannot train on Fireworks serverless. Choose one of: "
            + ", ".join(f"{fid} ({tok})" for fid, (tok, _, _) in MODELS.items())
            + ". Small models (under ~27B) need dedicated Fireworks training, or use trainer modal.")
    return b


def full_name(fid: str) -> str:
    return f"accounts/fireworks/models/{fid}"


def api_key() -> str | None:
    from ..core import credentials

    return credentials.env_for("fireworks", KEY_ENV)


def _shim() -> Path:
    """A directory whose only entry links to this ``opentrain`` package, for the 3.12 interpreter's
    path: it must import OpenTrain's pure modules without the newer interpreter's site-packages."""
    from ..server.service import state_home

    pkg = Path(__file__).resolve().parents[1]
    d = state_home() / "pyshim"
    d.mkdir(parents=True, exist_ok=True)
    link = d / "opentrain"
    if not (link.is_symlink() and link.resolve() == pkg):
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(pkg, target_is_directory=True)
    return d


def runner() -> tuple[list[str], dict[str, str]]:
    """The interpreter prefix and environment for the SDK side (worker, sampler)."""
    uv = shutil.which("uv")
    if not uv:
        raise FireworksError("Fireworks needs uv on PATH (https://docs.astral.sh/uv/) to run its SDK on Python 3.12")
    key = api_key()
    if not key:
        raise FireworksError(f"no Fireworks key: /connect fireworks or set {KEY_ENV}")
    cmd = [uv, "run", "--quiet", "--no-project", "--python", PYTHON,
           "--with", SDK, "--with", "transformers", "--with", "jinja2", "python"]
    return cmd, {KEY_ENV: key, "PYTHONPATH": str(_shim()), "TOKENIZERS_PARALLELISM": "false",
                 "TRANSFORMERS_VERBOSITY": "error"}


class FireworksBackend:
    name = "fireworks"

    def __init__(self, state: Path):
        self.store = JobStore(state)

    def available(self) -> tuple[bool, str]:
        if not shutil.which("uv"):
            return False, "install uv (https://docs.astral.sh/uv/): the Fireworks SDK runs on Python 3.12 through it"
        if not api_key():
            return False, f"connect a key: /connect fireworks (or set {KEY_ENV})"
        return True, ("ready: serverless LoRA on " + ", ".join(MODELS)
                      + "; small models need trainer modal")

    def estimate(self, spec: TrainSpec) -> float:
        _, train, _ = MODELS[model_id(spec.base_model)]
        tokens = 0
        for line in Path(spec.data).read_text().splitlines():
            if line.strip():
                tokens += min(int(len(line) / CHARS_PER_TOKEN) + 1, spec.max_tokens)
        return round(tokens * spec.epochs * train / 1e6, 6)

    def launch(self, spec: TrainSpec) -> str:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"fireworks backend unavailable: {why}")
        if not Path(spec.data).is_file():
            raise FileNotFoundError(f"training data {spec.data} does not exist")
        fid = model_id(spec.base_model)
        tok, train, sample = MODELS[fid]
        job = self.store.new(self.name, spec, usd_per_mtok=train, fireworks_model=fid, tokenizer=tok,
                             detail=f"starting; cost from tokens x ${train}/Mtok (Fireworks prices {PRICES_AS_OF})")
        cmd, env = runner()
        self.store.spawn(WORKER, job["job_id"], python=cmd, env=env)
        return job["job_id"]

    def status(self, job_id: str) -> JobStatus:
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)


# ---------------------------------------------------------------- answers from a trained snapshot


class _Sampler:
    """One ``fireworks_sampler`` process for one snapshot: JSON lines in, JSON lines out."""

    def __init__(self, snapshot: str, base: str):
        fid = model_id(base)
        cmd, env = runner()
        self.proc = subprocess.Popen([*cmd, "-m", SAMPLER, snapshot, fid, MODELS[fid][0]],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, env={**os.environ, **env}, start_new_session=True)
        self.lock = threading.Lock()
        ready = self._read(timeout=600)
        if not ready.get("ready"):
            self.close()
            raise FireworksError(f"the Fireworks sampler did not start: {ready.get('error', 'no reply')}")

    def _read(self, timeout: float) -> dict[str, Any]:
        out: list[str] = []
        t = threading.Thread(target=lambda: out.append(self.proc.stdout.readline()), daemon=True)
        t.start()
        t.join(timeout)
        if not out or not out[0]:
            return {"error": "no answer from the sampler (exited or timed out)"}
        return json.loads(out[0])

    def ask(self, messages: list[dict[str, str]], max_tokens: int, timeout: float) -> dict[str, Any]:
        with self.lock:
            if self.proc.poll() is not None:
                raise FireworksError("the Fireworks sampler exited")
            self.proc.stdin.write(json.dumps({"messages": messages, "max_tokens": max_tokens}) + "\n")
            self.proc.stdin.flush()
            r = self._read(timeout)
        if r.get("error"):
            raise FireworksError(r["error"])
        return r

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()


_samplers: dict[str, _Sampler] = {}
_samplers_lock = threading.Lock()


def sample(checkpoint: str, base: str, messages: list[dict[str, str]], *, max_tokens: int = 512,
           timeout: float = 300.0) -> tuple[str, int, int, float]:
    """(text, prompt tokens, completion tokens, cost) from a trained snapshot."""
    snapshot = checkpoint.removeprefix(CHECKPOINT)
    with _samplers_lock:
        s = _samplers.get(snapshot)
        if s is None or s.proc.poll() is not None:
            s = _samplers[snapshot] = _Sampler(snapshot, base)
    r = s.ask(messages, max_tokens, timeout)
    price = MODELS[model_id(base)][2]
    p, c = int(r.get("prompt_tokens", 0)), int(r.get("completion_tokens", 0))
    return r["text"], p, c, round((p + c) * price / 1e6, 6)
