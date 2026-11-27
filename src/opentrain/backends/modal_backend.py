"""Modal: train on a Modal GPU, and serve base models and trained versions from Modal.

Modal is compute, not a fine-tuning API: OpenTrain ships its own trainer (TRL + PEFT LoRA, in
``opentrain_modal.py``) and runs it there. One code path, any GPU Modal offers, billed per second
on your own Modal account. Credentials are Modal's own (``~/.modal.toml`` from ``modal setup``, or
``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``).

Training. ``launch`` spawns a detached worker (``python -m opentrain.backends.modal_backend
<job.json>``) like the local and Tinker backends. The worker runs the Modal app ephemerally and
streams the trainer's steps (loss, progress) into the job file. Cancelling stops the app, which
stops the GPU. The adapter lands on the ``opentrain`` Modal volume; the checkpoint is
``modal://<job_id>``.

Serving. ``endpoint(base, adapter)`` returns an OpenAI-compatible endpoint: one deployed app per
base model (vLLM, scale to zero), serving the base as ``base`` and every adapter trained on it under
its job id. Adding an adapter redeploys that app (rolling, no downtime). Requests carry a key
OpenTrain generated and keeps in the auth store (``OPENTRAIN_MODAL_KEY``). The first request after
idle waits for a cold start (a few minutes the first time the image builds, then about a minute).

Cost. Training is billed GPU seconds at Modal's list price (https://modal.com/pricing, read
2026-11-26), measured from the container's start to the adapter save. Serving is attributed per
answer as its latency times the GPU's price divided by the answers in flight at once, so a
batch of eight concurrent answers splits the GPU between them. Cold starts and idle time before
scale-down are not attributed to any answer; Modal's bill is the truth.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..models import serve_context, vllm_tool_args
from ..plugs import Endpoint, JobStatus, TrainSpec
from . import JobStore, missing_modules, spec_from_dict, spec_to_dict, worker_main

APP_FILE = Path(__file__).with_name("opentrain_modal.py")
KEY_ENV = "OPENTRAIN_MODAL_KEY"
KEY_ID = "modal-serve"

# $/second, modal.com/pricing (read 2026-11-26)
RATES = {"T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583,
         "A100-80GB": 0.000694, "H100": 0.001097, "H200": 0.001261, "B200": 0.001736}
CPU_RATE = 0.0000131 * 2 + 0.00000222 * 16  # the container's cores and memory alongside the GPU


class ModalError(RuntimeError):
    pass


def params_b(model: str) -> float | None:
    """Billions of parameters, read from the model's name (``Qwen3-8B`` -> 8, ``30B-A3B`` -> 30)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model.split("/")[-1])
    return float(m.group(1)) if m else None


def train_gpu(model: str) -> list[str]:
    """GPUs to train on, in order of preference: Modal takes the first one it has free."""
    p = params_b(model) or 8
    if p <= 8:
        return ["L40S", "A100-80GB", "H100"]
    if p <= 16:
        return ["H100", "A100-80GB", "H200"]
    if p <= 40:
        return ["H200", "B200"]
    raise ModalError(f"{model}: LoRA on one GPU supports models up to about 40B here")


def loop_gpu(model: str, method: str, teacher: str | None = None) -> list[str]:
    """GPUs for rejection sampling, distillation and RL: the policy, a vLLM sampler (about a third of
    the GPU) and, for distillation, the teacher share one GPU."""
    if method == "sft":
        return train_gpu(model)
    p = (params_b(model) or 8) + ((params_b(teacher) or 8) if method == "opd" else 0)
    if p <= 6:
        return ["L40S", "A100-80GB", "H100"]
    if p <= 16:
        return ["A100-80GB", "H100", "H200"]
    if p <= 30:
        return ["H200", "B200"]
    raise ModalError(f"{model} ({method}{', teacher ' + teacher if teacher else ''}): too large for one GPU here; "
                     "use a smaller student or teacher")


def gpu_seconds(spec: TrainSpec, rows: list[dict[str, Any]]) -> float:
    """A rough duration. Throughputs are rules of thumb for LoRA on one GPU (training ~8k tokens/s
    for a 1B model; sampling ~1.5k tokens/s with vLLM in eager mode with a LoRA, as measured on RL
    runs), scaled by model size; startup is image + weights."""
    from .gpu_train import settings

    p = max(params_b(spec.base_model) or 8, 0.5)
    train_tps, sample_tps = 8000 / p, 1500 / max(1.0, p / 2)
    prompt = sum(len(json.dumps(r.get("messages", [])[:-1])) / 4 for r in rows) / max(1, len(rows))
    answer = sum(len(str(r.get("messages", [{}])[-1].get("content", ""))) / 4 for r in rows) / max(1, len(rows))
    if spec.method == "sft":
        return 240 + len(rows) * (prompt + answer) * spec.epochs / train_tps
    s = settings({"method": spec.method, "extra": spec.extra})
    gen = int(s["max_new_tokens"]) * 0.6  # most answers stop before the cap
    if spec.method == "rejection":
        n = len(rows) * int(s["samples"]) * int(s["rounds"])
        kept = len(rows) * int(s["keep"]) * int(s["rounds"])
        return 420 + n * gen / sample_tps + kept * (prompt + gen) / train_tps
    n = int(s["steps"]) * int(s["prompts_per_step"]) * int(s["samples"])
    seconds = 420 + n * gen / sample_tps + n * (prompt + gen) * 2 / train_tps
    if spec.method == "opd":
        t = max(params_b(s.get("teacher") or "") or 8, 0.5)
        seconds += 120 + n * (prompt + gen) / (24000 / t)
    return seconds


def serve_gpu(model: str) -> list[str]:
    p = params_b(model) or 8
    if p <= 4:
        return ["L4", "A10", "L40S"]
    if p <= 14:
        return ["L40S", "A100-80GB", "H100"]
    return ["H200", "H100"]


def rate(gpu: str | list[str]) -> float:
    """$/s for a GPU (``L40S``, ``H100:2``) or, for a list, the dearest one (an upper bound)."""
    if isinstance(gpu, list):
        return max(rate(g) for g in gpu)
    return RATES[gpu.split(":")[0]] * int(gpu.split(":")[1] if ":" in gpu else 1) + CPU_RATE


def device_gpu(name: str) -> str | None:
    """Modal's name for the GPU torch reports (``NVIDIA A100-SXM4-80GB`` -> ``A100-80GB``)."""
    n = name.upper()
    for key, test in (("B200", "B200"), ("H200", "H200"), ("H100", "H100"), ("L40S", "L40S"), ("A10", "A10G"),
                      ("L4", "L4"), ("T4", "T4")):
        if test in n:
            return key
    if "A100" in n:
        return "A100-80GB" if "80" in n else "A100-40GB"
    if n.endswith(" A10") or "A10 " in n:
        return "A10"
    return None


def configured() -> tuple[bool, str]:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True, "MODAL_TOKEN_ID"
    cfg = Path(os.environ.get("MODAL_CONFIG_PATH") or Path.home() / ".modal.toml")
    if cfg.is_file():
        text = cfg.read_text()
        prof = os.environ.get("MODAL_PROFILE") or next(
            (m.group(1) for m in re.finditer(r"^\[(.+?)\]\s*$((?:\n(?!\[).*)*)", text, re.M)
             if re.search(r"^active\s*=\s*true", m.group(2), re.M)), None)
        return True, f"~/.modal.toml{f' (profile {prof})' if prof else ''}"
    return False, "not signed in: run `modal setup`, or /connect modal with token_id:token_secret"


def load_app(**env: str):
    """Import the Modal app module with this configuration (a fresh module each time)."""
    for k, v in env.items():
        os.environ[k] = v
    spec = importlib.util.spec_from_file_location("opentrain_modal", APP_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["opentrain_modal"] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ training


class ModalBackend:
    name = "modal"

    def __init__(self, state: Path):
        self.state = Path(state)
        self.store = JobStore(state)

    def available(self) -> tuple[bool, str]:
        if missing_modules("modal"):
            return False, "missing modal: pip install 'opentrain-cli[modal]'"
        ok, how = configured()
        return (True, f"ready ({how})") if ok else (False, how)

    def estimate(self, spec: TrainSpec) -> float:
        """GPU seconds at list price, padded 20-30%: see ``gpu_seconds``. The job reports the real cost."""
        rows = _rows(spec.data)
        # Modal may run it on any GPU in the list: price the dearest, so the budget's hold covers the run
        gpu = spec.extra.get("gpu") or loop_gpu(spec.base_model, spec.method, spec.extra.get("teacher"))
        pad = 1.2 if spec.method == "sft" else 1.3
        return round(gpu_seconds(spec, rows) * rate(gpu) * pad, 4)

    def launch(self, spec: TrainSpec) -> str:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"modal backend unavailable: {why}")
        if not Path(spec.data).is_file():
            raise FileNotFoundError(f"training data {spec.data} does not exist")
        gpus = ([spec.extra["gpu"]] if spec.extra.get("gpu")
                else loop_gpu(spec.base_model, spec.method, spec.extra.get("teacher")))
        job = self.store.new(self.name, spec, detail=f"starting on Modal ({gpus[0]})", gpu=gpus)
        self.store.spawn("opentrain.backends.modal_backend", job["job_id"])
        return job["job_id"]

    def status(self, job_id: str) -> JobStatus:
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)

    def stop_early(self, job_id: str) -> bool:
        """End at the next step and keep the adapter (the budget ran out); False if it could not ask."""
        return request_stop(job_id)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def is_trace(row: dict[str, Any]) -> bool:
    """An agent trace: several assistant turns (tool calls between them). Every one is trained on."""
    return sum(1 for m in row.get("messages") or [] if m.get("role") == "assistant") > 1


def sft_row(row: dict[str, Any]) -> dict[str, Any]:
    """What the Modal trainer gets for one example: a prompt-completion pair (loss on the answer),
    or, for an agent trace, the whole conversation (loss on every assistant turn)."""
    if is_trace(row):
        from .tinker import hf_messages

        out: dict[str, Any] = {"messages": hf_messages(row["messages"])}
        if row.get("tools"):
            out["tools"] = row["tools"]
        return out
    return prompt_completion(row)


def prompt_completion(row: dict[str, Any]) -> dict[str, Any]:
    """A chat example as TRL's prompt-completion pair: the loss falls on the last assistant turn only,
    for any chat template (no generation markers needed)."""
    from .tinker import hf_messages

    msgs = hf_messages(row["messages"])
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError("a training example must end with the assistant's answer")
    out = {"prompt": msgs[:-1], "completion": msgs[-1:]}
    if row.get("tools"):
        out["tools"] = row["tools"]
    return out


FOLLOW_SILENCE = 3 * 3600  # following by polling: give up after this long with no new event


def run(store: JobStore, job: dict[str, Any]) -> None:
    import modal

    jid = job["job_id"]
    spec = spec_from_dict(job["spec"])
    gpus = job.get("gpu") or loop_gpu(spec.base_model, spec.method, spec.extra.get("teacher"))
    gpus = [gpus] if isinstance(gpus, str) else gpus
    rows = [sft_row(r) if spec.method == "sft" else prompt_completion(r) for r in _rows(spec.data)]
    store.log(jid, f"{len(rows)} examples; {spec.method} on {spec.base_model}; LoRA rank {spec.lora_rank}; "
                   f"GPU {' or '.join(gpus)}")
    mod = load_app(OT_MODAL_MODE="train", OT_TRAIN_GPU=json.dumps(gpus), OT_LOOP_GPU=json.dumps(gpus))
    fn = mod.train if spec.method == "sft" else mod.train_loop
    st: dict[str, Any] = {"gpu": gpus[0], "per_s": rate(gpus), "started": None, "seen": 0, "app": None}

    def handle(ev: dict[str, Any]) -> bool:
        """Record one event; True when the job is over (done). An error event raises."""
        st["seen"] += 1
        now = time.time()
        if ev["event"] == "started":
            st["started"] = now
            st["gpu"] = device_gpu(ev["gpu"]) or st["gpu"]
            st["per_s"] = rate(st["gpu"])
            store.log(jid, f"container up on {ev['gpu']}: {ev['versions']}")
            store.update(jid, detail=f"loading {spec.base_model} on {ev['gpu']}")
        cost = round((now - st["started"]) * st["per_s"], 4) if st["started"] else 0.0
        gpu = st["gpu"]
        if ev["event"] == "step":
            total = ev["total"] or 1
            extra = {k: ev[k] for k in ("reward", "kl") if k in ev}
            shown = (f" · reward {ev['reward']:.0%}" if "reward" in ev else "") + (
                f" · kl {ev['kl']:.3f}" if "kl" in ev else "")
            store.update(jid, progress=round(ev["step"] / total, 4), loss=ev["loss"], cost_usd=cost,
                         detail=f"step {ev['step']}/{total} · loss {ev['loss']:.3f}{shown} · {gpu}", **extra)
            store.log(jid, f"step {ev['step']}/{total} loss {ev['loss']:.4f}"
                           + "".join(f" {k}={v}" for k, v in extra.items()) + (f" ({ev['note']})" if ev.get("note") else ""))
        elif ev["event"] == "log":
            store.log(jid, ev["note"])
            store.update(jid, detail=f"{ev['note']} · {gpu}", cost_usd=cost)
        elif ev["event"] == "error":
            if ev.get("trace"):
                store.log(jid, ev["trace"])
            raise ModalError(ev["error"])
        elif ev["event"] == "done":
            store.log(jid, f"adapter saved on the opentrain volume: {ev['adapter']} ({ev['seconds']}s)")
            if ev.get("rounds"):
                store.update(jid, rounds=ev["rounds"])
            early = "stopped early (budget), adapter kept · " if ev.get("stopped_early") else ""
            store.update(jid, state="succeeded", progress=1.0, checkpoint=f"modal://{jid}", cost_usd=cost,
                         adapter_rank=spec.lora_rank, stopped_early=bool(ev.get("stopped_early")),
                         detail=f"{early}done in {round(ev['seconds'] / 60, 1)} min on {gpu} · ${cost:.2f}")
            return True
        return False

    def cancelled() -> bool:
        if not store.cancel_requested(jid):
            return False
        store.log(jid, "cancelled; stopping the Modal app")
        stop_app(st["app"])
        cost = round((time.time() - st["started"]) * st["per_s"], 4) if st["started"] else 0.0
        store.update(jid, state="cancelled", cost_usd=cost, detail="cancelled; no adapter saved")
        return True

    try:
        with modal.enable_output(), mod.app.run(detach=True):
            st["app"] = mod.app.app_id
            store.update(jid, modal_app=st["app"])
            for ev in fn.remote_gen(jid, spec_to_dict(spec), rows):
                if handle(ev) or cancelled():
                    break
            else:
                raise ModalError("the Modal app ended without saving an adapter")
        stop_app(st["app"])
        return
    except ModalError:
        stop_app(st["app"])
        raise
    except Exception as exc:  # noqa: BLE001 - a dropped connection; the detached job goes on
        if not st["app"]:
            raise
        store.log(jid, f"lost the stream from Modal ({type(exc).__name__}: {exc}); the job keeps running "
                       "there, following its event log instead")
    _follow(store, jid, st, handle, cancelled)


def _follow(store: JobStore, jid: str, st: dict[str, Any], handle, cancelled) -> None:
    """Poll the job's events from the Modal Dict until it ends, riding out network errors."""
    import modal

    d = modal.Dict.from_name("opentrain-events", create_if_missing=True)
    last_new, wait = time.time(), 10
    while True:
        try:
            n = d.get(f"{jid}/n", 0)
            while st["seen"] < n:
                if handle(d[f"{jid}/{st['seen']}"]):
                    stop_app(st["app"])
                    return
                last_new = time.time()
            wait = 10
        except ModalError:
            stop_app(st["app"])
            raise
        except Exception as exc:  # noqa: BLE001 - still offline: back off and try again
            store.update(jid, detail=f"reconnecting to Modal ({type(exc).__name__})")
            wait = min(wait * 2, 120)
        if cancelled():
            return
        if time.time() - last_new > FOLLOW_SILENCE:
            stop_app(st["app"])
            raise ModalError(f"no news from the Modal job for {FOLLOW_SILENCE // 3600} hours; stopped it")
        time.sleep(wait)


def request_stop(job_id: str) -> bool:
    """Ask a running Modal job to end at its next step and save its adapter (see ``EventLog``)."""
    import modal

    try:
        modal.Dict.from_name("opentrain-events", create_if_missing=True)[f"{job_id}/stop"] = True
        return True
    except Exception:  # noqa: BLE001 - offline: the caller falls back to cancelling
        return False


def stop_app(app_id: str | None) -> None:
    """Stop a detached training app (it stops by itself when its call ends; this makes sure)."""
    if not app_id:
        return
    try:
        subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        pass


# ------------------------------------------------------------------ serving


_lock = threading.Lock()
_inflight = {"n": 0}


def _deployments_path() -> Path:
    from ..server.service import data_home

    return Path(os.environ.get("OPENTRAIN_MODAL_STATE") or data_home() / "modal.json")


def _deployments() -> dict[str, Any]:
    try:
        return json.loads(_deployments_path().read_text())
    except (OSError, ValueError):
        return {}


def serving_key() -> str:
    """The key every OpenTrain endpoint on Modal requires; generated once, kept in the auth store."""
    from ..core import credentials

    key = os.environ.get(KEY_ENV) or credentials.get_or_create(KEY_ID, lambda: "ot-" + secrets.token_urlsafe(24))
    os.environ[KEY_ENV] = key
    return key


def deploy(base: str, adapters: dict[str, int] | None = None) -> str:
    """Deploy (or redeploy) the serving app for ``base`` with these adapters; return its base URL."""
    env = {**os.environ, "OT_MODAL_MODE": "serve", "OT_SERVE_BASE": base,
           "OT_SERVE_ADAPTERS": json.dumps(adapters or {}), "OT_SERVE_GPU": json.dumps(serve_gpu(base)),
           "OPENTRAIN_SERVE_KEY": serving_key(), "OT_SERVE_ARGS": json.dumps(vllm_tool_args(base)),
           "OT_SERVE_MAX_LEN": str(serve_context(base))}
    p = subprocess.run([sys.executable, "-m", "modal", "deploy", str(APP_FILE)], env=env, capture_output=True,
                       text=True, timeout=1800)
    if p.returncode != 0:
        raise ModalError(f"modal deploy failed: {(p.stderr or p.stdout).strip()[-800:]}")
    import modal

    name = f"opentrain-serve-{_slug(base)}"
    try:
        return modal.Function.from_name(name, "serve").get_web_url().rstrip("/")
    except Exception as exc:  # noqa: BLE001 - the SDK raises several types here
        raise ModalError(f"deployed {name}, but could not read its URL: {exc}") from None


def _slug(base: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:40]


def endpoint(base: str, adapter: str | None = None, rank: int = 32) -> Endpoint:
    """The endpoint answering for ``base`` (``adapter=None``) or for one adapter on it, deploying
    or redeploying its app first when needed."""
    serving_key()
    with _lock:
        d = _deployments()
        cur = d.get(base) or {}
        want = dict(cur.get("adapters") or {})
        if adapter:
            want[adapter] = rank
        if not cur.get("url") or want != (cur.get("adapters") or {}):
            url = deploy(base, want)
            d[base] = {"url": url, "adapters": want, "gpu": serve_gpu(base), "deployed": time.time()}
            path = _deployments_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(d, indent=1))
            cur = d[base]
    return Endpoint(base_url=cur["url"] + "/v1", model=adapter or "base", api_key_env=KEY_ENV)


@contextmanager
def answering(base: str):
    """Count answers in flight on a Modal endpoint; yields a function giving this answer's cost."""
    with _lock:
        _inflight["n"] += 1
    t0 = time.time()
    peak = [_inflight["n"]]
    try:
        yield lambda: round((time.time() - t0) * rate(serve_gpu(base)[0]) / max(1, peak[0]), 6)
    finally:
        with _lock:
            _inflight["n"] -= 1


_ready: dict[tuple[str, str], float] = {}


def wait_ready(ep: Endpoint, timeout: float = 1500) -> None:
    """Block until a scaled-to-zero endpoint answers ``/models`` (cold start), or raise. Skipped for
    an endpoint seen ready in the last minute (it scales down after minutes of idle, not seconds)."""
    import urllib.request

    if time.time() - _ready.get((ep.base_url, ep.model), 0) < 60:
        return
    t_end = time.time() + timeout
    last = ""
    while time.time() < t_end:
        req = urllib.request.Request(f"{ep.base_url}/models", headers={"Authorization": f"Bearer {serving_key()}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                names = [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
                if ep.model in names:
                    _ready[(ep.base_url, ep.model)] = time.time()
                    return
                last = f"{ep.model} not among {names}"
        except Exception as exc:  # noqa: BLE001 - cold starts surface as several error types
            last = str(exc)
        time.sleep(5)
    raise ModalError(f"{ep.base_url} did not come up: {last}")


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:], run))
