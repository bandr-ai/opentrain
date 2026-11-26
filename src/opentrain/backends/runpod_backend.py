"""RunPod: train on a RunPod GPU pod, and serve base models and trained versions from one.

Like Modal, RunPod is compute: OpenTrain runs its own trainer (``gpu_train``: SFT, rejection
sampling, on-policy distillation, RL) on a pod, from the public image ``vllm/vllm-openai`` (torch,
vLLM, transformers), on your RunPod account (key: ``/connect runpod`` or ``RUNPOD_API_KEY``).

Training. A detached worker creates a pod (the first of the GPUs that has capacity; community cloud
first, then secure), whose start command runs ``runpod_agent`` (shipped in the pod's environment).
The worker posts the trainer, the graders and the rows to the agent, follows its events, downloads
the adapter (kept under the project's state, so a later pod can serve it), then tells the same pod to
serve it. The checkpoint is ``runpod://<job_id>``. Cancelling or failing deletes the pod.

Serving. ``endpoint(base, adapter)`` reuses the live pod serving ``base`` (uploading the adapter and
restarting vLLM when it is missing) or creates one. Pods do not scale to zero: the agent deletes its
own pod after 15 idle minutes (``OPENTRAIN_RUNPOD_IDLE``) and after 12 hours in any case, so nothing
bills on after OpenTrain is gone. For that it is given your key in the pod's environment (visible
only to your account). Requests carry a key OpenTrain generated (``OPENTRAIN_RUNPOD_KEY``).

Cost. Billed per second at the pod's own hourly price (the ``cost`` RunPod reports for it), from
creation to deletion; serving is attributed per answer like Modal's (latency x price / answers in
flight). RunPod's bill is the truth.

API: https://docs.runpod.io/api-reference-v2 (read 2026-11-26): ``POST /v2/pods`` with ``name``,
``image``, ``gpu {id, count}``, ``cloud``, ``disk``, ``ports ["8000/http"]``, ``env``,
``entrypoint``/``cmd``; ``GET /v2/pods/{id}`` (``status``, ``cost``); ``DELETE /v2/pods/{id}``.
HTTP ports are reached at ``https://<id>-<port>.proxy.runpod.net`` (100 s per request).
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..plugs import Endpoint, JobStatus, TrainSpec
from . import JobStore, spec_from_dict, spec_to_dict, worker_main

API = "https://api.runpod.io/v2"
IMAGE = "vllm/vllm-openai:v0.21.0"
KEY_ENV = "RUNPOD_API_KEY"
SERVE_KEY_ENV = "OPENTRAIN_RUNPOD_KEY"
SERVE_KEY_ID = "runpod-serve"
HERE = Path(__file__).resolve().parent

# OpenTrain GPU name -> RunPod GPU type ids (https://docs.runpod.io/references/gpu-types.md)
GPU_IDS = {
    "L4": ["NVIDIA L4"],
    "RTX4090": ["NVIDIA GeForce RTX 4090"],
    "L40S": ["NVIDIA L40S"],
    "A100-80GB": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    "H100": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe"],
    "H200": ["NVIDIA H200"],
    "B200": ["NVIDIA B200"],
}
# $/hour, community cloud, https://www.runpod.io/pricing (updated 2026-11-12, read 2026-11-26).
# Estimates only: a pod's real price is what RunPod reports for it.
HOURLY = {"L4": 0.44, "RTX4090": 0.34, "L40S": 0.79, "A100-80GB": 1.19, "H100": 1.99, "H200": 3.59, "B200": 5.98}


class RunPodError(RuntimeError):
    pass


def api_key() -> str | None:
    """``/connect runpod``, ``RUNPOD_API_KEY``, or the key ``runpodctl`` keeps in ``~/.runpod/config.toml``
    (``runpodctl doctor`` writes it), so a machine already signed in to RunPod's CLI needs nothing more."""
    from ..core import credentials

    return credentials.env_for("runpod", KEY_ENV) or runpodctl_key()


def runpodctl_key() -> str | None:
    """The key RunPod's own tools keep in ``~/.runpod/config.toml``: ``[default] api_key`` (``flash
    login``, runpod-python) or the top-level ``apikey`` (``runpodctl``)."""
    import tomllib

    path = Path(os.environ.get("RUNPOD_CONFIG") or Path.home() / ".runpod" / "config.toml")
    try:
        doc = tomllib.loads(path.read_text())
    except (OSError, ValueError):
        return None
    key = (doc.get("default") or {}).get("api_key") or doc.get("apikey") or doc.get("apiKey")
    return str(key).strip() if key and str(key).strip() else None


GRAPHQL = "https://api.runpod.io/graphql"
CONSOLE = "https://console.runpod.io"


def _graphql(query: str, variables: dict | None = None) -> dict[str, Any]:
    req = urllib.request.Request(GRAPHQL, data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "opentrain"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed https host
        out = json.loads(r.read())
    if out.get("errors"):
        raise RunPodError("RunPod: " + "; ".join(e.get("message", "?") for e in out["errors"]))
    return out.get("data") or {}


def login_start() -> dict[str, str]:
    """Start a browser sign-in (the approval flow RunPod's own ``flash login`` uses): returns the
    request id and the console URL where the user approves it."""
    req = _graphql("mutation { createFlashAuthRequest { id status expiresAt } }").get("createFlashAuthRequest") or {}
    if not req.get("id"):
        raise RunPodError("RunPod did not start a sign-in request")
    return {"request": req["id"], "url": f"{CONSOLE}/flash/login?request={req['id']}"}


def login_check(request: str) -> str | None:
    """``None`` while waiting; once approved, keeps the key in OpenTrain's auth store and returns a
    hint of it. Raises when the request was denied or expired."""
    from ..core import integrations

    q = "query($id: String!) { flashAuthRequestStatus(flashAuthRequestId: $id) { status apiKey } }"
    st = _graphql(q, {"id": request}).get("flashAuthRequestStatus") or {}
    if st.get("apiKey") and st.get("status") in ("APPROVED", "CONSUMED"):
        integrations.connect("runpod", st["apiKey"])
        return st["apiKey"][:6] + "…"
    if st.get("status") in ("DENIED", "EXPIRED", "CONSUMED"):
        raise RunPodError(f"RunPod sign-in {str(st['status']).lower()}")
    return None


def browser_login(*, open_browser: bool = True, timeout: float = 600, say=print) -> str:
    """Sign in to RunPod in the browser, no key pasted. Returns a hint of the key."""
    import webbrowser

    start = login_start()
    say(f"Approve OpenTrain's access in your browser:\n  {start['url']}")
    if open_browser:
        webbrowser.open(start["url"])
    end = time.time() + timeout
    while time.time() < end:
        hint = login_check(start["request"])
        if hint:
            return hint
        time.sleep(2)
    raise RunPodError("RunPod sign-in timed out")


def _api(method: str, path: str, body: dict | None = None, *, timeout: float = 60) -> tuple[int, Any]:
    key = api_key()
    if not key:
        raise RunPodError(f"no RunPod key: /connect runpod or set {KEY_ENV}")
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https host
            raw = r.read()
            return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw.decode(errors="replace")[:500]}


def serving_key() -> str:
    from ..core import credentials

    key = os.environ.get(SERVE_KEY_ENV) or credentials.get_or_create(SERVE_KEY_ID,
                                                                     lambda: "ot-" + secrets.token_urlsafe(24))
    os.environ[SERVE_KEY_ENV] = key
    return key


def create_pod(name: str, gpus: list[str], *, log=print, clouds=("COMMUNITY", "SECURE")) -> dict[str, Any]:
    """A pod running the agent on the first GPU with capacity. Raises with every refusal if none."""
    agent = base64.b64encode((HERE / "runpod_agent.py").read_bytes()).decode()
    env = {"OT_BOOT": agent, "OT_TOKEN": serving_key(), "OT_RUNPOD_KEY": api_key() or "",
           "OT_IDLE": os.environ.get("OPENTRAIN_RUNPOD_IDLE", "900"),
           "OT_MAX_HOURS": os.environ.get("OPENTRAIN_RUNPOD_MAX_HOURS", "12"), "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    refusals = []
    for gpu in gpus:
        for gid in GPU_IDS.get(gpu, [gpu]):
            for cloud in clouds:
                body = {"name": name[:60], "image": IMAGE, "gpu": {"id": gid, "count": 1}, "cloud": cloud,
                        "disk": 80, "ports": ["8000/http"], "env": env, "entrypoint": ["/bin/bash", "-c"],
                        "cmd": ["echo \"$OT_BOOT\" | base64 -d > /opt/ot_agent.py && exec python3 /opt/ot_agent.py"]}
                for attempt in range(3):
                    code, out = _api("POST", "/pods", body)
                    if code != 429:
                        break
                    time.sleep(10 * (attempt + 1))
                if code in (200, 201) and out and out.get("id"):
                    log(f"pod {out['id']} created: {gid} ({cloud.lower()} cloud), ${out.get('cost', '?')}/h")
                    return {"id": out["id"], "gpu": gpu, "gpu_id": gid, "cloud": cloud,
                            "hourly": float(out.get("cost") or HOURLY.get(gpu, 2.0)),
                            "url": f"https://{out['id']}-8000.proxy.runpod.net"}
                if code == 402:
                    raise RunPodError("RunPod says the account balance is too low to start a pod")
                if code == 401:
                    raise RunPodError("RunPod refused the key (401): check /connect runpod")
                refusals.append(f"{gid}/{cloud.lower()}: {code} {str((out or {}).get('error') or out)[:120]}")
    raise RunPodError("no RunPod GPU available: " + "; ".join(refusals))


def delete_pod(pod_id: str) -> None:
    try:
        _api("DELETE", f"/pods/{pod_id}")
    except RunPodError:
        pass


def _agent(url: str, method: str, path: str, body: bytes | dict | None = None, *, timeout: float = 90,
           raw: bool = False) -> Any:
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(url + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {serving_key()}",
                                          "Content-Type": "application/json" if isinstance(body, dict)
                                          else "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - the pod's proxy URL
        out = r.read()
    return out if raw else json.loads(out)


def wait_agent(pod: dict[str, Any], *, timeout: float = 1800, log=print, cancelled=lambda: False) -> dict:
    """Until the agent answers and has installed what it needs (image pull + boot: minutes)."""
    end, last, said = time.time() + timeout, "", 0.0
    while time.time() < end:
        if cancelled():
            raise RunPodError("cancelled")
        try:
            h = _agent(pod["url"], "GET", "/health", timeout=20)
            if h.get("ready"):
                if h.get("boot_error"):
                    raise RunPodError(f"the pod could not install its packages: {h['boot_error']}")
                return h
            last = "installing packages"
        except RunPodError:
            raise
        except Exception as e:  # noqa: BLE001 - proxy 404/502 until the container listens
            last = str(e)[:120]
            code, info = _api("GET", f"/pods/{pod['id']}")
            if code == 404 or (info or {}).get("status") in ("TERMINATED", "ERROR", "EXITED"):
                raise RunPodError(f"pod {pod['id']} stopped while starting: {(info or {}).get('status', code)}")
        if time.time() - said > 60:
            log(f"waiting for the pod: {last}")
            said = time.time()
        time.sleep(10)
    raise RunPodError(f"pod {pod['id']} did not come up in {int(timeout / 60)} minutes: {last}")


class RunPodBackend:
    name = "runpod"

    def __init__(self, state: Path):
        self.store = JobStore(state)
        self.state = Path(state)

    def available(self) -> tuple[bool, str]:
        if not api_key():
            return False, f"connect a key: /connect runpod (or set {KEY_ENV})"
        return True, "ready: GPU pods on your RunPod account (sft, rejection, opd, rl)"

    def estimate(self, spec: TrainSpec) -> float:
        from .modal_backend import _rows, gpu_seconds, loop_gpu

        gpus = loop_gpu(spec.base_model, spec.method, spec.extra.get("teacher"))
        seconds = gpu_seconds(spec, _rows(spec.data)) + 300  # the 10 GB image pull, on top of Modal's startup
        # the pod lands on whichever GPU has capacity: price the dearest, so the budget's hold covers it
        return round(seconds / 3600 * max(HOURLY.get(g, 2.0) for g in gpus) * 1.3, 4)

    def launch(self, spec: TrainSpec) -> str:
        from .modal_backend import loop_gpu

        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"runpod backend unavailable: {why}")
        if not Path(spec.data).is_file():
            raise FileNotFoundError(f"training data {spec.data} does not exist")
        gpus = [spec.extra["gpu"]] if spec.extra.get("gpu") else loop_gpu(spec.base_model, spec.method,
                                                                          spec.extra.get("teacher"))
        job = self.store.new(self.name, spec, detail=f"creating a RunPod pod ({gpus[0]})", gpu=gpus)
        self.store.spawn("opentrain.backends.runpod_backend", job["job_id"],
                         env={KEY_ENV: api_key() or "", SERVE_KEY_ENV: serving_key()})
        return job["job_id"]

    def status(self, job_id: str) -> JobStatus:
        return self.store.status(job_id)

    def cancel(self, job_id: str) -> None:
        self.store.request_cancel(job_id)


def adapter_path(job_id: str) -> Path:
    """Where OpenTrain keeps a RunPod-trained adapter, so a later pod can serve it (job ids are unique)."""
    from ..server.service import data_home

    d = Path(os.environ.get("OPENTRAIN_RUNPOD_ADAPTERS") or data_home() / "runpod-adapters")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job_id}.tar.gz"


def run(store: JobStore, job: dict[str, Any]) -> None:
    from .modal_backend import _rows, prompt_completion

    jid = job["job_id"]
    spec = spec_from_dict(job["spec"])
    rows = [prompt_completion(r) for r in _rows(spec.data)]
    gpus = job.get("gpu") or ["L40S"]

    def log(line: str) -> None:
        store.log(jid, line)

    log(f"{len(rows)} examples; {spec.method} on {spec.base_model}; LoRA rank {spec.lora_rank}; GPU {' or '.join(gpus)}")
    pod = create_pod(f"opentrain-{jid}", gpus, log=log)
    store.update(jid, pod=pod["id"], detail=f"pod {pod['id']} starting on {pod['gpu_id']}")
    t0, per_s = time.time(), pod["hourly"] / 3600
    keep = False
    try:
        wait_agent(pod, log=log, cancelled=lambda: store.cancel_requested(jid))
        files = {"gpu_train.py": (HERE / "gpu_train.py").read_text(),
                 "graders.py": (HERE.parent / "graders.py").read_text()}
        _agent(pod["url"], "POST", "/train", {"spec": spec_to_dict(spec), "rows": rows, "files": files,
                                              "job_id": jid}, timeout=95)
        store.update(jid, detail=f"training on {pod['gpu_id']}")
        n = 0
        while True:
            cost = round((time.time() - t0) * per_s, 4)
            if store.cancel_requested(jid):
                log("cancelled; deleting the pod")
                store.update(jid, state="cancelled", cost_usd=cost, detail="cancelled; no adapter saved")
                return
            try:
                got = _agent(pod["url"], "GET", f"/events?after={n}", timeout=30)
            except Exception as e:  # noqa: BLE001 - a proxy hiccup; the pod's status decides
                try:
                    code, info = _api("GET", f"/pods/{pod['id']}")
                except Exception:  # noqa: BLE001 - this machine is offline too: the pod trains on; wait
                    code, info = 0, {}
                if code == 404 or (info or {}).get("status") in ("TERMINATED", "ERROR", "EXITED"):
                    raise RunPodError(f"pod {pod['id']} stopped during training") from None
                log(f"no events ({str(e)[:80]}); retrying")
                time.sleep(10)
                continue
            n = got["next"]
            for ev in got["events"]:
                if ev["event"] == "step":
                    total = ev["total"] or 1
                    extra = {k: ev[k] for k in ("reward", "kl") if k in ev}
                    shown = (f" · reward {ev['reward']:.0%}" if "reward" in ev else "") + (
                        f" · kl {ev['kl']:.3f}" if "kl" in ev else "")
                    store.update(jid, progress=round(ev["step"] / total, 4), loss=ev["loss"], cost_usd=cost,
                                 detail=f"step {ev['step']}/{total} · loss {ev['loss']:.3f}{shown} · {pod['gpu']}",
                                 **extra)
                    log(f"step {ev['step']}/{total} loss {ev['loss']:.4f}"
                        + "".join(f" {k}={v}" for k, v in extra.items()))
                elif ev["event"] == "log":
                    log(ev["note"])
                    store.update(jid, detail=f"{ev['note']} · {pod['gpu']}", cost_usd=cost)
                elif ev["event"] == "error":
                    if ev.get("trace"):
                        log(ev["trace"])
                    raise RunPodError(ev["error"])
                elif ev["event"] == "done":
                    tar = _agent(pod["url"], "GET", f"/adapter?name={jid}", timeout=95, raw=True)
                    path = adapter_path(jid)
                    path.write_bytes(tar)
                    log(f"adapter downloaded: {path} ({len(tar) // 1024} KB)")
                    cost = round((time.time() - t0) * per_s, 4)
                    _remember(spec.base_model, pod, {jid: spec.lora_rank})
                    _agent(pod["url"], "POST", "/serve", {"base": spec.base_model,
                                                          "adapters": {jid: spec.lora_rank}})
                    keep = True
                    patch = {"rounds": ev["rounds"]} if ev.get("rounds") else {}
                    store.update(jid, state="succeeded", progress=1.0, checkpoint=f"runpod://{jid}", cost_usd=cost,
                                 adapter_rank=spec.lora_rank, **patch,
                                 detail=f"done in {round((time.time() - t0) / 60, 1)} min on {pod['gpu']} · "
                                        f"${cost:.2f}; the pod now serves it and stops itself after 15 idle minutes")
                    return
            time.sleep(5)
    finally:
        if not keep:
            delete_pod(pod["id"])


# ------------------------------------------------------------------ serving

_lock = threading.Lock()
_inflight = {"n": 0}


def _pods_path() -> Path:
    from ..server.service import data_home

    return Path(os.environ.get("OPENTRAIN_RUNPOD_STATE") or data_home() / "runpod.json")


def _pods() -> dict[str, Any]:
    try:
        return json.loads(_pods_path().read_text())
    except (OSError, ValueError):
        return {}


def _remember(base: str, pod: dict[str, Any], adapters: dict[str, int]) -> None:
    d = _pods()
    d[base] = {**pod, "adapters": adapters, "since": time.time()}
    p = _pods_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1))


def _alive(pod: dict[str, Any]) -> dict | None:
    try:
        return _agent(pod["url"], "GET", "/health", timeout=20)
    except Exception:  # noqa: BLE001
        return None


def endpoint(base: str, adapter: str | None = None, rank: int = 32) -> Endpoint:
    """The pod serving ``base`` (and ``adapter``), creating it or adding the adapter when needed."""
    from .modal_backend import serve_gpu

    serving_key()
    with _lock:
        cur = _pods().get(base)
        health = _alive(cur) if cur else None
        want = dict((cur or {}).get("adapters") or {}) if health else {}
        if adapter:
            want[adapter] = rank
        if health and set(want) <= set(health.get("serving") or []) and health.get("base") == base:
            return Endpoint(base_url=cur["url"] + "/v1", model=adapter or "base", api_key_env=SERVE_KEY_ENV)
        if not health:
            cur = create_pod(f"opentrain-serve-{base.split('/')[-1]}", serve_gpu(base))
            try:
                wait_agent(cur)
            except RunPodError:
                delete_pod(cur["id"])
                raise
        for name in want:
            if name not in ((health or {}).get("serving") or []):
                src = adapter_path(name)
                if not src.is_file():
                    raise RunPodError(f"the adapter for {name} is not on this machine ({src})")
                _agent(cur["url"], "POST", f"/adapter?name={name}", src.read_bytes(), timeout=95)
        _agent(cur["url"], "POST", "/serve", {"base": base, "adapters": want})
        _remember(base, cur, want)
        return Endpoint(base_url=cur["url"] + "/v1", model=adapter or "base", api_key_env=SERVE_KEY_ENV)


def hourly(base: str) -> float:
    return float((_pods().get(base) or {}).get("hourly") or 1.0)


@contextmanager
def answering(base: str):
    """Count answers in flight on a pod; yields a function giving this answer's share of its price."""
    with _lock:
        _inflight["n"] += 1
    t0 = time.time()
    peak = [_inflight["n"]]
    try:
        yield lambda: round((time.time() - t0) * hourly(base) / 3600 / max(1, peak[0]), 6)
    finally:
        with _lock:
            _inflight["n"] -= 1


_ready: dict[tuple[str, str], float] = {}


def wait_ready(ep: Endpoint, timeout: float = 1500) -> None:
    """Until vLLM on the pod lists the model (it restarts when adapters are added)."""
    if time.time() - _ready.get((ep.base_url, ep.model), 0) < 60:
        return
    t_end, last = time.time() + timeout, ""
    while time.time() < t_end:
        req = urllib.request.Request(f"{ep.base_url}/models", headers={"Authorization": f"Bearer {serving_key()}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
                names = [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
                if ep.model in names:
                    _ready[(ep.base_url, ep.model)] = time.time()
                    return
                last = f"{ep.model} not among {names}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:160]
        time.sleep(5)
    raise RunPodError(f"{ep.base_url} did not come up: {last}")


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:], run))
