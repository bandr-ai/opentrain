"""The agent OpenTrain runs inside a RunPod pod: train with ``gpu_train``, then serve with vLLM.

Standard library only; ``runpod_backend`` ships it in the pod's environment (base64) and the pod's
start command writes it to disk and runs it on the image ``vllm/vllm-openai`` (torch, vLLM and
transformers already installed; peft is added at boot). It listens on port 8000, which RunPod
exposes at ``https://<pod id>-8000.proxy.runpod.net``. Every request needs
``Authorization: Bearer $OT_TOKEN``: the proxy URL is public.

* ``GET  /health``                    ``{"ready": bool, "training": bool, "serving": [names]}``
* ``POST /train`` ``{spec, rows, files}``   writes ``files`` (gpu_train.py, graders.py) and starts
  ``python gpu_train.py job.json /work/adapters/<job>``; events go to ``GET /events?after=N``.
* ``GET  /adapter?name=X``            the adapter as a .tar.gz (OpenTrain keeps a copy)
* ``POST /adapter?name=X``            upload an adapter (.tar.gz) to serve
* ``POST /serve`` ``{base, adapters: {name: rank}}``  (re)start vLLM on 127.0.0.1:8001
* ``/v1/...``                         proxied to vLLM (requests that take over 100 s hit RunPod's
  proxy limit; answers are short)

It deletes its own pod after ``OT_IDLE`` seconds without a request while not training (default 15
minutes), and after ``OT_MAX_HOURS`` in any case, so a pod cannot bill on after OpenTrain is gone.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

WORK = os.environ.get("OT_WORK", "/work")
TOKEN = os.environ.get("OT_TOKEN", "")
IDLE = int(os.environ.get("OT_IDLE", "900"))
MAX_HOURS = float(os.environ.get("OT_MAX_HOURS", "12"))
STARTED = time.time()

state = {"ready": False, "boot_error": None, "training": False, "events": [], "serving": {}, "base": None,
         "last": time.time(), "vllm": None}
lock = threading.Lock()


def boot() -> None:
    """Add what the image lacks (peft; the transformers OpenTrain trains with)."""
    pkgs = os.environ.get("OT_PIP", "peft transformers==5.8.1").split()
    if not pkgs:
        state["ready"] = True
        return
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    if p.returncode != 0:
        state["boot_error"] = (p.stderr or p.stdout)[-1500:]
    state["ready"] = True


def train(body: dict) -> None:
    job = body["spec"].get("name") or "job"
    for name, src in (body.get("files") or {}).items():
        with open(os.path.join(WORK, os.path.basename(name)), "w") as f:
            f.write(src)
    path = os.path.join(WORK, "job.json")
    with open(path, "w") as f:
        json.dump({"spec": body["spec"], "rows": body["rows"]}, f)
    out = os.path.join(WORK, "adapters", body.get("job_id") or job)
    proc = subprocess.Popen([sys.executable, os.path.join(WORK, "gpu_train.py"), path, out], cwd=WORK,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        try:
            ev = json.loads(line)
            if not isinstance(ev, dict) or "event" not in ev:
                raise ValueError
        except ValueError:
            tail = (tail + [line])[-40:]
            continue
        if ev["event"] == "error":
            ev["trace"] = "\n".join(tail)
        with lock:
            state["events"].append(ev)
    code = proc.wait()
    with lock:
        if not state["events"] or state["events"][-1]["event"] not in ("done", "error"):
            state["events"].append({"event": "error", "error": f"trainer exited with {code}",
                                    "trace": "\n".join(tail)})
        state["training"] = False
        state["last"] = time.time()


def serve(base: str, adapters: dict[str, int]) -> None:
    old = state.get("vllm")
    if old and old.poll() is None:
        old.terminate()
        try:
            old.wait(60)
        except subprocess.TimeoutExpired:
            old.kill()
    cmd = ["vllm", "serve", base, "--served-model-name", "base", "--host", "127.0.0.1", "--port", "8001",
           "--max-model-len", os.environ.get("OT_SERVE_MAX_LEN", "8192"), "--enforce-eager",
           "--generation-config", "vllm"]
    if adapters:
        rank = next(r for r in (8, 16, 32, 64, 128, 256, 512) if r >= max(adapters.values()))
        cmd += ["--enable-lora", "--max-loras", str(min(8, len(adapters))), "--max-lora-rank", str(rank),
                "--lora-modules", *[f"{n}={WORK}/adapters/{n}" for n in adapters]]
    log = open(os.path.join(WORK, "vllm.log"), "a")
    state["vllm"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    state["base"], state["serving"] = base, dict(adapters)


def vllm_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=3) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError):
        return False


def suicide(reason: str) -> None:
    """Delete this pod through RunPod's API (the pod's own key, else the one OpenTrain passed)."""
    pod = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("OT_RUNPOD_KEY") or os.environ.get("RUNPOD_API_KEY")
    print(f"opentrain agent: stopping the pod ({reason})", flush=True)
    if pod and key:
        req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}", method="DELETE",
                                     headers={"Authorization": f"Bearer {key}"})
        try:
            urllib.request.urlopen(req, timeout=30)  # noqa: S310 - fixed https host
        except (OSError, urllib.error.URLError) as e:
            print(f"opentrain agent: could not delete the pod: {e}", flush=True)
    os._exit(0)


def watchdog() -> None:
    while True:
        time.sleep(30)
        now = time.time()
        if now - STARTED > MAX_HOURS * 3600:
            suicide(f"ran {MAX_HOURS} hours")
        if not state["training"] and now - state["last"] > IDLE:
            suicide(f"idle {IDLE} s")


def tar_of(path: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name in sorted(os.listdir(path)):
            t.add(os.path.join(path, name), arcname=name)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode())

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _ok(self) -> bool:
        if not TOKEN or self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._json(401, {"error": "unauthorized"})
            return False
        state["last"] = time.time()
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._ok():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/health":
            return self._json(200, {"ready": state["ready"], "boot_error": state["boot_error"],
                                    "training": state["training"], "serving": list(state["serving"]),
                                    "base": state["base"], "vllm": vllm_up(),
                                    "gpu": os.environ.get("RUNPOD_GPU_NAME") or ""})
        if u.path == "/events":
            after = int(q.get("after", 0))
            with lock:
                evs = state["events"][after:]
            return self._json(200, {"events": evs, "next": after + len(evs)})
        if u.path == "/adapter":
            path = os.path.join(WORK, "adapters", os.path.basename(q.get("name", "")))
            if not os.path.isdir(path):
                return self._json(404, {"error": "no such adapter"})
            return self._send(200, tar_of(path), "application/gzip")
        if u.path.startswith("/v1/"):
            return self._proxy(None)
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._ok():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/train":
            if state["training"]:
                return self._json(409, {"error": "already training"})
            if not state["ready"]:
                return self._json(503, {"error": "still installing"})
            body = json.loads(self._body())
            state["training"], state["events"] = True, []
            threading.Thread(target=train, args=(body,), daemon=True).start()
            return self._json(200, {"started": True})
        if u.path == "/adapter":
            name = os.path.basename(q.get("name", ""))
            dest = os.path.join(WORK, "adapters", name)
            os.makedirs(dest, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(self._body()), mode="r:gz") as t:
                t.extractall(dest, filter="data")
            return self._json(200, {"saved": name})
        if u.path == "/serve":
            body = json.loads(self._body())
            serve(body["base"], body.get("adapters") or {})
            return self._json(200, {"starting": True})
        if u.path.startswith("/v1/"):
            return self._proxy(self._body())
        self._json(404, {"error": "not found"})

    def _proxy(self, body: bytes | None) -> None:
        req = urllib.request.Request("http://127.0.0.1:8001" + self.path, data=body,
                                     method="POST" if body is not None else "GET",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=95) as r:
                self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read())
        except OSError as e:
            self._json(503, {"error": f"model server not ready: {e}"})
        state["last"] = time.time()


def main() -> None:
    os.makedirs(os.path.join(WORK, "adapters"), exist_ok=True)
    threading.Thread(target=boot, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("OT_PORT", "8000"))), Handler).serve_forever()


if __name__ == "__main__":
    main()
