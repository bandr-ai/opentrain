"""Serving: where each model version answers, one stable local endpoint in front of them, and the
config a coding agent needs to use it.

Three parts:

* ``endpoint_for(version)``: the real OpenAI-compatible endpoint behind a registry version. Tinker
  sampler checkpoints are served by Tinker itself (no deploy step); a Fireworks model needs an
  on-demand deployment; a local PEFT adapter needs a local server (``vllm serve --enable-lora``),
  which ``start_local_server`` launches when vLLM is installed and otherwise says what is missing.
  A version's ``serving`` dict (``base_url``, ``model``, ``api_key_env``) overrides all of this,
  for self-hosted endpoints.

* ``make_server(state)``: the ``opentrain serve`` proxy, a stdlib ``ThreadingHTTPServer`` on
  127.0.0.1 exposing ``/v1/models`` and ``/v1/chat/completions`` (streaming passed through as it
  arrives). Agents always ask for ``opentrain/current`` (or a pinned ``opentrain/v3``); the proxy
  maps that to the live version on every request, so ``opentrain model use v4`` or
  ``POST /v1/opentrain/switch`` changes the model with no restart and no dropped connection:
  requests in flight finish on the old version, new ones go to the new one. If the specialist
  fails (unreachable, 5xx, 429) and a frontier handoff is configured, the request is replayed
  there and marked as a handoff (chat completions only). Every request is logged to ``.opentrain/serving/requests.jsonl``
  with latency, so "share of the work your model did" can be computed later. The proxy holds the
  provider keys (from the environment); clients send a dummy key.

  ``/v1/messages`` (Anthropic Messages, for Claude Code) is forwarded only to versions whose
  provider speaks it natively: Tinker documents an Anthropic-compatible endpoint for sampler
  checkpoints. There is no OpenAI-to-Anthropic translation here; other versions get a 501 that
  says so.

* ``client_config(kind)``: ready-to-paste config for pi, opencode and Claude Code.

Docs relied on (read 2026-11-25):
  Tinker OpenAI-compatible API  https://tinker-docs.thinkingmachines.ai/inference/openai/overview/
  Tinker Anthropic-compatible   https://tinker-docs.thinkingmachines.ai/inference/anthropic/
  Tinker + opencode             https://tinker-docs.thinkingmachines.ai/inference/opencode/
  pi models.json                https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models-configuration.md
                                https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/configuration.md
  Claude Code gateways          https://code.claude.com/docs/en/llm-gateway
                                https://code.claude.com/docs/en/llm-gateway-connect.md
  Fireworks deployment          https://docs.fireworks.ai/fine-tuning/migrate-from-managed-training.md
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .plugs import Endpoint
from .registry import Registry, RegistryError, Version

DEFAULT_PORT = 8377
LOCAL_SERVER_PORT = 8400
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
# unverified: FIREWORKS_BASE is Fireworks' long-standing OpenAI-compatible URL; the pages read in
# this session describe deployments but do not restate the inference base URL.


class ServingError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- endpoints


def endpoint_for(version: Version) -> Endpoint:
    """The OpenAI-compatible endpoint that answers for this version. Raises ``ServingError`` with
    what to do when the version cannot be served as it stands."""
    s = version.serving or {}
    if s.get("base_url"):
        return Endpoint(base_url=s["base_url"].rstrip("/"), model=s.get("model") or version.checkpoint,
                        api_key_env=s.get("api_key_env"))
    ck = version.checkpoint
    if ck.startswith("fake://"):
        raise ServingError(f"{version.version} came from the fake backend: simulated, no weights to serve")
    if version.backend == "tinker" or ck.startswith("tinker://"):
        from .backends import tinker

        return Endpoint(base_url=tinker.OPENAI_BASE, model=ck, api_key_env=tinker.KEY_ENV)
    if version.backend == "modal" or ck.startswith("modal://"):
        from .backends import modal_backend

        try:
            return modal_backend.endpoint(version.base_model, ck.removeprefix("modal://"),
                                          int(s.get("lora_rank", 32)))
        except modal_backend.ModalError as exc:
            raise ServingError(str(exc)) from None
    if version.backend == "runpod" or ck.startswith("runpod://"):
        from .backends import runpod_backend

        try:
            return runpod_backend.endpoint(version.base_model, ck.removeprefix("runpod://"),
                                           int(s.get("lora_rank", 32)))
        except runpod_backend.RunPodError as exc:
            raise ServingError(str(exc)) from None
    if ck.startswith("fireworks-sampler://"):
        raise ServingError(f"{version.version} is a Fireworks training snapshot: OpenTrain evals answer from it, "
                           "but serving it to other apps needs a dedicated Fireworks deployment ($8/h and up, "
                           "billed while up); set serving.base_url once you create one")
    if version.backend == "fireworks":
        return Endpoint(base_url=FIREWORKS_BASE, model=ck, api_key_env="FIREWORKS_API_KEY")
    if version.backend == "local":
        port = int(s.get("port", LOCAL_SERVER_PORT))
        model = s.get("model") or f"opentrain-{version.version}"
        return Endpoint(base_url=f"http://127.0.0.1:{port}/v1", model=model, api_key_env=None)
    raise ServingError(f"don't know how to serve backend {version.backend!r}; set serving.base_url on {version.version}")


def anthropic_base_for(version: Version) -> str | None:
    """Base URL of a native Anthropic Messages endpoint for this version, if its provider has one."""
    s = version.serving or {}
    if s.get("anthropic_base_url"):
        return s["anthropic_base_url"].rstrip("/")
    if version.backend == "tinker" and version.checkpoint.startswith("tinker://"):
        from .backends import tinker

        return tinker.ANTHROPIC_BASE
    return None


def local_server_command(version: Version, port: int = LOCAL_SERVER_PORT) -> list[str]:
    """The command that serves a local adapter, or ``ServingError`` naming what to install."""
    s = version.serving or {}
    name = s.get("model") or f"opentrain-{version.version}"
    if shutil.which("vllm"):
        return ["vllm", "serve", version.base_model, "--port", str(port),
                "--enable-lora", "--lora-modules", f"{name}={version.checkpoint}",
                "--max-lora-rank", str(s.get("lora_rank", 64)),
                "--enable-auto-tool-choice", "--tool-call-parser", s.get("tool_parser", "hermes")]
    if shutil.which("mlx_lm.server"):
        # unverified: mlx_lm.server --adapter-path expects an MLX-format adapter; a PEFT adapter from
        # the local backend may need converting first. Logged loudly by start_local_server.
        return ["mlx_lm.server", "--model", version.base_model, "--adapter-path", version.checkpoint,
                "--port", str(port)]
    raise ServingError("no local inference server found. Install vLLM (Linux + CUDA: pip install vllm) to "
                       "serve LoRA adapters with --enable-lora, or mlx-lm on Apple silicon (pip install mlx-lm; "
                       "PEFT adapters may need converting to MLX format)")


def start_local_server(version: Version, state: Path, port: int = LOCAL_SERVER_PORT) -> tuple[Endpoint, int]:
    """Launch the local server for a version, detached, logging to ``.opentrain/serving/``."""
    if version.backend != "local":
        raise ServingError(f"{version.version} is a {version.backend} version; nothing to start locally")
    cmd = local_server_command(version, port)
    folder = Path(state) / "serving"
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / f"local-{version.version}.log").open("a")
    if cmd[0] == "mlx_lm.server":
        log.write(f"{_now()} WARNING: mlx_lm.server may not load a PEFT adapter directly (unverified)\n")
        log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    log.close()
    model = (version.serving or {}).get("model") or f"opentrain-{version.version}"
    if cmd[0] == "mlx_lm.server":
        model = version.base_model
    Registry(Path(state)).update(version.version, serving={**(version.serving or {}), "port": port,
                                                           "model": model, "pid": proc.pid})
    return Endpoint(base_url=f"http://127.0.0.1:{port}/v1", model=model), proc.pid


# ---------------------------------------------------------------- which version is live


def _live_path(state: Path) -> Path:
    return Path(state) / "serving" / "live.json"


def live_version(state: Path) -> str | None:
    """What ``opentrain/current`` means right now: the serving pin if one is set, else the registry's
    current (promoted) version. The pin lets you switch what agents use without touching promotion,
    which only the gate may change."""
    p = _live_path(state)
    if p.exists():
        v = json.loads(p.read_text()).get("version")
        if v:
            return v
    cur = Registry(Path(state)).current()
    return cur.version if cur else None


def use_version(state: Path, version: str | None, why: str = "manual") -> None:
    """Pin ``opentrain/current`` to a version (None: follow the registry's current again)."""
    if version is not None:
        Registry(Path(state)).get(version)
    p = _live_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"version": version, "at": _now(), "why": why}))
    tmp.replace(p)


# ---------------------------------------------------------------- the proxy


@dataclass(frozen=True)
class Handoff:
    """A frontier OpenAI-compatible endpoint to replay a request on when the specialist fails."""

    base_url: str
    model: str
    api_key_env: str | None = None


class UpstreamUnreachable(Exception):
    pass


def _auth_headers(api_key_env: str | None, anthropic: bool = False) -> dict[str, str]:
    key = os.environ.get(api_key_env) if api_key_env else None
    if not key:
        return {}
    return {"x-api-key": key} if anthropic else {"Authorization": f"Bearer {key}"}


def _post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e
    except (urllib.error.URLError, OSError) as e:
        raise UpstreamUnreachable(str(getattr(e, "reason", e))) from e


def _rename_model(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        if "model" in obj:
            obj["model"] = name
        if isinstance(obj.get("message"), dict) and "model" in obj["message"]:
            obj["message"]["model"] = name
    return obj


class Proxy:
    """Routing, handoff and the request log, shared by every handler thread."""

    def __init__(self, state: Path, handoff: Handoff | None = None, timeout: float = 600.0):
        self.state = Path(state)
        self.handoff = handoff
        self.timeout = timeout
        self.log_path = self.state / "serving" / "requests.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def resolve(self, name: str) -> Version:
        reg = Registry(self.state)
        if name in ("opentrain", "opentrain/current"):
            live = live_version(self.state)
            if not live:
                raise LookupError("no version is live yet; promote one or run opentrain model use <version>")
            return reg.get(live)
        if name.startswith("opentrain/"):
            try:
                return reg.get(name.split("/", 1)[1])
            except RegistryError as e:
                raise LookupError(str(e)) from None
        raise LookupError(f"unknown model {name!r}; ask for opentrain/current or opentrain/<version>")

    def models(self) -> list[dict[str, Any]]:
        out = [{"id": "opentrain/current", "object": "model", "owned_by": "opentrain",
                "version": live_version(self.state)}]
        for v in Registry(self.state).versions():
            out.append({"id": v.model_name, "object": "model", "owned_by": "opentrain",
                        "backend": v.backend, "decision": v.decision})
        return out

    def record(self, entry: dict[str, Any]) -> None:
        line = json.dumps({"at": _now(), **entry}, default=str)
        with self._lock, self.log_path.open("a") as fh:
            fh.write(line + "\n")


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


class _Handler(BaseHTTPRequestHandler):
    server_version = "opentrain-serve"
    proxy: Proxy  # set on the subclass by make_server
    token: str = ""  # set on the subclass by make_server; required to switch versions over HTTP

    def _local_only(self) -> bool:
        """Refuse requests whose Host is not this machine: a web page using DNS rebinding would
        otherwise reach the proxy (and the provider keys it adds) through the browser."""
        header = self.headers.get("Host") or ""
        host = header.split("]")[0] + "]" if header.startswith("[") else header.rsplit(":", 1)[0]
        if self.server.server_address[0] not in ("127.0.0.1", "::1", "localhost"):
            return True  # bound to a public interface on purpose: the operator owns that choice
        if host in _LOCAL_HOSTS:
            return True
        self._error(403, "requests must address this machine (Host: 127.0.0.1 or localhost)")
        return False

    def log_message(self, fmt, *args):  # the request log is requests.jsonl, not stderr
        pass

    def _json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._json(status, {"error": {"message": message, "type": kind}})

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if not self._local_only():
            return
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": self.proxy.models()})
        if self.path.rstrip("/") == "/v1/opentrain/live":
            return self._json(200, {"version": live_version(self.proxy.state)})
        self._error(404, f"no route {self.path}")

    def do_POST(self):
        if not self._local_only():
            return
        path = self.path.split("?")[0].rstrip("/")
        # JSON only: a browser can send text/plain across sites without a preflight, but not JSON.
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._error(415, "send Content-Type: application/json")
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._error(400, "body is not JSON")
        if path == "/v1/opentrain/switch":
            import hmac

            if not hmac.compare_digest(self.headers.get("X-OpenTrain-Token") or "", self.token):
                return self._error(403, "switching versions needs the X-OpenTrain-Token from "
                                        ".opentrain/serving/token (or use `opentrain model use`)")
            try:
                use_version(self.proxy.state, body.get("version"), why=body.get("why", "switch via serve"))
            except RegistryError as e:
                return self._error(404, str(e))
            return self._json(200, {"version": live_version(self.proxy.state)})
        if path == "/v1/chat/completions":
            return self._forward(body, "/chat/completions", anthropic=False)
        if path in ("/v1/messages", "/v1/messages/count_tokens"):
            return self._forward(body, path, anthropic=True)
        self._error(404, f"no route {self.path}")

    def _forward(self, body: dict[str, Any], suffix: str, anthropic: bool) -> None:
        asked = body.get("model", "")
        t0 = time.monotonic()
        entry: dict[str, Any] = {"model": asked, "stream": bool(body.get("stream")), "api": suffix,
                                 "handoff": False}
        try:
            version = self.proxy.resolve(asked)
        except LookupError as e:
            self.proxy.record({**entry, "status": 404, "error": str(e), "latency_s": 0.0})
            return self._error(404, str(e), "model_not_found")
        entry.update(version=version.version, backend=version.backend)

        try:
            if anthropic:
                base = anthropic_base_for(version)
                if not base:
                    raise ServingError(
                        f"{version.version} ({version.backend}) has no Anthropic Messages endpoint; only Tinker "
                        "checkpoints do. Use an OpenAI-compatible agent (pi, opencode) for this version.")
                url = f"{base}{suffix}"
                upstream_model = version.checkpoint if version.backend == "tinker" else \
                    (version.serving or {}).get("model", version.checkpoint)
                key_env = "TINKER_API_KEY" if version.backend == "tinker" else (version.serving or {}).get("api_key_env")
                headers = {"anthropic-version": self.headers.get("anthropic-version", "2023-06-01"),
                           **_auth_headers(key_env, anthropic=True)}
                if self.headers.get("anthropic-beta"):
                    headers["anthropic-beta"] = self.headers["anthropic-beta"]
            else:
                ep = endpoint_for(version)
                url, upstream_model = f"{ep.base_url}{suffix}", ep.model
                headers = _auth_headers(ep.api_key_env)
        except ServingError as e:
            self.proxy.record({**entry, "status": 501, "error": str(e), "latency_s": 0.0})
            return self._error(501, str(e), "not_supported")
        entry["upstream"] = url

        resp, failure = None, None
        try:
            resp = _post(url, {**body, "model": upstream_model}, headers, self.proxy.timeout)
            if resp.status >= 500 or resp.status == 429:
                failure = f"upstream {resp.status}"
        except UpstreamUnreachable as e:
            failure = f"unreachable: {e}"

        rename = asked
        ho = self.proxy.handoff
        if failure and ho and not anthropic:
            entry.update(handoff=True, specialist_error=failure)
            if resp is not None:
                resp.close()
            try:
                resp = _post(f"{ho.base_url.rstrip('/')}{suffix}", {**body, "model": ho.model},
                             _auth_headers(ho.api_key_env), self.proxy.timeout)
                rename, entry["upstream"] = None, ho.base_url
            except UpstreamUnreachable as e:
                resp, failure = None, f"specialist {failure}; frontier unreachable: {e}"
        if resp is None:
            self.proxy.record({**entry, "status": 502, "error": failure,
                               "latency_s": round(time.monotonic() - t0, 4)})
            return self._error(502, f"{version.version}: {failure}", "upstream_error")

        extra = {"X-OpenTrain-Version": version.version}
        if entry["handoff"]:
            extra["X-OpenTrain-Handoff"] = "frontier"
        ctype = resp.headers.get("Content-Type", "application/json")
        with resp:
            if "text/event-stream" in ctype:
                ttfb, usage = self._stream(resp, ctype, extra, rename, t0)
                entry.update(ttfb_s=ttfb)
            else:
                data = resp.read()
                usage = None
                try:
                    obj = json.loads(data)
                    usage = obj.get("usage") if isinstance(obj, dict) else None
                    if rename:
                        data = json.dumps(_rename_model(obj, rename)).encode()
                except json.JSONDecodeError:
                    pass
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)
        self.proxy.record({**entry, "status": resp.status, "usage": usage,
                           "latency_s": round(time.monotonic() - t0, 4)})

    def _stream(self, resp, ctype: str, extra: dict[str, str], rename: str | None, t0: float):
        """Pass server-sent events through line by line as they arrive, renaming the model."""
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        ttfb, usage = None, None
        for raw in resp:
            line = raw
            if raw.startswith(b"data:") and raw[5:].strip() not in (b"", b"[DONE]"):
                try:
                    obj = json.loads(raw[5:])
                    if isinstance(obj, dict) and obj.get("usage"):
                        usage = obj["usage"]
                    if rename:
                        line = b"data: " + json.dumps(_rename_model(obj, rename)).encode() + b"\n"
                except json.JSONDecodeError:
                    pass
            if ttfb is None:
                ttfb = round(time.monotonic() - t0, 4)
            self.wfile.write(line)
            self.wfile.flush()
        return ttfb, usage


def make_server(state: Path, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                handoff: Handoff | None = None) -> ThreadingHTTPServer:
    """The ``opentrain serve`` server, not yet started (call ``serve_forever``). Port 0 picks a free one."""
    import secrets

    proxy = Proxy(Path(state), handoff)
    token = secrets.token_urlsafe(24)
    tok = Path(state) / "serving" / "token"
    tok.parent.mkdir(parents=True, exist_ok=True)
    tok.write_text(token)
    tok.chmod(0o600)
    handler = type("OpenTrainHandler", (_Handler,), {"proxy": proxy, "token": token})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


# ---------------------------------------------------------------- client config


def client_config(kind: str, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                  model: str = "opentrain/current") -> str:
    """Config to paste into a coding agent so it uses ``opentrain serve``. ``kind``: pi, opencode, claude."""
    base = f"http://{host}:{port}"
    if kind == "pi":
        cfg = {"providers": {"opentrain": {
            "baseUrl": f"{base}/v1", "api": "openai-completions", "apiKey": "opentrain-local",
            "models": [{"id": model, "name": f"OpenTrain ({model})", "contextWindow": 32768, "maxTokens": 8192,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}
        return ("# ~/.pi/agent/models.json (or $PI_CODING_AGENT_DIR/models.json); merge into existing providers.\n"
                "# The key is a placeholder: opentrain serve holds the real provider keys.\n"
                f"{json.dumps(cfg, indent=2)}\n"
                f"# Then run `opentrain serve` and, in pi, /model -> {model}")
    if kind == "opencode":
        cfg = {"$schema": "https://opencode.ai/config.json", "provider": {"opentrain": {
            "npm": "@ai-sdk/openai-compatible", "name": "OpenTrain",
            "options": {"baseURL": f"{base}/v1", "apiKey": "opentrain-local"},
            "models": {model: {"name": f"OpenTrain ({model})", "tool_call": True, "attachment": False,
                               "limit": {"context": 32768, "output": 8192}}}}}}
        return ("# opencode.json in your project root (or ~/.config/opencode/opencode.json)\n"
                f"{json.dumps(cfg, indent=2)}\n"
                f"# Then run `opentrain serve` and pick opentrain/{model} in opencode's model picker")
    if kind == "claude":
        return (
            "# Claude Code speaks only the Anthropic Messages API, so this works only while the live\n"
            "# version is a Tinker checkpoint: opentrain serve forwards /v1/messages to Tinker's\n"
            "# Anthropic-compatible endpoint. Local and Fireworks versions get a clear 501 instead.\n"
            "# Anthropic does not support routing Claude Code to non-Claude models; expect gaps\n"
            "# (no prompt caching, thinking signatures empty, tool-use quality is your model's).\n"
            f"export ANTHROPIC_BASE_URL={base}\n"
            "export ANTHROPIC_AUTH_TOKEN=opentrain-local   # placeholder; the proxy holds TINKER_API_KEY\n"
            f"export ANTHROPIC_MODEL={model}\n"
            f"export ANTHROPIC_DEFAULT_HAIKU_MODEL={model}   # background tasks use the same model\n"
            "export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1  # upstream rejects unknown beta fields\n"
            "# Then run `opentrain serve` and start `claude` from this shell."
        )
    raise ValueError(f"unknown client {kind!r}; choose pi, opencode or claude")
