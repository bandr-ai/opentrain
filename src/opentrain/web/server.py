"""The ``opentrain web`` server: stdlib only, local only.

    GET  /                          the single-page app (with this process's token baked in)
    GET  /app.js, /app.css          its assets
    GET  /api/home | runs | versions | race | data | skills | policy
    GET  /api/runs/<id>             one run with its full event timeline
    GET  /api/runs/<id>/events      Server-Sent Events: new events as the run appends them
    GET  /api/evals?a=v1&b=v2       held-out task ids that flipped between two versions
    GET  /api/tasks/<id>            one TRAINING task (held-out tasks are never served)
    POST /api/versions/revert       Registry.revert: put back the version live before this one
    POST /api/policy                validate a Trainfile with the real parser, then save it

Binds 127.0.0.1. Every request must carry a Host header naming this machine (blocks DNS
rebinding); every POST must carry ``X-OpenTrain-Token`` with the random token handed to the page,
and a JSON body (blocks CSRF from other local sites, which cannot read the page to learn it).
"""

from __future__ import annotations

import hmac
import json
import secrets
import sys
import threading
import time
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .. import __version__
from ..registry import RegistryError
from ..trainfile import TrainfileError
from . import api

STATIC = Path(__file__).with_name("static")
ASSETS = {"/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8")}
TOKEN_MARK = "__OPENTRAIN_TOKEN__"
MAX_BODY = 256_000
SSE_POLL_S = 0.5
SSE_HEARTBEAT_S = 15.0
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    ),
}


def _asset(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = f"opentrain/{__version__}"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, state: Path, token: str, quiet: bool = True, **kw):
        self.state, self.token, self.quiet = state, token, quiet
        super().__init__(*args, **kw)

    # ------------------------------------------------------------ plumbing

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        if not self.quiet:
            super().log_message(fmt, *args)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if not host:
            return False
        name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        bound = str(self.server.server_address[0]).lower()
        return name in LOCAL_HOSTS or name == bound

    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in {**SECURITY_HEADERS, **(extra or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        self._send(status, json.dumps(data, default=str).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    # ------------------------------------------------------------ GET

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
        url = urlsplit(self.path)
        path, q = url.path, {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if path in ("/", "/index.html"):
                html = _asset("index.html").replace(TOKEN_MARK, self.token)
                return self._send(200, html.encode(), "text/html; charset=utf-8")
            if path in ASSETS:
                name, ctype = ASSETS[path]
                return self._send(200, _asset(name).encode(), ctype)
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not path.startswith("/api/"):
                return self._error(404, "not found")
            parts = [unquote(p) for p in path[len("/api/"):].strip("/").split("/")]
            return self._api_get(parts, q)
        except api.NotFound as e:
            return self._error(404, str(e) or "not found")
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as e:  # a view must never take the server down
            return self._error(500, f"{type(e).__name__}: {e}")

    def _api_get(self, parts: list[str], q: dict[str, str]) -> None:
        s = self.state
        simple = {"home": api.home, "runs": api.runs, "versions": api.versions, "race": api.race,
                  "data": api.data, "skills": api.skills, "policy": api.policy}
        if len(parts) == 1 and parts[0] in simple:
            return self._json(simple[parts[0]](s))
        if parts == ["evals"]:
            return self._json(api.evals(s, q.get("a") or None, q.get("b") or None))
        if len(parts) == 2 and parts[0] == "runs":
            return self._json(api.run(s, parts[1]))
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "events":
            return self._sse(parts[1], q)
        if len(parts) == 2 and parts[0] == "tasks":
            return self._json(api.task(s, parts[1]))
        return self._error(404, "no such endpoint")

    def _sse(self, run_id: str, q: dict[str, str]) -> None:
        api.get_run(self.state, run_id)  # 404 before we commit to a stream
        last = self.headers.get("Last-Event-ID") or q.get("after") or "-1"
        try:
            after = int(last)
        except ValueError:
            after = -1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True
        path = api.get_run(self.state, run_id).events_path
        size = -1
        beat = time.monotonic()
        stop = getattr(self.server, "stopping", None)
        try:
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            while not (stop and stop.is_set()):
                cur = path.stat().st_size if path.exists() else 0
                if cur != size:
                    size = cur
                    new, closed = api.events_since(self.state, run_id, after)
                    for e in new:
                        after = e["seq"]
                        self.wfile.write(f"id: {after}\nevent: run\ndata: {json.dumps(e, default=str)}\n\n".encode())
                    if closed:
                        self.wfile.write(b"event: end\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    if new:
                        self.wfile.flush()
                        beat = time.monotonic()
                if time.monotonic() - beat > SSE_HEARTBEAT_S:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    beat = time.monotonic()
                time.sleep(SSE_POLL_S)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    # ------------------------------------------------------------ POST

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
        given = self.headers.get("X-OpenTrain-Token") or ""
        if not hmac.compare_digest(given.encode(), self.token.encode()):
            return self._error(HTTPStatus.FORBIDDEN, "missing or wrong X-OpenTrain-Token")
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).hostname not in LOCAL_HOSTS | {str(self.server.server_address[0])}:
            return self._error(HTTPStatus.FORBIDDEN, "cross-origin request refused")
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send application/json")
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._error(400, "body is not JSON")
        if not isinstance(body, dict):
            return self._error(400, "body must be a JSON object")
        path = urlsplit(self.path).path.rstrip("/")
        try:
            if path == "/api/versions/revert":
                why = str(body.get("why") or "rolled back from the web UI")[:500]
                return self._json(api.revert(self.state, why))
            if path == "/api/policy":
                return self._json(api.save_policy(self.state, body.get("text")))
            return self._error(404, "no such endpoint")
        except TrainfileError as e:
            return self._json({"error": str(e), "invalid": True}, 400)
        except RegistryError as e:
            return self._error(409, str(e))
        except Exception as e:
            return self._error(500, f"{type(e).__name__}: {e}")


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        # A browser closing a tab mid-stream (SSE) is normal, not an error worth a traceback.
        import sys

        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)

    def __init__(self, addr: tuple[str, int], state: Path, token: str | None = None, quiet: bool = True):
        self.token = token or secrets.token_urlsafe(24)
        self.stopping = threading.Event()
        super().__init__(addr, partial(Handler, state=Path(state), token=self.token, quiet=quiet))

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{'127.0.0.1' if host in ('', '0.0.0.0') else host}:{port}/"

    def shutdown(self) -> None:
        self.stopping.set()
        super().shutdown()


def make_server(state: Path, port: int = 0, host: str = "127.0.0.1", quiet: bool = True) -> WebServer:
    """Bind ``port`` (0: any free port; if taken, the next free one of the ten after it)."""
    last: OSError | None = None
    candidates = [port] if port == 0 else list(range(port, port + 10))
    for p in candidates:
        try:
            return WebServer((host, p), Path(state), quiet=quiet)
        except OSError as e:
            last = e
    raise OSError(f"no free port from {port}: {last}")


def serve_web(state: Path, port: int = 7878, open_browser: bool = True, host: str = "127.0.0.1") -> int:
    state = Path(state)
    srv = make_server(state, port, host)
    if not state.is_dir():
        print(f"  (no {state} yet; screens will tell you what to run first)", file=sys.stderr)
    print(f"OpenTrain   {srv.url}")
    print(f"State       {state}")
    print("            Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(srv.url)).start()
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print()
    finally:
        srv.stopping.set()
        srv.server_close()
    return 0
