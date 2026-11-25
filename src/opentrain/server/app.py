"""The OpenTrain server: one process per user, every client talks to it.

HTTP under ``/api``, one SSE stream at ``/api/event`` carrying every event of every project, HTTP
Basic auth with the service password (``opentrain:<password>``), and one-time pairing links that
give a browser a signed cookie instead. Requests name their project with ``?directory=`` (or the
``x-opentrain-directory`` header); the server keeps one :class:`Project` per directory.

Route handlers stay thin: each calls a service on the project. Services are attached by
``attachers`` when a project is first opened, so the engine, permissions, jobs and so on each
live in their own module.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import web

from .. import __version__
from ..core.project import Project, default_directory

Attacher = Callable[["Hub", Project], None]
QUEUE_SIZE = 4096
HEARTBEAT = 15.0
PAIR_TTL = 300.0
COOKIE = "opentrain_session"
COOKIE_TTL = 30 * 24 * 3600
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request"):
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind


class Hub:
    """Projects by directory, and the fan-out of their events to SSE subscribers."""

    def __init__(self, default: Path | None = None):
        self.default = Path(default or default_directory())
        self.projects: dict[str, Project] = {}
        self.attachers: list[Attacher] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.subscribers: set[asyncio.Queue] = set()

    def project(self, directory: str | Path | None = None) -> Project:
        key = str(Path(directory or self.default).expanduser().resolve())
        if key not in self.projects:
            if not Path(key).is_dir():
                raise ApiError(404, f"no such directory: {key}", "not_found")
            p = Project(key)
            p.store.subscribe(self.broadcast)
            self.projects[key] = p
            for attach in self.attachers:
                attach(self, p)
        return self.projects[key]

    def broadcast(self, event: dict[str, Any]) -> None:
        if self.loop is None or self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(self._fanout, event)
        except RuntimeError:  # loop shutting down
            pass

    def _fanout(self, event: dict[str, Any]) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # a stalled client is dropped, never the server
                self.subscribers.discard(q)  # its stream notices at the next heartbeat and ends

    def close(self) -> None:
        for p in self.projects.values():
            for svc in p.services.values():
                if hasattr(svc, "close"):
                    try:
                        svc.close()
                    except Exception:
                        pass
            p.store.close()


# ---------------------------------------------------------------------- auth


class Auth:
    def __init__(self, password: str):
        self.password = password
        self.key = hashlib.sha256(b"opentrain-cookie:" + password.encode()).digest()
        self.codes: dict[str, float] = {}

    def basic_ok(self, header: str | None) -> bool:
        if not header or not header.startswith("Basic "):
            return False
        try:
            user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
        except ValueError:
            return False
        return user == "opentrain" and hmac.compare_digest(pw, self.password)

    def token(self) -> str:
        exp = str(int(time.time() + COOKIE_TTL))
        sig = hmac.new(self.key, exp.encode(), hashlib.sha256).hexdigest()
        return f"{exp}.{sig}"

    def token_ok(self, token: str | None) -> bool:
        if not token or "." not in token:
            return False
        exp, sig = token.split(".", 1)
        good = hmac.new(self.key, exp.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time()

    def pair(self) -> str:
        now = time.time()
        self.codes = {c: t for c, t in self.codes.items() if t > now}
        code = secrets.token_urlsafe(18)
        self.codes[code] = now + PAIR_TTL
        return code

    def redeem(self, code: str) -> bool:
        exp = self.codes.pop(code, None)
        return bool(exp and exp > time.time())


# ---------------------------------------------------------------------- app


def _directory(request: web.Request) -> str | None:
    return request.query.get("directory") or request.headers.get("x-opentrain-directory")


def project_of(request: web.Request) -> Project:
    return request.app[HUB].project(_directory(request))


async def body_of(request: web.Request) -> dict[str, Any]:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise ApiError(400, "body must be JSON") from None
    if not isinstance(data, dict):
        raise ApiError(400, "body must be a JSON object")
    return data


def ok(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=lambda o: json.dumps(o, default=str))


@web.middleware
async def guard(request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]):
    app = request.app
    host = (request.host or "").rsplit(":", 1)[0] if not (request.host or "").startswith("[") else request.host.split("]")[0] + "]"
    if app[LOCAL_ONLY] and host not in LOCAL_HOSTS:  # DNS rebinding: a page on evil.com resolving to us
        return ok({"error": {"type": "forbidden", "message": f"host {request.host!r} not allowed"}}, 403)
    open_paths = request.path.startswith("/auth/connect/") or request.path in ("/", "/favicon.ico") \
        or request.path.startswith("/static/")
    auth: Auth = app[AUTH]
    if not open_paths and not (auth.basic_ok(request.headers.get("Authorization"))
                               or auth.token_ok(request.cookies.get(COOKIE))):
        return ok({"error": {"type": "unauthorized", "message": "missing or wrong credentials"}}, 401)
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.can_read_body \
            and not request.content_type.startswith("application/json"):
        return ok({"error": {"type": "invalid_request", "message": "POST bodies must be application/json"}}, 415)
    try:
        return await handler(request)
    except ApiError as exc:
        return ok({"error": {"type": exc.kind, "message": exc.message}}, exc.status)
    except KeyError as exc:
        return ok({"error": {"type": "not_found", "message": str(exc).strip("'\"")}}, 404)
    except web.HTTPException:
        raise
    except Exception as exc:  # never a stack trace to the client; the log has it
        app.logger.exception("handler failed")
        return ok({"error": {"type": "server_error", "message": f"{type(exc).__name__}: {exc}"}}, 500)


HUB = web.AppKey("hub", Hub)
AUTH = web.AppKey("auth", Auth)
LOCAL_ONLY = web.AppKey("local_only", bool)
PID = web.AppKey("pid", int)
DEFAULT_AGENT = web.AppKey("default_agent", object)

routes = web.RouteTableDef()


@routes.get("/api/health")
async def health(request: web.Request):
    return ok({"ok": True, "version": __version__, "pid": request.app[PID],
               "projects": sorted(request.app[HUB].projects)})


@routes.get("/api/project")
async def project_info(request: web.Request):
    return ok(project_of(request).info())


@routes.get("/api/event")
async def events(request: web.Request):
    hub: Hub = request.app[HUB]
    only = set(filter(None, request.query.get("types", "").split(",")))
    directory = _directory(request)
    location = str(Path(directory).expanduser().resolve()) if directory else None
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
    await resp.prepare(request)
    q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    hub.subscribers.add(q)

    async def send(ev: dict[str, Any]) -> None:
        await resp.write(f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n".encode())

    try:
        await send({"type": "server.connected", "data": {"version": __version__}, "time": time.time()})
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), HEARTBEAT)
            except asyncio.TimeoutError:
                if q not in hub.subscribers:
                    break
                await resp.write(b": heartbeat\n\n")
                continue
            if ev is None:  # server shutting down
                break
            if (only and ev["type"] not in only) or (location and ev.get("location") not in (None, location)):
                continue
            await send(ev)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        hub.subscribers.discard(q)
    return resp


@routes.get("/api/event/log")
async def event_log(request: web.Request):
    """Durable events after ``after`` (a seq): replay what a client missed while disconnected."""
    p = project_of(request)
    after = int(request.query.get("after", 0))
    return ok({"events": p.store.events(after=after, aggregate=request.query.get("aggregate"),
                                        limit=min(int(request.query.get("limit", 500)), 5000)),
               "verified": p.store.verify()[0]})


@routes.get("/api/session")
async def list_sessions(request: web.Request):
    return ok(project_of(request).sessions(include_children=request.query.get("children") == "1"))


@routes.post("/api/session")
async def create_session(request: web.Request):
    b = await body_of(request)
    p = project_of(request)
    s = p.create_session(title=b.get("title"), agent=b.get("agent") or request.app[DEFAULT_AGENT](p),
                         model=b.get("model"), parent_id=b.get("parent_id"))
    return ok(s, 201)


@routes.get("/api/session/{sid}")
async def get_session(request: web.Request):
    return ok(project_of(request).session(request.match_info["sid"]))


@routes.patch("/api/session/{sid}")
async def patch_session(request: web.Request):
    b = await body_of(request)
    p = project_of(request)
    p.session(request.match_info["sid"])
    allowed = {k: b[k] for k in ("title", "agent", "model", "metadata") if k in b}
    return ok(p.store.update("session", request.match_info["sid"], allowed))


@routes.delete("/api/session/{sid}")
async def delete_session(request: web.Request):
    p = project_of(request)
    p.session(request.match_info["sid"])
    p.store.delete("session", request.match_info["sid"])
    return ok({"deleted": request.match_info["sid"]})


@routes.get("/api/session/{sid}/message")
async def session_messages(request: web.Request):
    p = project_of(request)
    p.session(request.match_info["sid"])
    return ok(p.messages(request.match_info["sid"]))


@routes.post("/api/pair")
async def pair(request: web.Request):
    code = request.app[AUTH].pair()
    base = f"{request.scheme}://{request.host}"
    return ok({"code": code, "url": f"{base}/auth/connect/{code}", "expires_in": PAIR_TTL})


@routes.get("/auth/connect/{code}")
async def connect(request: web.Request):
    auth: Auth = request.app[AUTH]
    if not auth.redeem(request.match_info["code"]):
        return web.Response(status=403, text="This link was already used or has expired. Run `opentrain web` again.")
    resp = web.HTTPFound(request.query.get("next", "/"))
    resp.set_cookie(COOKIE, auth.token(), max_age=COOKIE_TTL, httponly=True, samesite="Strict")
    raise resp


def make_app(*, password: str, hub: Hub | None = None, local_only: bool = True,
             extra_routes: list[web.RouteTableDef] | None = None) -> web.Application:
    import os

    app = web.Application(middlewares=[guard], client_max_size=32 * 1024 * 1024)
    app[HUB] = hub or Hub()
    app[AUTH] = Auth(password)
    app[LOCAL_ONLY] = local_only
    app[PID] = os.getpid()
    app[DEFAULT_AGENT] = lambda p: "train"
    app.add_routes(routes)
    for r in extra_routes or []:
        app.add_routes(r)

    async def on_startup(app: web.Application) -> None:
        app[HUB].loop = asyncio.get_running_loop()

    async def on_shutdown(app: web.Application) -> None:
        for q in list(app[HUB].subscribers):  # end every open event stream
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def on_cleanup(app: web.Application) -> None:
        app[HUB].close()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.on_cleanup.append(on_cleanup)
    return app
