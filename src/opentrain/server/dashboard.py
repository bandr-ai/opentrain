"""The web dashboard, served by the same server as everything else.

``opentrain web`` opens a one-time pairing link; the browser gets a signed cookie and, with
``?directory=``, a cookie naming the project. The page (runs, evals, versions, race, data,
skills, policy) reads the same state the terminal app and the engine use.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from aiohttp import web

from ..registry import RegistryError
from ..trainfile import TrainfileError
from ..web import api
from .app import ApiError, body_of, ok, project_of

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
APP = Path(__file__).resolve().parent.parent / "web" / "app"
"""The web app (built from ``web/`` with ``pnpm build``). The older single-file dashboard in
``static`` is the fallback when the app was not built."""
TYPES = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png",
         ".woff2": "font/woff2", ".json": "application/json", ".ico": "image/x-icon"}
TOKEN_MARK = "__OPENTRAIN_TOKEN__"
DIR_COOKIE = "opentrain_dir"
SECURITY = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                        "connect-src 'self'; img-src 'self' data:; font-src 'self'; frame-src 'self'; base-uri 'none'; "
                                        "form-action 'none'; frame-ancestors 'none'")}

routes = web.RouteTableDef()

PACKAGE = Path(__file__).resolve().parent.parent


def _code_stamp() -> float:
    """When OpenTrain's code on disk last changed (an upgrade, or a checkout under an editable install)."""
    return max((p.stat().st_mtime for p in PACKAGE.rglob("*.py")), default=0.0)


def _page_script() -> str | None:
    built = APP / "index.html"
    m = re.search(r"/assets/(index-[\w-]+\.js)", built.read_text()) if built.exists() else None
    return m.group(1) if m else None


STARTED = _code_stamp()  # what this server process is running


def _state(request: web.Request) -> Path:
    return project_of(request).state


@routes.get("/")
async def index(request: web.Request):
    built = APP / "index.html"
    html = built.read_text() if built.exists() else (STATIC / "index.html").read_text().replace(TOKEN_MARK, "cookie")
    resp = web.Response(text=html, content_type="text/html", headers={**SECURITY, "Cache-Control": "no-cache"})
    if request.query.get("directory"):
        resp.set_cookie(DIR_COOKIE, request.query["directory"], max_age=30 * 24 * 3600, httponly=True, samesite="Strict")
    return resp


@routes.get("/api/build")
async def build(request: web.Request):
    """Whether this server runs older code than is on disk (restart it), and which page script is
    current (a tab running another one should reload)."""
    stamp, script = await asyncio.to_thread(lambda: (_code_stamp(), _page_script()))
    return ok({"restart": stamp > STARTED + 1, "script": script})


@routes.get("/assets/{name}")
async def asset(request: web.Request):
    name = request.match_info["name"]
    p = (APP / "assets" / name).resolve()
    if not p.is_file() or p.parent != (APP / "assets").resolve():
        raise web.HTTPNotFound()
    return web.Response(body=p.read_bytes(), content_type=TYPES.get(p.suffix, "application/octet-stream"),
                        headers={**SECURITY, "Cache-Control": "public, max-age=31536000, immutable"})


@routes.get("/app.js")
async def js(request: web.Request):
    return web.Response(text=(STATIC / "app.js").read_text(), content_type="text/javascript", headers=SECURITY)


@routes.get("/app.css")
async def css(request: web.Request):
    return web.Response(text=(STATIC / "app.css").read_text(), content_type="text/css", headers=SECURITY)


async def _call(fn, *args):
    try:
        return ok(await asyncio.to_thread(fn, *args))
    except api.NotFound as exc:
        raise ApiError(404, str(exc) or "not found", "not_found") from None


SIMPLE = {"home": api.home, "runs": api.runs, "versions": api.versions, "race": api.race, "data": api.data,
          "skills": api.skills, "policy": api.policy}

for _name, _fn in SIMPLE.items():
    def _make(fn):
        async def handler(request: web.Request):
            return await _call(fn, _state(request))
        return handler
    routes.get(f"/api/{_name}")(_make(_fn))


@routes.get("/api/evals")
async def evals(request: web.Request):
    return await _call(api.evals, _state(request), request.query.get("a") or None, request.query.get("b") or None)


@routes.get("/api/runs/{rid}")
async def run(request: web.Request):
    return await _call(api.run, _state(request), request.match_info["rid"])


@routes.get("/api/tasks/{tid}")
async def task(request: web.Request):
    return await _call(api.task, _state(request), request.match_info["tid"])


@routes.get("/api/runs/{rid}/events")
async def run_events(request: web.Request):
    state, rid = _state(request), request.match_info["rid"]
    try:
        r = await asyncio.to_thread(api.get_run, state, rid)
    except api.NotFound as exc:
        raise ApiError(404, str(exc), "not_found") from None
    try:
        after = int(request.headers.get("Last-Event-ID") or request.query.get("after") or -1)
    except ValueError:
        after = -1
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store", **SECURITY})
    await resp.prepare(request)
    await resp.write(b"retry: 2000\n\n")
    size, beat = -1, 0.0
    try:
        while True:
            cur = r.events_path.stat().st_size if r.events_path.exists() else 0
            if cur != size:
                size = cur
                new, closed = await asyncio.to_thread(api.events_since, state, rid, after)
                for e in new:
                    after = e["seq"]
                    await resp.write(f"id: {after}\nevent: run\ndata: {json.dumps(e, default=str)}\n\n".encode())
                if closed:
                    await resp.write(b"event: end\ndata: {}\n\n")
                    return resp
            beat += 0.5
            if beat >= 15:
                beat = 0
                await resp.write(b": keep-alive\n\n")
            await asyncio.sleep(0.5)
    except (ConnectionResetError, asyncio.CancelledError):
        return resp


@routes.post("/api/versions/revert")
async def revert(request: web.Request):
    b = await body_of(request)
    try:
        return await _call(api.revert, _state(request), str(b.get("why") or "rolled back from the web UI")[:500])
    except RegistryError as exc:
        raise ApiError(409, str(exc)) from None


@routes.post("/api/policy")
async def save_policy(request: web.Request):
    b = await body_of(request)
    try:
        return await _call(api.save_policy, _state(request), b.get("text"))
    except TrainfileError as exc:
        return ok({"error": str(exc), "invalid": True}, 400)
