"""Engine routes: prompting a session, interrupting it, its tools, agents and models."""

from __future__ import annotations

import asyncio

from aiohttp import web

from .app import ApiError, body_of, ok, project_of

routes = web.RouteTableDef()


def _runner(request: web.Request):
    return project_of(request).services["runner"]


@routes.post("/api/session/{sid}/prompt")
async def prompt(request: web.Request):
    b = await body_of(request)
    text = (b.get("text") or "").strip()
    if not text:
        raise ApiError(400, "text is required")
    item = _runner(request).prompt(request.match_info["sid"], text, delivery=b.get("delivery", "steer"),
                                   agent=b.get("agent"), model=b.get("model"))
    return ok(item, 202)


@routes.post("/api/session/{sid}/interrupt")
async def interrupt(request: web.Request):
    return ok({"interrupted": await _runner(request).interrupt(request.match_info["sid"])})


@routes.post("/api/session/{sid}/wait")
async def wait(request: web.Request):
    """Block until the session is idle (for ``opentrain run`` and tests)."""
    b = await body_of(request)
    r = _runner(request)
    try:
        await r.wait_idle(request.match_info["sid"], timeout=float(b.get("timeout", 3600)))
    except asyncio.TimeoutError:
        raise ApiError(408, "still running") from None
    return ok(project_of(request).session(request.match_info["sid"]))


@routes.get("/api/session/{sid}/tools")
async def session_tools(request: web.Request):
    p = project_of(request)
    agent = p.services["agents"].get(p.session(request.match_info["sid"]).get("agent"))
    return ok([t.spec() for t in p.services["runner"].tools_for(agent)])


@routes.post("/api/session/{sid}/tool-call")
async def tool_call(request: web.Request):
    """An agent provider's tool call, arriving through the MCP bridge."""
    b = await body_of(request)
    r = _runner(request)
    try:
        res = await r.external_tool_call(request.match_info["sid"], b.get("name", ""), b.get("input") or {},
                                         b.get("call_id") or "")
    except Exception as exc:
        return ok({"output": str(exc), "is_error": True})
    return ok({"output": res.output, "is_error": bool(res.metadata.get("error"))})


@routes.get("/api/agent")
async def agents(request: web.Request):
    return ok([a.public() for a in project_of(request).services["agents"].list()])


@routes.get("/api/provider")
async def providers(request: web.Request):
    from ..core.providers import default_model, providers as all_providers

    return ok({"providers": [p.public() for p in all_providers()], "default": default_model()})
