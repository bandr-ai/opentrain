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


@routes.get("/api/outside/tools")
async def outside_tools(request: web.Request):
    """The tools an agent outside OpenTrain may call through ``opentrain mcp``."""
    p = project_of(request)
    return ok([t.spec() for t in p.services["runner"].tools_for(p.services["agents"].get("outside"))])


@routes.post("/api/outside/call")
async def outside_call(request: web.Request):
    b = await body_of(request)
    try:
        res = await _runner(request).outside_call(b.get("name", ""), b.get("input") or {})
    except Exception as exc:
        return ok({"output": str(exc), "is_error": True})
    return ok({"output": res.output, "is_error": bool(res.metadata.get("error"))})


@routes.get("/api/agent")
async def agents(request: web.Request):
    return ok([a.public() for a in project_of(request).services["agents"].list()])


@routes.get("/api/provider")
async def providers(request: web.Request):
    from ..core.providers import default_model
    from ..core.providers import providers as all_providers

    return ok({"providers": [p.public() for p in all_providers()], "default": default_model()})


@routes.post("/api/session/{sid}/question/{qid}/reply")
async def question_reply(request: web.Request):
    b = await body_of(request)
    qs = project_of(request).services["questions"]
    return ok(qs.reply(request.match_info["qid"], b.get("answers"), rejected=bool(b.get("reject"))))


@routes.get("/api/question")
async def questions(request: web.Request):
    return ok(project_of(request).services["questions"].pending(request.query.get("session")))


@routes.post("/api/session/{sid}/proposal/{pid}/reply")
async def proposal_reply(request: web.Request):
    b = await body_of(request)
    ps = project_of(request).services["proposals"]
    try:
        return ok(ps.reply(request.match_info["pid"], str(b.get("reply") or ""), b.get("message")))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None


@routes.get("/api/proposal")
async def proposals(request: web.Request):
    return ok(project_of(request).services["proposals"].pending(request.query.get("session")))


@routes.get("/api/job")
async def jobs(request: web.Request):
    return ok(project_of(request).services["jobs"].list(request.query.get("session"),
                                                        active=request.query.get("active") == "1"))


@routes.get("/api/job/{jid}")
async def job(request: web.Request):
    return ok(project_of(request).services["jobs"].get(request.match_info["jid"]))


@routes.get("/api/job/{jid}/logs")
async def job_logs(request: web.Request):
    tail = int(request.query.get("tail", 200))
    return ok({"lines": project_of(request).services["jobs"].logs(request.match_info["jid"], tail)})


@routes.post("/api/job/{jid}/cancel")
async def job_cancel(request: web.Request):
    return ok(project_of(request).services["jobs"].cancel(request.match_info["jid"]))


@routes.get("/api/session/{sid}/todo")
async def todos(request: web.Request):
    p = project_of(request)
    t = p.store.get("todo", f"todo_{request.match_info['sid']}")
    return ok((t or {}).get("todos", []))


@routes.get("/api/skill")
async def skills(request: web.Request):
    from ..core.skills import Skills

    return ok([s.public() for s in Skills(project_of(request).directory).all().values()])


@routes.get("/api/tool")
async def tools(request: web.Request):
    return ok([t.spec() for t in project_of(request).services["tools"].tools.values()])


@routes.get("/api/permission")
async def permissions(request: web.Request):
    return ok(project_of(request).services["permissions"].pending(request.query.get("session")))


@routes.post("/api/session/{sid}/permission/{pid}/reply")
async def permission_reply(request: web.Request):
    b = await body_of(request)
    perms = project_of(request).services["permissions"]
    try:
        return ok(perms.reply(request.match_info["pid"], b.get("reply", ""), b.get("message")))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None


@routes.get("/api/approval")
async def approvals(request: web.Request):
    return ok(project_of(request).services["permissions"].approvals())


@routes.delete("/api/approval/{aid}")
async def revoke(request: web.Request):
    project_of(request).services["permissions"].revoke(request.match_info["aid"])
    return ok({"revoked": request.match_info["aid"]})


@routes.get("/api/integration")
async def integrations(request: web.Request):
    from ..core.integrations import listing

    return ok(listing())


@routes.post("/api/integration/{iid}/connect")
async def connect_integration(request: web.Request):
    from ..core.integrations import connect

    b = await body_of(request)
    try:
        return ok(connect(request.match_info["iid"], b.get("key") or ""))
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None


@routes.delete("/api/integration/{iid}")
async def disconnect_integration(request: web.Request):
    from ..core.integrations import disconnect

    return ok({"removed": disconnect(request.match_info["iid"])})


@routes.get("/api/model")
async def models(request: web.Request):
    from ..core.providers import default_model
    from ..core.providers import providers as all_providers

    session_model = None
    if request.query.get("session"):
        p = project_of(request)
        s = p.session(request.query["session"])
        session_model = p.services["runner"].resolve_model(s, p.services["agents"].get(s.get("agent")))
    rows = []
    for prov in all_providers():
        for m in prov.models:
            rows.append({"ref": m.ref, "name": m.name, "provider": prov.id, "provider_name": prov.name,
                         "connected": prov.connected, "context": m.context,
                         "input_price": m.input_price, "output_price": m.output_price})
    return ok({"models": rows, "default": default_model(), "session": session_model})


@routes.post("/api/model/default")
async def set_default_model(request: web.Request):
    from ..core import config
    from ..core.llm import parse_ref

    b = await body_of(request)
    ref = b.get("ref") or ""
    try:
        parse_ref(ref)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
    config.save_global({"model": ref})
    return ok({"default": ref})


@routes.get("/api/trainer")
async def trainers(request: web.Request):
    from ..core.integrations import trainers as list_trainers

    p = project_of(request)
    return ok({"trainers": list_trainers(p.state), "policy": p.info().get("policy")})


@routes.post("/api/trainer")
async def set_trainer(request: web.Request):
    """Set the project's trainer and/or base model in its Trainfile."""
    from .. import trainfile

    b = await body_of(request)
    p = project_of(request)
    path = None
    try:
        if b.get("trainer"):
            from ..backends import BACKENDS

            if b["trainer"] not in BACKENDS:
                raise ApiError(400, f"no trainer {b['trainer']!r}; known: {', '.join(BACKENDS)}")
            path = trainfile.set_value(p.directory, "trainer", b["trainer"])
        if b.get("base_model"):
            path = trainfile.set_value(p.directory, "base model", b["base_model"])
    except trainfile.TrainfileError as exc:
        raise ApiError(400, str(exc)) from None
    return ok({"trainfile": str(path) if path else None, "policy": p.info().get("policy")})


@routes.get("/api/datasource")
async def datasources(request: web.Request):
    from ..core.integrations import data_sources

    return ok(data_sources())


@routes.get("/api/command")
async def commands(request: web.Request):
    from ..core.commands import load

    return ok([c.public() for c in load(project_of(request).directory).values()])


@routes.post("/api/session/{sid}/command")
async def run_command(request: web.Request):
    from ..core.commands import load

    b = await body_of(request)
    p = project_of(request)
    cmds = load(p.directory)
    name = (b.get("name") or "").lstrip("/")
    if name not in cmds:
        raise ApiError(404, f"no command /{name}; known: {', '.join('/' + n for n in sorted(cmds))}", "not_found")
    cmd = cmds[name]
    text = cmd.expand(b.get("arguments") or "")
    item = p.services["runner"].prompt(request.match_info["sid"], text, delivery=b.get("delivery", "steer"),
                                       agent=cmd.agent, model=cmd.model)
    return ok({**item, "command": name}, 202)


@routes.get("/api/config")
async def get_config(request: web.Request):
    from ..core import config

    p = project_of(request)
    try:
        cfg = config.load(p.directory)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
    providers = {k: {kk: vv for kk, vv in v.items() if kk not in ("api_key", "key")}
                 for k, v in (cfg.get("providers") or {}).items() if isinstance(v, dict)}
    agents = p.services["agents"]
    return ok({**cfg, "providers": providers, "agent_problems": agents.problems,
               "global_dir": str(config.global_dir())})


@routes.get("/api/mcp")
async def mcp_status(request: web.Request):
    m = project_of(request).services.get("mcp")
    return ok(m.status if m else {})


@routes.get("/api/status")
async def project_status(request: web.Request):
    import asyncio as _asyncio

    from ..core.tools.train import project_status as status

    p = project_of(request)
    out = await _asyncio.to_thread(status, p)
    out["project"] = p.info()
    return ok(out)
