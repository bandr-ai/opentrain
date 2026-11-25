"""Which services the server attaches to each project, and which extra routes it serves.

Kept in one place so the order is explicit.
"""

from __future__ import annotations

from aiohttp import web

from ..core.project import Project
from .app import Hub


def attach_all(hub: Hub) -> None:
    from ..core.integrations import load_into_env, load_plugins

    load_into_env()
    load_plugins()
    hub.attachers.append(_engine)


def _engine(hub: Hub, p: Project) -> None:
    from ..core.agent import Agents
    from ..core.jobs import Jobs
    from ..core.questions import Questions
    from ..core.runner import Runner
    from ..core.tool import Registry
    from ..core.tools import register_all

    p.services["agents"] = Agents(p)
    p.services["tools"] = Registry()
    register_all(p)
    from ..core.permission import BudgetHook, Permissions

    p.services["questions"] = Questions(p)
    p.services["permissions"] = Permissions(p)
    p.services["runner"] = Runner(p, hub.loop)
    p.services["runner"].on_usage.append(BudgetHook(p))
    p.services["jobs"] = Jobs(p, hub.loop)
    from ..core.mcp_client import McpServers

    mcp = McpServers(p)
    p.services["mcp"] = mcp
    import threading

    threading.Thread(target=mcp.start, name=f"mcp {p.name}", daemon=True).start()


def extra_routes() -> list[web.RouteTableDef]:
    from .dashboard import routes as web_routes
    from .data_routes import routes as data_routes
    from .routes import routes

    return [routes, data_routes, web_routes]


def configure_app(app: web.Application) -> None:
    from .app import DEFAULT_AGENT

    app[DEFAULT_AGENT] = lambda p: p.services["agents"].default()
