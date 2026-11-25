"""Which services the server attaches to each project, and which extra routes it serves.

Kept in one place so the order is explicit.
"""

from __future__ import annotations

from aiohttp import web

from ..core.project import Project
from .app import Hub


def attach_all(hub: Hub) -> None:
    hub.attachers.append(_engine)


def _engine(hub: Hub, p: Project) -> None:
    from ..core.agent import Agents
    from ..core.runner import Runner
    from ..core.tool import Registry

    p.services["agents"] = Agents(p)
    p.services["tools"] = Registry()
    p.services["runner"] = Runner(p, hub.loop)


def extra_routes() -> list[web.RouteTableDef]:
    from .routes import routes

    return [routes]


def configure_app(app: web.Application) -> None:
    from .app import DEFAULT_AGENT

    app[DEFAULT_AGENT] = lambda p: p.services["agents"].default()
