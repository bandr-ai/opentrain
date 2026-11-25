"""Which services the server attaches to each project, and which extra routes it serves.

Kept in one place so the order is explicit: later commits add the engine, permissions, jobs,
the providers and the web UI here.
"""

from __future__ import annotations

from aiohttp import web

from .app import Hub


def attach_all(hub: Hub) -> None:
    pass


def extra_routes() -> list[web.RouteTableDef]:
    return []


def configure_app(app: web.Application) -> None:
    pass
