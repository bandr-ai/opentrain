"""Run the OpenTrain server: ``opentrain serve`` (foreground) or ``--service`` (registered, background)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from pathlib import Path

from aiohttp import web

from .. import __version__
from . import service
from .app import Hub, make_app


def build_app(password: str, *, default: Path | None = None, local_only: bool = True) -> web.Application:
    """The app with every service attached. Kept separate so tests can run it in-process."""
    from .wiring import attach_all, extra_routes

    hub = Hub(default)
    attach_all(hub)
    app = make_app(password=password, hub=hub, local_only=local_only, extra_routes=extra_routes())
    from .wiring import configure_app

    configure_app(app)
    return app


async def _serve(host: str, port: int, password: str, register: bool, default: Path | None) -> None:
    app = build_app(password, default=default, local_only=host in ("127.0.0.1", "localhost", "::1"))
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        await site.start()
    except OSError:
        if port == 0:
            raise
        site = web.TCPSite(runner, host, 0)  # the port is taken: take any
        await site.start()
    sock = site._server.sockets[0]  # type: ignore[union-attr]
    real_port = sock.getsockname()[1]
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{real_port}"
    if register:
        service.write_registration(service.Registration(url=url, pid=os.getpid(), version=__version__,
                                                        password=password))
    logging.getLogger("opentrain").info("OpenTrain %s listening on %s", __version__, url)
    print(f"OpenTrain server {__version__} on {url}", flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await stop.wait()
    finally:
        if register:
            service.clear_registration(os.getpid())
        await runner.cleanup()


def run(host: str = "127.0.0.1", port: int = service.DEFAULT_PORT, *, password: str | None = None,
        register: bool = False, default: Path | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    password = password or os.environ.get("OPENTRAIN_SERVER_PASSWORD") or service.new_password()
    asyncio.run(_serve(host, port, password, register, default))
    return 0
