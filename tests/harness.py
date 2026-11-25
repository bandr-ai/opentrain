"""Run a real OpenTrain server in a background thread for tests."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from aiohttp import web

from opentrain.client import Client
from opentrain.server.app import HUB


class LiveServer:
    def __init__(self, default: Path, password: str = "pw"):
        from opentrain.server import build_app

        self.password = password
        self.app = build_app(password, default=default)
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self.ready.wait(10), "server did not start"

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.runner = web.AppRunner(self.app)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        self.loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.app[HUB].url = self.url
        self.ready.set()
        self.loop.run_forever()

    @property
    def hub(self):
        return self.app[HUB]

    def client(self, directory: Path) -> Client:
        return Client(self.url, self.password, directory)

    def stop(self):
        fut = asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop)
        try:
            fut.result(10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(10)
