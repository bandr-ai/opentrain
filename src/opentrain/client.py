"""The client every front end uses: the TUI, ``opentrain run``, scripts, tests.

``Client`` is synchronous (urllib) for commands and scripts. ``AsyncClient`` uses aiohttp and adds
the SSE event stream for the terminal app. Both scope every request to one project directory.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, AsyncIterator, Iterator


class ClientError(RuntimeError):
    def __init__(self, status: int, message: str, kind: str = ""):
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind


def _auth(password: str) -> str:
    return "Basic " + base64.b64encode(f"opentrain:{password}".encode()).decode()


class _Base:
    def __init__(self, url: str, password: str, directory: str | Path):
        self.url = url.rstrip("/")
        self.password = password
        self.directory = str(Path(directory).expanduser().resolve())

    def _url(self, path: str, query: dict[str, Any] | None = None) -> str:
        q = {"directory": self.directory, **{k: v for k, v in (query or {}).items() if v is not None}}
        return f"{self.url}{path}?{urllib.parse.urlencode(q)}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": _auth(self.password), "Content-Type": "application/json"}


class Client(_Base):
    """Blocking client."""

    def request(self, method: str, path: str, body: dict | None = None, query: dict | None = None,
                timeout: float = 60) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._url(path, query), data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read()).get("error", {})
            except ValueError:
                err = {}
            raise ClientError(exc.code, err.get("message") or str(exc), err.get("type", "")) from None
        return json.loads(raw) if raw else None

    def get(self, path: str, **query: Any) -> Any:
        return self.request("GET", path, query=query)

    def post(self, path: str, body: dict | None = None, **query: Any) -> Any:
        return self.request("POST", path, body or {}, query=query)

    def patch(self, path: str, body: dict) -> Any:
        return self.request("PATCH", path, body)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)

    def events(self, types: list[str] | None = None, timeout: float | None = None) -> Iterator[dict[str, Any]]:
        """The SSE stream, one event dict at a time (blocking)."""
        req = urllib.request.Request(self._url("/api/event", {"types": ",".join(types) if types else None}),
                                     headers=self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            yield from _parse_sse(line.decode("utf-8", "replace") for line in r)


def _parse_sse(lines) -> Iterator[dict[str, Any]]:
    data: list[str] = []
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            if data:
                try:
                    yield json.loads("\n".join(data))
                except ValueError:
                    pass
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


class AsyncClient(_Base):
    """aiohttp client; create inside a running loop and ``await close()`` when done."""

    def __init__(self, url: str, password: str, directory: str | Path):
        import aiohttp

        super().__init__(url, password, directory)
        self._session = aiohttp.ClientSession(headers={"Authorization": _auth(password)})

    async def close(self) -> None:
        await self._session.close()

    async def request(self, method: str, path: str, body: dict | None = None, query: dict | None = None) -> Any:
        async with self._session.request(method, self._url(path, query), json=body) as r:
            raw = await r.read()
            data = json.loads(raw) if raw else None
            if r.status >= 400:
                err = (data or {}).get("error", {}) if isinstance(data, dict) else {}
                raise ClientError(r.status, err.get("message") or f"HTTP {r.status}", err.get("type", ""))
            return data

    async def get(self, path: str, **query: Any) -> Any:
        return await self.request("GET", path, query=query)

    async def post(self, path: str, body: dict | None = None, **query: Any) -> Any:
        return await self.request("POST", path, body or {}, query=query)

    async def patch(self, path: str, body: dict) -> Any:
        return await self.request("PATCH", path, body)

    async def delete(self, path: str) -> Any:
        return await self.request("DELETE", path)

    async def events(self, types: list[str] | None = None) -> AsyncIterator[dict[str, Any]]:
        import aiohttp

        url = self._url("/api/event", {"types": ",".join(types) if types else None})
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=None, sock_read=None)) as r:
            if r.status >= 400:
                raise ClientError(r.status, f"event stream refused: HTTP {r.status}")
            data: list[str] = []
            async for raw in r.content:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if not line:
                    if data:
                        try:
                            yield json.loads("\n".join(data))
                        except ValueError:
                            pass
                        data = []
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())


def connect(directory: str | Path, *, url: str | None = None, password: str | None = None,
            start: bool = True) -> Client:
    """A client for ``directory`` on the given server, or on the user's service (started if needed)."""
    import os

    from .server import service

    url = url or os.environ.get("OPENTRAIN_SERVER")
    if url:
        return Client(url, password or os.environ.get("OPENTRAIN_SERVER_PASSWORD", ""), directory)
    reg = service.ensure() if start else service.discover()
    if reg is None:
        raise RuntimeError("no OpenTrain server is running")
    return Client(reg.url, reg.password, directory)
