"""The OpenTrain terminal app: a client of the server, like every other front end."""

from __future__ import annotations

from pathlib import Path


def run(directory: Path, url: str, password: str, session: str | None = None, prompt: str | None = None) -> int:
    from .app import OpenTrainApp

    app = OpenTrainApp(directory, url, password, session_id=session, initial_prompt=prompt)
    app.run()
    return app.return_code or 0
