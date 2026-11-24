"""``opentrain web``: the local dashboard over ``.opentrain/``."""

from __future__ import annotations

from ..state import state_dir


def cmd_web(args) -> int:
    from ..web.server import serve_web

    return serve_web(state_dir(args.state), port=args.port, open_browser=not args.no_open)


def register(sub) -> None:
    p = sub.add_parser("web", help="open the local dashboard: runs, evals, versions, data, policy")
    p.add_argument("--port", type=int, default=7878, help="port on 127.0.0.1 (default 7878; next free if taken)")
    p.add_argument("--no-open", action="store_true", help="do not open a browser")
    p.add_argument("--state", default=None, help="folder that holds .opentrain/ (default: here)")
    p.set_defaults(fn=cmd_web)
