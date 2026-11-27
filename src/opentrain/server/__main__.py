"""``python -m opentrain.server [--service] [--host H] [--port P]``: run the server directly."""

import argparse
import sys

from . import run, service


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="opentrain-server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=service.DEFAULT_PORT)
    ap.add_argument("--service", action="store_true", help="register as the user's background server")
    a = ap.parse_args(argv)
    return run(a.host, a.port, register=a.service)


sys.exit(main())
