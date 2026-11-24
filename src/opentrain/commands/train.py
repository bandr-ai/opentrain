"""Training and serving commands: ``backends``, ``job``, ``serve``, ``model`` and ``connect``.

Jobs run in detached workers, so these commands only read and signal what is on disk under
``.opentrain/``: any shell can see a job another shell launched, and closing the terminal that
launched it does not stop it.
"""

from __future__ import annotations

import json
import sys

from ..state import ensure


def _state(p) -> None:
    p.add_argument("--state", default=None, help="folder that holds .opentrain/ (default: here)")


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([fmt(header), fmt(["-" * w for w in widths]), *(fmt(r) for r in rows)])


def cmd_backends(args) -> int:
    from ..backends import list_backends

    rows = [[name, "yes" if ok else "no", why] for name, ok, why in list_backends(ensure(args.state))]
    print(_table(rows, ["Backend", "Ready", "Why"]))
    return 0


def cmd_job_ps(args) -> int:
    from ..backends import JobStore

    store = JobStore(ensure(args.state))
    rows = []
    for job in store.jobs():
        st = store.status(job["job_id"])
        pct = "" if st.progress is None else f"{st.progress * 100:.0f}%"
        rows.append([st.job_id, job["backend"], st.state, pct, f"${st.cost_usd:.2f}", st.checkpoint or "",
                     st.detail[:60]])
    if not rows:
        print("No training jobs yet.")
        return 0
    print(_table(rows, ["Job", "Backend", "State", "Done", "Cost", "Checkpoint", "Detail"]))
    return 0


def cmd_job_logs(args) -> int:
    from ..backends import JobStore

    try:
        lines = JobStore(ensure(args.state)).logs(args.job_id, tail=args.lines)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 1
    print("\n".join(lines) if lines else "(no log lines yet)")
    return 0


def cmd_job_cancel(args) -> int:
    from ..backends import JobStore, get_backend

    d = ensure(args.state)
    try:
        job = JobStore(d).load(args.job_id)
        get_backend(job["backend"], d).cancel(job["job_id"])
    except (KeyError, NotImplementedError) as e:
        print(e.args[0] if e.args else str(e), file=sys.stderr)
        return 1
    print(f"Cancel requested for {job['job_id']}; the worker stops at its next step.")
    return 0


def cmd_serve(args) -> int:
    from ..serving import Handoff, live_version, make_server

    d = ensure(args.state)
    handoff = None
    if args.handoff:
        if not args.handoff_model:
            print("--handoff needs --handoff-model (the frontier model name at that endpoint)", file=sys.stderr)
            return 2
        handoff = Handoff(args.handoff, args.handoff_model, args.handoff_key_env)
    server = make_server(d, args.host, args.port, handoff)
    host, port = server.server_address[:2]
    print(f"Serving   http://{host}:{port}/v1  (opentrain/current -> {live_version(d) or 'nothing live yet'})")
    if handoff:
        print(f"Handoff   {handoff.model} at {handoff.base_url} when the specialist fails")
    print(f"Log       {d / 'serving' / 'requests.jsonl'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_model_ls(args) -> int:
    from ..registry import Registry
    from ..serving import live_version

    d = ensure(args.state)
    reg = Registry(d)
    versions = reg.versions()
    if args.json:
        print(json.dumps([v.__dict__ for v in versions], indent=2, default=str))
        return 0
    if not versions:
        print("No model versions yet.")
        return 0
    live = live_version(d)
    cur = reg.current()
    rows = []
    for v in versions:
        mark = ("live " if v.version == live else "") + ("current" if cur and v.version == cur.version else "")
        rows.append([v.version, v.backend, v.base_model, v.decision, mark.strip(), v.checkpoint])
    print(_table(rows, ["Version", "Backend", "Base", "Decision", "", "Checkpoint"]))
    return 0


def cmd_model_use(args) -> int:
    from ..backends.fake import SIMULATED
    from ..registry import Registry, RegistryError
    from ..serving import ServingError, endpoint_for, start_local_server, use_version

    d = ensure(args.state)
    try:
        v = Registry(d).get(args.version)
        if v.checkpoint.startswith("fake://"):
            print(f"warning: {v.version} is {SIMULATED}", file=sys.stderr)
        elif v.backend == "local" and not (v.serving or {}).get("base_url"):
            ep, pid = start_local_server(v, d)
            print(f"Started   local server pid {pid} at {ep.base_url} (model {ep.model}); log in "
                  f"{d / 'serving'}")
        else:
            endpoint_for(v)
        use_version(d, v.version, why=args.why)
    except (RegistryError, ServingError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"opentrain/current -> {v.version}. A running `opentrain serve` picks this up on the next request.")
    return 0


def cmd_connect(args) -> int:
    from ..serving import client_config

    print(client_config(args.client, host=args.host, port=args.port))
    return 0


def register(sub) -> None:
    from ..serving import DEFAULT_PORT

    p = sub.add_parser("backends", help="trainer backends and whether each can run here")
    _state(p)
    p.set_defaults(fn=cmd_backends)

    job = sub.add_parser("job", help="training jobs")
    jsub = job.add_subparsers(dest="sub", required=True)
    p = jsub.add_parser("ps", help="list training jobs")
    _state(p)
    p.set_defaults(fn=cmd_job_ps)
    p = jsub.add_parser("logs", help="show a job's log")
    p.add_argument("job_id")
    p.add_argument("-n", "--lines", type=int, default=200, help="last N lines (0: all)")
    _state(p)
    p.set_defaults(fn=cmd_job_logs)
    p = jsub.add_parser("cancel", help="stop a job at its next step")
    p.add_argument("job_id")
    _state(p)
    p.set_defaults(fn=cmd_job_cancel)

    p = sub.add_parser("serve", help="OpenAI-compatible endpoint for opentrain/current, hot-switchable")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--handoff", metavar="URL", help="frontier OpenAI-compatible base URL to use when the specialist fails")
    p.add_argument("--handoff-model", help="model name at the handoff endpoint")
    p.add_argument("--handoff-key-env", help="environment variable that holds the handoff endpoint's key")
    _state(p)
    p.set_defaults(fn=cmd_serve)

    model = sub.add_parser("model", help="model versions")
    msub = model.add_subparsers(dest="sub", required=True)
    p = msub.add_parser("ls", help="list versions")
    p.add_argument("--json", action="store_true", help="print machine-readable output")
    _state(p)
    p.set_defaults(fn=cmd_model_ls)
    p = msub.add_parser("use", help="point opentrain/current at a version (no restart needed)")
    p.add_argument("version")
    p.add_argument("--why", default="manual", help="recorded with the switch")
    _state(p)
    p.set_defaults(fn=cmd_model_use)

    p = sub.add_parser("connect", help="print config for a coding agent to use opentrain serve")
    p.add_argument("client", choices=["pi", "opencode", "claude"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    _state(p)
    p.set_defaults(fn=cmd_connect)

