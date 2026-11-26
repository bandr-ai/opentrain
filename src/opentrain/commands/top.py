"""The ``opentrain`` command: a handful of verbs, shaped like opencode's.

    opentrain [dir]              the terminal app (starts the background server if needed)
    opentrain run "..."          one instruction, no interface: prints the answer, exits
    opentrain serve              the server in the foreground (for a remote or headless box)
    opentrain web [dir]          the dashboard in a browser (a one-time sign-in link)
    opentrain attach URL         the terminal app against a server elsewhere
    opentrain pair               print a one-time link to sign a browser in
    opentrain stop               stop the background server
    opentrain export [--out D]   tasks and training set in open formats, for any trainer
    opentrain watch              the unattended loop
    opentrain model ...          serve a trained version (OpenAI-compatible), list, switch, connect an agent
    opentrain debug ...          every pipeline step on its own (bench, mine, sample, eval, ...)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .. import __version__

VERBS = ("run", "serve", "web", "mcp", "login", "attach", "pair", "stop", "export", "watch", "model", "debug", "help")


def _client(directory: Path, url: str | None = None, password: str | None = None):
    from ..client import connect

    return connect(directory, url=url, password=password)


def _tui(directory: Path, url: str | None, password: str | None, session: str | None, prompt: str | None) -> int:
    from ..server import service
    from ..tui import run as run_tui

    if url:
        return run_tui(directory, url, password or os.environ.get("OPENTRAIN_SERVER_PASSWORD", ""), session, prompt)
    reg = service.ensure()
    return run_tui(directory, reg.url, reg.password, session, prompt)


# ---------------------------------------------------------------------- run


def cmd_run(a: argparse.Namespace) -> int:
    from ..client import ClientError

    directory = Path(a.dir).resolve()
    c = _client(directory, a.server, a.password)
    text = " ".join(a.message).strip() or (sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not text:
        print("opentrain run: nothing to do (give a message or pipe one in)", file=sys.stderr)
        return 2
    interactive = sys.stdin.isatty() and not a.yes
    meta: dict[str, Any] = {}
    if a.yes:
        meta = {"unattended": True, "auto_approve": ["*"]}
    elif not sys.stdin.isatty():
        meta = {"unattended": True}
    if a.session:
        sid = a.session
    else:
        body: dict[str, Any] = {"metadata": meta}
        if a.agent:
            body["agent"] = a.agent
        if a.model:
            body["model"] = a.model
        sid = c.post("/api/session", body)["id"]
    connected = threading.Event()
    done = threading.Event()
    errors: list[str] = []

    from ..core.store import merge

    parts: dict[str, dict[str, Any]] = {}
    shown_jobs: dict[str, str] = {}
    follow: dict[str, Any] = {"idle_since": None}

    def listen() -> None:
        busy_seen = False
        try:
            for ev in c.events():
                if ev["type"] == "server.connected":
                    connected.set()
                    continue
                d = ev.get("data") or {}
                if d.get("session_id") != sid and d.get("id") != sid:
                    continue
                if a.json:
                    print(json.dumps(ev, default=str), flush=True)
                t = ev["type"]
                if t == "part.delta" and not a.json:
                    sys.stdout.write(d.get("delta", ""))
                    sys.stdout.flush()
                if t in ("part.created", "part.updated"):
                    parts[d["id"]] = merge(parts.get(d["id"], {}), d)
                if t == "part.updated" and (d.get("tool") or {}).get("state") in ("completed", "error") and not a.json:
                    tool = parts[d["id"]].get("tool") or {}
                    mark = "✓" if tool["state"] == "completed" else "✗"
                    print(f"\n  {mark} {tool.get('name', '')}  {tool.get('title') or _subject(tool.get('input'))}"
                          + (f"  {tool.get('error')}" if tool["state"] == "error" else ""), file=sys.stderr)
                elif t == "permission.created" and d.get("status") == "pending":
                    _answer_permission(c, sid, d, interactive)
                elif t == "question.created" and d.get("status") == "pending" and not a.json:
                    _answer_question(c, sid, d, interactive)
                elif t == "proposal.created" and not a.json:
                    _show_proposal(c, sid, d, interactive)
                elif t == "proposal.updated" and not a.json and d.get("status") in ("approved", "skipped", "changed"):
                    print(f"  → {d['status']}" + (f": {d['message']}" if d.get("message") else ""), file=sys.stderr)
                elif t == "message.updated" and d.get("error"):
                    errors.append(str(d["error"]))
                elif t in ("job.created", "job.updated") and not a.json:
                    _show_job(d, shown_jobs)
                elif t == "session.status":
                    if d.get("status") == "busy":
                        busy_seen = True
                        follow["idle_since"] = None
                    elif d.get("status") == "idle" and busy_seen:
                        if a.detach or not _active_jobs(c, sid):
                            done.set()
                            return
                        # the agent started work that outlives its turn: the server wakes it when the
                        # work ends, so keep following until nothing of this session is running
                        busy_seen = False
                        follow["idle_since"] = time.time()
                        if not a.json:
                            print("\n  … waiting for the job; the agent continues when it ends (ctrl+c to leave it "
                                  "running)", file=sys.stderr)
        except Exception as exc:  # the stream dropped: fall back to waiting
            errors.append(f"event stream: {exc}")
            connected.set()

    th = threading.Thread(target=listen, daemon=True)
    th.start()
    connected.wait(10)
    try:
        if text.startswith("/") and " " not in text.split("\n")[0][:1]:
            name, _, args = text[1:].partition(" ")
            c.post(f"/api/session/{sid}/command", {"name": name, "arguments": args})
        else:
            c.post(f"/api/session/{sid}/prompt", {"text": text})
        while not done.wait(1.0):
            since, now = follow["idle_since"], time.time()
            if since and now - since > FOLLOW_MAX:
                print("\n  stopped following after 24 hours; the job keeps running on the server", file=sys.stderr)
                break
            if since and now - since > 30 and now - follow.get("checked", 0) > 15:
                follow["checked"] = now
                quiet = not _active_jobs(c, sid) and c.get(f"/api/session/{sid}").get("status") == "idle"
                follow["quiet"] = follow.get("quiet", 0) + 1 if quiet else 0
                if follow["quiet"] >= 2:  # twice in a row: the jobs ended and nothing woke the agent
                    break
            if not th.is_alive():
                c.post(f"/api/session/{sid}/wait", {"timeout": 24 * 3600})
                break
    except KeyboardInterrupt:
        c.post(f"/api/session/{sid}/interrupt")
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ClientError as exc:
        print(f"opentrain run: {exc.message}", file=sys.stderr)
        return 1
    msgs = c.get(f"/api/session/{sid}/message")
    last = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    print()
    if not a.json:
        s = c.get(f"/api/session/{sid}")
        print(f"\n[{sid} · {s.get('agent')} · ${s.get('cost') or 0:.4f}]", file=sys.stderr)
    return 1 if (last and last.get("error") and last.get("finish") != "interrupted") else 0


FOLLOW_MAX = 24 * 3600


def _subject(inp: Any) -> str:
    """What a tool call is about, when the tool gives no title: its dataset, model, query or id."""
    if not isinstance(inp, dict):
        return ""
    for k in ("name", "model", "hf", "kaggle", "query", "path", "version", "job_id", "row_id"):
        v = inp.get(k)
        if isinstance(v, str) and v:
            extra = f" ({inp['method']})" if k == "name" and isinstance(inp.get("method"), str) else ""
            return v[:60] + extra
    return ""


def _active_jobs(c, sid: str) -> list[dict[str, Any]]:
    try:
        return c.get("/api/job", session=sid, active="1")
    except Exception:  # noqa: BLE001 - an unreachable server ends the follow, not the run
        return []


def _show_job(j: dict[str, Any], shown: dict[str, str]) -> None:
    """One line per job state change: started, then how it ended."""
    if not j.get("id") or not j.get("state"):
        return
    state = j["state"]
    if shown.get(j["id"]) == state:
        return
    shown[j["id"]] = state
    if j.get("title"):
        shown[j["id"] + ":title"] = j["title"]
    title = shown.get(j["id"] + ":title") or j.get("kind") or "job"
    if state == "running":
        print(f"\n  ▸ {title} started", file=sys.stderr)
    else:
        cost = j.get("cost_usd")
        tail = j.get("error") or j.get("detail") or ""
        print(f"\n  ▸ {title} {state}" + (f" · ${cost:.2f}" if isinstance(cost, (int, float)) else "")
              + (f" · {tail}" if tail else ""), file=sys.stderr)


def _answer_question(c, sid: str, q: dict[str, Any], interactive: bool) -> None:
    """The agent's question with numbered options; on a terminal, answer by number or in words."""
    answers: list[list[str]] = []
    for item in q.get("questions") or []:
        print(f"\n? {item.get('question')}", file=sys.stderr)
        opts = item.get("options") or []
        for i, o in enumerate(opts, 1):
            print(f"  {i}. {o}", file=sys.stderr)
        if not interactive:
            continue
        try:
            with open("/dev/tty") as tty:
                sys.stderr.write("  answer (number or words): ")
                sys.stderr.flush()
                ans = tty.readline().strip()
        except OSError:
            return
        pick = [opts[int(x) - 1] for x in ans.replace(",", " ").split() if x.isdigit() and 0 < int(x) <= len(opts)]
        answers.append(pick or [ans or (opts[0] if opts else "")])
    if interactive and answers:
        try:
            c.post(f"/api/session/{sid}/question/{q['id']}/reply", {"answers": answers})
        except Exception as exc:  # noqa: BLE001 - answered elsewhere (the web app) meanwhile
            print(f"  ({exc})", file=sys.stderr)


def _show_proposal(c, sid: str, p: dict[str, Any], interactive: bool) -> None:
    """A proposed step, as a card; on a terminal, approve, skip or change it right here."""
    cost = f"~${p['cost_usd']:.2f}" if p.get("cost_usd") else "free"
    mins = f" · {p['minutes']:g} min" if isinstance(p.get("minutes"), (int, float)) else ""
    needs = f" · needs {', '.join(p['needs'])}" if p.get("needs") else ""
    print(f"\n┌ next step: {p.get('title')}\n│ {cost}{mins}{needs}" + (f"\n│ {p['why']}" if p.get("why") else ""),
          file=sys.stderr)
    if p.get("status") != "pending" or not interactive:
        return
    try:
        with open("/dev/tty") as tty:
            sys.stderr.write("└ [a]pprove / [s]kip / or type a change: ")
            sys.stderr.flush()
            ans = tty.readline().strip()
    except OSError:
        return
    if ans.lower() in ("a", "approve", "y", "yes", ""):
        body: dict[str, Any] = {"reply": "approve"}
    elif ans.lower() in ("s", "skip", "n", "no"):
        body = {"reply": "skip"}
    else:
        body = {"reply": "change", "message": ans}
    try:
        c.post(f"/api/session/{sid}/proposal/{p['id']}/reply", body)
    except Exception as exc:  # noqa: BLE001 - settled elsewhere (the web app) meanwhile
        print(f"  ({exc})", file=sys.stderr)


def _answer_permission(c, sid: str, req: dict[str, Any], interactive: bool) -> None:
    if not interactive:
        return  # unattended: the server denies asks itself
    from ..tui.app import describe_permission

    print("\n" + describe_permission(req).plain, file=sys.stderr)
    try:
        with open("/dev/tty") as tty:
            sys.stderr.write("allow? [y]es once / [a]lways / [n]o: ")
            sys.stderr.flush()
            ans = tty.readline().strip().lower()
    except OSError:
        ans = "n"
    reply = {"y": "once", "yes": "once", "a": "always", "always": "always"}.get(ans, "reject")
    c.post(f"/api/session/{sid}/permission/{req['id']}/reply", {"reply": reply})


# ---------------------------------------------------------------------- server verbs


def cmd_serve(a: argparse.Namespace) -> int:
    from ..server import run

    if a.password:
        os.environ["OPENTRAIN_SERVER_PASSWORD"] = a.password
    if a.host not in ("127.0.0.1", "localhost", "::1") and not (a.password or os.environ.get("OPENTRAIN_SERVER_PASSWORD")):
        print("opentrain serve: set --password (or OPENTRAIN_SERVER_PASSWORD) to listen beyond this machine",
              file=sys.stderr)
        return 2
    return run(a.host, a.port, register=a.register, default=Path(a.dir).resolve() if a.dir else None)


def cmd_web(a: argparse.Namespace) -> int:
    import urllib.parse
    import webbrowser

    directory = Path(a.dir).resolve()
    c = _client(directory, a.server, a.password)
    link = c.post("/api/pair")["url"] + "?next=" + urllib.parse.quote(f"/?directory={directory}")
    base = link.split("/auth/", 1)[0]
    print(f"OpenTrain: {link}\n(signs this browser in for 30 days; after that just open {base})")
    if not a.no_open:
        webbrowser.open(link)
    return 0


def cmd_pair(a: argparse.Namespace) -> int:
    import urllib.parse

    directory = Path(a.dir).resolve()
    c = _client(directory, a.server, a.password)
    print(c.post("/api/pair")["url"] + "?next=" + urllib.parse.quote(f"/?directory={directory}"))
    return 0


def cmd_stop(a: argparse.Namespace) -> int:
    from ..server import service

    print("stopped" if service.stop() else "no server was running")
    return 0


def cmd_mcp(a: argparse.Namespace) -> int:
    """OpenTrain's tools for Claude Code (or any MCP client), against this project's server."""
    import os
    import shutil
    import subprocess

    directory = Path(a.dir).resolve()
    if a.install:
        claude = shutil.which("claude")
        if not claude:
            print("Claude Code is not installed (https://claude.com/claude-code)", file=sys.stderr)
            return 1
        cmd = [claude, "mcp", "add", "--scope", a.scope, "opentrain", "--", "opentrain", "mcp", "--dir", str(directory)]
        r = subprocess.run(cmd, cwd=directory)
        if r.returncode == 0:
            print("Added. In Claude Code, ask for example: \"use opentrain to find the cheapest model that handles "
                  "my data\". Watch it in the browser with: opentrain web")
        return r.returncode
    from ..core.bridge import Bridge

    os.environ["OPENTRAIN_DIRECTORY"] = str(directory)
    Bridge(outside=True).serve()
    return 0


def cmd_attach(a: argparse.Namespace) -> int:
    return _tui(Path(a.dir).resolve(), a.url, a.password, a.session, None)


def cmd_export(a: argparse.Namespace) -> int:
    from ..export import export

    directory = Path(a.dir).resolve()
    out = Path(a.out).resolve() if a.out else directory / "opentrain-export"
    m = export(directory / ".opentrain", out, include_heldout=False)
    print(f"Wrote {out}")
    for k in ("tasks", "examples"):
        if k in m:
            print(f"  {k}: {m[k] if not isinstance(m[k], list) else len(m[k])}")
    return 0


def cmd_model(a: argparse.Namespace) -> int:
    from ..cli import debug_main

    mapping = {"serve": ["serve"], "connect": ["connect"], "ls": ["model", "ls"], "use": ["model", "use"]}
    return debug_main(mapping[a.what] + a.rest)


def cmd_login(a: argparse.Namespace) -> int:
    """Sign in to a compute provider in the browser (no key to paste)."""
    from ..backends import runpod_backend

    try:
        hint = runpod_backend.browser_login(open_browser=not a.no_open)
    except (runpod_backend.RunPodError, OSError) as exc:
        print(f"RunPod sign-in failed: {exc}", file=sys.stderr)
        return 1
    print(f"Signed in to RunPod (key {hint} kept in OpenTrain's auth store). Trainer runpod is ready.")
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    from ..cli import debug_main

    return debug_main(["watch"] + a.rest)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opentrain", description="OpenTrain: the open harness for training models. "
                                "With no command, opens the terminal app in the current directory.")
    p.add_argument("--version", action="version", version=f"opentrain {__version__}")
    sub = p.add_subparsers(dest="verb")

    def server_opts(sp, with_dir: bool = True):
        if with_dir:
            sp.add_argument("--dir", default=".", help="project directory (default: here)")
        sp.add_argument("--server", default=None, help="a server URL (default: your background server)")
        sp.add_argument("--password", default=None, help="that server's password (or OPENTRAIN_SERVER_PASSWORD)")

    r = sub.add_parser("run", help="one instruction to the engine, no interface")
    r.add_argument("message", nargs="*", help='e.g. "mine my sessions" or "/status"')
    r.add_argument("--agent", help="train (default) or plan, or any agent")
    r.add_argument("--model", help="provider/model for the engine")
    r.add_argument("--session", help="continue this session")
    r.add_argument("--yes", action="store_true", help="approve every ask (spend, upload, run) without prompting")
    r.add_argument("--json", action="store_true", help="print events as JSON lines")
    r.add_argument("--detach", action="store_true",
                   help="return when the agent's turn ends, even if a job it started is still running")
    server_opts(r)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve", help="run the server in the foreground")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7477)
    s.add_argument("--password", default=None)
    s.add_argument("--register", action="store_true", help="become this user's background server")
    s.add_argument("--dir", default=None, help="default project for clients that do not name one")
    s.set_defaults(fn=cmd_serve)

    w = sub.add_parser("web", help="open the dashboard in a browser")
    w.add_argument("--no-open", action="store_true", help="print the link only")
    server_opts(w)
    w.set_defaults(fn=cmd_web)

    mc = sub.add_parser("mcp", help="OpenTrain's tools for Claude Code or any MCP client (stdio)")
    mc.add_argument("--install", action="store_true", help="add it to Claude Code (claude mcp add) and exit")
    mc.add_argument("--scope", default="project", choices=["local", "project", "user"],
                    help="Claude Code config scope for --install (default: project, in .mcp.json)")
    mc.add_argument("--dir", default=".", help="the project (default: here)")
    mc.set_defaults(fn=cmd_mcp)

    lg = sub.add_parser("login", help="sign in to a GPU provider in the browser (runpod)")
    lg.add_argument("provider", choices=["runpod"])
    lg.add_argument("--no-open", action="store_true", help="print the link only")
    lg.set_defaults(fn=cmd_login)

    at = sub.add_parser("attach", help="the terminal app against a server elsewhere")
    at.add_argument("url")
    at.add_argument("--password", default=None)
    at.add_argument("--session", default=None)
    at.add_argument("--dir", default=".", help="project directory on that server")
    at.set_defaults(fn=cmd_attach)

    pa = sub.add_parser("pair", help="print a one-time link that signs a browser in")
    server_opts(pa)
    pa.set_defaults(fn=cmd_pair)

    st = sub.add_parser("stop", help="stop the background server (running jobs on remote trainers go on)")
    st.set_defaults(fn=cmd_stop)

    ex = sub.add_parser("export", help="tasks and training set in open formats")
    ex.add_argument("--dir", default=".")
    ex.add_argument("--out", default=None)
    ex.set_defaults(fn=cmd_export)

    wa = sub.add_parser("watch", help="the unattended loop (see opentrain debug watch --help)")
    wa.add_argument("rest", nargs=argparse.REMAINDER)
    wa.set_defaults(fn=cmd_watch)

    m = sub.add_parser("model", help="serve a trained version, list or switch versions, connect a coding agent")
    m.add_argument("what", choices=["serve", "ls", "use", "connect"])
    m.add_argument("rest", nargs=argparse.REMAINDER)
    m.set_defaults(fn=cmd_model)

    sub.add_parser("debug", help="every pipeline step on its own: opentrain debug --help")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "debug":
        from ..cli import debug_main

        return debug_main(argv[1:])
    if argv and argv[0] == "help":
        parser().print_help()
        return 0
    first = argv[0] if argv else None
    if first is None or (first not in VERBS and not first.startswith("-") and Path(first).is_dir()) \
            or (first and first.startswith("--") and first in ("--session", "--prompt", "--server", "--password")):
        tui = argparse.ArgumentParser(prog="opentrain")
        tui.add_argument("dir", nargs="?", default=".")
        tui.add_argument("--session", default=None)
        tui.add_argument("--prompt", default=None, help="send this as the first message")
        tui.add_argument("--server", default=None)
        tui.add_argument("--password", default=None)
        a = tui.parse_args(argv)
        if not sys.stdout.isatty():
            parser().print_help()
            return 0
        return _tui(Path(a.dir).resolve(), a.server, a.password, a.session, a.prompt)
    a = parser().parse_args(argv)
    if not getattr(a, "fn", None):
        parser().print_help()
        return 0
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        return 130
