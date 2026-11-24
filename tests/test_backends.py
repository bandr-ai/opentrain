from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from opentrain.backends import BACKENDS, JobStore, get_backend, list_backends
from opentrain.backends import tinker as tk
from opentrain.backends.fake import SIMULATED
from opentrain.plugs import TrainerBackend, TrainSpec
from opentrain.registry import Registry
from opentrain.serving import (
    Handoff,
    ServingError,
    client_config,
    endpoint_for,
    live_version,
    make_server,
    use_version,
)

ROW = {
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
        {"role": "assistant", "content": "done"},
    ],
    "tools": [{"type": "function", "function": {"name": "bash", "description": "run", "parameters": {
        "type": "object", "properties": {"cmd": {"type": "string"}}}}}],
}


@pytest.fixture
def data(tmp_path: Path) -> Path:
    p = tmp_path / "train.jsonl"
    p.write_text("\n".join(json.dumps(ROW) for _ in range(3)) + "\n")
    return p


def spec(data: Path, **kw) -> TrainSpec:
    return TrainSpec(base_model="Qwen/Qwen3-8B", data=data, **kw)


# ---------------------------------------------------------------- registry and fake backend


def test_registry_lists_every_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENTRAIN_FAKE_BACKEND", raising=False)
    rows = {name: (ok, why) for name, ok, why in list_backends(tmp_path)}
    assert set(rows) == set(BACKENDS) == {"tinker", "local", "fireworks", "fake"}
    assert rows["fake"][0] is False and "OPENTRAIN_FAKE_BACKEND" in rows["fake"][1]
    assert rows["fireworks"][0] is False and "not implemented" in rows["fireworks"][1]
    assert all(isinstance(get_backend(n, tmp_path), TrainerBackend) for n in BACKENDS)
    with pytest.raises(KeyError):
        get_backend("nope", tmp_path)


def test_fake_refuses_without_env(tmp_path, data, monkeypatch):
    monkeypatch.delenv("OPENTRAIN_FAKE_BACKEND", raising=False)
    fake = get_backend("fake", tmp_path)
    with pytest.raises(RuntimeError, match="OPENTRAIN_FAKE_BACKEND"):
        fake.launch(spec(data))
    with pytest.raises(RuntimeError):
        fake.estimate(spec(data))
    assert not (tmp_path / "jobs").exists()


def test_fake_lifecycle_persists_across_instances(tmp_path, data, monkeypatch):
    monkeypatch.setenv("OPENTRAIN_FAKE_BACKEND", "1")
    jid = get_backend("fake", tmp_path).launch(spec(data))
    st = get_backend("fake", tmp_path).status(jid)  # a fresh instance, as a later CLI call would make
    assert st.state == "succeeded" and st.done
    assert st.checkpoint == f"fake://{jid}"
    assert SIMULATED in st.detail and any(SIMULATED in line for line in st.logs)
    assert json.loads((tmp_path / "jobs" / f"{jid}.json").read_text())["spec"]["base_model"] == "Qwen/Qwen3-8B"


def test_fake_hold_then_cancel(tmp_path, data, monkeypatch):
    monkeypatch.setenv("OPENTRAIN_FAKE_BACKEND", "1")
    fake = get_backend("fake", tmp_path)
    jid = fake.launch(spec(data, extra={"hold": True}))
    assert fake.status(jid).state == "running"
    fake.cancel(jid)
    assert (tmp_path / "jobs" / f"{jid}.cancel").exists()
    st = fake.status(jid)
    assert st.state == "cancelled" and SIMULATED in st.detail


# ---------------------------------------------------------------- job store


def test_job_store_cancel_flag_and_dead_worker(tmp_path, data):
    store = JobStore(tmp_path)
    job = store.new("tinker", spec(data))
    jid = job["job_id"]
    assert store.load(jid[:12])["job_id"] == jid  # prefix lookup, like run ids
    assert not store.cancel_requested(jid)
    store.request_cancel(jid)
    assert store.cancel_requested(jid)

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    other = store.new("local", spec(data))["job_id"]
    store.update(other, state="running", pid=gone.pid)
    st = store.status(other)
    assert st.state == "failed" and "exited without reporting" in st.detail
    store.update(jid, state="running", pid=gone.pid)
    assert store.status(jid).state == "cancelled"  # it was asked to stop, and it did


def test_worker_main_records_crash(tmp_path, data):
    from opentrain.backends import worker_main

    store = JobStore(tmp_path)
    jid = store.new("tinker", spec(data))["job_id"]

    def boom(store, job):
        raise RuntimeError("no key")

    with pytest.raises(RuntimeError):
        worker_main([str(store.path(jid))], boom)
    st = store.status(jid)
    assert st.state == "failed" and "no key" in st.detail


# ---------------------------------------------------------------- tinker and local availability, data


def _block(monkeypatch, *mods):
    for m in mods:
        monkeypatch.setitem(sys.modules, m, None)


def test_tinker_unavailable_without_sdk_or_key(tmp_path, monkeypatch):
    _block(monkeypatch, "tinker")
    ok, why = get_backend("tinker", tmp_path).available()
    assert not ok and "opentrain[tinker]" in why
    monkeypatch.setattr(tk, "missing_modules", lambda *names: [])
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    ok, why = get_backend("tinker", tmp_path).available()
    assert not ok and "TINKER_API_KEY" in why


def test_tinker_launch_refuses_when_unavailable(tmp_path, data, monkeypatch):
    _block(monkeypatch, "tinker")
    with pytest.raises(RuntimeError, match="unavailable"):
        get_backend("tinker", tmp_path).launch(spec(data))


def test_local_unavailable_without_torch(tmp_path, monkeypatch):
    _block(monkeypatch, "torch")
    local = get_backend("local", tmp_path)
    ok, why = local.available()
    assert not ok and "torch" in why and "opentrain[local]" in why
    assert local.estimate(spec(Path("x"))) == 0.0


def test_tinker_estimate_uses_documented_price(tmp_path, data):
    t = get_backend("tinker", tmp_path)
    chars = sum(len(line) for line in data.read_text().splitlines())
    est = t.estimate(spec(data, epochs=2))
    assert est == pytest.approx(chars / tk.CHARS_PER_TOKEN * 2 * 0.44 / 1e6, rel=0.05)
    assert t.estimate(spec(data, extra={"usd_per_mtok": 10.0})) > est
    with pytest.raises(ValueError, match="no Tinker training price"):
        t.estimate(TrainSpec(base_model="someone/unknown", data=data))


def test_split_example_unwraps_tools_and_system():
    system, tools, msgs = tk.split_example(ROW)
    assert system == "be brief"
    assert tools == [{"name": "bash", "description": "run",
                      "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert tk.tool_names_by_call_id(msgs) == {"c1": "bash"}
    hf = tk.hf_messages(ROW["messages"])
    assert hf[2]["tool_calls"][0]["function"]["arguments"] == {"cmd": "ls"} and hf[2]["content"] == ""


class CharTokenizer:
    """A prefix-stable chat template over characters, standing in for an HF tokenizer."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        text = "".join(f"<{m['role']}>{m.get('content') or ''}{json.dumps(m.get('tool_calls') or '')}</>"
                       for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_hf_fallback_weights_only_assistant_tokens():
    msgs = tk.hf_messages(ROW["messages"])
    tokens, weights = tk.hf_supervised_tokens(CharTokenizer(), msgs, ROW["tools"])
    text = "".join(chr(t) for t in tokens)
    trained = "".join(c for c, w in zip(text, weights) if w)
    assert "done" in trained and "fix it" not in trained and "a.py" not in trained and '"cmd"' in trained
    inp, target, w = tk.datum_parts(tokens, weights, max_tokens=10_000)
    assert len(inp) == len(target) == len(w) == len(tokens) - 1
    assert sum(w) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no assistant tokens"):
        tk.datum_parts(tokens, weights, max_tokens=5)


def test_hf_fallback_refuses_unstable_template():
    class Rewriting(CharTokenizer):
        def apply_chat_template(self, messages, **kw):
            return super().apply_chat_template(messages, **kw) + str(len(messages))

    with pytest.raises(ValueError, match="prefix-stable"):
        tk.hf_supervised_tokens(Rewriting(), tk.hf_messages(ROW["messages"]), None)


# ---------------------------------------------------------------- serving


class Upstream:
    """A tiny OpenAI-compatible server that records what it was asked."""

    def __init__(self, name: str, fail: int | None = None):
        self.name, self.fail, self.seen = name, fail, []
        up = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                up.seen.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization"),
                                "x-api-key": self.headers.get("x-api-key")})
                if up.fail:
                    data = b'{"error": {"message": "down"}}'
                    self.send_response(up.fail)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for word in ("hel", "lo"):
                        chunk = {"model": body["model"], "choices": [{"delta": {"content": word}}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                out = {"model": body["model"], "choices": [{"message": {"role": "assistant",
                       "content": f"from {up.name}"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("UPSTREAM_KEY", "sekret")
    a, b, down = Upstream("a"), Upstream("b"), Upstream("down", fail=503)
    reg = Registry(tmp_path)
    reg.add(run_id=None, base_model="m", backend="local", checkpoint="/x/a",
            serving={"base_url": a.url, "model": "model-a", "api_key_env": "UPSTREAM_KEY"})
    reg.add(run_id=None, base_model="m", backend="local", checkpoint="/x/b",
            serving={"base_url": b.url, "model": "model-b", "anthropic_base_url": b.url[:-3]})
    reg.add(run_id=None, base_model="m", backend="local", checkpoint="/x/c",
            serving={"base_url": down.url, "model": "model-c"})
    reg.promote("v1", "test")
    yield tmp_path, a, b, down
    for u in (a, b, down):
        u.close()


def _proxy(state, handoff=None):
    server = make_server(state, port=0, handoff=handoff)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_serve_maps_names_and_hot_switches(served):
    state, a, b, _ = served
    server, base = _proxy(state)
    try:
        with urllib.request.urlopen(f"{base}/v1/models") as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        assert ids == ["opentrain/current", "opentrain/v1", "opentrain/v2", "opentrain/v3"]

        msg = {"model": "opentrain/current", "messages": [{"role": "user", "content": "hi"}]}
        status, headers, body = _post(f"{base}/v1/chat/completions", msg)
        out = json.loads(body)
        assert status == 200 and out["model"] == "opentrain/current" and "from a" in json.dumps(out)
        assert a.seen[-1]["body"]["model"] == "model-a" and a.seen[-1]["auth"] == "Bearer sekret"
        assert headers["X-OpenTrain-Version"] == "v1"

        status, _, body = _post(f"{base}/v1/opentrain/switch", {"version": "v2"})
        assert status == 200 and json.loads(body)["version"] == "v2" and live_version(state) == "v2"
        _, headers, body = _post(f"{base}/v1/chat/completions", msg)
        assert "from b" in body.decode() and headers["X-OpenTrain-Version"] == "v2"
        assert b.seen[-1]["body"]["model"] == "model-b"

        _, _, body = _post(f"{base}/v1/chat/completions", {**msg, "model": "opentrain/v1"})
        assert "from a" in body.decode()  # pinned versions bypass the live pointer

        use_version(state, "v1")  # what `opentrain model use v1` does from another process
        _, _, body = _post(f"{base}/v1/chat/completions", msg)
        assert "from a" in body.decode()

        assert _post(f"{base}/v1/chat/completions", {**msg, "model": "gpt-x"})[0] == 404
        assert _post(f"{base}/v1/opentrain/switch", {"version": "v9"})[0] == 404
    finally:
        server.shutdown()
        server.server_close()

    log = [json.loads(x) for x in (state / "serving" / "requests.jsonl").read_text().splitlines()]
    assert [e["status"] for e in log] == [200, 200, 200, 200, 404]
    assert [e.get("version") for e in log[:4]] == ["v1", "v2", "v1", "v1"]
    assert log[0]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2}
    assert all("latency_s" in e for e in log) and log[0]["handoff"] is False


def test_serve_streams_through(served):
    state, a, _, _ = served
    server, base = _proxy(state)
    try:
        req = urllib.request.Request(f"{base}/v1/chat/completions", method="POST",
                                     data=json.dumps({"model": "opentrain/current", "stream": True,
                                                      "messages": []}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.headers["Content-Type"].startswith("text/event-stream")
            events = [line for line in r.read().decode().splitlines() if line.startswith("data:")]
    finally:
        server.shutdown()
        server.server_close()
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(e[5:]) for e in events[:-1]]
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == "hello"
    assert {c["model"] for c in chunks} == {"opentrain/current"}
    entry = json.loads((state / "serving" / "requests.jsonl").read_text().splitlines()[-1])
    assert entry["stream"] is True and entry["status"] == 200 and entry["ttfb_s"] is not None


def test_serve_hands_off_to_frontier_when_specialist_fails(served):
    state, a, _, down = served
    use_version(state, "v3")
    msg = {"model": "opentrain/current", "messages": []}
    server, base = _proxy(state)
    try:
        assert _post(f"{base}/v1/chat/completions", msg)[0] == 503  # no handoff: the error goes back
    finally:
        server.shutdown()
        server.server_close()
    server, base = _proxy(state, Handoff(a.url, "frontier-model", "UPSTREAM_KEY"))
    try:
        status, headers, body = _post(f"{base}/v1/chat/completions", msg)
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200 and headers["X-OpenTrain-Handoff"] == "frontier"
    assert json.loads(body)["model"] == "frontier-model"  # not passed off as the specialist's answer
    assert a.seen[-1]["body"]["model"] == "frontier-model"
    entry = json.loads((state / "serving" / "requests.jsonl").read_text().splitlines()[-1])
    assert entry["handoff"] is True and "503" in entry["specialist_error"]


def test_serve_anthropic_messages_only_where_native(served):
    state, _, b, _ = served
    server, base = _proxy(state)
    try:
        status, _, body = _post(f"{base}/v1/messages", {"model": "opentrain/v1", "messages": []})
        assert status == 501 and "Anthropic" in body.decode()
        status, _, _ = _post(f"{base}/v1/messages", {"model": "opentrain/v2", "messages": []})
        assert status == 200 and b.seen[-1]["path"] == "/v1/messages"
    finally:
        server.shutdown()
        server.server_close()


def test_endpoint_for_each_backend(tmp_path):
    reg = Registry(tmp_path)
    t = reg.add(run_id=None, base_model="Qwen/Qwen3-8B", backend="tinker",
                checkpoint="tinker://run:train:0/sampler_weights/job")
    ep = endpoint_for(t)
    assert ep.base_url == tk.OPENAI_BASE and ep.model == t.checkpoint and ep.api_key_env == "TINKER_API_KEY"
    loc = reg.add(run_id=None, base_model="m", backend="local", checkpoint="/a")
    assert endpoint_for(loc).base_url.startswith("http://127.0.0.1:")
    fake = reg.add(run_id=None, base_model="m", backend="fake", checkpoint="fake://x")
    with pytest.raises(ServingError, match="simulated"):
        endpoint_for(fake)


def test_client_configs():
    pi = client_config("pi", port=9000)
    assert json.loads(pi[pi.index("{"):pi.rindex("}") + 1])["providers"]["opentrain"]["baseUrl"] == \
        "http://127.0.0.1:9000/v1"
    oc = client_config("opencode")
    assert '"@ai-sdk/openai-compatible"' in oc and "opentrain/current" in oc
    cl = client_config("claude")
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8377" in cl and "Tinker" in cl and "does not support" in cl
    with pytest.raises(ValueError):
        client_config("vim")


# ---------------------------------------------------------------- commands


def _run(argv, capsys):
    from opentrain.commands import train

    p = argparse.ArgumentParser()
    train.register(p.add_subparsers(dest="cmd", required=True))
    args = p.parse_args(argv)
    code = args.fn(args)
    return code, capsys.readouterr()


def test_commands(tmp_path, data, monkeypatch, capsys):
    monkeypatch.setenv("OPENTRAIN_FAKE_BACKEND", "1")
    st = ["--state", str(tmp_path)]
    code, out = _run(["backends", *st], capsys)
    assert code == 0 and "fireworks" in out.out and "tinker" in out.out

    state = tmp_path / ".opentrain"
    jid = get_backend("fake", state).launch(spec(data, extra={"hold": True}))
    code, out = _run(["job", "ps", *st], capsys)
    assert code == 0 and jid in out.out and "running" in out.out
    code, out = _run(["job", "cancel", jid, *st], capsys)
    assert code == 0 and "Cancel requested" in out.out
    code, out = _run(["job", "logs", jid, *st], capsys)
    assert "cancel requested" in out.out

    reg = Registry(state)
    reg.add(run_id=None, base_model="Qwen/Qwen3-8B", backend="tinker", checkpoint="tinker://r/sampler_weights/a")
    code, out = _run(["model", "use", "v1", *st], capsys)
    assert code == 0 and live_version(state) == "v1"
    code, out = _run(["model", "ls", *st], capsys)
    assert "live" in out.out and "tinker://r/sampler_weights/a" in out.out
    assert _run(["model", "use", "v7", *st], capsys)[0] == 1
    code, out = _run(["connect", "opencode", "--port", "9001", *st], capsys)
    assert "http://127.0.0.1:9001/v1" in out.out


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def _stub_tinker(calls: list, cancel_after: int | None = None, store=None, jid=None):
    """Just enough of the tinker SDK surface the worker uses (names checked against tinker 0.40)."""
    import types as pytypes

    class ModelInput:
        def __init__(self, tokens):
            self.tokens = tokens
            self.length = len(tokens)

        @classmethod
        def from_ints(cls, tokens):
            return cls(tokens)

    class Training:
        def get_console_url(self):
            return "https://console/run"

        def get_tokenizer(self):
            return CharTokenizer()

        def forward_backward(self, batch, loss_fn):
            calls.append(("fb", len(batch), loss_fn))
            if cancel_after is not None and sum(c[0] == "fb" for c in calls) == cancel_after:
                store.request_cancel(jid)
            return _Future(pytypes.SimpleNamespace(metrics={"loss:sum": 1.5}))

        def optim_step(self, params):
            calls.append(("opt", params["learning_rate"]))
            return _Future(None)

        def save_weights_for_sampler(self, name):
            calls.append(("save", name))
            return _Future(pytypes.SimpleNamespace(path=f"tinker://run:train:0/sampler_weights/{name}"))

    class Service:
        def create_lora_training_client(self, base_model, rank):
            calls.append(("create", base_model, rank))
            return _Future(Training())

    mod = pytypes.ModuleType("tinker")
    mod.ServiceClient = Service
    mod.ModelInput = ModelInput
    mod.Datum = lambda model_input, loss_fn_inputs: pytypes.SimpleNamespace(
        model_input=model_input, loss_fn_inputs=loss_fn_inputs)
    mod.AdamParams = lambda learning_rate: {"learning_rate": learning_rate}
    return mod


def test_tinker_worker_loop_reports_progress_cost_and_checkpoint(tmp_path, data, monkeypatch):
    from opentrain.backends import tinker_worker

    calls: list = []
    monkeypatch.setitem(sys.modules, "tinker", _stub_tinker(calls))
    monkeypatch.setattr(tinker_worker, "missing_modules", lambda *n: list(n))  # no cookbook: HF fallback
    store = JobStore(tmp_path)
    jid = store.new("tinker", spec(data, epochs=2, batch_size=2), usd_per_mtok=0.44)["job_id"]
    tinker_worker.run(store, store.load(jid))
    st = store.status(jid)
    assert st.state == "succeeded" and st.progress == 1.0
    assert st.checkpoint == f"tinker://run:train:0/sampler_weights/{jid}"
    job = store.load(jid)
    assert [c[1] for c in calls if c[0] == "fb"] == [2, 1, 2, 1]  # 3 examples, batch 2, 2 epochs
    assert job["cost_usd"] == pytest.approx(round(job["tokens"] * 0.44 / 1e6, 4))
    assert ("create", "Qwen/Qwen3-8B", 32) in calls and "estimate" in st.detail


def test_tinker_worker_stops_on_cancel_flag(tmp_path, data, monkeypatch):
    from opentrain.backends import tinker_worker

    calls: list = []
    store = JobStore(tmp_path)
    jid = store.new("tinker", spec(data, epochs=3, batch_size=1), usd_per_mtok=0.44)["job_id"]
    monkeypatch.setitem(sys.modules, "tinker", _stub_tinker(calls, cancel_after=2, store=store, jid=jid))
    monkeypatch.setattr(tinker_worker, "missing_modules", lambda *n: list(n))
    tinker_worker.run(store, store.load(jid))
    st = store.status(jid)
    assert st.state == "cancelled" and st.checkpoint is None and "step 2/9" in st.detail
    assert sum(c[0] == "fb" for c in calls) == 2 and not any(c[0] == "save" for c in calls)
