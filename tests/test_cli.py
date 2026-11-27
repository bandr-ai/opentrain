import json
import subprocess
import sys
import urllib.request
from http.cookiejar import CookieJar

from opentrain.core.providers.scripted import ScriptedProvider


def _run(live, *args, input_text=None):
    return subprocess.run([sys.executable, "-m", "opentrain", *args, "--server", live.url, "--password", live.password,
                           "--dir", str(live.proj)], capture_output=True, text=True, timeout=60,
                          input=input_text if input_text is not None else "")


def test_run_prints_the_answer_and_tool_lines(live):
    ScriptedProvider.register("cli1", [["Checking.", {"tool": "status", "input": {}}], ["Nothing is live yet."]])
    r = _run(live, "run", "--model", "scripted/cli1", "how", "are", "things")
    assert r.returncode == 0, r.stderr
    assert "Nothing is live yet." in r.stdout
    assert "✓ status" in r.stderr


def test_run_unattended_denies_asks(live):
    ScriptedProvider.register("cli2", [[{"tool": "shell", "input": {"command": "echo hi"}}], ["could not run it"]])
    r = _run(live, "run", "--model", "scripted/cli2", "run echo")
    assert r.returncode == 0
    assert "✗ shell" in r.stderr and "not allowed" in r.stderr


def test_run_yes_approves(live):
    ScriptedProvider.register("cli3", [[{"tool": "shell", "input": {"command": "echo approved-run"}}], ["ran"]])
    r = _run(live, "run", "--yes", "--model", "scripted/cli3", "run echo")
    assert r.returncode == 0 and "✓ shell" in r.stderr


def test_run_json_and_slash_commands(live):
    ScriptedProvider.register("cli4", [["status here"]])
    r = _run(live, "run", "--json", "--model", "scripted/cli4", "/status")
    types = [json.loads(line)["type"] for line in r.stdout.splitlines() if line.startswith("{")]
    assert "message.created" in types and "part.delta" in types


def test_web_link_signs_in_and_serves_the_dashboard(live):
    r = _run(live, "web", "--no-open")
    link = next(w for w in r.stdout.split() if w.startswith("http"))
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    page = opener.open(link).read().decode()
    assert "<html" in page
    home = json.loads(opener.open(live.url + "/api/home").read())
    assert home["state"] == str(live.proj.resolve() / ".opentrain")


def test_help_and_debug(live):
    r = subprocess.run([sys.executable, "-m", "opentrain", "--help"], capture_output=True, text=True)
    assert "run" in r.stdout and "debug" in r.stdout
    r = subprocess.run([sys.executable, "-m", "opentrain", "debug", "--help"], capture_output=True, text=True)
    assert "bench" in r.stdout and "mine" in r.stdout


def test_run_follows_a_job_to_the_end(live):
    """The agent starts an eval and ends its turn; `run` waits for the job and the turn it wakes."""
    c = live.client(live.proj)
    content = "".join(json.dumps({"question": f"q{i}", "answer": "done"}) + "\n" for i in range(30))
    c.post("/api/dataset/fj/upload", {"filename": "fj.jsonl", "content": content})
    c.patch("/api/dataset/fj", {"grader": {"type": "exact"}})
    ScriptedProvider.register("fj-model", [])
    ScriptedProvider.register("cli5", [
        [{"tool": "dataset_eval", "input": {"name": "fj", "model": "scripted/fj-model"}}],
        ["Started the eval; waiting for it."],
        ["The eval finished: all rows passed."]])
    r = _run(live, "run", "--yes", "--model", "scripted/cli5", "evaluate fj")
    assert r.returncode == 0, r.stderr
    assert "eval scripted/fj-model on fj succeeded" in r.stderr, r.stderr
    assert "The eval finished" in r.stdout, r.stdout
    ScriptedProvider.register("cli6", [[{"tool": "dataset_eval", "input": {"name": "fj", "model": "scripted/fj-model"}}],
                                       ["started"]])
    r = _run(live, "run", "--yes", "--detach", "--model", "scripted/cli6", "x")
    assert r.returncode == 0 and "waiting for the job" not in r.stderr
