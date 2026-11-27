import json
import threading
import time
import urllib.request

import pytest

from opentrain.client import Client, ClientError
from opentrain.server import service


def test_health_and_auth(live):
    c = live.client(live.proj)
    assert c.get("/api/health")["ok"] is True
    bad = Client(live.url, "wrong", live.proj)
    with pytest.raises(ClientError) as e:
        bad.get("/api/health")
    assert e.value.status == 401


def test_host_header_is_checked(live):
    req = urllib.request.Request(live.url + "/api/health", headers={"Host": "evil.example:80"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403


def test_sessions_crud_and_project_scoping(live, tmp_path):
    c = live.client(live.proj)
    s = c.post("/api/session", {"title": "first"})
    assert s["id"].startswith("ses_") and s["agent"] == "train"
    assert [x["id"] for x in c.get("/api/session")] == [s["id"]]
    other = tmp_path / "other"
    other.mkdir()
    assert live.client(other).get("/api/session") == []
    c.patch(f"/api/session/{s['id']}", {"title": "renamed"})
    assert c.get(f"/api/session/{s['id']}")["title"] == "renamed"
    assert c.get(f"/api/session/{s['id']}/message") == []
    with pytest.raises(ClientError) as e:
        c.get("/api/session/ses_nope")
    assert e.value.status == 404
    c.delete(f"/api/session/{s['id']}")
    assert c.get("/api/session") == []
    log = c.get("/api/event/log")
    assert log["verified"] and [e["type"] for e in log["events"]][:2] == ["session.created", "session.updated"]


def test_event_stream_delivers_project_events(live):
    c = live.client(live.proj)
    got = []

    def listen():
        for ev in c.events():
            got.append(ev)
            if ev["type"] == "session.created":
                return

    t = threading.Thread(target=listen, daemon=True)
    t.start()
    time.sleep(0.3)
    c.post("/api/session", {"title": "x"})
    t.join(5)
    assert got[0]["type"] == "server.connected" and got[-1]["type"] == "session.created"


def test_pairing_gives_a_cookie_once(live):
    c = live.client(live.proj)
    link = c.post("/api/pair")["url"]
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    opener.open(link.replace("/auth/connect", "/auth/connect") + "?next=/api/health")
    with pytest.raises(urllib.error.HTTPError) as e:  # second use
        opener.open(link)
    assert e.value.code == 403
    # the cookie from the first use authenticates
    assert json.loads(opener.open(live.url + "/api/health").read())["ok"]


def test_registration_file_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTRAIN_SERVICE_FILE", str(tmp_path / "svc.json"))
    reg = service.Registration(url="http://127.0.0.1:1", pid=1, version="x", password="p")
    path = service.write_registration(reg)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert service.read_registration() == reg
    assert service.discover() is None  # nothing answers
