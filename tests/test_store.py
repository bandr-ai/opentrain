import sqlite3
import threading

from opentrain.core import ids
from opentrain.core.store import Store, merge


def test_ids_are_prefixed_and_time_ordered():
    a = [ids.new("session") for _ in range(200)]
    assert all(x.startswith("ses_") for x in a)
    assert a == sorted(a)
    assert ids.kind_of(a[0]) == "session"


def test_merge_deep_and_delete():
    assert merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"c": 3}, "d": None}) == {"a": {"b": 1, "c": 3}}


def test_projection_and_chain(tmp_path):
    s = Store(tmp_path / ".opentrain")
    seen = []
    s.subscribe(seen.append)
    ses = s.create("session", {"title": "t", "agent": "train"})
    s.update("session", ses["id"], {"title": "renamed", "tokens": {"input": 5}})
    msg = s.create("message", {"session_id": ses["id"], "role": "user"})
    s.create("part", {"session_id": ses["id"], "message_id": msg["id"], "type": "text", "text": "hi"})
    s.publish("part.delta", {"session_id": ses["id"], "delta": "x"}, durable=False)
    assert s.get("session", ses["id"])["title"] == "renamed"
    assert s.get("session", ses["id"])["tokens"] == {"input": 5}
    assert [p["text"] for p in s.find("part", parent_id=msg["id"])] == ["hi"]
    assert len(seen) == 5 and seen[-1]["durable"] is False
    assert len(s.events()) == 4  # the delta was not stored
    assert s.verify() == (True, "")
    s.delete("session", ses["id"])
    assert s.get("session", ses["id"]) is None and s.find("part", session_id=ses["id"]) == []


def test_tampering_breaks_the_chain(tmp_path):
    s = Store(tmp_path / ".opentrain")
    for i in range(3):
        s.create("session", {"title": str(i)})
    db = sqlite3.connect(s.path)
    db.execute("UPDATE event SET data='{\"title\":\"forged\"}' WHERE seq=2")
    db.commit()
    ok, where = s.verify()
    assert not ok and "event 2" in where


def test_concurrent_writers(tmp_path):
    s = Store(tmp_path / ".opentrain")
    def w():
        for _ in range(50):
            s.create("session", {"title": "x"})
    ts = [threading.Thread(target=w) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(s.find("session")) == 200 and s.verify()[0]
