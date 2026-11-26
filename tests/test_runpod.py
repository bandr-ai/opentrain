"""RunPod and the training methods, without the network: RunPod's API is faked."""

import json

import pytest

from opentrain.backends import get_backend
from opentrain.backends import runpod_backend as rb
from opentrain.plugs import TrainSpec
from opentrain.recipes import recipe, task_kind, teacher_for


@pytest.fixture
def keys(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test")
    monkeypatch.setenv("OPENTRAIN_AUTH_FILE", str(tmp_path / "auth.json"))
    monkeypatch.delenv(rb.SERVE_KEY_ENV, raising=False)


def test_a_pod_goes_to_the_first_gpu_with_capacity(keys, monkeypatch):
    seen = []

    def api(method, path, body=None, timeout=60):
        seen.append((body["gpu"]["id"], body["cloud"]))
        if body["gpu"]["id"] == "NVIDIA A100 80GB PCIe" and body["cloud"] == "SECURE":
            return 201, {"id": "p1", "cost": 2.09}
        return 400, {"error": "no capacity"}

    monkeypatch.setattr(rb, "_api", api)
    pod = rb.create_pod("t", ["L40S", "A100-80GB"], log=lambda s: None)
    assert pod["id"] == "p1" and pod["hourly"] == 2.09 and pod["url"] == "https://p1-8000.proxy.runpod.net"
    assert seen == [("NVIDIA L40S", "COMMUNITY"), ("NVIDIA L40S", "SECURE"),
                    ("NVIDIA A100 80GB PCIe", "COMMUNITY"), ("NVIDIA A100 80GB PCIe", "SECURE")]


def test_the_pod_carries_the_agent_and_a_generated_key(keys, monkeypatch):
    bodies = []
    monkeypatch.setattr(rb, "_api", lambda m, p, body=None, timeout=60: bodies.append(body) or (201, {"id": "p"}))
    rb.create_pod("t", ["L4"], log=lambda s: None)
    env = bodies[0]["env"]
    assert env["OT_TOKEN"].startswith("ot-") and env["OT_RUNPOD_KEY"] == "rp-test"
    assert b"def watchdog" in __import__("base64").b64decode(env["OT_BOOT"])
    assert bodies[0]["ports"] == ["8000/http"] and bodies[0]["image"].startswith("vllm/vllm-openai:")


def test_no_money_and_no_capacity_say_so(keys, monkeypatch):
    monkeypatch.setattr(rb, "_api", lambda *a, **k: (402, {"error": "balance"}))
    with pytest.raises(rb.RunPodError, match="balance"):
        rb.create_pod("t", ["L4"], log=lambda s: None)
    monkeypatch.setattr(rb, "_api", lambda *a, **k: (400, {"error": "none"}))
    with pytest.raises(rb.RunPodError, match="no RunPod GPU available"):
        rb.create_pod("t", ["L4"], log=lambda s: None)


def test_estimates_per_method(keys, tmp_path):
    data = tmp_path / "t.jsonl"
    data.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": "q " * 60},
                                                       {"role": "assistant", "content": "a " * 100}]})
                              for _ in range(200)))
    b = get_backend("runpod", tmp_path)
    costs = {}
    for m in ("sft", "rejection", "rl", "opd"):
        r = recipe("reason", 200, "Qwen/Qwen3-1.7B", grader={"type": "number"}, method=m)
        costs[m] = b.estimate(TrainSpec(base_model="Qwen/Qwen3-1.7B", data=data, method=m, extra=r["extra"]))
    assert 0 < costs["sft"] < costs["rejection"] and costs["sft"] < costs["rl"] and all(c < 5 for c in costs.values())


def test_backends_that_only_do_sft_refuse_the_rest(tmp_path):
    data = tmp_path / "t.jsonl"
    data.write_text(json.dumps({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}))
    with pytest.raises(ValueError, match="modal or runpod"):
        get_backend("local", tmp_path).estimate(TrainSpec(base_model="Qwen/Qwen3-0.6B", data=data, method="rl"))


def test_recipes_pick_a_method_plan():
    gsm = ["Natalia sold 48/2 = <<48/2=24>>24 clips in May. Altogether she sold 48+24 = 72 clips. #### 72"] * 20
    assert task_kind({"type": "number"}, gsm) == "reason"
    r = recipe("reason", 7000, "Qwen/Qwen3-1.7B", grader={"type": "number"})
    assert r["plan"] == ["sft", "opd", "rl", "rejection"] and r["method"] == "sft"
    rl = recipe("reason", 7000, "Qwen/Qwen3-1.7B", grader={"type": "number"}, method="rl")
    assert rl["extra"]["grader"] == {"type": "number"} and rl["learning_rate"] < 1e-4
    assert recipe("reason", 7000, "Qwen/Qwen3-1.7B", method="opd")["extra"]["teacher"] == "Qwen/Qwen3-8B"
    assert teacher_for("Qwen/Qwen3.5-2B") == "Qwen/Qwen3.5-9B" and teacher_for("Qwen/Qwen3-14B") is None
    with pytest.raises(ValueError, match="grader"):
        recipe("answer", 100, "Qwen/Qwen3-1.7B", grader={"type": "judge"}, method="rl")
    with pytest.raises(ValueError, match="teacher"):
        recipe("label", 100, "Qwen/Qwen3-14B", method="opd")


def test_kaggle_reads_the_largest_table_in_the_zip(monkeypatch):
    import io
    import zipfile

    from opentrain.datasets import DatasetError, kaggle_auth, read_kaggle

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "hi")
        z.writestr("small.csv", "q,a\n1,2\n")
        z.writestr("data/train.csv", "question,answer\n" + "".join(f"what is {i}+1,{i + 1}\n" for i in range(50)))
    urls = []
    rows, src = read_kaggle("someone/sums", fetch=lambda u: urls.append(u) or buf.getvalue(), limit=10)
    assert urls == ["https://www.kaggle.com/api/v1/datasets/download/someone/sums"]
    assert len(rows) == 10 and rows[0] == {"question": "what is 0+1", "answer": "1"} and src["file"] == "data/train.csv"
    assert read_kaggle("someone/sums", file="small.csv", fetch=lambda u: buf.getvalue())[1]["file"] == "small.csv"
    with pytest.raises(DatasetError):
        read_kaggle("not a dataset id")
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KAGGLE_KEY", "me:abc")
    assert kaggle_auth()["Authorization"].startswith("Basic ")


def test_a_runpodctl_sign_in_is_enough(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("OPENTRAIN_AUTH_FILE", str(tmp_path / "auth.json"))
    cfg = tmp_path / "config.toml"
    cfg.write_text('apikey = "rpa_fromcli"\napiurl = "https://api.runpod.io/graphql"\n')
    monkeypatch.setenv("RUNPOD_CONFIG", str(cfg))
    assert rb.api_key() == "rpa_fromcli"
