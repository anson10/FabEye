import importlib
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.wm_models import WaferCNN  # noqa: E402
from serving.export_onnx import export  # noqa: E402
from serving.preprocess import to_tensor  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("onnx")
    ckpt, onnx_path = str(d / "m.pt"), str(d / "m.onnx")
    torch.save(WaferCNN().state_dict(), ckpt)
    export(ckpt, onnx_path)
    os.environ["WAFER_ONNX"] = onnx_path
    from fastapi.testclient import TestClient

    import serving.app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


def disc(n=30):
    yy, xx = np.mgrid[:n, :n]
    m = np.where(np.hypot(yy - n / 2, xx - n / 2) < n / 2, 1, 0)
    m[10:14, 10:14] = 2
    return m.tolist()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_valid_distribution(client):
    r = client.post("/predict", json={"wafer_map": disc()})
    assert r.status_code == 200
    body = r.json()
    assert body["pattern"] in client.get("/classes").json()["classes"]
    assert 0 < body["confidence"] <= 1
    assert len(body["top3"]) == 3
    assert isinstance(body["needs_review"], bool)
    assert body["needs_review"] == (not body["auto_accept"])
    assert all(
        p in client.get("/classes").json()["classes"] for p in body["prediction_set"]
    )


def test_predict_accepts_any_wafer_size(client):
    for n in (12, 26, 49):
        assert client.post("/predict", json={"wafer_map": disc(n)}).status_code == 200


def test_rejects_bad_values(client):
    bad = disc()
    bad[0][0] = 5
    assert client.post("/predict", json={"wafer_map": bad}).status_code == 422


def test_rejects_non_2d(client):
    assert client.post("/predict", json={"wafer_map": [1, 2, 3]}).status_code == 422


def test_batch_matches_single(client):
    maps = [disc(26), disc(40)]
    batch = client.post("/predict/batch", json={"wafer_maps": maps}).json()["results"]
    for m, b in zip(maps, batch):
        single = client.post("/predict", json={"wafer_map": m}).json()
        assert single["pattern"] == b["pattern"]
        assert abs(single["confidence"] - b["confidence"]) < 1e-4


def test_preprocessing_matches_training_pipeline():
    from data.wm811k import resize_map

    m = np.array(disc(33), dtype=np.uint8)
    r = resize_map(m)
    ref = np.stack([r == 0, r == 1, r == 2]).astype(np.float32)
    assert np.array_equal(to_tensor(m), ref)


def test_alpha_controls_set_and_rejects_unknown(client):
    ok = client.post("/predict?alpha=0.05", json={"wafer_map": disc()})
    assert ok.status_code == 200 and ok.json()["alpha"] == 0.05
    assert (
        client.post("/predict?alpha=0.3", json={"wafer_map": disc()}).status_code == 422
    )


def test_calibration_endpoint_states_limits(client):
    r = client.get("/calibration").json()
    assert r["calibrated_on"]["lots"] > 0
    assert set(r["prediction_set"]) == {"0.1", "0.05"}
    assert r["limits"]


def test_health_flags_mismatched_calibration(client):
    # the test model is untrained random weights, so it cannot match the shipped calibration
    assert client.get("/health").json()["calibration_matches_model"] is False
