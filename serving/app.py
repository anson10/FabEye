"""FastAPI service for wafer map pattern classification, served with ONNX Runtime.

Run: uvicorn serving.app:app --port 8000
Env: WAFER_ONNX (model path), WAFER_CALIBRATION (calibration file), ORT_THREADS,
     WAFER_API_KEY (required X-API-Key header for /predict*, unset disables auth),
     WAFER_CORS_ORIGINS (comma-separated allowed origins, default "*").

Each prediction carries two calibrated signals, both computed on lots the model
never trained on:
  prediction_set  class-conditional conformal set. Contains the true pattern with
                  about 1 - alpha probability per class, for wafers from new lots.
  auto_accept     True when confidence clears the selective-risk threshold, chosen so
                  the error rate among auto-accepted wafers stays under the target.

Every request is logged as one JSON line with a request ID (see serving/observability.py)
and counted in Prometheus metrics served at /metrics.
"""

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from serving.auth import require_api_key
from serving.observability import (
    ObservabilityMiddleware,
    logger,
    metrics_response,
    record_predictions,
)
from serving.preprocess import CLASSES, to_tensor

MODEL_PATH = os.environ.get("WAFER_ONNX", "serving/wafer_cnn.onnx")
CALIBRATION_PATH = os.environ.get("WAFER_CALIBRATION", "serving/calibration.json")
state = {}


@asynccontextmanager
async def lifespan(app):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
    state["sess"] = ort.InferenceSession(
        MODEL_PATH, so, providers=["CPUExecutionProvider"]
    )
    with open(CALIBRATION_PATH) as f:
        state["cal"] = json.load(f)
    digest = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
    state["cal_matches_model"] = digest == state["cal"]["model_sha256"]
    if os.environ.get("WAFER_API_KEY") is None:
        logger.info(
            json.dumps({"event": "startup", "warning": "WAFER_API_KEY not set — auth disabled"})
        )
    yield
    state.clear()


app = FastAPI(title="Wafer Map Classifier", version="2.0", lifespan=lifespan)
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("WAFER_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class WaferRequest(BaseModel):
    wafer_map: list[list[int]] = Field(
        ...,
        description="2D die grid. 0 outside the wafer, 1 good die, 2 defective die.",
    )


class BatchRequest(BaseModel):
    wafer_maps: list[list[list[int]]] = Field(..., max_length=64)


AlphaQuery = Query(
    0.1, description="Miscoverage level for the prediction set. Supported: 0.1, 0.05."
)


def check_alpha(alpha):
    key = str(alpha)
    if key not in state["cal"]["conformal"]:
        raise HTTPException(
            status_code=422,
            detail=f"alpha must be one of {sorted(state['cal']['conformal'])}",
        )
    return key


def infer(maps):
    try:
        x = np.stack([to_tensor(m) for m in maps])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    logits = state["sess"].run(None, {"wafer": x})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def describe(p, alpha_key):
    order = np.argsort(-p)
    q = np.asarray(state["cal"]["conformal"][alpha_key]["classwise_thresholds"])
    in_set = (1 - p) <= q
    accept = bool(p[order[0]] >= state["cal"]["selective"]["confidence_threshold"])
    return {
        "pattern": CLASSES[order[0]],
        "confidence": round(float(p[order[0]]), 4),
        "auto_accept": accept,
        "needs_review": not accept,
        "alpha": float(alpha_key),
        "prediction_set": [CLASSES[i] for i in order if in_set[i]],
        "top3": [
            {"pattern": CLASSES[i], "probability": round(float(p[i]), 4)}
            for i in order[:3]
        ],
    }


@app.get("/")
def root():
    return {"message": "Wafer Map Classifier. See /docs for usage."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "calibration_matches_model": state["cal_matches_model"],
    }


@app.get("/classes")
def classes():
    return {"classes": CLASSES}


@app.get("/calibration")
def calibration():
    """What the calibrated signals guarantee, and how they behaved on unseen lots."""
    c = state["cal"]
    return {
        "calibrated_on": c["calibration"],
        "matches_loaded_model": state["cal_matches_model"],
        "prediction_set": {
            a: {
                "coverage_on_unseen_lots": v["unseen_lot_coverage"],
                "worst_class_coverage_on_unseen_lots": v[
                    "unseen_lot_worst_class_coverage"
                ],
            }
            for a, v in c["conformal"].items()
        },
        "auto_accept": c["selective"],
        "limits": [
            "Sets and the accept rule assume new lots resemble the calibration lots.",
            "Coverage for the rarest classes rests on few calibration wafers and can fall short.",
            "The accept rule guarantee needs calibration on well over 200 lots to be useful.",
        ],
    }


@app.get("/metrics")
def metrics():
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(req: WaferRequest, alpha: float = AlphaQuery):
    key = check_alpha(alpha)
    t0 = time.perf_counter()
    out = describe(infer([req.wafer_map])[0], key)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    record_predictions([out])
    return out


@app.post("/predict/batch", dependencies=[Depends(require_api_key)])
def predict_batch(req: BatchRequest, alpha: float = AlphaQuery):
    key = check_alpha(alpha)
    t0 = time.perf_counter()
    probs = infer(req.wafer_maps)
    results = [describe(p, key) for p in probs]
    record_predictions(results)
    return {
        "results": results,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }
