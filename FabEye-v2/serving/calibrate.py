"""Compute conformal and selective-risk thresholds for the served ONNX model.

Calibrates on labelled validation-lot wafers and reports how the thresholds behave
on labelled test-lot wafers, which are different lots. Writes serving/calibration.json.

Usage: WM_SPLIT=lot python serving/calibrate.py --onnx serving/wafer_cnn.onnx
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.conformal import (  # noqa: E402
    classwise_thresholds,
    evaluate_sets,
    marginal_threshold,
    prediction_sets,
    select_threshold,
)
from serving.preprocess import CLASSES  # noqa: E402

ALPHAS = (0.1, 0.05)
RISK_TARGET, RISK_DELTA = 0.02, 0.1


def probs_for(sess, maps, bs=2048):
    out = []
    for s in range(0, len(maps), bs):
        m = maps[s : s + bs]
        x = np.stack([m == 0, m == 1, m == 2], 1).astype(np.float32)
        lg = sess.run(None, {"wafer": x})[0]
        e = np.exp(lg - lg.max(1, keepdims=True))
        out.append(e / e.sum(1, keepdims=True))
    return np.concatenate(out)


def main(onnx_path, cache, out_path):
    k = len(CLASSES)
    if not os.path.exists(cache):
        sys.exit(
            f"{cache} not found. Build it with: python training/scarce_labels.py --cache-only"
        )
    z = np.load(cache)
    lab, split = z["label"] >= 0, z["split"]
    cal, te = np.flatnonzero(lab & (split == 1)), np.flatnonzero(lab & (split == 2))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    maps = z["maps"]
    p_cal, y_cal = probs_for(sess, maps[cal]), z["label"][cal].astype(int)
    p_te, y_te = probs_for(sess, maps[te]), z["label"][te].astype(int)

    result = {
        "model_sha256": hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
        "calibration": {
            "wafers": int(len(cal)),
            "lots": int(len(np.unique(z["lot"][cal]))),
            "source": "labelled wafers from validation lots, disjoint from training and test lots",
        },
        "classes": CLASSES,
        "conformal": {},
    }
    for a in ALPHAS:
        q = marginal_threshold(p_cal, y_cal, a)
        qc = classwise_thresholds(p_cal, y_cal, a, k)
        ev = evaluate_sets(prediction_sets(p_te, qc), y_te, k)
        result["conformal"][str(a)] = {
            "marginal_threshold": q,
            "classwise_thresholds": qc.tolist(),
            "unseen_lot_coverage": ev["coverage"],
            "unseen_lot_worst_class_coverage": float(
                np.nanmin(ev["per_class_coverage"])
            ),
            "unseen_lot_review_rate": ev["review_rate"],
        }
    conf_cal, wrong_cal = p_cal.max(1), p_cal.argmax(1) != y_cal
    t = select_threshold(conf_cal, wrong_cal, RISK_TARGET, RISK_DELTA)
    acc = p_te.max(1) >= t
    result["selective"] = {
        "confidence_threshold": float(t),
        "target_error": RISK_TARGET,
        "delta": RISK_DELTA,
        "unseen_lot_accept_rate": float(acc.mean()),
        "unseen_lot_error_among_accepted": (
            float((p_te.argmax(1) != y_te)[acc].mean()) if acc.any() else None
        ),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(
        json.dumps({k: v for k, v in result.items() if k != "classes"}, indent=1)[:1600]
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="serving/wafer_cnn.onnx")
    ap.add_argument("--cache", default="data/wm811k/ctx_cache_lot.npz")
    ap.add_argument("--out", default="serving/calibration.json")
    a = ap.parse_args()
    main(a.onnx, a.cache, a.out)
