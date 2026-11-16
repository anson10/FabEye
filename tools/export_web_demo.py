"""Export sample wafers and reference predictions for the in-browser demo on ansonantony.tech.

Writes two JSON files into --out:
  fabeye_samples.json  gallery for the demo page: per pattern, wafers drawn at random (seed 0)
                       from the lot-disjoint TEST split, 2 the service auto-accepts and 1 it
                       sends to review where the class has one. Each carries the raw die grid
                       and the reference output of this repo's ONNX pipeline, which the page
                       shows if the browser cannot run the model.
  fabeye_parity.json   gallery wafers plus --parity-extra more random test wafers, with
                       reference probabilities, for the browser/Python parity test.
  fabeye_threshold_curve.json  for the review-threshold slider: at each confidence threshold,
                       the share of ALL test-lot wafers classified without review and the
                       error rate among those, from the served ONNX model.

The reference output uses serving/preprocess.py, serving/wafer_cnn.onnx and
serving/calibration.json exactly as serving/app.py does (alpha 0.1).

Run: python tools/export_web_demo.py --out /path/to/website/data
"""

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.preprocess import CLASSES, to_tensor  # noqa: E402

ALPHA = "0.1"


def encode(m):
    return ["".join(str(int(v)) for v in row) for row in m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--processed", default="data/wm811k/processed_lot.pkl")
    ap.add_argument("--model", default="serving/wafer_cnn.onnx")
    ap.add_argument("--calibration", default="serving/calibration.json")
    ap.add_argument("--accepted-per-class", type=int, default=2)
    ap.add_argument("--review-per-class", type=int, default=1)
    ap.add_argument("--parity-extra", type=int, default=60)
    args = ap.parse_args()

    d = pickle.load(open(args.processed, "rb"))
    cal = json.load(open(args.calibration))
    digest = hashlib.sha256(open(args.model, "rb").read()).hexdigest()
    assert digest == cal["model_sha256"], "calibration does not match the model"
    assert d["classes"] == CLASSES == cal["classes"]

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    thr_conf = cal["selective"]["confidence_threshold"]
    q = np.asarray(cal["conformal"][ALPHA]["classwise_thresholds"])

    def predict(i):
        x = to_tensor(d["maps"][i])[None]
        logits = sess.run(None, {"wafer": x})[0][0].astype(np.float64)
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        top = int(np.argmax(p))
        return {
            "probs": [round(float(v), 6) for v in p],
            "pattern": CLASSES[top],
            "auto_accept": bool(p[top] >= thr_conf),
            "prediction_set": [CLASSES[k] for k in np.argsort(-p) if (1 - p[k]) <= q[k]],
        }

    def record(i, ref):
        m = d["maps"][i]
        return {
            "id": f"wm811k-test-{i}",
            "lot": str(d["lots"][i]),
            "label": CLASSES[int(d["y"][i])],
            "rows": int(m.shape[0]),
            "cols": int(m.shape[1]),
            "map": encode(m),
            "reference": ref,
        }

    test = np.where(d["split"] == 2)[0]
    rng = np.random.default_rng(0)
    gallery = []
    for c in range(len(CLASSES)):
        idx = rng.permutation(test[d["y"][test] == c])
        acc, rev = [], []
        for i in idx:
            if len(acc) >= args.accepted_per_class and len(rev) >= args.review_per_class:
                break
            ref = predict(int(i))
            if ref["auto_accept"] and len(acc) < args.accepted_per_class:
                acc.append(record(int(i), ref))
            elif not ref["auto_accept"] and len(rev) < args.review_per_class:
                rev.append(record(int(i), ref))
        gallery += acc + rev

    chosen = {g["id"] for g in gallery}
    extra = []
    for i in np.random.default_rng(1).permutation(test):
        if len(extra) >= args.parity_extra:
            break
        if f"wm811k-test-{i}" not in chosen:
            extra.append(record(int(i), predict(int(i))))

    meta = {
        "dataset": "WM-811K (Wu, Jang and Chen, IEEE Trans. Semiconductor Manufacturing, 2015)",
        "dataset_url": "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map",
        "split": "lot-disjoint test split, data/wm811k.py (WM_SPLIT=lot, seed 0)",
        "selection": (
            f"per pattern, drawn at random (seed 0) from test lots: {args.accepted_per_class} "
            f"auto-accepted and {args.review_per_class} sent to review where the class has one"
        ),
        "model_sha256": digest,
        "alpha": float(ALPHA),
        "classes": CLASSES,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fabeye_samples.json"), "w") as f:
        json.dump({"meta": meta, "samples": gallery}, f, separators=(",", ":"))
    with open(os.path.join(args.out, "fabeye_parity.json"), "w") as f:
        json.dump({"meta": meta, "samples": gallery + extra}, f, separators=(",", ":"))
    # Review-threshold curve over the whole test split, with the served model.
    conf, right = [], []
    for start in range(0, len(test), 256):
        idx = test[start : start + 256]
        x = np.stack([to_tensor(d["maps"][i]) for i in idx])
        logits = sess.run(None, {"wafer": x})[0].astype(np.float64)
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        conf.append(p.max(1))
        right.append(p.argmax(1) == d["y"][idx])
    conf, right = np.concatenate(conf), np.concatenate(right)
    grid = sorted({round(t, 3) for t in np.arange(0.50, 0.9951, 0.005)})
    rows = []
    for t in [*grid, thr_conf]:
        acc = conf >= t
        rows.append({
            "threshold": round(float(t), 4),
            "accepted": round(float(acc.mean()), 4),
            "error": round(float(1 - right[acc].mean()), 4) if acc.any() else None,
        })
    at = rows.pop()  # the served threshold: must reproduce calibration.json
    assert abs(at["accepted"] - cal["selective"]["unseen_lot_accept_rate"]) < 5e-4, at
    assert abs(at["error"] - cal["selective"]["unseen_lot_error_among_accepted"]) < 5e-4, at
    with open(os.path.join(args.out, "fabeye_threshold_curve.json"), "w") as f:
        json.dump({"meta": {**meta, "wafers": int(len(test)), "served_threshold": thr_conf,
                            "at_served_threshold": at}, "curve": rows}, f, separators=(",", ":"))

    n_rev = sum(not g["reference"]["auto_accept"] for g in gallery)
    print(f"gallery {len(gallery)} wafers ({n_rev} sent to review), parity {len(gallery) + len(extra)}")


if __name__ == "__main__":
    main()
