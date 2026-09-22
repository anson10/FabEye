"""Conformal and selective-risk experiments on lot-disjoint data.

Calibration pool: labelled validation-lot wafers. Test: labelled test-lot wafers.
Usage: WM_SPLIT=lot python training/run_conformal.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.wm811k import CLASSES, SPLIT  # noqa: E402
from evaluation.conformal import (
    classwise_thresholds,
    evaluate_sets,
    marginal_threshold,  # noqa: E402
    prediction_sets,
    select_threshold,
)

K = len(CLASSES)
rng = np.random.default_rng(0)
z = np.load(f"data/wm811k/embeddings_{SPLIT}.npz")
lab = z["label"] >= 0


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


cal_i = np.flatnonzero(lab & (z["split"] == 1))
te_i = np.flatnonzero(lab & (z["split"] == 2))
P_cal, y_cal, lot_cal = (
    softmax(z["logits"][cal_i].astype(np.float32)),
    z["label"][cal_i].astype(int),
    z["lot"][cal_i],
)
P_te, y_te = softmax(z["logits"][te_i].astype(np.float32)), z["label"][te_i].astype(int)
lots_cal = np.unique(lot_cal)
by_lot = {l: np.flatnonzero(lot_cal == l) for l in lots_cal}
print(
    f"calibration wafers {len(y_cal)} in {len(lots_cal)} lots, test wafers {len(y_te)}",
    flush=True,
)
out = {
    "n_cal": int(len(y_cal)),
    "n_cal_lots": int(len(lots_cal)),
    "n_test": int(len(y_te)),
}

# E1: guarantee on unseen lots, marginal vs class-conditional
out["e1"] = {}
for alpha in (0.1, 0.05):
    q = marginal_threshold(P_cal, y_cal, alpha)
    qc = classwise_thresholds(P_cal, y_cal, alpha, K)
    out["e1"][str(alpha)] = {
        "marginal": evaluate_sets(prediction_sets(P_te, q), y_te, K),
        "classwise": evaluate_sets(prediction_sets(P_te, qc), y_te, K),
    }
    for name in ("marginal", "classwise"):
        r = out["e1"][str(alpha)][name]
        print(
            f"E1 alpha {alpha} {name:9s} coverage {r['coverage']:.3f} set {r['mean_set_size']:.2f} "
            f"review {r['review_rate']:.3f} worst-class {np.nanmin(r['per_class_coverage']):.3f}",
            flush=True,
        )

# E2: what a random within-lot split would have reported (same lots on both sides)
perm = rng.permutation(len(y_te))
a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
q = marginal_threshold(P_te[a], y_te[a], 0.1)
r = evaluate_sets(prediction_sets(P_te[b], q), y_te[b], K)
out["e2_within_lot"] = r
print(
    f"E2 within-lot calibration coverage {r['coverage']:.3f} (target 0.90)", flush=True
)

# E3: coverage spread across calibration sets, lots vs iid wafers with the same wafer count
out["e3"] = {}
sizes = [5, 10, 20, 50, 100, 200]
target = 0.9
for m in sizes:
    lot_cov, waf_cov = [], []
    for _ in range(300):
        ls = rng.choice(lots_cal, m, replace=False)
        idx = np.concatenate([by_lot[l] for l in ls])
        q = marginal_threshold(P_cal[idx], y_cal[idx], 0.1)
        lot_cov.append(evaluate_sets(prediction_sets(P_te, q), y_te, K)["coverage"])
        idx2 = rng.choice(len(y_cal), len(idx), replace=False)
        q2 = marginal_threshold(P_cal[idx2], y_cal[idx2], 0.1)
        waf_cov.append(evaluate_sets(prediction_sets(P_te, q2), y_te, K)["coverage"])
    row = {}
    for name, v in (("lot_draws", lot_cov), ("wafer_draws", waf_cov)):
        v = np.array(v)
        row[name] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "p5": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
            "frac_below_target_minus_2pts": float((v < target - 0.02).mean()),
        }
    row["mean_wafers"] = float(
        np.mean(
            [
                len(
                    np.concatenate(
                        [by_lot[l] for l in rng.choice(lots_cal, m, replace=False)]
                    )
                )
                for _ in range(20)
            ]
        )
    )
    out["e3"][str(m)] = row
    print(
        f"E3 {m:4d} lots (~{row['mean_wafers']:.0f} wafers) lot-draw cov {row['lot_draws']['mean']:.3f} +- {row['lot_draws']['std']:.3f} "
        f"p5 {row['lot_draws']['p5']:.3f} | wafer-draw std {row['wafer_draws']['std']:.3f} p5 {row['wafer_draws']['p5']:.3f}",
        flush=True,
    )

fig, ax = plt.subplots(figsize=(7, 4))
xs = np.arange(len(sizes))
for name, color, off in (
    ("lot_draws", "#d9534f", -0.08),
    ("wafer_draws", "#7fb7c9", 0.08),
):
    mid = [out["e3"][str(m)][name]["mean"] for m in sizes]
    lo = [out["e3"][str(m)][name]["p5"] for m in sizes]
    hi = [out["e3"][str(m)][name]["p95"] for m in sizes]
    ax.errorbar(
        xs + off,
        mid,
        yerr=[np.array(mid) - lo, np.array(hi) - mid],
        fmt="o",
        color=color,
        capsize=3,
        label=(
            "calibrate on whole lots"
            if name == "lot_draws"
            else "calibrate on random wafers"
        ),
    )
ax.axhline(target, color="grey", ls="--", lw=1)
ax.set_xticks(xs, [str(m) for m in sizes])
ax.set_xlabel("calibration lots (wafer count matched for the wafer curve)")
ax.set_ylabel("coverage on unseen lots, 5th to 95th pct")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/wm811k_conformal_coverage.png", dpi=150)

# E4: selective risk control. Auto-accept only when the upper bound on error <= 2%.
conf_cal, wrong_cal = P_cal.max(1), P_cal.argmax(1) != y_cal
conf_te, wrong_te = P_te.max(1), P_te.argmax(1) != y_te
target_err, delta = 0.02, 0.1
out["e4"] = {}
t_all = select_threshold(conf_cal, wrong_cal, target_err, delta)
acc = conf_te >= t_all
out["e4"]["full_calibration"] = {
    "threshold": float(t_all),
    "accept_rate": float(acc.mean()),
    "realized_error": float(wrong_te[acc].mean()) if acc.any() else None,
}
print(
    f"E4 full calibration: threshold {t_all:.4f} accept {acc.mean():.3f} realized error {out['e4']['full_calibration']['realized_error']:.4f} (target {target_err})",
    flush=True,
)
for m in (10, 50, 200):
    rows = {"lot_draws": [], "wafer_draws": []}
    for _ in range(300):
        ls = rng.choice(lots_cal, m, replace=False)
        idx = np.concatenate([by_lot[l] for l in ls])
        idx2 = rng.choice(len(y_cal), len(idx), replace=False)
        for name, ii in (("lot_draws", idx), ("wafer_draws", idx2)):
            t = select_threshold(conf_cal[ii], wrong_cal[ii], target_err, delta)
            acc = conf_te >= t
            rows[name].append(
                (float(wrong_te[acc].mean()) if acc.any() else 0.0, float(acc.mean()))
            )
    res = {}
    for name, v in rows.items():
        v = np.array(v)
        res[name] = {
            "violation_rate": float((v[:, 0] > target_err).mean()),
            "mean_accept_rate": float(v[:, 1].mean()),
            "mean_error": float(v[:, 0].mean()),
        }
    out["e4"][f"{m}_lots"] = res
    print(
        f"E4 {m:4d} lots: violation lot-draws {res['lot_draws']['violation_rate']:.3f} wafer-draws {res['wafer_draws']['violation_rate']:.3f} "
        f"(nominal {delta}) accept {res['lot_draws']['mean_accept_rate']:.3f}",
        flush=True,
    )

with open(f"results/wm811k_{SPLIT}_conformal.json", "w") as f:
    json.dump(out, f, indent=1)
print("done")
