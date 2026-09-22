"""
Week 5 master integration script.

Runs GNN and CNN on the same 1,500 test wafers and produces:
  1. Alignment metrics (type-based primary, full type+location secondary)
  2. Confusion matrices for both models vs ground truth
  3. Parameter-defect correlation heatmap
  4. All results saved to SQLite database

Usage:
    PYTHONPATH=. python3 evaluation/analyze_alignment.py
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

from evaluation.alignment import GNNCNNComparison, DEFECT_NAMES
from evaluation.visualizations import plot_confusion_matrix
from database.db_utils import DatabaseManager

# Process step and parameter names — same order as generator.py PARAMETER_RANGES
PROCESS_STEPS = [
    "oxidation", "lithography", "etching", "deposition",
    "doping", "cmp", "cleaning", "annealing",
]
STEP_PARAMS = {
    "oxidation":   ["temperature", "pressure", "duration"],
    "lithography": ["exposure_dose", "focus_offset", "wavelength"],
    "etching":     ["etch_rate", "selectivity", "duration"],
    "deposition":  ["temperature", "rate", "thickness"],
    "doping":      ["concentration", "energy", "dose"],
    "cmp":         ["pressure", "velocity", "slurry_conc"],
    "cleaning":    ["chemical_conc", "temperature", "duration"],
    "annealing":   ["temperature", "duration", "atmosphere"],
}
FEATURE_NAMES = [
    f"{step}/{param}"
    for step in PROCESS_STEPS
    for param in STEP_PARAMS[step]
]  # 24 feature labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn-ckpt",    default="checkpoints/best_gnn.pt")
    p.add_argument("--cnn-ckpt",    default="checkpoints/best_cnn.pt")
    p.add_argument("--gnn-data",    default="data/raw/synthetic_wafers.json")
    p.add_argument("--image-dir",   default="data/wafer_images")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device",      default=None)
    p.add_argument("--loc-thresh",  type=float, default=0.35)
    p.add_argument("--db-url",      default=None,
                   help="SQLAlchemy DB URL (default: sqlite:///fabeye.db)")
    return p.parse_args()


# ------------------------------------------------------------------
# 1. Alignment plots
# ------------------------------------------------------------------
def _plot_alignment(metrics: dict, save_dir: str, loc_thresh: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GNN-CNN Integration Analysis", fontsize=14)

    labels = ["True Negative", "Aligned", "Type Mismatch", "Loc Mismatch", "GNN Only", "CNN Only"]
    counts = [
        metrics["true_negative"], metrics["aligned"],
        metrics["type_mismatch"], metrics["loc_mismatch"],
        metrics["gnn_only"],      metrics["cnn_only"],
    ]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#e67e22", "#9b59b6", "#f39c12"]
    nonzero = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]
    if nonzero:
        c_, l_, col_ = zip(*nonzero)
        axes[0].pie(c_, labels=l_, colors=col_, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Wafer Outcome Distribution")

    names   = ["Type\nAlignment", "Full\nAlignment", "FP Rate", "FN Rate"]
    values  = [metrics["alignment_rate"], metrics["full_alignment_rate"],
               metrics["fp_rate"],        metrics["fn_rate"]]
    targets = [0.65, None, 0.30, 0.25]
    bar_col = ["#3498db", "#5dade2", "#e74c3c", "#e74c3c"]

    bars = axes[1].bar(names, values, color=bar_col, alpha=0.8)
    for target in targets:
        if target is not None:
            axes[1].axhline(target, color="red", linestyle="--", alpha=0.4, linewidth=1)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    axes[1].set_ylim(0, 1.1)
    axes[1].set_title(f"Key Metrics vs Targets (loc_thresh={loc_thresh:.2f})")
    axes[1].set_ylabel("Rate")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "alignment_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Alignment plot   → {path}")


# ------------------------------------------------------------------
# 2. Parameter-defect correlation
# ------------------------------------------------------------------
def _parameter_correlation(gnn_data_path: str, seed: int, save_dir: str):
    """
    For each defect type, compute the mean normalized feature value across
    wafers of that class. Visualises which process parameters are elevated
    for each defect type.
    """
    with open(gnn_data_path) as f:
        samples = json.load(f)

    n = len(samples)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    test_idx = set(indices[n_train + n_val:].tolist())

    features, labels = [], []
    for i, s in enumerate(samples):
        if i not in test_idx:
            continue
        feat = np.array(s["node_features"], dtype=np.float32)  # [8, 3]
        features.append(feat.flatten())                          # [24]
        labels.append(s["defect"]["defect_type"])

    X = np.stack(features)   # [N_test, 24]
    y = np.array(labels)     # [N_test]

    n_classes = len(DEFECT_NAMES)
    n_feats   = len(FEATURE_NAMES)

    # Class-conditional means: how elevated is each feature for each defect type?
    class_means = np.zeros((n_classes, n_feats))
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            class_means[c] = X[mask].mean(axis=0)

    # Normalise across classes per feature so each column spans [0, 1]
    col_min = class_means.min(axis=0, keepdims=True)
    col_max = class_means.max(axis=0, keepdims=True)
    class_means_norm = (class_means - col_min) / (col_max - col_min + 1e-8)

    # Shorten feature names for the plot x-axis
    short_names = [f.replace("_contamination", "_cont").replace("_defect", "_def")
                   for f in FEATURE_NAMES]

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(class_means_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalised mean (0=lowest class, 1=highest)")

    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(DEFECT_NAMES, fontsize=9)
    ax.set_title("Parameter-Defect Correlation: class-conditional feature means (test set)")

    # Annotate causal rules from generator.py
    causal = {
        (4, FEATURE_NAMES.index("oxidation/temperature")): "★",
        (4, FEATURE_NAMES.index("oxidation/duration")):    "★",
        (2, FEATURE_NAMES.index("cmp/pressure")):          "★",
        (2, FEATURE_NAMES.index("cmp/slurry_conc")):       "★",
        (1, FEATURE_NAMES.index("cleaning/chemical_conc")):"★",
        (3, FEATURE_NAMES.index("doping/concentration")):  "★",
        (5, FEATURE_NAMES.index("deposition/rate")):       "★",
    }
    for (row, col), marker in causal.items():
        ax.text(col, row, marker, ha="center", va="center", fontsize=8, color="blue")

    plt.tight_layout()
    path = os.path.join(save_dir, "parameter_correlation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Correlation plot → {path}")
    return class_means_norm


# ------------------------------------------------------------------
# 3. Per-type breakdown helper
# ------------------------------------------------------------------
def _print_per_type(per_wafer: list):
    from collections import defaultdict
    by_type = defaultdict(lambda: {"aligned": 0, "type_mismatch": 0,
                                   "loc_mismatch": 0, "gnn_only": 0})
    for w in per_wafer:
        t = w["gnn_type_pred"]
        if t == 0:
            continue
        s = w["status"]
        if s in by_type[t]:
            by_type[t][s] += 1

    print("\n  Per-type breakdown (GNN-predicted defects only):")
    print(f"  {'Type':<25} {'Aligned':>8} {'TypeErr':>8} {'LocErr':>8} {'GNN-Only':>9}")
    print("  " + "-" * 62)
    for t in sorted(by_type):
        name = DEFECT_NAMES[t] if t < len(DEFECT_NAMES) else str(t)
        d = by_type[t]
        print(f"  {name:<25} {d['aligned']:>8} {d['type_mismatch']:>8} "
              f"{d['loc_mismatch']:>8} {d['gnn_only']:>9}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 60)
    print("FabEye — Week 5: GNN-CNN Integration Analysis")
    print("=" * 60)

    comparator = GNNCNNComparison(
        gnn_ckpt=args.gnn_ckpt,
        cnn_ckpt=args.cnn_ckpt,
        gnn_data=args.gnn_data,
        image_dir=args.image_dir,
        device=args.device,
        location_threshold=args.loc_thresh,
    )
    print(f"Test set size: {comparator.n_test} wafers\n")

    metrics = comparator.run()

    # ── Print results ──────────────────────────────────────────────
    print("\n=== Integration Results ===")
    print(f"  Test wafers:          {metrics['n_test']}")
    print(f"  Type alignment rate:  {metrics['alignment_rate']:.3f}  (target >0.65) "
          + ("✅" if metrics["alignment_rate"] >= 0.65 else "❌"))
    print(f"  Full alignment rate:  {metrics['full_alignment_rate']:.3f}  "
          f"(type + location within {args.loc_thresh:.2f})")
    print(f"  FP rate:              {metrics['fp_rate']:.3f}  (target <0.30) "
          + ("✅" if metrics["fp_rate"] < 0.30 else "❌"))
    print(f"  FN rate:              {metrics['fn_rate']:.3f}  (target <0.25) "
          + ("✅" if metrics["fn_rate"] < 0.25 else "❌"))
    print(f"\n  True negative:  {metrics['true_negative']}")
    print(f"  Aligned:        {metrics['aligned']}  (type ✓ + loc ✓)")
    print(f"  Loc mismatch:   {metrics['loc_mismatch']}  (type ✓ + loc ✗)")
    print(f"  Type mismatch:  {metrics['type_mismatch']}  (type ✗)")
    print(f"  GNN only:       {metrics['gnn_only']}  (GNN detects, CNN misses)")
    print(f"  CNN only:       {metrics['cnn_only']}  (CNN detects, GNN misses)")

    _print_per_type(metrics["per_wafer"])

    # ── Save metrics JSON ──────────────────────────────────────────
    save_metrics = {k: v for k, v in metrics.items()
                    if k not in ("per_wafer", "gnn_cm", "cnn_cm")}
    path = os.path.join(args.results_dir, "alignment_metrics.json")
    with open(path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nOutputs:")
    print(f"  Metrics JSON     → {path}")

    # ── 1. Alignment plot ──────────────────────────────────────────
    _plot_alignment(metrics, save_dir=args.results_dir, loc_thresh=args.loc_thresh)

    # ── 2. Confusion matrices ──────────────────────────────────────
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(
        metrics["gnn_cm"], title="GNN Confusion Matrix (test set)",
        save_path=os.path.join(args.results_dir, "gnn_confusion_matrix.png"),
    )
    print(f"  GNN confusion    → {args.results_dir}/gnn_confusion_matrix.png")

    plot_confusion_matrix(
        metrics["cnn_cm"], title="CNN Confusion Matrix (test set)",
        save_path=os.path.join(args.results_dir, "cnn_confusion_matrix.png"),
    )
    print(f"  CNN confusion    → {args.results_dir}/cnn_confusion_matrix.png")

    # ── 3. Parameter-defect correlations ──────────────────────────
    print("\nComputing parameter-defect correlations...")
    _parameter_correlation(
        gnn_data_path=args.gnn_data,
        seed=42,
        save_dir=args.results_dir,
    )

    # ── 4. Save to database ────────────────────────────────────────
    print("\nSaving results to database...")
    db = DatabaseManager(args.db_url)
    row_id = db.log_alignment(
        run_name="week5_integration",
        n_test=metrics["n_test"],
        alignment_rate=metrics["alignment_rate"],
        full_alignment_rate=metrics["full_alignment_rate"],
        fp_rate=metrics["fp_rate"],
        fn_rate=metrics["fn_rate"],
        true_negative=metrics["true_negative"],
        aligned=metrics["aligned"],
        type_mismatch=metrics["type_mismatch"],
        loc_mismatch=metrics["loc_mismatch"],
        gnn_only=metrics["gnn_only"],
        cnn_only=metrics["cnn_only"],
        location_threshold=args.loc_thresh,
        notes=f"GNN ckpt={args.gnn_ckpt}, CNN ckpt={args.cnn_ckpt}",
    )
    print(f"  DB row id={row_id} → fabeye.db (alignment_results table)")

    print("\nWeek 5 complete.")


if __name__ == "__main__":
    main()
