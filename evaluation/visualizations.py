"""
Shared visualization utilities for GNN and CNN evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


DEFECT_NAMES = ["none", "particle_contamination", "scratch", "pit", "oxide_defect", "metal_contamination"]
DEFECT_COLORS = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix",
                          save_path: str = None, normalize: bool = True):
    """Plot a confusion matrix with optional row normalization."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot = cm.astype(float) / row_sums
    else:
        cm_plot = cm.astype(float)

    n = cm.shape[0]
    names = DEFECT_NAMES[:n]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            raw = cm[i, j]
            label = f"{val:.2f}\n({raw})" if normalize else str(raw)
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_wafer_prediction(image: np.ndarray, gnn_loc: tuple, gnn_type: int,
                          cnn_boxes: list, cnn_labels: list,
                          save_path: str = None):
    """
    Overlay GNN location prediction and CNN bounding boxes on a wafer image.

    Args:
        image:      HxWx3 uint8 array
        gnn_loc:    (x, y) normalized [0, 1]
        gnn_type:   predicted defect class (0-based)
        cnn_boxes:  list of [x1, y1, x2, y2] pixel coords
        cnn_labels: list of 1-based class labels matching cnn_boxes
    """
    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    # GNN predicted location (cross marker)
    gx, gy = gnn_loc[0] * w, gnn_loc[1] * h
    color = DEFECT_COLORS[gnn_type % len(DEFECT_COLORS)]
    ax.plot(gx, gy, "x", color=color, markersize=16, markeredgewidth=3,
            label=f"GNN: {DEFECT_NAMES[gnn_type]}")

    # CNN detected bounding boxes
    for box, lbl in zip(cnn_boxes, cnn_labels):
        x1, y1, x2, y2 = box
        c = DEFECT_COLORS[(lbl - 1) % len(DEFECT_COLORS)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=c, facecolor="none")
        ax.add_patch(rect)
        name = DEFECT_NAMES[(lbl - 1) % len(DEFECT_NAMES)]
        ax.text(x1, y1 - 4, name, color=c, fontsize=8)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("GNN prediction (×) vs CNN detections (boxes)")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
