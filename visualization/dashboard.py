"""
FabEye Streamlit Dashboard — 6 tabs covering the full ML pipeline.

Run:
    PYTHONPATH=. streamlit run visualization/dashboard.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FabEye — Wafer Defect Analysis",
    page_icon="🔬",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
RESULTS     = ROOT / "results"
DATA_RAW    = ROOT / "data" / "raw" / "synthetic_wafers.json"
IMAGE_DIR   = ROOT / "data" / "wafer_images"
ANN_PATH    = IMAGE_DIR / "annotations.json"
GNN_CKPT    = ROOT / "checkpoints" / "best_gnn.pt"
CNN_CKPT    = ROOT / "checkpoints" / "best_cnn.pt"

DEFECT_NAMES  = ["none", "particle_contamination", "scratch", "pit",
                 "oxide_defect", "metal_contamination"]
DEFECT_COLORS = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
IMG_SIZE      = 512


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def load_gnn_metrics():
    with open(RESULTS / "gnn_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_cnn_metrics():
    with open(RESULTS / "cnn_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_alignment_metrics():
    with open(RESULTS / "alignment_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_wafer_data():
    with open(DATA_RAW) as f:
        return json.load(f)

@st.cache_data
def load_annotations():
    with open(ANN_PATH) as f:
        return json.load(f)

@st.cache_resource
def load_models():
    """Load both models once and cache them for the wafer inspector tab."""
    import torch
    from models.gnn import DefectPredictionGNN
    from models.cnn import DefectDetectionCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(GNN_CKPT, map_location=device, weights_only=False)
    hidden = ckpt["model_state"]["conv1.bias"].shape[0]
    gnn = DefectPredictionGNN(hidden_channels=hidden)
    gnn.load_state_dict(ckpt["model_state"])
    gnn = gnn.to(device).eval()

    ckpt = torch.load(CNN_CKPT, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if any(k.startswith("model.module.") for k in state):
        state = {k.replace("model.module.", "model.", 1): v for k, v in state.items()}
    cnn = DefectDetectionCNN(pretrained=False)
    cnn.load_state_dict(state)
    cnn = cnn.to(device).eval()

    return gnn, cnn, device


# ── Helpers ───────────────────────────────────────────────────────────────────
def _confusion_img(path: Path, title: str):
    if path.exists():
        st.image(str(path), caption=title, use_column_width=True)
    else:
        st.info(f"Run `python3 evaluation/analyze_alignment.py` to generate {path.name}")


def _metric_row(cols, labels, values, fmts=None):
    for col, label, val in zip(cols, labels, values):
        fmt = fmts[labels.index(label)] if fmts else "{:.4f}"
        col.metric(label, fmt.format(val) if isinstance(val, float) else str(val))


# ─────────────────────────────────────────────────────────────────────────────
# TAB DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dataset", "🧠 GNN Results", "🔭 CNN Results",
    "🔗 Integration", "📈 Parameter Correlations", "🔍 Wafer Inspector",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview")

    if not DATA_RAW.exists():
        st.warning("Raw data not found. Run `python3 data/generator.py --n 10000` first.")
    else:
        wafers = load_wafer_data()
        n = len(wafers)
        defect_types = [w["defect"]["defect_type"] for w in wafers]
        has_defect   = [w["defect"]["has_defect"]  for w in wafers]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Wafers", f"{n:,}")
        c2.metric("Defective", f"{sum(has_defect):,}")
        c3.metric("Clean", f"{n - sum(has_defect):,}")
        c4.metric("Defect Rate", f"{sum(has_defect)/n*100:.1f}%")

        st.divider()
        col_l, col_r = st.columns(2)

        # Defect type distribution bar chart
        with col_l:
            st.subheader("Defect Type Distribution")
            counts = [defect_types.count(i) for i in range(len(DEFECT_NAMES))]
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.bar(DEFECT_NAMES, counts, color=DEFECT_COLORS, alpha=0.85)
            ax.set_ylabel("Count"); ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        str(c), ha="center", va="bottom", fontsize=8)
            ax.grid(axis="y", alpha=0.3); fig.tight_layout()
            st.pyplot(fig); plt.close()

        # Severity distribution
        with col_r:
            st.subheader("Defect Severity Distribution")
            severities = [w["defect"]["severity"] for w in wafers if w["defect"]["has_defect"]]
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(severities, bins=40, color="#3498db", alpha=0.8, edgecolor="white")
            ax.set_xlabel("Severity [0–1]"); ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.3); fig.tight_layout()
            st.pyplot(fig); plt.close()

        st.divider()
        st.subheader("Sample Wafer Images")
        if IMAGE_DIR.exists():
            imgs = sorted(IMAGE_DIR.glob("W_*.png"))[:8]
            ann  = load_annotations()
            ann_idx = {a["image_id"]: a for a in ann["annotations"]}
            img_id_map = {img["file_name"]: img["id"] for img in ann["images"]}

            cols = st.columns(8)
            for col, img_path in zip(cols, imgs):
                pil = Image.open(img_path).resize((128, 128))
                img_id = img_id_map.get(img_path.name, -1)
                ann_entry = ann_idx.get(img_id)
                label = DEFECT_NAMES[ann_entry["category_id"]] if ann_entry else "none"
                col.image(pil, caption=label, use_column_width=True)
        else:
            st.info("Wafer images not found locally (gitignored). Generate with `python3 data/image_generator.py`.")

        st.divider()
        st.subheader("Process Step Feature Statistics")
        step_names = ["oxidation","lithography","etching","deposition",
                      "doping","cmp","cleaning","annealing"]
        feat_matrix = np.array([w["node_features"] for w in wafers])  # [N, 8, 3]
        means = feat_matrix.mean(axis=0)
        stds  = feat_matrix.std(axis=0)
        rows  = []
        for i, step in enumerate(step_names):
            rows.append({"Step": step,
                         "Feat 0 mean": f"{means[i,0]:.3f}", "Feat 0 std": f"{stds[i,0]:.3f}",
                         "Feat 1 mean": f"{means[i,1]:.3f}", "Feat 1 std": f"{stds[i,1]:.3f}",
                         "Feat 2 mean": f"{means[i,2]:.3f}", "Feat 2 std": f"{stds[i,2]:.3f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GNN Results
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("GNN — Defect Prediction Results")

    gnn_m = load_gnn_metrics()
    test  = gnn_m["test"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type Accuracy",   f"{test['type_accuracy']*100:.2f}%", delta="target >85%")
    c2.metric("Location MSE",    f"{test['location_mse']:.4f}",       delta="target <0.05")
    c3.metric("Severity RMSE",   f"{test['severity_rmse']:.4f}",      delta="target <0.6")
    c4.metric("Inference",       f"{test['avg_inference_ms']:.1f} ms",delta="target <30ms")

    st.divider()
    hist = gnn_m["history"]
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Loss Curves")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(epochs, hist["train_loss"], label="Train", color="#3498db")
        ax.plot(epochs, hist["val_loss"],   label="Val",   color="#e74c3c")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Type Accuracy")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(epochs, hist["train_type_acc"], label="Train", color="#3498db")
        ax.plot(epochs, hist["val_type_acc"],   label="Val",   color="#e74c3c")
        ax.axhline(0.85, color="green", linestyle="--", alpha=0.6, label="Target 85%")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Confusion Matrix")
    _confusion_img(RESULTS / "gnn_confusion_matrix.png", "GNN — Predicted vs True defect type")

    with st.expander("Training history table"):
        df = pd.DataFrame({
            "Epoch":        epochs,
            "Train Loss":   [f"{v:.4f}" for v in hist["train_loss"]],
            "Val Loss":     [f"{v:.4f}" for v in hist["val_loss"]],
            "Train Acc":    [f"{v:.3f}" for v in hist["train_type_acc"]],
            "Val Acc":      [f"{v:.3f}" for v in hist["val_type_acc"]],
            "LR":           hist["lr"],
        })
        st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CNN Results
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("CNN — Defect Detection Results")

    cnn_m = load_cnn_metrics()
    test  = cnn_m["test"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision",     f"{test['precision']:.4f}")
    c2.metric("Recall",        f"{test['recall']:.4f}",  delta="target >0.75")
    c3.metric("F1",            f"{test['f1']:.4f}")
    c4.metric("Cls Accuracy",  f"{test['cls_acc']:.4f}", delta="target >0.80")
    c5.metric("Inference",     f"{test['inference_ms']:.1f} ms", delta="target <150ms")

    c1b, c2b, c3b = st.columns(3)
    c1b.metric("TP", test["tp"])
    c2b.metric("FP", test["fp"])
    c3b.metric("FN", test["fn"])

    st.divider()
    hist = cnn_m["history"]
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Loss Curves")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(epochs, hist["train_loss"], label="Train", color="#3498db")
        ax.plot(epochs, hist["val_loss"],   label="Val",   color="#e74c3c")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Detection Metrics")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(epochs, hist["val_precision"], label="Precision", color="#9b59b6")
        ax.plot(epochs, hist["val_recall"],    label="Recall",    color="#3498db")
        ax.plot(epochs, hist["val_f1"],        label="F1",        color="#2ecc71")
        ax.axhline(0.75, color="red", linestyle="--", alpha=0.5, label="Target 75%")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Confusion Matrix")
    _confusion_img(RESULTS / "cnn_confusion_matrix.png", "CNN — Predicted vs True defect type (IoU ≥ 0.5)")

    with st.expander("Training history table"):
        df = pd.DataFrame({
            "Epoch":      epochs,
            "Train Loss": [f"{v:.4f}" for v in hist["train_loss"]],
            "Val Loss":   [f"{v:.4f}" for v in hist["val_loss"]],
            "Val F1":     [f"{v:.4f}" for v in hist["val_f1"]],
            "Val Recall": [f"{v:.4f}" for v in hist["val_recall"]],
            "LR":         hist["lr"],
        })
        st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Integration
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("GNN-CNN Integration Analysis")

    al = load_alignment_metrics()

    c1, c2, c3, c4 = st.columns(4)
    align_ok = al["alignment_rate"] >= 0.65
    fp_ok    = al["fp_rate"] < 0.30
    fn_ok    = al["fn_rate"] < 0.25
    c1.metric("Type Alignment",  f"{al['alignment_rate']*100:.1f}%",
              delta="✅ >65% target" if align_ok else "❌ below target")
    c2.metric("Full Alignment",  f"{al['full_alignment_rate']*100:.1f}%")
    c3.metric("FP Rate",         f"{al['fp_rate']*100:.1f}%",
              delta="✅ <30% target" if fp_ok else "❌ above target")
    c4.metric("FN Rate",         f"{al['fn_rate']*100:.1f}%",
              delta="✅ <25% target" if fn_ok else "❌ above target")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Wafer Outcome Distribution")
        labels = ["True Negative", "Aligned", "Type Mismatch",
                  "Loc Mismatch",  "GNN Only", "CNN Only"]
        counts = [al["true_negative"], al["aligned"],   al["type_mismatch"],
                  al["loc_mismatch"],  al["gnn_only"],  al["cnn_only"]]
        colors = ["#2ecc71","#3498db","#e74c3c","#e67e22","#9b59b6","#f39c12"]
        nz = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]
        c_, l_, col_ = zip(*nz)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(c_, labels=l_, colors=col_, autopct="%1.1f%%", startangle=90)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.subheader("Outcome Counts")
        df = pd.DataFrame({"Outcome": labels, "Count": counts,
                           "% of Test": [f"{c/al['n_test']*100:.1f}%" for c in counts]})
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Targets vs Achieved")
        tgt_df = pd.DataFrame({
            "Metric":   ["Type Alignment", "FP Rate", "FN Rate"],
            "Target":   [">65%", "<30%", "<25%"],
            "Achieved": [f"{al['alignment_rate']*100:.1f}%",
                         f"{al['fp_rate']*100:.1f}%",
                         f"{al['fn_rate']*100:.1f}%"],
            "Status":   ["✅" if align_ok else "❌",
                         "✅" if fp_ok    else "❌",
                         "✅" if fn_ok    else "❌"],
        })
        st.dataframe(tgt_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Alignment Analysis Plot")
    _confusion_img(RESULTS / "alignment_analysis.png", "Alignment breakdown")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Parameter Correlations
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Parameter-Defect Correlation Explorer")

    st.markdown("""
    The heatmap below shows the **class-conditional mean** of each process parameter
    for each defect type. A bright cell means wafers with that defect type tend to have
    elevated values of that parameter. ★ markers indicate the known causal rules from
    the simulator — they should align with the brightest cells.
    """)

    corr_path = RESULTS / "parameter_correlation.png"
    if corr_path.exists():
        st.image(str(corr_path), use_column_width=True)
    else:
        st.info("Run `python3 evaluation/analyze_alignment.py` to generate the correlation plot.")

    st.divider()
    st.subheader("Causal Rules Reference")
    rules = pd.DataFrame({
        "Defect Type":    ["oxide_defect", "scratch", "particle_contamination",
                           "pit", "metal_contamination"],
        "Process Step":   ["oxidation", "CMP", "cleaning", "doping", "deposition"],
        "Trigger":        ["temperature > 1100 AND duration > 90",
                           "pressure > 8 AND slurry_conc < 0.1",
                           "chemical_conc < 0.5",
                           "concentration > 5e17",
                           "rate > 4.0"],
    })
    st.dataframe(rules, use_container_width=True, hide_index=True)

    if DATA_RAW.exists():
        st.divider()
        st.subheader("Interactive Feature Explorer")
        wafers = load_wafer_data()
        step_names = ["oxidation","lithography","etching","deposition",
                      "doping","cmp","cleaning","annealing"]
        param_names = [
            ["temperature","pressure","duration"],
            ["exposure_dose","focus_offset","wavelength"],
            ["etch_rate","selectivity","duration"],
            ["temperature","rate","thickness"],
            ["concentration","energy","dose"],
            ["pressure","velocity","slurry_conc"],
            ["chemical_conc","temperature","duration"],
            ["temperature","duration","atmosphere"],
        ]

        col1, col2 = st.columns(2)
        sel_step  = col1.selectbox("Process step", step_names)
        step_idx  = step_names.index(sel_step)
        sel_param = col2.selectbox("Parameter", param_names[step_idx])
        param_idx = param_names[step_idx].index(sel_param)

        feat_by_type = {i: [] for i in range(len(DEFECT_NAMES))}
        for w in wafers:
            t = w["defect"]["defect_type"]
            feat_by_type[t].append(w["node_features"][step_idx][param_idx])

        fig, ax = plt.subplots(figsize=(8, 3))
        for t in range(len(DEFECT_NAMES)):
            vals = feat_by_type[t]
            if vals:
                ax.hist(vals, bins=30, alpha=0.5, label=DEFECT_NAMES[t],
                        color=DEFECT_COLORS[t], density=True)
        ax.set_xlabel(f"{sel_step} / {sel_param} (normalised)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3); fig.tight_layout()
        st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Wafer Inspector
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Wafer Inspector")
    st.markdown("Select a wafer to see the GNN's parameter-based prediction overlaid with the CNN's visual detection.")

    if not IMAGE_DIR.exists():
        st.warning("Wafer images not found locally. They are gitignored — generate them with `python3 data/image_generator.py`.")
    elif not GNN_CKPT.exists() or not CNN_CKPT.exists():
        st.warning("Model checkpoints not found. They are gitignored — train the models first.")
    else:
        import torch
        import torchvision.transforms.functional as TF

        ann_data   = load_annotations()
        img_lookup = {img["wafer_id"]: img for img in ann_data["images"]}
        ann_lookup = {}
        for a in ann_data["annotations"]:
            img_obj = next(i for i in ann_data["images"] if i["id"] == a["image_id"])
            wid = img_obj["wafer_id"]
            ann_lookup.setdefault(wid, []).append(a)

        wafer_ids = sorted(img_lookup.keys())

        col1, col2 = st.columns([1, 3])
        with col1:
            selected = st.selectbox("Wafer ID", wafer_ids)
            run_btn  = st.button("Run both models")

        with col2:
            if run_btn and selected:
                with st.spinner("Loading models and running inference..."):
                    gnn_model, cnn_model, device = load_models()

                    # ── Load GNN graph for this wafer ────────────────
                    wafers = load_wafer_data()
                    wafer  = next(w for w in wafers if w["wafer_id"] == selected)
                    from torch_geometric.data import Data

                    x = torch.tensor(wafer["node_features"], dtype=torch.float)
                    n_steps = x.shape[0]
                    x = torch.cat([x, torch.eye(n_steps)], dim=1)
                    edges = wafer["adjacency"]
                    if edges:
                        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    else:
                        edge_index = torch.zeros((2, 0), dtype=torch.long)

                    batch_vec = torch.zeros(n_steps, dtype=torch.long)
                    with torch.no_grad():
                        tl, lp, sp = gnn_model(
                            x.to(device), edge_index.to(device), batch_vec.to(device)
                        )
                    gnn_type = int(tl.argmax(dim=-1).item())
                    gnn_loc  = lp[0].cpu().numpy()
                    gnn_sev  = float(sp[0].squeeze().item())

                    # ── Load image and run CNN ───────────────────────
                    img_path = IMAGE_DIR / img_lookup[selected]["file_name"]
                    pil_img  = Image.open(img_path).convert("RGB")
                    img_t    = TF.to_tensor(pil_img).to(device)
                    with torch.no_grad():
                        dets = cnn_model([img_t])[0]

                    boxes  = dets["boxes"].cpu().numpy()
                    labels = dets["labels"].cpu().numpy()
                    scores = dets["scores"].cpu().numpy()

                    # ── Draw overlay ─────────────────────────────────
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(np.array(pil_img))

                    # GNN location (×)
                    gx = gnn_loc[0] * IMG_SIZE
                    gy = gnn_loc[1] * IMG_SIZE
                    gcolor = DEFECT_COLORS[gnn_type % len(DEFECT_COLORS)]
                    ax.plot(gx, gy, "x", color=gcolor, markersize=20,
                            markeredgewidth=3,
                            label=f"GNN: {DEFECT_NAMES[gnn_type]} (sev={gnn_sev:.2f})")

                    # CNN boxes
                    for box, lbl, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box
                        lbl0 = int(lbl) - 1
                        c    = DEFECT_COLORS[lbl0 % len(DEFECT_COLORS)]
                        rect = mpatches.Rectangle(
                            (x1, y1), x2-x1, y2-y1,
                            linewidth=2, edgecolor=c, facecolor="none"
                        )
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 4, f"{DEFECT_NAMES[lbl0]} {score:.2f}",
                                color=c, fontsize=8,
                                bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

                    ax.legend(loc="upper right", fontsize=8,
                              bbox_to_anchor=(1, 1), framealpha=0.8)
                    ax.set_title(f"{selected}"); ax.axis("off")
                    fig.tight_layout()
                    st.pyplot(fig); plt.close()

                # ── Side panel: ground truth + predictions ───────────
                st.divider()
                c1, c2, c3 = st.columns(3)

                gt_defect = wafer["defect"]
                c1.markdown("**Ground Truth**")
                c1.write(f"Type: `{gt_defect['defect_type_name']}`")
                c1.write(f"Location: `({gt_defect['location_x']:.3f}, {gt_defect['location_y']:.3f})`")
                c1.write(f"Severity: `{gt_defect['severity']:.3f}`")

                c2.markdown("**GNN Prediction**")
                c2.write(f"Type: `{DEFECT_NAMES[gnn_type]}`")
                c2.write(f"Location: `({gnn_loc[0]:.3f}, {gnn_loc[1]:.3f})`")
                c2.write(f"Severity: `{gnn_sev:.3f}`")

                c3.markdown("**CNN Detections**")
                if len(boxes) == 0:
                    c3.write("No detections above threshold")
                else:
                    for lbl, score in zip(labels, scores):
                        c3.write(f"`{DEFECT_NAMES[int(lbl)-1]}` — score {score:.3f}")
            else:
                st.info("Select a wafer ID and click **Run both models** to inspect it.")
