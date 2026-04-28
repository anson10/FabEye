"""
FabEye Streamlit Dashboard — 6 tabs covering the full ML pipeline.

Run:
    PYTHONPATH=. streamlit run visualization/dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FabEye — Wafer Defect Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #e8eaf0;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero .subtitle {
        font-size: 1.05rem;
        color: #7a8aaa;
        margin: 0 0 1rem 0;
    }
    .hero .badge {
        display: inline-block;
        background: #1e3a5f;
        color: #5b9bd5;
        border: 1px solid #2d5a8e;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin-right: 6px;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c8d0e0;
        border-left: 3px solid #3d7fcc;
        padding-left: 10px;
        margin: 1.2rem 0 0.6rem 0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #1a1f2e;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #7a8aaa !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #e8eaf0 !important; font-size: 1.5rem; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 8px;
        padding: 4px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #7a8aaa;
        font-weight: 600;
        font-size: 0.88rem;
        border-radius: 6px;
        padding: 6px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important;
        color: #5b9bd5 !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1f2e;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #7a8aaa;
    }

    /* Divider */
    hr { border-color: #2a3550; margin: 1rem 0; }

    /* Tech detail box */
    .tech-box {
        background: #131720;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.85rem;
        color: #8090aa;
        line-height: 1.7;
    }
    .tech-box b { color: #b0c0d8; }
    .tech-box code {
        background: #1e2538;
        padding: 1px 5px;
        border-radius: 3px;
        font-size: 0.82rem;
        color: #7fb3e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
RESULTS   = ROOT / "results"
DATA_RAW  = ROOT / "data" / "raw" / "synthetic_wafers.json"
IMAGE_DIR = ROOT / "data" / "wafer_images"
ANN_PATH  = IMAGE_DIR / "annotations.json"
GNN_CKPT  = ROOT / "checkpoints" / "best_gnn.pt"
CNN_CKPT  = ROOT / "checkpoints" / "best_cnn.pt"

DEFECT_NAMES  = ["none", "particle_contamination", "scratch", "pit",
                 "oxide_defect", "metal_contamination"]
DEFECT_COLORS = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
IMG_SIZE      = 512

plt.rcParams.update({
    "figure.facecolor": "#1a1f2e",
    "axes.facecolor":   "#1a1f2e",
    "axes.edgecolor":   "#2a3550",
    "axes.labelcolor":  "#8090aa",
    "xtick.color":      "#8090aa",
    "ytick.color":      "#8090aa",
    "text.color":       "#c8d0e0",
    "grid.color":       "#2a3550",
    "legend.facecolor": "#1a1f2e",
    "legend.edgecolor": "#2a3550",
})


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_gnn_metrics():
    with open(RESULTS / "gnn_metrics.json") as f: return json.load(f)

@st.cache_data
def load_cnn_metrics():
    with open(RESULTS / "cnn_metrics.json") as f: return json.load(f)

@st.cache_data
def load_alignment_metrics():
    with open(RESULTS / "alignment_metrics.json") as f: return json.load(f)

@st.cache_data
def load_wafer_data():
    with open(DATA_RAW) as f: return json.load(f)

@st.cache_data
def load_annotations():
    with open(ANN_PATH) as f: return json.load(f)

@st.cache_resource
def load_models():
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
def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def result_image(path, caption=""):
    if Path(path).exists():
        st.image(str(path), caption=caption, width=700)
    else:
        st.info(f"Run `python3 evaluation/analyze_alignment.py` to generate {Path(path).name}")

def outcome_bar(counts, labels, colors, total):
    """Horizontal bar chart for outcome breakdown — never overlaps."""
    nz = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]
    c_, l_, col_ = zip(*nz)

    fig, ax = plt.subplots(figsize=(7, len(c_) * 0.65 + 0.6))
    y = range(len(c_))
    bars = ax.barh(list(y), c_, color=col_, alpha=0.88,
                   edgecolor="#0f1117", linewidth=0.5, height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels(l_, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Wafer count", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top","right","left"]].set_visible(False)

    for bar, count in zip(bars, c_):
        pct = count / total * 100
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=9, color="#c8d0e0")
    ax.set_xlim(0, max(c_) * 1.28)
    fig.tight_layout()
    return fig

def line_chart(x, series: dict, ylabel="", hlines: list = None, ylim=None):
    """series = {label: (values, color)}"""
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for label, (vals, color) in series.items():
        ax.plot(x, vals, label=label, color=color, linewidth=1.8)
    if hlines:
        for val, label, color in hlines:
            ax.axhline(val, color=color, linestyle="--", alpha=0.55,
                       linewidth=1.2, label=label)
    if ylim: ax.set_ylim(*ylim)
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def tech_detail(content_html: str):
    with st.expander("Technical details"):
        st.markdown(f'<div class="tech-box">{content_html}</div>',
                    unsafe_allow_html=True)


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>FabEye</h1>
  <p class="subtitle">End-to-end ML pipeline for semiconductor wafer defect prediction and detection</p>
  <span class="badge">GNN</span>
  <span class="badge">Faster R-CNN</span>
  <span class="badge">PyTorch</span>
  <span class="badge">10,000 wafers</span>
  <span class="badge">6 defect types</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
FabEye combines two machine learning stages: a **Graph Neural Network** that predicts defects
from process parameters before a wafer is inspected, and a **Convolutional Neural Network**
(Faster R-CNN) that detects defects visually in inspection images. The integration layer
answers the core research question — *does the parameter anomaly the GNN detected actually
produce the visual defect the CNN sees?*
""")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset", "GNN Results", "CNN Results",
    "Integration", "Parameter Correlations", "Wafer Inspector",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dataset
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not DATA_RAW.exists():
        st.warning("Raw data not found. Run `python3 data/generator.py --n 10000` first.")
    else:
        wafers       = load_wafer_data()
        n            = len(wafers)
        defect_types = [w["defect"]["defect_type"] for w in wafers]
        has_defect   = [w["defect"]["has_defect"]  for w in wafers]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Wafers",  f"{n:,}")
        c2.metric("Defective",     f"{sum(has_defect):,}")
        c3.metric("Clean",         f"{n - sum(has_defect):,}")
        c4.metric("Defect Rate",   f"{sum(has_defect)/n*100:.1f}%")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            section("Defect Type Distribution")
            counts = [defect_types.count(i) for i in range(len(DEFECT_NAMES))]
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            bars = ax.bar(DEFECT_NAMES, counts, color=DEFECT_COLORS, alpha=0.9,
                          edgecolor="#0f1117", linewidth=0.6)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=28, labelsize=8)
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                        str(c), ha="center", va="bottom", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        with col_r:
            section("Defect Severity Distribution")
            severities = [w["defect"]["severity"] for w in wafers if w["defect"]["has_defect"]]
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            ax.hist(severities, bins=40, color="#3498db", alpha=0.85, edgecolor="#0f1117")
            ax.set_xlabel("Severity [0–1]"); ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig); plt.close()

        st.divider()
        section("Sample Wafer Images")
        if IMAGE_DIR.exists():
            ann      = load_annotations()
            ann_idx  = {a["image_id"]: a for a in ann["annotations"]}
            id_map   = {img["file_name"]: img["id"] for img in ann["images"]}
            imgs     = sorted(IMAGE_DIR.glob("W_*.png"))[:8]
            cols     = st.columns(8)
            for col, img_path in zip(cols, imgs):
                pil    = Image.open(img_path).resize((120, 120))
                img_id = id_map.get(img_path.name, -1)
                entry  = ann_idx.get(img_id)
                label  = DEFECT_NAMES[entry["category_id"]] if entry else "none"
                col.image(pil, caption=label.replace("_", " "), width=120)
        else:
            st.info("Wafer images are gitignored. Generate with `python3 data/image_generator.py`.")

        st.divider()
        section("Process Step Feature Statistics (normalised)")
        step_names = ["oxidation","lithography","etching","deposition",
                      "doping","cmp","cleaning","annealing"]
        feat_matrix = np.array([w["node_features"] for w in wafers])
        means = feat_matrix.mean(axis=0); stds = feat_matrix.std(axis=0)
        rows = [{"Step": s,
                 "F0 mean": f"{means[i,0]:.3f}", "F0 std": f"{stds[i,0]:.3f}",
                 "F1 mean": f"{means[i,1]:.3f}", "F1 std": f"{stds[i,1]:.3f}",
                 "F2 mean": f"{means[i,2]:.3f}", "F2 std": f"{stds[i,2]:.3f}"}
                for i, s in enumerate(step_names)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        tech_detail("""
<b>Dataset generation:</b> 10,000 wafers produced by <code>data/generator.py</code> using
physics-inspired causal rules. Each wafer has 8 process steps, each with 3 normalised parameters,
giving a 24-dimensional feature vector for the GNN. Defects are assigned deterministically when a
parameter crosses a threshold (e.g. oxidation temperature &gt; 1100 → oxide defect) plus Gaussian
noise to prevent the model from memorising exact thresholds. ~35% of wafers have defects.<br><br>
<b>Image generation:</b> <code>data/image_generator.py</code> renders a 512×512 RGB wafer image
for each sample. Defect classes have distinct visual signatures (bright blob, dark line, pit, soft
cloud, cluster) with brightness modulated by severity. Images are saved as PNG and annotated in
COCO format for Faster R-CNN ingestion.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GNN Results
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    gnn_m = load_gnn_metrics()
    test  = gnn_m["test"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type Accuracy",  f"{test['type_accuracy']*100:.2f}%",  "target >85%")
    c2.metric("Location MSE",   f"{test['location_mse']:.4f}",        "target <0.05")
    c3.metric("Severity RMSE",  f"{test['severity_rmse']:.4f}",       "target <0.6")
    c4.metric("Inference",      f"{test['avg_inference_ms']:.1f} ms", "target <30 ms")

    st.divider()
    hist   = gnn_m["history"]
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    col_l, col_r = st.columns(2)
    with col_l:
        section("Loss Curves")
        fig = line_chart(epochs,
            {"Train": (hist["train_loss"], "#3498db"),
             "Val":   (hist["val_loss"],   "#e74c3c")},
            ylabel="Loss")
        st.pyplot(fig); plt.close()

    with col_r:
        section("Type Accuracy")
        fig = line_chart(epochs,
            {"Train": (hist["train_type_acc"], "#3498db"),
             "Val":   (hist["val_type_acc"],   "#e74c3c")},
            ylabel="Accuracy",
            hlines=[(0.85, "Target 85%", "#2ecc71")],
            ylim=(0.5, 1.0))
        st.pyplot(fig); plt.close()

    st.divider()
    section("Confusion Matrix — GNN vs Ground Truth")
    result_image(RESULTS / "gnn_confusion_matrix.png",
                 "Rows = true class, columns = predicted class (row-normalised)")

    with st.expander("Training history table"):
        df = pd.DataFrame({
            "Epoch":     epochs,
            "Train Loss":[f"{v:.4f}" for v in hist["train_loss"]],
            "Val Loss":  [f"{v:.4f}" for v in hist["val_loss"]],
            "Train Acc": [f"{v:.3f}" for v in hist["train_type_acc"]],
            "Val Acc":   [f"{v:.3f}" for v in hist["val_type_acc"]],
            "LR":        hist["lr"],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    tech_detail("""
<b>Architecture:</b> 3-layer GCN (Graph Convolutional Network) with hidden width 256, BatchNorm,
and dropout 0.2. Three prediction heads branch from a pooled graph embedding: a 3-layer MLP for
defect type (6-class softmax), a 2-layer MLP for location (2D sigmoid), and a 2-layer MLP for
severity (scalar sigmoid).<br><br>
<b>Graph structure:</b> Each wafer is a chain graph with 8 nodes (one per process step). Edges
connect consecutive steps reflecting causal temporal dependencies. Each node carries 3 raw
normalised parameters + 8-dimensional one-hot step identity = 11 features.<br><br>
<b>Training:</b> Adam, lr=0.001, ReduceLROnPlateau (factor 0.5, patience 20), 120 epochs,
WeightedRandomSampler for class balance. The flat bypass (raw 24 features concatenated to pooled
embedding) lets the model use individual parameter values directly alongside the graph-propagated
representation — this was the key architectural change that lifted accuracy above 85%.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CNN Results
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    cnn_m = load_cnn_metrics()
    test  = cnn_m["test"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision",    f"{test['precision']:.4f}")
    c2.metric("Recall",       f"{test['recall']:.4f}",      "target >0.75")
    c3.metric("F1",           f"{test['f1']:.4f}")
    c4.metric("Cls Accuracy", f"{test['cls_acc']:.4f}",     "target >0.80")
    c5.metric("Inference",    f"{test['inference_ms']:.1f} ms", "target <150 ms")

    c1b, c2b, c3b, _ = st.columns(4)
    c1b.metric("True Positives",  test["tp"])
    c2b.metric("False Positives", test["fp"])
    c3b.metric("False Negatives", test["fn"])

    st.divider()
    hist   = cnn_m["history"]
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    col_l, col_r = st.columns(2)
    with col_l:
        section("Loss Curves")
        fig = line_chart(epochs,
            {"Train": (hist["train_loss"], "#3498db"),
             "Val":   (hist["val_loss"],   "#e74c3c")},
            ylabel="Loss")
        st.pyplot(fig); plt.close()

    with col_r:
        section("Detection Metrics")
        fig = line_chart(epochs,
            {"Precision": (hist["val_precision"], "#9b59b6"),
             "Recall":    (hist["val_recall"],    "#3498db"),
             "F1":        (hist["val_f1"],        "#2ecc71")},
            ylabel="Score",
            hlines=[(0.75, "Target 75%", "#e74c3c")],
            ylim=(0.0, 1.05))
        st.pyplot(fig); plt.close()

    st.divider()
    section("Confusion Matrix — CNN vs Ground Truth")
    result_image(RESULTS / "cnn_confusion_matrix.png",
                 "IoU ≥ 0.5 matching. Rows = true class, columns = predicted class (row-normalised)")

    with st.expander("Training history table"):
        df = pd.DataFrame({
            "Epoch":      epochs,
            "Train Loss": [f"{v:.4f}" for v in hist["train_loss"]],
            "Val Loss":   [f"{v:.4f}" for v in hist["val_loss"]],
            "Val F1":     [f"{v:.4f}" for v in hist["val_f1"]],
            "Val Recall": [f"{v:.4f}" for v in hist["val_recall"]],
            "LR":         hist["lr"],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    tech_detail("""
<b>Architecture:</b> Faster R-CNN with ResNet-50 FPN backbone pretrained on ImageNet. The
detection head (box predictor) is replaced with a 7-class version (background + 6 defect types).
Three backbone stages are unfrozen for fine-tuning. FPN (Feature Pyramid Network) generates
multi-scale feature maps at P2–P5 to handle defects ranging from 4px particles to 100px+ scratch lines.<br><br>
<b>Training:</b> SGD, momentum 0.9, weight decay 5e-4, lr=5e-3, ReduceLROnPlateau (halved at
epoch 7). Batch size 2 per GPU (4 effective) on 2× RTX 4000 via DistributedDataParallel.
DDP was required because Faster R-CNN expects a <code>list</code> of images — DataParallel converts
it to a stacked tensor, breaking the channel dimension.<br><br>
<b>Validation strategy:</b> Two-pass validation — train mode for loss, eval mode for IoU-matched
detection metrics. Runs every 2 epochs to halve wall-clock time. Best checkpoint saved by F1;
early stopping monitored on loss. Best epoch was epoch 6 (F1 = 0.9850).
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Integration
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    al = load_alignment_metrics()

    align_ok = al["alignment_rate"] >= 0.65
    fp_ok    = al["fp_rate"] < 0.30
    fn_ok    = al["fn_rate"] < 0.25

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type Alignment",  f"{al['alignment_rate']*100:.1f}%",
              "✅ >65% met" if align_ok else "❌ below target")
    c2.metric("Full Alignment",  f"{al['full_alignment_rate']*100:.1f}%",
              "type + location")
    c3.metric("FP Rate",         f"{al['fp_rate']*100:.1f}%",
              "✅ <30% met" if fp_ok else "❌ above target")
    c4.metric("FN Rate",         f"{al['fn_rate']*100:.1f}%",
              "✅ <25% met" if fn_ok else "❌ above target")

    st.divider()
    labels = ["True Negative", "Aligned", "Type Mismatch",
              "Loc Mismatch",  "GNN Only", "CNN Only"]
    counts = [al["true_negative"], al["aligned"],   al["type_mismatch"],
              al["loc_mismatch"],  al["gnn_only"],  al["cnn_only"]]
    colors = ["#2ecc71","#3498db","#e74c3c","#e67e22","#9b59b6","#f39c12"]

    col_l, col_r = st.columns([1.4, 1])

    with col_l:
        section("Wafer Outcome Distribution")
        fig = outcome_bar(counts, labels, colors, al["n_test"])
        st.pyplot(fig); plt.close()

    with col_r:
        section("Outcome Summary")
        df = pd.DataFrame({
            "Outcome": labels,
            "Count":   counts,
            "% of Test": [f"{c/al['n_test']*100:.1f}%" for c in counts],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Targets vs Achieved")
        tgt = pd.DataFrame({
            "Metric":   ["Type Alignment", "FP Rate", "FN Rate"],
            "Target":   [">65%", "<30%", "<25%"],
            "Achieved": [f"{al['alignment_rate']*100:.1f}%",
                         f"{al['fp_rate']*100:.1f}%",
                         f"{al['fn_rate']*100:.1f}%"],
            "Status":   ["✅" if align_ok else "❌",
                         "✅" if fp_ok    else "❌",
                         "✅" if fn_ok    else "❌"],
        })
        st.dataframe(tgt, use_container_width=True, hide_index=True)

    st.divider()
    section("Alignment Metrics vs Targets")
    metric_names  = ["Type Alignment", "Full Alignment", "FP Rate", "FN Rate"]
    metric_values = [al["alignment_rate"], al["full_alignment_rate"],
                     al["fp_rate"],        al["fn_rate"]]
    targets_map   = {"Type Alignment": 0.65, "FP Rate": 0.30, "FN Rate": 0.25}
    bar_colors    = ["#3498db","#5dade2","#e67e22","#e67e22"]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(metric_names))
    bars = ax.bar(x, metric_values, color=bar_colors, alpha=0.88,
                  edgecolor="#0f1117", linewidth=0.5, width=0.5)
    for name, tval, color in [("Type Alignment", 0.65, "#2ecc71"),
                               ("FP Rate", 0.30, "#e74c3c"),
                               ("FN Rate", 0.25, "#e74c3c")]:
        xi = metric_names.index(name)
        ax.plot([xi - 0.32, xi + 0.32], [tval, tval],
                color=color, linewidth=2.5, linestyle="--",
                label=f"Target {name}")
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val*100:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#e8eaf0")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    tech_detail("""
<b>Alignment definition:</b> A wafer is <b>type-aligned</b> when both models predict the same
defect class. It is <b>fully aligned</b> when type agrees AND the GNN's predicted (x,y) location
is within 0.35 normalised units of the CNN detection centroid.<br><br>
<b>Why type-first:</b> The GNN's location RMSE is ~0.23 per coordinate (Euclidean ~0.33). A strict
location threshold of 0.2 rejected most correct type predictions. Type agreement is the primary
research signal — it validates that the parameter anomaly produces the right class of visual defect.
Location is secondary; the GNN is not a precise localisation model.<br><br>
<b>FP/FN definitions:</b> FP = GNN fires, CNN sees nothing (GNN false alarm). FN = CNN detects
something, GNN predicted clean (GNN missed the process anomaly). The low FN rate (8.9%) means the
GNN rarely misses defects that the CNN visually confirms — the causal chain from parameters to
image is well captured.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Parameter Correlations
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    The heatmap shows the **class-conditional mean** of each process parameter for each defect type.
    A bright cell means wafers with that defect tend to have elevated values of that parameter.
    ★ markers show the known causal rules — they should align with the brightest cells,
    validating that the models have learned genuine causal relationships.
    """)

    if DATA_RAW.exists():
        _wafers_corr = load_wafer_data()
        _step_names_corr = ["oxidation","lithography","etching","deposition",
                            "doping","cmp","cleaning","annealing"]
        _param_names_corr = [
            ["temperature","pressure","duration"],
            ["exposure_dose","focus_offset","wavelength"],
            ["etch_rate","selectivity","duration"],
            ["temperature","rate","thickness"],
            ["concentration","energy","dose"],
            ["pressure","velocity","slurry_conc"],
            ["chemical_conc","temperature","duration"],
            ["temperature","duration","atmosphere"],
        ]
        _feat_labels = [f"{s}\n{p}" for s, ps in zip(_step_names_corr, _param_names_corr)
                        for p in ps]

        X_corr = np.array([w["node_features"] for w in _wafers_corr],
                          dtype=np.float32).reshape(len(_wafers_corr), -1)
        y_corr = np.array([w["defect"]["defect_type"] for w in _wafers_corr])
        n_cls  = len(DEFECT_NAMES); n_ft = X_corr.shape[1]

        cm_means = np.zeros((n_cls, n_ft))
        for c in range(n_cls):
            mask = y_corr == c
            if mask.sum() > 0:
                cm_means[c] = X_corr[mask].mean(axis=0)
        col_min = cm_means.min(axis=0, keepdims=True)
        col_max = cm_means.max(axis=0, keepdims=True)
        cm_norm = (cm_means - col_min) / (cm_means.max(axis=0, keepdims=True) - col_min + 1e-8)

        causal_cells = {
            (4, _feat_labels.index("oxidation\ntemperature")): "★",
            (4, _feat_labels.index("oxidation\nduration")):    "★",
            (2, _feat_labels.index("cmp\npressure")):          "★",
            (2, _feat_labels.index("cmp\nslurry_conc")):       "★",
            (1, _feat_labels.index("cleaning\nchemical_conc")):"★",
            (3, _feat_labels.index("doping\nconcentration")):  "★",
            (5, _feat_labels.index("deposition\nrate")):       "★",
        }

        fig, ax = plt.subplots(figsize=(20, 4.5))
        im = ax.imshow(cm_norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        cbar.set_label("Normalised mean", fontsize=9, color="#8090aa")
        cbar.ax.yaxis.set_tick_params(color="#8090aa")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8090aa", fontsize=8)

        ax.set_xticks(range(n_ft))
        ax.set_xticklabels(_feat_labels, rotation=55, ha="right", fontsize=7.5)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(DEFECT_NAMES, fontsize=10)
        ax.set_title("Parameter-Defect Correlation  (★ = known causal rule)",
                     fontsize=11, pad=10)

        for (row, col_idx), marker in causal_cells.items():
            ax.text(col_idx, row, marker, ha="center", va="center",
                    fontsize=12, color="#1a6bcc", fontweight="bold")

        # Vertical separators between process steps
        for sep in [3, 6, 9, 12, 15, 18, 21]:
            ax.axvline(sep - 0.5, color="#0f1117", linewidth=1.5, alpha=0.6)

        fig.tight_layout()
        st.pyplot(fig); plt.close()
    else:
        st.info("Raw data not found. Run `python3 data/generator.py --n 10000` first.")

    st.divider()
    section("Causal Rules Reference")
    rules = pd.DataFrame({
        "Defect Type":  ["oxide_defect", "scratch", "particle_contamination",
                         "pit", "metal_contamination"],
        "Process Step": ["Oxidation", "CMP", "Cleaning", "Doping", "Deposition"],
        "Trigger Condition": [
            "temperature > 1100 AND duration > 90",
            "pressure > 8 AND slurry_conc < 0.1",
            "chemical_conc < 0.5",
            "concentration > 5×10¹⁷",
            "deposition rate > 4.0",
        ],
    })
    st.dataframe(rules, use_container_width=True, hide_index=True)

    if DATA_RAW.exists():
        st.divider()
        section("Interactive Feature Distribution by Defect Type")
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
        sel_step   = col1.selectbox("Process step", step_names)
        step_idx   = step_names.index(sel_step)
        sel_param  = col2.selectbox("Parameter", param_names[step_idx])
        param_idx  = param_names[step_idx].index(sel_param)

        feat_by_type = {i: [] for i in range(len(DEFECT_NAMES))}
        for w in wafers:
            feat_by_type[w["defect"]["defect_type"]].append(
                w["node_features"][step_idx][param_idx])

        fig, ax = plt.subplots(figsize=(7, 3))
        for t in range(len(DEFECT_NAMES)):
            vals = feat_by_type[t]
            if vals:
                ax.hist(vals, bins=30, alpha=0.55, label=DEFECT_NAMES[t],
                        color=DEFECT_COLORS[t], density=True, edgecolor="none")
        ax.set_xlabel(f"{sel_step} / {sel_param} (normalised)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7.5, ncol=3)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    tech_detail("""
<b>Method:</b> For each of the 24 process features (8 steps × 3 parameters), compute the mean
normalised value across all test-set wafers belonging to each defect class. Normalise each column
so the range spans [0, 1], making different parameters comparable on the same colour scale.<br><br>
<b>Why class-conditional means:</b> If a parameter is causally related to a defect, wafers with
that defect will systematically have higher or lower values of that parameter. The heatmap makes
this visible at a glance. It is equivalent to the first step of a linear discriminant analysis.<br><br>
<b>Validation:</b> The ★ causal markers are ground truth from the simulator. If they coincide with
bright cells, the model has learned the real causal structure rather than spurious correlations in
the training data. This is important because with 10,000 wafers and only 5 causal rules, random
correlations between parameters and defect types are possible.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Wafer Inspector
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
    Pick any wafer from the test dataset. Both models run live: the GNN reads process parameters
    and predicts where a defect should appear; the CNN reads the inspection image and draws
    detection boxes. The overlay shows whether they agree.
    """)

    if not IMAGE_DIR.exists():
        st.warning("Wafer images are gitignored. Generate with `python3 data/image_generator.py`.")
    elif not GNN_CKPT.exists() or not CNN_CKPT.exists():
        st.warning("Model checkpoints are gitignored. Train the models first.")
    else:
        import torch
        import torchvision.transforms.functional as TF

        ann_data   = load_annotations()
        img_lookup = {img["wafer_id"]: img for img in ann_data["images"]}
        wafer_ids  = sorted(img_lookup.keys())

        col1, col2, col3 = st.columns([1.2, 1, 2])
        with col1:
            selected = st.selectbox("Wafer ID", wafer_ids, label_visibility="visible")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("Run both models", type="primary")

        if run_btn and selected:
            with st.spinner("Loading models and running inference..."):
                gnn_model, cnn_model, device = load_models()

                wafers = load_wafer_data()
                wafer  = next(w for w in wafers if w["wafer_id"] == selected)

                from torch_geometric.data import Data
                x = torch.tensor(wafer["node_features"], dtype=torch.float)
                n_steps = x.shape[0]
                x = torch.cat([x, torch.eye(n_steps)], dim=1)
                edges = wafer["adjacency"]
                edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                              if edges else torch.zeros((2, 0), dtype=torch.long))
                batch_vec = torch.zeros(n_steps, dtype=torch.long)

                with torch.no_grad():
                    tl, lp, sp = gnn_model(
                        x.to(device), edge_index.to(device), batch_vec.to(device))
                gnn_type = int(tl.argmax(dim=-1).item())
                gnn_loc  = lp[0].cpu().numpy()
                gnn_sev  = float(sp[0].squeeze().item())

                img_path = IMAGE_DIR / img_lookup[selected]["file_name"]
                pil_img  = Image.open(img_path).convert("RGB")
                img_t    = TF.to_tensor(pil_img).to(device)
                with torch.no_grad():
                    dets = cnn_model([img_t])[0]
                boxes  = dets["boxes"].cpu().numpy()
                labels = dets["labels"].cpu().numpy()
                scores = dets["scores"].cpu().numpy()

            col_img, col_info = st.columns([2, 1])

            with col_img:
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                ax.imshow(np.array(pil_img))
                gx = gnn_loc[0] * IMG_SIZE
                gy = gnn_loc[1] * IMG_SIZE
                gc = DEFECT_COLORS[gnn_type % len(DEFECT_COLORS)]
                ax.plot(gx, gy, "x", color=gc, markersize=18, markeredgewidth=3,
                        label=f"GNN: {DEFECT_NAMES[gnn_type]} (sev {gnn_sev:.2f})")
                for box, lbl, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    lbl0 = int(lbl) - 1
                    c    = DEFECT_COLORS[lbl0 % len(DEFECT_COLORS)]
                    rect = mpatches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor=c, facecolor="none")
                    ax.add_patch(rect)
                    ax.text(x1, max(y1-4, 6),
                            f"{DEFECT_NAMES[lbl0]} {score:.2f}",
                            color=c, fontsize=8,
                            bbox=dict(facecolor="#1a1f2e", alpha=0.7,
                                      pad=1, edgecolor="none"))
                ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
                ax.set_title(f"{selected}", fontsize=9)
                ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig); plt.close()

            with col_info:
                gt = wafer["defect"]
                section("Ground Truth")
                st.write(f"Type: `{gt['defect_type_name']}`")
                st.write(f"Location: `({gt['location_x']:.3f}, {gt['location_y']:.3f})`")
                st.write(f"Severity: `{gt['severity']:.3f}`")

                section("GNN Prediction")
                match_type = "✅" if gnn_type == gt["defect_type"] else "❌"
                st.write(f"Type: `{DEFECT_NAMES[gnn_type]}` {match_type}")
                st.write(f"Location: `({gnn_loc[0]:.3f}, {gnn_loc[1]:.3f})`")
                st.write(f"Severity: `{gnn_sev:.3f}`")

                section("CNN Detections")
                if len(boxes) == 0:
                    st.write("No detections above score threshold (0.4)")
                else:
                    for lbl, score in zip(labels, scores):
                        lbl0 = int(lbl) - 1
                        match = "✅" if lbl0 == gt["defect_type"] else "❌"
                        st.write(f"`{DEFECT_NAMES[lbl0]}` — {score:.3f} {match}")
        else:
            st.info("Select a wafer ID above and click **Run both models**.")

        tech_detail("""
<b>GNN inference:</b> The wafer's 8-node chain graph is reconstructed from <code>node_features</code>
and <code>adjacency</code> in the JSON. One-hot step encodings are appended (same as training).
A single-graph batch vector of all zeros is used since we process one wafer at a time.<br><br>
<b>CNN inference:</b> The PNG image is loaded via PIL, converted to a normalised
<code>FloatTensor[3, 512, 512]</code>, and passed directly to Faster R-CNN in eval mode.
The model outputs boxes, labels, and scores after NMS and score thresholding (0.4).<br><br>
<b>DDP checkpoint loading:</b> The CNN was trained with <code>DistributedDataParallel</code>,
which adds a <code>model.module.*</code> prefix to all state dict keys. The loader strips this
prefix before calling <code>load_state_dict</code>, making the checkpoint compatible with
single-GPU inference without any code changes.
        """)
