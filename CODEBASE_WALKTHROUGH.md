# FabEye Codebase Walkthrough

This document explains how to explore the FabEye codebase from scratch, in the order that makes the most sense conceptually. Each section tells you what the file does, what to read carefully, and what to run to see it work.

---

## Mental model before you start

The project is a pipeline. Data flows in one direction:

```
Raw parameters → GNN → type + severity prediction
Raw images     → CNN → type + location + confidence

Both outputs → Alignment analysis → how well do they agree?
```

Read and run files in pipeline order. Don't jump to the models before understanding the data.

---

## Step 1 — Understand the data (`data/generator.py`)

**Read first:** the docstring at the top, then `_compute_defect()`, then `_step_to_feature_vector()`.

`_compute_defect()` is the heart of the whole project. It encodes the physics — five if-conditions that map parameter extremes to defect types. Understand this function and you understand what the GNN is trying to learn.

`_step_to_feature_vector()` normalises each parameter to [0,1] by dividing by its known range. This is the feature engineering step.

**Run it:**
```bash
python3 data/generator.py --n 100 --out /tmp/test_wafers.json
```

Look at the printed class distribution. Notice that none (~65%) dominates — this imbalance affects every downstream decision.

**Key insight:** severity = `defect_score - 0.55`. It encodes how far the process deviated from safe operating ranges, not how large the defect is visually. The GNN predicts severity from parameters because both come from the same source.

---

## Step 2 — Understand the images (`data/image_generator.py`)

**Read:** `_make_wafer_base()` and each `_draw_*` function.

`_make_wafer_base()` creates the background texture using layered Gaussian blurs. Note the weight on the sigma=128 term (0.15) — this was tuned down to reduce false positives where smooth background noise looked like an oxide defect.

Each `_draw_*` function renders one defect type. Notice how `severity` controls size and brightness in every renderer. High severity = larger, more obvious defect.

**Run it:**
```bash
python3 data/image_generator.py --n 10 --out /tmp/test_images
```

Open a few images. You should see the wafer circle clearly. On defective wafers, the defect is usually visible near the centre of the image.

**Key insight:** location is randomly placed in [0.1, 0.9] regardless of parameters. This is why the GNN has no location head — there is nothing to learn.

---

## Step 3 — Understand data loading

### GNN side (`data/loader.py`)

**Read:** `WaferGraphDataset.get()`.

The critical line is:
```python
x = torch.cat([x, torch.eye(n_steps)], dim=1)
```
Without this one-hot identity encoding, every node looks identical to the GNN and it cannot distinguish which step caused the anomaly. With it, each node has a unique name.

The output is a PyG `Data` object. If you haven't used PyTorch Geometric before: a `Data` object is like a named-tuple that holds node features (`x`), edges (`edge_index`), and whatever labels you attach (`y_type`, `y_severity`).

**Run it:**
```bash
python3 data/loader.py
```

Check the shapes. `x` should be [8, 11] — 8 nodes, 3 parameters + 8 one-hot = 11 features.

### CNN side (`data/image_loader.py`)

**Read:** `WaferImageDataset.__getitem__()`.

The label offset is important: `labels.append(ann["category_id"] + LABEL_OFFSET)`. Faster R-CNN reserves label 0 for background. Defect type 0 (none) becomes label 1. Defect type 4 (oxide) becomes label 5. Whenever you see `lbl - 1` in the dashboard or alignment code, it is undoing this offset.

**Run it:**
```bash
python3 data/image_loader.py
```

---

## Step 4 — Understand the GNN model (`models/gnn.py`)

**Read:** `DefectPredictionGNN.__init__()` then `forward()`, then `DefectLoss.forward()`.

In `__init__`, count the components: 3 GCN layers, 3 batch norms, 2 heads (type, severity). The pool_dim calculation is worth working out by hand: `hidden_channels * 2 + n_steps * n_raw = 256*2 + 8*3 = 536`. This is the size of the vector fed into both prediction heads.

In `forward()`, the graph embedding is built from three sources concatenated:
1. Max-pooled GCN output [256] — captures peak activations across all nodes
2. Mean-pooled GCN output [256] — captures average behaviour
3. Raw parameter bypass [24] — gives direct access to original parameter values

In `DefectLoss.forward()`, read the anchor loss section. The `none_anchor_weight=0.2` means clean wafers contribute 0.2x as much to the severity loss as defective wafers. Without this, clean wafers produce zero gradient and the severity head drifts to a non-zero default.

**Run the smoke test:**
```bash
python3 models/gnn.py
```

Should print shape [2,6] for type logits and [2,1] for severity.

---

## Step 5 — Understand the CNN model (`models/cnn.py`)

**Read:** `build_faster_rcnn()` then `DefectDetectionCNN.__init__()`.

The key modification is the two lines that replace the box predictor head:
```python
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

The pretrained model was trained on 91 COCO classes. We replace the final head with a new one sized for 7 classes (background + 6 defect types). The backbone weights are kept; only the new head is randomly initialised.

`trainable_backbone_layers=3` freezes the first 2 ResNet layers and trains layers 3–5 plus the FPN. Frozen layers preserve low-level features (edges, textures) that transfer well from ImageNet. Trainable layers adapt to wafer-specific patterns.

**Run the smoke test:**
```bash
python3 models/cnn.py
```

Note the difference in output between training mode (loss dict) and eval mode (prediction list).

---

## Step 6 — Understand GNN training

### Trainer (`training/gnn_trainer.py`)

**Read:** `train_epoch()` and `validate_epoch()`.

`train_epoch` is standard PyTorch: zero grad → forward → loss → backward → clip → step.

The gradient clip `clip_grad_norm_(model.parameters(), max_norm=1.0)` is important for GNNs — without it, message passing on small graphs can cause gradient spikes.

`validate_epoch` runs without gradients, accumulates `DefectMetrics`, and returns the computed results.

Checkpointing saves on val accuracy improvement, not val loss improvement. The justification: accuracy is the final reported metric, so we want the best-accuracy model, not the lowest-loss model (they don't always coincide).

### Main script (`training/train_gnn.py`)

**Read top-to-bottom.** Follow the flow: args → seed → device → loaders → class weights → model → loss → optimizer → trainer → fit → eval → save.

The class weight computation iterates through training indices and counts labels. This is slow (7000 PyG `get()` calls) but only runs once before training starts.

**Run it:**
```bash
python3 training/train_gnn.py --epochs 5
```

5 epochs takes about 2 minutes and confirms the full pipeline works end to end.

---

## Step 7 — Understand CNN training

### Trainer (`training/cnn_trainer.py`)

**Read:** `validate_epoch()`.

The two-pass structure is non-obvious. Faster R-CNN only returns losses in train mode and only returns predictions in eval mode. Validation therefore needs two passes: one in train mode for loss, one in eval mode for detection metrics. This doubles validation time but is unavoidable with the torchvision API.

IoU matching in `validate_epoch`: for each predicted box, find the best-matching GT box. If IoU >= 0.5 and the GT hasn't been matched yet, count as TP. This is standard PASCAL VOC evaluation.

### Main script (`training/train_cnn.py`)

Uses `torchrun` for distributed training. On a single machine run:
```bash
python3 training/train_cnn.py
```

When training with torchrun, state dict keys are prefixed with `model.module.`. The loading code in the dashboard and alignment scripts strips this prefix automatically.

---

## Step 8 — Understand evaluation

### GNN metrics (`evaluation/metrics.py`)

**Read:** `DefectMetrics.update()` and `compute()`.

`update()` accumulates raw tensors across batches. `compute()` converts them to numpy and calculates final metrics. Severity RMSE is only computed on defective wafers (where ground truth severity > 0).

### Alignment (`evaluation/alignment.py`)

**Read:** `GNNCNNComparison.__init__()` then `compute_alignment()`.

`__init__` reproduces the exact train/val/test split from both the GNN and CNN datasets using the same seed. This ensures index i in `gnn_test` corresponds to the same wafer as index i in `cnn_test`.

`compute_alignment()` iterates through paired results and assigns each wafer to one of 6 status categories. Read the status logic carefully — the boundary cases (gnn_only, cnn_only) are the most informative for understanding model failure modes.

**Note:** `alignment.py` still uses the old 3-output GNN forward call. If you re-run `analyze_alignment.py`, update `_run_gnn()` to unpack `type_logits, severity = self.gnn_model(...)` instead of `type_logits, loc_pred, sev_pred = ...`.

---

## Step 9 — Understand the dashboard (`visualization/dashboard.py`)

The dashboard is 900+ lines but follows a clear structure. Read it in sections:

| Lines | Content |
|---|---|
| 1–170 | Imports, constants, colour maps, CSS injection. Skim this. |
| 175–215 | Cached loader functions. Note `@st.cache_data` and `@st.cache_resource`. |
| 290–390 | Tab 1 (Overview). Dataset stats and defect gallery. |
| 385–440 | Tab 2 (GNN Results). Reads `gnn_metrics.json`. |
| 450–530 | Tab 3 (CNN Results). Reads `cnn_metrics.json`. |
| 535–640 | Tab 4 (Integration). Reads `alignment_metrics.json`. |
| 640–780 | Tab 5 (Explorer). Interactive parameter plots, no inference. |
| 780–920 | Tab 6 (Wafer Inspector). Live model inference. Read carefully. |

**Wafer Inspector key flow:**
```python
# GNN: graph → type + severity
tl, sp = gnn_model(x, edge_index, batch_vec)
gnn_type = NAMES[tl.argmax().item()]
gnn_sev  = sp[0].item()

# CNN: image → boxes + labels + scores
dets = cnn_model([img_t])[0]

# Location derived from CNN bbox centre
cnn_loc = ((x1 + x2) / 2 / 512, (y1 + y2) / 2 / 512)
```

**Run the dashboard:**
```bash
streamlit run visualization/dashboard.py
```

---

## Step 10 — Run the tests

```bash
pytest tests/ -v
```

Tests cover data generation, model forward passes, and metric calculations. They are fast (no training). If a test fails after code changes, it tells you which component broke.

---

## Files you can skip initially

| File | Why |
|---|---|
| `database/db_utils.py` | PostgreSQL logging, not needed for model understanding |
| `database/schema.sql` | DB schema, only needed if setting up Postgres |
| `evaluation/visualizations.py` | Plot helpers, only relevant when regenerating figures |
| `evaluation/analyze_alignment.py` | Orchestration script, read after understanding `alignment.py` |
| `training/log_results.py` | SQL logging wrapper, not part of the model pipeline |
| `watch_training.sh` | Utility script for monitoring training runs |

---

## Recommended reading order

```
data/generator.py              understand the physics and labels
data/image_generator.py        understand the image rendering
data/loader.py                 understand GNN input format
data/image_loader.py           understand CNN input format
models/gnn.py                  understand GNN architecture and loss
models/cnn.py                  understand CNN architecture
training/gnn_trainer.py        understand GNN train/val loop
training/train_gnn.py          understand full GNN training pipeline
training/cnn_trainer.py        understand CNN train/val loop
training/train_cnn.py          understand full CNN training pipeline
evaluation/metrics.py          understand how GNN is evaluated
evaluation/alignment.py        understand cross-model comparison
visualization/dashboard.py     understand the full system end to end
```
