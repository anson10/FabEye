# FabEye — Detailed Project Report

**Last Updated:** May 2026  
**Status:** Active — post-refactor (GNN location head removed)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Generation](#3-data-generation)
4. [Data Loading](#4-data-loading)
5. [GNN Model](#5-gnn-model)
6. [CNN Model](#6-cnn-model)
7. [Training — GNN](#7-training--gnn)
8. [Training — CNN](#8-training--cnn)
9. [Evaluation and Alignment](#9-evaluation-and-alignment)
10. [Dashboard](#10-dashboard)
11. [Results](#11-results)
12. [Design Decisions and Known Limitations](#12-design-decisions-and-known-limitations)

---

## 1. Project Overview

FabEye is a two-stage machine learning pipeline for semiconductor wafer defect analysis. It mirrors the two sources of information available in a real fabrication facility:

- **Process parameters** — temperature, pressure, duration, chemical concentration at each manufacturing step. These are recorded by the equipment and stored in the MES (Manufacturing Execution System).
- **Inspection images** — optical or SEM images taken after processing to visually confirm what defects are present and where.

The key research question is: **can process parameter anomalies (upstream) predict what visual defects will appear in inspection images (downstream)?** Validating this causal chain would allow fabs to intervene earlier in the process rather than waiting for the inspection stage.

The pipeline has two models with a deliberately clean division of responsibility:

| Model | Input | Outputs | Why |
|---|---|---|---|
| GNN (Graph Neural Network) | Process parameters from 8 manufacturing steps | Defect type, severity | Parameters encode what went wrong and how badly, not where |
| CNN (Faster R-CNN) | 512x512 wafer inspection image | Defect type, bounding box location, confidence | Images encode where the defect appears spatially |

---

## 2. System Architecture

```
data/generator.py          →  synthetic_wafers.json   (10,000 wafers)
data/image_generator.py    →  wafer_images/*.png       (10,000 images)
                               annotations.json         (COCO format)

data/loader.py             →  WaferGraphDataset        (PyG Data objects)
data/image_loader.py       →  WaferImageDataset        (image + target dicts)

models/gnn.py              →  DefectPredictionGNN      (2 heads: type, severity)
models/cnn.py              →  DefectDetectionCNN       (Faster R-CNN wrapper)

training/train_gnn.py      →  checkpoints/best_gnn.pt
training/train_cnn.py      →  checkpoints/best_cnn.pt

evaluation/metrics.py      →  DefectMetrics            (GNN evaluation)
evaluation/alignment.py    →  GNNCNNComparison         (cross-model comparison)

visualization/dashboard.py →  Streamlit app (6 tabs)
```

Both models are trained independently on the same 70/15/15 train/val/test split (seed=42). The test set is held out and only used for final evaluation and the live inspector.

---

## 3. Data Generation

### 3.1 Process Parameter Generator (`data/generator.py`)

The generator creates synthetic wafers with physics-inspired causal rules. Each wafer has 8 process steps in sequence:

```
oxidation → lithography → etching → deposition → doping → cmp → cleaning → annealing
```

Each step has 3 parameters sampled from realistic ranges (e.g., oxidation temperature 800–1200°C). Parameters that span orders of magnitude (dopant concentration, dose) are sampled log-uniformly to avoid the low end being underrepresented.

**Defect scoring logic** (`_compute_defect`):

The generator evaluates 5 physics rules, each contributing a score to a specific defect class:

| Rule | Condition | Defect | Base score |
|---|---|---|---|
| High oxidation | temp > 1100°C AND duration > 90 min | oxide_defect | 0.6 + noise |
| High CMP pressure + low slurry | pressure > 8 AND slurry < 0.1 | scratch | 0.5 + noise |
| Poor cleaning | chemical_conc < 0.5 | particle_contamination | 0.4 + noise |
| Extreme doping | concentration > 5e17 | pit | 0.45 + noise |
| Fast deposition | rate > 4.0 | metal_contamination | 0.4 + noise |

Gaussian noise `U(0, 0.15)` is added to all 6 scores (including none) to prevent exact threshold memorisation. The defect with the highest score wins. If that score is below 0.55 the wafer is labelled "none".

**Severity** is `best_score - 0.55`, clipped to [0, 1]. This means severity directly encodes how far the triggering parameter deviated from the threshold — a useful regression target for the GNN because the deviation is directly caused by the process.

**Location** is sampled uniformly from [0.1, 0.9] for both axes, independent of all parameters. This is a deliberate limitation of the synthetic data: real fabs have spatial signatures (edge effects, tool zones) but modelling them was out of scope. The consequence is that the GNN has no signal to predict location, which informed the architecture decision to remove the location head.

**Class distribution** with 10,000 wafers:

| Class | Count | % |
|---|---|---|
| none | 6,495 | 65.0% |
| metal_contamination | 1,435 | 14.3% |
| oxide_defect | 634 | 6.3% |
| particle_contamination | 565 | 5.7% |
| pit | 534 | 5.3% |
| scratch | 337 | 3.4% |

The heavy imbalance (65% none) is handled in training via class weights: all defect classes are upweighted to 2.0 relative to none at 1.0.

### 3.2 Image Generator (`data/image_generator.py`)

Each wafer gets a 512×512 grayscale-on-black PNG rendered as a simulated SEM image. The rendering has two layers:

**Base texture** (`_make_wafer_base`):
Three Gaussian blurs of random noise at scales 128, 64, 32 are summed with weights 0.15, 0.35, 0.50 respectively. Normalised to [0.3, 0.6] to give a mid-grey wafer surface. A circular mask at radius 230px is applied to simulate the wafer boundary.

Note: the largest scale (sigma=128) can produce smooth low-frequency brightness variations that visually resemble oxide defects. This causes some CNN false positives on clean wafers. The weights (0.15 for sigma=128) were chosen to reduce but not eliminate this artifact — reducing it further would make clean wafers look unrealistically uniform.

**Defect rendering** (per type):

| Type | Visual signature | Implementation |
|---|---|---|
| particle_contamination | Bright circular blob | `cv2.circle` + halo ring |
| scratch | Thin bright diagonal line | `cv2.line` at random angle |
| pit | Dark depression + bright rim | Dark filled circle + bright outline |
| oxide_defect | Soft cloudy bright patch | Filled circle + large Gaussian blur |
| metal_contamination | Irregular bright cluster | Multiple small circles scattered |

Defect size scales with severity. A severity of 0.4 produces an oxide patch radius of ~36px; severity 0.05 produces ~22px.

**COCO annotations** are saved alongside the images with bounding boxes in `[x, y, w, h]` format, category IDs matching the defect type integers, and one annotation per defective image. Clean wafers have no annotation entry.

---

## 4. Data Loading

### 4.1 GNN Loader (`data/loader.py`)

`WaferGraphDataset` extends PyG's `Dataset`. Each `get(idx)` call:

1. Reads the wafer's `node_features` — a list of 8 feature vectors, one per process step. Each vector has 3 normalized parameters (values in [0,1]).
2. Appends an 8×8 identity matrix (one-hot step encoding) to the feature matrix. This gives each node a unique identity so the GNN can distinguish "high temperature at oxidation" from "high temperature at annealing". Without this, all nodes share the same learned weights and the model can't identify which step caused the defect.
3. Converts the adjacency list to a PyG edge_index tensor [2, E].
4. Returns a `Data` object with `x` (shape [8, 11]), `edge_index`, `y_type`, `y_loc`, `y_severity`.

The train/val/test split uses `torch.Generator().manual_seed(42)` with a random permutation, giving a reproducible 7000/1500/1500 split.

### 4.2 CNN Loader (`data/image_loader.py`)

`WaferImageDataset` reads from `annotations.json` (COCO format). Each `__getitem__` call:

1. Loads the PNG with PIL, converts to RGB, applies `TF.to_tensor` to get a float32 [3, H, W] tensor in [0,1].
2. Builds a target dict with `boxes` (xyxy float32), `labels` (int64), and `image_id`. Bounding boxes smaller than 4px² are discarded.
3. Applies label offset +1 so that label 0 is reserved for Faster R-CNN's background class and never appears as a target.

`collate_fn` keeps images and targets as Python lists rather than stacking them — required by Faster R-CNN which expects variable numbers of detections per image.

The same seed=42 split is used so GNN and CNN test sets contain exactly the same 1500 wafers.

---

## 5. GNN Model

### 5.1 Architecture (`models/gnn.py` — `DefectPredictionGNN`)

```
Input: 8 nodes, 11 features each (3 params + 8 one-hot)

GCNConv(11 → 256) → BN → ReLU → Dropout(0.3)
GCNConv(256 → 256) → BN → ReLU → Dropout(0.3)
GCNConv(256 → 256) → BN

Global max pool  [256]
Global mean pool [256]     → concat → [512 + 24 = 536]
Raw feature bypass [24]

type_head:     Linear(536→256) → ReLU → Dropout → Linear(256→128) → ReLU → Linear(128→6)
severity_head: Linear(536→64)  → ReLU → Dropout → Linear(64→1) → Sigmoid
```

**Three GCN layers** are used. Four would cause over-smoothing on an 8-node chain graph — all nodes would converge to the same representation after enough message passing steps. Three layers give each node a 3-hop receptive field, which on an 8-node chain means every node can see the full graph.

**Dual pooling** (max + mean) captures complementary information. Mean pooling averages the step representations; max pooling picks the most extreme value at each feature dimension, which is useful for flagging anomalous steps.

**Raw feature bypass** concatenates the unnormalized 3-parameter vectors for all 8 steps directly to the pooled representation. This gives the prediction heads direct access to the raw parameter values without them being transformed by the GCN message passing. It mirrors the 24-feature vector used by a random forest baseline (3 params × 8 steps).

**No location head.** Location was removed because process parameters have no causal link to where on the wafer a defect appears in the synthetic data. Including a location head caused it to collapse to predicting the mean (0.5, 0.5) because 65% of training samples (none class) were masked out of the location loss, leaving the head gradient-starved.

### 5.2 Loss Function (`DefectLoss`)

```python
total_loss = type_weight * CE(type_logits, y_type)
           + severity_weight * (MSE_defective + 0.2 * MSE_none)
```

**Type loss**: standard cross-entropy with class weights [1.0, 2.0, 2.0, 2.0, 2.0, 2.0] to penalise misclassifying rare defect types.

**Severity loss**: two components. The primary MSE is computed only on defective samples (where severity > 0). A lightweight anchor term (weight 0.2) supervises clean wafers toward severity=0. Without the anchor, the severity head receives zero gradient for 65% of samples and tends to output a non-zero default value for clean wafers.

---

## 6. CNN Model

### 6.1 Architecture (`models/cnn.py` — `DefectDetectionCNN`)

A thin wrapper around torchvision's Faster R-CNN with ResNet-50 FPN backbone.

**Faster R-CNN pipeline:**
1. ResNet-50 extracts feature maps at 5 scales (FPN).
2. Region Proposal Network (RPN) proposes ~2000 candidate bounding boxes.
3. RoI Align extracts fixed-size features for each proposal.
4. Box predictor head classifies each proposal and regresses the bounding box offsets.

**Modifications from the pretrained model:**
- Box predictor head replaced with `FastRCNNPredictor(in_features, 7)` — 7 classes: background (0) + none (1) + 5 defect types (2–6).
- `trainable_backbone_layers=3` — the last 3 layers of ResNet-50 plus the FPN are trainable. The first 2 layers are frozen. This is the standard transfer learning setup: low-level edge/texture features are shared with ImageNet, high-level semantic features are fine-tuned for wafer images.
- Score threshold set to 0.4 at inference — detections below this confidence are discarded.

**Label offset**: JSON defect_type (0–5) maps to Faster R-CNN label (1–6) by adding 1. Label 0 is background and is never a training target.

**Training mode** forward pass takes `(images, targets)` and returns a loss dict with 4 components: `loss_classifier` (RoI classification), `loss_box_reg` (RoI box regression), `loss_objectness` (RPN objectness), `loss_rpn_box_reg` (RPN box regression).

**Eval mode** forward pass takes only `images` and returns prediction dicts with `boxes`, `labels`, `scores` after NMS.

---

## 7. Training — GNN

### 7.1 Script (`training/train_gnn.py`)

**Hyperparameters:**
- Epochs: 100 (early stopping with patience=25)
- Batch size: 32
- Learning rate: 1e-3 with ReduceLROnPlateau (factor=0.5, patience=7)
- Hidden channels: 256
- Dropout: 0.3
- Weight decay: 1e-3

**Class weight computation**: The training indices are extracted from the random split, labels are counted, and inverse-frequency-inspired weights [1.0, 2.0, 2.0, 2.0, 2.0, 2.0] are passed to the cross-entropy loss.

### 7.2 Trainer (`training/gnn_trainer.py`)

`GNNTrainer.train_epoch` runs forward, computes `DefectLoss`, clips gradients to norm 1.0, and steps the optimizer.

`GNNTrainer.validate_epoch` runs in no-grad eval mode, accumulates `DefectMetrics`, and returns accuracy + severity RMSE + loss.

Checkpointing saves on every improvement in val type accuracy (not val loss). This is intentional — the primary metric is classification accuracy, not the combined loss.

Early stopping monitors val loss with patience=25. The model typically converges around epoch 60–80 and stops early.

### 7.3 Gradient clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` prevents exploding gradients. GNNs on small graphs can have unstable gradient magnitudes because a small number of message passing steps means fewer gradient-dampening operations compared to deep CNNs.

---

## 8. Training — CNN

### 8.1 Script (`training/train_cnn.py`)

Faster R-CNN is significantly heavier than the GNN. Key differences:

- Batch size: 4 (GPU memory constraint)
- Epochs: 10
- Optimizer: SGD with momentum=0.9, weight_decay=1e-4
- LR: 0.005 with step decay
- Trained with `torchrun` for optional multi-GPU (DDP)

### 8.2 Trainer (`training/cnn_trainer.py`)

`CNNTrainer.validate_epoch` runs two passes over the validation set:
1. **Train mode** — to compute loss (Faster R-CNN only returns losses in train mode, not eval mode)
2. **Eval mode** — to run actual detection and compute precision/recall/F1 via IoU matching

IoU matching: for each predicted box, find the ground truth box with the highest IoU. If IoU >= 0.5 and the GT box hasn't been matched yet, it's a true positive. Unmatched predictions are false positives, unmatched GT boxes are false negatives.

Validation runs every 2 epochs (not every epoch) to reduce training time. The last cached metrics are reused on odd epochs.

**DDP checkpoint loading**: when training with `torchrun`, `DistributedDataParallel` wraps the model and prefixes all state dict keys with `model.module.`. The dashboard and alignment scripts strip this prefix when loading for single-GPU inference.

---

## 9. Evaluation and Alignment

### 9.1 GNN Metrics (`evaluation/metrics.py`)

`DefectMetrics` accumulates predictions across batches and computes:

- **Type accuracy**: fraction of wafers with correct defect type (6-class, including none)
- **Severity RMSE**: root mean squared error on defective wafers only

`timed_inference` wraps the forward pass with `time.perf_counter` to measure milliseconds per batch, averaged to get per-sample inference time.

### 9.2 GNN-CNN Alignment (`evaluation/alignment.py`)

`GNNCNNComparison` loads both checkpoints, reproduces the exact same test split, and runs both models on their respective test inputs. Per-wafer results are paired by index (both splits use the same seed so index i in the GNN test set corresponds to the same wafer as index i in the CNN test set).

**Status categories per wafer:**

| Status | Condition |
|---|---|
| true_negative | Both predict no defect |
| aligned | Same defect type, CNN bbox centre within 0.2 normalised distance of GNN location |
| loc_mismatch | Same defect type, location too far |
| type_mismatch | Different defect types, both detected |
| gnn_only | GNN predicts defect, CNN detects nothing |
| cnn_only | CNN detects defect, GNN predicts none |

Note: after the location head was removed from the GNN, `loc_mismatch` and `aligned` distinction is no longer meaningful in the live inspector. The alignment script `alignment.py` still references the old 3-output GNN forward signature and would need updating to match the new 2-output model before re-running `analyze_alignment.py`.

**Primary metric is type alignment rate** (aligned + loc_mismatch) / total detected. Location was demoted to secondary because the GNN location RMSE of ~0.23 per coordinate meant strict location matching rejected most correct type predictions.

**False positive rate**: GNN fires but CNN sees nothing / all GNN fires. A high FP rate means the GNN is over-predicting defects from parameters that don't produce visible defects.

**False negative rate**: CNN fires but GNN predicts none / all CNN fires. A low FN rate means the GNN rarely misses process anomalies that the CNN visually confirms — the causal chain is working.

---

## 10. Dashboard

`visualization/dashboard.py` is a Streamlit app with 6 tabs:

| Tab | Content |
|---|---|
| Overview | Project description, dataset stats, defect gallery |
| GNN Results | Type accuracy, severity RMSE, confusion matrix, training curves |
| CNN Results | Precision/recall/F1, training curves, example detections |
| Integration | GNN-CNN alignment rate, outcome breakdown, parameter correlations |
| Explorer | Interactive process parameter visualiser |
| Wafer Inspector | Live inference on any wafer from the test set |

**Wafer Inspector** flow:
1. User selects a wafer ID from the dropdown (populated from `annotations.json`)
2. Clicks "Run both models"
3. GNN processes the 8-node graph → type + severity
4. CNN processes the 512×512 image → boxes + labels + scores
5. Result displayed: GNN type and severity in the info panel, CNN bounding box drawn on the image with the box centre reported as location

**Caching**: all data and model loading functions use `@st.cache_data` or `@st.cache_resource`. Models are loaded once per session and cached in memory. The data JSON is cached after first read. This makes re-runs of the inspector fast after the first load.

---

## 11. Results

### GNN (test set, 1500 samples)

| Metric | Value | Target |
|---|---|---|
| Type accuracy | 87.27% | >85% |
| Severity RMSE | 0.107 | <0.60 |
| Inference time | 10.5 ms | <30 ms |

**Per-class breakdown:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| none | 0.95 | 0.91 | 0.93 | 975 |
| particle_contamination | 0.69 | 0.83 | 0.76 | 89 |
| scratch | 0.89 | 0.56 | 0.68 | 45 |
| pit | 0.77 | 0.79 | 0.78 | 92 |
| oxide_defect | 0.79 | 0.71 | 0.75 | 97 |
| metal_contamination | 0.72 | 0.91 | 0.80 | 202 |

Scratch has the lowest recall (0.56) because its rule requires two conditions simultaneously (high CMP pressure AND low slurry), which produces fewer training examples of the exact trigger pattern.

### CNN (test set)

| Metric | Value |
|---|---|
| Precision | 0.983 |
| Recall | 0.985 |
| F1 | 0.984 |
| Classification accuracy | ~1.000 |

The CNN metrics are very high because the model scores are bimodal — either ~0.997 or ~1e-10. Wafers the model recognises get near-perfect confidence; wafers it doesn't recognise produce essentially zero score and fall below the 0.4 threshold. This is a sign of overfitting to the training distribution of synthetic texture patterns.

### Integration (1500 test wafers)

| Metric | Value |
|---|---|
| Type alignment rate | 67.2% |
| Full alignment rate | 61.9% |
| False positive rate | 15.6% |
| False negative rate | 8.9% |

897 wafers (59.8%) were true negatives — both models agreed no defect. Of the 603 wafers where at least one model detected a defect, 67.2% had matching types. The low FN rate (8.9%) is the key finding: the GNN rarely misses process anomalies that the CNN visually confirms.

---

## 12. Design Decisions and Known Limitations

### Why the location head was removed from the GNN

The synthetic generator assigns defect location uniformly at random after deciding the defect type. There is no causal link between process parameters and spatial location. The location head was trained with a masked loss (only on defective samples), which meant 65% of training samples contributed zero gradient. The head collapsed to predicting the mean of the location distribution (~0.5, 0.5) for all inputs. Removing it simplified the model, reduced the number of parameters, and made the architecture honest about what the GNN can actually predict.

In production with real fab data, spatial signatures do exist (edge effects, specific tool regions) and a location head could be meaningful.

### Why the CNN scores are bimodal

Faster R-CNN with a pretrained ResNet-50 backbone fine-tuned on 7000 synthetic images with identical texture generation patterns tends to overfit to those specific visual fingerprints. It either recognises a pattern with near-certainty or produces near-zero scores. More diverse data augmentation, a lighter backbone, or training on more varied synthetic textures would improve generalisation.

### Why 88% accuracy overstates GNN performance

65% of all wafers are labelled "none". A trivial model that always predicts "none" scores 65% accuracy. The 87.3% headline is only 22 percentage points above that baseline. The more informative metrics are per-class F1 and the FN rate in the alignment analysis.

### Why CNN precision/recall appears near-perfect

Precision and recall are computed only on detections that passed the 0.4 score threshold. Low-severity defects never reach this threshold and are effectively invisible to these metrics. The metrics describe how well the model performs when it fires, not how often it fires correctly on all inputs.
