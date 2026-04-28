# FabEye CNN Architecture — Faster R-CNN with ResNet-50 FPN

---

## Overview

FabEye uses **Faster R-CNN** with a **ResNet-50 FPN** backbone for defect detection on synthetic wafer SEM images. The model takes a 512×512 RGB image as input and outputs a variable number of bounding boxes, each tagged with a defect class and a confidence score.

```
Input: 512×512 RGB image
  │
  ▼
ResNet-50 Backbone       ← feature extraction (pretrained on ImageNet)
  │
  ▼
Feature Pyramid Network  ← multi-scale feature maps P2–P5
  │
  ▼
Region Proposal Network  ← ~2000 candidate boxes + objectness scores
  │
  ▼
ROI Align + Detection Head ← per-region classification + box refinement
  │
  ▼
Output: [{boxes: [N,4], labels: [N], scores: [N]}]
```

---

## Total Parameters: 41,324,786

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| ResNet-50 backbone (layer1–layer2) | ~8M | Frozen |
| ResNet-50 backbone (layer3–layer4) | ~17M | Yes |
| Feature Pyramid Network | ~3M | Yes |
| Region Proposal Network | ~1M | Yes |
| ROI Detection Head | ~12M | Yes |
| **Total** | **~41M** | **~33M active** |

`trainable_backbone_layers=3` freezes only the first 2 layers of ResNet (basic edge/texture detectors). Everything from layer3 onwards is fine-tuned on our wafer images.

---

## Stage 1 — ResNet-50 Backbone

ResNet-50 is a 50-layer deep residual network. It processes the input image and produces a hierarchy of feature maps at decreasing spatial resolutions.

### Residual Blocks

The key innovation in ResNet is the **skip connection** (residual shortcut):

```
input x
  │
  ├──► Conv → BN → ReLU → Conv → BN → Conv → BN ──►+──► ReLU ──► output
  │                                                  ↑
  └──────────────────────────────────────────────────┘ (identity shortcut)
```

The block learns the **residual** `F(x) = output - x` rather than the full mapping. This solves the vanishing gradient problem — gradients flow directly through the shortcut path back to early layers, making 50-layer training stable.

### Feature Map Hierarchy

ResNet-50 produces 4 feature maps at different scales:

| Stage | Output Size | Channels | Stride | What it encodes |
|-------|------------|----------|--------|----------------|
| C2 (layer1) | 128×128 | 256 | 4× | Edges, corners, simple textures |
| C3 (layer2) | 64×64 | 512 | 8× | Surface patterns, grain boundaries |
| C4 (layer3) | 32×32 | 1024 | 16× | Object parts, defect shapes |
| C5 (layer4) | 16×16 | 2048 | 32× | High-level semantics, full defect context |

C2 has high spatial resolution but shallow features. C5 has low resolution but rich semantic features. Neither alone is ideal for detecting defects that vary in size (a particle is ~15px, an oxide patch is ~80px).

---

## Stage 2 — Feature Pyramid Network (FPN)

FPN solves the multi-scale problem by **merging top-down semantic information with bottom-up spatial detail**.

### Top-down pathway

```
C5 (16×16, 2048ch) ──► 1×1 conv ──► P5 (16×16, 256ch)
                                          │ 2× upsample
C4 (32×32, 1024ch) ──► 1×1 conv ──►  +  ──► P4 (32×32, 256ch)
                                          │ 2× upsample
C3 (64×64,  512ch) ──► 1×1 conv ──►  +  ──► P3 (64×64, 256ch)
                                          │ 2× upsample
C2 (128×128, 256ch) ──► 1×1 conv ──► +  ──► P2 (128×128, 256ch)
```

Each level P2–P5 has exactly **256 channels** and sees **both** local texture (from Ci) and global context (from higher Pj via upsampling + addition). The 1×1 convolutions reduce channel dimensionality before the merge.

### Why this matters for defects

| Defect | Typical size | Best FPN level |
|--------|-------------|---------------|
| Particle | 8–20px | P2 (128×128) |
| Scratch | 30–130px long | P3 (64×64) |
| Pit | 10–40px | P2/P3 |
| Oxide defect | 40–120px | P3/P4 |
| Metal contamination | 20–60px cluster | P3 |

Without FPN, you'd need to pick one scale and miss everything else.

---

## Stage 3 — Region Proposal Network (RPN)

The RPN slides a small network over every position of every FPN level and asks: *"is there an object here?"*

### Anchors

At each spatial position on each FPN level, the RPN places a set of **anchor boxes** — pre-defined rectangles of different aspect ratios:

| FPN level | Anchor size | Aspect ratios |
|-----------|------------|---------------|
| P2 | 32×32 | 0.5, 1.0, 2.0 |
| P3 | 64×64 | 0.5, 1.0, 2.0 |
| P4 | 128×128 | 0.5, 1.0, 2.0 |
| P5 | 256×256 | 0.5, 1.0, 2.0 |

Total anchors per image ≈ 200,000. The RPN produces two outputs per anchor:

1. **Objectness score** — probability that any object (not yet classified) exists in this anchor
2. **Box delta** — (Δx, Δy, Δw, Δh) offsets to refine the anchor into a tighter proposal box

### NMS on proposals

After scoring all anchors, Non-Maximum Suppression (NMS) removes duplicate proposals with IoU > 0.7. The top ~2,000 proposals pass to Stage 4.

### RPN losses

```
loss_objectness   = binary cross-entropy(anchor_scores, is_foreground)
loss_rpn_box_reg  = smooth_L1(predicted_deltas, gt_deltas)  ← only on foreground anchors
```

Smooth L1 loss is used for box regression instead of MSE. It behaves like L1 for large errors (less sensitive to outliers) and L2 for small errors (smooth gradients near zero).

---

## Stage 4 — ROI Align + Detection Head

### ROI Align

For each of the ~2,000 proposals, the detection head needs a fixed-size feature representation. ROI Align crops the relevant region from the FPN feature map and resamples it to a fixed **7×7 grid** using bilinear interpolation.

```
FPN feature map (any scale)
     │
     ▼
Crop region corresponding to proposal box
     │
     ▼
Bilinear interpolation → 7×7×256 feature grid
```

Unlike the older ROI Pooling (which had misalignment due to integer rounding), ROI Align uses exact floating-point coordinates — critical for small defects where a 1-pixel misalignment can lose the entire defect.

### Detection Head

```
7×7×256 feature
     │
     ▼
Flatten → FC(12544 → 1024) → ReLU
     │
     ▼
FC(1024 → 1024) → ReLU
     │
  ┌──┴──────────┐
  ▼             ▼
FC(1024 → 7)   FC(1024 → 7×4)
class logits   box deltas (one set per class)
  │             │
Softmax        + anchor offset
  │             │
class label    refined box [x1, y1, x2, y2]
```

**Two outputs per proposal:**
- **Class logits** → softmax → 7-class probability (background, none, particle, scratch, pit, oxide, metal)
- **Box deltas** → added to proposal box → final refined bounding box in xyxy pixel coordinates

### Detection losses

```
loss_classifier = cross_entropy(class_logits, gt_labels)
loss_box_reg    = smooth_L1(box_deltas, gt_deltas)   ← only on foreground proposals
```

### Post-processing (inference only)

1. Apply box deltas to get final box coordinates
2. Filter by `score_thresh=0.4` — discard low-confidence detections
3. NMS with `nms_thresh=0.5` — remove duplicate boxes for the same defect
4. Return surviving (box, label, score) tuples

---

## Class Index Mapping

TorchVision reserves index 0 for background. All JSON defect types are shifted by +1:

| JSON `defect_type` | Name | FRCNN label |
|--------------------|------|-------------|
| — | background | 0 |
| 0 | none | 1 |
| 1 | particle_contamination | 2 |
| 2 | scratch | 3 |
| 3 | pit | 4 |
| 4 | oxide_defect | 5 |
| 5 | metal_contamination | 6 |

---

## Training Configuration

| Hyperparameter | Value | Reason |
|----------------|-------|--------|
| Optimizer | SGD + momentum 0.9 | Better generalisation than Adam for pretrained fine-tuning |
| Learning rate | 5e-3 | Standard for SGD on Faster R-CNN |
| Weight decay | 5e-4 | L2 regularisation, same as original paper |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=4) | Halves LR when val loss plateaus |
| Batch size | 4 per GPU | Memory budget: ~1.5GB per image at 512×512 |
| Trainable backbone layers | 3 | layer3, layer4, FPN unfrozen; layer1, layer2 frozen |
| Score threshold | 0.4 | Balances precision vs recall |
| NMS IoU threshold | 0.5 | Standard COCO setting |
| Early stopping patience | 8 epochs | Stops if val loss doesn't improve |
| Checkpoint metric | Val F1 | Saves best detection balance, not just lowest loss |

---

## Full Forward Pass (Training Mode)

```
image [3, 512, 512]
  │
  ▼  ImageNet normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  │
  ▼  ResNet-50 → C2[128²,256] C3[64²,512] C4[32²,1024] C5[16²,2048]
  │
  ▼  FPN lateral + top-down → P2[128²,256] P3[64²,256] P4[32²,256] P5[16²,256]
  │
  ▼  RPN (per level):
  │    anchors → objectness scores + deltas
  │    NMS → top 2000 proposals
  │    loss_objectness + loss_rpn_box_reg
  │
  ▼  ROI Align (per proposal):
  │    crop + bilinear → 7×7×256
  │
  ▼  Detection Head:
  │    FC×2 → class logits + box deltas
  │    loss_classifier + loss_box_reg
  │
  ▼  Total loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
  │
  ▼  backward() → SGD.step()
```

---

## Full Forward Pass (Inference Mode)

```
image [3, 512, 512]
  │
  ▼  ResNet-50 → FPN → RPN → top 300 proposals (post-NMS)
  │
  ▼  ROI Align → Detection Head → (class, box, score) per proposal
  │
  ▼  Filter: score > 0.4
  │
  ▼  NMS: IoU > 0.5 → keep highest score per overlap group
  │
  ▼  Output: {boxes: [N,4], labels: [N], scores: [N]}
         N = number of detected defects (0 if clean wafer)
```

---

## Why Faster R-CNN for Wafer Defects

| Alternative | Why not used |
|-------------|-------------|
| YOLOv8 | Single-shot, less accurate on small objects (<20px) |
| SSD | Same issue — anchor matching less precise at small scales |
| Simple CNN classifier | Can't output multiple detections per image |
| ViT-based detector | Too data-hungry for 10,000 images |
| Faster R-CNN ✓ | Two-stage accuracy, FPN multi-scale, pretrained backbone |

The two-stage design (propose then classify) is slower than YOLO but more accurate for the small, low-contrast defects in SEM images. At inference ~100ms per image is well within the <150ms target.
