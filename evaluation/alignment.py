"""
GNN-CNN alignment module.

Compares GNN parameter-based defect predictions against CNN visual detections
on the same test wafers to measure how well manufacturing parameter anomalies
predict the visual defects that appear in inspection images.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GNNLoader

from models.gnn import DefectPredictionGNN
from models.cnn import DefectDetectionCNN
from data.loader import WaferGraphDataset
from data.image_loader import WaferImageDataset, collate_fn


DEFECT_NAMES = ["none", "particle_contamination", "scratch", "pit", "oxide_defect", "metal_contamination"]
IMG_SIZE = 512


class GNNCNNComparison:
    """
    Loads both trained models and runs them on the same test set split.

    A wafer is "aligned" when both models agree on the defect type AND the GNN's
    predicted location is within `location_threshold` (normalized) of the CNN
    detection centroid.
    """

    def __init__(
        self,
        gnn_ckpt: str = "checkpoints/best_gnn.pt",
        cnn_ckpt: str = "checkpoints/best_cnn.pt",
        gnn_data: str = "data/raw/synthetic_wafers.json",
        image_dir: str = "data/wafer_images",
        device: str = None,
        seed: int = 42,
        location_threshold: float = 0.2,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.location_threshold = location_threshold

        # Load GNN — infer hidden_channels from checkpoint weights
        ckpt = torch.load(gnn_ckpt, map_location=self.device, weights_only=False)
        hidden_channels = ckpt["model_state"]["conv1.bias"].shape[0]
        self.gnn_model = DefectPredictionGNN(hidden_channels=hidden_channels)
        self.gnn_model.load_state_dict(ckpt["model_state"])
        self.gnn_model = self.gnn_model.to(self.device).eval()

        # Load CNN — strip DDP "module." prefix if checkpoint was saved under torchrun
        self.cnn_model = DefectDetectionCNN(pretrained=False)
        ckpt = torch.load(cnn_ckpt, map_location=self.device, weights_only=False)
        cnn_state = ckpt["model_state"]
        if any(k.startswith("model.module.") for k in cnn_state):
            cnn_state = {k.replace("model.module.", "model.", 1): v for k, v in cnn_state.items()}
        self.cnn_model.load_state_dict(cnn_state)
        self.cnn_model = self.cnn_model.to(self.device).eval()

        # Reproduce training/val/test split (same seed + ratios)
        gnn_ds = WaferGraphDataset(gnn_data)
        cnn_ds = WaferImageDataset(image_dir)
        n = len(gnn_ds)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        n_test  = n - n_train - n_val

        gen = torch.Generator().manual_seed(seed)
        _, _, self.gnn_test = random_split(gnn_ds, [n_train, n_val, n_test], generator=gen)

        gen = torch.Generator().manual_seed(seed)
        _, _, self.cnn_test = random_split(cnn_ds, [n_train, n_val, n_test], generator=gen)

        self.n_test = n_test

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_gnn(self, batch_size: int = 64) -> list:
        loader = GNNLoader(self.gnn_test, batch_size=batch_size, shuffle=False)
        results = []
        for batch in loader:
            batch = batch.to(self.device)
            type_logits, loc_pred, sev_pred = self.gnn_model(
                batch.x, batch.edge_index, batch.batch
            )
            type_preds = type_logits.argmax(dim=-1)
            for i in range(type_preds.shape[0]):
                results.append({
                    "type_pred": int(type_preds[i].item()),
                    "type_gt":   int(batch.y_type[i].item()),
                    "loc_pred":  loc_pred[i].cpu().numpy().tolist(),
                    "loc_gt":    batch.y_loc[i].cpu().numpy().tolist(),
                    "sev_pred":  float(sev_pred[i].squeeze().item()),
                })
        return results

    @torch.no_grad()
    def _run_cnn(self, batch_size: int = 8) -> list:
        loader = DataLoader(
            self.cnn_test, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        results = []
        self.cnn_model.eval()
        for images, targets in loader:
            images = [img.to(self.device) for img in images]
            preds = self.cnn_model(images)
            for pred, tgt in zip(preds, targets):
                results.append({
                    "boxes":     pred["boxes"].cpu().numpy().tolist(),
                    "labels":    pred["labels"].cpu().numpy().tolist(),  # 1-based (0=background)
                    "scores":    pred["scores"].cpu().numpy().tolist(),
                    "gt_boxes":  tgt["boxes"].numpy().tolist(),
                    "gt_labels": tgt["labels"].numpy().tolist(),
                })
        return results

    # ------------------------------------------------------------------
    def compute_alignment(self, gnn_results: list, cnn_results: list) -> dict:
        """
        Per-wafer comparison of GNN predictions vs CNN detections.

        Status categories:
          true_negative  — both models predict no defect
          aligned        — same defect type, location within threshold
          loc_mismatch   — same type, location too far
          type_mismatch  — different defect types
          gnn_only       — GNN detects, CNN sees nothing
          cnn_only       — CNN detects, GNN predicts none
        """
        assert len(gnn_results) == len(cnn_results)
        n_classes = len(DEFECT_NAMES)

        true_negative = aligned = type_mismatch = loc_mismatch = gnn_only = cnn_only = 0
        per_wafer = []

        # Confusion matrix accumulators: rows=true, cols=predicted
        gnn_cm = np.zeros((n_classes, n_classes), dtype=int)
        cnn_cm = np.zeros((n_classes, n_classes), dtype=int)

        for gnn, cnn in zip(gnn_results, cnn_results):
            gnn_type   = gnn["type_pred"]
            gnn_loc    = np.array(gnn["loc_pred"])
            cnn_labels = cnn["labels"]       # 1-based, empty list if no detection

            gnn_defect = gnn_type > 0
            cnn_defect = len(cnn_labels) > 0

            if not gnn_defect and not cnn_defect:
                true_negative += 1
                status = "true_negative"

            elif gnn_defect and not cnn_defect:
                gnn_only += 1
                status = "gnn_only"

            elif not gnn_defect and cnn_defect:
                cnn_only += 1
                status = "cnn_only"

            else:
                # Both detected — compare highest-scoring CNN detection
                best = int(np.argmax(cnn["scores"])) if cnn["scores"] else 0
                cnn_label_0 = cnn_labels[best] - 1   # 0-based defect class
                box = cnn["boxes"][best]              # xyxy pixel coords
                cx = (box[0] + box[2]) / (2 * IMG_SIZE)
                cy = (box[1] + box[3]) / (2 * IMG_SIZE)
                cnn_loc = np.array([cx, cy])
                dist = float(np.linalg.norm(gnn_loc - cnn_loc))

                type_ok = (gnn_type == cnn_label_0)
                loc_ok  = (dist <= self.location_threshold)

                if type_ok and loc_ok:
                    aligned += 1
                    status = "aligned"
                elif type_ok:
                    loc_mismatch += 1
                    status = "loc_mismatch"
                else:
                    type_mismatch += 1
                    status = "type_mismatch"

            per_wafer.append({
                "gnn_type_pred": gnn_type,
                "gnn_type_gt":   gnn["type_gt"],
                "cnn_labels":    cnn_labels,
                "status":        status,
            })

            # Accumulate confusion matrices against ground truth
            gt = gnn["type_gt"]
            gnn_cm[gt, gnn_type] += 1

            # CNN: map each GT label to the best matched prediction via IoU
            gt_labels_0 = [l - 1 for l in cnn["gt_labels"]]  # 0-based
            pred_labels  = cnn["labels"]
            pred_boxes   = cnn["boxes"]
            gt_boxes     = cnn["gt_boxes"]

            matched_gt = set()
            for gi, (gt_box, gt_lbl) in enumerate(zip(gt_boxes, gt_labels_0)):
                best_iou, best_pred_lbl = 0.0, 0
                for pi, (pd_box, pd_lbl) in enumerate(zip(pred_boxes, pred_labels)):
                    if pi in matched_gt:
                        continue
                    ix1 = max(gt_box[0], pd_box[0]); iy1 = max(gt_box[1], pd_box[1])
                    ix2 = min(gt_box[2], pd_box[2]); iy2 = min(gt_box[3], pd_box[3])
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    area_g = (gt_box[2]-gt_box[0]) * (gt_box[3]-gt_box[1])
                    area_p = (pd_box[2]-pd_box[0]) * (pd_box[3]-pd_box[1])
                    iou = inter / (area_g + area_p - inter + 1e-8)
                    if iou > best_iou:
                        best_iou, best_pred_lbl, best_pi = iou, pd_lbl - 1, pi
                if best_iou >= 0.5:
                    matched_gt.add(best_pi)
                    cnn_cm[gt_lbl, best_pred_lbl] += 1
                else:
                    cnn_cm[gt_lbl, 0] += 1  # missed → predicted "none"
            # False positives: unmatched predictions counted as predicted but true=none
            for pi, pd_lbl in enumerate(pred_labels):
                if pi not in matched_gt:
                    cnn_cm[0, pd_lbl - 1] += 1

        # --- Summary metrics ---
        one_or_both = aligned + type_mismatch + loc_mismatch + gnn_only + cnn_only

        # Primary: type alignment — both models agree on defect class (location is secondary
        # because GNN location RMSE ~0.23/coord makes strict spatial matching too strict)
        type_aligned = aligned + loc_mismatch
        type_alignment_rate = type_aligned / max(one_or_both, 1)

        # Full alignment — type AND location match within threshold
        full_alignment_rate = aligned / max(one_or_both, 1)

        # FP rate  = GNN fires but CNN misses / all GNN fires
        gnn_fires = aligned + type_mismatch + loc_mismatch + gnn_only
        fp_rate = gnn_only / max(gnn_fires, 1)

        # FN rate  = CNN fires but GNN misses / all CNN fires
        cnn_fires = aligned + type_mismatch + loc_mismatch + cnn_only
        fn_rate = cnn_only / max(cnn_fires, 1)

        # Keep alignment_rate pointing to the primary (type) metric for downstream use
        alignment_rate = type_alignment_rate

        return {
            "n_test":              len(gnn_results),
            "alignment_rate":      alignment_rate,
            "full_alignment_rate": full_alignment_rate,
            "fp_rate":             fp_rate,
            "fn_rate":             fn_rate,
            "true_negative":       true_negative,
            "aligned":             aligned,
            "type_mismatch":       type_mismatch,
            "loc_mismatch":        loc_mismatch,
            "gnn_only":            gnn_only,
            "cnn_only":            cnn_only,
            "gnn_cm":              gnn_cm,
            "cnn_cm":              cnn_cm,
            "per_wafer":           per_wafer,
        }

    # ------------------------------------------------------------------
    def run(self, batch_size_gnn: int = 64, batch_size_cnn: int = 8) -> dict:
        print("Running GNN on test set...")
        gnn_results = self._run_gnn(batch_size_gnn)
        print(f"  GNN complete: {len(gnn_results)} wafers")

        print("Running CNN on test set...")
        cnn_results = self._run_cnn(batch_size_cnn)
        print(f"  CNN complete: {len(cnn_results)} wafers")

        print("Computing alignment metrics...")
        metrics = self.compute_alignment(gnn_results, cnn_results)
        return metrics
