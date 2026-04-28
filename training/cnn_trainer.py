"""
CNNTrainer — training loop, validation, early stopping, and checkpointing
for the DefectDetectionCNN (Faster R-CNN) model.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from models.cnn import DefectDetectionCNN, label_to_defect_type


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = np.inf
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of xyxy boxes: [N,4] x [M,4] → [N,M]."""
    inter_x1 = torch.max(box_a[:, None, 0], box_b[None, :, 0])
    inter_y1 = torch.max(box_a[:, None, 1], box_b[None, :, 1])
    inter_x2 = torch.min(box_a[:, None, 2], box_b[None, :, 2])
    inter_y2 = torch.min(box_a[:, None, 3], box_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - intersection

    return intersection / (union + 1e-8)


class CNNTrainer:
    """
    Args:
        model:          DefectDetectionCNN instance
        optimizer:      torch optimizer
        scheduler:      LR scheduler (ReduceLROnPlateau or StepLR)
        device:         torch.device
        checkpoint_dir: directory to save best checkpoint
        iou_threshold:  IoU threshold for counting a detection as correct
    """

    def __init__(
        self,
        model:          DefectDetectionCNN,
        optimizer:      torch.optim.Optimizer,
        scheduler,
        device:         torch.device,
        checkpoint_dir: str   = "checkpoints",
        iou_threshold:  float = 0.5,
    ):
        self.model          = model.to(device)
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.iou_threshold  = iou_threshold
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _to_device(self, images, targets):
        images  = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets

    def train_epoch(self, loader) -> dict:
        self.model.train()
        total_loss = cls_loss = box_loss = obj_loss = rpn_loss = 0.0
        n_batches = 0

        for images, targets in tqdm(loader, desc="train", leave=False):
            images, targets = self._to_device(images, targets)

            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            loss = self.model.total_loss(loss_dict)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            cls_loss   += loss_dict.get("loss_classifier",     0.0 if isinstance(loss_dict.get("loss_classifier"), float) else loss_dict["loss_classifier"].item())
            box_loss   += loss_dict.get("loss_box_reg",        0.0 if isinstance(loss_dict.get("loss_box_reg"),    float) else loss_dict["loss_box_reg"].item())
            obj_loss   += loss_dict.get("loss_objectness",     0.0 if isinstance(loss_dict.get("loss_objectness"), float) else loss_dict["loss_objectness"].item())
            rpn_loss   += loss_dict.get("loss_rpn_box_reg",    0.0 if isinstance(loss_dict.get("loss_rpn_box_reg"),float) else loss_dict["loss_rpn_box_reg"].item())
            n_batches  += 1

        return {
            "loss":     total_loss / max(n_batches, 1),
            "cls_loss": cls_loss   / max(n_batches, 1),
            "box_loss": box_loss   / max(n_batches, 1),
            "obj_loss": obj_loss   / max(n_batches, 1),
            "rpn_loss": rpn_loss   / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate_epoch(self, loader) -> dict:
        """
        Approximate validation: run in train mode to get losses, then eval mode
        for detection accuracy.  Faster R-CNN doesn't output losses in eval mode.
        """
        # --- loss pass (train mode, no backward) ---
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        for images, targets in loader:
            images, targets = self._to_device(images, targets)
            loss_dict   = self.model(images, targets)
            total_loss += self.model.total_loss(loss_dict).item()
            n_batches  += 1
        avg_loss = total_loss / max(n_batches, 1)

        # --- detection accuracy pass (eval mode) ---
        self.model.eval()
        tp = fp = fn = 0
        correct_cls = total_gt = 0

        for images, targets in loader:
            images, targets = self._to_device(images, targets)
            preds = self.model(images)

            for pred, tgt in zip(preds, targets):
                gt_boxes  = tgt["boxes"]
                gt_labels = tgt["labels"]
                pd_boxes  = pred["boxes"]
                pd_labels = pred["labels"]

                n_gt = len(gt_boxes)
                n_pd = len(pd_boxes)
                total_gt += n_gt

                if n_gt == 0 and n_pd == 0:
                    continue
                if n_gt == 0:
                    fp += n_pd
                    continue
                if n_pd == 0:
                    fn += n_gt
                    continue

                iou = _compute_iou(pd_boxes, gt_boxes)  # [n_pd, n_gt]
                matched_gt = set()
                for pi in range(n_pd):
                    best_gt  = iou[pi].argmax().item()
                    best_iou = iou[pi, best_gt].item()
                    if best_iou >= self.iou_threshold and best_gt not in matched_gt:
                        matched_gt.add(best_gt)
                        tp += 1
                        if pd_labels[pi].item() == gt_labels[best_gt].item():
                            correct_cls += 1
                    else:
                        fp += 1
                fn += n_gt - len(matched_gt)

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        cls_acc   = correct_cls / (tp + 1e-8)

        return {
            "loss":      avg_loss,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "cls_acc":   cls_acc,
            "tp": tp, "fp": fp, "fn": fn,
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 20,
        patience: int = 10,
    ) -> dict:
        early_stop   = EarlyStopping(patience=patience)
        best_val_f1  = 0.0
        history = {
            "train_loss": [], "val_loss": [],
            "val_precision": [], "val_recall": [], "val_f1": [],
            "val_cls_acc": [], "lr": [],
        }

        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate_epoch(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["loss"])

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_cls_acc"].append(val_metrics["cls_acc"])
            history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"Loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"P {val_metrics['precision']:.3f}  R {val_metrics['recall']:.3f}  "
                f"F1 {val_metrics['f1']:.3f}  Cls {val_metrics['cls_acc']:.3f} | "
                f"LR {current_lr:.2e}"
            )

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                self.save_checkpoint("best_cnn.pt", epoch, val_metrics)

            if early_stop.step(val_metrics["loss"]):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"\nBest val F1: {best_val_f1:.4f}")
        return history

    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "metrics":     metrics,
        }, path)

    def load_checkpoint(self, filename: str = "best_cnn.pt") -> dict:
        path = os.path.join(self.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["metrics"]
