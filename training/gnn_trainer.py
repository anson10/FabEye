"""
GNNTrainer — training loop, validation, early stopping, and checkpointing
for the DefectPredictionGNN model.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm

from models.gnn import DefectPredictionGNN, DefectLoss
from evaluation.metrics import DefectMetrics


class EarlyStopping:
    """Stop training if validation loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = np.inf
        self.counter    = 0
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


class GNNTrainer:
    """
    Args:
        model:          DefectPredictionGNN instance
        criterion:      DefectLoss instance
        optimizer:      torch optimizer
        scheduler:      LR scheduler (ReduceLROnPlateau)
        device:         torch.device
        checkpoint_dir: directory to save best checkpoint
    """

    def __init__(
        self,
        model: DefectPredictionGNN,
        criterion: DefectLoss,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model          = model.to(device)
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _move_batch(self, batch):
        return (
            batch.x.to(self.device),
            batch.edge_index.to(self.device),
            batch.batch.to(self.device),
            batch.y_type.to(self.device),
            batch.y_loc.view(-1, 2).to(self.device),   # flatten fix: [B*2] → [B, 2]
            batch.y_severity.to(self.device),
        )

    def train_epoch(self, loader) -> dict:
        self.model.train()
        total_loss = loc_loss_sum = sev_loss_sum = type_loss_sum = 0.0
        correct = total = 0

        for batch in loader:
            x, edge_index, b, y_type, y_loc, y_sev = self._move_batch(batch)

            self.optimizer.zero_grad()
            type_logits, location, severity = self.model(x, edge_index, b)
            loss, breakdown = self.criterion(type_logits, location, severity, y_type, y_loc, y_sev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            bs = y_type.size(0)
            total_loss     += breakdown["total_loss"]    * bs
            type_loss_sum  += breakdown["type_loss"]     * bs
            loc_loss_sum   += breakdown["location_loss"] * bs
            sev_loss_sum   += breakdown["severity_loss"] * bs
            correct        += (type_logits.argmax(-1) == y_type).sum().item()
            total          += bs

        n = len(loader.dataset)
        return {
            "loss":     total_loss    / n,
            "type_loss":  type_loss_sum  / n,
            "loc_loss":   loc_loss_sum   / n,
            "sev_loss":   sev_loss_sum   / n,
            "type_acc": correct / total,
        }

    @torch.no_grad()
    def validate_epoch(self, loader) -> dict:
        self.model.eval()
        metrics = DefectMetrics()
        total_loss = 0.0

        for batch in loader:
            x, edge_index, b, y_type, y_loc, y_sev = self._move_batch(batch)
            type_logits, location, severity = self.model(x, edge_index, b)
            loss, _ = self.criterion(type_logits, location, severity, y_type, y_loc, y_sev)
            total_loss += loss.item() * y_type.size(0)
            metrics.update(type_logits, location, severity, y_type, y_loc, y_sev)

        result = metrics.compute()
        result["loss"] = total_loss / len(loader.dataset)
        return result

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs:  int = 100,
        patience:  int = 15,
    ) -> dict:
        early_stop = EarlyStopping(patience=patience)
        best_val_loss = np.inf
        history = {
            "train_loss": [], "val_loss": [],
            "train_type_acc": [], "val_type_acc": [],
            "train_loc_mse": [], "val_loc_mse": [],
            "lr": [],
        }

        for epoch in range(1, n_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate_epoch(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["loss"])

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_type_acc"].append(train_metrics["type_acc"])
            history["val_type_acc"].append(val_metrics["type_accuracy"])
            history["train_loc_mse"].append(train_metrics["loc_loss"])
            history["val_loc_mse"].append(val_metrics["location_mse"])
            history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"Loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                f"Acc {train_metrics['type_acc']:.3f}/{val_metrics['type_accuracy']:.3f} | "
                f"LR {current_lr:.2e}"
            )

            # Save best checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_gnn.pt", epoch, val_metrics)

            if early_stop.step(val_metrics["loss"]):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print(f"\nBest val loss: {best_val_loss:.4f}")
        return history

    # ------------------------------------------------------------------
    def save_checkpoint(self, filename: str, epoch: int, metrics: dict):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch":      epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "metrics":    metrics,
        }, path)

    def load_checkpoint(self, filename: str = "best_gnn.pt") -> dict:
        path = os.path.join(self.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
        return ckpt["metrics"]
