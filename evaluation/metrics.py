"""
Evaluation metrics for the GNN defect prediction model.
"""

import time
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


DEFECT_NAMES = ["none", "particle", "scratch", "pit", "oxide_defect", "metal_contam"]


class DefectMetrics:
    """
    Accumulates predictions across batches and computes final metrics.

    Usage:
        metrics = DefectMetrics()
        for batch in loader:
            ...
            metrics.update(type_logits, location_pred, severity_pred,
                           batch.y_type, batch.y_loc, batch.y_severity)
        results = metrics.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._type_preds    = []
        self._type_targets  = []
        self._loc_preds     = []
        self._loc_targets   = []
        self._sev_preds     = []
        self._sev_targets   = []
        self._inference_times_ms = []

    def update(
        self,
        type_logits:   torch.Tensor,
        location_pred: torch.Tensor,
        severity_pred: torch.Tensor,
        y_type:        torch.Tensor,
        y_loc:         torch.Tensor,
        y_severity:    torch.Tensor,
        inference_time_ms: float = 0.0,
    ):
        type_pred = type_logits.argmax(dim=-1)
        self._type_preds.extend(type_pred.cpu().numpy().tolist())
        self._type_targets.extend(y_type.cpu().numpy().tolist())

        self._loc_preds.extend(location_pred.detach().cpu().numpy().tolist())
        self._loc_targets.extend(y_loc.cpu().numpy().tolist())

        self._sev_preds.extend(severity_pred.detach().cpu().squeeze(-1).numpy().tolist())
        self._sev_targets.extend(y_severity.cpu().numpy().tolist())

        if inference_time_ms > 0:
            self._inference_times_ms.append(inference_time_ms)

    def compute(self) -> dict:
        preds   = np.array(self._type_preds)
        targets = np.array(self._type_targets)
        loc_p   = np.array(self._loc_preds)
        loc_t   = np.array(self._loc_targets)
        sev_p   = np.array(self._sev_preds)
        sev_t   = np.array(self._sev_targets)

        type_accuracy = float((preds == targets).mean())

        # Location MSE only on truly defective wafers
        defect_mask = targets > 0
        if defect_mask.sum() > 0:
            location_mse = float(np.mean((loc_p[defect_mask] - loc_t[defect_mask]) ** 2))
            severity_rmse = float(np.sqrt(np.mean((sev_p[defect_mask] - sev_t[defect_mask]) ** 2)))
        else:
            location_mse  = 0.0
            severity_rmse = 0.0

        avg_inference_ms = (
            float(np.mean(self._inference_times_ms)) if self._inference_times_ms else 0.0
        )

        return {
            "type_accuracy":    type_accuracy,
            "location_mse":     location_mse,
            "severity_rmse":    severity_rmse,
            "avg_inference_ms": avg_inference_ms,
            "n_samples":        int(len(preds)),
        }

    def classification_report(self) -> str:
        return classification_report(
            self._type_targets,
            self._type_preds,
            target_names=DEFECT_NAMES,
            zero_division=0,
        )

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self._type_targets, self._type_preds)


def timed_inference(model, batch, device: str = "cpu") -> tuple[tuple, float]:
    """Run a forward pass and return (outputs, inference_time_ms)."""
    model.eval()
    x          = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    b          = batch.batch.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(x, edge_index, b)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return outputs, elapsed_ms
