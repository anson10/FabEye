"""Tests for evaluation metrics (DefectMetrics) and alignment logic (GNNCNNComparison.compute_alignment)."""

import numpy as np
import pytest
import torch

from evaluation.metrics import DefectMetrics


# ── DefectMetrics ────────────────────────────────────────────────────────────

class TestDefectMetrics:
    def setup_method(self):
        self.metrics = DefectMetrics()

    def _feed(self, n_samples=16, n_defective=8, n_classes=6):
        """Push synthetic predictions into the metrics accumulator."""
        type_logits = torch.zeros(n_samples, n_classes)
        y_type      = torch.zeros(n_samples, dtype=torch.long)

        for i in range(n_defective):
            cls = (i % (n_classes - 1)) + 1  # classes 1..5
            type_logits[i, cls] = 10.0
            y_type[i] = cls

        loc_pred  = torch.rand(n_samples, 2)
        sev_pred  = torch.rand(n_samples, 1)
        y_loc     = torch.rand(n_samples, 2)
        y_sev     = torch.rand(n_samples)

        self.metrics.update(type_logits, loc_pred, sev_pred, y_type, y_loc, y_sev)
        return y_type

    def test_compute_returns_required_keys(self):
        self._feed()
        result = self.metrics.compute()
        assert set(result.keys()) >= {
            "type_accuracy", "location_mse", "severity_rmse",
            "avg_inference_ms", "n_samples"
        }

    def test_n_samples_correct(self):
        self._feed(n_samples=20)
        result = self.metrics.compute()
        assert result["n_samples"] == 20

    def test_n_samples_across_multiple_updates(self):
        self._feed(n_samples=10)
        self._feed(n_samples=15)
        result = self.metrics.compute()
        assert result["n_samples"] == 25

    def test_perfect_type_predictions_give_accuracy_one(self):
        n = 12
        type_logits = torch.zeros(n, 6)
        y_type      = torch.randint(0, 6, (n,))
        for i, t in enumerate(y_type):
            type_logits[i, t] = 10.0

        self.metrics.update(
            type_logits,
            torch.rand(n, 2),
            torch.rand(n, 1),
            y_type,
            torch.rand(n, 2),
            torch.rand(n),
        )
        result = self.metrics.compute()
        assert result["type_accuracy"] == pytest.approx(1.0)

    def test_random_type_predictions_give_low_accuracy(self):
        """Random predictions should give accuracy ~1/6 ≈ 0.17 on 6 classes."""
        n = 200
        torch.manual_seed(0)
        type_logits = torch.randn(n, 6)
        y_type      = torch.randint(0, 6, (n,))

        self.metrics.update(
            type_logits,
            torch.rand(n, 2),
            torch.rand(n, 1),
            y_type,
            torch.rand(n, 2),
            torch.rand(n),
        )
        result = self.metrics.compute()
        assert result["type_accuracy"] < 0.5

    def test_type_accuracy_in_range(self):
        self._feed()
        result = self.metrics.compute()
        assert 0.0 <= result["type_accuracy"] <= 1.0

    def test_location_mse_only_on_defective(self):
        """All-clean batch: location_mse should be 0 (no defective samples)."""
        n = 16
        type_logits = torch.zeros(n, 6)
        type_logits[:, 0] = 10.0  # strongly predict "none"
        y_type      = torch.zeros(n, dtype=torch.long)

        self.metrics.update(
            type_logits,
            torch.rand(n, 2),
            torch.rand(n, 1),
            y_type,
            torch.rand(n, 2),
            torch.rand(n),
        )
        result = self.metrics.compute()
        assert result["location_mse"] == 0.0
        assert result["severity_rmse"] == 0.0

    def test_perfect_location_gives_zero_mse(self):
        n = 8
        type_logits     = torch.zeros(n, 6)
        y_type          = torch.ones(n, dtype=torch.long)  # all defective
        type_logits[:, 1] = 10.0

        y_loc  = torch.rand(n, 2)
        y_sev  = torch.rand(n)
        sev_pred = y_sev.unsqueeze(-1)

        self.metrics.update(type_logits, y_loc.clone(), sev_pred, y_type, y_loc, y_sev)
        result = self.metrics.compute()
        assert result["location_mse"] == pytest.approx(0.0, abs=1e-6)

    def test_severity_rmse_is_non_negative(self):
        self._feed()
        result = self.metrics.compute()
        assert result["severity_rmse"] >= 0.0

    def test_inference_time_recorded(self):
        n = 4
        type_logits = torch.zeros(n, 6)
        y_type      = torch.zeros(n, dtype=torch.long)
        self.metrics.update(
            type_logits, torch.rand(n, 2), torch.rand(n, 1),
            y_type, torch.rand(n, 2), torch.rand(n),
            inference_time_ms=5.0,
        )
        result = self.metrics.compute()
        assert result["avg_inference_ms"] == pytest.approx(5.0)

    def test_reset_clears_state(self):
        self._feed(n_samples=10)
        self.metrics.reset()
        self._feed(n_samples=10, n_defective=4)
        result = self.metrics.compute()
        assert result["n_samples"] == 10

    def test_classification_report_returns_string(self):
        self._feed()
        report = self.metrics.classification_report()
        assert isinstance(report, str)
        assert "precision" in report

    def test_confusion_matrix_shape(self):
        self._feed()
        cm = self.metrics.confusion_matrix()
        assert cm.shape == (6, 6)

    def test_confusion_matrix_diagonal_is_true_positives(self):
        """Perfect predictions → all counts on diagonal."""
        n = 12
        type_logits = torch.zeros(n, 6)
        y_type      = torch.arange(n) % 6
        for i, t in enumerate(y_type):
            type_logits[i, t] = 10.0

        self.metrics.update(
            type_logits,
            torch.rand(n, 2),
            torch.rand(n, 1),
            y_type,
            torch.rand(n, 2),
            torch.rand(n),
        )
        cm = self.metrics.confusion_matrix()
        off_diagonal = cm - np.diag(np.diag(cm))
        assert off_diagonal.sum() == 0


# ── Alignment logic (GNNCNNComparison.compute_alignment) ─────────────────────
# We test the pure compute_alignment method without loading model checkpoints.

from evaluation.alignment import GNNCNNComparison


def _make_gnn_result(type_pred, type_gt=0, loc_pred=None, sev_pred=0.5):
    return {
        "type_pred": type_pred,
        "type_gt":   type_gt,
        "loc_pred":  loc_pred or [0.5, 0.5],
        "loc_gt":    [0.5, 0.5],
        "sev_pred":  sev_pred,
    }


def _make_cnn_result(labels=None, boxes=None, scores=None,
                     gt_labels=None, gt_boxes=None):
    return {
        "labels":    labels    or [],
        "boxes":     boxes     or [],
        "scores":    scores    or [],
        "gt_labels": gt_labels or [],
        "gt_boxes":  gt_boxes  or [],
    }


class TestComputeAlignment:
    """Unit tests for GNNCNNComparison.compute_alignment using synthetic result lists."""

    def _comparator(self, threshold=0.35):
        # Create an instance with dummy paths — we only call compute_alignment directly
        # so __init__ is bypassed via object.__new__
        obj = object.__new__(GNNCNNComparison)
        obj.location_threshold = threshold
        return obj

    def test_both_clean_is_true_negative(self):
        comp = self._comparator()
        gnn = [_make_gnn_result(type_pred=0, type_gt=0)]
        cnn = [_make_cnn_result(labels=[], boxes=[], scores=[])]
        result = comp.compute_alignment(gnn, cnn)
        assert result["true_negative"] == 1
        assert result["aligned"] == 0

    def test_gnn_detects_cnn_misses_is_gnn_only(self):
        comp = self._comparator()
        gnn = [_make_gnn_result(type_pred=2, type_gt=2)]  # GNN sees defect
        cnn = [_make_cnn_result()]                         # CNN sees nothing
        result = comp.compute_alignment(gnn, cnn)
        assert result["gnn_only"] == 1

    def test_cnn_detects_gnn_misses_is_cnn_only(self):
        comp = self._comparator()
        gnn = [_make_gnn_result(type_pred=0, type_gt=2)]  # GNN says clean
        cnn = [_make_cnn_result(
            labels=[3], boxes=[[100, 100, 200, 200]], scores=[0.9],
            gt_labels=[3], gt_boxes=[[100, 100, 200, 200]],
        )]  # CNN sees scratch (label 3 = 0-based class 2 after -1)
        result = comp.compute_alignment(gnn, cnn)
        assert result["cnn_only"] == 1

    def test_type_and_location_match_is_aligned(self):
        comp = self._comparator(threshold=0.35)
        # GNN predicts type 1 at (0.5, 0.5)
        gnn = [_make_gnn_result(type_pred=1, type_gt=1, loc_pred=[0.5, 0.5])]
        # CNN detects label 2 (1-based) → 0-based class 1 (type 1 matches GNN)
        # Box centroid: ((100+164)/2, (100+164)/2) / 512 = (0.258, 0.258)
        # Distance from (0.5, 0.5) = sqrt(2*(0.242)^2) = 0.342 — just under threshold
        cnn = [_make_cnn_result(
            labels=[2], boxes=[[90, 90, 174, 174]], scores=[0.9],
            gt_labels=[2], gt_boxes=[[90, 90, 174, 174]],
        )]
        result = comp.compute_alignment(gnn, cnn)
        assert result["aligned"] == 1

    def test_type_match_location_far_is_loc_mismatch(self):
        comp = self._comparator(threshold=0.1)  # tight threshold
        gnn = [_make_gnn_result(type_pred=1, type_gt=1, loc_pred=[0.1, 0.1])]
        # CNN box centroid at ~(0.78, 0.78) — far from (0.1, 0.1)
        cnn = [_make_cnn_result(
            labels=[2], boxes=[[380, 380, 420, 420]], scores=[0.9],
            gt_labels=[2], gt_boxes=[[380, 380, 420, 420]],
        )]
        result = comp.compute_alignment(gnn, cnn)
        assert result["loc_mismatch"] == 1

    def test_type_mismatch_is_type_mismatch(self):
        comp = self._comparator()
        gnn = [_make_gnn_result(type_pred=1, type_gt=1)]  # GNN: particle
        # CNN: scratch (label 3 → 0-based class 2, not 1)
        cnn = [_make_cnn_result(
            labels=[3], boxes=[[200, 200, 300, 300]], scores=[0.9],
            gt_labels=[3], gt_boxes=[[200, 200, 300, 300]],
        )]
        result = comp.compute_alignment(gnn, cnn)
        assert result["type_mismatch"] == 1

    def test_counts_sum_to_n_test(self):
        comp = self._comparator()
        n = 10
        gnn = [_make_gnn_result(0)] * n
        cnn = [_make_cnn_result()] * n
        result = comp.compute_alignment(gnn, cnn)
        total = (result["true_negative"] + result["aligned"] +
                 result["type_mismatch"] + result["loc_mismatch"] +
                 result["gnn_only"] + result["cnn_only"])
        assert total == n

    def test_alignment_rate_between_zero_and_one(self):
        comp = self._comparator()
        gnn = [_make_gnn_result(i % 3) for i in range(20)]
        cnn = [_make_cnn_result() for _ in range(20)]
        result = comp.compute_alignment(gnn, cnn)
        assert 0.0 <= result["alignment_rate"] <= 1.0
        assert 0.0 <= result["full_alignment_rate"] <= 1.0
        assert 0.0 <= result["fp_rate"] <= 1.0
        assert 0.0 <= result["fn_rate"] <= 1.0

    def test_all_true_negatives_gives_zero_alignment_rate(self):
        """When all wafers are clean, there are no defect events to align."""
        comp = self._comparator()
        n = 10
        gnn = [_make_gnn_result(0)] * n
        cnn = [_make_cnn_result()] * n
        result = comp.compute_alignment(gnn, cnn)
        assert result["alignment_rate"] == 0.0
        assert result["true_negative"] == n

    def test_confusion_matrices_shape(self):
        comp = self._comparator()
        gnn = [_make_gnn_result(i % 6, type_gt=i % 6) for i in range(12)]
        cnn = [_make_cnn_result() for _ in range(12)]
        result = comp.compute_alignment(gnn, cnn)
        assert result["gnn_cm"].shape == (6, 6)
        assert result["cnn_cm"].shape == (6, 6)

    def test_fp_rate_is_zero_when_gnn_never_fires(self):
        comp = self._comparator()
        n = 8
        gnn = [_make_gnn_result(0)] * n  # GNN always says clean
        cnn = [_make_cnn_result()] * n
        result = comp.compute_alignment(gnn, cnn)
        assert result["fp_rate"] == 0.0

    def test_fn_rate_is_zero_when_cnn_never_fires(self):
        comp = self._comparator()
        n = 8
        gnn = [_make_gnn_result(0)] * n
        cnn = [_make_cnn_result()] * n
        result = comp.compute_alignment(gnn, cnn)
        assert result["fn_rate"] == 0.0

    def test_per_wafer_list_length(self):
        comp = self._comparator()
        n = 7
        gnn = [_make_gnn_result(0)] * n
        cnn = [_make_cnn_result()] * n
        result = comp.compute_alignment(gnn, cnn)
        assert len(result["per_wafer"]) == n

    def test_per_wafer_status_is_valid(self):
        comp = self._comparator()
        valid = {"true_negative", "aligned", "type_mismatch",
                 "loc_mismatch", "gnn_only", "cnn_only"}
        gnn = [_make_gnn_result(i % 2) for i in range(6)]
        cnn = [_make_cnn_result() for _ in range(6)]
        result = comp.compute_alignment(gnn, cnn)
        for w in result["per_wafer"]:
            assert w["status"] in valid
