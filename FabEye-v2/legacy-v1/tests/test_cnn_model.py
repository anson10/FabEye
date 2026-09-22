"""Tests for CNN model architecture (DefectDetectionCNN and helpers)."""

import pytest
import torch
from models.cnn import (
    DefectDetectionCNN,
    build_faster_rcnn,
    label_to_name,
    defect_type_to_label,
    label_to_defect_type,
    DEFECT_NAMES,
    _FRCNN_CLASSES,
)


def _dummy_images(n=2, h=128, w=128):
    """Small images to keep tests fast (full 512×512 is slow)."""
    return [torch.rand(3, h, w) for _ in range(n)]


def _dummy_targets(n=2):
    return [
        {
            "boxes":  torch.tensor([[10., 10., 60., 60.]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.int64),  # particle
        }
        for _ in range(n)
    ]


class TestBuildFasterRCNN:
    def test_builds_without_pretrained(self):
        model = build_faster_rcnn(pretrained=False)
        assert model is not None

    def test_num_classes_in_predictor(self):
        model = build_faster_rcnn(num_classes=7, pretrained=False)
        out_features = model.roi_heads.box_predictor.cls_score.out_features
        assert out_features == 7

    def test_custom_num_classes(self):
        model = build_faster_rcnn(num_classes=4, pretrained=False)
        out_features = model.roi_heads.box_predictor.cls_score.out_features
        assert out_features == 4


class TestDefectDetectionCNN:
    def setup_method(self):
        self.model = DefectDetectionCNN(pretrained=False)

    def test_parameter_count_is_positive(self):
        assert self.model.n_parameters > 0

    def test_training_mode_returns_loss_dict(self):
        self.model.train()
        images = _dummy_images()
        targets = _dummy_targets()
        loss_dict = self.model(images, targets)
        expected_keys = {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
        assert set(loss_dict.keys()) == expected_keys

    def test_all_training_losses_are_positive(self):
        self.model.train()
        loss_dict = self.model(_dummy_images(), _dummy_targets())
        for k, v in loss_dict.items():
            assert v.item() >= 0.0, f"{k} is negative"

    def test_total_loss_is_sum(self):
        self.model.train()
        loss_dict = self.model(_dummy_images(), _dummy_targets())
        total = self.model.total_loss(loss_dict)
        expected = sum(v.item() for v in loss_dict.values())
        assert abs(total.item() - expected) < 1e-5

    def test_total_loss_is_scalar(self):
        self.model.train()
        loss_dict = self.model(_dummy_images(), _dummy_targets())
        total = self.model.total_loss(loss_dict)
        assert total.shape == ()

    def test_eval_mode_returns_predictions(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(_dummy_images())
        assert isinstance(preds, list)
        assert len(preds) == 2
        for pred in preds:
            assert "boxes" in pred
            assert "labels" in pred
            assert "scores" in pred

    def test_eval_prediction_shapes_consistent(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(_dummy_images(n=1))
        pred = preds[0]
        n = pred["boxes"].shape[0]
        assert pred["labels"].shape == (n,)
        assert pred["scores"].shape == (n,)

    def test_eval_scores_in_range(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(_dummy_images(n=2))
        for pred in preds:
            if len(pred["scores"]) > 0:
                assert pred["scores"].min() >= 0.0
                assert pred["scores"].max() <= 1.0

    def test_eval_boxes_have_four_coords(self):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(_dummy_images(n=1))
        pred = preds[0]
        if len(pred["boxes"]) > 0:
            assert pred["boxes"].shape[1] == 4

    def test_predict_convenience_method(self):
        preds = self.model.predict(_dummy_images(n=2))
        assert isinstance(preds, list)
        assert len(preds) == 2

    def test_score_threshold_filters_low_confidence(self):
        """Setting a very high threshold should produce fewer (or zero) detections."""
        self.model.eval()
        self.model.model.roi_heads.score_thresh = 0.99
        with torch.no_grad():
            preds_strict = self.model(_dummy_images(n=1))

        self.model.model.roi_heads.score_thresh = 0.01
        with torch.no_grad():
            preds_lenient = self.model(_dummy_images(n=1))

        # Strict threshold should give <= detections compared to lenient
        assert len(preds_strict[0]["boxes"]) <= len(preds_lenient[0]["boxes"])

    def test_grad_flows_through_loss(self):
        self.model.train()
        loss_dict = self.model(_dummy_images(), _dummy_targets())
        total = self.model.total_loss(loss_dict)
        total.backward()
        # At least one backbone parameter should have a gradient
        grads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0

    def test_single_image_batch(self):
        """Model must handle batch size of 1."""
        self.model.train()
        loss_dict = self.model(_dummy_images(n=1), _dummy_targets(n=1))
        assert "loss_classifier" in loss_dict

    def test_num_classes_matches_constant(self):
        out_features = self.model.model.roi_heads.box_predictor.cls_score.out_features
        assert out_features == _FRCNN_CLASSES


class TestLabelHelpers:
    def test_label_to_name_background(self):
        assert label_to_name(0) == "background"

    def test_label_to_name_known_labels(self):
        assert label_to_name(2) == "particle_contamination"
        assert label_to_name(3) == "scratch"
        assert label_to_name(4) == "pit"
        assert label_to_name(5) == "oxide_defect"
        assert label_to_name(6) == "metal_contamination"

    def test_label_to_name_unknown(self):
        assert "unknown" in label_to_name(99)

    def test_defect_type_to_label_offset(self):
        # JSON type 0 → FRCNN label 1
        for json_type in range(6):
            assert defect_type_to_label(json_type) == json_type + 1

    def test_label_to_defect_type_inverse(self):
        for json_type in range(6):
            label = defect_type_to_label(json_type)
            recovered = label_to_defect_type(label)
            assert recovered == json_type

    def test_label_offset_constant(self):
        assert DefectDetectionCNN.LABEL_OFFSET == 1

    def test_defect_names_length(self):
        # background + 6 defect classes = 7
        assert len(DEFECT_NAMES) == 7
