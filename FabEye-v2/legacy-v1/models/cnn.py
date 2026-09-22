"""
CNN architecture for semiconductor wafer defect detection.

Uses Faster R-CNN with a ResNet-50 FPN backbone pretrained on ImageNet.
The detection head is fine-tuned for 6 classes (none + 5 defect types).

Output per image:
  boxes:  [N, 4]  xyxy format
  labels: [N]     class index
  scores: [N]     confidence in [0, 1]
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


NUM_CLASSES = 6  # background(0) + none(1) + particle(2) + scratch(3) + pit(4) + oxide(5) + metal(6)
# Note: torchvision Faster R-CNN uses class 0 as background internally,
# so we pass num_classes = n_defect_types + 1 = 7
_FRCNN_CLASSES = NUM_CLASSES + 1  # 7


def build_faster_rcnn(
    num_classes:   int   = _FRCNN_CLASSES,
    pretrained:    bool  = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    """
    Build Faster R-CNN with ResNet-50 FPN backbone.

    Args:
        num_classes:               number of classes including background (default 7)
        pretrained:                load ImageNet weights for backbone
        trainable_backbone_layers: how many backbone layers to unfreeze (0-5)
                                   0 = frozen backbone, 5 = fully trainable
                                   3 is a good balance: trains FPN + last 2 ResNet stages

    Returns:
        model ready for training (call model.train() / model.eval())
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Replace the box predictor head with one sized for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class DefectDetectionCNN(nn.Module):
    """
    Thin wrapper around Faster R-CNN that matches FabEye's calling conventions.

    In training mode:
        forward(images, targets) → loss_dict
        loss_dict keys: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg

    In eval mode:
        forward(images) → list of dicts, each with:
            boxes:  Tensor[N, 4]   xyxy pixel coords
            labels: Tensor[N]      class index (0=background, 1=none, 2=particle, ...)
            scores: Tensor[N]      confidence

    Args:
        pretrained:                use ImageNet backbone weights
        trainable_backbone_layers: backbone layers to unfreeze (0-5)
        score_threshold:           minimum confidence for a detection at inference
        nms_iou_threshold:         IoU threshold for NMS
    """

    # Maps our defect_type int (from JSON) → Faster R-CNN label index
    # JSON: 0=none, 1=particle, 2=scratch, 3=pit, 4=oxide, 5=metal
    # FRCNN: 0=background (never predicted), 1..6 = our classes in same order
    LABEL_OFFSET = 1   # JSON defect_type + 1 = FRCNN label

    def __init__(
        self,
        pretrained:                bool  = True,
        trainable_backbone_layers: int   = 3,
        score_threshold:           float = 0.4,
        nms_iou_threshold:         float = 0.5,
    ):
        super().__init__()
        self.model = build_faster_rcnn(
            num_classes=_FRCNN_CLASSES,
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        # Expose score/nms thresholds so they can be tuned without subclassing
        self.model.roi_heads.score_thresh = score_threshold
        self.model.roi_heads.nms_thresh   = nms_iou_threshold

    def forward(self, images: list, targets: list = None):
        """
        Args:
            images:  list of float32 tensors [3, H, W] in [0, 1]
            targets: list of dicts (only required during training)
                     each dict must have:
                       boxes:  FloatTensor[N, 4]  xyxy
                       labels: Int64Tensor[N]      class index (with LABEL_OFFSET applied)

        Returns:
            training: dict of 4 scalar losses
            eval:     list of prediction dicts
        """
        return self.model(images, targets)

    def predict(self, images: list, device: torch.device = None) -> list:
        """
        Convenience method for single-pass inference without targets.
        Moves images to device if provided.
        """
        self.eval()
        if device is not None:
            images = [img.to(device) for img in images]
        with torch.no_grad():
            return self.forward(images)

    def total_loss(self, loss_dict: dict) -> torch.Tensor:
        """Sum all component losses into a single scalar for backward()."""
        return sum(loss_dict.values())

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── label helpers ────────────────────────────────────────────────────────────

DEFECT_NAMES = ["background", "none", "particle_contamination", "scratch", "pit", "oxide_defect", "metal_contamination"]


def label_to_name(label: int) -> str:
    """FRCNN label index → human-readable name."""
    if 0 <= label < len(DEFECT_NAMES):
        return DEFECT_NAMES[label]
    return f"unknown_{label}"


def defect_type_to_label(defect_type: int) -> int:
    """JSON defect_type → FRCNN label (adds background offset)."""
    return defect_type + DefectDetectionCNN.LABEL_OFFSET


def label_to_defect_type(label: int) -> int:
    """FRCNN label → JSON defect_type (removes background offset)."""
    return label - DefectDetectionCNN.LABEL_OFFSET


if __name__ == "__main__":
    import torch

    model = DefectDetectionCNN(pretrained=False)
    print(f"Model parameters: {model.n_parameters:,}")

    # Smoke test — training mode
    model.train()
    dummy_images = [torch.rand(3, 512, 512) for _ in range(2)]
    dummy_targets = [
        {
            "boxes":  torch.tensor([[50., 80., 150., 180.]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.int64),   # particle
        },
        {
            "boxes":  torch.tensor([[200., 100., 280., 160.]], dtype=torch.float32),
            "labels": torch.tensor([4], dtype=torch.int64),   # pit
        },
    ]
    loss_dict = model(dummy_images, dummy_targets)
    total = model.total_loss(loss_dict)
    print(f"Training losses: { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")
    print(f"Total loss: {total.item():.4f}")

    # Smoke test — eval mode
    model.eval()
    with torch.no_grad():
        preds = model(dummy_images)
    for i, p in enumerate(preds):
        print(f"Image {i}: {len(p['boxes'])} detections, labels={p['labels'].tolist()}, scores={[round(s.item(),3) for s in p['scores']]}")
