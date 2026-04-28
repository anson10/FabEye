"""
PyTorch data loaders for the CNN (Faster R-CNN) pipeline.

Reads images from data/wafer_images/ and annotations from annotations.json.
Returns (image_tensor, target_dict) pairs in the format Faster R-CNN expects:
  image:  FloatTensor[3, H, W]  normalised to [0, 1]
  target: {
    boxes:    FloatTensor[N, 4]  xyxy pixel coords
    labels:   Int64Tensor[N]     class index (1-based; 0 = background)
    image_id: Int64Tensor[1]
  }

Usage:
    python3 data/image_loader.py
"""

import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF


# Faster R-CNN expects 0 = background; defect_type from JSON is 0-based (0=none…5=metal).
# We shift by +1 so label 0 is never used as a target.
LABEL_OFFSET = 1


class WaferImageDataset(Dataset):
    """
    Args:
        image_dir:      directory containing PNG images + annotations.json
        transforms:     optional callable applied to (image, target) pairs
        min_box_area:   discard annotations whose bbox area is below this (px²)
    """

    def __init__(
        self,
        image_dir: str = "data/wafer_images",
        transforms=None,
        min_box_area: int = 4,
    ):
        self.image_dir   = Path(image_dir)
        self.transforms  = transforms
        self.min_box_area = min_box_area

        ann_path = self.image_dir / "annotations.json"
        with open(ann_path) as f:
            coco = json.load(f)

        # index: image_id → {file_name, wafer_id}
        self.images = {img["id"]: img for img in coco["images"]}

        # index: image_id → list of annotations
        self.ann_index: dict[int, list] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            self.ann_index[ann["image_id"]].append(ann)

        self.ids = sorted(self.images.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id   = self.ids[idx]
        img_info = self.images[img_id]
        anns     = self.ann_index[img_id]

        # Load image
        img_path = self.image_dir / img_info["file_name"]
        image = TF.to_tensor(Image.open(img_path).convert("RGB"))  # [3, H, W] float32 [0,1]

        # Build target
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]          # COCO [x,y,w,h]
            x2, y2 = x + w, y + h
            if w * h < self.min_box_area:
                continue
            boxes.append([x, y, x2, y2])
            labels.append(ann["category_id"] + LABEL_OFFSET)

        if boxes:
            boxes_t  = torch.tensor(boxes,  dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """Custom collate: keeps images and targets as separate lists (required by Faster R-CNN)."""
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def create_image_loaders(
    image_dir:   str   = "data/wafer_images",
    batch_size:  int   = 4,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
    num_workers: int   = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset into train/val/test and return DataLoaders.

    Faster R-CNN is heavy — default batch_size=4 works on 8 GB VRAM.
    Set num_workers>0 on Linux for faster I/O (not recommended on Windows).
    """
    dataset = WaferImageDataset(image_dir)
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"Image dataset split — train: {n_train}, val: {n_val}, test: {n_test}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_image_loaders(batch_size=2)
    images, targets = next(iter(train_loader))
    print(f"Batch size:     {len(images)}")
    print(f"Image shape:    {images[0].shape}")
    print(f"Boxes:          {targets[0]['boxes']}")
    print(f"Labels:         {targets[0]['labels']}")
