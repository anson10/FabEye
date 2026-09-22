"""Wafer map preprocessing for inference. Must match data/wm811k.py exactly."""

import cv2
import numpy as np

IMG = 64
CLASSES = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def to_tensor(wafer_map, size=IMG):
    """Convert a 2D die grid (0 outside, 1 good, 2 defective) to a (3, size, size) array."""
    m = np.asarray(wafer_map, dtype=np.uint8)
    if m.ndim != 2:
        raise ValueError("wafer_map must be a 2D grid")
    if m.min() < 0 or m.max() > 2:
        raise ValueError("wafer_map values must be 0, 1 or 2")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.stack([m == 0, m == 1, m == 2]).astype(np.float32)
