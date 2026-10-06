"""Wafer map preprocessing for inference. Must match data/wm811k.py exactly."""

import cv2
import numpy as np

IMG = 64
MAX_SIDE = 512  # largest WM-811K map is 300 per side; bounds request cost
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
    m = np.asarray(wafer_map)  # validate before the uint8 cast, which would wrap or overflow
    if m.ndim != 2 or m.size == 0:
        raise ValueError("wafer_map must be a non-empty 2D grid")
    if max(m.shape) > MAX_SIDE:
        raise ValueError(f"wafer_map sides must be at most {MAX_SIDE}")
    if not np.isin(m, (0, 1, 2)).all():
        raise ValueError("wafer_map values must be 0, 1 or 2")
    m = cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return np.stack([m == 0, m == 1, m == 2]).astype(np.float32)
