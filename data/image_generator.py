"""
Synthetic wafer SEM image generator.

Reads synthetic_wafers.json and produces one 512x512 grayscale-on-black PNG
per wafer. Each image simulates a top-down SEM view of a silicon wafer:
  - Circular wafer boundary with gaussian surface texture
  - Defect rendered at (location_x, location_y) with appearance tuned per type
  - COCO-format annotation JSON saved alongside images

Defect visual signatures:
  particle_contamination : bright circular blob
  scratch                : thin bright diagonal line segment
  pit                    : dark circular depression
  oxide_defect           : cloudy bright patch with soft edges
  metal_contamination    : irregular bright cluster
  none                   : clean surface only

Usage:
    python3 data/image_generator.py
    python3 data/image_generator.py --n 500 --out data/wafer_images
"""

import argparse
import json
import os
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np


IMAGE_SIZE   = 512
WAFER_RADIUS = 230   # pixels, leaves ~12px margin
DEFECT_NAMES = ["none", "particle_contamination", "scratch", "pit", "oxide_defect", "metal_contamination"]


# ── surface texture ──────────────────────────────────────────────────────────

def _make_wafer_base(rng: np.random.Generator) -> np.ndarray:
    """512x512 float32 image with circular wafer + subtle surface noise."""
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # Silicon surface: low-frequency perlin-like noise via layered gaussians
    for scale in [128, 64, 32]:
        noise = rng.standard_normal((IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), scale)
        img  += noise * (scale / 128.0)

    # Normalise to [0.3, 0.6] — mid-grey wafer surface
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = img * 0.3 + 0.3

    # Circular mask
    cx = cy = IMAGE_SIZE // 2
    Y, X = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 > WAFER_RADIUS ** 2
    img[mask] = 0.0

    return img


# ── defect renderers ─────────────────────────────────────────────────────────

def _loc_to_px(lx: float, ly: float) -> tuple[int, int]:
    """Normalised [0,1] location → pixel coords inside wafer circle."""
    cx = cy = IMAGE_SIZE // 2
    # Map [0,1] → [-WAFER_RADIUS*0.85, +WAFER_RADIUS*0.85] so defects stay inside
    r = WAFER_RADIUS * 0.85
    px = int(cx + (lx - 0.5) * 2 * r)
    py = int(cy + (ly - 0.5) * 2 * r)
    return px, py


def _draw_particle(img: np.ndarray, px: int, py: int, severity: float, rng):
    """Bright circular blob — particle sitting on surface."""
    radius = max(4, int(6 + severity * 14))
    brightness = 0.55 + severity * 0.35
    cv2.circle(img, (px, py), radius, brightness, -1)
    # Slight halo
    cv2.circle(img, (px, py), radius + 2, brightness * 0.6, 1)
    img[:] = cv2.GaussianBlur(img, (3, 3), 0.8)


def _draw_scratch(img: np.ndarray, px: int, py: int, severity: float, rng):
    """Thin bright line — mechanical scratch across surface."""
    length = int(30 + severity * 100)
    angle  = rng.uniform(0, np.pi)
    dx = int(np.cos(angle) * length / 2)
    dy = int(np.sin(angle) * length / 2)
    x1, y1 = px - dx, py - dy
    x2, y2 = px + dx, py + dy
    thickness = max(1, int(1 + severity * 3))
    brightness = 0.75 + severity * 0.20
    cv2.line(img, (x1, y1), (x2, y2), brightness, thickness)
    img[:] = cv2.GaussianBlur(img, (3, 3), 0.5)


def _draw_pit(img: np.ndarray, px: int, py: int, severity: float, rng):
    """Dark circular depression — material removal / void."""
    radius = max(5, int(8 + severity * 18))
    # Dark centre
    cv2.circle(img, (px, py), radius, 0.05 + (1 - severity) * 0.10, -1)
    # Bright rim (edge charging in SEM)
    cv2.circle(img, (px, py), radius, 0.80, 2)
    img[:] = cv2.GaussianBlur(img, (5, 5), 1.2)


def _draw_oxide(img: np.ndarray, px: int, py: int, severity: float, rng):
    """Soft cloudy bright patch — thin-film oxide variation."""
    radius = max(12, int(20 + severity * 40))
    patch  = np.zeros_like(img)
    cv2.circle(patch, (px, py), radius, 0.5 + severity * 0.4, -1)
    patch = cv2.GaussianBlur(patch, (0, 0), radius // 2)
    img  += patch * 0.55


def _draw_metal(img: np.ndarray, px: int, py: int, severity: float, rng):
    """Irregular bright cluster — metal residue / sputtered particles."""
    n_spots = int(3 + severity * 8)
    spread  = int(10 + severity * 25)
    for _ in range(n_spots):
        ox = int(rng.integers(-spread, spread))
        oy = int(rng.integers(-spread, spread))
        r  = max(2, int(rng.integers(2, max(3, int(6 * severity)))))
        brightness = 0.70 + rng.uniform(0, 0.25)
        cv2.circle(img, (px + ox, py + oy), r, brightness, -1)
    img[:] = cv2.GaussianBlur(img, (3, 3), 0.7)


_RENDERERS = {
    1: _draw_particle,
    2: _draw_scratch,
    3: _draw_pit,
    4: _draw_oxide,
    5: _draw_metal,
}


# ── bounding box ─────────────────────────────────────────────────────────────

def _defect_bbox(defect_type: int, px: int, py: int, severity: float) -> list[int]:
    """Return [x, y, w, h] bounding box for the defect at (px, py)."""
    if defect_type == 1:   # particle
        r = max(4, int(6 + severity * 14)) + 4
        hw = hh = r
    elif defect_type == 2: # scratch
        length = int(30 + severity * 100)
        hw, hh = length // 2 + 4, max(4, int(1 + severity * 3)) + 4
    elif defect_type == 3: # pit
        r = max(5, int(8 + severity * 18)) + 4
        hw = hh = r
    elif defect_type == 4: # oxide
        r = max(12, int(20 + severity * 40)) + 8
        hw = hh = r
    elif defect_type == 5: # metal
        s = int(10 + severity * 25) + int(6 * severity) + 6
        hw = hh = s
    else:
        return [px, py, 1, 1]

    x = max(0, px - hw)
    y = max(0, py - hh)
    w = min(IMAGE_SIZE - x, hw * 2)
    h = min(IMAGE_SIZE - y, hh * 2)
    return [x, y, w, h]


# ── main generator ────────────────────────────────────────────────────────────

def _render_one(args: tuple) -> dict | None:
    """Render a single wafer image — top-level for multiprocessing."""
    idx, wafer, out_dir, seed = args
    wafer_id = wafer["wafer_id"]
    defect   = wafer["defect"]
    dtype    = defect["defect_type"]
    lx, ly   = defect["location_x"], defect["location_y"]
    severity = defect["severity"]

    rng = np.random.default_rng(seed + idx)
    img = _make_wafer_base(rng)

    px = py = 0
    if dtype > 0:
        px, py = _loc_to_px(lx, ly)
        _RENDERERS[dtype](img, px, py, severity, rng)

    img_u8  = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    fname   = f"{wafer_id}.png"
    cv2.imwrite(os.path.join(out_dir, fname), img_rgb)

    img_entry = {"id": idx, "file_name": fname, "width": IMAGE_SIZE, "height": IMAGE_SIZE, "wafer_id": wafer_id}

    ann_entry = None
    if dtype > 0:
        bbox = _defect_bbox(dtype, px, py, severity)
        ann_entry = {
            "id":          idx,
            "image_id":    idx,
            "category_id": dtype,
            "bbox":        bbox,
            "area":        bbox[2] * bbox[3],
            "iscrowd":     0,
            "severity":    severity,
        }
    return img_entry, ann_entry


def generate_images(
    json_path: str = "data/raw/synthetic_wafers.json",
    out_dir:   str = "data/wafer_images",
    n:         int = None,
    seed:      int = 42,
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path) as f:
        wafers = json.load(f)
    if n is not None:
        wafers = wafers[:n]

    workers = cpu_count()
    print(f"Generating {len(wafers)} wafer images → {out_dir}  (workers={workers})")

    tasks = [(idx, wafer, out_dir, seed) for idx, wafer in enumerate(wafers)]
    with Pool(workers) as pool:
        results = pool.map(_render_one, tasks)

    coco = {
        "info":        {"description": "FabEye synthetic wafer images"},
        "categories":  [{"id": i, "name": name} for i, name in enumerate(DEFECT_NAMES)],
        "images":      [],
        "annotations": [],
    }
    for img_entry, ann_entry in results:
        coco["images"].append(img_entry)
        if ann_entry:
            coco["annotations"].append(ann_entry)

    ann_path = os.path.join(out_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    total_defective = sum(1 for w in wafers if w["defect"]["defect_type"] > 0)
    print(f"Done — {len(wafers)} images, {total_defective} with defect annotations")
    print(f"Annotations → {ann_path}")
    return ann_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json",  default="data/raw/synthetic_wafers.json")
    p.add_argument("--out",   default="data/wafer_images")
    p.add_argument("--n",     type=int, default=None, help="limit to first N wafers")
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()
    generate_images(args.json, args.out, args.n, args.seed)
