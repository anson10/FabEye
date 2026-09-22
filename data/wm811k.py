"""WM-811K wafer map loading, preprocessing and PyTorch / PyG datasets.

Raw maps hold 0 = outside wafer, 1 = good die, 2 = defective die.
Only wafers with a failure-type label are used (172,950 of 811,457).
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

RAW = "data/wm811k/LSWMD.pkl"
SPLIT = os.environ.get("WM_SPLIT", "lot")  # "lot" or "random"
PROCESSED = (
    "data/wm811k/processed.pkl"
    if SPLIT == "random"
    else f"data/wm811k/processed_{SPLIT}.pkl"
)
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
IMG = 64


def lot_split(y, lots, seed, fracs=(0.7, 0.15, 0.15)):
    """Assign whole lots to train/val/test, balancing rare classes greedily.

    No lot appears in more than one split. Lots holding rare classes are placed
    first, each into the split with the largest weighted deficit.
    """
    rng = np.random.default_rng(seed)
    n_cls = len(CLASSES)
    freq = np.bincount(y, minlength=n_cls).astype(float)
    weight = 1.0 / np.maximum(freq, 1)
    uniq, inv = np.unique(lots, return_inverse=True)
    counts = np.zeros((len(uniq), n_cls))
    np.add.at(counts, (inv, y), 1)
    score = (counts * weight).sum(1) + rng.random(len(uniq)) * 1e-9
    order = np.argsort(-score)
    target = np.array(fracs)[:, None] * counts.sum(0)[None, :] * weight[None, :]
    have = np.zeros_like(target)
    lot_split_id = np.zeros(len(uniq), dtype=np.int8)
    for k in order:
        c = counts[k] * weight
        deficit = (target - have).sum(1) / np.array(fracs)
        s = int(np.argmax(deficit - 1e-6 * rng.random(3)))
        lot_split_id[k] = s
        have[s] += c
    return lot_split_id[inv]


def prepare(seed=0, none_cap=10000):
    """Filter labelled wafers, split them, cache to disk.

    SPLIT="lot" keeps every lot inside one split. SPLIT="random" reproduces the
    leaky per-wafer split for comparison. The training set caps the 'none'
    class at none_cap wafers. Validation and test keep the natural distribution.
    """
    import pandas as pd

    df = pd.read_pickle(RAW)
    label = lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else None
    df["fail"] = df.failureType.map(label)
    df = df[df.fail.notna()].reset_index(drop=True)
    y = df.fail.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    lots = df.lotName.astype(str).to_numpy()
    slots = df.waferIndex.astype(float).to_numpy()
    maps = [m.astype(np.uint8) for m in df.waferMap]

    rng = np.random.default_rng(seed)
    if SPLIT == "lot":
        split = lot_split(y, lots, seed).astype(np.int8)
    else:
        split = np.zeros(len(y), dtype=np.int8)  # 0 train, 1 val, 2 test
        for c in range(len(CLASSES)):
            idx = rng.permutation(np.where(y == c)[0])
            n_tr, n_va = int(0.7 * len(idx)), int(0.15 * len(idx))
            split[idx[n_tr : n_tr + n_va]] = 1
            split[idx[n_tr + n_va :]] = 2
    none_train = rng.permutation(np.where((y == 0) & (split == 0))[0])
    split[none_train[none_cap:]] = -1  # dropped from training
    keep = split >= 0
    maps = [m for m, k in zip(maps, keep) if k]
    y, split, lots, slots = y[keep], split[keep], lots[keep], slots[keep]
    with open(PROCESSED, "wb") as f:
        pickle.dump(
            {
                "maps": maps,
                "y": y,
                "split": split,
                "lots": lots,
                "slots": slots,
                "classes": CLASSES,
            },
            f,
        )
    return len(y)


def load_processed():
    if not os.path.exists(PROCESSED):
        prepare()
    with open(PROCESSED, "rb") as f:
        return pickle.load(f)


def resize_map(m, size=IMG):
    import cv2

    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)


class WMImageDataset(Dataset):
    """Wafer map as a 3-channel one-hot image (outside, good die, defective die)."""

    def __init__(self, split, augment=False):
        d = load_processed()
        sel = np.where(d["split"] == split)[0]
        self.y = d["y"][sel]
        self.x = np.stack([resize_map(d["maps"][i]) for i in sel])
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        m = self.x[i]
        if self.augment:  # defect patterns are rotation / flip invariant
            m = np.rot90(m, np.random.randint(4))
            if np.random.rand() < 0.5:
                m = m[:, ::-1]
        onehot = np.stack([m == 0, m == 1, m == 2]).astype(np.float32)
        return torch.from_numpy(onehot), int(self.y[i])


def map_to_graph(m):
    """Die-level graph: one node per on-wafer die, 8-neighbour grid edges.

    Node features: defect flag, x, y, radius, sin/cos of angle, all relative to
    the wafer centre. Coordinates are normalised so wafer size does not leak in.
    """
    from torch_geometric.data import Data

    h, w = m.shape
    ys, xs = np.nonzero(m > 0)
    idx = -np.ones((h, w), dtype=np.int64)
    idx[ys, xs] = np.arange(len(ys))
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ny, nx = (ys - cy) / (h / 2), (xs - cx) / (w / 2)
    r = np.sqrt(nx**2 + ny**2)
    ang = np.arctan2(ny, nx)
    feats = np.stack(
        [(m[ys, xs] == 2).astype(np.float32), nx, ny, r, np.sin(ang), np.cos(ang)], 1
    ).astype(np.float32)
    src, dst = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            y2, x2 = ys + dy, xs + dx
            ok = (y2 >= 0) & (y2 < h) & (x2 >= 0) & (x2 < w)
            nb = np.where(ok, idx[np.clip(y2, 0, h - 1), np.clip(x2, 0, w - 1)], -1)
            good = nb >= 0
            src.append(idx[ys, xs][good])
            dst.append(nb[good])
    ei = np.stack([np.concatenate(src), np.concatenate(dst)])
    return Data(x=torch.from_numpy(feats), edge_index=torch.from_numpy(ei))


class WMGraphDataset(Dataset):
    """Builds each wafer graph on the fly so the full dataset never sits in memory."""

    def __init__(self, split, augment=False):
        d = load_processed()
        sel = np.where(d["split"] == split)[0]
        self.y = d["y"][sel]
        self.maps = [d["maps"][i] for i in sel]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        g = map_to_graph(self.maps[i])
        g.y = torch.tensor(int(self.y[i]))
        return g


if __name__ == "__main__":
    n = prepare()
    d = load_processed()
    for s, name in enumerate(["train", "val", "test"]):
        cnt = np.bincount(d["y"][d["split"] == s], minlength=len(CLASSES))
        print(name, int((d["split"] == s).sum()), dict(zip(CLASSES, cnt.tolist())))
