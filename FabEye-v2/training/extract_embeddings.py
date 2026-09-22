"""Embed every wafer in WM-811K (labelled and unlabelled) with a trained WaferCNN.

The lot-context model needs all wafers of a lot, not only the labelled ones.
Output: data/wm811k/embeddings_<split>.npz with per-wafer embedding, logits,
lot id, slot, label (-1 if unlabelled) and the split of the wafer's lot
(-1 if the lot has no labelled wafers).

Usage: WM_SPLIT=lot python training/extract_embeddings.py --ckpt checkpoints/wm_lot_cnn_seed0.pt
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.wm811k import CLASSES, PROCESSED, RAW, SPLIT, resize_map  # noqa: E402
from models.wm_models import WaferCNN  # noqa: E402


def main(ckpt, chunk=4096):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaferCNN().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    with open(PROCESSED, "rb") as f:
        d = pickle.load(f)
    lot_to_split = dict(zip(d["lots"], d["split"]))  # lots never straddle splits
    del d

    df = pd.read_pickle(RAW)
    label = lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else None
    fail = df.failureType.map(label)
    cls = {c: i for i, c in enumerate(CLASSES)}
    y = fail.map(cls).fillna(-1).astype(np.int8).to_numpy()
    lots = df.lotName.astype(str).to_numpy()
    slots = df.waferIndex.astype(np.float32).to_numpy()
    maps = df.waferMap.to_numpy()
    del df

    lot_ids = {l: i for i, l in enumerate(pd.unique(lots))}
    lot_id = np.array([lot_ids[l] for l in lots], dtype=np.int32)
    lot_split = np.array([lot_to_split.get(l, -1) for l in lots], dtype=np.int8)

    n = len(maps)
    emb = np.zeros((n, 256), dtype=np.float16)
    logits = np.zeros((n, len(CLASSES)), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, n, chunk):
            xs = np.stack([resize_map(m.astype(np.uint8)) for m in maps[s : s + chunk]])
            x = torch.from_numpy(
                np.stack([xs == 0, xs == 1, xs == 2], 1).astype(np.float32)
            ).to(dev)
            e = model.embed(x)
            emb[s : s + len(xs)] = e.cpu().numpy()
            logits[s : s + len(xs)] = model.head(e).cpu().numpy()
            if (s // chunk) % 40 == 0:
                print(f"{s}/{n}", flush=True)
    out = f"data/wm811k/embeddings_{SPLIT}.npz"
    np.savez(
        out, emb=emb, logits=logits, lot=lot_id, slot=slots, label=y, split=lot_split
    )
    print("saved", out, "wafers", n, "in labelled lots", int((lot_split >= 0).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/wm_lot_cnn_seed0.pt")
    main(ap.parse_args().ckpt)
