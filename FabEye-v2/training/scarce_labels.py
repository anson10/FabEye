"""Label-scarcity experiment for the lot-context head.

For each label fraction f, retrain the CNN on f of the training labels, embed the
wafers of all labelled lots, then compare encoder-only, none, shuffled and
context heads. The head also sees only f of its labels. No model selection on
labels is used, so every fraction is treated the same.

Usage: WM_SPLIT=lot python training/scarce_labels.py --seeds 3
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.wm811k import CLASSES, PROCESSED, RAW, SPLIT, resize_map  # noqa: E402
from models.wm_models import WaferCNN  # noqa: E402
from training.train_lot_context import build_members, run, score  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = f"data/wm811k/ctx_cache_{SPLIT}.npz"


def build_cache():
    """64x64 maps of every wafer in a labelled lot, with lot, slot, label and split."""
    with open(PROCESSED, "rb") as f:
        d = pickle.load(f)
    lot_to_split = dict(zip(d["lots"], d["split"]))
    del d
    df = pd.read_pickle(RAW)
    lots = df.lotName.astype(str).to_numpy()
    keep = np.array([l in lot_to_split for l in lots])
    df, lots = df[keep].reset_index(drop=True), lots[keep]
    label = lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else None
    cls = {c: i for i, c in enumerate(CLASSES)}
    y = df.failureType.map(label).map(cls).fillna(-1).astype(np.int8).to_numpy()
    maps = np.stack([resize_map(m.astype(np.uint8)) for m in df.waferMap])
    uniq = {l: i for i, l in enumerate(pd.unique(lots))}
    np.savez(
        CACHE,
        maps=maps,
        lot=np.array([uniq[l] for l in lots], dtype=np.int32),
        slot=df.waferIndex.astype(np.float32).to_numpy(),
        label=y,
        split=np.array([lot_to_split[l] for l in lots], dtype=np.int8),
    )


def onehot(m):
    return torch.stack([m == 0, m == 1, m == 2], 1).float()


def train_encoder(maps, y, train_ids, seed, bs=128):
    torch.manual_seed(seed)
    n = len(train_ids)
    model = WaferCNN().to(dev)
    steps = max(800, 20 * (n // bs + 1))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    counts = np.bincount(y[train_ids], minlength=len(CLASSES)).astype(np.float32)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, device=dev)
    w = w / w.mean()
    X = torch.from_numpy(maps[train_ids]).to(dev)
    Y = torch.from_numpy(y[train_ids].astype(np.int64)).to(dev)
    model.train()
    for _ in range(steps):
        b = torch.randint(0, n, (min(bs, n),), device=dev)
        x = X[b]
        x = torch.rot90(x, int(np.random.randint(4)), (1, 2))
        if np.random.rand() < 0.5:
            x = x.flip(2)
        loss = F.cross_entropy(model(onehot(x)), Y[b], weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def embed_all(model, maps, bs=4096):
    embs, logs = [], []
    for s in range(0, len(maps), bs):
        e = model.embed(onehot(torch.from_numpy(maps[s : s + bs]).to(dev)))
        embs.append(e)
        logs.append(model.head(e))
    return torch.cat(embs), torch.cat(logs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument(
        "--cache-only", action="store_true", help="build the wafer cache and exit"
    )
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.01, 0.05, 0.2, 1.0])
    a = ap.parse_args()
    if not os.path.exists(CACHE):
        build_cache()
    if a.cache_only:
        sys.exit(0)
    z = np.load(CACHE)
    maps, lot, slot, y, split = (
        z["maps"],
        z["lot"],
        z["slot"],
        z["label"].astype(np.int64),
        z["split"],
    )
    members = build_members(lot)
    te = np.flatnonzero((y >= 0) & (split == 2))
    rng = np.random.default_rng(0)
    tr_all = np.flatnonzero((y >= 0) & (split == 0))
    none_ids = rng.permutation(tr_all[y[tr_all] == 0])[
        :10000
    ]  # same cap as the main experiments
    base = np.concatenate([tr_all[y[tr_all] != 0], none_ids])

    out = {}
    for f in a.fracs:
        keep = []
        for c in range(len(CLASSES)):
            ids_c = base[y[base] == c]
            k = min(len(ids_c), max(5, int(round(f * len(ids_c)))))
            keep.append(rng.choice(ids_c, k, replace=False))
        train_ids = np.sort(np.concatenate(keep))
        model = train_encoder(maps, y, train_ids, seed=0)
        emb, logits = embed_all(model, maps)
        enc = score(y[te], logits[te].argmax(1).cpu().numpy())
        print(
            f"frac {f} train_wafers {len(train_ids)} encoder macro_f1 {enc['macro_f1']:.4f}",
            flush=True,
        )
        data = (emb, logits, lot, slot, y, split, members)
        res = {"train_wafers": int(len(train_ids)), "encoder": enc}
        for mode in ("none", "shuffled", "context"):
            runs = []
            for seed in range(a.seeds):
                ids, pr, _ = run(mode, seed, data, a.epochs, frac=f, select=False)
                runs.append(score(y[ids], pr))
            res[mode] = {
                "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
                "macro_f1_std": float(np.std([r["macro_f1"] for r in runs])),
                "runs": runs,
            }
            print(
                f"frac {f} {mode:9s} macro_f1 {res[mode]['macro_f1_mean']:.4f} +- {res[mode]['macro_f1_std']:.4f}",
                flush=True,
            )
        out[str(f)] = res
        with open(f"results/wm811k_{SPLIT}_scarce_labels.json", "w") as fh:
            json.dump(out, fh, indent=1)
