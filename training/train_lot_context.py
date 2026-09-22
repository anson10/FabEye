"""Train and evaluate the lot-context head against controls.

Modes, all using the same frozen CNN and the same test wafers:
  encoder   frozen CNN logits only
  none      head on the wafer's own embedding, no neighbours (controls for extra head training)
  shuffled  neighbours drawn from a random OTHER lot, real slot offsets (controls for "more context")
  context   the wafer's true lot neighbours

The head trains on a portion of the validation lots and is selected on the rest.
Test lots are never touched until the final score. Labels are never model inputs.

Usage: WM_SPLIT=lot python training/train_lot_context.py --seeds 3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.wm811k import CLASSES, SPLIT  # noqa: E402
from models.lot_context import MAX_OFF, PAD_OFF, LotContextHead  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 24


def build_members(lot):
    order = np.argsort(lot, kind="stable")
    bounds = np.flatnonzero(np.diff(lot[order])) + 1
    groups = np.split(order, bounds)
    return {int(lot[g[0]]): g for g in groups}


def neighbours(targets, lot, slot, members, mode, pool_lots, rng):
    """Return (nb_idx, off_idx, pad) for each target wafer. nb_idx -1 marks padding."""
    n = len(targets)
    nb = -np.ones((n, K), dtype=np.int64)
    off = np.full((n, K), PAD_OFF, dtype=np.int64)
    if mode in ("none", "encoder"):
        return nb, off, np.ones((n, K), dtype=bool)
    for r, i in enumerate(targets):
        mem = members[int(lot[i])]
        mem = mem[mem != i]
        mem = mem[np.argsort(np.abs(slot[mem] - slot[i]), kind="stable")][:K]
        real_off = (
            np.clip(slot[mem] - slot[i], -MAX_OFF, MAX_OFF).astype(np.int64) + MAX_OFF
        )
        m = len(mem)
        if mode == "shuffled":
            other = members[int(rng.choice(pool_lots))]
            while int(lot[other[0]]) == int(lot[i]):
                other = members[int(rng.choice(pool_lots))]
            mem = rng.choice(other, m, replace=len(other) < m)
        nb[r, :m] = mem
        off[r, :m] = real_off
    return nb, off, nb < 0


def batches(idx, bs, shuffle, rng):
    idx = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for s in range(0, len(idx), bs):
        yield idx[s : s + bs]


def predict(model, emb, logits, targets, nb, off, pad, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for b in batches(targets, bs, False, None):
            t = targets[b]
            nbi = torch.from_numpy(nb[b]).to(dev)
            e = emb[torch.from_numpy(t).to(dev)]
            ne = emb[nbi.clamp(min=0)] * (~torch.from_numpy(pad[b]).to(dev))[..., None]
            out.append(
                model(
                    logits[torch.from_numpy(t).to(dev)],
                    e,
                    ne,
                    torch.from_numpy(off[b]).to(dev),
                    torch.from_numpy(pad[b]).to(dev),
                )
                .argmax(1)
                .cpu()
            )
    return torch.cat(out).numpy()


def run(mode, seed, data, epochs, frac=1.0, select=True):
    emb, logits, lot, slot, y, split, members = data
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lab = y >= 0
    val_lots = np.unique(lot[lab & (split == 1)])
    rng.shuffle(val_lots)
    cut = int(0.6 * len(val_lots))
    head_train_lots, head_val_lots = set(val_lots[:cut]), set(val_lots[cut:])
    is_tr = np.array([l in head_train_lots for l in lot])
    is_hv = np.array([l in head_val_lots for l in lot])
    tr = np.flatnonzero(lab & is_tr & (split == 1))
    hv = np.flatnonzero(lab & is_hv & (split == 1))
    te = np.flatnonzero(lab & (split == 2))
    if frac < 1:  # label-scarce head: stratified subsample, at least 3 per class
        keep = []
        for c in range(len(CLASSES)):
            ids_c = tr[y[tr] == c]
            k = min(len(ids_c), max(3, int(round(frac * len(ids_c)))))
            keep.append(rng.choice(ids_c, k, replace=False))
        tr = np.sort(np.concatenate(keep))
    pool = {1: val_lots, 2: np.unique(lot[lab & (split == 2)])}
    res = {}
    for name, ids, sp in (("tr", tr, 1), ("hv", hv, 1), ("te", te, 2)):
        res[name] = (ids, *neighbours(ids, lot, slot, members, mode, pool[sp], rng))

    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(np.float32)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, device=dev)
    w = w / w.mean()
    model = LotContextHead().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state = -1, None
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    ids, nb, off, pad = res["tr"]
    for ep in range(epochs):
        model.train()
        for b in batches(ids, 256, True, rng):
            t = torch.from_numpy(ids[b]).to(dev)
            p = torch.from_numpy(pad[b]).to(dev)
            ne = emb[torch.from_numpy(nb[b]).to(dev).clamp(min=0)] * (~p)[..., None]
            out = model(logits[t], emb[t], ne, torch.from_numpy(off[b]).to(dev), p)
            loss = F.cross_entropy(out, yt[t], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        hv_ids, *hv_rest = res["hv"]
        pr = predict(model, emb, logits, hv_ids, *hv_rest)
        f1 = f1_score(
            y[hv_ids], pr, average="macro", labels=range(len(CLASSES)), zero_division=0
        )
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    te_ids, *te_rest = res["te"]
    pr = predict(model, emb, logits, te_ids, *te_rest)
    return te_ids, pr, best


def score(y, p):
    f1s = f1_score(y, p, average=None, labels=range(len(CLASSES)), zero_division=0)
    return {
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1s.mean()),
        "per_class_f1": dict(zip(CLASSES, map(float, f1s))),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    z = np.load(f"data/wm811k/embeddings_{SPLIT}.npz")
    emb = torch.from_numpy(z["emb"]).to(dev).float()
    logits = torch.from_numpy(z["logits"]).to(dev).float()
    lot, slot, y, split = z["lot"], z["slot"], z["label"].astype(np.int64), z["split"]
    y_all = y
    members = build_members(lot)
    data = (emb, logits, lot, slot, y_all, split, members)

    te = np.flatnonzero((y_all >= 0) & (split == 2))
    out = {"encoder": [score(y_all[te], z["logits"][te].argmax(1))]}
    for mode in ("none", "shuffled", "context"):
        out[mode] = []
        for seed in range(a.seeds):
            ids, pr, hv = run(mode, seed, data, a.epochs)
            s = score(y_all[ids], pr)
            s["head_val_macro_f1"] = float(hv)
            out[mode].append(s)
            print(
                f"{mode:9s} seed {seed} test macro_f1 {s['macro_f1']:.4f} acc {s['accuracy']:.4f}",
                flush=True,
            )
    summary = {
        m: {
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in rs])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in rs])),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rs])),
        }
        for m, rs in out.items()
    }
    with open(f"results/wm811k_{SPLIT}_lot_context.json", "w") as f:
        json.dump({"summary": summary, "runs": out}, f, indent=1)
    for m, s in summary.items():
        print(
            f"{m:9s} macro_f1 {s['macro_f1_mean']:.4f} +- {s['macro_f1_std']:.4f}  acc {s['accuracy_mean']:.4f}"
        )
