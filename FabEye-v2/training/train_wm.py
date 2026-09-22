"""Train and evaluate WM-811K classifiers.

Usage: python training/train_wm.py --model {cnn,gnn,rf} --seed 0
Writes results/wm811k_<model>_seed<seed>.json and checkpoints/wm_<model>_seed<seed>.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.wm811k import (
    SPLIT,
    CLASSES,
    WMGraphDataset,
    WMImageDataset,
    load_processed,
    resize_map,
)  # noqa: E402
from models.wm_models import WaferCNN, WaferGNN  # noqa: E402
from torch_geometric.loader import DataLoader as GraphLoader  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize(y, p, extra):
    f1s = f1_score(y, p, labels=range(len(CLASSES)), average=None, zero_division=0)
    return {
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1s.mean()),
        "per_class_f1": dict(zip(CLASSES, map(float, f1s))),
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(CLASSES))).tolist(),
        **extra,
    }


@torch.no_grad()
def predict(model, loader, graph):
    model.eval()
    ys, ps, probs = [], [], []
    for batch in loader:
        if graph:
            batch = batch.to(dev)
            out, y = model(batch), batch.y
        else:
            x, y = batch
            out = model(x.to(dev))
        ps.append(out.argmax(1).cpu())
        probs.append(F.softmax(out, 1).cpu())
        ys.append(y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def run_deep(kind, seed, epochs, bs):
    graph = kind == "gnn"
    Ds = WMGraphDataset if graph else WMImageDataset
    Loader = GraphLoader if graph else DataLoader
    kw = dict(batch_size=bs, num_workers=4, persistent_workers=True)
    tr = Ds(0, augment=True)
    train_dl = Loader(tr, shuffle=True, **kw)
    val_dl, test_dl = Loader(Ds(1), **kw), Loader(Ds(2), **kw)

    counts = np.bincount(tr.y, minlength=len(CLASSES)).astype(np.float32)
    w = torch.tensor((counts.sum() / counts) ** 0.5, device=dev)
    w = w / w.mean()

    model = (WaferGNN() if graph else WaferCNN()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, 2e-3, total_steps=epochs * len(train_dl)
    )
    ckpt = f"checkpoints/wm_{SPLIT}_{kind}_seed{seed}.pt"
    best, hist = -1, []
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in train_dl:
            if graph:
                batch = batch.to(dev)
                out, y = model(batch), batch.y
            else:
                x, y = batch[0].to(dev), batch[1].to(dev)
                out = model(x)
            loss = F.cross_entropy(out, y, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(y), n + len(y)
        yv, pv, _ = predict(model, val_dl, graph)
        vf1 = f1_score(
            yv, pv, average="macro", labels=range(len(CLASSES)), zero_division=0
        )
        hist.append(
            {"epoch": ep + 1, "train_loss": tot / n, "val_macro_f1": float(vf1)}
        )
        print(f"ep {ep+1}/{epochs} loss {tot/n:.4f} val_macro_f1 {vf1:.4f}", flush=True)
        if vf1 > best:
            best = vf1
            torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt))
    t0 = time.time()
    y, p, probs = predict(model, test_dl, graph)
    ms = (time.time() - t0) / len(y) * 1000
    np.save(
        f"results/wm811k_{SPLIT}_{kind}_seed{seed}_probs.npy", probs.astype(np.float16)
    )
    return summarize(
        y,
        p,
        {
            "inference_ms_per_wafer": ms,
            "history": hist,
            "params": sum(q.numel() for q in model.parameters()),
        },
    )


_yy, _xx = np.mgrid[:64, :64]
_r = np.hypot(_yy - 31.5, _xx - 31.5) / 32
_a = (np.arctan2(_yy - 31.5, _xx - 31.5) + np.pi) / (2 * np.pi)
_RI = np.clip((_r * 8).astype(int), 0, 7).ravel()
_AI = np.clip((_a * 8).astype(int), 0, 7).ravel()
_CELL = _RI * 8 + _AI


def rf_features(m):
    """Radial x angular defect density, computed on the 64x64 map."""
    m = resize_map(m).ravel()
    on, bad = m > 0, m == 2
    tot = np.bincount(_CELL[on], minlength=64)
    dead = np.bincount(_CELL[bad], minlength=64)
    cells = dead / np.maximum(tot, 1)
    rt = np.bincount(_RI[on], minlength=8)
    rd = np.bincount(_RI[bad], minlength=8)
    return np.concatenate(
        [cells, rd / np.maximum(rt, 1), [bad.sum() / max(on.sum(), 1)]]
    ).astype(np.float32)


def run_rf(seed):
    from sklearn.ensemble import RandomForestClassifier

    d = load_processed()
    X = np.stack([rf_features(m) for m in d["maps"]])
    tr, te = d["split"] == 0, d["split"] == 2
    clf = RandomForestClassifier(
        300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
    )
    clf.fit(X[tr], d["y"][tr])
    t0 = time.time()
    p = clf.predict(X[te])
    ms = (time.time() - t0) / te.sum() * 1000
    np.save(
        f"results/wm811k_{SPLIT}_rf_seed{seed}_probs.npy",
        clf.predict_proba(X[te]).astype(np.float16),
    )
    return summarize(d["y"][te], p, {"inference_ms_per_wafer": ms})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "gnn", "rf"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    res = (
        run_rf(a.seed) if a.model == "rf" else run_deep(a.model, a.seed, a.epochs, a.bs)
    )
    res.update(model=a.model, seed=a.seed, split=SPLIT)
    with open(f"results/wm811k_{SPLIT}_{a.model}_seed{a.seed}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"TEST macro_f1 {res['macro_f1']:.4f} acc {res['accuracy']:.4f}")
