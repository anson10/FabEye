"""How strongly failure patterns cluster within a lot, on the raw WM-811K labels.

For every pair of defective wafers (label other than "none") in the same lot, count
how often both carry the same pattern. Chance is the probability that two wafers
drawn independently from the overall defect-label distribution match.

Usage: python evaluation/lot_clustering.py
Writes results/wm811k_lot_clustering.json
"""

import json

import numpy as np
import pandas as pd

RAW = "data/wm811k/LSWMD.pkl"


def main():
    df = pd.read_pickle(RAW)
    label = lambda x: x[0][0] if len(x) > 0 and len(x[0]) > 0 else None
    fail = df.failureType.map(label)
    keep = fail.notna() & (fail != "none")
    d = pd.DataFrame({"lot": df.lotName[keep], "fail": fail[keep]})

    same = pairs = 0
    for counts in d.groupby("lot").fail.value_counts().groupby(level=0):
        c = counts[1].to_numpy()
        same += int((c * (c - 1)).sum())
        pairs += int(c.sum() * (c.sum() - 1))
    p = d.fail.value_counts(normalize=True).to_numpy()
    out = {
        "defective_wafers": int(len(d)),
        "lots": int(d.lot.nunique()),
        "within_lot_same_pattern": same / pairs,
        "chance_same_pattern": float((p**2).sum()),
    }
    with open("results/wm811k_lot_clustering.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
