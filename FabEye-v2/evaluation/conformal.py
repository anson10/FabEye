"""Split conformal prediction and selective risk control for wafer pattern classification.

Scores are LAC: s = 1 - p(true class). Calibration and test data must come from
different lots. Functions take probability arrays and return numpy results.
"""

import numpy as np
from scipy.stats import beta


def conformal_quantile(scores, alpha):
    """Finite-sample corrected (1 - alpha) quantile of calibration scores."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return 1.0 if k > n else float(np.sort(scores)[k - 1])


def marginal_threshold(probs, y, alpha):
    return conformal_quantile(1 - probs[np.arange(len(y)), y], alpha)


def classwise_thresholds(probs, y, alpha, n_cls):
    """One threshold per true class (Mondrian). A class with no calibration data gets 1.0."""
    out = np.ones(n_cls)
    for c in range(n_cls):
        m = y == c
        if m.any():
            out[c] = conformal_quantile(1 - probs[m, c], alpha)
    return out


def prediction_sets(probs, q):
    """Boolean (n, K) matrix. Class k is in the set if 1 - p_k <= q (q scalar or per-class)."""
    return (1 - probs) <= np.asarray(q)


def evaluate_sets(sets, y, n_cls):
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    per_class = [
        float(covered[y == c].mean()) if (y == c).any() else float("nan")
        for c in range(n_cls)
    ]
    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "review_rate": float((size != 1).mean()),
        "per_class_coverage": per_class,
    }


def clopper_pearson_upper(k, n, delta):
    """Upper (1 - delta) confidence bound on an error rate with k errors in n trials."""
    return (
        1.0
        if n == 0
        else float(beta.ppf(1 - delta, k + 1, max(n - k, 1)) if k < n else 1.0)
    )


def select_threshold(conf, wrong, target, delta, grid=200):
    """Fixed-sequence selection of the lowest confidence threshold whose error UCB <= target.

    Scans thresholds from strict to lenient and stops at the first failure, which keeps
    the guarantee valid without a multiplicity correction.
    """
    cand = np.unique(np.quantile(conf, np.linspace(1, 0, grid)))[::-1]
    chosen = np.inf
    for t in cand:
        acc = conf >= t
        n, k = int(acc.sum()), int(wrong[acc].sum())
        if n == 0 or clopper_pearson_upper(k, n, delta) > target:
            break
        chosen = t
    return chosen
