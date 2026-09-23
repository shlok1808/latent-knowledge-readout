"""AUC variants used everywhere: pooled, within-task (pair-weighted), ECE, bootstrap.

Weights let one episode count once even when it contributes several rows to a
stage (a failing episode has up to 15 'late' rounds, a success usually 0-2).
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def pooled_auc(y, s, w=None):
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return np.nan
    return float(roc_auc_score(y[m], s[m], sample_weight=None if w is None else w[m]))


def within_task(y, s, groups, w=None):
    """Pair-weighted within-task AUC: failures vs successes of the same task only.
    Returns (auc, per-task rows for the bootstrap, number of mixed tasks)."""
    m = np.isfinite(s) & np.isfinite(y)
    y, s, g = y[m], s[m], groups[m]
    w = np.ones(len(y)) if w is None else w[m]
    rows = []
    for task in np.unique(g):
        ix = g == task
        pos, neg = ix & (y == 1), ix & (y == 0)
        if not pos.any() or not neg.any():
            continue
        diff = s[pos][:, None] - s[neg][None, :]
        ww = w[pos][:, None] * w[neg][None, :]
        wins = float(np.sum(ww * ((diff > 0) + 0.5 * (diff == 0))))
        rows.append((wins, float(ww.sum())))
    if not rows:
        return np.nan, rows, 0
    return sum(r[0] for r in rows) / sum(r[1] for r in rows), rows, len(rows)


def bootstrap_within(rows, n=1000, seed=42):
    if len(rows) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    wins = np.array([r[0] for r in rows]); pairs = np.array([r[1] for r in rows])
    idx = rng.integers(len(rows), size=(n, len(rows)))
    vals = wins[idx].sum(1) / pairs[idx].sum(1)
    return tuple(np.quantile(vals, [0.025, 0.975]).tolist())


def ece(y, p, w=None, bins=10):
    m = np.isfinite(p) & np.isfinite(y)
    y, p = y[m], p[m]
    w = np.ones(len(y)) if w is None else w[m]
    if not len(y):
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    k = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = w.sum()
    return float(sum(abs(np.average(y[k == b], weights=w[k == b]) - np.average(p[k == b], weights=w[k == b]))
                     * w[k == b].sum() / total for b in range(bins) if (k == b).any()))


def summarize(y, s, groups, w=None, prob=False, boot=True):
    auc_w, rows, n_mixed = within_task(y, s, groups, w)
    lo, hi = bootstrap_within(rows) if boot else (np.nan, np.nan)
    out = {"n": int(np.isfinite(s).sum()), "n_pos": int(np.nansum(y[np.isfinite(s)])),
           "pooled_auc": pooled_auc(y, s, w), "within_auc": auc_w, "within_lo": lo, "within_hi": hi,
           "mixed_tasks": n_mixed, "underpowered": n_mixed < 15}
    if prob:
        out["ece"] = ece(y, s, w)
    return out
