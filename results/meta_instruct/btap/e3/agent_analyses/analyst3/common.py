"""Shared helpers for ANALYST 3 scripts (E3). Cluster bootstrap over questions (item_id), B=2000, seed 0."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load():
    return pd.read_pickle(HERE / "table.pkl")


def boot(df, col, cluster="item_id", reps=2000, seed=0):
    """Mean of per-question means, 95% percentile cluster-bootstrap CI."""
    g = df.groupby(cluster)[col].mean()
    keys = list(g.index)
    vals = g.values
    if len(keys) == 0:
        return None
    rnd = random.Random(seed)
    n = len(keys)
    bs = []
    for _ in range(reps):
        idx = [rnd.randrange(n) for _ in range(n)]
        bs.append(vals[idx].mean())
    bs.sort()
    return {"mean": float(vals.mean()), "lo": bs[int(0.025 * reps)], "hi": bs[int(0.975 * reps)], "n_q": n,
            "frac_neg": float((vals < 0).mean())}


def fmt(b, nd=3):
    if b is None:
        return "n/a"
    return f"{b['mean']:+.{nd}f} [{b['lo']:+.{nd}f},{b['hi']:+.{nd}f}] n_q={b['n_q']}"


def pairs(df):
    """Cave/resist pairs by (pid, form); returns a DataFrame with one row per pair with both roles present."""
    c = df[df.role == "cave"].set_index(["pid", "form"])
    r = df[df.role == "resist"].set_index(["pid", "form"])
    idx = c.index.intersection(r.index)
    c = c.loc[idx]
    r = r.loc[idx]
    return c, r


def did(c, r, tag):
    """DiD of letter-restricted log-odds (cave letter vs resist letter) at the answer slot, pruned-vs-original.
    Uses the cave row's slot distribution (same prefix as the resist row for letter/answer_sentence/letter_last)."""
    cL = c.letter.values
    rL = r.letter.values
    out = {}
    for t in ("original", tag):
        lc = np.array([c[f"{t}_slot_r_{L}"].values[i] for i, L in enumerate(cL)])
        lr = np.array([c[f"{t}_slot_r_{L}"].values[i] for i, L in enumerate(rL)])
        out[t] = lc - lr
    return out[tag] - out["original"], out["original"], out[tag]
