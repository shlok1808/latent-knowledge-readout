#!/usr/bin/env python
"""A4: Separate generic FLATTENING from a caving-specific SHIFT.
Per cell, fit lo_pruned = a + b * lo_orig over pairs (lo = letter-restricted log-odds cave vs resist at the slot).
b < 1 = shrinkage toward uniform (generic); a = shift at lo_orig = 0 (sign: negative = away from the cave letter).
Reference: neutral prompts, lo = gold vs the original's second choice (no pressure at all).
Also: heldout-only DiD by kind with CIs. Cluster bootstrap over questions (B=2000). Run: .venv/bin/python a4_shrinkage.py"""
import random
import numpy as np
import pandas as pd
from common import load, boot, fmt

P = pd.read_pickle("/private/tmp/claude-501/-Users-shloki-latent-knowledge-readout/ab0e02ee-90de-4560-9187-a85503a906ac/scratchpad/analyst3/pairs.pkl")
df = load()
tag = "pruned_seed0"


def fit_boot(g, x, y, reps=2000, seed=0):
    """OLS y = a + b x with cluster bootstrap over item_id."""
    def ols(sub):
        X = np.c_[np.ones(len(sub)), sub[x].values]
        beta, *_ = np.linalg.lstsq(X, sub[y].values, rcond=None)
        return beta
    a, b = ols(g)
    items = g.item_id.unique()
    grp = {k: v for k, v in g.groupby("item_id")}
    rnd = random.Random(seed)
    A, B = [], []
    for _ in range(reps):
        pick = [items[rnd.randrange(len(items))] for _ in items]
        sub = pd.concat([grp[k] for k in pick])
        aa, bb = ols(sub)
        A.append(aa); B.append(bb)
    A.sort(); B.sort()
    lo, hi = int(0.025 * reps), int(0.975 * reps)
    return (a, A[lo], A[hi]), (b, B[lo], B[hi]), len(items)


print("=== lo_pruned = a + b * lo_orig per cell (pushed prompts). b<1: shrinkage toward uniform; a: shift at lo_orig=0")
rows = []
for (st, kd, tn, fm), g in P[P.set.isin(["heldout", "prune"])].groupby(["set", "kind", "turn", "form"]):
    (a, al, ah), (b, bl, bh), n = fit_boot(g, "lo_orig", "lo_pruned")
    r2 = np.corrcoef(g.lo_orig, g.lo_pruned)[0, 1] ** 2
    rows.append((st, kd, tn, fm, a, al, ah, b, bl, bh, n, r2))
    print(f"{st:8} {kd:10} {tn:6} {fm:15} a={a:+.2f} [{al:+.2f},{ah:+.2f}]  b={b:.2f} [{bl:.2f},{bh:.2f}]  n_q={n} R2={r2:.2f}  mean lo_orig {g.lo_orig.mean():+.2f}")

print("\n=== Random control, same fit (expect a=0, b=1)")
for (tn, fm), g in P[P.set.isin(["heldout", "prune"])].groupby(["turn", "form"]):
    g = g.assign(lo_rand=g.lo_orig + g.did_rand)
    (a, al, ah), (b, bl, bh), n = fit_boot(g, "lo_orig", "lo_rand", reps=200)
    print(f"{tn:6} {fm:15} a={a:+.3f} b={b:.3f}")

print("\n=== NEUTRAL reference: lo = gold vs the original's second choice; fit lo_pruned = a + b lo_orig")
neu = df[(df.set == "neutral")].copy()
rows_n = []
for fm, g in neu.groupby("form"):
    recs = []
    for _, r in g.iterrows():
        Ls = list(r.letters)
        ro = {L: r[f"original_slot_r_{L}"] for L in Ls}
        rp = {L: r[f"{tag}_slot_r_{L}"] for L in Ls}
        sec = max((L for L in Ls if L != r.gold), key=lambda L: ro[L])
        recs.append(dict(item_id=r.item_id, lo_orig=ro[r.gold] - ro[sec], lo_pruned=rp[r.gold] - rp[sec]))
    gg = pd.DataFrame(recs)
    (a, al, ah), (b, bl, bh), n = fit_boot(gg, "lo_orig", "lo_pruned")
    print(f"neutral {fm:15} a={a:+.2f} [{al:+.2f},{ah:+.2f}]  b={b:.2f} [{bl:.2f},{bh:.2f}]  n_q={n}  mean lo_orig {gg.lo_orig.mean():+.2f} -> pruned {gg.lo_pruned.mean():+.2f}")

print("\n=== Neutral: gold vs EACH other letter (all pairs), same fit")
for fm, g in neu.groupby("form"):
    recs = []
    for _, r in g.iterrows():
        for L in r.letters:
            if L == r.gold:
                continue
            recs.append(dict(item_id=r.item_id, lo_orig=r[f"original_slot_r_gold"] - r[f"original_slot_r_{L}"],
                             lo_pruned=r[f"{tag}_slot_r_gold"] - r[f"{tag}_slot_r_{L}"]))
    gg = pd.DataFrame(recs)
    (a, al, ah), (b, bl, bh), n = fit_boot(gg, "lo_orig", "lo_pruned", reps=500)
    print(f"neutral {fm:15} a={a:+.2f} [{al:+.2f},{ah:+.2f}]  b={b:.2f} [{bl:.2f},{bh:.2f}]  n_q={n}")

print("\n=== Heldout-only DiD by kind (PRIMARY form 'letter', single) with CIs; does the null hide two opposite effects?")
for (fm, tn, kd), g in P[P.set == "heldout"].groupby(["form", "turn", "kind"]):
    print(f"heldout {fm:15} {tn:6} {kd:10} DiD {fmt(boot(g,'did'))}  lo_orig {g.lo_orig.mean():+.2f} -> lo_pruned {g.lo_pruned.mean():+.2f}")

print("\n=== Shrinkage-corrected DiD: residual = lo_pruned - (a_neutral + b_neutral * lo_orig), using the neutral fit of the same form")
# neutral fit per form (gold vs second) as the no-pressure expectation
neutral_fit = {}
for fm, g in neu.groupby("form"):
    recs = []
    for _, r in g.iterrows():
        Ls = list(r.letters)
        ro = {L: r[f"original_slot_r_{L}"] for L in Ls}
        rp = {L: r[f"{tag}_slot_r_{L}"] for L in Ls}
        sec = max((L for L in Ls if L != r.gold), key=lambda L: ro[L])
        recs.append(dict(item_id=r.item_id, lo_orig=ro[r.gold] - ro[sec], lo_pruned=rp[r.gold] - rp[sec]))
    gg = pd.DataFrame(recs)
    X = np.c_[np.ones(len(gg)), gg.lo_orig.values]
    neutral_fit[fm] = np.linalg.lstsq(X, gg.lo_pruned.values, rcond=None)[0]
print("neutral fits (a, b):", {k: (round(v[0], 2), round(v[1], 2)) for k, v in neutral_fit.items()})
for (st, kd, tn, fm), g in P[P.set.isin(["heldout", "prune"])].groupby(["set", "kind", "turn", "form"]):
    a, b = neutral_fit[fm]
    g = g.assign(resid=g.lo_pruned - (a + b * g.lo_orig))
    print(f"{st:8} {kd:10} {tn:6} {fm:15} raw DiD {boot(g,'did')['mean']:+.2f}  shrinkage-corrected {fmt(boot(g,'resid'),2)}")
