#!/usr/bin/env python
"""A1: Does pruning move probability between SPELLINGS of the same letter ("(B" vs "B" vs " B"), and how much letter
mass is lost at the first reply position, by set/kind/turn? Run: .venv/bin/python a1_spelling.py"""
import numpy as np
import pandas as pd
from common import load, boot, fmt

df = load()
tags = [t for t in ("pruned_seed0", "random_seed0") if f"{t}_sum" in df]

# one row per prompt (pid): first-position stats are the same for every reply of a prompt; use the letter form
one = df[df.form.isin(["letter", "pres_qa"])].drop_duplicates("pid").copy()
print("rows (one per prompt):", len(one))

print("\n=== First-position spelling SHARES within letter mass (exp of mean log-share), original vs pruned vs random")
print("single-turn canonical spelling is '(X' (paren); multi-turn canonical is bare 'X'.")
for (st, kd, tn), g in one.groupby(["set", "kind", "turn"]):
    line = f"{st:8} {kd:10} {tn:6} n={len(g):4} | "
    for t in ("original",) + tuple(tags):
        shares = {sp: np.exp(g[f"{t}_first_sp_{sp}"]).mean() for sp in ("bare", "space", "paren")}
        line += f"{t[:8]}: bare {shares['bare']:.3f} space {shares['space']:.3f} paren {shares['paren']:.3f} | "
    print(line)

print("\n=== First-position TOTAL letter mass (mean prob, all spellings of the item's letters)")
for (st, kd, tn), g in one.groupby(["set", "kind", "turn"]):
    line = f"{st:8} {kd:10} {tn:6} n={len(g):4} | "
    for t in ("original",) + tuple(tags):
        line += f"{t[:8]}: {np.exp(g[f'{t}_first_mass']).mean():.3f} (median {np.exp(g[f'{t}_first_mass']).median():.3f}) | "
    print(line)

print("\n=== Log-ratio of first-position letter mass (pruned/original), cluster bootstrap over questions")
for (st, kd, tn), g in one.groupby(["set", "kind", "turn"]):
    g = g.assign(lr=g["pruned_seed0_first_mass"] - g["original_first_mass"])
    print(f"{st:8} {kd:10} {tn:6} {fmt(boot(g, 'lr'))}")

print("\n=== Same, by dataset (single-turn pushed prompts, heldout+prune)")
sub = one[(one.turn == "single") & one.set.isin(["heldout", "prune"])]
for ds, g in sub.groupby("dataset"):
    g = g.assign(lr=g["pruned_seed0_first_mass"] - g["original_first_mass"])
    print(f"{ds:14} {fmt(boot(g, 'lr'))}  orig mass {np.exp(g.original_first_mass).mean():.3f} -> pruned {np.exp(g.pruned_seed0_first_mass).mean():.3f}")

print("\n=== Spelling at the ANSWER SLOT of the answer_sentence form (after 'The correct answer is ('; canonical = bare)")
asx = df[df.form == "answer_sentence"]
for (st, tn), g in asx.groupby(["set", "turn"]):
    line = f"{st:8} {tn:6} n={len(g):4} | "
    for t in ("original",) + tuple(tags):
        shares = {sp: np.exp(g[f"{t}_slot_sp_{sp}"]).mean() for sp in ("bare", "space", "paren")}
        line += f"{t[:8]}: bare {shares['bare']:.4f} space {shares['space']:.4f} paren {shares['paren']:.4f} mass {np.exp(g[f'{t}_slot_mass']).mean():.3f} | "
    print(line)

print("\n=== Within-spelling vs across-spelling: for single-turn letter form, where did the lost paren mass go?")
print("mean prob of '(X' tokens (all item letters), of bare, of space, and of NON-letter (1 - letter mass)")
sub = one[(one.turn == "single") & (one.kind != "neutral")]
for t in ("original",) + tuple(tags):
    raw = np.stack(sub[f"{t}_first_raw"].values)  # n x 15 log-probs
    from common import HERE
    import json
    slot = json.load(open("/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/e3/slot_ids.json"))
    order = [v for _, v in sorted(((int(k), v) for k, v in slot.items()))]
    letters = sub.letters.values
    p = np.exp(raw)
    mask = np.array([[L in ls for (_, L) in order] for ls in letters])
    p = p * mask
    sp = np.array([s for s, _ in order])
    tot = p.sum(1)
    print(f"{t:13} paren {p[:, sp == 'paren'].sum(1).mean():.3f}  bare {p[:, sp == 'bare'].sum(1).mean():.3f}  "
          f"space {p[:, sp == 'space'].sum(1).mean():.3f}  non-letter {(1 - tot).mean():.3f}")
