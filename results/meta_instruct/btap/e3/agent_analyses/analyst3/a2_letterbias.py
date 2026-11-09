#!/usr/bin/env python
"""A2: After pruning, does the letter-restricted distribution at the answer slot shift toward particular letters
(A..E position), toward the user's cue letter, toward the model's earlier letter (multi-turn), toward gold, or
toward uniform (entropy)? Run: .venv/bin/python a2_letterbias.py"""
import numpy as np
import pandas as pd
from common import load, boot, fmt

df = load()
tag = "pruned_seed0"
# one row per prompt+form (cave and resist rows share the slot distribution for letter/answer_sentence/letter_last;
# the cave row is used)
base = df[(df.role != "resist") & df.form.isin(["letter", "answer_sentence", "letter_last"])].copy()

print("=== Mean letter-restricted PROB by option position (A..E), 5-option items only, slot of the letter form")
for (st, kd, tn), g in base[(base.form == "letter") & (base.n_opt == 5)].groupby(["set", "kind", "turn"]):
    line = f"{st:8} {kd:10} {tn:6} n={len(g):3} | "
    for t in ("original", tag, "random_seed0"):
        ps = [np.exp(g[f"{t}_slot_r_{L}"]).mean() for L in "ABCDE"]
        line += f"{t[:8]}: " + " ".join(f"{p:.3f}" for p in ps) + " | "
    print(line)
print("\n=== Same, 4-option items")
for (st, kd, tn), g in base[(base.form == "letter") & (base.n_opt == 4)].groupby(["set", "kind", "turn"]):
    line = f"{st:8} {kd:10} {tn:6} n={len(g):3} | "
    for t in ("original", tag, "random_seed0"):
        ps = [np.exp(g[f"{t}_slot_r_{L}"]).mean() for L in "ABCD"]
        line += f"{t[:8]}: " + " ".join(f"{p:.3f}" for p in ps) + " | "
    print(line)

print("\n=== Change in restricted log-prob of GOLD, CUE letter, PREV (multi) letter, and mean OTHER letter at the slot")
print("    (pruned - original; per-question cluster bootstrap)")
for (st, kd, tn, fm), g in base[base.set.isin(["heldout", "prune", "neutral"])].groupby(["set", "kind", "turn", "form"]):
    g = g.copy()
    g["d_gold"] = g[f"{tag}_slot_r_gold"] - g["original_slot_r_gold"]
    g["d_cue"] = g[f"{tag}_slot_r_cue"] - g["original_slot_r_cue"]
    g["d_prev"] = g[f"{tag}_slot_r_prev"] - g["original_slot_r_prev"]
    # other letters: neither gold nor cue
    def other(row):
        Ls = [L for L in row.letters if L not in (row.gold, row.cue_letter)]
        return np.mean([row[f"{tag}_slot_r_{L}"] - row[f"original_slot_r_{L}"] for L in Ls])
    g["d_other"] = g.apply(other, axis=1)
    g["dH"] = g[f"{tag}_slot_H"] - g["original_slot_H"]
    print(f"{st:8} {kd:10} {tn:6} {fm:15} gold {fmt(boot(g,'d_gold'),2)}  cue {fmt(boot(g,'d_cue'),2) if g.d_cue.notna().any() else 'n/a'}  "
          f"other {fmt(boot(g,'d_other'),2)}  prev {fmt(boot(g,'d_prev'),2) if g.d_prev.notna().any() else 'n/a'}  dH {fmt(boot(g,'dH'),2)}")

print("\n=== Restricted ENTROPY at the slot (nats): original vs pruned vs random, by set/kind/turn/form")
for (st, kd, tn, fm), g in base.groupby(["set", "kind", "turn", "form"]):
    print(f"{st:8} {kd:10} {tn:6} {fm:15} H orig {g.original_slot_H.mean():.3f}  pruned {g[f'{tag}_slot_H'].mean():.3f}  "
          f"random {g.random_seed0_slot_H.mean():.3f}")

print("\n=== Argmax letter at the slot: fraction where pruned argmax == original argmax; and what the pruned argmax is")
for (st, kd, tn, fm), g in base[base.set.isin(["heldout", "prune"])].groupby(["set", "kind", "turn", "form"]):
    same = (g[f"{tag}_slot_argmax"] == g.original_slot_argmax).mean()
    o_gold = (g.original_slot_argmax == g.gold).mean()
    p_gold = (g[f"{tag}_slot_argmax"] == g.gold).mean()
    o_cue = (g.original_slot_argmax == g.cue_letter).mean()
    p_cue = (g[f"{tag}_slot_argmax"] == g.cue_letter).mean()
    o_prev = (g.original_slot_argmax == g.prev_letter).mean() if tn == "multi" else float("nan")
    p_prev = (g[f"{tag}_slot_argmax"] == g.prev_letter).mean() if tn == "multi" else float("nan")
    print(f"{st:8} {kd:10} {tn:6} {fm:15} same {same:.3f} | argmax=gold {o_gold:.3f}->{p_gold:.3f} | =cue {o_cue:.3f}->{p_cue:.3f} | =prev {o_prev:.3f}->{p_prev:.3f}")
