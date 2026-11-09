#!/usr/bin/env python
"""A3: DiD heterogeneity. Reproduces the pre-named DiD per pair, then asks: does it scale with the original's
log-odds (dose-response)? by dataset / n_opt / cue letter / gold position / option length? Why is the multi-turn
letter-form DiD so large? Run: .venv/bin/python a3_did.py"""
import numpy as np
import pandas as pd
from common import load, boot, fmt, pairs, did

df = load()
c, r = pairs(df)
tag = "pruned_seed0"
d, lo, lp = did(c, r, tag)
dr, _, lrnd = did(c, r, "random_seed0")
P = c.reset_index()[["pid", "form", "set", "kind", "turn", "dataset", "n_opt", "item_id", "letter", "gold",
                      "cue_letter", "prev_letter", "letter_pos", "gold_pos", "opt_len", "mean_opt_len", "prompt_len",
                      "q_len", "written_by", "real_style", "cue"]].copy()
P["did"] = d
P["did_rand"] = dr
P["lo_orig"] = lo      # original log-odds cave vs resist (restricted)
P["lo_pruned"] = lp
P["resist_letter"] = r.reset_index().letter.values
P["cave_pos"] = P.letter_pos
P["resist_pos"] = [P.letters[i].index(L) if False else 0 for i, L in enumerate(P.resist_letter)] if False else [
    c.reset_index().letters.values[i].index(L) for i, L in enumerate(P.resist_letter)]

print("=== Reproduce the pre-named DiD table (pruned) and the random control")
for (st, fm, tn), g in P[P.set != "real"].groupby(["set", "form", "turn"]):
    print(f"{st:8} {fm:15} {tn:6} pruned {fmt(boot(g,'did'))}   random {fmt(boot(g,'did_rand'))}")

print("\n=== Dose-response: DiD vs original log-odds (cave - resist), heldout+prune, by form/turn")
print("    quartiles of lo_orig; mean DiD per quartile. Caveat: conditioning on the original's value.")
for (fm, tn), g in P[P.set.isin(["heldout", "prune"])].groupby(["form", "turn"]):
    g = g.copy()
    g["q"] = pd.qcut(g.lo_orig, 4, labels=False, duplicates="drop")
    s = g.groupby("q").agg(lo=("lo_orig", "mean"), did=("did", "mean"), n=("did", "size"))
    corr = np.corrcoef(g.lo_orig, g.did)[0, 1]
    # also correlate with the AVERAGE of orig and pruned log-odds (less regression-to-mean bias)
    corr2 = np.corrcoef((g.lo_orig + g.lo_pruned) / 2, g.did)[0, 1]
    print(f"{fm:15} {tn:6} corr(lo_orig, did)={corr:+.3f}  corr(avg lo, did)={corr2:+.3f}  " +
          "  ".join(f"Q{int(q)}: lo {row.lo:+.2f} did {row.did:+.2f} (n {int(row.n)})" for q, row in s.iterrows()))

print("\n=== DiD by dataset (heldout+prune)")
for (fm, tn, ds), g in P[P.set.isin(["heldout", "prune"])].groupby(["form", "turn", "dataset"]):
    print(f"{fm:15} {tn:6} {ds:14} {fmt(boot(g,'did'),2)}  lo_orig {g.lo_orig.mean():+.2f}")

print("\n=== DiD by kind (suggestion vs doubt), heldout+prune")
for (fm, tn, kd), g in P[P.set.isin(["heldout", "prune"])].groupby(["form", "turn", "kind"]):
    print(f"{fm:15} {tn:6} {kd:10} {fmt(boot(g,'did'),2)}  lo_orig {g.lo_orig.mean():+.2f} lo_pruned {g.lo_pruned.mean():+.2f}")

print("\n=== DiD by cave-letter position and resist-letter position (letter form, single and multi, heldout+prune)")
for tn in ("single", "multi"):
    g = P[(P.form == "letter") & (P.turn == tn) & P.set.isin(["heldout", "prune"])]
    print(f"-- {tn}: by cave letter:", {L: f"{g[g.letter==L].did.mean():+.2f} (n{(g.letter==L).sum()})" for L in "ABCDE"})
    print(f"   {tn}: by resist letter:", {L: f"{g[g.resist_letter==L].did.mean():+.2f} (n{(g.resist_letter==L).sum()})" for L in "ABCDE"})
    print(f"   {tn}: cave before resist (pos) {g[g.cave_pos < g.resist_pos].did.mean():+.2f} (n{(g.cave_pos < g.resist_pos).sum()}) "
          f"vs cave after resist {g[g.cave_pos > g.resist_pos].did.mean():+.2f} (n{(g.cave_pos > g.resist_pos).sum()})")

print("\n=== Multi-turn letter form: decompose. Restricted log-prob of the PREV (= resist) letter and of the cave letter")
g = P[(P.form == "letter") & (P.turn == "multi") & P.set.isin(["heldout", "prune"])].copy()
cc = c.reset_index()
cc = cc[(cc.form == "letter") & (cc.turn == "multi") & cc.set.isin(["heldout", "prune"])]
for t in ("original", tag, "random_seed0"):
    rp = np.array([cc[f"{t}_slot_r_{L}"].values[i] for i, L in enumerate(cc.prev_letter.values)])
    rc = np.array([cc[f"{t}_slot_r_{L}"].values[i] for i, L in enumerate(cc.letter.values)])
    print(f"{t:13} mean restricted p(prev=resist) {np.exp(rp).mean():.3f}  mean restricted p(cave) {np.exp(rc).mean():.3f}  "
          f"mean log-odds cave-resist {(rc-rp).mean():+.2f}  letter mass at slot {np.exp(cc[f'{t}_slot_mass']).mean():.3f}")
print("multi-turn letter form, by kind:")
for kd, gg in cc.groupby("kind"):
    for t in ("original", tag):
        rp = np.array([gg[f"{t}_slot_r_{L}"].values[i] for i, L in enumerate(gg.prev_letter.values)])
        rc = np.array([gg[f"{t}_slot_r_{L}"].values[i] for i, L in enumerate(gg.letter.values)])
        print(f"   {kd:10} {t:13} p(prev) {np.exp(rp).mean():.3f} p(cave) {np.exp(rc).mean():.3f} lo {(rc-rp).mean():+.2f}")

print("\n=== Single-turn letter form by kind: restricted p(cave), p(resist=gold)")
cc = c.reset_index()
cc = cc[(cc.form == "letter") & (cc.turn == "single") & cc.set.isin(["heldout", "prune"])]
for (st, kd), gg in cc.groupby(["set", "kind"]):
    for t in ("original", tag):
        rg = gg[f"{t}_slot_r_gold"].values
        rc = np.array([gg[f"{t}_slot_r_{L}"].values[i] for i, L in enumerate(gg.letter.values)])
        print(f"   {st:8} {kd:10} {t:13} p(cave) {np.exp(rc).mean():.3f} p(gold) {np.exp(rg).mean():.3f} lo {(rc-rg).mean():+.2f}")

print("\n=== Correlation of DiD across forms for the same prompt (is the slot-level 'belief' change consistent?)")
w = P[P.set.isin(["heldout", "prune"])].pivot_table(index="pid", columns="form", values="did")
print(w.corr().round(3))
print("\n=== Correlation of DiD with first-position letter-mass loss (format term) per prompt")
one = df[df.form == "letter"].drop_duplicates("pid").set_index("pid")
fmt_loss = one[f"{tag}_first_mass"] - one["original_first_mass"]
for fm in ("letter", "answer_sentence", "letter_last"):
    for tn in ("single", "multi"):
        g = P[(P.form == fm) & (P.turn == tn) & P.set.isin(["heldout", "prune"])]
        x = fmt_loss.loc[g.pid].values
        print(f"{fm:15} {tn:6} corr(format loss, DiD) = {np.corrcoef(x, g.did)[0,1]:+.3f}")

P.to_pickle("/private/tmp/claude-501/-Users-shloki-latent-knowledge-readout/ab0e02ee-90de-4560-9187-a85503a906ac/scratchpad/analyst3/pairs.pkl")
