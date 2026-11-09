#!/usr/bin/env python
"""A7: Is the first-position letter-mass loss pressure-specific, or a uniform shift? Compare on the LOGIT scale:
d = logit(p_pruned) - logit(p_orig) of total letter mass at the first reply position, per prompt, by set/kind/turn.
Also a regression of d on prompt features (kind, turn, dataset, n_opt, prompt_len, original logit).
Run: .venv/bin/python a7_format_logit.py"""
import numpy as np
import pandas as pd
from common import load, boot, fmt

df = load()
tag = "pruned_seed0"
one = df[df.form.isin(["letter", "pres_qa"])].drop_duplicates("pid").copy()


def logit_from_logp(lp):
    p = np.exp(lp)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


one["lg_o"] = logit_from_logp(one.original_first_mass.values)
one["lg_p"] = logit_from_logp(one[f"{tag}_first_mass"].values)
one["lg_r"] = logit_from_logp(one.random_seed0_first_mass.values)
one["d"] = one.lg_p - one.lg_o
one["d_rand"] = one.lg_r - one.lg_o
one["lr"] = one[f"{tag}_first_mass"] - one.original_first_mass
print("=== Change in LOGIT of first-position letter mass (pruned - original), per prompt, cluster bootstrap")
for (st, kd, tn), g in one.groupby(["set", "kind", "turn"]):
    print(f"{st:8} {kd:10} {tn:6} logit d {fmt(boot(g,'d'),2)}   (log-ratio {boot(g,'lr')['mean']:+.2f}; orig p {np.exp(g.original_first_mass).mean():.3f} logit {g.lg_o.mean():+.2f} -> pruned p {np.exp(g[f'{tag}_first_mass']).mean():.3f} logit {g.lg_p.mean():+.2f}; random d {boot(g,'d_rand',reps=200)['mean']:+.3f})")

print("\n=== Pooled: neutral+pres vs pushed single vs pushed multi")
for name, m in (("no pressure (neutral+pres)", one.set.isin(["neutral", "pres"])),
                ("pushed single", (one.turn == "single") & one.set.isin(["heldout", "prune"])),
                ("pushed multi doubt", (one.turn == "multi") & (one.kind == "doubt")),
                ("pushed multi suggestion", (one.turn == "multi") & (one.kind == "suggestion"))):
    print(f"{name:28} {fmt(boot(one[m],'d'),2)}")

print("\n=== OLS of logit change on features (pushed + neutral single-turn prompts)")
sub = one[one.set.isin(["heldout", "prune", "neutral"]) & (one.turn == "single")].copy()
X = pd.DataFrame({
    "const": 1.0,
    "doubt": (sub.kind == "doubt").astype(float),
    "suggestion": (sub.kind == "suggestion").astype(float),
    "arc": (sub.dataset == "arc_challenge").astype(float),
    "obqa": (sub.dataset == "obqa").astype(float),
    "n_opt5": (sub.n_opt == 5).astype(float),
    "prompt_len_z": (sub.prompt_len - sub.prompt_len.mean()) / sub.prompt_len.std(),
    "lg_o": sub.lg_o,
})
beta, *_ = np.linalg.lstsq(X.values, sub.d.values, rcond=None)
pred = X.values @ beta
r2 = 1 - ((sub.d - pred) ** 2).sum() / ((sub.d - sub.d.mean()) ** 2).sum()
print("R2", round(r2, 3))
for k, b in zip(X.columns, beta):
    print(f"   {k:14} {b:+.3f}")
print("\n=== Same, pushed single only, with cue wording (template) dummies: which wordings break the format most?")
sub2 = one[one.set.isin(["heldout", "prune"]) & (one.turn == "single")].copy()
# template = the cue with the option letter and quoted text removed
import re
sub2["tmpl"] = sub2.cue.str.replace(r'\("[^"]*"\)', '("…")', regex=True).str.replace(r"[Oo]ption [A-E]", "option X", regex=True)
for t, g in sub2.groupby("tmpl"):
    if len(g) >= 20:
        print(f"n={len(g):3} d {boot(g,'d',reps=500)['mean']:+.2f}  orig p {np.exp(g.original_first_mass).mean():.2f} -> {np.exp(g[f'{tag}_first_mass']).mean():.2f} | {t[:90]}")
print("\nmulti-turn cue wordings:")
sub3 = one[one.set.isin(["heldout", "prune"]) & (one.turn == "multi")].copy()
sub3["tmpl"] = sub3.cue.str.replace(r'\("[^"]*"\)', '("…")', regex=True).str.replace(r"[Oo]ption [A-E]", "option X", regex=True)
for t, g in sub3.groupby("tmpl"):
    if len(g) >= 20:
        print(f"n={len(g):3} d {boot(g,'d',reps=500)['mean']:+.2f}  orig p {np.exp(g.original_first_mass).mean():.2f} -> {np.exp(g[f'{tag}_first_mass']).mean():.2f} | {t[:90]}")
