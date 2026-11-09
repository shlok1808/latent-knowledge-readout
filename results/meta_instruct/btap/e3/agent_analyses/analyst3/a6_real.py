#!/usr/bin/env python
"""A6: the REAL set. Replies actually written by the original / pruned model (256-token runs, user source, initially
correct cells) plus a twin with only the final letter swapped. For each (reply, twin) pair and each scoring model:
pref = logp(as_written) - logp(twin) = how strongly the scorer prefers the letter the writer chose, given the writer's
own reasoning. Also: how much more does the pruned scorer like the pruned model's reasons-first replies than the
original scorer does (style preference), and letter-slot restricted log-odds. Run: .venv/bin/python a6_real.py"""
import numpy as np
import pandas as pd
from common import load, boot, fmt

df = load()
tag = "pruned_seed0"
real = df[df.set == "real"].copy()
w = real[real.form == "real_as_written"].set_index("pid")
t = real[real.form == "real_twin"].set_index("pid")
idx = w.index.intersection(t.index)
w, t = w.loc[idx], t.loc[idx]
R = pd.DataFrame({"item_id": w.item_id, "kind": w.kind, "written_by": w.written_by, "style": w.real_style,
                  "written_role": w.role, "n_reply": w.n_reply})
for m in ("original", tag, "random_seed0"):
    R[f"pref_{m}"] = w[f"{m}_sum"].values - t[f"{m}_sum"].values          # whole-reply preference for the written letter
    R[f"slot_{m}"] = w[f"{m}_slot_r_letter"].values - t[f"{m}_slot_r_letter"].values  # restricted log-odds at the slot
    R[f"sum_{m}"] = w[f"{m}_sum"].values
R["d_pref"] = R[f"pref_{tag}"] - R.pref_original
R["d_slot"] = R[f"slot_{tag}"] - R.slot_original
R["d_sum"] = R[f"sum_{tag}"] - R.sum_original
R["d_sum_pertok"] = R.d_sum / R.n_reply

print("counts:")
print(R.groupby(["written_by", "style", "kind", "written_role"]).size())

print("\n=== Preference for the WRITTEN final letter over the swapped twin (whole reply, nats), by scorer")
for (wb, st, kd, rl), g in R.groupby(["written_by", "style", "kind", "written_role"]):
    if len(g) < 5:
        continue
    print(f"written_by {wb:13} {st:13} {kd:10} writer {rl:6} n={len(g):3} | orig {g.pref_original.mean():+.2f}  pruned {g[f'pref_{tag}'].mean():+.2f}  "
          f"random {g.pref_random_seed0.mean():+.2f} | pruned-orig {fmt(boot(g,'d_pref'),2)}")

print("\n=== Same at the letter SLOT only (restricted log-odds written vs twin letter)")
for (wb, st, kd, rl), g in R.groupby(["written_by", "style", "kind", "written_role"]):
    if len(g) < 5:
        continue
    print(f"written_by {wb:13} {st:13} {kd:10} writer {rl:6} n={len(g):3} | orig {g.slot_original.mean():+.2f}  pruned {g[f'slot_{tag}'].mean():+.2f} | d {fmt(boot(g,'d_slot'),2)}")

print("\n=== Style preference: whole-reply log-prob of the AS-WRITTEN reply, pruned - original (per token too)")
for (wb, st), g in R.groupby(["written_by", "style"]):
    print(f"written_by {wb:13} {st:13} n={len(g):3} | d_sum {fmt(boot(g,'d_sum'),2)}  per token {g.d_sum_pertok.mean():+.3f}  mean len {g.n_reply.mean():.1f}")

print("\n=== Key comparison: pruned model's reasons-first replies. Does the ORIGINAL model, given that reasoning, agree with the")
print("    letter the pruned model wrote? (pref_original > 0 means yes). Split by whether the pruned writer caved or resisted.")
g = R[(R.written_by == tag) & (R["style"] == "reasons_first")]
for (kd, rl), gg in g.groupby(["kind", "written_role"]):
    print(f"{kd:10} writer {rl:6} n={len(gg):3} | orig agrees (pref>0) {(gg.pref_original>0).mean():.2f}  pruned agrees {(gg[f'pref_{tag}']>0).mean():.2f} | "
          f"slot orig>0 {(gg.slot_original>0).mean():.2f} pruned>0 {(gg[f'slot_{tag}']>0).mean():.2f}")
print("\n=== And the original model's letter-first replies: does the PRUNED model agree with the original's letter?")
g = R[(R.written_by == "original") & (R["style"] == "letter_first")]
for (kd, rl), gg in g.groupby(["kind", "written_role"]):
    print(f"{kd:10} writer {rl:6} n={len(gg):3} | orig agrees {(gg.pref_original>0).mean():.2f}  pruned agrees {(gg[f'pref_{tag}']>0).mean():.2f} | "
          f"mean pref orig {gg.pref_original.mean():+.2f} pruned {gg[f'pref_{tag}'].mean():+.2f}")
