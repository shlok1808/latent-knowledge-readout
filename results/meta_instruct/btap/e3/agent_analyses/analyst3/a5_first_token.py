#!/usr/bin/env python
"""A5: Where does the lost first-position letter mass go? The answer_sentence reply starts with 'The' and the
letter_last reply with 'Looking', so logp[0] of those rows IS log p(first token = 'The' / 'Looking' | prompt).
Also: per-token-position decomposition of the whole-reply delta for the sentence forms. Run: .venv/bin/python a5_first_token.py"""
import numpy as np
import pandas as pd
from common import load, boot, fmt

df = load()
tag = "pruned_seed0"
one_letter = df[df.form == "letter"].drop_duplicates("pid").set_index("pid")
one_as = df[df.form == "answer_sentence"].drop_duplicates("pid").set_index("pid")
one_ll = df[df.form == "letter_last"].drop_duplicates("pid").set_index("pid")
idx = one_letter.index.intersection(one_as.index).intersection(one_ll.index)
T = pd.DataFrame({
    "set": one_letter.loc[idx, "set"], "kind": one_letter.loc[idx, "kind"], "turn": one_letter.loc[idx, "turn"],
    "item_id": one_letter.loc[idx, "item_id"],
})
for t in ("original", tag, "random_seed0"):
    T[f"{t}_letters"] = np.exp(one_letter.loc[idx, f"{t}_first_mass"].values)
    T[f"{t}_The"] = np.exp([v[0] for v in one_as.loc[idx, f"{t}_logp"].values])
    T[f"{t}_Looking"] = np.exp([v[0] for v in one_ll.loc[idx, f"{t}_logp"].values])
print("=== Mean first-token probability: all letter spellings vs 'The' vs 'Looking' (one row per prompt)")
for (st, kd, tn), g in T.groupby(["set", "kind", "turn"]):
    line = f"{st:8} {kd:10} {tn:6} n={len(g):3} | "
    for t in ("original", tag):
        line += f"{t[:8]}: letters {g[f'{t}_letters'].mean():.3f} The {g[f'{t}_The'].mean():.3f} Looking {g[f'{t}_Looking'].mean():.4f} rest {1-g[f'{t}_letters'].mean()-g[f'{t}_The'].mean()-g[f'{t}_Looking'].mean():.3f} | "
    print(line)
print("\n=== Share of the LOST letter mass that 'The' picks up: mean over prompts of dThe / (-dLetters), pushed single-turn")
g = T[(T.turn == "single") & (T.kind != "neutral")].copy()
g["dL"] = g[f"{tag}_letters"] - g.original_letters
g["dThe"] = g[f"{tag}_The"] - g.original_The
print(f"sum over prompts: letters lost {-g.dL.sum():.1f}, 'The' gained {g.dThe.sum():.1f}  -> ratio {g.dThe.sum()/(-g.dL.sum()):.3f}")
g2 = T[(T.kind == "neutral")].copy()
g2["dL"] = g2[f"{tag}_letters"] - g2.original_letters
g2["dThe"] = g2[f"{tag}_The"] - g2.original_The
print(f"neutral: letters lost {-g2.dL.sum():.1f}, 'The' gained {g2.dThe.sum():.1f} -> ratio {g2.dThe.sum()/(-g2.dL.sum()):.3f}")
g3 = T[(T.turn == "multi")].copy()
g3["dL"] = g3[f"{tag}_letters"] - g3.original_letters
g3["dThe"] = g3[f"{tag}_The"] - g3.original_The
print(f"multi: letters lost {-g3.dL.sum():.1f}, 'The' gained {g3.dThe.sum():.1f} -> ratio {g3.dThe.sum()/(-g3.dL.sum()):.3f}")

print("\n=== Per-token-position delta (pruned - original, mean log-prob) for the answer_sentence form, by set/kind/turn/role")
print("    tokens: 0 'The' 1 ' correct' 2 ' answer' 3 ' is' 4 ' (' 5 LETTER 6 ')' 7.. option text ... '.' eot")
for (st, kd, tn, rl), g in df[(df.form == "answer_sentence") & df.set.isin(["heldout", "neutral"])].groupby(["set", "kind", "turn", "role"]):
    o = np.array([v[:7] for v in g.original_logp.values])
    p = np.array([v[:7] for v in g[f"{tag}_logp"].values])
    d = (p - o).mean(0)
    tail_o = np.array([v[7:].sum() for v in g.original_logp.values])
    tail_p = np.array([v[7:].sum() for v in g[f"{tag}_logp"].values])
    print(f"{st:8} {kd:10} {tn:6} {rl:6} n={len(g):3} " + " ".join(f"{x:+.2f}" for x in d) + f"  | tail(option text+eot) {(tail_p-tail_o).mean():+.2f}  | whole {(np.array([v.sum() for v in g[f'{tag}_logp'].values]) - np.array([v.sum() for v in g.original_logp.values])).mean():+.2f}")

print("\n=== Same for letter_last form: first 4 tokens, then the letter slot and the rest")
for (st, kd, tn, rl), g in df[(df.form == "letter_last") & df.set.isin(["heldout", "neutral"])].groupby(["set", "kind", "turn", "role"]):
    o0 = np.array([v[:4] for v in g.original_logp.values]); p0 = np.array([v[:4] for v in g[f"{tag}_logp"].values])
    dslot = np.mean([g[f"{tag}_logp"].values[i][a] - g.original_logp.values[i][a] for i, a in enumerate(g.answer_idx.values)])
    dmid = np.mean([g[f"{tag}_logp"].values[i][4:a].sum() - g.original_logp.values[i][4:a].sum() for i, a in enumerate(g.answer_idx.values)])
    print(f"{st:8} {kd:10} {tn:6} {rl:6} first4 " + " ".join(f"{x:+.2f}" for x in (p0 - o0).mean(0)) + f"  mid(option text) {dmid:+.2f}  LETTER {dslot:+.2f}")

print("\n=== pres set (protection check): whole-reply delta and first-token delta")
g = df[df.set == "pres"].copy()
g["d"] = g[f"{tag}_sum"] - g.original_sum
g["d0"] = [p[0] - o[0] for p, o in zip(g[f"{tag}_logp"], g.original_logp)]
g["drand"] = g.random_seed0_sum - g.original_sum
print("whole", fmt(boot(g, "d")), " first-token", fmt(boot(g, "d0")), " random whole", fmt(boot(g, "drand")))
print("pres: mean p(first letter token) original", np.exp(g.original_logp.apply(lambda v: v[0])).mean().round(3),
      "pruned", np.exp(g[f"{tag}_logp"].apply(lambda v: v[0])).mean().round(3))
