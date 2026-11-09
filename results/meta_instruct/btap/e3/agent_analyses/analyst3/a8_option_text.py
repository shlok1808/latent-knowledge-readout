#!/usr/bin/env python
"""A8: Belief signal in the OPTION-TEXT span of the letter_last form ("Looking at the options, <text> fits the
question best, so the answer is (X)."). The text comes BEFORE any letter, so its log-prob is where the model's
choice is expressed without a letter token. DiD_text = [lp_pruned(cave text) - lp_orig(cave text)] -
[lp_pruned(resist text) - lp_orig(resist text)], span = tokens 4 .. answer_idx-1 (after 'Looking at the options,').
Also the same for the answer_sentence tail (text AFTER the letter) as a contrast. Run: .venv/bin/python a8_option_text.py"""
import numpy as np
import pandas as pd
from common import load, boot, fmt

df = load()
tag = "pruned_seed0"


def span_sum(row, m, a, b):
    v = row[f"{m}_logp"]
    return float(np.sum(v[a:b]))


for form, name in (("letter_last", "letter_last: option text BEFORE the letter (tokens 4..answer_idx)"),
                   ("answer_sentence", "answer_sentence: option text AFTER the letter (tokens answer_idx+2..end-1)")):
    print(f"\n=== {name}")
    sub = df[df.form == form].copy()
    if form == "letter_last":
        sub["d_text"] = [span_sum(r, tag, 4, r.answer_idx) - span_sum(r, "original", 4, r.answer_idx) for _, r in sub.iterrows()]
        sub["d_text_rand"] = [span_sum(r, "random_seed0", 4, r.answer_idx) - span_sum(r, "original", 4, r.answer_idx) for _, r in sub.iterrows()]
        sub["ntok"] = sub.answer_idx - 4
    else:
        sub["d_text"] = [span_sum(r, tag, r.answer_idx + 2, r.n_reply - 1) - span_sum(r, "original", r.answer_idx + 2, r.n_reply - 1) for _, r in sub.iterrows()]
        sub["d_text_rand"] = [span_sum(r, "random_seed0", r.answer_idx + 2, r.n_reply - 1) - span_sum(r, "original", r.answer_idx + 2, r.n_reply - 1) for _, r in sub.iterrows()]
        sub["ntok"] = sub.n_reply - 1 - (sub.answer_idx + 2)
    c = sub[sub.role == "cave"].set_index("pid")
    r = sub[sub.role == "resist"].set_index("pid")
    idx = c.index.intersection(r.index)
    P = pd.DataFrame({"item_id": c.loc[idx, "item_id"], "set": c.loc[idx, "set"], "kind": c.loc[idx, "kind"],
                      "turn": c.loc[idx, "turn"], "cave_d": c.loc[idx, "d_text"], "resist_d": r.loc[idx, "d_text"],
                      "did_text": c.loc[idx, "d_text"].values - r.loc[idx, "d_text"].values,
                      "did_text_rand": c.loc[idx, "d_text_rand"].values - r.loc[idx, "d_text_rand"].values,
                      "cave_ntok": c.loc[idx, "ntok"], "resist_ntok": r.loc[idx, "ntok"]})
    P["did_text_pertok"] = c.loc[idx, "d_text"].values / c.loc[idx, "ntok"].values - r.loc[idx, "d_text"].values / r.loc[idx, "ntok"].values
    for (st, kd, tn), g in P.groupby(["set", "kind", "turn"]):
        print(f"{st:8} {kd:10} {tn:6} cave text d {g.cave_d.mean():+.2f}  resist text d {g.resist_d.mean():+.2f}  DiD_text {fmt(boot(g,'did_text'),2)}  "
              f"per-token DiD {boot(g,'did_text_pertok')['mean']:+.3f}  random DiD {boot(g,'did_text_rand',reps=200)['mean']:+.3f}  ntok cave/resist {g.cave_ntok.mean():.1f}/{g.resist_ntok.mean():.1f}")
    neu = sub[sub.set == "neutral"]
    print(f"neutral (correct option text only): d {fmt(boot(neu,'d_text'),2)}  per token {(neu.d_text/neu.ntok).mean():+.3f}")
    if form == "letter_last":
        print("\n--- Correlation between the option-text DiD and the letter-slot DiD for the same prompt (letter form, single+multi)")
        pairs = pd.read_pickle("/private/tmp/claude-501/-Users-shloki-latent-knowledge-readout/ab0e02ee-90de-4560-9187-a85503a906ac/scratchpad/analyst3/pairs.pkl")
        for fm in ("letter", "answer_sentence"):
            q = pairs[(pairs.form == fm) & pairs.set.isin(["heldout", "prune"])].set_index("pid")
            j = q.index.intersection(P.index)
            for tn in ("single", "multi"):
                m = q.loc[j, "turn"] == tn
                print(f"   text DiD vs {fm:15} slot DiD, {tn:6}: r = {np.corrcoef(P.loc[j][m.values].did_text, q.loc[j][m.values].did)[0,1]:+.3f}  (n {int(m.sum())})")
