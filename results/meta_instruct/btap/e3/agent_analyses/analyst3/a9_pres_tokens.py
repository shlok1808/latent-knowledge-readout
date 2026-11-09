#!/usr/bin/env python
"""A9: pres set (the 256 protected QA examples, reply '(X)<|eot_id|>'): per-token delta pruned - original.
Run: .venv/bin/python a9_pres_tokens.py"""
import numpy as np
from common import load, boot, fmt
df = load(); tag = "pruned_seed0"
g = df[df.set == "pres"].copy()
print("reply token counts:", g.n_reply.value_counts().to_dict())
g3 = g[g.n_reply == 3].copy()
for i, name in enumerate(["'(X'", "')'", "eot"]):
    g3[f"d{i}"] = [p[i] - o[i] for p, o in zip(g3[f"{tag}_logp"], g3.original_logp)]
    print(f"token {i} {name:5} delta {fmt(boot(g3, f'd{i}'),3)}  mean p orig {np.exp([o[i] for o in g3.original_logp]).mean():.3f} -> pruned {np.exp([p[i] for p in g3[f'{tag}_logp']]).mean():.3f}")
g3["dall"] = g3[f"{tag}_sum"] - g3.original_sum
print("whole", fmt(boot(g3, "dall"), 3))
# compare with the neutral letter form ('(X)' + eot) on test questions
n = df[(df.set == "neutral") & (df.form == "letter")].copy()
for i, name in enumerate(["'(X'", "')'", "eot"]):
    n[f"d{i}"] = [p[i] - o[i] for p, o in zip(n[f"{tag}_logp"], n.original_logp)]
    print(f"neutral letter token {i} {name:5} delta {fmt(boot(n, f'd{i}'),3)}  p orig {np.exp([o[i] for o in n.original_logp]).mean():.3f} -> {np.exp([p[i] for p in n[f'{tag}_logp']]).mean():.3f}")
