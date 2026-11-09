#!/usr/bin/env python3
"""Verifier part 2: real-reply agreement, neutral gold shift, within-cell fits, flattening-adjusted DiDs."""
import json, math
from collections import defaultdict
import numpy as np

E3 = "/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/e3/"
ex = [json.loads(l) for l in open(E3 + "examples.jsonl")]
slot_ids = json.loads(open(E3 + "slot_ids.json").read())
ORDER = [tuple(v) for _, v in sorted(((int(k), v) for k, v in slot_ids.items()))]
SC = {t: [json.loads(l) for l in open(E3 + f"scores_{t}.jsonl")] for t in ("original", "pruned_seed0")}
LET = {r["item_id"]: r["letters"] for r in ex}

def lse(a):
    a = np.asarray(a, float); m = a.max(); return m + math.log(np.exp(a - m).sum())
def restricted(vec, letters):
    by = defaultdict(list)
    for v, (_, L) in zip(vec, ORDER):
        if L in letters: by[L].append(v)
    lp = {L: lse(v) for L, v in by.items()}; tot = lse(list(lp.values()))
    return {L: v - tot for L, v in lp.items()}, tot
def fit(x, y):
    x, y = np.asarray(x), np.asarray(y); A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]; return a, b
RNG = np.random.default_rng(7)
def cboot(by_q, B=2000):
    keys = sorted(by_q); qm = np.array([np.mean(by_q[k]) for k in keys]); n = len(qm)
    est = qm.mean(); bs = qm[RNG.integers(0, n, size=(B, n))].mean(1); bs.sort()
    return f"{est:+.3f} [{bs[int(.025*B)]:+.3f},{bs[int(.975*B)]:+.3f}] n_q={n}"

print("==== A. Neutral: change in restricted log p(gold) (pruned - original), letter and sentence slots")
for form in ("letter", "answer_sentence"):
    d = defaultdict(list); acc = {"original": [], "pruned_seed0": []}
    for i, r in enumerate(ex):
        if r["set"] != "neutral" or r["form"] != form: continue
        L = LET[r["item_id"]]; ro, _ = restricted(SC["original"][i]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][i]["slot"], L)
        d[r["item_id"]].append(rp[r["gold"]] - ro[r["gold"]])
        for t, rr in (("original", ro), ("pruned_seed0", rp)): acc[t].append(max(rr, key=rr.get) == r["gold"])
    print(f"   {form:16} d lr(gold) {cboot(d)} | restricted argmax==gold: orig {np.mean(acc['original']):.3f} pruned {np.mean(acc['pruned_seed0']):.3f}")

print("\n==== B. Within-cell fit lo_pruned = a + b lo_orig (cave - resist), heldout single, and neutral-adjusted DiD")
NEU = {"letter": (-0.09, 0.726), "answer_sentence": (0.00, 0.918)}
pairs = defaultdict(dict)
for i, r in enumerate(ex): pairs[(r["pid"], r["form"])][r["role"]] = i
for form in ("letter", "answer_sentence"):
    for turn in ("single", "multi"):
        for kind in ("suggestion", "doubt"):
            xs, ys, adj, within = [], [], defaultdict(list), defaultdict(list)
            for (pid, f), d in pairs.items():
                if f != form or "cave" not in d or "resist" not in d: continue
                r = ex[d["cave"]]
                if r["set"] != "heldout" or r["turn"] != turn or r["kind"] != kind: continue
                L = LET[r["item_id"]]; ro, _ = restricted(SC["original"][d["cave"]]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][d["cave"]]["slot"], L)
                c, g = r["letter"], ex[d["resist"]]["letter"]
                x, y = ro[c] - ro[g], rp[c] - rp[g]; xs.append(x); ys.append(y)
                a0, b0 = NEU[form]; adj[r["item_id"]].append(y - (a0 + b0 * x))
            a, b = fit(xs, ys)
            # within-cell adjusted = residual at each point from the within-cell fit = a (intercept), so report a
            print(f"   heldout {turn:6} {kind:10} {form:16} within-cell a={a:+.2f} b={b:.2f} | raw DiD {np.mean(np.array(ys)-np.array(xs)):+.3f} | neutral-fit-adjusted DiD {cboot(adj)}")

print("\n==== C. Real replies: at the answer slot, does the OTHER model's restricted argmax agree with the written letter?")
for by in ("original", "pruned_seed0"):
    other = "pruned_seed0" if by == "original" else "original"
    agree_other, agree_self, n = defaultdict(list), defaultdict(list), 0
    for i, r in enumerate(ex):
        if r["set"] != "real" or r["written_by"] != by or r["form"] != "real_as_written": continue
        L = LET[r["item_id"]]; ro, _ = restricted(SC[other][i]["slot"], L); rs, _ = restricted(SC[by][i]["slot"], L)
        agree_other[r["real_style"]].append(max(ro, key=ro.get) == r["letter"]); agree_self[r["real_style"]].append(max(rs, key=rs.get) == r["letter"]); n += 1
    for st in agree_other:
        print(f"   written_by {by:13} style {st:14} n={len(agree_other[st]):3d}  {other} argmax==written {np.mean(agree_other[st]):.3f} | author argmax==written {np.mean(agree_self[st]):.3f}")
# raw slot DiD for real, split by author and by role as written
print("\n==== D. Real replies: slot DiD (cave minus resist) by author x written role; whole-reply delta per row by author")
rows_by_pid = defaultdict(dict)
for i, r in enumerate(ex):
    if r["set"] == "real": rows_by_pid[r["pid"]][r["form"]] = i
sl = defaultdict(lambda: defaultdict(list)); wh = defaultdict(lambda: defaultdict(list))
for pid, d in rows_by_pid.items():
    if "real_as_written" not in d or "real_twin" not in d: continue
    iw, it = d["real_as_written"], d["real_twin"]; r = ex[iw]; q = r["item_id"]; L = LET[q]
    ro, _ = restricted(SC["original"][iw]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][iw]["slot"], L)
    ic, ir = (iw, it) if r["role"] == "cave" else (it, iw)
    v = (rp[ex[ic]["letter"]] - rp[ex[ir]["letter"]]) - (ro[ex[ic]["letter"]] - ro[ex[ir]["letter"]])
    sl[(r["written_by"], "all")][q].append(v); sl[(r["written_by"], "written=" + r["role"])][q].append(v); sl[("pooled", "all")][q].append(v)
    for j, lab in ((iw, "as_written"), (it, "twin")):
        wh[(r["written_by"], lab)][q].append(sum(SC["pruned_seed0"][j]["logp"]) - sum(SC["original"][j]["logp"]))
for k in sorted(sl): print(f"   slot DiD {k[0]:13} {k[1]:16} {cboot(sl[k])}")
for k in sorted(wh): print(f"   whole-reply delta (pruned-orig) {k[0]:13} {k[1]:11} {cboot(wh[k])}")
