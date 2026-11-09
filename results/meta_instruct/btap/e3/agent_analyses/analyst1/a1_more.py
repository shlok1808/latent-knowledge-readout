"""Analyst 1, part 2: neutral per form, slot letter mass per form, multi-turn spelling/repeat test, placebo by kind,
letter_last prefix-belief DiD, preservation decomposition.  Run: python3 a1_more.py > a1_more.out"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_lib import cluster_boot, fmt, letter_lp, load, pairs, restricted  # noqa

ex, spell, let, sc = load()
T = "pruned_seed0"
P = pairs(ex)

# ---- A. letter mass at the ANSWER slot and at FIRST position, per set/form/turn (log ratio pruned/orig) ----
print("=== A. log letter mass (item letters, all spellings): orig mean | pruned-orig log-ratio  [slot] and [first] ===")
cells = defaultdict(lambda: defaultdict(list))
for i, r in enumerate(ex):
    if r["set"] not in ("heldout", "neutral", "pres", "prune"):
        continue
    k = (r["set"], r["form"], r["turn"])
    _, so = letter_lp(sc["original"][i]["slot"], let, r["letters"]); _, sp = letter_lp(sc[T][i]["slot"], let, r["letters"])
    _, fo = letter_lp(sc["original"][i]["first"], let, r["letters"]); _, fp = letter_lp(sc[T][i]["first"], let, r["letters"])
    cells[k]["so"].append(so); cells[k]["ds"].append(sp - so); cells[k]["fo"].append(fo); cells[k]["df"].append(fp - fo)
for k in sorted(cells):
    c = cells[k]
    print(f"  {'|'.join(k):34} slot: orig p_letters {np.mean(np.exp(c['so'])):.3f}  logratio {np.mean(c['ds']):+.3f}"
          f" | first: orig {np.mean(np.exp(c['fo'])):.3f} logratio {np.mean(c['df']):+.3f} (x{np.exp(np.mean(c['df'])):.2f})")

# ---- B. neutral fits per form (gold row's own slot): m(X) = lr(X)-lr(gold) ----
print("\n=== B. neutral per form: m_p = a + b m_o over non-gold X; and mean shift of lr(gold) (restricted) ===")
for form in ("letter", "answer_sentence", "letter_last"):
    mo, mp, dg = [], [], defaultdict(list)
    for i, r in enumerate(ex):
        if r["set"] == "neutral" and r["form"] == form:
            o, _ = restricted(sc["original"][i]["slot"], let, r["letters"]); p, _ = restricted(sc[T][i]["slot"], let, r["letters"])
            g = r["gold"]
            dg[r["item_id"]].append(p[g] - o[g])
            for X in r["letters"]:
                if X != g:
                    mo.append(o[X] - o[g]); mp.append(p[X] - p[g])
    b, a = np.polyfit(mo, mp, 1)
    print(f"  neutral {form:16} a={a:+.3f} b={b:.3f} mean m_o {np.mean(mo):+.2f}; mean d(m) {np.mean(np.array(mp) - mo):+.3f};"
          f"  d lr(gold) {fmt(cluster_boot(dg))}")

# ---- C. placebo split by kind (suggestion: cave letter is named by user; doubt: gold is named), with intercepts ----
print("\n=== C. placebo by kind: d = a + b m_o + c is_cave; also matched-bin cave minus placebo (bins of m_o width 1) ===")
for set_ in ("heldout", "prune"):
    for form in ("letter", "answer_sentence"):
        for turn in ("single", "multi"):
            for kind in ("suggestion", "doubt"):
                rows = defaultdict(list)
                for (pid, f), d in P.items():
                    if f != form or "cave" not in d:
                        continue
                    ic, ir = d["cave"], d["resist"]; r = ex[ic]
                    if r["set"] != set_ or r["turn"] != turn or r["kind"] != kind:
                        continue
                    o, _ = restricted(sc["original"][ic]["slot"], let, r["letters"]); p, _ = restricted(sc[T][ic]["slot"], let, r["letters"])
                    res = ex[ir]["letter"]
                    for X in r["letters"]:
                        if X == res:
                            continue
                        m = o[X] - o[res]
                        rows[r["item_id"]].append((m, (p[X] - p[res]) - m, float(X == r["letter"])))
                keys = sorted(rows)
                A = np.array([x for k in keys for x in rows[k]])
                Xm = np.c_[np.ones(len(A)), A[:, 0], A[:, 2]]
                co = np.linalg.lstsq(Xm, A[:, 1], rcond=None)[0]
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(500):
                    kk = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
                    B = np.array([x for k in kk for x in rows[k]])
                    bs.append(np.linalg.lstsq(np.c_[np.ones(len(B)), B[:, 0], B[:, 2]], B[:, 1], rcond=None)[0][2])
                # matched bins
                diffs, w = [], []
                for lo in np.arange(-12, 12, 1.0):
                    sel = (A[:, 0] >= lo) & (A[:, 0] < lo + 1)
                    cv, pl = A[sel & (A[:, 2] == 1)], A[sel & (A[:, 2] == 0)]
                    if len(cv) >= 5 and len(pl) >= 5:
                        diffs.append(cv[:, 1].mean() - pl[:, 1].mean()); w.append(len(cv))
                mb = np.average(diffs, weights=w) if w else float("nan")
                print(f"  {set_:7} {form:15} {turn:6} {kind:10} a={co[0]:+.3f} b={co[1]:+.3f} c={co[2]:+.3f} "
                      f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]  matched-bin cave-placebo {mb:+.3f} (cave n in bins {sum(w)})")

# ---- D. multi-turn: per-spelling change for the resist (=previous answer=gold) letter vs cave letter at first position ----
print("\n=== D. letter form, first position: mean change in log p per spelling (pruned - orig), resist vs cave letter ===")
for set_ in ("heldout",):
    for turn in ("single", "multi"):
        acc = defaultdict(list)
        for (pid, f), d in P.items():
            if f != "letter" or "cave" not in d:
                continue
            ic, ir = d["cave"], d["resist"]; r = ex[ic]
            if r["set"] != set_ or r["turn"] != turn:
                continue
            vo, vp = np.array(sc["original"][ic]["first"]), np.array(sc[T][ic]["first"])
            for role, L in (("resist", ex[ir]["letter"]), ("cave", r["letter"])):
                for sp in ("bare", "space", "paren"):
                    j = np.where((let == L) & (spell == sp))[0][0]
                    acc[(role, sp)].append((vo[j], vp[j]))
        print(f"  {set_} {turn}:")
        for k in sorted(acc):
            a = np.array(acc[k])
            print(f"     {k[0]:6} {k[1]:5}  orig mean p {np.mean(np.exp(a[:, 0])):.4f}  pruned mean p {np.mean(np.exp(a[:, 1])):.4f}"
                  f"  mean dlogp {np.mean(a[:, 1] - a[:, 0]):+.3f}")
# neutral, no previous turn: gold per spelling
acc = defaultdict(list)
for i, r in enumerate(ex):
    if r["set"] == "neutral" and r["form"] == "letter":
        vo, vp = np.array(sc["original"][i]["first"]), np.array(sc[T][i]["first"])
        for sp in ("bare", "space", "paren"):
            j = np.where((let == r["gold"]) & (spell == sp))[0][0]
            acc[sp].append((vo[j], vp[j]))
for sp, a in acc.items():
    a = np.array(a)
    print(f"  neutral gold {sp:5} orig p {np.mean(np.exp(a[:, 0])):.4f} pruned p {np.mean(np.exp(a[:, 1])):.4f} dlogp {np.mean(a[:, 1] - a[:, 0]):+.3f}")

# ---- E. letter_last: belief read at the OPTION TEXT (tokens before the answer slot), cave vs resist rows ----
print("\n=== E. letter_last/answer_sentence: DiD of summed log p of reply tokens BEFORE the answer slot (cave row - resist row) ===")
for set_ in ("heldout", "prune", ):
    for form in ("letter_last", "answer_sentence"):
        for turn in ("single", "multi"):
            byq, mo = defaultdict(list), []
            for (pid, f), d in P.items():
                if f != form or "cave" not in d:
                    continue
                ic, ir = d["cave"], d["resist"]; r = ex[ic]
                if r["set"] != set_ or r["turn"] != turn:
                    continue
                def pre(t, i):
                    return sum(sc[t][i]["logp"][:ex[i]["answer_idx"]])
                def post(t, i):  # after the slot (option text in answer_sentence)
                    return sum(sc[t][i]["logp"][ex[i]["answer_idx"] + 1:])
                m_o = pre("original", ic) - pre("original", ir)
                m_p = pre(T, ic) - pre(T, ir)
                byq[r["item_id"]].append(m_p - m_o); mo.append(m_o)
            print(f"  {set_:7} {form:15} {turn:6} prefix-DiD {fmt(cluster_boot(byq))}   orig prefix margin mean {np.mean(mo):+.2f}")

# ---- F. preservation (pres_qa = the original's own short correct reply): token 0 (the letter) vs the rest;
# token 0 log p = log letter mass + log restricted p(gold) + log p(this spelling | gold) (exact identity) ----
print("\n=== F. preservation pres_qa: whole-reply delta split into token 0 and rest; decomposition of token 0 ===")
import json as _j
slot = _j.load(open("/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/e3/slot_ids.json"))
ids_sorted = sorted(int(k) for k in slot)
D = defaultdict(list)
for i, r in enumerate(ex):
    if r["set"] != "pres":
        continue
    j = ids_sorted.index(r["reply_ids"][0])
    for t, tag in (("original", "o"), (T, "p")):
        lp = sc[t][i]["logp"]; v = np.array(sc[t][i]["first"])
        d, tot = letter_lp(v, let, r["letters"])
        g = r["gold"]
        D[tag + "tok0"].append(lp[0]); D[tag + "rest"].append(sum(lp[1:])); D[tag + "eot"].append(lp[-1])
        D[tag + "mass"].append(tot); D[tag + "restr"].append(d[g] - tot); D[tag + "spell"].append(v[j] - d[g])
        D[tag + "check"].append(lp[0] - v[j])
        D[tag + "argmax_gold"].append(float(max(d, key=d.get) == g))
        D[tag + "whole"].append(sum(lp))
print(f"  n={len(D['owhole'])}; spelling of token 0: " + str(dict(__import__('collections').Counter(spell[ids_sorted.index(r['reply_ids'][0])] for r in ex if r['set']=='pres'))))
for k in ("whole", "tok0", "rest", "eot", "mass", "restr", "spell"):
    o, p = np.array(D["o" + k]), np.array(D["p" + k])
    print(f"  {k:6} orig {o.mean():+.3f}  pruned {p.mean():+.3f}  delta {np.mean(p - o):+.3f}")
print(f"  check tok0 == first[spelling id]: max abs diff orig {np.max(np.abs(D['ocheck'])):.2e} pruned {np.max(np.abs(D['pcheck'])):.2e}")
print(f"  letter-argmax == gold (restricted, first pos): orig {np.mean(D['oargmax_gold']):.3f} pruned {np.mean(D['pargmax_gold']):.3f}")
