"""Analyst 1: DiD re-derivation, flattening model, placebo-letter control.
Run: python3 a1_did.py > a1_did.out
All margins m(X) = lr(X) - lr(resist) with lr = letter-restricted log prob (spellings summed) at the ANSWER slot of the
CAVE row (as in analyze_e3.py), unless stated. Positive m(cave) = leaning to cave.
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_lib import cluster_boot, fmt, load, pairs, restricted  # noqa

ex, spell, let, sc = load()
TAGS = [t for t in ("pruned_seed0", "random_seed0") if t in sc]
print("tags present:", list(sc))
P = pairs(ex)
out = {}

# ---------- per-prompt records ----------
recs = []  # one per (pid, form) with cave+resist
for (pid, form), d in P.items():
    if "cave" not in d or "resist" not in d:
        continue
    ic, ir = d["cave"], d["resist"]
    r = ex[ic]
    if r["set"] not in ("prune", "heldout"):
        continue
    Ls = r["letters"]
    c, s = ex[ic]["letter"], ex[ir]["letter"]
    rec = {"set": r["set"], "form": form, "turn": r["turn"], "kind": r["kind"], "q": r["item_id"], "cave": c,
           "res": s, "letters": Ls}
    for t in ["original"] + TAGS:
        lc, _ = restricted(sc[t][ic]["slot"], let, Ls)   # cave-row slot
        lr_, _ = restricted(sc[t][ir]["slot"], let, Ls)  # resist-row slot
        rec[t] = {"cr": lc, "rr": lr_}
    recs.append(rec)


def cell(rec):
    return (rec["set"], rec["form"], rec["turn"])


print("\n=== (1) DiD re-derived, cave-row slot (= analyze_e3 definition) and own-slot variant ===")
for t in TAGS:
    print(f"\n-- {t}")
    for key in sorted({cell(r) for r in recs}):
        byq, byq_own, mo = defaultdict(list), defaultdict(list), []
        for r in recs:
            if cell(r) != key:
                continue
            o, p = r["original"], r[t]
            m_o = o["cr"][r["cave"]] - o["cr"][r["res"]]
            m_p = p["cr"][r["cave"]] - p["cr"][r["res"]]
            byq[r["q"]].append(m_p - m_o)
            own_o = o["cr"][r["cave"]] - o["rr"][r["res"]]
            own_p = p["cr"][r["cave"]] - p["rr"][r["res"]]
            byq_own[r["q"]].append(own_p - own_o)
            mo.append(m_o)
        b = cluster_boot(byq)
        out.setdefault(t, {}).setdefault("did", {})["|".join(key)] = b
        print(f"  {'|'.join(key):32} DiD {fmt(b)}   own-slot {cluster_boot(byq_own)['mean']:+.3f}"
              f"   orig margin m_o mean {np.mean(mo):+.2f} median {np.median(mo):+.2f} frac(m_o>0) {np.mean(np.array(mo)>0):.2f}")

# ---------- (2) flattening: regress pruned margin on orig margin; split by orig preference ----------
print("\n=== (2) flattening. m_p = a + b*m_o (OLS over prompts); DiD split by sign of m_o ===")
for t in TAGS:
    print(f"\n-- {t}")
    for key in sorted({cell(r) for r in recs}):
        mo, mp, q = [], [], []
        for r in recs:
            if cell(r) != key:
                continue
            o, p = r["original"]["cr"], r[t]["cr"]
            mo.append(o[r["cave"]] - o[r["res"]]); mp.append(p[r["cave"]] - p[r["res"]]); q.append(r["q"])
        mo, mp = np.array(mo), np.array(mp)
        b, a = np.polyfit(mo, mp, 1)
        pos, neg = defaultdict(list), defaultdict(list)
        for x, y, qq in zip(mo, mp, q):
            (pos if x > 0 else neg)[qq].append(y - x)
        bp, bn = cluster_boot(pos), cluster_boot(neg)
        print(f"  {'|'.join(key):32} a={a:+.3f} b={b:.3f}  | orig prefers CAVE (n={sum(map(len, pos.values()))}): DiD {fmt(bp)}"
              f" | orig prefers RESIST (n={sum(map(len, neg.values()))}): DiD {fmt(bn)}")
        out[t].setdefault("flatten", {})["|".join(key)] = {"a": a, "b": b, "pos": bp, "neg": bn}

# ---------- neutral calibration ----------
print("\n=== neutral calibration: for neutral letter-form prompts, every non-gold X: m = lr(X)-lr(gold) ===")
neu = {}
for t in TAGS:
    mo, mp, sec_byq = [], [], defaultdict(list)
    for i, r in enumerate(ex):
        if r["set"] == "neutral" and r["form"] == "letter":
            o, _ = restricted(sc["original"][i]["slot"], let, r["letters"])
            p, _ = restricted(sc[t][i]["slot"], let, r["letters"])
            g = r["gold"]
            for X in r["letters"]:
                if X != g:
                    mo.append(o[X] - o[g]); mp.append(p[X] - p[g])
            sec = max((X for X in o if X != g), key=lambda X: o[X])
            sec_byq[r["item_id"]].append((p[sec] - p[g]) - (o[sec] - o[g]))
    mo, mp = np.array(mo), np.array(mp)
    b, a = np.polyfit(mo, mp, 1)
    neu[t] = (a, b)
    print(f"  {t}: neutral fit m_p = {a:+.3f} + {b:.3f} m_o (n={len(mo)});  orig gold-vs-X margin mean {np.mean(mo):+.2f};"
          f"  floor DiD (2nd vs gold) {fmt(cluster_boot(sec_byq))}")
    out[t]["neutral_fit"] = {"a": a, "b": b}

print("\n  heldout/prune residual after neutral-calibrated flattening: resid = m_p - (a_n + b_n m_o)")
for t in TAGS:
    a_n, b_n = neu[t]
    for key in sorted({cell(r) for r in recs}):
        byq, pred = defaultdict(list), []
        for r in recs:
            if cell(r) != key:
                continue
            o, p = r["original"]["cr"], r[t]["cr"]
            m_o = o[r["cave"]] - o[r["res"]]; m_p = p[r["cave"]] - p[r["res"]]
            byq[r["q"]].append(m_p - (a_n + b_n * m_o)); pred.append((a_n + b_n * m_o) - m_o)
        print(f"  {t} {'|'.join(key):32} predicted DiD from flattening {np.mean(pred):+.3f};  residual {fmt(cluster_boot(byq))}")

# ---------- placebo letters: same prompt, unnamed letters ----------
print("\n=== placebo: DiD for named cave letter vs unnamed letters X (each vs resist), same prompt & slot ===")
print("    c = cave-specific shift: OLS d = a + b*m_o + c*is_cave over (prompt, letter); cluster bootstrap over questions")
for t in TAGS:
    print(f"\n-- {t}")
    for key in sorted({cell(r) for r in recs}):
        rows = defaultdict(list)  # q -> [(m_o, d, is_cave)]
        for r in recs:
            if cell(r) != key:
                continue
            o, p = r["original"]["cr"], r[t]["cr"]
            for X in r["letters"]:
                if X == r["res"]:
                    continue
                m_o = o[X] - o[r["res"]]; d = (p[X] - p[r["res"]]) - m_o
                rows[r["q"]].append((m_o, d, 1.0 if X == r["cave"] else 0.0))
        keys = sorted(rows)

        def fit(ks):
            A = np.array([x for k in ks for x in rows[k]])
            Xm = np.c_[np.ones(len(A)), A[:, 0], A[:, 2]]
            return np.linalg.lstsq(Xm, A[:, 1], rcond=None)[0]

        coef = fit(keys)
        rng = np.random.default_rng(0)
        bs = np.array([fit([keys[j] for j in rng.integers(0, len(keys), len(keys))])[2] for _ in range(500)])
        A = np.array([x for k in keys for x in rows[k]])
        cav, pla = A[A[:, 2] == 1], A[A[:, 2] == 0]
        print(f"  {'|'.join(key):32} raw DiD cave {cav[:, 1].mean():+.3f} (m_o {cav[:, 0].mean():+.2f}) | placebo {pla[:, 1].mean():+.3f}"
              f" (m_o {pla[:, 0].mean():+.2f}) | b={coef[1]:.3f} c(cave-specific)={coef[2]:+.3f} "
              f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
        out[t].setdefault("placebo", {})["|".join(key)] = {"cave": float(cav[:, 1].mean()), "placebo": float(pla[:, 1].mean()),
                                                         "b": float(coef[1]), "c": float(coef[2]),
                                                         "c_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

json.dump(out, open(__file__.rsplit("/", 1)[0] + "/a1_did.json", "w"), indent=1, default=float)
