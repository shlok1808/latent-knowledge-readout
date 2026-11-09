#!/usr/bin/env python3
"""Verifier: independent recomputation of E3 load-bearing numbers. Own code, own bootstrap seed."""
import json, math, sys
from collections import defaultdict
import numpy as np

E3 = "/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/e3/"
TAGS = ["original", "pruned_seed0", "random_seed0"]
ex = [json.loads(l) for l in open(E3 + "examples.jsonl")]
slot_ids = json.loads(open(E3 + "slot_ids.json").read())
ORDER = [tuple(v) for _, v in sorted(((int(k), v) for k, v in slot_ids.items()))]  # (spelling, letter)
SC = {}
for t in TAGS:
    rows = [json.loads(l) for l in open(E3 + f"scores_{t}.jsonl")]
    assert len(rows) == len(ex)
    for i, r in enumerate(rows):
        assert r["row"] == i and r["pid"] == ex[i]["pid"] and r["role"] == ex[i]["role"] and r["form"] == ex[i]["form"]
    SC[t] = rows
N = len(ex)
print("rows", N)

def lse(a):
    a = np.asarray(a, float); m = a.max(); return m + math.log(np.exp(a - m).sum())

def letter_lp(vec, letters):
    """per-letter log p (spellings summed) + total letter mass log p"""
    by = defaultdict(list)
    for v, (_, L) in zip(vec, ORDER):
        if L in letters: by[L].append(v)
    lp = {L: lse(v) for L, v in by.items()}
    return lp, lse(list(lp.values()))

def restricted(vec, letters):
    lp, tot = letter_lp(vec, letters)
    return {L: v - tot for L, v in lp.items()}, tot

RNG = np.random.default_rng(20261109)
def cboot(by_q, B=2000):
    """cluster bootstrap over questions: mean of per-question means"""
    keys = sorted(by_q); qm = np.array([np.mean(by_q[k]) for k in keys]); n = len(qm)
    if n == 0: return None
    est = qm.mean()
    idx = RNG.integers(0, n, size=(B, n)); bs = qm[idx].mean(1); bs.sort()
    return est, bs[int(0.025 * B)], bs[int(0.975 * B)], n

def fmt(r, d=3):
    if r is None: return "None"
    return f"{r[0]:+.{d}f} [{r[1]:+.{d}f},{r[2]:+.{d}f}] n_q={r[3]}"

# ---- index pairs (pid, form) -> role -> row
pairs = defaultdict(dict)
for i, r in enumerate(ex): pairs[(r["pid"], r["form"])][r["role"]] = i
LET = {r["item_id"]: r["letters"] for r in ex}

def did_slot(tag, ic, ir, use_own_slot=False):
    """DiD of letter-restricted log-odds cave vs resist at the answer slot.
    default: cave-row slot for both letters (as analyze_e3). own_slot: cave letter from cave row, resist letter from resist row."""
    L = LET[ex[ic]["item_id"]]; cL, rL = ex[ic]["letter"], ex[ir]["letter"]
    ro_c, _ = restricted(SC["original"][ic]["slot"], L); rp_c, _ = restricted(SC[tag][ic]["slot"], L)
    if use_own_slot:
        ro_r, _ = restricted(SC["original"][ir]["slot"], L); rp_r, _ = restricted(SC[tag][ir]["slot"], L)
    else:
        ro_r, rp_r = ro_c, rp_c
    lo_o = ro_c[cL] - ro_r[rL]; lo_p = rp_c[cL] - rp_r[rL]
    return lo_p - lo_o, lo_o, lo_p

SKIP = "--tail" in sys.argv
if not SKIP:
  pass
print("\n==== 1. PRIMARY DiD (heldout, letter, single), own bootstrap; plus random; plus by kind; all 12 cells")
for tag in ["pruned_seed0", "random_seed0"]:
    cells = defaultdict(lambda: defaultdict(list)); kind = defaultdict(lambda: defaultdict(list))
    for (pid, form), d in pairs.items():
        if "cave" not in d or "resist" not in d: continue
        r0 = ex[d["cave"]]
        if r0["set"] not in ("heldout", "prune"): continue
        v, lo_o, lo_p = did_slot(tag, d["cave"], d["resist"])
        cells[(r0["set"], form, r0["turn"])][r0["item_id"]].append(v)
        kind[(r0["set"], form, r0["turn"], r0["kind"])][r0["item_id"]].append(v)
    print(f"-- {tag}")
    for k in sorted(cells): print(f"   {'|'.join(k):40} {fmt(cboot(cells[k]))}")
    if tag == "pruned_seed0":
        for k in sorted(kind):
            if k[0] == "heldout" and k[1] in ("letter", "answer_sentence"): print(f"   by kind {'|'.join(k):45} {fmt(cboot(kind[k]))}")

print("\n==== 2. letter_last: shared prefix? and own-slot DiD")
share = defaultdict(lambda: [0, 0])
for (pid, form), d in pairs.items():
    if "cave" in d and "resist" in d:
        c, r = ex[d["cave"]], ex[d["resist"]]
        same = c["reply_ids"][:c["answer_idx"]] == r["reply_ids"][:r["answer_idx"]]
        share[form][0 if same else 1] += 1
print("   shared prefix up to answer slot [same, different] per form:", dict(share))
for use_own in (False, True):
    cells = defaultdict(lambda: defaultdict(list))
    for (pid, form), d in pairs.items():
        if form != "letter_last" or "cave" not in d or "resist" not in d: continue
        r0 = ex[d["cave"]]
        if r0["set"] not in ("heldout", "prune"): continue
        v, _, _ = did_slot("pruned_seed0", d["cave"], d["resist"], use_own_slot=use_own)
        cells[(r0["set"], r0["turn"])][r0["item_id"]].append(v)
    for k in sorted(cells): print(f"   letter_last {'own-slot' if use_own else 'cave-row-slot'} {'|'.join(k):16} {fmt(cboot(cells[k]))}")
# option-text prefix DiD for letter_last: sum of logp over tokens before answer slot, cave minus resist, pruned minus original
cells = defaultdict(lambda: defaultdict(list))
for (pid, form), d in pairs.items():
    if form != "letter_last" or "cave" not in d or "resist" not in d: continue
    r0 = ex[d["cave"]]
    if r0["set"] not in ("heldout", "prune"): continue
    def pre(tag, i): return sum(SC[tag][i]["logp"][:ex[i]["answer_idx"]])
    v = (pre("pruned_seed0", d["cave"]) - pre("pruned_seed0", d["resist"])) - (pre("original", d["cave"]) - pre("original", d["resist"]))
    cells[(r0["set"], r0["turn"])][r0["item_id"]].append(v)
for k in sorted(cells): print(f"   letter_last option-text-prefix DiD {'|'.join(k):16} {fmt(cboot(cells[k]))}")

print("\n==== 3. FORMAT: first-position letter mass, one row per prompt (letter form, cave row; pres row)")
def first_mass(tag, i):
    L = LET[ex[i]["item_id"]]; _, m = letter_lp(SC[tag][i]["first"], L); return m
grp = defaultdict(lambda: defaultdict(list))
for i, r in enumerate(ex):
    if (r["form"] == "letter" and r["role"] in ("cave", "correct")) or r["form"] == "pres_qa":
        mo, mp = first_mass("original", i), first_mass("pruned_seed0", i)
        po, pp = math.exp(mo), math.exp(mp)
        logit = lambda p: math.log(p / (1 - p))
        key = (r["set"], r["turn"], r["kind"])
        grp[key][r["item_id"]].append((po, pp, mp - mo, logit(pp) - logit(po), pp - po))
def agg(key, j): return cboot({q: [x[j] for x in v] for q, v in grp[key].items()})
print(f"   {'cell':34} {'p_orig':>7} {'p_prun':>7} | log-ratio | logit shift | abs drop")
for key in sorted(grp):
    po = np.mean([x[0] for v in grp[key].values() for x in v]); pp = np.mean([x[1] for v in grp[key].values() for x in v])
    lr, lg, ad = agg(key, 2), agg(key, 3), agg(key, 4)
    print(f"   {'|'.join(key):34} {po:7.3f} {pp:7.3f} | {lr[0]:+.3f} [{lr[1]:+.2f},{lr[2]:+.2f}] | {lg[0]:+.2f} [{lg[1]:+.2f},{lg[2]:+.2f}] | {ad[0]:+.3f}")
# paired within-item: the 37 items that are both neutral and heldout
neu = {}; held = defaultdict(list)
for i, r in enumerate(ex):
    if r["form"] == "letter" and r["set"] == "neutral": neu[r["item_id"]] = i
    if r["form"] == "letter" and r["set"] == "heldout" and r["role"] == "cave" and r["turn"] == "single": held[r["item_id"]].append(i)
common = [q for q in neu if q in held]
dl, dg, da = [], [], []
for q in common:
    mo, mp = first_mass("original", neu[q]), first_mass("pruned_seed0", neu[q])
    for i in held[q]:
        ho, hp = first_mass("original", i), first_mass("pruned_seed0", i)
        dl.append((hp - ho) - (mp - mo))
        lg = lambda m: m - math.log1p(-math.exp(m))
        dg.append((lg(hp) - lg(ho)) - (lg(mp) - lg(mo)))
        da.append((math.exp(hp) - math.exp(ho)) - (math.exp(mp) - math.exp(mo)))
print(f"   paired same-item pushed-single minus neutral (n={len(dl)} prompts, {len(common)} items): "
      f"log-ratio diff {np.mean(dl):+.3f} (se {np.std(dl)/math.sqrt(len(dl)):.3f}); logit diff {np.mean(dg):+.3f} (se {np.std(dg)/math.sqrt(len(dg)):.3f}); abs-drop diff {np.mean(da):+.3f}")

print("\n==== 4. NEUTRAL flattening: lo(gold vs original 2nd choice) pruned vs original at letter slot and sentence slot; binned means")
def fit(x, y):
    x, y = np.asarray(x), np.asarray(y); A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]; r2 = 1 - ((y - a - b * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return a, b, r2
def binned(x, y, edges):
    x, y = np.asarray(x), np.asarray(y); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 8: out.append(f"[{lo:+.0f},{hi:+.0f}) n={m.sum():3d} x={x[m].mean():+.2f} y={y[m].mean():+.2f} y-x={(y[m]-x[m]).mean():+.2f}")
    return out
for form in ("letter", "answer_sentence", "letter_last"):
    xs, ys, xs_all, ys_all = [], [], [], []
    for i, r in enumerate(ex):
        if r["set"] != "neutral" or r["form"] != form: continue
        L = LET[r["item_id"]]; ro, _ = restricted(SC["original"][i]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][i]["slot"], L)
        g = r["gold"]; sec = max((x for x in ro if x != g), key=lambda x: ro[x])
        xs.append(ro[g] - ro[sec]); ys.append(rp[g] - rp[sec])
        for a_ in L:
            for b_ in L:
                if a_ < b_: xs_all.append(ro[a_] - ro[b_]); ys_all.append(rp[a_] - rp[b_])
    a, b, r2 = fit(xs, ys); a2, b2, r22 = fit(xs_all, ys_all)
    print(f"   neutral {form:16} gold-vs-2nd: a={a:+.2f} b={b:.3f} R2={r2:.2f} n={len(xs)} mean lo_o={np.mean(xs):+.2f} lo_p={np.mean(ys):+.2f} | all pairs: a={a2:+.2f} b={b2:.3f} R2={r22:.2f}")
    if form != "letter_last":
        for s in binned(xs_all, ys_all, [-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12]): print("        all-pairs bin", s)
# entropy of restricted letter dist at first position, neutral letter form
H = {t: [] for t in ("original", "pruned_seed0")}
for i, r in enumerate(ex):
    if r["set"] == "neutral" and r["form"] == "letter":
        for t in H:
            rp, _ = restricted(SC[t][i]["slot"], LET[r["item_id"]]); p = np.exp(np.array(list(rp.values())))
            H[t].append(-(p * np.log(p + 1e-30)).sum())
print(f"   neutral letter-slot restricted entropy: orig {np.mean(H['original']):.3f} pruned {np.mean(H['pruned_seed0']):.3f}")

print("\n==== 5. Does neutral flattening transfer to pushed prompts? wrong-vs-wrong pairs (neither gold nor cave) on pushed single prompts")
for form in ("letter", "answer_sentence"):
    for kind in ("suggestion", "doubt"):
        xs, ys = [], []
        for (pid, f), d in pairs.items():
            if f != form or "cave" not in d: continue
            r = ex[d["cave"]]
            if r["set"] != "heldout" or r["turn"] != "single" or r["kind"] != kind: continue
            L = LET[r["item_id"]]; ro, _ = restricted(SC["original"][d["cave"]]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][d["cave"]]["slot"], L)
            W = [x for x in L if x not in (r["gold"], r["letter"])]
            for a_ in W:
                for b_ in W:
                    if a_ < b_: xs.append(ro[a_] - ro[b_]); ys.append(rp[a_] - rp[b_])
        a, b, r2 = fit(xs, ys)
        print(f"   heldout single {kind:10} {form:16} wrong-vs-wrong (excl gold & cave): a={a:+.2f} b={b:.3f} R2={r2:.2f} n_pairs={len(xs)}")
        for s in binned(xs, ys, [-8, -4, -2, -1, 0, 1, 2, 4, 8]): print("        bin", s)

print("\n==== 6. Placebo at letter and sentence slots (heldout single suggestion): DiD(cave vs gold) vs DiD(unsuggested wrong vs gold)")
for form in ("letter", "answer_sentence"):
    for kind in ("suggestion", "doubt"):
        cav, plc, diff, xs_c, ys_c, xs_w, ys_w = defaultdict(list), defaultdict(list), defaultdict(list), [], [], [], []
        for (pid, f), d in pairs.items():
            if f != form or "cave" not in d or "resist" not in d: continue
            r = ex[d["cave"]]
            if r["set"] != "heldout" or r["turn"] != "single" or r["kind"] != kind: continue
            L = LET[r["item_id"]]; ro, _ = restricted(SC["original"][d["cave"]]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][d["cave"]]["slot"], L)
            g, c = r["gold"], r["letter"]
            dc = (rp[c] - rp[g]) - (ro[c] - ro[g]); cav[r["item_id"]].append(dc); xs_c.append(ro[c] - ro[g]); ys_c.append(rp[c] - rp[g])
            W = [x for x in L if x not in (g, c)]
            dw = np.mean([(rp[w] - rp[g]) - (ro[w] - ro[g]) for w in W]); plc[r["item_id"]].append(dw)
            for w in W: xs_w.append(ro[w] - ro[g]); ys_w.append(rp[w] - rp[g])
            diff[r["item_id"]].append(dc - dw)
        print(f"   {form:16} {kind:10} DiD cave-vs-gold {fmt(cboot(cav))} | unsuggested-wrong-vs-gold {fmt(cboot(plc))} | cave minus placebo {fmt(cboot(diff))}")
        print(f"        orig lo(cave-gold) mean {np.mean(xs_c):+.2f}; orig lo(wrong-gold) mean {np.mean(xs_w):+.2f}")
        for s in binned(xs_c, ys_c, [-8, -4, -2, -1, 0, 1, 2, 4, 8]): print("        cave bins   ", s)
        for s in binned(xs_w, ys_w, [-8, -4, -2, -1, 0, 1, 2, 4, 8]): print("        placebo bins", s)

print("\n==== 7. MULTI-turn letter form: restricted p(prev=resist=gold), p(cave); confound check")
for tag in TAGS:
    pp, pc, lo = [], [], []
    for (pid, f), d in pairs.items():
        if f != "letter" or "cave" not in d or "resist" not in d: continue
        r = ex[d["cave"]]
        if r["set"] not in ("heldout", "prune") or r["turn"] != "multi": continue
        L = LET[r["item_id"]]; rp, _ = restricted(SC[tag][d["cave"]]["slot"], L)
        pp.append(math.exp(rp[ex[d["resist"]]["letter"]])); pc.append(math.exp(rp[r["letter"]])); lo.append(rp[r["letter"]] - rp[ex[d["resist"]]["letter"]])
    print(f"   {tag:13} p(prev) {np.mean(pp):.3f}  p(cave) {np.mean(pc):.3f}  lo(cave-prev) {np.mean(lo):+.2f}  n={len(pp)}")
# in multi-turn, does gold also gain vs unsuggested wrong letters (placebo)? and is the gain concentrated on paren spelling
cav, plc = defaultdict(list), defaultdict(list)
for (pid, f), d in pairs.items():
    if f != "letter" or "cave" not in d or "resist" not in d: continue
    r = ex[d["cave"]]
    if r["set"] != "heldout" or r["turn"] != "multi": continue
    L = LET[r["item_id"]]; ro, _ = restricted(SC["original"][d["cave"]]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][d["cave"]]["slot"], L)
    g, c = r["gold"], r["letter"]
    cav[r["item_id"]].append((rp[c] - rp[g]) - (ro[c] - ro[g]))
    W = [x for x in L if x not in (g, c)]; plc[r["item_id"]].append(np.mean([(rp[w] - rp[g]) - (ro[w] - ro[g]) for w in W]))
print(f"   heldout multi letter: DiD cave-vs-gold {fmt(cboot(cav))} | unsuggested-wrong-vs-gold {fmt(cboot(plc))}")

print("\n==== 8. PRES: whole-reply drop and components")
whole, eot, tok0, mass0, choice0, mid = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list); ntok = {}
for i, r in enumerate(ex):
    if r["set"] != "pres": continue
    lo_, lp_ = SC["original"][i]["logp"], SC["pruned_seed0"][i]["logp"]
    assert r["reply_ids"][-1] == 128009, r["reply_ids"]
    whole[r["item_id"]].append(sum(lp_) - sum(lo_)); eot[r["item_id"]].append(lp_[-1] - lo_[-1]); tok0[r["item_id"]].append(lp_[0] - lo_[0])
    mid[r["item_id"]].append(sum(lp_[1:-1]) - sum(lo_[1:-1])); ntok[len(lp_)] = ntok.get(len(lp_), 0) + 1
    L = LET[r["item_id"]]; _, mo = letter_lp(SC["original"][i]["first"], L); _, mp = letter_lp(SC["pruned_seed0"][i]["first"], L)
    mass0[r["item_id"]].append(mp - mo); choice0[r["item_id"]].append((lp_[0] - mp) - (lo_[0] - mo))
print("   reply lengths", ntok, "| mid tokens (e.g. ')')", fmt(cboot(mid)))
print(f"   whole {fmt(cboot(whole))} | eot {fmt(cboot(eot))} | tok0 {fmt(cboot(tok0))} = letter mass {fmt(cboot(mass0))} + choice-of-spelled-letter {fmt(cboot(choice0))}")
po = np.mean([math.exp(SC['original'][i]['logp'][0]) for i, r in enumerate(ex) if r['set'] == 'pres']); pq = np.mean([math.exp(SC['pruned_seed0'][i]['logp'][0]) for i, r in enumerate(ex) if r['set'] == 'pres'])
eo = np.mean([math.exp(SC['original'][i]['logp'][1]) for i, r in enumerate(ex) if r['set'] == 'pres']); eq = np.mean([math.exp(SC['pruned_seed0'][i]['logp'][1]) for i, r in enumerate(ex) if r['set'] == 'pres'])
print(f"   pres mean p(first token as written) {po:.3f} -> {pq:.3f}; mean p(eot after ')') {eo:.3f} -> {eq:.3f}")

print("\n==== 9. REAL replies: whole-reply and slot DiD by author")
for by in ("original", "pruned_seed0"):
    wh, sl = defaultdict(list), defaultdict(list)
    for (pid, f), d in pairs.items():
        r0 = ex[next(iter(d.values()))]
        if r0["set"] != "real" or r0["written_by"] != by: continue
        # per pid: as_written row (role = author's choice) and twin row (opposite role), both forms in pairs keyed by form; join by pid
        pass
    rows_by_pid = defaultdict(dict)
    for i, r in enumerate(ex):
        if r["set"] == "real" and r["written_by"] == by: rows_by_pid[r["pid"]][r["form"]] = i
    for pid, d in rows_by_pid.items():
        if "real_as_written" not in d or "real_twin" not in d: continue
        iw, it = d["real_as_written"], d["real_twin"]; q = ex[iw]["item_id"]
        wh[q].append((sum(SC["pruned_seed0"][iw]["logp"]) - sum(SC["pruned_seed0"][it]["logp"])) - (sum(SC["original"][iw]["logp"]) - sum(SC["original"][it]["logp"])))
        L = LET[q]; ro, _ = restricted(SC["original"][iw]["slot"], L); rp, _ = restricted(SC["pruned_seed0"][iw]["slot"], L)
        wL, tL = ex[iw]["letter"], ex[it]["letter"]
        sgn = 1 if ex[iw]["role"] == "cave" else -1   # DiD oriented as cave minus resist
        sl[q].append(sgn * ((rp[wL] - rp[tL]) - (ro[wL] - ro[tL])))
    print(f"   written_by {by:13} whole-reply DiD (as-written minus twin) {fmt(cboot(wh))} | slot DiD (cave minus resist) {fmt(cboot(sl))}")
