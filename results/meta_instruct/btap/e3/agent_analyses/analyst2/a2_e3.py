#!/usr/bin/env python
"""Analyst 2, E3. Read-only on the repo. Output: a2_out.json + printed tables next to this script.
Sections: REAL (pairs as_written+twin), BOX (synthetic per dataset/kind/turn), SHRINK (DiD regressed on original gap),
PLACEBO (suggested vs unsuggested wrong letter), FORMAT (first-position letter mass). Each for pruned_seed0 and random_seed0
(if present). Cluster bootstrap over item_id (B=2000, seed 0)."""
import json, math, random, re, sys
from collections import defaultdict
from pathlib import Path

E3 = Path("/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/e3")
OUT = Path(__file__).resolve().parent / "a2_out.json"
TAGS = ("pruned_seed0", "random_seed0")

def lse(xs):
    m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))

ex = [json.loads(l) for l in open(E3 / "examples.jsonl")]
slot = json.loads((E3 / "slot_ids.json").read_text())
order = [v for _, v in sorted((int(k), v) for k, v in slot.items())]
sc = {}
for t in ("original",) + TAGS:
    p = E3 / f"scores_{t}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p)]
        if len(rows) != len(ex):
            print("incomplete", t, len(rows)); continue
        assert all(r["row"] == i and r["pid"] == ex[i]["pid"] for i, r in enumerate(rows))
        sc[t] = rows
print("tags present:", list(sc))

def restricted(vals, letters):
    by = defaultdict(list)
    for v, (_, L) in zip(vals, order):
        if L in letters: by[L].append(v)
    lp = {L: lse(v) for L, v in by.items()}; tot = lse(list(lp.values()))
    return {L: v - tot for L, v in lp.items()}, tot

def boot(byq, reps=2000, seed=0):
    keys = sorted(byq)
    if not keys: return None
    rnd = random.Random(seed)
    m = {k: sum(byq[k]) / len(byq[k]) for k in keys}
    mean = lambda ks: sum(m[k] for k in ks) / len(ks)
    bs = sorted(mean([keys[rnd.randrange(len(keys))] for _ in keys]) for _ in range(reps))
    return {"mean": round(mean(keys), 4), "ci": [round(bs[int(.025 * reps)], 4), round(bs[int(.975 * reps)], 4)], "nq": len(keys),
            "n": sum(len(v) for v in byq.values())}

def boot_reg(pts, reps=2000, seed=0):
    """pts: q -> [(x,y)]. OLS y = a + b x, cluster bootstrap. Returns intercept a (shift at original gap 0) and slope b."""
    keys = sorted(pts)
    def fit(ks):
        xy = [p for k in ks for p in pts[k]]; n = len(xy)
        mx = sum(x for x, _ in xy) / n; my = sum(y for _, y in xy) / n
        sxx = sum((x - mx) ** 2 for x, _ in xy); b = sum((x - mx) * (y - my) for x, y in xy) / sxx
        return my - b * mx, b
    if len(keys) < 10: return None
    a, b = fit(keys); rnd = random.Random(seed)
    bs = []
    for _ in range(reps):
        try: bs.append(fit([keys[rnd.randrange(len(keys))] for _ in keys]))
        except ZeroDivisionError: pass
    reps = len(bs)
    A = sorted(x[0] for x in bs); Bs = sorted(x[1] for x in bs)
    q = lambda s: [round(s[int(.025 * reps)], 4), round(s[int(.975 * reps)], 4)]
    return {"intercept": round(a, 4), "int_ci": q(A), "slope": round(b, 4), "slope_ci": q(Bs), "nq": len(keys)}

def fixed_style(r):
    body = r["reply"].split("<|eot_id|>")[0].lstrip()
    if re.match(r"^(\([A-E]\)|[A-E]([).:]|$))", body) or re.match(r"^the (correct )?answer is:?\s*\([A-E]\)", body, re.I):
        return "letter_first"
    return "reasons_first"

# --- group pairs
pairs = defaultdict(dict)
for i, r in enumerate(ex):
    if r["set"] in ("prune", "heldout"):
        pairs[(r["pid"], r["form"])][r["role"]] = i
    elif r["set"] == "real":
        pairs[(r["pid"], "real")][r["form"]] = i

# sanity: cave and resist rows share the slot vector (shared prefix)
mx = 0.0
for (pid, f), d in pairs.items():
    a, b = list(d.values())
    for t in sc:
        mx = max(mx, max(abs(x - y) for x, y in zip(sc[t][a]["slot"], sc[t][b]["slot"])))
print("max |slot(cave)-slot(resist)| within pair, any model:", mx)

# relabel real_style
relabel = {}
for (pid, f), d in pairs.items():
    if f != "real": continue
    r = ex[d["real_as_written"]]; new = fixed_style(r)
    relabel[pid] = new
nchg = [(pid, ex[pairs[(pid, 'real')]['real_as_written']]['real_style'], s, ex[pairs[(pid, 'real')]['real_as_written']]['reply'][:70])
        for pid, s in relabel.items() if s != ex[pairs[(pid, 'real')]['real_as_written']]['real_style']]
print("real_style relabelled:", len(nchg))
for x in nchg: print("   ", x)

out = {"relabelled": nchg}
for tag in [t for t in TAGS if t in sc]:
    R = {}
    O, P = sc["original"], sc[tag]
    # ---------- REAL
    real = defaultdict(lambda: defaultdict(list))
    for (pid, f), d in pairs.items():
        if f != "real": continue
        iw, it = d["real_as_written"], d["real_twin"]; r = ex[iw]
        ic, ir = (iw, it) if r["role"] == "cave" else (it, iw)
        L = set(r["letters"]); cL, rL = ex[ic]["letter"], ex[ir]["letter"]
        ro, _ = restricted(O[ic]["slot"], L); rp, _ = restricted(P[ic]["slot"], L)
        slot_did = (rp[cL] - rp[rL]) - (ro[cL] - ro[rL])
        W = lambda S, i: sum(S[i]["logp"])
        whole_did = (W(P, ic) - W(P, ir)) - (W(O, ic) - W(O, ir))
        d_aw = W(P, iw) - W(O, iw); d_tw = W(P, it) - W(O, it)
        # post-slot part of the reply only (tokens after the answer token), cave minus resist
        ai = r["answer_idx"]
        post = lambda S, i: sum(S[i]["logp"][ai + 1:])
        post_did = (post(P, ic) - post(P, ir)) - (post(O, ic) - post(O, ir))
        orig_gap = ro[cL] - ro[rL]
        st = relabel[pid]; wb = r["written_by"]; aw = r["role"]
        for key in [("all",), (wb,), (wb, st), (wb, st, "aw=" + aw), (wb, "kind=" + r["kind"]), (wb, "ds=" + r["box"].split("/")[0])]:
            k = "|".join(key)
            real["slot_did"][k] and None
            real[("slot_did", k)][r["item_id"]].append(slot_did)
            real[("whole_did", k)][r["item_id"]].append(whole_did)
            real[("post_did", k)][r["item_id"]].append(post_did)
            real[("d_as_written", k)][r["item_id"]].append(d_aw)
            real[("d_twin", k)][r["item_id"]].append(d_tw)
            real[("orig_gap", k)][r["item_id"]].append(orig_gap)
    R["real"] = {f"{m}::{k}": boot(v) for (m, k), v in sorted((kk, vv) for kk, vv in real.items() if isinstance(kk, tuple))}
    # shrink-adjusted real slot DiD
    rp_pts = defaultdict(lambda: defaultdict(list))
    for (pid, f), d in pairs.items():
        if f != "real": continue
        iw, it = d["real_as_written"], d["real_twin"]; r = ex[iw]
        ic, ir = (iw, it) if r["role"] == "cave" else (it, iw)
        L = set(r["letters"]); cL, rL = ex[ic]["letter"], ex[ir]["letter"]
        ro, _ = restricted(O[ic]["slot"], L); rp, _ = restricted(P[ic]["slot"], L)
        g = ro[cL] - ro[rL]
        for k in ("all", r["written_by"], f"{r['written_by']}|{relabel[pid]}"):
            rp_pts[k][r["item_id"]].append((g, (rp[cL] - rp[rL]) - g))
    R["real_shrink"] = {k: boot_reg(v) for k, v in rp_pts.items()}
    # ---------- SYNTHETIC per box, SHRINK, PLACEBO
    did = defaultdict(lambda: defaultdict(list)); pts = defaultdict(lambda: defaultdict(list))
    ll = defaultdict(lambda: defaultdict(list)); plpts = defaultdict(lambda: defaultdict(list))
    plc = defaultdict(lambda: defaultdict(list)); gapo = defaultdict(lambda: defaultdict(list))
    for (pid, f), d in pairs.items():
        if f == "real" or "cave" not in d: continue
        ic, ir = d["cave"], d["resist"]; r = ex[ic]; L = set(r["letters"])
        cL, rL = r["letter"], ex[ir]["letter"]
        ro, _ = restricted(O[ic]["slot"], L); rp, _ = restricted(P[ic]["slot"], L)
        g = ro[cL] - ro[rL]; v = (rp[cL] - rp[rL]) - g
        ds, kind, turn = r["box"].split("/")
        if f == "letter_last":  # prefixes differ (option text precedes the letter): own-row letter binding instead
            ro2, _ = restricted(O[ir]["slot"], L); rp2, _ = restricted(P[ir]["slot"], L)
            bc = rp[cL] - ro[cL]; br = rp2[rL] - ro2[rL]
            for key in [(r["set"], turn), (r["set"], turn, kind), (r["set"], turn, ds)]:
                ll["|".join(key) + "|cave_row"][r["item_id"]].append(bc)
                ll["|".join(key) + "|resist_row"][r["item_id"]].append(br)
                ll["|".join(key) + "|cave_minus_resist"][r["item_id"]].append(bc - br)
            continue
        others = [x for x in L if x not in (cL, rL)]
        pl = sum((rp[cL] - rp[x]) - (ro[cL] - ro[x]) for x in others) / len(others)
        if kind == "suggestion" and f == "letter":
            for x in others:
                plpts[f"{r['set']}|{turn}"][r["item_id"]].append((ro[cL] - ro[x], (rp[cL] - rp[x]) - (ro[cL] - ro[x])))
        for key in [(r["set"], f, turn), (r["set"], f, turn, kind), (r["set"], f, turn, ds), (r["set"], f, turn, ds, kind)]:
            k = "|".join(key)
            did[k][r["item_id"]].append(v); pts[k][r["item_id"]].append((g, v))
            plc[k][r["item_id"]].append(pl); gapo[k][r["item_id"]].append(g)
    R["placebo_shrink_letter"] = {k: boot_reg(v) for k, v in plpts.items()}
    R["letter_last_binding"] = {k: boot(v) for k, v in sorted(ll.items())}
    R["did"] = {k: boot(v) for k, v in sorted(did.items())}
    R["orig_gap"] = {k: boot(v) for k, v in sorted(gapo.items())}
    R["shrink"] = {k: boot_reg(v) for k, v in sorted(pts.items()) if k.count("|") <= 3}
    R["placebo_vs_unsuggested_wrong"] = {k: boot(v) for k, v in sorted(plc.items())}
    # neutral shrink: gold vs each wrong letter (no push) -> slope/intercept for comparison
    npts = defaultdict(list)
    for i, r in enumerate(ex):
        if r["set"] == "neutral" and r["form"] == "letter":
            L = set(r["letters"]); ro, _ = restricted(O[i]["slot"], L); rp, _ = restricted(P[i]["slot"], L)
            for x in L - {r["gold"]}:
                g = ro[x] - ro[r["gold"]]; npts[r["item_id"]].append((g, (rp[x] - rp[r["gold"]]) - g))
    R["neutral_shrink_all_wrong"] = boot_reg(npts)
    ww = defaultdict(list)
    for i, r in enumerate(ex):
        if r["set"] == "neutral" and r["form"] == "letter":
            L = set(r["letters"]); ro, _ = restricted(O[i]["slot"], L); rp, _ = restricted(P[i]["slot"], L)
            W = sorted(L - {r["gold"]})
            for a in W:
                for b in W:
                    if a != b: ww[r["item_id"]].append((ro[a] - ro[b], (rp[a] - rp[b]) - (ro[a] - ro[b])))
    R["neutral_shrink_wrong_vs_wrong"] = boot_reg(ww)
    # ---------- FORMAT per box (first-position letter mass, letter form, once per prompt)
    fmt = defaultdict(lambda: defaultdict(list)); seen = set()
    for i, r in enumerate(ex):
        if r["form"] not in ("letter", "pres_qa") or r["pid"] in seen: continue
        seen.add(r["pid"]); L = set(r["letters"])
        _, mo = restricted(O[i]["first"], L); _, mp = restricted(P[i]["first"], L)
        parts = r["box"].split("/")
        keys = [(r["set"], r["turn"])]
        if len(parts) == 3: keys += [(r["set"], r["turn"], parts[1]), (r["set"], r["turn"], parts[0]), (r["set"], r["turn"], parts[0], parts[1])]
        else: keys += [(r["set"], r["turn"], parts[0])]
        for key in keys: fmt["|".join(key)][r["item_id"]].append(mp - mo)
    R["format_logratio"] = {k: boot(v) for k, v in sorted(fmt.items())}
    out[tag] = R
OUT.write_text(json.dumps(out, indent=1))

def show(tag, sec, filt=lambda k: True, fields=("mean", "ci", "nq")):
    if tag not in out: return
    print(f"\n== {tag} :: {sec}")
    for k, v in out[tag][sec].items():
        if v and filt(k): print(f"  {k:55} " + "  ".join(f"{f}={v[f]}" for f in fields if f in v))
for tag in TAGS:
    show(tag, "real", lambda k: k.split("::")[0] in ("slot_did", "whole_did", "post_did", "d_as_written", "d_twin", "orig_gap"))
    show(tag, "real_shrink", fields=("intercept", "int_ci", "slope", "slope_ci", "nq"))
    show(tag, "did")
    show(tag, "letter_last_binding")
    show(tag, "orig_gap", lambda k: k.count("|") == 2)
    show(tag, "shrink", fields=("intercept", "int_ci", "slope", "slope_ci", "nq"))
    show(tag, "placebo_vs_unsuggested_wrong", lambda k: "suggestion" in k and k.count("|") <= 3)
    if tag in out:
        print("neutral shrink:", out[tag]["neutral_shrink_all_wrong"]); print("neutral wrong-vs-wrong:", out[tag]["neutral_shrink_wrong_vs_wrong"])
        print("placebo shrink (letter, suggestion):", out[tag]["placebo_shrink_letter"])
    show(tag, "format_logratio")

# reply shape per author (author is confounded with reply shape)
from collections import Counter
shape = Counter()
for r in ex:
    if r["set"] == "real" and r["form"] == "real_as_written":
        b = r["reply"].split("<|eot_id|>")[0].strip()
        t = "bare" if re.fullmatch(r"\(?[A-E]\)?\.?", b) else ("sentence_open" if re.match(r"^the (correct )?answer is", b, re.I) else "other")
        shape[f"{r['written_by']}|{t}"] += 1
print("real reply shape by author:", dict(shape))
out["real_shape_by_author"] = dict(shape); OUT.write_text(json.dumps(out, indent=1))
