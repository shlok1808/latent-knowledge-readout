"""I4: is the uniform ~25% drop at the fixed slot just a confidence/temperature change?
For each model: neutral rows -> log-odds of the model's own top letter (confidence). Fit one temperature T so that
original logits / T match the pruned model's neutral confidence; then predict every D1 cell's shift under pure
temperature scaling and compare with the observed pruned shifts."""
import json, math, sys
from collections import defaultdict
sys.path.insert(0, ".")
from knowledge_readout.syco import metrics as M

def rows(tag):
    return [json.loads(l) for l in open(f"runs/prefix_{tag}/scores.jsonl")]

def lo(p):
    p = min(max(p, 1e-12), 1 - 1e-12); return math.log(p / (1 - p))

data = {t: rows(t) for t in ("original", "pruned_seed0", "random_seed0")}
for t, rs in data.items():
    neu = [r for r in rs if r["kind"] == "neutral"]
    conf = [lo(max(r["probs"].values())) for r in neu]
    conf.sort()
    print(f"{t:14} neutral top-letter log-odds: mean {sum(conf)/len(conf):.2f}  median {conf[len(conf)//2]:.2f}  n {len(conf)}")

# temperature fit: scale ORIGINAL letter logprobs by 1/T, renormalise, match pruned mean neutral confidence
orig = {r["item_id"]: r for r in data["original"] if r["kind"] == "neutral"}
pr = {r["item_id"]: r for r in data["pruned_seed0"] if r["kind"] == "neutral"}
def temp_probs(lp, T):
    v = {k: x / T for k, x in lp.items()}; m = max(v.values()); z = sum(math.exp(x - m) for x in v.values())
    return {k: math.exp(x - m) / z for k, x in v.items()}
target = sum(lo(max(r["probs"].values())) for r in pr.values()) / len(pr)
best = None
for T100 in range(80, 300):
    T = T100 / 100
    c = sum(lo(max(temp_probs({k: math.log(max(v, 1e-30)) for k, v in r["probs"].items()}, T).values())) for r in orig.values()) / len(orig)
    if best is None or abs(c - target) < best[1]:
        best = (T, abs(c - target), c)
T = best[0]
print(f"fitted T = {T:.2f} (orig confidence at T: {best[2]:.2f} vs pruned {target:.2f})")

# predicted D1 shifts under temperature-only change vs observed
def cells(rs, transform=None):
    by = {(r["item_id"]): r for r in rs if r["kind"] == "neutral"}
    acc = defaultdict(list)
    for r in rs:
        if r["kind"] not in ("suggestion", "doubt") or r["turn"] != "single":
            continue
        n = by.get(r["item_id"])
        if n is None: continue
        pn, pc = n["probs"], r["probs"]
        if transform:
            pn = transform(pn); pc = transform(pc)
        a = max(pn, key=pn.get)
        prior = "correct" if a == r["gold"] else "wrong"
        if r["kind"] == "suggestion":
            s = lo(pc[r["target"]]) - lo(pn[r["target"]])
        else:
            s = lo(pn[a]) - lo(pc[a])
        col = "user" if r["source"] == "user" else ("rel12" if r["source"] != "tool" else None)
        if col: acc[(prior, r["kind"], col)].append(s)
    return {k: sum(v) / len(v) for k, v in acc.items()}
tf = lambda p: temp_probs({k: math.log(max(v, 1e-30)) for k, v in p.items()}, T)
o = cells(data["original"]); pT = cells(data["original"], tf); p = cells(data["pruned_seed0"])
print(f"{'cell':34} {'orig':>6} {'T-pred':>7} {'pruned':>7} {'ratio obs':>9} {'ratio T':>8}")
for k in sorted(o):
    print(f"{str(k):34} {o[k]:6.2f} {pT[k]:7.2f} {p[k]:7.2f} {p[k]/o[k]:9.2f} {pT[k]/o[k]:8.2f}")
