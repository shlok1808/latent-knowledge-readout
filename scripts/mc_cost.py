"""Projected MC cost for a run: continuation rounds needed (upper bound: every
continuation runs to max_rounds) and hours at the collection's measured round rate.

    python scripts/mc_cost.py runs/ms-tc-1p7b [min_depth]
"""
import glob, json, sys
run = sys.argv[1]; min_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cfg = json.load(open(f"{run}/manifest.json"))["config"]
N, R = cfg["mc"]["n"], cfg["max_rounds"]
eps = [json.load(open(f)) for f in glob.glob(f"{run}/episodes/*.json")]
prog = [json.loads(l) for l in open(f"{run}/progress.jsonl")]
rounds_done = sum(p["rounds"] for p in prog); secs = max(p["seconds"] for p in prog)
rate = rounds_done / secs
by = {}
for e in eps:
    d = e["recipe_depth"]
    n = len(e["rounds"])
    ts = {t for t in cfg["mc"]["rounds"] if t <= n}
    ts |= {r["round"] for r in e["rounds"] if r["observation"].startswith("Crafted ") and not (r["done"] and r["reward"] == 1)}
    cont = sum(N * (R - t) for t in ts if not (t == n and e["rounds"][-1]["done"]) and t < R)
    b = by.setdefault(d, [0, 0, 0]); b[0] += 1; b[1] += cont; b[2] += n
print(f"collection rate {rate:.1f} rounds/s over {rounds_done} rounds")
tot = 0
for d in sorted(by):
    n_eps, cont, rds = by[d]
    inscope = d >= min_depth
    tot += cont if inscope else 0
    print(f"depth {d}: {n_eps} eps, {rds} rounds collected, MC continuation rounds (upper bound) {cont:,} {'IN' if inscope else 'out of'} scope")
print(f"MC total (min_depth={min_depth}): {tot:,} rounds -> {tot / rate / 3600:.1f} h at collection rate (upper bound; successes end early)")
