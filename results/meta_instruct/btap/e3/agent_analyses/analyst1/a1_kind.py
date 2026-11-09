"""Analyst 1, part 3: DiD (cave-row slot, as analyze_e3) split by kind; pruned and random. Run: python3 a1_kind.py > a1_kind.out"""
import sys
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from a1_lib import cluster_boot, fmt, load, pairs, restricted  # noqa
ex, spell, let, sc = load()
P = pairs(ex)
for T in ("pruned_seed0", "random_seed0"):
    print("--", T)
    acc = defaultdict(lambda: defaultdict(list))
    for (pid, f), d in P.items():
        if "cave" not in d or "resist" not in d:
            continue
        ic, ir = d["cave"], d["resist"]; r = ex[ic]
        if r["set"] not in ("heldout", "prune") or f == "letter_last":
            continue
        o, _ = restricted(sc["original"][ic]["slot"], let, r["letters"]); p, _ = restricted(sc[T][ic]["slot"], let, r["letters"])
        c, s = r["letter"], ex[ir]["letter"]
        acc[(r["set"], f, r["turn"], r["kind"])][r["item_id"]].append((p[c] - p[s]) - (o[c] - o[s]))
    for k in sorted(acc):
        print(f"  {'|'.join(k):40} DiD {fmt(cluster_boot(acc[k]))}")
