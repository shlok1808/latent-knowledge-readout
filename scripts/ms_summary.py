"""Outcome summary of an mstep run: success, mixed-outcome tasks, lengths, by depth.

    python scripts/ms_summary.py runs/ms-tc-1p7b
"""
import collections
import glob
import json
import sys

run = sys.argv[1]
eps = [json.load(open(f)) for f in glob.glob(f"{run}/episodes/*.json")]
by = collections.defaultdict(list)
for e in eps:
    by[e["recipe_depth"]].append(e)
for d in sorted(by):
    E = by[d]
    succ = [e for e in E if e["success"]]
    tasks = collections.defaultdict(set)
    for e in E:
        tasks[e["task_id"]].add(e["success"])
    mixed = sum(len(v) == 2 for v in tasks.values())
    allf = sum(v == {False} for v in tasks.values())
    sl = sorted(collections.Counter(len(e["rounds"]) for e in succ).items())
    term = dict(collections.Counter(e["termination"] for e in E))
    print(f"depth {d}: eps {len(E)} success {len(succ) / len(E):.3f} mixed {mixed}/{len(tasks)} "
          f"all-fail {allf} term {term}\n   success lengths {sl}")
rounds = [r for e in eps for r in e["rounds"]]
print("rounds", len(rounds), "token-limit", sum(r["hit_token_limit"] for r in rounds),
      "no-action-span", sum(not r["action_token_indices"] for r in rounds))
