"""Relabel BabyAI episodes collected before the success-rule fix (2026-09-24).

Old rule: success = done and reward >= 1.0. But MiniGrid's completion reward is
1 - 0.9*steps/max_steps (0.84-0.99 observed), so every success was recorded as a
failure with termination 'env_done'. New rule (mstep/httpenv.py): success = done and
"The task is completed" in the observation. Rewrites episodes in place, keeping the
old values under `relabel_2026_09_24`.

    python scripts/relabel_babyai.py runs/ms-babyai-1p7b-pilot
"""
import glob, json, sys
from failure_probes.common import write_json

run = sys.argv[1]
changed = 0
for f in sorted(glob.glob(f"{run}/episodes/*.json")):
    e = json.load(open(f))
    if "relabel_2026_09_24" in e:
        continue
    last = e["rounds"][-1]
    ok = bool(last["done"] and "The task is completed" in last["observation"])
    e["relabel_2026_09_24"] = {"success": e["success"], "termination": e["termination"], "last_reward": last["reward"]}
    for r in e["rounds"]:
        r["reward"] = int(bool(r["done"] and "The task is completed" in r["observation"]))
    e["success"] = ok
    e["termination"] = "success" if ok else e["termination"]
    changed += ok != e["relabel_2026_09_24"]["success"]
    write_json(f, e)
print(f"relabelled {run}: {changed} episodes flipped to success")
