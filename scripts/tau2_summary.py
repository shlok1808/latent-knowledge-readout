"""Per-domain outcome summary of a τ² run: success (deterministic), full τ² reward, mixed tasks,
lengths, terminations, step-label rate.   python scripts/tau2_summary.py runs/ms-tau2-qwen3-8b-pilot"""
import collections, glob, json, sys
eps = [json.load(open(f)) for f in glob.glob(f"{sys.argv[1]}/episodes/*.json")]
for dom in ("retail", "airline"):
    E = [e for e in eps if e["task_id"].startswith("tau2_" + dom)]
    if not E:
        continue
    t = collections.defaultdict(set)
    for e in E:
        t[e["task_id"]].add(e["success"])
    L = [len(e["rounds"]) for e in E]
    sl = sorted(len(e["rounds"]) for e in E if e["success"])
    full = sum(e["rounds"][-1].get("tau2_reward", 0) >= 1 for e in E)
    m = sum(r["expected_action_match"] for e in E for r in e["rounds"]); n = sum(L)
    inv = sum(r["validity"] == "invalid" for e in E for r in e["rounds"])
    print(f"{dom}: eps {len(E)} | success {sum(e['success'] for e in E) / len(E):.3f} (full tau2 reward {full / len(E):.3f}) "
          f"| mixed {sum(len(v) == 2 for v in t.values())}/{len(t)} all-fail {sum(v == {False} for v in t.values())} "
          f"all-success {sum(v == {True} for v in t.values())} | rounds mean {n / len(E):.1f} | success lengths {sl} "
          f"| term {dict(collections.Counter(e['termination'] for e in E))} | step-match {m / n:.2f} invalid {inv / n:.2f}")
