"""Smoke test for mstep.httpenv against running AgentGym servers (CPU only)."""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from mstep import httpenv

CASES = {
    "alfworld": ([2420, 2500, 2600], ["Thought: look.\n\nAction:\nlook", "Thought: x\n\nAction:\ninventory",
                                      "Thought: bad\n\nAction:\nfly to the moon"]),
    "sciworld": ([3491, 624, 1058], ["Thought: x\n\nAction:\nlook around", "Thought: x\n\nAction:\ninventory",
                                     "Thought: bad\n\nAction:\nfly to the moon"]),
    "babyai": ([1804, 1840, 12755], ["Thought: x\nAction: turn left", "Thought: x\nAction: move forward",
                                     "Thought: bad\nAction: fly to the moon"]),
}
only = sys.argv[1:] or list(CASES)
for env in only:
    idxs, resps = CASES[env]
    for i in idxs:
        ep = httpenv.Episode(env, {"data_idx": i})
        print(f"=== {env} {i} INITIAL: {ep.initial_observation[:300]!r}")
        for r in resps:
            out = ep.step(r)
            out["observation"] = out["observation"][:160]
            print("   ", json.dumps(out))
        ep.close()

if "sciworld" in only and "--noconc" not in sys.argv:
    free = lambda: subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1]
    print("RAM before:", free())
    def run(k):
        ep = httpenv.Episode("sciworld", {"data_idx": [3491, 624, 1058][k % 3]})
        for _ in range(5):
            ep.step("Thought: x\n\nAction:\nlook around")
        ep.close()
        return k
    t = time.time()
    with ThreadPoolExecutor(32) as pool:
        done = list(pool.map(run, range(64)))
    print(f"64 concurrent sciworld episodes x5 steps OK in {time.time()-t:.1f}s; RAM after:", free())
