"""Per-round table with every label, MC value, stage assignment and baseline text.

One row per (episode, round), in the same order as acts/index.json, so row i of
the table is row i of the activation memmaps.

Label polarity: every binary label is coded 1 = the BAD outcome (episode fails,
action invalid, no progress, MC value < 0.5), matching stage-2's `failure`
coding. AUC is symmetric, so this only fixes the sign of diff-of-means scores.

  python -m mstep.dataset --run runs/ms-tc-1p7b
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from failure_probes.common import read_json
from mstep import tc as tcmod

ROUND_STAGES = [("early", 1, 2), ("mid", 3, 5), ("late", 6, 10**6)]
# ALFWorld has no native progress signal; its own state-change messages mark sub-goals (A-21).
ALFWORLD_PROGRESS = ("You pick up", "You put", "You clean", "You heat", "You cool", "You turn on", "You slice")


def http_round_labels(env, r):
    """Validity / progress / milestone event for AgentGym HTTP envs (A-21)."""
    if env == "alfworld":
        progress = r["observation"].startswith(ALFWORLD_PROGRESS)
    else:  # sciworld score 0-100, babyai subgoal reward in [0,1]: native per-step progress
        progress = (r.get("score_delta") or 0) > 0
    return {"valid": int(r["validity"] == "executable"), "progress": int(progress), "crafted": int(progress)}


def round_stage(r):
    return next(name for name, lo, hi in ROUND_STAGES if lo <= r <= hi)


def build(run):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    env = manifest["config"].get("env", "textcraft")
    tags = manifest.get("item_id_to_tag", {})
    index = read_json(run / "acts" / "index.json") if (run / "acts" / "index.json").exists() else None
    rows, cache = [], {}
    order = index or [{"episode_id": p.stem, "round": None} for p in sorted((run / "episodes").glob("*.json"))]
    if index is None:  # no extraction yet: enumerate every round
        expanded = []
        for item in order:
            e = read_json(run / "episodes" / f"{item['episode_id']}.json")
            expanded += [{"episode_id": e["episode_id"], "round": r["round"]} for r in e["rounds"]]
        order = expanded
    for i, item in enumerate(order):
        eid = item["episode_id"]
        if eid not in cache:
            cache.clear()
            e = read_json(run / "episodes" / f"{eid}.json")
            mc_path = run / "mc" / f"{eid}.json"
            mc = read_json(mc_path)["values"] if mc_path.exists() else {}
            crafts, texts, prior = 0, [], []
            per_round = {}
            task = tasks[e["task_id"]]
            first_obs = task.get("initial_observation") or e.get("initial_observation", "")
            for r in e["rounds"]:
                lab = tcmod.round_labels(task, r, tags) if env == "textcraft" else http_round_labels(env, r)
                terminal_goal = bool(r["done"] and r["reward"] == 1)
                milestone = None
                if lab["crafted"] and not terminal_goal:
                    crafts += 1
                    milestone = crafts
                # Observable text up to and including round t's RESPONSE, but not round t's
                # observation: the activation has not seen it, and for IMMEDIATE labels it
                # would leak the label (A-15).
                text = "\n".join([first_obs] + texts + [r["response"]])
                surface = [len(r["generated_token_ids"]), len(r["prompt_token_ids"]),
                           sum(p["validity"] != "executable" for p in prior),
                           sum(p["validity"] in ("impossible", "invalid") for p in prior),
                           len({p["action"] for p in prior})]
                per_round[r["round"]] = {
                    "valid_bad": 1 - lab["valid"], "noprogress_bad": 1 - lab["progress"],
                    "milestone": milestone, "text": text, "surface": surface,
                    "mc_value": mc.get(str(r["round"]), np.nan), "action": r["action"],
                    "validity": r["validity"], "terminal_goal": terminal_goal}
                texts += [r["response"], r["observation"]]
                prior.append(r)
            cache[eid] = (e, per_round)
        e, per_round = cache[eid]
        rnd = item["round"]
        d = per_round[rnd]
        rows.append({"row": i, "episode_id": eid, "task_id": e["task_id"], "depth": e["recipe_depth"],
                     "attempt": e["attempt"], "round": rnd, "n_rounds": len(e["rounds"]),
                     "stage": round_stage(rnd), "final_bad": int(not e["success"]),
                     "mc_value": d["mc_value"],
                     "mc_bad": (np.nan if np.isnan(d["mc_value"]) else int(d["mc_value"] < 0.5)),
                     **{k: d[k] for k in ("valid_bad", "noprogress_bad", "milestone", "text", "surface",
                                           "action", "validity", "terminal_goal")}})
    df = pd.DataFrame(rows)
    # Milestone of the episode's latest non-terminal craft at or before this round.
    df["milestones_reached"] = df.groupby("episode_id")["milestone"].transform(lambda s: s.ffill().fillna(0))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    df = build(a.run)
    df.to_pickle(Path(a.run) / "rounds.pkl")
    ep = df.groupby("episode_id").first()
    mixed = ep.groupby("task_id")["final_bad"].agg(lambda s: s.nunique() == 2).sum()
    print(f"{len(df)} rows, {len(ep)} episodes, success {1 - ep.final_bad.mean():.3f}, mixed tasks {mixed}")
    print(df.groupby("depth").agg(rows=("row", "size"), eps=("episode_id", "nunique"),
                                  fail=("final_bad", "mean"), mc=("mc_value", "count")))


if __name__ == "__main__":
    main()
