"""Analyst 1 shared loader (independent of scripts/analyze_e3.py). numpy only."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

E3 = Path("/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/e3")
TAGS = ("original", "pruned_seed0", "random_seed0")


def load():
    ex = [json.loads(l) for l in open(E3 / "examples.jsonl")]
    slot = json.loads((E3 / "slot_ids.json").read_text())
    ids = sorted(int(k) for k in slot)
    spell = [slot[str(i)][0] for i in ids]
    let = [slot[str(i)][1] for i in ids]
    sc = {}
    for t in TAGS:
        p = E3 / f"scores_{t}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        if len(rows) != len(ex):
            print(f"[warn] {t}: {len(rows)} rows != {len(ex)}; skipped")
            continue
        for i, r in enumerate(rows):
            assert r["row"] == i and r["pid"] == ex[i]["pid"] and r["role"] == ex[i]["role"] and r["form"] == ex[i]["form"]
        sc[t] = rows
    return ex, np.array(spell), np.array(let), sc


def lse(a):
    a = np.asarray(a, float)
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def letter_lp(vec, let, letters):
    """{L: log sum over spellings p(L)} restricted to item letters (unnormalised), and log total letter mass."""
    vec = np.asarray(vec, float)
    d = {L: lse(vec[let == L]) for L in letters}
    return d, lse(list(d.values()))


def restricted(vec, let, letters):
    d, tot = letter_lp(vec, let, letters)
    return {L: v - tot for L, v in d.items()}, tot


def cluster_boot(by_q, reps=2000, seed=0):
    """by_q: {question: [values]}. Mean of per-question means, percentile CI resampling questions."""
    keys = sorted(by_q)
    if not keys:
        return None
    m = np.array([np.mean(by_q[k]) for k in keys])
    rng = np.random.default_rng(seed)
    bs = np.array([m[rng.integers(0, len(m), len(m))].mean() for _ in range(reps)])
    return {"mean": float(m.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "n_q": len(keys)}


def fmt(b):
    return "NA" if b is None else f"{b['mean']:+.3f} [{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}] nq={b['n_q']}"


def pairs(ex):
    """(pid, form) -> {role: row index}"""
    P = defaultdict(dict)
    for i, r in enumerate(ex):
        P[(r["pid"], r["form"])][r["role"]] = i
    return P
