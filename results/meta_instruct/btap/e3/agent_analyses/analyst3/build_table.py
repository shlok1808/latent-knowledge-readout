#!/usr/bin/env python
"""ANALYST 3 / E3: build a per-row feature table from examples.jsonl + scores_*.jsonl.

Run from the repo root with the project venv:
    .venv/bin/python <this file>
Writes table.pkl (pandas) next to this script. Read-only on the repo.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shloki/latent-knowledge-readout")
E3 = ROOT / "results/meta_instruct/btap/e3"
OUT = Path(__file__).resolve().parent
TAGS = ("original", "pruned_seed0", "random_seed0")

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)

ex = [json.loads(l) for l in open(E3 / "examples.jsonl")]
slot = json.loads((E3 / "slot_ids.json").read_text())
order = [v for _, v in sorted(((int(k), v) for k, v in slot.items()))]  # [(spelling, letter)] x15
SPELL = [s for s, _ in order]
LET = [L for _, L in order]

sc = {}
for t in TAGS:
    p = E3 / f"scores_{t}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in open(p)]
        assert len(rows) == len(ex) and all(r["row"] == i and r["pid"] == ex[i]["pid"] for i, r in enumerate(rows))
        sc[t] = rows
print("tags:", list(sc))


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def letter_stats(vals, letters):
    """Per-letter log mass (spellings summed), total letter log mass, per-spelling log mass (item letters only)."""
    by_L, by_S = {}, {}
    for v, s, L in zip(vals, SPELL, LET):
        if L in letters:
            by_L.setdefault(L, []).append(v)
            by_S.setdefault(s, []).append(v)
    lpL = {L: lse(v) for L, v in by_L.items()}
    lpS = {s: lse(v) for s, v in by_S.items()}
    tot = lse(list(lpL.values()))
    return lpL, lpS, tot


OPT_RE = re.compile(r"^\(([A-E])\) (.*)$", re.M)
CUE_LETTER_RE = re.compile(r"[Oo]ption ([A-E])\b")

recs = []
for i, r in enumerate(ex):
    prompt = tok.decode(r["prompt_ids"])
    # user turns
    user_turns = re.findall(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", prompt, re.S)
    asst_turns = re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", prompt, re.S)
    first_user = user_turns[0]
    last_user = user_turns[-1]
    opts = dict(OPT_RE.findall(first_user))
    question = first_user.split("\n(A)")[0].strip()
    # cue sentence: the last user turn minus the instruction and minus the question block
    cue = last_user
    cue = cue.replace("Give your final answer using exactly one option letter.", "").strip()
    if r["turn"] == "single" and r["kind"] != "neutral":
        # drop the question + options block
        parts = cue.split("\n\n")
        cue = parts[-1].strip() if len(parts) > 1 else ""
    elif r["kind"] == "neutral":
        cue = ""
    m = CUE_LETTER_RE.search(cue)
    cue_letter = m.group(1) if m else None
    prev_letter = asst_turns[0].strip() if asst_turns else None
    letters = r["letters"]
    rec = dict(
        row=i, set=r["set"], pid=r["pid"], kind=r["kind"], turn=r["turn"], box=r["box"], dataset=r["box"].split("/")[0],
        item_id=r["item_id"], letters=letters, n_opt=len(letters), role=r["role"], form=r["form"], letter=r["letter"],
        gold=r["gold"], written_by=r.get("written_by"), real_style=r.get("real_style"), reply=r["reply"],
        answer_idx=r["answer_idx"], n_reply=len(r["reply_ids"]), prompt_len=len(r["prompt_ids"]),
        question=question, cue=cue, cue_letter=cue_letter, prev_letter=prev_letter,
        letter_pos=letters.index(r["letter"]), gold_pos=letters.index(r["gold"]),
        opt_text=opts.get(r["letter"], ""), opt_len=len(opts.get(r["letter"], "")),
        gold_text=opts.get(r["gold"], ""), q_len=len(question),
        mean_opt_len=float(np.mean([len(v) for v in opts.values()])) if opts else np.nan,
        slot_tok=tok.decode([r["reply_ids"][r["answer_idx"]]]),
    )
    for t, rows in sc.items():
        s = rows[i]
        rec[f"{t}_logp"] = np.array(s["logp"])
        rec[f"{t}_sum"] = float(sum(s["logp"]))
        rec[f"{t}_slot_tok_lp"] = float(s["logp"][r["answer_idx"]])
        for which in ("slot", "first"):
            vals = s[which]
            lpL, lpS, tot = letter_stats(vals, letters)
            rec[f"{t}_{which}_mass"] = tot  # log total letter mass (item letters, all spellings)
            for L in "ABCDE":
                rec[f"{t}_{which}_lp_{L}"] = lpL.get(L, np.nan)          # log mass of letter L (spellings summed)
                rec[f"{t}_{which}_r_{L}"] = (lpL[L] - tot) if L in lpL else np.nan  # letter-restricted log prob
            for sp in ("bare", "space", "paren"):
                rec[f"{t}_{which}_sp_{sp}"] = lpS[sp] - tot  # log share of spelling within letter mass
            rec[f"{t}_{which}_r_letter"] = lpL[r["letter"]] - tot
            rec[f"{t}_{which}_r_gold"] = lpL[r["gold"]] - tot
            if cue_letter in lpL:
                rec[f"{t}_{which}_r_cue"] = lpL[cue_letter] - tot
            else:
                rec[f"{t}_{which}_r_cue"] = np.nan
            if prev_letter in lpL:
                rec[f"{t}_{which}_r_prev"] = lpL[prev_letter] - tot
            else:
                rec[f"{t}_{which}_r_prev"] = np.nan
            # restricted entropy over letters
            ps = np.exp(np.array([lpL[L] - tot for L in letters]))
            rec[f"{t}_{which}_H"] = float(-(ps * np.log(ps + 1e-30)).sum())
            rec[f"{t}_{which}_argmax"] = max(lpL, key=lpL.get)
            # raw 15-vector kept for spelling analyses
            rec[f"{t}_{which}_raw"] = np.array(vals)
    recs.append(rec)

df = pd.DataFrame(recs)
df.to_pickle(OUT / "table.pkl")
print(df.shape)
print(df[["set", "kind", "turn", "form", "role"]].value_counts().head(30))
print("cue_letter missing:", df[(df.kind != "neutral")].cue_letter.isna().sum())
print("cue letter == reply letter by role/kind:")
print(df[df.kind != "neutral"].assign(eq=lambda d: d.cue_letter == d.letter).groupby(["kind", "role"]).eq.mean())
