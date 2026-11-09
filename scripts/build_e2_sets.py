#!/usr/bin/env python
"""E2 (new) example sets: format-safe margin examples for B-TAP (knowledge_readout/syco/PRUNING_PLAN.md, "E2 (new)").

    python scripts/build_e2_sets.py caving                      # laptop OK: results/meta_instruct/btap/e2/caving_margin.jsonl
    python scripts/build_e2_sets.py useful --device cuda:0      # GPU: results/meta_instruct/btap/e2/useful_margin.jsonl

caving  The same 512 pushed prompts as the paper-style pruning set (seed 0). Margin example:
        pos = every spelling of the letter the model caved to, neg = every spelling of the gold letter,
        base = the question's neutral prompt. Loss (pruning.batch_margin_loss) = -(m(pushed) - m(neutral)).
useful  Useful-updating examples for the E2-C preservation set. Pruning-pool questions the model answers WRONG with no
        push (its own written neutral reply, ended, parsed); a reliable source (the 12 Table C3 text sources) asserts
        the gold ("correct assertion") or doubts the model's wrong answer ("doubting a wrong answer"), single and
        multi-turn; kept only when the model's greedy written reply (max 48 new tokens, ended) picks the gold.
        Margin example: pos = gold spellings, neg = the model's initial wrong letter, base = neutral prompt.
        256 examples, balanced over kind x turn (64 each), sources rotated, a question at most once per (kind, turn).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_readout.syco import items as I  # noqa: E402
from knowledge_readout.syco import metrics as M  # noqa: E402
from knowledge_readout.syco import prompts as P  # noqa: E402
from knowledge_readout.syco import pruning_data as D  # noqa: E402
from knowledge_readout.syco import scorer as S  # noqa: E402
from knowledge_readout.syco.types import Cue  # noqa: E402

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
OUT = ROOT / "results" / "meta_instruct" / "btap" / "e2"
SPLITS = {"csqa": "validation", "arc_challenge": "test"}
N_USEFUL = 256
TEXT_SOURCES = tuple(s for s in M.EXTERNAL_SOURCES if s != P.TOOL_SOURCE)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def key(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def pool_items():
    out = {}
    for ds, split in SPLITS.items():
        p, _ = I.split_pool_test(I.load_items(ds, split=split), seed=0, pool_frac=0.5)
        out.update({it.item_id: it for it in p})
    return out


def spellings(tok, letter):
    ids = []
    for text in (f"({letter}", letter, f" {letter}"):
        e = tok.encode(text, add_special_tokens=False)
        if len(e) != 1:
            raise ValueError(f"{text!r} is not one token")
        ids.append(int(e[0]))
    return ids


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def cmd_caving(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLAMA)
    pool = pool_items()
    rows = []
    for i, r in enumerate(json.loads(l) for l in open(ROOT / "results/meta_instruct/btap/sets/seed0/prune.jsonl")):
        it = pool[r["item_id"]]
        cave = r["parsed"]
        if cave == it.answer or cave not in it.letters:
            raise ValueError(f"row {i}: cave letter {cave!r} vs gold {it.answer!r}")
        base = S.encode_text(tok, D.neutral_prompt(it, tok).text)[0]
        rows.append({"loss": "margin", "prompt_ids": r["prompt_ids"], "base_prompt_ids": base,
                     "pos_ids": spellings(tok, cave), "neg_ids": spellings(tok, it.answer),
                     "item_id": it.item_id, "box": r["box"], "kind": r["kind"], "turn": r["turn"], "cave": cave,
                     "gold": it.answer})
    write(OUT / "caving_margin.jsonl", rows)
    log(f"caving margin examples: {len(rows)}; boxes {Counter(r['box'] for r in rows)}")


def cmd_useful(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(LLAMA)
    pool = pool_items()
    neutral = {json.loads(l)["item_id"]: json.loads(l) for l in open(ROOT / "runs/btap/neutral_llama.jsonl")}
    wrong = []
    for iid, g in neutral.items():
        it = pool[iid]
        a = D.parse_letter(g["reply_text"], it) if g["ended"] else None
        if a is not None and a != it.answer:
            wrong.append((it, a))
    log(f"pool questions answered wrong (written, parsed): {len(wrong)}")
    cands, texts = [], []
    for it, a in wrong:
        for kind in ("suggestion", "doubt"):
            for turn in ("single", "multi"):
                for src in TEXT_SOURCES:
                    cue = Cue(kind=kind, source=src, turn=turn, target=it.answer if kind == "suggestion" else a,
                              prior="wrong")
                    p = P.build_prompt(it, cue, "native", tokenizer=tok, neutral_answer=a)
                    cands.append({"item_id": it.item_id, "initial": a, "gold": it.answer, "kind": kind, "turn": turn,
                                  "source": src, "prompt_text": p.text})
                    texts.append(p.text)
    log(f"candidates: {len(cands)}")
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32).to(args.device).eval()
    gens = D.generate_replies(model, tok, texts, 48, args.batch_size,
                              progress=lambda d, n: log(f"  {d}/{n}") if d == n or d % 2048 == 0 else None)
    followed = defaultdict(list)
    for c, g in zip(cands, gens):
        it = pool[c["item_id"]]
        r = D.parse_letter(g["reply_text"], it) if g["ended"] else None
        c.update({"reply_text": g["reply_text"], "read": r})
        if r == it.answer:
            followed[(c["kind"], c["turn"])].append(c)
    rate = {f"{k[0]}/{k[1]}": len(v) for k, v in followed.items()}
    totals = Counter(c["kind"] + "/" + c["turn"] for c in cands)
    log(f"followed (picked gold): {rate} of {dict(totals)}")
    rows = []
    per = N_USEFUL // 4
    for (kind, turn), lst in sorted(followed.items()):
        lst = sorted(lst, key=lambda c: key("e2_useful", c["item_id"], kind, turn, c["source"]))
        seen, picked = set(), []
        for c in lst:
            if c["item_id"] in seen:
                continue
            seen.add(c["item_id"])
            picked.append(c)
            if len(picked) == per:
                break
        if len(picked) < per:
            log(f"WARNING: {kind}/{turn} only {len(picked)} of {per}")
        for c in picked:
            it = pool[c["item_id"]]
            rows.append({"loss": "margin", "prompt_ids": S.encode_text(tok, c["prompt_text"])[0],
                         "base_prompt_ids": S.encode_text(tok, D.neutral_prompt(it, tok).text)[0],
                         "pos_ids": spellings(tok, it.answer), "neg_ids": spellings(tok, c["initial"]),
                         **{k: c[k] for k in ("item_id", "kind", "turn", "source", "initial", "gold", "reply_text")}})
    write(OUT / "useful_margin.jsonl", rows)
    write(ROOT / "runs/btap/e2_useful_candidates.jsonl", cands)
    (OUT / "useful_report.json").write_text(json.dumps({"wrong_questions": len(wrong), "candidates": len(cands),
                                                        "followed": rate, "kept": Counter(f"{r['kind']}/{r['turn']}" for r in rows),
                                                        "sources": Counter(r["source"] for r in rows)}, indent=1))
    log(f"useful margin examples: {len(rows)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("caving")
    u = sub.add_parser("useful")
    u.add_argument("--device", default="cuda:0")
    u.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    cmd_caving(args) if args.cmd == "caving" else cmd_useful(args)


if __name__ == "__main__":
    main()
