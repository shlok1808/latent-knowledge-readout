#!/usr/bin/env python
"""B-TAP step A: build the pruning and preservation sets on a GPU (PRUNING_PLAN.md, knowledge_readout/syco/pruning_data.py).

    uv run --no-sync python scripts/build_pruning_sets.py --device cuda:0 --qwen-device cuda:1

1. CSQA (validation) + ARC-Challenge (test), cut 50/50 with items.split_pool_test(seed=0): the PRUNING POOL half only.
   The pool ids are checked against results/meta_instruct/bare_letters_everywhere/question_split.json (the eval's).
2. Llama-3.1-8B-Instruct (fp32 + TF32, like the eval): greedy neutral replies on the pool.
3. For every pool question Llama answers correctly: the 16 pruning candidates (2 kinds x 2 turns x 4 wordings),
   greedy replies (max --max-new-tokens), plus the bare-letter reading at the first generated position.
4. Qwen2.5-7B-Instruct: greedy neutral replies on the pool (P2a filter). Qwen's chat template inserts its default
   system prompt when none is given; we keep the template's default ("no system prompt" = none passed by us).
5. Judge, keep both-correct questions, fill the 8 boxes for every seed in --seeds (seed 0 = main, others = the
   3-seed stability check, D.2), build the preservation set per seed (Alpaca resampled + QA resampled).
Outputs:
  runs/btap/candidates.jsonl (all judged candidates, local), runs/btap/neutral_{llama,qwen}.jsonl
  results/meta_instruct/btap/sets/seed{s}/{prune,preserve}.jsonl   (token ids + text; what scoring reads)
  results/meta_instruct/btap/sets/report.json (counts, box report per seed, agreement, settings)
  results/meta_instruct/btap/sets/eyeball_seed0.json (20 random selected examples per box, for step B)
--limit N runs on the first N pool questions per dataset (smoke test; writes under .../sets_smoke).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_readout.syco import items as I  # noqa: E402
from knowledge_readout.syco import pruning_data as D  # noqa: E402
from knowledge_readout.syco import scorer as S  # noqa: E402

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
SPLITS = {"csqa": "validation", "arc_challenge": "test"}
EVAL_SPLIT = ROOT / "results" / "meta_instruct" / "bare_letters_everywhere" / "question_split.json"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load(hf_id, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32).to(device).eval()
    return model, tok


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def progress(name):
    t0 = time.time()

    def cb(done, total):
        if done == total or done % (32 * 20) == 0:
            log(f"  {name}: {done}/{total} ({time.time() - t0:.0f}s)")
    return cb


def pool_items(limit=None):
    pool, ids = [], {}
    for ds, split in SPLITS.items():
        items = I.load_items(ds, split=split)
        p, _ = I.split_pool_test(items, seed=0, pool_frac=0.5)
        ids[ds] = sorted(it.item_id for it in p)
        pool += p[:limit] if limit else p
    if EVAL_SPLIT.exists():
        saved = json.loads(EVAL_SPLIT.read_text())
        for ds in SPLITS:
            if sorted(saved[ds]["pool"]) != ids[ds]:
                raise SystemExit(f"{ds}: pool ids differ from {EVAL_SPLIT}; refusing to build on a different split")
        log("pool ids match the eval's question_split.json")
    else:
        log(f"WARNING: {EVAL_SPLIT} not found; pool split not cross-checked")
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen-device", default=None, help="default: same as --device (Qwen loads after Llama is freed)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--wrong-option-seed", type=int, default=0, help="items.wrong_option seed, same as the eval")
    ap.add_argument("--max-new-tokens", type=int, default=48,
                    help="long enough for '(L) option text<eot>' echoes; replies cut off here never count as caved")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    import torch

    out_res = ROOT / "results" / "meta_instruct" / "btap" / ("sets_smoke" if args.limit else "sets")
    out_runs = ROOT / "runs" / ("btap_smoke" if args.limit else "btap")
    pool = pool_items(args.limit)
    by_id = {it.item_id: it for it in pool}
    log(f"pool questions: {len(pool)}")
    from datasets import load_dataset
    alpaca = list(load_dataset("tatsu-lab/alpaca", split="train"))  # before any GPU work: a network error fails fast
    log(f"alpaca rows: {len(alpaca)}")
    gen_configs = {}

    # ---- Llama: neutral + candidates
    model, tok = load(LLAMA, args.device)
    bare_ids = S.letter_token_ids(tok, "ABCDE", ("{L}", " {L}"))
    gen_configs["llama"] = model.generation_config.to_dict()
    neutral_texts = [D.neutral_prompt(it, tok).text for it in pool]
    llama_neutral = D.generate_replies(model, tok, neutral_texts, args.max_new_tokens, args.batch_size,
                                       bare_letter_ids=bare_ids, progress=progress("llama neutral"),
                                       letters=[it.letters for it in pool])
    llama_by_id = {it.item_id: g for it, g in zip(pool, llama_neutral)}
    write_jsonl(out_runs / "neutral_llama.jsonl", [{"item_id": k, **v} for k, v in llama_by_id.items()])
    llama_ok = [it for it in pool if llama_by_id[it.item_id]["ended"]
                and D.parse_letter(llama_by_id[it.item_id]["reply_text"], it) == it.answer]
    log(f"llama neutral correct: {len(llama_ok)}/{len(pool)}")

    cfg = D.pruning_templates()
    cands, texts, cand_letters = [], [], []
    for it in llama_ok:
        for box, w, cue in D.candidate_cues(it, args.wrong_option_seed):
            p = D.build(it, cue, tok, cfg, neutral_answer=it.answer)
            cands.append({"box": list(box), "wording": w, "item_id": it.item_id, "dataset": it.dataset,
                          "kind": cue.kind, "turn": cue.turn, "source": cue.source, "target": cue.target,
                          "gold": it.answer, "prompt_text": p.text})
            texts.append(p.text)
            cand_letters.append(it.letters)
    log(f"candidates: {len(cands)}")
    gens = D.generate_replies(model, tok, texts, args.max_new_tokens, args.batch_size, bare_letter_ids=bare_ids,
                              progress=progress("candidates"), letters=cand_letters)
    eot_llama = sorted(D.end_of_turn_ids(model, tok))
    del model
    torch.cuda.empty_cache()

    judged = []
    for c, g in zip(cands, gens):
        c = {**c, **g}
        judged.append(D.judge(c, by_id[c["item_id"]]))

    # ---- Qwen neutral (P2a)
    qmodel, qtok = load(QWEN, args.qwen_device or args.device)
    gen_configs["qwen"] = qmodel.generation_config.to_dict()
    q_texts = [D.neutral_prompt(it, qtok).text for it in pool]
    qwen_neutral = D.generate_replies(qmodel, qtok, q_texts, args.max_new_tokens, args.batch_size,
                                      progress=progress("qwen neutral"))
    qwen_by_id = {it.item_id: g for it, g in zip(pool, qwen_neutral)}
    write_jsonl(out_runs / "neutral_qwen.jsonl", [{"item_id": k, **v} for k, v in qwen_by_id.items()])
    del qmodel
    torch.cuda.empty_cache()

    kept, filt = D.both_correct(pool, {k: v["reply_text"] for k, v in llama_by_id.items()},
                                {k: v["reply_text"] for k, v in qwen_by_id.items()},
                                llama_ended={k: v["ended"] for k, v in llama_by_id.items()})
    unparsed_q = [(it.item_id, qwen_by_id[it.item_id]["reply_text"]) for it in pool
                  if D.parse_letter(qwen_by_id[it.item_id]["reply_text"], it) is None]
    log(f"QWEN UNPARSED neutral replies: {len(unparsed_q)} of {len(pool)}; examples:")
    for iid, txt in unparsed_q[:10]:
        log(f"    {iid}: {txt!r}")
    unparsed_l = [(it.item_id, llama_by_id[it.item_id]["reply_text"]) for it in pool
                  if D.parse_letter(llama_by_id[it.item_id]["reply_text"], it) is None]
    log(f"LLAMA UNPARSED neutral replies: {len(unparsed_l)}; examples: {unparsed_l[:5]}")
    kept_ids = {it.item_id for it in kept}
    log(f"both-correct: {filt}")
    judged_kept = [c for c in judged if c["item_id"] in kept_ids]
    write_jsonl(out_runs / "candidates.jsonl", judged)

    # ---- sets per seed
    report = {"settings": {**vars(args), "llama": LLAMA, "qwen": QWEN, "dtype": "float32+tf32",
                           "eot_ids_llama": eot_llama, "wordings": D.WORDINGS, "generation_configs": gen_configs,
                           "decoding": "greedy, repetition_penalty=1.0, sampling off"},
              "unparsed_examples": {"qwen": unparsed_q[:30], "llama": unparsed_l[:30]},
              "pool": len(pool), "filter": filt, "candidates": len(judged), "candidates_both_correct": len(judged_kept),
              "agreement": D.agreement(judged_kept), "seeds": {}}
    for seed in args.seeds:
        selected, box_rep = D.fill_boxes(judged_kept, seed=seed)
        prune = []
        for box in D.BOXES:
            for c in selected[box]:
                prune.append({"set": "prune", "box": "/".join(box), **{k: c[k] for k in (
                    "item_id", "dataset", "kind", "turn", "wording", "target", "gold", "parsed", "bare_letter",
                    "prompt_text", "reply_text", "prompt_ids", "ended")}, "target_ids": c["reply_ids"]})
        n_alp = D.PRESERVATION_ALPACA if not args.limit else 8
        n_qa = D.PRESERVATION_QA_PER_DATASET if not args.limit else min(
            D.PRESERVATION_QA_PER_DATASET, *(sum(it.dataset == ds for it in kept) for ds in D.DATASETS))
        pres = D.alpaca_examples(alpaca, tok, eot_id=tok.convert_tokens_to_ids("<|eot_id|>"), n=n_alp, seed=seed)
        pres += D.qa_preservation(kept, llama_by_id, n_per_dataset=n_qa, seed=seed)
        d = out_res / f"seed{seed}"
        write_jsonl(d / "prune.jsonl", prune)
        write_jsonl(d / "preserve.jsonl", pres)
        report["seeds"][seed] = {"boxes": box_rep, "n_prune": len(prune), "n_preserve": len(pres),
                                 "pres_qa_overlapping_prune_questions": D.overlap_count(prune, pres),
                                 "alpaca_truncated": sum(r.get("truncated", False) for r in pres),
                                 "shortfall_total": sum(v["shortfall"] for v in box_rep.values())}
        log(f"seed {seed}: prune {len(prune)}, preserve {len(pres)}, shortfall {report['seeds'][seed]['shortfall_total']}")
        if seed == args.seeds[0]:
            eye = D.eyeball_sample(selected, per_box=20, seed=seed)
            (out_res / f"eyeball_seed{seed}.json").write_text(json.dumps([{k: c[k] for k in (
                "box", "item_id", "kind", "turn", "wording", "target", "gold", "parsed", "bare_letter", "reply_text",
                "prompt_text")} for c in eye], indent=1))
    (out_res / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"done: {out_res}")


if __name__ == "__main__":
    main()
