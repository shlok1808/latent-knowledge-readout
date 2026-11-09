#!/usr/bin/env python
"""E3 (lab fork #28, bet #29): teacher-forced log-probs of fixed CAVE / RESIST replies under the original, pruned and
random models. knowledge_readout/syco/reply_probe.py has the reply builders and the scorer.

    python scripts/run_e3.py build                    # laptop or GPU box: writes results/meta_instruct/btap/e3/examples.jsonl
    python scripts/run_e3.py score --tag original --device cuda:0
    python scripts/run_e3.py score --tag pruned_seed0 --prune-indices results/meta_instruct/btap/seed0/selected.json
    python scripts/run_e3.py score --tag random_seed0 --prune-indices results/meta_instruct/btap/seed0/random.json

Example sets (field "set"):
  prune      the 512 pruning-set prompts (seed 0), cave letter = the letter the model wrote when it caved
  heldout    pushed TEST prompts (user source; suggestion+doubt x single+multi) for questions the ORIGINAL model
             answers correctly with no push (strict written reading); cave letter = items.wrong_option(item, 0) for
             suggestions (= the suggested option, as in the eval); for doubts the ORIGINAL model's second-choice
             letter on the neutral prompt (bare-letter Gate 1 run), so the cave reply is not a near-zero floor (critic)
  pres       the 256 QA preservation examples (seed 0), target = the model's own neutral reply: did protection hold?
  neutral    neutral TEST prompts, correct answer only
  real       replies the original / pruned model actually wrote (256-token runs, user, initially-correct cells) whose
             final answer is the cave or the resist letter, plus a twin with only the final answer swapped; tagged
             real_style = letter_first (reply opens with the answer) or reasons_first (critic: analyse separately)
Letter spelling for the letter form: single-turn "(B)", multi-turn "B" (the model's usual spelling there).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_readout.syco import items as I  # noqa: E402
from knowledge_readout.syco import prompts as P  # noqa: E402
from knowledge_readout.syco import reply_probe as R  # noqa: E402
from knowledge_readout.syco import scorer as S  # noqa: E402
from knowledge_readout.syco.types import Cue  # noqa: E402

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
OUT = ROOT / "results" / "meta_instruct" / "btap" / "e3"
SPLITS = {"csqa": "validation", "arc_challenge": "test", "obqa": "test"}
N_HELDOUT_Q = 200
N_NEUTRAL_Q = 300
N_REAL_PER_SOURCE = 300


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def key(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def load_all_items():
    pool, test = {}, {}
    for ds, split in SPLITS.items():
        items = I.load_items(ds, split=split)
        if ds == "obqa":
            p, t = [], items
        else:
            p, t = I.split_pool_test(items, seed=0, pool_frac=0.5)
        pool.update({it.item_id: it for it in p})
        test.update({it.item_id: it for it in t})
    return pool, test


def add(rows, tok, eot, prompt_text, item, letter, role, form, spelling, meta):
    reply = R.reply_text(item, letter, form, spelling)
    ids, ans = R.encode_reply(tok, reply, letter, eot)
    rows.append({**meta, "item_id": item.item_id, "letters": "".join(item.letters), "role": role, "form": form,
                 "letter": letter, "gold": item.answer,
                 "reply": reply, "prompt_ids": S.encode_text(tok, prompt_text)[0], "reply_ids": ids,
                 "answer_idx": ans})


def cmd_build(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLAMA)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    pool, test = load_all_items()
    assert not set(pool) & set(test), "pool and test questions overlap"
    slot = R.letter_slot_ids(tok)
    second = {}
    for line in open(ROOT / "runs/sycophancy_eval_meta_instruct_bare_everywhere/scores.jsonl"):
        x = json.loads(line)
        if x["kind"] == "neutral":
            ranked = sorted(x["probs"], key=lambda L: -x["probs"][L])
            second[x["item_id"]] = next(L for L in ranked if L != x["gold"])
    rows = []

    # (a) pruning-set prompts
    for i, r in enumerate(json.loads(l) for l in open(ROOT / "results/meta_instruct/btap/sets/seed0/prune.jsonl")):
        it = pool[r["item_id"]]
        cave, gold = r["parsed"], r["gold"]
        spelling = "bare" if r["reply_text"].lstrip()[0] != "(" else "paren"
        meta = {"set": "prune", "pid": f"prune{i}", "kind": r["kind"], "turn": r["turn"], "box": r["box"]}
        for role, L in (("cave", cave), ("resist", gold)):
            for form in R.FORMS:
                add(rows, tok, eot, r["prompt_text"], it, L, role, form, spelling, meta)

    # (b) held-out pushed prompts on questions the original model answers correctly (strict written reading)
    orig = {x["item_id"]: x for x in json.load(open(ROOT / "results/meta_instruct/btap/written_accuracy/original.json"))["rows"]}
    ok = sorted((iid for iid, x in orig.items() if x["written"] == x["gold"]), key=lambda i: key("e3_heldout", i))
    for iid in ok[:N_HELDOUT_Q]:
        it = test[iid]
        for kind in ("suggestion", "doubt"):
            wrong = I.wrong_option(it, 0) if kind == "suggestion" else second[iid]
            target = I.wrong_option(it, 0) if kind == "suggestion" else it.answer
            for turn in ("single", "multi"):
                cue = Cue(kind=kind, source="user", turn=turn, target=target, prior="correct")
                text = P.build_prompt(it, cue, "native", tokenizer=tok, neutral_answer=it.answer).text
                spelling = "paren" if turn == "single" else "bare"
                meta = {"set": "heldout", "pid": f"held:{iid}:{kind}:{turn}", "kind": kind, "turn": turn,
                        "box": f"{it.dataset}/{kind}/{turn}"}
                for role, L in (("cave", wrong), ("resist", it.answer)):
                    for form in R.FORMS:
                        add(rows, tok, eot, text, it, L, role, form, spelling, meta)

    # (c) neutral prompts, correct answer only
    for iid in sorted(test, key=lambda i: key("e3_neutral", i))[:N_NEUTRAL_Q]:
        it = test[iid]
        text = P.build_prompt(it, P.neutral_cue(), "native", tokenizer=tok).text
        meta = {"set": "neutral", "pid": f"neutral:{iid}", "kind": "neutral", "turn": "single", "box": f"{it.dataset}/neutral"}
        for form in R.FORMS:
            add(rows, tok, eot, text, it, it.answer, "correct", form, "paren", meta)

    # (c2) preservation QA examples: protection check
    for i, r in enumerate(json.loads(l) for l in open(ROOT / "results/meta_instruct/btap/sets/seed0/preserve.jsonl")):
        if r.get("kind") != "qa":
            continue
        rows.append({"set": "pres", "pid": f"pres{i}", "kind": "neutral", "turn": "single", "box": f"{r['dataset']}/pres",
                     "item_id": r["item_id"], "letters": "".join(pool[r["item_id"]].letters), "role": "correct",
                     "form": "pres_qa", "letter": r["gold"],
                     "gold": r["gold"], "reply": r["reply_text"], "prompt_ids": r["prompt_ids"],
                     "reply_ids": r["target_ids"], "answer_idx": 0})

    # (d) real written replies + minimal twins (user, single-turn, initially-correct cells)
    n_real = {}
    for src in ("original", "pruned_seed0"):
        data = json.load(open(ROOT / f"results/meta_instruct/btap/written_flips/{src}_long.json"))["rows"]
        cands = []
        for x in data:
            it = test[x["item_id"]]
            if x["cell"] not in ("incorrect assertion", "doubting a correct answer") or x["initial"] != it.answer:
                continue
            read = x.get("read")
            if read is None or read not in it.letters:
                continue
            if read == it.answer:  # resisted: the twin caves (to the suggested option, or the fixed wrong option)
                other = x["target"] if x["cell"] == "incorrect assertion" else second[it.item_id]
            else:  # caved: suggestion caves only to the suggested option; doubt to any wrong option
                if x["cell"] == "incorrect assertion" and read != x["target"]:
                    continue
                other = it.answer
            if x["reply"].count(f"({read})") != 1 or any(f"({L})" in x["reply"] for L in it.letters if L != read):
                continue  # replies naming more than one option (or the same one twice) would give mixed twins
            twin = R.swap_final_answer(x["reply"], it, read, other)
            if twin is None:
                continue
            cands.append((x, it, read, twin, other))
        cands.sort(key=lambda c: key("e3_real", src, c[0]["item_id"], c[0]["cell"]))
        n_real[src] = len(cands)
        for x, it, read, twin, other in cands[:N_REAL_PER_SOURCE]:
            kind = "suggestion" if x["cell"] == "incorrect assertion" else "doubt"
            cue = Cue(kind=kind, source="user", turn="single", target=x["target"], prior="correct")
            text = P.build_prompt(it, cue, "native", tokenizer=tok, neutral_answer=it.answer).text
            head = x["reply"].lstrip()
            style = "letter_first" if (head[:1] == "(" or head[:1] in it.letters and head[1:2] in ("", ")", ".", ":", "<")
                                       or head.lower().startswith("the correct answer is")) else "reasons_first"
            meta = {"set": "real", "pid": f"real:{src}:{x['item_id']}:{kind}", "kind": kind, "turn": "single",
                    "box": f"{it.dataset}/{kind}/single", "written_by": src, "real_style": style}
            for txt, L, is_orig in ((x["reply"], read, True), (twin, other, False)):
                body = txt.split("<|eot_id|>")[0]
                ids, ans = R.encode_reply(tok, body, L, eot)
                if "<|eot_id|>" not in txt:
                    ids = ids[:-1]  # the model did not end there; keep the reply exactly as written
                rows.append({**meta, "item_id": it.item_id, "letters": "".join(it.letters),
                             "role": "resist" if L == it.answer else "cave",
                             "form": "real_as_written" if is_orig else "real_twin", "letter": L, "gold": it.answer,
                             "reply": txt, "prompt_ids": S.encode_text(tok, text)[0], "reply_ids": ids,
                             "answer_idx": ans})
    for r in rows:  # letter-only replies must open with a canonical letter token
        if r["form"] in ("letter", "pres_qa"):
            assert r["reply_ids"][0] in slot, (r["pid"], r["reply"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slot_ids.json").write_text(json.dumps({str(k): v for k, v in sorted(slot.items())}))
    with open(OUT / "examples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    summary = {"n": len(rows), "by_set": Counter(r["set"] for r in rows),
               "by_set_role_form": Counter(f"{r['set']}|{r['role']}|{r['form']}" for r in rows),
               "real_candidates": n_real}
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary["by_set"]) + f" total {len(rows)}; real candidates {n_real}")


def cmd_score(args):
    import torch
    from transformers import AutoModelForCausalLM

    from knowledge_readout.syco import pruning as PR

    torch.backends.cuda.matmul.allow_tf32 = True
    rows = [json.loads(l) for l in open(OUT / "examples.jsonl")]
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32).to(args.device).eval()
    if args.prune_indices:
        idx, meta = PR.load_indices(args.prune_indices)
        PR.check_indices_match(model, meta)
        PR.apply_zeroing(model, idx, expected_shapes=meta.get("shapes"))
    t0 = time.time()
    slot = [int(k) for k in json.loads((OUT / "slot_ids.json").read_text())]
    lp = R.score_replies(model, rows, batch_size=args.batch_size, pad_id=128004, slot_ids=slot,
                         progress=lambda d, n: log(f"  {d}/{n} ({time.time() - t0:.0f}s)") if d == n or d % 1024 == 0 else None)
    with open(OUT / f"scores_{args.tag}.jsonl", "w") as f:
        for n, (r, l) in enumerate(zip(rows, lp)):
            f.write(json.dumps({"row": n, "pid": r["pid"], "role": r["role"], "form": r["form"], **l}) + "\n")
    log(f"{args.tag}: scored {len(rows)} in {time.time() - t0:.0f}s; peak {torch.cuda.max_memory_allocated(args.device) / 2**30:.1f} GiB")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--tag", required=True)
    s.add_argument("--prune-indices", type=Path, default=None)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--batch-size", type=int, default=32, help="32 fits fp32 8B on a 40 GB A100 (64 would OOM)")
    args = ap.parse_args(argv)
    cmd_build(args) if args.cmd == "build" else cmd_score(args)


if __name__ == "__main__":
    main()
