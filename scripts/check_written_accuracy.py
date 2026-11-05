#!/usr/bin/env python
"""Gate 2 diagnostic: neutral accuracy on the TEST questions read from what the model WRITES (greedy reply, parsed),
next to the bare-letter reading at the first generated position, for the original / pruned / random model.

Why: after B-TAP pruning, the bare-letter eval showed 7 points lower neutral accuracy and half the letter mass; the
paper says the initially-correct sets stay "nearly identical". This separates real damage from a reading artifact.

    uv run --no-sync python scripts/check_written_accuracy.py --tag pruned_seed0 \
        --prune-indices results/meta_instruct/btap/seed0/selected.json --device cuda:0
Writes results/meta_instruct/btap/written_accuracy/<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_readout.syco import items as I  # noqa: E402
from knowledge_readout.syco import pruning as PR  # noqa: E402
from knowledge_readout.syco import pruning_data as D  # noqa: E402
from knowledge_readout.syco import scorer as S  # noqa: E402

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
SPLITS = {"csqa": "validation", "arc_challenge": "test", "obqa": "test"}


def test_items():
    out = []
    for ds, split in SPLITS.items():
        items = I.load_items(ds, split=split)
        out += items if ds == "obqa" else I.split_pool_test(items, seed=0, pool_frac=0.5)[1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prune-indices", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(LLAMA)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32).to(args.device).eval()
    if args.prune_indices:
        idx, meta = PR.load_indices(args.prune_indices)
        PR.check_indices_match(model, meta)
        PR.apply_zeroing(model, idx, expected_shapes=meta.get("shapes"))
    items = test_items()
    bare = S.letter_token_ids(tok, "ABCDE", ("{L}", " {L}"))
    gens = D.generate_replies(model, tok, [D.neutral_prompt(it, tok).text for it in items], args.max_new_tokens, 64,
                              bare_letter_ids=bare, letters=[it.letters for it in items])
    per = {}
    rows = []
    for it, g in zip(items, gens):
        w = D.parse_letter(g["reply_text"], it) if g["ended"] else None
        c = per.setdefault(it.dataset, Counter())
        c["n"] += 1
        c["written_correct"] += w == it.answer
        c["written_unparsed"] += w is None
        c["bare_correct"] += g["bare_letter"] == it.answer
        c["bare_differs_from_written"] += (w is not None and g["bare_letter"] != w)
        c["reply_starts_with_paren"] += g["reply_text"].lstrip().startswith("(")
        rows.append({"item_id": it.item_id, "gold": it.answer, "written": w, "bare": g["bare_letter"],
                     "reply": g["reply_text"]})
    total = sum((v for v in per.values()), Counter())
    out = {"tag": args.tag, "prune_indices": str(args.prune_indices) if args.prune_indices else None,
           "per_dataset": {k: dict(v) for k, v in per.items()}, "total": dict(total),
           "written_accuracy": total["written_correct"] / total["n"], "bare_accuracy": total["bare_correct"] / total["n"],
           "rows": rows}
    path = ROOT / "results" / "meta_instruct" / "btap" / "written_accuracy" / f"{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"{args.tag}: written acc {out['written_accuracy']:.3f}, bare acc {out['bare_accuracy']:.3f}, "
          f"unparsed {total['written_unparsed']}, bare != written {total['bare_differs_from_written']}, "
          f"starts with '(' {total['reply_starts_with_paren']} of {total['n']}", flush=True)


if __name__ == "__main__":
    main()
