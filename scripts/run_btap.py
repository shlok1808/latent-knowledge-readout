#!/usr/bin/env python
"""B-TAP steps C (score) and the selection that feeds step D (PRUNING_PLAN.md; knowledge_readout/syco/pruning.py).

Score half the layers per GPU, in parallel (2x H100), then merge and select:

    uv run --no-sync python scripts/run_btap.py score  --seed 0 --layers 0-15  --device cuda:0
    uv run --no-sync python scripts/run_btap.py score  --seed 0 --layers 16-31 --device cuda:1
    uv run --no-sync python scripts/run_btap.py select --seed 0 --device cuda:0

score:  reads results/meta_instruct/btap/sets/seed{S}/{prune,preserve}.jsonl, loads Llama-3.1-8B-Instruct in fp32 with
        TF32 matmuls (same numerics as the eval), scores the given layers (main + first-token variant for seed 0),
        saves runs/btap/seed{S}/score_L{a}-{b}.pt (shortlists only, small).
select: merges every score part for the seed (must cover all layers exactly once), selects the top 1,000 (main), the
        first-token top 1,000 (seed 0) and their overlap, builds the magnitude-matched random control (seed S),
        writes results/meta_instruct/btap/seed{S}/{selected,random,first_token}.json + report.json.
Overlap across seeds (stability, D.2): `select` also reports overlap with every other seed already selected.
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

from knowledge_readout.syco import pruning as PR  # noqa: E402

LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
SETS = ROOT / "results" / "meta_instruct" / "btap" / "sets"
OUT_RES = ROOT / "results" / "meta_instruct" / "btap"
OUT_RUNS = ROOT / "runs" / "btap"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(device):
    import torch
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32).to(device).eval()
    model.config.use_cache = False
    return model


def parse_layers(spec):
    return PR.parse_layer_spec(spec)


EX_KEYS = ("loss", "prompt_ids", "target_ids", "base_prompt_ids", "pos_ids", "neg_ids")


def load_examples(paths):
    out = []
    for p in paths:
        out += [{k: r[k] for k in EX_KEYS if k in r} for r in read_jsonl(p)]
    return out


def arm_name(args):
    """Output folder name: seed{S} for the paper runs, or an E2 arm name (e.g. e2b) if --arm is given."""
    return args.arm or f"seed{args.seed}"


def cmd_score(args):
    import torch

    d = SETS / f"seed{args.seed}"
    prune = load_examples(args.prune_file or [d / "prune.jsonl"])
    if args.neg_prune_file:  # E2-E: these examples are SUBTRACTED, scaled so both groups weigh the same in total
        neg = load_examples(args.neg_prune_file)
        w = -len(prune) / len(neg)
        prune += [dict(ex, weight=w) for ex in neg]
        log(f"contrastive pruning set: {len(prune) - len(neg)} positive + {len(neg)} negative (weight {w:.3f})")
    pres = load_examples(args.pres_file or [d / "preserve.jsonl"])
    log(f"seed {args.seed}: {len(prune)} pruning, {len(pres)} preservation examples; layers {args.layers}")
    if len(prune) != 512 or len(pres) != 512:
        log(f"WARNING: expected 512 + 512 examples, got {len(prune)} + {len(pres)}")
    model = load_model(args.device)
    from transformers import AutoTokenizer
    pad_id = AutoTokenizer.from_pretrained(LLAMA).convert_tokens_to_ids("<|finetune_right_pad_id|>")
    t0 = time.time()
    last = [0.0]

    def progress(phase, i, n):
        now = time.time()
        if phase == "group" or i == n or now - last[0] > 60:
            last[0] = now
            log(f"  {phase} {i}/{n} ({now - t0:.0f}s)")

    first = args.first_token if args.first_token is not None else (args.seed == 0)
    if any(PR.is_margin(ex) for ex in prune):
        first = False
    result = PR.score(model, prune, pres, layers=parse_layers(args.layers), group_size=args.group_size,
                      first_token_variant=first, prune_batch_size=args.prune_batch_size, pad_id=pad_id,
                      progress=progress, protect_frac=args.protect_frac, pres_split=args.pres_split)
    result.settings.update({"arm": arm_name(args), "prune_files": [str(x) for x in (args.prune_file or [])],
                            "pres_files": [str(x) for x in (args.pres_file or [])]})
    out = OUT_RUNS / arm_name(args) / f"score_L{args.layers.replace(',', '+')}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    log(f"saved {out} in {(time.time() - t0) / 60:.1f} min; peak GPU memory "
        f"{torch.cuda.max_memory_allocated(args.device) / 2**30:.1f} GiB")


def cmd_select(args):
    import torch

    parts = sorted((OUT_RUNS / arm_name(args)).glob("score_L*.pt"))
    if not parts:
        raise SystemExit(f"no score parts for {arm_name(args)}")
    results = [torch.load(p, weights_only=False) for p in parts]
    merged = PR.merge_results(*results)
    log(f"merged {len(parts)} parts: {len(merged.matrices)} matrices")
    selected, info = PR.select(merged, n=args.n, variant="main", return_info=True)
    model = load_model(args.device)
    if len(merged.matrices) != len(PR.eligible_matrices(model)):
        raise SystemExit(f"score parts cover {len(merged.matrices)} matrices, model has "
                         f"{len(PR.eligible_matrices(model))}; score the missing layers first")
    rand = PR.random_matched(model, selected, seed=args.seed)
    d = OUT_RES / arm_name(args)
    meta = {"seed": args.seed, "n": args.n, "model": LLAMA, "settings": merged.settings,
            "n_prune": merged.n_prune, "n_pres": merged.n_pres}
    PR.save_indices(d / "selected.json", selected, {**meta, "kind": "btap"}, shapes=merged.shapes, model_id=LLAMA)
    PR.save_indices(d / "random.json", rand, {**meta, "kind": "random_magnitude_matched", "random_seed": args.seed},
                    shapes=merged.shapes, model_id=LLAMA)
    report = {"meta": meta, "select_info": info, "overlap_random_vs_selected": PR.overlap(rand, selected),
              "per_matrix_type": {}}
    for s in selected:
        t = s["matrix"].rsplit(".", 2)[-2]
        report["per_matrix_type"][t] = report["per_matrix_type"].get(t, 0) + 1
    if merged.first is not None:
        first = PR.select(merged, n=args.n, variant="first")
        PR.save_indices(d / "first_token.json", first, {**meta, "kind": "btap_first_token"}, shapes=merged.shapes,
                        model_id=LLAMA)
        report["overlap_first_token_vs_main"] = PR.overlap(first, selected)
    report["overlap_with_other_seeds"] = {}
    for other in sorted(OUT_RES.glob("seed*/selected.json")):
        if other.parent.name != arm_name(args):
            report["overlap_with_other_seeds"][other.parent.name] = PR.overlap(PR.load_indices(other)[0], selected)
    (d / "report.json").write_text(json.dumps(report, indent=1, default=str))
    log(json.dumps({k: v for k, v in report.items() if k != "select_info"}, default=str))


def cmd_skills(args):
    """General-capability PROXY (not the paper's Table D5 suite): mean reference-reply cross-entropy on held-out Alpaca
    rows (never in any seed's preservation set), for the original model or with --prune-indices zeroed."""
    import hashlib

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from knowledge_readout.syco import pruning_data as D
    from knowledge_readout.syco import scorer as S

    tok = AutoTokenizer.from_pretrained(LLAMA)
    rows = list(load_dataset("tatsu-lab/alpaca", split="train"))
    used = set()
    for f in SETS.glob("seed*/preserve.jsonl"):
        used |= {r["alpaca_index"] for r in read_jsonl(f) if r.get("kind") == "alpaca"}
    if not used:
        raise SystemExit("no preservation sets found; build them first")
    key = lambda i: hashlib.sha256(f"skills\x1f{i}".encode()).hexdigest()
    pick = sorted((i for i, r in enumerate(rows) if i not in used and r["instruction"].strip() and r["output"].strip()),
                  key=key)[: args.n]
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    exs = []
    for i in pick:
        text = D.P.render_native([{"role": "user", "content": D.alpaca_prompt_text(rows[i])}], tok)
        exs.append({"prompt_ids": S.encode_text(tok, text)[0],
                    "target_ids": (tok.encode(rows[i]["output"].strip(), add_special_tokens=False) + [eot])[:512]})
    model = load_model(args.device)
    kind = "original"
    if args.prune_indices:
        idx, meta = PR.load_indices(args.prune_indices)
        PR.check_indices_match(model, meta)
        PR.apply_zeroing(model, idx, expected_shapes=meta.get("shapes"))
        kind = meta.get("kind", "pruned")
    losses = []
    with torch.no_grad():
        for ex in exs:
            losses.append(float(PR.example_loss(model, ex)))
    out = {"kind": kind, "prune_indices": str(args.prune_indices) if args.prune_indices else None, "n": len(losses),
           "mean_ce": sum(losses) / len(losses), "alpaca_indices": pick, "losses": losses,
           "note": "proxy for general capability, not the paper's Table D5 suite"}
    path = OUT_RES / "skills" / f"{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    log(f"skills {args.tag}: mean CE {out['mean_ce']:.4f} over {len(losses)} held-out Alpaca replies")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--arm", default=None, help="output folder name for an E2 arm (default seed{S})")
    s.add_argument("--prune-file", type=Path, nargs="+", default=None)
    s.add_argument("--pres-file", type=Path, nargs="+", default=None)
    s.add_argument("--protect-frac", type=float, default=PR.PROTECT_FRAC)
    s.add_argument("--neg-prune-file", type=Path, nargs="+", default=None,
                   help="E2-E: examples subtracted from the pruning score (weight -n_pos/n_neg)")
    s.add_argument("--pres-split", action="store_true",
                   help="protect the top p of EACH preservation subset (CE / margin) and take the union")
    s.add_argument("--layers", default="0-31")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--group-size", type=int, default=8)
    s.add_argument("--prune-batch-size", type=int, default=4,
                   help="4 keeps activation memory low enough for fp32 8B on 80 GB without gradient checkpointing")
    s.add_argument("--first-token", type=int, choices=(0, 1), default=None, help="default: on for seed 0 only")
    t = sub.add_parser("select")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--arm", default=None)
    t.add_argument("--n", type=int, default=PR.N_PRUNE)
    t.add_argument("--device", default="cuda:0")
    k = sub.add_parser("skills")
    k.add_argument("--prune-indices", type=Path, default=None)
    k.add_argument("--tag", required=True)
    k.add_argument("--n", type=int, default=200)
    k.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    if args.cmd == "skills":
        return cmd_skills(args)
    if args.cmd == "score":
        if args.first_token is not None:
            args.first_token = bool(args.first_token)
        cmd_score(args)
    else:
        cmd_select(args)


if __name__ == "__main__":
    main()
