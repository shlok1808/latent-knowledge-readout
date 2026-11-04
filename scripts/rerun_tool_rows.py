"""Rescore only the single-turn tool-row prompts of a finished run with the corrected layout, then compare.

Lab fork #19 (2026-11-04): the native tool row's single-turn layout now follows Prompt C1's order
(question -> tool call -> tool result -> instruction). The first run put the instruction before the tool
call, and Llama answered "The final answer is E." (about 1% of its probability on letters).

The prompts are rebuilt from the finished run's OWN rows, so the selection is exact: the same questions,
groups (initially right / wrong), targets and neutral answers. Only the layout changes. Checks (each one
stops the script if it fails):
  - exactly 2 rows (suggestion, doubt) per test question are replaced: 1,685 x 2 = 3,370 for Meta Instruct;
  - every other row is kept byte for byte, so the merged file has the same row count as the original;
  - each new row keeps the old row's item, kind, source, turn, target, group, neutral answer and gold;
  - each rebuilt prompt differs from the old one (it hashes differently) and ends with the instruction turn.
Outputs: runs/<run>/instruction_after_tool/scores.jsonl (merged rows) and
results/<checkpoint>/instruction_after_tool/compare_to_paper.json (+ all_cells.json, rerun_checks.json).

Usage (GPU machine, repo root):
    uv run --no-sync python scripts/rerun_tool_rows.py --checkpoint meta_instruct
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

from knowledge_readout.syco import items as I
from knowledge_readout.syco import prompts as P
from knowledge_readout.syco import scorer as S
from knowledge_readout.syco.types import Cue

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_sycophancy_eval", ROOT / "scripts" / "run_sycophancy_eval.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)

KEEP_FIELDS = ("item_id", "dataset", "checkpoint", "fmt", "kind", "source", "turn", "target", "prior",
               "neutral_answer", "gold", "prior_rule")


def is_tool_single(row):
    return row["source"] == "tool" and row["turn"] == "single" and row["kind"] in ("suggestion", "doubt")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="meta_instruct")
    ap.add_argument("--run-dir", type=Path, default=None, help="default runs/sycophancy_eval_<checkpoint>")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--allow-tf32", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    args = ap.parse_args(argv)

    run_dir = args.run_dir or ROOT / "runs" / f"sycophancy_eval_{args.checkpoint}"
    out_dir = run_dir / "instruction_after_tool"
    res_dir = ROOT / "results" / args.checkpoint / "instruction_after_tool"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    old_lines = (run_dir / "scores.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in old_lines]
    old_tool = [r for r in rows if is_tool_single(r)]
    n_questions = len({r["item_id"] for r in rows if r["kind"] == "neutral"})
    checks = {"rows_in_original": len(rows), "test_questions": n_questions, "tool_single_rows_found": len(old_tool)}
    per_item = {}
    for r in old_tool:
        per_item.setdefault(r["item_id"], []).append(r["kind"])
    assert len(old_tool) == 2 * n_questions, f"expected {2 * n_questions} tool single-turn rows, found {len(old_tool)}"
    assert all(sorted(k) == ["doubt", "suggestion"] for k in per_item.values()), "an item lacks one of the 2 rows"
    fmt = {r["fmt"] for r in rows}
    assert fmt == {"native"}, fmt

    by_id = {}
    for ds in ("csqa", "arc_challenge", "obqa"):
        for it in I.load_items(ds):
            by_id[it.item_id] = it
    entry = RUN.load_checkpoint_entry(args.checkpoint) if hasattr(RUN, "load_checkpoint_entry") else None
    hf_id = (entry or {}).get("hf_id") or {"meta_instruct": "meta-llama/Llama-3.1-8B-Instruct"}[args.checkpoint]
    model, tok = RUN.load_model_and_tokenizer(hf_id, "float32", "cuda", args.allow_tf32)

    prompts = []
    for r in old_tool:
        cue = Cue(kind=r["kind"], source=r["source"], turn=r["turn"], target=r["target"], prior=r["prior"])
        p = P.build_prompt(by_id[r["item_id"]], cue, "native", tokenizer=tok, neutral_answer=r["neutral_answer"],
                           tool_mode="native")
        assert [m["role"] for m in p.messages][-2:] == ["tool", "user"], "rebuilt prompt does not end with the instruction"
        assert S.prompt_sha256(p) != (r.get("extra") or {}).get("prompt_sha256"), f"{r['item_id']}: prompt unchanged"
        prompts.append(p)

    t0 = time.time()
    scores = S.score_prompts(model, tok, prompts, args.checkpoint, batch_size=args.batch_size,
                             variants=S.VARIANTS_WITH_PAREN)
    checks["rescoring_minutes"] = round((time.time() - t0) / 60, 2)

    new_by_key = {}
    assert len(scores) == len(old_tool)
    for i, (r, sc) in enumerate(zip(old_tool, scores)):
        assert sc.item_id == r["item_id"] and sc.cue.kind == r["kind"], f"score order mismatch at {i}"
        new = S.score_row(sc, by_id[r["item_id"]], r["neutral_answer"], dataset=r["dataset"],
                          prior_rule=r.get("prior_rule"))
        for f in KEEP_FIELDS:
            assert new.get(f) == r.get(f), f"{r['item_id']}: field {f} changed ({r.get(f)!r} -> {new.get(f)!r})"
        new.setdefault("extra", {})["prompt_sha256"] = S.prompt_sha256(prompts[i])
        new["extra"]["layout"] = "instruction_after_tool"
        new_by_key[S.row_key(r)] = new

    merged_lines, replaced = [], 0
    for line, r in zip(old_lines, rows):
        if is_tool_single(r):
            merged_lines.append(json.dumps(new_by_key[S.row_key(r)]))
            replaced += 1
        else:
            merged_lines.append(line)   # kept byte for byte
    assert replaced == len(old_tool) and len(merged_lines) == len(old_lines)
    (out_dir / "scores.jsonl").write_text("\n".join(merged_lines) + "\n")

    old_mass = sum(r["letter_mass"] for r in old_tool) / len(old_tool)
    new_mass = sum(v["letter_mass"] for v in new_by_key.values()) / len(new_by_key)
    checks.update({"rows_replaced": replaced, "rows_in_merged": len(merged_lines),
                   "other_rows_unchanged": len(merged_lines) - replaced,
                   "tool_single_letter_mass_before": round(old_mass, 4),
                   "tool_single_letter_mass_after": round(new_mass, 4),
                   "share_below_half_after": round(sum(v["letter_mass"] < 0.5 for v in new_by_key.values())
                                                   / len(new_by_key), 4),
                   "example_prompt": prompts[0].text[-700:]})
    (res_dir / "rerun_checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps({k: v for k, v in checks.items() if k != "example_prompt"}, indent=1))

    merged = [json.loads(line) for line in merged_lines]
    result, all_cells = RUN.compare_to_paper(merged, args.checkpoint, "native", "native",
                                             args.bootstrap_samples, args.seed)
    (res_dir / "compare_to_paper.json").write_text(json.dumps(result, indent=1, default=str))
    (res_dir / "all_cells.json").write_text(json.dumps(all_cells, indent=1, default=str))
    for row in result.get("table", []):
        print(f'{row["condition"]:26} {row["column"]:24} {row["role"]:9} paper {row["theirs"]:.2f} '
              f'ours {row["ours"]:.2f} {row["status"]}')
    print("saved", res_dir)


if __name__ == "__main__":
    main()
