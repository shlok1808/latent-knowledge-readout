"""Would reading only bare letters ("A", " A"; not "(A") change Table D1? A side check (lab bet #21).

Why: every doubt cell runs a bit high, and the doubt shift tracks how confident the model's starting answer is.
The paper never says how it read P(option). If it counted only bare letters, it was rescaling the small slice of
probability left on "A"/" A" (Llama writes "(A" as one token), which may be less confident. This check rescores
the SAME single-turn prompts (neutral, suggestion, doubt; all 14 sources, the tool row in its corrected layout)
counting only bare letters, keeps the groups (initially right / wrong), targets and neutral answers of the main
run, and compares to the paper. Only the letter spellings change; precision (fp32 + TF32) and prompts are the same.

Input: the merged rows of the main run with the corrected tool row,
       runs/sycophancy_eval_meta_instruct/instruction_after_tool/scores.jsonl
Output: results/meta_instruct/bare_letters/compare_to_paper.json (+ all_cells.json, checks.json)

Usage (GPU machine, repo root): uv run --no-sync python scripts/check_bare_letters.py
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

BARE = ("{L}", " {L}")
KEEP_FIELDS = ("item_id", "dataset", "checkpoint", "fmt", "kind", "source", "turn", "target", "prior",
               "neutral_answer", "gold")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/sycophancy_eval_meta_instruct/instruction_after_tool/scores.jsonl")
    ap.add_argument("--out", default="results/meta_instruct/bare_letters")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(ROOT / args.rows)]
    single = [r for r in rows if r["turn"] == "single" and r["kind"] in ("neutral", "suggestion", "doubt")]
    n_q = len({r["item_id"] for r in single if r["kind"] == "neutral"})
    assert len(single) == n_q * (1 + 14 * 2), f"expected {n_q * 29} single-turn rows, found {len(single)}"

    by_id = {}
    for ds in ("csqa", "arc_challenge", "obqa"):
        for it in I.load_items(ds):
            by_id[it.item_id] = it
    model, tok = RUN.load_model_and_tokenizer("meta-llama/Llama-3.1-8B-Instruct", "float32", "cuda", True)

    prompts = []
    for r in single:
        if r["kind"] == "neutral":
            cue = P.neutral_cue()
        else:
            cue = Cue(kind=r["kind"], source=r["source"], turn="single", target=r["target"], prior=r["prior"])
        p = P.build_prompt(by_id[r["item_id"]], cue, "native", tokenizer=tok, neutral_answer=r["neutral_answer"],
                           tool_mode="native")
        old_sha = (r.get("extra") or {}).get("prompt_sha256")
        assert old_sha is None or S.prompt_sha256(p) == old_sha, f"{r['item_id']} {r['kind']} {r['source']}: prompt differs"
        prompts.append(p)

    t0 = time.time()
    scores = S.score_prompts(model, tok, prompts, "meta_instruct", batch_size=args.batch_size, variants=BARE)
    minutes = (time.time() - t0) / 60

    new_rows, label_changes = [], 0
    for r, sc in zip(single, scores):
        assert sc.item_id == r["item_id"]
        new = S.score_row(sc, by_id[r["item_id"]], r["neutral_answer"], dataset=r["dataset"],
                          prior_rule=r.get("prior_rule"))
        for f in KEEP_FIELDS:
            assert new.get(f) == r.get(f), f"{r['item_id']}: field {f} changed"
        # keep the main run's labels (groups and targets must not move); store the bare-letter greedy pick apart
        new["extra"]["greedy_bare_letters"] = new["extra"].get("greedy")
        new["extra"]["greedy"] = (r.get("extra") or {}).get("greedy")
        if r["kind"] == "neutral" and (new["extra"]["greedy_bare_letters"] or {}).get("top_letter_variant_letter") \
                != r["neutral_answer"]:
            label_changes += 1
        new_rows.append(new)

    neutral = [x for x in new_rows if x["kind"] == "neutral"]
    checks = {"single_turn_rows_rescored": len(new_rows), "test_questions": n_q, "minutes": round(minutes, 1),
              "letter_spellings": list(BARE),
              "neutral_letter_mass_mean_bare": sum(x["letter_mass"] for x in neutral) / len(neutral),
              "neutral_letter_mass_mean_main_run": sum(r["letter_mass"] for r in single if r["kind"] == "neutral") / n_q,
              "questions_whose_bare_letter_pick_differs_from_the_main_label": label_changes,
              "note": "groups, targets and neutral answers are the main run's; only P(option) is read from bare letters"}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))

    result, all_cells = RUN.compare_to_paper(new_rows, "meta_instruct", "native", "native",
                                             args.bootstrap_samples, args.seed)
    (out / "compare_to_paper.json").write_text(json.dumps(result, indent=1, default=str))
    (out / "all_cells.json").write_text(json.dumps(all_cells, indent=1, default=str))
    for row in result.get("table", []):
        print(f'{row["condition"]:26} {row["column"]:24} paper {row["theirs"]:.2f} ours {row["ours"]:.2f} {row["status"]}')
    print("saved", out)


if __name__ == "__main__":
    main()
