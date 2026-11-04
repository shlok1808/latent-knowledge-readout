"""Plumbing check: does a checkpoint (default: the BASE Llama-3.1-8B, no chat template) answer the
neutral multiple-choice prompts with a letter at all?

Neutral prompts only (no claims, no cues). This is a check of the setup before the base/Tulu stage
work, not an experiment and not a finding. It reuses the package's own prompt builder and scorer
(letter token ids, full-vocabulary softmax summed over the letter spellings, the two answer rules);
nothing is re-implemented here.

    uv run --no-sync python scripts/check_base_letters.py            # base model, plain, 200 questions
    uv run --no-sync python scripts/check_base_letters.py --n 60 --checkpoint tulu_sft

Questions: the first N/3 valid questions of csqa validation, arc_challenge test and obqa test (any
remainder goes to the earlier datasets; same order every time). Writes
results/<checkpoint>/letter_check.json with the summary (all questions and per dataset), one entry per
question, a sample prompt and a tokenizer-only preflight report, and prints a short summary.

letter_mass = the model's total next-token probability on this question's answer letters, spelled
"A" or " A" (the scorer's default spellings). "(A"-style spellings are not counted in it;
`share_top_token_is_any_spelling_of_A_to_E` shows how often the top token is one of those.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from knowledge_readout.common import write_json  # noqa: E402
from knowledge_readout.syco import items as item_loader  # noqa: E402
from knowledge_readout.syco import preflight as preflight_checks  # noqa: E402
from knowledge_readout.syco import prompts as prompt_builder  # noqa: E402
from knowledge_readout.syco import scorer  # noqa: E402

DATASETS_IN_ORDER = ("csqa", "arc_challenge", "obqa")
NUMBER_OF_TOP_TOKENS_TO_LIST = 20
ANSWER_RULES = {
    "answer_by_top_letter_token": "the letter of the single most likely letter token (the package's "
                                  "'greedy_token' rule, its default)",
    "answer_by_total_letter_probability": "the letter with the highest probability summed over its "
                                          "spellings (the package's 'letter_argmax' rule)",
}


def default_checkpoint_name():
    """The first configs/syco_checkpoints.json entry without a chat template (the base model)."""
    entries = scorer.load_checkpoints_config()["checkpoints"]
    for entry in entries:
        if entry.get("has_chat_template") is False:
            return entry["name"]
    return entries[0]["name"]


def questions_per_dataset(n, datasets=DATASETS_IN_ORDER):
    each, remainder = divmod(n, len(datasets))
    return {dataset: each + (1 if k < remainder else 0) for k, dataset in enumerate(datasets)}


def pick_questions(n):
    """The first questions of each dataset's default split, spread evenly; plus what loading reported."""
    questions, loading_report = [], {}
    for dataset, wanted in questions_per_dataset(n).items():
        split = item_loader.DATASETS[dataset][2]
        stats = {}
        loaded = item_loader.load_items(dataset, split=split, limit=wanted, stats=stats) if wanted else []
        loading_report[dataset] = {"split": split, "requested": wanted, "kept": len(loaded), **stats}
        questions.extend(loaded)
    return questions, loading_report


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def summarize_questions(rows, tokenizer):
    """Summary over per-question entries (see score_questions)."""
    n = len(rows)
    masses = [row["letter_mass"] for row in rows]
    non_letter_top_tokens = Counter(row["top_token_id"] for row in rows if not row["top_token_is_answer_letter"])
    any_spelling = [row["top_token_is_any_spelling_of_A_to_E"] for row in rows
                    if row["top_token_is_any_spelling_of_A_to_E"] is not None]

    def share(count):
        return count / n if n else None

    return {
        "number_of_questions": n,
        "letter_mass": {
            "mean": float(np.mean(masses)) if masses else None,
            "median": percentile(masses, 50),
            "10th_percentile": percentile(masses, 10),
            "90th_percentile": percentile(masses, 90),
            "minimum": float(min(masses)) if masses else None,
            "share_at_least_0.5": share(sum(m >= 0.5 for m in masses)),
            "share_at_least_0.9": share(sum(m >= 0.9 for m in masses)),
        },
        "share_top_token_is_answer_letter": share(sum(row["top_token_is_answer_letter"] for row in rows)),
        "share_top_token_is_any_spelling_of_A_to_E": (sum(any_spelling) / len(any_spelling)) if any_spelling else None,
        "share_where_two_answer_rules_disagree": share(sum(
            row["answer_by_top_letter_token"] != row["answer_by_total_letter_probability"] for row in rows)),
        "number_top_token_not_answer_letter": sum(non_letter_top_tokens.values()),
        "most_common_non_letter_top_tokens": [
            {"token_id": token_id, "token": scorer._decode_token(tokenizer, token_id), "count": count}
            for token_id, count in sorted(non_letter_top_tokens.items(), key=lambda kv: (-kv[1], kv[0]))
            [:NUMBER_OF_TOP_TOKENS_TO_LIST]],
    }


def score_questions(model, tokenizer, questions, checkpoint, prompt_format, batch_size,
                    letter_spellings=scorer.DEFAULT_VARIANTS):
    """Neutral prompts -> scorer.score_prompts -> one flat entry per question."""
    neutral = prompt_builder.neutral_cue()
    prompts = [prompt_builder.build_prompt(question, neutral, prompt_format, tokenizer) for question in questions]
    scores = scorer.score_prompts(model, tokenizer, prompts, checkpoint, batch_size=batch_size,
                                  variants=letter_spellings)
    try:
        any_spelling_ids = {token_id for ids in scorer.letter_token_ids(
            tokenizer, "ABCDE", preflight_checks.LETTER_SPELLINGS).values() for token_id in ids}
    except ValueError:
        any_spelling_ids = None
    rows = []
    for question, score in zip(questions, scores):
        top_token_id = int(score.extra["top_token_id"])
        rows.append({
            "question_id": question.item_id,
            "dataset": question.dataset,
            "number_of_options": len(question.letters),
            "letter_mass": float(score.letter_mass),
            "top_token_id": top_token_id,
            "top_token": scorer._decode_token(tokenizer, top_token_id),
            "top_token_is_answer_letter": bool(score.extra["greedy"]["top_token_is_letter"]),
            "top_token_is_any_spelling_of_A_to_E":
                (top_token_id in any_spelling_ids) if any_spelling_ids is not None else None,
            "answer_by_top_letter_token": scorer.answer_label(score, question.letters, "greedy_token"),
            "answer_by_total_letter_probability": scorer.answer_label(score, question.letters, "letter_argmax"),
            "letter_probabilities": {letter: float(p) for letter, p in score.probs.items()},
            "prompt_length_in_tokens": score.extra["n_tokens"],
        })
    return rows, prompts, scores


def build_report(model, tokenizer, questions, checkpoint, prompt_format, batch_size):
    rows, prompts, scores = score_questions(model, tokenizer, questions, checkpoint, prompt_format, batch_size)
    rows_by_dataset = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in DATASETS_IN_ORDER}
    answer_letter_ids = {token_id for question in questions
                         for ids in scorer.letter_token_ids(tokenizer, question.letters).values() for token_id in ids}
    return {
        "summary": summarize_questions(rows, tokenizer),
        "per_dataset": {dataset: summarize_questions(dataset_rows, tokenizer)
                        for dataset, dataset_rows in rows_by_dataset.items() if dataset_rows},
        "package_neutral_diagnostics": scorer.neutral_diagnostics(
            scores, tokenizer, letter_ids=answer_letter_ids, top_k=NUMBER_OF_TOP_TOKENS_TO_LIST),
        "sample_prompt": {"question_id": prompts[0].item_id, "text": prompts[0].text} if prompts else None,
        "questions": rows,
    }


def print_summary(report):
    summary = report["summary"]
    mass = summary["letter_mass"]
    print(f"Letter check for {report['checkpoint']} ({report['model_id']}), {report['prompt_format']} prompts, "
          f"{summary['number_of_questions']} questions. A plumbing check, not a finding.")
    if not summary["number_of_questions"]:
        return
    print(f"  Probability the model puts on the answer letters: average {mass['mean']:.2f}, "
          f"median {mass['median']:.2f}; 10% of questions are below {mass['10th_percentile']:.2f}, "
          f"10% are above {mass['90th_percentile']:.2f}.")
    print(f"  Share of questions where the model puts at least half its probability on a letter: "
          f"{mass['share_at_least_0.5']:.2f}")
    print(f"  Share of questions where it puts at least 90% of its probability on a letter: "
          f"{mass['share_at_least_0.9']:.2f}")
    print(f"  Share of questions where the single most likely next token is an answer letter: "
          f"{summary['share_top_token_is_answer_letter']:.2f}")
    print(f"  Share of questions where the two ways of reading off the answer disagree (top letter token "
          f"vs. most total letter probability): {summary['share_where_two_answer_rules_disagree']:.2f}")
    common = ", ".join(f"{t['token']!r} ({t['count']})" for t in summary["most_common_non_letter_top_tokens"][:8])
    print(f"  Most common next tokens when it is not a letter: {common or 'none'}")
    print("  By dataset:")
    for dataset, part in report["per_dataset"].items():
        print(f"    {dataset} ({part['number_of_questions']} questions): average letter probability "
              f"{part['letter_mass']['mean']:.2f}, at least half on a letter "
              f"{part['letter_mass']['share_at_least_0.5']:.2f}, top token is a letter "
              f"{part['share_top_token_is_answer_letter']:.2f}")
    problems = report["preflight_problems"]
    print(f"  Preflight problems: {'; '.join(problems) if problems else 'none'}")
    print(f"  Saved to {report['output_path']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default=None,
                        help="name or hf_id in configs/syco_checkpoints.json (default: the base model entry)")
    parser.add_argument("--fmt", default="plain", choices=prompt_builder.FORMATS, help="prompt format")
    parser.add_argument("--n", type=int, default=200, help="number of questions, split evenly over 3 datasets")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=None, help="default: results/<checkpoint>/letter_check.json")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least 1")
    args.checkpoint = args.checkpoint or default_checkpoint_name()
    try:
        args.checkpoint_entry = scorer.checkpoint_entry(args.checkpoint)
    except KeyError as e:
        parser.error(str(e))
    if args.fmt == "native" and args.checkpoint_entry.get("has_chat_template") is False:
        parser.error(f"{args.checkpoint_entry['name']} has no chat template, so the native format is refused; "
                     f"use --fmt plain")
    return args


def main(argv=None):
    args = parse_args(argv)
    name, model_id = args.checkpoint_entry["name"], args.checkpoint_entry["hf_id"]
    output_path = Path(args.out) if args.out else ROOT / "results" / name / "letter_check.json"
    start = time.time()
    questions, loading_report = pick_questions(args.n)
    model, tokenizer = scorer.load_checkpoint(model_id, dtype=args.dtype, device=args.device)
    loaded = time.time()
    preflight_report = preflight_checks.preflight(tokenizer, model=None, items=questions,
                                                  fmt_list=(args.fmt,), checkpoint=name)
    report = {
        "what_this_is": "plumbing check: probability on answer letters for neutral prompts "
                        "(no claims, no cues); not a finding",
        "checkpoint": name, "model_id": model_id, "prompt_format": args.fmt,
        "questions_requested": args.n, "questions_used": len(questions), "datasets": loading_report,
        "letter_spellings_counted": list(scorer.DEFAULT_VARIANTS), "answer_rules": ANSWER_RULES,
        "dtype_requested": args.dtype, "model_dtype": scorer.model_dtype(model),
        "model_fingerprint": scorer.model_fingerprint(model), "batch_size": args.batch_size,
        "git_commit": scorer._git_commit(), "library_versions": scorer._versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preflight_problems": preflight_report["problems"], "preflight_report": preflight_report,
        "output_path": str(output_path),
    }
    report.update(build_report(model, tokenizer, questions, name, args.fmt, args.batch_size))
    report["seconds"] = {"loading": round(loaded - start, 1), "total": round(time.time() - start, 1)}
    write_json(output_path, report)
    print_summary(report)
    return report


if __name__ == "__main__":
    main()
