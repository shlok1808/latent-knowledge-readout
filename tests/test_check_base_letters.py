"""scripts/check_base_letters.py on the offline tiny Llama (tests/syco_tiny.py). CPU, no network:
load_items and load_checkpoint are monkeypatched."""
import importlib.util
import json
from pathlib import Path

import pytest

from knowledge_readout.syco import items as item_loader
from knowledge_readout.syco import prompts as prompt_builder
from knowledge_readout.syco import scorer
from knowledge_readout.syco.types import Item
from syco_tiny import tiny_model, tiny_tokenizer

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_base_letters.py"
_spec = importlib.util.spec_from_file_location("check_base_letters", SCRIPT_PATH)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

OPTION_WORDS = ["red", "blue", "green", "cat", "dog"]
OPTIONS_PER_DATASET = {"csqa": 5, "arc_challenge": 4, "obqa": 4}


def make_fake_load_items(calls):
    def fake_load_items(dataset, split=None, limit=None, stats=None):
        calls.append((dataset, split, limit))
        letters = "ABCDE"[:OPTIONS_PER_DATASET[dataset]]
        questions = [Item(item_id=f"{dataset}:{split}:{i}", dataset=dataset, split=split,
                          question=f"Which one is it {i}?" + " What" * (i % 3),
                          options=tuple((letter, OPTION_WORDS[k]) for k, letter in enumerate(letters)),
                          answer="ABCD"[i % 4])
                     for i in range(limit or 0)]
        if stats is not None:
            stats.update({"rows": len(questions), "kept": len(questions), "skipped": 0})
        return questions
    return fake_load_items


@pytest.fixture
def tiny_setup(monkeypatch):
    tokenizer = tiny_tokenizer(chat_template=False)  # like the base model
    model = tiny_model(tokenizer)
    load_items_calls, load_checkpoint_calls = [], []
    monkeypatch.setattr(item_loader, "load_items", make_fake_load_items(load_items_calls))

    def fake_load_checkpoint(hf_id, dtype="bfloat16", device="cuda"):
        load_checkpoint_calls.append((hf_id, dtype, device))
        return model, tokenizer
    monkeypatch.setattr(scorer, "load_checkpoint", fake_load_checkpoint)
    return tokenizer, model, load_items_calls, load_checkpoint_calls


def test_defaults_and_even_split():
    assert script.default_checkpoint_name() == "base"
    assert script.questions_per_dataset(200) == {"csqa": 67, "arc_challenge": 67, "obqa": 66}
    args = script.parse_args([])
    assert (args.checkpoint, args.fmt, args.n, args.device, args.dtype) == ("base", "plain", 200, "cuda", "bfloat16")
    assert args.checkpoint_entry["hf_id"] == "meta-llama/Llama-3.1-8B"
    with pytest.raises(SystemExit):
        script.parse_args(["--fmt", "native"])  # the base model has no chat template


def test_main_on_tiny_model(tiny_setup, tmp_path, capsys):
    tokenizer, model, load_items_calls, load_checkpoint_calls = tiny_setup
    output_path = tmp_path / "base" / "letter_check.json"
    report = script.main(["--n", "10", "--device", "cpu", "--dtype", "float32", "--batch-size", "3",
                          "--out", str(output_path)])
    assert load_checkpoint_calls == [("meta-llama/Llama-3.1-8B", "float32", "cpu")]
    assert load_items_calls == [("csqa", "validation", 4), ("arc_challenge", "test", 3), ("obqa", "test", 3)]
    saved = json.loads(output_path.read_text())
    assert saved["checkpoint"] == "base" and saved["prompt_format"] == "plain" and saved["questions_used"] == 10
    assert saved["sample_prompt"]["text"].endswith("Assistant:")
    assert [saved["per_dataset"][d]["number_of_questions"] for d in ("csqa", "arc_challenge", "obqa")] == [4, 3, 3]
    assert report["preflight_report"]["checkpoint_config"]["name"] == "base"

    # the numbers are the package scorer's, unchanged (scored here unbatched)
    questions = [q for dataset, n in (("csqa", 4), ("arc_challenge", 3), ("obqa", 3))
                 for q in item_loader.load_items(dataset, item_loader.DATASETS[dataset][2], n)]
    prompts = [prompt_builder.build_prompt(q, prompt_builder.neutral_cue(), "plain", tokenizer) for q in questions]
    scores = scorer.score_prompts(model, tokenizer, prompts, "base", batch_size=1)
    rows = saved["questions"]
    assert [row["question_id"] for row in rows] == [q.item_id for q in questions]
    for row, score, question in zip(rows, scores, questions):
        assert row["letter_mass"] == pytest.approx(score.letter_mass, abs=1e-5)
        assert row["top_token_is_answer_letter"] == score.extra["greedy"]["top_token_is_letter"]
        assert row["answer_by_top_letter_token"] == scorer.answer_label(score, question.letters, "greedy_token")
        assert row["answer_by_total_letter_probability"] == \
            scorer.answer_label(score, question.letters, "letter_argmax")

    summary = saved["summary"]
    masses = sorted(row["letter_mass"] for row in rows)
    mass = summary["letter_mass"]
    assert mass["10th_percentile"] <= mass["median"] <= mass["90th_percentile"]
    assert mass["minimum"] == pytest.approx(masses[0])
    assert mass["share_at_least_0.5"] == sum(m >= 0.5 for m in masses) / 10
    assert mass["share_at_least_0.9"] == sum(m >= 0.9 for m in masses) / 10
    assert summary["share_top_token_is_answer_letter"] == sum(row["top_token_is_answer_letter"] for row in rows) / 10
    assert summary["share_where_two_answer_rules_disagree"] == sum(
        row["answer_by_top_letter_token"] != row["answer_by_total_letter_probability"] for row in rows) / 10
    assert sum(t["count"] for t in summary["most_common_non_letter_top_tokens"]) == \
        summary["number_top_token_not_answer_letter"] == sum(not row["top_token_is_answer_letter"] for row in rows)
    assert "share of questions where the model puts at least half its probability on a letter" in \
        capsys.readouterr().out.lower()
