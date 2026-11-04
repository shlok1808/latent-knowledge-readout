"""End-to-end test of scripts/run_sycophancy_eval.py on the offline tiny model (CPU, no network)."""
import importlib.util
import json
from pathlib import Path

import pytest

from knowledge_readout.syco import items as I
from knowledge_readout.syco import preflight as PF
from knowledge_readout.syco.types import Item
from syco_tiny import tiny_model, tiny_tokenizer

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_sycophancy_eval", ROOT / "scripts" / "run_sycophancy_eval.py")
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)

# Llama-3.1-like enough for the native tool row: tools in the system block, tool call, ipython result.
TOOL_TEMPLATE = (
    "{{ bos_token }}"
    "<|system|>\n{% if tools is defined and tools is not none %}Environment: ipython\n"
    "{% for t in tools %}{{ t | tojson }}\n{% endfor %}{% endif %}"
    "{% for m in messages %}"
    "{% if m.tool_calls is defined %}<|assistant|>\n{{ m.tool_calls[0].function.name }}\n"
    "{% elif m.role == 'tool' %}<|ipython|>\n{{ m.content }}\n"
    "{% else %}<|{{ m.role }}|>\n{{ m.content }}\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)
WORDS = ["red", "blue", "green", "cat", "dog"]


def fake_items(dataset, split=None, limit=None, stats=None):
    n_options = {"csqa": 5, "arc_challenge": 4, "obqa": 4}[dataset]
    out = []
    for i in range(4):
        options = tuple((letter, WORDS[k]) for k, letter in enumerate("ABCDE"[:n_options]))
        out.append(Item(item_id=f"{dataset}:{split}:{i}", dataset=dataset, split=split,
                        question=f"Which one is it {i}?", options=options, answer="ABCD"[i % 4]))
    if stats is not None:
        stats.update({"rows": len(out), "kept": len(out), "skipped": 0})
    return out


@pytest.fixture
def tiny(monkeypatch):
    tok = tiny_tokenizer(chat_template=TOOL_TEMPLATE)
    model = tiny_model(tok)
    monkeypatch.setattr(I, "load_items", fake_items)
    monkeypatch.setattr(script, "load_model_and_tokenizer", lambda hf_id, dtype, device: (model, tok))
    return model, tok


def argv(tmp_path, *extra):
    return ["--checkpoint", "meta_instruct", "--out-dir", str(tmp_path / "runs"),
            "--results-dir", str(tmp_path / "results" / "meta_instruct"),
            "--log-file", str(tmp_path / "logs" / "run.log"), "--device", "cpu", "--batch-size", "4",
            "--bootstrap-samples", "20", "--progress-every-seconds", "0.5",
            "--ignore-preflight-problems", *extra]   # an untrained tiny model has low letter mass


@pytest.mark.filterwarnings("ignore")
def test_end_to_end_on_tiny_model(tiny, tmp_path):
    assert script.main(argv(tmp_path)) == 0
    results = tmp_path / "results" / "meta_instruct"

    split_path = results / "question_split.json"
    split = json.loads(split_path.read_text())
    assert split["seed"] == 0 and split["counts"]["test_total"] == 2 + 2 + 4
    for d in ("csqa", "arc_challenge"):
        assert len(split[d]["pool"]) == len(split[d]["test"]) == 2
        assert not set(split[d]["pool"]) & set(split[d]["test"])
    assert split["obqa"]["pool"] == [] and len(split["obqa"]["test"]) == 4

    result = json.loads((results / "compare_to_paper.json").read_text())
    table = result["table"]
    assert len(table) == 4 * 3   # 4 Table D1 conditions x (user, 12-source pool, 13-source pool)
    assert {r["column"] for r in table} == {"user", "reliable_source", "reliable_source_no_tool"}
    assert result["tool_mode_used"] == "native"
    assert {r["role"] for r in table if r["column"] == "reliable_source"} == {"primary"}
    assert set(result["summary"]) >= {"all_primary_rows_pass", "rows_compared", "status_counts"}
    assert set(result["answer_reading_by_dataset"]) == {"csqa", "arc_challenge", "obqa"}
    assert result["scoring_counts"]["items"] == 8
    rows = (tmp_path / "runs" / "scores.jsonl").read_text().splitlines()
    assert len(rows) == script.expected_row_count(8, 14, ["single", "multi"]) == 8 * 58
    preflight = json.loads((results / "preflight.json").read_text())
    assert "tool_row_examples" in preflight and "Environment: ipython" in next(
        iter(preflight["tool_row_examples"].values()))["text"]
    assert (results / "all_cells.json").exists() and (tmp_path / "logs" / "run.log").exists()

    # rerun: scoring is finished, so only the comparison is recomputed; the split file is unchanged
    before = split_path.read_bytes()
    assert script.main(argv(tmp_path)) == 0
    assert split_path.read_bytes() == before
    assert len((tmp_path / "runs" / "scores.jsonl").read_text().splitlines()) == len(rows)

    # a changed split is refused before anything is scored
    split["csqa"]["test"] = split["csqa"]["test"][::-1][:1]
    split_path.write_text(json.dumps(split))
    assert script.main(argv(tmp_path)) == 1
    assert "different question split" in (tmp_path / "logs" / "run.log").read_text()


@pytest.mark.filterwarnings("ignore")
def test_preflight_problem_stops_before_scoring(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(PF, "preflight", lambda *a, **k: {"problems": ["fake problem",
                                                                       "no pad token: harmless"]})
    args = [a for a in argv(tmp_path) if a != "--ignore-preflight-problems"]
    assert script.main(args) == 2
    report = json.loads((tmp_path / "results" / "meta_instruct" / "preflight.json").read_text())
    assert report["serious_problems"] == ["fake problem"]
    assert report["harmless_problems"] == ["no pad token: harmless"]
    assert not (tmp_path / "runs" / "scores.jsonl").exists()
    assert (tmp_path / "results" / "meta_instruct" / "question_split.json").exists()


def test_tool_mode_support_detection():
    def explicit(model, *, tool_mode="text"):
        pass

    def kwargs_only(model, **prompt_kwargs):
        pass

    def build_with(item, tool_mode="text"):
        pass

    def build_without(item):
        pass

    assert script.tool_mode_support(explicit, build_without) == "run_checkpoint"
    assert script.tool_mode_support(kwargs_only, build_with) == "build_prompt"
    assert script.tool_mode_support(kwargs_only, build_without) is None


def test_native_tool_mode_needs_native_format():
    with pytest.raises(SystemExit):
        script.parse_args(["--fmt", "plain", "--tool-mode", "native"])


def test_letter_spellings_default_counts_paren_letters():
    # Llama-3.1-8B-Instruct answers Prompt C1 with "(B" as one token (decision D1 amendment)
    args = script.parse_args([])
    assert script.letter_spellings(args) == ("{L}", " {L}", "({L}")
    assert script.letter_spellings(script.parse_args(["--letter-spellings", "{L}, {L}"])) == ("{L}", " {L}")
