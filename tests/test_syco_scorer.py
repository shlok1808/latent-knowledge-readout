"""Scorer tests on the offline tiny Llama (tests/syco_tiny.py). CPU, no network."""
import json
import math

import numpy as np
import pytest
import torch

from failure_probes.syco import scorer
from failure_probes.syco.prompts import build_prompt
from failure_probes.syco.types import Cue, Item, Prompt, Score
from syco_tiny import tiny_model, tiny_tokenizer

NEUTRAL = Cue("neutral", "none", "single", None, None)


def make_item(i, n_options=5, answer="B", dataset="csqa"):
    letters = "ABCDE"[:n_options]
    words = ["red", "blue", "green", "cat", "dog"]
    options = tuple((letter, words[k]) for k, letter in enumerate(letters))
    return Item(item_id=f"{dataset}:validation:{i}", dataset=dataset, split="validation",
                question=f"Which one is it {i}?" + " What" * (i % 4), options=options, answer=answer)


ITEMS = [make_item(0), make_item(1, 3, "A", "arc_challenge"), make_item(2, 4, "D"), make_item(3), make_item(4, 5, "E", "obqa")]


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return tiny_model(tok)


def prompts_for(tok, fmt):
    out = []
    for item in ITEMS:
        out.append(build_prompt(item, NEUTRAL, fmt, tokenizer=tok))
        sug = Cue("suggestion", "user", "multi", item.letters[-1], "wrong")
        out.append(build_prompt(item, sug, fmt, tokenizer=tok, neutral_answer="A"))
    return out


# ----------------------------------------------------------------- letter ids


def test_letter_token_ids_bare_and_space_variants(tok):
    ids = scorer.letter_token_ids(tok, "ABCDE")
    for letter in "ABCDE":
        assert ids[letter] == [tok.convert_tokens_to_ids(letter), tok.convert_tokens_to_ids(" " + letter)]
    flat = [i for v in ids.values() for i in v]
    assert len(flat) == len(set(flat)) == 10


def test_letter_token_ids_skips_multi_token_and_unk(tok):
    # " Z" is two tokens, so only "Z" counts.
    assert scorer.letter_token_ids(tok, "Z") == {"Z": [tok.convert_tokens_to_ids("Z")]}
    # "é" is <unk>, " é" is two tokens: no usable variant.
    with pytest.raises(ValueError, match="no single-token variant"):
        scorer.letter_token_ids(tok, "é")


def test_letter_token_ids_raises_on_shared_id():
    class Collide:
        unk_token_id = None

        def encode(self, text, add_special_tokens=False):
            return [7]

    with pytest.raises(ValueError, match="share token id"):
        scorer.letter_token_ids(Collide(), "AB")


# ----------------------------------------------------------------- distribution


def test_letter_distribution_full_vocab_softmax_sum_and_renorm():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=50) * 3
    ids = {"A": [3, 10], "B": [4], "C": [5, 11]}
    dist = scorer.letter_distribution(logits, ids)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    raw = {L: p[v].sum() for L, v in ids.items()}
    mass = sum(raw.values())
    assert dist.letter_mass == pytest.approx(mass, rel=1e-12)
    for L in ids:
        assert dist.logprobs[L] == pytest.approx(math.log(raw[L]), rel=1e-12)
        assert dist.probs[L] == pytest.approx(raw[L] / mass, rel=1e-12)
    assert sum(dist.probs.values()) == pytest.approx(1.0, abs=1e-12)
    assert list(dist.probs) == ["A", "B", "C"]
    # tuple unpacking order matches Score field order (probs, letter_mass, logprobs)
    probs, letter_mass, logprobs = dist
    assert probs is dist.probs and logprobs is dist.logprobs


def test_letter_distribution_torch_matches_numpy_and_upcasts_bf16():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=300).astype(np.float32) * 5
    ids = {"A": [1, 2], "B": [3, 4]}
    ref = scorer.letter_distribution(logits, ids)
    t32 = scorer.letter_distribution(torch.tensor(logits), ids)
    for L in ids:
        assert t32.probs[L] == pytest.approx(ref.probs[L], rel=1e-9)
        assert t32.logprobs[L] == pytest.approx(ref.logprobs[L], rel=1e-9)
    bf = torch.tensor(logits).to(torch.bfloat16)
    tbf = scorer.letter_distribution(bf, ids)
    refbf = scorer.letter_distribution(bf.float().numpy(), ids)  # same rounded logits, float64 math
    assert tbf.letter_mass == pytest.approx(refbf.letter_mass, rel=1e-9)
    assert all(isinstance(v, float) for v in tbf.probs.values())


def test_letter_distribution_no_underflow_for_tiny_letter_mass():
    logits = np.zeros(10)
    logits[0] = 800.0  # letters get ~e^-800 each: float32 exp would underflow to 0
    dist = scorer.letter_distribution(logits, {"A": [1], "B": [2]})
    assert dist.probs == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}
    assert math.isfinite(dist.logprobs["A"]) and dist.logprobs["A"] == pytest.approx(-800.0)


# ----------------------------------------------------------------- scoring


@pytest.mark.parametrize("fmt", ["plain", "native"])
def test_batched_equals_unbatched(model, tok, fmt):
    prompts = prompts_for(tok, fmt)
    lengths = {len(scorer.encode_prompt(tok, p)) for p in prompts}
    assert len(lengths) > 3  # the batch really needs padding
    one = scorer.score_prompts(model, tok, prompts, "tiny", batch_size=1)
    many = scorer.score_prompts(model, tok, prompts, "tiny", batch_size=4)
    for a, b, p in zip(one, many, prompts):
        assert (a.item_id, a.cue) == (b.item_id, b.cue) == (p.item_id, p.cue)
        assert set(a.probs) == set(p.letters)
        for L in p.letters:
            assert a.probs[L] == pytest.approx(b.probs[L], abs=1e-4)
            assert a.logprobs[L] == pytest.approx(b.logprobs[L], abs=1e-4)
        assert a.letter_mass == pytest.approx(b.letter_mass, abs=1e-4)


def test_matches_unpadded_forward(model, tok):
    # Left-padded batch == each prompt alone with no padding at all.
    prompts = prompts_for(tok, "plain")
    scores = scorer.score_prompts(model, tok, prompts, "tiny", batch_size=len(prompts))
    ids_by_letter = scorer.letter_token_ids(tok, "ABCDE")
    for p, s in zip(prompts, scores):
        ids = torch.tensor([scorer.encode_prompt(tok, p)])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1]
        ref = scorer.letter_distribution(logits, {L: ids_by_letter[L] for L in p.letters})
        for L in p.letters:
            assert s.probs[L] == pytest.approx(ref.probs[L], abs=1e-4)



def test_padding_side_and_pad_token_restored():
    tok = tiny_tokenizer(pad=False)
    model = tiny_model(tok)
    tok.padding_side = "right"
    assert tok.pad_token is None
    prompts = prompts_for(tok, "plain")[:4]
    scores = scorer.score_prompts(model, tok, prompts, "tiny", batch_size=4)
    assert len(scores) == 4
    assert tok.padding_side == "right" and tok.pad_token is None
    ref = scorer.score_prompts(model, tok, prompts, "tiny", batch_size=1)
    for a, b in zip(scores, ref):
        assert a.probs == pytest.approx(b.probs, abs=1e-4)


def test_bos_rule_by_format(tok):
    item = ITEMS[0]
    plain = build_prompt(item, NEUTRAL, "plain", tokenizer=tok)
    native = build_prompt(item, NEUTRAL, "native", tokenizer=tok)
    assert scorer.needs_special_tokens("plain") is True
    assert scorer.needs_special_tokens("native") is False
    for p in (plain, native):
        ids = scorer.encode_prompt(tok, p)
        assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id
    # a native-rendered text mislabelled as plain would get a second BOS
    mislabelled = Prompt(item.item_id, NEUTRAL, native.messages, native.text, "plain", item.letters)
    with pytest.raises(ValueError, match="double BOS"):
        scorer.encode_prompt(tok, mislabelled)
    with pytest.raises(ValueError, match="double BOS"):
        scorer.score_prompts(tiny_model(tok), tok, [mislabelled], "tiny")


def test_score_extra_and_types(model, tok):
    (s,) = scorer.score_prompts(model, tok, prompts_for(tok, "plain")[:1], "ckpt")
    assert isinstance(s, Score) and s.checkpoint == "ckpt" and s.fmt == "plain"
    assert 0 < s.letter_mass <= 1
    assert sum(s.probs.values()) == pytest.approx(1.0)
    assert set(s.extra) == {"n_tokens", "top_token_id"}
    json.dumps(scorer.score_row(s, ITEMS[0], "A"))


# ----------------------------------------------------------------- labels


def test_neutral_labels():
    items = ITEMS[:2]
    scores = [
        Score(items[0].item_id, NEUTRAL, "plain", "c", {"A": .1, "B": .6, "C": .1, "D": .1, "E": .1}, .9, {}),
        Score(items[1].item_id, NEUTRAL, "plain", "c", {"A": .2, "B": .4, "C": .4}, .9, {}),  # tie -> B
        Score(items[1].item_id, Cue("suggestion", "user", "single", "A", "wrong"), "plain", "c",
              {"A": 1.0, "B": 0.0, "C": 0.0}, .9, {}),  # non-neutral: ignored
    ]
    assert scorer.neutral_labels(scores, items) == {items[0].item_id: ("B", "correct"), items[1].item_id: ("B", "wrong")}
    with pytest.raises(ValueError, match="two neutral"):
        scorer.neutral_labels(scores + scores[:1], items)


# ----------------------------------------------------------------- orchestration


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_run_checkpoint_end_to_end(model, tok, tmp_path):
    from failure_probes.syco.items import wrong_option

    out = tmp_path / "rows.jsonl"
    manifest = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single", "multi"],
                                     out, seed=3, batch_size=3)
    rows = read_rows(out)
    per_item = 1 + 2 * 1 * 2 + 1  # neutral + (suggestion, doubt) x sources x turns + answer_again
    assert len(rows) == per_item * len(ITEMS)
    neutral = {r["item_id"]: r for r in rows if r["kind"] == "neutral"}
    assert len(neutral) == len(ITEMS)
    by_id = {i.item_id: i for i in ITEMS}
    for r in rows:
        item = by_id[r["item_id"]]
        assert {"item_id", "dataset", "checkpoint", "fmt", "kind", "source", "turn", "target", "prior",
                "neutral_answer", "gold", "probs", "letter_mass", "logprobs", "extra"} <= set(r)
        assert r["dataset"] == item.dataset and r["gold"] == item.answer
        assert r["checkpoint"] == "tiny" and r["fmt"] == "plain"
        n = neutral[r["item_id"]]
        answer = max(item.letters, key=lambda L: (n["probs"][L], -item.letters.index(L)))
        assert r["neutral_answer"] == answer
        prior = "correct" if answer == item.answer else "wrong"
        if r["kind"] == "neutral":
            assert r["prior"] is None and r["target"] is None and r["turn"] == "single"
            continue
        assert r["prior"] == prior
        if r["kind"] == "suggestion":
            assert r["target"] == (wrong_option(item, 3) if prior == "correct" else item.answer)
        elif r["kind"] == "doubt":
            assert r["target"] == answer
        else:
            assert r["kind"] == "answer_again" and r["turn"] == "multi" and r["target"] is None

    path = scorer.manifest_path(out, "tiny", "plain")
    assert path.exists() and json.loads(path.read_text()) == manifest
    c = manifest["counts"]
    assert c["items"] == len(ITEMS) and c["neutral_rows"] == len(ITEMS)
    assert c["cue_rows"] == (per_item - 1) * len(ITEMS)
    assert c["prior_correct"] + c["prior_wrong"] == len(ITEMS)
    assert manifest["seed"] == 3 and manifest["sources"] == ["user"] and "commit" in manifest["git"]

    # same (checkpoint, fmt) again into the same file refuses; another fmt may share the file
    with pytest.raises(FileExistsError):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single"], out, seed=3)
    scorer.run_checkpoint(model, tok, ITEMS, "tiny", "native", ["user"], ["single"], out, seed=3)
    assert len(read_rows(out)) == per_item * len(ITEMS) + 3 * len(ITEMS)


def test_run_checkpoint_neutral_rows_match_score_prompts(model, tok, tmp_path):
    out = tmp_path / "rows.jsonl"
    scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single"], out, seed=0, batch_size=2)
    direct = scorer.score_prompts(model, tok, [build_prompt(i, NEUTRAL, "plain") for i in ITEMS], "tiny")
    rows = {r["item_id"]: r for r in read_rows(out) if r["kind"] == "neutral"}
    for s in direct:
        assert rows[s.item_id]["probs"] == pytest.approx(s.probs, abs=1e-6)


def test_run_checkpoint_fails_fast_without_writing(tmp_path):
    tok = tiny_tokenizer(chat_template=False)
    model = tiny_model(tok)
    out = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="chat_template"):
        scorer.run_checkpoint(model, tok, ITEMS, "base", "native", ["user"], ["single"], out, seed=0)
    assert not out.exists()
    with pytest.raises(ValueError, match="unknown source"):
        scorer.run_checkpoint(model, tok, ITEMS, "base", "plain", ["nope"], ["single"], out, seed=0)
    assert not out.exists()
