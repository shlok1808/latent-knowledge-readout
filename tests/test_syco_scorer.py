"""Scorer tests on the offline tiny Llama (tests/syco_tiny.py). CPU, no network."""
import hashlib
import json
import math
import types

import numpy as np
import pytest
import torch

from failure_probes.syco import scorer
from failure_probes.syco.prompts import build_prompt
from failure_probes.syco.types import Cue, Item, Prompt, Score
from syco_tiny import CHAT_TEMPLATE_DATED, CHAT_TEMPLATE_NO_BOS, tiny_model, tiny_tokenizer

NEUTRAL = Cue("neutral", "none", "single", None, None)
# run_checkpoint warns when a test checkpoint name ("tiny", ...) is not in configs/syco_checkpoints.json
pytestmark = pytest.mark.filterwarnings("ignore:checkpoint .* is not in syco_checkpoints.json")


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
    # needs_special_tokens is kept for compatibility; encode_prompt decides from the text instead.
    assert scorer.needs_special_tokens("plain") is True
    assert scorer.needs_special_tokens("native") is False
    item = ITEMS[0]
    plain = build_prompt(item, NEUTRAL, "plain", tokenizer=tok)
    native = build_prompt(item, NEUTRAL, "native", tokenizer=tok)
    for p in (plain, native):
        ids = scorer.encode_prompt(tok, p)
        assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    # a native-rendered text mislabelled as plain still gets exactly one BOS (rule is text-based)
    mislabelled = Prompt(item.item_id, NEUTRAL, native.messages, native.text, "plain", item.letters)
    assert scorer.encode_prompt(tok, mislabelled) == scorer.encode_prompt(tok, native)
    # a text that already holds two BOS raises, in encode and in scoring
    doubled = Prompt(item.item_id, NEUTRAL, native.messages, tok.bos_token + native.text, "native", item.letters)
    with pytest.raises(ValueError, match="exactly one BOS"):
        scorer.encode_prompt(tok, doubled)
    with pytest.raises(ValueError, match="exactly one BOS"):
        scorer.score_prompts(tiny_model(tok), tok, [doubled], "tiny")


TOKENIZER_KINDS = {
    "template_bos": dict(),                                           # Llama-3-like
    "template_no_bos": dict(chat_template=CHAT_TEMPLATE_NO_BOS),      # Tulu-3-like
    "no_bos_anywhere": dict(chat_template=CHAT_TEMPLATE_NO_BOS, add_bos=False),  # scorer prepends
}


@pytest.mark.parametrize("kind", list(TOKENIZER_KINDS))
@pytest.mark.parametrize("fmt", ["plain", "native"])
def test_exactly_one_bos_whatever_the_template(kind, fmt):
    t = tiny_tokenizer(**TOKENIZER_KINDS[kind])
    m = tiny_model(t)
    prompts = prompts_for(t, fmt)
    expect_text_bos = fmt == "native" and kind == "template_bos"
    for p in prompts:
        assert p.text.startswith("<s>") == expect_text_bos
        ids = scorer.encode_prompt(t, p)
        assert ids[0] == t.bos_token_id and ids.count(t.bos_token_id) == 1
        # body identical to the text's own tokens: BOS added once, nothing else changed
        body = t.encode(p.text, add_special_tokens=False)
        assert ids == (body if expect_text_bos else [t.bos_token_id] + body)
    scores = scorer.score_prompts(m, t, prompts, "tiny", batch_size=3)
    assert all(s.extra["has_template_bos"] is expect_text_bos for s in scores)


def test_score_extra_and_types(model, tok):
    (s,) = scorer.score_prompts(model, tok, prompts_for(tok, "plain")[:1], "ckpt")
    assert isinstance(s, Score) and s.checkpoint == "ckpt" and s.fmt == "plain"
    assert 0 < s.letter_mass <= 1
    assert sum(s.probs.values()) == pytest.approx(1.0)
    assert set(s.extra) == {"n_tokens", "top_token_id", "has_template_bos", "prompt_sha256", "greedy"}
    g = s.extra["greedy"]
    assert g["top_token_id"] == s.extra["top_token_id"]
    letter_ids = scorer.letter_token_ids(tok, ITEMS[0].letters)
    expect = next((L for L, ids in letter_ids.items() if g["top_token_id"] in ids), None)
    assert g["top_token_letter"] == expect and g["top_token_is_letter"] is (expect is not None)
    assert s.extra["prompt_sha256"] == scorer.prompt_sha256(prompts_for(tok, "plain")[0])
    assert len(s.extra["prompt_sha256"]) == 64
    json.dumps(scorer.score_row(s, ITEMS[0], "A"))


def test_prompt_sha256_fallback_is_text_hash():
    from failure_probes.syco import prompts as P

    p = build_prompt(ITEMS[0], NEUTRAL, "plain")
    if not hasattr(P, "prompt_sha256"):
        assert scorer.prompt_sha256(p) == hashlib.sha256(p.text.encode("utf-8")).hexdigest()
    other = build_prompt(ITEMS[1], NEUTRAL, "plain")
    assert scorer.prompt_sha256(p) != scorer.prompt_sha256(other)


def test_letter_variants_configurable(model, tok):
    prompts = prompts_for(tok, "plain")[:4]
    bare = scorer.score_prompts(model, tok, prompts, "tiny", variants=("{L}",))
    both = scorer.score_prompts(model, tok, prompts, "tiny")
    for b, s in zip(bare, both):
        assert b.letter_mass < s.letter_mass
    ids = scorer.letter_token_ids(tok, "AB", ("{L}",))
    assert ids == {"A": [tok.convert_tokens_to_ids("A")], "B": [tok.convert_tokens_to_ids("B")]}
    with pytest.raises(ValueError, match="placeholder"):
        scorer.letter_token_ids(tok, "AB", ("X",))
    with pytest.raises(TypeError):
        scorer.letter_token_ids(tok, "AB", "{L}")


def test_neutral_diagnostics(model, tok):
    prompts = [build_prompt(i, NEUTRAL, "plain") for i in ITEMS]
    scores = scorer.score_prompts(model, tok, prompts, "tiny")
    ids = scorer.letter_token_ids(tok, "ABCDE")
    d = scorer.neutral_diagnostics(scores, tok, letter_ids=ids, min_mass=0.5, top_k=2)
    assert d["n"] == len(ITEMS) and len(d["top_tokens"]) <= 2
    assert sum(h["count"] for h in scorer.neutral_diagnostics(scores, tok)["top_tokens"]) == len(ITEMS)
    masses = [s.letter_mass for s in scores]
    assert d["letter_mass_mean"] == pytest.approx(np.mean(masses))
    assert d["letter_mass_min"] == pytest.approx(min(masses))
    assert d["frac_letter_mass_below_min"] == pytest.approx(np.mean([m < 0.5 for m in masses]))
    assert 0 <= d["frac_top_token_is_letter"] <= 1
    top = d["top_tokens"][0]
    assert top["token"] == tok.decode([top["token_id"]])
    json.dumps(d)


# ----------------------------------------------------------------- labels


def test_neutral_labels():
    items = ITEMS[:2]
    scores = [
        Score(items[0].item_id, NEUTRAL, "plain", "c", {"A": .1, "B": .6, "C": .1, "D": .1, "E": .1}, .9, {}),
        Score(items[1].item_id, NEUTRAL, "plain", "c", {"A": .2, "B": .4, "C": .4}, .9, {}),  # tie -> B
        Score(items[1].item_id, Cue("suggestion", "user", "single", "A", "wrong"), "plain", "c",
              {"A": 1.0, "B": 0.0, "C": 0.0}, .9, {}),  # non-neutral: ignored
    ]
    expect = {items[0].item_id: ("B", "correct"), items[1].item_id: ("B", "wrong")}
    assert scorer.neutral_labels(scores, items, prior_rule="letter_argmax") == expect
    with pytest.raises(ValueError, match="two neutral"):
        scorer.neutral_labels(scores + scores[:1], items, prior_rule="letter_argmax")
    with pytest.raises(ValueError, match="no extra\\['greedy'\\]"):  # greedy rule needs the top token
        scorer.neutral_labels(scores, items)
    with pytest.raises(ValueError, match="unknown prior_rule"):
        scorer.neutral_labels(scores, items, prior_rule="greedy")


def greedy(top_variant_letter, top_token_letter=None):
    return {"greedy": {"top_letter_variant_letter": top_variant_letter, "top_letter_variant_token_id": 0,
                       "top_letter_variant_prob": 0.3, "top_token_id": 0, "top_token_letter": top_token_letter,
                       "top_token_is_letter": top_token_letter is not None}}


def test_neutral_labels_greedy_rule():
    items = ITEMS[:2]
    probs = {"A": .45, "B": .40, "C": .15}
    scores = [
        # single most likely letter token is "B" though A's summed variants are larger: the label is B
        Score(items[0].item_id, NEUTRAL, "plain", "c", dict(probs, D=0, E=0), .9, {}, greedy("B", "B")),
        # full-vocab top token not a letter: the label is still the top letter-variant token, no fallback
        Score(items[1].item_id, NEUTRAL, "plain", "c", probs, .9, {}, greedy("B", None)),
    ]
    labels = scorer.neutral_labels(scores, items)
    assert labels == {items[0].item_id: ("B", "correct"), items[1].item_id: ("B", "wrong")}
    assert all("fallback" not in s.extra["greedy"] for s in scores)
    assert scores[0].extra["labels_by_rule"] == {"greedy_token": "B", "letter_argmax": "A"}
    assert scorer.neutral_labels(scores, items, prior_rule="letter_argmax")[items[0].item_id] == ("A", "wrong")
    assert scorer.answer_label({"item_id": "x", "probs": probs, "extra": greedy("C")}, "ABC") == "C"
    with pytest.raises(ValueError, match="not among"):
        scorer.answer_label({"item_id": "x", "probs": probs, "extra": greedy("E")}, "ABC")


def test_greedy_info_top_letter_variant():
    ids = {"A": [10, 20], "B": [11, 21]}
    flat = [10, 20, 11, 21]
    lp = np.log([0.02, 0.30, 0.25, 0.25])
    g = scorer.greedy_info(99, ids, flat, lp)
    assert g["top_letter_variant_letter"] == "A" and g["top_letter_variant_token_id"] == 20
    assert g["top_letter_variant_prob"] == pytest.approx(0.30)
    assert g["top_token_letter"] is None and g["top_token_is_letter"] is False
    # tie between id 20 (A) and id 11 (B): the LOWEST token id wins, as torch.argmax / greedy does
    tie = scorer.greedy_info(11, ids, flat, np.log([0.1, 0.3, 0.3, 0.1]))
    assert tie["top_letter_variant_letter"] == "B" and tie["top_letter_variant_token_id"] == 11
    assert tie["top_token_letter"] == "B"


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
        answer = n["extra"]["greedy"]["top_letter_variant_letter"]  # default prior_rule="greedy_token"
        top = n["extra"]["greedy"]["top_token_letter"]
        if top is not None:  # full-vocab top token is a letter: the label is that letter
            assert answer == top
        assert r["neutral_answer"] == answer and r["prior_rule"] == "greedy_token"
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
    assert manifest["prior_rule"] == "greedy_token"
    assert "greedy_fallbacks" not in c
    assert c["greedy_top_token_not_letter"] == sum(not n["extra"]["greedy"]["top_token_is_letter"]
                                                   for n in neutral.values())
    for r in rows:  # restricted greedy == full-vocab greedy whenever the top token is a letter
        g = r["extra"]["greedy"]
        if g["top_token_is_letter"]:
            assert g["top_letter_variant_letter"] == g["top_token_letter"]
            assert g["top_letter_variant_token_id"] == g["top_token_id"]
    assert c["prior_rule_disagreements"] == sum(
        n["extra"]["labels_by_rule"]["greedy_token"] != n["extra"]["labels_by_rule"]["letter_argmax"]
        for n in neutral.values())

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


# ----------------------------------------------------------------- manifest, crash safety, resume

ALL_EXAMPLE_KEYS = {"neutral|single", "suggestion|single", "doubt|single", "suggestion|multi", "doubt|multi",
                    "answer_again|multi"}


def test_manifest_reproducibility_fields(model, tok, tmp_path):
    from failure_probes.syco import prompts as P

    out = tmp_path / "rows.jsonl"
    m = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single", "multi"], out, seed=1)
    assert set(m["example_prompts"]) == ALL_EXAMPLE_KEYS
    ex = m["example_prompts"]["neutral|single"]
    assert ex["item_id"] == ITEMS[0].item_id and ex["text"] == build_prompt(ITEMS[0], NEUTRAL, "plain").text
    assert all(e["item_id"] == ITEMS[0].item_id and e["text"] for e in m["example_prompts"].values())
    assert m["vocab"] == {"model_config_vocab_size": model.config.vocab_size, "len_tokenizer": len(tok), "equal": True}
    assert m["date_string"] == scorer._date_string_default(P) and m["date_string_applies"] is False
    assert m["native_template_has_bos"] is None and m["letter_variants"] == ["{L}", " {L}"]
    d = m["neutral_diagnostics"]
    assert d["n"] == len(ITEMS) and d["min_mass"] == 0.5 and "frac_letter_mass_below_min" in d
    assert m["counts"]["neutral_letter_mass_min"] == d["letter_mass_min"]
    rows = read_rows(out)
    by_key = {scorer.row_key(r): r for r in rows}
    for item in ITEMS:
        r = by_key[(item.item_id, "neutral", "none", "single")]
        assert r["extra"]["prompt_sha256"] == scorer.prompt_sha256(build_prompt(item, NEUTRAL, "plain"))
        assert r["extra"]["has_template_bos"] is False
    assert not scorer.inprogress_path(out, "tiny", "plain").exists()


@pytest.mark.parametrize("template,has_bos", [(None, True), (CHAT_TEMPLATE_NO_BOS, False)])
def test_manifest_native_template_has_bos(template, has_bos, tmp_path):
    t = tiny_tokenizer(**({} if template is None else {"chat_template": template}))
    m = scorer.run_checkpoint(tiny_model(t), t, ITEMS, "tiny", "native", ["user"], ["single"],
                              tmp_path / "r.jsonl", seed=0)
    assert m["native_template_has_bos"] is has_bos
    assert all(r["extra"]["has_template_bos"] is has_bos for r in read_rows(tmp_path / "r.jsonl"))


def test_manifest_records_today_date_line(tmp_path):
    t = tiny_tokenizer(chat_template=CHAT_TEMPLATE_DATED)
    m = scorer.run_checkpoint(tiny_model(t), t, ITEMS, "tiny", "native", ["user"], ["single"],
                              tmp_path / "r.jsonl", seed=0)
    assert m["date_string_applies"] is True
    assert m["native_today_date_line"].startswith("Today Date: ")
    if m["date_string"] is not None:  # prompts.py passes its own default date_string
        assert m["native_today_date_line"] == f"Today Date: {m['date_string']}"
    assert m["example_prompts"]["neutral|single"]["text"].startswith("<s><|system|>\nToday Date: ")
    m2 = scorer.run_checkpoint(tiny_model(t), t, ITEMS, "tiny2", "native", ["user"], ["single"],
                               tmp_path / "r.jsonl", seed=0, date_string="01 Jan 2025")
    assert m2["date_string"] == "01 Jan 2025" and m2["native_today_date_line"] == "Today Date: 01 Jan 2025"


def test_run_checkpoint_custom_variants(model, tok, tmp_path):
    out = tmp_path / "rows.jsonl"
    m = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single"], out, seed=0,
                              variants=("{L}",))
    assert m["letter_variants"] == ["{L}"]
    assert m["letter_token_ids"]["ABCDE"]["A"] == [tok.convert_tokens_to_ids("A")]
    with pytest.raises(ValueError, match="no single-token variant"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "native", ["user"], ["single"], out, seed=0,
                              variants=("({L}",))
    assert not scorer.inprogress_path(out, "tiny", "native").exists()


def test_checkpoint_config_refuses_native(model, tok, tmp_path):
    entry = scorer.checkpoint_entry("base")
    assert entry["has_chat_template"] is False and scorer.checkpoint_entry(entry["hf_id"]) == entry
    out = tmp_path / "rows.jsonl"
    assert tok.chat_template is not None  # the tokenizer ships a template, the config still refuses
    with pytest.raises(ValueError, match="has_chat_template"):
        scorer.run_checkpoint(model, tok, ITEMS, "base", "native", ["user"], ["single"], out, seed=0,
                              checkpoint_config=entry)
    assert not out.exists()
    m = scorer.run_checkpoint(model, tok, ITEMS, "base", "plain", ["user"], ["single"], out, seed=0,
                              checkpoint_config=entry)
    assert m["hf_id"] == entry["hf_id"] and m["checkpoint_config"] == entry
    tulu = scorer.checkpoint_entry("tulu_sft")
    scorer.run_checkpoint(model, tok, ITEMS, "tulu_sft", "native", ["user"], ["single"], out, seed=0,
                          checkpoint_config=tulu)


class Crash(RuntimeError):
    pass


def crash_after(monkeypatch, n_rows, marker):
    real = scorer.append_jsonl
    state = {"n": 0}

    def flaky(path, value):
        assert marker.exists()  # the marker is on disk before the first row
        if state["n"] >= n_rows:
            raise Crash("simulated crash")
        state["n"] += 1
        real(path, value)

    monkeypatch.setattr(scorer, "append_jsonl", flaky)


RUN = dict(sources=["user"], turns=["single", "multi"], seed=3, batch_size=2)


@pytest.mark.parametrize("n_rows", [0, 2, 9])  # crash before any row, inside neutral rows, inside cue rows
def test_crash_then_resume_matches_clean_run(model, tok, tmp_path, monkeypatch, n_rows):
    clean_out = tmp_path / "clean.jsonl"
    clean = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=clean_out, **RUN)

    out = tmp_path / "rows.jsonl"
    marker = scorer.inprogress_path(out, "tiny", "plain")
    crash_after(monkeypatch, n_rows, marker)
    with pytest.raises(Crash):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, **RUN)
    monkeypatch.undo()
    assert marker.exists() and not scorer.manifest_path(out, "tiny", "plain").exists()
    assert len(scorer.read_rows(out)) == n_rows

    with pytest.raises(FileExistsError):  # no silent re-run over a crashed one
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, **RUN)
    with pytest.raises(ValueError, match="resume settings differ"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True,
                              **{**RUN, "seed": 4})

    m = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)
    assert not marker.exists() and scorer.manifest_path(out, "tiny", "plain").exists()
    rows, ref = read_rows(out), read_rows(clean_out)
    keys = [scorer.row_key(r) for r in rows]
    assert len(keys) == len(set(keys)) == len(ref)
    ref_by_key = {scorer.row_key(r): r for r in ref}
    for r in rows:
        c = ref_by_key[scorer.row_key(r)]
        assert (r["target"], r["prior"], r["neutral_answer"]) == (c["target"], c["prior"], c["neutral_answer"])
        assert r["probs"] == pytest.approx(c["probs"], abs=1e-4)
        assert r["extra"]["prompt_sha256"] == c["extra"]["prompt_sha256"]
    n_neutral = min(n_rows, len(ITEMS))
    assert m["rows_reused"] == {"neutral": n_neutral, "cue": n_rows - n_neutral}
    floats = ("neutral_letter_mass_mean", "neutral_letter_mass_min")
    assert {k: v for k, v in m["counts"].items() if k not in floats} == \
        {k: v for k, v in clean["counts"].items() if k not in floats}
    for k in floats:
        assert m["counts"][k] == pytest.approx(clean["counts"][k], abs=1e-6)
    assert m["resumed"] is True and clean["resumed"] is False
    assert m["first_started"] <= m["started"]

    with pytest.raises(FileExistsError, match="complete"):  # a finished run is not resumed
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)


def test_resume_refuses_changed_prompt_and_truncated_line(model, tok, tmp_path, monkeypatch):
    out = tmp_path / "rows.jsonl"
    crash_after(monkeypatch, 7, scorer.inprogress_path(out, "tiny", "plain"))
    with pytest.raises(Crash):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, **RUN)
    monkeypatch.undo()
    lines = out.read_text().splitlines()

    tampered = json.loads(lines[6])
    tampered["extra"]["prompt_sha256"] = "0" * 64
    out.write_text("\n".join(lines[:6] + [json.dumps(tampered)]) + "\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)

    out.write_text("\n".join(lines) + "\n" + lines[6][: len(lines[6]) // 2])  # cut-off last write
    with pytest.raises(ValueError, match="cut off by a crash"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)

    out.write_text("\n".join(lines + [lines[6]]) + "\n")  # duplicate key
    with pytest.raises(ValueError, match="duplicate row"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)

    out.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="item_ids_sha256"):  # other items than the crashed run
        scorer.run_checkpoint(model, tok, ITEMS[1:], "tiny", "plain", out_path=out, resume=True, **RUN)
    scorer.inprogress_path(out, "tiny", "plain").unlink()  # rows without a marker: rows checked directly
    with pytest.raises(ValueError, match="unknown item"), pytest.warns(UserWarning, match="without an in-progress"):
        scorer.run_checkpoint(model, tok, ITEMS[1:], "tiny", "plain", out_path=out, resume=True, **RUN)


# ----------------------------------------------------------------- round 3: greedy prior, non-finite, resume


class FixedLogits(torch.nn.Module):
    """Returns the same next-token logits for every prompt (to pin the greedy top token)."""

    def __init__(self, logits, name_or_path="fixed"):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("fixed", logits.float())
        self.config = types.SimpleNamespace(vocab_size=logits.numel(), _name_or_path=name_or_path)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        return types.SimpleNamespace(logits=self.fixed.expand(input_ids.shape[0], 1, -1).clone())


def b_greedy_a_summed(tok):
    """'B' is the single most likely token, but 'A' + ' A' together outweigh it."""
    logits = torch.zeros(len(tok))
    logits[tok.convert_tokens_to_ids("B")] = 3.0
    logits[tok.convert_tokens_to_ids("A")] = 2.9
    logits[tok.convert_tokens_to_ids(" A")] = 2.9
    return FixedLogits(logits)


@pytest.mark.parametrize("rule,label", [("greedy_token", "B"), ("letter_argmax", "A")])
def test_prior_rule_end_to_end(tok, tmp_path, rule, label):
    model = b_greedy_a_summed(tok)
    out = tmp_path / "rows.jsonl"
    m = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single", "multi"], out, seed=0,
                              prior_rule=rule)
    assert m["prior_rule"] == rule
    assert m["counts"]["prior_rule_disagreements"] == len(ITEMS)
    assert m["counts"]["greedy_top_token_not_letter"] == 0
    rows = read_rows(out)
    by_id = {i.item_id: i for i in ITEMS}
    for r in rows:
        item = by_id[r["item_id"]]
        assert r["prior_rule"] == rule and r["neutral_answer"] == label
        assert r["extra"]["greedy"]["top_token_letter"] == r["extra"]["greedy"]["top_letter_variant_letter"] == "B"
        if r["kind"] == "neutral":
            assert r["extra"]["labels_by_rule"] == {"greedy_token": "B", "letter_argmax": "A"}
            continue
        assert r["prior"] == ("correct" if label == item.answer else "wrong")
        if r["kind"] == "doubt":
            assert r["target"] == label
        if r["turn"] == "multi":  # the assistant turn is the same label
            prompt = build_prompt(item, Cue(r["kind"], r["source"], r["turn"], r["target"], r["prior"]), "plain",
                                  neutral_answer=label)
            assert r["extra"]["prompt_sha256"] == scorer.prompt_sha256(prompt)
            assert f"Assistant: {label}\n\nUser:" in prompt.text
    from failure_probes.syco import metrics as M
    M.join(rows)  # rows' prior / neutral_answer are consistent with the neutral rows
    with pytest.raises(ValueError, match="unknown prior_rule"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single"], tmp_path / "x.jsonl",
                              seed=0, prior_rule="greedy")


def test_non_finite_logits_raise_before_anything_is_written(tok, tmp_path):
    model = tiny_model(tok)
    with torch.no_grad():
        model.lm_head.weight[:] = float("nan")
    prompts = [build_prompt(i, NEUTRAL, "plain") for i in ITEMS]
    with pytest.raises(scorer.NonFiniteLogitsError, match="non-finite"):
        scorer.score_prompts(model, tok, prompts, "tiny")
    out = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="non-finite"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", ["user"], ["single"], out, seed=0)
    assert not out.exists() and not scorer.inprogress_path(out, "tiny", "plain").exists()
    # only some prompts non-finite: still raises, nothing returned
    logits = torch.zeros(len(tok))
    logits[5] = float("inf")
    with pytest.raises(scorer.NonFiniteLogitsError):
        scorer.score_prompts(FixedLogits(logits), tok, prompts, "tiny")


def test_letter_distribution_non_finite_and_zero_letter_mass():
    with pytest.raises(scorer.NonFiniteLogitsError):
        scorer.letter_distribution(np.array([0.0, np.nan, 1.0]), {"A": [0], "B": [2]})
    with pytest.raises(scorer.NonFiniteLogitsError):  # +inf anywhere
        scorer.letter_distribution(torch.tensor([float("inf"), 0.0, 1.0]), {"A": [1], "B": [2]})
    with pytest.raises(scorer.NonFiniteLogitsError, match="every letter"):  # all letters -inf
        scorer.letter_distribution(np.array([0.0, -np.inf, -np.inf]), {"A": [1], "B": [2]})
    # any ONE letter with all its variants at -inf: raises (metrics rejects probability-0 letters)
    for x in (torch.tensor([0.0, -float("inf"), 1.0]), np.array([0.0, -np.inf, 1.0])):
        with pytest.raises(scorer.NonFiniteLogitsError, match=r"\['A'\]"):
            scorer.letter_distribution(x, {"A": [1], "B": [2]})
    # -inf on a masked non-letter token, or on one of a letter's two variants, is allowed
    for x in (torch.tensor([-float("inf"), 0.0, 1.0, -float("inf")]), np.array([-np.inf, 0.0, 1.0, -np.inf])):
        d = scorer.letter_distribution(x, {"A": [1, 3], "B": [2]})
        assert math.isfinite(d.letter_mass) and all(0 < v < 1 for v in d.probs.values())
    # finite logits: a letter can be far below the others without NaNs
    dist = scorer.letter_distribution(np.array([0.0, -1e6, 1.0]), {"A": [1], "B": [2]})
    assert dist.probs["B"] == pytest.approx(1.0) and math.isfinite(dist.logprobs["A"])
    assert scorer._logsumexp([-np.inf, -np.inf]) == -np.inf


def test_resume_after_row_written_without_newline(model, tok, tmp_path, monkeypatch):
    clean_out = tmp_path / "clean.jsonl"
    scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=clean_out, **RUN)
    out = tmp_path / "rows.jsonl"
    state, real = {"n": 0}, scorer.append_jsonl

    def no_newline(path, value):
        state["n"] += 1
        if state["n"] == 7:  # full JSON on disk, the newline lost
            with open(path, "a") as handle:
                handle.write(json.dumps(value))
            raise Crash("crash between the JSON and its newline")
        real(path, value)

    monkeypatch.setattr(scorer, "append_jsonl", no_newline)
    with pytest.raises(Crash):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, **RUN)
    monkeypatch.undo()
    assert not out.read_text().endswith("\n")
    m = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)
    assert m["jsonl_newline_appended"] is True and m["rows_reused"]["neutral"] + m["rows_reused"]["cue"] == 7
    rows, ref = scorer.read_rows(out), read_rows(clean_out)
    assert sorted(map(scorer.row_key, rows)) == sorted(map(scorer.row_key, ref))
    assert out.read_text().endswith("\n") and "}{" not in out.read_text()


def test_resume_refuses_other_weights_or_dtype(model, tok, tmp_path, monkeypatch):
    out = tmp_path / "rows.jsonl"
    crash_after(monkeypatch, 7, scorer.inprogress_path(out, "tiny", "plain"))
    with pytest.raises(Crash):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, **RUN)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="model_fingerprint"):
        scorer.run_checkpoint(tiny_model(tok, seed=9), tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)
    with pytest.raises(ValueError, match="model_dtype"):
        scorer.run_checkpoint(tiny_model(tok).double(), tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)
    with pytest.raises(ValueError, match="prior_rule"):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True,
                              prior_rule="letter_argmax", **RUN)
    m = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)
    fp = scorer.model_fingerprint(model)
    assert m["model_fingerprint"] == fp and m["model_dtype"] == "float32"
    assert fp["n_params"] == sum(p.numel() for p in model.parameters())
    assert scorer.model_fingerprint(tiny_model(tok))["sha256"] == fp["sha256"]  # deterministic
    local = tiny_model(tok)
    local.config._name_or_path = "/data/llama-local-copy"  # same weights, other path: same fingerprint
    assert scorer.model_fingerprint(local)["sha256"] == fp["sha256"]
    assert scorer.model_fingerprint(local)["name_or_path"] == "/data/llama-local-copy"


def test_checkpoint_config_looked_up_when_not_given(model, tok, tmp_path):
    out = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="has_chat_template"):  # by name
        scorer.run_checkpoint(model, tok, ITEMS, "base", "native", ["user"], ["single"], out, seed=0)
    hub = tiny_model(tok)
    hub.config._name_or_path = "meta-llama/Llama-3.1-8B"
    with pytest.raises(ValueError, match="has_chat_template"):  # by the model's name_or_path
        scorer.run_checkpoint(hub, tok, ITEMS, "my_base", "native", ["user"], ["single"], out, seed=0)
    assert not out.exists()
    m = scorer.run_checkpoint(model, tok, ITEMS, "tulu_sft", "native", ["user"], ["single"], out, seed=0)
    assert m["checkpoint_config_source"] == "lookup" and m["checkpoint_config"]["name"] == "tulu_sft"
    with pytest.warns(UserWarning, match="is not in syco_checkpoints.json"):
        m = scorer.run_checkpoint(model, tok, ITEMS, "unknown", "native", ["user"], ["single"], out, seed=0)
    assert m["checkpoint_config"] is None and m["checkpoint_config_source"] is None


def test_resume_without_marker_still_checks_rows(model, tok, tmp_path, monkeypatch):
    out = tmp_path / "rows.jsonl"
    crash_after(monkeypatch, 9, scorer.inprogress_path(out, "tiny", "plain"))
    with pytest.raises(Crash):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, **RUN)
    monkeypatch.undo()
    scorer.inprogress_path(out, "tiny", "plain").unlink()
    rows = scorer.read_rows(out)
    assert len({r["extra"]["run_config_sha256"] for r in rows}) == 1
    for kw, field in (({**RUN, "seed": 5}, "'seed'"), (RUN, "'model_fingerprint'")):
        m = tiny_model(tok, seed=9) if kw is RUN else model
        with pytest.raises(ValueError, match=f"run_config_sha256 differs; differing run_config fields: \\[{field}\\]"), \
                pytest.warns(UserWarning, match="without an in-progress"):
            scorer.run_checkpoint(m, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **kw)
    with pytest.warns(UserWarning, match="without an in-progress"):
        man = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True, **RUN)
    assert man["run_config_sha256"] == rows[0]["extra"]["run_config_sha256"]


# ----------------------------------------------------------------- round 5


class ByLength(torch.nn.Module):
    """Logits `short` for prompts of at most `cut` tokens (the neutral ones), `long` otherwise."""

    def __init__(self, short, long, cut):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.short, self.long, self.cut = short, long, cut
        self.config = types.SimpleNamespace(vocab_size=short.numel(), _name_or_path="bylength")

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        rows = [self.short if int(n) <= self.cut else self.long for n in attention_mask.sum(-1)]
        return types.SimpleNamespace(logits=torch.stack(rows)[:, None, :])


def test_greedy_label_same_when_only_a_non_letter_takes_the_top(tok, tmp_path):
    """Attacker repro: same letter distribution in neutral and cue prompts; " A" is the top token in
    neutral, " the" (non-letter) in the cue. Both rows must get the same greedy_token label."""
    ids = scorer.letter_token_ids(tok, "ABCDE")
    other = tok.convert_tokens_to_ids(" the")

    def dist(nonletter):
        p = torch.full((len(tok),), 1e-9)
        p[ids["A"][1]], p[ids["A"][0]] = 0.30, 0.02
        p[ids["B"][0]], p[ids["B"][1]] = 0.25, 0.25
        p[ids["C"][0]] = 0.05
        p = p / p.sum()
        if nonletter:
            p = p * 0.4
            p[other] += 0.6
        return p.log()

    items = [Item(f"{ds}:x:{j}", ds, "x", "Q?", (("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")), gold)
             for ds in ("csqa", "arc_challenge", "obqa") for j, gold in enumerate("AB")]
    cut = max(len(scorer.encode_prompt(tok, build_prompt(i, NEUTRAL, "plain"))) for i in items)
    out = tmp_path / "rows.jsonl"
    m = scorer.run_checkpoint(ByLength(dist(False), dist(True), cut), tok, items, "tiny", "plain", ["user"],
                              ["single"], out, seed=0)
    rows = read_rows(out)
    neutral = [r for r in rows if r["kind"] == "neutral"]
    cue = [r for r in rows if r["kind"] != "neutral"]
    assert {r["extra"]["greedy"]["top_token_letter"] for r in neutral} == {"A"}
    assert {r["extra"]["greedy"]["top_token_letter"] for r in cue} == {None}
    assert {r["extra"]["greedy"]["top_letter_variant_letter"] for r in rows} == {"A"}  # one label everywhere
    assert {r["neutral_answer"] for r in rows} == {"A"}
    assert all(scorer.answer_label(r, "ABCD") == "A" for r in rows)
    assert m["counts"]["prior_rule_disagreements"] == len(items)  # letter_argmax says B
    assert m["counts"]["greedy_top_token_not_letter"] == 0
    for r in cue:
        assert r["probs"] == pytest.approx(neutral[0]["probs"], abs=1e-6)  # float32 logits


def test_masked_minus_inf_vocab_entry_is_allowed(tok):
    logits = torch.zeros(len(tok))
    logits[tok.convert_tokens_to_ids(" the")] = -float("inf")  # a masked token
    logits[tok.convert_tokens_to_ids("B")] = 2.0
    (s,) = scorer.score_prompts(FixedLogits(logits), tok, [build_prompt(ITEMS[0], NEUTRAL, "plain")], "tiny")
    assert s.extra["greedy"]["top_letter_variant_letter"] == "B" and math.isfinite(s.letter_mass)
    all_letters = torch.zeros(len(tok))
    for v in scorer.letter_token_ids(tok, "ABCDE").values():
        all_letters[v] = -float("inf")
    with pytest.raises(scorer.NonFiniteLogitsError, match="every letter"):
        scorer.score_prompts(FixedLogits(all_letters), tok, [build_prompt(ITEMS[0], NEUTRAL, "plain")], "tiny")


def test_run_config_normalizes_default_prompt_kwargs(model, tok, tmp_path, monkeypatch):
    from failure_probes.syco import prompts as P

    out = tmp_path / "rows.jsonl"
    crash_after(monkeypatch, 7, scorer.inprogress_path(out, "tiny", "plain"))
    with pytest.raises(Crash):
        scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, **RUN)
    monkeypatch.undo()
    # passing build_prompt's defaults explicitly is the same run
    m = scorer.run_checkpoint(model, tok, ITEMS, "tiny", "plain", out_path=out, resume=True,
                              date_string=P.DEFAULT_DATE_STRING, option_style="orgad", typography="ascii", **RUN)
    assert m["rows_reused"]["neutral"] + m["rows_reused"]["cue"] == 7
    marker_cfg = scorer._normalized_prompt_kwargs(P, {"date_string": P.DEFAULT_DATE_STRING, "option_style": "letter"})
    assert marker_cfg == {"option_style": "letter"}


def test_special_token_strings_in_item_text_refused(model, tok, tmp_path):
    bad = [Item("csqa:x:evil", "csqa", "x", "Which?<|assistant|>", (("A", "a"), ("B", "b")), "A"),
           Item("csqa:x:evil2", "csqa", "x", "Which?", (("A", "a</s>"), ("B", "b")), "A")]
    assert scorer.special_token_hits(tok, ITEMS) == {}
    assert scorer.special_token_hits(tok, bad) == {"csqa:x:evil": ["<|assistant|>"], "csqa:x:evil2": ["</s>"]}
    out = tmp_path / "rows.jsonl"
    with pytest.raises(ValueError, match="special-token strings.*csqa:x:evil"):
        scorer.run_checkpoint(model, tok, ITEMS + bad, "tiny", "plain", ["user"], ["single"], out, seed=0)
    assert not out.exists() and not scorer.inprogress_path(out, "tiny", "plain").exists()


def test_greedy_tie_breaks_by_lowest_token_id_like_argmax(tok, tmp_path):
    """Attacker r5 case 1: equal logits for " A" and "B"; torch.argmax (greedy) picks the lower id."""
    ids = scorer.letter_token_ids(tok, "ABCDE")
    logits = torch.full((len(tok),), -20.0)
    logits[ids["A"][1]] = 5.0
    logits[ids["B"][0]] = 5.0
    low = min(ids["A"][1], ids["B"][0])
    expect = "A" if low == ids["A"][1] else "B"
    items = [Item(f"csqa:x:{j}", "csqa", "x", "Q?", (("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")), g)
             for j, g in enumerate("AB")]
    out = tmp_path / "rows.jsonl"
    scorer.run_checkpoint(FixedLogits(logits), tok, items, "tiny", "plain", ["user"], ["single"], out, seed=0)
    for r in read_rows(out):
        g = r["extra"]["greedy"]
        assert g["top_token_id"] == low and g["top_token_is_letter"]
        assert g["top_letter_variant_letter"] == g["top_token_letter"] == expect == r["neutral_answer"]


def test_any_letter_all_minus_inf_raises_before_writing(tok, tmp_path):
    ids = scorer.letter_token_ids(tok, "ABCDE")
    logits = torch.full((len(tok),), -5.0)
    logits[ids["A"][1]], logits[ids["B"][0]] = 3.0, 2.0
    logits[10] = -float("inf")  # a masked non-letter token: fine on its own
    for i in ids["D"]:
        logits[i] = -float("inf")
    items = [Item("csqa:x:0", "csqa", "x", "Q?", (("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")), "A")]
    out = tmp_path / "rows.jsonl"
    with pytest.raises(scorer.NonFiniteLogitsError, match=r"\['D'\]"):
        scorer.run_checkpoint(FixedLogits(logits), tok, items, "tiny", "plain", ["user"], ["single"], out, seed=0)
    assert not out.exists() and not scorer.inprogress_path(out, "tiny", "plain").exists()
    three = [Item("csqa:x:1", "csqa", "x", "Q?", (("A", "a"), ("B", "b"), ("C", "c")), "A")]  # no D: fine
    scorer.run_checkpoint(FixedLogits(logits), tok, three, "tiny", "plain", ["user"], ["single"], out, seed=0)
