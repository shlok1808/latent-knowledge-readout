"""Preflight checks on the offline tiny Llama (tests/syco_tiny.py). CPU, no network, no writes."""
import json
import os

import pytest

from knowledge_readout.syco import scorer
from knowledge_readout.syco.preflight import preflight
from knowledge_readout.syco.types import Item
from syco_tiny import CHAT_TEMPLATE_DATED, CHAT_TEMPLATE_NO_BOS, tiny_model, tiny_tokenizer


def make_item(i, n_options=5, answer="B"):
    letters = "ABCDE"[:n_options]
    words = ["red", "blue", "green", "cat", "dog"]
    return Item(item_id=f"csqa:validation:{i}", dataset="csqa", split="validation",
                question=f"Which one is it {i}?" + " What" * (i % 4),
                options=tuple((letter, words[k]) for k, letter in enumerate(letters)), answer=answer)


ITEMS = [make_item(i, 3 + i % 3, "ABC"[i % 3]) for i in range(24)]
# the tiny tokenizer's name_or_path is "": no checkpoint_config is found, which warns
pytestmark = pytest.mark.filterwarnings("ignore:checkpoint .* is not in syco_checkpoints.json")


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    return tiny_model(tok)


def test_tokenizer_only_report(tok, tmp_path):
    before = set(os.listdir(tmp_path))
    rep = preflight(tok)
    json.dumps(rep)
    assert set(os.listdir(tmp_path)) == before
    L = rep["letters"]
    for letter in "ABCDE":
        assert L[letter]["single_token"] and L[letter]["ids"] == [tok.convert_tokens_to_ids(letter)]
        assert L[" " + letter]["single_token"] and L[" " + letter]["token"] == " " + letter
        assert not L["(" + letter]["single_token"] and not L[" (" + letter]["single_token"]
    assert set(L) == {p.format(L=x) for p in ("{L}", " {L}", "({L}", " ({L}") for x in "ABCDE"}
    assert rep["collisions"] == []
    assert rep["letter_token_ids"] == scorer.letter_token_ids(tok, "ABCDE")
    t = rep["tokenizer"]
    assert t["has_chat_template"] is True and t["pad_token"] == "<pad>" and t["pad_falls_back_to_eos"] is False
    assert t["bos_token"] == "<s>" and t["len_tokenizer"] == len(tok)
    plain, native = rep["formats"]["plain"], rep["formats"]["native"]
    assert plain["neutral"]["starts_with_bos"] is False and plain["neutral"]["bos_added_by_scorer"] is True
    assert native["neutral"]["starts_with_bos"] is True and native["neutral"]["bos_added_by_scorer"] is False
    for f in (plain, native):
        for which in ("neutral", "multi"):
            assert f[which]["encode_ok"] and f[which]["n_bos"] == 1
    assert native["neutral"]["generation_prompt"] == "<|assistant|>\n"
    assert native["neutral"]["tail"].endswith("<|assistant|>\n")
    assert plain["neutral"]["tail"].endswith("Assistant:")
    assert native["neutral"]["contains_today_date"] is False
    assert rep["problems"] == []
    assert "neutral_diagnostics" not in plain  # no model given


def test_template_without_bos_and_dated_template():
    rep = preflight(tiny_tokenizer(chat_template=CHAT_TEMPLATE_NO_BOS), fmt_list=("native",))
    n = rep["formats"]["native"]["neutral"]
    assert n["starts_with_bos"] is False and n["bos_added_by_scorer"] is True and n["n_bos"] == 1
    assert rep["problems"] == []

    rep = preflight(tiny_tokenizer(chat_template=CHAT_TEMPLATE_DATED), fmt_list=("native",))
    n = rep["formats"]["native"]["neutral"]
    assert n["contains_today_date"] is True and n["today_date_line"].startswith("Today Date: ")
    rep = preflight(tiny_tokenizer(chat_template=CHAT_TEMPLATE_DATED), fmt_list=("native",), date_string="01 Jan 2025")
    assert rep["formats"]["native"]["neutral"]["today_date_line"] == "Today Date: 01 Jan 2025"
    assert rep["formats"]["native"]["date_string"] == "01 Jan 2025"


def test_no_template_no_pad_and_config_refusal(tok):
    rep = preflight(tiny_tokenizer(chat_template=False, pad=False))
    assert rep["tokenizer"]["has_chat_template"] is False and rep["tokenizer"]["pad_falls_back_to_eos"] is True
    assert "chat_template" in rep["formats"]["native"]["error"]
    assert any("pad" in p for p in rep["problems"])
    rep = preflight(tok, checkpoint_config=scorer.checkpoint_entry("base"))
    assert "refused" in rep["formats"]["native"]
    assert rep["formats"]["plain"]["neutral"]["encode_ok"]


def test_collision_and_missing_letter_reported():
    class Tok:
        """Every spelling of every letter encodes to the same id."""
        unk_token_id = None
        bos_token = bos_token_id = eos_token = eos_token_id = pad_token = pad_token_id = None
        padding_side = "right"
        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [7]

        def decode(self, ids):
            return "?"

        def __len__(self):
            return 10

    rep = preflight(Tok(), fmt_list=())
    assert rep["collisions"] and rep["letter_token_ids"] is None
    assert any("share token id" in p for p in rep["problems"])


def test_with_model_and_items(model, tok):
    side, pad = tok.padding_side, tok.pad_token
    rep = preflight(tok, model, ITEMS, n_compare=20, batch_size=4)
    assert (tok.padding_side, tok.pad_token) == (side, pad)  # tokenizer state restored
    json.dumps(rep)
    assert rep["tokenizer"]["model_config_vocab_size"] == model.config.vocab_size
    assert rep["tokenizer"]["equal"] is True
    for fmt in ("plain", "native"):
        f = rep["formats"][fmt]
        d = f["neutral_diagnostics"]
        assert d["n"] == len(ITEMS) and len(d["top_tokens"]) <= 10
        assert 0 < d["letter_mass_min"] <= d["letter_mass_mean"] < 1
        assert 0 <= d["frac_letter_mass_below_min"] <= 1
        assert f["n_compared"] == 20
        assert f["max_abs_batched_minus_unbatched"] < 1e-4
    # the untrained tiny model puts little mass on letters: flagged, not raised
    assert any("letter_mass" in p for p in rep["problems"])


def test_with_model_custom_variants(model, tok):
    rep = preflight(tok, model, ITEMS[:5], fmt_list=("plain",), variants=("{L}",))
    assert rep["letter_token_ids"]["A"] == [tok.convert_tokens_to_ids("A")]
    assert rep["formats"]["plain"]["neutral_diagnostics"]["n"] == 5


def test_expected_letter_spelling_checked(tok, monkeypatch):
    from knowledge_readout.syco import prompts as P

    rep = preflight(tok)
    exp = rep["expected_letter_spelling"]
    assert exp["plain"]["pattern"] == " {L}" and exp["native"]["pattern"] == "{L}"
    assert all(v["single_token"] for f in exp.values() for v in f["letters"].values())
    assert rep["problems"] == []
    # scorer variants without the plain format's " A": flagged
    rep = preflight(tok, fmt_list=("plain",), variants=("{L}",))
    assert any("not among the scorer variants" in p for p in rep["problems"])
    # an expected spelling that is several tokens: flagged per letter
    monkeypatch.setitem(P.EXPECTED_LETTER_PREFIX, "plain", "  ")
    rep = preflight(tok, fmt_list=("plain",))
    assert sum("is not a single token" in p for p in rep["problems"]) == 5


def test_no_names_no_lookup_warning(tok):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing to look up: no "checkpoint None is not in ..." warning
        rep = preflight(tok, fmt_list=("plain",))
    assert rep["checkpoint_config"] is None and rep["checkpoint_config_source"] is None


def test_checkpoint_config_looked_up(tok):
    rep = preflight(tok, checkpoint="base")
    assert rep["checkpoint_config_source"] == "lookup" and rep["checkpoint_config"]["name"] == "base"
    assert "refused" in rep["formats"]["native"]
    hub = tiny_tokenizer()
    hub.name_or_path = "meta-llama/Llama-3.1-8B"
    assert "refused" in preflight(hub)["formats"]["native"]  # by the tokenizer's name_or_path
    with pytest.warns(UserWarning, match="is not in syco_checkpoints.json"):
        rep = preflight(tok, checkpoint="nope")
    assert rep["checkpoint_config"] is None and "neutral" in rep["formats"]["native"]


def test_special_token_items_flagged(tok):
    bad = Item("csqa:x:evil", "csqa", "x", "Which?<s>", (("A", "a"), ("B", "b")), "A")
    rep = preflight(tok, items=[bad], fmt_list=("plain",))
    assert rep["items_with_special_tokens"] == {"csqa:x:evil": ["<s>"]}
    assert any("special-token" in p for p in rep["problems"])
