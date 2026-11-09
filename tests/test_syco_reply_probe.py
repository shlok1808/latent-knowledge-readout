"""Offline tests for knowledge_readout.syco.reply_probe (E3)."""
import pytest
import torch

from knowledge_readout.syco import pruning as PR
from knowledge_readout.syco import reply_probe as R
from knowledge_readout.syco.types import Item

from syco_tiny import tiny_model, tiny_tokenizer

ITEM = Item(item_id="csqa:validation:q1", dataset="csqa", split="validation", question="Where do you keep milk?",
            options=(("A", "fridge"), ("B", "oven"), ("C", "garage.")), answer="A")


def test_reply_text_forms():
    assert R.reply_text(ITEM, "B", "letter") == "(B)"
    assert R.reply_text(ITEM, "B", "letter", "bare") == "B"
    assert R.reply_text(ITEM, "B", "answer_sentence") == "The correct answer is (B) oven."
    assert R.reply_text(ITEM, "C", "letter_last") == \
        "Looking at the options, garage fits the question best, so the answer is (C)."
    # cave and resist replies of one form differ only in the answer part
    a, b = R.reply_text(ITEM, "A", "letter_last"), R.reply_text(ITEM, "B", "letter_last")
    assert a.replace("fridge", "X").replace("(A)", "(Y)") == b.replace("oven", "X").replace("(B)", "(Y)")
    with pytest.raises(ValueError):
        R.reply_text(ITEM, "D", "letter")


@pytest.mark.parametrize("reply,letter", [("(B)", "B"), ("B", "B"), ("The correct answer is (B) oven.", "B"),
                                          ("Looking at it, (A) fridge or not, so the answer is (B).", "B")])
def test_answer_index_points_at_letter(reply, letter):
    tok = tiny_tokenizer()
    ids, idx = R.encode_reply(tok, reply, letter, tok.eos_token_id)
    assert ids[-1] == tok.eos_token_id
    assert letter in tok.decode([ids[idx]])
    # the LAST "(B)" is the answer, not an earlier mention
    if reply.count("(") > 1:
        assert idx > 3


def test_swap_final_answer():
    assert R.swap_final_answer("(B)<|eot_id|>", ITEM, "B", "A") == "(A)<|eot_id|>"
    assert R.swap_final_answer("B<|eot_id|>", ITEM, "B", "A") == "A<|eot_id|>"
    assert R.swap_final_answer("I agree. The answer is (B) oven.<|eot_id|>", ITEM, "B", "A") == \
        "I agree. The answer is (A) fridge.<|eot_id|>"
    # only the LAST mention changes
    assert R.swap_final_answer("(A) is wrong; the answer is (B) Oven, really.", ITEM, "B", "A") == \
        "(A) is wrong; the answer is (A) fridge, really."
    assert R.swap_final_answer("no letters here", ITEM, "B", "A") is None
    assert R.swap_final_answer("(B)", ITEM, "B", "B") is None


def test_score_replies_matches_token_ce_and_is_batch_invariant():
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    exs = [{"prompt_ids": [tok.bos_token_id] + tok.encode(q, add_special_tokens=False),
            "reply_ids": tok.encode(r, add_special_tokens=False) + [tok.eos_token_id]}
           for q, r in [("Where is milk?", "(A)"), ("Q", "B fridge answer"), ("What is the answer?", "C")]]
    got = R.score_replies(model, exs, batch_size=2, pad_id=0)
    one = [R.score_replies(model, [e], batch_size=1)[0] for e in exs]
    for g, o, e in zip(got, one, exs):
        ref = (-PR.token_ce(model, {"prompt_ids": e["prompt_ids"], "target_ids": e["reply_ids"]})).tolist()
        assert len(g) == len(e["reply_ids"])
        assert g == pytest.approx(o, abs=1e-4)
        assert g == pytest.approx(ref, abs=1e-4)


def test_score_replies_slot_logprobs():
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    slot = sorted(R.letter_slot_ids(tok, "ABC"))
    reply = "x so the answer is (B)."
    ids, a = R.encode_reply(tok, reply, "B", tok.eos_token_id)
    ex = {"prompt_ids": [tok.bos_token_id] + tok.encode("Q?", add_special_tokens=False), "reply_ids": ids, "answer_idx": a}
    out = R.score_replies(model, [ex, ex], batch_size=2, slot_ids=slot)
    plain = R.score_replies(model, [ex])[0]
    assert out[0]["logp"] == pytest.approx(plain, abs=1e-5)
    # the answer token is one of the slot ids, and its slot log-prob equals its own per-token log-prob
    j = slot.index(ids[a]) if ids[a] in slot else None
    if j is not None:
        assert out[0]["slot"][j] == pytest.approx(plain[a], abs=1e-5)
    assert len(out[0]["slot"]) == len(slot) == len(out[0]["first"])
