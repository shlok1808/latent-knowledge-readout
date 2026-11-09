"""E3: does B-TAP pruning hit caving or the one-letter answer format? (lab fork #28, bet #29)

Teacher-forced log-probabilities of FIXED replies, no generation. For each pushed prompt we score a CAVE reply and a
RESIST reply in three forms, built by code from the item's own option text (correct by construction):
  letter       "(B)" (single-turn spelling) or "B" (multi-turn spelling; the earlier assistant turn is a bare letter)
  answer_sentence "The correct answer is (B) oven."  (what the pruned model writes; letter read at its slot)
  letter_last  "Looking at the options, oven fits the question best, so the answer is (B)."
every reply followed by the end-of-turn token. We also score real replies the models wrote, each paired with a minimal
twin where only the final answer is swapped (`swap_final_answer`).

Every scored example keeps its per-token log-probs and the index of the ANSWER token (the token carrying the letter),
so the analysis can use the whole reply, the first token, or the log-prob at the answer slot.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

FORMS = ("letter", "answer_sentence", "letter_last")
ANSWER_SENTENCE = "The correct answer is ({L}) {text}."   # the pruned model's own modal opener (critic, E3)
LETTER_LAST_HEAD = "Looking at the options, {text} fits the question best, so the answer is ({L})."


def reply_text(item, letter: str, form: str, spelling: str = "paren") -> str:
    """The reply text (without end-of-turn) for one form. spelling: "paren" -> "(B)", "bare" -> "B" (letter form only;
    the sentence forms always write "(B)")."""
    if letter not in item.letters:
        raise ValueError(f"{item.item_id}: letter {letter!r} not in {item.letters}")
    text = dict(item.options)[letter]
    if form == "letter":
        if spelling not in ("paren", "bare"):
            raise ValueError(f"unknown spelling {spelling!r}")
        return f"({letter})" if spelling == "paren" else letter
    if form == "answer_sentence":
        return ANSWER_SENTENCE.format(L=letter, text=text.rstrip("."))
    if form == "letter_last":
        return LETTER_LAST_HEAD.format(text=text.rstrip("."), L=letter)
    raise ValueError(f"unknown form {form!r}; expected one of {FORMS}")


def answer_char_index(reply: str, letter: str) -> int:
    """Character index of the answer letter in a reply: the LAST "(L)" if present, else a bare single-letter reply."""
    i = reply.rfind(f"({letter})")
    if i >= 0:
        return i + 1
    if reply.strip() == letter:
        return reply.index(letter)
    m = None
    for m in re.finditer(rf"(?<![A-Za-z]){re.escape(letter)}(?![A-Za-z])", reply):
        pass
    if m is None:
        raise ValueError(f"letter {letter!r} not found in reply {reply!r}")
    return m.start()


def encode_reply(tokenizer, reply: str, letter: str, eot_id: int):
    """(token ids incl. end-of-turn, index of the token that contains the answer letter)."""
    enc = tokenizer(reply, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
    c = answer_char_index(reply, letter)
    idx = next((k for k, (a, b) in enumerate(offs) if a <= c < b), None)
    if idx is None:
        raise ValueError(f"no token covers the answer letter in {reply!r}")
    return ids + [int(eot_id)], idx


def swap_final_answer(reply: str, item, old: str, new: str):
    """Minimal-pair twin: replace only the FINAL answer old -> new (the last "(old)" and, if it directly follows,
    old's option text; or a bare one-letter reply). Returns None when the reply's final answer can't be located.
    The text before and after stays word-for-word identical."""
    if old == new or old not in item.letters or new not in item.letters:
        return None
    body, eot = (reply.split("<|eot_id|>")[0], "<|eot_id|>" in reply)
    if body.strip() == old:
        out = body.replace(old, new, 1)
    else:
        i = body.rfind(f"({old})")
        if i < 0:
            return None
        head, tail = body[:i], body[i + len(old) + 2:]
        opts = dict(item.options)
        old_t, new_t = opts[old].rstrip("."), opts[new].rstrip(".")
        if tail.lstrip().lower().startswith(old_t.lower()):
            lead = tail[: len(tail) - len(tail.lstrip())]
            tail = lead + new_t + tail.lstrip()[len(old_t):]
        out = f"{head}({new}){tail}"
    return out + ("<|eot_id|>" if eot else "")


def letter_slot_ids(tokenizer, letters="ABCDE"):
    """{token_id: (spelling, letter)} for the letter tokens read at an answer slot: "A", " A" and "(A"."""
    out = {}
    for L in letters:
        for sp, text in (("bare", L), ("space", " " + L), ("paren", "(" + L)):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                out[int(ids[0])] = (sp, L)
    return out


def score_replies(model, examples, batch_size=64, pad_id=0, progress=None, slot_ids=None):
    """Teacher-forced per-token log-probs of each example's reply_ids given prompt_ids. Right padding + attention
    mask (padding after every real token cannot change real positions under the causal mask). Batches in length
    order; returns one list of floats per example, in input order. With slot_ids (list of token ids), returns dicts
    {"logp": [...], "slot": [log p(id) at the answer slot], "first": [log p(id) at the first reply position]}, the
    answer slot being the position that predicts reply_ids[answer_idx]."""
    device = next(model.parameters()).device
    order = sorted(range(len(examples)), key=lambda i: -(len(examples[i]["prompt_ids"]) + len(examples[i]["reply_ids"])))
    out = [None] * len(examples)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            seqs = [list(examples[i]["prompt_ids"]) + list(examples[i]["reply_ids"]) for i in idx]
            width = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), width), int(pad_id), dtype=torch.long)
            attn = torch.zeros_like(ids)
            for r, x in enumerate(seqs):
                ids[r, :len(x)] = torch.tensor(x)
                attn[r, :len(x)] = 1
            hidden = model.model(input_ids=ids.to(device), attention_mask=attn.to(device), use_cache=False)[0]
            rows, spans = [], []
            for r, i in enumerate(idx):  # lm_head only on the positions that predict reply tokens (memory + speed)
                p, t = len(examples[i]["prompt_ids"]), len(examples[i]["reply_ids"])
                rows.append(hidden[r, p - 1:p + t - 1])
                spans.append(t)
            logp = F.log_softmax(model.lm_head(torch.cat(rows)).float(), dim=-1)
            tgt = torch.tensor([tok for i in idx for tok in examples[i]["reply_ids"]], device=device)
            vals = logp.gather(1, tgt[:, None])[:, 0].tolist()
            sl = logp[:, slot_ids].tolist() if slot_ids is not None else None
            k = 0
            for i, t in zip(idx, spans):
                if slot_ids is None:
                    out[i] = vals[k:k + t]
                else:
                    a = int(examples[i]["answer_idx"])
                    out[i] = {"logp": vals[k:k + t], "slot": sl[k + a], "first": sl[k]}
                k += t
            if progress:
                progress(min(s + batch_size, len(order)), len(order))
    return out
