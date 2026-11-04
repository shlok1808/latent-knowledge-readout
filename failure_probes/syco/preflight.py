"""Cheap pre-run checks for one checkpoint on Lambda (no writes, no generation).

    from failure_probes.syco.preflight import preflight
    report = preflight(tokenizer, model, items[:64])
    report["problems"]   # human-readable list; empty means nothing obviously wrong

The report is a plain JSON-serializable dict:
- `letters`: for each spelling "A".."E", " A".." E", "(A".."(E", " (A".." (E": its ids, whether it is
  one real token (not unk), and the decoded token; `collisions` lists ids shared by spellings of
  different letters; `letter_token_ids` is what the scorer would use with `variants` (or the error).
- `expected_letter_spelling[fmt]`: the spelling each format's answer is expected in
  (prompts.EXPECTED_LETTER_PREFIX: plain " A", native "A") for "A".."E": a problem when one is not
  a single real token, or when `variants` does not include that spelling (the scorer would miss it).
- `checkpoint_config` / `checkpoint_config_source`: the given entry, else the
  configs/syco_checkpoints.json entry looked up by `checkpoint`, the tokenizer's or the model's
  name_or_path ("lookup"); not found -> a warning and None. has_chat_template=false refuses native.
- `items_with_special_tokens` (with items): item_id -> tokenizer special-token strings found in its
  text; any hit is a problem (run_checkpoint refuses such items).
- `tokenizer`: chat_template present, bos/eos/pad tokens and ids, padding side, len(tokenizer),
  and model.config.vocab_size when a model is given.
- `formats[fmt]`: a rendered neutral (and, with items, a multi-turn) prompt: whether the text
  starts with BOS, contains "Today Date" (and that line), its tail, the generation prompt suffix
  (native: apply_chat_template with minus without add_generation_prompt), the encoded length and
  BOS count after `scorer.encode_text`, and the last tokens. Errors are recorded, not raised.
- With model and items: per format, the neutral-pass diagnostics (top-token histogram, letter
  mass mean/min, fraction below `min_mass`) and the max |batched - unbatched| letter-prob
  difference on up to `n_compare` prompts.
"""
from __future__ import annotations

from failure_probes.syco import scorer
from failure_probes.syco.types import Cue, Item

LETTER_SPELLINGS = ("{L}", " {L}", "({L}", " ({L}")
PREFLIGHT_LETTERS = "ABCDE"
DUMMY_ITEM = Item(item_id="preflight:dummy:0", dataset="preflight", split="dummy",
                  question="Which color is the sky on a clear day?",
                  options=(("A", "green"), ("B", "blue"), ("C", "red"), ("D", "black"), ("E", "white")),
                  answer="B")


def _err(e):
    return f"{type(e).__name__}: {e}"


def _letter_report(tokenizer, letters, spellings):
    unk = getattr(tokenizer, "unk_token_id", None)
    rows, owner, collisions = {}, {}, []
    for pattern in spellings:
        for letter in letters:
            text = pattern.format(L=letter)
            ids = [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]
            single = len(ids) == 1 and ids[0] != unk
            rows[text] = {"letter": letter, "pattern": pattern, "ids": ids, "single_token": single,
                          "is_unk": bool(ids) and unk is not None and unk in ids,
                          "token": scorer._decode_token(tokenizer, ids[0]) if single else None}
            if single:
                prev = owner.get(ids[0])
                if prev is not None and prev[0] != letter:
                    collisions.append({"token_id": ids[0], "spellings": [prev[1], text]})
                owner.setdefault(ids[0], (letter, text))
    return rows, collisions


def _build(P, item, cue, fmt, tokenizer, neutral_answer=None, date_string=None, **kw):
    extra = {"date_string": date_string} if date_string is not None else {}
    required = tuple(extra) + tuple(kw)
    return scorer._call(P.build_prompt, required=required, item=item, cue=cue, fmt=fmt, tokenizer=tokenizer,
                        neutral_answer=neutral_answer, **extra, **kw)


def _generation_prompt(tokenizer, messages, date_string=None):
    kw = {"date_string": date_string} if date_string is not None else {}
    with_gp = tokenizer.apply_chat_template([dict(m) for m in messages], tokenize=False,
                                            add_generation_prompt=True, **kw)
    without = tokenizer.apply_chat_template([dict(m) for m in messages], tokenize=False,
                                            add_generation_prompt=False, **kw)
    return with_gp[len(without):] if with_gp.startswith(without) else None


def _text_report(tokenizer, prompt, date_string=None):
    text = prompt.text
    bos = tokenizer.bos_token
    rep = {"starts_with_bos": bool(bos) and text.startswith(bos),
           "contains_today_date": "Today Date" in text,
           "today_date_line": scorer._today_date_line(text),
           "tail": text[-60:],
           "n_chars": len(text)}
    if prompt.fmt == "native":
        try:
            rep["generation_prompt"] = _generation_prompt(tokenizer, prompt.messages, date_string)
        except Exception as e:  # noqa: BLE001
            rep["generation_prompt"] = None
            rep["generation_prompt_error"] = _err(e)
    try:
        ids, had_bos = scorer.encode_text(tokenizer, text, label=f"preflight {prompt.fmt}")
        rep.update({"encode_ok": True, "n_tokens": len(ids),
                    "n_bos": sum(1 for i in ids if i == tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else 0,
                    "bos_added_by_scorer": tokenizer.bos_token_id is not None and not had_bos,
                    "last_tokens": [scorer._decode_token(tokenizer, i) for i in ids[-5:]]})
    except Exception as e:  # noqa: BLE001
        rep.update({"encode_ok": False, "encode_error": _err(e)})
    return rep


def _max_batch_diff(model, tokenizer, prompts, batch_size, variants):
    one = scorer.score_prompts(model, tokenizer, prompts, "preflight", batch_size=1, variants=variants)
    many = scorer.score_prompts(model, tokenizer, prompts, "preflight", batch_size=max(batch_size, 2),
                                variants=variants)
    diff = 0.0
    for a, b in zip(one, many):
        for letter in a.probs:
            diff = max(diff, abs(a.probs[letter] - b.probs[letter]))
        diff = max(diff, abs(a.letter_mass - b.letter_mass))
    return diff


def _expected_spelling_report(tokenizer, fmt, variants, problems, P):
    prefix = getattr(P, "EXPECTED_LETTER_PREFIX", {}).get(fmt)
    if prefix is None:
        return None
    pattern = prefix + "{L}"
    unk = getattr(tokenizer, "unk_token_id", None)
    rep = {"pattern": pattern, "in_variants": pattern in variants, "letters": {}}
    for letter in PREFLIGHT_LETTERS:
        text = pattern.format(L=letter)
        ids = [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]
        single = len(ids) == 1 and ids[0] != unk
        rep["letters"][letter] = {"text": text, "ids": ids, "single_token": single}
        if not single:
            problems.append(f"{fmt}: expected answer spelling {text!r} is not a single token (ids {ids})")
    if not rep["in_variants"]:
        problems.append(f"{fmt}: expected answer spelling {pattern!r} is not among the scorer variants "
                        f"{tuple(variants)}")
    return rep


def preflight(tokenizer, model=None, items=None, fmt_list=("plain", "native"), *, checkpoint_config=None,
              checkpoint=None, variants=scorer.DEFAULT_VARIANTS, date_string=None, n_compare=20, batch_size=8,
              min_mass=scorer.DEFAULT_MIN_MASS, max_batch_diff=1e-3):
    """Run the cheap checks and return a dict report (see module docstring). Writes nothing."""
    from failure_probes.syco import prompts as P

    items = list(items) if items is not None else None
    problems = []
    checkpoint_config, config_source = scorer._resolve_checkpoint_config(
        checkpoint, checkpoint_config, getattr(tokenizer, "name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None))
    report = {"variants": list(variants), "checkpoint": checkpoint, "checkpoint_config": checkpoint_config,
              "checkpoint_config_source": config_source, "problems": problems,
              "expected_letter_spelling": {}}

    letters, collisions = _letter_report(tokenizer, PREFLIGHT_LETTERS, LETTER_SPELLINGS)
    report["letters"] = letters
    report["collisions"] = collisions
    if collisions:
        problems.append(f"letter spellings share token ids: {collisions}")
    try:
        report["letter_token_ids"] = scorer.letter_token_ids(tokenizer, PREFLIGHT_LETTERS, variants)
    except ValueError as e:
        report["letter_token_ids"] = None
        report["letter_token_ids_error"] = _err(e)
        problems.append(f"scorer letter ids: {e}")

    has_template = getattr(tokenizer, "chat_template", None) is not None
    tok = {"has_chat_template": has_template,
           "bos_token": tokenizer.bos_token, "bos_token_id": tokenizer.bos_token_id,
           "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
           "pad_token": tokenizer.pad_token, "pad_token_id": tokenizer.pad_token_id,
           "pad_falls_back_to_eos": tokenizer.pad_token is None,
           "padding_side": tokenizer.padding_side, "len_tokenizer": len(tokenizer)}
    if model is not None:
        tok.update({k: v for k, v in scorer._vocab_info(model, tokenizer).items() if k != "len_tokenizer"})
        if tok["model_config_vocab_size"] is not None and tok["model_config_vocab_size"] < len(tokenizer):
            problems.append(f"len(tokenizer)={len(tokenizer)} > model vocab_size={tok['model_config_vocab_size']}")
    report["tokenizer"] = tok
    if tokenizer.pad_token is None:
        problems.append("no pad token: scoring pads with eos (fine under left padding + mask, but check)")

    if items:
        hits = scorer.special_token_hits(tokenizer, items)
        report["items_with_special_tokens"] = hits
        if hits:
            problems.append(f"{len(hits)} item(s) contain tokenizer special-token strings (run_checkpoint "
                            f"refuses them): {dict(list(hits.items())[:5])}")

    gp_date = date_string if date_string is not None else scorer._date_string_default(P)
    sample_item = items[0] if items else DUMMY_ITEM
    neutral = P.neutral_cue() if hasattr(P, "neutral_cue") else scorer.NEUTRAL_CUE
    multi = Cue("suggestion", "user", "multi", sample_item.answer, "wrong")
    report["formats"] = {}
    for fmt in fmt_list:
        frep = {}
        report["formats"][fmt] = frep
        if fmt == "native" and checkpoint_config is not None and checkpoint_config.get("has_chat_template") is False:
            frep["refused"] = "checkpoint_config has_chat_template=false (run_checkpoint refuses native)"
            continue
        if not (fmt == "native" and not has_template):
            report["expected_letter_spelling"][fmt] = _expected_spelling_report(tokenizer, fmt, variants, problems, P)
        try:
            neutral_prompt = _build(P, sample_item, neutral, fmt, tokenizer, date_string=date_string)
            frep["neutral"] = _text_report(tokenizer, neutral_prompt, gp_date)
            wrong = next(letter for letter in sample_item.letters if letter != sample_item.answer)
            multi_prompt = _build(P, sample_item, multi, fmt, tokenizer, neutral_answer=wrong,
                                  date_string=date_string)
            frep["multi"] = _text_report(tokenizer, multi_prompt, gp_date)
        except Exception as e:  # noqa: BLE001
            frep["error"] = _err(e)
            if not (fmt == "native" and not has_template):
                problems.append(f"{fmt}: building prompts failed: {e}")
            continue
        for which in ("neutral", "multi"):
            if not frep[which].get("encode_ok"):
                problems.append(f"{fmt}/{which}: {frep[which].get('encode_error')}")
        if fmt == "native" and frep["neutral"]["contains_today_date"]:
            frep["date_string"] = gp_date

        if model is None or not items:
            continue
        try:
            prompts = [_build(P, item, neutral, fmt, tokenizer, date_string=date_string) for item in items]
            scores = scorer.score_prompts(model, tokenizer, prompts, "preflight", batch_size, variants=variants)
            ids = {i for item_letters in {it.letters for it in items}
                   for v in scorer.letter_token_ids(tokenizer, item_letters, variants).values() for i in v}
            diag = scorer.neutral_diagnostics(scores, tokenizer, letter_ids=ids, min_mass=min_mass)
            frep["neutral_diagnostics"] = diag
            diff = _max_batch_diff(model, tokenizer, prompts[:n_compare], batch_size, variants)
            frep["max_abs_batched_minus_unbatched"] = diff
            frep["n_compared"] = min(n_compare, len(prompts))
            if diff > max_batch_diff:
                problems.append(f"{fmt}: batched vs unbatched letter probs differ by {diff:.2e} > {max_batch_diff:g}")
            if diag["letter_mass_mean"] is not None and diag["letter_mass_mean"] < min_mass:
                problems.append(f"{fmt}: mean neutral letter_mass {diag['letter_mass_mean']:.3f} < {min_mass}")
        except Exception as e:  # noqa: BLE001
            frep["scoring_error"] = _err(e)
            problems.append(f"{fmt}: neutral scoring failed: {e}")
    return report
