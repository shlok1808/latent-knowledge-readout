"""Prompt building for the Orgad 2610.28325 replication (CONTRACT.md, prompts.py section).

Templates live in `configs/syco_templates.json`: all 14 sources verbatim from 2610.28325 v1
Table C3 (see papers/2610.28325_templates.md). Any source with `"verified": false` (none today,
kept for future sources) is refused unless `allow_unverified=True`. The tool source's text
wrapper is ours (`wrapper_verified: false`): tool prompts need `allow_unverified=True` or
`allow_tool_wrapper=True`.

Layout (Prompt C1). QB = question + options "(A) text", one per line:
- single-turn suggestion/doubt: QB, blank line, claim, blank line, instruction.
- neutral (single-turn, our layout choice): QB, blank line, instruction.
- multi-turn suggestion/doubt: user = QB + "\n" + instruction (no blank line, per Prompt C1);
  assistant = the model's own neutral answer letter only (e.g. "B"); user = claim, blank line,
  instruction.
- answer_again (ours): multi-turn only; second user turn = "Please answer the question again.",
  blank line, instruction.

{option} rendering (option_style): "orgad" (default, Prompt C1) -> option B ("text"),
capitalized to "Option B (...)" when it starts a sentence; "letter" -> (B); "letter_text" -> (B) text.

Formats:
- "plain": one shared plain-text dialogue for all checkpoints:
      [System: ...\\n\\n]User: ...\\n\\nAssistant: B\\n\\nUser: ...\\n\\nAssistant:
  The text ends in "Assistant:" with no trailing space, so the answer is expected as the
  space-prefixed letter " A" (the scorer reads both "A" and " A"). Tokenize with
  add_special_tokens=True so the tokenizer adds BOS (the text has none).
- "native": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).
  Chat templates already insert BOS, so tokenize with add_special_tokens=False or BOS doubles.
  Templates end with the assistant header (Llama-3: "...<|end_header_id|>\\n\\n", Tulu: "<|assistant|>\\n"),
  so the answer is expected as the bare letter "A". Use `needs_special_tokens(fmt)`.
The tool source is plain text inside the user turn in both formats, never a native tool role.
"""
from __future__ import annotations

import json
import re
import warnings
from functools import lru_cache
from pathlib import Path

from failure_probes.syco.types import Cue, Item, Prompt

TEMPLATES_PATH = Path(__file__).resolve().parents[2] / "configs" / "syco_templates.json"

FORMATS = ("plain", "native")
TURNS = ("single", "multi")
KINDS = ("neutral", "answer_again", "suggestion", "doubt")
PRIORS = ("correct", "wrong")
OPTION_STYLES = ("orgad", "letter", "letter_text")
PLAIN_ROLE_NAMES = {"system": "System", "user": "User", "assistant": "Assistant"}

# Which tokenization of the answer letter each format expects next (documentation; the scorer
# reads both variants regardless).
EXPECTED_LETTER_PREFIX = {"plain": " ", "native": ""}


class UnverifiedTemplateError(ValueError):
    """Raised when a prompt would use a template not yet checked against the paper."""


@lru_cache(maxsize=8)
def _load_templates_cached(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    _validate_templates(cfg, path)
    return cfg


def load_templates(path: str | Path | None = None) -> dict:
    """Load and validate the template config (cached per path)."""
    return _load_templates_cached(str(Path(path) if path is not None else TEMPLATES_PATH))


def _validate_templates(cfg: dict, path: str) -> None:
    for key in ("instruction", "answer_again", "sources"):
        if key not in cfg:
            raise ValueError(f"{path}: missing key {key!r}")
    for name, entry in cfg["sources"].items():
        if "verified" not in entry or not isinstance(entry["verified"], bool):
            raise ValueError(f"{path}: source {name!r} needs a boolean 'verified'")
        for kind in ("suggestion", "doubt"):
            tmpl = entry.get(kind)
            if not isinstance(tmpl, str) or "{option}" not in tmpl:
                raise ValueError(f"{path}: source {name!r} {kind} template must contain {{option}}")
            if entry["verified"] and "PLACEHOLDER" in tmpl.upper():
                raise ValueError(f"{path}: source {name!r} is marked verified but holds a placeholder")
            if entry.get("format") == "json":
                try:
                    json.loads(tmpl)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}: source {name!r} {kind} is not valid JSON: {e}") from e
        if "wrapper" in entry:
            if "{result}" not in entry["wrapper"]:
                raise ValueError(f"{path}: source {name!r} wrapper must contain {{result}}")
            if not isinstance(entry.get("wrapper_verified"), bool):
                raise ValueError(f"{path}: source {name!r} needs a boolean 'wrapper_verified'")


def available_sources(templates: dict | None = None, verified_only: bool = False) -> tuple:
    cfg = templates if templates is not None else load_templates()
    return tuple(n for n, e in cfg["sources"].items() if e["verified"] or not verified_only)


def needs_special_tokens(fmt: str) -> bool:
    """add_special_tokens for tokenizing Prompt.text: plain -> True (text has no BOS),
    native -> False (the chat template already wrote BOS; adding it again doubles it)."""
    if fmt == "plain":
        return True
    if fmt == "native":
        return False
    raise ValueError(f"unknown fmt {fmt!r}; expected one of {FORMATS}")


def neutral_cue() -> Cue:
    return Cue(kind="neutral", source="none", turn="single", target=None, prior=None)


def answer_again_cue(prior: str) -> Cue:
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
    return Cue(kind="answer_again", source="none", turn="multi", target=None, prior=prior)


def _check_prior(item: Item, prior: str, neutral_answer: str) -> None:
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}, got {prior!r}")
    if neutral_answer not in item.letters:
        raise ValueError(f"{item.item_id}: neutral_answer {neutral_answer!r} not in {item.letters}")
    expected = "correct" if neutral_answer == item.answer else "wrong"
    if prior != expected:
        raise ValueError(
            f"{item.item_id}: prior={prior!r} inconsistent with neutral_answer={neutral_answer!r} "
            f"vs gold {item.answer!r} (expected {expected!r})")


def build_cues(item: Item, prior: str, neutral_answer: str, sources, turns, seed: int,
               *, include_answer_again: bool = False, templates: dict | None = None) -> list:
    """Suggestion and doubt cues for every (source, turn), in that nesting order.

    prior correct: suggestion -> wrong_option(item, seed), doubt -> neutral_answer (= gold).
    prior wrong:   suggestion -> item.answer,              doubt -> neutral_answer (model's wrong answer).
    With include_answer_again=True and "multi" in turns, one answer_again cue is appended.
    The neutral cue is not included (see `neutral_cue()`).
    """
    _check_prior(item, prior, neutral_answer)
    if isinstance(sources, str) or isinstance(turns, str):
        raise TypeError("sources and turns must be sequences, not a single string")
    cfg = templates if templates is not None else load_templates()
    for s in sources:
        if s not in cfg["sources"]:
            raise ValueError(f"unknown source {s!r}; known: {tuple(cfg['sources'])}")
    for t in turns:
        if t not in TURNS:
            raise ValueError(f"unknown turn {t!r}; expected one of {TURNS}")

    if prior == "correct":
        from failure_probes.syco.items import wrong_option  # lazy: built in parallel
        suggest_target = wrong_option(item, seed)
        if suggest_target == item.answer or suggest_target not in item.letters:
            raise ValueError(f"{item.item_id}: wrong_option returned {suggest_target!r}")
    else:
        suggest_target = item.answer
    doubt_target = neutral_answer

    cues = []
    for s in sources:
        for t in turns:
            cues.append(Cue(kind="suggestion", source=s, turn=t, target=suggest_target, prior=prior))
            cues.append(Cue(kind="doubt", source=s, turn=t, target=doubt_target, prior=prior))
    if include_answer_again and "multi" in turns:
        cues.append(answer_again_cue(prior))
    return cues


def _render_option(item: Item, letter: str, option_style: str, sentence_start: bool = False) -> str:
    if option_style == "orgad":
        text = f'option {letter} ("{dict(item.options)[letter]}")'
        return text[0].upper() + text[1:] if sentence_start else text
    if option_style == "letter":
        return f"({letter})"
    if option_style == "letter_text":
        return f"({letter}) {dict(item.options)[letter]}"
    raise ValueError(f"unknown option_style {option_style!r}; expected one of {OPTION_STYLES}")


_SENTENCE_END = re.compile(r"[.!?]\s*$")


def _fill_option(template: str, item: Item, letter: str, option_style: str) -> str:
    """Replace every {option}; it counts as sentence-initial at the string start or after . ! ?"""
    parts = template.split("{option}")
    out = [parts[0]]
    for rest in parts[1:]:
        before = "".join(out)
        start = before.strip() == "" or bool(_SENTENCE_END.search(before))
        out.append(_render_option(item, letter, option_style, sentence_start=start))
        out.append(rest)
    return "".join(out)


def _fill_json(template: str, item: Item, letter: str, option_style: str) -> str:
    """Fill {option} inside each JSON string value, then re-serialize compactly (escapes quotes)."""
    obj = json.loads(template)
    filled = {k: _fill_option(v, item, letter, option_style) if isinstance(v, str) else v
              for k, v in obj.items()}
    return json.dumps(filled, separators=(",", ":"), ensure_ascii=False)


def _question_block(item: Item) -> str:
    return "\n".join([item.question] + [f"({letter}) {text}" for letter, text in item.options])


def _claim(item: Item, cue: Cue, cfg: dict, option_style: str, allow_unverified: bool,
           allow_tool_wrapper: bool = False) -> str:
    entry = cfg["sources"].get(cue.source)
    if entry is None:
        raise ValueError(f"unknown source {cue.source!r}; known: {tuple(cfg['sources'])}")
    if not entry["verified"]:
        if not allow_unverified:
            raise UnverifiedTemplateError(
                f"source {cue.source!r} template is unverified (placeholder, not the paper's Table C5 "
                f"wording). Paste the real wording into {TEMPLATES_PATH.name} and set verified=true, "
                f"or pass allow_unverified=True for plumbing tests only.")
        warnings.warn(f"building prompt from UNVERIFIED template for source {cue.source!r}", stacklevel=3)
    if cue.target not in item.letters:
        raise ValueError(f"{item.item_id}: cue target {cue.target!r} not in {item.letters}")
    tmpl = entry[cue.kind]
    if entry.get("format") == "json":
        claim = _fill_json(tmpl, item, cue.target, option_style)
    else:
        claim = _fill_option(tmpl, item, cue.target, option_style)
    if "wrapper" in entry:
        if not entry["wrapper_verified"] and not (allow_unverified or allow_tool_wrapper):
            raise UnverifiedTemplateError(
                f"source {cue.source!r}: the plain-text wrapper {entry['wrapper']!r} is ours, not the "
                f"paper's. Pass allow_tool_wrapper=True (or allow_unverified=True) to use it.")
        claim = entry["wrapper"].replace("{result}", claim)
    return claim


def _messages(item: Item, cue: Cue, cfg: dict, neutral_answer, system, option_style, allow_unverified,
              allow_tool_wrapper=False):
    if cue.kind not in KINDS:
        raise ValueError(f"unknown cue kind {cue.kind!r}; expected one of {KINDS}")
    if cue.turn not in TURNS:
        raise ValueError(f"unknown turn {cue.turn!r}; expected one of {TURNS}")
    instruction = cfg["instruction"]
    qblock = _question_block(item)
    msgs = [] if system is None else [{"role": "system", "content": system}]

    if cue.kind == "neutral":
        if cue.turn != "single":
            raise ValueError("neutral cue is single-turn only")
        msgs.append({"role": "user", "content": f"{qblock}\n\n{instruction}"})
        return msgs

    if cue.kind == "answer_again":
        if cue.turn != "multi":
            raise ValueError("answer_again cue is multi-turn only")
        second = f"{cfg['answer_again']}\n\n{instruction}"
    else:
        claim = _claim(item, cue, cfg, option_style, allow_unverified, allow_tool_wrapper)
        if cue.turn == "single":
            msgs.append({"role": "user", "content": f"{qblock}\n\n{claim}\n\n{instruction}"})
            return msgs
        second = f"{claim}\n\n{instruction}"

    if neutral_answer is None:
        raise ValueError(f"{cue.kind}/multi needs neutral_answer (the model's own neutral letter)")
    if neutral_answer not in item.letters:
        raise ValueError(f"{item.item_id}: neutral_answer {neutral_answer!r} not in {item.letters}")
    msgs.append({"role": "user", "content": f"{qblock}\n{instruction}"})
    msgs.append({"role": "assistant", "content": neutral_answer})
    msgs.append({"role": "user", "content": second})
    return msgs


def render_plain(messages) -> str:
    """Shared plain-text dialogue; ends with "Assistant:" so the next token is the letter."""
    parts = [f"{PLAIN_ROLE_NAMES[m['role']]}: {m['content']}" for m in messages]
    parts.append("Assistant:")
    return "\n\n".join(parts)


def render_native(messages, tokenizer) -> str:
    if tokenizer is None:
        raise ValueError("native format needs a tokenizer")
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "native format refused: tokenizer has no chat_template (base model?). Use fmt='plain'.")
    text = tokenizer.apply_chat_template(
        [dict(m) for m in messages], tokenize=False, add_generation_prompt=True)
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, expected str")
    return text


def build_prompt(item: Item, cue: Cue, fmt: str, tokenizer=None, neutral_answer=None, system=None,
                 option_style: str = "orgad", allow_unverified: bool = False,
                 templates: dict | None = None, allow_tool_wrapper: bool = False) -> Prompt:
    if fmt not in FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; expected one of {FORMATS}")
    if option_style not in OPTION_STYLES:
        raise ValueError(f"unknown option_style {option_style!r}; expected one of {OPTION_STYLES}")
    cfg = templates if templates is not None else load_templates()
    msgs = _messages(item, cue, cfg, neutral_answer, system, option_style, allow_unverified,
                     allow_tool_wrapper)
    text = render_plain(msgs) if fmt == "plain" else render_native(msgs, tokenizer)
    return Prompt(item_id=item.item_id, cue=cue, messages=tuple(msgs), text=text, fmt=fmt,
                  letters=item.letters)
