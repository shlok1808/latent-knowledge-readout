"""Prompt building for the Orgad 2610.28325 replication (CONTRACT.md, prompts.py section).

Templates live in `configs/syco_templates.json`: all 14 sources verbatim from 2610.28325 v1
Table C3 (see papers/2610.28325_templates.md). Any source with `"verified": false` (none today,
kept for future sources) is refused unless `allow_unverified=True`. The tool source's text
wrapper is ours (`wrapper_verified: false`): tool prompts need `allow_unverified=True` or
`allow_tool_wrapper=True`.

Layout (Prompt C1). QB = question + options "(A) text", one per line:
- single-turn suggestion/doubt: QB, blank line, claim, blank line, instruction (Prompt C1).
- neutral (single-turn): QB + "\n" + instruction, no blank line. This is exactly the multi-turn
  first user turn (`_first_turn`), so the assistant letter in the multi-turn prompts is the model's
  own answer to the very same string.
- multi-turn suggestion/doubt: user = QB + "\n" + instruction (no blank line, per Prompt C1);
  assistant = the model's own neutral answer letter only (e.g. "B"); user = claim, blank line,
  instruction.
- answer_again (ours): multi-turn only; second user turn = "Please answer the question again.",
  blank line, instruction.

{option} rendering (option_style): "orgad" (default, Prompt C1) -> option B ("text"),
capitalized to "Option B (...)" when it starts a sentence (never inside the tool JSON);
"letter" -> (B); "letter_text" -> (B) text.

Typography (`typography`): "ascii" (default) keeps the config's ASCII apostrophe (don't) and ASCII
option quotes ("text"); "pdf" uses U+2019 (don’t) and U+201C/U+201D (“text”) as the PDF shows, for a
later robustness check. The config records the choice under its top-level "typography" field.

Tool source: {option} is substituted literally into the verbatim Table C3 JSON string (no
json.dumps, no escaping, no capitalisation), so it renders like the other 13 sources:
{"result":"the answer is option B ("oven")",...}. The result is then not strictly valid JSON when the
option text carries quotes (the orgad style always does); presumably the paper's prompt was the same.

Formats:
- "plain": one shared plain-text dialogue for all checkpoints:
      [System: ...\\n\\n]User: ...\\n\\nAssistant: B\\n\\nUser: ...\\n\\nAssistant:
  The text ends in "Assistant:" with no trailing space, so the answer is expected as the
  space-prefixed letter " A" (the scorer reads both "A" and " A"). The text has no BOS.
- "native": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
  date_string=...). date_string (default "26 Jul 2024") pins Llama-3.1's automatic
  "Today Date" system block so prompts are reproducible; templates that ignore it are unaffected.
  Some chat templates write BOS (Llama 3), some do not (Tulu 3). Templates end with the assistant
  header (Llama-3: "...<|end_header_id|>\\n\\n", Tulu: "<|assistant|>\\n"), so the answer is
  expected as the bare letter "A" (EXPECTED_LETTER_PREFIX).
Tokenization is not decided here: scorer.encode_text decides from the rendered text (a text that
starts with BOS is encoded without special tokens, any other text gets exactly one BOS), so every
prompt starts with exactly one BOS in both formats. `needs_special_tokens(fmt)` is a legacy
per-format rule that the scorer no longer uses (it would be wrong for Tulu 3's BOS-less template).
The tool source is plain text inside the user turn in both formats, never a native tool role.

Item text guard (`check_item_text`, called by build_prompt and build_cues in both formats): a
question or option text holding a newline followed by a plain role marker ("User:", "Assistant:",
"System:") would forge a dialogue turn in the plain format, and a newline inside an option text
breaks the one-option-per-line layout; both raise ItemTextError. items.py skips (role marker) or
repairs (option newline -> space) such rows when loading, so this only fires on hand-built Items.

Templates: `load_templates` validates the file and returns a deep copy (the cache is never
exposed). A dict passed as `templates=` to build_prompt / build_cues / available_sources is
validated the same way.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import warnings
from functools import lru_cache
from pathlib import Path

from knowledge_readout.syco.types import Cue, Item, Prompt

TEMPLATES_PATH = Path(__file__).resolve().parents[2] / "configs" / "syco_templates.json"

FORMATS = ("plain", "native")
TURNS = ("single", "multi")
KINDS = ("neutral", "answer_again", "suggestion", "doubt")
PRIORS = ("correct", "wrong")
OPTION_STYLES = ("orgad", "letter", "letter_text")
TYPOGRAPHIES = ("ascii", "pdf")
DEFAULT_DATE_STRING = "26 Jul 2024"
# typography -> (apostrophe, open quote, close quote) used when rendering
_TYPO = {"ascii": ("'", '"', '"'), "pdf": ("\u2019", "\u201c", "\u201d")}
PLAIN_ROLE_NAMES = {"system": "System", "user": "User", "assistant": "Assistant"}

# Which tokenization of the answer letter each format expects next (documentation; the scorer
# reads both variants regardless).
EXPECTED_LETTER_PREFIX = {"plain": " ", "native": ""}


class UnverifiedTemplateError(ValueError):
    """Raised when a prompt would use a template not yet checked against the paper."""


class ItemTextError(ValueError):
    """An item's text would break the prompt layout (forged role turn, newline inside an option)."""


# a newline, optional spaces/tabs, then a plain-format role name and a colon (see render_plain)
ROLE_MARKER = re.compile(r"[\r\n][ \t]*(?:" + "|".join(PLAIN_ROLE_NAMES.values()) + r")[ \t]*:")


def has_role_marker(text: str) -> bool:
    """True if `text` holds a newline followed by a plain role marker (User:/Assistant:/System:)."""
    return bool(ROLE_MARKER.search(text))


def check_item_text(item: Item) -> None:
    """Raise ItemTextError if the question or an option forges a plain role turn, or an option
    text contains a newline (options are rendered one per line). Same rule in both formats, so the
    item set never depends on the format."""
    for where, text in [("question", item.question)] + [(f"option {L}", t) for L, t in item.options]:
        if has_role_marker(text):
            raise ItemTextError(f"{item.item_id}: {where} holds a newline + plain role marker "
                                f"({ROLE_MARKER.search(text).group().strip()!r}); it would forge a dialogue turn")
    for letter, text in item.options:
        if "\n" in text or "\r" in text:
            raise ItemTextError(f"{item.item_id}: option {letter} contains a newline; options are rendered one per line")


@lru_cache(maxsize=8)
def _load_templates_cached(path: str) -> dict:
    """Validated config, cached per path. PRIVATE: callers that might mutate get a deep copy."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    _validate_templates(cfg, path)
    return cfg


def load_templates(path: str | Path | None = None, typography: str = "ascii") -> dict:
    """Load and validate the template config (cached per path) and return a DEEP COPY, so changing
    the result never changes later prompts. typography="pdf" uses U+2019 for the apostrophe (see
    `apply_typography`)."""
    cfg = _load_templates_cached(str(Path(path) if path is not None else TEMPLATES_PATH))
    return copy.deepcopy(apply_typography(cfg, typography))


def _templates(templates: dict | None) -> dict:
    """Read-only config for internal use: a caller's dict is validated like the file; None -> the
    cached default (not copied, never handed out)."""
    if templates is None:
        return _load_templates_cached(str(TEMPLATES_PATH))
    _validate_templates(templates, "templates argument")
    return templates


def _check_typography(typography: str) -> None:
    if typography not in TYPOGRAPHIES:
        raise ValueError(f"unknown typography {typography!r}; expected one of {TYPOGRAPHIES}")


def apply_typography(cfg: dict, typography: str = "ascii") -> dict:
    """"ascii": cfg unchanged (same object). "pdf": deep copy with every ASCII apostrophe in the
    instruction, answer_again, and source suggestion/doubt templates replaced by U+2019. Option quotes
    are applied at render time (`_render_option`). Item text (questions, options) is never touched."""
    _check_typography(typography)
    if typography == "ascii":
        return cfg
    apos = _TYPO[typography][0]
    out = copy.deepcopy(cfg)
    for key in ("instruction", "answer_again"):
        out[key] = out[key].replace("'", apos)
    for entry in out["sources"].values():
        for kind in ("suggestion", "doubt"):
            entry[kind] = entry[kind].replace("'", apos)
    return out


def _validate_templates(cfg: dict, path: str) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: templates must be a dict, got {type(cfg).__name__}")
    for key in ("instruction", "answer_again", "sources"):
        if key not in cfg:
            raise ValueError(f"{path}: missing key {key!r}")
    for key in ("instruction", "answer_again"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{path}: {key!r} must be a non-empty string")
    if not isinstance(cfg["sources"], dict) or not cfg["sources"]:
        raise ValueError(f"{path}: 'sources' must be a non-empty dict")
    for name, entry in cfg["sources"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: source {name!r} must be a dict")
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
            if not isinstance(entry["wrapper"], str) or "{result}" not in entry["wrapper"]:
                raise ValueError(f"{path}: source {name!r} wrapper must contain {{result}}")
            if not isinstance(entry.get("wrapper_verified"), bool):
                raise ValueError(f"{path}: source {name!r} needs a boolean 'wrapper_verified'")
    typo = cfg.get("typography")
    if typo is not None:
        if not isinstance(typo, dict) or typo.get("apostrophe") not in TYPOGRAPHIES \
                or typo.get("option_quotes") not in TYPOGRAPHIES:
            raise ValueError(f"{path}: 'typography' must be a dict with apostrophe/option_quotes in "
                             f"{TYPOGRAPHIES}")


def available_sources(templates: dict | None = None, verified_only: bool = False) -> tuple:
    cfg = _templates(templates)
    return tuple(n for n, e in cfg["sources"].items() if e["verified"] or not verified_only)


def needs_special_tokens(fmt: str) -> bool:
    """LEGACY, unused by the scorer: plain -> True, native -> False. Wrong for chat templates that
    write no BOS (Tulu 3); scorer.encode_text decides from the rendered text instead."""
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
    check_item_text(item)
    if isinstance(sources, str) or isinstance(turns, str):
        raise TypeError("sources and turns must be sequences, not a single string")
    cfg = _templates(templates)
    for s in sources:
        if s not in cfg["sources"]:
            raise ValueError(f"unknown source {s!r}; known: {tuple(cfg['sources'])}")
    for t in turns:
        if t not in TURNS:
            raise ValueError(f"unknown turn {t!r}; expected one of {TURNS}")

    if prior == "correct":
        from knowledge_readout.syco.items import wrong_option  # lazy: built in parallel
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


def _render_option(item: Item, letter: str, option_style: str, sentence_start: bool = False,
                   typography: str = "ascii") -> str:
    if option_style == "orgad":
        _, lq, rq = _TYPO[typography]
        text = f'option {letter} ({lq}{dict(item.options)[letter]}{rq})'
        return text[0].upper() + text[1:] if sentence_start else text
    if option_style == "letter":
        return f"({letter})"
    if option_style == "letter_text":
        return f"({letter}) {dict(item.options)[letter]}"
    raise ValueError(f"unknown option_style {option_style!r}; expected one of {OPTION_STYLES}")


_SENTENCE_END = re.compile(r"[.!?]\s*$")


def _fill_option(template: str, item: Item, letter: str, option_style: str,
                 typography: str = "ascii") -> str:
    """Replace every {option}; it counts as sentence-initial at the string start or after . ! ?"""
    parts = template.split("{option}")
    out = [parts[0]]
    for rest in parts[1:]:
        before = "".join(out)
        start = before.strip() == "" or bool(_SENTENCE_END.search(before))
        out.append(_render_option(item, letter, option_style, sentence_start=start,
                                  typography=typography))
        out.append(rest)
    return "".join(out)


def _fill_json(template: str, item: Item, letter: str, option_style: str,
               typography: str = "ascii") -> str:
    """Substitute {option} LITERALLY into the verbatim Table C3 JSON template string: no parsing, no
    json.dumps, no quote escaping, no sentence-initial capitalisation. The option renders exactly as
    for the other 13 sources, e.g. {"result":"the answer is option B ("oven")",...}.
    NOTE: the result is not strictly valid JSON when the rendered option contains quotes (the orgad
    style always does); the paper's prompt presumably looked the same, since Table C3 shows the
    template with {option} filled in place."""
    return template.replace("{option}", _render_option(item, letter, option_style,
                                                       sentence_start=False, typography=typography))


def _question_block(item: Item) -> str:
    return "\n".join([item.question] + [f"({letter}) {text}" for letter, text in item.options])


def _first_turn(qblock: str, instruction: str) -> str:
    """The neutral prompt AND the multi-turn first user turn (Prompt C1): QB + newline + instruction."""
    return f"{qblock}\n{instruction}"


def _claim(item: Item, cue: Cue, cfg: dict, option_style: str, allow_unverified: bool,
           allow_tool_wrapper: bool = False, typography: str = "ascii") -> str:
    entry = cfg["sources"].get(cue.source)
    if entry is None:
        raise ValueError(f"unknown source {cue.source!r}; known: {tuple(cfg['sources'])}")
    if not entry["verified"]:
        if not allow_unverified:
            raise UnverifiedTemplateError(
                f"source {cue.source!r} template is unverified (placeholder, not the paper's Table C3 "
                f"wording). Paste the real wording into {TEMPLATES_PATH.name} and set verified=true, "
                f"or pass allow_unverified=True for plumbing tests only.")
        warnings.warn(f"building prompt from UNVERIFIED template for source {cue.source!r}", stacklevel=3)
    if cue.target not in item.letters:
        raise ValueError(f"{item.item_id}: cue target {cue.target!r} not in {item.letters}")
    tmpl = entry[cue.kind]
    if entry.get("format") == "json":
        claim = _fill_json(tmpl, item, cue.target, option_style, typography)
    else:
        claim = _fill_option(tmpl, item, cue.target, option_style, typography)
    if "wrapper" in entry:
        if not entry["wrapper_verified"] and not (allow_unverified or allow_tool_wrapper):
            raise UnverifiedTemplateError(
                f"source {cue.source!r}: the plain-text wrapper {entry['wrapper']!r} is ours, not the "
                f"paper's. Pass allow_tool_wrapper=True (or allow_unverified=True) to use it.")
        claim = entry["wrapper"].replace("{result}", claim)
    return claim


def _messages(item: Item, cue: Cue, cfg: dict, neutral_answer, system, option_style, allow_unverified,
              allow_tool_wrapper=False, typography="ascii"):
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
        msgs.append({"role": "user", "content": _first_turn(qblock, instruction)})
        return msgs

    if cue.kind == "answer_again":
        if cue.turn != "multi":
            raise ValueError("answer_again cue is multi-turn only")
        second = f"{cfg['answer_again']}\n\n{instruction}"
    else:
        claim = _claim(item, cue, cfg, option_style, allow_unverified, allow_tool_wrapper, typography)
        if cue.turn == "single":
            msgs.append({"role": "user", "content": f"{qblock}\n\n{claim}\n\n{instruction}"})
            return msgs
        second = f"{claim}\n\n{instruction}"

    if neutral_answer is None:
        raise ValueError(f"{cue.kind}/multi needs neutral_answer (the model's own neutral letter)")
    if neutral_answer not in item.letters:
        raise ValueError(f"{item.item_id}: neutral_answer {neutral_answer!r} not in {item.letters}")
    msgs.append({"role": "user", "content": _first_turn(qblock, instruction)})
    msgs.append({"role": "assistant", "content": neutral_answer})
    msgs.append({"role": "user", "content": second})
    return msgs


def render_plain(messages) -> str:
    """Shared plain-text dialogue; ends with "Assistant:" so the next token is the letter."""
    parts = [f"{PLAIN_ROLE_NAMES[m['role']]}: {m['content']}" for m in messages]
    parts.append("Assistant:")
    return "\n\n".join(parts)


def render_native(messages, tokenizer, date_string: str | None = DEFAULT_DATE_STRING) -> str:
    """apply_chat_template(..., add_generation_prompt=True). date_string is passed as a template kwarg
    (Llama-3.1 reads it for its automatic "Today Date" system block; other templates ignore it), so
    the rendered prompt does not depend on the day it is built. None omits it."""
    if tokenizer is None:
        raise ValueError("native format needs a tokenizer")
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "native format refused: tokenizer has no chat_template (base model?). Use fmt='plain'.")
    extra = {} if date_string is None else {"date_string": date_string}
    text = tokenizer.apply_chat_template(
        [dict(m) for m in messages], tokenize=False, add_generation_prompt=True, **extra)
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, expected str")
    return text


def build_prompt(item: Item, cue: Cue, fmt: str, tokenizer=None, neutral_answer=None, system=None,
                 option_style: str = "orgad", allow_unverified: bool = False,
                 templates: dict | None = None, allow_tool_wrapper: bool = False,
                 typography: str = "ascii", date_string: str | None = DEFAULT_DATE_STRING) -> Prompt:
    """typography: "ascii" (default) | "pdf" (U+2019 apostrophe, U+201C/D option quotes).
    date_string: passed to the chat template in the native format only (see render_native)."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; expected one of {FORMATS}")
    if option_style not in OPTION_STYLES:
        raise ValueError(f"unknown option_style {option_style!r}; expected one of {OPTION_STYLES}")
    _check_typography(typography)
    check_item_text(item)
    cfg = apply_typography(_templates(templates), typography)
    msgs = _messages(item, cue, cfg, neutral_answer, system, option_style, allow_unverified,
                     allow_tool_wrapper, typography)
    text = render_plain(msgs) if fmt == "plain" else render_native(msgs, tokenizer, date_string)
    return Prompt(item_id=item.item_id, cue=cue, messages=tuple(msgs), text=text, fmt=fmt,
                  letters=item.letters)


def prompt_sha256(prompt) -> str:
    """Hex SHA-256 of the rendered prompt text (UTF-8). Accepts a Prompt or a str."""
    text = prompt.text if isinstance(prompt, Prompt) else prompt
    if not isinstance(text, str):
        raise TypeError(f"expected Prompt or str, got {type(prompt).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
