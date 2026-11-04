"""Item loaders for the Orgad 2610.28325 replication (CONTRACT.md, items.py section).

Pure row converters turn one Hugging Face row into an `Item` (or None when the row is malformed).
`load_items` wraps `datasets.load_dataset` (imported lazily so the rest of the package works without it).
`wrong_option` picks a deterministic random wrong letter (OPEN DETAIL #5).

Lettering rule (all three datasets): options are put in label order (letters alphabetically, ARC's
numeric labels "1".."5" numerically) and relettered contiguously from "A"; the gold is relettered the same way.
Labels must be ASCII: one ASCII letter, or ASCII decimal digits; anything else (e.g. "²", "٣", "É")
is skipped as "mixed_labels", never a crash.

Text hygiene (prompts render one option per line in a plain "User:/Assistant:" dialogue):
- a question or option text holding a newline followed by a plain role marker ("User:",
  "Assistant:", "System:") is skipped as "role_marker_in_text" (it would forge a dialogue turn);
- newlines inside an option text are replaced by a space (counted in stats as
  "option_newlines_replaced", not a skip);
- two options with the same text (after that, whitespace runs collapsed; case kept, since ARC has
  options like "Co" vs "CO") are skipped as "duplicate_option_text".
"""
import hashlib
import logging
import re

from failure_probes.syco.prompts import has_role_marker
from failure_probes.syco.types import LETTERS, Item

log = logging.getLogger(__name__)

# dataset key -> (HF path, HF config or None, default split). OPEN DETAIL #2: splits stay configurable.
DATASETS = {
    "csqa": ("tau/commonsense_qa", None, "validation"),  # test split is unlabelled
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "obqa": ("allenai/openbookqa", "main", "test"),
}

SKIP_REASONS = (
    "bad_row", "missing_id", "missing_question", "missing_answer", "bad_choices", "empty_option_text",
    "duplicate_labels", "mixed_labels", "too_few_options", "too_many_options", "answer_not_in_labels",
    "role_marker_in_text", "duplicate_option_text",
)
# non-skip counters that may also appear in `stats`
COUNTERS = ("option_newlines_replaced",)
# has_role_marker comes from prompts (the plain format's role names)
_NEWLINES = re.compile(r"\s*[\r\n]+\s*")


class _Malformed(Exception):
    def __init__(self, reason):
        assert reason in SKIP_REASONS, reason
        super().__init__(reason)
        self.reason = reason


def _clean(value):
    return value.strip() if isinstance(value, str) else ""


def _label_key(label):
    """Sort key for one normalized label: ("num", int), ("let", str), or ("bad", str) for anything
    that is not ASCII decimal digits or one ASCII letter (unicode digits such as "²" or "٣" are bad:
    str.isdigit() accepts "²" but int() cannot parse it)."""
    if not label.isascii():
        return ("bad", label)
    if label.isdecimal():
        return ("num", int(label))
    if len(label) == 1 and label.isalpha():
        return ("let", label)
    return ("bad", label)



def _letter_options(choices, answer_key, counters=None):
    """Return (options, gold) or raise _Malformed. `choices` is HF's {"label": [...], "text": [...]}."""
    if not isinstance(choices, dict):
        raise _Malformed("bad_choices")
    labels, texts = choices.get("label"), choices.get("text")
    if labels is None or texts is None or isinstance(labels, str) or isinstance(texts, str):
        raise _Malformed("bad_choices")
    labels, texts = list(labels), list(texts)
    if len(labels) != len(texts):
        raise _Malformed("bad_choices")
    labels = [_clean(lab).upper() for lab in labels]
    texts = [_clean(t) for t in texts]
    if any(not lab for lab in labels):
        raise _Malformed("bad_choices")
    if any(not t for t in texts):
        raise _Malformed("empty_option_text")
    if any(has_role_marker(t) for t in texts):
        raise _Malformed("role_marker_in_text")
    n_newline = sum(1 for t in texts if _NEWLINES.search(t))
    texts = [_NEWLINES.sub(" ", t) for t in texts]
    if len({" ".join(t.split()) for t in texts}) != len(texts):  # exact text, whitespace runs collapsed
        raise _Malformed("duplicate_option_text")
    if len(set(labels)) != len(labels):
        raise _Malformed("duplicate_labels")
    keys = [_label_key(lab) for lab in labels]
    kinds = {k[0] for k in keys}
    if "bad" in kinds or len(kinds) > 1:
        raise _Malformed("mixed_labels")
    if len(labels) < 2:
        raise _Malformed("too_few_options")
    if len(labels) > len(LETTERS):
        raise _Malformed("too_many_options")
    gold_label = _clean(answer_key).upper()
    if not gold_label:
        raise _Malformed("missing_answer")
    if gold_label not in labels:
        raise _Malformed("answer_not_in_labels")
    order = sorted(range(len(labels)), key=lambda i: keys[i])
    options, gold = [], None
    for new_idx, old_idx in enumerate(order):
        letter = LETTERS[new_idx]
        options.append((letter, texts[old_idx]))
        if labels[old_idx] == gold_label:
            gold = letter
    if counters is not None and n_newline:
        counters["option_newlines_replaced"] = counters.get("option_newlines_replaced", 0) + n_newline
    return tuple(options), gold


def _convert(row, split, dataset, question_field, counters=None):
    if not isinstance(row, dict):
        raise _Malformed("bad_row")
    native_id = _clean(row.get("id"))
    if not native_id:
        raise _Malformed("missing_id")
    question = _clean(row.get(question_field))
    if not question:
        raise _Malformed("missing_question")
    if has_role_marker(question):
        raise _Malformed("role_marker_in_text")
    counted = {}
    options, gold = _letter_options(row.get("choices"), row.get("answerKey"), counted)
    if counters is not None:  # only rows that are kept count
        for key, n in counted.items():
            counters[key] = counters.get(key, 0) + n
    return Item(item_id=f"{dataset}:{split}:{native_id}", dataset=dataset, split=split,
                question=question, options=options, answer=gold)


def _safe(row, split, dataset, question_field, stats=None):
    try:
        return _convert(row, split, dataset, question_field, stats)
    except _Malformed as exc:
        if stats is not None:
            stats[exc.reason] = stats.get(exc.reason, 0) + 1
        return None


def item_from_csqa_row(row, split, stats=None):
    """tau/commonsense_qa row (id, question, question_concept, choices{label,text}, answerKey) -> Item | None."""
    return _safe(row, split, "csqa", "question", stats)


def item_from_arc_row(row, split, stats=None):
    """allenai/ai2_arc ARC-Challenge row (id, question, choices{text,label}, answerKey) -> Item | None.
    Numeric labels "1".."4" map to A..D; 3- and 5-option items are kept."""
    return _safe(row, split, "arc_challenge", "question", stats)


def item_from_obqa_row(row, split, stats=None):
    """allenai/openbookqa main row (id, question_stem, choices{text,label}, answerKey) -> Item | None."""
    return _safe(row, split, "obqa", "question_stem", stats)


_CONVERTERS = {
    "csqa": item_from_csqa_row,
    "arc_challenge": item_from_arc_row,
    "obqa": item_from_obqa_row,
}


def load_items(dataset, split=None, limit=None, stats=None):
    """Load one dataset as a list of Items, in the dataset's native row order (deterministic).

    split: defaults to DATASETS[dataset][2]. limit: keep the first `limit` valid items (skipped rows
    don't count). stats: optional dict, filled with {"rows", "kept", "skipped", <reason>: n} plus
    the non-skip counter "option_newlines_replaced" when it is non-zero.
    Raises ValueError on an unknown dataset or a duplicate item_id.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    path, config, default_split = DATASETS[dataset]
    split = split or default_split
    import datasets  # lazy: keep the package importable without `datasets`

    if config is None:
        rows = datasets.load_dataset(path, split=split)
    else:
        rows = datasets.load_dataset(path, config, split=split)

    convert = _CONVERTERS[dataset]
    reasons = {}
    items, seen = [], set()
    n_rows = 0
    for row in rows:
        if limit is not None and len(items) >= limit:
            break
        n_rows += 1
        item = convert(row, split, reasons)
        if item is None:
            continue
        if item.item_id in seen:
            raise ValueError(f"duplicate item_id {item.item_id!r} in {path} {split}")
        seen.add(item.item_id)
        items.append(item)
    skipped = sum(n for reason, n in reasons.items() if reason in SKIP_REASONS)
    if skipped:
        log.warning("%s/%s: skipped %d of %d rows as malformed: %s", dataset, split, skipped, n_rows, reasons)
    if stats is not None:
        stats.update(reasons)
        stats.update({"rows": n_rows, "kept": len(items), "skipped": skipped})
    return items


def wrong_option(item, seed):
    """Deterministic random wrong letter for `item`: sha256 of (item_id, seed), never the gold.

    Uniform over the item's wrong letters up to a negligible modulo bias (2**64 mod k)."""
    wrong = [letter for letter in item.letters if letter != item.answer]
    if not wrong:
        raise ValueError(f"item {item.item_id!r} has no wrong option")
    digest = hashlib.sha256(f"{item.item_id}\x1f{seed}".encode("utf-8")).digest()
    return wrong[int.from_bytes(digest[:8], "big") % len(wrong)]
