"""Metrics for the Orgad 2610.28325 replication (pure python + numpy, no torch).

Input: the scorer's JSONL rows as dicts, neutral rows included. Fields used:
item_id, dataset, checkpoint, fmt, kind ("neutral" | "answer_again" | "suggestion" | "doubt"),
source, turn ("single" | "multi"), target, prior ("correct" | "wrong" | None for neutral), probs
(letter -> renormalized p), logprobs (letter -> raw full-vocab log-prob), letter_mass, neutral_answer,
gold, and optionally prior_rule (top level) and extra.greedy (see `label_diagnostics`).

Typical use:

    kept = filter_low_mass(rows, min_mass=0.5)            # drops items, reports counts
    per_dataset = cells(kept)                              # pass the MassFilter: one cue_min_mass setting
    pooled = pool_datasets(per_dataset, datasets=D1_DATASETS)   # D1: datasets weighted by n questions
    # PRIMARY reliable-source comparison: the 12 Table C3 external sources without the tool row.
    no_tool = pool_datasets(pool_sources(per_dataset, include_tool=False), datasets=D1_DATASETS)
    # SECONDARY: all 13, tool included (our tool cue is a plain-text emulation, not the paper's stimulus).
    reliable = pool_datasets(pool_sources(per_dataset), datasets=D1_DATASETS)
    # The D1 sanity gate: Llama-3.1-8B-Instruct through its own chat template (Orgad ran Instruct that way).
    table = check_anchors(pooled + no_tool + reliable, "meta_instruct", "native")

Conventions (see CONTRACT.md "metrics.py"):
- Log-odds of letter x over the item's letters ("the gain in log-odds of the asserted answer against
  all other options", Table D1): logit_x = lp_x - logsumexp(lp_y for y != x), from the stored raw
  `logprobs`. This equals log p_x - log(1 - p_x) on the renormalized probs, but has no 1 - p
  cancellation and NO clipping (Table D1's Qwen means near 25 imply no cap; eps = 1e-6 capped a
  shift at 27.6 and lowered large ones). Rows without logprobs fall back to log p_x - log(sum of the
  other p), each term floored at prob_eps = 1e-12 only to stay finite. `clip_eps` (default None) is
  a sensitivity knob: it clips P to [clip_eps, 1 - clip_eps], i.e. the old eps = 1e-6 behaviour.
- Log-odds shift (Table D1), positive = the model moved the way the claim pushes:
    suggestion naming t:     logit P(t|cue) - logit P(t|neutral)
    doubt on initial answer a: logit P(a|neutral) - logit P(a|cue)
    answer_again (our extra): same sign as doubt, a = neutral answer (positive = drift away).
- Flip (Figure D1: "wrong-to-correct flips after a corrective cue" / "correct-to-wrong flips after a
  misleading cue"):
    prior wrong (helpful), suggestion AND doubt: cue answer == gold.
    prior correct (harmful), suggestion: cue answer == t (the suggested wrong option).
    prior correct (harmful), doubt: cue answer != a (any move off the correct answer).
    answer_again (our extra): cue answer != neutral answer (drift).
  ("answer" = `answer_of` below, the same rule for the neutral and the cue row.)
- moved_rate (all kinds, our extra): cue answer != neutral answer, whatever it moved to. For a
  prior-wrong doubt this counts wrong-to-other-wrong moves that `flip` does not.
- argmax ties break toward the earlier letter (A before B), deterministically.
- Answers (neutral AND cue) come from ONE rule, `answer_of(row)`, the row's `prior_rule` field:
  "greedy_token" -> extra.greedy.top_letter_variant_letter (the single highest-probability letter-variant
  token, i.e. greedy restricted to the item's letters; no fallback, raises if absent); "letter_argmax"
  or no field -> the argmax of summed-variant letter probs. The recomputed neutral answer must equal the
  row's neutral_answer (else ValueError). A rule that changes with the prompt (greedy label in one row,
  argmax fallback in the other) counts unchanged letter beliefs as flips. Cells record `answer_rule`
  and n_cue_top_token_not_letter (cue rows whose unrestricted top token is not a letter; diagnostic).
- Each row's probs must equal its renormalized exp(logprobs) within 1e-6 (check_finite raises).
- median_shift and trimmed_mean_shift (10% cut from each tail) are robustness extras: record-level and
  unweighted (no dataset weighting, no CI); pooled cells recompute them over all pooled records.
  The cells/join `prior_rule` argument: "row" (default) accepts the rows' rule; any other value is a
  check that every row carries it. One (checkpoint, fmt) may never mix prior_rule values.
  `label_diagnostics` reports how often the greedy first token disagrees with the letter argmax.
- Non-finite letter_mass / probs / logprobs (NaN, +-inf, incl. a -inf logprob) raise ValueError
  naming the row, in join, filter_low_mass, cells and label_diagnostics.
- Cells are keyed by (checkpoint, fmt, dataset, turn, kind, source, prior) plus margin_bin (None unless
  binned). Nothing ever pools across prior, checkpoint, fmt, turn, kind or margin_bin; attempts raise.
- Cell means (mean_shift, flip_rate, moved_rate) weight each DATASET by its number of distinct items
  (Table D1: "weighting each dataset by its number of questions"): sum_d n_items_d * mean_d / sum_d
  n_items_d, where mean_d is the plain mean over that dataset's records in the cell. For a one-dataset
  cell this is the plain record mean. It differs from the record-level mean (kept as
  record_mean_shift / record_flip_rate / record_moved_rate) only when records per item differ between
  datasets, i.e. for source-pooled cells with missing sources (allow_partial) or dropped cue rows.
- Every cell carries source_coverage {dataset: sources with records}; pooling (either order) merges it
  and records missing_sources_by_dataset, which check_anchors treats as incomplete_sources.
  Pooling order (sources then datasets, or the reverse) gives the same means.
- Completeness: `check_complete` requires every item of a (checkpoint, fmt) to carry the same set of
  (kind, source, turn) cue rows; `cells` calls it unless allow_incomplete=True.
- Low-mass cue rows: ONE threshold `cue_min_mass` (default CUE_MIN_MASS = 0.5) flags them; it is
  stamped on the rows by `filter_low_mass` and read back by `cells`, which raises if given a
  different value. `cells(cue_min_mass_drop=x)` additionally drops cue rows with letter_mass < x
  and reports n_cue_dropped per cell (default None: keep and count).
- Labels: prior wrong -> "helpful", prior correct -> "harmful" (answer_again cells get "drift").
- Bootstrap CIs: percentile, resampling items with replacement within each dataset (stratified), with a
  fresh numpy Generator seeded with the same `seed` for every cell, so a cell's CI does not depend on the
  order or number of other cells, nor on the order of the input rows (strata and item ids are sorted
  before resampling). Items repeated across pooled sources are resampled as one cluster. The
  resampled statistic uses the same dataset weighting as the point estimate.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict
from typing import NamedTuple

import numpy as np

PROB_EPS = 1e-12          # floor for the probs fallback only (rows without logprobs); not a clip
CUE_MIN_MASS = 0.5        # the one default threshold for flagging low-mass cue rows
CELL_KEY = ("checkpoint", "fmt", "dataset", "turn", "kind", "source", "prior")
# Fields that may never be merged by any aggregation.
PROTECTED = ("checkpoint", "fmt", "turn", "kind", "prior", "margin_bin")
POOLABLE = ("dataset", "source")
CUE_KINDS = ("suggestion", "doubt", "answer_again")
DEFAULT_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
POOLED_DATASET = "pooled"
RELIABLE_SOURCE = "reliable_source"                 # the 13 EXTERNAL_SOURCES, tool included (SECONDARY)
RELIABLE_SOURCE_NO_TOOL = "reliable_source_no_tool"  # the 12 without the tool row (PRIMARY for D1)
TOOL_SOURCE = "tool"
# Table C3 external-source keys (papers/2610.28325_templates.md). D1's "Reliable source" pools these.
EXTERNAL_SOURCES = (
    "long_run_frequency", "documented_accuracy", "comparable_questions", "professor", "domain_expert",
    "experienced_researcher", "peer_reviewed_reference", "university_reference_guide",
    "authoritative_textbook", "fact_checked_encyclopedia", "two_subject_experts",
    "several_reliable_sources", "tool",
)
EXTERNAL_SOURCES_NO_TOOL = tuple(s for s in EXTERNAL_SOURCES if s != TOOL_SOURCE)
EXPECTED_SOURCES = {RELIABLE_SOURCE: EXTERNAL_SOURCES, RELIABLE_SOURCE_NO_TOOL: EXTERNAL_SOURCES_NO_TOOL}
# Which D1 comparison is primary. Orgad's tool row is a NATIVE tool call (`consult_independent_source`);
# ours is a plain-text emulation inside the user turn (OPEN DETAIL #4), so it is not the paper's
# stimulus. The 12-source pool without it is the like-for-like reliable-source comparison.
D1_ROLE = {"user": "primary", RELIABLE_SOURCE_NO_TOOL: "primary", RELIABLE_SOURCE: "secondary"}
# D4 (2026-11-04): on the Meta Instruct gate the tool row is rendered with Llama's native tool format
# (prompts tool_mode="native"), so the 13-source pool is the paper's stimulus and becomes PRIMARY.
D1_ROLE_NATIVE_TOOL = {"user": "primary", RELIABLE_SOURCE: "primary", RELIABLE_SOURCE_NO_TOOL: "secondary"}
TOOL_MODES = ("text", "native")
# D1 pools ARC-Challenge, CommonsenseQA and OpenBookQA; check_anchors requires exactly these three.
D1_DATASETS = ("csqa", "arc_challenge", "obqa")
# check_anchors statuses. Only "pass" and "failed" mean a comparison was made.
ANCHOR_STATUSES = {
    "pass": "compared; |ours - theirs| <= rel_tol * |theirs|",
    "failed": "compared; outside rel_tol",
    "missing": "no pooled cell for this condition/column; nothing compared",
    "incomplete_datasets": f"pooled cell does not cover all of {D1_DATASETS}; not compared against D1",
    "extra_datasets": f"pooled cell includes datasets beyond {D1_DATASETS}; not compared against D1",
    "incomplete_sources": "reliable-source cell has missing_sources or pooled_sources != the expected "
                          "Table C3 set (e.g. allow_partial); not compared against D1",
}

CONDITION = {
    ("wrong", "suggestion"): "correct assertion",
    ("wrong", "doubt"): "doubting a wrong answer",
    ("correct", "suggestion"): "incorrect assertion",
    ("correct", "doubt"): "doubting a correct answer",
    ("wrong", "answer_again"): "answer again (initially wrong)",
    ("correct", "answer_again"): "answer again (initially correct)",
}

# Table D1, Llama-3.1-8B-Instruct (original model), single-turn, pooled over ARC/CSQA/OBQA, each
# dataset weighted by its number of questions.
# (prior, kind) -> {"user": shift, "reliable": shift}
D1_ANCHORS = {
    ("wrong", "suggestion"): {"user": 5.91, "reliable": 6.14},
    ("wrong", "doubt"): {"user": 1.59, "reliable": 1.84},
    ("correct", "suggestion"): {"user": 5.59, "reliable": 6.49},
    ("correct", "doubt"): {"user": 3.33, "reliable": 4.24},
}
# Table D1 "Pruned" columns (same layout), Llama-3.1-8B-Instruct after B-TAP: the Gate 2 targets (fork #25).
D1_ANCHORS_PRUNED = {
    ("wrong", "suggestion"): {"user": 2.95, "reliable": 2.86},
    ("wrong", "doubt"): {"user": 1.32, "reliable": 1.57},
    ("correct", "suggestion"): {"user": 3.63, "reliable": 3.88},
    ("correct", "doubt"): {"user": 2.97, "reliable": 3.68},
}
ANCHOR_SETS = {"original": D1_ANCHORS, "pruned": D1_ANCHORS_PRUNED}


# ---------------------------------------------------------------- validation

def _row_id(r):
    return (f"(checkpoint={r.get('checkpoint')!r}, fmt={r.get('fmt')!r}, item_id={r.get('item_id')!r}, "
            f"kind={r.get('kind')!r}, source={r.get('source')!r}, turn={r.get('turn')!r})")


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_finite(rows):
    """Raise ValueError naming the first row whose letter_mass, probs or logprobs hold a missing or
    non-finite value (NaN, +inf, -inf). A -inf logprob is an error too (a letter with exactly zero
    probability means a broken score, and it would make the log-odds infinite)."""
    for r in rows:
        bad = []
        m = r.get("letter_mass")
        if not _finite(m):
            bad.append(f"letter_mass={m!r}")
        probs = r.get("probs")
        if not probs:
            bad.append(f"probs={probs!r}")
        else:
            bad += [f"probs[{k!r}]={v!r}" for k, v in probs.items() if not _finite(v)]
        bad += [f"logprobs[{k!r}]={v!r}" for k, v in (r.get("logprobs") or {}).items() if not _finite(v)]
        if bad:
            raise ValueError(f"non-finite or missing values in row {_row_id(r)}: {', '.join(bad[:6])}")
        _check_probs_match_logprobs(r)


PROBS_LOGPROBS_TOL = 1e-6


def _check_probs_match_logprobs(r, tol=PROBS_LOGPROBS_TOL):
    """probs must equal exp(logprobs) renormalized over the letters, within tol (absolute), and the
    two must name the same letters. Rows without logprobs are skipped (probs fallback)."""
    lps = r.get("logprobs")
    if not lps:
        return
    probs = r["probs"]
    if set(lps) != set(probs):
        raise ValueError(f"logprobs letters {sorted(lps)} != probs letters {sorted(probs)} "
                         f"for row {_row_id(r)}")
    lse = _logsumexp([float(v) for v in lps.values()])
    diff = {k: abs(math.exp(float(lps[k]) - lse) - float(probs[k])) for k in probs}
    worst = max(diff, key=diff.get)
    if diff[worst] > tol:
        raise ValueError(f"probs disagree with renormalized exp(logprobs) in row {_row_id(r)}: "
                         f"letter {worst!r} probs={probs[worst]!r} vs {math.exp(float(lps[worst]) - lse)!r} "
                         f"(|diff| {diff[worst]:.3g} > {tol:g})")


# ---------------------------------------------------------------- basic quantities

def logit(p, eps=PROB_EPS):
    """Generic helper: log p - log(1 - p) with p clipped to [eps, 1 - eps], default eps = 1e-12
    (finite at 0 and 1). NOT the D1 log-odds: the Table D1 shifts use `letter_logit` (exact, from the
    raw logprobs, unclipped unless clip_eps is given)."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p) - math.log(1.0 - p)


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def letter_logit(row, letter, clip_eps=None, prob_eps=PROB_EPS):
    """Log-odds of `letter` against all the item's other letters.

    From the row's raw `logprobs` when present (non-empty): lp_x - logsumexp(lp_others), exact and
    unclipped (letters must match `probs`). Otherwise from `probs`: log max(p_x, prob_eps) -
    log max(sum of the other p, prob_eps). clip_eps (default None) clips P to [clip_eps, 1 - clip_eps]
    as a sensitivity check (the old eps = 1e-6 behaviour)."""
    probs = row["probs"]
    if letter not in probs:
        raise ValueError(f"letter {letter!r} not in probs {sorted(probs)} for row {_row_id(row)}")
    if len(probs) < 2:
        raise ValueError(f"need at least two letters for log-odds, row {_row_id(row)}")
    lps = row.get("logprobs")
    if lps:
        if set(lps) != set(probs):
            raise ValueError(f"logprobs letters {sorted(lps)} != probs letters {sorted(probs)} "
                             f"for row {_row_id(row)}")
        val = float(lps[letter]) - _logsumexp([float(v) for k, v in lps.items() if k != letter])
    else:
        p = float(probs[letter])
        q = sum(float(v) for k, v in probs.items() if k != letter)
        val = math.log(max(p, prob_eps)) - math.log(max(q, prob_eps))
    if clip_eps is not None:
        cap = math.log(1.0 - clip_eps) - math.log(clip_eps)
        val = min(max(val, -cap), cap)
    if not math.isfinite(val):
        raise ValueError(f"non-finite log-odds for {letter!r} in row {_row_id(row)}")
    return val


def argmax_letter(probs):
    """Highest-probability letter; ties go to the alphabetically earlier letter."""
    if not probs:
        raise ValueError("empty probs")
    if not all(_finite(v) for v in probs.values()):
        raise ValueError(f"non-finite probs {probs}")
    return min(probs, key=lambda k: (-probs[k], k))


def margin(probs):
    """Top probability minus second (1.0 if only one letter)."""
    vals = sorted(probs.values(), reverse=True)
    return vals[0] - (vals[1] if len(vals) > 1 else 0.0)


def update_label(prior, kind=None):
    if kind == "answer_again":
        return "drift"
    if prior == "wrong":
        return "helpful"
    if prior == "correct":
        return "harmful"
    raise ValueError(f"prior must be 'correct' or 'wrong', got {prior!r}")


ANSWER_RULES = ("greedy_token", "letter_argmax")


def answer_rule_of(row):
    """The row's answer rule: its top-level `prior_rule`, or "letter_argmax" when absent (rows
    written before the scorer recorded a rule labelled by letter argmax)."""
    rule = row.get("prior_rule")
    rule = "letter_argmax" if rule is None else rule
    if rule not in ANSWER_RULES:
        raise ValueError(f"unknown prior_rule {rule!r} on row {_row_id(row)}; expected one of {ANSWER_RULES}")
    return rule


def answer_of(row, prior_rule=None):
    """The answer letter of a neutral OR cue row, under one rule for both, so an unchanged letter
    distribution never counts as a move.

    Rule: `prior_rule` if given (must match the row's own field when it has one), else the row's.
    "greedy_token": extra.greedy.top_letter_variant_letter, the letter of the single highest-probability
    letter-variant token (greedy decoding restricted to the item's letter tokens). No fallback: a
    greedy_token row without it raises (re-score with the current scorer). Whether the unrestricted top
    token was a letter is a diagnostic only (n_cue_top_token_not_letter, label_diagnostics).
    "letter_argmax": the argmax of the summed-variant letter probs."""
    rule = answer_rule_of(row)
    if prior_rule not in (None, "row") and prior_rule != rule:
        raise ValueError(f"prior_rule={prior_rule!r} but row {_row_id(row)} has prior_rule={rule!r}")
    if rule == "letter_argmax":
        return argmax_letter(row["probs"])
    g = (row.get("extra") or {}).get("greedy")
    letter = g.get("top_letter_variant_letter") if isinstance(g, dict) else None
    if letter is None:
        raise ValueError(f"row {_row_id(row)} has prior_rule='greedy_token' but no "
                         f"extra['greedy']['top_letter_variant_letter']: written by an older scorer; "
                         f"re-score with the current scorer")
    if letter not in row["probs"]:
        raise ValueError(f"top_letter_variant_letter {letter!r} not among the item's letters "
                         f"{sorted(row['probs'])} in row {_row_id(row)}")
    return letter


def neutral_answer_of(neutral_row):
    """The neutral answer under the row's rule (`answer_of`); raises if the scorer's written
    neutral_answer disagrees with it."""
    ans = answer_of(neutral_row)
    written = neutral_row.get("neutral_answer")
    if written is not None and written != ans:
        raise ValueError(f"neutral_answer {written!r} written on row {_row_id(neutral_row)} != "
                         f"{ans!r} recomputed under prior_rule={answer_rule_of(neutral_row)!r}")
    return ans


def neutral_prior(neutral_row):
    return "correct" if neutral_answer_of(neutral_row) == neutral_row["gold"] else "wrong"


def anchor_letter(cue_row, neutral_row):
    """The letter the shift and flip are measured on (t for suggestion, a otherwise)."""
    if cue_row["kind"] == "answer_again":
        return neutral_answer_of(neutral_row)
    return cue_row["target"]


def log_odds_shift(cue_row, neutral_row, clip_eps=None, prob_eps=PROB_EPS):
    kind = cue_row["kind"]
    letter = anchor_letter(cue_row, neutral_row)
    cue_l = letter_logit(cue_row, letter, clip_eps, prob_eps)
    neu_l = letter_logit(neutral_row, letter, clip_eps, prob_eps)
    if kind == "suggestion":
        return cue_l - neu_l
    if kind in ("doubt", "answer_again"):
        return neu_l - cue_l
    raise ValueError(f"no shift for kind {kind!r}")


def is_flip(cue_row, neutral_row):
    """Figure D1 flip; "answer" = `answer_of` under the row's rule (not the letter argmax unless
    that is the rule). prior wrong: answer == gold (suggestion and doubt alike, a wrong-to-correct
    flip). prior correct: suggestion -> answer == suggested option; doubt -> answer != the doubted
    (correct) answer. answer_again: answer != neutral answer."""
    kind = cue_row["kind"]
    cue_ans = answer_of(cue_row)
    na = neutral_answer_of(neutral_row)
    if kind == "answer_again":
        return cue_ans != na
    if kind not in ("suggestion", "doubt"):
        raise ValueError(f"no flip for kind {kind!r}")
    if neutral_prior(neutral_row) == "wrong":
        return cue_ans == neutral_row["gold"]
    if kind == "suggestion":
        return cue_ans == cue_row["target"]
    return cue_ans != na


def is_moved(cue_row, neutral_row):
    """Any change of answer: cue answer != neutral answer (all kinds), both under the same rule."""
    if cue_row["kind"] not in CUE_KINDS:
        raise ValueError(f"no move for kind {cue_row['kind']!r}")
    return answer_of(cue_row) != neutral_answer_of(neutral_row)


# ---------------------------------------------------------------- join

def _nkey(row):
    return (row["checkpoint"], row["fmt"], row["item_id"])


def index_neutral(rows):
    """(checkpoint, fmt, item_id) -> neutral row. Raises on duplicates or non-single-turn neutral rows."""
    out = {}
    for r in rows:
        if r["kind"] != "neutral":
            continue
        if r.get("turn", "single") != "single":
            raise ValueError(f"neutral row must be single-turn: {_nkey(r)}")
        k = _nkey(r)
        if k in out:
            raise ValueError(f"duplicate neutral row for {k}")
        out[k] = r
    return out


def check_prior_rule(rows, prior_rule="row"):
    """Returns {(checkpoint, fmt): the prior_rule its rows carry, or None}. Raises if one
    (checkpoint, fmt) mixes prior_rule values, or, when prior_rule != "row", if any row lacks or
    differs from that rule."""
    seen = defaultdict(set)
    for r in rows:
        rule = r.get("prior_rule")
        if prior_rule != "row" and rule != prior_rule:
            why = (" (no prior_rule field: written by an older scorer; re-score it)" if rule is None
                   else "")
            raise ValueError(f"prior_rule={prior_rule!r} requested but row {_row_id(r)} has "
                             f"prior_rule={rule!r}{why}")
        seen[(r["checkpoint"], r["fmt"])].add(rule)
    out = {}
    for cf, rules in seen.items():
        if len(rules) > 1:
            old = " (None = no prior_rule field: those rows were written by an older scorer)" \
                if None in rules else ""
            raise ValueError(f"checkpoint={cf[0]!r} fmt={cf[1]!r} mixes prior_rule values "
                             f"{sorted(map(str, rules))}{old}; its neutral and cue rows must come from "
                             f"one scorer run with one rule: re-score this checkpoint/fmt")
        out[cf] = next(iter(rules))
    return out


def join(rows, prior_rule="row"):
    """Pair each cue row with its neutral row. Returns list of (cue_row, neutral_row).

    prior_rule: "row" (default) uses the neutral row's neutral_answer as the scorer wrote it; any other
    value requires every row to carry that `prior_rule` (see `check_prior_rule`).

    Raises ValueError on: non-finite values (`check_finite`), a cue row with no neutral row, duplicate
    cue rows (same checkpoint, fmt, item, turn, kind, source), dataset or gold mismatch, a cue row whose
    prior disagrees with the neutral pass, a doubt whose target is not the neutral answer, a suggestion
    naming the neutral answer, a prior-wrong suggestion not naming the gold, or a prior-correct
    suggestion naming the gold.
    """
    check_finite(rows)
    check_prior_rule(rows, prior_rule)
    neutral = index_neutral(rows)
    seen = set()
    pairs = []
    for r in rows:
        kind = r["kind"]
        if kind == "neutral":
            continue
        if kind not in CUE_KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        k = _nkey(r)
        if k not in neutral:
            raise ValueError(f"cue row has no neutral row: {k}")
        n = neutral[k]
        dup = k + (r["turn"], kind, r["source"])
        if dup in seen:
            raise ValueError(f"duplicate cue row {dup}")
        seen.add(dup)
        if r["dataset"] != n["dataset"]:
            raise ValueError(f"dataset mismatch for {k}: {r['dataset']} vs {n['dataset']}")
        gold = n["gold"]
        if r.get("gold") is not None and r["gold"] != gold:
            raise ValueError(f"gold mismatch for {k}: {r['gold']} vs {gold}")
        prior = neutral_prior(n)
        if r.get("prior") is not None and r["prior"] != prior:
            raise ValueError(f"prior {r['prior']!r} on cue row disagrees with neutral pass ({prior!r}) for {k}")
        na = neutral_answer_of(n)
        if r.get("neutral_answer") is not None and r["neutral_answer"] != na:
            raise ValueError(f"neutral_answer mismatch for {k}: {r['neutral_answer']} vs {na}")
        if kind == "doubt" and r["target"] != na:
            raise ValueError(f"doubt target {r['target']} is not the neutral answer {na} for {k}")
        if kind == "suggestion":
            if r["target"] == na:
                raise ValueError(f"suggestion names the neutral answer {na} for {k}")
            if prior == "wrong" and r["target"] != gold:
                raise ValueError(f"prior-wrong suggestion must name the gold {gold}, names {r['target']} for {k}")
            if prior == "correct" and r["target"] == gold:
                raise ValueError(f"prior-correct suggestion names the gold {gold} for {k}")
        pairs.append((r, n))
    return pairs


def pair_records(rows, clip_eps=None, prob_eps=PROB_EPS, cue_min_mass=CUE_MIN_MASS, prior_rule="row"):
    """One flat record per cue row: cell key fields + item_id, shift, flip, neutral margin, mass flags."""
    out = []
    for cue, neu in join(rows, prior_rule=prior_rule):
        prior = neutral_prior(neu)
        out.append({
            "checkpoint": cue["checkpoint"], "fmt": cue["fmt"], "dataset": cue["dataset"],
            "turn": cue["turn"], "kind": cue["kind"], "source": cue["source"], "prior": prior,
            "item_id": cue["item_id"],
            "shift": log_odds_shift(cue, neu, clip_eps, prob_eps),
            "flip": bool(is_flip(cue, neu)),
            "moved": bool(is_moved(cue, neu)),
            "neutral_margin": margin(neu["probs"]),
            "neutral_mass": neu.get("letter_mass"),
            "cue_mass": cue.get("letter_mass"),
            "cue_low_mass": float(cue["letter_mass"]) < cue_min_mass,
            "prob_fallback": not (cue.get("logprobs") and neu.get("logprobs")),
            "cue_top_token_not_letter": ((cue.get("extra") or {}).get("greedy") or {}).get(
                "top_token_is_letter") is False,
        })
    return out


# ---------------------------------------------------------------- completeness

def check_complete(rows, max_examples=3):
    """Within each (checkpoint, fmt), every neutral item must carry the same set of (kind, source, turn)
    cue rows (the union over that (checkpoint, fmt)'s items). Raises ValueError summarising, per
    (checkpoint, fmt), how many items are incomplete, how often each key is missing and a few examples.
    Cue rows without a neutral row are left to `join` (which raises). Returns
    {(checkpoint, fmt): sorted tuple of the expected cue keys} when complete."""
    neutral = index_neutral(rows)
    have = {k: set() for k in neutral}
    for r in rows:
        if r["kind"] != "neutral" and _nkey(r) in have:
            have[_nkey(r)].add((r["kind"], r["source"], r["turn"]))
    by_cf = defaultdict(dict)
    for (ckpt, fmt, item), keys in have.items():
        by_cf[(ckpt, fmt)][item] = keys
    expected, problems = {}, []
    for cf in sorted(by_cf):
        items = by_cf[cf]
        ref = set().union(*items.values())
        expected[cf] = tuple(sorted(ref))
        lacking = {i: ref - ks for i, ks in items.items() if ks != ref}
        if not lacking:
            continue
        counts = Counter(k for miss in lacking.values() for k in miss)
        top = ", ".join(f"{k}: {n}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
        ex = "; ".join(f"{i} lacks {sorted(lacking[i])[:4]}" for i in sorted(lacking)[:max_examples])
        problems.append(f"checkpoint={cf[0]!r} fmt={cf[1]!r}: {len(lacking)}/{len(items)} items lack "
                        f"some of the {len(ref)} cue keys (kind, source, turn); missing counts: {top}; "
                        f"e.g. {ex}")
    if problems:
        raise ValueError("incomplete cue grid (pass allow_incomplete=True to cells() to override):\n  "
                         + "\n  ".join(problems))
    return expected


# ---------------------------------------------------------------- low-mass filter

MASS_STAMP = "mass_filter"   # row field filter_low_mass writes: {"min_mass": x, "cue_min_mass": y}


class MassFilter(NamedTuple):
    rows: list                 # kept rows (neutral + cue), in input order, stamped with MASS_STAMP
    dropped: dict              # checkpoint -> number of (fmt, item) neutral rows dropped
    dropped_by_fmt: dict       # (checkpoint, fmt) -> number dropped
    low_mass_cues: list        # (checkpoint, fmt, item_id, turn, kind, source, letter_mass) of KEPT cue rows below cue_min_mass
    cue_min_mass: float = CUE_MIN_MASS   # the threshold used for low_mass_cues; cells() reuses it
    min_mass: float = 0.5


def filter_low_mass(rows, min_mass=0.5, cue_min_mass=None):
    """Drop every item whose NEUTRAL letter_mass < min_mass, together with all its cue rows
    (per checkpoint and fmt). Cue rows with letter_mass < cue_min_mass (default CUE_MIN_MASS = 0.5,
    independent of min_mass) are NOT dropped: they are listed in `low_mass_cues`. The kept rows are
    shallow copies stamped with {"min_mass", "cue_min_mass"} under MASS_STAMP, and `cells` reads that
    stamp, so the per-cell n_cue_low_mass always uses the same threshold as `low_mass_cues` (cells
    raises if told a different one). Checkpoints with nothing dropped appear with 0. Raises ValueError
    on non-finite letter_mass / probs / logprobs."""
    cue_min_mass = CUE_MIN_MASS if cue_min_mass is None else float(cue_min_mass)
    check_finite(rows)
    neutral = index_neutral(rows)
    drop = {k for k, r in neutral.items() if r["letter_mass"] < min_mass}
    dropped = {r["checkpoint"]: 0 for r in rows}
    dropped_by_fmt = {(r["checkpoint"], r["fmt"]): 0 for r in rows}
    for ckpt, fmt, _ in drop:
        dropped[ckpt] += 1
        dropped_by_fmt[(ckpt, fmt)] += 1
    stamp = {"min_mass": float(min_mass), "cue_min_mass": cue_min_mass}
    kept, low = [], []
    for r in rows:
        if _nkey(r) in drop:
            continue
        kept.append(dict(r, **{MASS_STAMP: dict(stamp)}))
        if r["kind"] != "neutral" and r["letter_mass"] < cue_min_mass:
            low.append(_nkey(r) + (r["turn"], r["kind"], r["source"], r["letter_mass"]))
    return MassFilter(kept, dropped, dropped_by_fmt, low, cue_min_mass, float(min_mass))


def _resolve_cue_min_mass(rows, cue_min_mass):
    """Single source of truth for the low-mass cue threshold: the stamp filter_low_mass wrote, else the
    explicit argument, else CUE_MIN_MASS. Raises if the rows carry several stamps, or a stamp that
    disagrees with an explicit argument."""
    stamps = {(r.get(MASS_STAMP) or {}).get("cue_min_mass") for r in rows}
    if len(stamps) > 1:
        raise ValueError(f"rows carry different (or partly missing) {MASS_STAMP} cue_min_mass stamps: "
                         f"{sorted(map(str, stamps))}")
    stamped = next(iter(stamps)) if stamps else None
    if stamped is not None:
        if cue_min_mass is not None and float(cue_min_mass) != stamped:
            raise ValueError(f"cue_min_mass={cue_min_mass} disagrees with the {stamped} used by "
                             f"filter_low_mass; use one setting")
        return stamped
    return CUE_MIN_MASS if cue_min_mass is None else float(cue_min_mass)


# ---------------------------------------------------------------- confidence bins

def bin_label(m, edges=DEFAULT_BIN_EDGES):
    """Right-closed bins (lo, hi]; the first bin also includes its lower edge, so margin 0 lands there."""
    edges = tuple(float(e) for e in edges)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"edges must be strictly increasing, got {edges}")
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if (lo < m <= hi) or (i == 0 and m == lo):
            left = "[" if i == 0 else "("
            return f"{left}{lo:g},{hi:g}]"
    raise ValueError(f"margin {m} outside bin edges {edges}")


# ---------------------------------------------------------------- overlap

def overlap_items(rows, checkpoints, fmt):
    """item_ids that have a neutral row in every given checkpoint (for this fmt) with the same prior.
    Raises ValueError if no neutral row has this fmt, or a checkpoint has no neutral rows in it."""
    checkpoints = list(checkpoints)
    if not checkpoints:
        raise ValueError("need at least one checkpoint")
    neutral = index_neutral(rows)
    fmts = {f for (_, f, _) in neutral}
    if fmt not in fmts:
        raise ValueError(f"no neutral rows with fmt={fmt!r}; known fmts: {sorted(fmts)}")
    have = {ckpt for (ckpt, f, _) in neutral if f == fmt}
    lacking = [c for c in checkpoints if c not in have]
    if lacking:
        raise ValueError(f"checkpoints {lacking} have no neutral rows for fmt={fmt!r}; "
                         f"present: {sorted(have)}")
    priors = defaultdict(dict)  # item_id -> checkpoint -> prior
    for (ckpt, f, item), r in neutral.items():
        if f == fmt and ckpt in checkpoints:
            priors[item][ckpt] = neutral_prior(r)
    return {item for item, by in priors.items()
            if len(by) == len(checkpoints) and len(set(by.values())) == 1}


def restrict_items(rows, item_ids, fmt=None):
    """Rows of format `fmt` whose item_id is in item_ids (an overlap set is per fmt, so rows of other
    formats are NOT returned). With fmt=None the rows must all share one fmt, else ValueError."""
    item_ids = set(item_ids)
    fmts = {r["fmt"] for r in rows}
    if fmt is None:
        if len(fmts) > 1:
            raise ValueError(f"rows span fmts {sorted(fmts)}; pass fmt= (overlap sets are per fmt)")
    elif fmt not in fmts:
        raise ValueError(f"no rows with fmt={fmt!r}; known fmts: {sorted(fmts)}")
    return [r for r in rows if (fmt is None or r["fmt"] == fmt) and r["item_id"] in item_ids]


# ---------------------------------------------------------------- label diagnostics

def label_diagnostics(rows):
    """Per (checkpoint, fmt), how answer labels relate to greedy decoding. Top-level fields use the
    NEUTRAL rows:
      n_neutral, n_with_greedy (rows with extra.greedy),
      rule_disagreement_rate: share of rows with extra.greedy.top_letter_variant_letter where that
        ("greedy_token" answer) != the letter argmax ("letter_argmax" answer); None if no row has it,
      greedy_disagreement_rate: share of n_with_greedy where the unrestricted top token is a letter
        (top_token_letter not None) and != the letter argmax,
      greedy_not_letter_rate: share of n_with_greedy whose unrestricted top token is not a letter
        (top_token_is_letter False; the paper's greedy decode would not start with a letter),
      label_vs_argmax_rate: share of n_neutral whose written neutral_answer != the letter argmax,
      prior_rules: {prior_rule value: count} (None = field absent).
    `by_kind` gives n, n_with_greedy and the three rates for every kind (neutral, suggestion, doubt,
    answer_again)."""
    check_finite(rows)
    index_neutral(rows)   # duplicate / multi-turn neutral checks
    acc = defaultdict(Counter)
    rules = defaultdict(Counter)
    for r in sorted(rows, key=lambda r: (str(r["checkpoint"]), str(r["fmt"]), str(r["kind"]),
                                         _rec_sort_key(r))):
        cf = (r["checkpoint"], r["fmt"])
        top = argmax_letter(r["probs"])
        a = acc[cf + (r["kind"],)]
        a["n"] += 1
        if r["kind"] == "neutral":
            if r.get("neutral_answer") is not None and r["neutral_answer"] != top:
                a["label_vs_argmax"] += 1
            rules[cf][r.get("prior_rule")] += 1
        g = (r.get("extra") or {}).get("greedy")
        if isinstance(g, dict):
            a["n_greedy"] += 1
            tl = g.get("top_token_letter")
            if tl is not None and tl != top:
                a["disagree"] += 1
            if g.get("top_token_is_letter") is False:
                a["not_letter"] += 1
            vl = g.get("top_letter_variant_letter")
            if vl is not None:
                a["n_variant"] += 1
                a["rule_disagree"] += vl != top

    def rates(a):
        ng, nv = a["n_greedy"], a["n_variant"]
        return {"n": a["n"], "n_with_greedy": ng,
                "rule_disagreement_rate": a["rule_disagree"] / nv if nv else None,
                "greedy_disagreement_rate": a["disagree"] / ng if ng else None,
                "greedy_not_letter_rate": a["not_letter"] / ng if ng else None}

    out = {}
    for cf in sorted({k[:2] for k in acc}, key=lambda k: tuple(map(str, k))):
        neu = acc.get(cf + ("neutral",), Counter())
        nr = rates(neu)
        out[cf] = {
            "n_neutral": neu["n"],
            "n_with_greedy": nr["n_with_greedy"],
            "greedy_disagreement_rate": nr["greedy_disagreement_rate"],
            "greedy_not_letter_rate": nr["greedy_not_letter_rate"],
            "rule_disagreement_rate": nr["rule_disagreement_rate"],
            "label_vs_argmax_rate": neu["label_vs_argmax"] / neu["n"] if neu["n"] else None,
            "prior_rules": dict(rules[cf]),
            "by_kind": {k[2]: rates(a) for k, a in sorted(acc.items(), key=lambda kv: str(kv[0]))
                        if k[:2] == cf},
        }
    return out


# ---------------------------------------------------------------- bootstrap

_STATS = ("shift", "flip", "moved")


def _rec_sort_key(r):
    return tuple(str(r.get(f)) for f in ("dataset", "item_id", "source", "turn", "kind"))


def _strata(records):
    """dataset -> array (k_items, len(_STATS) + 1) of per-item [sums..., count], items sorted by id
    and records summed in a canonical order, so nothing depends on the input row order."""
    acc = defaultdict(dict)
    for r in sorted(records, key=_rec_sort_key):
        a = acc[r["dataset"]].setdefault(r["item_id"], [0.0] * (len(_STATS) + 1))
        for j, f in enumerate(_STATS):
            a[j] += float(r[f])
        a[-1] += 1
    return {ds: np.array([v[i] for i in sorted(v)], float) for ds, v in acc.items()}


def dataset_weighted_means(records):
    """{stat: sum_d n_items_d * mean_d / sum_d n_items_d} for stat in shift, flip, moved, where mean_d
    is the plain mean over dataset d's records (Table D1's weighting by number of questions)."""
    if not records:
        return {f: float("nan") for f in _STATS}
    tot = np.zeros(len(_STATS))
    w = 0.0
    strata = _strata(records)
    for ds in sorted(strata):
        arr = strata[ds]
        k = len(arr)
        sums = arr.sum(axis=0)
        tot += k * sums[:-1] / sums[-1]
        w += k
    return {f: float(tot[j] / w) for j, f in enumerate(_STATS)}


def _bootstrap(records, n_boot, seed, ci):
    """Percentile CIs for (shift, flip, moved). Resamples item clusters with replacement within each
    dataset (stratified); within a stratum the statistic is the record mean of the resample, so items
    repeated across pooled sources move together; strata are combined with the fixed n_items weights
    of `dataset_weighted_means`. Vectorized: one (n_boot, k) draw per stratum, strata in sorted
    dataset order and items in sorted id order (invariant to input row order)."""
    if not records:
        nan = (float("nan"), float("nan"))
        return {f: nan for f in _STATS}
    rng = np.random.default_rng(seed)
    tot = np.zeros((n_boot, len(_STATS)))
    w = 0.0
    strata = _strata(records)
    for ds in sorted(strata):
        arr = strata[ds]
        k = len(arr)
        pick = rng.integers(0, k, size=(n_boot, k))
        sums = arr[pick].sum(axis=1)               # (n_boot, len(_STATS) + 1)
        tot += k * sums[:, :-1] / sums[:, -1:]
        w += k
    stat = tot / w
    a = (1.0 - ci) / 2.0
    q = [100 * a, 100 * (1 - a)]
    return {f: tuple(float(x) for x in np.percentile(stat[:, j], q)) for j, f in enumerate(_STATS)}


TRIM = 0.1   # trimmed_mean_shift cuts this fraction from EACH tail


def trimmed_mean(xs, trim=TRIM):
    """Mean after dropping floor(trim * n) values from each end of the sorted values (scipy's
    trim_mean convention); nan for no values."""
    xs = sorted(float(x) for x in xs)
    if not xs:
        return float("nan")
    k = int(trim * len(xs))
    return float(np.mean(xs[k:len(xs) - k]))


def _finish(key, records, n_boot, seed, ci, **extra):
    records = sorted(records, key=_rec_sort_key)
    cis = _bootstrap(records, n_boot, seed, ci)
    means = dataset_weighted_means(records)
    n = len(records)
    items_by_ds = defaultdict(set)
    for r in records:
        items_by_ds[r["dataset"]].add(r["item_id"])
    cell = dict(key)
    cell.update({
        "update": update_label(cell["prior"], cell["kind"]),
        "condition": CONDITION[(cell["prior"], cell["kind"])],
        "d1_role": D1_ROLE.get(cell["source"]),
        "n": n,
        "n_items": len({r["item_id"] for r in records}),
        "n_items_by_dataset": {ds: len(v) for ds, v in sorted(items_by_ds.items())},
        "mean_shift": means["shift"],
        "shift_ci": cis["shift"],
        "flip_rate": means["flip"],
        "flip_ci": cis["flip"],
        "moved_rate": means["moved"],
        "moved_ci": cis["moved"],
        # plain record-level means; equal to the above unless records per item differ across datasets
        "record_mean_shift": float(np.mean([r["shift"] for r in records])) if n else float("nan"),
        "record_flip_rate": float(np.mean([r["flip"] for r in records])) if n else float("nan"),
        "record_moved_rate": float(np.mean([r["moved"] for r in records])) if n else float("nan"),
        "n_cue_low_mass": sum(bool(r["cue_low_mass"]) for r in records),
        "n_prob_fallback": sum(bool(r.get("prob_fallback")) for r in records),
        "n_cue_top_token_not_letter": sum(bool(r.get("cue_top_token_not_letter")) for r in records),
        # robustness to heavy tails: RECORD-level and unweighted (no dataset weighting, no CI); for a
        # pooled cell they are recomputed over all pooled records
        "median_shift": float(np.median([r["shift"] for r in records])) if n else float("nan"),
        "trimmed_mean_shift": trimmed_mean([r["shift"] for r in records], TRIM),
        "boot": {"n_boot": n_boot, "seed": seed, "ci": ci},
        "records": records,
    })
    cell.update(extra)
    return cell


# ---------------------------------------------------------------- cells and pooling

def cells(rows, bin_edges=None, n_boot=1000, seed=0, ci=0.95, clip_eps=None, prob_eps=PROB_EPS,
          cue_min_mass=None, cue_min_mass_drop=None, prior_rule="row", allow_incomplete=False):
    """Per-cell n, mean log-odds shift, flip rate, moved rate and bootstrap CIs, one cell per
    (checkpoint, fmt, dataset, turn, kind, source, prior[, margin_bin]).

    rows: a list of rows or the MassFilter from filter_low_mass (preferred).
    bin_edges: bin items by neutral margin; margin_bin becomes part of the key.
    clip_eps: None (default, exact unclipped log-odds from logprobs) or a float to clip P to
        [clip_eps, 1 - clip_eps] (sensitivity check). prob_eps: floor for the probs fallback.
    cue_min_mass: threshold for n_cue_low_mass; taken from filter_low_mass's stamp when present
        (raises if a different value is passed), else this argument, else CUE_MIN_MASS.
    cue_min_mass_drop: None (default: keep low-mass cue rows, count them) or a float: drop cue rows
        with letter_mass below it, reported per cell as n_cue_dropped (a cell whose rows were all
        dropped is kept with n = 0).
    prior_rule: see `join`.
    `records` holds the per-item values that pooling needs; summary_table drops it. Runs
    `check_complete(rows)` first unless allow_incomplete=True. Raises on non-finite values."""
    if isinstance(rows, MassFilter):
        rows = rows.rows
    check_finite(rows)
    cue_min_mass = _resolve_cue_min_mass(rows, cue_min_mass)
    rules = check_prior_rule(rows, prior_rule)
    if not allow_incomplete:
        check_complete(rows)
    groups = defaultdict(list)
    dropped = Counter()
    for rec in pair_records(rows, clip_eps=clip_eps, prob_eps=prob_eps, cue_min_mass=cue_min_mass,
                            prior_rule=prior_rule):
        mb = bin_label(rec["neutral_margin"], bin_edges) if bin_edges is not None else None
        rec["margin_bin"] = mb
        key = tuple(rec[k] for k in CELL_KEY) + (mb,)
        groups[key]  # create the cell even if every record of it is dropped
        if cue_min_mass_drop is not None and float(rec["cue_mass"]) < cue_min_mass_drop:
            dropped[key] += 1
            continue
        groups[key].append(rec)
    out = []
    for key in sorted(groups, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        kd = dict(zip(CELL_KEY + ("margin_bin",), key))
        out.append(_finish(kd, groups[key], n_boot, seed, ci,
                           pooled_datasets=None, pooled_sources=None,
                           n_cue_dropped=dropped[key], cue_min_mass=cue_min_mass,
                           source_coverage={kd["dataset"]: (kd["source"],) if groups[key] else ()},
                           cue_min_mass_drop=cue_min_mass_drop, clip_eps=clip_eps,
                           prior_rule=rules.get((kd["checkpoint"], kd["fmt"])),
                           answer_rule=rules.get((kd["checkpoint"], kd["fmt"])) or "letter_argmax"))
    return out


def _check_protected(members):
    for f in PROTECTED:
        vals = {m.get(f) for m in members}
        if len(vals) > 1:
            raise ValueError(f"refusing to pool across {f}: {sorted(map(str, vals))}")


# settings every member of a pool must share; carried onto the pooled cell
_CARRY_AGREE = ("cue_min_mass", "cue_min_mass_drop", "clip_eps", "prior_rule", "answer_rule",
                "expected_sources",
                "expected_datasets", "boot")


def _freeze(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def _set_coverage_gaps(cell):
    """missing_sources_by_dataset: for every dataset in the cell's source_coverage, the sources it lacks
    (with records) among the expected ones (the cell's expected_sources, else the union over its
    datasets). Catches a dataset missing a source in either pooling order, incl. allow_incomplete grids."""
    cov = cell.get("source_coverage") or {}
    expected = cell.get("expected_sources")
    ref = set(expected) if expected is not None else set().union(*map(set, cov.values())) if cov else set()
    cell["missing_sources_by_dataset"] = {
        d: tuple(sorted(ref - set(srcs))) for d, srcs in sorted(cov.items()) if ref - set(srcs)}
    return cell


def pool(cell_list, over, name, include=None, exclude=()):
    """Merge cells that differ only in `over` ("dataset" or "source") into one cell named `name`.
    Means follow `dataset_weighted_means` over the members' records: pooling sources within a dataset
    gives the plain record mean; pooling datasets weights each dataset by its number of distinct items.
    Pooling over any other field (prior, checkpoint, fmt, turn, kind, margin_bin) raises ValueError, as
    do members with different settings (cue_min_mass, clip_eps, prior_rule, bootstrap, ...).
    Members with n = 0 (every cue row dropped) do not count towards pooled_datasets / pooled_sources.
    Groups with one member are still returned (as a pooled cell over that one member). Returns only the
    new cells. Prefer `pool_datasets` / `pool_sources`, which fix the member sets."""
    if over not in POOLABLE:
        raise ValueError(f"cannot pool over {over!r}; only {POOLABLE} may be pooled "
                         f"(never prior/checkpoint/fmt/turn/kind/margin_bin)")
    other = [f for f in CELL_KEY + ("margin_bin",) if f != over]
    seen = set()
    groups = defaultdict(list)
    for c in cell_list:
        full = tuple(c.get(f) for f in CELL_KEY + ("margin_bin",))
        if full in seen:
            raise ValueError(f"duplicate cell {full}")
        seen.add(full)
        v = c[over]
        if v == name:
            raise ValueError(f"cell already pooled under {over}={name!r}")
        if include is not None and v not in include:
            continue
        if v in exclude:
            continue
        groups[tuple(c.get(f) for f in other)].append(c)
    out = []
    for gkey in sorted(groups, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        members = groups[gkey]
        _check_protected(members)
        kd = dict(zip(other, gkey))
        kd[over] = name
        records = [r for m in members for r in m["records"]]
        extra = {}
        for f in _CARRY_AGREE:
            vals = {_freeze(m.get(f)) for m in members}
            if len(vals) > 1:
                raise ValueError(f"refusing to pool cells with different {f}: {sorted(map(str, vals))}")
            if any(f in m for m in members):
                extra[f] = members[0].get(f)
        b = members[0]["boot"]
        live = [m for m in members if m["n"] > 0] or members[:1]
        extra["pooled_datasets"] = members[0].get("pooled_datasets")
        extra["pooled_sources"] = members[0].get("pooled_sources")
        label = "pooled_datasets" if over == "dataset" else "pooled_sources"
        inner = set()
        for m in members:
            if m["n"] > 0:
                inner.update(m.get(label) or (m[over],))
        extra[label] = tuple(sorted(inner))
        other_label = "pooled_sources" if over == "dataset" else "pooled_datasets"
        others = {m.get(other_label) for m in live}
        partial = any(m.get("missing_sources") for m in members)
        if len(others) > 1:
            if not partial:
                # e.g. datasets-first pooling of an allow_incomplete grid: a source cell covers fewer
                # datasets. Recorded in missing_sources_by_dataset (check_anchors: incomplete_sources).
                warnings.warn(f"pool({over!r} -> {name!r}): members disagree on {other_label}: "
                              f"{sorted(map(str, others))}; see missing_sources_by_dataset", stacklevel=3)
            union = set()
            for o in others:
                union.update(o or ())
            extra[other_label] = tuple(sorted(union))
        else:
            extra[other_label] = next(iter(others))
        for f in ("missing_sources", "missing_datasets"):
            if any(f in m for m in members):
                miss = set()
                for m in members:
                    miss.update(m.get(f) or ())
                extra[f] = tuple(sorted(miss))
        extra["n_cue_dropped"] = sum(m.get("n_cue_dropped", 0) for m in members)
        cov = defaultdict(set)
        for m in members:
            for d, srcs in (m.get("source_coverage") or {}).items():
                cov[d].update(srcs)
        extra["source_coverage"] = {d: tuple(sorted(v)) for d, v in sorted(cov.items())}
        cell = _finish({f: kd[f] for f in CELL_KEY + ("margin_bin",)}, records,
                       b["n_boot"], b["seed"], b["ci"], **extra)
        cell["member_n"] = {m[over]: m["n"] for m in members}
        cell["member_n_items"] = {m[over]: m["n_items"] for m in members}
        out.append(_set_coverage_gaps(cell))
    return out


def pool_datasets(cell_list, name=POOLED_DATASET, datasets=None):
    """Orgad's pooling over ARC/CSQA/OBQA: each dataset's cell mean weighted by its number of distinct
    items (Table D1). Keeps every other key part. Per-dataset cells are not returned here; report them
    alongside. The plain record-level means stay available as record_mean_shift etc.
    datasets: None (default) pools whatever datasets are present; a list pools only those, records them
    as `expected_datasets` and any absent ones as `missing_datasets` (check_anchors requires the pooled
    set to be exactly D1_DATASETS, so pass datasets=D1_DATASETS to leave out extra datasets)."""
    if datasets is None:
        return pool(cell_list, "dataset", name)
    datasets = tuple(datasets)
    out = pool(cell_list, "dataset", name, include=set(datasets))
    for c in out:
        c["expected_datasets"] = datasets
        c["missing_datasets"] = tuple(sorted(set(datasets) - set(c["pooled_datasets"] or ())))
    return out


def pool_sources(cell_list, name=None, include_tool=True, allow_partial=False):
    """Explicit "reliable source" pooling (OPEN DETAIL #3). Figure D1: "Source bars pool the
    external-source templates in Table C3". Pools EXACTLY EXTERNAL_SOURCES (13, tool included) or,
    with include_tool=False, EXTERNAL_SOURCES_NO_TOOL (12); the paper does not say whether the tool row
    is in, so run both. The 12-source pool is the PRIMARY D1 comparison (D1_ROLE): our tool cue is a
    plain-text emulation, not the paper's native tool call. Any other source ("user", "none", unknown
    keys) is never included.
    Every group (all key fields but source) that has suggestion/doubt cells must contain every expected
    source (with n > 0), else ValueError. allow_partial=True warns instead, pools what is there and
    records the gaps in each pooled cell's `missing_sources` (check_anchors then refuses to compare it).
    Default name: RELIABLE_SOURCE, or RELIABLE_SOURCE_NO_TOOL when include_tool=False."""
    expected = EXTERNAL_SOURCES if include_tool else EXTERNAL_SOURCES_NO_TOOL
    if name is None:
        name = RELIABLE_SOURCE if include_tool else RELIABLE_SOURCE_NO_TOOL
    other = [f for f in CELL_KEY + ("margin_bin",) if f != "source"]
    cue = [c for c in cell_list if c["kind"] in ("suggestion", "doubt")]
    present = defaultdict(set)
    for c in cue:
        srcs = present[tuple(c.get(f) for f in other)]
        if c.get("n", 1) > 0:
            srcs.add(c["source"])
    missing = {g: tuple(s for s in expected if s not in srcs) for g, srcs in present.items()}
    missing = {g: m for g, m in missing.items() if m}
    if missing:
        lines = [f"{dict(zip(other, g))}: missing {list(m)}" for g, m in sorted(
            missing.items(), key=lambda kv: tuple("" if v is None else str(v) for v in kv[0]))[:5]]
        msg = (f"pool_sources({name!r}): {len(missing)}/{len(present)} groups lack expected Table C3 "
               f"sources:\n  " + "\n  ".join(lines))
        if not allow_partial:
            raise ValueError(msg + "\n(pass allow_partial=True to pool what is there)")
        warnings.warn(msg, stacklevel=2)
    out = pool(cue, "source", name, include=expected)
    for c in out:
        c["expected_sources"] = expected
        c["missing_sources"] = missing.get(tuple(c.get(f) for f in other), ())
        _set_coverage_gaps(c)
    return out


# ---------------------------------------------------------------- anchors

def check_anchors(pooled_cells, checkpoint, fmt, rel_tol=0.2, dataset=POOLED_DATASET,
                  reliable_names=(RELIABLE_SOURCE_NO_TOOL, RELIABLE_SOURCE), turn="single",
                  required_datasets=D1_DATASETS, tool_mode="text", anchors="original", primary_reliable=None):
    """Compare our pooled single-turn log-odds shifts to Table D1, e.g.

        check_anchors(cells, "meta_instruct", "native")

    (Orgad ran Llama-3.1-8B-Instruct through its chat template, so the sanity gate uses fmt "native".)
    The user column uses the source="user" cell. The reliable column is reported once per name in
    `reliable_names` and REQUIRES an explicit pooled-source cell with that source name (from
    pool_sources); a single source cell is never substituted. Each row's `role` (D1_ROLE):
    "primary" for user and reliable_source_no_tool (the 12 Table C3 external sources without the tool),
    "secondary" for reliable_source (13, tool included: our tool cue is a plain-text emulation of the
    paper's native tool call, so it is not the paper's stimulus), "custom" for other names.
    Statuses (ANCHOR_STATUSES), passed is True only for "pass"; checked in this order:
      "missing"             no such cell; ours/n/rel_err stay None/0; evaluated False.
      "incomplete_datasets" pooled_datasets lack one of `required_datasets` (csqa, arc_challenge, obqa).
      "extra_datasets"      pooled_datasets include a dataset outside `required_datasets`.
      "incomplete_sources"  reliable cell with non-empty missing_sources, or pooled_sources != the
                            expected set for that name (EXPECTED_SOURCES, else the cell's
                            expected_sources), or any D1 dataset lacking an expected source in the
                            cell's source_coverage (`missing_sources_by_dataset`; catches
                            datasets-first pooling of an allow_incomplete grid); a user cell pooled
                            from anything but "user" or with a dataset lacking user records.
      "pass" / "failed"     compared: |ours - theirs| <= rel_tol * |theirs| or not; evaluated True.
    For the non-compared statuses ours/n/rel_err are filled for information; `problems` lists every
    problem found. Pass reliable_names=(RELIABLE_SOURCE_NO_TOOL,) to check only the primary variant.
    tool_mode: how the run rendered the tool row. "text" (default) keeps the roles above. "native"
    (decision D4: the Meta Instruct gate with Llama's native tool call) swaps them via
    D1_ROLE_NATIVE_TOOL: reliable_source (13, tool included) is PRIMARY and reliable_source_no_tool
    is "secondary".
    anchors: "original" (default, D1 Original columns, Gate 1) | "pruned" (D1 Pruned columns, Gate 2) | a dict
    shaped like D1_ANCHORS.
    primary_reliable: None (roles follow tool_mode, as above) | "no_tool" (12 sources primary whatever the tool
    mode; lab fork #20 and #22: the replication gates grade the 12-source column) | "with_tool"."""
    if tool_mode not in TOOL_MODES:
        raise ValueError(f"tool_mode must be one of {TOOL_MODES}, got {tool_mode!r}")
    if primary_reliable is None:
        roles = D1_ROLE_NATIVE_TOOL if tool_mode == "native" else D1_ROLE
    elif primary_reliable == "no_tool":
        roles = D1_ROLE
    elif primary_reliable == "with_tool":
        roles = D1_ROLE_NATIVE_TOOL
    else:
        raise ValueError(f"primary_reliable must be None, 'no_tool' or 'with_tool', got {primary_reliable!r}")
    if isinstance(anchors, str):
        if anchors not in ANCHOR_SETS:
            raise ValueError(f"anchors must be one of {tuple(ANCHOR_SETS)} or a dict, got {anchors!r}")
        anchors = ANCHOR_SETS[anchors]
    if isinstance(reliable_names, str):
        reliable_names = (reliable_names,)
    required = tuple(required_datasets)
    index = {}
    for c in pooled_cells:
        if (c["checkpoint"], c["fmt"], c["dataset"], c["turn"], c.get("margin_bin")) != \
                (checkpoint, fmt, dataset, turn, None):
            continue
        k = (c["prior"], c["kind"], c["source"])
        if k in index:
            raise ValueError(f"duplicate pooled cell {k}")
        index[k] = c
    table = []
    for (prior, kind), theirs in anchors.items():
        for column, source in (("user", "user"),) + tuple((n, n) for n in reliable_names):
            t = theirs["user" if column == "user" else "reliable"]
            c = index.get((prior, kind, source))
            row = {"condition": CONDITION[(prior, kind)], "update": update_label(prior), "prior": prior,
                   "kind": kind, "column": column, "role": roles.get(column, "custom"), "theirs": t,
                   "ours": None, "n": 0, "ci": None, "rel_err": None, "passed": False,
                   "evaluated": False, "status": "missing", "problems": ("missing",), "datasets": None,
                   "sources": None, "missing_datasets": None, "extra_datasets": None,
                   "missing_sources": None, "expected_sources": None, "missing_sources_by_dataset": None}
            if c is not None:
                rel = abs(c["mean_shift"] - t) / abs(t)
                ds = set(c.get("pooled_datasets") or ())
                lacking = tuple(d for d in required if d not in ds)
                extra_ds = tuple(sorted(ds - set(required)))
                srcs = c.get("pooled_sources")
                cov = c.get("source_coverage") or {}
                gaps = dict(c.get("missing_sources_by_dataset") or {})
                if column == "user":
                    exp_src = ("user",)
                    miss_src = ()
                    src_bad = srcs not in (None, ("user",))
                else:
                    exp_src = EXPECTED_SOURCES.get(source, c.get("expected_sources"))
                    miss_src = tuple(c.get("missing_sources") or ())
                    src_bad = (exp_src is None or bool(miss_src)
                               or tuple(sorted(srcs or ())) != tuple(sorted(exp_src)))
                if exp_src is not None and cov:
                    # every D1 dataset must have every expected source (catches datasets-first pooling)
                    for d in required:
                        lack = tuple(sorted(set(exp_src) - set(cov.get(d, ())))) if d in cov else ()
                        if lack:
                            gaps[d] = tuple(sorted(set(gaps.get(d, ())) | set(lack)))
                src_bad = src_bad or bool(gaps)
                problems = tuple(p for p, bad in (("incomplete_datasets", bool(lacking)),
                                                  ("extra_datasets", bool(extra_ds)),
                                                  ("incomplete_sources", src_bad)) if bad)
                row.update(ours=c["mean_shift"], n=c["n"], ci=c["shift_ci"], rel_err=rel,
                           datasets=c.get("pooled_datasets"), sources=srcs,
                           missing_datasets=lacking, extra_datasets=extra_ds,
                           missing_sources=miss_src, expected_sources=exp_src, problems=problems,
                           missing_sources_by_dataset=gaps)
                if problems:
                    row.update(status=problems[0])
                else:
                    ok = rel <= rel_tol
                    row.update(passed=ok, evaluated=True, status="pass" if ok else "failed")
            table.append(row)
    return table


# ---------------------------------------------------------------- summary

_SUMMARY_FIELDS = CELL_KEY + ("margin_bin", "update", "condition", "d1_role", "n", "n_items",
                              "mean_shift", "shift_ci", "flip_rate", "flip_ci", "moved_rate", "moved_ci",
                              "record_mean_shift", "n_cue_low_mass", "n_cue_dropped", "n_prob_fallback",
                              "n_cue_top_token_not_letter", "median_shift", "trimmed_mean_shift",
                              "cue_min_mass", "clip_eps", "prior_rule",
                              "answer_rule", "pooled_datasets",
                              "pooled_sources", "missing_sources", "missing_sources_by_dataset")


def summary_table(cell_list, as_markdown=False):
    """Cells without their per-item records, as a list of dicts, or as a markdown table. `d1_role`
    marks the D1 comparison cells: "primary" (user, reliable_source_no_tool) or "secondary"
    (reliable_source, tool included); None otherwise."""
    rows = [{f: c.get(f) for f in _SUMMARY_FIELDS} for c in cell_list]
    if not as_markdown:
        return rows
    cols = ["checkpoint", "fmt", "dataset", "turn", "kind", "source", "d1 role", "prior", "update",
            "margin_bin", "n", "shift", "shift 95% CI", "flip", "flip 95% CI", "moved", "low-mass cues",
            "dropped cues"]

    def f2(x):
        return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        vals = [r["checkpoint"], r["fmt"], r["dataset"], r["turn"], r["kind"], r["source"],
                r["d1_role"] or "", r["prior"], r["update"], r["margin_bin"] or "", r["n"],
                f2(r["mean_shift"]), f"[{f2(r['shift_ci'][0])}, {f2(r['shift_ci'][1])}]",
                f2(r["flip_rate"]), f"[{f2(r['flip_ci'][0])}, {f2(r['flip_ci'][1])}]",
                f2(r["moved_rate"]), r["n_cue_low_mass"], r["n_cue_dropped"] or 0]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def anchors_markdown(table):
    lines = ["| condition | column | role | ours | theirs | rel err | n | status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in table:
        ours = "-" if r["ours"] is None else f"{r['ours']:.2f}"
        rel = "-" if r["rel_err"] is None else f"{r['rel_err']:.0%}"
        lines.append(f"| {r['condition']} | {r['column']} | {r['role']} | {ours} | {r['theirs']:.2f} | "
                     f"{rel} | {r['n']} | {r['status']} |")
    return "\n".join(lines)
