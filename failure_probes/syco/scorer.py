"""Next-token letter scoring (no generation) and per-checkpoint orchestration.

Contract: CONTRACT.md, section `scorer.py`. P(option) is the next-token probability at the last
prompt position: softmax over the FULL vocab, summed over each letter's single-token spellings
("B", " B" by default; configurable with `variants`), then renormalized over the item's letters
(spec OPEN DETAIL #1). `letter_mass` is the unnormalized total, so low-mass items (e.g. a base
model that wants to say "The") can be dropped downstream.

BOS rule (both formats): every encoded prompt starts with exactly one BOS at position 0. It is
decided from the rendered text, not from the format: a text that already starts with
`tokenizer.bos_token` (Llama-3 chat template) is encoded with add_special_tokens=False; any
other text (plain dialogue, Tulu-3 chat template) is encoded with add_special_tokens=True, and if
the tokenizer did not add BOS, its id is prepended. Anything else raises. Tokenizers without a
bos_token (none of ours) are encoded with add_special_tokens=False and no BOS.

Prior label (`prior_rule`). The paper labels the initial answer by one greedy decode. Default
"greedy_token": the letter of the single highest-probability letter-variant token among the
item's letter ids (greedy decoding restricted to the letter tokens), stored per row as
`extra["greedy"]["top_letter_variant_letter"]`. It is always defined (no fallback), it equals the
greedy letter whenever the full-vocab top token is a letter, and it is the same function of the
letter-token probabilities in every row, so a non-letter token taking the top spot in a cue
prompt cannot switch the rule within a (neutral, cue) pair. "letter_argmax": argmax of the summed
variant probabilities. The rule is written into every row (`prior_rule`), the run_config and the
manifest (with counts of top tokens that are not letters and of greedy_token vs letter_argmax
disagreements). The neutral rows' `neutral_answer`, the cue priors/targets and the multi-turn
assistant letter all use the label under the chosen rule.

Non-finite logits: NaN or +inf anywhere in a prompt's last-position logits, or -inf on ALL variant
tokens of ANY one of the item's letters (that letter's probability would be exactly 0, which
metrics rejects), raise in `score_prompts` before it returns, so nothing from that call is
written. A -inf on non-letter tokens (a masked vocab entry) is allowed.

Item text: run_checkpoint refuses items whose question or option text contains one of the
tokenizer's special-token strings (e.g. "<|eot_id|>", "<s>"), listing the item ids, before
anything is written.

Crash safety: `run_checkpoint` writes `<stem>.<checkpoint>.<fmt>.inprogress` when it starts and
replaces it by `<stem>.<checkpoint>.<fmt>.manifest.json` only on completion. `resume=True`
continues a crashed run: rows already in the JSONL for that (checkpoint, fmt) are kept (keyed by
item_id, kind, source, turn), their stored prompt hashes are checked against the rebuilt prompts,
and the neutral rows already there are reused for the prior labels. The run_config in the marker
(settings, prior_rule, model dtype and a cheap weight fingerprint) must match, and so must every
existing row's extra["run_config_sha256"] (so a resume without the marker is checked too). If the JSONL does not
end with a newline (a write cut off after the JSON but before "\n"), one is appended before any
new row.

Not wired into `failure-probes collect` (the lab mod gates collect). `run_checkpoint` takes an
already loaded model and tokenizer; `load_checkpoint` is the only thing here that touches the hub.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import subprocess
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from failure_probes.common import ROOT, append_jsonl, read_json, write_json
from failure_probes.syco.types import LETTERS, Cue, Score

DEFAULT_VARIANTS = ("{L}", " {L}")
NEUTRAL_CUE = Cue(kind="neutral", source="none", turn="single", target=None, prior=None)
CHECKPOINTS_CONFIG = ROOT / "configs" / "syco_checkpoints.json"
DEFAULT_MIN_MASS = 0.5
PRIOR_RULES = ("greedy_token", "letter_argmax")
DEFAULT_PRIOR_RULE = "greedy_token"


# --------------------------------------------------------------------------- letters


def letter_token_ids(tokenizer, letters=LETTERS, variants=DEFAULT_VARIANTS):
    """letter -> list of single-token ids, one per variant spelling that is a single real token.

    Variants that encode to several tokens, or to the unk token, are skipped. Raises if a letter
    has no single-token variant, or if two letters share an id (their mass would be confounded).
    """
    if isinstance(variants, str):
        raise TypeError("variants must be a sequence of patterns like ('{L}', ' {L}'), not a string")
    for variant in variants:
        if "{L}" not in variant:
            raise ValueError(f"letter variant {variant!r} has no '{{L}}' placeholder")
    unk = getattr(tokenizer, "unk_token_id", None)
    out, owner = {}, {}
    for letter in letters:
        ids = []
        for variant in variants:
            encoded = tokenizer.encode(variant.format(L=letter), add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] != unk and encoded[0] not in ids:
                ids.append(int(encoded[0]))
        if not ids:
            raise ValueError(f"letter {letter!r} has no single-token variant among {tuple(variants)}")
        for token_id in ids:
            if token_id in owner and owner[token_id] != letter:
                raise ValueError(f"letters {owner[token_id]!r} and {letter!r} share token id {token_id}")
            owner[token_id] = letter
        out[letter] = ids
    return out


class NonFiniteLogitsError(ValueError):
    """Last-position logits hold NaN or +inf, or some letter of the item has all its variants at -inf."""


class LetterDist(NamedTuple):
    probs: dict        # letter -> renormalized probability over the given letters
    letter_mass: float  # total full-vocab probability on all letters' tokens
    logprobs: dict     # letter -> log of full-vocab probability summed over its variants


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    top = values.max()
    if top == -np.inf:  # every value is -inf (probability 0): avoid -inf - -inf = nan
        return float("-inf")
    return float(top + np.log(np.exp(values - top).sum()))


def _letter_token_logprobs(logits_1d, ids_by_letter):
    """(flat ids, their full-vocab log-probs as float64 numpy), letters in order, variants in order.
    Raises NonFiniteLogitsError on NaN or +inf anywhere, or when any letter has all its variant
    tokens at -inf (probability exactly 0)."""
    flat = [i for ids in ids_by_letter.values() for i in ids]
    if hasattr(logits_1d, "detach"):  # torch: log-softmax on the logits' device, then move a few values
        import torch

        x = logits_1d.detach().reshape(-1).to(torch.float64)
        bad = torch.isnan(x) | torch.isposinf(x)
        if bool(bad.any()):
            raise NonFiniteLogitsError(f"NaN/+inf logits ({int(bad.sum())} entries)")
        logp_all = torch.log_softmax(x, dim=-1)
        picked = logp_all[torch.tensor(flat, device=logp_all.device)].cpu().numpy()
    else:
        x = np.asarray(logits_1d, dtype=np.float64).reshape(-1)
        bad = np.isnan(x) | np.isposinf(x)
        if bad.any():
            raise NonFiniteLogitsError(f"NaN/+inf logits ({int(bad.sum())} entries)")
        picked = x[flat] - _logsumexp(x)
    if not np.isfinite(picked).any():
        raise NonFiniteLogitsError("every letter token has log-probability -inf (probability 0)")
    pos, dead = 0, []
    for letter, ids in ids_by_letter.items():
        if not np.isfinite(picked[pos: pos + len(ids)]).any():
            dead.append(letter)
        pos += len(ids)
    if dead:
        raise NonFiniteLogitsError(f"letter(s) {dead} have log-probability -inf on every variant token "
                                   f"(probability exactly 0; metrics cannot use such a row)")
    return flat, picked


def letter_distribution(logits_1d, ids_by_letter):
    """Letter probabilities from one position's full-vocab logits (torch tensor or numpy array).

    Computed in float64 whatever the input dtype (bf16 logits are upcast before the softmax).
    Raises NonFiniteLogitsError on NaN or +inf anywhere, or if any letter has all its variants at
    -inf (-inf on non-letter tokens, or on some but not all variants of a letter, is fine).
    """
    return _dist_from_picked(_letter_token_logprobs(logits_1d, ids_by_letter)[1], ids_by_letter)


def _dist_from_picked(picked, ids_by_letter):
    logprobs, pos = {}, 0
    for letter, ids in ids_by_letter.items():
        logprobs[letter] = _logsumexp(picked[pos: pos + len(ids)])
        pos += len(ids)
    log_mass = _logsumexp(list(logprobs.values()))
    probs = {letter: math.exp(lp - log_mass) for letter, lp in logprobs.items()}
    return LetterDist(probs=probs, letter_mass=math.exp(log_mass), logprobs=logprobs)


# --------------------------------------------------------------------------- encoding


def needs_special_tokens(fmt):
    """Legacy per-format rule (plain -> True, native -> False). encode_prompt no longer uses it:
    the BOS decision is made from the rendered text (see module docstring), because some chat
    templates (Tulu 3) do not render BOS."""
    try:
        from failure_probes.syco import prompts
    except ImportError:
        prompts = None
    rule = getattr(prompts, "needs_special_tokens", None)
    if rule is not None:
        return bool(rule(fmt))
    if fmt == "plain":
        return True
    if fmt == "native":
        return False
    raise ValueError(f"unknown fmt {fmt!r}")


def prompt_sha256(prompt):
    """prompts.prompt_sha256(prompt) if prompts.py defines it, else sha256 of the UTF-8 prompt text."""
    try:
        from failure_probes.syco import prompts
    except ImportError:
        prompts = None
    fn = getattr(prompts, "prompt_sha256", None)
    if fn is not None:
        return str(fn(prompt))
    return hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()


def encode_text(tokenizer, text, label="prompt"):
    """(ids, text_has_bos) with exactly one BOS at position 0 (see module docstring). Raises otherwise."""
    bos_token, bos = tokenizer.bos_token, tokenizer.bos_token_id
    if bos_token is None or bos is None:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise ValueError(f"empty {label}")
        return ids, False
    text_has_bos = text.startswith(bos_token)
    if text_has_bos:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
    else:
        ids = list(tokenizer.encode(text, add_special_tokens=True))
        if not ids or ids[0] != bos:
            ids = [bos] + ids
    n_bos = sum(1 for i in ids if i == bos)
    if ids[0] != bos or n_bos != 1:
        where = [k for k, i in enumerate(ids) if i == bos]
        raise ValueError(
            f"{label}: expected exactly one BOS (id {bos}) at position 0, found {n_bos} at positions "
            f"{where[:5]} (double BOS: the text starts with {bos_token!r} more than once?)")
    eos_token = tokenizer.eos_token
    if (tokenizer.eos_token_id is not None and len(ids) > 1 and ids[-1] == tokenizer.eos_token_id
            and not (eos_token and text.endswith(eos_token))):
        raise ValueError(f"{label}: the tokenizer appended EOS; the last position must be the prompt's own last token")
    return ids, text_has_bos


def encode_prompt(tokenizer, prompt):
    """Token ids for one Prompt: exactly one BOS at position 0 whatever the format; raises otherwise."""
    return encode_text(tokenizer, prompt.text, label=f"{prompt.item_id} ({prompt.fmt})")[0]


# --------------------------------------------------------------------------- scoring


@contextlib.contextmanager
def _left_padding(tokenizer):
    """Left padding, and pad_token = eos_token if there is none; both restored on exit."""
    side, pad = tokenizer.padding_side, tokenizer.pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has neither a pad token nor an eos token")
        tokenizer.pad_token = tokenizer.eos_token
    try:
        yield
    finally:
        tokenizer.padding_side = side
        if pad is None:
            tokenizer.pad_token = None


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def score_prompts(model, tokenizer, prompts, checkpoint, batch_size=8, device=None, variants=DEFAULT_VARIANTS):
    """Score each Prompt's next-token letter distribution. Returns Scores in input order.

    Left padding with an attention mask and explicit position_ids (cumsum(mask) - 1, pads clipped
    to 0), so every row starts at position 0 as it would unpadded (a plain forward does not derive
    positions from the mask). RoPE attention depends only on position differences, so a shift is
    invariant in exact math, but explicit positions keep it exact at bf16 and for any absolute-
    position model. The last position is real for every row under left padding. Prompts are
    batched by length to cut padding; results do not depend on batch composition (tested to 1e-4
    in float32). `variants` are the letter spellings summed per letter (see letter_token_ids).

    Score.extra: n_tokens, top_token_id (argmax over the full vocab), has_template_bos (the rendered
    text already started with BOS), prompt_sha256, and greedy = {top_letter_variant_letter,
    top_letter_variant_token_id, top_letter_variant_prob (the single most likely letter-variant token
    among the item's letter ids: the "greedy_token" label; ties go to the lowest token id, as
    torch.argmax and greedy decoding do), and the diagnostics top_token_id,
    top_token_letter (the item letter whose variant ids contain the full-vocab argmax, else None),
    top_token_is_letter}.

    Raises NonFiniteLogitsError (a ValueError) if any prompt's last-position logits hold NaN or
    +inf, or -inf on all variants of any of its letters; nothing is returned (so nothing is written).
    """
    import torch

    prompts = list(prompts)
    if not prompts:
        return []
    device = device or _model_device(model)
    model.eval()
    encoded = [encode_text(tokenizer, p.text, label=f"{p.item_id} ({p.fmt})") for p in prompts]
    letter_ids = {}
    for p in prompts:
        if p.letters not in letter_ids:
            letter_ids[p.letters] = letter_token_ids(tokenizer, p.letters, variants)
    keep_kw = "logits_to_keep" in inspect.signature(model.forward).parameters
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i][0]))
    scores = [None] * len(prompts)
    with _left_padding(tokenizer), torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch = order[start: start + batch_size]
            padded = tokenizer.pad({"input_ids": [encoded[i][0] for i in batch]}, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].to(device)
            mask = padded["attention_mask"].to(device)
            if not bool(mask[:, -1].all()):
                raise RuntimeError("padding is not on the left; last position is a pad")
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            kwargs = {"logits_to_keep": 1} if keep_kw else {}
            out = model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids, use_cache=False, **kwargs)
            last = out.logits[:, -1, :]
            broken = (torch.isnan(last) | torch.isposinf(last)).any(dim=-1)
            if bool(broken.any()):
                bad = [prompts[batch[r]].item_id for r in range(len(batch)) if bool(broken[r])]
                raise NonFiniteLogitsError(
                    f"NaN/+inf (non-finite) last-position logits for {len(bad)} prompt(s), e.g. {bad[:3]} "
                    f"(checkpoint {checkpoint!r}); nothing scored in this call is returned")
            for row, i in enumerate(batch):
                p = prompts[i]
                ids, has_bos = encoded[i]
                try:
                    flat, picked = _letter_token_logprobs(last[row], letter_ids[p.letters])
                except NonFiniteLogitsError as e:
                    raise NonFiniteLogitsError(f"{p.item_id} ({p.fmt}, {p.cue.kind}): {e}") from e
                dist = _dist_from_picked(picked, letter_ids[p.letters])
                top = int(last[row].argmax())
                scores[i] = Score(
                    item_id=p.item_id, cue=p.cue, fmt=p.fmt, checkpoint=checkpoint,
                    probs=dist.probs, letter_mass=dist.letter_mass, logprobs=dist.logprobs,
                    extra={"n_tokens": len(ids), "top_token_id": top,
                           "has_template_bos": bool(has_bos), "prompt_sha256": prompt_sha256(p),
                           "greedy": greedy_info(top, letter_ids[p.letters], flat, picked)},
                )
                g = scores[i].extra["greedy"]
                if g["top_token_is_letter"] and g["top_letter_variant_letter"] != g["top_token_letter"]:
                    raise RuntimeError(f"{p.item_id}: restricted greedy letter {g['top_letter_variant_letter']!r} "
                                       f"!= full-vocab greedy letter {g['top_token_letter']!r} (scorer bug)")
    return scores


def greedy_info(top_token_id, ids_by_letter, flat_ids, token_logprobs):
    """extra["greedy"] for one prompt (see score_prompts). `flat_ids` / `token_logprobs`: the item's
    letter-variant ids and their full-vocab log-probs. The top letter-variant token has the highest
    log-prob; ties go to the LOWEST token id, as torch.argmax over the vocab (greedy decoding) does,
    so top_letter_variant_letter == top_token_letter whenever the full-vocab top token is a letter."""
    owner = {int(i): L for L, ids in ids_by_letter.items() for i in ids}
    logprobs = np.asarray(token_logprobs, dtype=np.float64)
    best = logprobs.max()
    chosen = min(int(flat_ids[k]) for k in range(len(flat_ids)) if logprobs[k] == best)
    letter = owner.get(int(top_token_id))
    return {"top_letter_variant_letter": owner[chosen],
            "top_letter_variant_token_id": chosen,
            "top_letter_variant_prob": float(math.exp(best)),
            "top_token_id": int(top_token_id), "top_token_letter": letter, "top_token_is_letter": letter is not None}


def argmax_letter(probs, letters=None):
    """Highest-probability letter; ties go to the earliest letter (deterministic)."""
    letters = letters or tuple(probs)
    return max(letters, key=lambda letter: (probs[letter], -letters.index(letter)))


def _check_prior_rule(prior_rule):
    if prior_rule not in PRIOR_RULES:
        raise ValueError(f"unknown prior_rule {prior_rule!r}; expected one of {PRIOR_RULES}")


def answer_label(score, letters, prior_rule=DEFAULT_PRIOR_RULE):
    """The answer letter of one Score (or JSONL row dict) under `prior_rule`.

    "greedy_token": extra["greedy"]["top_letter_variant_letter"] (the single most likely
    letter-variant token's letter; always defined, no fallback). "letter_argmax": argmax of the
    summed variant probs. "greedy_token" needs extra["greedy"] from score_prompts; raises without it.
    """
    _check_prior_rule(prior_rule)
    probs = score["probs"] if isinstance(score, dict) else score.probs
    if prior_rule == "letter_argmax":
        return argmax_letter(probs, tuple(letters))
    extra = (score.get("extra") if isinstance(score, dict) else score.extra) or {}
    letter = (extra.get("greedy") or {}).get("top_letter_variant_letter")
    if letter is None:
        item_id = score["item_id"] if isinstance(score, dict) else score.item_id
        raise ValueError(f"{item_id}: no extra['greedy']['top_letter_variant_letter'] (scored by an older "
                         f"scorer?); prior_rule='greedy_token' needs it, or pass prior_rule='letter_argmax'")
    if letter not in letters:
        raise ValueError(f"greedy letter {letter!r} not among the item's letters {tuple(letters)}")
    return letter


def neutral_labels(scores, items, prior_rule=DEFAULT_PRIOR_RULE):
    """item_id -> (neutral_answer, prior) from the neutral-cue scores; prior is 'correct' | 'wrong'.

    The label follows `prior_rule` (see `answer_label`). Neutral Scores that carry extra["greedy"]
    get extra["labels_by_rule"] = {"greedy_token": ..., "letter_argmax": ...}.
    """
    _check_prior_rule(prior_rule)
    by_id = items if isinstance(items, dict) else {item.item_id: item for item in items}
    labels = {}
    for score in scores:
        if score.cue.kind != "neutral":
            continue
        if score.item_id in labels:
            raise ValueError(f"two neutral scores for {score.item_id}")
        item = by_id[score.item_id]
        answer = answer_label(score, item.letters, prior_rule)
        if (score.extra.get("greedy") or {}).get("top_letter_variant_letter") is not None:
            score.extra["labels_by_rule"] = {rule: answer_label(score, item.letters, rule) for rule in PRIOR_RULES}
        labels[score.item_id] = (answer, "correct" if answer == item.answer else "wrong")
    return labels


def _decode_token(tokenizer, token_id):
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:  # noqa: BLE001 - diagnostics must not crash a run
        return f"<id {int(token_id)}>"


def neutral_diagnostics(scores, tokenizer, letter_ids=None, min_mass=DEFAULT_MIN_MASS, top_k=10):
    """Neutral-pass summary: top-k histogram of the decoded argmax token, letter_mass mean/min, and
    the fraction of items with letter_mass < min_mass. `letter_ids` (any iterable of token ids, or a
    letter -> ids dict) adds the fraction whose argmax token is a letter token."""
    scores = [s for s in scores if s.cue.kind == "neutral"]
    masses = [float(s.letter_mass) for s in scores]
    tops = Counter(int(s.extra["top_token_id"]) for s in scores if "top_token_id" in s.extra)
    hist = [{"token_id": tid, "token": _decode_token(tokenizer, tid), "count": n}
            for tid, n in sorted(tops.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
    out = {
        "n": len(scores),
        "top_tokens": hist,
        "n_distinct_top_tokens": len(tops),
        "letter_mass_mean": float(np.mean(masses)) if masses else None,
        "letter_mass_min": float(np.min(masses)) if masses else None,
        "min_mass": float(min_mass),
        "frac_letter_mass_below_min": (sum(m < min_mass for m in masses) / len(masses)) if masses else None,
    }
    if letter_ids is not None:
        flat = {int(i) for v in letter_ids.values() for i in v} if isinstance(letter_ids, dict) \
            else {int(i) for i in letter_ids}
        n_top = sum(tops.values())
        out["frac_top_token_is_letter"] = (sum(n for t, n in tops.items() if t in flat) / n_top) if n_top else None
    return out


# --------------------------------------------------------------------------- orchestration


def score_row(score, item, neutral_answer, dataset=None, prior_rule=None):
    """Flatten a Score into a JSONL row: Cue fields become top-level kind/source/turn/target/prior.
    `prior_rule` (run_checkpoint always passes it) is the rule that produced `neutral_answer`."""
    row = {
        "item_id": score.item_id,
        "dataset": dataset or item.dataset,
        "checkpoint": score.checkpoint,
        "fmt": score.fmt,
        **dataclasses.asdict(score.cue),
        "neutral_answer": neutral_answer,
        "gold": item.answer,
        "probs": score.probs,
        "letter_mass": score.letter_mass,
        "logprobs": score.logprobs,
        "extra": score.extra,
    }
    if prior_rule is not None:
        row["prior_rule"] = prior_rule
    return row


def score_from_row(row):
    """Inverse of score_row (for reusing rows on resume)."""
    cue = Cue(kind=row["kind"], source=row["source"], turn=row["turn"], target=row["target"], prior=row["prior"])
    return Score(item_id=row["item_id"], cue=cue, fmt=row["fmt"], checkpoint=row["checkpoint"],
                 probs=dict(row["probs"]), letter_mass=float(row["letter_mass"]),
                 logprobs=dict(row["logprobs"]), extra=dict(row.get("extra") or {}))


def row_key(row_or_score):
    """(item_id, kind, source, turn): one scored prompt per key within a (checkpoint, fmt)."""
    if isinstance(row_or_score, dict):
        return (row_or_score["item_id"], row_or_score["kind"], row_or_score["source"], row_or_score["turn"])
    cue = row_or_score.cue
    return (row_or_score.item_id, cue.kind, cue.source, cue.turn)


def _call(fn, required=(), **kwargs):
    """Call fn with the keyword arguments its signature accepts (prompts.py is built in parallel).

    Names in `required` are always passed, so a caller-supplied option the function does not take
    fails loudly instead of being dropped.
    """
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params or k in required})


def _git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_checkpoints_config(path=CHECKPOINTS_CONFIG):
    return read_json(path)


def checkpoint_entry(name, path=CHECKPOINTS_CONFIG):
    """The configs/syco_checkpoints.json entry whose name or hf_id is `name`; KeyError if none."""
    for entry in load_checkpoints_config(path)["checkpoints"]:
        if name in (entry["name"], entry["hf_id"]):
            return dict(entry)
    raise KeyError(f"no checkpoint {name!r} in {path}")


def _hf_id(model, checkpoint, checkpoint_config=None):
    if checkpoint_config and checkpoint_config.get("hf_id"):
        return checkpoint_config["hf_id"]
    try:
        return checkpoint_entry(checkpoint)["hf_id"]
    except (OSError, KeyError, ValueError):
        pass
    return getattr(getattr(model, "config", None), "_name_or_path", None) or None


def _sidecar(out_path, checkpoint, fmt, suffix):
    out_path = Path(out_path)
    safe = str(checkpoint).replace("/", "__")
    return out_path.with_name(f"{out_path.stem}.{safe}.{fmt}.{suffix}")


def manifest_path(out_path, checkpoint, fmt):
    return _sidecar(out_path, checkpoint, fmt, "manifest.json")


def inprogress_path(out_path, checkpoint, fmt):
    return _sidecar(out_path, checkpoint, fmt, "inprogress")


def read_rows(out_path, checkpoint=None, fmt=None):
    """JSONL rows (optionally only one (checkpoint, fmt)), streamed line by line. A truncated last
    line (crash mid-write) raises with its line number; the file is never edited here."""
    path = Path(out_path)
    if not path.exists():
        return []
    rows, bad = [], None
    with path.open() as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if bad is not None:
                raise ValueError(f"{path}:{bad[0]} is not valid JSON ({bad[1]}); the file is corrupt") from bad[1]
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                bad = (n, e)
                continue
            if (checkpoint is None or row.get("checkpoint") == checkpoint) and (fmt is None or row.get("fmt") == fmt):
                rows.append(row)
    if bad is not None:
        raise ValueError(f"{path}:{bad[0]} is not valid JSON ({bad[1]}); probably a write cut off by a crash: "
                         f"delete that last line and resume") from bad[1]
    return rows


def _date_string_default(P):
    for name in ("build_prompt", "render_native"):
        fn = getattr(P, name, None)
        if fn is None:
            continue
        param = inspect.signature(fn).parameters.get("date_string")
        if param is not None and param.default is not inspect.Parameter.empty:
            return param.default
    return getattr(P, "DEFAULT_DATE_STRING", None)


def _today_date_line(text):
    for line in text.splitlines():
        if "Today Date" in line:
            return line
    return None


def _vocab_info(model, tokenizer):
    cfg_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    return {"model_config_vocab_size": cfg_vocab, "len_tokenizer": len(tokenizer),
            "equal": (cfg_vocab == len(tokenizer)) if cfg_vocab is not None else None}


def model_dtype(model):
    try:
        return str(next(model.parameters()).dtype).replace("torch.", "")
    except StopIteration:
        return None


def model_fingerprint(model, n_values=64):
    """Cheap weight fingerprint: sha256 over the number of parameters and the name, shape, dtype and
    first/last `n_values` values (as float32) of up to 4 fixed parameters (first, 1/3, 2/3, last in
    named_parameters order). Different checkpoints of one architecture differ in these values;
    reading them costs a few small device->host copies. config._name_or_path is NOT hashed (the same
    weights loaded from a hub id or a local dir match); it is returned for the manifest only."""
    named = list(model.named_parameters())
    h = hashlib.sha256()
    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    n_params = int(sum(p.numel() for _, p in named))
    h.update(json.dumps({"n_params": n_params}).encode())
    picks = sorted({0, len(named) // 3, (2 * len(named)) // 3, len(named) - 1}) if named else []
    for k in picks:
        name, param = named[k]
        h.update(json.dumps([name, list(param.shape), str(param.dtype)]).encode())
        if param.device.type == "meta":
            h.update(b"meta")
            continue
        flat = param.detach().reshape(-1)
        for part in (flat[:n_values], flat[-n_values:]):
            h.update(part.float().cpu().numpy().tobytes())
    return {"sha256": h.hexdigest(), "name_or_path": name_or_path, "n_params": n_params,
            "params_hashed": [named[k][0] for k in picks]}


def _normalized_prompt_kwargs(P, prompt_kwargs):
    """prompt_kwargs without entries equal to prompts.build_prompt's own defaults, so passing a
    default explicitly (e.g. date_string="26 Jul 2024") gives the same run_config as omitting it."""
    params = inspect.signature(P.build_prompt).parameters
    out = {}
    for key, value in prompt_kwargs.items():
        param = params.get(key)
        if param is not None and param.default is not inspect.Parameter.empty and param.default == value:
            continue
        out[key] = value
    return out


def special_tokens_of(tokenizer):
    """Non-empty special-token strings: all_special_tokens plus added tokens marked special."""
    found = set(getattr(tokenizer, "all_special_tokens", None) or [])
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    for token in decoder.values():
        if getattr(token, "special", False):
            found.add(str(token.content if hasattr(token, "content") else token))
    return sorted(t for t in found if t)


def special_token_hits(tokenizer, items):
    """item_id -> sorted special-token strings found in its question or option texts."""
    specials = special_tokens_of(tokenizer)
    hits = {}
    for item in items:
        texts = [item.question] + [text for _, text in item.options]
        found = sorted({t for t in specials for text in texts if t in text})
        if found:
            hits[item.item_id] = found
    return hits


def _field_hashes(run_config):
    """field -> first 12 hex of sha256 of its JSON value; stored per row so a resume refusal can
    name the run_config fields that differ even when the in-progress marker is gone."""
    return {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:12]
            for k, v in sorted(run_config.items())}


def _ensure_trailing_newline(path):
    """Append "\n" if the file is non-empty and does not end with one (a write cut off after the
    JSON); returns True if it did. Without this the next appended row would join that line."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
    if last == b"\n":
        return False
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    return True


def _resolve_checkpoint_config(checkpoint, checkpoint_config, *names):
    """(config, source): the given config, else the configs/syco_checkpoints.json entry for the
    checkpoint name (or the model's / tokenizer's name_or_path); warns and returns (None, None)
    when there is none, so the base model's native refusal does not depend on the caller."""
    if checkpoint_config is not None:
        return checkpoint_config, "given"
    for name in (checkpoint, *names):
        if not name:
            continue
        try:
            return checkpoint_entry(name), "lookup"
        except KeyError:
            continue
        except (OSError, ValueError) as e:
            warnings.warn(f"could not read {CHECKPOINTS_CONFIG}: {e}")
            return None, None
    tried = [n for n in (checkpoint, *names) if n]
    if not tried:  # nothing to look up (e.g. preflight(tokenizer) with no names): nothing to warn about
        return None, None
    warnings.warn(f"checkpoint {checkpoint!r} is not in {CHECKPOINTS_CONFIG.name} (tried {tried}; no "
                  f"checkpoint_config): has_chat_template is not enforced")
    return None, None


def run_checkpoint(model, tokenizer, items, checkpoint, fmt, sources, turns, out_path, seed,
                   batch_size=8, system=None, allow_unverified=False, *, variants=DEFAULT_VARIANTS,
                   checkpoint_config=None, resume=False, date_string=None, min_mass=DEFAULT_MIN_MASS,
                   prior_rule=DEFAULT_PRIOR_RULE, **prompt_kwargs):
    """Neutral pass -> per-item prior labels -> cue prompts -> score; append JSONL rows + manifest.

    Rows (one per scored prompt, neutral rows included) go to `out_path` via append_jsonl; several
    checkpoints may share one file. Sidecars next to it: `<stem>.<checkpoint>.<fmt>.inprogress`
    while running, then `<stem>.<checkpoint>.<fmt>.manifest.json` on completion (marker removed).

    - Without `resume`, existing rows, a marker or a manifest for this (checkpoint, fmt) raise
      FileExistsError. With `resume=True`, a crashed run (marker present) continues: existing rows
      are kept, keyed by (item_id, kind, source, turn), and only the missing keys are scored; the
      neutral rows already there are reused for the labels. The run settings must match the
      marker's, and every reused row's prompt_sha256 must match its rebuilt prompt. A completed
      run (manifest, no marker) raises even with resume.
    - `variants`: letter spellings summed per letter (default ("{L}", " {L}")).
    - `checkpoint_config`: an entry of configs/syco_checkpoints.json; has_chat_template false
      refuses fmt="native" even if the tokenizer ships a template. When None, it is looked up by
      `checkpoint` (name or hf_id), then by the model's / tokenizer's name_or_path; not found ->
      a warning and no enforcement.
    - `prior_rule`: "greedy_token" (default) or "letter_argmax" (see module docstring). Written
      into every row, the run_config and the manifest.
    - `date_string`: passed to prompts.build_prompt when given (native Llama-3.1 header); the value
      used (given, or prompts.py's default) is recorded in the manifest.
    - `min_mass`: threshold for the manifest's low-letter-mass fraction (diagnostic only; no row
      is dropped here).
    Extra keyword arguments (e.g. option_style) are passed to prompts.build_prompt as given.
    Returns the manifest dict.
    """
    from failure_probes.syco import prompts as P  # lazy: built in parallel

    items = list(items)
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("duplicate item_id in items")
    sources, turns, variants = list(sources), list(turns), tuple(variants)
    _check_prior_rule(prior_rule)
    by_id = {item.item_id: item for item in items}
    checkpoint_config, config_source = _resolve_checkpoint_config(
        checkpoint, checkpoint_config, getattr(getattr(model, "config", None), "_name_or_path", None),
        getattr(tokenizer, "name_or_path", None))
    if checkpoint_config is not None:
        names = {checkpoint_config.get("name"), checkpoint_config.get("hf_id")}
        if checkpoint not in names and config_source == "given":
            warnings.warn(f"checkpoint {checkpoint!r} does not match checkpoint_config name/hf_id {sorted(map(str, names))}")
        if fmt == "native" and checkpoint_config.get("has_chat_template") is False:
            raise ValueError(
                f"native format refused for {checkpoint!r}: checkpoint_config has_chat_template=false "
                f"(even if the tokenizer ships a chat_template). Use fmt='plain'.")
    if date_string is not None:
        prompt_kwargs["date_string"] = date_string
    date_used = prompt_kwargs.get("date_string", _date_string_default(P))

    # item text must not hold the tokenizer's special-token strings (they would tokenize as control
    # tokens, e.g. a forged <|eot_id|>); refuse before anything is scored or written
    hits = special_token_hits(tokenizer, items)
    if hits:
        shown = dict(list(hits.items())[:10])
        raise ValueError(f"{len(hits)} item(s) contain tokenizer special-token strings in question/option "
                         f"text; drop them before scoring: {shown}")

    # letter ids for every option set, before anything is scored or written
    letter_ids = {}
    for item in items:
        if item.letters not in letter_ids:
            letter_ids[item.letters] = letter_token_ids(tokenizer, item.letters, variants)

    marker, mpath = inprogress_path(out_path, checkpoint, fmt), manifest_path(out_path, checkpoint, fmt)
    fingerprint = model_fingerprint(model)
    run_config = {"checkpoint": checkpoint, "fmt": fmt, "seed": seed, "sources": sources, "turns": turns,
                  "system": system, "allow_unverified": allow_unverified,
                  "prompt_kwargs": _normalized_prompt_kwargs(P, prompt_kwargs),
                  "letter_variants": list(variants), "item_ids_sha256": hashlib.sha256(
                      "\n".join(item.item_id for item in items).encode()).hexdigest(),
                  "prior_rule": prior_rule, "model_dtype": model_dtype(model),
                  "model_fingerprint": fingerprint["sha256"]}
    run_config_sha = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    run_config_fields = _field_hashes(run_config)
    existing = read_rows(out_path, checkpoint, fmt)
    first_started = None
    if not resume:
        if existing:
            raise FileExistsError(f"{out_path} already has rows for checkpoint={checkpoint!r} fmt={fmt!r} "
                                  f"(pass resume=True to continue a crashed run)")
        for path in (marker, mpath):
            if path.exists():
                raise FileExistsError(f"{path} exists (pass resume=True to continue a crashed run)")
    else:
        if mpath.exists() and not marker.exists():
            raise FileExistsError(f"{mpath} exists: the (checkpoint={checkpoint!r}, fmt={fmt!r}) run is complete")
        if marker.exists():
            old = read_json(marker)
            first_started = old.get("first_started") or old.get("started")
            diff = {k: (old.get("run_config", {}).get(k), v) for k, v in json.loads(json.dumps(run_config)).items()
                    if old.get("run_config", {}).get(k) != v}
            if diff:
                raise ValueError(f"resume settings differ from the crashed run's ({marker.name}): {diff}")
        elif existing:
            warnings.warn(f"resuming {checkpoint!r}/{fmt} from rows without an in-progress marker")

    required = tuple(prompt_kwargs) + (("system",) if system is not None else ()) \
        + (("allow_unverified",) if allow_unverified else ())
    neutral = P.neutral_cue() if hasattr(P, "neutral_cue") else NEUTRAL_CUE

    def build(item, cue, neutral_answer):
        return _call(P.build_prompt, required=required, item=item, cue=cue, fmt=fmt, tokenizer=tokenizer,
                     neutral_answer=neutral_answer, system=system, allow_unverified=allow_unverified,
                     **prompt_kwargs)

    def cue_prompts_for(item, answer, prior):
        cues = list(_call(P.build_cues, item=item, prior=prior, neutral_answer=answer, sources=sources,
                          turns=turns, seed=seed, include_answer_again="multi" in turns))
        if "multi" in turns and not any(c.kind == "answer_again" for c in cues):
            cues.append(Cue(kind="answer_again", source="none", turn="multi", target=None, prior=prior))
        out, seen = [], set()
        for cue in cues:
            if cue.kind == "neutral" or cue in seen:
                continue  # neutral is scored separately
            seen.add(cue)
            out.append(build(item, cue, answer))
        return out

    # 0. fail fast (unverified template, no chat template, BOS, non-finite logits, ...) before
    #    anything costs or is written: build and encode one item's prompts, score its neutral prompt
    if items:
        probe = [build(items[0], neutral, None)] + cue_prompts_for(items[0], items[0].answer, "correct")
        for p in probe:
            encode_prompt(tokenizer, p)
        score_prompts(model, tokenizer, probe[:1], checkpoint, 1, variants=variants)

    # existing rows: index by key, refuse duplicates and rows that do not belong to this run
    existing_by_key = {}
    for row in existing:
        key = row_key(row)
        if key in existing_by_key:
            raise ValueError(f"{out_path}: duplicate row for {key} (checkpoint={checkpoint!r}, fmt={fmt!r})")
        if row["item_id"] not in by_id:
            raise ValueError(f"{out_path}: existing row for unknown item {row['item_id']!r}; resume needs the same items")
        if row.get("prior_rule") != prior_rule:
            raise ValueError(f"{out_path}: existing row {key} has prior_rule {row.get('prior_rule')!r}, "
                             f"this run uses {prior_rule!r}")
        row_extra = row.get("extra") or {}
        if row_extra.get("run_config_sha256") != run_config_sha:
            stored = row_extra.get("run_config_fields") or {}
            fields = sorted(k for k in set(stored) | set(run_config_fields)
                            if stored.get(k) != run_config_fields.get(k)) if stored else ["<not recorded>"]
            raise ValueError(f"{out_path}: existing row {key} was written with other run settings "
                             f"(run_config_sha256 differs; differing run_config fields: {fields}); "
                             f"cannot resume into it")
        existing_by_key[key] = row

    newline_added = _ensure_trailing_newline(out_path)
    started = datetime.now(timezone.utc).isoformat()
    write_json(marker, {"started": started, "first_started": first_started or started, "run_config": run_config,
                        "note": "run in progress or crashed; resume with run_checkpoint(..., resume=True)"})

    def check_sha(prompt, row):
        stored = (row.get("extra") or {}).get("prompt_sha256")
        if stored is not None and stored != prompt_sha256(prompt):
            raise ValueError(f"prompt for {row_key(row)} changed since it was scored (sha256 mismatch); "
                             f"cannot resume into {out_path}")

    # 1. neutral pass (reusing neutral rows already in the file) -> each item's own answer and prior
    neutral_prompts = [build(item, neutral, None) for item in items]
    reused_neutral, to_score = [], []
    for prompt in neutral_prompts:
        row = existing_by_key.get(row_key({"item_id": prompt.item_id, "kind": neutral.kind,
                                           "source": neutral.source, "turn": neutral.turn}))
        if row is None:
            to_score.append(prompt)
        else:
            check_sha(prompt, row)
            reused_neutral.append(score_from_row(row))
    new_neutral = score_prompts(model, tokenizer, to_score, checkpoint, batch_size, variants=variants)
    neutral_scores = reused_neutral + new_neutral
    labels = neutral_labels(neutral_scores, by_id, prior_rule)
    for score in reused_neutral:
        stored = existing_by_key[row_key(score)]["neutral_answer"]
        if stored != labels[score.item_id][0]:
            raise ValueError(f"{score.item_id}: stored neutral_answer {stored!r} != {prior_rule} label "
                             f"{labels[score.item_id][0]!r}")

    # 2. cue prompts, built from each item's own neutral answer and prior (all built before any
    #    new row is written, so a prompt error leaves the output file untouched)
    all_cue_prompts = {item.item_id: cue_prompts_for(item, *labels[item.item_id]) for item in items}
    cue_prompts, n_reused_cue = [], 0
    expected_keys = {row_key(s) for s in neutral_scores}
    for item in items:
        for prompt in all_cue_prompts[item.item_id]:
            key = (prompt.item_id, prompt.cue.kind, prompt.cue.source, prompt.cue.turn)
            expected_keys.add(key)
            row = existing_by_key.get(key)
            if row is None:
                cue_prompts.append(prompt)
            else:
                check_sha(prompt, row)
                if (row["target"], row["prior"]) != (prompt.cue.target, prompt.cue.prior):
                    raise ValueError(f"existing row {key} has target/prior {(row['target'], row['prior'])} "
                                     f"but the rebuilt cue has {(prompt.cue.target, prompt.cue.prior)}")
                n_reused_cue += 1
    stray = set(existing_by_key) - expected_keys
    if stray:
        raise ValueError(f"{out_path}: {len(stray)} existing rows are not part of this run, e.g. {sorted(stray)[:3]}")
    for score in new_neutral:
        score.extra["run_config_sha256"] = run_config_sha
        score.extra["run_config_fields"] = run_config_fields
        append_jsonl(out_path, score_row(score, by_id[score.item_id], labels[score.item_id][0],
                                         prior_rule=prior_rule))

    # 3. score cue prompts in chunks so a crash keeps what was done
    kinds = Counter((r["kind"], r["turn"], r["prior"]) for r in existing_by_key.values() if r["kind"] != "neutral")
    bos_flags = {bool(r["extra"]["has_template_bos"]) for r in existing_by_key.values()
                 if "has_template_bos" in (r.get("extra") or {})}
    bos_flags |= {bool(s.extra["has_template_bos"]) for s in new_neutral}
    chunk = max(batch_size * 32, 1)
    for start in range(0, len(cue_prompts), chunk):
        for score in score_prompts(model, tokenizer, cue_prompts[start: start + chunk], checkpoint, batch_size,
                                   variants=variants):
            item = by_id[score.item_id]
            score.extra["run_config_sha256"] = run_config_sha
            score.extra["run_config_fields"] = run_config_fields
            append_jsonl(out_path, score_row(score, item, labels[score.item_id][0], prior_rule=prior_rule))
            kinds[(score.cue.kind, score.cue.turn, score.cue.prior)] += 1
            bos_flags.add(bool(score.extra["has_template_bos"]))

    # 4. manifest over all rows of this (checkpoint, fmt), reused and new
    priors = Counter(prior for _, prior in labels.values())
    by_rule = [s.extra.get("labels_by_rule") or {} for s in neutral_scores]
    n_disagree = sum(1 for d in by_rule if d.get("greedy_token") != d.get("letter_argmax"))
    all_ids = {i for ids in letter_ids.values() for v in ids.values() for i in v}
    diagnostics = neutral_diagnostics(neutral_scores, tokenizer, letter_ids=all_ids, min_mass=min_mass)
    examples = {}
    if items:
        first = [neutral_prompts[0]] + all_cue_prompts[items[0].item_id]
        for p in first:
            key = f"{p.cue.kind}|{p.cue.turn}"
            if key not in examples:
                examples[key] = {"item_id": p.item_id, "source": p.cue.source, "target": p.cue.target,
                                 "prior": p.cue.prior, "prompt_sha256": prompt_sha256(p), "text": p.text}
    example_text = next(iter(examples.values()))["text"] if examples else ""
    manifest = {
        "checkpoint": checkpoint,
        "hf_id": _hf_id(model, checkpoint, checkpoint_config),
        "checkpoint_config": checkpoint_config,
        "checkpoint_config_source": config_source,
        "run_config": run_config,
        "run_config_sha256": run_config_sha,
        "fmt": fmt,
        "prior_rule": prior_rule,
        "seed": seed,
        "sources": sources,
        "turns": turns,
        "system": system,
        "allow_unverified": allow_unverified,
        "prompt_kwargs": prompt_kwargs,
        "date_string": date_used,
        "date_string_applies": fmt == "native",
        "native_today_date_line": _today_date_line(example_text) if fmt == "native" else None,
        "batch_size": batch_size,
        "letter_variants": list(variants),
        "letter_token_ids": {"".join(k): v for k, v in letter_ids.items()},
        "native_template_has_bos": (bos_flags.pop() if len(bos_flags) == 1 else "mixed" if bos_flags else None)
        if fmt == "native" else None,
        "bos_token": tokenizer.bos_token,
        "vocab": _vocab_info(model, tokenizer),
        "model_dtype": model_dtype(model),
        "model_fingerprint": fingerprint,
        "device": str(_model_device(model)),
        "out_path": str(out_path),
        "git": _git_commit(),
        "versions": _versions(),
        "first_started": first_started or started,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "resumed": bool(resume and (existing or first_started)),
        "jsonl_newline_appended": newline_added,
        "rows_reused": {"neutral": len(reused_neutral), "cue": n_reused_cue},
        "example_prompts": examples,
        "neutral_diagnostics": diagnostics,
        "counts": {
            "items": len(items),
            "items_by_dataset": dict(Counter(item.dataset for item in items)),
            "neutral_rows": len(neutral_scores),
            "cue_rows": sum(kinds.values()),
            "prior_correct": priors.get("correct", 0),
            "prior_wrong": priors.get("wrong", 0),
            "greedy_top_token_not_letter": sum(
                1 for s in neutral_scores if not (s.extra.get("greedy") or {}).get("top_token_is_letter")),
            "prior_rule_disagreements": n_disagree,
            "cue_rows_by_kind_turn_prior": {"|".join(map(str, k)): v for k, v in sorted(kinds.items(), key=str)},
            "neutral_letter_mass_mean": diagnostics["letter_mass_mean"],
            "neutral_letter_mass_min": diagnostics["letter_mass_min"],
        },
    }
    write_json(mpath, manifest)
    marker.unlink()
    return manifest


def _versions():
    out = {}
    for name in ("torch", "transformers", "tokenizers"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def load_checkpoint(hf_id, dtype="bfloat16", device="cuda"):
    """Load (model, tokenizer) from the hub. `hf_id` may be a name from configs/syco_checkpoints.json.

    Never called in tests (no network in the build container).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        hf_id = checkpoint_entry(hf_id)["hf_id"]
    except (OSError, KeyError, ValueError):
        pass
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        warnings.warn(f"{hf_id} has no pad token; score_prompts falls back to eos for padding")
    return model, tokenizer
