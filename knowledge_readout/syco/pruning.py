"""B-TAP weight scoring, selection, random control and zeroing (Orgad et al. 2610.28325 §4, App. C.1, C.5).

Plan: knowledge_readout/syco/PRUNING_PLAN.md ("2. pruning.py"). Decisions: papers/2610.28325.md P1, P3a, P3b, P4.

What the paper fixes (App. C.1):
- Eligible weights: the 7 linear projections of every transformer block (q, k, v, o, gate, up, down). Nothing else.
- Per example, per scalar weight w: Delta(w; x, y) = -w * dl/dw, with l = mean cross-entropy over the target
  (response) tokens (Eq 1; "cross-entropy loss of the response tokens", §4).
- Pruning score S_prune(w) = mean of Delta over the pruning set (signed, Eq 2). Preservation score
  S_pres(w) = mean of |Delta| over the preservation set (Eq 3).
- Per matrix: protect the top p fraction by S_pres (p = 0.005% -> PROTECT_FRAC = 5e-5), rank the remaining
  candidates by S_prune as percentile ranks WITHIN the matrix, pool all matrices, take the global top n (n = 1,000),
  zero them. Scores are computed once on the original model; no recomputation after pruning.
- Random control (App. C.5): per matrix, 10 quantile bins of |w|; for every (matrix, bin) sample uniformly without
  replacement as many NON-targeted coordinates as the targeted set has there.

Our choices (the paper does not fix these; each is a judgment call, flagged here):
- Accumulation is at least float32 whatever the model dtype (fp64 only for fp64 test models); harm_pruning sums
  in bf16, P1 says use fp32.
- Protection size per matrix: k = floor(p * numel) (0 for matrices smaller than 1/p = 20,000 weights).
- Ties are broken deterministically everywhere, never "whatever topk returns":
  protection order = (S_pres desc, flat index asc); candidate rank order = (S_prune desc, flat index asc), so tied
  S_prune values get DIFFERENT percentile ranks (the lower flat index ranks higher).
- Percentile rank: R = 1 - (r - 1) / |C_m| for rank r = 1 (highest S_prune) .. |C_m|. So every matrix's best
  candidate has R = 1.0 exactly. This makes many weights tie at the top; `select` breaks R ties by
  (S_prune desc, matrix order, flat index asc) and reports how many weights tie at the cutoff R.
- Memory: only one group of layers is scored at a time (P1), and after each group only the top `keep_top`
  candidates per matrix are kept (a shortlist). `select` raises if a shortlist could have been too short to hold
  every weight above the global cutoff.
- Quantile bins (C.5): bin = floor(rank_by_|w| * 10 / numel), rank ascending by |w|, ties by flat index.
  Computed by sorting (torch.quantile refuses inputs above 16M elements).
- First-token variant (P3a): the same pruning-set scores with l = CE of the first target token only; computed from
  the same forward pass. It is a check (overlap of the two top-n sets), not the main selection.

Example format: dicts with "prompt_ids" (list[int], non-empty) and "target_ids" (list[int], non-empty). The prompt
is the exact eval prompt token ids (BOS included); the target is the reply token ids (e.g. "(C)<eot>").
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
ELIGIBLE = ATTN + MLP
PROTECT_FRAC = 5e-5   # p = 0.005% per matrix (App. C.1 / C.3)
N_PRUNE = 1000        # n (App. C.1)
N_BINS = 10           # magnitude bins for the random control (App. C.5)


# ----------------------------------------------------------------------------------------------- matrices

def eligible_matrices(model):
    """Ordered [(name, weight_param)] for every block's 7 projections: layer 0 q, k, v, o, gate, up, down, layer 1 ...
    Names are the state_dict names, e.g. "model.layers.3.mlp.up_proj.weight"."""
    try:
        blocks = model.model.layers
    except AttributeError as e:
        raise ValueError("expected a Llama-style model with model.model.layers") from e
    out = []
    for i, block in enumerate(blocks):
        for proj in ATTN:
            out.append((f"model.layers.{i}.self_attn.{proj}.weight", getattr(block.self_attn, proj).weight))
        for proj in MLP:
            out.append((f"model.layers.{i}.mlp.{proj}.weight", getattr(block.mlp, proj).weight))
    if len(out) != len(ELIGIBLE) * len(blocks):
        raise ValueError(f"found {len(out)} eligible matrices, expected {len(ELIGIBLE)} x {len(blocks)}")
    return out


def layer_of(name):
    """Layer index from an eligible matrix name."""
    parts = name.split(".")
    if len(parts) < 3 or parts[0] != "model" or parts[1] != "layers":
        raise ValueError(f"not an eligible matrix name: {name!r}")
    return int(parts[2])


def _model_device(model):
    return next(model.parameters()).device


# -------------------------------------------------------------------------------------------------- loss

def is_margin(ex):
    return ex.get("loss", "ce") == "margin"


def _check_example(ex):
    if is_margin(ex):
        for key in ("prompt_ids", "pos_ids", "neg_ids"):
            if not ex.get(key):
                raise ValueError(f"margin example is missing or has empty {key!r}")
        if set(ex["pos_ids"]) & set(ex["neg_ids"]):
            raise ValueError("margin example: pos_ids and neg_ids overlap")
        if "base_prompt_ids" in ex and not ex["base_prompt_ids"]:
            raise ValueError("margin example: empty base_prompt_ids")
        return
    for key in ("prompt_ids", "target_ids"):
        if key not in ex:
            raise ValueError(f"example is missing {key!r}")
        if len(ex[key]) == 0:
            raise ValueError(f"example {key!r} is empty")


def _upcast(t):
    """At least float32 (bf16/fp16 logits -> fp32; fp64 stays fp64 for the float64 tests)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def _target_logits(model, hidden_rows):
    """lm_head on the gathered hidden-state rows only (the vocab projection is the memory peak otherwise)."""
    return model.lm_head(hidden_rows)


def _decoder(model):
    try:
        return model.model
    except AttributeError as e:
        raise ValueError("expected a Llama-style model with .model (decoder) and .lm_head") from e


def token_ce(model, ex):
    """Per-token cross-entropy (at least float32) of the target tokens given the prompt; one forward, no KV
    cache. Hidden states at positions P-1 .. P+T-2 predict target tokens 0 .. T-1; lm_head runs on those rows only."""
    _check_example(ex)
    prompt, target = list(ex["prompt_ids"]), list(ex["target_ids"])
    ids = torch.tensor([prompt + target], dtype=torch.long, device=_model_device(model))
    hidden = _decoder(model)(input_ids=ids, use_cache=False)[0][0]
    p, t = len(prompt), len(target)
    pred = _upcast(_target_logits(model, hidden[p - 1:p + t - 1]))
    return F.cross_entropy(pred, ids[0, p:p + t], reduction="none")


def example_loss(model, ex, first_token_only=False):
    """l(x, y): mean CE over the target tokens, or the CE of the first target token only."""
    ce = token_ce(model, ex)
    return ce[0] if first_token_only else ce.mean()


def batch_ce(model, examples, pad_id=0):
    """One right-padded forward for several examples. Returns (mean_ce, first_ce), each of shape [B] (at least
    float32): each example's own mean CE over its target tokens, and its first-target-token CE. Padding sits after
    every real token, so with the causal mask (and attention_mask) it cannot change any real position.
    lm_head runs only on the rows that predict target tokens."""
    for ex in examples:
        _check_example(ex)
    seqs = [list(ex["prompt_ids"]) + list(ex["target_ids"]) for ex in examples]
    n, width = len(seqs), max(len(s) for s in seqs)
    device = _model_device(model)
    ids = torch.full((n, width), int(pad_id), dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    b_idx, pos, n_tgt, first_pos = [], [], [], []
    for b, (ex, s) in enumerate(zip(examples, seqs)):
        ids[b, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[b, :len(s)] = 1
        p, t = len(ex["prompt_ids"]), len(ex["target_ids"])
        first_pos.append(len(pos))
        b_idx += [b] * t
        pos += list(range(p - 1, p + t - 1))
        n_tgt.append(t)
    ids, attn = ids.to(device), attn.to(device)
    b_idx = torch.tensor(b_idx, dtype=torch.long, device=device)
    pos = torch.tensor(pos, dtype=torch.long, device=device)
    hidden = _decoder(model)(input_ids=ids, attention_mask=attn, use_cache=False)[0]
    pred = _upcast(_target_logits(model, hidden[b_idx, pos]))
    ce = F.cross_entropy(pred, ids[b_idx, pos + 1], reduction="none")
    sums = torch.zeros(n, dtype=ce.dtype, device=device).index_add(0, b_idx, ce)
    mean_ce = sums / torch.tensor(n_tgt, dtype=ce.dtype, device=device)
    first_ce = ce[torch.tensor(first_pos, dtype=torch.long, device=device)]
    return mean_ce, first_ce


def _check_loss(values, what, labels):
    """Fail fast on a non-finite loss (one host sync per batch / example)."""
    bad = ~torch.isfinite(values.detach())
    if bool(bad.any()):
        which = [labels[i] for i in torch.nonzero(bad, as_tuple=True)[0].tolist()]
        raise ValueError(f"non-finite loss in {what}: examples {which}")


def next_token_logprobs(model, prompts, pad_id=0):
    """log-softmax of the next-token logits after each prompt (list of id lists), [len(prompts), vocab], fp32+.
    One right-padded forward; lm_head only on each prompt's last real position."""
    n, width = len(prompts), max(len(x) for x in prompts)
    device = _model_device(model)
    ids = torch.full((n, width), int(pad_id), dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    for r, x in enumerate(prompts):
        ids[r, :len(x)] = torch.tensor(list(x), dtype=torch.long)
        attn[r, :len(x)] = 1
    hidden = _decoder(model)(input_ids=ids.to(device), attention_mask=attn.to(device), use_cache=False)[0]
    last = torch.tensor([len(x) - 1 for x in prompts], device=device)
    rows = hidden[torch.arange(n, device=device), last]
    return F.log_softmax(_upcast(_target_logits(model, rows)), dim=-1)


def batch_margin_loss(model, examples, pad_id=0):
    """Margin loss per example, shape [B] (E2-B/C, ours, not the paper's). For example x:
        m(prompt) = logsumexp(log p(pos_ids | prompt)) - logsumexp(log p(neg_ids | prompt))   (next token)
        loss = -(m(prompt_ids) - m(base_prompt_ids))   (base term omitted when the example has no base prompt).
    pos_ids/neg_ids are the token ids of every spelling of the two letters ("(C", "C", " C"), so a change that
    moves all letters together (the answer format, total letter mass) cancels exactly: the softmax normaliser drops
    out of a log-probability difference. The base (neutral) prompt removes what holds without the push."""
    prompts, owner = [], []
    for k, ex in enumerate(examples):
        prompts.append(ex["prompt_ids"]); owner.append((k, 1.0))
        if ex.get("base_prompt_ids"):
            prompts.append(ex["base_prompt_ids"]); owner.append((k, -1.0))
    lp = next_token_logprobs(model, prompts, pad_id=pad_id)
    device = lp.device
    m = torch.zeros(len(examples), dtype=lp.dtype, device=device)
    for r, (k, sign) in enumerate(owner):
        ex = examples[k]
        pos = torch.logsumexp(lp[r, torch.tensor(ex["pos_ids"], device=device)], dim=0)
        neg = torch.logsumexp(lp[r, torch.tensor(ex["neg_ids"], device=device)], dim=0)
        m = m.index_add(0, torch.tensor([k], device=device), (sign * (pos - neg)).reshape(1))
    return -m


# ------------------------------------------------------------------------------------------------ scoring

@dataclass
class Shortlist:
    """Top `keep_top` pruning candidates of one matrix, best first."""
    flat_index: torch.Tensor   # int64, CPU
    s_prune: torch.Tensor      # float32, CPU
    r: torch.Tensor            # float64 percentile ranks, CPU, strictly decreasing
    n_candidates: int          # |C_m| = numel - n_protected
    n_protected: int
    numel: int
    shape: tuple

    @property
    def truncated(self):
        return len(self.flat_index) < self.n_candidates


@dataclass
class ScoreResult:
    matrices: list                       # eligible matrix names in model order (only the scored layers)
    shapes: dict                         # name -> (rows, cols)
    main: dict                           # name -> Shortlist (l = mean CE over the full target)
    first: dict | None                   # name -> Shortlist (l = first-token CE), if requested
    n_prune: int
    n_pres: int
    settings: dict = field(default_factory=dict)


def _check_finite(name, t):
    if not torch.isfinite(t).all():
        raise ValueError(f"{name}: non-finite scores")


def protected_mask(s_pres, protect_frac=PROTECT_FRAC):
    """Bool mask (flat) of the k = floor(protect_frac * numel) weights with the highest S_pres; ties by flat index."""
    flat = s_pres.reshape(-1)
    n = flat.numel()
    k = int(math.floor(protect_frac * n))
    mask = torch.zeros(n, dtype=torch.bool, device=flat.device)
    if k > 0:
        order = torch.sort(-flat, stable=True).indices   # S_pres desc, ties keep flat-index order
        mask[order[:k]] = True
    return mask


def shortlist(s_prune, s_pres, protect_frac=PROTECT_FRAC, keep_top=4000):
    """Protect, percentile-rank the remaining candidates within the matrix, keep the top `keep_top`."""
    if s_prune.shape != s_pres.shape:
        raise ValueError("s_prune and s_pres must have the same shape")
    if keep_top < 1:
        raise ValueError("keep_top must be >= 1")
    _check_finite("s_prune", s_prune)
    _check_finite("s_pres", s_pres)
    shape = tuple(s_prune.shape)
    sp = s_prune.reshape(-1).float()
    prot = protected_mask(s_pres, protect_frac)
    cand = torch.nonzero(~prot, as_tuple=True)[0]            # ascending flat index
    n_cand = cand.numel()
    if n_cand == 0:
        raise ValueError("every weight in the matrix is protected")
    order = torch.sort(-sp[cand], stable=True).indices        # S_prune desc, ties by flat index asc
    top = order[:keep_top]
    flat = cand[top]
    ranks = torch.arange(top.numel(), dtype=torch.float64)    # r - 1
    r = 1.0 - ranks / n_cand
    return Shortlist(flat_index=flat.cpu(), s_prune=sp[flat].cpu(), r=r, n_candidates=int(n_cand),
                     n_protected=int(prot.sum()), numel=sp.numel(), shape=shape)


def _acc_dtype(p):
    """Accumulator dtype: at least float32 (bf16/fp32 models -> fp32; fp64 test models -> fp64)."""
    return torch.promote_types(p.dtype, torch.float32)


def _group_scores(model, named, prune_examples, pres_examples, first_token_variant=False, progress=None,
                  prune_batch_size=16, pad_id=0):
    """Full fp32 score tensors for the given (name, param) list: S_prune (mean Delta), S_pres (mean |Delta|) and,
    if requested, S_prune_first. Only the params in `named` may require grad (caller's job).

    Speed: S_prune is linear in the gradient, mean_i(-w g_i) = -w * (sum_i g_i) / n, so the pruning set runs in
    batches of `prune_batch_size` (loss = sum of each example's own mean CE; one backward per batch). The
    preservation score needs |Delta| per example, so it runs one example per backward. Pruning examples are
    batched longest first (stable), so the first batch is the memory peak; the means do not depend on the order.
    A non-finite loss raises at once, naming the example (index into the caller's list)."""
    params = [p for _, p in named]
    margin = [is_margin(ex) for ex in prune_examples]
    if any(margin) and not all(margin):
        raise ValueError("the pruning set must be all CE examples or all margin examples")
    margin = bool(margin and margin[0])
    if margin and first_token_variant:
        raise ValueError("first_token_variant applies to CE pruning examples only")
    seq_len = (lambda ex: len(ex["prompt_ids"]) + len(ex.get("base_prompt_ids") or [])) if margin else \
        (lambda ex: len(ex["prompt_ids"]) + len(ex["target_ids"]))
    by_len = sorted(range(len(prune_examples)), key=lambda i: -seq_len(prune_examples[i]))
    acc_dtype = {n: _acc_dtype(p) for n, p in named}
    acc_prune = {n: torch.zeros_like(p, dtype=acc_dtype[n]) for n, p in named}
    acc_pres = {n: torch.zeros_like(p, dtype=acc_dtype[n]) for n, p in named}
    acc_first = {n: torch.zeros_like(p, dtype=acc_dtype[n]) for n, p in named} if first_token_variant else None

    bs = max(1, int(prune_batch_size))
    for start in range(0, len(by_len), bs):
        idx = by_len[start:start + bs]
        if margin:
            mean_ce, first_ce = batch_margin_loss(model, [prune_examples[i] for i in idx], pad_id=pad_id), None
        else:
            mean_ce, first_ce = batch_ce(model, [prune_examples[i] for i in idx], pad_id=pad_id)
        _check_loss(mean_ce, "pruning set", [f"prune[{i}]" for i in idx])
        if first_token_variant:
            _check_loss(first_ce, "pruning set (first token)", [f"prune[{i}]" for i in idx])
        grads = torch.autograd.grad(mean_ce.sum(), params, retain_graph=first_token_variant)
        for (n, p), g in zip(named, grads):
            acc_prune[n].sub_(p.detach().to(acc_dtype[n]) * g.to(acc_dtype[n]))          # += -w * sum_i dl_i/dw
        del grads
        if first_token_variant:
            grads_f = torch.autograd.grad(first_ce.sum(), params)
            for (n, p), g in zip(named, grads_f):
                acc_first[n].sub_(p.detach().to(acc_dtype[n]) * g.to(acc_dtype[n]))
            del grads_f
        del mean_ce, first_ce
        if progress:
            progress("prune", min(start + bs, len(prune_examples)), len(prune_examples))
    for i, ex in enumerate(pres_examples):
        loss = batch_margin_loss(model, [ex], pad_id=pad_id)[0] if is_margin(ex) else token_ce(model, ex).mean()
        _check_loss(loss.reshape(1), "preservation set", [f"pres[{i}]"])
        grads = torch.autograd.grad(loss, params)
        for (n, p), g in zip(named, grads):
            acc_pres[n].add_((p.detach().to(acc_dtype[n]) * g.to(acc_dtype[n])).abs())   # += |-w * dl/dw|
        del loss, grads
        if progress:
            progress("pres", i + 1, len(pres_examples))

    n_pr, n_ps = len(prune_examples), len(pres_examples)
    for n in acc_prune:
        acc_prune[n].div_(n_pr)
        acc_pres[n].div_(n_ps)
        if first_token_variant:
            acc_first[n].div_(n_pr)
    return acc_prune, acc_pres, acc_first


def score(model, prune_examples, pres_examples, layers=None, group_size=8, first_token_variant=False,
          keep_top=4000, protect_frac=PROTECT_FRAC, progress=None, prune_batch_size=16, pad_id=0):
    """B-TAP scores for the eligible matrices of `layers` (default: all), one group of `group_size` layers at a time.

    Returns a ScoreResult with per-matrix shortlists. Full score tensors live only for one group. requires_grad
    flags of every parameter are restored afterwards (also on error).
    Cost: n_groups x (ceil(n_prune / prune_batch_size) + n_pres) forward+backward passes.
    Splitting across GPUs/processes: give each process a disjoint `layers` list, then `merge_results`.
    TF32 and other global numeric flags are the caller's choice; this module sets none.
    progress: optional callable(phase, i, n) with phase in {"group", "prune", "pres"}."""
    if not prune_examples or not pres_examples:
        raise ValueError("need non-empty pruning and preservation sets")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    for ex in list(prune_examples) + list(pres_examples):
        _check_example(ex)
    all_named = eligible_matrices(model)
    n_layers = len(all_named) // len(ELIGIBLE)
    layers = list(range(n_layers)) if layers is None else sorted(set(layers))
    if any(not 0 <= l < n_layers for l in layers):
        raise ValueError(f"layers must be in [0, {n_layers})")

    named_params = list(model.named_parameters())
    saved_flags = [(p, p.requires_grad) for _, p in named_params]
    was_training = model.training
    main, first, shapes, order = {}, ({} if first_token_variant else None), {}, []
    groups = [layers[i:i + group_size] for i in range(0, len(layers), group_size)]
    hook = None
    try:
        for _, p in named_params:
            p.requires_grad_(False)
        if getattr(model, "is_gradient_checkpointing", False):
            # Reentrant checkpointing only builds a graph when a block's INPUT requires grad; with every embedding
            # weight frozen, nothing would. Mark the embedding output as requiring grad (what HF's
            # enable_input_require_grads does). Cost: backward then runs down to the embeddings for every group.
            def _req_grad(_module, _inputs, output):
                output.requires_grad_(True)
            hook = model.get_input_embeddings().register_forward_hook(_req_grad)
        for gi, group in enumerate(groups):
            if progress:
                progress("group", gi + 1, len(groups))
            named = [(n, p) for n, p in all_named if layer_of(n) in set(group)]
            for _, p in named:
                p.requires_grad_(True)
            s_pr, s_ps, s_first = _group_scores(model, named, prune_examples, pres_examples,
                                                first_token_variant, progress, prune_batch_size, pad_id)
            for n, p in named:
                order.append(n)
                shapes[n] = tuple(p.shape)
                main[n] = shortlist(s_pr[n], s_ps[n], protect_frac, keep_top)
                if first_token_variant:
                    first[n] = shortlist(s_first[n], s_ps[n], protect_frac, keep_top)
                p.requires_grad_(False)
            del s_pr, s_ps, s_first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if hook is not None:
            hook.remove()
        for p, flag in saved_flags:
            p.requires_grad_(flag)
        model.train(was_training)
    settings = {"protect_frac": protect_frac, "keep_top": keep_top, "group_size": group_size, "layers": layers,
                "first_token_variant": first_token_variant,
                "accumulator_dtype": str(_acc_dtype(next(model.parameters()))),
                "prune_batch_size": prune_batch_size, "model_dtype": str(next(model.parameters()).dtype),
                "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False))}
    return ScoreResult(matrices=order, shapes=shapes, main=main, first=first, n_prune=len(prune_examples),
                       n_pres=len(pres_examples), settings=settings)


_MERGE_MUST_MATCH = ("protect_frac", "keep_top", "first_token_variant", "accumulator_dtype", "model_dtype")


def _matrix_sort_key(name):
    proj = name.split(".")[-2]
    return layer_of(name), ELIGIBLE.index(proj)


def merge_results(*results):
    """Combine ScoreResults computed over DISJOINT layers (e.g. one per GPU) into one that `select` accepts.
    Exact: percentile ranks are within-matrix, so nothing depends on which process scored which layer.
    The examples (counts) and the settings that change scores must match."""
    if not results:
        raise ValueError("nothing to merge")
    base = results[0]
    for r in results[1:]:
        if (r.n_prune, r.n_pres) != (base.n_prune, base.n_pres):
            raise ValueError("results were scored on different example counts")
        for key in _MERGE_MUST_MATCH:
            if r.settings.get(key) != base.settings.get(key):
                raise ValueError(f"setting {key!r} differs: {r.settings.get(key)!r} vs {base.settings.get(key)!r}")
    names = [n for r in results for n in r.matrices]
    if len(set(names)) != len(names):
        raise ValueError("results overlap (a matrix was scored twice)")
    order = sorted(names, key=_matrix_sort_key)
    main, shapes = {}, {}
    first = {} if base.first is not None else None
    for r in results:
        main.update(r.main)
        shapes.update(r.shapes)
        if first is not None:
            first.update(r.first)
    settings = dict(base.settings)
    settings["layers"] = sorted(l for r in results for l in r.settings.get("layers", []))
    settings["merged_from"] = len(results)
    return ScoreResult(matrices=order, shapes=shapes, main=main, first=first, n_prune=base.n_prune,
                       n_pres=base.n_pres, settings=settings)


# ---------------------------------------------------------------------------------------------- selection

def _rc(flat, shape):
    cols = shape[1]
    return int(flat) // cols, int(flat) % cols


def select(result, n=N_PRUNE, variant="main", return_info=False):
    """Global top n by percentile rank R over all matrices' shortlists (Eq 4).

    R ties (every matrix's best candidate has R = 1.0) are broken by (S_prune desc, matrix order, flat index asc);
    this tie rule is ours, the paper does not give one. Raises if a truncated shortlist could hide a weight above
    the cutoff (raise keep_top in `score`). Returns [{matrix, flat_index, row, col, shape, s_prune, r}] best first, and
    with return_info=True also a dict (cutoff_r, tied_at_cutoff_in_pool, tied_at_cutoff_chosen, per_matrix)."""
    lists = result.main if variant == "main" else result.first if variant == "first" else None
    if lists is None:
        raise ValueError(f"variant {variant!r} not available (main | first, first needs first_token_variant)")
    if n < 1:
        raise ValueError("n must be >= 1")
    rs, sps, mis, flats = [], [], [], []
    for mi, name in enumerate(result.matrices):
        sl = lists[name]
        k = len(sl.flat_index)
        rs.append(sl.r.numpy())
        sps.append(sl.s_prune.double().numpy())
        mis.append(np.full(k, mi, dtype=np.int64))
        flats.append(sl.flat_index.numpy())
    r, sp, mi, fl = (np.concatenate(x) for x in (rs, sps, mis, flats))
    if len(r) < n:
        raise ValueError(f"only {len(r)} candidates in the shortlists, need {n}")
    order = np.lexsort((fl, mi, -sp, -r))          # last key is primary: R desc, S_prune desc, matrix, flat
    chosen = order[:n]
    cutoff = r[chosen[-1]]
    per_matrix = {}
    for j in chosen:
        name = result.matrices[mi[j]]
        per_matrix[name] = per_matrix.get(name, 0) + 1
    for name, cnt in per_matrix.items():
        sl = lists[name]
        if sl.truncated and (cnt >= len(sl.flat_index) or sl.r[-1].item() > cutoff):
            raise ValueError(f"{name}: shortlist of {len(sl.flat_index)} may be too short for the cutoff "
                             f"R={cutoff}; rerun score() with a larger keep_top")
    for name in result.matrices:   # a truncated shortlist whose last R is above the cutoff hides better weights
        sl = lists[name]
        if sl.truncated and sl.r[-1].item() > cutoff:
            raise ValueError(f"{name}: shortlist too short (last R {sl.r[-1].item()} > cutoff {cutoff})")
    out = []
    for j in chosen:
        name = result.matrices[mi[j]]
        row, col = _rc(fl[j], result.shapes[name])
        out.append({"matrix": name, "flat_index": int(fl[j]), "row": row, "col": col,
                    "shape": list(result.shapes[name]), "s_prune": float(sp[j]), "r": float(r[j])})
    if not return_info:
        return out
    info = {"cutoff_r": float(cutoff), "tied_at_cutoff_in_pool": int((r == cutoff).sum()),
            "tied_at_cutoff_chosen": int((r[chosen] == cutoff).sum()), "per_matrix": per_matrix}
    return out, info


# -------------------------------------------------------------------------------------------- random control

def magnitude_bins(weight, n_bins=N_BINS):
    """Flat int64 bin per weight (CPU): floor(rank * n_bins / numel), rank ascending by |w|, ties by flat index.
    Sorted on the weight's own device (GPU in practice), then moved to CPU."""
    flat = weight.detach().reshape(-1).float().abs()
    n = flat.numel()
    order = torch.sort(flat, stable=True).indices
    rank = torch.empty(n, dtype=torch.long, device=flat.device)
    rank[order] = torch.arange(n, dtype=torch.long, device=flat.device)
    return ((rank * n_bins) // n).cpu()


def random_matched(model, selected, seed, n_bins=N_BINS):
    """Magnitude-matched random control (App. C.5): same count per (matrix, |w|-bin), non-targeted coordinates,
    uniform without replacement. Deterministic for a seed (CPU generator; matrices in model order, bins ascending)."""
    params = dict(eligible_matrices(model))
    order = [n for n, _ in eligible_matrices(model)]
    by_matrix = {}
    for s in selected:
        if s["matrix"] not in params:
            raise ValueError(f"unknown matrix {s['matrix']!r}")
        by_matrix.setdefault(s["matrix"], []).append(int(s["flat_index"]))
    gen = torch.Generator().manual_seed(int(seed))
    out = []
    for name in order:
        if name not in by_matrix:
            continue
        w = params[name]
        shape = list(w.shape)
        targeted = torch.tensor(sorted(set(by_matrix[name])), dtype=torch.long)
        if len(targeted) != len(by_matrix[name]):
            raise ValueError(f"{name}: duplicate targeted indices")
        bins = magnitude_bins(w, n_bins)            # one matrix at a time; freed below
        tmask = torch.zeros(bins.numel(), dtype=torch.bool)
        tmask[targeted] = True
        t_bins = bins[targeted]
        want = torch.bincount(t_bins, minlength=n_bins)
        picked_bins = []
        for b in range(n_bins):
            c = int(want[b])
            if c == 0:
                continue
            pool = torch.nonzero((bins == b) & ~tmask, as_tuple=True)[0]
            if pool.numel() < c:
                raise ValueError(f"{name} bin {b}: only {pool.numel()} non-targeted weights, need {c}")
            pick = pool[torch.randperm(pool.numel(), generator=gen)[:c]]
            for f in sorted(pick.tolist()):
                row, col = _rc(f, tuple(shape))
                out.append({"matrix": name, "flat_index": f, "row": row, "col": col, "shape": shape,
                            "s_prune": None, "r": None, "bin": b})
                picked_bins.append(b)
        got = torch.bincount(torch.tensor(picked_bins, dtype=torch.long), minlength=n_bins)
        if not torch.equal(want, got):
            raise AssertionError(f"{name}: random control counts differ from the targeted set")
        del bins, tmask, t_bins
    return out


# ------------------------------------------------------------------------------------------------- zeroing

def _check_shape(name, actual, expected, where):
    if expected is not None and tuple(expected) != tuple(actual):
        raise ValueError(f"{name}: shape {tuple(actual)} in the model but {tuple(expected)} in {where}")


def apply_zeroing(model, indices, expected_shapes=None):
    """Zero the given weights in place. Returns [(name, flat_idx_tensor, old_values)] for `restore`.
    Safety: every touched matrix's shape must match `expected_shapes[name]` (if given) and each index's own
    "shape" field (if present), else ValueError before anything is changed."""
    params = dict(eligible_matrices(model))
    by_matrix = {}
    for s in indices:
        if s["matrix"] not in params:
            raise ValueError(f"unknown matrix {s['matrix']!r}")
        name = s["matrix"]
        _check_shape(name, params[name].shape, s.get("shape"), "the index entry")
        if expected_shapes is not None:
            if name not in expected_shapes:
                raise ValueError(f"{name}: no expected shape given")
            _check_shape(name, params[name].shape, expected_shapes[name], "expected_shapes")
        by_matrix.setdefault(name, []).append(int(s["flat_index"]))
    for name, idxs in by_matrix.items():           # validate everything before touching any weight
        p = params[name]
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"{name}: duplicate indices")
        if not p.is_contiguous():
            raise ValueError(f"{name}: weight is not contiguous")
        if max(idxs) >= p.numel() or min(idxs) < 0:
            raise ValueError(f"{name}: index out of range")
    saved = []
    with torch.no_grad():
        for name, idxs in by_matrix.items():
            p = params[name]
            flat = p.view(-1)
            idx = torch.tensor(idxs, dtype=torch.long, device=p.device)
            saved.append((name, idx, flat[idx].clone()))
            flat[idx] = 0
    return saved


def restore(model, saved):
    params = dict(eligible_matrices(model))
    with torch.no_grad():
        for name, idx, old in saved:
            params[name].view(-1)[idx] = old


def overlap(a, b):
    """|a ∩ b| / |a| on (matrix, flat_index)."""
    if not a:
        raise ValueError("first index set is empty")
    sa = {(s["matrix"], int(s["flat_index"])) for s in a}
    sb = {(s["matrix"], int(s["flat_index"])) for s in b}
    return len(sa & sb) / len(sa)


def save_indices(path, indices, meta=None, shapes=None, model_id=None):
    """Write {"meta", "indices"} JSON. meta["shapes"] = {matrix: [rows, cols]} for every touched matrix (from
    `shapes`, e.g. result.shapes, or from the entries' own "shape" fields); meta["model_id"] if given."""
    meta = dict(meta or {})
    touched = sorted({s["matrix"] for s in indices}, key=_matrix_sort_key)
    out_shapes = {}
    for name in touched:
        if shapes is not None and name in shapes:
            out_shapes[name] = list(shapes[name])
        else:
            found = {tuple(s["shape"]) for s in indices if s["matrix"] == name and s.get("shape") is not None}
            if len(found) != 1:
                raise ValueError(f"{name}: no single shape known; pass shapes= (e.g. result.shapes)")
            out_shapes[name] = list(found.pop())
    meta["shapes"] = out_shapes
    if model_id is not None:
        meta["model_id"] = model_id
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "indices": indices}, indent=1))


def load_indices(path):
    data = json.loads(Path(path).read_text())
    return data["indices"], data.get("meta", {})


def check_indices_match(model, meta):
    """Raise ValueError unless meta["shapes"] exists and every listed matrix exists in `model` with that shape."""
    shapes = meta.get("shapes")
    if not shapes:
        raise ValueError("index file has no 'shapes' in meta; cannot check it against the model")
    params = dict(eligible_matrices(model))
    for name, shape in shapes.items():
        if name not in params:
            raise ValueError(f"{name}: not an eligible matrix of this model")
        _check_shape(name, params[name].shape, shape, "the index file")


def parse_layer_spec(spec):
    """"0-7,24-31" -> [0, 1, ..., 7, 24, ..., 31] (sorted, unique). Single numbers are allowed ("3,5-6")."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("empty layer spec")
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty item in layer spec {spec!r}")
        if "-" in part:
            lo, hi = part.split("-", 1)
            if not lo.strip().isdigit() or not hi.strip().isdigit():
                raise ValueError(f"bad range {part!r} in layer spec {spec!r}")
            lo, hi = int(lo), int(hi)
            if lo > hi:
                raise ValueError(f"range {part!r} goes backwards")
            out.update(range(lo, hi + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"bad layer {part!r} in layer spec {spec!r}")
            out.add(int(part))
    return sorted(out)
