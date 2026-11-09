"""Tests for knowledge_readout/syco/pruning.py (B-TAP), on the tiny offline model, CPU."""
import pytest
import torch

from knowledge_readout.syco import pruning as PR
from syco_tiny import tiny_model, tiny_tokenizer


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


def _model(tok, dtype=torch.float32):
    m = tiny_model(tok)
    return m.to(dtype)


def _ex(tok, prompt, target):
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"prompt_ids": p, "target_ids": t}


@pytest.fixture(scope="module")
def sets(tok):
    prune = [_ex(tok, "What is the answer? I believe the answer is C", " C"),
             _ex(tok, "Which option? I think B", "(B)"),
             _ex(tok, "Give your final answer", " A")]
    pres = [_ex(tok, "What is the question again", " D"),
            _ex(tok, "Please answer the question", "(E)")]
    return prune, pres


def _brute(model, examples, signed):
    """Per-example full backward on the whole model; mean of -w*g (signed) or |-w*g|."""
    named = PR.eligible_matrices(model)
    acc = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in named}
    for ex in examples:
        model.zero_grad()
        PR.example_loss(model, ex).backward()
        for n, p in named:
            d = -(p.detach().double() * p.grad.double())
            acc[n] += d if signed else d.abs()
    model.zero_grad()
    return {n: a / len(examples) for n, a in acc.items()}


def test_eligible_matrices(tok):
    m = _model(tok)
    named = PR.eligible_matrices(m)
    assert len(named) == 14
    assert named[0][0] == "model.layers.0.self_attn.q_proj.weight"
    assert named[-1][0] == "model.layers.1.mlp.down_proj.weight"
    assert [PR.layer_of(n) for n, _ in named] == [0] * 7 + [1] * 7


def test_example_loss_targets_only(tok, sets):
    m = _model(tok)
    ex = sets[0][0]
    ids = torch.tensor([ex["prompt_ids"] + ex["target_ids"]])
    with torch.no_grad():
        logits = m(input_ids=ids).logits[0]
        P = len(ex["prompt_ids"])
        lp = torch.log_softmax(logits.float(), -1)
        manual = torch.stack([-lp[P - 1 + i, t] for i, t in enumerate(ex["target_ids"])])
        assert torch.allclose(PR.example_loss(m, ex), manual.mean(), atol=1e-6)
        assert torch.allclose(PR.example_loss(m, ex, first_token_only=True), manual[0], atol=1e-6)
    with pytest.raises(ValueError):
        PR.example_loss(m, {"prompt_ids": [1, 2], "target_ids": []})


def _true_float64(m):
    """HF LlamaRMSNorm computes in float32 even for float64 models; patch it to stay in the input dtype so the
    finite-difference check can be tight."""
    def fwd(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.variance_epsilon))
    for mod in m.modules():
        if type(mod).__name__.endswith("RMSNorm"):
            mod.forward = fwd.__get__(mod)
    return m


def test_attribution_matches_finite_difference_float64(tok, sets):
    m = _true_float64(_model(tok, torch.float64))
    ex = sets[0][0]
    named = dict(PR.eligible_matrices(m))
    for name in ("model.layers.1.mlp.up_proj.weight", "model.layers.0.self_attn.k_proj.weight"):
        p = named[name]
        m.zero_grad()
        PR.example_loss(m, ex).backward()
        eps = 1e-5
        for flat in (0, 17, 333, 1000):
            w = p.view(-1)[flat].item()
            delta = -(w * p.grad.view(-1)[flat].item())
            with torch.no_grad():
                p.view(-1)[flat] = w * (1 + eps)
                lp = PR.example_loss(m, ex).item()
                p.view(-1)[flat] = w * (1 - eps)
                lm = PR.example_loss(m, ex).item()
                p.view(-1)[flat] = w
            fd = -(lp - lm) / (2 * eps)
            assert delta == pytest.approx(fd, rel=1e-6, abs=1e-11)
    m.zero_grad()


def test_group_scores_match_brute_force(tok, sets):
    prune, pres = sets
    m = _model(tok, torch.float64)
    named = PR.eligible_matrices(m)
    for _, p in m.named_parameters():
        p.requires_grad_(False)
    for _, p in named:
        p.requires_grad_(True)
    s_pr, s_ps, _ = PR._group_scores(m, named, prune, pres)
    for _, p in m.named_parameters():
        p.requires_grad_(True)
    want_pr = _brute(m, prune, signed=True)
    want_ps = _brute(m, pres, signed=False)
    for n, _ in named:
        assert torch.allclose(s_pr[n].double(), want_pr[n], rtol=1e-5, atol=1e-9)
        assert torch.allclose(s_ps[n].double(), want_ps[n], rtol=1e-5, atol=1e-9)
        assert (s_ps[n] >= 0).all()


def _same_result(a, b, which="main"):
    la, lb = getattr(a, which), getattr(b, which)
    assert a.matrices == b.matrices
    for n in a.matrices:
        assert torch.equal(la[n].flat_index, lb[n].flat_index)
        assert torch.allclose(la[n].s_prune, lb[n].s_prune, rtol=1e-5, atol=1e-8)
        assert torch.equal(la[n].r, lb[n].r)


def test_score_group_size_invariant_and_flags_restored(tok, sets):
    prune, pres = sets
    m = _model(tok, torch.float64)
    # make one non-eligible param frozen to check flags are restored exactly
    m.model.norm.weight.requires_grad_(False)
    before = {n: p.requires_grad for n, p in m.named_parameters()}
    a = PR.score(m, prune, pres, group_size=1, protect_frac=0.01, keep_top=50)
    b = PR.score(m, prune, pres, group_size=99, protect_frac=0.01, keep_top=50)
    _same_result(a, b)
    assert {n: p.requires_grad for n, p in m.named_parameters()} == before
    assert a.n_prune == 3 and a.n_pres == 2
    sl = a.main["model.layers.0.self_attn.q_proj.weight"]
    assert sl.n_protected == int(0.01 * 1024) and sl.n_candidates == 1024 - sl.n_protected
    assert sl.r[0].item() == 1.0 and (sl.r[1:] < sl.r[:-1]).all()


def test_score_shortlist_is_protect_then_rank(tok, sets):
    prune, pres = sets
    m = _model(tok, torch.float64)
    res = PR.score(m, prune, pres, protect_frac=0.02, keep_top=30)
    s_pr = _brute(m, prune, signed=True)
    s_ps = _brute(m, pres, signed=False)
    name = "model.layers.1.mlp.gate_proj.weight"
    sl = res.main[name]
    flat_ps = s_ps[name].reshape(-1)
    k = int(0.02 * flat_ps.numel())
    prot = set(torch.sort(-flat_ps, stable=True).indices[:k].tolist())
    assert prot.isdisjoint(set(sl.flat_index.tolist()))
    cand = [i for i in range(flat_ps.numel()) if i not in prot]
    sp = s_pr[name].reshape(-1)
    want = sorted(cand, key=lambda i: (-sp[i].item(), i))[:30]
    assert sl.flat_index.tolist() == want


def test_zeroing_small_weight_matches_first_order(tok, sets):
    m = _model(tok, torch.float64)
    ex = sets[0][1]
    named = dict(PR.eligible_matrices(m))
    p = named["model.layers.0.self_attn.v_proj.weight"]
    flat = int(p.detach().abs().reshape(-1).argmin())
    m.zero_grad()
    loss = PR.example_loss(m, ex)
    loss.backward()
    delta = -(p.view(-1)[flat].item() * p.grad.view(-1)[flat].item())
    m.zero_grad()
    saved = PR.apply_zeroing(m, [{"matrix": "model.layers.0.self_attn.v_proj.weight", "flat_index": flat}])
    with torch.no_grad():
        actual = PR.example_loss(m, ex).item() - loss.item()
    PR.restore(m, saved)
    assert actual == pytest.approx(delta, rel=0.05, abs=1e-10)


def _fake_result(tok_shapes, sl_map, first=None):
    names = list(sl_map)
    return PR.ScoreResult(matrices=names, shapes=tok_shapes, main=sl_map, first=first, n_prune=1, n_pres=1)


def test_shortlist_protection_percentile_ties():
    s_prune = torch.tensor([[5.0, 4.0, 4.0, 1.0, 0.5]])
    s_pres = torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0]])
    sl = PR.shortlist(s_prune, s_pres, protect_frac=0.2, keep_top=10)   # k = 1 protects index 0
    assert sl.n_protected == 1 and sl.n_candidates == 4
    assert sl.flat_index.tolist() == [1, 2, 3, 4]                        # tie 4.0: lower index first
    assert sl.r.tolist() == [1.0, 0.75, 0.5, 0.25]
    # protection ties broken by flat index
    sl2 = PR.shortlist(s_prune, torch.ones(1, 5), protect_frac=0.4, keep_top=10)
    assert sl2.flat_index.tolist() == [2, 3, 4]
    assert not sl.truncated
    assert PR.shortlist(s_prune, s_pres, protect_frac=0.2, keep_top=2).truncated
    with pytest.raises(ValueError):
        PR.shortlist(torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 2))


def test_select_global_topn_and_ties():
    a = PR.shortlist(torch.tensor([[3.0, 2.0, 1.0, 0.0]]), torch.zeros(1, 4), protect_frac=0.0, keep_top=4)
    b = PR.shortlist(torch.tensor([[9.0, 8.0]]), torch.zeros(1, 2), protect_frac=0.0, keep_top=2)
    res = _fake_result({"model.layers.0.self_attn.q_proj.weight": (1, 4),
                        "model.layers.0.self_attn.k_proj.weight": (1, 2)},
                       {"model.layers.0.self_attn.q_proj.weight": a, "model.layers.0.self_attn.k_proj.weight": b})
    out, info = PR.select(res, n=3, return_info=True)
    # R: a = [1, .75, .5, .25], b = [1, .5]. Top: both R=1 (tie -> S_prune desc: b's 9 first), then a's .75
    assert [(o["matrix"].split(".")[-2], o["flat_index"]) for o in out] == [("k_proj", 0), ("q_proj", 0),
                                                                          ("q_proj", 1)]
    assert info["cutoff_r"] == 0.75 and info["tied_at_cutoff_in_pool"] == 1
    out4, info4 = PR.select(res, n=4, return_info=True)
    # R=0.5 tie between a[2] (s=1) and b[1] (s=8): b first
    assert out4[-1]["matrix"].endswith("k_proj.weight") and info4["tied_at_cutoff_in_pool"] == 2
    assert PR.select(res, n=4) == PR.select(res, n=4)
    with pytest.raises(ValueError):
        PR.select(res, n=7)


def test_select_raises_when_shortlist_too_short():
    big = PR.shortlist(torch.arange(100.0).reshape(1, 100), torch.zeros(1, 100), protect_frac=0.0, keep_top=3)
    small = PR.shortlist(torch.tensor([[1.0, 0.5, 0.2, 0.1]]), torch.zeros(1, 4), protect_frac=0.0, keep_top=4)
    res = _fake_result({"model.layers.0.mlp.up_proj.weight": (1, 100), "model.layers.0.mlp.down_proj.weight": (1, 4)},
                       {"model.layers.0.mlp.up_proj.weight": big, "model.layers.0.mlp.down_proj.weight": small})
    # big's R: 1, .99, .98 (truncated). small's R: 1, .75, .5, .25. n=5 needs cutoff .75 -> big's hidden .97 > cutoff
    with pytest.raises(ValueError, match="keep_top|too short"):
        PR.select(res, n=5)
    PR.select(res, n=2)   # cutoff 1.0: fine


def test_random_matched(tok, sets):
    m = _model(tok)
    prune, pres = sets
    res = PR.score(m, prune, pres, protect_frac=0.01, keep_top=50)
    sel = PR.select(res, n=40)
    r1 = PR.random_matched(m, sel, seed=0)
    r2 = PR.random_matched(m, sel, seed=0)
    r3 = PR.random_matched(m, sel, seed=1)
    assert r1 == r2 and r1 != r3 and len(r1) == 40
    params = dict(PR.eligible_matrices(m))
    want, got = {}, {}
    for s in sel:
        b = int(PR.magnitude_bins(params[s["matrix"]])[s["flat_index"]])
        want[(s["matrix"], b)] = want.get((s["matrix"], b), 0) + 1
    for s in r1:
        got[(s["matrix"], s["bin"])] = got.get((s["matrix"], s["bin"]), 0) + 1
    assert want == got
    assert PR.overlap(sel, r1) == 0.0


def test_magnitude_bins_equal_sizes():
    w = torch.randn(7, 13)
    bins = PR.magnitude_bins(w)
    counts = torch.bincount(bins, minlength=10)
    assert counts.sum() == 91 and counts.max() - counts.min() <= 1
    order = torch.argsort(w.reshape(-1).abs())
    assert (bins[order][1:] >= bins[order][:-1]).all()


def test_zeroing_roundtrip_and_save_load(tok, tmp_path):
    m = _model(tok)
    before = {n: p.detach().clone() for n, p in m.state_dict().items()}
    idx = [{"matrix": "model.layers.0.mlp.up_proj.weight", "flat_index": 5},
           {"matrix": "model.layers.1.self_attn.o_proj.weight", "flat_index": 1000}]
    saved = PR.apply_zeroing(m, idx)
    assert m.state_dict()["model.layers.0.mlp.up_proj.weight"].view(-1)[5] == 0
    PR.restore(m, saved)
    for n, p in m.state_dict().items():
        assert torch.equal(p, before[n])
    with pytest.raises(ValueError):
        PR.apply_zeroing(m, idx + idx[:1])
    shapes = {n: tuple(p.shape) for n, p in PR.eligible_matrices(m)}
    PR.save_indices(tmp_path / "x.json", idx, {"seed": 0}, shapes=shapes, model_id="tiny")
    back, meta = PR.load_indices(tmp_path / "x.json")
    assert back == idx and meta["seed"] == 0 and meta["model_id"] == "tiny"
    assert meta["shapes"] == {"model.layers.0.mlp.up_proj.weight": [64, 32],
                              "model.layers.1.self_attn.o_proj.weight": [32, 32]}
    PR.check_indices_match(m, meta)
    with pytest.raises(ValueError):
        PR.save_indices(tmp_path / "y.json", idx)          # no shape known
    assert PR.overlap(idx, idx[:1]) == 0.5


def test_first_token_variant(tok, sets):
    prune, pres = sets
    m = _model(tok, torch.float64)
    res = PR.score(m, prune, pres, first_token_variant=True, protect_frac=0.01, keep_top=50)
    assert res.first is not None and res.first is not res.main
    assert PR.select(res, n=20, variant="first") != PR.select(res, n=20)
    one = [{"prompt_ids": ex["prompt_ids"], "target_ids": ex["target_ids"][:1]} for ex in prune]
    res1 = PR.score(m, one, pres, first_token_variant=True, protect_frac=0.01, keep_top=50)
    _same_result(PR.ScoreResult(res1.matrices, res1.shapes, res1.first, None, 3, 2), res1)
    with pytest.raises(ValueError):
        PR.select(PR.score(m, prune, pres, protect_frac=0.01, keep_top=50), n=5, variant="first")


def test_batch_ce_matches_single(tok, sets):
    prune, _ = sets
    m = _model(tok, torch.float64)
    with torch.no_grad():
        mean_ce, first_ce = PR.batch_ce(m, prune)
        for b, ex in enumerate(prune):
            ce = PR.token_ce(m, ex)
            assert mean_ce[b].item() == pytest.approx(ce.mean().item(), rel=1e-6)
            assert first_ce[b].item() == pytest.approx(ce[0].item(), rel=1e-6)


def test_batched_prune_scores_equal_per_example(tok, sets):
    prune, pres = sets
    m = _model(tok, torch.float64)
    named = PR.eligible_matrices(m)
    outs = []
    for bs in (1, 2, 16):
        s_pr, s_ps, s_f = PR._group_scores(m, named, prune, pres, first_token_variant=True, prune_batch_size=bs)
        outs.append((s_pr, s_ps, s_f))
    for s_pr, s_ps, s_f in outs[1:]:
        for n, _ in named:
            assert torch.allclose(s_pr[n], outs[0][0][n], rtol=1e-5, atol=1e-9)
            assert torch.allclose(s_f[n], outs[0][2][n], rtol=1e-5, atol=1e-9)
            assert torch.equal(s_ps[n], outs[0][1][n])


def test_merge_equals_full_run(tok, sets):
    prune, pres = sets
    m = _model(tok, torch.float64)
    kw = dict(protect_frac=0.01, keep_top=50, first_token_variant=True)
    full = PR.score(m, prune, pres, **kw)
    b = PR.score(m, prune, pres, layers=[1], **kw)
    a = PR.score(m, prune, pres, layers=[0], **kw)
    merged = PR.merge_results(b, a)
    assert merged.matrices == full.matrices
    _same_result(merged, full)
    _same_result(merged, full, "first")
    assert PR.select(merged, n=30) == PR.select(full, n=30)
    with pytest.raises(ValueError):
        PR.merge_results(a, a)
    c = PR.score(m, prune, pres, layers=[1], protect_frac=0.02, keep_top=50, first_token_variant=True)
    with pytest.raises(ValueError):
        PR.merge_results(a, c)


def test_shape_safety(tok, sets):
    m = _model(tok)
    prune, pres = sets
    res = PR.score(m, prune, pres, protect_frac=0.01, keep_top=50)
    sel = PR.select(res, n=10)
    assert all(tuple(s["shape"]) == res.shapes[s["matrix"]] for s in sel)
    rnd = PR.random_matched(m, sel, seed=3)
    assert all("shape" in s for s in rnd)
    before = {n: p.detach().clone() for n, p in m.state_dict().items()}
    bad = [dict(sel[0], shape=[1, 2])]
    with pytest.raises(ValueError, match="shape"):
        PR.apply_zeroing(m, bad)
    with pytest.raises(ValueError, match="shape"):
        PR.apply_zeroing(m, sel, expected_shapes={n: (3, 3) for n in res.shapes})
    with pytest.raises(ValueError):          # a later bad entry must not leave earlier weights zeroed
        PR.apply_zeroing(m, sel + [{"matrix": sel[0]["matrix"], "flat_index": 10**9}])
    for n, p in m.state_dict().items():
        assert torch.equal(p, before[n])
    saved = PR.apply_zeroing(m, sel, expected_shapes=res.shapes)
    PR.restore(m, saved)
    with pytest.raises(ValueError):
        PR.check_indices_match(m, {"shapes": {sel[0]["matrix"]: [5, 5]}})
    with pytest.raises(ValueError):
        PR.check_indices_match(m, {})
    with pytest.raises(ValueError):
        PR.check_indices_match(m, {"shapes": {"model.layers.9.mlp.up_proj.weight": [64, 32]}})


def test_parse_layer_spec():
    assert PR.parse_layer_spec("0-7,24-31") == list(range(8)) + list(range(24, 32))
    assert PR.parse_layer_spec("3, 5-6,3") == [3, 5, 6]
    for bad in ("", "1-", "a", "5-2", "1,,2", "-3"):
        with pytest.raises(ValueError):
            PR.parse_layer_spec(bad)


def test_gradient_checkpointing_same_scores_and_state_restored(tok, sets):
    prune, pres = sets
    kw = dict(protect_frac=0.01, keep_top=50, first_token_variant=True, prune_batch_size=2)
    ref = PR.score(_model(tok, torch.float64), prune, pres, **kw)
    m = _model(tok, torch.float64)
    m.gradient_checkpointing_enable()
    m.train()
    m.model.norm.weight.requires_grad_(False)
    flags = {n: p.requires_grad for n, p in m.named_parameters()}
    hooks_before = dict(m.get_input_embeddings()._forward_hooks)   # HF adds its own input-require-grads hook
    res = PR.score(m, prune, pres, **kw)
    _same_result(res, ref)
    _same_result(res, ref, "first")
    assert m.training
    assert {n: p.requires_grad for n, p in m.named_parameters()} == flags
    assert res.settings["gradient_checkpointing"] is True
    assert dict(m.get_input_embeddings()._forward_hooks) == hooks_before


def test_nonfinite_loss_fails_fast(tok, sets):
    prune, pres = sets
    m = _model(tok)
    bad_tok = prune[1]["target_ids"][0]
    with torch.no_grad():
        m.lm_head.weight[bad_tok] = float("nan")
    with pytest.raises(ValueError, match=r"prune\[1\]"):
        PR.score(m, prune, pres, protect_frac=0.01, keep_top=50)
    assert all(p.requires_grad for p in m.parameters())


def test_batches_longest_first_same_result(tok, sets):
    prune, pres = sets
    m = _model(tok, torch.float64)
    named = PR.eligible_matrices(m)
    a = PR._group_scores(m, named, prune, pres, prune_batch_size=2)[0]
    b = PR._group_scores(m, named, list(reversed(prune)), pres, prune_batch_size=2)[0]
    for n, _ in named:
        assert torch.allclose(a[n], b[n], rtol=1e-10, atol=1e-13)


# ------------------------------------------------------------------ margin loss (E2-B / E2-C, ours)

def _margin_examples(tok):
    q = lambda s: [tok.bos_token_id] + tok.encode(s, add_special_tokens=False)
    ids = lambda L: [tok.convert_tokens_to_ids(L), tok.convert_tokens_to_ids(" " + L)]
    return [
        {"loss": "margin", "prompt_ids": q("I believe the answer is B. What"), "base_prompt_ids": q("What"),
         "pos_ids": ids("B"), "neg_ids": ids("A")},
        {"loss": "margin", "prompt_ids": q("Which one? I think C"), "base_prompt_ids": q("Which one?"),
         "pos_ids": ids("C"), "neg_ids": ids("A")},
        {"loss": "margin", "prompt_ids": q("Please answer the question"), "pos_ids": ids("D"), "neg_ids": ids("E")},
    ]


def test_margin_loss_matches_full_model_logits():
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    exs = _margin_examples(tok)
    got = PR.batch_margin_loss(model, exs, pad_id=0)
    for k, ex in enumerate(exs):
        def m(prompt):
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([prompt])).logits[0, -1].float(), -1)
            return torch.logsumexp(lp[ex["pos_ids"]], 0) - torch.logsumexp(lp[ex["neg_ids"]], 0)
        want = -(m(ex["prompt_ids"]) - (m(ex["base_prompt_ids"]) if ex.get("base_prompt_ids") else 0.0))
        assert float(got[k].detach()) == pytest.approx(float(want), abs=1e-4)
    # batch invariance
    one = torch.stack([PR.batch_margin_loss(model, [e])[0] for e in exs])
    assert torch.allclose(got, one, atol=1e-4)


def test_margin_loss_ignores_a_shift_shared_by_all_letters():
    """A change that raises every letter logit equally (a pure 'answer with a letter' change) leaves the loss unchanged.
    Implemented by adding the same vector to every letter row of lm_head and a final hidden direction that is
    constant: we add c * mean(hidden) contribution via a bias-like shift, i.e. scale every letter row by the same
    additive vector v and check the margin difference is unchanged up to the term v . h, which is identical for all
    letters in one prompt and therefore cancels."""
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    ex = _margin_examples(tok)[0]
    before = float(PR.batch_margin_loss(model, [ex])[0].detach())
    letters = sorted(set(ex["pos_ids"] + ex["neg_ids"] + [tok.convert_tokens_to_ids(x) for x in ("C", "D", "E")]))
    v = torch.randn(model.lm_head.weight.shape[1]) * 0.5
    with torch.no_grad():
        model.lm_head.weight[letters] += v          # every letter logit moves by the same v . h
    after = float(PR.batch_margin_loss(model, [ex])[0].detach())
    assert after == pytest.approx(before, abs=1e-4)
    with torch.no_grad():
        model.lm_head.weight[ex["pos_ids"]] += v    # only the pushed letter moves -> the loss changes
    assert abs(float(PR.batch_margin_loss(model, [ex])[0].detach()) - before) > 1e-3


def test_score_with_margin_prune_set_and_mixed_preservation():
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    prune = _margin_examples(tok)[:2]
    pres = [{"prompt_ids": [tok.bos_token_id] + tok.encode("Hello", add_special_tokens=False),
             "target_ids": tok.encode(" A", add_special_tokens=False)}, _margin_examples(tok)[2]]
    res = PR.score(model, prune, pres, group_size=2, keep_top=50, prune_batch_size=2)
    sel = PR.select(res, n=20)
    assert len(sel) == 20 and all(s["s_prune"] > 0 or s["s_prune"] <= 0 for s in sel)
    with pytest.raises(ValueError):
        PR.score(model, prune + [pres[0]], pres, keep_top=50)
    with pytest.raises(ValueError):
        PR.score(model, prune, pres, keep_top=50, first_token_variant=True)


def test_margin_scores_match_finite_difference():
    """-w * d(loss)/dw from score() equals the loss change from scaling w (central difference), float64."""
    tok = tiny_tokenizer()
    model = _true_float64(tiny_model(tok).double())
    exs = _margin_examples(tok)[:2]
    named = PR.eligible_matrices(model)
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for _, p in named]
    for p in params:
        p.requires_grad_(True)
    s_prune, _, _ = PR._group_scores(model, named, exs, exs, prune_batch_size=2)
    name, w = named[3]
    for flat in (0, 7, 31):
        eps = 1e-4
        with torch.no_grad():
            orig = w.view(-1)[flat].item()
            w.view(-1)[flat] = orig * (1 + eps); up = PR.batch_margin_loss(model, exs).mean().item()
            w.view(-1)[flat] = orig * (1 - eps); dn = PR.batch_margin_loss(model, exs).mean().item()
            w.view(-1)[flat] = orig
        fd = -(up - dn) / (2 * eps)   # -w * dl/dw ~ -(l(w(1+e)) - l(w(1-e))) / (2e)
        assert s_prune[name].view(-1)[flat].item() == pytest.approx(fd, rel=1e-4, abs=1e-9)
