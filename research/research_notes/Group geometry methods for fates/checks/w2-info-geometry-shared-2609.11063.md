# Check: w2-info-geometry-shared-2609.11063

Paper: Picozzi, "The information geometry of large language models is shared, learned, and controllable", arXiv 2609.11063v1 (91 PDF pages: main text 1-41 incl. Extended Data, Supplementary 42-91).
Reading checked: `reading/w2-info-geometry-shared-2609.11063.json`. Page numbers below are PDF pages (form-feed index of paper.txt), the same convention the reading uses.

## VERDICT: NEEDS FIXES

The reading is accurate on numbers, quotes, figures and page references. It needs fixes because several interpretive statements go past what the paper shows, and one control is described in a way that misleads the Stage B / H2 design. One high-stakes item (E1), four medium, the rest low. No REDO needed; no figure or section is missing.

## 1. Quote result

- `quote_check.py`: 29 quotes, 0 failed.
- Stricter pass (my own script, scratchpad `pagecheck.py`): the script only tests the first 120 normalised characters. I tested the FULL text of all 29 quotes against the page the reading claims. All 29 are found in full, on the claimed page (key_results 16, pitfalls 11, limitations 1, future_work 1).
- Claim-versus-quote fit: 27 of 29 fit. Two claims stretch their quote (key_results[8] and [11], see E5, E6).

## 2. Figures and tables versus manifest

- Manifest lists Figure 1-5 and Table 1. All six are in the reading. I opened all six PNGs (p03, p07, p09, p10, p12, p14).
- Values read from figures match the reading: Fig 1a largest green about 19.5e-3 at d^2 about 1.75, grey about 16.5e-3; Fig 1b identity line; Fig 2a ranges (output 0.73-0.95, activation 0.38-0.88, all points above diagonal); Fig 2c, 2d; Fig 3c (most points above diagonal, so probability-only error is lower) and 3d (mixed cloud, claim holds for family medians); Fig 4a-c; Fig 5a (one high outlier), 5b and 5c (I upscaled 5c: every point is below the diagonal), 5d; Table 1 (all six rows).
- Extended Data 1-10, Supp Tables 1-4 and Supp Figs 1-2 were read from text only (the reading says so). Contents verified against captions and text: ED1-10 numbers all match (7.4e-13, 1.23, 0.910 vs 0.899, 0.080 to 0.013, 89%, 0.00493 vs 0.0814, 2.10 / 4.3x, 0.81-1.51, 10.57, 0.997 / 0.935 / 0.506, 1.57x, 11.79 / 11.08 / 6.62, 0.918 / 0.860 / 0.755, 55% / +0.069 / 0.59, 0.49 vs 0.33, 0.29 vs 1.02).
- One page slip: see E10.

## 3. Errors (reading says / paper says / page)

### E1 (HIGH, affects Stage B and H2): disturbance of the reusable update is measured in-sample, and the reading does not say so
- Reading says (controls_worth_copying): "Separate donor, reference and target prompts so a fix is tested on prompts it never saw." Also methods[2], bears_on_fates Stage B ("2.9-6.3x less reference-sequence KL") and actionable_fix ("Fisher averaged over a reference set ... to protect other behaviour").
- Paper says: the cost (sequence KL) is teacher-forced KL on continuations of the SAME four prompts that built the reference metric (Fig 5 caption p12; Methods p23). Supp Note 10.5 (p78) states it outright: "Reference prompts also define the metric" so the readout is disturbance on that reference set. Only the target EFFECT is held out (8 unseen prompts), and its criterion is a positive mean change; the smallest mean is 0.0128 nats (p78). The donor-only Fisher comparator is at least 2.1x worse (p78).
- Consequence: the paper gives no held-out disturbance number for the reusable update. Held-out neutral-prompt KL is used only for the per-prompt instruct-model frontiers (p22, p81), which are not reusable updates. The reading should say "preservation shown on the reference prompts themselves; held-out disturbance not measured", and should not list this setup as a held-out-prompt control.

### E2 (MEDIUM): "metric matters more than linearity" is not supported
- Reading says (bears_on_group_angle, point 3): relinearised paths cost 11.5-127x less path KL, "evidence that the metric matters more than linearity of a single step".
- Paper says: Fig 5b compares Fisher against Euclidean with BOTH relinearised after every accepted step (p12, p23, p77). No comparison of relinearised versus single-step under the same metric is reported. What is shown is that the Fisher advantage persists along relinearised paths (p11).
- Also, "manifold" in our sense (nonlinear steering along a learned manifold) is not tested here; relinearisation is piecewise-linear integration under a metric with a 0.02-nat trust region (p23). The paper only cites nonlinear steering (ref 14, Curveball steering, arXiv 2603.09313) on p2.

### E3 (MEDIUM): the warning about Euclidean principal angles is the reading's extension, but relevance_to_project attributes it to the paper
- Reading says (relevance_to_project): the paper "warns that Euclidean activation geometry, including Euclidean principal angles, is coordinate-dependent".
- Paper says: Proposition 1 (p49) and p1, p4 are about Euclidean distances, inner products, Gram matrices and cosine relational geometry; the reparameterisation test degrades CKA, mutual k-NN and cosine RSA only (ED1c p32; Supp Note 17 p83: about -10%, -15% by kappa 1e4, -28% by kappa 1e3, with the cosine recovery at 1e4 tied to near-singular maps). "Principal angle" appears only for read-out subspaces (Corollary 2 p49, control p50, chordal distance p3-4), never for internal activation subspaces.
- bears_on_group_angle point 1 ("hence Euclidean principal angles") marks it as inference, which is fine. relevance_to_project should say it is our inference. Also worth adding: Proposition 1 is an abstract factorisation argument; the paper does not discuss how norm layers restrict the gauge group in real transformers. A G-metric is also layer-specific (p17), so "principal angles in the G-metric" across layers needs a stated choice of layer for G; the reading leaves that out.

### E4 (MEDIUM): fate strengths for the flip-margin screen and the alignment factorisation are generous
- Flip margin (methods[4], bears_on_fates fate 1 "moderate"). Paper (Theorem 23, p72): if the argmax changes then rho >= m/2. This is a necessary condition only; it is an inequality with zero violations over 45,315 cross-seed flips, so the LAMBADA "0.519 to zero" shows where flips can occur, not independent validation. z = m/(2 rho) is a function of the two output distributions alone, so it cannot separate threshold crossing from override by itself. The reading's "flips with large m ... point to override or transformation" has no support in the paper. Also "z < 1" should be "z <= 1".
- Alignment factorisation (methods[5], fate 3 "moderate (method)", fate 4). Paper (Theorem 24, p73): the update is the preconditioned training-loss step A^-1 g; the empirical check is 80 held-out Pythia-70M examples (mean alignment^2 0.6811; correlation of log sensitivity with log squared rate 0.541). The reading's adaptation (observed pushback displacement as the update) is flagged as ours, good, but: any displacement that flips the answer has negative alignment with the truth log-odds by definition, so "strong negative alignment = override" does not discriminate override from bypass or readout failure; "near-zero alignment = non-mediation" only says no first-order change in truth log-odds. Also the no-pushback and pushback runs have different contexts (extra tokens), so G and the displacement are not defined on one shared context without a convention. Suggest "weak" for both.

### E5 (LOW-MEDIUM): key_results[8] wording
- Reading says: "Output flips (argmax changes) are forced by margin vs distribution distance". Paper (p72): a flip requires distance of at least half the margin; margin and distance do not force a flip. The quoted sentence links a smooth acquisition trajectory to an abrupt-looking benchmark transition; it concerns training-time acquisition, not pushback.

### E6 (LOW): key_results[11] drops a qualifier and context
- Reading says: depth delays commitment, "not the onset of movement along the answer margin (gradual drift vs commitment)". Paper (p8): not the earlier onset of UNSIGNED movement along the target-alternative margin direction. The result is training-time acquisition in randomised controlled synthetic languages (p8, p21, Supp Note 7), not inference-time behaviour. "Gradual drift vs commitment" is our gloss.

### E7 (LOW): key_results[10] "does not determine use"
- Paper (p73): the sentence concerns controllability versus learning rate (acquisition), under a training-loss covector. "Use" and "mediation" are our reading (see E4).

### E8 (LOW): bf16 pitfall is thinner than stated
- Reading says (methods[0], pitfalls[1]): bf16 directions "disagree badly"; "our 9B-27B runs in bf16 would be unreliable".
- Paper (p79): in two models (Pythia-410M and 1.4B), two contexts each, bf16 and float32 directions had absolute cosines 0.056-0.542, which the paper describes as weak to moderate alignment. G is severely ill-conditioned (kappa 1e5 to 1e7). So the direction of the warning is right, but it is four comparisons, and the 9B-27B statement is an extrapolation. Cite the numbers.

### E9 (LOW): compute statements that are inference, presented as fact
- Reading says: full GPU solves "fit only up to 2.8B on a 32 GB RTX 5000 Ada; 9B-27B needs 80 GB GPUs or the low-rank top-|S| Woodbury approximation".
- Paper (p25, p88): most GPU runs used one RTX 5000 Ada (32 GB) with additional A100 runs; it does not say which card carried the 2.8B solves. The 80 GB figure is ours. Woodbury (p22, p78) cuts the vocabulary-side cost but still needs |S| vector-Jacobian products through the network, so it is not shown to relieve network memory. The "at most 85 iterations" figure is an exact-arithmetic bound (p78), not an observed count.

### E10 (LOW): page slip and position slip
- other_figures_tables_read_from_text_only: Supp. Table 1 is listed on p54. The table and its caption are on p55 (p54 holds Remark 2).
- architecture_location: "residual stream, default at about 45% depth, final prompt position". The final prompt position is stated only for the reusable control (p23, p77). Elsewhere the paper says residual stream at a middle layer (p16) and does not state the position in what I read. Also, "top five eigenmodes carry 55% of eigenvalue mass and most answer-relevant probability motion": ED9 (p39) reports the largest answer-motion value (0.59) among the bands; by the definition on p18 this is the fraction of the band's probability motion on the highest-probability tokens, largest across bands, not "most of the answer motion". These modes are of the final-layer output Fisher H, so they describe output geometry, not a place in the architecture.

### E11 (LOW): fate 4 explanation
- Reading says (bears_on_fates fate 4): summed CAA moving sycophancy the wrong way shows "competing objectives interact through the metric". Paper (ED5b p36, p82): reports that summed CAA vectors move sycophancy the opposite way; gives no mechanism. Sycophancy and style are also not competing signals in the pushback sense.

## 4. Omissions

1. Size of the effects in the fix evidence. Reusable-update target effects are 0.013-0.21 nats (p78, Fig 5d); finite paths go to 0.4 nats (p23); matched-effect steering sweeps are small relative to flips under pushback. No answer-flip or accuracy outcome is reported for any steering result; the one discrete endpoint (20-item multiple choice) showed no Fisher-Euclidean difference (p81, already in pitfalls). The erosion predictor reverses sign beyond the moderate-intervention regime (p77). Together these limit the "Stage B fix" claim; the reading's attenuation pitfall (0.918 to 0.755) covers only part of it.
2. The anti-sycophancy objective is defined as preferring honest over flattering continuations (p81, already in the reading's setup) on Qwen2-1.5B-Instruct and Qwen1.5-1.8B-Chat with 12 prompts. It is not user pushback on a factual answer. key_results[5], actionable_fix and relevance_to_project do not repeat this caveat.
3. Scale precedent: p2 says FishBack (arXiv 2605.17231, ref 12) extends natural-gradient steering to 8B models, and ref 13 (Park et al., 2602.15293) is the other Fisher-steering paper. The reading lists these only in sections_read; reuse_verdict says "nothing at 9B-27B" without pointing to the 8B precedent. Worth a read before we reimplement.
4. Same-position limit of G: the paper's G is the Jacobian from layer l to the final state, computed for an intervention at one position. A probe direction at an earlier token could act through attention at later positions while having low G-sensitivity at the final position. The reading's bypass test ("near-zero Fisher sensitivity at the action position is output-inert by construction") should be limited to "same-position, first-order".
5. Numbers the reading could add: the paper's own check of alignment (0.6811, 0.541, p73), the same-tokenizer stability range already given (Supp Table 1), and the paper's statement that code and data are only on request (p26; already in code_available fields).
6. The reading says the Grassmannian tools cover subspace distance and speed, and that Procrustes and forking are not covered (correct). It could also say that curvature of subspace paths is not covered either.

## 5. Verified OK (no change needed)

- Title, author, date, page counts; setup.core_object, damping c about 1e-2 (0.03 for the reference Fisher); models and sizes (largest full GPU solve 2.8B, spectra only at 6.9B, Mistral-7B only in cross-model output geometry, nothing at 9B or above); data list; baselines; controls list (matched-random subspaces preserving principal angles, reparameterisation up to kappa 1e4, permutation-null CKA 0.316 versus 0.001, displacement-magnitude-matched null 0.636 and residual +0.063 about 9%, held-out neutral off-target KL, Fisher-orthogonal residual).
- All Fig 1-5 and Table 1 entries (what is plotted, takeaway, numbers, text_matches_figure verdicts, including "partly" for Fig 3d).
- key_results numbers: 0.88 / 0.62, 99 cells, 3,515 / 1,188, 8-74x and 30-190x, 0.997 vs 0.896, 11.79 / 11.08 / 6.62, 24 of 24 settings.
- Proposition 6 (speed squared equals 2 times the Frobenius norm of the normal part) and its reading as read-out subspace only; "no Procrustes, no path-forking" confirmed by search (no hits for Procrustes or fork anywhere in the text).
- limitations_quote, limitations_note (no dedicated limitations section; the p25, p81 and 6.3 items are correct), future_work_quote.
- Section coverage: the reading lists main text, Methods, all ten ED figures and Supp Notes 1-20; proofs are declared skimmed. I found no skipped section that carries a claim used in the reading.

## 6. Suggested fixes (for whoever edits the reading)

1. Add to controls_worth_copying, methods[2] and Stage B text: preservation is measured on the reference prompts used to build the metric (p12, p78); held-out target effect only, min 0.0128 nats.
2. Rewrite bears_on_group_angle point 3 to: "advantage persists along relinearised paths (both arms relinearised); no linear-vs-relinearised comparison; no manifold steering tested".
3. In relevance_to_project, mark the principal-angle point as our inference.
4. Downgrade fate 1 to "weak-moderate" and fates 3-4 alignment use to "weak"; state that z <= 1 is necessary, not sufficient.
5. Fix key_results[8] wording, key_results[11] ("unsigned", training-time, synthetic), key_results[10] ("learning", not "use").
6. Add numbers and n to the bf16 pitfall; mark the 80 GB and Woodbury remarks as our inference; mark 85 iterations as exact-arithmetic bound.
7. Fix Supp Table 1 page (55), "final prompt position" scope, and the ED9 "largest answer motion" wording.
8. Add the omissions in section 4 (items 1-4) to pitfalls or reuse_verdict.
