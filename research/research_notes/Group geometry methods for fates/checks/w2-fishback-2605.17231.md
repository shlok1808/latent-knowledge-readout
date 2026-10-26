# Check: w2-fishback-2605.17231 (FishBack, arXiv 2605.17231v2)

Verdict: **NEEDS FIXES** (no REDO). Every number, quote and figure/table entry in the reading is faithful to the paper. The fixes are to five wording errors and to omissions that touch the high-stakes fields (`architecture_location`, `actionable_fix`, group-angle summary). Edits are listed in section 5.

## 1. Quote check
- `python3 quote_check.py w2-fishback-2605.17231`: 35 quotes, 0 failed.
- That script tests only the first 120 normalised characters and not page numbers. I re-checked with per-page `pdftotext` of `paper.pdf`: all 35 full quotes are present and all 35 sit on the page the reading gives (including `limitations_quote` p15).

## 2. Coverage checks
- Manifest lists 20 figures/tables (Fig 1-6, Tables 1-14). The reading has all 20, with correct pages. `grep` of `paper.txt` finds no other float. Nothing missing.
- `sections_read` covers the main text and Appendices A-K, including the theorem/remark numbering. Nothing skipped.
- Images opened: p02 (Fig 1), p06 (Figs 2-4; re-rendered at 300 dpi and cropped), p07 (Figs 5-6, same), p17 (Tables 7-9), p18 (Table 10), p20 (Table 11), p21 (Tables 12-14).
- Recomputed or compared against text/images:
  - Table 1: all 21 median/IQR/win%/n entries match.
  - Tables 2-14: all numbers match, including Table 4 column order (tiny 1183, small 82, medium 288) and Table 8 (the page image resolves the Table 7/8 interleave in `paper.txt`; the reading is right).
  - Fig 3, 4, 5, 6 read-offs agree with the crops within reading error.
  - Fig 2(a): the ~6-order drop does happen only in the last tail (rank ~700 of 768), so the reading's "partly" is correct. The text claims it happens "within the first few hundred eigendirections" (p6).
  - Fig 2(b) bar labels match Tables 2-3.
  - Arithmetic: Llama kappa 1.2e4/3.9e3 = 3.1x, Qwen 1.2e5/2.9e3 = 41x (the paper says 3x and 40x). Cases 600+450+900 = 1,950 ("nearly two thousand").
- Real mismatches the reading flags are real: Fig 2 text vs plot, Table 3 vs J.2 "falls to single digits" (p20), "every cell measured" vs Llama L8 = 0.98 (K.4, p21), the K.2 cohort mixing, the 8B rank-50 cap, and the unexplained GPT-2 L11 negative correlation.
- Fate names match the project list. Strengths are hedged (weak / weak-moderate / moderate "as a method; no evidence in this paper"). Vocabulary is "decodable". Causal controllability is kept apart from endogenous mediation (methods[0], md). Group-angle claim "does not track subspaces, principal angles, Grassmannian paths, speed or curvature, or compare item classes" is right: `grep` finds no "principal angle", "Grassmann", "geodesic", "eigenvector" or "eigenspace" in the paper, and "curvature" only in the related-work/Gauss-Newton sense.

## 3. Errors (what the reading says / what the paper says)
1. **`architecture_location` ignores GPT-2's opposite depth trend (moderate; high-stakes field).**
   - Reading: "The gain is strongest at early and middle layers (GPT-2 L3-L11 throughout; Llama-3-8B L8/L16; Qwen3-8B L6/L18)"; md: "early and middle layers (about 25-50% depth)". The reading's own Fig 3 and Fig 5 entries say the GPT-2 gain grows with depth (pooled ~1.10 at L3 to ~1.45 at L11), so the sentence contradicts them.
   - Paper: p17 (G.2) "On GPT-2 Small the proportion rises with depth and is complete at the deepest layer" (18/18 at L11, Table 8). p20 (J.3) "Its condition numbers remain of order 10^7 at every layer, varying by less than a factor of two across depth, so the flattening that drives the pattern above does not occur here". Table 2 (p13): kappa 9.5e7 to 4.9e7. GPT-2 L11 is about 92% depth, and nothing collapses there.
   - So "collapse at about 75% depth" is an 8B finding only, and each 8B model has just one sampled layer there (L24, L27; none between 16 and 24 or beyond 27). Suggested wording: "8B: gain at L8-L16 / L6-L18, collapse at L24 / L27 (1/18, 2/18 cells; about 75% depth; one sampled layer each). GPT-2: gain rises with depth, largest at L11, no flattening. Depth profile is model-dependent; nothing tested at 9B-27B."
2. **Conditioning drop given as the explanation without the Llama counter-evidence (moderate).**
   - Reading (`architecture_location`): collapses "where G's conditioning flattens (condition number down ~3x on Llama and ~40x on Qwen vs early layers)".
   - Paper: the numbers are quoted correctly (p7). But Table 3 (p13) gives Llama kappa 1.2e4 (L8), 4.0e3 (L16), **3.9e3 (L24)**. L16 and L24 have the same kappa, yet significant cells are 10/18 vs 1/18 (Table 8, p17). Effective rank does not separate them either (4.25, 5.63, 4.45). On Qwen the drop L18 to L27 is about 6x (1.8e4 to 2.9e3). The paper's own Remark 9 (App. B) says the conditioning bounds do not say where correction helps.
   - Suggest: "paper's account; on Llama kappa does not separate L16 from L24."
3. **"G-orthogonal" should be "orthogonal" (minor; matters for re-implementation).**
   - Reading (`setup.approximation_8B`): the applied direction mixes c_min with "the normalized G-orthogonal part of the Fisher direction".
   - Paper p15 (E.3, eq. 54-55): delta_perp = delta_F - (delta_F . q_hat) q_hat, "the part of the Fisher direction orthogonal to the concept". That is Euclidean-orthogonal to q_hat. "G-orthogonal" appears in the paper only in I.1 (error orthogonal to the optimum).
4. **"No probes" contradicts the paper and the reading's own setup (minor, but it is in the headline line).**
   - Reading (md verdict line; `relevance_to_project`): "It has no probes, no sycophancy and no fates" / "no probes, no factual or sycophancy task, no fates."
   - Paper: Fig 1 (p2) has a "Concept Probe"; p3 (Sec 3) "is read by a mean-difference probe beta_W on the final representation"; the covector is q = J^T beta_W. The reading's `core_objects` says so. What the paper lacks is intermediate-layer decodability probes. Suggest "no layer-wise decodability probes".
5. **Qwen reachability overstated (minor).**
   - Reading (`key_results`): "On Qwen3-8B, the geometric component is needed to reach targets at all."
   - Paper p18 (Table 10 text): FishBack reaches 67.9-75.2%, the Euclidean ablation 22.7-36.3%, fixed baselines about 41-58%. The authors' own conclusion: "evidence that the geometric component contributes to progress on this model, not a separate advantage of the method." Suggest "contributes to reaching more targets (68-75% vs 23-36%)".
6. **Proxies do not predict injection success (minor; mislabelled use).**
   - Reading (`methods[2].causal_followup_it_allows`): "Predict which items an injection can rescue (the paper shows these proxies predict the steering gain...)".
   - Paper p20 (J.1): the realized gain is "the log-ratio of off-target divergences between the Euclidean ablation and FishBack". It is a relative off-target-KL gain of the metric correction, not reachability or success. Reword as "predict where the metric correction helps over flat steering; rescue prediction is our extension and untested".
7. **"Argues against data-fitted manifolds" is too strong, and the manifold-vs-linear question is untested (minor-moderate; this wave's focus).**
   - Reading (`bears_on_group_angle`): "It argues against data-fitted manifolds."
   - Paper p2: Wurgaft et al. and Raval et al. are "confirming curvature but obtaining the geometry from data rather than the model, with no optimality guarantee". The five baselines are all linear (CAA, ActAdd, ITI, ReFT-r1, RepSurgery). No manifold method is run, so the paper gives no evidence on model-derived local metric vs data-fitted manifold. State that explicitly.
   - Also "re-solved along the trajectory": exact on GPT-2 only. At 8B the Jacobian is frozen at h0 and only the Fisher and covector are updated (E.4, p15).
8. **Table 9 takeaway (nit).**
   - Reading: "Roughly 43-77% of target-level cells are significant".
   - Paper p17: baselines 77/43/64%, but ablation rows are 79/56/67%, so the span is 43-79%. The caption says Table 9 includes the deepest 8B layers, so part of "8B weaker than GPT-2" is the L24/L27 collapse.
9. **Table 2 n (nit).** Reading marks Table 2 "yes" and records "medians over 100 contexts per layer" (p13). Table 6 (p16) says 50 per concept per layer, 150 per layer evaluated. The paper does not explain the 100. Note it.
10. **Compute attribution (nit).** Reading (`methods[5]`) attaches "10.8 GPU-h for 450 Llama cases, 30.8 for 900 Qwen" to FishBack steering. E.7 (p16) says "Total compute", which probably covers all seven methods and the ablation. Say "total for the whole comparison".

## 4. Important omissions
1. **The 8B approximation's fidelity is asserted, not measured; eigenspace stability is never examined (high; core for the group angle).**
   - Paper p5: "Each component is validated in Appendix E"; p2: "A faithful approximation ... recovers the same optimal direction". Appendix E gives no exact-vs-approximate comparison at any scale. Its checks are one-line assertions without numbers: "reproducible to machine precision", "stable under changes of K", drift "measured to be smaller than the noise of re-estimation" (p15). Table 4 is a synthetic exact model, and Theorem 16 is a bound with unmeasured eta and xi. No exact G exists at 8B.
   - The paper reports eigenvalues and the direction G^-1 q only. It never reports eigenvector or subspace accuracy (the words "eigenvector" and "eigenspace" do not occur). Our plan (principal angles between top-k eigenspaces across layers, flipped vs resisted) rests on exactly that. Add as a pitfall: eigenspace stability at rank 50 and effective rank 3-6 is untested, so fork-layer claims need a repeat-seed check on our side.
2. **Ablation structure, c_min and what "metric-only" means (medium-low).**
   - Paper p15 (E.3): the Euclidean ablation "is exactly (55) with this second component removed", so FishBack = the same c_min * q_hat term plus a G-informed orthogonal correction. The gain is therefore the value of that correction. c_min differs by model (0.3 GPT-2, 0.5 8B) with no stated selection procedure or sensitivity. The reading's pitfall 6 notes the forced component but not these two points.
3. **Project-facing caveats for the proposed bypass test (medium; our extension, not a paper error).**
   - H = Cov_p(w) is taken under the current predictive distribution, so G's scale and rank depend on predictive entropy. The paper deliberately selects mid-entropy contexts (top-3 tokens all concept forms, mass >= 0.7; p16). Flipped and resisted items after pushback will differ in entropy by construction, so matching on "prior confidence" does not cover this. Use predictive-entropy matching or scale-free statistics.
   - "Little mass in G's top eigenspace" needs a null. A random direction has about k/d (about 0.1% at 8B), so compare to random directions and to q itself.
   - The paper does not say whether f (layer l to final) holds other positions fixed (`lambda = f(h)`, p3). If so, G covers only the same-position path, and pushback evidence at earlier tokens is outside it. Paper silent; check before use.
4. **`actionable_fix` is a per-item optimiser, not a cheap patch (medium-low).** About 140 products (about 280 partial forwards) plus 30 steps and bisection per case (p15), one next-token step, no generation test. Say it is a research tool or oracle upper bound for Stage B, not a deployable replacement for CAA. Also "do not apply the output-layer Fisher upstream" is significant on GPT-2 and Qwen only; on Llama it is directionally consistent and not significant (K.4, p21).
5. **Appendix K details left out (low-medium).**
   - The untransported metric reaches only 22-37% (Llama) and 13-20% (Qwen) of targets vs 98-100% for the pullback (p21).
   - It is indistinguishable from flat steering on both 8B models and worse than flat on GPT-2 (Euclidean lower at 3 of 4 layers; p21).
   - The ratio grows toward the output layer and "we ... offer no mechanism for it" (p21).
6. **Table 11 selection (low-medium).**
   - Qwen n = 106/67/68 of 300 contexts tracks the Euclidean ablation's reach rate (36.3/22.7/23.3% x 300 = 109/68/70). The deep-layer correlations (0.56, 0.62) are computed on the subset where the flat baseline succeeded.
   - GPT-2 n = 13-28 of 150 is unexplained attrition; J.3 itself says r = 0.3 is not distinguishable from zero there.
   - The "named in advance" predictor (p5) is the authors' claim, and 4 of 9 layer-level tests are non-significant.

## 5. Verdict and edits
**NEEDS FIXES.** The reading's numbers and quotes can be trusted. Edits, in priority order:
1. Rewrite `architecture_location` and the md "Where in the architecture" (errors 1 and 2).
2. Add pitfall for omission 1 (fidelity unmeasured; eigenspace stability untested) and for omission 3 (entropy dependence, null baseline, same-position Jacobian).
3. Fix wording: errors 3 ("orthogonal"), 4 ("no layer-wise probes"), 5, 6, 7 (including "no manifold baseline run").
4. Add a line to `actionable_fix` for omission 4, and the K details of omission 5.
5. Optional: errors 8-10 and omissions 2 and 6.
