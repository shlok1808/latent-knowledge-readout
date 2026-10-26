# FishBack: Pullback Fisher Geometry for Optimal Activation Steering (Wang et al., arXiv 2605.17231v2)

**Verdict: adapt as a tool, not as evidence.** This is a steering paper about verb morphology. It has no layer-wise decodability probes (only a mean-difference concept probe on the final representation), no sycophancy and no fates. Its metric, though, gives us a cheap per-item test of whether a decodable direction is one the output actually reads.

## In one paragraph
Additive steering (CAA, ActAdd, ITI) treats the residual stream as flat. The paper argues that the right geometry at layer l is the softmax Fisher pulled back through the Jacobian, **G = JᵀHJ**. To second order, G measures how much a displacement at layer l moves the next-token distribution. The steering step that hits a target concept change with the least output movement is **δh ∝ G⁺q**, where q = Jᵀβ is the concept probe pulled back to layer l. On GPT-2 Small (exact) and on Llama-3-8B and Qwen3-8B (matrix-free, rank 50), it lowers off-target KL at matched concept change by 1.4-6.5x (GPT-2) and 1.8-3.6x (8B, early and middle layers only) against CAA, ActAdd, ITI, ReFT and RepSurgery. The gain disappears at about 75% depth on the 8B models.

## Setup
- **Task:** push the concept probability P_W (target forms over base+target forms) to 0.3, 0.5, 0.7 or 0.9 at one next-token step. Score the off-target KL on the renormalized non-concept tokens at exactly that level, found by bisection.
- **Concepts:** third-person singular (run→runs), progressive (run→running) and past tense (run→ran). Single-token pairs only (Park et al. 2026 protocol).
- **Data:** C4 validation positions where the top-3 next tokens are all concept forms with probability ≥ 0.7. They use 50 contexts per layer per concept (GPT-2, Llama) and 100 on Qwen. The same contexts are reused across layers.
- **Models and layers:** GPT-2 Small (d = 768) at L3, L6, L9, L11; Llama-3-8B at L8, L16, L24; Qwen3-8B at L6, L18, L27 (d = 4096). Base vs instruct is not stated.
- **8B approximation:** finite-difference JVP + top-1000-token Fisher + exact VJP, then a rank-50 randomized eigensolve (140 metric-vector products, about 280 partial forwards per case). The Jacobian is frozen at h0 across 30 steps, and a fixed concept component is mixed in (c_min = 0.5; 0.3 on GPT-2) with the part of the Fisher direction orthogonal to q̂ (Euclidean-orthogonal, eq. 54-55). The Euclidean ablation is exactly this with the orthogonal part removed (p15). No selection procedure or sensitivity is given for c_min.
- **Baselines:** CAA, ActAdd, ITI, ReFT-r1 and RepSurgery. Reference runs: a Euclidean ablation (G = I, all else equal) and a best-per-case oracle. Appendix K adds the untransported output metric (M = H).
- **Stats:** one-sided Wilcoxon on log-ratios with Benjamini-Hochberg within explicit families; cells with fewer than 6 pairs are marked underpowered.
- **Compute:** total for the whole comparison (all methods, not FishBack alone): 31.4, 10.8 and 30.8 GPU-hours (GPT-2, Llama, Qwen) on RTX PRO 6000 and H200; 48 GB minimum for 8B.

## Figures and tables (20 of 20 in the manifest covered)

| ID | Page | What it shows | Takeaway | Matches text? |
|---|---|---|---|---|
| Fig 1 | 2 | Schematic: h and probe β → J, H → G = JᵀHJ → q = Jᵀβ → δh ∝ G⁺q → h + δh | Pipeline only | Yes |
| Table 1 | 5 | Median [IQR] KL_method/KL_FishBack. GPT-2: CAA 1.42, ActAdd 6.52, ITI 2.01, ReFT 2.67, RepS 1.97, Euclid 1.30, oracle 1.25. Llama: 2.22/3.50/3.08/2.20/2.75, Euclid 1.79, oracle 1.64. Qwen: 1.77/2.93/3.61/1.93/2.11, Euclid 2.45, oracle 1.49 | FishBack wins every pooled median; many IQRs cross 1 | Yes (pooled rows are descriptive only) |
| Fig 2 | 6 | (a) G eigen-spectra for GPT-2 L3-L11; (b) effective rank as % of d: GPT-2 2.37/3.72/15.17/17.06%, Llama 0.10/0.14%, Qwen 0.08/0.11% | Very anisotropic. **8B bars are capped at 1.2% by the rank-50 truncation** | Partly: most of the "six orders of magnitude" drop is in the last tail (rank ~700 of 768), not "within the first few hundred" |
| Fig 3 | 6 | GPT-2 box plots of off-target KL per layer. Medians approx: L3 FishBack ~1.8 vs Euclid ~2.2; L11 ~0.14 vs ~1.2 (approx, read from figure) | Lowest median everywhere; heavy overlap | Yes. The marginal medians overstate the paired effect (~8x gap at L11 vs paired ~1.45x) |
| Fig 4 | 6 | Same for Llama L8/L16 and Qwen L6/L18. FishBack ~0.7-1.5 vs others ~1.3-7 (approx, read from figure) | Advantage persists at early and middle 8B layers | Yes |
| Fig 5 | 7 | GPT-2 Euclid/FishBack ratio per concept. Pooled ~1.10, 1.24, 1.27, 1.45 (L3→L11; approx, read from figure) | Metric-only gain is modest and grows with depth | Partly: past tense at L3 sits at about parity |
| Fig 6 | 7 | Same on 8B. Llama pooled ~1.7, ~2.0; Qwen ~2.2, ~2.7 (approx, read from figure) | Bigger gain at 8B | Partly: Llama L16 lower IQR is below 1 |
| Table 2 | 13 | GPT-2 exact: κ 4.9-9.5e7; r_eff 18.2, 28.6, 116.5, 131.0; Dev 0.978-0.996 | Far from any scaled identity | Yes, but the caption's "100 contexts per layer" does not match Table 6 (50 per concept, 150 per layer); unexplained |
| Table 3 | 13 | 8B within rank 50. Llama κ 1.2e4/4.0e3/3.9e3, r_eff 4.25/5.63/4.45; Qwen κ 1.2e5/1.8e4/2.9e3, r_eff 3.15/4.54/6.07 | κ flattens with depth; r_eff is single-digit everywhere | Partly: J.2 says r_eff "falls to single digits" at the deep layers, but it is single-digit at every layer and Qwen's rises |
| Table 4 | 14 | Synthetic model: cos to the exact minimizer 0.997-1.000; Euclid/Fisher 1.58-2.39; Spearman(κ, gain) 0.29-0.56; step/r_KL 82-1183 | Direction stays right far beyond the second-order radius | Yes |
| Table 5 | 15 | Hyperparameters: K 5000/1000, rank 50, oversampling 20, c_min 0.3/0.5, α = median eigenvalue, ε = 1e-4, T = 30, Jacobian frozen at 8B | Recipe | Yes |
| Table 6 | 16 | Contexts scanned/accepted/used, e.g. GPT-2 third 165,925/500/50 | Small eval sets | Yes |
| Table 7 | 17 | BH family sizes: GPT-2 288/72; 8B 216/45+9 | Multiplicity stated | Yes |
| Table 8 | 17 | Folded cells significant: GPT-2 10/16, 14/16, 16/18, 18/18; Llama 15/17, 10/18, **1/18**; Qwen 15/18, 18/18, **2/18** | GPT-2: rises with depth, complete at L11. 8B: collapse at ~75% depth (one sampled layer each) | Yes (paper.txt garbles the layout; the page image is clear) |
| Table 9 | 17 | Target-level significant: GPT-2 173/224 + 38/48; Llama 75/176 + 20/36; Qwen 115/180 + 24/36 | 43-79% significant (baselines 77/43/64%, ablation 79/56/67%); weaker at 8B, partly because the deepest 8B layers are included | Yes |
| Table 10 | 18 | Reachability. GPT-2 FishBack 100/100/99.6/**91.2** vs Euclid 100 everywhere; ActAdd 12.7% at L3; Qwen Euclid 22.7-36.3% vs FishBack 67.9-75.2% | Matched comparisons drop the cases weak baselines miss | Yes |
| Table 11 | 20 | Spearman(proxy, gain): Llama L24 0.42/0.50, Qwen L27 0.56/0.62; near 0 early; GPT-2 inconsistent (L11 align −0.41*) | Deep-layer gain is predictable from concept-spectrum alignment | Yes |
| Table 12 | 21 | Output metric upstream vs pullback (all-4 cohort): GPT-2 1.35-2.72*; Llama 0.98/1.16/1.33 (n = 18/5/5); Qwen 1.41-3.56* | The Jacobian transport matters | Partly: K.4 says "every cell", but Llama L8 = 0.98 |
| Table 14 | 21 | cos(q, δ_H) ≈ 0.95 on 8B, 0.32-0.84 on GPT-2; cos(q, δ_G) 0.13-0.87 | The output metric barely rotates q at 8B | Yes |
| Table 13 | 21 | Single-target cohort: Llama 1.11/2.60/2.98 (cells 2, 1, 0 of 12); Qwen 1.52/1.91/2.45 (4, 6, 5 of 12) | Directionally consistent | Yes, but K.2 mixes cohorts in its depth trend |

Page images: `../papers/w2-fishback-2605.17231/pages/` (p02, p05, p06, p07, p13-p18, p20, p21).

## Key results (with quotes)
- "On GPT-2 Small, this non-Euclidean geometry is borne out empirically, and our method lowers off-target KL divergence by median factors of 1.4–6.5x against individual steering baselines." (p1)
- "On Llama-3-8B and Qwen3-8B, it lowers off-target KL by median factors of 1.8–3.6x against individual baselines at the early and middle layers." (p1)
- Metric-only ablation: "On GPT-2 Small the ratio sits above parity at every layer, with a pooled median of 1.30 and an 80% win rate" (p7), and "on the 8B models it widens to 1.79 (Llama-3-8B) and 2.45 (Qwen3-8B) at the early and middle layers." (p7)
- Oracle: "the median ratio is 1.25 with a 64% win rate—a thinner margin" (p6)
- Deep layers: "the Euclidean gap of Table 1 is no longer significant" (p7); "one cell of eighteen survives on Llama-3-8B and two of eighteen on Qwen3-8B" (p17)
- "Cosine similarity to the optimal direction is therefore the wrong diagnostic; the relevant quantity is the G-weighted spectral error of (60)." (p18)
- "The geometry that Park et al. (2026) identify is the right geometry, but it is the right geometry for the final representation." (p21)
- Cost: "140 metric-vector products or roughly 280 partial forward-pass equivalents" (p15)
- GPT-2 depth trend is the opposite of 8B: "On GPT-2 Small the proportion rises with depth and is complete at the deepest layer." (p17); its G does not flatten: "varying by less than a factor of two across depth, so the flattening that drives the pattern above does not occur here" (p20)
- Qwen reachability: FishBack 67.9-75.2% vs Euclidean ablation 22.7-36.3% (p18). The authors read this as the geometric component contributing to progress, not a separate advantage of the method.
- Untransported output metric upstream: reaches "only 22–37% on Llama-3-8B and 13–20% on Qwen3-8B, against 98–100% for the pullback" (p21); no better than flat on 8B, and worse than flat on GPT-2 ("the Euclidean ablation achieving lower off-target divergence at three of four layers", p21). The transport gain grows toward the output layer, and they "offer no mechanism for it" (p21).

## Methods we could use
1. **G = JᵀHJ and its top-k eigenspace (the "output-sensitive subspace").** This gives a bypass test per item: does the correct-answer probe direction have mass in G's top eigenspace at the answer position after pushback? Compare flipped vs resisted items, matched on prior confidence. It is local, second order and about the whole next-token distribution. It measures causal controllability, not endogenous mediation. This use is our extension.
2. **q = Jᵀβ with β = w_correct − w_user (unembedding difference).** This is the residual direction the output reads for the answer. Compare it with the probe direction in the G-metric, not by Euclidean cosine. For override, compute both covectors and project the pushback-induced Δh onto each.
3. **Alignment proxies qᵀGq and ‖H^½Jq‖/‖Jq‖.** Cheap per-item scalars. Use them together with the logit-margin control. In the paper they predict where the metric correction helps over flat steering (log-ratio of off-target KL, Euclidean vs FishBack, deep 8B layers only), not injection success. Predicting which items an injection can rescue is our extension and untested.
4. **Fisher-Pythagorean identity and proxy cost ratio.** These tell us when two directions are functionally the same (they differ only along flat directions). This matters for transformed vs bypass.
5. **Layer recursion G(l) = A_lᵀG(l+1)A_l.** This is what we would track for the group angle: the path of the top-k eigenspace across layers and tokens on the Grassmannian, its principal angles to the probe subspace, and where flipped and resisted items fork. The paper does none of this.
6. **FishBack steering as a Stage B fix** for not-routed and threshold items at early or middle layers. Compare against CAA and plain DPO at matched flip-back rate, scoring off-target KL. It is a per-item optimiser (~280 partial forwards + 30 steps + bisection per case, one next-token step, no generation test): a research tool or oracle upper bound, not a deployable CAA replacement.

## Controls worth copying
Matched-target bisection; metric-only ablation (same covector, calibration and concept efficiency); per-method step calibration; separate reachability reporting; Wilcoxon on log-ratios + BH; underpowered flag (fewer than 6 pairs); best-per-case oracle; the three-rung metric ladder (I / H / JᵀHJ); a synthetic exact model; naming the predictor in advance.

## Pitfalls
- **The 8B "concentration" is partly an artifact:** "computed within the rank-50 subspace retained by the approximation" (p6). Effective rank is capped at 50 of 4096.
- Matched comparisons select: "the ratios reported for it describe the cases where it did reach the target, which are the cases where it was competitive." (p17)
- Pooled rows are not independent: "a context contributes one case per layer" (p16).
- Raw-ratio Wilcoxon gave false positives in their own first pass (p17).
- The evaluated direction is not pure G⁺q, because a fixed concept component is forced in (p15).
- Text-table mismatches: effective rank "falls to single digits" (p20) vs Table 3; "every cell measured" (p21) vs Llama L8 = 0.98; Figure 2's "within the first few hundred eigendirections" (p6).
- A low-G direction is not erasure: "exact nullity therefore does not accumulate merely because normalization layers are traversed." (p12)
- The separability condition behind Theorem 3 holds exactly only when all pairs share the same within-pair log-odds (p4). Single-token concepts only (p16).
- **The 8B approximation's fidelity is asserted, not measured.** "Each component is validated in Appendix E." (p5), but E.6 gives one-line checks without numbers (e.g. drift "measured to be smaller than the noise of re-estimation", p15) and no exact-vs-approximate comparison at 8B. Eigenvector or subspace accuracy is never reported. Our principal-angle tracking at rank 50 (effective rank 3-6) needs a repeat-seed stability check first.
- **Caveats for our bypass test (our extension):** H depends on the predictive distribution, so G's scale and rank depend on entropy; the paper selects mid-entropy contexts ("a cumulative probability of at least 0.7", p16), while flipped and resisted items will differ in entropy. Match on entropy or use scale-free statistics. "Little mass in G's top eigenspace" needs a null (random direction ≈ k/d, ~0.1% at 8B; also compare to q). Whether f holds other positions fixed is not stated; if so, G covers only the same-position path.
- **Table 11 selection:** Qwen n = 106/67/68 of 300 tracks the Euclidean ablation's reach rate, so deep-layer correlations sit on the subset where flat steering succeeded; GPT-2 n = 13-28 of 150, where "a correlation of 0.3 is not distinguishable from zero" (p20). 4 of 9 layer-level tests are non-significant.
- The "don't apply the output metric upstream" result is significant on GPT-2 and Qwen only; on Llama it is "directionally consistent but does not reach significance in either cohort" (p21).

## Limitations and future work
There is no limitations section. The closest statement: "The fourth approximation also marks the boundary of the method. At roughly 75% relative depth the conditioning of G has flattened considerably" (p15). Reachability is "not uniformly in our favor" (p17). The GPT-2 L11 negative correlation is left "unexplained" (p20). **No future work is stated** anywhere (Conclusion and appendices checked); future_work_quote = "none stated".

## Bears on our fates
| Fate | How | Strength |
|---|---|---|
| 1 readout/threshold | qᵀGq and q tell how strongly the output reads the correct direction; FishBack can push the margin | weak (tool) |
| 2 bypass | Probe direction outside G's top eigenspace = decodable but locally inert. This is the main use | moderate (method), no evidence here |
| 3 not routed | FishBack injection works at early and middle layers (98-100% reach on GPT-2 and Llama), which covers the "injection works" half | weak-moderate |
| 4 override | Compare two covectors under the same G | weak |
| 5 transformed | G-weighted distances, Procrustes after G^½ whitening | weak |
| 6 erased | Caution only: low G ≠ erased | weak |

## Group angle
It gives a model-derived per-layer, per-token metric and a candidate subspace to track (the top-k eigenvectors of G). It does **not** track subspaces, principal angles, Grassmannian paths or curvature, and it does not compare item classes. On manifold vs linear steering it is a third option: a linear step under a local metric, re-solved each step exactly only on GPT-2 (at 8B the Jacobian is frozen at h0; only the Fisher and covector update). It contrasts itself with data-fitted manifold steering (Wurgaft et al. 2026 = 2605.05115, our c5; Raval et al. 2026), which obtains the geometry "from data rather than the model" (p2), but it runs no manifold baseline (all five baselines are linear), so it gives no evidence on model-derived metric vs data-fitted manifold. It never measures eigenspace stability.

## Where in the architecture, and what fix
Residual stream at the prediction position; the depth profile is model-dependent. **8B:** gain at L8-L16 (Llama) / L6-L18 (Qwen), collapse at L24 / L27 (1/18, 2/18 cells; about 75% depth; one sampled layer each). **GPT-2:** gain rises with depth, largest at L11 (about 92% depth), no flattening (κ ≈ 1e7 at every layer). The paper explains the 8B collapse by flatter conditioning, but on Llama κ does not separate L16 (4.0e3) from L24 (3.9e3), nor does effective rank (5.63 vs 4.45). Nothing tested at 9B-27B. **Fix:** pullback-Fisher injection of the correct answer at early or middle layers, as a fate-matched Stage B candidate. **Monitor:** whether the correct-answer probe signal sits inside G's top eigenspace (with an entropy match and a random-direction null).

## Relevance (honest)
Medium. The paper says nothing about knowledge-action divergence. What it gives us is a cheap, principled bypass diagnostic, the right inner product for comparing probe and causal directions, and a lower-collateral injection. Its 8B geometry numbers should not be quoted as model facts, because they are bounded by the rank-50 truncation. If sycophantic flips are decided late in the network, the correction may buy little there; we have not tested this.
