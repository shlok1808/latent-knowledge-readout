# The information geometry of large language models is shared, learned, and controllable

Dario Picozzi (UCL), arXiv 2609.11063v1, 10 Sep 2026. 91 PDF pages: 41 pages of main text and Extended Data, then 50 pages of Supplementary Information (Supp. page k = PDF page 41+k).

**One-line summary:** Measure activation changes by how much they change the output distribution (the pullback Fisher metric). This metric predicts the cost of an intervention before you make it, and steering with it changes the target behaviour with far less collateral change than Euclidean steering.

**Most useful thing for us:** a per-direction output-sensitivity test, v^T G v. A probe direction that decodes the right answer but scores about zero here is output-inert to first order at the same position, which is the bypass signature. The paper also gives a cheap output-only screen for threshold flips, and an anti-sycophancy steer we could use as a Stage B arm next to DPO.

---

## What I read
Everything in the main text:
- abstract, intro, the six Results sections, Table 1, Discussion, all Methods subsections, and the availability statements
- references, skimmed
- Extended Data Figs. 1-10 and the Extended Data note on cyclic concepts

All 20 Supplementary Notes. For the proof-heavy ones (Notes 2-4, 6 and 8) I read the statements and remarks and skimmed the proofs. The full ordered list is in the JSON (`sections_read`).

## Setup
- **Core object.** H is the covariance of unembedding rows under the next-token distribution p (the output Fisher). Pulled back to layer l it becomes G = J^T H J. For a small change, KL is about one half of δh^T G δh.
- **The steer.** δh ∝ (G + αI)^-1 q, where q is the gradient of a contrastive log-odds objective and α = 0.01·λmax.
- **Models.** Pythia 70M-6.9B, GPT-2 family, GPT-Neo, Qwen2 0.5B/1.5B, Qwen2-1.5B-Instruct, Qwen1.5-1.8B-Chat, BLOOM, Mamba, RWKV, Mistral-7B and OLMo-2-1B, plus small synthetic models.
  - Full GPU pullback solves only go up to 2.8B. At 6.9B there are only spectra.
  - **Nothing at 9B-27B.**
- **Layers.** Residual stream at a middle layer, round(0.45 × depth); the final prompt position is stated for the reusable control. Sweeps at 1/4, 1/2 and 3/4 depth, and Pythia-410M layers 4/8/12/20. The SAE work uses Pythia-410M layer 12.
- **Objectives.**
  - capital-city true/false preference
  - sycophancy, defined as honest-over-flattering continuations (not factual pushback)
  - TruthfulQA, style, sentiment, CounterFact edits
- **Baselines.** Euclidean steering, CAA, donor-only Fisher, attribution patching, activation magnitude, AdamW+KL, K-FAC. ROME/MEMIT appear only as descriptive context.

## Figures (6 in the manifest; I looked at every page image)

| Fig | Page | What is plotted | Takeaway | Numbers |
|---|---|---|---|---|
| 1 | 3 | **a.** Held-out profiled loss vs squared chordal subspace distance d² (Pythia 70M-1.4B); real read-out deviation (green) vs random directions with matched principal angles (grey). **b.** Measured vs predicted KL for 99 small perturbations. | **a.** Behaviour pins down the read-out subspace, and real deviations cost more than random ones with the same principal angles. **b.** One half δhᵀGδh predicts real KL with no fitted scale. | **a.** Green peaks at about 19.5e-3 and grey at about 16.5e-3, both at d² ≈ 1.75 (approx, read from figure). R² ≥ 0.9999; slope ratios 1.14-1.27. **b.** Median ratio 1.000, range 0.948-1.224. |
| 2 | 7 | **a.** Output Fisher-Rao agreement vs activation agreement for 29 dyads. **b.** Error of the six-scalar prediction across training. **c.** Model-human semantic distance vs size. **d.** Predicted vs measured human alignment. | Output geometry is shared across architectures far more than activation geometry. Bigger and better-trained models sit closer to human completions. | **a.** Every point is above the diagonal; output 0.73-0.95 vs activation 0.38-0.88 (approx). Means are 0.88 vs 0.62. **b.** Error falls from about 0.25 to about 0.01 after step 256. **d.** r = 0.758. |
| 3 | 9 | **a.** Predicted vs measured Fisher spectral exponent. **b.** Predicted vs measured effective dimension. **c.** Effective-dimension error: weighted vs probability-only profile. **d.** Spectral-exponent error: weighted vs probability-only. | Token statistics predict the spectrum with no fitting. Probability concentration predicts how many modes there are; read-out structure predicts how fast they decay. | **a.** Pearson 0.968. **b.** RMSE 0.027-0.052. **c.** Most points are above the diagonal. **d.** The point cloud is mixed; the claim rests on family medians (partial match). |
| 4 | 10 | **a.** n-gram-predicted vs measured fact margin. **b, c.** Cumulative facts acquired vs step and vs held-out loss (Pythia 70M/160M/410M). | Corpus statistics forecast when a fact is acquired. Curves from different sizes line up by loss, not by step. | **a.** R² 0.775-0.792. **b.** About 25 facts at step 0, a jump to about 80 by step 32, then about 175-200 by 128k (approx). |
| 5 | 12 | **a.** Predicted vs measured cost ratio, Euclidean vs natural gradient. **b.** Path KL on relinearised paths. **c.** Reference-metric vs comparator sequence KL. **d.** Target change vs sequence KL on unseen prompts. | The cost advantage is predicted before intervening. It holds along curved paths, and a single shared truth or anti-sycophancy update transfers to unseen prompts. | **a.** Median 0.967, slope 0.982. **b.** All points are below the diagonal; ratios 11.5-127×. **c.** Ratios 2.9-6.3×. **d.** All 24 changes are positive (0.013-0.21 nats). Sequence KL in c and d is on continuations of the same reference prompts that build the metric. |
| Table 1 | 14 | One correction applied across six operations | The same Fisher substitution improves steering, editing, feature importance, attribution, SAE training and LoRA | Steering: up to 72.7× lower off-target KL. Editing: 10.57×. Feature importance: ρ 0.997 vs 0.896. Attribution: 0.935 vs 0.506. SAE training: matched cosine 0.167→0.480. LoRA: 1.57× better AUC. |

**Also read, as text only (no page images):**
- **Extended Data Fig. 8 (layer dependence):**
  - off-target advantage: 11.79× at 1/4 depth, 11.08× at 1/2, 6.62× at 3/4
  - measured-to-predicted ratio falls (0.918, 0.86, 0.755) as intervention size grows
- **Extended Data Fig. 10 (cyclic concepts vs relations):**
  - cyclic concepts are rotations, with isometry defect 0.29
  - relations share a displacement direction: 0.49 vs 0.33 for the permuted control
- **Extended Data Figs. 1-7 and 9, Supp. Tables 1-4, Supp. Figs. 1-2:** summarised in the JSON.

## Key results (exact quotes in the JSON)
1. Activation coordinates are arbitrary. Any invertible linear change gives the same model, so Euclidean activation geometry (distances, inner products, cosines) is not fixed by behaviour (p. 1). That this extends to Euclidean principal angles between internal subspaces is our inference.
2. Output geometries agree across 10 models at 0.88, against 0.62 for mid-layer activations (p. 2).
3. One half δhᵀGδh predicts the real KL change, with a median ratio of 1.000 (p. 4).
4. A measured anisotropy ratio predicts, with no fitting, how much better natural-gradient steering will be than Euclidean steering (p. 2, Fig. 5a).
5. **Anti-sycophancy and truth (instruct models up to 1.8B).** At equal behaviour change, off-target divergence is 8-74× lower for anti-sycophancy and 30-190× lower for capital-city truth (p. 11). These are per-prompt frontiers on 12 prompts; anti-sycophancy means honest-over-flattering continuations, not user pushback on a factual answer. A shared update learned on 4 donor prompts (base models) transfers a positive mean change to unseen prompts in all 24 settings, but the effects are small (smallest mean 0.0128 nats) and no answer-flip or accuracy outcome is reported.
6. **Where to intervene.** The advantage is largest at early-to-mid layers and shrinks toward the read-out (p. 13).
7. **Flip-margin theorem (Supp. Note 9).** An argmax flip is impossible unless the root-probability distance is at least half the margin. This is a necessary condition, not a cause. On LAMBADA the flip rate falls from 0.519 to 0 as z = m/(2ρ) crosses 1, where zero is forced by the theorem. The paper uses it to link smooth training-time acquisition to an abrupt-looking benchmark transition, not pushback.
8. **Controllability is not learning.** "The missing quantity linking controllability to learning is objective–update alignment." (Theorem 24, a training-loss update; checked on 80 Pythia-70M examples: mean alignment² 0.6811, log-sensitivity vs log-rate correlation 0.541.) Reading this as "use" or mediation is ours.
9. **Subspace kinematics.** Only the normal component of a change rotates a subspace; updates that stay inside the span (weight decay included) do not rotate it.
10. **Evidence depth (synthetic languages, training time).** Deeper evidence delays persistent preference for the designated answer but not the earlier onset of unsigned movement along the target-alternative margin. "Drift then commitment" is our gloss, not an inference-time result.

## Methods we could reuse
- **Bypass test: Fisher sensitivity of the probe direction.** At each layer, compute vᵀGv for the correct-answer probe direction and compare it to random and top-eigen directions. A direction that is decodable but has about zero sensitivity is output-inert, which is fate 2. (Our adaptation; the paper never applies G to probe directions.) Limit: G is for an intervention at one position, so this is a same-position, first-order test; a direction at an earlier token could still act through attention later.
- **Threshold screen: flip-margin z.**
  - z = m/(2ρ), where m is the pre-pushback top-1 vs runner-up margin and ρ is the root-probability distance between the pre- and post-pushback outputs.
  - Output only, so it is nearly free.
  - A flip needs z ≤ 1 (necessary, not sufficient). Small-margin, small-move flips are candidate threshold crossings (fate 1), but z uses only the two output distributions, so it cannot separate threshold crossing from override by itself. Pairs well with matching items on prior confidence.
- **Controllability vs mediation: alignment factorisation.** Score how well the natural no-pushback→pushback activation displacement aligns with the truth covector, measured in the G-metric.
  - This is our adaptation; the paper's update is a training-loss step.
  - Weak as a fate test: any displacement that flips the answer has negative alignment by definition, so it cannot tell override from bypass or readout failure. Near-zero alignment only means no first-order change in truth log-odds.
  - The two runs have different contexts (extra pushback tokens), so we need a convention for which context G lives on.
- **Stage B fix: natural-gradient steering or LoRA.** Steer the truth-vs-flip log-odds covector at about 45% depth, final position, with a reference-prompt Fisher to protect other behaviour. Compare at matched effect with CAA/Euclidean steering and plain DPO. The paper measures disturbance only on the reference prompts that build the metric, so we must add held-out disturbance ourselves.
- **Feature ranking.** Use Fisher sensitivity and the effect-to-cost ratio on SAE features to find bypass candidates (high activation, low sensitivity) and override candidates.
- **Grassmannian tools.** Chordal distance, principal-vector paths, angle-preserving random controls, and subspace speed (‖Ṗ‖ = √2‖B‖).

## Controls worth copying
- Matched-response comparison: binary-search each method's step size to the same objective change before comparing costs.
- Random subspaces that preserve every principal angle, as the null for "the subspace rotated" claims.
- Random invertible reparameterisation, κ up to 1e4. Any of our geometry measures that changes under it is a coordinate artefact.
- A displacement-magnitude-matched null rather than a shuffled null. The shuffled null gives 0.025 against 0.636 for the matched one, so it makes sharing look everywhere.
- Permutation-null calibration of similarity scores: CKA's null is 0.316, while rank agreement's is 0.001.
- Fix every prediction before measuring outcomes.
- Measure off-target KL on held-out neutral prompts (the paper does this for the per-prompt instruct frontiers; Supp. Note 14 says localised objectives use on-prompt KL, the Supp. Fig. 2 caption says held-out neutral prompts).
- Keep separate donor, reference and target prompt sets. But note: in the paper only the target effect is held out; disturbance is measured on the reference prompts themselves.

## Pitfalls (quoted in the JSON)
- Local predictions weaken as interventions grow (0.918 → 0.755).
- **bf16 vs fp32 directions agree only weakly (absolute cosine 0.056-0.542, Pythia-410M and 1.4B, two contexts each).** Only four comparisons; doing our 9B-27B G-solves in fp32 is our extrapolation.
- Full solves were only done up to 2.8B. Which GPU carried them is not stated; the 85-iteration CG figure is an exact-arithmetic bound, not an observed count.
- **The reusable update's disturbance is in-sample:** "Reference prompts also define the metric" (p. 78). No held-out disturbance number.
- The erosion predictor works only for moderate interventions and reverses sign beyond them (p. 77).
- A 20-item multiple-choice check showed no difference where off-target KL differed 36-50×. Don't judge fixes by accuracy alone.
- **The pullback metric describes reweighting. Moving mass between disjoint tokens (Canberra → Sydney) also needs a ground metric on tokens.**
- Fisher cost measures how much a feature matters to the output, not what it means. Attribution patching still predicts raw effect size better.
- Euclidean steering depends on the coordinate chart. Raw cosine comparisons are dominated by a shared mean direction.
- Too much damping erases the natural-gradient advantage.

**Limitations:** there is no dedicated section. The LoRA result "does not establish safety-specific retention or broader capability preservation" (p. 15). Code is available only on request.

**Future work:** apply the reference-set construction in parameter space, giving a "behaviour-level trust region" for preference fine-tuning (p. 15). Other proposals are Fisher-weighted SAEs and output-geometry audits across checkpoints.

## Bearing on the six fates
| Fate | How | Strength |
|---|---|---|
| 1 readout/threshold | Flip-margin z is an output-only necessary condition; shows where flips can occur but cannot separate threshold from override alone | weak-moderate |
| 2 bypass | The null space of G is exactly the output-invisible directions; vᵀGv on probe directions (same-position, first-order) | moderate (method only) |
| 3 not routed | Cleaner injection test (less collateral); alignment factorisation (our adaptation, does not discriminate fates) | moderate (injection), weak (alignment) |
| 4 override | Objectives combine as covectors; summed CAA moved sycophancy the opposite way (no mechanism given); effect-to-cost ranking | weak |
| 5 transformed | Rotation (isometry) fits for cyclic concepts, output space only | weak |
| 6 erased | Outputs alone cannot tell erased from masked | none |
| Stage B / H2 | Reusable anti-sycophancy/truth update (disturbance in-sample, small effects, no flip outcome); natural-gradient LoRA as a trust region; explicitly distinct from DPO | moderate (≤1.8B) |

## Group angle
- **Mainly a caution.** The paper shows Euclidean distances, inner products and cosines are not fixed by behaviour (GL(d) gauge freedom). That Euclidean principal angles between internal subspaces inherit this is our inference; the paper uses principal angles only for read-out subspaces, and does not discuss how norm layers restrict the gauge in real transformers.
  - Within one model's residual stream the basis is shared, so cross-layer angles can be computed.
  - But they answer a different question than angles in the pullback metric, which weights directions by their effect on the output.
  - For fate claims, measure angles in the G-metric (with a stated layer for G, since G is layer-specific), or at least run the reparameterisation test.
- **Tools it offers:**
  - chordal distance
  - angle-preserving random controls
  - subspace speed (only the normal component rotates)
- **Linear vs curved steering.** With both arms relinearised after every step, natural-gradient paths cost 11.5-127× less path KL than Euclidean paths: the Fisher advantage persists along relinearised paths. There is no relinearised-vs-single-step comparison, and no nonlinear manifold steering is tested (only cited).
- **Not covered:** Procrustes alignment, fork points, curvature of subspace paths, and tracking internal truth subspaces across layers or tokens.

## Where in the architecture, and what fix
- **Where:** residual stream at a middle layer, about 45% depth (final prompt position for the reusable control). The advantage is largest early-to-mid and smallest near the read-out. The top 5 output-Fisher modes carry 55% of eigenvalue mass and the largest answer-motion value (0.59) among the bands; these are output-geometry modes, not a place in the architecture.
- **Fix:** a damped natural-gradient truth steer with a reference-prompt Fisher, or natural-gradient LoRA, compared with DPO and CAA at matched effect, with all solves in fp32 and disturbance measured on held-out prompts.

## Verdict and relevance
**Verdict:** borrow the methods and controls, not the findings. Nothing was tested at 9B-27B, on pressure-induced flips, or on internal truth signals. The code is available only on request, but the formulas and solver recipe are fully specified. Before reimplementing, read the cited scale precedents: FishBack (2605.17231) extends natural-gradient steering to 8B, and Park et al. (2602.15293).

**Relevance:** indirect but useful. The paper studies output geometry and minimum-disturbance control, never failures of acting on what is decodable. Its output-sensitivity measure is a natural bypass test, its anti-sycophancy steer is a ready Stage B arm, and its coordinate-dependence result (for distances and cosines; for principal angles by our inference) applies to how this wave measures subspace paths.
