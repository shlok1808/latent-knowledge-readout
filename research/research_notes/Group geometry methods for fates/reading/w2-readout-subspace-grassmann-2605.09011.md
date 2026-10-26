# A Geometric Perspective on Next-Token Prediction in LLMs: Three Emerging Phases

Lombardo, Trimigno, Cagnoni (Univ. Parma), arXiv 2605.09011, May 2026. Slug: `w2-readout-subspace-grassmann-2605.09011`.

## In one paragraph

The authors train a tuned lens (a learned affine readout) for every layer of eight models (Qwen2.5 1.5B-32B, OLMo-2 1B-32B; base checkpoints inferred from the names). They take the SVD of each layer's lens correction matrix and treat its top-k right singular vectors as a point on the Grassmannian Gr(k, d). They then track how similar consecutive layers' subspaces are (mean cosine of principal angles, "RSS"). The profile rises, plateaus and falls, which gives three phases. The middle phase grows fastest with depth. The authors say the first and last stay about fixed, but that holds only for Qwen2.5; in OLMo-2 both grow (b1 4 to 17 layers, Phase 3 up to 12). The tuned lens reads the final token from layer 1, but the logit lens cannot read it until the last phase.

**Most useful for us:** the tuned-lens vs logit-lens gap per layer, used per item, as a "decodable but not yet readable" trace (fate 1 vs fate 4). Also the published phase bands for Qwen2.5-14B and 32B as a WHERE axis.

**Main catch:** the readout subspace comes from lens *weights*, so every input gets the same profile. It cannot compare flipped with resisted items unless we rebuild it from item activations.

## Sections read
Abstract; 1 Introduction; 2 Background and Related Work (2.1, 2.2); 3 Methodology (3.1 readout subspace and RSS; 3.2 phase boundaries; 3.3 functional setup); 4 Results (4.1 phase detection; 4.2 geometric characterisation; 4.3 functional interpretation); 5 Discussion (mean-field comparison, conclusions, limitations); References; Appendices A (lens training), B (pairwise matrices), C (Pareto frontier), D and D.1 (unimodality, breakpoints), E (full results).

## Setup
- **Data:** lens training on 226.49M SlimPajama tokens; all metrics on 16M held-out validation tokens. Generic web/code text only: no facts, no failures, no pressure.
- **Models:** Qwen2.5-1.5B/7B/14B/32B, OLMo-2-1B/7B/13B/32B (base inferred from names; the paper does not say; OLMo-2 sizes come from three release dates).
- **Where measured:** every layer, every token position (averaged); k = 1-50% of d.
- **Baselines:** logit lens; four distribution fits for the profile shape.
- **Controls:** held-out data; train/val lens gap; sweep over k; breakpoint stability across k (sigma <= 3.9%). **Missing:** any causal test; any null baseline for RSS.
- **Compute:** 2x L40S. Lens training from a few hours (1B) to about 4 days (32B); 500 steps x 2^18 tokens is about 131M tokens, a subset of the 226M-token set.

## Figures

| Fig | Page | What it shows | Numbers | Matches text? |
|---|---|---|---|---|
| 1 | 2 | Hit@k vs layer, tuned lens vs logit lens, Qwen2.5-32B / OLMo-2-32B | tuned Hit@1 ~0.2 to ~0.8; logit lens ~0 until layer ~45, then ~0.43 (Qwen) / ~0.47 (OLMo) (approx) | yes |
| 2 | 4 | Pairwise RSS matrices, OLMo-2-32B, k = 1-50% | Visibility 0.48 down to 0.19 | yes |
| 3 | 4 | Visibility vs energy trade-off, 4 models | monotone | yes |
| 4 | 5 | Unimodal fits and chord breakpoints, Qwen2.5-14B / OLMo-2-13B | 14B: b1 = 5, b2 = 40; 13B: b1 = 10, b2 = 32 | yes |
| 5 | 5 | R2 of 4 fits at k = 50% | best fit 0.75 (OLMo-2-32B) to ~1.0 | partly: Qwen2.5-7B missing; Kumaraswamy beats unimodal for Qwen2.5-32B |
| 6 | 6 | Breakpoints and phase widths vs depth | b2 slope 1.00 (Qwen), 0.88 (OLMo); b1 slope 0.11 / 0.26; OLMo b1 4 to 17 layers, P3 up to 12; Qwen P3 R2 'nan' | partly: P1 and P3 flat in Qwen only |
| 7 | 6 | FFN/MHA norm ratio per phase | P3 ratio ~2-3.8 in all 8 (approx) | partly: OLMo-2-32B P1 ~3.2 is not "balanced"; OLMo-2-7B P1 ~0.4 |
| 8 | 7 | Breakpoint robustness; consensus split | Qwen2.5-14B 11%/85%; Qwen2.5-32B 13%/89%; mean P1 19.8%, P2 59.0%, P3 21.2%; counts are transitions (L-1) | yes |
| 9 | 7 | Update rank and norm, 4 models | rank peaks in P2, falls in P3 (Qwen2.5-7B ~2880 to ~2700); norm up to ~600 (Qwen2.5-32B), ~35 (OLMo-2-32B) (approx) | yes |
| 10 | 8 | Raw RSS profiles, 4 models | moves only ~0.1-0.2 | partly: caption says all eight |
| 11 | 8 | Tuned-lens Hit@1/Hit@5 and ratio vs depth | ratio ~2.0 down to ~1.2-1.35 | yes |
| 12 | 9 | RSS and abs. cos(update, residual), 4 models | abs. cos up to 0.3-0.4 | yes (loosely "orthogonal") |
| 13 | 9 | Effective rank of hidden states, 4 models | Qwen2.5-7B also dips in P2 | partly |
| T1 | 12 | SlimPajama token counts | train 226.49M, eval 18.66M | yes |
| 14 | 12 | Lens train/val loss per layer | gap +0.012 to +0.089 | yes |
| 15 | 13 | Pairwise matrices, Qwen2.5 family | Visibility ~0.46-0.59 to ~0.15-0.17 | partly: 35% column missing |
| 16 | 14 | Pairwise matrices, OLMo-2 family | ~0.48-0.58 to ~0.18-0.19 | partly: 35% column missing |
| 17 | 14 | Pareto curves, all 8 | near-linear frontiers | yes |
| 18 | 15 | Fits at k = 15%, all 8 | OLMo-2-32B best 0.775 | partly: no breakpoint panels; Kumaraswamy wins 3/8 |
| 19 | 16 | Raw RSS profiles, all 8 | P3 drop ~0.05-0.15 | yes |
| 20 | 17 | RSS and abs. cos, all 8 | OLMo-2-1B peaks in P2 | partly: contradicts "highest in P3 (8/8)" |
| 21 | 17 | Hidden-state effective rank, all 8 | Qwen ranks rise in P3 | partly |
| 22 | 18 | Update rank and norm, all 8 | final norms ~16 to ~600 | mostly |

Page images: `../papers/w2-readout-subspace-grassmann-2605.09011/pages/p02.png` ... `p18.png`.

## Key results (exact quotes)
- "The representation lens recovers the final token early, while the LogitLens remains nearly blind until the final phase, indicating predictive information is present but hidden in superposition." (p. 2)
- "On average, Phase 1 spans 19.8% ± 6.9%, Phase 2 59.0% ± 12.4%, and Phase 3 21.2% ± 8.6%, with the intermediate phase dominating." (p. 7)
- "Overall, as depth increases, Phase 2 expands while Phases 1 and 3 remain roughly constant." (p. 8) True for Qwen2.5 only; OLMo-2 Phase 1 and Phase 3 grow with depth (Figure 6).
- Authors' abstract claim that phases differ in effective rank, not in alignment with the residual stream, is partly contradicted by their Figures 12, 20, 21: abs. cos(update, residual) also splits by phase, and hidden-state rank does not concentrate in Phase 3 (it rises in Qwen2.5). Only update rank and norm concentrate.
- "Hit@1 rises to ≈ 0.56 and Hit@5/Hit@1 decreases, showing progressive concentration, while LogitLens remains low (≈ 0.03–0.10), implying disambiguation within the subspace without explicit decoding." (p. 9)
- "Contributions are more balanced in Phase 2, and FFN dominates across both families in Phase 3, suggesting activation of distributed memories to write the final prediction into the residual stream." (p. 8)

## Methods we could use
1. **Readout subspace + RSS (principal-angle mean cosine).** As built here it is input-independent, so it gives no fate signal. Rebuilt on item or condition subspaces (neutral vs pushback, same layer), it could separate *transformed* (rotated) from *erased* (gone) and show where paths fork.
2. **Visibility vs energy rule for k.** Use it to pick the subspace size.
3. **Double-chord Kneedle breakpoints.** A reproducible way to cut any layer profile into before / fork / after.
4. **Tuned lens vs logit lens rank per item.** Track the correct answer vs the user's claimed answer. A crossing in the middle band fits override; losing only in the last band fits late readout failure. The lens reads what the model *will say*, not correctness.
5. **Update geometry (abs. cos, effective rank, norm) and FFN/MHA ratio.** Helps localise the flip to attention vs MLP. Only useful if computed per item and per condition.

## Controls worth copying
- Fit the lens or probe on one split and measure on held-out data; report the train/val gap per layer.
- Sweep the subspace size and show the result holds across it.
- Always show the learned readout next to the fixed output readout.
- Test the shape assumption against more flexible fits.
- Use two model families.

## Pitfalls
- Stated: "Breakpoint estimates may vary by 1–3 layers across values of k due to residual noise." (p. 6)
- Our reading: the subspace is input-independent (one trajectory per model).
- Our reading: a tuned lens is trained toward the model's own output. On flipped items it decodes the flip, so pair it with a correctness probe and the logit-margin control.
- Our reading: the lens was trained on web text; chat pushback prompts are out of distribution.
- Our reading: the Hit@k target is undefined ("how often the final token appears in the top-k at each layer", p. 2); last-layer tuned Hit@1 ~0.8 suggests agreement with the model's own output, not the data label (inferred).
- Our reading: absolute RSS changes are small (~0.1-0.2) and there is no null baseline. Several text claims are looser than the figures (Figs 5, 7, 13, 18, 20, 21).

## Limitations / future work (quotes)
- "Phase interpretations are based on correlational evidence without causal interventions; activation patching or causal mediation analyses could provide stronger mechanistic grounding." (p. 10)
- "Our analysis is restricted to two decoder-only families with dense attention and standard residual connections; extension to MoE, state-space, or non-residual architectures is left to future work." (p. 10)

## Bears on fates
| Fate | How | Strength |
|---|---|---|
| 1 readout/threshold | tuned vs logit lens gap per layer, per item | moderate (method only) |
| 4 override | Phase 2 "hoisting overriding" = candidate competition; test with a rank crossing | weak |
| 5 transformed | principal angles are the right tool, but only used across layers, not conditions | weak |
| 6 erased | effective rank, averaged over all tokens | weak |
| 2 bypass, 3 not routed | no interventions | none |

## Group angle
A cheap template: SVD, then top-k subspace, then principal angles, then RSS and pairwise matrix, then Kneedle breakpoints. Swap the object (lens weights) for condition-level subspaces to get "where do neutral and pushback paths fork". There is nothing on manifold vs linear steering. Its late reorientation phase is directionally consistent with 2605.12714 (late curvature peaks in decoder models); qualitative only, since objects and statistics differ. Cited leads, not read: Gurnee et al. [10] (manifolds), Shai et al. [18] (belief-state geometry), Bruno et al. [5] (mean-field phases), LoRA-Lens [20].

## Where in the architecture / what fix
- **Where:** Phase 1 ends at 11-29% of depth; Phase 3 starts at 60-89%. Qwen2.5-14B: about layers 5 and 40. Qwen2.5-32B: about layers 8 and 56. These index layer-to-layer transitions (e.g. 14B Phase 3 = transition 40 to 41 onward: 7 transitions, 8 layers). Phase 3 is FFN-dominated in all models.
- **Fix:** none in the paper. Our proposal (untested): if flips concentrate in Phase 3, restrict the Stage B fix to late MLPs, then recheck the middle-band readout to see whether the problem is fixed or only hidden. If flips happen in Phase 2, target the middle band and the attention that reads the user turn.

## Verdict
Reuse partly: the tuned-vs-logit lens per-item trace, the phase bands and the principal-angle recipe. Do not reuse RSS on lens weights as a fate test. Relevance: low as evidence, medium as a toolkit.
