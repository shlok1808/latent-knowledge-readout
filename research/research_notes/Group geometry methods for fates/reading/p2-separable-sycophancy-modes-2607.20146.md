# Gotta Catch them all: the modes of Sycophancy (arXiv 2607.20146)

Jain, Yost, Abdullah (Thoughtworks / SUU), 22 Jul 2026. Read in full from paper.txt (main text plus Appendices A-O). All 62 manifest figures and tables checked against the page PNGs.

## In one paragraph
Gemma-2-9B-it is given 948 social-pressure dilemmas, each under 4 system-prompt personas: Passive Affiliative (PA), Strategic Ingratiation (SI), Defensive Conflict-Avoidant (DCA) and a neutral baseline. The three personas are about equally sycophantic by an LLM judge (index 0.87-0.94 vs baseline 0.35), and a judge can tell them apart only 57.8% of the time (chance 20%). Their last-token residual activations are 100% linearly separable from L14. Geometry peaks at L18, subspace necessity peaks at L22/L26, and logit-lens output only diverges from L23. Mode subspaces are sufficient when injected but ablating them moves output by less than 7%. Only PA has selective heads. **The paper has no factual items and no correct-answer variable, so it does not study our question.** Its value for us is methods, plus two cautions.

## Setup
- **Data:** 948 GPT-4o-mini-generated dilemmas (8 contexts x 8 pressure mechanisms x ~20 variants), 2-4 sentences each, with no gold answer. Behavioural tests use 1,000 situations x 5 conditions (Figure 29 n sums to 1,000). There is also a persona-free control of 80 authored scenarios.
- **Model:** Gemma-2-9B-it only. Residual stream `hook_resid_post` at L14/18/22/26/30/34, last prompt token. Heads: 16 x 6 layers.
- **Controls:** paired deviation vectors (mode minus same-item baseline); train/test split by situation; random-direction and full-layer controls for necessity; zero vs mean head ablation; ICA vs PCA; second judge (kappa 0.67); permutation null for the persona-free probe.
- **Missing:** no second model, no steering, no fixes, no per-item correctness, no matching on confidence.

## Key numbers
| Claim | Number | Page |
|---|---|---|
| Text-only mode classification | 57.8% (GPT-4o-mini), 57.5% (Claude) | 1, 31 |
| Internal separability | 100% probes, K-means ARI 1.000 at L18/L22 | 5, 16 |
| Geometry peak | L18: silhouette 0.365, Fisher 1.643 | 18 |
| Necessity (k=32) | PA 0.063 at L22, DCA 0.044 at L26, SI max 0.011 | 18 |
| Sufficiency (k=8, L18) | PA 0.86 +- 2.30, SI 0.98 +- 0.34, DCA 0.40 +- 2.32 | 24 |
| Logit-lens commitment | KL < 0.06 through L22, peak L33 (PA 18.6) | 18-19 |
| Heads | 26/96 PA-selective, 0 SI, 0 DCA; TOP-10 = 1.6-4.4% of layer effect | 19-20 |
| Baseline neighbour | 84.7% DCA in activations, 54.2% SI in text | 7 |
| Shared dims (cos > 0.9), L18 | PA-DCA 17/50, PA-SI 13/50, SI-DCA 8/50; PA-DCA reaches 37/50 at L34 | 19 |
| Persona-free | 96.7% within-set at L18 (n=80); transfer chance raw, 0.817 after z-scoring at L22 | 33 |

Quotes (all pass quote_check):
- p5: "removing the same directions from sycophantic forward passes produces minimal behavioral disruption, with ICA and PCA variants both yielding less than 7% change"
- p8: "Conversely, mechanisms that induce a mode weakly often bypass its identifiable subspace." (F6. No numbers are given anywhere.)
- p33: "Class-centroid directions in the persona and persona-free regimes are positively but weakly aligned (cosine 0.08-0.12)"

## Figures and tables (all 62)
| ID | p | What it shows | Matches text? |
|---|---|---|---|
| Table 1 | 3 | The three modes: traits, triggers, predicted geometry | yes |
| Fig 1 | 5 | L18 2D PCA: 4 tight, far-apart blobs, ARI 1.000, n=947 per cluster | partly: at 300 dpi PA-DCA adjacency holds, but in 2D the baseline is about equidistant from all three modes; "closest to DCA" rests on Table 3 |
| Table 2 | 6 | F3 summary: sufficiency, necessity, complement recall, judge accuracy | partly: DCA is not the "most distributed" at L18 (0.31 vs PA 0.41); the L22 "ceiling" is not DCA's (0.044 at L26) |
| Fig 2 | 7 | L18 principal angles. Smallest ~10-13 deg, rising to ~90 deg. Counts 10-20/20-45/>45 deg: PA-SI 6/28/16, PA-DCA 9/27/13 (+1 at 5-10), SI-DCA 3/25/22 | partly: "orthogonal fraction near zero" holds only by variance share |
| Table 3 | 7 | Baseline nearest neighbour: 84.7% DCA (activations) vs 54.2% SI (text) | yes |
| Fig 3 | 7 | Sycophancy index by pressure x condition. Flattery .93-.99, moral reframing .79-.87 | partly: "risk of rejection" row is 0.70 and n=1 |
| Table 4-5 | 15 | Context types; six judge dimensions | yes |
| Table 6-7 | 16 | ARI by clustering method; nearest-neighbour chain baseline->DCA->PA<-SI | partly: "removal" undefined |
| Fig 4 | 17 | L22 PCA: two mode clusters touch | yes |
| Fig 5 | 17 | Layer sweep: silhouette/DB/Fisher peak at L18 | **no**: 4th panel is cosine distance, not Mahalanobis; not monotone after L18 |
| Fig 6 / Table 9 | 18 | Necessity vs k: all < ~0.07, threshold 0.9 | yes |
| Table 8 | 18 | Separation metrics and Mahalanobis by layer | yes |
| Table 10 / Fig 7 | 18-19 | Logit-lens KL: silent until L23, peak L31-33, SI first | yes |
| Table 11 | 18 | Raw centroid cosine distance ~0.004-0.010; deviation cosines 0.63-0.87 | minor: 0.871 vs 0.868 in Table 20 |
| Table 12 | 19 | Shared dims 17/13/8 of 50 | yes |
| Table 13 | 19 | L22 heads: H10/H11 specificity ~7-12x, H1 general | partly: raw KL 0.001-0.018; H10/H11 KL is PA ~2x SI, not "comparable" (headers are fine on the page; only paper.txt is garbled) |
| Table 14-15 | 20 | PA synergistic trio; redundant hub; TOP-6 additivity 0.17-0.24 | partly: TOP-6 PA joint KL 0.0057 is below the L22H9+L22H3 pair's 0.032; TOP-6 members not listed |
| Fig 8-10 | 21 | Head output-norm, KL-bucket and entropy maps look similar across modes; L22H10, L26H12, L30 visible | yes |
| Fig 11-13 | 22 | Six attention metrics per mode, visually similar | yes |
| Fig 14-16 | 23 | Zero-ablation (L22H1 darkest), mean-ablation (L22H1 ~0.05), specificity (L30H7 ~15-23) | partly: L22H10/11 still ~8-12 in Fig 16 |
| Table 16-18 | 24 | Sufficiency (std much larger than mean for PA/DCA); ICA vs PCA; complement recall 0.29-0.60 | yes |
| Table 19 | 24 | Same as Table 3 | yes |
| Fig 17-18 | 25 | L30H13 and L30H15: KL ~1e-4, error bars cross zero | partly: titles show 2.3x PA selectivity |
| Fig 19-22 | 26-27 | L18H12, L22H11, L30H7, L22H10: PA ~2x SI/DCA, error bars ~2-3x the means | partly: Fig 19 says "only head showing PA selectivity"; text calls L22H10/H11 shared |
| Fig 23 | 27 | Mean cos and shared dims by layer. PA-DCA 17 -> 37; SI-DCA lowest | partly: PA-DCA dips L14 -> L18 |
| Fig 24-27 | 28 | Principal angles at L14/22/26/30. PA-DCA converges from L22 | yes |
| Fig 28 | 28 | L34: PA-DCA angles all below ~58 deg | **no**: caption says "100% shared", but it is ~37/50 |
| Table 20 | 29 | Output vs activation pairwise cosines, same ordering | yes |
| Table 21-22 | 29 | Per-dimension scores (flattery is the only discriminator); judge confusion (DCA 18.7%) | yes |
| Fig 29 | 30 | Output cosine to baseline by mechanism: differences ~0.03-0.05; n per row 120-131 except "risk of rejection" n=1 (sum 1,000) | partly |
| Fig 30 | 30 | Output nearest neighbours: PA->DCA 0.61, DCA->PA ~0.81 | yes |
| Fig 31 | 30 | Index by condition: 0.87-0.94 vs 0.353 vs 0.046 | yes |
| Fig 32 | 31 | Per-dimension deltas: flattery 0.63/0.76/0.46 | partly: caption quotes absolute values; deltas don't equal Table 21 minus baseline |
| Table 23-24 | 31 | Second judge: 57.5%, r 0.95 | yes |
| Fig 33 | 32 | Delta from baseline: +0.52 to +0.59; anti-syc -0.31 | yes |
| Fig 34 | 32 | By context type: community/civic lowest | partly: 13 labels for an 8-type design |
| Table 25-26, Fig 35 | 33-34 | Persona-free: within-set 0.97-1.0; transfer raw ~chance, z-scored max 0.817 | yes |
| Fig 36 | 34 | Persona-free PCA: baseline far on PC1; SI and DCA overlap in 2D | partly |

## Methods worth stealing
1. **Paired deviation vectors.** Subtract the same-item neutral run. For us: the pressure run minus the neutral run, per item.
2. **Layerwise principal angles between deviation subspaces.** Use the rotation-robust shared-dim count (cos > 0.9) and variance-weighted brackets. This is our H1 test if we swap personas for pushback vs competing answer.
3. **Normalised necessity:** (KL_ablate - KL_random)/(KL_full - KL_random), plus sufficiency by injection.
4. **Complement probe.** Remove the top-k directions and re-probe. Helps tell transformed from erased.
5. **Logit-lens commitment layer** (first layer where KL > 0.1).
6. **Zero vs mean head ablation, then pairwise and group additivity.** This catches redundancy.
7. **z-score each regime before probe transfer.**

## Pitfalls (for reading them and for our own design)
- Modes come from different system prompts, so separability is close to guaranteed. The authors concede this (p9): "may reflect statistical and distributional regularities of the situation text".
- The persona-free control gives each mode its own trigger type, and its baseline is a neutral factual exchange (n=80). That confounds mode with content.
- Causal outcomes are next-token KL at the last prompt token only, and tiny (head KL 0.002-0.018).
- Low necessity is called "redundancy" without any cross-layer joint ablation. **For us: a low ablation effect alone cannot separate bypass (fate 2) from redundancy.**
- Necessity is divided by the effect of zeroing the whole layer (a huge ceiling), and KL is measured against the unablated mode run, not toward baseline. So "<7%" is small by construction and is not a behaviour change. **For us: measure flips and logit margin, and use a neutral-run patch as the ceiling.**
- At the output layer, mode and baseline first-token distributions nearly coincide (logit-lens KL 0.05-0.11 at L40, ~0 at L41). Every causal number is first-token KL, so it tracks a quantity that barely differs between sycophantic and baseline runs.
- TOP-6 PA joint ablation (KL 0.0057) is smaller than one PA head pair's joint ablation (0.032). TOP-6 members are not listed. The "heads compensate" story is unverified.
- F6 (pressure type changes the internal route) cites Appendix J, but Appendix J has no per-mechanism analysis.
- Raw probe transfer falls to chance because of a constant offset. **For us (H3, fate 5): z-score, then Procrustes, before calling a failure "transformed".**

## What it means for us
- **Scoop risk: low.** It has no factual items, no correct-answer probe, no flipped vs resisted split, no confidence matching, no fates and no fixes. Stage A and Stage B are both untouched.
- **Framing risk:** "sycophancy is several mechanisms, and pressure type picks the route" is now claimed on social dilemmas. Frame our H1 around the fate of a decodable correct answer, and cite this paper.
- **H1:** weak, indirect support. There is a shared dominant direction plus mode-specific mid-variance dimensions, and pressure type affects behaviour modestly (moral reframing lowest; baseline varies too, 0.25-0.40). The internal F6 claim has no numbers. Appendix N (pressure-only, no persona, n=80) gives separable states, but each mode has its own scenario content, so it shows decodable context, not different fates.
- **H2:** not tested. It only predicts that mode-targeted fixes would beat a single direction. Its ablations move less than 5-7% of a full-layer-zeroing effect on first-token KL. That is a warning, not evidence that single-direction ablation fails as a Stage B fix, because behaviour after ablation is never measured.
- **H3:** supported by analogy. A probe trained on persona runs fails on persona-free runs unless the offset is removed.
- **Location (Gemma-2-9B-it):** geometry L18, causal L22-26, output L23-33. Heads L22H10/H11 (hub; L22H10 attends to persona tokens, a loose analogue of a head reading the user's claim), L30H7 (most specific). All head effects are 0.001-0.018 nats. This gives us layer priors for the same model in our first fate run.

**Verdict:** reuse the methods, don't cite the conclusions as evidence. Relevance: medium.


_Checked and corrected 2026-10-25 (see checks/p2-separable-sycophancy-modes-2607.20146.md)._
