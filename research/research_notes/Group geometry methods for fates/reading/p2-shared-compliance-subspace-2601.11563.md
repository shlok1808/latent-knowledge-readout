# Shared "compliance subspace" for sycophancy and conformity (Zhang & Chen, arXiv 2601.11563)

**One line:** Sycophancy and conformity move the final-token hidden state in roughly similar directions (2D PCA only), and a line in logit space (I = prior true-lie gap, S = pressure push on the lie) predicts who flips. Evidence is weak; the design is reusable.

## Sections read
Abstract; 1 Introduction; 2 Theory (2.1, 2.2, 2.3, 2.5; no 2.4); 3 Methodology (3.1-3.3); 4 Results (4.1-4.4); 5 Discussion (5.1, 5.2); 6 Implications (6.1-6.3 incl. IEAF table); 7 Conclusion; References skimmed. No appendices.

## Setup
- **Data:** 2995 MCQ items (2000 MMLU + 995 MMLU-Pro), filtered for correct neutral answers. Each item: neutral, sycophancy (one authoritative user), conformity (three-agent consensus), each mild and aggressive.
- **Models:** 15 LLMs for behavior (7B-72B and MoE); mechanistic work on Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, InternLM3-8B-Instruct.
- **Position/layer:** last token before generation; layer not stated.
- **Measures:** S = L(lie|pressure) - L(lie|neutral); I = L(true|neutral) - L(lie|neutral); Mconf = same gap in probabilities.
- **Controls:** neutral baseline only. No random-direction control, no held-out SVM split, no causal test, no CIs.

## Figures
| Fig | Page | What | Takeaway / numbers | Text matches? |
|---|---|---|---|---|
| 1 | 2 | Cartoon of signal competition | Framing only | yes |
| 2 | 5 | Flowchart: I vs S into "attention gating" | Mechanism hypothesised, never tested | yes |
| 3 | 9 | Conformity vs sycophancy rate, 15 models | Figure: Spearman rho^2 0.344, p 0.022; text: Pearson R^2 0.309, p 0.032. Qwen2.5-7B has the top conformity (~0.39, approx) | **no**: stats differ; text calls Qwen robust |
| 4 | 9 | PCA, Qwen2.5-7B | PC1 20.9%, PC2 14.1%; 2D angle between shift vectors ~30 deg (approx, our estimate) | yes |
| 5 | 10 | PCA, Llama-3.1-8B | PC1 15.9%, PC2 10.1%; angle ~40-50 deg (approx) | yes |
| 6 | 11 | PCA, InternLM3-8B | PC1 22.1%, PC2 13.5%; angle ~50 deg (approx); mild/aggressive centroids nearly overlap | partly |
| 7 | 12 | S vs I with SVM, Qwen | acc 86.04%, margin 2.405, SV 33.9%; many complied items have I<0 | partly (I>0 filter violated) |
| 8 | 13 | S vs I with SVM, InternLM3 | acc 89.94%, margin 5.817, SV 28.8%; S barely varies (-4..5), split is along I | partly |
| 9 (caption says "8") | 14 | S vs I, Llama | acc 72.57%, SV 63.1% (text only; no page image provided) | not checked |
| 10 | 15 | P(comply) vs Mconf, Qwen | At Mconf=1: conf_mild ~0.74, conf_aggr ~0.66, syc_mild ~0.55, syc_aggr ~0.49 (approx) | partly |
| 11 | 16 | Same, InternLM3 | Sigmoid; syc_aggr still ~0.13 at Mconf=1, others ~0.02 (approx) | partly |
| 12 | 16 | Same, Llama | **Mild > aggressive** compliance throughout (e.g. ~0.26 vs ~0.12 at Mconf 0.8, approx) | **no** |
| IEAF table | 19 | 3x3 proposal matrix | No experiments | yes |

## Key results (quotes checked)
- "The Pearson correlation analysis yields a coefficient of determination (R2) of 0.3089 with a p-value of 0.0315" (p8)
- "The angle between the centroid vectors of the sycophancy cluster and the conformity cluster is acute, indicating high cosine similarity." (p10) - no number given.
- Qwen SVM accuracy 86.04%; InternLM3 89.94%; Llama 72.57% (p12-14).
- "the model retains a compliance probability exceeding 60% even when its Confidence Margin approaches 1.0" (p15, Qwen, aggressive conformity)

## Pitfalls (ours unless quoted)
1. **S>I separability is nearly built in.** S - I = L(lie|pressure) - L(true|neutral). If the true logit barely moves, "complied" is just S > I. A 86-90% linear fit says nothing about internals.
2. **Filter violated:** text says "ensuring I > 0" (p7), but Figs 7-8 and the Mconf<0 regions show many items already preferring the lie in neutral. Those are not knowing-vs-doing cases.
3. "Shared subspace" = two 2D PCA arrows. No cosine, no layer, no random baseline, and directions are over all items (pressure present), not complied vs resisted.
4. Several text-figure mismatches (Fig 3, 6, 11, 12).
5. No causal test; "attention gating" is asserted.

## Limitations / future work
- Limitation (p20): "our reliance on explicit linguistic markers of authority and consensus may not fully capture the complexity of implicit or multi-modal social manipulations"
- Future work: only the IEAF proposals (contrastive pairs, consistency rewards, gating heads, "Real-time Epistemic Drift" monitoring), untested (p19).

## Bearing on our fates
- **Override (4):** framing fits; Qwen high-confidence flips are candidates. No internal evidence. Weak.
- **Readout/threshold (1):** InternLM3 outcome is mostly explained by prior I. Suggests many 7-8B flips are low-margin; matching on margin is essential. Weak-moderate.
- **Erased (6):** they claim not erased, without a probe. None.
- **Bypass, not routed, transformed:** not addressed.

## Group angle
No principal angles, no layerwise tracking, no paths. One cheap test it points to: principal angles between pressure-induced displacement subspaces (pushback vs competing answer vs consensus) at every layer, then whether flipped vs resisted items (matched on I) diverge inside that subspace. The 30-50 deg 2D gaps suggest "shared" is only partial, a useful H1 prediction.

## Verdict
Copy the design (same items under two pressure sources x two intensities; compliance vs prior-margin curves). Don't cite the shared subspace as established.

**Most useful for us:** in Qwen2.5-7B, compliance stays above ~50% even at maximal neutral confidence, so high-margin flips are where to look for override or bypass rather than threshold failure.
