# The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models

Cho, Wu, Da Costa, Koshiyama. arXiv 2602.08159v2 (29 Jul 2026). Code: https://github.com/seonglae/confidence-manifold. Slug: `c15-confidence-manifold-2602.08159`

## In one line
For a *given* (teacher-forced) answer, whether it is correct is decodable from the last-token residual stream as a **mean shift inside a 2-8D supervised subspace** at mid-to-late depth. Two class centroids work as well as a trained probe. Directions fit on one dataset are mis-oriented on others.

## Relevance to our project (honest)
Moderate, and less than the title suggests. The paper describes the static geometry of correctness for given answers: single turn, no pressure, up to 14B. It does not ask what happens to that signal when the model acts against it. Most of its "causal" tests act on probes, not on the model. Use it for two things:
1. A baseline for what our correct-answer subspace should look like at 9B-14B.
2. Two warnings that bear on H3. A direction can be present but pointed elsewhere in another condition. And a teacher-forced detector loses 27 points on the model's own answers.

## Sections read (all, in order)
Abstract; 1 Introduction; 2 Background; 3 Method; 3.1 Data Construction and Direction Learning; 3.2 Geometric Analysis; 3.3 Causal Validation; 4 Experiments; 5 Results; 5.1 The Correctness Signal is 2-8 Dimensional; 5.2 Two Centroids Suffice; 5.3 What the Interventions Do and Do Not Show; 5.4 Internal vs Output-Based; 5.5 Transfer Requires Joint Training; 5.6 Where the Signal Sits; 6 Discussion; 7 Limitations; 8 Conclusion; References; A (A.1, A.2); B (B.1-B.3); C (C.1-C.3); D (D.1-D.5); E (E.1-E.6); F (F.1-F.3); G Pooled LID; H Full-space DoM; I Steering details; J Nested CV; K Paraphrase control; L DAS details; M RLACE/INLP; N Linear alternatives; O Llama-3.1-8B; P Generative evaluation. (57 headings)

## Setup
- **Data:** teacher-forced (question, given answer) pairs labeled correct or incorrect. Main set: TruthfulQA (817 questions, misconception-heavy). Transfer sets: SciQ, CSQA, FEVER, HaluEval. Steering: direction fit on 200 questions, evaluated on 617 held out. Own-generation check: multiple-choice CSQA.
- **Models (11):** GPT-2 124M/355M/774M (base); Llama-3.2-1B/3B, Qwen2-1.5B/7B, Gemma-2-2B, Mistral-7B-v0.3, Llama-3.1-8B, Qwen2.5-14B (all instruction-tuned). Scale and tuning are confounded.
- **Positions/layers:** last token, every layer; best layer picked by CV test AUC. Steering is applied at every token at the best layer.
- **Baselines:** P(True), NLL, token entropy, verbalized confidence, semantic entropy, CCS, L2 norm, reconstruction error, LOF, cluster uncertainty, pooled LID, LLM-Check, LDA, full-space DoM.
- **Controls:** GroupKFold by question, nested CV, random and orthogonal steering directions, length probe, paraphrase control, contamination check.

## Key results (with quotes)
1. "two class centroids in a 2–8 dimensional subspace match a trained linear probe, and 25 labeled examples recover 90% of full-data AUC on GPT-2." (p1)
2. "A correctness probe reaches 0.80–0.97 AUC across the 11 models" (p4)
3. "Every model peaks between 2 and 8 dimensions and then degrades" (p4)
4. The mean shift is in the raw stream: "difference-of-means in the raw residual stream, with no projection, already reaches 0.72–0.91 AUC, and PLS adds only +0.034 to +0.081" (p4)
5. Unimodal classes: "KMeans-2 and KMeans-3 within each class cost 1.3–2.4pp, so the per-class distributions behave as single unimodal clusters" (p4)
6. Steering (Qwen2-7B): "the error rate rises from 0.56 to 0.61 at α = −5 and falls to 0.52 at α = +5" (p5). Controls: "random directions move the error rate by −0.8pp and orthogonal directions by −0.3pp" (p5). Six models: "+2.8 to +10.0pp" (p5).
7. RLACE: "Across all 5 tested models this drops AUC to chance" (p5). But INLP: "After 32 iterations, AUC degrades substantially: GPT-2 0.58, GPT-2-Large 0.60, Qwen2-7B 0.75, Mistral-7B 0.72, Llama-3B 0.73." (p25)
8. Regime: "On TruthfulQA, output methods (P(True), token entropy, semantic entropy) fall to 0.44–0.64 while probes reach 0.80–0.94" (p6). Why: "the model confidently asserts the wrong answer, so any signal derived from the output distribution encodes the belief rather than its correctness." (p6). On Qwen2-7B: "P(True) drops to 0.48 (below chance) while centroid remains at 0.93, a +45.2pp gap." (p21)
9. Transfer: "A probe fit on TruthfulQA alone transfers close to random: 0.39–0.66 AUC on SciQ, CSQA, and FEVER" (p6). "The failure is one of orientation, not of absence." (p6)
10. HaluEval: "...yields 0.99 in-domain AUC, and the two directions are essentially orthogonal (| cos θ| = 0.08)" (p7). "a below-chance score is the signature of a direction that is present but pointed elsewhere." (p7)
11. Depth: "...the GPT-2 family peaks at its final layer, larger models at 46–75% depth" (p7). "Layer selection is 11× more impactful than dimension selection." (p14)
12. Across models: "What is shared across models is the abstract structure of the encoding, its dimensionality and mean-shift form, not the parameters of any particular probe." (p7)
13. Own answers: "...reaches 0.957 AUC (n = 242) while the same subspace applied to the model's own generated answers reaches 0.688 (n = 199), a 27-point drop." (p26)

## Figures and tables (29 listed, 29 covered; all page images viewed)
| id | page | what is plotted | takeaway | numbers | text matches? |
|---|---|---|---|---|---|
| Figure 1 | 2 | Output-method vs probe AUC, 4 models | Probes beat outputs by 27-37 pts on TruthfulQA | 0.52/0.79, 0.58/0.95, 0.61/0.92, 0.55/0.92 (printed) | partly: Qwen2-7B values match no Table 4 entry; output method unnamed |
| Table 1 | 4 | Best-layer probe AUC, layer/depth, 11 models | 0.80-0.97; 46-75% depth for 1B+ | e.g. Qwen2-7B 0.94 at L20/28 | yes |
| Table 2 | 5 | PLS dim sweep 1-32D | Peak at 2-8D; big drop only for small models | GPT-2 0.758 -> 0.621; 14B 0.951 -> 0.939 | yes |
| Table 3 | 5 | DAS sweep summary | Peaks 1-5D | Qwen2-7B 3D 0.916, 32D 0.894 | yes (but DAS acts on a classifier, curves nearly flat) |
| Figure 2 | 6 | Error rate vs steering alpha, learned/random/orthogonal | Learned direction shifts error; controls near baseline at the extremes | learned ~0.627 at -5, ~0.517 at +5, base ~0.558 (approx, read from figure) | partly: plot shows ~0.627, not 0.61; not monotonic; controls drift to ~0.59 mid-range |
| Table 4 | 6 | P(T), entropy, SE, CCS, probe on TruthfulQA | Outputs near chance | Qwen2-7B 0.64/0.52/0.58/0.80/0.94 | yes |
| Table 5 | 7 | In-domain, single, multi transfer (PLS-5D) | Joint training fixes transfer for 1B+ only | single 0.39-0.66; multi up to 0.905 | yes |
| Figure 3 | 14 | Intrinsic dim by depth; AUC vs dim; cross-layer probe cosine | Compression with depth; dimension explains little; gradual rotation | mean ~0.6 -> ~0.15-0.2, late bump ~0.35; r=-0.43 (approx) | partly: (c) is a smooth band, not clear blocks |
| Table 6 | 15 | 6 geometric classifiers in PLS-8 | Centroid = probe | Qwen2-7B 0.885 vs 0.891; NCH 0.849 | partly: NCH is up to 0.045 below, not within +-0.02 |
| Table 7 | 16 | Unsupervised features | 0.51-0.62 | max 0.621 | yes |
| Figure 4 | 17 | 3D PLS hulls, 6 models | Heavy overlap in 3D | AUC labels 0.94-0.80 | no: caption says GPT-2 separates more clearly; image shows Llama-3B clearest, GPT-2 overlapping |
| Table 8 | 17 | GPT-2 family AUC and dim by depth | AUC up, dim down | GPT-2 0.632 -> 0.772; 24.3 -> 8.9 | partly: GPT-2 dips at 75% |
| Figure 5 | 18 | 3D PLS for 1-2B models | Overlap despite AUC 0.91-0.93 | labels 0.91, 0.93, 0.93 | yes |
| Table 9 | 18 | Dim sweep with stds | Variance grows with dim | +-0.003 -> +-0.034 | yes |
| Table 10 | 19 | Full in/cross-domain | HaluEval below chance cross-domain | 0.181-0.472 | yes |
| Table 11 | 19 | Few-shot GPT-2 | Centroid = probe at all N | 0.60/0.69/0.76/0.78 | yes |
| Table 12 | 19 | Full 3584D vs PLS-5D transfer | PLS +10-14 pts | 0.50 -> 0.64 etc. | yes |
| Table 13 | 20 | Native PLS-8 per dataset | Signal everywhere in-domain | 0.61-0.96 | yes |
| Table 14 | 20 | Qwen2-7B per-dataset dim sweep | Low-rank peaks | TruthfulQA peak 12D 0.929 | partly: 12D is outside 2-8D; FEVER not degraded |
| Table 15 | 20 | LR, centroid, LDA, QDA, KMeans | Single centroid enough | 0.910/0.913/0.913/0.921/0.886/0.897 | yes (diffs within ~1 std) |
| Table 16 | 21 | Centroid vs LR over K | Centroid catches up and edges ahead | K=400 0.907 vs 0.894 | yes |
| Table 17 | 21 | P(True) vs centroid, 4 datasets | Internal wins only on TruthfulQA | 0.477 vs 0.929 | yes |
| Table 18 | 22 | Pooled LID vs PLS-DoM | LID ~chance | 0.51-0.58 vs 0.78-0.94 | yes |
| Table 19 | 22 | Full DoM vs PLS-DoM vs probe | Mean shift is in the raw stream | +0.057 mean gain | yes |
| Table 20 | 23 | Steering, 6 models | 2.8-10pp effects | Gemma +2.8 with orth -4.5 | yes (controls move up to 4.5pp) |
| Table 21 | 23 | Nested vs standard CV | No inflation | +0.005, -0.026 | yes |
| Table 22 | 24 | Full DAS sweep | Nearly flat curves | Llama-3B 0.935-0.944 | yes |
| Table 23 | 25 | RLACE before/after | To ~0.5 | 0.908 -> 0.505 (Qwen2-7B) | yes (but INLP leaves 0.72-0.75) |
| Table 24 | 25 | LDA, DoM, PLS-1/5/8 | PLS >> LDA; unscaled DoM can fail | GPT-2 unscaled DoM 0.469 | yes |

## Methods worth stealing (and which fate each diagnoses)
- **PLS dimension sweep.** Shows how many supervised directions carry correctness. Fate 5: does the subspace change dimension or orientation under pressure?
- **Two-centroid / standardized DoM detector.** Fate 1: are the centroids still separated while the answer flips? A first group comparison of flipped vs resisted items.
- **KMeans-per-class / QDA unimodality test.** Discrete states vs continuum: do flipped items form their own cluster?
- **Cross-condition transfer + direction cosine + below-chance check.** Fate 5 vs 6: a frozen no-pressure probe below chance while a native refit is high means re-pointed, not erased. A direct test design for H3.
- **Cross-layer probe-weight cosine matrix.** Tracks the direction along depth (only at the last token).
- **Steering with random and orthogonal controls; specificity = learned minus orthogonal.** Causal controllability (fate 3 vs 2). It is not endogenous mediation.
- **RLACE/INLP.** A positive control for what linear erasure looks like. INLP shows redundancy.
- **DAS as done here:** do not reuse. It is fit on stored embeddings and flips a classifier.
- **Teacher-forced vs own-generation (multiple-choice) check.**

## Controls worth copying
GroupKFold by question; preprocessing fit on training folds only; nested CV; random and orthogonal steering controls; a standardized full-space DoM baseline; output and unsupervised baselines on the same items; length and paraphrase controls; native refit next to transfer; a judge-free multiple-choice check on own generations.

## Pitfalls
- DAS is on "pre-extracted embeddings at the optimal layer (no GPU model loading needed)" and flips "the classifier prediction" (p24). The rank claim is about a probe, not the model.
- RLACE shows necessity "for detection" (p5), not for behavior. INLP leaves 0.72-0.75 after 32 removals (p25).
- The steering judge is keyword overlap (p23).
- Teacher-forced to own answers: 0.957 to 0.688 (p26).
- Scale is confounded with instruction tuning (p8).
- A PLS basis is defined only up to rotation, so its components have no meaning (p4).
- 3D projections look overlapping even at high AUC (p18).
- Unscaled DoM "is dominated by high-norm dimensions" (p25). Standardize first.
- The abstract overstates steering ("9.1 points on six models", p1). Table 20 gives 2.8-10.0.
- The paper says both "a single direction suffices" (p15) and 2-8D.
- The LID sign is chosen on the full data (p22).

## Limitations and future work (quoted)
- Limitation (p8): "the geometry is thus a property of the teacher-forced representation, and extending detection to free-form generations is open"
- Future work (p26): "we leave free-form generative validation to future work once a trustworthy automatic judge is available"

## How it bears on our six fates
| fate | how | strength |
|---|---|---|
| 1 readout/threshold | Centroid 0.93 vs P(True) 0.48 on TruthfulQA: decodable but not expressed (static, no pressure) | moderate |
| 5 transformed | Below-chance transfer + high refit + near-orthogonal directions = "present but pointed elsewhere" | moderate |
| 6 erased | RLACE positive control; INLP residual warns erasure needs iterative checks | weak |
| 3 not routed | Steering 2.8-10pp = some causal controllability | weak |
| 2 bypass | Not tested (no check of the model's own use) | weak |
| 4 override | Not addressed | weak |

## Group/subspace angle
- **Useful priors.** Correctness is a mean shift in a 2-8D subspace, and classes are unimodal. Start flipped-vs-resisted comparisons with two centroids in a PLS subspace before any manifold method.
- **Orientation matters more than dimension.** Intrinsic dimension explains R2=0.18 of AUC, and dataset-specific directions are near-orthogonal. For H1, compare subspace *orientation* across pressures (principal angles, joint fitting) rather than dimensionality.
- **Warnings.**
  - The "manifold" is a linear subspace.
  - Label-free geometry (pooled LID, norms, LOF) misses correctness.
  - 3D plots mislead.
  - The rank claim rests on a probe-only DAS.
- **Gap they leave.** Directions rotate gradually across layers, but nothing is tracked across tokens.

## Where in the architecture
- **Component:** last-token residual stream only; there is no head or MLP analysis.
- **Depth:**
  - The best layer sits at 46-75% depth for 1B-14B instruction-tuned models (Qwen2-7B L20/28, Mistral-7B L23/32, Llama-3.1-8B L16/32, Qwen2.5-14B L32/48). For GPT-2 it is the final layer.
  - Intrinsic dimension falls from 20-55D early to 8-12D at the best layer.
  - Nominal phase boundaries sit near 30% and 70% depth.
  - Mistral re-expands after its best layer.
- **Across models:** the low-rank, mean-shift form holds across five families. Probe weights do not transfer between models.

## Actionable fix
- **Monitor.** Use a cheap two-centroid detector at mid-to-late depth. Fit it jointly on several datasets and refit it per model.
- **Where it pays off.** Internal monitors earn their keep mainly on confidently-wrong items. For our controls, the logit margin or P(True) will look worst exactly there.
- **Training data for the detector.** A detector trained on no-pressure, teacher-forced items may lose a lot on the model's own pressured answers. Fit it on own generations.
- **Steering.** It gives only 2.8-10pp, so it is not a fix by itself.

## Reuse verdict
**Adapt.**
- **Reuse:** the protocol, the code, the controls, native-vs-transfer refits, and the multiple-choice own-generation check.
- **Do not reuse:** their DAS or RLACE results as causal evidence about behavior, or teacher-forced probes for our model-generated flips.
