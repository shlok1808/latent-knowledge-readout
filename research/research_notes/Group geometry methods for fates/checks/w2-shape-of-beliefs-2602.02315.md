# Check: w2-shape-of-beliefs-2602.02315 (Sarfati et al., Goodfire, "The Shape of Beliefs")

Verdict: NEEDS FIXES (minor to moderate; no fabricated numbers or quotes; the problems are over-strong labels and missing caveats)

## Quote check
- `quote_check.py`: 26 quotes, 0 failed (it only tests the first 120 normalised characters).
- Stricter pass (my own script, scratchpad `chk/fullcheck.py`): every quote matches the full text and sits on the page the reading gives. 26/26 OK, including `limitations_quote` and `future_work_quote` (p9).

## Figures vs manifest
- Manifest lists 13 figures (Fig 1-13), no tables. The reading has 13 figure entries. None missing. Page PNGs exist for all (Fig 3 and 4 share p06).
- I opened Fig 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13 (and read the Fig 12 caption). Numbers the reading gives match the plots within reading error: Fig 3A (0.875 at L0, dip ~0.905 at L9, ~0.985 at L15), 3C (0.81/0.67/0.72/0.83), 3D (0.89 -> 0.55 -> 0.50), Fig 4 first eigenvalue per layer (0.59/0.46/0.30/0.23/0.39) and two-eigenvalue sums (0.88 at L0 vs 0.44 at L14), Fig 5 linear peak std ~165, Fig 6 probe peak ~185 at mean ~490 and field-aware collapse to ~60 at mean ~600, Fig 7, Fig 9 PC percentages (checked against the text layer for all nine models), Fig 10 (ideal observer ends ~500 / ~225), Fig 11 (first switch keeps std ~150 for several hundred numbers; later switches show narrow spikes).
- The reading's own flags are correct: Sec 4.2 cites "Fig. 4" for the alpha sweep (it is Fig 6); App D text says N(700,300) while the Fig 11 caption says N(700,100); Fig 9 caption "second layer (L = 1)" vs panel "idx=1"; Fig 2 caption "300 tokens" vs axis in numbers.

## Errors (reading says / paper says / evidence)

1. **Overstated key claim, and the bypass label built on it.**
   - Reading says (key_results[9].claim): "decodable does not mean linearly controllable." Methods[2] adds "a messy or null result from straight-line steering is not proof that the information is causally inert." `bears_on_fates` 2 bypass is rated "moderate".
   - Paper says (p9, quote the reading itself uses): "Linear separability does not entail geometrically faithful control". The claim is about side effects on a coupled covariate, not about failure to control the target.
   - Evidence: in Fig 5B and Fig 6 the linear steering paths DO move the mean from 300 to ~700; only the std is distorted (peaks ~165 and ~185). The paper has no null or inert case, so it gives no direct support for "a null from straight-line steering is not proof of inertness". Fix: reword the claim to "linear steering moves the target but distorts coupled covariates"; downgrade bypass to "weak" (analogy only) and keep the manifold-steering control as our own design advice, not a paper finding.

2. **"Manifolds are visible at most layers" (architecture_location) is unsupported.**
   - Paper shows manifold PCA at layer 14 (Fig 1B, Fig 11C) and at only three depths per model in Fig 9 (idx 1, L-2, L-1). Separability (Fig 3A) is shown at all 16 layers, but separability is not a visible manifold.
   - Fig 9 at idx=1: 6 of 9 models are a mixed blob (LFM2 x3, Llama-3.1-8B, Llama-3.2-3B, OLMo-2-0425-1B); OLMo-2-1124-7B and OLMo-3 are clearly structured at idx=1; Llama-3.2-1B is partly structured. The reading's Fig 9 takeaway ("at layer 1 it is mostly mixed") is fair, but then "most layers" contradicts it. Fix: "visible at the three depths shown; sharper late; layer-1 mixed in most models".

3. **Transfer result lacks its layer caveat where it carries weight.**
   - Fig 3D is a single binary probe trained on mu={300,350} at layer 0 only (caption says layer 0). The reading's figure entry says "(layer 0)", but `bears_on_fates` 5 transformed ("moderate", "exactly how a probe ... could miss information") and methods[1] ("H3 and 5 transformed") quote the 0.89 -> chance decay with no layer or pair caveat.
   - Layer-0 transfer at a comma token mostly reflects shallow context statistics. Transfer at deeper layers, and for other probe pairs, is not shown. Also this is shift along the task variable (delta-mu), not shift under a context change like pushback, so it supports the H3 test design only by analogy. Fix: add "layer 0, one probe pair" to the strength note; keep "moderate" at most as "design idea", not evidence.

4. **"All interventions are on Llama-3.2-1B only" (setup.models_and_sizes) is inferred, not stated.** Sec 4 and the Fig 5/6 captions never name the model; the paper only says it relies "primarily" on Llama-3.2-1B (p3). Likely true; mark it as inferred.

5. **Pitfall gloss goes beyond the quote (pitfalls[6], p14).** Quote: "even 100 input numbers is insufficient to sketch an underlying distribution with sigma = 100, even though the model's output is faithful to it". Reading's gloss: "outputs are not a clean posterior readout". The paper says the output IS faithful and attributes this to a normality prior ("model believes the input data is Gaussian"). Fix: restate the pitfall as "the model's output can match the target before the data alone supports it, so a strong prior is in play", or drop the "not a clean readout" wording.

6. **Fig 2 numbers and takeaway.**
   - Reading: "crossing path reaches std ~240 near mean ~600". Plot (Fig 2D): the peak is std ~240 at mean ~510-520; a second cluster of ~225 sits at mean ~580-620. Minor.
   - Reading takeaway: activations "show ... a path ... through a high-variance (high-entropy) region". The paper claims high variance for the OUTPUT only (p4); panel C is a PCA arc with no variance measure. Minor wording fix.

7. **Small numeric looseness.** `bears_on_group_angle` says spline/field-aware std "~100-115"; Fig 5B spline reaches ~122 near mean 500 (field-aware in Fig 6 is ~100-113). Say "~100-122".

## Omissions
- **Base models only.** The paper's own limitation says "base models without finetuning". Our target is aligned/instruct models under pushback. The reading says "base" in setup and reuse_verdict but never flags this as a transfer risk for sycophancy fates.
- **Last-layer steering.** Sec 4.1 edits "last-layer representations", which is one step from the unembedding, so the causal test is close to logit editing. This limits the "WHERE" value for mid-network routing. The reading notes "last-layer" but not this consequence.
- **No control or sample sizes for Fig 3A.** Layer-0 accuracy is already 0.875 (9 classes, chance 0.11). No shuffled-label or recent-numbers baseline, and the number of series per class is not reported. The reading says "no seeds, error bars" but not these two.
- **Fig 4 "low-dimensional".** The reading echoes the paper ("readout field is low-dimensional") while its own numbers show L13/L14 need about 5 components for 0.8 of the variance. Fine as paper paraphrase; do not reuse as our claim without a number.
- `actionable_fix` "For monitoring: use a field of local probes" is our inference; the paper proposes only steering schemes. Label it.

## Fate labels, architecture_location, actionable_fix (summary)
- Fates 3 and 6 "none", 1 "weak", 4 "weak", 5 "moderate", 2 "moderate". Recommended: 2 -> weak (item 1), 5 -> weak-to-moderate with caveat (item 3). Group-angle statement ("no principal angles, Grassmannian, Procrustes, speed, curvature measured") is correct: curvature is visual only.
- architecture_location: residual stream at the com2num token is correct; "layer 14" for PCA is correct; fix item 2.
- reuse_verdict "adapt" and relevance "Medium" are reasonable.
- Vocabulary: uses "decodable", no "knows". Good.
