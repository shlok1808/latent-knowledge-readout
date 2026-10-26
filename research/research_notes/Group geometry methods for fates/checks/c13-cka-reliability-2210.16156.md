# Check: c13-cka-reliability-2210.16156 (Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky, "Reliability of CKA as a Similarity Measure in Deep Learning", arXiv v2, Nov 2022)

Verdict: NEEDS FIXES (minor to moderate). No fabricated quotes or numbers, every section and every figure is covered, and the reader notes are good. The problems are a few over-strong labels, one overstated takeaway (Fig 8), two numbers slightly off, and missing links from the paper's evidence to fates 6 and 3/H3. Not a REDO.

## Quote check
- `quote_check.py`: 28 quotes, 0 failed (it only tests the start of each quote).
- Stricter pass (my own script, scratchpad `c13/qc.py`): all 28 quotes match the full text after whitespace normalisation, and each sits on the page the reading gives (form-feed page split of paper.txt). 28/28 OK, including `limitations_quote`, `limitations_extra` x2 and `future_work_quote` (all p9 or p5).

## Figures vs manifest
- Manifest lists 16 figure/table entries (Fig 1, 2, 4, 5, 6, 7, 8, 10-17 minus 9 and 3, plus Table 1). The reading has an entry for all 16, plus Fig 3 (merged into the Fig 2 caption in the manifest, correctly noted) and Fig 9 (not in manifest). None missing.
- PNGs given: p04, p06, p07, p08, p17-p21. I opened all of them. I also rendered p16 (Fig 9, which the reading could not see) and re-cropped Fig 4 at 300 dpi.
- Skipped sections: none. `sections_read` covers body, Appendix A-C and references (skimmed). The proof of Cor 4 is the one-line "pick v orthogonal to w" argument, which the reading reflects.
- Numbers checked against figure labels or plots and found correct: Fig 2 curves, Fig 4a/4b, Fig 5 (85.9 -> 85.4/85.5/85.4; RBF 85.5/85.4/84.3), Fig 6, Fig 7, Fig 8 (class-9 stream to about PC1 150, PC2 100), Fig 11, Fig 12 (including 81.7 at sigma 0.4), Fig 13, Fig 14 (88.7 -> 88.3/88.5/85.2), Fig 15 (all six bars each side), Fig 16, Fig 17 (66.0/65.7/66.0/65.5, per-corruption range about 38 to 84), Table 1.
- Reader notes that I confirmed as correct: Fig 4b text says "quickly drop to 0" but the whole-class curve plateaus near 0.17 and the single-point curve only reaches about 0.03 at 5000; Fig 2 "high" early CKA is high for generalised vs memorised (1.0, 0.85) but only moderate vs random (0.67, 0.52); Fig 11 originals already resemble the target; Fig 14 minimise costs 3.5 points; Fig 16 random-init probe at chance is unexplained; Fig 17 caption typo.

## Errors (reading says / paper says / evidence)

1. **Fig 8 takeaway says "the same mechanism as the theorem". Overstated.** (figures, Fig 8)
   - Reading says: "The optimiser raised CKA by pushing a subset of one class far out along one direction, the same mechanism as the theorem (emergent, not imposed)."
   - Paper says (p9): "This mechanism of manipulating the CKA aligns with our theoretical analysis. We emphasize, that in this case it is a completely emergent behavior." It does not say "same".
   - Evidence: Thm 1 and Cor 3 are about lowering CKA between X and a shifted copy of X. Fig 8 is a PCA (PC1/PC2 only) of one network that was trained to raise CKA between the first and last layer. Different quantity and opposite direction. The paper does not measure whether the displaced class-9 points moved along a direction that is behaviourally inert.
   - Fix: "consistent with (paper: 'aligns with') outlier-driven CKA; here the displacement raises inter-layer CKA, while Thm 1 lowers X-vs-copy CKA; one network, 2-D view, no inertness test." This matters because the project wants to treat "flipped items shifted along an agree-with-user direction" as a subset translation.

2. **bears_on_fates fate 5 (transformed) is labelled "strong" but part of it is the reading's own inference.**
   - Reading says: strong; "CKA alone cannot diagnose 'transformed'; use Procrustes alignment plus probe transfer".
   - Paper says (p3): CKA is invariant to "orthogonal transformations such as permutations, rotations and reflections and to isotropic scaling". That is a statement about a transform of the whole set. It says nothing about item-specific re-encodings, and the paper's own Thm 1 shows that changing only a subset of items is visible to CKA.
   - Our "transformed" fate is probably item- or condition-specific (the pressure changes the flipped items), so CKA would not be blind to it in general. Also the Procrustes residual after optimal rotation is itself rotation-invariant; a rotation shows up only in the fitted R (or in probe transfer). The paper only calls Procrustes "straightforward to interpret" (p9).
   - Fix: "strong for whole-set rotation/reflection invisibility (p3); moderate for the claim that CKA cannot diagnose item-specific transformed; to detect a rotation read the fitted R or use probe transfer, not the residual. Reading's inference." Same wording point applies to bears_on_group_angle (2).

3. **bears_on_fates fate 4 (override): "CKA ... would drop even if correctness stays decodable" is flagged as inference, but the size of the drop is not guaranteed.**
   - Paper says (Thm 1, p3): the limit as c goes to infinity is Gamma(rho) x ||E_S x||^2 / E||x||^2 x sqrt(PR), with S the untranslated part and rho at most 1/2 (Cor 2 extends rho). That limit is not generally near 0. It is small when rho is small, when the untranslated mean is small relative to the spread, or (in the synthetic case) when PR is large. My back-of-envelope check on the two-cube data (E||x||^2 about 83, ||mean||^2 about 0.3, PR about 1000, rho = 1/2) gives about 0.11, which matches the Fig 4a plateau (about 0.11-0.12).
   - Fix: add "predicted limit = Gamma(rho) ||E_S x||^2 / E||x||^2 sqrt(PR); compute it from our activations before reading a CKA drop as a shift; the finite-c curve (Fig 4a) shows the drop happens already at distance about 5-10 for unit-cube data, but that is scale-specific". Also say Thm 1 holds only in the limit c to infinity.

4. **architecture_location: "about 0.5-1.0 at layers 1-2" is slightly off.**
   - Fig 15 (different seeds) gives generalised vs random 0.57 (L1) and 0.42 (L2); Fig 2 gives 0.67 and 0.52. So the range is about 0.4-1.0.
   - Related small slip: Fig 4a entry says plateau "about 0.1-0.15", but the RBF 0.5 x median curve plateaus near 0.17 (linear and RBF median near 0.11-0.12).
   - Fig 4b single-point curve is not "roughly linear from 0": it stays near 1.0 until about 500, is 0.57 at 1000, 0.12 at 2500, 0.03 at 5000. The two curves cross near 2400. Minor.

5. **Fig 17 takeaway "so the CKA change is not a hidden functional change" is stronger than the evidence.**
   - Paper says (p21): average performance on CIFAR-10C "remains effectively unchanged". The comparison is accuracy only (per-corruption and average). The training objective keeps outputs by distillation on the training set. No per-item agreement, no logit comparison, no probe or patching comparison on the manipulated nets.
   - This matters for Stage B ("removed or only hidden"): the paper supports "accuracy and OOD accuracy preserved", not "mechanism preserved".
   - Fix: "accuracy-level function preserved on CIFAR10 test and CIFAR-10C; internal mechanism not tested."

6. **actionable_fix / relevance: "at most 0.5 accuracy points" is true for linear CKA on the CNN only, and the summary fields do not carry the exceptions.**
   - The exceptions are in the figure entries (RBF 0.8 minimise -1.6; RBF 0.4 minimise 85.9 -> 81.7, -4.2; ViT minimise 88.7 -> 85.2, -3.5) but not in `actionable_fix`, `relevance_to_project` or the headline key result. The ViT is the only transformer, and the paper itself says "minimal loss of performance" for it (p19) even though minimising costs 3.5 points.
   - Also the monitor-design rule says "training can move CKA maps to targets". The paper shows this when CKA is an explicit loss term (Eq 3, with a validation-accuracy-driven lambda). It does not show that ordinary fixes (DPO or a fate-matched fix) decouple CKA from behaviour. Fix: "if a geometry score is in the loss or used for selection it can be gamed; no evidence that ordinary training does it by accident."

## Omissions (things the project would want, missing from the reading)

1. **The other direction of fate 6 (erased) and H3.** The reading only says "low CKA does not mean the information is gone". The paper also shows high CKA with no usable information: Fig 15/16 give CKA 0.87-0.98 between the generalised and memorised nets at layers 1-2 while the memorised net's linear probe is about 10% (chance). And Ding et al. (p3, in `pitfalls`) shows CKA ignores removal of low-variance components even when probing accuracy drops a lot. The reading never links these to erased or to H3 (a monitor or probe misses a failure). If the correctness signal sits in low-variance directions, an erased or transformed failure could leave CKA unchanged. Add to `bears_on_fates` for fates 6 and 5, and to `bears_on_group_angle`.
2. **The subset translations in the experiments were added by hand.** Paper (p7): "the subset translations presented here were added manually to be able to run the experiments in a controlled fashion". The only emergent case in this paper is Fig 8, under an explicit CKA objective; natural occurrence is argued from prior work (Nguyen et al. 2022 block structure, Ding et al. 2021). The reading's pitfall "they can occur naturally and go unnoticed" repeats the paper's claim without noting that this paper does not show it in an ordinarily trained network. Add to `pitfalls` or `limitations_extra`.
3. **Fig 9 content.** The reading says "not read from figure (no page image)". From my render of p16: the map (x = sigma, y = mu, both 1 to 5000, 10 instantiations in the caption, mean only plotted) is near 0 (dark) where mu is comparable to or larger than sigma and about 0.5-0.6 (green) where sigma is clearly larger than mu. The maximum is about 0.6, never near 1, even for a map the paper calls "simple, invertible". So "drops to 0" holds in the mean-dominated region and CKA is moderate elsewhere. A random Gaussian square matrix is also badly conditioned, and the paper itself (p16) says it sees "no reason to completely dismiss" invariance to invertible maps in some contexts. The pitfall "an information-preserving invertible linear map can drive linear CKA to about 0" should say this is regime-specific and uses random Gaussian maps, not near-orthogonal re-encodings.
4. **Depth dependence in Fig 2.** Generalised vs memorised CKA falls to about 0.03 at layer 8 and generalised vs random to about 0.17, so in this CNN CKA does separate the nets in later layers and fails mainly early. `architecture_location` only gives the early-layer part. For "where" claims it is worth saying "CKA's failure here is early-layer specific (CNN, CIFAR10); the same measure separates the nets by layer 7-8".
5. **No seed or error bars for the manipulation results.** Seeds are reported for Fig 2 (5) and Fig 9 (10) only. Figs 5-7 and 11-14 are single networks with differences of 0.2-0.5 points; no repetition is reported in paper.txt. Add to `pitfalls` or `setup.controls`.
6. **Thm 1 conditions.** `key_results` states the limit but omits c to infinity and rho at most 1/2 (Cor 2 extends it). Small, but it feeds error 3.
7. **Internal inconsistency in the paper (low priority).** Appendix B.5 (p19) calls the Fig 5 network a "VGG style model", while Sec 4.2 and A.2 describe a 9-layer CNN; Fig 5 and the VGG x1 target of Fig 11 both show 85.9% but have 8 and 17 layers, so they are different maps. The reading lists them as separate models, which is the safer reading; a one-line note would help.

## What is good (keep)
- Every figure and appendix is covered, including the merged Fig 3 and the missing Fig 9.
- Numbers are labelled "from figure labels" vs "approx, read from figure", and they hold up.
- The relevance statement is honest: a methods warning from a small vision model on CIFAR10, not evidence about LLMs; "our inference, not the paper's" tags are used on fates 4 and 2.
- The `reuse_verdict` (copy the stress tests, pair CKA with probes and Procrustes, do not use CKA alone for transformed vs erased or as a Stage B monitor) is sound, once errors 2 and 6 are applied.

## Suggested edits (short)
1. figures[Fig 8].takeaway: replace "the same mechanism as the theorem" with "consistent with outlier-driven CKA (paper: 'aligns with'); direction differs (raises inter-layer CKA)".
2. bears_on_fates[5 transformed]: "strong" for whole-set invariance, "moderate" for the diagnostic claim; add "Procrustes residual is rotation-invariant too; read R or use probe transfer".
3. bears_on_fates[4 override]: add the limit formula and "only in the limit".
4. bears_on_fates[6 erased] and [5]: add Fig 15/16 (high CKA, chance probe) and Ding et al. low-variance point (H3 link).
5. pitfalls: add "translations added manually (p7)", "Fig 9 is regime-specific", "no seeds for Sec 4.3".
6. figures[Fig 17].takeaway, actionable_fix: "accuracy-level only"; list the minimise exceptions (-1.6, -3.5, -4.2) and "gaming requires CKA in the loss".
7. architecture_location: "about 0.4-1.0"; add the late-layer separation (about 0.03 and 0.17 at layer 8).
