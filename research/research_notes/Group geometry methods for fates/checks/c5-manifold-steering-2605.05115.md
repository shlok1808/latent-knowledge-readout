# Check: c5-manifold-steering-2605.05115 (Manifold Steering, Wurgaft et al., arXiv 2605.05115v1)

Verdict: NEEDS FIXES (minor). No redo needed. All numbers, quotes and pages hold up; 3 claims need rewording, 1 needs a qualifier, and a few ideas are missing.

## 1. Quote check
- `quote_check.py`: 36 quotes, 0 failed. Caveat: the script only tests the first 120 normalized characters and does not test pages.
- Extra page check: I mapped every quote to its page (page-number markers in paper.txt, then `pdftotext` per page for the ones that disagreed). All 36 page numbers are right. Four apparent mismatches (key_results 0, 1, 2, 18, reading says p6) were a stray "6" in figure text; `pdftotext -f 6 -l 6` finds "highly correlated", "mediates the causal" and "teleportation" on p6. The reading is correct.
- Numbers re-checked against the text and figures: r values (0.99/0.89/.999/.999 vs 0.89/0.53/0.71/0.36), E_BC (0.34/0.36/2.42/5.21 vs 0.93/1.09/6.95/13.49, 2.8x), pullback R2 (0.77/0.75/0.47/0.78 vs 0.42/0.32/0.24/0.23), grid r=.99 vs 0.90, cylinder .99 vs 0.81, world model r=0.99 vs 0.06, 2.22/0.20/0.29, 95.4%, 30/30, K=50, 16 and 5 base prompts, 50 pairs, top-64 vs full-residual replacement, 32 of 64 PCA dims in pullback, Table 1 sizes (49/84/48/909). All match.
- Figures: manifest lists 13 (Fig 1-12, Table 1); the reading has all 13 entries, none missing. I opened p02, p04, p05, p07, p08, p11, p13, p26, p32, p33, p34 (re-rendered at 300 dpi for Fig 11), p35. Not opened: p30 (Fig 8, architecture diagram only; caption matches the reading). Figure read-offs ("approx, read from figure") agree with what I see, including the months second band (Fig 2b), the spiky ages traces (Fig 4), the weak (2,0)->(2,4) heatmap contrast (Fig 6), and group 2 of Fig 11 where "other" is about 0.4 and above every concept.
- architecture_location is accurate: layer 28 residual, last token (Llama 3.1 8B); layer 70 (70B, Fig 11 only); world-model encoder output. No component sweep, depth sweep, seeds or checkpoints (grep for seed finds nothing).
- Skipped sections: none. Related Work and Discussion are listed as read, but only a few of their points reach any field (see section 3).

## 2. Errors (reading says / paper says / quote + page)

E1. Model type (matters: sycophancy is mostly an instruct/RLHF behaviour).
- Reading says (setup.models): "No 9B-27B models, no instruction-tuned or chat models."
- Paper says: only "Llama 3.1 8B". It never says base or Instruct, and gives no chat template; prompts are plain templates ("Q: What day is {k} days after {entity}?\nA:", Table 1).
- Quote: "We investigate Llama 3.1 8B (Touvron et al., 2023) activations at layer 28 in bfloat16 for all tasks." (A.2, p26). grep for "instruct" and "chat" finds nothing.
- Fix: "base vs instruct not stated; plain Q/A templates, no chat template; 8B only for LM results."

E2. Pullback pitfall is one-sided (pitfalls[3]).
- Reading says: pullback is "searched inside the manifold's own PCA subspace and initialized at the chord, which partly builds in the result".
- Paper says the opposite constraint too: pullback gets a smaller search space than the comparison paths, and the other 32 PCA components plus the residual are held fixed.
- Quote: "We note that the linear and manifold-steering paths used as comparisons span the full 64-dimensional subspace, so the pullback optimization is operating within a strictly smaller search space." (A.8, p28).
- Fix: keep the build-in concern (it is a fair inference) but add this sentence and mark the concern as the reader's inference.

E3. "Confidence is a manifold axis" is stronger than the evidence (key_results[15], methods[5], bears_on_group_angle).
- Reading says: "Confidence/uncertainty forms an extra manifold dimension at 70B"; "suggests confidence is an extra manifold axis, which supports matching flipped vs resisted items on prior confidence".
- Paper says: the groups are addition values (1-5, 6-10, 11-15, 16-20), so the axis is input difficulty, not a measured per-item confidence. One model, one layer, weekdays only. The caption itself limits ordered transitions to "the first three groups". At 300 dpi: group 0 and 1 intermediate peaks about 0.4-0.58 with "other" about 0.3-0.4; group 2 peaks about 0.2-0.27 with "other" about 0.4 (above every concept); group 3 has no ordered peaks. The 70B weekday increments (1-20) are also larger than Table 1's weekday increments (1-7), and no appendix section describes this dataset.
- Quote: "This suggests manifold geometry can serve as a handle for calibrating model confidence in a controlled fashion." (p34).
- Fix: say "difficulty (addition-value) groups form a second axis; ordered transitions weaken by group 2 and vanish by group 3; per-item confidence not tested". Keep the matching-on-prior-confidence advice, but as the reading's own design suggestion, not a paper result.

E4. Bypass wording (methods[4], bears_on_fates fate 2).
- Reading says: null-space directions of the output Jacobian are "decodable-but-inert by construction".
- Paper says only that the geometry "remains Euclidean in its null space" and that J_F "has rank at most |Z| - 1 << n" (p10). It says nothing about decodability or inertness. The paper's own bound looks off by one for |Z|+1 output bins (concept tokens plus "other"); low priority.
- Fix: "to first order, locally, per item". Add: any random direction in R^n also has a tiny projection on a row space of at most |Z| dimensions, so a bypass test needs a random-direction baseline, and the Jacobian must be taken per item and per layer. Strength "weak" is right.

## 3. Calibration and omissions

Calibration (judgement, not errors)
- bears_on_fates strength "moderate" for fates 3 and 5 is generous given (a) the paper has no failure cases, (b) with two answers the manifold collapses to a path (the reading says this itself), and (c) layer-28 last-token injection is 4 layers from the output, so "inject and see" at that site is close to guaranteed to work and cannot separate fate 3 from others. Fate 3 needs injection at earlier layers or at the claim-token positions. Suggest "weak-to-moderate" with the layer caveat.
- E_BC (key_results[3-5], methods[2]): "natural" is defined against a behaviour manifold built from class-mean outputs on the same prompts, so low E_BC for steering that ends at class centroids is partly by construction. A confident wrong answer is on the manifold, so E_BC cannot flag a sycophantic flip; it flags hedged or mixed outputs only. The reading's text hints at this, but it should say it outright because Stage B plans to use E_BC as a side-effect check. (My inference.)

Omissions worth adding
- O1. Pullback alpha sweep (App C.3, p35-36): at alpha = 0 the recovered path leaves Mh to match an off-manifold target; as alpha grows it tracks Mh. This is a dial between "natural" and "unnatural" states that give the same output, which is relevant to fates 2 and 5. The reading only mentions C.3 for an inconsistency.
- O2. ICLR (Sec 4, Discussion p15): the paper argues the model builds novel geometry from context, with token meanings "wildly different from the meaning learned during training". This is the closest analogue in the paper to user pushback re-defining the answer in context (fates 4 and 5). Not linked in the reading. (My suggestion, not a paper claim.)
- O3. Discussion overclaims not recorded: "we establish a causal bridge between representation and behavior" and that linear-steering pathologies "stem from the mismatch between assumed flat geometry and the true curved geometry ... rather than an inherent challenge" (p15). These sit in tension with the reading's own best catch (the linear baseline overwrites the whole residual vector, A.6 p28). Add one line tying them together. Also p15: "the appropriate units of causal analysis are intrinsic coordinates on manifolds", from templated tasks, one model family.
- O4. No reference-leads field. Leads named in the paper: Park et al. 2026 (called the closest work, toy softmax model, p14); Bigelow et al. 2025 (linear steering is linear in concept log-probability when LRH holds; relevant to the logit-margin first control); Sarfati et al. 2026 (linear steering leaves higher moments unconstrained); Vennemeyer et al. 2025, arXiv 2509.21305, "Sycophancy is not one thing" (cited for sycophancy, p16); Yocum et al. 2025 (activation geometry constrains output); Bethune et al. 2025 (energy-based metric). Only the Vennemeyer ID is printed in the reference list text I checked; other IDs not verified.

## 4. What the reading got right (keep)
- Unfair linear baseline (whole-residual replacement vs top-64 PCA edit), the chord baseline for pullback R2, near-circular world-model isometry (behaviour F built from the activation spline, tau = 0.5), supervised intrinsic coordinates, single late layer, and the C.3 inconsistency are all real and quoted correctly.
- "Causal controllability, not mediation" is kept separate; the paper's "mediates" sentence (p6) is about intervention effect only.
- reuse_verdict "adapt" and relevance "moderate" are fair: no failure cases, no sycophancy experiment, 8B only, sycophancy named only as hypothetical future work (p16).

## 5. Suggested edits (priority order)
1. E1 (setup.models wording). 2. E3 (three places). 3. E2 (add the sentence). 4. E4 (first-order caveat plus random baseline). 5. O1, O3, O4 (cheap additions). 6. Calibrate fate 3 and 5 strength; add the E_BC caution.
