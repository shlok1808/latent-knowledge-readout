# Ideas team, lens: contradictions (2026-10-25)

Sources: all 21 notes in `reading/*.md` (exact quotes and pages checked in the `.json` files for the load-bearing ones), plus `scout_pass2.md`. Items from scout_pass2 (#2, #3, #4, #7) are abstract-level only and have no page.

Every idea below is a **HYPOTHESIS**. None is sealed. The real design choices (model, items, controls) still go through `/decide`, and any collect needs Shlok's bet first.

## In short
- In most of these papers, the choice of method sets the fate label. Edits along a probe direction come out "bypass". Edits written into a subspace come out "not routed". A frozen probe gives "erased", a refit probe gives "decodable", and CKA gives "different".
- Run these two first, because both are cheap and every other idea depends on them:
  - **C7:** is the post-pushback "Canberra" signal really correctness, or is it just the model's own earlier answer sitting in the context?
  - **C2:** a frozen vs refit vs Procrustes test for erased vs transformed.
- C1 is the core idea: a ladder of interventions that shows whether "bypass" belongs to the item or to the tool.

## Contradiction map

| # | Where the papers disagree | What decides the result | Idea |
|---|---|---|---|
| 1 | Probe direction is inert (c9, c7) vs the same information is steerable (c6, c15, w2-nonlinear) | Which edit you try: adding along the probe, ablating the probe, a subspace write, or a nonlinear map | C1 |
| 2 | "Erased" (frozen probe at chance) vs "present but pointed elsewhere" (refit high) | Frozen vs refit probe. CKA cannot see rotations | C2 |
| 3 | Probes say decodable, so bypass; component methods find a competitor, so override | Each family looks only for its own fate. CKA drops under a push applied to a subset of items | C3 |
| 4 | One compliance subspace (scout #2) vs distinct linearly separable modes (scout #3) | Separability of prompt templates vs behaviour-conditioned causal transfer | C4 |
| 5 | Abrupt one-layer fork (c6) vs smooth drift (c8 representation, u3) | Principal angles saturate near 90° in high dimensions; SVD subspaces jitter | C5 |
| 6 | Manifold or Fisher steering beats linear (c5, w2-shape, w2-fishback, w2-info-geo) vs correctness is a linear mean shift (c15) | Each paper uses a different, often unfair, linear baseline | C6 |
| 7 | Decodable from the earliest checkpoints, even in untrained networks (c9, c7) vs "the probe reads the correct answer" | Trivial or copied input features, and labels taken from prompt conditions | C7 |

---

## C1. HYPOTHESIS: "Bypass" is a verdict of the instrument, not a property of the item. Test it with an intervention ladder.

**Idea.** On the same flipped items and at the same site, apply a fixed ladder of edits matched on off-target KL:
- R1: add along the probe direction
- R2: ablate the probe subspace
- R3: a KAPPA write into the prediction subspace
- R4: a pullback-Fisher step G⁺q
- R5: a nonlinear invertible interchange

An item's fate is the lowest rung that restores the correct answer. We then report how often an item's fate label changes from rung to rung.

**Why promising (the disagreement):**
- Adding along the probe direction does nothing (it is "null-equivalent in 43 of 48 model×checkpoint cells") (c9-lagged-coupling p1). Per c9 p11, a probe is "an excellent sensor ... and a useless lever".
- Ablating the probe costs 0.6 pp. Ablating an optimised subspace at the same layer takes accuracy from 42% to 0% (c7-readout-mediator-angle p5). c7 p4 adds that orthogonality "does not imply causal inertness".
- Yet writing the decodable answer into the *prediction* subspace lifts Llama-8B TruthfulQA from 56.7 to 73.5 (c6-kappa p7). The prediction subspace sits 65-70° from W_U, so this is not a logit swap (c6 p8).
- Steering along the correctness direction moves the error rate by 2.8-10 pp against random and orthogonal controls (c15-confidence-manifold p5).
- In the same pipeline, a map restricted to linear scores 0.010/0.072, while the nonlinear map scores 0.813/0.760 (w2-nonlinear-interventions p4).
- Truthfulness is the concept least like bypass: ablating shared nodes flips the token in 67% of cases, and even probe-exclusive nodes flip it in 25% (u5-user p19). So the prior for factual pushback is "not cleanly bypassed".
- Prediction tool: ½δhᵀGδh predicts the actual KL change (median ratio 1.000; w2-info-geometry-shared p4). vᵀGv on the probe direction is therefore a free forecast of whether R1 can work.

**Smallest first experiment.**
- **Setup:** Gemma-2-9B-it on the 400 TriviaQA pushback items (2604.19117). Keep flipped items where a refit probe is right.
- **Site:** one site, at about 55-65% depth (the KAPPA fork band, c6 p21), last prompt token.
- **First pass:** R1-R3 only (the KAPPA code is public, c6 p1).
- **Controls:** for each rung, 20 random directions or subspaces matched in norm and rank, a dose grid, and a per-unit noise floor (c9 p10). Off-target KL on 200 held-out neutral prompts, because w2-info-geo measures disturbance on its own reference prompts (p78).
- **Mediation, kept separate:** ablate the same subspace on *resisted* items. If they then flip, the model routes through it on its own (endogenous use). That is a different claim from "R3 can steer it" (controllability). c6 never runs this ablation.

**Kill.** If per-item fate labels agree across R1-R3 for at least 80% of items (Cohen's κ ≥ 0.6), the instrument does not decide the fate. Keep R1 plus its null and drop the ladder. The idea also dies if R3 is non-specific, meaning random subspaces of the same rank rescue flips as often.

**Cost.**
- R1-R3: about 3-6 A100-hours at 9B.
- R4 on a 100-item subset: about 10 GPU-hours (about 280 partial forwards per case, w2-fishback p15).
- R5: 1-2 GPU-hours per site.

**Scoop risk.** Medium-high. "Decodability is not use" is crowded (ICLR 2027 submissions; 2610.21708). None of the 21 papers runs this cross-instrument ladder on pressure flips.

**Where and what fix.**
- If R3 works only in a narrow depth band, the fate-matched Stage B fix is prediction-subspace alignment in that band, compared against DPO (H2).
- If only R5 works, the information is encoded nonlinearly (fate 5), and a linear fix or monitor will miss it (H3).

---

## C2. HYPOTHESIS: "Erased" and "transformed" are artefacts of reusing a frozen probe. They are real only when refit and Procrustes-aligned probes disagree with the frozen one.

**Idea.** At each layer, score the pushback activations four ways:
1. a frozen no-pressure probe;
2. a refit probe;
3. the frozen probe after an orthogonal Procrustes map, fit on *resisted* items only (neutral → pushback);
4. the sign of (frozen AUC − 0.5).

Fate rule:
- erased = the refit probe is at chance;
- transformed = refit high, frozen low, and Procrustes recovers it;
- if the map fitted on resisted items also fixes flipped items, the rotation is a property of the pushback context, not of the failure.

**Why promising (the disagreement):**
- "A frozen diagnostic would read re-encoding as erasure", and the subspace rotates by 20-77° (c8-represented-used p13).
- A TruthfulQA probe transfers at only 0.39-0.66, and "the failure is one of orientation, not of absence" (c15 p6). A below-chance score is "a direction that is present but pointed elsewhere" (c15 p7).
- Probes transfer only locally (w2-shape-of-beliefs p5).
- The one erasure claim in round 2 (2607.00415) had a below-chance probe.
- CKA is blind to rotations and reflections (c13-cka-reliability p3). It collapses under a subset translation while separability stays above 90% (c13 p7). So CKA cannot make this call.
- Euclidean geometry is not fixed by behaviour (w2-info-geometry-shared p1), and a cross-model Procrustes transplant fails, rescuing 2.9% (c7 p17). Do everything within one model.
- Polymorphism Is Rotation: one Procrustes fit restores alignment across seeds (scout #7, abstract only).

**Smallest first experiment.**
- **Setup:** Gemma-2-9B-it, 400 items, neutral vs pushback contexts. Cache every layer at the decision position. Everything after caching runs on CPU.
- **Extra check:** c15's own-generation check. Teacher-forced AUC was 0.957 and fell to 0.688 on the model's own answers (c15 p26). Fit on the model's own answers.
- **Causal follow-up:** a twin exchange inside the rotated coordinates (c8 p13), to see whether the re-encoded information is used.

**Kill.**
- If frozen AUC is within 0.03 of refit AUC at every layer for flipped items, there is no re-encoding: fate 5 is empty under pushback, and H3 dies for this setup.
- If the Procrustes map fitted on resisted items brings flipped items to within 0.03 of refit, "transformed" is context-wide. H3 then reduces to a monitor rule ("refit per context"), not a mechanism.

**Cost.** About 1 A100-hour of caching, then CPU. This is the cheapest idea.

**Scoop risk.** Low-medium. c8 does frozen vs refit for post-training, not inference pressure. "Rewriting or Reweighting?" (scout #1, 2608.01835, HIGH) tracks geometry through post-training. The pushback, inference-time version was not found.

**Where and what fix.**
- Report the layer where the rotation angle jumps.
- Monitor rule for Stage A/B: never call a failure "erased" without the refit probe and the Procrustes probe.
- After a Stage B fix, recheck R (frozen and refit) and U separately (c8).

---

## C3. HYPOTHESIS: Decide override vs a lost read from how the flip is shaped (slope and intercept, plus a mean-shift/subspace split), not from which tool found it.

**Idea.**
- For each item, regress the pushback log-odds (correct vs claimed answer) on the no-pressure log-odds, for flipped and resisted items matched on prior confidence.
- **Override:** the slope stays near 1 and the intercept is large and negative. In the group view, the pushback displacement Δh lies mostly along a user-claim direction, and the correct-answer coordinates do not change.
- **Lost read:** the slope drops toward 0, the correct-answer coordinates stay put, and their output sensitivity vᵀGv falls.

**Why promising (the disagreement):**
- Probe papers find "still decodable" and lean toward bypass (c9 p1; c7 p5).
- Papers that look for a competitor find override:
  - c6 hypothesises that knowledge is overridden by "a preference for hasty answering" but never tests it (c6-kappa p4);
  - in c9's prior arm, "the prior, not the probe direction, wins" (c9 p10);
  - IOI shows competing head classes acting on one logit difference (c12-tigges p5, p8).
- CKA's subset-translation theorem: pushing only the flipped items is a subset translation. It lowers CKA while separability holds (c13 p4-5, p7). A neutral-vs-pushback CKA drop is therefore just what override predicts, and is not evidence of erasure.
- c8 gives the discriminator without probes: slope γ and a history-independent intercept (c8 Fig 5 p8; Fig 7 p15; seed shifts Fig 15 p21).
- Per-item differencing before PCA, so the top PCs are not item content (c14-mags, Remark 1, p5).

**Smallest first experiment.**
- **Step 1, from logits only (free):** compute slope and intercept per pressure.
- **Step 2:** split Δh at three layers into three parts: the shift of the group mean (tracked separately, per c13), the change in correct-subspace coordinates, and the change in the user-claim subspace.
- **Step 3:** centre flipped and resisted items separately and recompute CKA. If the drop disappears, it was a group shift.
- **Setup:** Gemma-2-9B-it, 400 items, both pressures.

**Kill.** Suppose flipped and resisted items share the same slope and intercept, and most flips have a flip-margin z ≤ 1 with small prior margins. (z is a necessary condition for a flip: w2-info-geometry-shared p72.) Then the flips are near-threshold items (fate 1), and neither the override nor the lost-read story is needed.

**Cost.** Free on logits; about 2 A100-hours for activations.

**Scoop risk.** Medium. Two papers have these framings without this discriminator, as far as their abstracts show:
- "When Truth Is Overridden" (2508.02087, read in round 2) frames override;
- Signal Competition (scout #2) frames suppression.

**Where and what fix.**
- **Override:** a conditional projection removal at the heads that read the user's claim (c14 method). Candidate heads come from Genadi et al. (scout #4, abstract only).
- **Lost read:** a C1-style R3 write.
- These are two different Stage B arms to set against DPO.

---

## C4. HYPOTHESIS: The H1 dispute ("one compliance subspace" vs "distinct modes") comes from prompt-template separability. Behaviour-conditioned causal transfer settles it.

**Idea.**
- For each pressure, fit the flip-vs-resist subspace under the same prompt, on items matched on prior confidence. This removes the template signal.
- Compare the two pressures by:
  1. sum cos² / principal angles, against matched-random nulls and a split-half ceiling;
  2. a cross-ablation matrix: the pushback subspace applied to tempting-option items, and vice versa;
  3. the same comparison in G-metric coordinates.

**Why promising (the disagreement):**
- Signal Competition: sycophancy and conformity share one compliance subspace (scout #2, abstract only).
- Gotta Catch Them All: three sycophancy modes are perfectly linearly separable from layer 14, with distinct attention circuits (scout #3, abstract only).
- Why separability is weak evidence: labels taken from prompt conditions give 90-96% probes (c11-trajectory-signatures p9). The "deceptive" item was just the persona-prefixed prompt (c11 p4).
- Why overlap is weak evidence:
  - low-rank spans are near-orthogonal by default (c6 p19; c7 p4);
  - late-layer anisotropy inflates CKA (c6 Fig 9 p22);
  - CKA's null here is 0.316 (w2-info-geometry-shared p84).
- Task subspaces mostly fail to transfer (c6 p8). Dataset directions are near-orthogonal (|cos| 0.08), yet "the abstract structure" is shared (c15 p7).
- Output geometry agrees across models more than activation geometry does: 0.88 vs 0.62 (w2-info-geometry-shared Fig 2, p7).

**Smallest first experiment.**
- Gemma-2-9B-it, 400 items × 2 pressures. We need at least 100 flips per pressure; enlarge the item set if flips are rarer.
- Split-half ceilings within each pressure, bootstrapped. u2-subtrackpp p2 warns that single-snapshot SVD subspaces are noisy.
- If there is signal, repeat at Gemma-2-27B to check whether it sits at a stable relative depth.

**Kill.**
- H1 dies if the flip subspaces are within the split-half ceiling of each other and cross-ablation recovers at least 80% of the within-pressure effect. That would mean one subspace and one fix.
- If the split-half ceiling itself is low, the result is uninterpretable. Add items before concluding.

**Cost.** About 3-5 A100-hours at 9B, and about 2x that at 27B.

**Scoop risk.** Medium. Both scout papers address H1. A behaviour-conditioned causal transfer was not seen in either.

**Where and what fix.**
- Shared subspace: one fix suffices.
- Distinct subspaces: pressure-matched fixes (H2).
- Add c12's class-level path-patching share for "user-claim heads" across pressures. Compare head classes, not head indices (c12 p7).

---

## C5. HYPOTHESIS: The "one-layer fork" between decodable knowledge and action is partly a ceiling effect of the angle metric. The causal fork is a band.

**Idea.**
- Locate the layer × token fork between flipped and resisted items with three metrics side by side, each with a bootstrap noise floor and a split-half ceiling:
  - mean principal angle;
  - sum cos² with an exact Haar null;
  - angle in the G-metric.
- Then patch resisted activations into flipped items, matched on fact and confidence, at each layer and token. The causal fork is where patching stops rescuing.

**Why promising (the disagreement):**
- c6 reports the knowledge-prediction angle jumping within about one layer, from about 50-67° to 84-88° (c6-kappa Fig 8, p21). But c6 itself says random low-rank spans in about 4000 dimensions are near-orthogonal by default (p19), and c7 says the same (p4). The metric has no headroom near 90°.
- SVD snapshots jitter, and rank-1 smoothing hides jumps (u2-subtrackpp p2, p9).
- Phase breakpoints move 1-3 layers with k (w2-readout-subspace-grassmann p6).
- Behaviour can switch at a threshold while the representation varies smoothly (c8 p6).
- Smooth movement plus a margin produces abrupt-looking flips (w2-info-geometry-shared p72).
- In multi-turn tasks the gap opens gradually (u3-geometry-of-divergence Fig 3, p4).
- Sublayer sampling (resid_mid vs resid_post) can credit the fork to attention or to MLP (c10-curved-inference-ii, method 1).

**Smallest first experiment.**
- Gemma-2-9B-it: 150 flipped and 150 resisted items, matched on confidence.
- 3 positions: end of the pushback, start of the answer, first answer token.
- All 42 layers, sampled at both sublayers.

**Kill.** If the three metrics disagree on the fork layer by more than 3 layers and patching declines smoothly over at least 6 layers, drop "fork point" as a WHERE claim and report a band. An abrupt fork survives only if patching falls from above 70% to below 20% within 2 layers, at the layer all three metrics agree on.

**Cost.** The geometry runs on CPU. Patching is about 38k forward passes, roughly 2-3 A100-hours.

**Scoop risk.** Low-medium. c6 reports a benchmark-level jump without a causal test. "Road Not Taken" (scout #14) finds forks in chain-of-thought tokens, not pressure flips.

**Where and what fix.** We get the relative depth and component type (attention vs MLP) of the causal fork. Intervene one layer before it. Check that the relative depth holds at 27B, and do not assume a fixed depth across models: c11 found the best probe depth varies 12-fold (p8).

---

## C6. HYPOTHESIS: The "manifold beats linear" result is mostly a baseline artefact. With a fair linear baseline, the output metric matters more than curvature for flipping answers back.

**Idea.** Compare four edits on flipped items at a matched flip-back rate:
- (a) diff-in-means confined to the same k-dimensional subspace, keeping the rest of the residual;
- (b) a pullback-Fisher step;
- (c) a manifold spline through resisted centroids, binned by confidence × pushback strength;
- (d) an i-ResNet map.

Score off-target KL on held-out prompts and accuracy on unpressured controls, at 35%, 55% and 75% depth.

**Why promising (the disagreement):**
- Manifold steering is reported to beat linear by 2.8× in naturalness (c5-manifold-steering p8). But its linear baseline replaces the whole residual: "the entire residual-stream activation is replaced" (c5 p28).
- c5 also notes that two answers collapse the manifold to a line (c5, relevance section).
- Linear steering inflates output std, but this was measured at the last layer of a 1B model (w2-shape-of-beliefs Figs 5-6, pp7-8).
- A Fisher-metric *linear* step beats a Euclidean one by 1.4-6.5× (w2-fishback p1), but the gain disappears at about 75% depth (w2-fishback p17).
- Natural-gradient anti-sycophancy steering gives 8-74× lower off-target KL, on 12 prompts at ≤1.8B (w2-info-geometry-shared p11).
- The linear-restricted map fails outright, but may be under-tuned (w2-nonlinear p4).
- Correctness itself is a unimodal mean shift in 2-8 dimensions (c15 p4).

**Smallest first experiment.** Run (a) and (c) first on Gemma-2-9B-it flipped items. Both are cheap and use c5's public code. Run (b) on a 100-item subset.

**Kill.** If (c) beats both (a) and (b) by at least 2× in off-target KL at matched flip-back on held-out prompts, the geometry of pressure flips is really curved, and a manifold-based Stage B fix is justified.

**Cost.** (a) and (c): under 2 GPU-hours. (b): about 10 GPU-hours. (d): 1-2 GPU-hours per site.

**Scoop risk.** Low-medium. c5's future work names sycophancy (p16), so the Goodfire group could do it.

**Where and what fix.** This picks the type and depth of the inference-time Stage B arm. Per w2-fishback, metric-aware steering only pays before about 75% depth.

---

## C7. HYPOTHESIS: Part of the post-pushback "the probe still reads Canberra" signal is the model's own earlier answer sitting in the context, not correctness. A 2×2 design separates the two.

**Idea.**
- After pushback, both "Canberra" (the model's first answer) and "Sydney" (the user's claim) are literally in the context. A correct-answer probe at the decision position can therefore read context presence or copying. If so, "decodable but flipped" is guaranteed by the prompt.
- Cross the two factors: first-turn answer correct or wrong (wrong answers prefilled) × user claim correct or wrong.
- Fit probes for truth, "what I said" and "what the user said" on cells where those three are decorrelated, and test whether the cell centroids add up.

**Why promising (the disagreement):**
- Decodability saturates on trivial features. Probes reach AUROC ≥ 0.99 from step 1k, even at 160M (c9 p7), and the paper names trivial saturation as the field's weakness (c9 p2).
- An untrained network already gives probe R² 0.956 (c7 p8).
- Labels taken from prompt conditions make probes detect the persona (c11 p4, p9).
- Teacher-forced vs own-answer AUC: 0.957 vs 0.688 (c15 p26).
- Filler-role binding (u4-role-learning, fate-5 row) makes the alternative concrete: "Canberra" can be present but bound to "what I said before" rather than to "what is true".
- Frozen-attention attribution cannot see which answer a head copies (u5 p17).

**Smallest first experiment.** Gemma-2-9B-it, 400 items × 4 cells. Cache every layer at the decision position, then run probes and an additivity test (an interaction term means the answer has been re-bound to a different role) on CPU.

**Kill (of the threat).** If a truth probe trained on decorrelated cells keeps AUC ≥ 0.8 at mid-depth, correctness is decodable beyond copying, and the premise of Stage A holds. If it falls to ≤ 0.6 while the "what I said" probe stays ≥ 0.9, then Stage A's "decodable" calls are mostly copying. Redesign before any collect: for example, probe only items where the user's claim comes before any model answer, or rely more on the tempting-option format.

**Cost.** About 1-2 A100-hours.

**Scoop risk.** Low. It is a control, not a headline, but it protects C1-C6.

**Where and what fix.** If copying dominates, the mechanism is QK routing: heads copying the model's own answer vs the user's claim. Adapt c12's copy score and use the doubt-attending heads from scout #4. The fix target becomes those heads, not the residual readout.

---

## Ranking (value × cheapness)
1. **C7.** It checks the premise; about 1-2 GPU-hours, then CPU.
2. **C2.** Cheapest; it tests H3 directly.
3. **C3.** Nearly free on logits; separates override from a lost read.
4. **C1.** The core of Stage A; moderate cost; high scoop risk.
5. **C5.** Gives the WHERE claim.
6. **C4.** H1, needs enough flips per pressure.
7. **C6.** Choice of the Stage B arm.

## Not verified
- Scout #2, #3, #4 and #7 are judged from abstracts only, with no pages. Read them before citing to Dr. Lin.
- All flip rates and item counts assume the 400-item TriviaQA set gives at least 100 flips per pressure on Gemma-2-9B-it. This has not been checked.
