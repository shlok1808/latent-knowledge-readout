# Ideas team, lens: transfers (2026-10-25)

Each idea takes a method that one paper in the reading set developed and applies it where that paper (and the others) never tried it: to per-item flips under pushback or a tempting wrong option, at 9B-27B. Every idea is a **HYPOTHESIS**. Citations use the form `slug p.N` (page numbers from the reading notes and JSON). "Decodable" means a probe reads it, not that the model knows it. Causal controllability (an edit changes behaviour) is kept separate from endogenous mediation (the model routes through it).

Default testbed for every first experiment: **Gemma-2-9B-it, the 400 TriviaQA pairs from 2604.19117**. Each item is run in three conditions: neutral follow-up, user pushback ("I'm sure it's Sydney"), and tempting wrong option. Prompts are length-matched (`c10-curved-inference-ii-2608.24037` p.19 control). Flipped and resisted items are matched on their neutral logit margin. Replication goes to Gemma-2-27B (same family) and Qwen2.5-14B (a second family).

## Recommended order

| # | Idea | Fates it separates | Cost | Scoop |
|---|---|---|---|---|
| 1 | Plant known fates first, then check which metrics call them correctly | all (validates the instruments) | <1 GPU-h | none |
| 2 | Frozen vs refit probes plus paired Procrustes | 2 vs 5 vs 6 | ~1 GPU-h | medium |
| 3 | Pullback-Fisher (G-metric) output sensitivity of the correct-answer subspace | 2 vs 4, and re-tests KAPPA's 90° | ~2 GPU-h | low-medium |
| 4 | Fork-point localisation on the Grassmannian at sublayer resolution | WHERE (attention vs MLP, relative depth) | ~2 GPU-h | medium |
| 5 | Output-level slope/intercept and flip-margin screen across pressures | 1 vs 4 vs 2/3, H1 | ~0 extra | low |
| 6 | Two-root attribution and head classes tracked through Stage B | 4; Stage B removed / moved / masked | ~6-8 GPU-h | medium-high |
| 7 | Manifold vs discrete state on a built pressure × confidence grid | 1 vs 4; manifold vs discrete | ~3 GPU-h | high |
| 8 | Filler-role binding: "transformed" as re-binding | 5 (a specific form) | ~1 GPU-h | low |

---

## 1. Plant known fates in real activations before trusting any group metric (HYPOTHESIS)

**Idea.** We take real 9B activations from resisted items, plant each fate with a known transform, run the rest of the model forward, and score every metric we plan to use on whether it names the planted fate. Our hypothesis is that CKA and RSA call a planted subset shift "changed or lost" and miss a planted rotation, while frozen-vs-refit probes, Procrustes and the G-norm call both correctly.

**Builds on.**
- `c13-cka-reliability-2210.16156`:
  - linear CKA collapses when only a subset of points is translated (theorem p.4), and it does so on real activations while the classes stay more than 90% separable (p.7);
  - CKA is blind to rotations (p.3);
  - the authors' advice is Procrustes (p.9).
- `c8-represented-used-2610.05292`: an invertible map keeps the information but can set the covariance to almost anything (Prop. 1, p.3); the gauge control (p.11).
- `w2-info-geometry-shared-2609.11063`: random invertible reparameterisation, angle-preserving random subspaces (p.3), and the CKA permutation null of 0.316.
- `c15-confidence-manifold-2602.08159`: RLACE as a positive control for erasure (p.5); INLP leaves AUC at 0.72-0.75 after 32 removals (p.25).
- `c6-kappa-2509.23782`: the prediction-vs-random CKA reaches about 0.4-0.48 at late layers (Fig. 9, p.22).
- `c7-readout-mediator-angle-2605.29126`: anisotropy makes the angle null about 14× wider (p.27).
- Polymorphism Is Rotation (2605.24577, scout_pass2 #7): seeds differ by a rotation.

**Why promising.** Each of these papers shows one way a group metric fails, but none checks its metric against known ground truth on LLM activations. Our six-fate calls depend on separating rotation, loss, shift and rescaling. A cheap confusion matrix tells us which metrics are allowed to decide which fates.

**Smallest first experiment.**
- **Input:** pushback-condition activations of about 200 resisted items, Gemma-2-9B-it, at three layers (about 40%, 60% and 80% depth), answer position.
- **Plants**, each applied to a random 30% of items, with the edited state run through the remaining layers:
  1. rotate the correct-answer subspace (PLS-4) by 30° and by 60° (fate 5);
  2. translate the items along the prediction-probe direction (c13 subset translation; an override-like push);
  3. RLACE-erase the correct-answer subspace (fate 6);
  4. rescale it by 0.03 and compensate downstream (c8 gauge: no information change);
  5. apply a random invertible map with κ = 1e3.
- **Compute:**
  - the group metrics: linear CKA, RSA, principal angles (sum cos² against a matched-dimension Haar null), Procrustes residual, frozen and refit probe accuracy, and G-norm (idea 3);
  - the behaviour after each plant (flip rate).
- **Output:** a confusion matrix of planted fate vs called fate, for each metric.

**Kill result.** Any metric that miscalls more than 20% of plants at the planted size is barred from making fate calls. If CKA and RSA call every plant correctly, the c13 worry does not apply at 9B, and they can stay in the H1 comparison.

**Cost.** CPU plus about 5k partial forwards: under 1 GPU-hour.

**Scoop risk.** None. This is a methods control, and it becomes a methods subsection of the paper.

---

## 2. Frozen vs refit probes plus paired Procrustes: is a late drop rotation, loss, or non-use? (HYPOTHESIS)

**Idea.** `c8` built R/U/A (refit decodability, use, variance share) to tell apart erased, rerouted and rescaled information after *training*. We apply it to inference-time *pressure*, and add a paired Procrustes fit: the same item appears under neutral and pushback, so the point correspondences are exact. Our hypothesis is that most flipped items keep the frozen neutral probe above chance at mid-depth (bypass or override). Where accuracy does drop late, a Procrustes-rotated frozen probe recovers it (transformed by rotation), and true loss (fate 6) is rare.

**Builds on.**
- `c8-represented-used-2610.05292`:
  - "A frozen diagnostic would read re-encoding as erasure" (p.13);
  - the kept subspace "rotates by 20 to 77 degrees" (p.13);
  - the twin-exchange use measure, U = 0.15 after SFT while decodability stays at 0.98 (p.5, p.13);
  - the warning that a ridge probe misreads low variance as lost information (p.11).
- `c15-confidence-manifold-2602.08159`: "a below-chance score is the signature of a direction that is present but pointed elsewhere" (p.7); the probe loses 27 points from teacher-forced to own answers (p.26), so fit on own answers.
- `c7-readout-mediator-angle-2605.29126`: Procrustes transplant *across* models fails (2.9% rescue, p.17), so the fit should be within one model; untrained-checkpoint probe floor (p.8).
- `c13-cka-reliability-2210.16156` p.9 (use Procrustes, not CKA); `c6-kappa-2509.23782` p.3 (paired knowledge and prediction probes); `w2-nonlinear-interventions-2605.14749` p.4 (a linear map fails where an invertible non-linear map works).

**Why promising.** Round 2 found that the correct signal usually survives, and that the one paper claiming erasure (2607.00415) has a below-chance probe, which is the c15 signature of a direction pointed elsewhere. No paper runs frozen vs refit on pressure-induced flips, and no paper uses same-item pairing to make Procrustes well-posed at 9B+. The scout gap check (scout_pass2 p.1, "Gap check") found nothing.

**Smallest first experiment.**
- **Input:** Gemma-2-9B-it, 400 items × {neutral, pushback}, answer position, every layer.
- **Probe target:** "which of the two named options is correct", with option order randomised so the label is balanced (KAPPA design). Everything is split by item (GroupKFold).
- **Compute per layer:**
  - R_frozen: train on neutral, test on flipped and on resisted pushback items.
  - R_refit: train on pushback, test on held-out *flipped* items. Accuracy above chance on flipped items cannot come from reading the model's choice, because on those items the choice is wrong.
  - Orthogonal Procrustes from neutral to pushback in a joint top-64 PCA, with each condition centred separately. The mean-shift vector is tracked on its own, following c13.
  - Frozen probe composed with the rotation, and the rotation angle against an angle-preserving random-rotation null and a shuffled-pairing null.
  - Twin-exchange U: swap the correct-answer component between the two option orders of the same item, then read the answer change.
  - A non-linear (MLP) refit probe, plus RLACE and INLP as positive controls for erasure.
- **Calls** (pre-register via a lab-mod sealed bet):
  - frozen high, U low → bypass (2);
  - frozen low or below chance, refit high, rotated frozen probe recovers → transformed by rotation (5);
  - only the non-linear refit recovers → transformed non-linearly (5);
  - refit and non-linear refit both at the untrained floor → erased (6).

**Kill result.** If frozen accuracy on flipped items stays within 5 points of neutral accuracy at every layer, there is no late drop to explain. Fates 5 and 6 are then empty at 9B, and the Procrustes arm is dropped.

**Cost.** Forward passes for 800 prompts (minutes); twin exchange is about 800 more forwards. About 1 GPU-hour in total.

**Scoop risk.** Medium. The c8 authors (4 Oct 2026) could extend to inference. The scout's top risk, Rewriting or Reweighting? (2608.01835), works on post-training, not inference-time flips.

---

## 3. Pullback-Fisher output sensitivity of the correct-answer subspace, and KAPPA's 90° re-measured in the G-metric (HYPOTHESIS)

**Idea.** FishBack and 2609.11063 use G = JᵀHJ (the output Fisher pulled back to layer l) to steer with less collateral change. Neither applies it to probe directions. We use it to ask, per item, whether the decodable correct-answer subspace is one the output reads at all. Our hypothesis is that KAPPA's late-layer near-orthogonality between the knowledge and prediction subspaces is partly a coordinate artefact. In the G-metric, flipped items will split into two groups:
- (a) knowledge directions with near-null G-norm: bypass;
- (b) knowledge directions that are read by the output, but where the pushback displacement has a larger opposing component along the answer covector: override.

**Builds on.**
- `w2-fishback-2605.17231`:
  - G = JᵀHJ and the covector q = Jᵀβ;
  - "Cosine similarity to the optimal direction is therefore the wrong diagnostic" (p.18);
  - the rank-50 effective-rank artefact (p.6);
  - the entropy-selected contexts (p.16);
  - the gain collapses at about 75% depth on 8B models (p.15, p.17).
- `w2-info-geometry-shared-2609.11063`:
  - coordinates are arbitrary up to an invertible linear map (p.1);
  - one half δhᵀGδh predicts KL with a median ratio of 1.000 (p.4);
  - the advantage shrinks toward the read-out (p.13);
  - bf16 and fp32 directions disagree, so the solves must be fp32.
- `c6-kappa-2509.23782`: the knowledge-prediction angle reaches about 87° at decision layers (Fig. 8, p.21; p.20).
- `c7-readout-mediator-angle-2605.29126`: a probe at about 88° from the load-bearing subspace still sits at the Haar null (p.1, p.5).
- `c9-lagged-coupling-2609.01048`: headroom and write-in c = e(1)/C, matched-norm random nulls and a noise floor (p.9-10).
- `c10-curved-inference-ii-2608.24037`: G = UᵀU as a rough lens (our extension, p.3 metric).

**Why promising.** The bypass fate needs a test of "decodable but not read", and a Euclidean angle cannot give it (2609.11063 p.1). vᵀGv = Var_p(Jv) needs only one JVP per direction, so it is cheap. It also makes "not routed" a falsifiable claim about KAPPA's geometry rather than an angle on a coordinate chart.

**Smallest first experiment.**
- **Input:** about 200 flipped and 200 resisted pushback items, matched on the neutral margin *and* on next-token entropy (FishBack p.16 caveat), at three layers (about 45%, 60% and 75% depth), answer position, fp32.
- **Compute:**
  - For the correct-answer basis U (PLS-4), the prediction basis P (KAPPA), 20 random bases and q = Jᵀ(w_correct − w_user): the G-Gram matrices (Ju_i)ᵀH(Ju_j), each from one JVP per basis vector.
  - The G-norm fraction of U against the random null; principal angles between U and P in the G-metric; and the G-inner product of the pushback displacement Δh with q, and with U's G-projection.
  - For 50 items, a check with FishBack's rank-50 top eigenspace, with repeat-seed stability.
- **Reading:**
  - U's G-norm at the random null, plus c9 matched-null steering z ≈ 0 → bypass (2);
  - U's G-norm high and aligned with q, while Δh·q is strongly negative → override (4);
  - U's G-norm low where it is decodable but high one band earlier → the signal was not carried forward (3), to be confirmed by KAPPA-style injection;
  - a G-angle between U and P much smaller than the Euclidean 87° → KAPPA's "routed along geometrically distinct directions" (c6 p.5) is a chart artefact.

**Kill result.** If U's G-norm does not differ between flipped and resisted items (bootstrap CI) at any of the three layers, and the G-angle tracks the Euclidean angle, then output-sensitivity does not separate fates at the answer position. The G arm is then dropped.

**Cost.** About 400 items × 3 layers × about 30 JVPs, roughly 36k partial forwards, plus the rank-50 check on 50 items: about 2 GPU-hours.

**Scoop risk.** Low to medium. The FishBack and Picozzi lines are steering papers. Picozzi lists sycophancy audits as a use (p.15) but runs no per-item flips.

---

## 4. Fork-point localisation on the Grassmannian at sublayer resolution (HYPOTHESIS)

**Idea.** KAPPA found that the knowledge-prediction angle *jumps* within about one layer at the depth where the choice crystallises, but only per benchmark and without pressure. 2605.09011 and 2605.12714 track subspaces across depth with breakpoints and curvature, but with no condition or outcome contrast. We track per-condition and per-outcome subspaces after attention and after MLP. Our hypothesis is that under pushback the jump comes earlier for flipped than for resisted items, and that it first appears after an *attention* sublayer, at a relative depth stable across Gemma-2-9B and 27B. For the tempting wrong option, it first appears after an MLP sublayer (H1).

**Builds on.**
- `c6-kappa-2509.23782`: the one-layer jump from about 50-67° to about 84-88° (Fig. 8, p.21) at 55-65% relative depth.
- `w2-readout-subspace-grassmann-2605.09011`:
  - the RSS (principal-angle mean cosine) and Kneedle breakpoints (p.6);
  - the tuned lens reads the answer early while the logit lens stays blind (p.2, p.9);
  - Phase 3 is FFN-dominated and starts at 60-89% depth (p.8).
- `w2-layerwise-grassmann-dynamics-2605.12714`:
  - Grassmann speed and Menger curvature;
  - Random and last-k layer baselines (p.9);
  - GFMI against the behaviour label;
  - decoder models bend late, at 0.78-0.89 of depth (p.5).
- `u2-subtrackpp-2502.01586`: single-snapshot SVD subspaces are noisy (p.2), and rank-1 smoothing hides jumps (p.9).
- `c10-curved-inference-ii-2608.24037`: double-resolution sampling (p.3).
- `c11-trajectory-signatures-deception-acl2026`: DTW and near-tie decision points; the best depth varies 12-fold across models (p.8).
- `u3-geometry-of-divergence-2608.30650`: step-cosine curvature, which needs a null (about -0.5 for i.i.d. jitter).
- `u5-user-2608.27510`: a steepest-rise "belief point" (p.18).

**Why promising.** WHERE needs a component type and a relative depth. A fork found after attention points to heads reading the user's claim (fix: attention-level). A fork after MLP points to re-encoding (fix: late-MLP-targeted). The c6 jump gives a sharp prior to test against, and the tools are cheap once activations exist.

**Smallest first experiment.**
- **Input:** Gemma-2-9B-it, `resid_mid` and `resid_post` at every layer (85 points), answer position. Flipped vs resisted items, matched on confidence, under pushback and under the tempting option.
- **Compute per point:**
  - the knowledge subspace K (PLS-4) and the prediction subspace P (KAPPA probes), fit separately for each outcome group;
  - sum cos² of K vs P against a matched-dimension Haar null, with a split-half ceiling and bootstrap bands (u2 warning: no smoothing);
  - Kneedle breakpoints, and Grassmann speed and curvature of K along depth;
  - per-item tuned-lens rank of the correct vs the claimed token, which gives a per-item fork layer;
  - GFMI with the flip label.
- **Causal check:** patch resisted → flipped at the fork, at fork ± 3, and at Random and last-k layer baselines (2605.12714).
- **Token axis:** step-cosine and DTW across the assistant reply in K-coordinates, with a jitter null.
- **Repeat** on 27B: does the fork stay at the same relative depth?

**Kill result.** The idea fails if any of the following holds:
- no point shows a flipped-vs-resisted difference in the K-P angle beyond the split-half floor;
- patching at the fork does no better than the Random-layer baseline;
- the fork depth differs by more than 10% of relative depth between 9B and 27B.

In any of these cases the WHERE claim is dropped and only the descriptive profile is reported.

**Cost.** About 2.4k prompts × 85 hook points at one position (about 3 GB fp16), CPU for the geometry, and about 1 GPU-hour for patching. About 2 GPU-hours in total.

**Scoop risk.** Medium. Gotta Catch Them All (2607.20146, scout #3) separates sycophancy modes by layer and circuit but has no fork geometry. Signal Competition (2601.11563, scout #2) predicts one shared subspace, which is the opposite of H1, so cite it and test against it.

---

## 5. Output-level slope/intercept and flip-margin screen across pressures (HYPOTHESIS)

**Idea.** `c8` reads the fate from a regression of per-item policy log-odds on reference log-odds (a kept slope with a large negative intercept means a constant push; a slope near 0 means non-use). We apply this to pressure: the pushback margin regressed on the neutral margin, and the output margin regressed on the decoded internal margin. 2609.11063's flip-margin bound (a flip needs z = m/2ρ ≤ 1) adds an output-only screen for threshold flips. Our hypothesis is that pushback gives slope ≈ 1 with a negative intercept (a constant shove, so flips concentrate at small neutral margins: fates 1/4), while the tempting option gives a flattened slope (the output stops tracking item evidence: fates 2/3). That would be H1 at zero cost.

**Builds on.**
- `c8-represented-used-2610.05292`: the slope γ and intercept against β (Fig. 7, p.15); "the fate switches at a threshold in β ... whereas Rinv and A vary smoothly" (p.6).
- `w2-info-geometry-shared-2609.11063`: flip frequency falls from 0.519 to 0 as the normalised margin crosses 1 (p.72).
- `c9-lagged-coupling-2609.01048`: the logit-margin AUROC as a probe-free track (p.7).
- `c15-confidence-manifold-2602.08159`: output methods fall to 0.44-0.64 AUC on confidently wrong items (p.6).
- `c6-kappa-2509.23782`: two clusters in the knowledge-vs-prediction scatter (Fig. 3, p.5).

**Why promising.** It needs only the logits that Stage A already collects, plus one probe readout. It gives a population-level fate prior and an H1 test before any geometry. It also serves as the logit-margin baseline that every group metric must beat: at 8B a probe beats the margin by only +0.008 AUC (2609.04582, from round 2).

**Smallest first experiment.**
- **Fits:** m_push = γ·m_neutral + b, and the same for the tempting option. Fit pooled, then as a 2-component mixture regression (are there two populations?). Then fit m_push against the PLS-decoded internal margin at the best layer.
- **Screen:** z for every flip.
- **Reading:**
  - γ ≈ 1 with b < 0, and flips at z ≤ 1 with small m_neutral → threshold/override boundary (1/4);
  - γ → 0 while the internal margin stays high → bypass or not routed (2/3), passed to ideas 2-3;
  - a mixture with distinct (γ, b) components → several fates, so per-item sorting is required.

**Kill result.** If pushback and tempting-option fits share (γ, b) within bootstrap CIs and the mixture adds nothing (BIC), H1 gets no support at the output level. That alone does not kill H1, which may still hold internally.

**Cost.** Zero GPU beyond collect.

**Scoop risk.** Low.

---

## 6. Two-root attribution and head classes tracked through Stage B: removed, moved, or masked? (HYPOTHESIS)

**Idea.** `u5` roots one attribution graph at a probe score and another at the next-token logit, and finds shared nodes for truthfulness. `c12` shows that a function is stable as a *class* of heads while individual heads come and go, and that self-repair hides removals. We root one graph at the correct-answer subspace score and another at the log-odds (correct − user claim), on flipped vs resisted items. The shared and logit-exclusive nodes define head classes that read the user's claim. We then re-find the classes *on each fixed model* after DPO and after the fate-matched fix. Our hypothesis is that after DPO the user-claim class keeps its path-patching share, but a downstream component cancels it (masked, as 2604.19117 saw in Llama-70B). After a fate-matched head-level fix, the share falls (removed) (H2).

**Builds on.**
- `u5-user-2608.27510`: for truthfulness, ablating the shared nodes drops the probe score by 93% and flips 67% of tokens (p.19); attention is frozen in their graphs (p.17), so add patching.
- `c12-tigges-circuits-across-training-2407.10827`:
  - "models can clearly compensate for losses of and changes in individual circuit components" (p.7);
  - class shares are stable (p.8);
  - more heads take on the same role (p.23);
  - two-criterion head inclusion.
- `c14-mags-2605.21770`: per-head flip-vs-resist AUROC maps (p.5), used with both tails; conditional projection as a cheap fate-4 fix.
- Genadi et al. EACL 2026 (scout #4): mid-layer heads attend to doubt tokens.
- 2604.19117 (round 2): after RLHF, the heads' ablation effect grew while sycophancy fell.

**Why promising.** This is the one idea that answers "removed or only hidden?" at the component level, and it names a component type and depth for the fix. Without c12's re-find step, a Stage B recheck on the old heads would mislead.

**Smallest first experiment.**
- **Model:** Gemma-2-9B-it, 42 layers × 16 heads.
- **Step 1, discovery:**
  - attribution patching (gradient × activation) at head outputs and Gemma Scope residual-SAE features, with two roots (PLS-4 projection norm; log-odds);
  - exclusive and shared sets with size-matched random ablations, which u5 lacks;
  - per-head attention from the answer position to the user-claim span vs to the model's own earlier answer, and the OV copy score onto the answer logits;
  - signed AUROC per head.
- **Step 2:** exact path patching of the top 30 heads; class share of the pushback-induced log-odds change.
- **Step 3:** LoRA-DPO, and a MAGS-style conditional projection at the class heads. Re-find the classes on each model and compare the class shares. Downstream cancellation is tested by patching the class's pre-fix output into the fixed model.
- **Reading:**
  - share → 0 with behaviour fixed → removed;
  - share intact with a cancelling component → masked;
  - the share moves to new heads → moved.

**Kill result.** If no head class explains at least 20% of the pushback-induced log-odds change, the override is not attention-mediated at 9B. The head-class tracking is then dropped in favour of the MLP-band analysis from idea 4.

**Cost.** About 2 GPU-hours for discovery and patching, plus 2-4 GPU-hours per LoRA-DPO run, plus re-finding: about 6-8 GPU-hours.

**Scoop risk.** Medium-high. Project idea 3 ("remove or mask") is already flagged high-risk, and 2608.01835 tracks sycophancy geometry through post-training.

---

## 7. Manifold vs discrete state on a built pressure × confidence grid (HYPOTHESIS)

**Idea.** `c5` warns that a two-answer flip collapses its manifold to a line. We add structure: four pushback strengths × five prior-confidence bins. We fit the activation and behaviour manifolds on resisted items only, then place each flipped item. Our hypothesis is that flips under mild pushback lie *on* the resisted sheet at the claimed-answer end (continuous override), while flips under authority pushback form a separate unimodal cluster (a discrete "defer" state). The two would need different fixes.

**Builds on.**
- `c5-manifold-steering-2605.05115`:
  - isometry r and E_BC (p.6, p.8);
  - leaving out same-geodesic pairs (p.28);
  - the unfair linear baseline, which replaced the whole residual (p.28);
  - sycophancy named as future work (p.16).
- `w2-shape-of-beliefs-2602.02315`: probes transfer only locally along the manifold (p.5); linear steering is not geometrically faithful (p.9); additivity test (Fig. 13, p.22).
- `c15-confidence-manifold-2602.08159`: unimodal classes (KMeans-2/3 cost 1.3-2.4 points, p.4); 2-8D (p.4).
- `u4-role-learning-1910.09113`: continuous vs snapped vs discrete evaluation (Table 1, p.6).
- Spokes, not a Ring (NeurIPS 2026, scout #6): occupancy and sensitivity tests (not read; blocked).

**Why promising.** Whether failure is a "manifold or discrete state" decides the fix. A continuous push calls for dose-matched counter-steering or reweighting. A discrete state calls for a gated detector and intervention. The pressure grid also gives H3 directly: train the probe at pressure 0 and test its accuracy as pressure rises, against a refit at each level.

**Smallest first experiment.**
- **Input:** 400 items × 4 pressure levels, greedy outcomes, three layers around the idea-4 fork.
- **Manifold fit:** centroids per (pressure, confidence) cell on resisted items, splines, and a behaviour manifold in √p coordinates over {correct, claim, other}.
- **Tests:**
  - isometry against Euclidean distance;
  - each flipped item's distance to the sheet against held-out resisted items;
  - QDA/KMeans unimodality of flipped items per level;
  - grid additivity;
  - steering flipped items back along the spline vs a linear step *in the same subspace* (fixing the c5 baseline), scored by accuracy, E_BC and held-out off-target KL.

**Kill result.** If the top PC carries more than 90% of centroid variance (the grid is 1-D) and spline steering beats same-subspace linear steering by less than 1.2× in E_BC, the manifold framing is dropped and the result reported as linear.

**Cost.** About 1 GPU-hour of forwards plus about 2 GPU-hours of steering evaluation.

**Scoop risk.** High. 2608.01835 builds behavioural manifold charts for sycophancy, and the c5 group named sycophancy as next.

---

## 8. Filler-role binding: "transformed" as re-binding (HYPOTHESIS)

**Idea.** Under pushback, both candidate identities may stay decodable while their *roles* swap: Canberra bound to "what was said earlier", Sydney bound to "what is true". A single correctness probe that reads "Canberra present" cannot see such a swap, but a filler ⊗ role decomposition with ROLE's substitution test can. Our hypothesis is that some flips that probes classify as bypass are re-bindings, which a binding-targeted surgery reverses.

**Builds on.**
- `u4-role-learning-1910.09113`:
  - substitution accuracy through the model's own downstream: 94.83% learned vs 4.52% bag-of-words (Table 1, p.6);
  - constituent surgery (p.7, Fig. 5);
  - the bag-of-words baseline in the same range as the learned roles on sentence encoders (p.7);
  - the single-vector limit for transformers (p.8).
- `c8-represented-used-2610.05292`: the gauge control (p.11).
- `c5-manifold-steering-2605.05115`: Hellinger outputs for scoring.

The entity-binding literature (for example Feng and Steinhardt 2023) is outside this reading set and must be checked before any novelty claim.

**Why promising.** It is the only idea that gives fate 5 a concrete, editable form rather than "re-encoded somehow". Our planned prompts name both candidates, which gives us fillers and fixed mention positions, so the obstacle in u4 p.8 (no clean grammar) is milder.

**Smallest first experiment.**
- **Input:** Gemma-2-9B-it, pushback prompts naming both candidates, answer position at the idea-4 fork layer.
- **Fit:** ĥ = W Σ_j f(a_j) ⊗ r_j + b. Fillers f are PCA-64 of the model's own embedding rows for each candidate. There are three learned roles (model-earlier, user-claim, unbound), plus a bag-of-words baseline with one shared role.
- **Tests:**
  - replace h with ĥ and run the remaining layers (substitution accuracy on flip vs resist);
  - surgery that swaps the role vectors of the two fillers, compared with a random-swap control.
- **Reading:** surgery reverses at least twice as many flips as the random swap, and learned roles beat bag-of-words → re-binding (5), with a binding-targeted steer as the fate-matched fix.

**Kill result.** The idea is dropped if substitution accuracy with learned roles is at most the bag-of-words baseline + 5 points, or if surgery reverses less than twice as many flips as the random swap.

**Cost.** About 1 GPU-hour.

**Scoop risk.** Low.

---

## Cross-cutting rules these transfers rely on

- Every angle: report sum cos² against a matched-dimension Haar null, not the mean angle (`c7-readout-mediator-angle-2605.29126` p.4).
- Every ablation and patch: a 25-draw random envelope and additive drops (c7), or matched-norm random nulls with a noise floor reported next to each null claim (`c9-lagged-coupling-2609.01048` p.10).
- Re-fit coordinates on the model under test (c8 p.13). After any Stage B fix, re-find the circuits (`c12-tigges-circuits-across-training-2407.10827` p.7).
- Bring the baseline ladder to every group feature: PCA-k, a 1-D mean difference, a full probe and the logit margin. One direction beat the shape scalars, 90.3% vs 66.6% (`c11-trajectory-signatures-deception-acl2026` p.9).
- Gate every claim with idea 1's confusion matrix.

## Not covered

The full texts of 2608.01835, 2601.11563, 2607.20146, Spokes and Genadi et al. were not read (scout_pass2 abstracts only; OpenReview blocked). Their scoop and contradiction risk for ideas 2, 4, 6 and 7 is therefore based on abstracts only.
