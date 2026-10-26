# Lagged Coupling: Internal Representations Become Readable Before They Become Causal (Xining Xun, Tsingjiao Information Science; arXiv 2609.01048)

Slug: `c9-lagged-coupling-2609.01048`. Source: local `papers/c9-lagged-coupling-2609.01048/paper.txt` and page images only.

## In short

Across 6 Pythia sizes and 8 checkpoints, a linear probe reads the target almost perfectly from step 1k (AUROC >= 0.99), but adding the probe direction back at the same site does nothing beyond matched random directions in 43 of 48 cells. Early cells even backfire, and 'write-in' stays under about 0.11% of headroom. The output's own logit margin becomes informative only slowly. For us this is a recipe and a warning for fate 2 (bypass), not evidence about instruct models under pressure: probes saturate suspiciously early, the probe site and target are unspecified, and the 6.9B-12B nulls sit below the instrument's noise floor.

**Relevance.** Less relevant than the title suggests. It is a training-dynamics study on base Pythia with controlled answer-format contrasts, not a study of instruct models acting against a decodable answer under pressure. Its value for us is methodological: a clean, cheap recipe for testing fate 2 (bypass), with matched nulls, dose-response, a noise floor and headroom-normalized write-in, plus a well-argued warning that probe accuracy does not imply steerability. The substantive results are weakened by trivially saturated probes and an instrument that cannot resolve effects at 6.9B-12B.

**Reuse verdict.** adapt: copy the controls (matched-norm random-direction nulls as z, dose-response, noise floor, logit-margin AUROC as Track 2, headroom/write-in, pre-registration, weight hashing) into Stage A's bypass check. Do not lean on its claims as evidence about our setting, because the models are base Pythia, the target variable and site are unspecified, the large-scale nulls are below the noise floor, and the code is not released.

## Setup

- **data**: Four task families, described only by name and one line each: T template contrast, D distractor robustness, P prior-bias override, N negative transfer. No example items, item counts or target-variable definitions are given in the paper. Probes are trained on held-apart items; dev and test phases are kept separate.
- **models**: Base (pretraining) Pythia suite: 160M, 410M, 1B, 2.8B, 6.9B, 12B, each at 8 checkpoints (steps 1k, 2k, 4k, 8k, 16k, 32k, 64k, 143k). That is 6 x 8 = 48 model-checkpoint cells and x 4 families = 192 units. Replication arm: OLMo-2 at two sizes (sizes not stated). No instruction-tuned or chat models.
- **positions layers**: Answer position. A logistic probe is trained per unit at a 'pre-registered best site' (layer index never reported). Intervention: add lambda times the unit-norm probe direction at that same site, with lambda on a frozen strength grid (grid values not reported).
- **measures**: Track 1: probe AUROC (internal readability). Track 2: AUROC of the model's own logit margin s(x) = z_t* - max other logit against the gold label (behavioral readability). Track 3: paired effect e(lambda) on the logit margin, as a z-score against a null of random-direction edits matched for site and norm. Headroom C: RMS of the residual component along the probe direction. Write-in c = e(1)/C. Noise floor epsilon_m: median absolute within-unit null effect.
- **baselines**: Random-direction edits matched for site and norm (the null for every cell); the logit-margin AUROC acts as a probe-free baseline for readability.
- **controls**: Dose-response over the strength grid; per-unit noise floor; a dev/test probe-discrepancy flag (|AUROC_test - AUROC_dev| > 0.1, fired 3/192); a content-hash gate on checkpoint weights (it caught a corrupted Pythia-2.8B lineage); a pre-registered decision tree with an INDETERMINATE branch; an md5-chained audit ledger.
- **compute**: About 240 instance-hours on 2 x A100-SXM4-40GB for the full grid (quoted in key_results).

## Key results (quoted)

- Internal readability is saturated from the first checkpoint at every size.  
  > "probe inducibility is ≥ 0.990 at every size from step 1,000 onward (dev minimum 0.990; test minimum 0.982" (p7)
- Steering along the probe's own direction does nothing in 43 of 48 cells.  
  > "steering along that same reading direction remains null-equivalent in 43 of 48 model×checkpoint cells" (p1)
- Behavioral readability (the AUROC of the logit margin) develops slowly, and later at larger scale.  
  > "the 2.8B model climbs from 0.534 to 0.968, 6.9B from 0.569 to 0.896, and 12B from 0.524 to 0.909, crossing 0.9 only at the final checkpoint" (p7)
- The ordering is always read-before-write.  
  > "At no scale does causal efficacy precede readability, and at larger scales behavioral readability itself lags internal readability by tens of thousands of steps." (p8)
- Scale does not close the lag.  
  > "the pre-registered regression of catch-up magnitude on log parameters has 95% CI [−0.722, +0.438], crossing zero." (p8)
- Early in training, pushing along the probe direction has the opposite effect (backfire).  
  > "four of the five significant cells are negative, and all four occur in the first 2,000 steps" (p8)
- The nulls are not caused by too small a dose.  
  > "The effect curves e(λ) over the frozen strength grid are flat at all tested strengths in every null cell" (p8)
- Only one positive cell exists, and it does not persist.  
  > "The sole cell where steering significantly helps is 12B at step 8,000 (ze = +2.49 ; dev phase +2.04); it does not persist to step 16,000" (p9)
- Headroom along the probe direction grows a lot with training.  
  > "over training, C expands by factors of 2.0× (160M), 8.1× (410M), 3.0× (1B), 46.9× (2.8B), 37.7× (6.9B), and 56.8× (12B, from 0.038 to 2.16)" (p9)
- Write-in per unit of headroom stays tiny.  
  > "at the final checkpoint, ∣^c∣ is 0.070% (2.8B), 0.109% (6.9B), and 0.108% (12B), with per-cell effects e(1) never exceeding 0.016" (p9)
- The authors' reading: the probe direction describes something the readout ignores.  
  > "the probe direction becomes an increasingly accurate description of a component that the downstream computation increasingly ignores." (p10)
- The instrument's noise floor exceeds the largest effect at the top sizes.  
  > "the dashed line marks the largest causal effect observed anywhere on the grid (|e(1)| = 0.016), which sits at or below the noise floor for the three largest sizes." (p10)
- The prior-bias arm is the one pre-registered positive: larger models are harder to steer off their priors.  
  > "the WLS slope of bias magnitude on log parameters has 95% CI [+0.00062, +0.291]—excluding zero by a hair—with intercept −0.719." (p10)
- In the prior arm, the prior wins over the probe direction.  
  > "makes it progressively resistant to external direction, and it is the prior—not the probe direction—that wins." (p10)
- The OLMo-2 replication preserves the direction of every headline finding but is weak (two sizes).  
  > "the arm replicates the direction of every headline finding—probe saturation from the earliest checkpoint, a readable-before-causal ordering, null-equivalent steering effects—at attenuated magnitude" (p10)
- Development is not monotone.  
  > "The development index confirms the non-monotonicity directly: 34.55% of units (66/191) develop backwards by the frozen definition." (p6)
- Safety framing: probes are sensors, not levers.  
  > "A probe can be an excellent sensor at step 1,000 and a useless lever at step 143,000." (p11)
- Compute for the full grid.  
  > "Total compute was approximately 240 instance-hours on 2×A100-SXM4-40GB." (p4)

## Figures and tables

Numbers marked approx were read off plots; others come from printed tables or labels.

### Table 1 (p3)

- **Plotted**: Positioning table: lines of prior work (probing, activation steering, weight editing, developmental circuits, this work), the track each assumes, and whether it was measured across training.
- **Takeaway**: The authors claim that steering papers assume Track 3 (causal efficacy) holds wherever Track 1 (readability) holds, and that no prior work has tracked all three over training.
- **Numbers**: None, apart from 'yes (6 sizes x 8 checkpoints)' for this work (from table).
- **Text matches figure?** yes: the table restates the Related Work text.

### Table 2 (p4)

- **Plotted**: The pre-registered grid: six Pythia sizes with parameter counts, eight checkpoints and four task families.
- **Takeaway**: 192 units, all under one frozen pipeline. The families are listed but never specified as items.
- **Numbers**: From table: 160M = 1.62e8, 410M = 4.05e8, 1B = 1.01e9, 2.8B = 2.78e9, 6.9B = 6.85e9, 12B = 1.18e10 parameters; checkpoints at 1k, 2k, 4k, 8k, 16k, 32k, 64k, 143k.
- **Text matches figure?** yes

### Table 3 (p6)

- **Plotted**: The evaluated decision tree: scale-axis WLS slopes (main and sub lines), time-axis majority vote, and the fraction of units with a negative development index.
- **Takeaway**: Both pre-registered onset hypotheses come out INDETERMINATE, and a third of units develop backwards.
- **Numbers**: From table: scale main +0.240 [-0.600, +0.866], n = 6; scale sub +0.325 [-0.121, +1.182], n = 5; time 3 A / 3 IND, no majority; negative D fraction 0.3455 (66/191).
- **Text matches figure?** yes: the text rounds the CIs to [-0.60, +0.87] and [-0.12, +1.18].

### Figure 1 (p7)

- **Plotted**: Forest plot of per-size time-axis slopes with 95% CIs. The x-axis label reads 'Time-axis slope of macro index D per log10(step)'. Blue marks 'decided (A)', grey marks INDETERMINATE.
- **Takeaway**: Three sizes have CIs entirely below zero and three cross zero, so the majority vote fails.
- **Numbers**: Approx, read from figure: 160M about -0.06 [-0.16, +0.02]; 410M about -0.65 [-1.0, -0.46]; 1B about -0.45 [-0.92, +0.05]; 2.8B about 0.0 [-0.82, about 0.0]; 6.9B about -0.28 [-0.38, -0.09]; 12B about -0.55 [-0.80, +0.20].
- **Text matches figure?** partly: the verdicts match the text. However, the x-axis names the development index D while the text says 'milestone index'. The 2.8B point estimate is drawn at the upper end of its CI, which is odd for a WLS interval. The paper also never says what fork A or fork C mean in substance.

### Figure 2 (p7)

- **Plotted**: A schematic (not data): three horizontal tracks over log training step. Internal readability is saturated, behavioral readability ramps up, and causal efficacy sits in the null band with early backfire triangles and one 12B pulse at 8k.
- **Takeaway**: This is the paper's mental model. The quantitative versions are Figures 3, 4 and 5.
- **Numbers**: Labels only: AUROC >= 0.990; 12B pulse z = +2.49; inert in 43/48 cells.
- **Text matches figure?** yes, but it is a drawing and carries no evidence by itself.

### Figure 3 (p8)

- **Plotted**: For Pythia-6.9B and Pythia-12B over training: internal probe AUROC (green), behavioral answer-score AUROC (blue) and causal z_e (orange, right axis, grey band = |z| < 2).
- **Takeaway**: The probe sits at about 1.0 from step 1k. The logit-margin AUROC climbs slowly (to 0.896 for 6.9B and 0.909 for 12B). Causal z mostly stays inside the null band, apart from -2.43 at 12B 2k and +2.49 at 12B 8k.
- **Numbers**: Approx, read from figure; these agree with Tables B1 and C2. 6.9B behavioral 0.57, 0.48, 0.49, 0.57, 0.775, 0.875, 0.877, 0.896 and z_e -1.88, -0.55, -0.36, +0.80, -0.17, +0.70, +0.51, +0.45. 12B behavioral 0.524, 0.512, 0.632, 0.561, 0.662, 0.752, 0.815, 0.909 and z_e -0.12, -2.43, +0.63, +2.49, -0.31, +1.04, -0.06, -0.56. Internal probe about 0.99-1.0 throughout.
- **Text matches figure?** partly: the data match the appendix tables. The caption's claim that 'the gap widens with scale' rests on two sizes, and Section 4.3 only shows that the lag does not shrink (its CI crosses zero).

### Figure 4 (p9)

- **Plotted**: Heatmap of causal efficacy z_e for 6 sizes x 8 checkpoints. Boxed cells have |z| >= 2. The colorbar label reads 'z_e (family median, test phase)'.
- **Takeaway**: Steering along the probe direction is inert almost everywhere. The four significant negative cells are all within the first 2k steps, and the one significant positive cell is 12B at 8k.
- **Numbers**: From the figure's printed cell values, which match Table B1. Significant cells: 410M 2k -2.65; 2.8B 1k -2.68; 2.8B 2k -2.48; 12B 2k -2.43; 12B 8k +2.49. Largest non-significant cells include 1B 16k +1.38, 410M 143k +1.22 and 1B 8k +1.21.
- **Text matches figure?** yes. One thing the main text never states: each cell is a median across the four task families, so an effect confined to one family (for example P) could be hidden.

### Figure 5 (p10)

- **Plotted**: (a) Headroom C along the probe direction over training, log scale, one line per size, with growth factors annotated. (b) Noise floor epsilon_m against parameter count, log-log, with a dashed line at the largest causal effect on the grid (|e(1)| = 0.016).
- **Takeaway**: The direction becomes much more 'occupied' with training and scale, but the instrument gets coarser with scale too.
- **Numbers**: (a) Growth factors as annotated: x2.0 (160M), x8.1 (410M), x3.0 (1B), x46.9 (2.8B), x37.7 (6.9B), x56.8 (12B). Approx, read from figure: 12B goes from about 0.04 to about 2.1; 160M from about 0.05 to about 0.1. (b) Floors 0.00398, 0.00597, 0.00619, 0.01179, 0.0286, 0.0328; dashed line at 0.016.
- **Text matches figure?** partly: the caption says the largest effect 'sits at or below the noise floor for the three largest sizes'. In panel (b), though, the 2.8B floor (0.01179) lies below the 0.016 line, so the claim holds only for 6.9B and 12B.

### Table 4 (p11)

- **Plotted**: Two families of safety methods (monitoring with probes, runtime activation steering), the track each relies on, and whether the grid supports it.
- **Takeaway**: Probes work as sensors (Track 1 is supported). Steering along probe directions is not supported for most of training.
- **Numbers**: From table: AUROC >= 0.990 from step 1k at all scales; steering inert in 43/48 cells and counterproductive in 4 early cells.
- **Text matches figure?** yes

### Extra tables (not in the manifest; text only)

- **Table B1 (p15)**: z_e for all 48 cells (test phase). Read from paper.txt only; no page image was rendered for page 15. This is the source data for Figure 4. It notes the 12B 8k pulse is +2.04 in the dev phase. Numbers: As listed under Figure 4.
- **Table C1 (p15)**: Per-size noise floor epsilon_m and final-checkpoint |c| for the three largest sizes. Text only. At 12B the write-in (0.108% of C = 2.16, about 0.0023 in margin units by my arithmetic) is roughly 14 times below the 12B noise floor (0.0328). The 'pinned write-in' at large scale is therefore not resolved by the instrument. Numbers: epsilon_m: 0.00398, 0.00597, 0.00619, 0.01179, 0.0286, 0.0328. |c| at 143k: 0.070% (2.8B), 0.109% (6.9B), 0.108% (12B).
- **Table C2 (p15)**: Behavioral (logit-margin) AUROC over checkpoints for 1B, 2.8B, 6.9B and 12B. Text only. Behavioral readability starts near chance (0.47-0.65) and rises to 0.71-0.97 by 143k. Numbers: 1B 0.647 to 0.711; 2.8B 0.534 to 0.968; 6.9B 0.569 to 0.896; 12B 0.524 to 0.909 (first and last checkpoints).

## Methods we could use

### Three-track decomposition: probe AUROC, logit-margin AUROC, and matched-null steering z on the same items

- **Measures**: Whether a variable is (1) linearly decodable inside the model, (2) present in the model's own output score, and (3) causally effective when pushed along the decoding direction.
- **Fate it could diagnose**: Bypass: Track 1 high and Track 3 null. It also separates readout/threshold failure from the other fates: if Track 2 (margin AUROC) is high while accuracy is low, the cut-off is wrong. For us, Track 2 is exactly the logit-margin control already planned.
- **Causal follow-up**: Track 3 is itself the causal test, for the probe direction only. Follow up with a causally chosen direction (DAS or subspace patching) to tell 'this direction is inert' apart from 'the information is inert'.
- **Data and compute**: Per model: one forward pass per item for activations, a logistic probe, and a few forward passes per strength level lambda for steering, plus random-direction nulls. Cheap at 9B-27B for about 400 items.
- **Code**: No: 'Code, configurations, manifests, and the audit ledger will be released with the camera-ready.' (p12)

### Norm- and site-matched random-direction null, reported as z_e, with a dose-response curve

- **Measures**: Whether an edit along the probe direction moves the logit margin more than random edits of the same norm at the same site.
- **Fate it could diagnose**: Bypass (z about 0 at all doses). A strongly negative z, i.e. backfire, hints that the readout uses the variable with flipped sign, which is a cheap pointer toward 'transformed'.
- **Causal follow-up**: Doses beyond the grid; comparing z for probe-derived vs causally derived directions.
- **Data and compute**: Dozens of random directions per cell; forward passes only.
- **Code**: No (not yet released).

### Headroom C and normalized write-in c = e(1)/C

- **Measures**: How strongly the representation expresses the variable along the direction (RMS of the component), against how much the output moves per unit of that natural spread.
- **Fate it could diagnose**: Bypass: large C with c near 0 means the direction is occupied but unused. It extends to subspaces: replace C with the RMS norm of the projection onto the subspace.
- **Causal follow-up**: Compare c across flipped and resisted items, across pressure types, and before and after a fix (Stage B: was the circuit removed or only hidden?).
- **Data and compute**: Same activations as above. Negligible extra compute.
- **Code**: No (not yet released).

### Per-unit noise floor epsilon_m (median absolute within-unit null effect)

- **Measures**: The smallest effect the steering instrument can resolve in that unit.
- **Fate it could diagnose**: Any fate claim of 'no causal effect' (bypass, erased) is only meaningful when the floor is below the effect sizes that matter.
- **Causal follow-up**: Decide where more items or seeds are needed before calling a null.
- **Data and compute**: Comes free with the random-direction nulls.
- **Code**: No (not yet released).

### Pre-registered decision tree with a written INDETERMINATE branch, plus a content-hash gate on weights and an append-only md5 ledger

- **Measures**: Nothing about the model. It guards against moving thresholds after seeing data and against silently corrupted checkpoints.
- **Fate it could diagnose**: All of them: it fits the lab mod's sealed-bet workflow, where fate-sorting thresholds would be frozen before the collect step.
- **Causal follow-up**: n/a
- **Data and compute**: None beyond bookkeeping.
- **Code**: No (gate promised with pipeline release).

### Development index D (Spearman correlation of checkpoint order with milestone progress) and ordering patterns over training

- **Measures**: Whether each unit consolidates monotonically over training checkpoints.
- **Fate it could diagnose**: Weak for us: it is about training time, not token time. It loosely echoes direction 3 (jump vs drift) and Stage B checkpoints (before and after fine-tuning).
- **Causal follow-up**: Denser checkpoint sampling around transitions.
- **Data and compute**: Many checkpoints, which we will not have for 9B-27B instruct models apart from before and after our own fine-tunes.
- **Code**: No.

## Controls worth copying

- Random-direction edits matched for site and norm as the null for every steering or patching cell, reported as z.
- Dose-response curves over a fixed strength grid, so a null cannot be blamed on a timid dose. Report the grid values, which this paper does not.
- A per-unit noise floor (median absolute null effect), reported next to every null claim.
- The logit-margin AUROC as a probe-free readability baseline (their Track 2). This is the same as our planned logit-margin control.
- A dev/test discrepancy flag on probe AUROC (they used > 0.1).
- Pre-registered thresholds with a written INDETERMINATE branch (maps onto the lab mod's sealed bets).
- Content hashes of model weights and of every result file, plus atomic writes. This is cheap insurance for multi-day runs.

## Pitfalls

- Probe AUROC >= 0.99 from step 1k, even at 160M, strongly suggests the target is trivially decodable from surface features of the prompt. The paper reports no control-task, random-init or embedding-layer probe baseline, although its own introduction names this exact weakness of the field.  
  > "probe accuracies saturate trivially, steering nulls are rarely matched for norm and site, and post-hoc threshold tuning can manufacture “emergence” out of noise" (p2)
- At large sizes the instrument cannot resolve the effects it reports. The 6.9B and 12B noise floors (0.0286, 0.0328) exceed the largest effect on the grid (0.016), so 'write-in < 0.11%' at 12B is below resolution. The caption also miscounts: 2.8B (0.01179) is below 0.016.  
  > "so “null-equivalent” at 12B means |effect| below a coarser floor" (p12)
- The probe site (layer), the target variable, the item counts and the lambda grid are never reported, which makes the work hard to reproduce or map onto our setup.  
  > "is trained per unit on held-apart items at the pre-registered best site ℓ; inducibility i is its AUROC on the dev split" (p4)
- The nulls apply only to one additive edit along a probe direction at one site. They say nothing about subspace or DAS-style interventions or patching. The paper states this boundary itself.  
  > "Weight-space edits[21] and trained representation interventions[23][24] answer a different question (can the model be changed), and our nulls do not speak against them" (p6)
- The Figure 4 cells are medians across the four task families ('family median' on the colorbar), so family-specific effects could be hidden. Not stated in the main text.  
  (not stated in the paper; see p9)
- Figure 1's axis names the development index D while the text says milestone index, and the 2.8B point estimate sits at its CI's upper end. Forks A and C are never defined in substance.  
  (not stated in the paper; see p7)
- Code is not released yet, the author is a single company author, and only base Pythia models with controlled format contrasts are used, so the setting is far from instruct models under user pushback.  
  > "Code, configurations, manifests, and the audit ledger will be released with the camera-ready." (p12)

## Limitations and future work (authors' words)

- Limitations (p12): "One suite (Pythia) carries the main claims; the OLMo-2 replication is two sizes and capped at “weak” by the frozen rule."
- Future work (p12): "Open-ended generation and multi-turn control are exactly the settings where readability-to-causality transfer matters most, and extending the grid there is the natural next step."

## Bearing on the six fates

| Fate | How | Strength |
|---|---|---|
| 2 bypass | This is the paper's core result. The direction is decodable (AUROC >= 0.99), yet a matched-null edit along it is causally inert in 43/48 cells, while the output still comes to carry the answer (Track 2 rises to about 0.9). So the decision runs through something other than the probe direction. The headroom/write-in ratio is a ready-made bypass diagnostic. | strong |
| 4 override | In the prior-bias arm, larger models are harder to steer off their learned priors: 'the prior, not the probe direction, wins'. This is a training-time analogue of a competing signal winning, though not tested with a user-pushback competitor. | moderate |
| 5 transformed | The early backfire cells (pushing toward the answer pushes the output away) are what a sign-flipped readout would produce. This is only suggestive: base models early in pretraining, with no rotation or sign analysis. | weak |
| 3 not routed | Injecting along the probe direction fails, which argues against 'not routed' for that direction. The paper cannot say whether a causally chosen direction would work. | weak |
| 1 readout/threshold failure | Track 2 (the logit-margin AUROC) is the right probe-free tool for spotting threshold failures (a high margin AUROC with low accuracy), but the paper does not report accuracy against margin AUROC. | weak |
| 6 erased | Nothing is erased in this setting: readability stays saturated everywhere. | weak |

## Bearing on the group / subspace angle

Mostly a warning. A single readout direction can be an accurate description and still a useless lever, and the authors argue for directions 'chosen by causal criteria rather than readability criteria'. That supports moving from single probe directions to causally selected subspaces (DAS, subspace patching) and judging any subspace by its causal write-in, not only by angles or CKA. The headroom/write-in ratio and the per-unit noise floor carry over directly to subspaces: RMS of the projection against the effect of a matched-norm edit inside the subspace, compared with random subspaces of the same rank. The paper uses no group, subspace, manifold or token-path geometry itself.

## Where in the architecture

Weak on location. Everything is measured in the residual stream at a 'pre-registered best site' whose layer is never reported, at the answer position, with no component-level (attention, MLP, head) analysis. The structure (saturated readability, delayed behavioral readability, inert steering) holds across 6 Pythia sizes and 8 checkpoints, and in direction only on OLMo-2 at 2 sizes. There are no seeds beyond md5-derived unit seeds and no instruct models.

## Actionable fix or monitor

Monitoring, not steering: use probes as sensors (early-warning monitors). Do not treat probe directions as control handles. Before any steering fix, verify the direction's causal write-in against matched random directions with a dose-response curve. For fixes at scale, pick directions by causal criteria. For our Stage B, recheck after each fix both the decodability and the matched-null causal write-in of the fate-relevant direction, since a fix that only hides the behavior could leave readability intact while write-in changes.

## Sections read

Abstract, 1 Introduction, 2 Related Work, 3 Pre-registered Setup and Decision Protocol, 3.1 Grid and instrumentation, 3.2 Estimands, 3.3 The decision tree, 3.4 Why additive single-site surgery is the right estimand, 4 Results, 4.1 The pre-registered verdict: INDETERMINATE on both axes, 4.2 Three tracks, three clocks, 4.3 The ordering is read-before-write, and scale does not close the lag, 4.4 Interventions are largely inert, occasionally counterproductive, 4.5 Mechanism: headroom outruns write-in by orders of magnitude, 4.6 The prior-bias arm: a mechanical positive result, 4.7 Replication on OLMo-2: direction preserved, magnitude weak, 5 Discussion, 5.1 Implications for AI safety: sensors before levers, 6 Limitations and Threats to Validity, 7 Reproducibility and Audit, References, Appendix A The weight-corruption incident: a mandatory content-hash gate, Appendix B Full causal-efficacy grid, Appendix C Resolution, behavioral readability, and configuration summary
