# Rewriting or Reweighting? A Geometric Account in Language Models (Wang, Yang, Wang, Zhang; Peking Univ. / Beijing Jiaotong Univ.; arXiv 2608.01835v1, 3 Aug 2026)

Slug: `p2-sycophancy-manifold-posttraining-2608.01835` · Relevance: **high** · Code: https://github.com/ronglingze/Manifold-Analysis

## Bottom line for our plan

DOES: (1) asks the same Stage B question in general form ('remove, build a competing mechanism, rotate, shift occupancy, weaken readout'); (2) tracks a sycophancy subspace through controlled SFT, DPO and SFT->DPO checkpoints on Llama-3.1-8B with principal angles, projector overlap and anchored retention; (3) claims SFT rewrites and DPO reweights; (4) separates geometry, occupancy and causal gain. LEAVES OPEN: (a) no decodable correct answer: sycophancy is 'sycophantic vs truthful stance text' on opinion claims, so no knowing-vs-acting gap, no probe for the correct answer, no 'decodable but flipped' items; (b) no per-item fates and no six-way sort (Stage A untouched); (c) no fate-matched fixes, no H2; (d) only Llama-3.1-8B, one seed, DPO with 20x smaller lr and ~no behavior change, so 'DPO hides' is not established; (e) no pressure types (H1); (f) MLP neurons only, no attention heads reading the user claim, no layerwise subspace path, no Procrustes transformed-vs-erased test; (g) sycophancy chart causal validation is weak (NOC gain mean 0.062, sometimes negative; steering curves only for repetition); (h) no base-vs-base noise floor: DPO's ~10 deg is mostly a ~9 deg first-checkpoint offset shared by DPO-up, DPO-down and SFT, so 'DPO reweights' is weaker than the headline; (i) R^G uses fixed dataset labels, not the model's own flips. Scoop risk for Stage B: moderate on framing ('rewrite vs reweight' for SFT vs DPO on sycophancy geometry is now taken and must be cited), low on substance (removal vs hiding of a decodable correct answer, per item, with fate-matched fixes and matched update size is not done).

**H1/H2/H3:** H1: not tested (one pressure type: user states an opinion/claim; no pushback-after-answer or tempting-option conditions). H2: not tested (only SFT and DPO, up/down; no fate-matched fix). Their result predicts plain DPO leaves the inherited geometry in place, which is the premise of H2's 'plain DPO hides' arm, but with unmatched update size and DPO movement near the measurement floor. H3: anchored retention is a direct analogue: base-chart separability survives DPO (R^G 0.995-0.998) and mostly survives SFT (0.81-0.97), arguing that old probes keep working at the population level; Figure 13 shows per-item cases where the old chart stops separating after SFT, which is the H3 failure mode, but only 4 items. Caveat: R^G labels are fixed dataset labels on teacher-forced text (sycophantic vs truthful response), not the model's own behaviour, so it tests whether an old probe still separates the same text, not whether it catches the model's failures (our H3).

**Reuse verdict:** Reuse the measurement kit for Stage B: matched-rank principal angles + projector overlap across fix checkpoints, anchored retention R^G (pre-fix probe on post-fix model vs post-fix native probe), checkpoint-by-checkpoint trajectories, base-anchored teacher-forced paired candidates, random-coordinate controls. Do not copy: unsigned NOC proxy, unstandardized L1 selection, 3-vs-1 span labels, PCA-of-scaffold chart whose PC1 is orthogonal to the behavior direction (use a supervised or class-difference subspace instead), and unmatched SFT vs DPO budgets (match KL or behavior change before comparing geometry).

## Setup

- **data:** Repetition: 2,736 response-level records (1,368 repetitive from a 928,046-entry pool, 1,368 non-repetitive all-judge-true), prompts not matched across classes. Sycophancy: DataCreatorAI/Anti-Sycophancy-DPO, 3,480 chosen(truthful)/rejected(sycophantic) pairs on user claims such as 'All smartphones are spying on us all the time.'; responses are teacher-forced dataset text, not the model's own generations; a 'stance span' is extracted from each response by Llama-3.3-70B-Instruct (6,678 of 6,960 records kept). Post-training data: repetition 755,920 SFT records / DPO pairs (repaired continuations written by Llama-3.3-70B-Instruct); sycophancy 3,480 SFT examples / 3,480 DPO pairs from the same Anti-Sycophancy-DPO source.
- **models_and_sizes:** Atlas: 23 models from Gemma (2-2B, 2-2B-it, 3-4B-it, 2-9B, 2-9B-it, 2-27B, 2-27B-it), Qwen (2.5-0.5B, 0.5B-Instruct, 7B, 7B-Instruct, 2.5-32B, 32B-Instruct, Qwen3.5-27B), Mistral (7B-v0.3, 7B-Instruct, Nemo base/instruct, Small-3.1-24B base/instruct), Llama (3.1-8B, 3.1-8B-Instruct, 3.3-70B-Instruct). Controlled post-training: only Llama-3.1-8B base, one seed (42).
- **layers_positions:** Features are MLP down_proj inputs (FFN intermediate neurons) in all layers, flattened (e.g. 32 x 14,336 = 458,752 for Llama-3.1-8B), mean-pooled over a behavior span (sycophancy: extracted stance-text tokens; repetition: repeated span). NOC as implemented is |activation| x column norm of W_down / ||layer output||, a non-negative magnitude proxy. Sparse L1 logistic regression (C=1, saga) picks positive-weight coordinates (BACs: 20-61 for repetition, 42-101 for sycophancy); chart = PCA to 90% variance. Sycophancy classifier is 3-vs-1: positive = stance tokens of sycophantic records; negative = stance tokens of truthful records plus all non-stance tokens of both.
- **post_training_protocol:** Six runs per behavior from Llama-3.1-8B: SFT-up, SFT-down, DPO-up and DPO-down from base, SFT->DPO-up and SFT->DPO-down. SFT lr 1e-5 (sycophancy: 120 steps ~4.4 epochs up, 360 steps ~13.2 epochs down; repetition: 100 steps up, 600 steps down); DPO lr 5e-7, 60 steps (~2.2 epochs for sycophancy, but only ~0.010 epoch for repetition: 60 x 128 / 755,920 pairs), beta 0.1 (0.7 for SFT->DPO-down sycophancy); DPO reference model = frozen copy of the DPO initialisation. Effective batch 128, fp16, ZeRO-3, seed 42. Ten matched relative stages per run. Behaviour column in Table 2b is p_b/m_b: probability assigned to the behaviour-positive (sycophantic) target / sycophantic preference score. Base-model behaviour values are not reported; the first stored checkpoints (SFT step 10, DPO step 6) sit at sycophancy preference ~0.454 and Syco Prob 0.355.
- **baselines:** Matched-size random coordinate scaffolds (same model and layer distribution) for PCA, CCA, linear probing and steering; base model chart as anchor for post-training; public Base vs Instruct pairs (10 pairs) as uncontrolled ecological check.
- **controls:** Random-coordinate controls (reported numbers for CCA and probing); mean CCA and full-chart probing instead of primary CCA alone; anchored Fisher retention (base chart applied to checkpoint features vs checkpoint's native chart); dose-response steering with PPL tracking for repetition; base-anchored teacher-forced fixed candidate pairs across checkpoints (4 pairs). Label-permutation, rank-sensitivity, token-window and behavior-mismatch controls are described in Appendix I but no numbers for them are given in the text.

## Key results (quoted)

- DPO from base keeps the behavior chart; SFT moves it. (Checker caveat: ~9 deg of DPO's ~10 deg sycophancy angle is already present at step 6 for both up and down, so the DPO number is near the measurement floor.) — "Across the four behavior–direction settings, DPO initialized from Base is consistently geometry-conservative, retaining mean projector overlap 0.939 with mean angle 7.64◦" (p7)
- SFT displaces the chart about 3x more in angle. — "SFT produces substantially larger chart displacement, with mean overlap 0.745 and mean angle 25.07◦" (p7)
- DPO after SFT stays in the SFT geometry. — "DPO after SFT does not restore the Base chart, indicating that the sequential objective continues optimization within the SFT-induced geometry." (p7)
- The 'reweighting' mechanism for DPO is not directly measured: DPO changes all three measured axes only a little. — "In particular, reward-only changes are usually modest in all three measured axes, so the figures support geometric conservation more directly than a universal large occupancy- or gain-shift account." (p45)
- Reward-only sycophancy runs barely change behavior. — "Reward-only sycophancy trajectories preserve geometry while moving behavior modestly." (p18)
- Leaving the base chart does not mean less sycophancy. — "A trajectory that leaves the high-density Base support does not necessarily indicate weaker sycophancy." (p24)
- After SFT, sycophancy rises while paths move away from the base positive direction (the base chart stops being the right coordinate system). — "The fact that sycophantic preference can increase while the corresponding path moves away from the Base positive direction is particularly informative." (p26)
- Public instruct models usually have more, not less, causal gain on the repetition chart. — "This pattern is inconsistent with a simple account in which instruction alignment merely suppresses the behavior mechanism." (p42)
- Lower separability does not imply lower causal readout. — "This again shows that suppressing statistical separability does not necessarily suppress the causal readout of the behavior." (p42)
- Primary CCA alone is not evidence of a shared axis. — "The random primary scores of 0.667 in ACT and 0.817 in NOC are too large to regard a single canonical correlation as decisive evidence." (p33)
- Behavior-selected coordinates beat random on mean CCA. — "The ACT score increases from 0.222 under random selection to 0.515 under behavior-associated selection, an absolute increase of 0.293." (p32)
- Repetition steering works in all 23 models. — "Both operators produce a positive endpoint behavioral change for every model." (p36)
- Public instruct models usually have more causal gain on the repetition chart: 7 of 10 pairs in both NOC and ACT; reliability changes split 5/5 (NOC) and 4/6 (ACT). — "Seven of ten pairs exhibit positive changes in NOC causal gain, and seven of ten exhibit positive changes in ACT causal gain." (p42)

## Methods

### Sparse behavior-coordinate scaffold + PCA chart (ACT and NOC)
- Measures: L1-logistic selection of MLP neurons that predict a behavior label, then a low-rank PCA basis U of those neurons' pooled activations (ACT) or magnitude-normalized contributions (NOC).
- Fate it could diagnose: Gives the subspace to track; ACT-vs-NOC split (present vs functionally contributing) is a weak screen for bypass (fate 2): coordinates active but with little contribution to the layer update.
- Causal follow-up: Neuron scaling and chart-subspace add/remove steering on the selected coordinates.
- Data/compute: Few thousand labeled examples; forward passes with hooks on down_proj; sklearn L1 logistic on up to ~460k features per model (8B).
- Code: yes: https://github.com/ronglingze/Manifold-Analysis

### Grassmannian chart tracking across checkpoints
- Measures: Projector overlap S, mean principal angle, projection distance between base and checkpoint charts at matched rank; velocity, path length and excess path length over checkpoints.
- Fate it could diagnose: Transformed (fate 5): chart rotates after a fix; vs retained geometry (fate 1/4 style readout or competition change).
- Causal follow-up: Steer along base vs new chart at the new checkpoint to see which one still controls behavior.
- Data/compute: Charts at each checkpoint (one forward pass per checkpoint over the item set); cheap linear algebra.
- Code: yes (same repo)

### Anchored Fisher retention R^G
- Measures: Fisher separability of checkpoint features projected into the base chart, divided by separability in the checkpoint's own chart.
- Fate it could diagnose: Stage B removed vs hidden: R^G ~1 with behavior fixed = old code still there and readable (hidden); R^G << 1 with native G high = re-encoded (transformed); both low = erased (fate 6). Also a direct H3 test: does a pre-fix probe still work post-fix.
- Causal follow-up: Patch base-chart component back in post-fix to test whether behavior returns.
- Data/compute: Same as above; needs labels at each checkpoint.
- Code: yes (same repo)

### Geometry / occupancy / gain / threshold decomposition
- Measures: Chart movement (d_proj), class-conditional second-moment occupancy gap G_occ, finite-difference causal gain G_causal from steering, intercept tau from logistic fit.
- Fate it could diagnose: Readout/threshold failure (fate 1: gain or intercept change with geometry fixed) vs override (fate 4: occupancy shift) vs transformed (geometry move).
- Causal follow-up: Gain is itself causal (dose-response); threshold term is said to be estimable but no tau values are reported.
- Data/compute: Steering sweeps at 5 strengths per checkpoint per model.
- Code: yes (same repo)

### Cross-model CCA atlas with random-coordinate control
- Measures: Primary and mean canonical correlation between two models' chart coordinates on matched items; directed linear probing; same with matched random coordinates.
- Fate it could diagnose: Not a fate test; tells whether a fate-relevant axis found at 9B is the same axis at 27B.
- Causal follow-up: None directly.
- Data/compute: Same item set run through every model.
- Code: yes (same repo)

### Base-anchored teacher-forced candidate paths
- Measures: For fixed prompt + truthful and sycophantic candidate answers, token-by-token projections into the base chart at each checkpoint, with length-normalized likelihood as node size.
- Fate it could diagnose: Per-item view of Stage B: does the fix move items out of the old chart (transformed) or keep them inside while changing preference (readout/override)?
- Causal follow-up: Natural place to patch base-checkpoint states into the post-fix model per item.
- Data/compute: Teacher-forced passes of paired candidates at each checkpoint; trivial compute.
- Code: yes (same repo, presumably)

### Token-level trajectories over log-density-ratio landscape
- Measures: Per-token chart coordinates plotted over log p+(z)/p-(z) background.
- Fate it could diagnose: Where flipped vs resisted items split along the answer (group angle: token paths).
- Causal follow-up: Choose the token position for patching.
- Data/compute: Teacher-forced passes; 2-D KDE.
- Code: yes (same repo)

## Bears on fates

| Fate | How | Strength |
|---|---|---|
| Stage B (removed vs only hidden) | Closest prior method for the recheck: track the pre-fix subspace across fix checkpoints with principal angles + anchored retention. Their sycophancy-down DPO keeps the base chart (S 0.939, R^G 0.995) while behavior barely moves (Syco Prob 0.355 -> 0.342), consistent with 'hidden/reweighted, not removed', but the behavior change is too small to call it a fix, and ~9 of the 9.68 deg is a first-checkpoint offset shared with DPO-up, so 'conserved' is not distinguished from 'unchanged'. | moderate (method), weak (evidence) |
| 5 transformed | SFT rotates the chart by 22-31 deg; in Figure 13 SFT moves both truthful and sycophantic candidates out of the base chart's positive/negative separation while sycophantic preference rises, i.e. behavior re-encoded in new coordinates. 4 items, qualitative. | weak-moderate |
| 1 readout/threshold failure | Decomposition defines gain (slope) and intercept (threshold) inside a fixed chart; public instruct models gain causal readout while separability drops. Threshold values never reported. | weak |
| 2 bypass | ACT-high / NOC-low quadrant = active but little functional contribution; only conceptual, NOC is an unsigned proxy. | weak |
| 4 override | Occupancy shift inside a kept chart is their stand-in; no competing-signal analysis. | weak |
| 6 erased | Native Fisher G stays high after every run (10-18 for sycophancy), so no erasure observed; R^G with native G gives the test. | weak |

**Group angle:** Directly uses Grassmannian tools the group wants (projector overlap, principal angles, projection distance, path length/velocity over checkpoints) but over training time, not across layers: the chart concatenates selected neurons from all layers, so there is no layer-by-layer subspace path. No Procrustes (only CCA, which is not orthogonal). Token-path idea is present (Appendix K, L: teacher-forced candidate paths in a fixed base chart) but on 4 pairs with no flipped-vs-resisted split and no matching on prior confidence. No attention heads (MLP neurons only).

**Architecture location:** MLP down_proj inputs (FFN intermediate neurons), all layers concatenated; in Llama-3.1-8B the sycophancy scaffold is broad (nearly all 32 layers) with peaks around middle (~layer 13) and late (~layer 23, ~27) layers (approx, read from Figure 14; the paper warns counts are not causal importance); repetition scaffold is sparse. No attention components are analysed.

**Actionable fix:** No sycophancy fix is proposed or shown. Implied: if a fix should change the mechanism rather than reweight it, SFT moves the chart far more than DPO; DPO after SFT stays in the SFT geometry, so the order of stages matters. Repetition can be raised by ACT-chart steering in all 23 models (repetition is the only behavior with demonstrated causal control).

## Controls worth copying

- Matched-size random coordinate scaffolds from the same layers, run through the whole pipeline (PCA, CCA, probing, steering).
- Report mean CCA and full-chart probing, not primary CCA (random primary CCA reached 0.667 ACT / 0.817 NOC).
- Anchored retention: apply the pre-fix chart to post-fix activations and compare with the post-fix native chart.
- Track every checkpoint, not only endpoints (rules out 'moves then returns').
- Same prompts and same teacher-forced candidates across all checkpoints, plotted in base coordinates.
- Dose-response steering with PPL tracked, and local finite-difference gain instead of endpoint differences.
- Separate discovery data from post-training data (they checked overlap: 39 exact matches in the first 100k).

## Pitfalls

- Update size and behavior change are not matched between SFT and DPO: DPO uses lr 5e-7 for 60 steps vs SFT lr 1e-5 for 100-600 steps; repetition DPO sees only ~1% of its data (0.010 epoch); sycophancy DPO moves Syco Prob by about 0.015. 'Geometry preserved' may just mean 'little changed'. — "In particular, reward-only changes are usually modest in all three measured axes, so the figures support geometric conservation more directly than a universal large occupancy- or gain-shift account." (p45)
- Sycophancy SFT runs are many epochs on 3,480 examples (high overfit), DPO about 2 epochs. — "SFT-up runs for 120 steps (∼4.4 effective epochs) and SFT-down for 360 steps (∼13.2 effective epochs)." (p64)
- NOC as implemented is an unsigned magnitude proxy, not the signed contribution defined in Section 3.3, so it cannot show opposing/cancelling contributions. — "The implemented proxy is non-negative and does not preserve the sign or direction of an individual coordinate's contribution." (p12)
- L1 coordinate selection on unstandardized features. — "Because L1 regularization is not scale-invariant, the coordinate selection described below may be influenced by the relative scales of the per-neuron features." (p61)
- 3-vs-1 classifier: non-stance tokens of both classes are negatives, so selected neurons can partly code 'stance span vs rest of text', not sycophancy. — "Other-region features from sycophantic records (negative, label 0): activation vectors from all_except_answer_tokens for _rejected records." (p58)
- The chart is a PCA of the scaffold, not a discriminant subspace. In ACT the sycophancy PC1 is nearly orthogonal to the classifier direction (mean signed cosine 0.062, several 0.000, across the 23 atlas models), so the dominant chart direction is not the behaviour direction; the behaviour direction may sit in lower PCs. Chart rotation therefore need not mean the behaviour direction rotated (and vice versa). — "denotes the reported signed cosine similarity between PC1 and the supervised classifier direction." (p28)
- Sycophancy labels come from a fixed preference dataset of user opinions/claims, teacher-forced; no model-generated flips and no check of what the model itself believes. — "We constructed the sycophancy data from DataCreatorAI/Anti-Sycophancy-DPO, which provides 3 480 chosen–rejected preference pairs." (p43)
- Public Base vs Instruct comparisons are confounded. — "The comparison is paired at the model-family and parameter-scale level, but it is not a controlled training intervention." (p41)
- Discovery and intervention prompts not fully disjoint. — "Because this audit covered only a subset of the intervention corpus, we do not claim strict prompt-level disjointness." (p59)
- Layer counts of selected neurons are not causal importance. — "they should not be interpreted as direct measurements of layer-wise causal importance." (p34)
- Repetition DPO is ~0.010 epoch, so the repetition 'reward preserves geometry' result rests on a tiny update. — "All DPO runs complete 60 optimization steps, corresponding to an effective epoch count of ∼0.010 (60 × 128 / 755 920)." (p63)
- Measurement floor not reported: DPO-up and DPO-down sycophancy are identical at step 6 (S 0.9498, 9.06 deg, Syco Prob 0.3550) and the SFT first checkpoint also sits at ~9 deg, so ~9 deg / S~0.95 looks like a floor from chart re-estimation or checkpoint conversion. There is no base-vs-base (re-extraction or seed) control for S or angle. After step 6, DPO moves <=1 deg. — "Behavior moves gradually, but overlap remains around 0.94–0.95." (p16)
- Per-checkpoint chart pipeline may re-fit the sparse classifier and neuron set ('train or reuse'), then zero-pad bases to a common ambient space; it is not stated which checkpoints reuse the base mask, so S and angle mix neuron re-selection with representation change. — "extract activations for the base and checkpoint, train or reuse sparse classifiers, extract selected neurons, compute global padded indices, run PCA, zero-pad bases to a common ambient coordinate system" (p14)
- The two sycophancy behaviour metrics disagree for SFT-down: p_b (prob. of sycophantic target) rises to 0.442 vs 0.355 at the first DPO checkpoint, while the preference score falls to 0.330. Base-model behaviour is never reported. — "The Behavior entry reports pb /mb , where pb is the evaluation probability assigned to the behavior-positive target and mb is the realized behavioral metric." (p14)
- Overclaim in the headline: Figure 1 and the abstract say reward optimization redistributes occupancy and changes readout gain or threshold, but threshold is never estimated and Appendix S finds occupancy and gain changes for DPO are small and estimator-dependent. — "whereas reward-based optimization largely preserves the inherited chart while redistributing occupancy and modifying readout gain or threshold" (p1)
- Internal inconsistencies: Table 2a random baselines (0.367/0.317 primary CCA) conflict with Table 15 (0.667/0.817); Figure 5a repetition values do not match Table 2b; Table 6 Truth PPL 1485.1 vs Table 2b 1.48; Table 7 up/down step-6 rows identical; Figure 11/13 PC1 25.5% vs Table 4 Llama-3.1-8B ACT PC1 0.426; Figure 5d SFT axis ends at 120 steps while SFT-down ran 360. Single model, single seed, no confidence intervals; Appendix I robustness controls described without reported numbers. (p7 and elsewhere)

**Limitations (quote, p7):** "geometric conservation versus movement, but they do not by themselves prove which non-geometric mechanism causes the remaining behavioral change."  
No dedicated limitations section. Other admitted gaps: 'Threshold changes, subtle readout reconfiguration, or distributional changes outside the second-moment occupancy statistic remain plausible mechanisms' (p45); repaired repetition continuations not re-checked for repetition (p60); public pairs confounded (p41).

**Future work:** None stated (no future-work section; the paper ends at a two-sentence conclusion on p7-8).

## Figures and tables (every manifest item, plus Figure 11)

| ID | Page | What is plotted | Takeaway | Numbers | Text matches figure |
|---|---|---|---|---|---|
| Figure 1 | 1 | Pipeline schematic: token windows, ACT/NOC decomposition of FFN, sparse BAC selector, low-rank behavioral chart, ACT-NOC quadrant, cross-model CCA atlas, SFT vs RL chart cartoons. | Overview only; SFT pictured as reshaping the chart, RL as keeping it. | No data. | yes (schematic) |
| Figure 2 | 4 | Four-quadrant diagram: NOC alignment (x) vs ACT alignment (y): runtime-associated, active, weak, latent mechanism. | Frequent activation (ACT) and functional contribution (NOC) are separate axes; 'high NOC, low ACT' = latent pathway. | No data. | yes |
| Figure 3 | 4 | One repetition token trajectory projected onto the ACT chart (PC1 vs PC2) over a log-density-ratio background. | Trajectory goes from prefix/ordinary generation (blue basin) through lead-in to the established-repetition (red) region. | PC1 77.0% var, PC2 6.3% var; energy colour scale -6 to 6 (approx, read from figure). | yes |
| Table 1 | 6 | Per-model BAC counts and ACT chart statistics (G_uniF, G_occ, G_causal, PC1, Top-3, k0.9) for repetition and sycophancy across 23 models. | Both behaviors sit on small scaffolds with strongly compressed charts; sycophancy scaffold is larger. | Repetition #bac 20-61 (mean 37.04, median 34), mean G_uniF ACT 7.16; sycophancy #bac 42-101 (mean 66.09, median 62), k0.9 mean 9.17. | yes |
| Figure 4 | 6 | Repetition rate (left axis) and generated-token PPL (right) vs ACT-chart steering strength alpha in {-2..2} for Llama-3.1-8B, 8B-Instruct, 3.3-70B-Instruct. | Monotone dose-response, PPL flat. | R(+2)-R(-2): 8B-Instruct 0.142, 70B-Instruct 0.058, 8B 0.033 (text p38); axis 0.63-0.93. | yes |
| Table 2 | 7 | (a) Cross-model sycophancy atlas (classifier correlation, primary/mean CCA, top-2-PC, k-dim probe; BAC vs random; same/cross family/base-instruct). (b) Endpoint geometry S/angle, R^G, behavior, PPL- for SFT, DPO, SFT->DPO up/down, repetition and sycophancy. | DPO from base keeps the base chart (S 0.92-0.96); SFT and SFT->DPO move it (S 0.60-0.80). | Behaviour column = p_b/m_b (prob. of sycophantic target / sycophantic preference). Sycophancy: SFT up 0.773/24.69 deg, R^G 0.966, behav 0.886/0.646; SFT down 0.804/22.04, 0.814, 0.442/0.330; DPO up 0.938/10.10, 0.998, 0.370/0.474; DPO down 0.939/9.68, 0.995, 0.342/0.436; SFT->DPO up 0.721/29.03, 0.885, 0.851/0.666; SFT->DPO down 0.695/30.88, 0.819, 0.389/0.297. Random columns in (a): primary CCA 0.367 ACT / 0.317 NOC, top-2-PC 0.266/0.295. | partly: the random-baseline column conflicts with Table 15 and Appendix O text (random primary CCA 0.667/0.817, random top-2 NOC 0.495); SFT->DPO-up PPL- is 1.48 here but 1485.09937 in Table 6; also: SFT-down raises p_b to 0.442 (vs 0.355 at the first DPO checkpoint) while its preference falls to 0.330, so the two behaviour metrics disagree in direction for SFT-down |
| Figure 5 | 7 | (a,c) Behavior vs mean principal angle from base chart across checkpoints for SFT, RL(DPO), SFT->RL; (b,d) projector overlap S(0,t) vs checkpoint step, up and down. | DPO overlap stays ~0.94-0.96 throughout; SFT drops fast; SFT->DPO stays at the SFT level. | Sycophancy-up: SFT 9 deg->24.7 deg, preference 0.45->0.645; RL 9->10.1 deg, preference 0.454->0.474; SFT->RL 24.7->26.1 deg, 0.646->0.663. Sycophancy overlap endpoints: SFT up ~0.77, down ~0.80; RL ~0.94; SFT->RL 0.70-0.72. Repetition-up: SFT ends ~38 deg at n-gram rate ~0.85; RL 3.5-3.6 deg, 0.31->0.40; SFT->RL ~36 deg, ~0.94 (approx, read from figure). Both SFT (step 10, lr 1e-5) and RL (step 6, lr 5e-7) first stored checkpoints already sit at ~9 deg / preference ~0.454 in panel (c); in panel (d) every lineage drops from S=1.0 to ~0.95 at its first checkpoint (re-read at 220 dpi). | partly: sycophancy panels match Table 2 except SFT->RL-up angle (~26 vs 29.03); repetition panel (a) does not match Table 2 (SFT angle ~38 vs 27.78, SFT->RL rate ~0.94 vs 0.565) and (b) x-axis runs to 300 steps while SFT-up repetition ran 100 steps; panel (d) SFT x-axis ends at 120 steps for both directions although SFT-down ran 360 steps; the identical ~9 deg first-checkpoint offset for a 20x-smaller-lr DPO run suggests a measurement floor (chart re-estimation or checkpoint conversion), not objective-driven movement |
| Table 3 | 16 | Representative ACT repetition strength (G_uniF, G_occ, G_causal, T_base, N_eff) for 12 models. | Subset of Table 9. | e.g. Llama-3.1-8B G_uniF 3.63, G_causal 0.089; Qwen2.5-0.5B-Inst G_causal 0.599. | yes |
| Table 4 | 17 | Selected sycophancy PC1 variance and 90%-variance dimension in NOC and ACT for 8 models. | ACT more compressed than NOC. | Llama-3.1-8B NOC PC1 0.309 (Dim90 29), ACT PC1 0.426 (Dim90 8); Qwen2.5-32B ACT PC1 0.795 (Dim90 3). | yes |
| Table 5 | 17 | Final controlled repetition runs: S, angle, d_proj, R^G, G, repeat prob, PPL, n-gram rate. | Reward (DPO) runs keep geometry (S 0.92-0.96); SFT and SFT->Reward do not. | S: SFT up 0.693, down 0.711, Reward up 0.960, down 0.920, SFT->R up 0.598, down 0.604; R^G 0.938, 0.942, 1.096, 1.049, 0.900, 0.705; n-gram 0.723, 0.372, 0.403, 0.434, 0.942, 0.565. | yes (matches Table 2b) |
| Table 6 | 18 | Final controlled sycophancy runs: S, angle, d_proj, R^G, G, Syco Prob, Truth PPL, Syco Pref. | Reward-only keeps geometry but moves behavior little. | Syco Prob: SFT up 0.886, SFT down 0.442, Reward up 0.370, Reward down 0.342, SFT->R up 0.851, SFT->R down 0.389; Truth PPL 11.72, 1.28, 6.48, 5.69, 1485.10, 1.38; native G 10.96, 17.74, 13.88, 13.89, 10.40, 15.81. | partly: Truth PPL 1485.09937 for SFT->Reward-up vs 1.48 in Table 2b |
| Table 7 | 18 | Reward-only sycophancy trajectory at steps 6, 30, 60: S, angle, Syco Prob. | Geometry nearly fixed while behavior moves by about 0.015. | Reward-up: S 0.9498->0.9382, angle 9.06->10.10, Syco Prob 0.3550->0.3702; Reward-down: 0.9498->0.9389, 9.06->9.68, 0.3550->0.3416. | yes, but step-6 rows of up and down are identical to 4 decimals (0.9498, 9.06, 0.3550) for two opposite objectives, and the SFT first checkpoint in Figure 5c also sits at ~9 deg: most of DPO's reported displacement (9.06 of 9.68-10.10 deg) is present after 6 tiny steps in either direction, so it looks like a measurement floor; objective-driven DPO movement after step 6 is <=1 deg |
| Figure 6 | 21 | Repetition chart samples in Fisher direction e1 vs orthogonal PCA e2, ACT and NOC, with Fisher and logistic arrows. | Classes form separate clusters in both spaces. | ACT negatives near (-0.6, 2.3), positives near (0.6, -2.5); NOC negatives near (-0.08, -0.28), positives (0.08, 0.3) (approx, read from figure). | partly: in ACT the logistic arrow is very short and not visibly along e1 |
| Figure 7 | 22 | Sycophancy chart samples (Fisher e1 vs orthogonal PCA e2), ACT and NOC. | Sycophantic vs truthful stance tokens form overlapping-but-shifted clouds. | ACT negatives centered ~(-0.08, 0.7), positives ~(0.07, -0.6); NOC negatives ~(-0.025, -0.06), positives ~(0.02, 0.05) (approx, read from figure). | yes |
| Figure 8 | 23 | Six repetition token trajectories in ACT chart over density-ratio landscape. | Repeated spans move into and stay in the positive region; entry often starts at the first repeated unit, before the detector fires. | PC1 77.0%, PC2 6.3%. | yes |
| Figure 9 | 24 | Three more ACT repetition trajectories incl. a paragraph-level case. | Entry into the positive region can be direct or curved. | Same axes as Figure 8. | yes |
| Figure 10 | 25 | Six matched repetition trajectories in NOC chart. | NOC chart is dominated by PC1; compact negative region, separate positive region. | PC1 84.6%, PC2 3.3%. | yes |
| Figure 11 | 26 | Three sycophancy token trajectories in ACT chart (not listed in manifest). | Pre-onset tokens already drift toward the positive region; sycophantic span sits in positive region; paths are 2-D and heterogeneous. | PC1 25.5%, PC2 13.9%. Figure 13 uses the same axes (25.5%/13.9%) and is explicitly the base Llama-3.1-8B chart, so Figure 11 is very likely Llama-3.1-8B too; this conflicts with Table 4 (Llama-3.1-8B ACT PC1 0.426). Possibly token-level vs span-pooled PCA; the paper does not say. | yes for the figure itself; partly vs Table 4 (PC1 25.5% vs 42.6% for what appears to be the same model and space) |
| Figure 12 | 26 | Same sycophancy instance in ACT and NOC charts. | Path shapes differ; sycophantic span lands on positive side in both. | ACT PC1 25.5%/PC2 13.9%; NOC PC1 28.2%/PC2 12.3%. | yes |
| Figure 13 | 27 | Base-anchored ACT chart: teacher-forced truthful (blue) and sycophantic (red) candidate prefix paths for 4 fixed prompt pairs, columns Base, RL-up, SFT-up, SFT->RL-up. | Base: sycophantic path heads into positive cloud. RL-up: paths stay inside base support. SFT-up and SFT->RL-up: both candidates start upper-left (~PC1 -2.5, PC2 +1) and end together near the negative cloud, so the base chart no longer separates truthful from sycophantic for these items. Also RL-up: the two candidates are not clearly separated either (both end ~PC1 0-1). Tension: SFT-up R^G is 0.966 at the population level, yet these 4 items look poorly organised by the base chart. | 4 pairs only; PC1 25.5%, PC2 13.9% (approx, read from figure). | yes |
| Table 8 | 29 | Initial repetition probe accuracy and #BAC for 7 models. | Small scaffold predicts on-policy repetition. | Accuracy 0.9646-0.9879, mean 0.9767; #BAC mean 33.4. | yes |
| Table 9 | 29 | Full ACT repetition-chart statistics, 23 models. | Chart strength varies widely; Llama has the weakest G_uniF. | Mean G_uniF 7.16, G_occ 0.84, G_causal 0.191, T_base 0.856; Llama-3.1-8B G_uniF 3.63. | yes |
| Table 10 | 30 | Full NOC repetition-chart statistics, 23 models. | Similar to ACT; causal gain smaller. | Mean G_uniF 7.02, G_causal 0.093. | yes |
| Table 11 | 31 | Sycophancy ACT PCA statistics (#BAC, PC1, Top-3, k0.9, k0.9/N, cosine between PC1 and classifier direction). | ACT sycophancy charts are very compressed, but PC1 is almost orthogonal to the classifier direction. | Mean PC1 0.471, k0.9 9.17, k0.9/N 0.144; Cos mean 0.0623, median 0.0445, several exactly 0.0000. | yes |
| Table 12 | 32 | Sycophancy NOC PCA statistics. | NOC less compressed but PC1 better aligned with classifier. | Mean PC1 0.306, k0.9 25.35; Cos mean 0.4565, median 0.5667; Mistral-Nemo-Base -0.1646. | yes |
| Table 13 | 33 | Unified NOC statistics: repetition (left) and sycophancy (right): G_uniF, G_occ, G_causal, PC1, Top-3, k0.9. | Sycophancy NOC causal gain is tiny and sometimes negative. | Sycophancy NOC mean G_uniF 8.19, G_occ 0.565, G_causal 0.062 (median 0.030); Qwen2.5-7B -0.0009, Qwen2.5-32B-Instruct -0.0007; Llama-3.1-8B 0.040. | yes |
| Table 14 | 34 | Model ID letters A-W for heatmaps. | Roster. | 23 models. | partly: Figure 17 uses a different A-W ordering (A = Gemma-2-2B) despite O.1 saying the same order is used |
| Figure 14 | 34 | Number of positive-weight BACs per layer in Llama-3.1-8B, repetition vs sycophancy. | Sycophancy scaffold is broad, with peaks in middle and later layers; repetition sparse. | Sycophancy peak ~8 neurons at ~layer 13, ~7 at ~layer 23, ~6 at ~layers 8 and 27; spread over nearly all 32 layers. Repetition at most 3 per layer (~layers 21, 23, 27) (approx, read from figure at 220 dpi). | yes |
| Table 15 | 34 | Mean off-diagonal cross-model alignment, BAC vs random, ACT and NOC. | Random primary CCA is high, so mean CCA and k-dim probing are the real tests. | Primary CCA 0.938/0.974 vs random 0.667/0.817; mean CCA 0.515/0.648 vs 0.222/0.233; top-2-PC 0.784/0.958 vs 0.266/0.495; k-dim 0.473/0.581 vs 0.215/0.234. | yes (but conflicts with Table 2a random column) |
| Table 16 | 34 | Alignment stratified by same family, cross family, base/instruct pairs. | ACT has larger family effect; base/instruct pairs most aligned. | Mean CCA ACT 0.627 same / 0.480 cross / 0.754 base-inst; NOC 0.677/0.639/0.798. | yes |
| Figure 15 | 35 | 23x23 heatmap of absolute correlation of sparse classifier scores between models. | Near-uniformly high. | Mean off-diagonal 0.950; base/instruct 0.980; cells mostly >0.9 (approx, read from figure). | yes |
| Figure 16 | 36 | 23x23 heatmaps: primary and mean CCA, ACT and NOC, sycophancy BAC charts. | Primary CCA saturated; mean CCA shows family blocks in ACT. | Primary ~0.9-1.0 everywhere; mean CCA ACT ~0.3-0.8 with family blocks; NOC more uniform (approx, read from figure). | yes |
| Table 17 | 36 | Aggregate repetition steering: mean/median Delta R, positive count, median /Delta PPL/. | Both operators raise repetition in all 23 models. | ACT mean Delta R 0.119, median 0.102, 23/23; NOC BAC 0.029, 0.019, 23/23; median /Delta PPL/ 0.223/0.239. | yes |
| Figure 17 | 37 | Random-coordinate primary and mean CCA heatmaps. | Random primary CCA is high (esp. NOC); random mean CCA low. | Random primary ~0.6-0.8 ACT, ~0.8-0.9 NOC; random mean ~0.2 (approx, read from figure). | yes |
| Table 18 | 37 | Mean endpoint repetition change by family. | Positive in every family. | ACT: Qwen 0.148, Gemma 0.127, Llama 0.078, Mistral 0.096; NOC 0.028, 0.036, 0.032, 0.020. | yes |
| Figure 18 | 38 | Directed cross-model linear association heatmaps (top-2-PC and k-dim, ACT and NOC). | Leading coordinate transfers strongly; full chart less. | Means 0.784/0.958 (top-2), 0.473/0.581 (k-dim). | yes |
| Figure 19 | 39 | Random-coordinate linear association heatmaps. | Random top-2 NOC moderately high; random k-dim ~0.23. | Random top-2 NOC 0.495; k-dim 0.215/0.234. | yes |
| Figure 20 | 40 | Qwen family repetition steering, ACT chart and NOC BAC. | ACT strongest for Qwen2.5-0.5B-Instruct. | Qwen2.5-0.5B-Instruct 0.322 -> 0.840; NOC best Qwen2.5-7B-Instruct +0.055. | yes |
| Figure 21 | 40 | Gemma family repetition steering. | 2B models respond most. | ACT endpoint +0.182 (2B), +0.238 (2B-it); NOC ~+0.07. | yes |
| Figure 22 | 41 | Llama family repetition steering, both operators. | All monotone. | ACT +0.142 / +0.058 / +0.033; NOC +0.056 / +0.021 / +0.018 (8B-Inst / 70B-Inst / 8B). | yes |
| Figure 23 | 41 | Mistral family repetition steering. | Large ACT changes for Small-24B-Instruct and 7B-Instruct. | Mistral-Small-24B-Instruct 0.599 -> 0.834; Mistral-7B-Instruct 0.757 -> 0.900; NOC max +0.041. | yes |
| Table 19 | 42 | Instruct-minus-base changes in log G_uniF and G_causal (NOC, ACT) for 10 pairs. | Reliability changes have mixed signs; causal gain more often rises after instruction tuning. | Mean Delta log G_uniF NOC -0.049, ACT -0.027; Delta G_causal NOC +0.027, ACT +0.074; Qwen-0.5B ACT gain +0.389; Gemma-27B NOC log G -0.549. | yes |
| Figure 24 | 42 | Cross-model repetition strength landscape: log G_uniF NOC vs ACT, colour = baseline repetition, bubble = causal gain. | Chart reliability, baseline tendency and causal gain are distinct. | Axes ~1.4-2.4; Pearson r ~0.64 (text). | yes |
| Figure 25 | 43 | Bar charts of Table 19 per pair. | Heterogeneous signs. | Same values as Table 19. | yes |
| Table 20 | 43 | Repetition dataset construction counts. | Balanced 2,736 records. | Pool 928,046; 1,368 + 1,368. | yes |
| Table 21 | 43 | Sycophancy stance-extraction outcomes. | 3,206 complete pairs, 6,678 records. | Both 3,206; chosen only 138; rejected only 128; neither 8. | yes |
| Figure 26 | 45 | Final-checkpoint normalized geometry / occupancy / gain bars (NOC geometry+occupancy, activation-level gain) for SFT, RL, SFT->RL in 4 conditions. | Geometry is large for SFT and SFT->RL, near zero for RL; RL also shows near-zero occupancy and gain. | Syco-up: SFT geom ~1.0, RL ~0.2, SFT->RL ~1.0; Syco-down: SFT geom ~0.8, gain ~0.55; SFT->RL gain 1.0; RL all three ~0-0.15; Rep-down SFT->RL occupancy ~-1.0, gain ~+0.6 (approx, read from figure). | yes |
| Figure 27 | 45 | Same as Figure 26 with chart-subspace gain. | Gain conclusions depend on estimator; syco-down SFT->RL gain drops from 1.0 to ~0.15. | Rep-up SFT->RL gain ~-0.8; Rep-down SFT->RL gain ~1.0; syco gains <=0.2 (approx, read from figure). | yes |
| Figure 28 | 46 | Ten-stage z-scored trajectories of geometry, occupancy, activation-level gain (ACT geometry/occupancy), 4 conditions x 3 lineages. | RL geometry stays lowest at every stage; SFT geometry rises by stage 2; sequential syco-down gain rises late. | RL geometry z ~-1.2 throughout; syco-down SFT->RL gain z 1.4 -> 2.8; syco-down SFT gain -0.6 -> 1.4; rep-down SFT gain dips to ~-2.9 at stage 2 (approx, read from figure). | yes |
| Figure 29 | 47 | Same as 28 with chart-subspace gain. | Subspace gain for sycophancy is flat (~0.5); big gain only for sequential repetition-down. | Rep-down SFT->RL gain z ~2.5; rep-down SFT gain dips to ~-4.3 at stage 2; syco gains ~0-0.6 (approx, read from figure). | yes |
| Figure 30 | 48 | Ten-stage trajectories, NOC geometry/occupancy, activation-level gain. | Same asymmetry in NOC; RL geometry slowly rises from ~-1.5 to ~-0.9. | Syco-up SFT geometry -1.2 -> 1.2; RL -1.5 -> -0.9 (approx, read from figure). | yes |
| Figure 31 | 49 | Ten-stage trajectories, NOC geometry/occupancy, subspace gain. | Geometry and occupancy conclusions stable across gain estimator. | As Figure 30 for geometry; gain as Figure 29 (approx, read from figure). | yes |
| Figure 32 | 50 | Stage 1-5 snapshot bars, ACT geometry/occupancy, activation gain. | SFT vs RL geometry separate from stage 1-2. | RL geometry bar ~-1.2 at every stage; syco-down SFT->RL gain grows ~1.4 -> 2.6 (approx, read from figure). | yes |
| Figure 33 | 51 | Stage 6-10 snapshot bars, ACT, activation gain. | Hierarchy stable; late gain growth in sequential down runs. | Syco-down SFT->RL gain ~2.8; rep-down SFT->RL occupancy ~-2.5, gain ~+1.6 (approx, read from figure). | yes |
| Figure 34 | 52 | Stage 1-5 snapshots, ACT, subspace gain. | Gain profile changes with estimator. | Rep-down SFT gain ~-4.3 at stage 2 (approx, read from figure). | yes |
| Figure 35 | 53 | Stage 6-10 snapshots, ACT, subspace gain. | Sequential repetition-down has the only large positive gain. | Rep-down SFT->RL gain ~+2.4 (approx, read from figure). | yes |
| Figure 36 | 54 | Stage 1-5 snapshots, NOC, activation gain. | Early geometry separation; occupancy transients for repetition-down. | Rep-down SFT occupancy ~-2.8 at stage 3 (approx, read from figure). | yes |
| Figure 37 | 55 | Stage 6-10 snapshots, NOC, activation gain. | Late gain growth only in sequential down runs. | Syco-down SFT->RL gain ~2.8 (approx, read from figure). | yes |
| Figure 38 | 56 | Stage 1-5 snapshots, NOC, subspace gain. | Most functional view; sycophancy bars all small. | Syco bars mostly within +-1.2 (approx, read from figure). | yes |
| Figure 39 | 57 | Stage 6-10 snapshots, NOC, subspace gain. | Large gain shifts are the exception, not a property of reward-only training. | RL bars within ~+-1; rep-down SFT->RL gain ~+2.4 (approx, read from figure). | yes |
| Figure 40 | 59 | Repeat of the repetition-strength landscape (Figure 24). | Same as Figure 24. | Same as Figure 24. | yes |
| Figure 41 | 64 | Local behavioral chart scatter (Fisher e1 vs orthogonal PCA e2) with Fisher and logistic arrows. | Supervised directions align with separation axis. | Same layout as Figure 7a (approx, read from figure). | yes |
| Table 22 | 66 | Per-model classifier row counts for sycophancy 3-vs-1 training. | 9 models have 13,356 rows, 14 have 10,708 (missing other-token activation files). | All models: chosen 3,344, rejected 3,334, pairs 3,206. | yes |
| Table 23 | 67 | Checkpoint step per matched stage s. | SFT 10s/60s/12s/36s; reward and SFT->reward 6s. | s in 1..10. | yes |
| Table 24 | 67 | Repetition post-training dataset configurations (SFT-up/down, RL-up/down). | DPO uses repetitive vs repaired responses as chosen/rejected. | 755,920 records each. | yes |
| Table 25 | 67 | Repetition repair data counts by stage. | 98.62% retained. | 743,687 -> 766,869 -> 766,469 -> 755,920. | yes |
| Table 26 | 67 | Selected coordinate counts for archived classifiers. | Sycophancy base classifier: 92 positive, 57 negative, 149 nonzero. | Repetition 25/9/34; hallucination 20/32/52; SAE-repetition 218/556/774. | yes |
| Table 27 | 68 | Repetition training configuration. | SFT lr 1e-5 (100/600 steps); DPO lr 5e-7, 60 steps, beta 0.1. | Batch 128, fp16, ZeRO-3, seed 42, 8 GPUs. | yes |
| Table 28 | 68 | Sycophancy training configuration. | SFT lr 1e-5 (120/360 steps); DPO lr 5e-7, 60 steps, beta 0.1 (0.7 for SFT->DPO down). | 3,480 examples / pairs; seed 42. | yes |

## Sections read

- Abstract
- 1 Introduction
- 2 Related Work
- 3 Behavioral Manifold Framework
- 3.1 Behavioral Geometry as a Post-Training Object
- 3.2 Behavioral Probes and Class-Conditional Representation Measures
- 3.3 Activation and Normalized Output-Contribution Spaces
- 3.4 Selecting a Coordinate Scaffold
- 3.5 Estimating the Local Behavioral Chart
- 3.6 Comparing Charts on the Grassmannian
- 3.7 Separability, Occupancy, and Gain
- 3.8 Causal Validation of Behavioral Charts
- 3.9 Post-Training Trajectories and Anchored Retention
- 4 Behavioral Manifolds Across Models
- 4.1 Behavioral Anchor Coordinates Support Sparse, Low-Dimensional Charts
- 4.2 Steering Establishes Causal Relevance
- 5 Rewrites or Reweights
- 6 Conclusion
- References
- Appendix A Derivation of the Local Behavioral Model (A.1 Binary Behaviors and Their Log-Odds, A.2 Local Linearization, A.3 Low-Dimensional Behavioral Chart, A.4 Equivalent Statistical Derivation from a Local Gaussian Model, A.5 Consequences for the Geometry-Occupancy-Readout Decomposition)
- Appendix B Repetition Detection and Feature Extraction (B.1 Periodic n-gram Repetition Detection, B.2 Token Regions, B.3 Raw Activation Space, B.4 Normalized Output-Contribution Magnitude, B.5 Implementation Notes)
- Appendix C Neuron Selection, Subspace Estimation, and Cross-Model Alignment (C.1 Sparse Classifier, C.2 Physical Index Recovery, C.3 PCA Behavioral Chart, C.4 CCA Alignment, C.5 Classifier-Direction Alignment, C.6 Directed Linear Probing)
- Appendix D Steering and Intervention Details (D.1 Multiplicative Neuron Scaling, D.2 Additive Subspace Steering, D.3 Two-Pass Autoregressive Subspace Steering, D.4 Generated-Text Metrics, D.5 Greedy Challenge Prompts)
- Appendix E Post-Training Geometry Metrics and Checkpoint Analysis (E.1 Projector Metrics Without Forming Full Projectors, E.2 Anchored Fisher Retention, E.3 Pairwise Universal Strength Aggregation, E.4 Causal Gain From Steering Curves, E.5 Controlled Training Pipelines, E.6 End-to-End Fine-Tuning Analysis Pipeline, E.7 Endpoint-Summary Reporting Conventions)
- Appendix F Geometry-Occupancy-Gain Profiles of Controlled Post-Training (F.1 Derivation of Occupancy and Gain Estimators)
- Appendix G Numerical Tables (G.1 Repetition Strength Across Models, G.2 Sycophancy Low-Rank Statistics, G.3 Full Controlled Repetition Summary, G.4 Full Controlled Sycophancy Summary, G.5 Checkpoint-Level Sycophancy Trajectory)
- Appendix H Derivation of Universal Fisher Strength (H.1-H.6)
- Appendix I Leakage Control and Robustness Analysis (I.1 Coordinate Selection Robustness, I.2 Subspace Estimation Robustness, I.3 Token Extraction Robustness, I.4 Cross-Model Alignment Robustness, I.5 Behavior Specificity Controls, I.6 Causal Intervention Controls, I.7 Post-Training Geometry Robustness)
- Appendix J Visualization of Behavioral Charts (J.1 Visualization Protocol, J.2 Repetition Behavioral Charts, J.3 Sycophancy Behavioral Charts)
- Appendix K Token-Level Dynamics in Behavioral Landscapes (K.1 Behavioral Energy Landscape, K.2 Token-Level Trajectory Construction, K.3 Visual Encoding, K.4 Repetition Trajectories in ACT, K.5 Repetition Trajectories in NOC, K.6 Sycophancy Trajectories in ACT, K.7 Matched Sycophancy Trajectory Across ACT and NOC)
- Appendix L Base-Anchored Sycophancy Paths Across Post-Training Regimes (L.1 Base-Anchored Coordinate System, L.2 Candidate-Path and Preference Encoding, L.3 Matched Candidate-Path Gallery, L.4 Cross-Pair Interpretation)
- Appendix M Full Behavioral Anchor Coordinate Statistics (M.1 Initial Repetition Probe, M.2 Repetition Geometry in ACT, M.3 Repetition Geometry in NOC, M.4 Sycophancy Geometry in ACT, M.5 Sycophancy Geometry in NOC, M.6 Unified NOC Statistics)
- Appendix N Layer-Wise Distribution of Behavioral Anchor Coordinates
- Appendix O Full Cross-Model Behavioral Atlas (O.1 Model Roster and Common Ordering, O.2 Evaluation Protocol, O.3 Aggregate Results, O.4 Classifier-Direction Alignment, O.5 Canonical-Correlation Atlas, O.6 Random-Coordinate CCA Controls, O.7 Cross-Model Linear-Association Atlas, O.8 Random-Coordinate Linear-Association Controls, O.9 Family Structure and Matched Base-Instruction Pairs, O.10 Interpretation: Shared Core, Non-Universal Residuals)
- Appendix P Cross-Family Repetition Steering Results (Qwen, Gemma, Llama, Mistral families)
- Appendix Q Cross-Model Repetition Strength and Public Alignment Effects (Q.1 Construction of the Cross-Model Strength Landscape, Q.2 Matched Base-Instruct Pairing Protocol, Q.3 Numerical Paired Results, Q.4 Pair-Level Interpretation, Q.5 Aggregate Interpretation, Q.6 Relationship to the Controlled Post-Training Study)
- Appendix R Dataset Sizes, Filtering, and Sample Exclusions
- Appendix S Complete Geometry-Occupancy-Gain Analysis (S.1 Matched Training Stages and Normalization, S.2 Final-Checkpoint Profiles, S.3 Complete Ten-Stage Trajectories, S.4 Every Matched Stage Snapshot, S.5 Synthesis Across Resolutions)
- Appendix T Sycophancy Span Annotation and Tokenizer-Specific Alignment
- Appendix U (letter inferred) Geometry, Baseline Tendency, and Causal Gain Are Distinct
- Appendix V Construction of Repaired Non-Repetitive Continuations
- Appendix W Sparse Classifier Training and Coordinate-Selection Details
- Appendix X (letter inferred) Cross-Model Sample Alignment, CCA, and Directed Probing
- Appendix Y Controlled Post-Training Configurations (Y.1 Joint interpretation of ACT and NOC, Y.2 Token Windows and State Extraction)
- Appendix Z Additional Related Work (Repetition and neural degeneration; Sycophancy and preference failures)
- Tables 22-28 (pages 66-68)
