# Group geometry methods for fates

Run date 2026-10-25. Inputs: 21 full readings (figures viewed), 5 independent checks, quote check (581/581 verbatim), scout pass-2 list. Citations are `slug p.N`, using short slugs (e.g. `c6-kappa p4`). Full slugs are in section 3. "Our inference" marks claims that are ours, not the paper's.

---

## 1. Short summary

- **Groups beat single directions mainly as a measurement, not yet as a mechanism.** Correct-answer information usually sits in a small subspace (2-8D, a mean shift between two clouds) at 46-75% depth (`c15-confidence-manifold p1, p4, p7`). Two class centroids match a trained probe (`c15 p1`).
- **The closest prior work is KAPPA.** The "correct answer" and "model's choice" subspaces sit at 60-80 deg mid-depth and jump to near 90 deg (random level) in late layers (`c6-kappa p5, p20`). An input-specific edit that copies knowledge coordinates into the choice subspace gains up to 22.2 pts agreement (`c6 p7`). It does no pressure, no flips and no per-item fates.
- **Decodable is not used, again and again.** A probe direction reads at AUROC >= 0.99, yet steering along it does nothing in 43 of 48 cells (`c9-lagged-coupling p1, p7`). Ablating the probe subspace costs 0.6 pp, while ablating a DAS subspace at the same layer drops accuracy 42% to 0% (`c7-readout-mediator-angle p5`). This is the bypass (fate 2) toolkit.
- **Tool by fate (detail in section 2):** readout = tuned-vs-logit lens gap and logit margin; bypass = matched-random steering nulls, probe-vs-DAS ablation, Fisher (pullback) metric overlap; not routed = KAPPA-style or Fisher injection; override = per-item differencing at head outputs plus conditional projection; transformed vs erased = Procrustes plus refit probes plus interchange tests, never CKA alone.
- **Training mostly reroutes, rarely erases.** In controlled post-training, information is kept but unused (SFT: decodable R=0.98, use U=0.15) and erased only under prolonged weight decay (`c8-represented-used p1, p5`). A frozen probe reads a 20-77 deg rotation as erasure (`c8 p13`). This is direct support for Stage B's "removed or hidden" check.
- **Location is not stable across models.** Best sycophancy probe depth ranges 12-fold across models, and the sign of the effect flips between Llama-2-70B and Llama-3-70B (`c11-trajectory-signatures p7, p8`). Circuit roles move between heads over training while behaviour stays the same (`c12-tigges p7, p8`).
- **3 best ideas (after the second skeptic pass):** copy-vs-correctness 2x2 (R1); plant known fates, now including an offset-only plant, before trusting any group metric (R3); output-level slope/intercept screen plus a margin-scaled threshold-push test (R4 + A4). All three are cheap (under 5 GPU-h together). Details in section 5.
- **Biggest risks:** (1) CKA can be pushed to 0 by shifting one subset with classes still >90% separable (`c13-cka-reliability p4, p7`), which is exactly what a flipped-only shift looks like; (2) Euclidean principal angles depend on coordinates, so measure in the output-Fisher metric or run reparameterisation tests (`w2-info-geometry-shared p1`); (3) two-answer flips give a degenerate 1-D "manifold" (`c5-manifold-steering`, reading note on p10 rank bound); (4) scoop risk, now read and checked: 2608.01835 takes the "rewrite vs reweight" framing for SFT vs DPO on sycophancy geometry (moderate risk on framing, must be cited), but has no decodable correct answer, no per-item fates and DPO barely changed behaviour, so Stage B's substance is still open (`p2-syco-manifold p7, p18, p43`); 2607.20146 claims "several sycophancy modes" in Gemma-2-9B-it (framing risk for H1, low scoop risk: no facts, no fates, no fixes; `p2-syco-modes p5, p19`). The highest remaining scoop risk is the intervention ladder (idea 5, 2610.21708).

---

## 2. Method menu

| Tool | What it measures | Fate it can diagnose | Causal follow-up | Cost | Pitfalls | Verdict | Sources |
|---|---|---|---|---|---|---|---|
| Two-centroid / PLS subspace (2-8D) | Mean shift between correct and wrong clouds | Readout (1), baseline for all | Steer along it with random and orthogonal controls | Very low | Directions fit on one dataset point elsewhere on others; teacher-forced detector loses 27 pts on own answers | Copy | `c15 p1, p4, p6, p26` |
| Paired knowledge vs prediction probes + principal angles | Angle between "correct answer" and "will say" subspaces per layer | Not routed (3), transformed (5) | KAPPA minimal-norm edit | Low (public repo) | At k << d almost all pairs sit near 90 deg; compare sum cos^2 against an exact null | Adapt (per item, flipped vs resisted) | `c6 p3, p5, p6, p8`; `c7` reading (null warning) |
| Matched-random steering null + dose-response | Write-in of a probe direction beyond random | Bypass (2) | Is the null itself; add DAS | Low | Noise floor can exceed effects at large sizes | Copy | `c9 p1, p8, p10` |
| Probe-subspace vs DAS-subspace ablation | Is the decodable subspace the causal one? | Bypass (2), H3 | Additive ablation drop vs random envelope | Medium | Optimised subspace can be an "output format" subspace; untrained models already decode (R^2 0.956) | Adapt | `c7 p5, p8` |
| Output-Fisher (pullback) metric G | Which residual directions change the output | Bypass (probe outside G's top eigenspace), readout | Fisher injection (FishBack / natural-gradient steer) | Medium (~280 partial forwards) | Rank-truncated at 8B and eigenspace accuracy never reported (checker); G depends on predictive entropy, so flipped vs resisted need entropy matching (checker, our extension); 8B gains vanish near 75% depth but GPT-2 gains grow with depth | Adapt | `w2-fishback p1, p7, p15, p17`; `w2-info-geometry-shared p4, p13` |
| Procrustes + refit probes + interchange | Rotation/re-encoding vs loss | Transformed (5) vs erased (6) | Interchange intervention on the refit coordinates | Low-medium | Do it within one model; cross-model Procrustes transplant fails (2.9% rescue) | Copy | `c8 p13`; `c13 p9`; `c7 p17` |
| CKA / RSA across pressures | Whole-cloud similarity | H1 only, as a supporting measure | None by itself | Low | Subset shift drives CKA to 0; blind to rotations; can be trained to any value with behaviour fixed | Avoid as decider; use with bootstrap and separate centring | `c13 p1, p4, p7, p8` |
| Grassmannian speed/curvature, phase breakpoints | Where the subspace rotates fastest along depth | Fork-point localisation | Patch at the bend layer | Low | One number is not enough; stable is not dispensable | Adapt (light) | `w2-layerwise-grassmann p5, p9`; `w2-readout-subspace p7, p8`; `u2-subtrackpp` (formulas only) |
| Tuned vs logit lens per item | Information present but not yet in output basis | Readout (1) | Margin push at late layers | Low | Lens trained on generic text | Adapt | `w2-readout-subspace p2, p9` |
| Head-level per-item differencing + conditional projection | Directions where failed runs diverge at a head | Override (4) | Remove the component only when large | Low-medium | Weak evidence; tuned on test set; 128-head multiple comparisons | Adapt (as baseline) | `c14-mags p1, p5, p7` |
| Activation manifold fit + isometry + on-manifold steering | Shared shape of activations and outputs | Transformed (off-manifold), not routed | Steer along manifold, score naturalness | Medium (code public) | Degenerate for two answers; needs added structure | Adapt | `c5 p6, p8, p9`; `w2-shape-of-beliefs p4, p5` |
| Invertible non-linear map trained by interchange | A non-linear causal feature | Bypass/override where linear fails | Single-site clamp | Medium | One author, two models, one task | Adapt method | `w2-nonlinear-interventions p1, p3, p4` |
| Probe-graph vs logit-graph attribution (transcoders) | Feature overlap between "knowing" and "saying" | Bypass (2) | Three-subset ablation | High (needs CLTs; only 1-2B available) | Size confound (logit graph 1.77-4.15x larger) | Adapt design; blocked at 9-27B | `u5-user p1, p5, p19` |
| Trajectory scalars (path length, curvature, kappa) | Path statistics over tokens/layers | Monitoring only | None | Low | Length confound; one direction beats them | Avoid as evidence; baseline only | `c11 p9`; `u3-geometry-of-divergence p4`; `c10-curved-inference-ii p16` |
| Structured role decomposition (ROLE) | Which role an item is bound to | Transformed (re-bound) | Constituent surgery | High | RNN-era; needs one vector per input | Copy validation logic only | `u4-role-learning p6, p7, p14` |

---

## 3. Per-paper table

Checked column: "yes" only for papers with an independent check report.

| Paper (slug) | What it shows | Models / scale | Figures that matter | Fate | Architecture location | Reuse | Checked? |
|---|---|---|---|---|---|---|---|
| c5-manifold-steering-2605.05115 | Activations and outputs trace the same curve; manifold steering beats linear (2.8x more natural) | Llama 3.1 8B (+70B appendix) | Isometry and E_BC figures (p6, p8) | 5, 3 (moderate) | Late residual, last token (L28/32) | Adapt | no |
| c6-kappa-2509.23782 | Knowledge vs prediction subspaces diverge to ~90 deg late; KAPPA edit +22.2 pts | Llama 3.1 8B, Qwen2.5 7B; Qwen3 4B/14B/32B | Angle-by-layer (Fig 2, 7) | 3, 4 | Residual, answer-prefix token; jump at ~55-65% depth | Adapt | no |
| c7-readout-mediator-angle-2605.29126 | Probe subspace near 90 deg from causal DAS subspace; probe ablation inert | Gemma 2 2B mainly; 1.5-9B | Ablation and angle figures (p5) | 2, H3 | L11-12 QK bottleneck (~45%), late MLPs | Adapt (controls) | no |
| c8-represented-used-2610.05292 | Post-training reroutes or rescales, rarely erases; KL anchor keeps use | GRU, Qwen3 1.7B-4B (synthetic) | Fate-vs-beta (p6), rotation (p13) | 2, 5, 6 (strong method) | Decision position; later-half rank falls 12-19% | Adapt design | no |
| c9-lagged-coupling-2609.01048 | Readable from step 1k, steering null in 43/48 cells | Pythia 160M-12B, OLMo-2 | Write-in vs noise floor (p10) | 2 | Residual, unreported layer | Adapt controls | no |
| c10-curved-inference-ii-2608.24037 | Trajectory "surface area" separates judge labels; weak stats | 1-3B (LLaMA, Gemma3-1b) | p14, p32 tables | none solid | Summed over layers | Avoid | no |
| c11-trajectory-signatures-deception-acl2026 | Sycophancy gives trajectory signal; polarity flips across models | Llama-2 7/13/70B, Llama-3-70B, Qwen-0.5B | Effect sizes (p7, p8) | monitoring | Best depth 5-62% | Avoid as evidence; heed warning | no |
| c12-tigges-circuits-across-training-2407.10827 | Function stable at head-class level; heads turn over | Pythia up to 2.8B | Head turnover (p7) | Stage B, 4 | Heads (IOI), MLP (SVA) | Adapt metrics | no |
| c13-cka-reliability-2210.16156 | CKA manipulable by subset shift; blind to rotation | CIFAR10 CNNs | p7 drop to 0 | 5, 6 (warning) | none | Copy the warning | no |
| c14-mags-2605.21770 | Head-level conditional projection; small gains | Llama-3.1-8B, Gemma, GPT-OSS | Fig 3 heads (p5) | 4 | Head outputs, L8-31 | Adapt (baseline) | no |
| c15-confidence-manifold-2602.08159 | Correctness = mean shift in 2-8D subspace; orientation does not transfer | 11 models up to 14B | Dim curves (p4), transfer (p6-7) | 1, H3 | Last-token residual, 46-75% depth | Copy | no |
| u2-subtrackpp-2502.01586 | Grassmannian gradient-subspace optimizer | RoBERTa, Llama-style 60M-3B | Fig 3, 4 | none | none | Math only | yes: OK (minor edits; "3B" config label unresolved) |
| u3-geometry-of-divergence-2608.30650 | Path statistics separate agent success/failure; small gaps, length confound | Qwen3-14B/32B, Llama-3.1-8B | p4 gaps | direction 3 | One residual layer ~50-66% | Adapt (light) | no |
| u4-role-learning-1910.09113 | Learned roles approximate encoder (94.83%); surgery works | GRU seq2seq; sentence encoders | Table 1, Fig 5 | 5 (analogy) | none | Copy validation | yes: NEEDS FIXES (minor wording) |
| u5-user-2608.27510 | Probe vs logit circuits dissociate; truthfulness is the shared exception | Gemma-2-2B, Llama-3.2-1B | p19 ablation table | 2 | MLP/transcoder features; truth ~L12-14 of 26 | Adapt design | no |
| w2-fishback-2605.17231 | Fisher-pullback steering lowers off-target KL 1.8-3.6x at 8B | GPT-2, Llama-3-8B, Qwen3-8B | Ratio by layer (p7) | 2, 3 | Residual. 8B: gain at early-mid layers, gone at ~75% depth (one sampled layer each). GPT-2: gain rises with depth (p17). Model-dependent | Adapt | yes: NEEDS FIXES (depth claim overstated; 8B eigenspace accuracy never measured) |
| w2-info-geometry-shared-2609.11063 | Activation coordinates arbitrary; output geometry shared; anti-sycophancy steer | Instruct models up to 1.8B; 10 models | Fig 2, 5 | 2, H1 caution | Residual ~45% | Borrow methods | yes: NEEDS FIXES (overreach on fate-4 "competing objectives" reading; missed FishBack 8B precedent) |
| w2-layerwise-grassmann-dynamics-2605.12714 | Subspace curvature peaks late in decoder LLMs | Embedders, base LLMs | Table 1 (p5) | fork point | Late depth | Formulas only | no |
| w2-nonlinear-interventions-2605.14749 | Non-linear single-site edit removes refusal; linear version fails | Llama-3-8B-Instruct, Qwen2.5-7B-Instruct | Fig 1, Table 3 | 2, 4 | Mid residual, one prompt token | Adapt method | yes: OK (Qwen chosen layer not the sweep peak; some strength labels generous) |
| w2-readout-subspace-grassmann-2605.09011 | Three depth phases; info present before logit lens sees it | incl. Qwen2.5-14B/32B | Phase plots (p7-9) | 1 | Phase 3 starts 60-89%, FFN-dominated | Partly | no |
| w2-shape-of-beliefs-2602.02315 | Beliefs on curved manifold; probes transfer locally; linear steering off-manifold | Llama-3.2-1B base (+8 to 8B geometry only) | Fig 3-6 | 5 | Late layers | Adapt | yes: NEEDS FIXES (base models only, a transfer risk; Fig 3A lacks controls) |

---

## 4. Patterns across papers (facts)

1. **Decodable early, causal late or never.** Probes read from step 1k at AUROC >= 0.99 while steering stays null (`c9 p7, p1`). An untrained Pythia gives probe R^2 0.956 (`c7 p8`). A tuned lens finds the final token from layer 1 while the logit lens stays near 0.03-0.10 (`w2-readout-subspace p2, p9`).
2. **The probe subspace and the causal subspace are different.** Probe ablation costs 0.6 pp vs DAS 42% to 0% (`c7 p5`). Probe-only feature ablation flips 0% of tokens, logit-only flips 92-100% (`u5 p1, p24`). Truthfulness is partly the exception: 25% flips from probe-only nodes (`u5 p19`).
3. **Knowledge and action separate geometrically in late layers.** Knowledge-prediction angle goes from 60-80 deg to ~90 deg (`c6 p5`). Decoder subspace curvature peaks late (`w2-layerwise-grassmann p5`); Phase 3 (last ~21% of depth) is FFN-dominated (`w2-readout-subspace p7, p8`).
4. **Correctness is low-rank and a mean shift.** 2-8D peaks, unimodal classes (`c15 p4`); direction orientation, not dimension, is what fails to transfer (`c15 p6, p7`). KAPPA subspaces are also task-specific (`c6 p8`).
5. **Steering along a curve or under the right metric beats straight Euclidean steering, mostly early-to-mid depth.** Manifold steering is 2.8x more natural (`c5 p8`); linear steering goes off-manifold (`w2-shape-of-beliefs`, p5-6 per reading); Fisher steering 1.8-3.6x lower off-target KL at early/mid layers on 8B, not significant deep (`w2-fishback p1, p7`), though on GPT-2 the gain rises with depth (`w2-fishback p17`, flagged by the checker); natural-gradient anti-sycophancy steer 8-74x lower off-target divergence (`w2-info-geometry-shared p11`); the linear-restricted version of a non-linear edit does nothing (`w2-nonlinear-interventions p4`).
6. **Training hides more than it removes.** SFT keeps information decodable but unused (`c8 p5`); erasure needs prolonged weight decay (`c8 p1`); functions migrate between heads with self-repair (`c12 p7, p8`).
7. **Similarity scores can mislead.** CKA to 0 with >90% separability (`c13 p7`); variance share falls while information stays (153 of 172 runs keep R >= 0.9) (`c8 p7`); Euclidean geometry is not fixed by behaviour (`w2-info-geometry-shared p1`).
8. **Location varies across models.** Best probe depth varies 12-fold, polarity flips at same size (`c11 p8`); truthfulness belief layer does not replicate from Gemma to Llama (`u5` reading, p24-29); best correctness layer 46-75% (`c15 p7`).
9. **Output-based confidence fails exactly where we need it.** On TruthfulQA, P(True) and entropy fall to 0.44-0.64 while probes reach 0.80-0.94 (`c15 p6`); Qwen2-7B P(True) 0.48 vs centroid 0.93 (`c15 p21`).
10. **Commitment can look abrupt while drift is smooth.** Flip rate drops from 0.519 to 0 as the margin ratio crosses 1 (`w2-info-geometry-shared`, Supp. Note 9); use switches at a threshold in beta while decodability varies smoothly (`c8 p6`).

---

## 5. Ranked ideas

Second skeptic pass, 2026-10-25 (`ideas/skeptic_pass2.md`). It re-checked the six pass-1 ideas (R1-R6) against the two new papers and put the seven architecture-lens ideas (A1-A7, `ideas/actionability.md`) through the skeptic for the first time. Duplicates are merged, leaving 8 ideas. Order = evidence x actionability x feasibility, with prerequisites first. Every idea is a hypothesis. Costs are rough A100 estimates at 9B. Short slugs for the new papers: `p2-syco-manifold` = `p2-sycophancy-manifold-posttraining-2608.01835`; `p2-syco-modes` = `p2-separable-sycophancy-modes-2607.20146`. The skeptic checked citations against the reading JSONs and check files, not the PDFs.

**1. Copy-vs-correctness 2x2 (R1)**
- Idea: after pushback, "the probe still reads Canberra" may only be the model's own earlier answer sitting in the context. Cross "earlier answer right or wrong" with "fact true or false", and test whether a truth probe still works on the cells where the two disagree.
- Why promising: every fate call assumes the probe reads correctness (contradictions C7). `p2-syco-manifold` labels teacher-forced dataset text, so it never separated text from belief (`p2-syco-manifold p43`). Moving from teacher-forced answers to the model's own answers drops AUROC from 0.957 to 0.688 (`c15 p26`).
- First experiment: Gemma-2-9B-it, the 400 TriviaQA pushback pairs from 2604.19117. Build the four cells by prefilling a right or wrong first answer. Train a probe on the disagreeing cells at mid-depth.
- Kill: AUC on the disagreeing cells >= 0.8 means the threat is dead. Below 0.65, Stage A's probe premise fails.
- Cost: 1-2 GPU-h. Scoop: low.
- Skeptic: a control, not a finding, but skipping it could sink every later result.
- Serves: all fates; prerequisite for H1-H3.

**2. Plant known fates before trusting any group metric (R3)**
- Idea: write fake bypass, rotation, sign-flip, offset-only and erasure edits into real 9B activations. Score which tools call each one correctly: principal angles, CKA, RSA, Procrustes, frozen and refit probes.
- Why promising: CKA can fall to 0 while classes stay >90% separable (`c13 p7`). A frozen probe reads re-encoding as erasure (`c8 p13`). A constant offset alone sends probe transfer to chance (`p2-syco-modes p33`). A subspace's top PCA direction can be nearly orthogonal to the behaviour direction (cos 0.062; `p2-syco-manifold p28`), so "the subspace rotated" need not mean "the behaviour direction rotated".
- First experiment: about 5k partial forwards on Gemma-2-9B-it; six plant types, three doses each.
- Kill: any metric that miscalls more than 20% of plants is barred from fate calls.
- Cost: under 1 GPU-h. Scoop: none.
- Skeptic: the plants are linear and clean, so passing is necessary, not sufficient. The offset-only plant is new (from `p2-syco-modes`).
- Serves: tool validity for all six fates; prerequisite for H3.

**3. Output-level screen plus a threshold-push test (R4 + A4)**
- Idea: first, fit flip probability against the pre-pushback logit margin, per pressure. A shifted intercept points to override (fate 4); a flatter slope points to a readout/threshold failure (fate 1). Then the causal step: a late-layer push along W_U[own earlier answer] − W_U[user's claim], with the dose scaled to each item's margin so low-margin items get no free win.
- Why promising: the screen is nearly free. A flip needs the top answer's probability to move by at least half the margin (`w2-info-geometry-shared p72`; a necessary condition only). The last phase of depth is MLP-dominated in all 8 models studied (`w2-readout-subspace p7-8`). At 8B a probe beats the logit margin by only +0.008 AUC (2609.04582, round 2). `p2-syco-manifold` defines a threshold term but never estimates it (`p2-syco-manifold p45`).
- First experiment: Gemma-2-9B-it, logits for the competing-answer and user-pushback conditions; logistic fit with a pressure interaction. Then margin-scaled pushes at 80/90/95% depth with matched random nulls, measuring breakage on resisted items, off-target KL on neutral prompts, and **blocked correct updates** on a new "model wrong, user corrects it" item set.
- Kill: screen: both pressures share slope and intercept within bootstrap ranges and a mixture adds nothing (BIC). Push: at margin-scaled dose, rescue differs by less than 10 pp between threshold-labelled and other items; or threshold items are under 10% of flips; or the push blocks more than 5% of correct user corrections.
- Cost: about 0 GPU-h for the screen; under 3 GPU-h for the push. Scoop: low.
- Skeptic: the push direction uses the model's own earlier answer; using the gold answer would be an oracle no lab could deploy. Without the margin scaling, the A4 kill could not fail. The deployable form ("stick to your earlier answer") trades sycophancy for stubbornness; the correct-update set measures that trade.
- Serves: fate 1 vs fate 4; H1 at the output level; a cheap Stage B arm (H2).

**4. Monitor survival: frozen / recentred / Procrustes / refit probes (R2 + A5)**
- Idea: on flipped items, before and after one LoRA-DPO run, compare four probes: a frozen no-pressure probe; a recentred probe (only the bias refit); a within-model Procrustes-aligned probe (fit on resisted items); a fully refit probe. Gap closed by recentring = just an offset; closed by Procrustes = transformed (fate 5); closed by nothing, and patching cannot bring it back = erased (fate 6).
- Why promising: a frozen probe reads re-encoding as erasure (`c8 p13`, toy models). Transfer failure is about direction, not absence (`c15 p6-7`). Offset alone breaks transfer (`p2-syco-modes p33`). After SFT, `p2-syco-manifold`'s old-probe retention R^G is 0.81-0.97, and its 4-item Figure 13 shows single items where the old subspace stops separating (`p2-syco-manifold p7, p27`). The one round-2 erasure claim had a below-chance probe (2607.00415).
- First experiment: Gemma-2-9B-it, 400 items, flipped vs resisted matched on margin, cross-validation grouped by fact, at the depths from idea 5 (or 45% and 65%). Repeat after one LoRA-DPO run (shared with idea 8).
- Kill: the frozen probe stays within 0.03 AUROC of the refit probe on flipped items and after DPO (fates 5 and 6 are empty, and current monitors are fine); or every probe, including a non-linear one, is at chance.
- Cost: about 2 GPU-h plus the shared DPO run. Scoop: medium (R^G is close in spirit, but on teacher-forced labels, not the model's own failures).
- Skeptic: without margin matching, the refit gap is just a difficulty gap. The c8 rotation evidence is toy and benign. Use idea 2's plants to set the offset and rotation thresholds.
- Serves: fates 5 and 6; H3 directly.

**5. Intervention ladder + rescue band at sublayer resolution (R5 + A1 + A2's three tracks)**
- Idea: on the same items, run three rungs and compare their fate labels: probe-direction steering (matched nulls, dose curve); subspace (KAPPA-style) injection; patching in the item's own no-pushback state. Then find the depth band where injection still rescues, at attention-vs-MLP resolution (`resid_mid` / `resid_post`). Only subspace injection sets the band's upper edge.
- Why promising: probe ablation costs 0.6 pp vs DAS taking accuracy from 42% to 0% (`c7 p5`). Probe-direction steering is null in 43 of 48 cells (`c9 p1`). Single-site edits work best at middle depth (`w2-nonlinear-interventions p4`, refusal task). Layer priors for the same model: geometry peaks at L18, causal necessity at L22-26, output divergence from L23 (`p2-syco-modes p18-19`). The KAPPA fork is at 53-68% depth (`c6 p21`, corrected by its check).
- First experiment: Gemma-2-9B-it, 400 pairs matched on margin; 3 rungs x sublayer hook points over 25-75% depth first, with matched-norm random nulls and a noise floor (`c9 p8-10`). Add one joint cross-layer ablation rung, so a small ablation effect is not misread as bypass when it is really redundancy (`p2-syco-modes` pitfall, p16).
- Kill: rung labels agree for >= 80% of items (κ >= 0.6), so the tool does not set the fate; or the band's IQR is over 20% of depth or moves more than 15% between 9B and 27B (drop the WHERE claim); or injection rescues equally at every depth.
- Cost: 5-10 GPU-h at 9B, about 3x at 27B. Scoop: high (2610.21708; "decodability is not use" is crowded).
- Skeptic: lead with the per-item band, not the ladder. Full-state patching late in the network rescues by construction, so it cannot set the band's top. The "fork" may be the onset of anisotropy (c6 check), so add probe-vs-probe and random-angle ceilings. The 35-65% prior comes from other tasks.
- Serves: fates 2 and 3 (and 1 at the band's top); H3; the architecture WHERE.

**6. One fix or one per pressure (A7)**
- Idea: fit the fate-matched fix (KAPPA-style injection or a head-class projection) on one pressure and apply it to the other. The 2x2 rescue matrix, with side-effect accounting, says whether one anti-sycophancy fix covers both pressures.
- Why promising: KAPPA subspaces are task-specific (TruthfulQA to BBH-Algo transfer costs −12.02 agreement; `c6 p8`). Correctness directions from different datasets are near-orthogonal (|cos| 0.08; `c15 p7`). `p2-syco-modes` finds mode-specific dimensions on top of a shared direction (17/13/8 of 50 dims shared at L18; `p2-syco-modes p19`), but from personas, not pressures. The 2601.11563 shared-subspace claim is visual PCA only.
- First experiment: Gemma-2-9B-it, 200 items per pressure on shared facts, fix fit inside idea 5's band. Add a **same-pressure, different-wording** transfer as the ceiling.
- Kill: cross-pressure rescue is within 20% of the same-pressure, different-wording ceiling in both directions; one fix then suffices.
- Cost: under 3 GPU-h. Scoop: medium; neither new paper does it.
- Skeptic: without the wording ceiling, a transfer failure is just a prompt-format difference. Needs >= 100 flips per pressure (unchecked; C4).
- Serves: H1 in actionable form; H2 (which fix).

**7. Override vs not-routed head classes (A3 + the head-finding half of R6)**
- Idea: with class-level path patching, measure the share of the flip carried by two head classes: "claim readers" (attend to the pushback text) and "own-answer carriers" (attend to the earlier answer). Override = readers win while carriers are active. Not routed = carriers are silent while the answer is decodable upstream. Re-find the classes per model and per checkpoint.
- Why promising: functions are stable at class level while individual heads come and go (`c12 p7-8`). A conditional per-head projection fix exists (`c14 p13`). A QK routing bottleneck sits at about 45% depth (`c7 p6`, at 2B). CLT attribution freezes attention (`u5 p17`), so patching is required. `p2-syco-modes` gives a situation-attention template and an L22H10/H11 hub in the same model (`p2-syco-modes p19-20`). `p2-syco-manifold` is MLP-only, so heads are untouched.
- First experiment: Gemma-2-9B-it, 200 flipped + 200 margin-matched resisted items; two-criterion head selection (c12); path-patch the classes vs size-matched random head sets, with zero, mean and joint ablation.
- Kill: no class carries >= 20-30% of the effect beyond random sets; or the reader share is the same for flipped and resisted items.
- Cost: 10-20 GPU-h at 9B. Scoop: low-medium (Genadi, abstract only, has heads but no fates).
- Skeptic: real risk of a null. In `p2-syco-modes`, head effects are 0.001-0.018 nats, redundant, and only 1 of 3 modes has selective heads (`p2-syco-modes p19-20`); the c12 check notes factual tasks are MLP-heavy. Run only after idea 3 says override exists.
- Serves: fates 3 and 4; the architecture component (QK vs OV vs MLP); a measure for idea 8.

**8. Stage B removed-vs-hidden report card, matched on update size (A6 + the Stage B half of R6)**
- Idea: after each fix (plain LoRA-DPO at several β, SFT, fate-matched arms), report per fate: flip rate, refit decodability, frozen-probe survival (idea 4), re-found head-class share and ablation effect (idea 7), and injection rescue. "Hidden" = flips gone but the class share or ablation effect intact or larger; "removed" = both gone. **Arms are compared only at matched flip-rate drop or matched KL from the base model.**
- Why promising: it extends the round-2 masking result (Llama-70B sycophancy 39% to 3.5% while the heads' ablation effect rose from 10.5 to 27 pp; 2604.19117, one checkpoint pair) to a controlled 9B fix. Self-repair hides removals (`c12 p7`). `p2-syco-manifold` shows SFT moves the subspace about 3x more than DPO (25.07 vs 7.64 deg; `p2-syco-manifold p7`), but at unmatched update size and near a ~9 deg measurement floor (its check), so the matched version is open.
- First experiment: Gemma-2-9B-it LoRA-DPO on 2,000 separate pushback pairs, β in {0.01, 0.1, 0.5} with 3 seeds at β = 0.1; SFT; one fate-matched arm from idea 3 or 5. Save every checkpoint. Report a base-vs-base re-extraction noise floor for every internal metric.
- Kill: verdicts disagree in sign across seeds; or every arm removes the class share and keeps rescue, so "hidden" is not a risk at 9B.
- Cost: about 40-70 GPU-h at 9B plus evaluation. Cheaper option: OLMo-2-13B SFT/DPO/Instruct checkpoints as a free stage curve (availability unverified). Scoop: moderate on framing (cite `p2-syco-manifold`'s "rewrite vs reweight"); low on substance.
- Skeptic: drop the twin-exchange use measure U (the c8 check says it cannot be identified for facts; "SFT reroutes" and the β window are toy-only). Do not cite `p2-syco-manifold` as support for "DPO hides": DPO barely changed behaviour there (about 0.015 in sycophancy probability; `p2-syco-manifold p18`). The costliest idea; run it last.
- Serves: H2 directly; Stage B; fate 4 (masking) and fate 5 (rotation after a fix).

**Order of work:** ideas 1-2 before any fate call; idea 3 alongside collect; idea 5 picks the layers for ideas 4, 6 and 7; ideas 4 and 7 supply the measures for idea 8.

**Killed ideas**
- T3, pullback-Fisher G-metric (pass 1): the FishBack check disputed the layer claims and the 8B eigenspace accuracy was never measured.
- A2's vᵀGv triage (pass 2): relies on the same G-metric, and at >= 6.9B the noise floor exceeds the effects (c9). Its three probe tracks live on in idea 5.
- C6 / T7, manifold vs linear and manifold vs discrete grid (pass 1): the win may be a baseline artefact; it is a Stage B arm choice, not a fate test.
- T8, filler-role re-binding (pass 1): source reading needed fixes, covers one narrow form of "transformed", transfer to 9B unshown.
- C4 as a separate idea (pass 1): needs >= 100 flips per pressure (unchecked); its H1 part is in idea 3 and its transfer part in idea 6.
- Merged, not killed: A1 into idea 5, A4 into idea 3, A5 into idea 4, A3 into idea 7, A6 into idea 8.

---

## 6. What this changes in our plan

**Stage A (sorting failures into fates)**
- Add a fixed control ladder before any fate call: logit margin, P(True), two-centroid detector, full probe (`c15 p4, p6`). Report the probe's gain over the logit margin.
- Bypass (fate 2) needs three checks, not one: matched-random steering null with dose-response (`c9 p8`), probe-vs-DAS ablation with random/format/off-task controls (`c7 p5`), and whether the probe direction lies in the Fisher metric's top eigenspace (`w2-fishback` reading; `w2-info-geometry-shared p4`).
- Transformed vs erased: refit probes on the pressured condition, then Procrustes and interchange tests; a frozen probe reads rotation as erasure (`c8 p13`). Do not use CKA here (`c13 p7`).
- Not routed: KAPPA-style per-item injection into the prediction subspace as the "injection works" test (`c6 p6`).
- Train detectors on the model's own pressured answers, not teacher-forced ones (`c15 p26`).
- Search depth per model; do not assume a fixed relative depth (`c11 p8`).

**Stage B (fix and recheck)**
- Fate-matched arms to set beside DPO: KAPPA edit (not routed) (`c6 p6`); Fisher/natural-gradient steer, layer chosen per model (the location is model-dependent after correction: gain grows with depth in GPT-2, collapses at the one sampled deep layer at 8B; `w2-fishback p17, p20`; `w2-info-geometry-shared p11`); head-level conditional projection (override) (`c14 p7`).
- Recheck after each fix: decodability frozen and refit, use (interchange), and causal write-in vs random (`c8 p13`; `c9` reading). Re-run circuit discovery on the fixed model instead of reusing pre-fix heads (`c12 p7`).
- Expect SFT-style fixes to hide rather than remove (`c8 p5`); keep a KL-anchored arm.
- Never judge "removed vs hidden" with a single similarity score (`c13 p8`).

**Group analyses**
- Start with centroids in a PLS subspace for flipped vs resisted items matched on prior confidence (`c15 p4`). Track the mean-shift vector and the centred subspace separately (`c13` reading).
- Principal angles: compare sum cos^2 against exact nulls, at matched dimensions and token positions (`c7` reading); repeat in the Fisher metric (`w2-info-geometry-shared p1`).
- CKA/RSA for H1 only with bootstrap ranges and per-group centring (`c13 p7`). Compare pressures at matched relative depth (`u5` reading).
- Before any manifold claim, add structure beyond two answers (confidence bins, pushback strength, several candidate answers) (`c5` reading).
- Use Grassmannian speed/curvature only to choose where to patch (`w2-layerwise-grassmann p9`).
- Read before H1 claims: 2601.11563 (one compliance subspace) and 2607.20146 (separable sycophancy modes) disagree with each other (`scout_pass2.md`).

---

## 7. Coverage and trust

- **Read:** 21 papers in full, with figures viewed (readings in `research_notes/Group geometry methods for fates/reading/`).
- **Quote check:** 581 of 581 quotes verbatim (`quote_check_all.txt`).
- **Independent checks (final):** all 21 papers were checked independently. 6 readings were corrected after the checks (u4, info-geometry, shape-of-beliefs, fishback, and the two Grassmann papers). Quotes verified 100%.
- **Done this run:** full-text reading of 21 papers (340/340 figures and tables covered), quote check 100%, coverage check by script, independent checks of all 21 papers, corrections to 6 readings, 3-lens ideas team, skeptic ranking. The architecture lens was added after the skeptic pass.
- **Pass-2 run:** read in full and independently checked 2608.01835 and 2607.20146 (both GOOD, small fixes applied); the architecture-lens ideas (A1-A7) were critiqued by the skeptic (`ideas/skeptic_pass2.md`) and merged into section 5.
- **Blocked:** OpenReview ZP7VzpXmKR (bot check).
- **Pass-2 list (abstracts only, not read):** `scout_pass2.md`, 15 ranked papers. Top three: 2608.01835 (sycophancy manifold charts through post-training, HIGH scoop risk for Stage B), 2601.11563 (shared compliance subspace, against H1), 2607.20146 (separable sycophancy modes, for H1).


## 8. Pass-2 addendum (2026-10-25)

**2601.11563, *Human-like Social Compliance in LLMs*** (read in full, quote-verified 10/10, not independently checked). It claims sycophancy and conformity shift hidden states in similar directions, which is a prior claim against H1. Its evidence is weak: a visual PCA claim with no cosines, no layers and no causal tests. Its useful detail: in Qwen2.5-7B compliance stays above 50% even when neutral confidence is near maximal, so high-margin flips are where override or bypass should be looked for, and matching flipped vs resisted items on prior margin is essential. H1 stays open.

**2608.01835, *Rewriting or Reweighting? A Geometric Account in Language Models*** (`p2-syco-manifold`; read in full, checked GOOD with small fixes). What it shows: it picks sparse MLP neurons that predict sycophantic vs truthful text, takes a low-rank PCA subspace of them, and tracks that subspace through controlled SFT, DPO and SFT-then-DPO runs with principal angles, projector overlap and "anchored retention" R^G (does the old probe still separate classes on the new model). Headline: DPO keeps the subspace (overlap 0.939, 7.64 deg), SFT moves it about 3x more (overlap 0.745, 25.07 deg), and DPO after SFT stays in the SFT geometry (`p2-syco-manifold p7`). Models/scale: a 23-model atlas from 0.5B to 70B, but the post-training study is Llama-3.1-8B only, one seed. What it already does of our plan: the general Stage B question ("remove, rotate, reweight?"), Grassmannian tracking across fix checkpoints, and a population-level version of our frozen-vs-refit probe test; the "rewrite vs reweight" framing is now taken and must be cited. What it leaves open: there is no decodable correct answer (the items are opinion claims such as "all smartphones are spying on us", with teacher-forced dataset text; `p2-syco-manifold p43`), no per-item fates, no pressure types, no fate-matched fixes, no attention heads (MLP only), no layer-by-layer path and no Procrustes. Its evidence for "DPO reweights" is weak: DPO moved sycophancy probability by only about 0.015 (`p2-syco-manifold p18`), about 9 of DPO's ~10 deg already appears after 6 steps in both directions (a measurement floor with no base-vs-base control; its check, Table 7 p18), SFT and DPO had very different update sizes (lr 1e-5 vs 5e-7; `p2-syco-manifold p45`), and the threshold term is defined but never estimated (`p2-syco-manifold p45`). H1: not tested (one pressure type). H2: not tested; it cannot back the "plain DPO hides" premise, because DPO barely changed behaviour. H3: partial analogue. Old-probe retention stays high after DPO (R^G 0.995-0.998) and mostly after SFT (0.81-0.97), but on fixed dataset labels rather than the model's own failures; its 4-item Figure 13 shows single items where the old subspace stops separating after SFT, the H3 failure mode in miniature (`p2-syco-manifold p7, p27`).

**2607.20146, *Gotta Catch them all: the modes of Sycophancy*** (`p2-syco-modes`; read in full, checked GOOD with minor fixes). What it shows: Gemma-2-9B-it under three sycophantic personas on 948 social dilemmas. The three modes are about equally sycophantic and hard to tell apart from text (57.8%, chance 20%), but 100% linearly separable inside the model from L14 (`p2-syco-modes p1, p5`). Geometry peaks at L18, ablation necessity at L22-26, logit-lens output only diverges from L23 (`p2-syco-modes p18-19`). They share a dominant direction plus mode-specific dimensions (17/13/8 of 50 dims shared at L18; `p2-syco-modes p19`). Injecting a mode subspace is sufficient, but ablating it moves output by less than 7% (`p2-syco-modes p5`); only one mode has selective heads, with tiny effects of 0.001-0.018 nats (`p2-syco-modes p19-20`). Models/scale: Gemma-2-9B-it only, the same model as our first run. What it already does of our plan: none of Stage A or B. It has no factual items, no correct-answer probe, no flipped vs resisted split, no confidence matching, no fates and no fixes, so scoop risk is low. What it leaves open, and what we can reuse: paired deviation vectors (pressure run minus the same item's neutral run), layerwise principal angles between deviation subspaces with a shared-dimension count, normalised necessity plus sufficiency, a complement probe, zero vs mean vs joint head ablation, and z-scoring before probe transfer. Two cautions: its "<7% necessity" is divided by the effect of zeroing a whole layer and measured as first-token KL (`p2-syco-modes p16, p18`), so a small ablation effect is not evidence of bypass and may just be redundancy; and raw probe transfer falls to chance from a constant offset alone (`p2-syco-modes p33`). H1: weak, indirect support (shared direction plus mode-specific dimensions), but modes come from personas, not pressures, and its pressure-only control (n=80) gives each mode its own scenario content, so it shows the context is decodable, not that fates differ. It does create framing risk: "sycophancy is several mechanisms" is now claimed, so our H1 must be framed around the fate of a decodable correct answer. H2: not tested; behaviour after ablation is never measured. H3: supported by analogy (a probe trained on persona runs fails on persona-free runs until the offset is removed).

**What this changes in our plan.** Scoop risk for Stage B drops from HIGH to moderate on framing and low on substance: no paper yet tests whether a fix removes or only hides the failure to act on a decodable correct answer, per item, by fate, at matched update size. The highest remaining scoop risk is the intervention ladder (idea 5, 2610.21708). H1 stays open and unclaimed for correct-answer fates; we cite `p2-syco-modes` for "several sycophancy modes" and frame H1 as different fate mixes, not different internal states. Concrete design changes, all now in section 5: (1) Stage B compares arms only at matched flip-rate drop or matched KL from base, and reports a base-vs-base re-extraction noise floor for every internal metric (the gap in `p2-syco-manifold`); (2) we cite "rewrite vs reweight" as prior framing and do not cite it as evidence that DPO hides; (3) probe-transfer tests add a recentre-only step before Procrustes, and idea 2 adds an offset-only plant (`p2-syco-modes p33`); (4) the bypass test measures flips and logit margin with a neutral-run patch as the ceiling and adds a joint cross-layer ablation, so redundancy is not misread as bypass; (5) the head-class idea moves down the ranking (idea 7) because head effects in the same model are tiny and redundant; (6) the first Gemma-2-9B-it run uses `p2-syco-modes` layer priors (L18 geometry, L22-26 causal, L23+ output) as a starting band, as priors only, since they come from social dilemmas, not facts. The top three ideas are now the copy-vs-correctness 2x2, planted fates, and the output screen plus threshold push, all cheap (under 5 GPU-h together).
