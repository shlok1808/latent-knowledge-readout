# Skeptic pass 2: architecture + fixes lens, and re-check of the ranked ideas

Group geometry methods for fates, 2026-10-25.

Inputs:
- `ideas/actionability.md` (7 ideas, A1-A7);
- report section 5 (ranked ideas R1-R6);
- every verdict line in `checks/*.md`;
- the two new readings, both checked GOOD:
  - 2608.01835, *Rewriting or Reweighting?*
  - 2607.20146, *Modes of Sycophancy*

Every idea here is a HYPOTHESIS. "Decodable" never means "knows".

---

## 0. Citation check (the 3 most important claims in actionability.md)

I checked each claim against the `reading/<slug>.json` quotes and pages, and against the check files.

**1. The rescue-window depth band (A1, also used by A2, A3, A5, A7).**
- c6 p21 (Figure 8): the page is right.
- The c6 check corrects three things:
  - "55-65% depth" is really 53-68% (17/32 and 19/28).
  - The Qwen L19 value is inferred from KAPPA's layer rule, not measured.
  - The "jump" is usually a 5-20 deg climb over 4-6 layers, then a 15-25 deg jump.
- The check also finds that the random-control CKA rises at the same layer. So the "fork" may partly be anisotropy onset rather than a decision fork.
- 2605.17231 p17: the quote "collapses at roughly 75% relative depth" is verified. But the check finds this is an 8B-only result, from one sampled layer per model, and in GPT-2 the gain rises with depth.
- 2605.14749 p4: the quote "ineffective at early layers, peaks at middle layers, and declines at later layers" is verified. But the task is refusal, not facts.
- **Net: the pages are correct, but the band is stitched together from MCQ, verb-morphology and refusal tasks. Treat it as a prior, not as evidence.**

**2. Bypass ("sensors before levers", A2 and the fate-2 row).**
- c9 p1: "null-equivalent in 43 of 48" is verified.
- c9 p9: 56.8x (0.038 to 2.16) is verified.
- c9 p11: "sensors before levers" is verified.
- But c9's own pitfall says the 6.9B and 12B noise floors (0.0286 and 0.0328) are larger than the largest effect (0.016). So "write-in ≤0.11%" cannot be resolved at our scale.
- c7 p5: 0.6 pp vs 42% to 0% is verified. The Haar-null quote is on p1, and "88 deg" comes from figure labels.
- **Net: the citations are correct. The scale caveat is real, and A2 states it.**

**3. Transformed vs erased, and rerouting (A5, A6).**
- c8 p13: "rotates by 20 to 77 degrees" and "A frozen diagnostic would read re-encoding as erasure" are both verified.
- c8 p5: R_vis 0.98, U 0.15 is verified.
- But the c8 check finds these come from toy 4-layer models. In Qwen3, unanchored GRPO *amplifies* rather than reroutes, U is never reported, and the β window (0.003, 0.02] is the toy threshold only. The c8 check also says U needs hand-built twins, and those do not exist for facts under pushback.
- **Net: the pages are correct, but A6's "SFT reroutes / β decides" rests on toy evidence only.**

**Also checked:**
- 2609.11063 p72: 0.519 to 0 is verified. It is a necessary condition only, and the reading says z cannot separate threshold from override by itself.
- c6 p8: 65-70 deg from W_U is verified.

**Layer counts, from memory (not verified this pass):**
- Gemma-2-9B has 42 layers and 27B has 46. That matches my knowledge.
- OLMo-2-1124-13B -SFT/-DPO/-Instruct are, I believe, public. Verify on HF.

---

## 1. Verdicts on the 7 architecture-lens ideas

| # | Idea | Verdict | One-line reason |
|---|---|---|---|
| A1 | Rescue window in relative depth | **Merge into R5** | Same WHERE claim as R5's sublayer fork band. Three problems: the band is inferred from other tasks (§0.1); "fork" may be anisotropy; a mean-resisted-state patch is confounded by item difficulty. Patch the item's own no-pushback state instead. |
| A2 | Three-track + vᵀGv triage | **Merge T1-T3 into R5; kill the vᵀGv part** | The G-metric was already killed in pass 1 (T3): disputed layer claims, 8B eigenspace accuracy never measured, and bf16/fp32 cosines of 0.06-0.54. c9's noise floor beats its effects at ≥6.9B. T1-T3 is simply R5's probe-direction rung. |
| A3 | Override vs not-routed head classes | **Keep, merged with the head-finding half of R6** | Not done by either new paper; 2608.01835 is MLP-only. It is a Stage A component claim, and R6 only tracks heads through Stage B. Warning from 2607.20146: head effects are tiny (0.001-0.018 nats) and redundant, and only 1 of 3 modes has selective heads. The 20-30% class-share kill could well fire. |
| A4 | Threshold share + late margin push | **Merge into R4 (adds the causal step R4 lacked)** | As written, the kill test cannot fail, because low-margin items are always easier to push. The push also needs the correct answer at inference time, and it would also block correct updates when the user is right. Rewritten below. |
| A5 | Frozen vs refit vs Procrustes monitors | **Merge into R2** | Exact duplicate. New from 2607.20146: raw probe transfer falls to chance from a constant offset alone, so add a recentre-only step before Procrustes. New from 2608.01835: R^G (anchored retention) is the population analogue, but on teacher-forced labels. |
| A6 | Removed vs hidden report card | **Keep, merged with the Stage B half of R6** | This is the core of Stage B. 2608.01835 takes the "rewrite vs reweight" framing, so scoop risk drops from HIGH to moderate on framing and low on substance: no correct-answer variable, no per-item fates, and DPO barely moved behaviour, with a ~9 deg measurement floor. Problems: the U/twin measure cannot be identified for facts, and the β story is toy-only (§0.3). |
| A7 | One fix or one per pressure | **Keep** | Not done by either paper. 2607.20146's modes come from personas on social dilemmas, and its Appendix N confounds mode with content. Confound to fix: the pushback and tempting-option prompts differ in format, so needs a within-pressure, cross-wording transfer ceiling. |

### Per-idea checks in detail (done / confounded / scale / causal / disputed / duplicate)

**A1. Rescue window**
- Done by a paper? No paper reports a rescue window on pushback flips.
- 2607.20146 gives layer priors for the same model (Gemma-2-9B-it, 42 layers), from social dilemmas, not facts:
  - geometry peaks at L18 (43%);
  - causal necessity peaks at L22-26 (52-62%);
  - logit-lens divergence starts at L23 (55%).
- Confounded? Yes, as written. The resisted-item mean state carries a generic "stay" signal, and resisted items are more confident. Patch the item's own no-pushback state, and match flipped vs resisted items on margin.
- Testable at 9B-27B? Yes.
- Is there a causal step? Yes, but full-state patching at late layers rescues almost by construction. So the window's upper edge must come from subspace-restricted injection only.
- Relies on a disputed reading? Partly: the c6 fork reading and the FishBack 75% collapse (§0.1).
- Duplicate? Yes, of R5.

**A2. Triage**
- Not done by a paper.
- Confounded by entropy, which the idea itself handles.
- Testable at 9B-27B? Marginal: c9's floor issue.
- Causal step: only a validation against patching.
- Relies on a disputed reading? Yes: G, killed in pass 1.
- Duplicate? Partly, of R5.

**A3. Head classes**
- Not done by a paper. Genadi (abstract only) has heads but no fates.
- Confound: claim-reader heads may fire equally on resisted items. The idea's own kill test covers this; keep it.
- Testable at 9B-27B? Yes.
- Causal step: yes, path patching.
- Relies on a disputed reading? c12 is Pythia ≤2.8B, and its check says factual recall is MLP-heavy with empty circuits on MLP-dominant tasks. That raises the chance that override is MLP-borne, not head-borne.
- Duplicate? Partly, of R6.

**A4. Threshold push**
- Not done. 2608.01835 defines a threshold term τ but never estimates it.
- Confounded? Yes: "threshold" items are low-margin by definition.
- Testable at 9B-27B? Yes.
- Causal step: yes.
- Relies on a disputed reading? The 0.519 to 0 drop is a necessary condition only.
- Duplicate? Yes, of R4.

**A5. Monitor survival**
- Not done on flipped items.
- Confounded unless matched on margin. R2 already says so.
- Testable at 9B-27B? Yes.
- Causal step: partly; it is a monitor test.
- Relies on a disputed reading? c8's rotation result is toy, anchored and benign (c8 check). It supports the mechanism, but it is not an example of a transformed failure.
- Duplicate? Yes, of R2.

**A6. Report card**
- Done by a paper? Partly, in framing (2608.01835).
- Confound: unmatched update size, which is 2608.01835's own flaw. Match the DPO arms and the fate-matched arms on flip-rate drop or on KL from base.
- Testable at 9B-27B? Yes, with LoRA.
- Causal step: yes.
- Relies on a disputed reading? Yes: c8's U and β results are toy-only, and the masking result (2604.19117) is one checkpoint pair.
- Duplicate? Partly, of R6.

**A7. Pressure transfer**
- Not done.
- Confound: format, as above.
- Testable at 9B-27B? Yes.
- Causal step: yes.
- Relies on a disputed reading? c6's task-specificity result is on MCQ benchmarks, not on pressures.
- Duplicate? No. R4 tests H1 at the output level only.

---

## 2. The 6 ranked ideas vs the two new papers

**R1. Copy-vs-correctness 2x2**
- Neither paper touches it.
- Still needed. 2608.01835 uses teacher-forced dataset text, which is exactly the copy confound in another form.

**R2. Frozen / refit / Procrustes probes**
- Not done by either paper.
- 2608.01835's R^G is a population-level version:
  - DPO R^G is 0.995-0.998;
  - SFT R^G is 0.81-0.97;
  - its 4-item Figure 13 shows the per-item failure that H3 predicts.
- Add a recentre (bias-only refit) step, following 2607.20146.

**R3. Plant known fates**
- Neither paper touches it.
- 2608.01835 supplies one more reason for it: its chart PC1 is nearly orthogonal to the behaviour direction (cos 0.062), so chart rotation need not mean the behaviour direction rotated.

**R4. Output slope/intercept screen**
- Not done by either paper. 2608.01835 never estimates τ.
- Keep, now with A4's causal step.

**R5. Intervention ladder + fork band**
- Not done by either paper.
- 2607.20146 adds Gemma-2-9B-it layer priors and a design rule: its "<7% necessity" is divided by the effect of zeroing the whole layer, and it is measured as first-token KL. So:
  - measure flips and logit margin;
  - use a neutral-run patch as the ceiling;
  - do not read a low ablation effect as bypass, since it may be redundancy (it adds a joint, cross-layer ablation rung).

**R6 and Stage B (the main question)**
- Does 2608.01835 already do it? No.
  - It has opinion claims with no decodable correct answer, and no per-item fates.
  - It uses one model (Llama-3.1-8B) and one seed.
  - DPO moved behaviour by only about 0.015 in sycophancy probability.
  - 9.06 of DPO's 9.68-10.10 deg is a first-checkpoint floor that is shared with SFT and with the opposite objective.
- Does it undercut R6 and Stage B? Partly:
  1. "Plain DPO hides" has no real support from it, because DPO barely did anything. Our H2 premise has to stand on its own data.
  2. Its central confound is a warning for Stage B. Comparing SFT with DPO at unmatched update size makes "geometry conserved" mean "little changed". **Stage B must match arms on behaviour change, or on KL from base, before comparing internals, and it must report a base-vs-base re-extraction noise floor.**
  3. "Rewrite vs reweight" must be cited as prior framing.
  4. It is MLP-only, so the head-class half of R6 is untouched.
- Does 2607.20146 already do it? No: it has no fixes and no behaviour after ablation.
  - It undercuts R6's head half a little: for sycophancy modes in Gemma-2-9B-it, heads are redundant and their effects are tiny. Expect R6's ≥20% class-share kill to be a live risk, and add joint ablation.

---

## 3. Final merged ranking (8 surviving ideas)

Order = evidence × actionability × feasibility, with prerequisites first. Costs are rough estimates on an A100 at 9B.

### 1. Copy-vs-correctness 2x2 (R1)
- **Idea.** After pushback, "the probe still reads Canberra" may only be the model's earlier answer sitting in the context. Cross "earlier answer right or wrong" with "fact true or false", and test whether a truth probe survives on the decorrelated cells.
- **Why promising.**
  - Every fate call assumes the probe reads correctness (contradictions C7).
  - 2608.01835's labels are teacher-forced text, so it never separated text from belief (its pitfall, p43).
  - Moving from teacher-forced to the model's own answers drops AUROC from 0.957 to 0.688 (c15 p26).
- **First experiment.** Gemma-2-9B-it, the 400 TriviaQA pushback pairs (2604.19117). Build the four cells by prefilling a right or a wrong first answer. Train a probe on the decorrelated cells, at mid-depth.
- **Kill.** If the decorrelated-cell AUC is ≥0.8, the threat is dead. Below 0.65, Stage A's probe premise fails.
- **Cost.** 1-2 GPU-h.
- **Scoop.** Low.
- **Skeptic note.** It is a control, not a finding. Skipping it could sink every later result.
- **Serves.** All fates; it is a prerequisite for H1-H3.

### 2. Plant known fates before trusting any group metric (R3)
- **Idea.** Write synthetic bypass, rotation, sign-flip, offset-only and erasure edits into real 9B activations. Score which instruments call each one correctly: principal angles, CKA, RSA, Procrustes, frozen and refit probes.
- **Why promising.**
  - CKA can fall to 0 while separability stays above 90% (c13 p7).
  - A frozen probe reads re-encoding as erasure (c8 p13).
  - A constant offset alone sends probe transfer to chance (2607.20146 p33).
  - A chart's PC1 can be orthogonal to the behaviour direction (2608.01835 p28).
- **First experiment.** About 5k partial forwards at Gemma-2-9B-it. Six plant types, three doses each.
- **Kill.** Any metric that miscalls more than 20% of plants is barred from fate calls.
- **Cost.** Under 1 GPU-h.
- **Scoop.** None.
- **Skeptic note.** The plants are linear and clean, so passing is necessary, not sufficient. Add an offset-only plant (new from 2607.20146).
- **Serves.** Instrument validity for all six fates; prerequisite for H3.

### 3. Output-level screen plus the threshold-push test (R4 + A4)
- **Idea.**
  - First, fit flip probability against the pre-pushback margin, per pressure. A shifted intercept points to override; a flattened slope points to a readout failure.
  - Then test the causal step: a late-layer push along W_U[own prior answer] − W_U[claim]. Normalise the dose to each item's margin, so that low-margin items get no free win.
- **Why promising.**
  - The screen is nearly free.
  - A flip needs a root-probability move of at least half the margin (2609.11063 p72; necessary only).
  - Phase 3 is FFN-dominated in all 8 models (2605.09011 p7-8).
  - At 8B a probe beats the logit margin by only +0.008 AUC (2609.04582).
  - 2608.01835 defines a threshold term but never estimates it (p45).
- **First experiment.** Gemma-2-9B-it, collect logits for 1 competing answer + 2 user pushback. Fit a logistic with a pressure interaction. Then apply margin-normalised pushes at 80/90/95% depth with matched random nulls, measuring three things:
  - breakage on resisted items;
  - off-target KL on neutral prompts;
  - **blocked correct updates**, on a new "model wrong, user corrects" item set.
- **Kill.**
  - The screen: the two pressures share slope and intercept within bootstrap CIs, and a mixture adds nothing (BIC).
  - The push: at margin-normalised dose, rescue does not differ between threshold-labelled and other items by ≥10 pp; or threshold items are <10% of flips; or the push blocks >5% of correct user corrections.
- **Cost.** About 0 GPU-h for the screen; under 3 GPU-h for the push.
- **Scoop.** Low.
- **Skeptic note.**
  - The direction is built from the model's own prior answer. Using the gold answer would make it a non-deployable oracle.
  - Without the margin normalisation, the A4 kill could not fail.
  - The deployable form is "anchor to own prior answer", which trades sycophancy for stubbornness. The correct-update set measures that trade.
- **Serves.** Fate 1 vs fate 4; H1 (output level); a cheap Stage B arm (H2).

### 4. Monitor survival: frozen / recentred / refit / Procrustes probes (R2 + A5)
- **Idea.** On flipped items, before and after one LoRA-DPO run, compare four probes:
  - a frozen no-pressure probe;
  - a recentred probe (bias refit only);
  - a within-model Procrustes-aligned probe (fit on resisted items);
  - a fully refit probe.
- Reading the pattern:
  - a gap closed by recentring = offset;
  - closed by Procrustes = transformed (fate 5);
  - closed by nothing, plus a failed revival = erased (fate 6).
- **Why promising.**
  - A frozen diagnostic reads re-encoding as erasure (c8 p13; toy).
  - Transfer failure is orientation, not absence (c15 p6-7).
  - Offset-only transfer failure (2607.20146 p33).
  - SFT R^G is 0.81-0.97, and the 4-item Figure 13 shows per-item breakage (2608.01835 p7, p27).
  - The one erasure claim in round 2 had a below-chance probe (2607.00415).
- **First experiment.** Gemma-2-9B-it, 400 items, flipped vs resisted matched on margin. Use GroupKFold by fact, at the depths from idea 5 (or at 45% and 65%). Repeat after one LoRA-DPO run (shared with idea 8).
- **Kill.** Either of these:
  - the frozen probe stays within 0.03 AUROC of the refit probe on flipped items and after DPO (fates 5 and 6 are empty, and current monitors are fine);
  - everything, including the non-linear probe, is at chance.
- **Cost.** About 2 GPU-h, plus the shared DPO run.
- **Scoop.** Medium. R^G is close in spirit, but on teacher-forced labels and not on the model's own failures.
- **Skeptic note.** Without margin matching, the refit gap is a difficulty gap. The c8 rotation evidence is toy and benign. Use idea 2's plants to set the offset/rotation thresholds.
- **Serves.** Fates 5 and 6; H3 directly.

### 5. Intervention ladder + rescue band at sublayer resolution (R5 + A1 + A2's T1-T3)
- **Idea.** On the same items, run three rungs and compare their fate labels:
  - probe-direction steering (matched nulls, dose curve, write-in);
  - subspace (KAPPA-style) injection;
  - per-item patching of the item's own no-pushback state.
- Then localise where injection still rescues, as a band in relative depth at `resid_mid` / `resid_post` resolution (attention vs MLP). The upper edge comes from subspace injection only.
- **Why promising.**
  - Probe ablation costs 0.6 pp vs DAS taking accuracy from 42% to 0% (c7 p5).
  - Steering along the probe direction is null in 43 of 48 cells (c9 p1).
  - Single-site edits work only mid-depth (2605.14749 p4, refusal task).
  - Gemma-2-9B-it priors from 2607.20146: geometry L18, causal L22-26, output from L23 (p18-19).
  - The KAPPA fork is at 53-68% depth (c6 p21, corrected).
- **First experiment.** Gemma-2-9B-it, 400 pairs, matched on margin. 3 rungs × sublayer hook points in the 25-75% depth range first, with matched-norm random nulls and a noise floor (c9 p8-10). Add one joint, cross-layer ablation rung to separate bypass from redundancy (2607.20146 pitfall).
- **Kill.**
  - The rung labels agree for ≥80% of items (κ ≥ 0.6): the instrument does not set the fate.
  - The band's IQR is above 20% of depth, or it moves by more than 15% between 9B and 27B: drop the WHERE claim.
  - Subspace injection rescues equally at every depth.
- **Cost.** 5-10 GPU-h at 9B, about 3x that at 27B.
- **Scoop.** High (2610.21708; "decodability is not use" is crowded).
- **Skeptic note.**
  - Lead with the per-item band, not the ladder.
  - Full-state patching late in the network rescues by construction, so it cannot set the band's top.
  - The "fork" may be anisotropy onset (c6 check), so add probe-vs-probe and random-angle ceilings.
  - The 35-65% prior comes from other tasks.
- **Serves.** Fates 2 and 3 (and 1 at the band's top); H3; the architecture WHERE.

### 6. One fix or one per pressure (A7)
- **Idea.** Fit the fate-matched fix (KAPPA-style injection, or a head-class projection) on one pressure condition and apply it to the other. The 2x2 rescue matrix, with loss accounting, says whether one anti-sycophancy fix covers both pressures.
- **Why promising.**
  - KAPPA subspaces are task-specific: TruthfulQA to BBH-Algo transfer costs −12.02 AGR (c6 p8).
  - Correctness directions from different datasets are near-orthogonal (|cos| 0.08; c15 p7).
  - 2607.20146 finds mode-specific dimensions on top of a shared direction (17/13/8 of 50 dims shared at L18; p19), but from personas, not pressures.
  - The 2601.11563 shared-subspace claim is visual PCA only.
- **First experiment.** Gemma-2-9B-it, 200 items per condition on shared facts. Fit inside idea 5's band. Add a **cross-wording, within-pressure** transfer as the ceiling.
- **Kill.** Cross-pressure rescue is within 20% of the within-pressure, cross-wording ceiling in both directions. One fix then suffices.
- **Cost.** Under 3 GPU-h.
- **Scoop.** Medium. Neither new paper does it.
- **Skeptic note.** Without the wording ceiling, a transfer failure is just a format difference. It needs ≥100 flips per pressure (unchecked; C4).
- **Serves.** H1 in actionable form; H2 (which fix).

### 7. Override vs not-routed head classes (A3 + the head-finding half of R6)
- **Idea.** Use class-level path patching to measure the share of the flip's logit difference carried by two classes:
  - "claim readers" (attend to the pushback span);
  - "own-answer carriers" (attend to the earlier answer).
- Override = readers win while the carriers are active. Not routed = the carriers are silent while the answer is decodable upstream. Re-find the classes per model and per checkpoint.
- **Why promising.**
  - Functions are stable at class level, while individual heads come and go (c12 p7-8).
  - A conditional per-head projection fix exists (c14 p13).
  - A QK routing bottleneck at about 45% depth (c7 p6, at 2B).
  - CLT attribution freezes attention (u5 p17), so patching is required.
  - 2607.20146 has a situation-attention template and the L22H10/H11 hub in the same model (p19-20).
- **First experiment.** Gemma-2-9B-it, 200 flipped + 200 margin-matched resisted items. Two-criterion head selection (c12). Path-patch the classes vs size-matched random head sets, with zero and mean ablation plus joint ablation.
- **Kill.** Either of these:
  - no class carries ≥20-30% of the effect beyond random sets;
  - the reader share is the same for flipped and resisted items.
- **Cost.** 10-20 GPU-h at 9B.
- **Scoop.** Low-medium. Genadi (abstract only) has heads but no fates. 2608.01835 is MLP-only.
- **Skeptic note.** Real risk of a null:
  - in 2607.20146, head effects are 0.001-0.018 nats, redundant, and only 1 of 3 modes has selective heads;
  - c12's check notes factual tasks are MLP-heavy.
  - Run it only after idea 3 says override exists.
- **Serves.** Fates 3 and 4; the architecture component (QK vs OV vs MLP); it supplies a measure to idea 8.

### 8. Stage B removed-vs-hidden report card, matched on update size (A6 + the Stage B half of R6)
- **Idea.** After each fix, report per fate:
  - flip rate;
  - refit decodability R;
  - frozen-probe survival (idea 4);
  - the re-found head-class share and ablation effect (idea 7);
  - injection rescue.
- The fixes: plain LoRA-DPO at several β, SFT, and the fate-matched arms.
- "Hidden" = flips gone but the class share or ablation effect is intact or larger. "Removed" = both gone.
- **Arms are compared only at matched flip-rate drop, or matched KL from base.**
- **Why promising.**
  - It extends the round-2 masking result (Llama-70B: sycophancy 39% to 3.5% while the heads' ablation effect rose from 10.5 to 27 pp; 2604.19117, one checkpoint pair) to a controlled 9B fix.
  - Self-repair hides removals (c12 p7).
  - 2608.01835 shows SFT moves geometry about 3x more than DPO (25.07 vs 7.64 deg; p7), but at unmatched update size and near a ~9 deg floor. So the matched version is open.
- **First experiment.** Gemma-2-9B-it LoRA-DPO on 2,000 disjoint pushback pairs:
  - β ∈ {0.01, 0.1, 0.5}, with 3 seeds at β = 0.1;
  - SFT;
  - one fate-matched arm from idea 3 or idea 5.
  - Save every checkpoint.
  - Report a base-vs-base re-extraction noise floor for every internal metric.
- **Kill.** Either of these:
  - the verdicts disagree in sign across seeds;
  - every arm removes the class share and keeps rescue, so "hidden" is not a risk at 9B.
- **Cost.** About 40-70 GPU-h at 9B, plus evaluation. Cheaper option: OLMo-2-13B SFT/DPO/Instruct checkpoints as a free stage curve (verify availability).
- **Scoop.** Moderate on framing (cite 2608.01835's "rewrite vs reweight"); low on substance.
- **Skeptic note.**
  - Drop the twin-exchange U: the c8 check says it cannot be identified for facts, and "SFT reroutes" plus the β window are toy-only.
  - Do not cite 2608.01835 as support for "DPO hides": DPO barely changed behaviour there.
  - The costliest idea. Run it last.
- **Serves.** H2 directly; Stage B; fate 4 (masking) and fate 5 (rotation after a fix).

### Killed in this pass
- **A2's vᵀGv triage** (the G-metric was already killed in pass 1; scale floor). Its T1-T3 live on in idea 5.

### Dependencies
- Ideas 1-2 come before any fate call.
- Idea 3 runs alongside collect.
- Idea 5 picks the layers for ideas 4, 6 and 7.
- Ideas 4 and 7 supply the measures for idea 8.

## Not done / not verified
- I did not re-open PDFs. Citations were checked against the reading JSONs and the check files only.
- Layer counts and OLMo-2-13B checkpoint availability are from memory.
- All GPU-h figures are estimates.
- No check yet of whether 400 TriviaQA pairs give ≥100 flips per pressure at 9B.
