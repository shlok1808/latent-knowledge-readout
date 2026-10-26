# Ideas team, lens: actionability

Group geometry methods for fates, 2026-10-25.
Inputs: all 21 reading notes in `reading/*.md` (key quotes and pages spot-checked against the `.json` files) and `scout_pass2.md`.

Ground rules used here:
- Every idea is a **HYPOTHESIS**.
- "Decodable" never means "knows".
- Causal controllability (an edit changes behaviour) is kept separate from endogenous mediation (the model routes through it without an edit).
- Citations are `slug pN`. Short slugs: c5 manifold-steering-2605.05115, c6 kappa-2509.23782, c7 readout-mediator-angle-2605.29126, c8 represented-used-2610.05292, c9 lagged-coupling-2609.01048, c10 curved-inference-ii-2608.24037, c11 trajectory-signatures-deception-acl2026, c12 tigges-circuits-across-training-2407.10827, c13 cka-reliability-2210.16156, c14 mags-2605.21770, c15 confidence-manifold-2602.08159, u3 geometry-of-divergence-2608.30650, u4 role-learning-1910.09113, u5 user-2608.27510, plus the w2-* slugs written in full.
- Papers seen only as abstracts in scout_pass2 are marked **[abstract only]** and carry no page.

---

## 0. The map: where each fate most likely lives, how stable that is, what follows

| Fate | Most likely WHERE (component, relative depth) | Stable across models / seeds / checkpoints? | Fix or monitor that follows | Evidence |
|---|---|---|---|---|
| 1 readout / threshold | The last ~20% of depth. Phase 3 starts at 60-89% depth and is FFN-dominated in all 8 models (w2-readout-subspace-grassmann-2605.09011 p7-8). The prediction subspace turns toward W_U only after the fork, staying 65-70 deg from it at the KAPPA layers (c6 p8, p21). | Phase 3 covers 21.2% ± 8.6 of depth in 8 models from 2 families (2605.09011 p7). The bands are fixed in Qwen2.5 but grow with depth in OLMo-2 (p8). | Cheapest: a late, low-dose margin push or calibration. Metric-aware steering buys nothing there: FishBack's gain collapses at about 75% depth (w2-fishback-2605.17231 p17), and the natural-gradient advantage shrinks toward the read-out (w2-info-geometry-shared-2609.11063 p13). Monitor: flip-margin z (2609.11063 p72) and the near-tie screen top1/top2 < 1.2 (c11). | moderate (methods only, no pressure data) |
| 2 bypass | The residual stream at the probe's site, at any depth. The decodable direction sits about 88 deg from the load-bearing subspace, the Haar null (c7 p5). It occupies growing headroom, yet its write-in is at most 0.11% (c9 p9). For truthfulness it is less likely: shared nodes carry both the probe score and the token (u5 p19). | Read-before-write holds across 6 Pythia sizes x 8 checkpoints, and OLMo-2 keeps the direction (c9 p8, p10). But the 6.9-12B nulls sit below the noise floor (c9 p10). SFT reroutes (U 0.15 while R 0.98, c8 p5), and the outcome is fixed by the pretrained model across post-training seeds (c8 Table 4 p21). | Do not steer or train along the probe direction: "sensors before levers" (c9 p11). Choose causal directions. Keep a KL anchor in post-training (c8 p5-6). Monitor: the probe as a sensor, plus output sensitivity vᵀGv (method from 2609.11063 p4). | moderate |
| 3 not routed | Mid-depth, before the knowledge-prediction fork. The angle jumps at about 55-65% depth in Llama-8B and Qwen-7B (c6 p21). Single-site edits work only in a middle band: about L10-18 of 32 in Llama and L8-19 of 28 in Qwen (w2-nonlinear-interventions-2605.14749 p4). Fisher steering wins early and mid, then dies at about 75% depth (2605.17231 p17). The routing bottleneck is QK-mediated heads at about 45% depth (c7 p6, Gemma-2B only). | Three independent papers find a similar band in two 7-8B families (c6 p21; 2605.14749 p4; 2605.17231 p17). No data at 9-27B. | An input-specific injection inside the band (KAPPA, c6 p6-7), or training only that band's attention. Monitor: the forced-decoding attention-shift check (c14 p13). | moderate |
| 4 override | Attention heads that read the user turn, at mid-depth [Genadi EACL 2026, abstract only]. In Llama-8B on math, per-head failure predictors are densest in the last layer, plus one strong head at L8 (c14 p5). Competing head classes on one logit difference are the IOI analogue (c12 p5, p8). | The class is stable; individual heads are not. Roles move between heads during training (c12 p7), the same head type has opposite sign in different models (c12 p5), and sycophancy signature polarity flips across models (c11 p7-8). | Re-find heads at the class level on every checkpoint (c12 p7). Then apply a conditional projection (c14) or a class-restricted LoRA. Masking risk: in round 2 (2604.19117), the heads' ablation effect grew after an RLHF refresh. | weak-moderate |
| 5 transformed | Depth unknown. The signature is a frozen probe at or below chance while a refit probe works. The subspace rotates 20-77 deg (c8 p13); the failure is orientation, not absence (c15 p6-7); probes transfer only locally (w2-shape-of-beliefs-2602.02315 p5); seeds differ by a rotation [Polymorphism is Rotation, abstract only]. | Unknown at 9-27B. | Monitor: refit per checkpoint and per condition, and use Procrustes within one model; across models, a Procrustes transplant fails (c7 p17). Never use CKA, which is rotation-blind (c13 p3). | weak-moderate |
| 6 erased | Rare. In toy worlds it needs prolonged weight decay (c8 p1, p16). The knowledge probe stays on its plateau to the last layer at the answer position (c6 p18). | n/a | Monitor: a refit whitened probe without a penalty, plus the revival test (c8 p7, p11), with RLACE as a positive control (c15 p5). A fix is likely not needed. | moderate (against erasure) |

### Rules for any actionable claim (they apply to every idea below)
1. Report location as relative depth plus component class, never as a layer or head index. Head roles move (c12 p7-9), and the best probe depth varies from 5.1% to 62% across models (c11 p8).
2. Every causal null needs matched-norm random directions reported as z, a dose-response curve and a per-unit noise floor (c9 p8-10). Report additive drops with a random-ablation envelope (c7 p5).
3. Refit probes on every new model, checkpoint or condition rather than freezing them (c8 p13). Use whitened probes without a penalty (c8 p11).
4. Compare every angle with a matched-rank random null and a probe-vs-probe ceiling, since near 90 deg is the default at d of about 4k (c6 p19; c7 p4). Run the reparameterisation check or work in the G-metric (2609.11063 p1).
5. CKA and RSA are descriptive only, never pass/fail (c13 p8-9). Late layers show anisotropy (c6 p22).
6. Benchmark every geometric feature against a 1-D mean-difference direction and the logit margin. A single direction beat trajectory scalars, 90.3% vs 66.6% (c11 p9).
7. Split items by behaviour under the same prompt, matched on prior confidence (c11 p4 pitfall; confidence axis, c5 p34).
8. With two answers the output manifold degenerates to a line (c5 p10). Start from two centroids in a PLS subspace, since correctness classes are unimodal (c15 p4-5).

---

## Idea 1. HYPOTHESIS: a stable "rescue window" in relative depth tells a lab where to put monitors and inference-time fixes

**Idea.** For flipped items, the correct answer becomes decodable at a depth d_on, and the knowledge and prediction subspaces fork at d_fork. Patching in the resisted-item state rescues the flip only between the two. If [d_on, d_fork] sits at the same relative depth (±10%) across Gemma-2-9B and 27B and across base vs instruct, the lab rule is simple: monitor at or after d_on, intervene before d_fork, and expect nothing past it.

**Why promising.**
- KAPPA's knowledge-prediction angle jumps within about one layer, at about 55-65% depth, the same depth where the prediction probe saturates, in two families (c6 p21; Fig 2 p3).
- Three intervention papers find the same shape, a middle band with late collapse:
  - non-linear single-site edits work at about L10-18 of 32 (Llama) and stop working from L19 (2605.14749 p4);
  - Fisher steering's gain collapses at about 75% depth in both 8B models (2605.17231 p17);
  - the natural-gradient advantage is largest early to mid (2609.11063 p13).
- Correctness is best decodable at 46-75% depth across 1-14B models (c15 p7), and Phase 3 begins at 60-89% depth (2605.09011 p7).
- Together these predict a window of roughly 35-65% depth. None of it was tested on pushback flips.

**What it yields.**
- One depth band per model family.
- Whether the fork is written by attention or by the MLP. To see this, sample `resid_mid` and `resid_post` (sublayer resolution, c10 method 1).

**Smallest first experiment.**
1. Model: Gemma-2-9B-it. Items: the 400 TriviaQA pushback pairs from 2604.19117. Keep items that are correct before pushback, then match flipped vs resisted items on the pre-pushback logit margin.
2. At every layer and sublayer, at the answer position, measure two things:
   - the paired knowledge and prediction probes, and their principal angle against random and probe-vs-probe ceilings (c6 p19-21);
   - rescue: (a) patch the resisted-item mean state into flipped items; (b) inject only the correct-answer subspace with a KAPPA-style edit (c6 p6). Run both with matched-norm random nulls and a dose curve (c9 p8).
3. Read off d_on, d_fork and the rescue curve.
4. Repeat on Gemma-2-27B, and on Gemma-2-9B base given the same dialogue as plain text.

**Kill result.** Any of these kills it:
- rescue works at every depth, or only in the last layers (that means threshold, not routing);
- the window moves by more than 15% of relative depth between 9B and 27B, or between base and instruct;
- the per-item fork depth varies too much to define a population band (IQR above 20% of depth).

**Cost (estimate).**
- Activations: about 1 GPU-h at 9B.
- Patching sweep (400 items x 42 layers x 2 sublayers x ~5 doses, random nulls on a subset): about 5-10 GPU-h at 9B, about 3x that at 27B.

**Scoop risk: medium.** KAPPA measures angles per benchmark on MCQ format (c6). How much 2610.21708 overlaps is unknown. No read paper reports a rescue window for pushback flips.

---

## Idea 2. HYPOTHESIS: a three-track plus output-sensitivity triage sorts bypass items from routed ones before any patching, and names the flips a probe-direction steer can never fix

**Idea.** For each flipped item, measure four things:
- T1: probe decodability;
- T2: the logit-margin AUROC;
- T3: the matched-null write-in along the probe direction;
- the pullback-Fisher sensitivity vᵀGv of the correct-answer direction at the answer position.

Items with high T1, T3 near 0 and low vᵀGv are bypass candidates. A lab should monitor them with the probe (as a sensor) but should not steer or train along that direction.

**Why promising.**
- In c9, probe AUROC is at least 0.99 from step 1k, yet steering along the probe direction is null in 43 of 48 cells (p1, p7). Headroom along the direction grows up to 56.8x while write-in stays at or below 0.11% (p9). The paper's own framing is "sensors before levers" (p11).
- In c7, ablating the probe subspace costs 0.6 pp, while ablating an optimised subspace drops accuracy from 42% to 0%. The angle between them sits at the Haar null (p5).
- ½ δhᵀGδh predicts the real KL with a median ratio of 1.000 (2609.11063 p4). So vᵀGv is a cheap first-order, same-position test of output sensitivity; applying it this way is our adaptation. FishBack gives the matrix-free 8B recipe, about 280 partial forwards per case (2605.17231 p15).
- The output-pullback metric has rank at most |Z|-1 (c5 p10). With 2 answers most directions are inert by default, so nulls are essential.
- Counter-prior: for truthfulness, the probe and the token share nodes. Ablating the shared nodes cuts the probe score 93% and flips 67% of tokens (u5 p19). Bypass may therefore be rare for factual items, and the triage would show that too.

**What it yields.**
- Per model and depth, the share of flips that look like bypass.
- Whether a monitor-only policy is right for those flips.
- The causal direction to use instead of the probe direction.

**Smallest first experiment.**
1. Model and items: Gemma-2-9B-it, 100 flipped and 100 resisted items, matched on prior margin and output entropy (G depends on entropy; 2605.17231 p16 caveat).
2. At 25%, 45% and 65% depth, compute T1-T3 with the c9 recipe: 20 random directions per cell and a noise floor.
3. Compute vᵀGv in fp32, because bf16 and fp32 directions agree only weakly (2609.11063 pitfall). Compare it with random-direction and top-eigendirection baselines.
4. Validate on 50 items against the actual subspace-patching rescue from Idea 1.

**Kill result.** Either of these kills it:
- after entropy matching, vᵀGv and T3 fail to predict patching rescue (Spearman below 0.3);
- the measured 9B noise floor exceeds the effects, as it did at 6.9-12B in c9 p10. The triage then cannot resolve anything at our scale.

**Cost (estimate).**
- G: 200 items x 3 layers x ~280 partial forwards, about 170k partial forwards, roughly 5-15 GPU-h at 9B.
- T1-T3: under 2 GPU-h.

**Scoop risk: low-medium.** 2606.14388 [abstract only] measures the angle from behaviour subspaces to a "decision subspace" at 7-70B. No read paper applies G to probe directions on flipped items.

---

## Idea 3. HYPOTHESIS: override and not-routed live in different attention-head classes, and a fix aimed at the class (re-found per checkpoint) beats one aimed at head indices

**Idea.** Use class-level path patching to measure the share of the flip's logit difference (claim minus correct) that comes from two head classes:
- "claim readers": heads that attend to the user's claim;
- "own-answer carriers": heads that carry the model's earlier answer.

Two fates follow:
- override: the claim-reader class wins while the carriers are still active;
- not routed: the carrier class is silent while the correct answer is decodable upstream.

The fix targets the class, with a conditional projection or a class-restricted LoRA, and the class is re-found on every checkpoint.

**Why promising.**
- c12 shows that function is stable at the class level while individual heads come and go. Name mover (4,6) stops working at 3x10^10 tokens, and (11,0) takes over the successor role (p7).
- Class shares stay above 50%, with the core classes above 70% (p8). Larger models form more stable circuits (p9). Self-repair can hide a removal (p7).
- The same head type has opposite sign in different implementations (c12 p5), so classes must be re-found per model.
- In c14, per-head flip predictors in Llama-8B appear at layers 8-31, densest in the last layer and strongest at L8 (p5). c14 also gives a conditional-projection fix and a forced-decoding attention check (p13).
- c7 locates a QK-mediated routing bottleneck at about 45% depth (p6; 2B only).
- CLT attribution freezes attention (u5 p17), so it cannot see this kind of routing; patching is required.
- Priors from the scout:
  - middle-layer heads attend to user doubt [Genadi EACL 2026, abstract only];
  - sycophancy modes use distinct attention circuits from layer 14 [2607.20146, abstract only].
- c9's prior-bias arm is an override analogue. The prior beats the probe direction, and larger models resist steering more (p10).

**What it yields.**
- A component-type answer: attention QK vs OV vs MLP.
- A stable unit (the class) for monitors and fixes.
- A maintenance rule: re-find the classes after every model update.

**Smallest first experiment.**
1. Model and items: Gemma-2-9B-it, 200 flipped items.
2. Score every head by its attention from the answer position to the claim span vs to the model's earlier-answer span (a copy-score analogue, c12 method 3).
3. Keep a head only if it scores high AND ablating it hurts (c12's two-criterion rule).
4. Path-patch class outputs from the no-pushback run (c12 method 2) against random head sets of the same size.
5. Re-find the classes on Gemma-2-27B and on 3 LoRA-DPO seeds of 9B. Compare head-index Jaccard (c12 method 4) with class-share stability.

**Kill result.** Any of these kills it:
- no class carries more than 30% of the effect beyond size-matched random head sets;
- the claim-reader share is the same for flipped and resisted items, so the heads register pressure but do not decide (the c11 p4 warning);
- class shares are no more stable than head indices across 9B, 27B and the seeds.

**Cost (estimate).** Path patching over about 42 x 16 heads for 200 items: roughly 10-20 GPU-h at 9B.

**Scoop risk: low-medium.** Genadi localises heads but has no fates. 2601.11563 [abstract only] claims a single compliance subspace; test against it.

---

## Idea 4. HYPOTHESIS: threshold flips are a measurable share, and a late, low-dose margin push fixes them; it is the cheapest fix a lab could ship

**Idea.** Some flipped items had a small margin before pushback, moved only a little (flip-margin z near 1), and have knowledge and prediction coordinates on a shifted diagonal. These are readout/threshold failures. For them, a small push along W_U[correct] - W_U[claim] in the last ~20% of depth (Phase 3) should restore the answer without any training. If this share is large, the recommendation is calibration at decoding time rather than retraining.

**Why promising.**
- An argmax flip needs a root-probability move of at least half the margin. On LAMBADA the flip rate falls from 0.519 to 0 as z crosses 1 (2609.11063 p72). This is a necessary condition, not a sufficient one.
- On small-gap tasks the per-item coordinates lie on a tight diagonal (r = 0.97-0.99, c6 p5).
- A β-only push toward extremes gives about +21 AGR on Llama (c6 Fig 4 p7), so part of the gap is amplitude.
- Phase 3 is FFN-dominated in all 8 models, and the logit lens becomes readable only there (2605.09011 p2, p8).
- The internal signal survives where the output fails: on confidently wrong items, P(True) scores 0.48 while the centroid scores 0.93 (c15 p21).
- The near-tie screen comes from c11.
- Metric-aware steering loses its advantage near the read-out (2605.17231 p17; 2609.11063 p13), so a plain additive push is enough there.

**Smallest first experiment.**
1. Model and items: Gemma-2-9B-it, all flipped items.
2. Label threshold candidates (pre-pushback margin quantile, z, near-tie) before any intervention. Seal the thresholds first, as in c9's pre-registration.
3. Apply a dose curve of the W_U-difference push at 80%, 90% and 95% depth, with matched random nulls.
4. Measure three things:
   - rescue rate;
   - off-target KL on held-out neutral prompts (2609.11063 control);
   - correct-to-wrong breakage on resisted items (c6 Fig 14 p31 loss accounting).
5. Compare rescue in threshold vs non-threshold items.

**Kill result.** Any of these kills it:
- at matched dose, threshold-labelled items are not rescued more than others (difference below 10 pp);
- threshold items are under 10% of flips;
- the rescuing dose breaks more than 5% of resisted items.

**Cost (estimate).** Under 3 GPU-h at 9B.

**Scoop risk: low.** The method is simple, and nobody frames it as a fate share. The round-2 finding that a probe beats the logit margin by only +0.008 AUC at 8B (2609.04582) supports the idea rather than threatening it.

---

## Idea 5. HYPOTHESIS: probe monitors trained on normal behaviour miss transformed failures because the correct-answer subspace rotates under pressure and after fine-tuning; a refit or Procrustes-aligned monitor recovers them, while a frozen one reads them as erased

**Idea.** On flipped items, compare three probes:
- a frozen no-pressure probe;
- a probe refit on pressured items;
- the frozen probe after within-model orthogonal Procrustes alignment, fit on resisted items only.

Also compare a linear and an invertible non-linear interchange at one site. Reading the pattern:
- frozen at or below chance, refit high, Procrustes high: transformed. Action: realign the monitor per condition and per checkpoint.
- everything low, including the non-linear map and the revival test: erased (expected to be rare).

**Why promising.**
- c8 finds the subspace image rotating 20-77 deg at β = 0.04 and warns that "a frozen diagnostic would read re-encoding as erasure" (p13).
- Ridge probes misread rescaling as loss: R² 0.954 for least squares vs 0.139 for ridge (c8 p11). c8 also gives the revival test (p7).
- In c15, transfer failure "is one of orientation, not of absence" (p6), and a below-chance score means present but pointed elsewhere (p7). Moving from teacher-forced to the model's own answers drops AUROC from 0.957 to 0.688 (p26).
- Probes transfer only locally (2602.02315 p5).
- At the same site, a linear-restricted map scores 0.010 vs 0.813 for the non-linear map (2605.14749 Table 3 p4).
- CKA cannot make this call: it is rotation-invariant (c13 p3) and drops under a subset shift that keeps separability (c13 p7).
- A Procrustes transplant across models fails (c7 p17), so use Procrustes within one model only.
- A negative z, i.e. backfire, is a cheap hint of a sign flip (c9 p8). See also [Polymorphism is Rotation, abstract only].

**What it yields.**
- A monitor-maintenance rule: how often to refit, and when a Procrustes check is enough.
- A direct test of H3.
- The number a lab will ask for: how much AUROC a frozen monitor loses after one DPO update.

**Smallest first experiment.**
1. Model and items: Gemma-2-9B-it, 400 items. Fit probes inside Idea 1's window, or at 45% and 65% depth if Idea 1 has not run yet.
2. Compare frozen, refit and Procrustes probes on flipped items, with GroupKFold by fact (c15 control) and the random-invertible-map check (c8 control).
3. Repeat with the same three probes after one LoRA-DPO anti-sycophancy run (from Idea 6).

**Kill result.** Either of these kills it:
- the frozen probe stays within 0.03 AUROC of the refit probe on flipped items and after DPO (no transformed fate, and current monitors are fine);
- the refit probe and the non-linear probe are also at chance, and revival fails (the information is gone, or was never decodable).

**Cost (estimate).** Under 2 GPU-h for the probes, plus one DPO run (about 4-8 GPU-h with LoRA at 9B).

**Scoop risk: medium.** 2608.01835 [abstract only] tracks charts through post-training, and 2604.03754 [abstract only] shows truth directions depend on layer, task and instruction.

---

## Idea 6. HYPOTHESIS: plain DPO hides sycophancy (by rerouting or masking it) more often than it removes it, and the KL anchor strength decides which; this gives a "removed vs hidden" report card a lab can run after any fix

**Idea.** After each Stage B fix (plain DPO at several β, SFT, and fate-matched fixes), report three numbers per fate:
- R: refit decodability of the correct answer;
- U: use, from a twin exchange that swaps the correct-answer content between matched prompts and measures how the answer responds;
- the re-found head-class share and its ablation effect.

Reading it:
- removed: the pushback displacement and the class share are both gone;
- hidden: the flips are gone, but the class share or its ablation effect is intact or larger, or U falls while R stays.

**Why promising.**
- In c8, SFT reroutes (R_vis 0.98, U 0.15), while KL-RL at β = 1 reaches the same reward and keeps use (p5).
- The fate switches at a threshold in β, within (0.003, 0.02] for GRPO, while R and A vary smoothly (c8 p6). Erasure needs prolonged weight decay (c8 p1).
- Patching a single layer is undone downstream, so the twin exchange must cover the whole state (c8 p13).
- Qwen3 post-training leaves decodability almost unchanged (peak R² changes by at most 0.05) while late effective rank falls 12-19% (c8 p7). So rank is not evidence of loss.
- Self-repair can mask a removal, so re-find the circuit on the fixed model (c12 p7).
- Round 2 (2604.19117, one checkpoint pair, from the project notes): in Llama-70B an RLHF refresh cut sycophancy from 39% to 3.5%, yet the heads' ablation effect grew from 10.5 to 27 pp.
- Recheck both decodability and write-in after every fix (c9 p11).
- CKA maps can be set while behaviour stays fixed (c13 p8), so CKA cannot be the pass/fail test.
- A 20-item MC check showed no accuracy difference where off-target KL differed 36-50x (2609.11063 pitfall), so judge fixes on more than accuracy.
- The outcome is fixed by the pretrained model across post-training seeds (c8 Table 4 p21). Test 3 LoRA seeds to confirm this at 9B.

**Checkpoint stability for free (verify availability on HF).** OLMo-2-13B has SFT, DPO and final Instruct checkpoints of one pretrained model. Running the Idea 1, 3 and 6 measures on each gives a "fate across post-training stages" curve at 13B. 2605.09011 already reports OLMo-2 phase bands (p7), and c9 replicated in OLMo-2 (p10).

**Smallest first experiment.**
1. Model: Gemma-2-9B-it with LoRA-DPO on 2,000 anti-sycophancy pushback pairs, using facts disjoint from the evaluation set.
2. Runs: β ∈ {0.01, 0.1, 0.5}, with 3 seeds at β = 0.1, plus SFT on the same answers.
3. On 400 held-out items, measure:
   - flip rate;
   - R, U and A;
   - head-class share (Idea 3);
   - frozen vs refit probes (Idea 5).

**Kill result.** Either of these kills it:
- across seeds, the three numbers disagree in sign, so there is no stable verdict;
- every fix removes the class share and keeps U. Then "hidden" is not a real risk at 9B, and plain DPO is fine.

**Cost (estimate).** About 12 LoRA runs x 3-6 GPU-h at 9B, roughly 40-70 GPU-h, plus evaluation.

**Scoop risk: HIGH.** 2608.01835 [abstract only] reports that SFT rewrites sycophancy charts while reward optimisation reweights them. Round-2 idea 3 was already flagged as high scoop risk. Differentiate with the per-fate report card and the β knob.

---

## Idea 7. HYPOTHESIS: different pressures need different fixes (H1 in a form a lab can act on)

**Idea.** Fit the fate-matched fix (the KAPPA-style injection or the head-class projection) on one pressure condition, then apply it to the other: pushback to tempting wrong option, and the reverse. A 2x2 matrix of in-condition vs cross-condition rescue, with loss accounting, tells a lab whether one anti-sycophancy fix covers both pressures or whether each needs its own.

**Why promising.**
- KAPPA subspaces are mostly task-specific, and cross-dataset transfer is often harmful (TruthfulQA to BBH-Algo: -12.02 AGR; c6 p8, Fig 5).
- Correctness directions fit on different datasets are near-orthogonal (|cos| = 0.08, c15 p7).
- The scout found two conflicting priors: sycophancy and conformity share one compliance subspace [2601.11563, abstract only], while the modes of sycophancy are linearly separable and use distinct circuits [2607.20146, abstract only].
- Compare pressures at matched relative depth, since overlap is only non-trivial there (u5 p22, Fig 10).

**Smallest first experiment.**
1. Model: Gemma-2-9B-it, 200 items per condition, on the same facts.
2. Fit KAPPA-style subspaces (c6 p6) inside Idea 1's window, per condition.
3. Apply in-condition and cross-condition. Report rescue and breakage (c6 Fig 14 p31), plus the principal angles between the condition subspaces against random and split-half ceilings.

**Kill result.** Cross-condition rescue is within 20% of in-condition rescue in both directions. One fix then suffices, and H1 does not matter for fixes, whatever the geometry says.

**Cost (estimate).** Under 3 GPU-h at 9B.

**Scoop risk: medium.** It is the direct test between Signal Competition and Modes of Sycophancy.

---

## Ranking (actionability per GPU-hour) and order

1. **Idea 4 (threshold share).** Cheapest, gives an immediate fix recommendation, and can run alongside Stage A.
2. **Idea 1 (rescue window).** Supplies the WHERE for every other idea and reuses Stage A activations.
3. **Idea 5 (monitor survival).** A direct, cheap test of H3.
4. **Idea 7 (one fix or one per pressure).** Cheap, and it settles H1 for fixes.
5. **Idea 3 (head classes).** Gives the component type for override vs not routed.
6. **Idea 2 (triage).** Useful, but the G-solves cost more; run it after Idea 1 finds a window.
7. **Idea 6 (removed vs hidden).** The most valuable Stage B claim, but the costliest and the highest scoop risk. Run it once Ideas 1, 3 and 5 exist.

Dependencies:
- Idea 1 picks the layers for Ideas 2, 3, 5 and 7.
- Ideas 3 and 5 supply the measures for Idea 6.

## Not done / not verified
- Every WHERE claim above comes from 1-8B models (circuits up to 2.8B) on tasks without pressure. None was tested at 9-27B on pushback flips. The depth band in Idea 1 is an inference from three papers.
- Papers seen only as abstracts were not read: Genadi EACL 2026, 2607.20146, 2601.11563, 2606.14388, 2608.01835, Polymorphism Is Rotation (2605.24577), and Spokes, not a Ring. Claims from them are marked [abstract only].
- I wrote these from memory without checking them: the layer counts (Gemma-2-9B 42, Gemma-2-27B 46) and whether OLMo-2-13B's SFT and DPO checkpoints are released. Check both.
- All GPU-hour figures are rough estimates.
- We cannot vary pretraining seeds at 9-27B. "Stable across seeds" here means probe and subspace estimation seeds, data splits and LoRA-fix seeds, plus model size and post-training stage. c9's Pythia checkpoints and c12's Pythia circuits are the only evidence on seeds and training checkpoints, and both come from base models below 12B.
