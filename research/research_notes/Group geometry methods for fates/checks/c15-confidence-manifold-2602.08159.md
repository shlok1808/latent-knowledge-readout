# Check: c15-confidence-manifold-2602.08159 (Cho et al., The Confidence Manifold, v2)

Verdict: **NEEDS FIXES (minor)**. No REDO. The reading is accurate on numbers, quotes, figure coverage and the main caveats. The fixes are two small factual slips, one mischaracterised control, and several places where a paper overclaim is passed through or a fate strength is a notch too high.

## 1. Quote check
`quote_check.py`: 36 quotes, 0 failed. I also checked page attribution by hand against the page footers in paper.txt for the quotes on pages 1, 4, 5, 6, 7, 8, 14, 15, 21, 22, 23, 24, 25 and 26. All are correct.

## 2. Coverage
- Figures and tables: manifest has 29 (Figures 1-5, Tables 1-24). The reading has all 29. Nothing is missing.
- Sections: the reading's `sections_read` covers everything, including Appendices A-P. Nothing skipped. "E.1 Small Instruction-Tuned Models" holds only Figure 5, which is covered.
- Page PNGs are present for every figure page (2, 6, 14, 17, 18). I opened p02, p05, p06, p07, p14, p17, p19, p20 and p23.
- Numbers I spot-checked against text and images, all matching: Tables 1-6, 9-16, 19-24; Figure 2 endpoints (about 0.626 at alpha=-5 and 0.517 at +5, so about 11pp vs the stated 9.1, as the reading says); Figure 1 labels; Figure 3 panels (the reading's ~30%/~66% dashed lines are right: the paper text says 30/70, the image shows ~33/~65); Figure 4 (Llama-3B is clearly the most separated and the GPT-2 panels overlap heavily, so the reading's "caption contradicts image" is correct).
- Confirmed correct reading catches: Table 4 delta is the relative gain over CCS (0.80/0.54 = +48%, and so on); abstract "9.1 points on six models" overstates (range 2.8-10.0); DAS runs on stored embeddings and flips a classifier, not the model; Table 8 GPT-2 dips at 75%; "single direction suffices" (App. C.1) vs "2-8D".

## 3. Errors (reading says / paper says / quote + page)

1. **Table 14, FEVER at 32D.**
   - Reading: "FEVER does not degrade at 32D."
   - Paper: Table 14 (p20) has FEVER at 0.963 (5D) and 0.948 (32D), a drop of 1.5pp. It dips to 0.948 at 12D and stays flat. The text says "three of four show >=1.5pp degradation by k=32 (FEVER recovers to within noise of its peak)", but by that same 1.5pp threshold FEVER also qualifies: SciQ drops 2.5pp, TruthfulQA 3.7pp, CSQA 6.3pp.
   - Fix: "FEVER 0.963 -> 0.948 at 32D (-1.5pp, plateau from 12D); the paper's 'three of four' is inconsistent with its own table."

2. **Table 20 controls.**
   - Reading: "control directions move by up to 4.5pp, comparable to the learned effect in Llama-8B and Gemma-2B."
   - Paper: Table 20 (p23) gives 4.5pp (orthogonal, Gemma-2B) only for Gemma. Llama-3.1-8B controls are +1.6 (random) and +0.3 (orthogonal) against a learned +4.4. Gemma's controls move the opposite way from the learned effect and by more (-3.1, -4.5 vs +2.8). "Specificity" (learned minus orthogonal) hides this.
   - Fix: restrict the claim to Gemma-2B. Llama-1B's controls (+2.1, +3.2 vs +10.0) are also non-trivial.

3. **Nested CV scope** (`setup.controls` and `controls_worth_copying`: "Nested CV for layer and dimension selection").
   - Paper: App. J (p23) searches "PLS dimensions" only, {1..8,12,16}. The inner loop is "3-fold StratifiedKFold", not grouped by question, so paired answers can leak. It is run on only 2 models (Qwen2-7B, GPT-2-Large; Table 21). The main text (p4) says "layer/dimension selection", but the appendix does not show a layer search.
   - Fix: say "dimension only, 2 models, inner folds not question-grouped". Layer and dimension in Tables 1-2 are chosen by max CV test AUC on the reported folds (App. A.2, p13: "Layer and PLS dimension selection use CV test AUC"), which is optimistic for the peak-over-layers and peak-over-k statements.

4. **Table 6 NCH gap.**
   - Reading: "nearest convex hull is 0.02-0.045 below linear".
   - Paper: Table 6 (p15) gaps run from -0.022 (Llama-3B) to -0.051 (Qwen2-1.5B: 0.859 vs 0.808). Trivial, but the range is slightly off.

5. **`methods[0].data_and_compute_needed`:** "seconds of CPU per layer" is not in the paper. The paper reports no timings. Mark it as an estimate. `code_available: yes` rests only on the URL on the title page (not verified, no web).

## 4. Overclaims and pass-through problems (high-stakes fields)

- **`architecture_location`: "Layer choice matters about 11x more than dimension."**
  - Source: App. B.3 (p14). This is one unnamed "best-performing model", and the ratio is 77% (L0 to L23, where L0 is a near-chance embedding layer) over 7% (1D to 5D). The 11x depends on the arbitrary ranges.
  - Qualify it: one model, ratio depends on range chosen.
- **Optimal depth 46-75%.**
  - The AUC-vs-layer curves for the 1B-14B models are never shown. Only a sentence for Llama-8B gives them: "monotonically from L0 (0.59) through L16, then gradually decreases to L31 (0.87)" (p25). The depth numbers are an argmax of noisy, flat-topped curves selected on the reported folds, with fold SD 0.01-0.03.
  - Table 1's depth% is (L+1)/N, which is consistent with 0-indexed blocks. App. B.2 calls Mistral L23 "72%", while Table 1 says 75%. Treat depth as +/- a few percent.
- **"Orientation matters more than dimension" (`bears_on_group_angle`).**
  - The paper's only evidence is R2 = 0.18 for intrinsic dimension vs AUC (Fig. 3b), pooled over model-layer points that are both functions of depth and model. That does not test orientation.
  - The "orientation" claim should be marked as the paper's inference, not a finding.
- **"Teacher-forced detectors drop 27 points on the model's own answers"** (`relevance_to_project`, `actionable_fix`). In the paper this is one model (Qwen2-7B), one dataset (CSQA multiple choice), n=242 vs n=199 (unmatched), with no CI (p26).
  - The 0.957 baseline is itself higher than the paper's own CSQA numbers for the same model, layer and PLS-5 (Table 14: 0.886 at 5D, 0.914 peak; Table 13: 0.894). Against those, the drop is about 20-23pp, not 27.
  - Also, no pressure condition is tested anywhere, so "pressured answers" in `actionable_fix` is our extrapolation.
- **Fate 5 (transformed), strength "moderate".**
  - The paper's "signature" is a below-chance transfer AUC (0.18-0.47) together with |cos| = 0.08 (p7). The two do not sit comfortably together. A near-orthogonal direction (|cos| 0.08) should score about 0.5, whereas a score of 0.18 means a systematic sign flip. The paper does not say which model or space the 0.08 is computed in.
  - HaluEval is a different task format, not a pressure condition. I would downgrade to "weak-moderate" and add "cosine model/space not stated".
- **Fate 1 (readout), strength "moderate".** The evidence cites P(True) = 0.48 (Table 17). The same model and dataset gets P(True) 0.64 in Table 4 (p6), and Figure 1 plots 0.61 for the output method. The reading flags only the Figure 1 mismatch. The paper's output-baseline number is prompt-dependent (0.48-0.64), and P(True) is a separate verification prompt, not the model's answer action. This matters because our first control is the logit margin.
- **Steering evidence quality.**
  - Correctness is scored by keyword overlap (reading has this). App. P (p26) says inter-judge agreement between keyword overlap and an LLM judge was "low", which undercuts the steering effects directly; the reading does not connect the two.
  - Only Qwen2-7B has p-values (App. I, p23). The other five models are point estimates. Controls appear to be a single random draw per model.
  - The direction is scaled to 5% of mean activation norm and applied at every position, so alpha = 5 is a perturbation of about 25% of the mean norm. All causal experiments use seed 42 only (App. A.1, p13).
- **"Low-rank 2-8D" (`key_results`).** For 7B+ models the peak-to-32D gap is within about 1-2 fold SDs (14B: 0.951 vs 0.939, SD 0.02; Table 9). Fit n is only about 650-1,300 in d = 3584-5120, so degradation at large k is estimation variance. It is a statement about a supervised fit at small n, not a rank estimate. The paper itself says only DAS bounds rank (and DAS is probe-level), but the reading's "low-rank" wording is stronger than that.
- **Single-dataset transfer "near-random".** Table 10 (p19) gives Qwen2-7B 0.68-0.72 and Qwen2-1.5B up to 0.759 (FEVER) for TruthfulQA-trained transfer. These are partial transfer, not near-random. Table 5's "Single" for Qwen2-1.5B (0.532) disagrees with the mean of Table 10's three rows (0.647), and Qwen2-7B is 0.660 vs 0.693. The reading copies Table 5 without noting this.

## 5. Omissions worth adding
- **Cross-table AUC spread.** Qwen2-7B PLS-8 TruthfulQA AUC appears as 0.885 (T6), 0.890 (T2), 0.910 (T15, T21), 0.916 (T19 probe), 0.920 (T13) and 0.924 (T14). That is about 4pp for nominally the same quantity, comparable to one fold SD (about 0.04). Method gaps of 0.3-1pp (centroid vs LR, QDA, KMeans) are far below that. The full-dim probe in Table 1 (0.94) is also above every PLS-8 value, so "centroid matches the probe" holds only inside the PLS subspace.
- **Label efficiency.** "25 labels recover 90% of full-data AUC" is for GPT-2, the weakest model. Above chance (0.5) the figure is about 68% ((0.69-0.5)/(0.78-0.5)). Qwen2-7B at K=25 is 0.780 vs 0.907, which is about 69% above chance. The centroid-over-LR gap at K>=200 (+1.2-1.5pp) is within one LR SD (0.010-0.017).
- **Stability for our "where / which fix" question.** There is no checkpoint comparison. Classification uses 3 seeds and causal tests 1. Stability across models is shown only for the abstract form (mean shift, 2-8D). Parameters do not transfer (0.44-0.59).
- **Inference, not in the paper:** with balanced classes the first PLS direction is proportional to the standardised class-mean difference. The paper's Table 24 "PLS-1" column equals Table 19's standardised full-space DoM to three decimals for all 9 models. So "PLS adds +0.057" means components 2-8 on top of the mean-shift direction. 1D alone already gives 0.83-0.93 (Table 2). This supports the reading's suggestion to start with a plain two-centroid comparison.
- **No attention-head or MLP localisation and no token tracking.** The reading says this. It could add that the "three-phase block-diagonal" cosine map (Fig 3c) is a smooth band in the image.

## 6. Does the reading overclaim on our fates / group angle?
Mostly no. It already says the "manifold" is a linear subspace, that DAS and RLACE are not causal for behaviour, that steering is controllability only, and that relevance is moderate. The adjustments needed are in section 4 above: fate 5 and fate 1 strengths, the 27-point generalisation, the 11x and depth caveats, and the nested-CV description.

## 7. Suggested edits to the reading (do not apply here; readings are not edited by the checker)
1. Table 14 `text_matches_figure`: replace "FEVER does not degrade at 32D" with the 1.5pp dip.
2. Table 20 note: restrict "up to 4.5pp" to Gemma-2B and give the Llama-8B control values.
3. `setup.controls` and `controls_worth_copying`: "nested CV = dimension only, 2 models, inner folds not question-grouped".
4. `architecture_location`: add the 11x caveat (one model, range-dependent), the unshown layer curves, the select-on-test-folds caveat, and the single-seed causal tests.
5. `bears_on_fates`: fate 5 to weak-moderate (cos model/space unstated; below-chance vs near-orthogonal tension); fate 1 note that P(True) for Qwen2-7B is 0.48 / 0.61 / 0.64 depending on table.
6. `relevance_to_project` and `actionable_fix`: "27-point drop" is one model, one dataset, one MC set, unmatched n, no CI, and about 20-23pp against the paper's own CSQA numbers.
7. `methods[0]`: drop "seconds of CPU per layer" or mark it as an estimate.
