# Check: c6-kappa-2509.23782 (Park, Pyun, Jo, "Bridging the Knowledge-Prediction Gap ... MCQs", KAPPA, arXiv 2509.23782v4)

Verdict: NEEDS FIXES (moderate). No fabricated numbers or quotes. Tables, quotes and page numbers are reliable. The problems are in interpretive claims that our group-angle work will lean on: "jump, not drift", the Qwen depth numbers, the W_U angle, the CAA comparison, and the "two clusters" reading of GSM8k.

## Quote check
- `quote_check.py`: 33 quotes, 0 failed (it tests only the first 120 normalised characters and no pages).
- Stricter pass (my own script, per-page `pdftotext`): all 22 key_results quotes, 9 non-empty pitfall quotes, `limitations_quote` (p3) and `future_work_quote` (p9) match in full and sit on the page the reading gives. 33/33 OK.
- Two pitfalls have an empty quote (pitfalls[9] p6, [10] p9). Both are the reader's own claims, and both hold: the paper never ablates or patches the knowledge subspace (no "ablat" or "patch" in the text), and Table 4 (free-form GSM8k 91.6) against Table 1 (MCQ 47.8) is as stated.
- Cosmetic: `limitations_quote` stops mid-sentence ("most interpretable and").

## Figures vs manifest
- Manifest: 29 items (Figures 1-14, Tables 1-15). The reading has all 29. None missing. Pages without a PNG (p19, 20, 23, 25, 26) are text only, with no figures.
- I opened Figures 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11-14 and Tables 1, 2, 3, 8, 11, 12, 13, 14, 15. I re-rendered Figures 3, 8 and 9 at 200-300 dpi. I did not re-count Table 10.
- Numbers that check out: Table 1 (all rows the reading quotes), Table 2 (including "8.9-16.9" = Llama BBQ-Religion KAPPA(3) 76.5 minus 67.6, TruthfulQA 16.8, BBQ-Age 16.9, and Llama reasoning "3.4-5.0" = NLP 3.4, GSM8k 4.0, Algo 5.0, so the reading is right that "1.5-5.0" does not match), Table 3, Table 8, Tables 11-15, the Figure 5 cells (diagonal equals the Table 2 KAPPA(1) AGR gains), Figure 10 points, Figures 11-14 values, Figure 6 and 14 bar heights, Figure 4 curves.
- The reading's own flags are correct: Figure 2 caption "saturates" vs a peak at L13-14; Section 5.1 "intermediate" layers vs late layers; Appendix B.4 "clearly above random (<= 0.2)" vs Figure 9; Appendix E.1 "consistently improves AGR" vs Llama MMLU-STEM; the 1.5-5.0 reasoning range; the missing Table 12 accuracy loss.

## Errors (reading says / paper says / evidence)

1. **"Jump, not drift" is overstated.** (bears_on_group_angle (1), Figure 8 takeaway, methods[2])
   - Reading says: the angle "does not drift: it jumps within about one layer from about 50-67 deg to about 84-88 deg", and "an abrupt one-layer jump rather than a gradual climb".
   - Paper says (B.4, p20): the angle "rises sharply toward 90 deg" and "this pattern is consistent across both model families and all three benchmarks".
   - Evidence, Figure 8 at 200 dpi. A clean one-layer jump is in Llama BBH-Algo only (63 to 84 at L17). Elsewhere a climb comes first. Llama TruthfulQA goes 50 (L9) to 70 (L14-15), dips to 67 (L16), then jumps to 87 at L17. Qwen TruthfulQA goes 50 (L11-12) to 62 (L16-18), then jumps to 87 at L19. Qwen GSM8k goes 56 (L15) to 65 (L17-18), then to 81 at L19. Qwen BBH-Algo goes 62 (L17) to 66, then 81 at L19, then 85.6 at L21. Llama GSM8k is not a one-layer jump (85 at L17, 71 at L18, 79 at L19, 84 at L20, 87 at L21-22). In Figure 9 the Qwen GSM8k knowledge-prediction CKA climbs steadily from 0.12 at L3 to 0.82 at L25, with no jump.
   - Fix: "a 15-25 deg jump at one layer (L17 Llama, L19 Qwen), usually preceded by a 5-20 deg climb over 4-6 layers; clean only in Llama BBH-Algo; Llama GSM8k is noisy; CKA drifts rather than jumps." This matters for direction 3 (jump vs drift) and for the fork-point idea.

2. **Qwen "prediction probe saturates at about L19" is inferred, not shown.** (architecture_location, Figure 8 takeaway)
   - The paper gives layer-wise probe accuracy only for Llama (Figure 2, Figure 7; A.4 says "using Llama 3.1 8B Instruct"). There is no Qwen curve. L19 comes from the angle jump and from KAPPA's layer rule (top-1 TruthfulQA layer is L19, and layers are chosen only where the prediction probe is above 85%). It is a lower bound, not a measured saturation point. Mark it as inferred.
   - Arithmetic: the reading says "about 55-65% relative depth (17/32 and 19/28)". 17/32 is 53% and 19/28 is 68%. Say 53-68%.

3. **Prediction-vs-W_U angle "about 66-75 deg at the KAPPA layers" does not hold for Qwen TruthfulQA.** (Figure 8 numbers; the paper's own "65-70" in key_results[16] is quoted correctly)
   - Qwen TruthfulQA (Figure 8f): the prediction-W_U angle is 88.5, 88.3, 88.5, 88.3 at L19-22 and 86 and 83 at L23-24. The Top-6 band is L19-24, so about 83-88 deg. Qwen BBH-Algo at L22-27 is about 71-83. Llama panels are about 66-75.
   - Impact: the paper's "KAPPA is not a token-logit swap" argument (angle 65-70 deg) rests on Llama-like panels. On Qwen TruthfulQA the prediction subspace is close to random against W_U at the KAPPA layers, and it drops toward W_U only after about 4 layers. Keep the claim as "Llama; Qwen varies".

4. **Table 3 takeaway understates how often CAA matches or beats KAPPA(1).**
   - Reading says: "KAPPA helps every model, but CAA beats KAPPA(1) in ACC on Mistral."
   - Table 3 (p8): CAA beats KAPPA(1) in ACC on Mistral (52.8 vs 51.0), Qwen2.5 7B (61.1 vs 60.6), Qwen3 4B (60.1 vs 58.4) and Qwen3 14B (73.6 vs 73.3). That is 4 of 5 models. Only Llama has KAPPA(1) clearly ahead (67.8 vs 62.3). Table 2 shows the same for Qwen TruthfulQA (61.1 vs 60.6) and BBQ-Age (84.3 vs 84.1). KAPPA(3) and KAPPA(6) win, but they use 3 and 6 layers against CAA's 1.
   - The reading already notes that CAA's multiplier is fixed at 1.0 and not swept. Together this means a tuned CAA would probably match KAPPA(1) outside Llama. `actionable_fix` and `reuse_verdict` should say: KAPPA's edge over one-layer mean-difference steering is clear on Llama 3.1 8B only.

5. **GSM8k "two clusters = hint of discrete states" is an over-read.** (bears_on_group_angle (3), methods[1])
   - Figure 3a at 300 dpi: GSM8k is a bent continuous band, not two separated blobs. The prediction logit is high or low because the prediction probe is trained on the model's discrete argmax and is about 94-96% accurate (Figure 7, Figure 13). A bimodal prediction logit follows from the label. It is not evidence of discrete internal states versus a manifold. The "upper band" has knowledge logits of about 0-15, not especially high. The paper makes no such claim.
   - Fix: drop the discrete-vs-manifold hint, or label it "bimodal by construction of the prediction label".

6. **Figure 7 takeaway "prediction probe saturates near 0.96-0.99 by L17-18 on every benchmark".**
   - GSM8k-MC is about 0.89 at L17, 0.91 at L18, about 0.93 at L21 and reaches 0.96 only near L28-30 (Figure 13: 94.24% at the KAPPA(1) layer). The other five panels are 0.95-0.97 at L17-18.

7. **Figure 9 numbers: "knowledge vs random about 0.1-0.25".**
   - Qwen GSM8k knowledge-vs-random is 0.30 (L22), 0.33 (L23), 0.41 (L24) and 0.38 (L25) inside the KAPPA bands, and prediction-vs-random is 0.37-0.50. So the paper's isotropic "<= 0.2" baseline (B.3) disagrees with its own plotted controls in several panels, and not only for prediction-vs-random on TruthfulQA as the reading says. Extend the pitfall to Qwen GSM8k knowledge-vs-random.

8. **Figure 12 takeaway "at 10%, the TruthfulQA probe is barely above base".**
   - True for Llama (about 59 vs 56.7). Qwen is about 65 vs 58.8 (+6). Say "Llama".

9. **Table 12 takeaway: "when the knowledge probe is worse than the model (Qwen3 14B, and marginally Qwen3 4B), aligning with it lowers accuracy".**
   - Qwen3 4B: KAPPA(1) 47.5 and KAPPA(3) 47.2 are above base 47.1. Only KAPPA(6) 46.7 is below. Only Qwen3 14B loses in all three settings (53.0 / 61.3 / 60.8 vs 63.5).
   - Table 11 AGR also falls on more rows than the one listed: Llama MMLU Social Sciences KAPPA(6) 90.55 to 85.98, and Qwen MMLU Social Sciences KAPPA(6) 95.93 to 95.47.

10. **"No seeds" (architecture_location) is imprecise.**
   - Headline tables are single probe fits. The paper does have 3-seed training-data subsampling (Figures 11-13) and 10 distractor resamples (Tables 14-15). The 10 resamples reuse the same questions and the same probes, so the one-sided t-tests are not independent draws of probe or model. Say "headline results are single fits; seeds appear only in E.3".

11. **`relevance_to_project` "at 7-14B with a 32B check".**
   - All geometry (angles, CKA, W_U, scatter) is on Llama 3.1 8B and Qwen2.5 7B only. The 4B, 14B and 32B models have behavioural gap and KAPPA numbers (Tables 3, 12, 13) but no geometry. Our range is 9B-27B, so the geometry has not been shown in range.

12. **`actionable_fix` "Monitor: per-item KLD or disagreement ..." is the reader's suggestion.**
   - The paper tests no monitor. The knowledge probe is wrong about 20-25% of the time on TruthfulQA (KP 76-80%), so a disagreement flag would be noisy. Label it "untested idea".

## Fate labels (bears_on_fates)
- 3 not routed, "moderate": acceptable with its caveat. Note that near-orthogonal spans do not separate 3 from 2 bypass or 5 transformed (the reading says this for 5 only). The discriminating evidence is KAPPA controllability alone, so "moderate" is the ceiling.
- 1 readout/threshold, "moderate": too strong. The r = 0.97-0.99 diagonal is on no-gap tasks (MMLU-STEM, ARC) and says nothing about a threshold failure in the gap tasks. In Figure 3a the GSM8k prediction logits span about +-170 while knowledge logits span about +-40, so alpha = 30 may just compensate for probe logit scale. The beta-only result (+21 AGR on Llama, Figure 4) is consistent with a sign copy. Recommend "weak".
- 6 erased "moderate against" (knowledge probe stays on plateau, Figure 7): fine. 4 override "weak" (the paper's hypothesis is untested), 2 bypass "weak" and 5 transformed "weak": fine.

## Omissions
- **Fork point vs anisotropy onset.** In every panel of Figure 9 the random-control CKA rises at the same layer as the angle jump (Llama L17-18, Qwen L19; in Qwen GSM8k it goes from 0.03 to 0.3 across L18-20). The jump may partly reflect a residual-stream-wide change, not a decision-specific fork. The reading flags anisotropy for CKA but ties the "fork point" to decision formation without this control. Needed: a probe-vs-probe ceiling (two knowledge probes on disjoint halves) and random-direction angles at the same layers.
- **KAPPA layers are after the jump partly by construction.** Layers are picked only among those with prediction-probe validation accuracy above 85% (A.1, p15), so they cannot sit before saturation. The reading states the rule in `setup` but presents "KAPPA layers are all after the jump" as a finding.
- **Figure 4 range.** The alpha sweep is 0-20 on TruthfulQA only. The main setting alpha = 30 is outside it, and alpha = 80 appears only in Table 13.
- **Free-form transfer (Table 4).** KAPPA is applied at every generated token with probes from the MCQ position, judged by GPT-4o-mini, one run, no interval. The +2.5 on TruthfulQA is about the size of the distractor SDs (1.5-2.5, Table 14). The reading says "small" but not "not shown to be significant".
- **Angles are between probe-weight spans.** They reflect what a trained classifier uses (1000 epochs, best-validation checkpoint, no regularisation stated), not the variance subspace of activations. For our Grassmannian work, add activation-PCA subspaces as a second view.
- Instruct models only, MCQ format, no social pressure: the reading already says this in `relevance_to_project` and pitfalls[10].

## Summary for architecture_location and actionable_fix
- architecture_location: residual stream at the answer-prefix token is correct. Late-layer prediction-probe saturation at about L17 (Llama, measured) is correct. Fix items 1, 2 and 3: Qwen L19 is inferred, the angle usually climbs before it jumps, and the W_U angle differs by model.
- actionable_fix: KAPPA description, label requirement, weak transfer and the Qwen3 14B GSM8k loss (-10.5) are correct. Add item 4 (CAA competitive on 4 of 5 Table 3 models) and item 12 (monitor is untested).
- Vocabulary: uses "decodable", and "unused knowledge" appears only in quotes marks as a pitfall. One "knows-vs-acts" is the project's own idea name. Fine.
- reuse_verdict "adapt" and the lessons (per-layer random controls, probe-vs-probe ceiling, W_U comparison, loss accounting) stand. Add the fork-point control (omission 1).
