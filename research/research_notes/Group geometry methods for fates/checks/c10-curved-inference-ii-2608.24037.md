# Check: c10-curved-inference-ii-2608.24037 (Curved Inference II, Manson, arXiv 2608.24037v1)

Verdict: **NEEDS FIXES** (minor; no REDO). Quotes, tables, figure read-offs and the main call ("avoid as evidence; adapt only sublayer-resolution sampling") are faithful to the paper, and the reading catches the hard-to-see problems (swapped Fig 2/3 images, empty placeholder panel, mislabelled axes, p = 0.138 / 0.794 missing from every table, reversed "reduction" percentages, no probe baseline). The fixes are four wording errors and omissions, two of which sit in `architecture_location` / `key_results`. None changes fate mapping, `actionable_fix` or `reuse_verdict`.

## 1. Quote check
- `python3 quote_check.py c10-curved-inference-ii-2608.24037`: 20 quotes, 0 failed.
- That script tests only the first 120 normalised characters and not page numbers. I re-checked each quote against per-page text of `paper.pdf`: all 20 are present and all sit on the page the reading gives (including `limitations_quote` p19 and `future_work_quote` p20).

## 2. Coverage checks
- Manifest lists 8 floats (Fig 1-5, Tables 1-3). The reading has all 8, with the right pages (3, 4, 11, 11, 12, 13, 14, 15). Nothing missing.
- `sections_read` covers the main text and Appendices A-E and the Notes. Nothing skipped.
- Images opened: p03 (Fig 1), p11 (Tables 2-3), p12 (Fig 2, plus a 4x crop of the text box), p13 (Fig 3), p14 (Fig 4), p15 (Fig 5). Table 1 (p04) checked from `paper.txt`.
- Compared with text and images:
  - Tables 2 and 3 match Appendix E.1.1, E.1.2, E.2.1 and E.2.2 cell by cell (including the unanimous n values 34/39/31/47/50 and 63/60/57/61/52, which sum to 201 and 293).
  - Fig 2/3 swap confirmed. The Fig 2 caption (p12) says Gemma, N = 61, but the image is LLaMA, N = 47, three groups. The Fig 3 caption (p13) says LLaMA, N = 47, but the image is Gemma, one group, N = 61, mean 9,769, median 8,847, sigma 4,488, one outlier near 34,000. Top-right histogram of Fig 3 has y-label "Surface Area (A')" and x-label "Transparency Level" while plotting counts against A'. Fig 2 has the empty "Additional Analysis" panel. The reading is right on all of it.
  - Fig 4 is a single bar with a title that says "Separates Transparency Groups". Fig 5 matches its caption; the group sizes (2, 5, 40) are missing from the caption. Both match the reading.
  - Fig 1 is a schematic with no data (L = 28, T = 50 from the legend).
  - 0.138 and 0.794 appear only in the p15 text, in none of Tables 2-3 or E.1-E.2. The reading is right.
  - 201/500 = 40.2% kept and 293/500 = 58.6% kept, so the paper's "40% reduction" (LLaMA) and "60% reduction" (Gemma) are swapped. The reading is right.
  - The A' normalisation by response length appears only in Limitations (p19); Methods 3.5.1 says "total weighted path deformation". The reading is right.
- Fate names match the project list. Strengths are all "weak". Vocabulary is fine. No causal or probe claim is attributed to the paper. Code is marked "not checked".
- Table 1 sizes (LLaMA 2.3B, Gemma 1.3B, Gemma d = 2,048) are inconsistent with the paper's own "3B / 1B" text on p16. From outside knowledge LLaMA-3.2-3B is about 3.2B and Gemma-3-1B has a smaller hidden size. Not checkable here; the reading hedges it correctly.

## 3. Errors (what the reading says / what the paper says)
1. **Cross-model "direction" is echoed as established (moderate; `architecture_location`, `key_results[6]`).**
   - Reading: "Two small models (about 1B and 3B) agree only on direction (explanatory > evasive)", and "The only cross-model regularity: explanatory responses have higher A' than evasive ones."
   - Paper: p16 "Explanatory responses consistently showed higher surface area than evasive responses". No explanatory-vs-evasive group means exist anywhere; E.1.3 (p31) and E.2.3 (p32) give transparency-level means only. E.2.3 has reversals or non-monotone orderings: Gemma malicious high (n = 2) 9,425 vs low (n = 50) 10,636; Gemma strategic low 8,769, high (n = 1) 11,411, medium (n = 7) 17,471.
   - Fix: "The paper asserts this ordering with no response-type means; Gemma's transparency-level means (non-low groups n = 1-7) are not consistently ordered."
2. **η² values attributed only partly (minor; Table 2 `text_matches_figure`, `key_results[2]`).**
   - Reading: the η² values for strategic and malicious (0.847, 0.600) are full-consensus values.
   - Paper: the same holds for honest (p15 text 0.575 vs E.1.1 0.58) and for deceptive. Deceptive is the one that matters: the Fig 3 caption (p13), the p14 text and the p15 summary give "η² = 0.448" next to N = 47 (unanimous), but E.1.1 (p31) has 0.45 for the n = 100 full-consensus set, and E.1.2 gives no unanimous η². `key_results[2]` attaches 0.448 to the 40/5/2 group without that caveat.
3. **Fig 2 text box mis-transcribed (nit).**
   - Reading: 'high: 2396 [n<2]*'.
   - Paper p12 (image, zoomed): "high: 2396 [n=2]*", with the footnote "*n < 5: interpret with caution". The figure itself warns about n < 5, yet the p14 text interprets the n = 2 and n = 5 groups.
4. **r > 0.99 read as "path length (salience)" (minor wording; `key_results[7]`, `methods[1]`).**
   - Reading: "A' is almost perfectly correlated with its components, so it is in effect a path-length (salience) measure."
   - Paper p16: "Surface area correlations with salience and curvature contributions exceeded r > 0.99", that is, with both. A' = Σ(S + γκ) over the same steps, and the Fig 2 box (p12) says "Perfect correlations expected". r > 0.99 with both parts cannot show that salience dominates. It fits all three tracking the number of steps or tokens. Mark it "our inference: likely a length / step-count driver".
5. **Table 2 takeaway slightly broad (nit).**
   - Reading: "A' differs significantly between judge-label groups for every strategy under both label sets."
   - Paper p11 Table 2: unanimous strategic response p and persuasive transparency p are "(insuff.)", so each of those strategies has one testable dimension. Say "at least one testable dimension per strategy".

## 4. Important omissions
1. **Prompt manipulation barely works, and the paper presents that as success (medium; supports our "sort by observed behaviour" rule).**
   - Paper p16 ("Complete Classification Consensus"): Gemma honest 60/63 (95%), persuasive 55/57 (96%) and deceptive 61/61 are all "low transparency" and "evasive". It calls this "methodological success ... prompt strategies are working as designed". The "honest" prompt therefore produces evasive output. In LLaMA the unanimous "honest" set is 21 low / 9 high / 4 medium (E.1.3, p31).
   - LLaMA low-transparency means are almost identical across prompts: 1,418 / 1,269 / 1,276 / 1,220. High means are 2,490 / 2,235 / 2,396 / 2,400. A' tracks the judged class of the output, not the prompt strategy.
2. **Medium > high is a pattern, not a deceptive-only quirk (low-medium).**
   - E.1.3 (p31): medium exceeds high in all four LLaMA strategies with data (3,056 vs 2,490; 5,299 vs 2,235; 2,971 vs 2,396; 4,228 vs 2,400), with medium n = 2-5. The reading notes it only for deceptive (Fig 5). It supports the reading's worry that A' follows hedging or length rather than disclosure. Add the list.
3. **"Strengthened" covers 2 of 5 Gemma strategies, and filtering made significant results untestable (medium).**
   - Table 3 (p11): full-consensus persuasive response p = 0.033 and deceptive response p = 0.032 become "(insuff.)" and "Single class" after filtering. Strategic goes 0.006 → 0.003 on response type, while its transparency test goes from 0.001 to "(insuff.)". Gemma full-consensus η² is 0.01-0.05 except strategic 0.15 (E.2.1, p31), yet p16 says "effect sizes remained consistently large across both models". Large d values appear only in filtered subsets with 1-3-item minority groups. The reading says "weak or untestable" and covers honest; add these.
4. **Reliability is overstated by the paper and not flagged as a contradiction (medium).**
   - Limitations (p18): "While inter-rater reliability was good". C.2 (p27): Krippendorff α is 0.309 / 0.443 (Gemma) and 0.364 / 0.523 (LLaMA), labelled Poor, Tentative or Moderate. The reading quotes only the Gemma α.
   - C.2 also says responses without 2-of-3 agreement were excluded, yet E.1.1 and E.2.1 (p31) give n = 100 for every strategy under "full consensus".
5. **Metric definitions disagree inside the paper (medium; matters because the reading recommends borrowing the U^T U metric).**
   - 3.4.2 (p6) gives curvature with G-norms (parameter-invariant form). A.3.2 (p23) gives κ = ||Δx(ℓ) − Δx(ℓ−1)|| / ||Δx(ℓ)||², with no G.
   - A.1 notation (p22) defines A' = Σ||Δx||_G with no curvature term. 3.4.1 (p6) and A.3.1 (p23) include γκ.
   - Salience is x_i − x_{i−1} in 3.4.1 and x_{i+1} − x_i in 3.4.2 (p6-7).
   - A.2.1 (p22) feeds the MLP x(ℓ−1) while A.2.2 feeds it the post-attention state.
   - The paper never says which version the code runs. Add to `pitfalls`; keep the G-norm idea but label it "our extension, not validated here".
6. **Setup details missing (low).** Temperature, top-p, max length, seeds and base vs instruct are not stated (`grep` finds none; "instruction-tuned variants" appears only as future work, p19). Add to `setup`.
7. **Overclaims not listed in `pitfalls` (low; verdict already "avoid").**
   - "universal to transformer computation" (p16; A.6.3 p24) from two sub-4B models.
   - "provides definitive validation" (p8 and p17) from p = 0.048 on a 3-vs-60 split, uncorrected across about 20 tests.
   - "Without backdoors, labels, or probes" (abstract, p1) while using three LLM-judge labels.
   - "The shape of thought ... may be harder to hide" (p18) against the paper's own admission that the metric can be evaded (p19), with no adversarial test.

## 5. Suggested edits to the reading (not applied; readings are not edited here)
- `architecture_location`: replace "agree only on direction (explanatory > evasive)" with "the paper asserts explanatory > evasive in both models but reports no response-type means; Gemma's transparency-level means are not consistently ordered".
- `key_results[6]`: add "(no response-type means reported; see E.2.3 reversals)".
- `key_results[2]` and the Table 2 note: mark η² = 0.448 as the full-consensus (n = 100) value shown beside N = 47.
- `key_results[7]` and `methods[1]`: change "path-length (salience)" to "cannot be separated from its parts; likely length / step-count (our inference)".
- Fig 2 numbers: "[n=2]*" with the "*n < 5" footnote.
- `pitfalls`: add items 4.1-4.5 and the overclaims in 4.7.
- `setup`: add "no temperature, seeds or base/instruct stated".

## 6. What the reading gets right on the high-stakes fields
- `bears_on_fates`: all "weak" with reasons; the pullback-norm idea is labelled "our idea, not tested". Correct.
- `actionable_fix`: monitoring only, unvalidated. Correct. The paper has no probe, no causal test, no training or inference fix.
- `bears_on_group_angle`: one scalar per response, so no subspace, principal-angle or Procrustes information. Correct. The suggestions to keep sublayer sampling, report normalised and unnormalised residuals, control for residual norm and token count, and size and match groups before comparing geometry are sound.
- `reuse_verdict`: "avoid (as evidence); adapt only sublayer-resolution sampling". Agree.
