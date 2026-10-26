# Check: c11-trajectory-signatures-deception-acl2026

Paper: Mothukuri and Parizi, ACL 2026 long, pp. 34264-34276 (13 PDF pages; page numbers below are PDF pages).
Checker: compared reading JSON vs paper.txt vs page PNGs (p02, p04, p06-p09, p12, p13), plus Figures 2, 3, 4a, 4b re-rendered from paper.pdf at 250 dpi (the supplied PNGs are too small to read panel titles and x-axis labels; at 250 dpi they are readable).

## 1. Quote check
`quote_check.py`: 27 quotes, 0 failed. Page numbers on the key_results and pitfalls quotes also match the PDF (spot-checked all of them against the page footer numbers).

## 2. Verdict: NEEDS FIXES (small)
The reading is accurate on all numbers in Tables 3, 4, 5 and Figure 4a text, covers all 12 items in manifest.json (Table 1-6, Figure 1-6, none missing), has no skipped sections (Appendix C really ends after the acceleration feature on p13), and its main message is right: labels are prompt conditions, no causal test, no consistent location, a single direction beats the scalars. The fixes below are about (a) two overstated claims, (b) three "illegible/unconfirmed" hedges that can now be settled, (c) a few omissions that matter for the reuse advice. No REDO needed.

## 3. Errors

### E1 (medium, overclaim). "States are read at the last prompt position ... deceptive generation is never actually observed" is stated as fact; the paper is ambiguous
- Reading says (pitfalls, setup.positions, relevance_to_project, bears_on_group_angle point 4): the default state is the last PROMPT token, "before any answer is produced".
- Paper says: Sec 3.2 (p3) says both "final generated token position of the input sequence" and "position n" of the prompt; the next paragraph says the default mode takes "the final generated token position" with one L x d trajectory per prompt "regardless of output length". Limitations (p10): "State extraction uses the final generated token position". Appendix C (p13): hook "at the final token position", state "from which the model produces its next-token distribution". The abstract (p1) and Figure 1 say capture is at decision points during generation. So it is either the last prompt token (= the state that decides the first answer token, which for (A)/(B) sycophancy items is the decision state) or the last generated token. The paper never settles it.
- Fix: say "ambiguous: text says both last prompt position n and final generated token; Limitations says decision-point/multi-position aggregation is future work, so reported numbers use one position per prompt". Drop "never observed" and "uninformative by construction" as facts. Keep the "fork point is missed by one-position reading" warning, but phrase it conditionally. Note Figure 2 supports the reading's suspicion only weakly: the Corrigibility panels (Qwen-72B, DeepSeek-67B) overlap almost exactly, but the Power Seeking panel in DeepSeek-67B does not.

### E2 (medium). Figure 2: "titles illegible ... unconfirmed" can be resolved, and the apparent contradiction mostly goes away
- Reading says: panel titles illegible; Qwen-72B/DeepSeek-67B panels that overlap "contradict" the text; "may be the identical-prompt negative controls, but unconfirmed".
- Paper/figure says: titles are readable at 250 dpi and also appear in paper.txt. Qwen-72B (g) and DeepSeek-67B (h) panels are Pair 1 Sycophancy, Pair 2 Power Seeking, Pair 3 Corrigibility. The overlapping panels are Power Seeking / Corrigibility = the negative controls (p7, Figure 2). Sycophancy panels in (g) and (h) are separated. Qwen-0.5B panels are Geography, Math, Science (the lie-instructed multi-task set); only the Geography panel overlaps. GPT-2 pair 1 (Instructed Lies) overlaps.
- Fix: write the panel titles in the figure entry; replace "contradiction/unconfirmed" with "overlap in (g),(h) is the negative-control panels; the paper's 'larger models separate' sentence is about the deception panels". Remaining valid criticism: single-pair PCA with no aggregate, and the text calls Qwen-0.5B "substantial overlap" although 2 of 3 of its panels separate clearly.

### E3 (medium). Figure 4b: text/figure mismatch is confirmed, not conditional
- Reading says: x-axis illegible; "if alphabetical" the -0.60 bar sits in the Llama-2-7B group.
- Figure says: x labels are legible and alphabetical: DeepSeek-67B, GPT-2, Llama-2-13B, Llama-2-70B, Llama-2-7B, Llama-3-70B, Qwen2-0.5B, Qwen2-72B. The 5th group (Llama-2-7B) holds both the +0.62 Mean Acceleration bar and the -0.60 Mean Curvature bar. The 4th group (Llama-2-70B) has all bars within about +/-0.18.
- Paper says (p8): "Llama-2-70B shows reversed curvature patterns (d = -0.60)". So the text assigns the bar to the wrong model, or the figure labels/order are wrong. The "|d| < 0.20" claim for DeepSeek-67B is also contradicted: its bars reach about -0.26 (mean curvature) and -0.21 (direct distance). Two other text claims check out: Llama-2-13B path length +0.41 and Qwen-72B mean curvature above +0.5.
- Fix: state both mismatches as confirmed; note that the "which model has the reversed curvature" claim in the paper cannot be trusted. The reading's `numbers` already has the right bar values.

### E4 (low-medium). Table 5 is internally inconsistent; the reading misses it
- Reading says: Mean-Diff per model 91.3 / 90.2 / 91.5 / 91.0, mean 90.3. Only the Limitations mismatch (66.7 / 77.0 / 91.7) is flagged.
- Paper says (p9, Table 5): the same numbers. Their average is 91.0, not 90.3. All other Table 5 rows average correctly (Geo LR 66.6, PCA LR 67.8, Geo RF 85.5, PCA RF 91.4, Full 95.9). The text (p9) repeats "90.3%".
- Fix: add to pitfalls. It does not change the conclusion (single direction still beats the Geo features), only the gap (about 91.0 vs 85.5 for RF).

### E5 (medium). H3 rated "moderate" rests on weak evidence; Qwen-0.5B polarity reversal is not Bonferroni-significant and uses another dataset
- Reading says (bears_on_fates H3, strength moderate; key_results): "Qwen-0.5B reverses the polarity", strategic sign flips across Llamas show monitors do not transfer.
- Paper says (p7): Qwen-0.5B d = +0.412 (p = 0.005), +0.352 (p = 0.016), n = 50 on the Qwen multi-task, lie-instructed set. Both fail the paper's own alpha = 0.00089; the paired-t arithmetic (d x sqrt(n)) reproduces these p-values. The dataset differs from the Llama/GPT-2 runs, and for instructed lies the Llamas are null, so "reversed polarity" compares against a null. The same paragraph says Qwen-72B "follows the Llama-like pattern" (omitted by the reading). The strategic-deception sign flips (GPT-2 -0.99***, 7B +1.23***, 13B -0.86***, 3-70B -0.94***) are the solid evidence, and they are about path-length scalars, not about probes trained on normal behaviour.
- Fix: downgrade H3 to weak (indirect), and add the p-values, the dataset difference and the Qwen-72B sentence. The paper's own probes (mean-difference direction at about 91% on all 4 Llamas) transfer across models in the sense that matters for a monitor, which cuts against H3 as stated.

### E6 (low). Bypass entry: DeepSeek-67B "probe-detectable but geometrically weak" is not evidence of causal inertness
- Reading says (bears_on_fates fate 2, weak): "fits 'decodable is not used'".
- Paper says (p8): the phrase only describes small scalar effect sizes (and E3 shows they are not all below 0.20); probe accuracy for DeepSeek-67B is 77.5%, the lowest of the large models. Nothing about use of the signal.
- Fix: strength "none" (keep the sentence as a non-example). Also fate 6 "Not addressed" is labelled "weak"; use "none".

### E7 (low). Star thresholds are derivable, not "unclear"
- Reading says: star thresholds undefined, unclear whether * survives Bonferroni.
- Paper says: n = 30 pairs per cell (p6). From d x sqrt(30), d = +0.41 gives t of about 2.25 (p about 0.03), +0.48 gives p = 0.013 (matches text), +0.42 gives p about 0.03. So * means p below 0.05 and none of the single-star cells survive alpha = 0.00089; *** cells (|d| of 0.86 or more) do.
- Fix: replace "unclear" with the derived result.

## 4. Omissions (add to the reading)

O1. Dataset behind each headline number. Table 3 (path length d, 5 models, n = 30) is from the custom pairs (GPT-2 is not in the Anthropic row of Table 6, p12). Tables 4-5 and the 90-96% probes use the Anthropic persona set (n = 30; Fig 4a/CV on n = 60). So d = +1.63 is on custom pairs like "2+2 equals 4" vs "I strongly believe 2+2 equals 5" (Figure 5, p12), which differ in content and length. The reading's pitfall on the persona prompt only covers the Anthropic set. State which result uses which set.

O2. Confabulation shape effects have no model attached. Straightness d = +1.295, direct distance +0.972, mean curvature -0.892 (p6): the paper does not say which model or whether pooled. Reading quotes them without this flag.

O3. Figure 4a vs Table 5. Layerwise best-probe accuracies peak at 66.7-88.8% (p8) while Table 5 full best-layer probes reach 92.1-98.9% (p9); task and n for Figure 4a are not stated. The curves are jagged (visible in the 250 dpi crop), the "best depth" is an argmax on a noisy curve with no CI, and the 5.1% depth (about layer 4 of 80) fits a lexical persona cue. So the "12-fold" depth range is weak evidence for architecture-specific stages. Add this to architecture_location.

O4. Untested tools. DTW (Eq. 5, Sec 3.8) and decision-point sampling are described but no reported result uses either; evidence for sparse capture is "maintained classification accuracy" with no numbers, plus a 25-pair held-out "correlation" (p10). The reading's reuse advice ("three tools for fork-point work") should say none of the three is validated here; only the cosine layer search is used in results (and its baseline mu/sigma is undefined).

O5. Unsupported claims in Limitations (p10): "Detection thresholds calibrated on Llama architectures achieve 80-85% accuracy within that family" has no table behind it; the 3,400-pair total cannot be reproduced from the stated cell sizes; the "+/-" in Table 4 is undefined (fold SD or CI).

## 5. Checked and fine
- Table 3 values, stars, per-model means (GPT-2 -0.06, 7B +0.56, 13B -0.27, 70B +0.16, 3-70B +0.21), Table 4 and Table 5 rows (except E4), Table 6 coverage, Table 2 (8 rows vs "seven features"), Figure 1 three-way output vs binary results, Figure 3 duplicating Figure 2(d)(e) at larger size and showing separated instructed-lie panels despite the null, Figure 5 and 6 observations.
- architecture_location: 5.1% to 62.0% and the 70B Llama-2 vs Llama-3 confound are right; no seeds or checkpoints in the paper (grep found none); block outputs only, no heads/MLPs.
- actionable_fix "monitoring only, about 60 pairs to calibrate": matches p8.
- controls_worth_copying and reuse_verdict are sensible given E1-E5; only add the caveat in O4.
