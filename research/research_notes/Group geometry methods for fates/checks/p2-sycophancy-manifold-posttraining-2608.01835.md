# Check: p2-sycophancy-manifold-posttraining-2608.01835 (Wang, Yang, Wang, Zhang, "Rewriting or Reweighting? A Geometric Account in Language Models", arXiv 2608.01835v1)

Verdict: GOOD, small fixes applied. No made-up numbers or quotes. The tables, quotes and pages hold up. The reading's scoop-risk call stands: moderate on framing, low on substance. I added one caveat that matters for Stage B: the DPO "geometry conserved" number sits at a measurement floor the paper never reports. That makes "DPO reweights / hides" weaker evidence than the headline suggests.

## Quote check
- `quote_check.py` before the fixes: 23 quotes, 0 failed. After: 29 quotes, 0 failed. The script tests only the first 120 normalised characters.
- Stricter pass (my own script on per-page `pdftotext`): all 29 quotes match in full and sit on the page the reading gives, including the 6 new ones.
- One empty-quote pitfall (the internal-inconsistency list) is the reader's own claim. I re-verified every item in it.

## Figures vs manifest
- Manifest: 68 items. The reading has all 68, plus Figure 11 (p26), which the manifest missed because its caption is garbled in the text layer. None missing.
- I opened p07 at 220 dpi (Table 2, Figure 5), p27 at 200 dpi (Figure 13) and p34 at 220 dpi (Figure 14, Tables 14-16). Against paper.txt I checked Tables 6, 7, 15, 19, E.5-E.7, the Appendix L prose, the B.4 NOC proxy, R (dataset) and the Y training configs.
- Confirmed: all numbers in Table 2a/2b, Table 6, Table 7 and Table 15. The Figure 5c phase-portrait reads (SFT 9 to 24.7 deg, 0.454 to 0.645; RL 9 to 10.1, 0.454 to 0.474; SFT->RL 24.7 to 26.1, 0.646 to 0.663). The Figure 5a repetition mismatches with Table 2b. The Figure 13 description. The Table 2a vs Table 15 random-baseline conflict (0.367/0.317 vs 0.667/0.817; top-2 NOC 0.295 vs 0.495). Truth PPL 1485.1 vs 1.48. The NOC-as-unsigned-proxy claim (B.4).

## Errors and omissions found (and fixed)

1. **Missing: the DPO angle is mostly a first-checkpoint offset (high stakes for H2 and Stage B).**
   - Table 7: DPO-up and DPO-down are identical at step 6 (S 0.9498, 9.06 deg, Syco Prob 0.3550), even though the objectives are opposite. Figure 5c: the SFT first checkpoint (step 10, lr 1e-5) also sits at ~9 deg and preference ~0.454. Figure 5d: every lineage drops from S=1.0 to ~0.95 at its first stored checkpoint.
   - So 9.06 of DPO's 9.68-10.10 deg appears after 6 tiny steps in either direction. That looks like a floor from chart re-estimation or checkpoint conversion. After step 6, DPO moves 1 deg or less. The paper has no base-vs-base (re-extraction or seed) noise control. The reading had only flagged the identical rows as "odd".
   - Fixed: new pitfall; Table 7, Figure 5, key_results[0], the bears_on_fates Stage B row, H2 text and overlap item (h).

2. **Missing: repetition DPO is ~0.010 epoch** (60 x 128 / 755,920, p63). Repetition SFT is 100/600 steps. Added to the setup and the pitfalls. The reading's "unmatched update size" point is right and is now quantified for both behaviours.

3. **Missing: the behaviour column was unlabelled, and its two metrics disagree.** Table 2b "Behav." is p_b/m_b: the probability of the sycophantic target / the sycophantic preference score (E.7, p14). For SFT-down, p_b rises to 0.442 (vs 0.355 at the first DPO checkpoint) while preference falls to 0.330. Base-model behaviour is never reported. Added to setup, the Table 2 entry and the pitfalls.

4. **Missing: chart pipeline may re-select neurons per checkpoint.** E.6 says "train or reuse sparse classifiers ... zero-pad bases to a common ambient coordinate system". So S and angle can mix neuron re-selection with representation change. Added as a pitfall. This is a plausible source of the ~9 deg floor.

5. **Overclaim (pitfall wording).** The reading said the ACT chart "whose rotation is tracked is mostly non-behavioral variance", and attached a quote about compression that does not support it. The paper shows only that PC1 is nearly orthogonal to the classifier direction (signed cosine mean 0.062, across the 23 atlas models). Lower PCs may carry the behaviour direction. Reworded, and the quote swapped for the "Cos." definition (p28).

6. **Missing headline overclaim.** Figure 1 and the abstract say reward optimization "redistributes occupancy and modifies readout gain or threshold". Threshold is never estimated, and Appendix S finds DPO occupancy and gain changes small and estimator-dependent. Added as a pitfall with the p1 quote.

7. **H3 nuance added.** R^G uses fixed dataset labels on teacher-forced text, not the model's own flips. So it tests whether an old probe still separates the same text, not whether it catches the model's failures. I also noted a tension: SFT-up R^G is 0.966 at the population level, yet the 4 Figure 13 items look poorly organised by the base chart. Updated bears_on_H1_H2_H3 and the Figure 13 takeaway.

8. **Figure 11 note corrected.** The reading said its 25.5%/13.9% axes are "likely a different model". Figure 13 uses identical axes and is explicitly the base Llama-3.1-8B chart. So this is an internal inconsistency with Table 4 (Llama-3.1-8B ACT PC1 0.426), possibly token-level vs span-pooled PCA. The paper is silent.

9. **Figure 14 / architecture_location.** The sycophancy peak is at ~layer 13 (not ~12), with secondary peaks at ~23 and ~27, spread over nearly all 32 layers. I added that no attention components are analysed.

10. **Smaller additions.** Figure 5d's SFT x-axis ends at 120 steps although SFT-down ran 360. I added a key result with the Table 19 counts (instruct causal gain up in 7 of 10 pairs in both NOC and ACT; reliability split 5/5 and 4/6).

## Not changed (checked, correct)
- overlap_with_our_plan items (a)-(g) and the scoop-risk call. The data is Anti-Sycophancy-DPO opinion claims (e.g. "All smartphones are spying on us all the time."). There is no decodable correct answer the model held before pressure, no per-item fates, no pressure types, and no fate-matched fixes.
- Fate labels: bypass via the ACT/NOC quadrant is conceptual only (NOC is unsigned). Override is only an occupancy proxy. No erasure (native G 10.4-17.7). Transformed rests on SFT rotation plus 4 qualitative items. All strengths stay weak to moderate.
- Sections_read covers the full paper (A-Z appendices). No skipped sections found.

## Files changed
- `reading/p2-sycophancy-manifold-posttraining-2608.01835.json`: setup.post_training_protocol; figures Table 2, Figure 5, Table 7, Figure 11, Figure 13, Figure 14; key_results[0] plus 1 new; pitfalls (2 reworded, 5 new, inconsistency list extended); bears_on_fates[0]; bears_on_H1_H2_H3; overlap_with_our_plan (items h, i); architecture_location.
- `reading/p2-sycophancy-manifold-posttraining-2608.01835.md`: regenerated from the JSON. The renderer reproduced the old .md byte-for-byte before the fixes.
