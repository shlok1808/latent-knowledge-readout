# Check: p2-separable-sycophancy-modes-2607.20146 (Jain, Yost, Abdullah, "Gotta Catch them all: the modes of Sycophancy", arXiv 2607.20146v1)

Verdict: GOOD, minor fixes applied. No fabricated numbers or quotes. The scoop call (low), the fate labels and the H1/H2/H3 calls hold. The fixes tighten three interpretive points that our design will lean on: what "<7% necessity" means, how far the causal metric is from behaviour, and how much Appendix N overlaps with H1.

## Quote check
- Before: 20 quotes, 0 failed. After my edits: 23 quotes, 0 failed (3 new pitfall quotes, pages 16, 18, 20).
- The checker tests only the first 120 normalised characters. I spot-checked the high-stakes quotes in full against paper.txt (p1 abstract, p5 "<7% change", p8 F6 "bypass its identifiable subspace", p19 "26 of 96", p33 "cosine 0.08-0.12"). All are exact.

## Figures vs manifest
- Manifest: 62 items. The reading has all 62, with none missing and none extra.
- I opened p05, p07, p17, p19, p20, p23, p28, p30 and p34. I re-rendered Figures 1 and 2 at 300 dpi and Figure 28 at 400 dpi.
- Confirmed: Figure 2 curves, bars and variance shares; Figure 3 cells, including "risk of rejection" .70/.70/.80/.50/.00; Figure 5 panels (the 4th panel is cosine distance, not Mahalanobis, and DB is not monotone); Figure 7 (flat to ~L23, peak L31-33, ~0 at L41); Figure 28 PA-DCA counts 6/22/20/2 (so "100% shared" in the caption is wrong); Figures 14-16; Figure 29 cells; Figure 36 (SI and DCA overlap in 2D); Tables 12-15.

## Errors found and fixed
1. **Figure 1.** The reading said the legend is unreadable. At 300 dpi it is readable. PA-DCA adjacency holds and SI sits apart. In the two plotted PCs the baseline is about equidistant from all three modes, so "baseline closest to DCA" rests only on Table 3 (PCA-50 neighbours). n=947 per cluster, not 948. Fixed numbers and mismatch_note.
2. **Figure 2 counts.** The reading had PA-DCA as "8 (+~1 at 5-10)" in the 10-20 deg bracket. The figure shows 9 in 10-20 deg and 1 in 5-10 deg. The smallest angle is ~10-13 deg. Fixed.
3. **Table 13.** The reading said the headers were garbled in the PDF. They are clean on the page; only paper.txt is garbled. I added that H10/H11 zero-ablation KL is PA ~2x SI (0.005 vs 0.002, 0.011 vs 0.007), which contradicts the caption's "fire comparably". Fixed.
4. **Table 2.** I added that the "necessity ceiling (L22)" row is not DCA's ceiling (DCA peaks at 0.044 at L26) and not SI's under ICA (0.068).

## Omissions added
5. **Necessity denominator (new pitfall, p16).** Necessity is divided by the effect of zeroing the whole layer, which is a huge ceiling. KL_ablate is measured against the unablated mode run, not toward baseline. So "<7%" is small by construction and is not shown to be a change in behaviour. This changes how we design the fate-2 (bypass) test: measure flips and logit margin, and use a neutral-run patch as the ceiling.
6. **First-token KL barely differs between mode and baseline at the output (new pitfall, p18).** Logit-lens KL is 0.05-0.11 at L40 and ~0 at L41. Every causal number in the paper is next-token KL at that token. Interventions are never tied to the judged multi-token behaviour.
7. **TOP-6 oddity (new pitfall, p20, and Table 15 note).** TOP-6 PA joint KL (0.0057) is ~6x smaller than the L22H9+L22H3 pair's joint KL (0.032, Table 14), although both heads are top PA-selective. TOP-6 membership is never listed. The "heads compensate" (redundancy) story is therefore unverified.
8. **Sample sizes.** The behavioural set is 1,000 situations x 5 (Figure 29 n sums to 1,000), while activations use 948. Per-mechanism n is ~120-131, apart from a stray ninth mechanism with n=1. Added to setup.data and to Figure 29.
9. **Appendix N and H1.** The reading said "modes are persona prompts, not pressures". That misses Appendix N, where pressure type alone (vulnerable interlocutor, status reward, hostile threat) induces separable states (96.7-100%, n=80). The rest of the H1 assessment stands: each mode has its own content, so the result shows the pressure context is decodable, not that fates differ. It slightly raises framing overlap; it does not raise scoop risk. Updated bears_on_H1_H2_H3 and overlap_with_our_plan.

## Overclaims softened
10. **actionable_fix and md H2.** The reading claimed "single-direction ablation is unlikely to be an adequate Stage B fix". The paper never measures behaviour after ablation, so this is now worded as a warning, not evidence.
11. **bears_on_group_angle.** The reading called L22H10 "the analogue of our 'head that reads the user's claim'". It attends to persona (system-prompt) tokens, its ablation KL is ~0.005 nats, and it is redundant with L22H11, so it is now a "loose analogue". I pointed to their situation-attention-fraction metric as the better template (attention on the pushback turn, flipped vs resisted).
12. **architecture_location.** Added that L22H10/H11 are PA ~2x in ablation and form a redundant pair, that the logit lens collapses by L41, and that all head effects are 0.001-0.018 nats.

## Checked and left unchanged
- Fate labels: fate 2 as a method caution (now with the denominator point), fate 4 weak, fate 5 as a method (offset before Procrustes), fates 1/3/6 none. Correct.
- Scoop risk: low. The paper has no factual items, no correct-answer probe, no flipped/resisted split, no fixes and no DPO. Correct.
- H3: support by analogy only; it is a mode-label probe. I added that qualifier.
- Skipped sections: none. Sections A-O are all listed, and the references were skimmed.
- The reading's own caption flags (Figures 5, 19, 23, 28, 32, 34, 36) are correct.

## Files changed
- `reading/p2-separable-sycophancy-modes-2607.20146.json`: Figures 1, 2, 29 and Tables 2, 13, 15; setup.data; 3 new pitfalls; bears_on_fates[0]; bears_on_group_angle; bears_on_H1_H2_H3; overlap_with_our_plan; actionable_fix; architecture_location.
- `reading/p2-separable-sycophancy-modes-2607.20146.md`: the matching table rows, Setup, Pitfalls, and the H1/H2/Location bullets, plus a checked note at the end.
- Pre-edit backups: `/private/tmp/claude-501/-Users-shloki-agent-failure-probes/260ce1f6-34ed-4464-864c-359284c95eac/scratchpad/p2_backup.json` and `p2_backup.md` in the same folder.
