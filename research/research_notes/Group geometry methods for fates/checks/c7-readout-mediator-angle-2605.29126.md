# Check: c7-readout-mediator-angle-2605.29126 (Fadnavis, Kanakaraj, Wyss, "When and How Long? The Readout-Mediator Angle in Temporal Reasoning", arXiv 2605.29126v1)

Verdict: **NEEDS FIXES** (no REDO). All 45 quotes sit on the stated pages, all 47 manifest floats are covered, and the large majority of read-offs and numbers check out. The reading is also unusually good at catching the paper's own text/figure contradictions (I confirmed the ones I re-checked). The fixes are small: four wording or omission problems that touch `key_results`, `architecture_location` and `bears_on_group_angle`, plus one uncovered main-text figure (Fig 6). Suggested edits are in section 5.

## 1. Quote check
- `python3 quote_check.py c7-readout-mediator-angle-2605.29126`: 45 quotes, 0 failed.
- That script tests only the first 80-120 normalised characters and not page numbers. I re-checked every quote (including `limitations_quote` p23 and `future_work_quote` p9) against the per-page split of `paper.txt` (55 form feeds): all 45 are present and all 45 sit on the page the reading gives.

## 2. Coverage checks
- Manifest lists 47 floats (Figs 1-5, 7-38; Tables 1-10). The reading has all 47 with correct pages. No extras.
- Manifest and reading both omit **Figure 6** (page 9, "Adversarial dissociation"). It is a main-text figure that carries the "monitor blind spot is total" headline. There is no p09 image in `pages/`, so I rendered p9 from `paper.pdf` and looked at it (details in error 5).
- `sections_read` covers the main text, References and S1-S54 (every `S<n>` heading in `paper.txt` is listed). Nothing skipped.
- Images opened: p02 (Fig 1), p05 (Fig 2), p06 (Fig 3), p07 (Figs 4-5), p08 (Table 1), p13 (Table 2), p14 (Tables 3-4), p15 (Table 5, S3, S4), p17 (Fig 7), p19 (Fig 8, S11 table), p22 (Fig 10), p29 (Fig 13, S26-S27), p37 (Fig 18); p09 rendered separately.
- Not opened as images (checked against `paper.txt` and the reading only): Figs 9, 11, 12, 14-17, 19-38 and Tables 6-10 (p21, p23, p28, p30, p34-p38, p40-p55). For those the reading's text numbers (Table 6, Table 9/10, Fig 36 rho values, MLP stage numbers, S49 0.97 to 1.09) match `paper.txt`; the figure read-offs I did not re-verify are flagged as "approx" in the reading itself.
- Image checks that agree with the reading:
  - Fig 1: "Jun 27th" vs "June 22nd" in the schematic, and the QK panel's "+-10 days not found" vs the text's +-7, are both real.
  - Fig 2: panel A probe bar is 41.0 (text says 0.6 pp); panel C ratios 190x/1050x/262x; panel D drops 44/42/51/45 pp.
  - Fig 3D: IoU = recall = 1.0 at K=24 (circled), while S5 (p16) says IoU/recall 0.14/0.25 at K=24. Real contradiction.
  - Fig 4/5 text mismatches (D uses 4-vs-4 null for 10-dim TFA components; 5D rises ~0.97x to ~1.15x vs S49 "1.09x" on p45) are real.
  - Fig 8: caption probe rho=1.0 vs table 0; panel A 39.5/38.9/36.0/28.0/0.0 vs S11 table 42.0/5.0/11.5.
  - Fig 13: plot shows mean cos^2 for one setting, no null line; S27 text quotes a k=4 null of 89.6 deg.
  - Fig 18: the three prompts have near-identical per-token energy profiles (template-driven). Real.
  - Fig 10: caption (C) says accuracy by delta-quartile; panel C is per-tier Pearson r. Real.
- Fate names match the project list; strengths are hedged (mostly weak/moderate), and the reading says the paper has no pressure, no flips and no competing signal. Vocabulary is "decodable". The pitfall on 'DAS' (ablation-maximising search shows controllability, not mediation) keeps causal controllability apart from endogenous mediation, as required.

## 3. Errors (what the reading says / what the paper says)
1. **`key_results[5]` "seed-stable" is stronger than the paper's own numbers (low-moderate; bears on "stable across seeds").**
   - Reading: "The causal subspace saturates at k=4 and is seed-stable" (quote p5: "the basin is unique across seeds (CCA>0.94").
   - Paper, S1 (p14): across five seeds at k=4 "pairwise basis angle in [12 deg, 34 deg]. The solver finds different but equally causal k=4 bases". S28 (p30) says "compact basin (pairwise CCA > 0.85)", not 0.94. S29 (p31) calls the individual solutions "solver-dependent". S30 (p31) says two independently trained bases are "<15 deg" apart. So the stable object is the subspace's effect and a CCA-level overlap, not a single basis; the paper itself gives two different CCA thresholds (0.94 vs 0.85) and an angle range that reaches 34 deg.
   - Fix: say "five seeds give different k=4 bases (pairwise 12-34 deg) that are all causal and CCA-correlated (>0.94 in S1/S29, >0.85 in S28)".
2. **`architecture_location` checkpoint sentence mislocates the head change and omits that L* wanders (moderate; high-stakes field, the project wants "stable across checkpoints").**
   - Reading: "Across checkpoints (Pythia 1.4B): head counts and circularness change in a window of 10^3-5x10^4 steps; probe R^2 is high from init."
   - Paper S7 (p16-17) and Fig 7 (p17): boundary heads go 133 (step 0), 139 (16), 119 (128), **50 (512)**, 46 (1k), 44 (50k), 62 (143k). The head collapse is mostly done by step 512, before the 10^3-5x10^4 "emergence window", which is defined by the circularness rise (0.006 at 512, 0.021 at 1k, 0.070 at 50k). The paper's own label for the head change is "spurious ridges to task-tuned heads", i.e. a loss of heads, not a gain.
   - The same S7 line gives the probe-peak layer L* per checkpoint: 21, 21, 14, 12, 23, 11, 3, 4. L* is not stable across checkpoints, and the reading does not record it. This matters because L* is the layer where every Gemma/Qwen result is measured.
3. **Paper overclaim missed: DAS drop "emerges during training" has no data (moderate; stable-across-checkpoints).**
   - Paper p2 (intro, after Fig 1) and p8 (section 5 "Convergence..."): "boundary heads, DAS drops, and circularness all emerge only as training proceeds" / "and DAS ablation drop all emerge only as training proceeds (Supp. S7, Fig. S7)".
   - S7 and Fig 7 report only probe R^2, boundary-head count and FFT circularness. No ablation or DAS result at any Pythia checkpoint appears anywhere (`grep` for Pythia finds only the reference, S7, and the S37 sentence on p36 about circularness; none reports ablation). The reading's Fig 7 entry (panels A-C) is right, but it never flags that the text claims a third quantity the figure does not show. It should sit in `pitfalls` and in `architecture_location` ("no causal result across checkpoints; the checkpoint evidence is geometric only").
4. **`bears_on_group_angle` point (1) is partly wrong (low).**
   - Reading: "they mix 10-dim and 4-dim subspaces against a 4-vs-4 null".
   - Paper: true only for the mean-angle displays (Fig 4D, Fig 15; the reading's own Fig 4 and Fig 15 entries say so). Table 6 / Fig 16 (p34-35) use the matched sum cos^2 null: Haar 0.017 vs DAS (k=4 vs 10: 40/2304 = 0.0174) and 0.009 vs probe (k=2 vs 10: 20/2304 = 0.0087). The reading's own Table 6 entry lists those values. So the mismatch is limited to the angle figures, and the 7.1-7.6x enrichment is correctly normalised.
5. **Figure 6 is described only "from text" and one number differs from the figure (low).**
   - Reading (`figures_note`, key_results[20]): Fig 6 is "described from text only"; probe error "5.7-day".
   - Paper p9: the figure exists and is readable. Mechanism shift 0 / 27 / 71 days at alpha = 0 / 1 / 3; probe error labelled 0.5 / 2.6 / **5.5** days (text says 5.7 d at alpha=3, beta=2); "93% of damage is undetected". Caption says probe error "flatlines below 6 days". So the 5.7 vs 5.5 is another small text/figure mismatch. The pitfall already in the reading (the shift is measured in DAS's own coordinates, so it is partly true by construction) still stands and is right.

## 4. Omissions
- **Magnitude caveat for the L11-L12 head bottleneck.** `architecture_location` calls L11-L12 "a causal bottleneck". The paper's numbers (p6, S46) are top-10 heads 17.2 pp of the 42 pp drop and dNLL = 0.455 nats against 54-69 nats for the DAS ablation (about 0.7%). Fig 28 (reading entry has it) shows steering/freezing effects at the noise floor and a pathway decomposition the authors call uninterpretable (QK 126%, MLP 212%, other -238%). The location claim should carry this: the circuit trace is a partial account at best.
- **Erasure baselines are not in `bears_on_fates` for fate 6 / fate 2.** S11 table (p19) and S41: INLP 0.0 pp, LEACE -1.0 pp, probe 0.6 pp; the paper's sentence is "all three concept-erasure methods target the decoder subspace and have zero causal effect". Directly relevant to "erased" vs "bypass" (erasing the readable direction changes no behaviour). Currently fate 6 is only "energy above null".
- **Norm control numbers.** S26 (p29): DAS removes 19.4% of the activation norm vs probe 3.1% and random 4.1% (about 5x random), at L1 where it is mostly month-embedding directions. The paper concludes the 42 pp drop "cannot be attributed to a generic distributional shift", but the control is not norm-matched. The reading lists the norm control as worth copying but gives neither numbers nor this weakness.
- **Another internal mismatch not flagged.** S27 table (p29), layer 1: probe-DAS angle 89.01 / 88.42 / 87.59 deg at k = 2 / 4 / 6, whereas S3 and Fig 2B give 88.3 / 87.9 / 86.7 for the same layer and probe. (The reading flags the 89.6 null and Table 10's 87.6, not this one.)
- **Cross-template transfer result.** S30 (p31): the Set-F basis gives a 36.0 pp drop on Set-F (n=200, clean 0.36, not the n=332 / 42% of the main text) but only 9.0 pp on Set-G (clean 0.175, so about half the accuracy remains) and is untestable on Set-H (0% clean). The reading puts the clean accuracies in `setup` and flags 830 vs 332, but not the partial transfer.
- Minor: S3 says Qwen 2.5 1.5B has a probe-DAS angle of 86.9 deg at k=2 (below the 88.3 null by 1.4 deg); reading only says 7B/9B angles are missing, correctly.

## 5. Suggested edits
1. `key_results[5]`: replace "seed-stable" by "five seeds find different k=4 bases (pairwise 12-34 deg), all causal and CCA-correlated (>0.94 in S1/S29, >0.85 in S28); k=6 bases scatter like random subspaces".
2. `architecture_location` (checkpoint sentence): "Pythia 1.4B: boundary heads fall 133 to 50 by step 512 (spurious to task-tuned), circularness rises 0.006 to 0.070 between steps 512 and 50k; probe R^2 0.956 at init; probe-peak layer L* is 21, 21, 14, 12, 23, 11, 3, 4 across the 8 checkpoints; no ablation or DAS result exists at any checkpoint (the text claims one)."
3. `architecture_location` (heads): add "top-10 boundary heads explain 17.2 of 42 pp and 0.455 of about 69 nats; frozen-attention and steering decompositions are at the noise floor (Fig 28)".
4. `pitfalls`: add (a) the unsupported "DAS drops emerge during training" claim (p2, p8 vs S7/Fig 7), (b) the S27 table vs S3/Fig 2B angle mismatch, (c) the Fig 6 5.5 vs 5.7 day mismatch.
5. `bears_on_group_angle` (1): "Fig 4D and Fig 15 compare 10-dim TFA components with a 4-vs-4 angle null; Table 6/Fig 16 use the matched sum cos^2 null k1*k2/d (0.017 / 0.009)".
6. `bears_on_fates` fate 6 and fate 2: add the INLP/LEACE zero-effect rows (0.0 pp / -1.0 pp) as "linear erasure of the decodable direction changes behaviour by zero, in the paper's task".
7. `figures`: add a Figure 6 entry (p9): mechanism shift 0/27/71 d vs probe error 0.5/2.6/5.5 d at alpha = 0/1/3, measured in DAS coordinates.

## 6. What does not need changing
- Headline relevance judgement ("less relevant than its title suggests; methods-and-pitfalls source, not evidence about fates") is right: no pressure, no flips, no competing signal, models 1.5-9B with the circuit only at Gemma 2 2B, clean accuracy 42-51%, factual domain dropped at 0% clean.
- `actionable_fix` is faithful: reporting angle, additive drop and random envelope; probe regularisation toward the causal subspace is the authors' untested next step (p9); delta(x) has AUROC 0.63 and is "statistically indistinguishable from top-10 entropy" (p24).
- Pitfalls on the ablation-maximising "DAS", first-token integer parse failure under full ablation, the month-embedding / structural nature of the L1 basis, the inflated specificity ratios, anisotropy (14x wider null), and the self-graded Table 9 are all supported by `paper.txt`.
- `methods` and `controls_worth_copying` are accurate and usable. One mapping is loose but already caveated: ablation-maximising search cannot separate "not routed" from "override"; that needs injection or patching (the S38 test the paper proposes but did not run, as the reading says).
