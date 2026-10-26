# Check: w2-readout-subspace-grassmann-2605.09011

Paper: Lombardo, Trimigno, Cagnoni, "A Geometric Perspective on Next-Token Prediction in LLMs: Three Emerging Phases" (arXiv 2605.09011). Reading checked: `reading/w2-readout-subspace-grassmann-2605.09011.json` (and the .md, which agrees with it).

## Verdict: NEEDS FIXES (minor)
Numbers, quotes, page refs, fate labels, architecture_location and actionable_fix are reliable. Three wording edits are needed because two key_results/figure takeaways copy the authors' overclaims without a caveat, and one indexing convention is not stated. No REDO.

## Quote result
- `quote_check.py`: 17 quotes, 0 failed.
- That script only matches the first 120 characters and ignores page numbers, so I re-ran a stricter check (full-string match, then page number from `pdftotext` per page): 17/17 match in full, 17/17 on the claimed page (pitfalls[0] also appears on p.16 in App. D.1; fine).

## Figures checked
Manifest lists 23 figures/tables; the reading has all 23 with matching page numbers (none missing, none extra). I opened all 14 page images (p02, p04-p09, p12-p18) plus 200-300 dpi crops of Fig 6, Fig 8 and Fig 14. Verified against the images: Fig 1, 2, 3, 4 (b1/b2 and R2 legends), 5 (7 models only; Qwen2.5-7B missing), 6 (slopes), 7 (the reading's approximate bar values are consistent with the image), 8 (every consensus split and layer count), 9, 10, 11, 12/20, 13/21, 14 (all 8 gaps), 15/16 (Visibility at k=1% and 50%), 17, 18 (all best-two R2 values; 3 of 8 models where Kumaraswamy wins), 19, 22. The reading's "partly" flags (Fig 5, 7, 10, 13, 15, 16, 18, 20, 21) are all real and correct.

## Errors (reading says / paper says)
1. **Phases 1 and 3 "roughly fixed" is copied from the paper and is wrong for OLMo-2.** (medium)
   - Reading says: key_results[4] repeats "Phases 1 and 3 have roughly fixed size"; Fig 6 takeaway says "b1 and Phase 3 width barely grow".
   - Paper says (p.8): "Phase 2 expands while Phases 1 and 3 remain roughly constant."
   - Figures say: Fig 6 (p.6) OLMo-2 b1 = 4, 9, 9, 17 layers for 1B, 7B, 13B, 32B (slope 0.26, R2 0.97); Fig 8 (p.7) shows OLMo P1 at 27/29/23/27% of depth, i.e. it scales with depth; OLMo P3 = 7, 6, 8, 12 layers (Fig 6). Only Qwen is flat (P1 4 to 8 layers, P3 constant; Qwen P3 R2 is printed as "nan").
   - Fix: say Phase 2 grows fastest (slope 0.89 Qwen, 0.62 OLMo); Phase 1 is flat only in Qwen; Phase 3 flat in Qwen only.
2. **Abstract claim repeated without caveat.** (medium-minor)
   - Reading says: key_results[5] "Phases are told apart by what updates do to effective rank, not by their alignment with the residual stream."
   - Paper says (p.1 abstract, quote in the reading) the same, but its own Fig 12/20 and p.8 text report |cos(update, residual)| as the metric with a clean phase split (lowest in P2 7/8, highest in P3 claimed 8/8, anti-correlated with RSS; range about 0.02-0.4). Hidden-state rank does not "concentrate": it is flat or rises in P3 for all four Qwen models (Fig 13/21). Only update rank/norm concentrate.
   - The caveats exist elsewhere in the reading (Fig 12, 20, 21 entries), but key_results[5] should be labelled "authors' claim; partly contradicted by Figs 12, 20, 21".
3. **Off-by-one between Fig 6 and Fig 8 is quoted both ways.** (minor)
   - Reading says: Fig 6 "about 8 layers flat" for Qwen P3; Fig 8 "7 L"; architecture_location "P1/P2/P3 = 5/35/7" for Qwen2.5-14B.
   - Paper: Fig 6 plots L - b2 with L = 28/48/64 (Qwen P3 = 8); Fig 8 counts profile points (L-1), so 7. RSS_l compares layers l and l+1, so b1/b2 index transitions. For a WHERE axis, state the convention: "Phase 3 starts at transition 40 to 41 and runs to the output (7 transitions, 8 layers)".
4. **Fig 9 takeaway contradicts the reading's own Fig 22 entry.** (minor)
   - Reading says Fig 9: update rank "climbs in Phase 1 and plateaus in Phase 2". Paper (p.8, p.18) and Fig 9/22 images: update rank peaks in P2 and declines in P3 (7/8); Qwen2.5-7B about 2880 down to about 2700. Also OLMo-2-32B final update norm is "about 30" in the Fig 9 entry and "35" in the Fig 22 entry; the Fig 9 curve ends above the 30 tick, so about 35.
5. **Fig 1 numbers slightly high.** (minor) Reading: LogitLens Hit@1 "about 0.45-0.5 at the last layer". Image: Qwen2.5-32B about 0.43, OLMo-2-32B about 0.47 (logit-lens Hit@10 about 0.73 / 0.83).
6. **"Base models" is an inference.** (minor) The paper never says base vs instruct. The checkpoint names appear only in Fig 14 titles (OLMo-2-0425-1B, -1124-7B, -1124-13B, -0325-32B; the four OLMo-2 sizes are different release dates). Mark as "inferred from names".

## Omissions
- **Hit@k target is undefined in the paper.** p.2 says "how often the final token appears in the top-k" and never says whether "final token" is the data label or the model's own final-layer top-1. The last plotted layer shows tuned lens Hit@1 about 0.82 vs logit lens about 0.43 for the same layer, and 0.8 is far above plausible web-text next-token accuracy for a 32B model, which favours "agreement with the model's own output". The reading's key pitfall (the lens reads what the model would say) and its per-item proposal depend on this; flag it as undefined/inferred.
- **Lens training budget is inconsistent.** App. A (p.12): 500 steps x 2^18 tokens is about 131M tokens, less than the stated 226.49M-token corpus. Setup should say "500 steps on a 226M-token subset".
- Qwen P3 R2 = "nan" in Fig 6 (the "constant P3" claim is by eye).
- Cited leads that touch our manifold-vs-linear angle are not listed: Gurnee et al. 2026 "When models manipulate manifolds" [10], Shai et al. belief-state geometry [18], Bruno et al. mean-field three-phase analogy [5], LoRA-Lens [20] (same authors).
- The "agrees with 2605.12714" cross-paper line (in bears_on_group_angle and the .md) is only qualitative: lens-weight subspace on web text vs example-cloud PCA subspace on MTEB inputs, and RSS decline vs curvature peak are different statistics. Say "directionally consistent".
- Minor: the paper's "lowest in Phase 2 (7/8)" has one unnamed exception; in Fig 20 it looks like OLMo-2-7B (lowest at layers 0-2, in P1).

## High-stakes items, status
- Key results: all 13 quotes exact; claims fine except items 1 and 2 above.
- Controls worth copying: accurate (held-out lens split with per-layer gap, k sweep with consensus, tuned-vs-logit lens side by side, shape test, two families).
- Fate labels: hedged correctly ("moderate (method), no direct evidence", "weak", "none"). Correct that RSS as defined is input-independent.
- architecture_location: depth fractions and Qwen2.5-14B / 32B layer numbers match Fig 4 and Fig 8 (14B: b1 5, b2 40 of 47 transitions = 11% / 85%; 32B: 8 and 56 of 63 = 13% / 89%); apply the convention note in item 3.
- actionable_fix: correctly "none in the paper"; the two proposals are labelled as ours and untested.
- Skipped sections: none; sections_read covers everything including all appendices.
