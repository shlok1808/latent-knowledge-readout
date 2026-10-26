# Fixes: w2-readout-subspace-grassmann-2605.09011

All checker points verified against paper.txt and page images (p06, p07, p17). quote_check.py: 18 quotes, 0 failed.

## Changes (old -> new, page)
- key_results[4] claim: "Phases 1 and 3 have roughly fixed size" -> Phase 2 grows fastest (slope 0.89 Qwen, 0.62 OLMo); "roughly constant" holds for Qwen2.5 only; OLMo-2 b1 = 4, 9, 9, 17 layers (slope 0.26, R2 0.97) and P3 = 7, 6, 8, 12 layers (p.6 Fig 6, p.7 Fig 8, p.8 text).
- Figure 6 takeaway: "b1 and Phase 3 width barely grow" -> flat only in Qwen2.5; OLMo-2 b1 4 to 17, P3 6-7 to 12 layers (p.6).
- Figure 6 numbers: added OLMo-2 b1/P3 points, Qwen P3 R2 = "nan" (flat by eye), and the indexing note (Fig 6 counts layers L - b2, so Qwen P3 = 8; Fig 8 counts L-1 transitions, so 7) (p.6).
- Figure 6 text_matches_figure: "yes" -> "partly" (text's "remain roughly constant" fails for OLMo-2) (p.6, p.8).
- Figure 8 numbers: added "counts are transitions (sum to L-1); OLMo-2 P1 stays at 23-29% of depth, so it grows in layers" (p.7).
- key_results[5] claim: "Phases are told apart by effective rank, not by alignment" -> labelled "authors' claim (abstract), partly contradicted by Figs 12, 20, 21": |cos| also splits by phase (lowest P2 7/8, highest P3 claimed 8/8); hidden-state rank does not concentrate in P3 (rises in all four Qwen2.5); only update rank/norm concentrate (p.1, p.8, p.9, p.17, p.18).
- Figure 9 takeaway: "update rank climbs in Phase 1 and plateaus in Phase 2" -> "climbs in P1, peaks in P2, declines in P3 (7/8)" (p.7, p.8, p.18).
- Figure 9 numbers: OLMo-2-32B final norm "about 30" -> "about 35"; Qwen2.5-7B rank "plateaus about 2850" -> "peak about 2880 in P2, about 2700 in P3" (p.7 image).
- Figure 1 numbers: LogitLens last-layer Hit@1 "about 0.45-0.5" -> "about 0.43 (Qwen2.5-32B), about 0.47 (OLMo-2-32B); Hit@10 about 0.73 / 0.83" (p.2; values from the checker's zoomed read, consistent with the page).
- setup.models: "Base models" -> "base checkpoints inferred from the names; the paper never says base vs instruct"; added OLMo-2 checkpoint names from Fig 14 titles (three release dates) (p.12).
- setup.compute: added "500 x 2^18 = about 131M tokens, less than the 226.49M-token set; subset not specified" (p.12).
- New pitfall: Hit@k target "final token" is undefined; last-layer tuned Hit@1 ~0.8 vs logit ~0.43 favours "agreement with the model's own output" (inferred). Quote "how often the final token appears in the top-k at each layer" (p.2).
- Figure 20 numbers: added the unnamed 7/8 exception, likely OLMo-2-7B (|cos| lowest at layers 0-2, P1) (p.17 image).
- bears_on_group_angle: "agrees with 2605.12714" -> "directionally consistent ... qualitative only" (different objects and statistics); added cited leads [10] Gurnee et al. 2026, [18] Shai et al., [5] Bruno et al., [20] LoRA-Lens (p.2-3, refs).
- architecture_location: added the convention that b1, b2 and widths count layer-to-layer transitions (sum to L-1); Qwen2.5-14B P3 = transition 40 to 41 onward, 7 transitions, 8 layers (p.4, p.6, p.7).
- .md: same changes mirrored (one-paragraph summary, Setup models/compute, Fig 1/6/8/9 rows, caveat after the p.8 quote, new key-result caveat on the abstract claim, Hit@k pitfall, group angle wording and cited leads, Where convention).

## Rejected checker points
- None rejected. Not done: no lens-training-subset quote added (the paper gives no sentence on it; recorded as a computed note in setup.compute only).
