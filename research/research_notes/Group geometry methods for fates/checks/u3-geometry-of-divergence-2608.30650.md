# Check: u3-geometry-of-divergence-2608.30650 (Geometry of Divergence, arXiv 2608.30650v1)

Verdict: **NEEDS FIXES** (small and additive: no wrong number, quote or page; one missed sign inconsistency, one repeated overclaim, and several missing controls/caveats)

## 1. Quote check
- `python3 quote_check.py u3-geometry-of-divergence-2608.30650`: 18 quotes, 0 failed.
- That script tests only the first 120 normalised characters and no page numbers. I also checked every full quote and its page with per-page `pdftotext` of `paper.pdf`: all 18 full quotes are present and all 18 page numbers are correct (p1, p2, p3, p4, p6, p7).

## 2. Coverage checks
- Manifest lists 8 figures/tables (Fig 1-5, Tables 1-3); the reading has all 8. No figure is missing.
- `sections_read` covers every section. The reading correctly notes the "Appendix" is referenced (p3) but absent from the 8-page PDF.
- I opened p01, p04, p05, p06, p07 plus zoomed crops of Fig 3 and Fig 4a/4b and compared them with the reading's figure entries.
- Recomputed from Table 3: Never-think mean reward 0.241; best-of-two geometry per row (0.397, 0.392, 0.442, 0.352) = 0.396 (so the headline is the per-row better of the two triggers; the reading's inference is right); kappa-only mean 0.394, beta-only 0.381, Learned 0.397, Warm-up 0.3485, Always 0.353; gain vs Warm-up about +0.05. Token means 104.75k to 92.98k (11.2%) also reproduce with the same per-row choice. "Learned-HST best or tied in 3 of 4 rows" matches the bold marks (Airline 32B is a tie with Always-think at 0.364).
- Table 1, Table 2, Table 3 and Fig 5 numbers in the reading all match the paper/figures. The approximate Fig 4a/4b coordinates (ID 2, 6, 10, 1, 4 Retail; ID 2, 10, 5, 18, 1, 4 Airline) match my zoomed crop. Fig 3 numbers (turn 2 about -0.15/-0.12, L20 beta 22 to 49 vs 45, L24 40 to 88 vs 78, correct curves stop at turn 6) match.
- Fate labels: all "weak" except direction 3 "moderate"; this matches a paper that has no probes, patching or ablation. The layer fractions (22/40, 42/64, 16/32) are right for those models but the totals are not stated in the paper (see error 5).

## 3. Errors (what the reading says / what the paper says)
1. **"Acquisition chains carry little" repeated as a result, but Fig 4a contradicts it in Retail (moderate).**
   - Reading, key_results[4]: "Separability depends on action type: output chains separate, acquisition chains do not."
   - Paper p2/p5: "information-acquisition chains carry little" / "limited discriminative power (e.g., IDs 1 and 4)". Fig 4a (p5, zoomed): Retail ID 1 (Read-Read-Read) and ID 4 (Resp-Read-Read) are drawn as significant (outlined = "at least one metric p<.05") with variance-slope effects of about 0.09 and 0.08 (AUC about 0.59), as large as ID 8 (0.09) and larger than ID 10 (0.045), which the paper counts as a discriminative output chain. Only in Airline are ID 1 and 4 near zero (0.012, 0.048).
   - Suggested wording: "Output chains show larger effects; acquisition chains are weak in Airline but still p<.05 in Retail (effect about 0.08-0.09)."
2. **Fig 3 beta panel described as supporting the claim without the interval overlap (minor).**
   - Reading (Fig 3): takeaway "gap opens gradually from turn 3-4 and widens", `text_matches_figure: partly - supports the claims`.
   - Paper p4 (zoomed crop): beta curves for correct and incorrect lie on each other through turn 4 and their bands overlap through turn 5; at turn 6 the gap is about 4 on a base of 49 (L20) and 10 on 88 (L24), with the incorrect band touching the correct line. Kappa separates a little earlier, but the gap there is about 0.07-0.12 with bands that also overlap at turns 6-7. Also, beta is a cumulative prefix statistic, so it rises with turn index for both groups by construction.
3. **Fig 5b read-accuracy wording (nit).**
   - Reading: "read accuracy ~0.54-0.65, lowest (~0.565) under geometry/learned triggers."
   - Figure p6: the lowest point is Never-think (~0.54); geometry/Learned (~0.565-0.57) are the lowest among thinking policies.
4. **Fig 3 layer inconsistency: add the likely explanation (nit).**
   - Reading flags that Fig 3 (Math) shows layers 20 and 24 while Llama-3.1-8B (the Math model in Fig 2) uses layer 16.
   - Paper p3/p4: model for Fig 3 is not named. Layers 20 and 24 straddle the stated Qwen3-14B probe layer 22, so Fig 3 may be Qwen3-14B Math (which is not in Fig 2) or a layer-robustness check. Either way unresolved from the text; the reading should say both options.
5. **Layer totals are not from the paper (nit).**
   - Reading, architecture_location: "layer 22 of 40 ... 42 of 64 ... 16 of 32 ... (~55%, ~66%, 50%)".
   - Paper p3 gives only the layer indices. The totals (40, 64, 32) are correct for those model families but come from model cards; also indexing (0- vs 1-based) is unstated. Label them "from model cards".

## 4. Important omissions
1. **The variance-slope trigger fires in the opposite direction from the paper's own diagnostic (most important).**
   - Paper p4 (4.3): correct episodes have HIGHER beta; failed ones "contract". Paper p6 (Sec 6) and p7 (Table 3 header): the beta trigger fires "when beta_t > tau_beta". Thinking is therefore switched on when the state looks like a correct episode. No tau_beta value is reported anywhere (Table 3 gives tau_kappa only), and the paper never explains the sign.
   - Fig 3 shows beta rises with turn index in both groups (22 to about 45-49), so "beta > tau" probably acts as a "late in the episode" trigger, and kappa < tau (kappa falls with turn index in both groups) likely does too. The reading's methods entry lists only the kappa trigger and never mentions this. It matters for our Stage B "monitor then re-check" arm: before borrowing the trigger we need a turn-index trigger and a firing-rate-matched random trigger as controls (neither is in the paper), and firing counts per policy (not reported).
2. **No variance across the 5 trials, no tests for the policy comparisons (important).**
   - Paper p6/p7: "5 trials per task"; Table 3 and Fig 5 give single rewards. Differences such as 0.397 vs 0.398, or beta 0.412 vs Warm-up 0.405 (Retail 32B), have no error bars. From memory of tau-bench (not in the paper): Retail has about 115 tasks and Airline about 50, so a reward of 0.4 over 5 trials has a standard error of roughly 0.02 (Retail) to 0.03 (Airline); the +0.05 gain over Warm-up is within about two standard errors. The reading lists the other statistical weaknesses but not this one.
3. **Overclaims in Sec 6 not flagged (moderate).**
   - Paper p7: "Geometric thresholds are unimodal, transferable ... That optimum shifts predictably with the correct-minus-incorrect kappa gaps". Evidence: the unimodal claim rests on one 3-point sweep (-0.15/-0.20/-0.25 giving 0.388/0.397/0.363) in one setting; the "predictable shift" rests on 4 settings on a 0.05 grid (tau -0.20, -0.15, -0.15, -0.10 against gaps 0.030, 0.035, 0.049, 0.088); thresholds differ per setting, so "transferable" is not shown; "matches or beats the trained classifier" holds in 2 of 4 rows (Learned beats kappa in Airline 14B 0.420 vs 0.384, and Airline 32B 0.364 vs 0.352), and in Airline 32B Always-think (0.364) beats both geometry triggers.
   - The reading has "thresholds chosen by sweeping on the same evaluation" but none of these.
4. **Token saving is not attributed (minor).** Paper p7 itself says token cost depends on "both deliberation and failure length". Max-turn terminations fall from 75 to 34 (Retail 14B), so much of the 11.2% saving can come from fewer runaway episodes, not cheaper thinking. Firing frequency is not reported. The reading mentions token cost only as a control "worth copying".
5. **Fig 4 effect-size method is undefined (minor).** The paper never says what the AUC separates (windows? episodes?), how non-independent overlapping sliding windows from the same episode are handled, or whether the p<.05 flags are corrected across 20 chains x 2 metrics x 2 domains. The reading calls it "separability of windows" without saying this is inferred. Also, Respond-Respond-Respond (ID 2) is a variance-slope effect in Retail (about 0.19) but a curvature-only effect in Airline (about 0.16, beta near 0): the "stands out" chain flips its metric between domains. This fits the paper's own "signals are task-dependent" admission.
6. **Possible message-type confound for kappa (checker's inference, not in the paper).** h_t is taken at the last token of the user input q_t, before the model acts; in tau-bench the input alternates between simulated-user text and tool outputs. A state that alternates between two message types gives consecutive displacement cosines near -1 by construction, so negative kappa and its dependence on action chain may partly reflect what kind of input arrived rather than anything about the model's reasoning. The reading's null-model pitfall (about -0.5 for jitter) is right and related but does not cover this; the suggested check is kappa on user-only turns, or on matched-type turns.
7. **Leads cited by the paper that the reading does not record.** Dongre et al. 2026 "When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction" (arXiv 2605.12922), which the paper cites as using hidden states to detect and correct drift and is the nearest to our "attention heads reading the user's claim" idea; ICR Probe (Zhang et al. 2025b; cross-layer dynamics of hidden-state updates, closest to our path-across-layers idea); Orgad et al. 2025 (exact-answer-token localisation, already in the project). Worth adding to the leads list.

## 5. Ideas worth harvesting (for the "best ideas" goal; all already compatible with the reading's reuse verdict)
- Turn-aligned path statistics as a cheap baseline for the fork-point analysis, with two added controls the paper lacks: a null for kappa (stationary-jitter value near -0.5, shuffled turn order) and matching on turn index AND episode length, not only turn index.
- Stratify flipped vs resisted items by what the model just did before comparing geometry (the paper's own Retail/Airline contrast, Table 1, shows behaviour differs between groups).
- Stage B arm: "re-check own answer when a trigger fires", compared against a firing-rate-matched random trigger and a turn-index trigger, with firing counts reported.
- Do not copy the PCA-means reading in Fig 4c: the paper's text (p6) says correct episodes "expand outwards" but the plotted means converge by t+2.

## 6. Overall
The reading is accurate on all numbers, quotes, pages and figure content, and it already catches the main weaknesses (length confound, Never-think headline, best-of-two averaging, missing kappa null, PCA mismatch, no causal test). Fix list: add omission 4.1 (beta trigger direction), correct error 3.1 (acquisition chains), add 4.2/4.3 to pitfalls, tidy errors 3.2-3.5. No redo needed.
