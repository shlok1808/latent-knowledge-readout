# Fixes: w2-shape-of-beliefs-2602.02315

Files changed: reading/w2-shape-of-beliefs-2602.02315.json and .md. quote_check.py: 30 quotes, 0 failed (also checked at full length, 30/30).

## Changes (old -> new, page)
1. key_results[9].claim: "decodable does not mean linearly controllable" -> "linear steering moves the target variable but distorts coupled covariates (std inflates while the mean moves from 300 toward 700); readable does not mean geometrically faithful control" (p7 Fig 5B, p8 Fig 6, p9).
2. methods[2].which_fate: "a messy or null result from straight-line steering is not proof that the information is causally inert" -> labelled as our design advice, not a paper finding; the paper has no null or inert case (p7-8).
3. bears_on_fates 2 bypass: strength moderate -> weak; "how" reworded as analogy only plus our design advice (p7-9). MD fates table and "Most useful" bullet updated to match.
4. architecture_location: "Manifolds are visible at most layers" -> PCA shown at layer 14 (Fig 1, 11) and at idx 1, L-2, L-1 in App. B; structured late, mostly mixed at idx 1 (6 of 9 models); separability at all layers is not a visible manifold (p2, p18, p20, p6). MD Fig 9 row updated.
5. Fig 3 takeaway, key_results[2].claim, methods[1], bears_on_fates 5: added "one binary probe, mu = {300, 350}, layer 0 only" and "shift along the task variable, not a context change; design idea by analogy" (p6 caption). Fate 5 strength moderate -> weak-moderate. Added key result quoting the Fig 3D caption (p6). MD updated.
6. setup.models_and_sizes: "All interventions are on Llama-3.2-1B only" -> steering model not named in Sec 4 or Fig 5/6 captions; inferred from "We rely primarily on the Llama-3.2-1B" (p3). Added that quote as a key result. MD updated.
7. pitfalls[6]: "outputs are not a clean posterior readout" -> "output can match the target before the data alone supports it, so a strong (normality) prior is in play" (p14). Added key result quoting "the model believes the input data is Gaussian, even though it doesn't yet look like it" (p14).
8. Fig 2 numbers: "std ~240 near mean ~600" -> "peaks at std ~240 near mean ~510-520, second cluster ~225 near mean ~580-620" (p4, checked on the PNG). MD row updated.
9. Fig 2 takeaway: "Activations and outputs both show ... a high-variance region" -> high-variance phase is measured on the output only; activations show the two-attractor path (p4).
10. bears_on_group_angle: spline/field-aware std "~100-115" -> "~100-122" (p7 Fig 5B, spline peaks ~120-122 near mean 500, checked on the PNG). MD Fig 5 row: "spline ~100-120" -> "~100-122", and added "mean moves 300 to ~700".
11. Omission, base models: added pitfall quoting "Our analysis relies on true activations of base models without finetuning" as a transfer risk for aligned-model sycophancy fates; reuse_verdict and MD Setup/Pitfalls/Verdict updated (p9).
12. Omission, last-layer steering: layers_and_positions and MD Setup note that last-layer edits are close to logit editing and say little about mid-network routing (p7); reuse_verdict notes "last-layer steering only".
13. Omission, Fig 3A controls: setup.controls and MD note no shuffled-label or recent-numbers baseline and no count of series per class; layer-0 accuracy already ~0.875 vs chance ~0.11 (p6).
14. Omission, Fig 4 "low-dimensional": Fig 4 takeaway now marks it as the paper's wording; by our reading L13/L14 need about 5 components for 0.8 of the variance (p6, checked on the PNG). MD row updated.
15. Omission, actionable_fix: the "field of local probes for monitoring" idea labelled as our inference; the paper proposes only steering schemes. MD verdict updated.

## Checker points rejected
- None in full. Point 6b was only partly accepted: the paper does tie the switch trajectory to both probability and representation space (p4, "reflected both in probability space (Fig. 2D) and in representation space (Fig. 2C). They show that, after the switch, the model jumps to a phase of high variance"). So the reading still says both spaces show the two-attractor path, and only the high-variance claim is limited to the output.
