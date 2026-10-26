# Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior

**arXiv 2605.05115v1, 6 May 2026.** Wurgaft, Rager, Kowal, ..., Geiger, Fel, Goodman, Lubana (Stanford, Goodfire and others). Code: github.com/goodfire-ai/causalab, branch `manifold_steering`.

## In one line
On clean tasks where the answers have a known shape (days in a loop, ages in a line), the model's activations and its output probabilities trace the same curve, distances along the two curves match, and steering along the activation curve gives natural outputs. Steering along a straight line does not.

## Relevance to our project (honest)
Moderately relevant. This is the anchor paper for direction 2 ("where shared geometry breaks"), and its future-work section names sycophancy. But it only studies cases where the model behaves normally. It never looks at items where representation and behavior disagree, so it gives us a measuring tool, not evidence about our fates. With only two answers (Canberra vs Sydney), the "manifold" collapses to a line between two centroids, so the method needs extra structure before it can tell us much.

## Sections read (in order)
Abstract; 1 Introduction; 2 The Geometry of Representation and Behavior; 2.1 Setup; 2.2 Fitting the Manifolds; 2.3 Conceptual Structure Appears in Behavior and Activation Space; 3 Connecting Representation and Behavior via Intervention; 3.1 Steering Intervention Notation; 3.2 Steering Along the Activation Manifold Follows the Behavior Manifold; 3.3 Behavior Space Geometry Recovers the Activation Manifold; 3.4 Unifying Steering Strategies Through Geometry; 4 Factored Control in Multi-Dimensional Spaces; 5 Visual World Model: Mountain Car; 6 Related Work; 7 Discussion; 8 Future Work and Limitations; Acknowledgments; References; A Experimental Details for Language Tasks (A.1-A.9); B Vision Task (B.1 Mountain Car); C Additional Results (C.1 ICLR cylinder, C.2 uncertainty, C.3 Mountain Car). 34 headings in all.

## Setup
- **Data:** templated arithmetic with known geometry. Weekdays (49 prompts), months (84), letters C-Z (48), ages 10-100 (909). Plus in-context random walks on a 5x5 grid and a 9x9 cylinder (2048 tokens), and 100 Mountain Car rollouts.
- **Models:** Llama 3.1 8B for all main language results. Llama 3.1 70B in one appendix figure. A small GRU world model (64-d latent) trained by the authors.
- **Where they read:** residual stream, last token, layer 28 of 32 (8B), chosen in advance as "late". Layer 70 of 80 for the 70B figure. No layer sweep, no heads or MLPs.
- **Activation manifold:** 64-d PCA, one centroid per answer, splines through the centroids using known coordinates.
- **Behavior manifold:** the next-token distribution over the answer tokens plus an "other" bin, mapped to sqrt(p) (Hellinger) coordinates, then splined.
- **Baseline:** linear (diff-in-means) steering between centroids.
- **Metrics:** Pearson r between distance matrices, MDS plots, an output "energy" E_BC (how far the steered outputs are from natural outputs), and pullback R2.

## Figures and tables (13 of 13 covered, each page image viewed)
| id | page | what it shows | takeaway | numbers | text matches? |
|---|---|---|---|---|---|
| Fig 1 | 2 | Schematic of linear, manifold and pullback paths and the outputs they induce | The hypothesis as a picture | none | yes (illustration only) |
| Fig 2 | 4 | Weekday and month manifolds (PCA, MDS), path-length scatters | Geodesic distances match; straight-line distances do not | weekdays r=0.99 vs linear 0.89; months 0.89 vs 0.53 (figure labels) | yes; the months scatter has a second off-diagonal band that the text doesn't discuss (approx, read from figure) |
| Fig 3 | 5 | Letters and ages, same layout | Open curves; near-perfect isometry | letters 0.999 vs 0.71; ages 0.999 vs 0.36 | yes |
| Fig 4 | 7 | P(token) along steering paths, manifold vs linear | Manifold steering passes mass through the neighbours; linear skips them and inflates "other" | approx, read from figure: weekday manifold intermediate peaks ~0.70; linear "other" peaks ~0.33 at the midpoint; months linear "other" ~0.45; ages manifold traces very noisy | partly (ages is not smooth) |
| Fig 5 | 8 | Pullback paths (teal) vs manifold paths (black) | Paths optimized only for natural outputs bend like the manifold | R2 in text: 0.77/0.75/0.47/0.78 vs chord 0.42/0.32/0.24/0.23 | partly (letters and ages pullbacks are jagged) |
| Fig 6 | 11 | 5x5 grid task: MDS, surfaces, grid heatmaps along paths | Two intrinsic axes, "factored control" | r=.99 vs linear 0.90 | partly; the linear vs manifold contrast is unclear on the (2,0)->(2,4) path (approx, read from figure) |
| Fig 7 | 13 | Mountain Car: decoded frames and p(position) along paths | Linear midpoints give a blurred car; manifold gives a moving sharp peak | r=0.99 vs 0.06 | yes, but "behavior" here is a constructed readout (see pitfalls) |
| Table 1 | 26 | Task templates and sizes | Tiny templated datasets | 49 / 84 / 48 / 909 | yes |
| Fig 8 | 30 | World-model architecture | Reference only | 64-d latent, 16-d action | yes |
| Fig 9 | 32 | 9x9 cylinder heatmaps | Factored control on a periodic axis | >80% neighbour accuracy; r=.99 vs 0.81 | partly (clear on one path, weak on the other) |
| Fig 10 | 33 | Grid and cylinder steering paths, MDS | Manifold distances recover the lattice | none new | yes |
| Fig 11 | 34 | 70B layer 70: weekday circles stacked by difficulty | Uncertainty is a second manifold axis | approx, read from figure: group 0 peaks ~0.5-0.55; group 2 peaks ~0.25 under "other" ~0.4; group 3 order mostly lost | partly (the caption overstates group 2) |
| Fig 12 | 35 | World-model pullback in PCA | Pullback lands on the spiral manifold | distance to Mh: chord 2.22, geometric 0.20, pullback 0.29; 30/30 pairs | yes |

## Key results (with quotes)
- Isometry: "The two distances are highly correlated (r = 0.99 weekdays, r = 0.89 months, r = .999 letters, r = .999 ages)..." (p6). Straight lines do worse: r = 0.89 / 0.53 / 0.71 / 0.36 (p6).
- "linear steering instead exhibits ‘teleportation’: mass jumps between non-adjacent concepts..." (p6).
- Naturalness: E_BC manifold 0.34 / 0.36 / 2.42 / 5.21 vs linear 0.93 / 1.09 / 6.95 / 13.49, "an improvement of a factor of 2.8×, with all statistical comparisons yielding p < 0.001" (p8).
- Pullback R2 (pullback vs chord): weekdays 0.77 vs 0.42, months 0.75 vs 0.32, ages 0.47 vs 0.24, letters 0.78 vs 0.23 (p9).
- 2-D in-context graphs: r = .99 (grid and cylinder) vs linear 0.90 / 0.81 (p11).
- World model: chord distances correlate only r = 0.06 with the conceptual ordering (p14). Pullback "dominates the chord baseline on 30/30 pairs" (p36).
- 70B: confidence sits on its own manifold axis, "a handle for calibrating model confidence" (p34). This is one model and one layer.
- Theory: the output-pullback metric has rank at most |Z|-1 (p10), so with few answer options, almost all residual directions do nothing to the answer.

## Methods we could use
1. **Isometry test** (centroids + splines + Hellinger outputs + distance correlation). It could diagnose where shared geometry breaks. Fit it on resisted items, then place each flipped item: (a) on the manifold at the correct coordinate but answering wrong points to readout failure; (b) on the manifold at the claimed-answer coordinate points to override or transformed; (c) off the manifold means an unnatural state. Cheap: forward passes only. Code is available.
2. **Manifold steering** (edit only the top-64 PCA components, keep the rest of the residual). This is a cleaner "inject the correct answer" test for *not routed*.
3. **E_BC naturalness score.** Use it as a side-effect check for Stage B fixes, and as a monitor of whether sycophantic outputs look like natural confident answers.
4. **Pullback optimization.** It finds which states actually produce the correct output. That gives a weak handle on *bypass* when the natural correct centroid fails to work.
5. **Jacobian rank view.** If a decodable correctness direction barely projects onto the row space of the answer-logit Jacobian, it is a bypass candidate.
6. **Confidence axis.** It argues for matching flipped and resisted items on prior confidence.

## Controls worth copying
- Leave out same-geodesic sub-arc pairs from the isometry correlation, because they inflate r "by construction" (p28).
- Compute the straight-line (Euclidean) version of every statistic as the null.
- Sum the answer's spelling variants, keep an "other" bin, and work in sqrt(p) coordinates.
- Average over 16 base prompts with varied ground truth, and report mean ± SE across pairs with paired t-tests.
- Regularize the pullback norm (to avoid a "high-norm shortcut basin", p29).
- Run a probe sanity check (R2 0.95 / 0.90) before fitting a manifold.

## Pitfalls
- **The linear baseline isn't a fair comparison.** Manifold steering keeps the prompt's own off-subspace residual. Linear steering overwrites the whole residual ("the entire residual-stream activation is replaced by π(t) at each step", p28). Part of linear steering's "unnaturalness" may come from wiping prompt information. Give our linear baseline the same subspace.
- The pullback R2 baseline is a chord drawn in PCA space, "not the linear-steering trajectory after intervention" (p29).
- The pullback search is restricted to the manifold's own PCA subspace and starts from the chord (p28). That partly builds in the result.
- In the world model, "behavior" is a softmax over distances to the activation centroids ("F is a smooth, deterministic map from activations to position distributions", p30). The r = 0.996 isometry there is close to circular.
- The method needs known coordinates for each concept (supervised, p16).
- They use one layer chosen in advance and one model family. There are no heads or MLPs, no seeds and no checkpoints.
- Small inconsistency: App C.3 describes an alpha-weighted pullback target "via the procedure of §3.3" (p35), but §3.3 does not use one.
- Ages, the task with the most classes, has the noisiest steering and the weakest pullback (R2 0.47).

## Limitations and future work (authors)
- Limitations: "we focused on simple settings for which the concept of interest had a well-defined domain (e.g., weekdays)..." (p16)
- Future work: "future work is needed to explore more abstract concepts, e.g., refusals ..., sycophancy ..., and persuasion" (p16). They hedge on sycophancy: "(were it to exist)" (p16).

## Bears on the fates
| fate | how | strength |
|---|---|---|
| 5 transformed | Is the flipped item on the manifold at the claimed answer, or off it? | moderate |
| 3 not routed | On-manifold injection of the correct point | moderate |
| 1 readout | Per-item residuals of the activation-to-output map | weak |
| 2 bypass | Jacobian rank bound; pullback finds the states that actually work | weak |
| 4 override | Not studied (only the confidence axis) | weak |
| 6 erased | Not addressed | weak |

## Group / subspace angle
- **Helps:** it gives a concrete recipe for treating class point clouds as a manifold, comparing geometries through distance-matrix correlation (an RSA-like test), and checking causally with steering and pullback. It also warns that straight-line distances can be badly wrong on curved or folded geometry.
- **Warns:** two-answer flips give a degenerate 1-D manifold. To make the group angle work we need added structure: confidence bins, pushback strength, several candidate answers, or token position.

## Where in the architecture
Only the late-layer residual stream at the last token (8B layer 28; 70B layer 70). Nothing is tested about component type or depth, or about consistency across models.

## Actionable fix
For Stage B, steer flipped items back along the fitted manifold inside a small subspace while keeping the rest of the residual, and score the result with both accuracy and E_BC. As a monitor, flag a decision-position activation that is far from the natural activation manifold, or an output that is far from the natural output manifold. Neither has been tested on sycophancy.

## Verdict
**Adapt.** Reuse the code, the isometry test, E_BC and the pullback for direction 2. Give any linear baseline the same treatment. Don't expect it to work on binary fact flips without first adding structure to the output space.
