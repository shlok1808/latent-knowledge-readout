# The Shape of Beliefs (arXiv 2602.02315)

Sarfati, Bigelow, Wurgaft, Boppana, Merullo, Geiger, Lewis, McGrath, Lubana (Goodfire). v2, 5 May 2026.
Code (stated, not checked): github.com/raphael-goodfire/shape-of-beliefs

## In one line
In a toy numeric task, the model's "belief" about a distribution lies on a **curved manifold**. Probes decode it everywhere but **transfer only locally**. Straight-line steering **causes side effects**, while steering along the manifold mostly avoids them.

## Most useful for us
- **H3 / fate 5 (transformed):** a probe only works near the region it was trained on. It falls from ~0.89 to chance within a small parameter shift (approx; one binary probe at layer 0 only, and a shift along the task variable, not a context change like pushback). Design idea, by analogy. Cheap test for us: train the correctness probe on no-pressure items, then plot its accuracy as pressure grows. Compare with probes retrained at each pressure level.
- **Fate 2 (bypass) warning (analogy):** straight-line steering does move the target, but it distorts other outputs (std). The paper has no null or inert case. Our design advice: before calling an item "bypass", also steer along the data manifold, meaning the centroid path from flipped items to resisted items. Always read side effects too: entropy, hedging, and accuracy on control items.

## Setup
- **Data:** integer series 0-999 drawn from N(mu, sigma). mu runs 300-700 with sigma fixed at 100, and sigma runs 20-200 with mu fixed at 500. Also switching series: N(300,100) changes to N(700,100) at index 1000.
- **Models:** Llama-3.2-1B base (main). Geometry only (App. B): 8 more base models up to 8B (LFM2, Llama-3.1-8B, Llama-3.2-3B, OLMo-2/3). The steering sections do not name the model; we infer the 1B model (the paper relies "primarily" on it, p3).
- **Where:** residual stream at the comma token that predicts the next number. Probes at all 16 layers. PCA figures use layer 14. Steering uses last-layer representations, one step from the unembedding, so it is close to logit editing and says little about mid-network routing.
- **Baselines:** difference-of-means steering (CAA-like), single probe-direction steering, ideal Bayesian observer, random-cosine floor (~0.02), chance 0.5.
- **Missing:** seeds, error bars, any distance-to-manifold metric. Fig 3A has no shuffled-label or recent-numbers baseline and no count of series per class (layer-0 accuracy is already ~0.875; chance ~0.11).
- **Base models only:** no finetuned or instruct models. Our sycophancy fates are on aligned models, so this is a transfer risk.

## Figures
| Fig | Page | What it shows | Numbers (approx = read from plot) |
|---|---|---|---|
| 1 | 2 | Layer-14 activations form curved bands for mu and sigma; softmax outputs have the same shape | qualitative |
| 2 | 4 | Regime switch: mean jumps at once, std spikes then relaxes; two attractors | mean ~300 to ~690; std ~110, then ~230, back to ~100 by index ~1300; in (D) the peak is std ~240 near mean ~510-520 (approx). The high-variance phase is measured on the output only |
| 3 | 6 | Field probes: accuracy per layer, Gram matrix, interpolation, transfer | acc ~0.875 at L0 to ~0.985 at L15 (dip ~0.905 at L9); interp cos ~0.67-0.83 vs 0.02 floor; transfer (one probe, layer 0) ~0.89, then ~0.55 (delta 50), then ~0.50 (approx) |
| 4 | 6 | Readout-field dimension per layer; field shape at L15 | first eigenvalue ~0.59 (L0), ~0.30 (L13), ~0.23 (L14), ~0.39 (L15); L13/L14 need ~5 components for 0.8 of the variance, so "low-dimensional" is the paper's word (approx) |
| 5 | 7 | Linear vs manifold-spline steering, outputs plotted as (mean, std) | linear: mean moves 300 to ~700 but std peaks ~165; spline: std ~100-122 (approx) |
| 6 | 8 | Probe-direction vs field-aware steering over alpha | probe: std peaks ~185; field-aware: ~100-113 for mean 300-520, then falls to ~60 (approx) |
| 7 | 14 | Output converges to the true distribution after ~100 numbers | KL ~2.2 falls to ~0.05 (approx) |
| 8 | 15 | Hellinger distances between outputs over time | early vs late ~0.4-0.5 (approx) |
| 9 | 18 | Same branch-like manifolds in 9 models at late layers (idx L-2, L-1); idx 1 mostly mixed (6 of 9) | e.g. Llama-3.1-8B idx31 PC1 70% / PC2 20% |
| 10 | 19 | LLM vs ideal observer after the switch | observer: mean ~500, std ~225 at t=2000; LLM: ~690 / ~100 (approx) |
| 11 | 20 | 10 alternating segments: later switches are faster | qualitative |
| 12 | 21 | Schematic: curved data can still have simple readouts; rank-1 vs rank-2 probe fields | none |
| 13 | 22 | Additive (mu + sigma) interpolation misses true centroids | no error metric |

Page images: `../papers/w2-shape-of-beliefs-2602.02315/pages/p02.png` ... `p22.png`.

**Text vs figure mismatches:**
- Section 4.2 says the alpha sweep is in "Fig. 4". It is actually Fig. 6.
- Appendix D says N(700, 300), but Fig. 11 says N(700, 100), and the plot fits 100.
- Fig. 9's caption reuses L to mean a layer index, although L was defined as the number of layers.

## Key quotes
- p5: "probes transfer only locally along the manifold; transfer performance decays with distance in µ..."
- p9: "Linear separability does not entail geometrically faithful control..."

## Methods we could reuse
1. **Linear field probes + Gram spectrum.** One multiclass probe per layer over bins of a continuous variable (for us: confidence bins or pressure strength). Compare spectra or probe spans between resisted and flipped items.
2. **Transfer vs distance.** Our H3 check.
3. **Steering along a centroid spline vs a straight vector,** always reading an off-target covariate.
4. **Field-aware steering:** a kernel-weighted mix of point probes.
5. **Update trajectories:** attractors, a high-entropy crossing, and an ideal-observer reference. This could locate where flipped and resisted paths fork.
6. **Additivity test:** check whether a confidence x pressure grid is additive. If not, one global fix direction will not be enough.

## Pitfalls
- Steering scale (alpha) is arbitrary. Small alpha can do nothing ("activation plateaus") and large alpha breaks the model (p8).
- The last layer behaves differently (p6).
- Even the geometry-aware steering only works over part of the range, mean 300-500 (p8).
- One-vs-rest probes did worse than one multiclass probe (p16).
- "Off-manifold" is judged by eye, from plots.
- The output can match the target before the data supports it: the paper reads this as a normality prior (p14).
- Base models only (p9).

## Limitations (p9)
"Our analysis relies on true activations of base models without finetuning; nevertheless, it is currently constrained to numerical settings."

## Fates
| Fate | How | Strength |
|---|---|---|
| 1 readout | Encoding geometry is not the same as readout geometry | weak |
| 2 bypass | Analogy: linear steering moves the target but distorts covariates; no null case in the paper | weak |
| 3 not routed | not addressed | none |
| 4 override | Two attractors pulling the state (analogy only) | weak |
| 5 transformed | Probes transfer only locally (layer 0, one probe pair), and probe directions rotate; design idea | weak-moderate |
| 6 erased | not addressed | none |

## Group angle
- The paper computes no principal angles, Grassmannian paths or Procrustes.
- Extension for us: compute principal angles between probe spans across layers (a path on the Grassmannian), and between resisted and flipped items.
- Clean visual evidence that manifold steering beats linear steering, but only on a 1B model and a toy task.

## Verdict
**Adapt.** Reuse the tests and the controls. Do not cite it as evidence about factual tasks or aligned models. The "field of local probes for monitoring" idea is our inference, not the paper's.
