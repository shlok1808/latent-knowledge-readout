# Layer-wise Representation Dynamics: An Empirical Investigation Across Embedders and Base LLMs

Jiang, Yang, Tam (HKUST), arXiv 2605.12714, May 2026. Slug: `w2-layerwise-grassmann-dynamics-2605.12714`.

## In one paragraph

LRD runs three separate layer-wise measures on 31 models (13 encoder embedders, 12 decoder embedders, 6 base LLMs up to about 9B) over the inputs of 30 MTEB tasks:
- **Frenet:** Grassmann geodesic speed between adjacent layers' PCA subspaces, Menger curvature over consecutive triples, and end-to-end displacement d_{0,L}.
- **NRS:** per-anchor nearest-neighbour Jaccard between adjacent layers.
- **GFMI:** mutual information between each layer's kNN-graph components and the final layer's.

Decoder models bend their subspace path late (family-mean curvature peak at 0.78-0.89 of depth; 5 of 6 base LLMs, GPT-2 is the exception, and the decoder-embedder mean is driven by 8 Mistral-7B fine-tunes). For model selection, d_{0,L} is the strongest of the paper's three scores (mean Spearman +0.64); no external baseline was tested. For layer skipping, only GFMI beats random skipping, and only at 15-20% budgets. On the one base LLM (Mistral-7B, 3 tasks; Table D.6) no rule clearly beats random, but n = 3 is noise.

**Most useful for us:** the exact recipe for Grassmann speed and curvature, plus the Random/last-k layer baselines. Per-anchor kNN retention is the one item-level tool we could turn into a neutral-vs-pushback test.

**Main catch:** everything is population-level and about embedding quality, not failures. Distances are not normalised for subspace rank.

## Sections read
Abstract; 1 Introduction; 2 Related Work (2.1-2.3); 3 Measurement families (3.1 Frenet, 3.2 NRS, 3.3 GFMI); 4 Setup and observations (4.1, 4.2); 5 Model selection (5.1-5.3); 6 Layer pruning (6.1-6.3); 7 Limitations and Conclusion; References; Appendix A (A.1-A.3 formulas); B (B.1 models/tasks, B.2 protocols/compute); C (additional analyses); D (D.1 selection, D.2 pruning).

## Setup
- **Data:** 30 MTEB task inputs (13 CLS, 11 RET, 6 other); one pooled vector per example per layer.
- **Models:** 31 in total. Decoder embedders are mostly Mistral-7B based, plus Gemma-2-9B, Qwen2-7B and Qwen3 0.6B/8B. Base LLMs: Falcon-7B, Gemma-2-2B, GPT-2, GPT-2 XL, Llama-3.2-1B, Mistral-7B-v0.1.
- **Subspace:** top-r PCA of the centered example cloud. r is the smallest rank explaining 95% of final-layer variance and is reused at all layers.
- **Hyperparameters:** NRS k = 20 (up to 500 anchors); GFMI k = 30 with 20 percentile points over [5, 95].
- **Baselines:** none head-to-head for selection (the authors say so). For pruning: Random (3 seeds) and last-k, with first/last 3 layers protected, on a separate 5-model x 3-task panel (BGE-base, E5-large, Nemotron-Embed-1B, LLM2Vec-Mistral-7B, Mistral-7B-v0.1; the 3rd and 4th are not in the 31-model panel).
- **Pooling:** official or model-recommended "when available"; no rule stated for the base LLMs.
- **Compute:** 4x V100, about 300 GPU-hours; forward passes only.

## Figures (manifest items checked against page images; appendix items text only)

| Item | Page | What it shows | Numbers | Matches text? |
|---|---|---|---|---|
| Table 1 | 5 | Curvature-peak depth and NRS trend by family | encoder 0.56 / +0.015; decoder emb. 0.89 / +0.144; base LLM 0.78 / +0.121 | yes, but means hide spread: encoder peaks 0.08-0.99; 8 of 12 decoder emb. are Mistral fine-tunes at 0.867-0.868; GPT-2 0.38 |
| Table 2 | 6 | Mean curvature, speed, GFMI by family and task type | base LLM curvature ~0.35 vs ~0.13-0.14 for embedders | partly: encoder "Other" curvature (0.142) > CLS (0.137); encoder GFMI and base-LLM speed are highest for Other, not RET |
| Table 3 | 7 | Selection Spearman | d_{0,L} +0.64 (28/30 sig.), J_late +0.58, G_peak +0.52; MMLU-6 +0.64/+0.46/+0.64 (p ~0.17/0.35/0.17, not significant) | yes |
| Figure 1 | 8 | Per-model score vs MTEB scatter | rho +0.85 / +0.72 / +0.68; d_{0,L} in two clusters (~13-20, ~26-30) (approx) | yes; model-level rho is higher than Table 3's per-task means |
| Table 4 | 8 | Pruning mean/median relative change | GFMI mean -2.37% at 15% vs Random -5.90%; Frenet -21.33% at 20% | yes |
| Figure 2 | 9 | Pruning curves | as Table 4 | yes |
| Tables B.1, B.2 | 17-18 | Model and task panels | layers 6-48 | text only |
| Fig C.1, C.2, Table C.3 | 19-21 | Per-model summaries | Mistral-7B-v0.1 peak 0.856; GPT-2 XL 0.893; GPT-2 curvature 0.813 | checked on rendered pages (by the checker) |
| Table D.4, Figs D.3, D.4 | 21-24 | Per-task Spearman | d_{0,L} +0.12 to +0.88; GFMI -0.07 on SprintDup | checked on rendered pages (by the checker; Fig D.3 not opened) |
| Table D.5 | 24 | Base-LLM panel | GPT-2 XL: high d_{0,L} (20.11) at chance MMLU | checked on rendered page (by the checker) |
| Fig D.5, Table D.6 | 25-26 | Per-cell pruning | cells from -96.6% to +64.9%; Mistral-7B 3-task means show no rule beating Random | checked (pdftotext -layout) |

Page images: `../papers/w2-layerwise-grassmann-dynamics-2605.12714/pages/p05.png` ... `p09.png`.

## Key results (exact quotes)
- "For pruning, GFMI is the only measurement-guided rule that beats Random at the 15% and 20% budgets and has the best median change at every budget." (p. 1)
- "Curvature peak depth dpeak (κ) separates the three families: encoder embedders peak at mid-depth, whereas decoder embedders and base LLMs peak later (Table 1)." (p. 5)
- "the per-layer rule z(Jl ) does not carry over to pruning, since a layer that does not change local neighborhoods is not necessarily dispensable." (p. 9)
- "The GFMI advantage is concentrated at higher budgets and is largest on encoder embedders. On LLMs the three rules are harder to separate from Random." (p. 9) The stronger "no rule beats random on the base LLM" rests on Table D.6 (p. 26), not this sentence.
- "Final-layer graph alignment therefore captures structure beyond subspace displacement." (p. 19)

## Methods we could use
1. **Frenet (speed, Menger curvature, d_{0,L}).** Run it per condition, or on flipped vs resisted items matched on prior confidence, to find the layer where the population paths separate or bend. Use a fixed r and a normalised distance.
2. **NRS per anchor.** Compare an item's neighbours under neutral vs pushback at the same layer. Did it move into the region of the claimed answer (override or transformed), or lose stable neighbours (erased-like)?
3. **GFMI.** Swap the final-layer partition for the flip/resist label to find the earliest layer whose geometry already predicts behaviour. This marks a fork, not a fate.
4. **Block skipping with Random and last-k baselines.** A crude causal check on a layer band.

## Controls worth copying
- Compare any layer-band claim against Random (several seeds) and last-k; report the median as well as the mean.
- Fix the subspace rank once and reuse it at every layer.
- Keep global, local and alignment-to-final measures separate.
- Protect boundary layers; report per-task significance counts.

## Pitfalls
- Stated: "Random is competitive at light budgets, and the measurement-guided prior is supported only for GFMI past the 10% budget." (p. 9)
- Stated: GFMI thresholds must be per-layer percentiles because distance scales differ across layers (p. 4).
- Our reading: sqrt(sum theta^2) grows with r. d_{0,L} may partly track width or family (two clusters in Fig 1), and there is no control for that.
- Our reading: the measures are population-level; only NRS per anchor is item-level.
- Our reading: relative change is unstable on low baselines (some cells show about +65% after pruning).
- Our reading: pruning budgets collapse for small models (BGE-base: same numbers at 5% and 10%, and at 15% and 20%), so light-budget results partly reflect removing one layer.
- Our reading: Menger curvature 4A/(abc) blows up where speed is tiny; noisy on small between-condition distances.
- Our reading: family means hide spread and non-independence (Mistral fine-tunes dominate the decoder-embedder family).
- Minor: the Mistral-7B citation [30] is wrong.

## Limitations / future work (quotes)
- "The measurements also depend on input sampling, pooling, k-NN construction, and graph filtration percentile. Broader task coverage and stability analyses remain open." (p. 9)
- "Spearman captures global rank agreement, and top-k selection quality is left to future work." (p. 7)

## Bears on fates
| Fate | How | Strength |
|---|---|---|
| 5 transformed vs 6 erased | per-anchor NRS adapted to neutral vs pushback | weak |
| 4 override / H1 | per-condition speed and curvature to find the bend layer per pressure | weak |
| 2 bypass, 3 not routed | block skipping is crude; stable is not the same as dispensable | weak |
| 1 readout | not addressed | none |
| H3 | analogy only | none |

## Group angle
The clearest recipe for speed and curvature on the Grassmannian. Decoder LLMs bend late (5 of 6 base LLMs; GPT-2 does not), and base LLMs are more curved and slower than embedders. This is directionally consistent with 2605.09011's late reorientation phase; qualitative only, since objects and statistics differ. The paper also warns that one geometric number is not enough. It has nothing on manifold vs linear steering, Procrustes, or comparisons between conditions.

## Where / what fix
- **Where:** in decoder models the subspace bends most in the last ~10-25% of depth (n = 5 of 6 base LLMs; the decoder-embedder number is Mistral-driven), and neighbourhoods settle in the last third.
- **Fix:** none for safety. Our proposals (untested):
  - Use Random/last-k layer baselines for every Stage A/B localisation claim.
  - Find the fork layer with a normalised fixed-r Grassmann distance between neutral and pushback item clouds, or with GFMI on behaviour labels, and intervene there.

## Verdict
Reuse: the Frenet formulas (with normalisation), the pruning baselines, and a per-anchor NRS adapted to compare conditions. Skip GFMI as-is and the model-selection study. Relevance: low to medium.
