# Non-linear Interventions on LLMs (arXiv 2605.14749)

Sangwoo Kim (Seoul National University). v1, 14 May 2026. A 4-page preprint plus an appendix.
Code (stated, not checked): anonymous.4open.science/r/nonlinear-intervention-77AC/

## In one line
The method swaps the linear "change of basis" in steering for an **invertible non-linear map** (an i-ResNet) and trains it with interchange interventions. A **single-site, mid-layer** edit then removes refusal about as well as linear methods that edit hundreds of sites. The **same pipeline restricted to a linear map does nothing.**

## Most useful for us
- **A matched linear vs non-linear test:** same site, same data, same loss, with the map restricted to linear. Linear gets 0.010 / 0.072; non-linear gets 0.813 / 0.760 (Llama / Qwen).
- For fate sorting, run both maps at one site on flipped vs resisted items:
  - neither works: bypass or erased
  - only non-linear works: transformed, or encoded non-linearly
  - linear works: plain linear routing
- **Layer sweep = WHERE.** The edit only works in a middle band of layers.

## Setup
- **Data:** D+ is 2,000 harmless Alpaca prompts (the model complies). D- is 2,000 harmful SALAD-Bench prompts (the model refuses). Evaluation is StrongREJECT over 313 harmful prompts, scored by an LLM judge.
- **Models:** Llama-3-8B-Instruct and Qwen2.5-7B-Instruct, frozen.
- **Site:** residual stream block output. Llama: layer 10, token -3. Qwen: layer 14, token -4. Both are prompt tokens before generation.
- **Map:** 2 i-ResNet blocks (hidden width 128, spectral norm, Lipschitz 0.6) with k = 1 coordinate. At test time that coordinate is clamped to the mean over harmless prompts.
- **Loss:** hinge loss that pushes the class-mean-difference projections at downstream sites (those with AUC > tau) up to the harmless mean. The tau value is not reported.
- **Baselines:** DIM and RDO, each as ablation and as ActAdd.
- **Missing:** seeds, CIs, any capability check on benign prompts.

## Figures and tables
| Item | Page | What it shows | Numbers |
|---|---|---|---|
| Fig 1 | 3 | StrongREJECT by method | Llama: none ~0.01, DIM ~0.64/0.70, RDO ~0.66/0.83, non-linear 0.813. Qwen: none ~0.07, DIM ~0.74/0.75, RDO abl ~0.66, non-linear 0.760 (approx except non-linear) |
| Table 3 | 4 | Map restricted to linear vs i-ResNet | linear 0.010 / 0.072; non-linear 0.813 / 0.760 |
| Table 1 | 4 | Sites edited and total L2 | Llama non-linear: 1 site, L2 3.5 vs 572-59,396 sites, L2 1,812-2,899. Qwen: 1 site, 28.3 vs 15,474-41,376 |
| Table 2 | 4 | Linear shift at every loss site vs non-linear | 0.689 / 0.415 vs 0.813 / 0.760 |
| Fig 2 | 4 | Layer sweep | Llama: <=0.27 up to L9, ~0.80 at L10, ~0.44-0.78 at L10-18, ~0.05 from L19. Qwen: ~0.5-0.76 at L8-19, low outside (approx) |
| Table 4 | 8 | Intervention sites | Llama L10 / token -3; Qwen L14 / token -4 |

Page images: `../papers/w2-nonlinear-interventions-2605.14749/pages/p03.png`, `p04.png`, `p08.png`.

**Our own check of Table 1:**
- The "two orders of magnitude smaller" edit comes from editing 1 site instead of many.
- Per edited site, the non-linear L2 (3.5 Llama, 28.3 Qwen) is about the same as ActAdd's (~3.17 Llama, ~41.8 Qwen).

**Text vs table:** "Substantially short" (Table 2) fits Qwen (gap 0.35) better than Llama (gap 0.12).

## Key quotes
- p1: "Our non-linear intervention attains steering effectiveness comparable to strong linear baselines while editing activations at orders of magnitude fewer hidden-state locations."
- p4: "for both models, the intervention is largely ineffective at early layers, peaks at middle layers, and declines at later layers"

## Methods we could reuse
1. **Invertible non-linear interchange intervention:** inject the resisted-item coordinate into flipped items. This is a fate 3 test.
2. **Map restricted to linear, same objective:** the bypass vs transformed contrast.
3. **Linear shift at all downstream loss sites:** a control showing whether one upstream cause or many downstream symptoms carry the effect.
4. **Layer sweep:** localises where an edit works.
5. **Downstream-site hinge loss:** for implicit targets such as deference or hedging.

## Pitfalls
- The site is chosen "heuristically" (p3). The Llama site is the best layer in the sweep, so it may be tuned on the evaluation set.
- A baseline (RDO ActAdd on Qwen, 0.020) was dropped as unreliable.
- The linear-restricted control scores exactly the no-intervention level, and its edit size is not reported. It may be under-tuned.
- No shuffled-label control, so we cannot rule out that the flexible map creates the effect rather than finds it.
- No capability check, no seeds, and only one task.
- Success shows causal controllability, not endogenous mediation.

## Limitations (p4)
"We realize fθ only as an i-ResNet, and the same framework admits other invertible architectures."
Future work: choosing the intervention position automatically.

## Fates
| Fate | How | Strength |
|---|---|---|
| 1 readout | not addressed | none |
| 2 bypass | A linear null at a site does not mean the site is inert | moderate |
| 3 not routed | Injection test at one site | moderate |
| 4 override | Single upstream cause vs many downstream ones (Table 2), indirect | weak |
| 5 transformed | A non-linear coordinate could capture re-encoded information | weak |
| 6 erased | not addressed | none |

## Where in the architecture
- Residual stream at one prompt token near the end of the chat template.
- Middle layers: Llama ~10-18 of 32, Qwen ~8-19 of 28 (approx).
- No attention, MLP or head localisation.

## Verdict
**Adapt the method; do not lean on it as evidence.** Thin evidence: one author, two 7-8B models, one task, no seeds.
