# Geometry of Divergence: Tracking Hidden-State Trajectories for Adaptive Multi-Turn Reasoning

Liang, Yu, Nasiri, Garraghan (Lancaster / Huddersfield). arXiv 2608.30650v1, 31 Aug 2026. Slug: `u3-geometry-of-divergence-2608.30650`.

## In one paragraph
Each multi-turn agent episode is treated as a path of hidden states: one state per turn, at the last user-input token, at one middle layer. Two path statistics are computed. **Curvature** (kappa) is the cosine between consecutive turn-to-turn steps. **Variance slope** (beta) is how fast the spread of the visited states grows. Failed episodes have slightly more negative kappa and lower beta, across 6 task-model pairings (Qwen3-14B/32B, Llama-3.1-8B; Math, Code, tau-Bench Retail/Airline). A threshold on kappa decides when to turn on thinking mode. Reward rises from 0.241 to 0.396 averaged, against a never-think baseline. **For us:** low-to-moderate relevance, mainly for direction 3 (jump vs drift) and as a cheap path-statistics baseline. There are no probes, no causal tests, and no knowledge-vs-action setup.

## Sections read
Abstract; 1 Introduction; 2 Related Work (2.1-2.3); 3 Formulation (3.1-3.4); 4 Empirical Analysis (4.1-4.4); 5 Action Chains (5.1-5.2); 6 Geometry-Triggered Adaptive Reasoning (incl. Applicability & Future Direction); 7 Conclusion; References. The text cites an Appendix, but none is in paper.txt.

## Setup
- **Data:** Math and Code from "Lost in Conversation" sharded prompts; Airline and Retail from tau-Bench. Labels are binary episode correctness. Math has 259 correct vs 227 incorrect episodes.
- **Models:** Qwen3-14B, Qwen3-32B and Llama-3.1-8B, each acting as both agent and user simulator. The design is only partly crossed: Llama appears only on Math, Qwen3-32B only on tau-Bench.
- **Readout:** final token of each user input. Layer 22 (Qwen3-14B), 42 (Qwen3-32B), 16 (Llama-3.1-8B). Fig 3 shows layers 20/24.
- **Policies compared:** Never-think, Always-think, Warm-up (think on the first 2 turns), action-conditioned, geometry thresholds, and a learned classifier. 5 trials each. Mann-Whitney tests.
- **Controls:** turn alignment, the same warm-up for every policy except Never-think, identical windows for both metrics. No kappa null, no layer sweep, no interventions on hidden states.

## Figures and tables (all 8 manifest entries; "approx" values are read off plots)
- **Fig 1 (p1):** schematic of a success path vs a failure path, with an action chain and the "timing for adaptive reasoning". *Yes*.
- **Fig 2 (p4):**
  - (a) Medians of per-episode kappa, correct/incorrect: Math -0.17/-0.235; Code +0.13/-0.055; Retail-14B -0.215/-0.24; Airline-14B -0.12/-0.17; Retail-32B -0.145/-0.185; Airline-32B -0.04/-0.125.
  - (b) z-scored beta, correct ~+0.2 to +0.8 vs incorrect ~-0.05 to -0.48. p<.001 except Airline (0.30, 0.057).
  - *Partly*: the direction matches the text, but panel (a) has no tests and the boxes overlap heavily.
- **Fig 3 (p4):** Math, turns 2-7. kappa starts at ~-0.15 for both groups. By turn 4: correct ~-0.27, incorrect ~-0.34. Incorrect reaches ~-0.41 by turn 7. beta at L24 reaches ~88 (correct) vs ~78 (incorrect). The gap opens gradually: drift, not a jump. Both groups drift more negative with turn index. *Partly*: the figure uses layers 20/24, but the stated Llama layer is 16.
- **Fig 4 (p5):**
  - (a) Retail effect sizes (AUC - 1/2) are mostly on the variance-slope axis; Respond x3 is at ~0.19.
  - (b) Airline adds curvature; Respond x3 is at ~0.16 on curvature. The maximum effect is ~0.22 (AUC ~0.72).
  - (c) 2-D PCA of Respond x3: at t+0 correct and incorrect are in different clusters; by t+2 their means coincide.
  - *Partly*: the PCA shows convergence, not the "expanding outwards" the text describes, and the domain is unstated.
- **Table 1 (p5):** share of chains containing each action. Retail differences are small (-7.4 to +0.2). Airline differences are large: Write +23.7, Read -21.7, Transfer -21.7. *Yes*. Note that behaviour itself differs a lot in Airline.
- **Table 2 (p5):** top-20 chains. Respond x3 is 9.1/23.6% (correct/incorrect) in Airline; Read-Read-Transfer is 12.6/1.7%. *Yes*.
- **Fig 5 (p6):** Qwen3-14B rewards. Retail: Never 0.286, Warm-up 0.361, kappa 0.397, Learned 0.398. Airline: 0.196, 0.348, 0.384, 0.420. In Airline, read accuracy is lowest under geometry triggers (~0.565 vs 0.65 for Always-think). *Yes* for rewards; the read-accuracy drop is not discussed.
- **Table 3 (p7):** full reward and token table. *Partly*: the 0.396 headline is the per-row best of two triggers (reader's arithmetic). The triggers alone average 0.381 (beta) and 0.394 (kappa). Against the Warm-up policy built into every trigger (avg ~0.349), the gain is ~+0.05.

## Key results (quotes checked)
- Signals separate outcomes before completion (p1).
- Correct-minus-incorrect kappa gaps are small: "Math +0.047, Code +0.148, Retail (14B) +0.030, Airline (14B) +0.049, Retail (32B) +0.035, Airline (32B) +0.088" (p4).
- beta separation is not significant in Airline (p4).
- The authors acknowledge a length confound: "incorrect episodes are systematically longer than successful ones..." (p4).
- Output-type action chains carry the signal; acquisition chains do not (p2).
- Trigger reward 0.241 -> 0.396 (p7). A single kappa threshold ties the learned classifier, 0.397 vs 0.398 (p7).
- Warm-up alone beats Always-think on Qwen3-32B Retail, 0.405 vs 0.384 (p6). Action triggers do not transfer across domains (p7).

## Methods we could use
1. **Curvature kappa_t:** consecutive-step cosine along a path. Use for direction 3, or adapt to token-by-token or layer-by-layer paths of flipped vs resisted items. Negligible compute. Needs a null.
2. **Variance slope beta_t:** growth of the spread of the visited states. A weak contraction signal.
3. **Turn/position alignment:** a must-have control for any fork-point analysis.
4. **Behaviour stratification before geometry** (action chains). Analog for us: what the model did on the previous turn.
5. **Geometry-triggered intervention:** a monitor-to-fix pattern. Could be a Stage B inference-time arm, e.g. re-check the model's own answer when the representation reverses after pushback.

## Controls worth copying
Position/turn alignment; one shared warm-up across policies; stratify by behaviour; report cost alongside accuracy.

## Pitfalls
- **Length confound** (p4). Per-episode means still mix in more of the later, more-negative turns for the longer failures (reader's note from Fig 3).
- **Unfair baseline:** the headline compares against Never-think, but "every method except Never-thinking adopts this same Warm-up" (p6).
- **Same-data tuning:** thresholds are tuned by sweep on the same evaluation (p7), and the reported average takes the best of two triggers per row.
- **No kappa null.** Reader's note: jitter around a fixed point gives negative kappa by construction (about -0.5 for i.i.d. jitter), so "reversal" is not shown.
- **Task-dependent:** "these signals do not transfer universally across tasks..." (p2).
- **Simulated users:** the same LLM is the user simulator (p3).
- **Thin readout:** one layer and one token per model; the layer reporting is inconsistent; purely correlational; small effects (AUC - 1/2 <= ~0.22); the 2-D PCA story is not what the plotted means show (p6).

**Limitations (closest, p2):** "these signals do not transfer universally across tasks, which is consistent with the task-dependent behavior we observe for the variance slope". There is no limitations section.

**Future work (p7):** "trajectory geometry provides a new lens on multi-turn adversarial attacks, which is an active area in which jailbreaks are spread across turns to divert an agent away from its objective".

## Bearing on our fates
| Fate | How | Strength |
|---|---|---|
| Direction 3 (jump vs drift) | Cheap turn-level path statistics; in Math the gap widens gradually from turn 3-4 | moderate |
| 4 override | A kappa reversal at a pushback turn is a possible signature, not tested | weak |
| 6 erased | Variance contraction is only loosely related | weak |
| 1, 2, 3, 5 | Nothing (no probes, no patching) | weak |

## Group-geometry angle
**Helps:** it compares groups of paths with simple statistics (step cosine, spread growth) plus turn alignment. These are cheap baselines for our fork-point and token-path analysis.

**Warns:**
- Path statistics confound length and index.
- They need a null.
- They should be checked across layers.
- 2-D PCA pictures can contradict the text.

It uses no subspaces, principal angles, RSA/CKA or causal tests.

## Where in the architecture
One residual layer per model, at the last user token: ~55% depth (Qwen3-14B), ~66% (Qwen3-32B), 50% (Llama-3.1-8B). Layers 20/24 look similar for Math. There is no head or MLP analysis. The direction of the effect is consistent across the 6 pairings, but the design is only partly crossed.

## Actionable fix
A training-free monitor: when kappa drops below a threshold, switch on thinking. The gain over warm-up is modest (~+0.05 reward), at lower token cost. For us: a possible inference-time arm in Stage B, to compare against fate-matched fixes and DPO.

## Verdict
**Adapt (light).** Copy the alignment control and the kappa/beta baselines for direction 3. Avoid three things: reading negative kappa as reversal without a null, the never-think headline, and best-of selection.

## Relevance (honest)
Low to moderate. It studies multi-turn agent episodes with outcome labels, not correctness information that the model acts against. There are no probes and no causal tests, and only one layer per model is read. **Less relevant than its title suggests**; useful mainly for direction 3 and as a baseline.
