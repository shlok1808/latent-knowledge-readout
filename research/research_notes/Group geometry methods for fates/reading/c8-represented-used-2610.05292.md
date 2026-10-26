# Erased, Rerouted, or Rescaled? Post-Training and the Causal Quotient of a Language Model's Belief State

Li, Zheng, Wu, Chen. arXiv 2610.05292v1 (4 Oct 2026). Slug: `c8-represented-used-2610.05292`

## In one line
When post-training stops rewarding a distinction the model already encodes, the distinction is usually **kept but no longer used (rerouted)** or **kept and used at lower variance (rescaled)**. It is erased only under heavy extra pressure. A KL anchor decides whether the decision keeps using it.

## Relevance to our project (honest)
The methods are very close to our fate taxonomy: erased, rerouted (our bypass) and re-encoded (our transformed) become separately measurable, with a control for each. On substance it is less relevant than the title suggests. It studies what *training* does to information the reward ignores, not what happens at *inference* when pushback flips an answer. Its LM evidence is synthetic HMM streams on Qwen3 1.7B-4B. Take the measurement design, not the findings.

## Sections read (all, in order)
Abstract; 1 Introduction; 2 Quotients of a Belief State; 3 Which Reward-Null Distinctions Are Protected; 4 Experiments in Exact Worlds (Setup); 4.1 Compression Is Not Loss; 4.2 The Anchor Decides Use; 4.3 Erasure Needs Pressure Beyond the Objective; 4.4 A Recurrent State Spreads Freedom and Protection; 4.5 What Allocation Costs; 4.6 Optimization Settles What the Reward Leaves Open; 5 Open Language Models; 6 Related Work; 7 Conclusion; Limitations; References; A Theory; A.1 Anchor identity; A.2 Filter closure; A.3 The spectrum; A.4 Tied actions under shared parameters; B Worlds, Models and Training; C Measurements; D Additional Results in Exact Worlds; D.1 The direction without the anchor; E Language-Model Experiments. (26 headings)

## Setup
- **Data:** hidden-Markov worlds where the exact Bayesian belief state is known. The hidden state is a pair (task variable, nuisance variable). The reward at a query token reads only the task class, which defines a closed-form *reward-null kernel* with a next-token-visible part and a next-token-invisible part. Worlds: separable (main, second point, slow nuisance), visible-kernel control, entangled, Mess3. Action designs: tied, one-per-class, fresh tokens (a built-in control). LMs: in-context HMM streams (code from Balcells et al. 2026), plus a controlled GRPO task on six letters.
- **Models:** 4-layer transformers and 2-layer GRUs (width 128) pretrained to Bayes-optimal loss. Post-trained with SFT, DPO, GRPO, Dr. GRPO, REINFORCE and exact-gradient KL-RL over a range of beta that includes 0. LMs: base and post-trained pairs of Qwen3-1.7B, Qwen3-4B and Qwen3.5-4B; LoRA-32 GRPO on Qwen3-1.7B-Base and Qwen3-4B-Base.
- **Positions/layers:** the query (decision) position vs ordinary positions; all depths; LM probes on every layer by relative depth.
- **Baselines/controls:** all numbers are post/base ratios. Also: an untrained network, a last-8-tokens baseline, the exact Bayes filter, the theoretical anchored optimum, random subspaces of the same dimension, matched reward, matched KL, a gauge control, a random-invertible-map check, frozen vs refit probes, head-only training, the exact gradient, and pre-registered criteria.

## Key results (with quotes)
1. Mostly rerouted or rescaled, not erased: "In controlled worlds, post-training mostly reroutes or rescales reward-null information and leaves it decodable." (p1)
2. Erasure needs extra pressure: "Erasure appears only under prolonged weight decay, for distinctions that neither reward nor next-token prediction can see." (p1)
3. SFT reroutes: "After SFT the next-token-visible component is as decodable as before (Rvis = 0.98) and the decision has stopped responding to it (U = 0.15)" (p5)
4. Same reward, different fate: "KL-RL at β = 1 and SFT reach the same reward, the first with (Rinv , U, A) = (1.00, 1.00, 0.69) and the second with (0.75, 0.15, 0.13)." (p5)
5. Use flips at a threshold: "At the policy level the fate switches at a threshold in β, within (0.003, 0.02] for GRPO, whereas Rinv and A vary smoothly through it." (p6)
6. Anchor under stress: "...still reproduces the reference's log-odds (γ = 0.99), and retains 0.64 of the invisible component." (p6)
7. GRU, encoded but unused: "After SFT its filter all but stops using the invisible component (0.01 of the base) although half of it is still decodable from the same state" (p6)
8. Variance share is not information: "of the 172 runs whose kernel share fell below a tenth of the base, 153 keep R ≥ 0.9" (p7)
9. Decodable but inaccessible: "Once that variance has dropped two-hundredfold, a network that still encodes the nuisance belief linearly (R2 = 0.93) is indistinguishable from an untrained one." (p7)
10. LMs: "Post-training leaves both decodable: peak R2 changes by at most 0.05 for the belief state" (p7), while "the effective rank of the later half of the network falls by 12 to 19%" (p7).
11. In Qwen3, no anchor means amplification: "Without it they are amplified ten to twenty times and no run collapses" (p7)
12. Re-encoding: "At β = 0.04 the kernel's image in the residual stream rotates by 20 to 77 degrees." (p13)
13. "A frozen diagnostic would read re-encoding as erasure, which is why every coordinate here is refit on the model under test." (p13)
14. Two exits: "After SFT the policy responds to a twin exchange at 0.15 of the base and the next-token head at 0.56." (p13)
15. Ridge pitfall: "...the least-squares R2 of the kernel stays at 0.954 and the ridge R2 falls to 0.139." (p11)

## Figures and tables (20 listed, 20 covered; all page images viewed)
| id | page | what is plotted | takeaway | numbers | text matches? |
|---|---|---|---|---|---|
| Figure 1 | 2 | Schematic of fates; gamma vs beta; refit R2 of visible/invisible parts per objective; measured vs theoretical sensitivity | A small anchor keeps the log-odds; the visible part survives; theory predicts the policy | gamma ~0.1-0.5 at beta=0, ~1 for beta>=0.01; invisible R ~0.75-0.9, ~0 under weight decay (approx, read from figure) | yes |
| Table 1 | 3 | Definitions of R, U, A | Fate = signature in R/U/A | none | yes |
| Table 2 | 4 | Map of worlds and action designs | Each variant isolates one protection | none | yes |
| Figure 2 | 5 | A (log) vs U for each objective; marker fill = R | Moves left (rescale) and down (reroute); only SFT + weight decay empties | SFT A~0.13, U~0.15; GRPO beta=0 A~0.07, U~0.55; KL-RL beta=1 A~0.69, U=1 (approx, read from figure) | yes |
| Figure 3 | 6 | GRU/transformer filter use vs R_inv at ordinary positions | Encoded but no longer used (GRU) | GRU U_filt ~0.01-0.1 at R_inv 0.25-0.75 (approx) | yes |
| Figure 4 | 6 | Kernel share vs RankMe; revival steps vs retained variance | Allocation is a separate property and has a cost | slope ~-1; not reached below ~1% share (approx) | yes |
| Figure 5 | 8 | Policy vs reference log-odds; gamma over steps (toy, Qwen3-1.7B) | Anchor keeps log-odds; no anchor: toy collapses, LM amplifies | panels print gamma 0.45, 1.00, 16.0, 9.40, 0.895 | yes |
| Table 3 | 14 | R_vis, R_inv, U, gamma, A, fate per objective | Anchored = rescaled; SFT/unanchored = rerouted; weight decay = erased | e.g. SFT 0.96-0.98 / 0.59-0.75 / 0.15-0.22 / - / 0.03-0.13 | yes |
| Figure 6 | 14 | Figure 2 in 8 other world/architecture combos | Same pattern everywhere | SFT U ~0-0.35 (approx) | yes |
| Figure 7 | 15 | gamma, intercept, preference vs beta; matched-KL inset | Sharp anchor threshold; representation smooth | GRPO (0.003,0.02], KL-RL (0,0.003] (printed) | yes |
| Figure 8 | 15 | Policy exit vs next-token exit use of the kernel | Rerouting is visible inside one network | SFT 0.15 vs 0.56; GRPO beta=0 ~0.53 vs 0.75 (approx) | partly: text says GRPO near 0.9, which fits only the transfer-share panel |
| Figure 9 | 15 | Matched reward (SFT vs KL-RL); matched KL | Same reward with and without rerouting | as in text (1.00, 1.00, 0.69) vs (0.75, 0.15, 0.13) | yes |
| Figure 10 | 16 | R_inv, R_vis over 50k steps with/without weight decay | Erasure is late and needs weight decay | SFT R_inv 0.69 -> 0.30 -> 0.001 | yes |
| Figure 11 | 17 | GRU vs transformer decodability, next-token loss rescue, filter use | GRU spreads freedom and protection; next-token loss rescues GRU only | GRU gamma ~0.27 -> ~0.95 with next-token loss (approx) | yes |
| Figure 12 | 18 | Gauge control; share vs rank; logit-space share vs beta | Rescaling moves share 10-100x with R2 and interchange unchanged | share ~0.01 at s=0.1 (approx) | yes |
| Figure 13 | 19 | Revival curves; t95 vs kernel share | Low share = slow or no revival | base ~150-250 steps (approx) | yes |
| Figure 14 | 20 | Second-moment shares over training; gamma; rate ratio | Toy removes the mean and collapses; LM keeps the mean and amplifies | Qwen3 mean^2 0.80-0.95 (text) | yes |
| Table 4 | 21 | Seed-study outcomes | Outcome is fixed by the pretrained model | main GRPO gamma 0.39-0.51, collapse | yes |
| Figure 15 | 21 | Every seed run: gamma and shift over steps | Runs from one pretrained model overlap | shift up to ~6 nat (approx) | yes |
| Figure 16 | 22 | LM probe R2 by relative depth, base vs post; RankMe by depth | Decodability unchanged; late effective rank falls | RankMe ~0.80 (1.7B), ~0.85-0.9 (4B) late (approx) | yes |

## Methods worth stealing (and which fate each diagnoses)
- **R/U/A fate test.** R: refit whitened least-squares probe. U: twin-exchange use. A: variance share. All as ratios to a no-pressure reference. Diagnoses erased (6) vs bypass/not routed (2/3), and separates both from harmless rescaling.
- **Twin exchange.** A probe-free use measure built from prompt pairs matched on everything except the correctness-relevant content. Separates bypass from override.
- **Frozen vs refit probe, plus subspace rotation angle.** Tells transformed (5) from erased (6). A direct test of H3.
- **Two-exit test.** Compare use of the same subspace by two readouts in one forward pass. Diagnoses bypass at the decision only.
- **Slope/intercept regression of per-item log-odds (pressure vs no pressure).** Slope kept with a big negative intercept points to override by a constant push. Slope near 0 points to bypass.
- **Gauge control.** Rescale a subspace and compensate downstream. A metric that moves is measuring allocation, not information.
- **Revival test.** A new linear head on frozen activations, counting steps to 95%. Catches information that is decodable but practically inaccessible.
- **Whole-state filter-use patching.** Patch all layers and positions. Patching a single layer can be undone downstream.
- **Second-moment decomposition** (mean, class, nuisance, interaction) of the decision representation.

Code: no URL is given for the authors' own code. The LM part reuses Balcells et al. (2026) released code.

## Controls worth copying
Ratios to a reference condition. Refit plus frozen probes. Whitened unpenalized probes. Random-invertible-map check. Random-subspace interchange baseline. Gauge control. Matched-outcome comparisons (reward, KL). Untrained and last-k-tokens baselines. A built-in condition where the decision should never read the subspace. Exact-gradient and head-only controls. Pretraining-seed vs post-training-seed variance. Pre-registration.

## Pitfalls
- Frozen probes read re-encoding as erasure (p13, quoted above).
- Ridge probes read lower variance as lost information: 0.954 vs 0.139 (p11).
- "Spectral compression alone cannot identify information loss." (p3)
- "Patching a single layer or position is undone by the layers and positions left untouched, which is why the whole state is used." (p13)
- "A transformer recomputes part of an exchanged component from the tokens left in place" (p16)
- Interleaved within-sequence splits inflate decodability (p19). Note: their own LM probe for the invisible component uses ridge, which their appendix warns about.
- Linear probes only: "a distinction that survives only in a nonlinear code is counted as lost." (p9)
- Compare at matched distance: "The divergence at the policy level is not a consequence of unanchored runs travelling farther." (p16)

## Limitations and future work (quoted)
- Limitation (p9): "The exact ground truth that makes the three fates identifiable exists only in small hidden Markov worlds, with a one-step decision whose action is not fed back into the context."
- Future work (p9): "Whether a multi-step task, in which the policy also changes which histories are visited, moves the quotient or only the support of the belief state remains to be studied."

## How it bears on our six fates
| fate | how | strength |
|---|---|---|
| 2 bypass | "rerouted" = R stays, U falls; SFT gives U 0.15 with R_vis 0.98; the two-exit test | strong |
| 5 transformed | Subspace rotates 20-77 degrees; the frozen probe fails, the refit works | strong |
| 6 erased | Needs 50k steps of weight decay; rare | moderate |
| 3 not routed | GRU filter ignores a half-decodable component; revival cost | moderate |
| 4 override | A history-independent shift (4.6-6.1 nats) dominates while R stays; analogy only | weak |
| 1 readout/threshold | Rescaling warns that low variance is not a failure; no direct test | weak |

## Group/subspace angle
It works at the subspace level throughout and measures subspace rotation in degrees, which is a Grassmannian distance. It also decomposes the decision representation into group second moments. **Warning (Proposition 1):** an invertible linear map keeps all probe information and interchange effects but can set the covariance to almost anything. Variance-based group statistics (PCA share, effective rank, and covariance-based RSA/CKA comparisons across pressures) can therefore change with no change in information or use. Pair every group-geometry comparison with refit probes and interchange tests. Run the Procrustes idea as "frozen vs refit".

## Where in the architecture
There is no attention vs MLP localization. A GRU must carry the state, so it loses or keeps a distinction everywhere. A transformer can recompute at the decision position, so the decision position can behave differently from ordinary positions. In the Qwen3 pairs, decodability is unchanged at every relative depth while effective rank falls 12-19% in the later half. GRPO changes are written in the residual stream (head-only training does not collapse). Tuning only blocks 20-27 has a similar effect to LoRA on all blocks. Results are consistent across post-training seeds but depend on the pretrained model; the LM evidence covers one family.

## Actionable fix
- Keep a KL anchor in post-training, so the decision keeps using what the reference used.
- SFT on fixed labels and unanchored RLVR reroute rather than remove.
- For our Stage B: plain DPO acts like anchored RL here, and SFT-style fixes are likely to hide rather than remove. Recheck R (frozen and refit), U and A separately.
- Do not use effective rank or variance share as evidence of loss.

## Reuse verdict
**Adapt.** Copy the measurement design (R/U/A, twins, frozen vs refit, two exits, gauge control, slope/intercept). Do not carry over the findings: toy worlds, synthetic streams, 1.7-4B, no factual pressure task.
