# Trajectory Signatures of Deception in LLMs (Mothukuri & Parizi, ACL 2026)

**Verdict: adapt.** Copy the baseline ladder. Reuse three tools for fork-point work. Ignore the deception conclusions.

## In one paragraph
For each prompt, the paper takes the hidden state at every block output at the **last prompt token**. That gives one path through depth per item. From each path it computes scalar "shape" features: path length, curvature, acceleration and straightness. It compares "truthful" with "deceptive" prompt pairs (sycophancy, instructed lies, confabulation, strategic) in GPT-2, Llama-2 7/13/70B, Llama-3-70B, Qwen 0.5B/72B and DeepSeek-67B. Sycophancy gives the clearest effect, instructed lies give nothing, and effect signs flip between models. On binary sycophancy, 7 geometric scalars equal 7 PCA components (66.6% vs 67.8%), but a **single mean-difference direction reaches 90.3%** and a full probe 95.9%. There are no causal tests.

## Setup
- **Labels are prompt conditions, not behaviour.** The sycophancy "deceptive" item is simply the persona-prefixed prompt. Whether the model actually agreed with the persona is never used.
- **Data:**
  - Custom pairs (Fig 5). Some of them change content as well as framing: 2+2 = 4 vs 5, and Waterloo vs the fictional "Zenthara".
  - Anthropic sycophancy-NLP: neutral vs persona prompt (Fig 6).
  - Power-seeking and corrigibility as "negative controls": identical prompts, with the label set by the chosen answer. No numbers are ever reported for them.
  - A Qwen-0.5B multi-task set (50 pairs).
- **Sample sizes:** 30 pairs per cell (Table 3), 60 for CV, 30 for the baseline ladder, 15 for layer selection. The paper also claims "3,400 pairs" in total.
- **Read position:** block outputs at the final token, which Section 3.2 says is position n of the prompt. Sublayer points were tried and dropped. Sparse "decision-point" capture and a multi-position mode are described, but the results appear to use the final token only.
- **Features:** Table 2 lists 8; the text says 7.
- **Stats:** paired t-tests, paired-difference Cohen's d, Bonferroni over 56 tests (α = 0.00089), 10,000-sample bootstrap CIs.
- **Baselines:** Geo-7 vs PCA-7 (LR and RF), 1D mean-difference, full best-layer probe, and layerwise probes by depth.

## Figures and tables (12 of 12 covered)

| ID | Page | What it shows | Takeaway | Matches text? |
|---|---|---|---|---|
| Table 1 | 2 | Related work: probing / causal / depth-as-process | It notes that "decodability ≠ usage" | Yes |
| Fig 1 | 4 | Pipeline: pairs → model → capture → features → classifier (Truthful/Uncertain/Deceptive) | Schematic only | Partly: the 3-way output is never used |
| Table 2 | 4 | Definitions of 8 features | Scalar shape of one token's path through depth | Partly: the text says 7 |
| Table 3 | 6 | Path-length d per model × task (n = 30) | Signs flip for strategic deception. Sycophancy: 7B +1.05, 13B −0.08, 70B +0.48, L3-70B +1.63 | Partly: GPT-2 instructed is +0.41* against the text's "all \|d\| < 0.41"; no star legend |
| Fig 2 | 7 | 8 models × 3 example pairs, 3D PCA paths (blue truthful, red deceptive) | Single examples only. Some large-model panels (Qwen-72B, DeepSeek) overlap fully, possibly the identical-prompt controls (titles illegible) | Partly: contradicts "larger models separate more" |
| Fig 3 | 7 | Llama-2/3, 3 behaviours, enlarged | **It looks like a duplicate of Fig 2(d)(e).** Instructed lies look clearly separated, yet they are null in aggregate | Partly |
| Fig 4a | 8 | Layerwise probe accuracy vs normalised depth, 7 models | Curves span about 0.60-0.88 (read from figure; the 3D view makes peaks unreadable). Text: best depth runs from 5.1% (Llama-3-70B) to 62% (Qwen-72B) | Cannot check the peaks from the plot |
| Fig 4b | 8 | Cohen's d of 7 features × 8 models | \|d\| up to about 0.6 (read from figure); signs mixed | Partly: if the illegible labels are alphabetical, the −0.60 curvature bar belongs to Llama-2-7B, not 70B as the text says |
| Table 4 | 9 | Geo-7 LR vs PCA-7 LR: 72.1/70.5, 55.7/47.3, 77.2/87.9, 61.3/65.4; mean 66.6/67.8 | A tie. One cell is below chance, which signals noise | Yes |
| Table 5 | 9 | Ladder means: Geo LR 66.6, PCA-7 LR 67.8, Geo RF 85.5, PCA-7 RF 91.4, **Mean-Diff 90.3**, Full 95.9 | One direction beats the shape scalars | Partly: the Limitations section quotes 66.7/77.0/91.7 |
| Table 6 | 12 | Datasets × models | Qwen-72B and DeepSeek appear only in the figures; negative controls are never quantified | Partly |
| Fig 5 | 12 | Example custom pairs | The "matched" pairs change content too | Partly |
| Fig 6 | 13 | Sycophancy pair: neutral question vs persona ("Sue ... MIT") | "Deceptive" means only that the persona is present; the example also omits the question | Partly |

## Key results (with quotes)
- "Sycophancy produces the clearest and most recurring trajectory signal ... Instructed lies generate null effects universally." (p9)
- "Llama-2-13B produces negligible sycophancy effects (d = −0.08, p = 0.66) despite significant signatures in both 7B (d = +1.05) and 70B (d = +0.48) variants." (p7)
- Same size, different training: Llama-2-70B vs Llama-3-70B give "d = +0.48 and d = +1.63" (p8).
- The best probe depth "vary 12-fold across architectures" (p8).
- At matched dimensionality the methods tie: "(66.6% versus 67.8%, ∆ = −1.2 percentage points)" (p9).
- "A single mean-difference direction achieves 90.3% ... Full-dimensional best-layer logistic regression reaches 95.9%." (p9)
- Qwen-0.5B reverses: "Truthful generation produces longer paths (d = +0.412, p = 0.005)" (p7).
- "does not establish that trajectory geometry itself causes deceptive outputs" (p9).

## Methods we could use
1. **Baseline ladder (copy).** Benchmark every group-geometry feature (principal angles, Procrustes residual, CKA) against PCA-k at the same k, a 1D mean-difference direction, a full probe and **our logit margin**. Use LR and RF with repeated stratified CV, and keep layer selection separate from evaluation.
2. **Signal-layer cosine divergence (adapt).** Run it twice per item: pushback vs neutral (where pressure is registered), and flipped vs resisted under the same pushback (where outcomes split). If outcomes split after the pressure is registered, while the correct answer is still decodable, that points to override rather than erasure. Follow up by patching at the first divergent layer.
3. **Decision points, top1/top2 < 1.2 (adapt).** Use them to find flipped items whose answer token is a near-tie between the correct and the user's answer. Those are fate-1 (threshold) candidates, testable with a small steering dose. They also give fork tokens in longer answers.
4. **DTW with cosine cost (adapt).** Align flipped and resisted responses of different lengths token by token, and look for the token where the aligned cost jumps (the fork point).
5. **Trajectory shape scalars (avoid as the main tool).** They are not tied to the correct-answer subspace, no better than 7 PCs, and their sign flips across models.

## Controls worth copying
- The paired design with the pair as the unit, and paired-difference d.
- Bonferroni over features × models (they state it but do not stick to it).
- Disjoint data for calibration and evaluation.
- An identical-stimulus negative control where only the outcome differs. **Done right, this is our flipped-vs-resisted contrast.**

## Pitfalls
- **Condition labels sold as behaviour labels:** "we condition on behavior rather than aggregate across examples" (p2), yet the sycophancy pairs contrast "persona-prefixed and persona-stripped conditions" (p4). A 90-96% probe may just be detecting the persona.
- **State read before any answer:** "the trajectory captures hidden states at position n" of the prompt (p3).
- **Negative controls may be trivial:** they use "identical prompts", so trajectories read before generation would be identical by construction (our inference; no numbers are reported, p13).
- **Inconsistent use of Bonferroni:** p = 0.013 is called "a moderate effect" although α = 0.00089 (p6). The 56-test family assumes 8 models, but Table 3 has 5.
- **Inconsistent numbers:** Limitations (66.7/77.0/91.7) vs Table 5 (66.6/–/95.9). Table 2 has 8 features vs "7" in the text. The 4-way classification is claimed but never quantified.
- **Visual pitfall:** one-pair PCA plots (Fig 2/3) show separation where the aggregate is null. Fig 3 looks duplicated from Fig 2.
- **Small n:** layer selection uses 15 pairs (admitted, p10) and cells have 30 pairs.

## Limitations and future work (quoted)
- Limitations (p10): "The current methodology provides correlational rather than causal evidence;"
- Future work (p10): "causal intervention experiments manipulating trajectory properties at specific layers would determine whether trajectory signatures drive behavioral outcomes or emerge from upstream latent factors."

## Bearing on our fates
- **Override:** weak. It shows user pressure is registered and decodable, not that pressure beats the correct answer. Our same-pressure flipped-vs-resisted split is the missing test.
- **Readout/threshold failure:** weak, but the near-tie criterion is a ready-made screen.
- **Bypass:** weak. DeepSeek-67B is "probe-detectable but geometrically weak", which fits "decodable but not used". There is no causal test.
- **Transformed:** weak. Confabulation changes shape rather than length, so different failure types touch different geometric properties.
- **Erased:** not addressed.
- **H3:** moderate (indirect). Polarity flips across models and scales, so a geometric monitor calibrated on one model or failure type will not transfer.

## Bearing on the group-geometry angle
1. Benchmark group features against a single direction and the logit margin. Here, shape scalars lost to one direction (66.6% vs 90.3%).
2. Split by behaviour under the same prompt, matched on prior confidence. Contrasting different prompts makes the geometry track the input.
3. Never use single-item trajectory pictures as evidence.
4. n = 30 is too noisy. RSA or CKA across pressures (H1) needs hundreds of items and bootstrap CIs.
5. The last prompt token misses the fork. Use decision points, multi-position capture and DTW instead.

## Where in the architecture
There is no consistent location. The best probe depth runs from 5.1% to 62% of depth across models, and polarity differs within the Llama-2 family and between same-size Llama-2-70B and Llama-3-70B (training, tokenizer and data are confounded). Only block outputs are read; no heads or MLPs are localised. There are no seeds or checkpoints. **Lesson:** search depth per model, and do not assume a fixed relative depth.

## Actionable fix
Monitoring only, calibrated per model with about 60 pairs, and weaker than a single direction. No fix is implied. For us: use geometry to find the layer and token, then test causally.

## Relevance
Moderate to low. It covers sycophancy at 7B-70B, but it uses condition labels, reads states before any answer, and runs no causal tests, so it says nothing about the fate of the correct answer. The most useful pieces are the baseline ladder (one direction beats the trajectory scalars) and the cross-model polarity flips (a warning for claims that a finding holds across models).
