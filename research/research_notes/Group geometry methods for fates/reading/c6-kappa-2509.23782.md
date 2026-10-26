# Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions (KAPPA; Park, Pyun, Jo, Seoul National University; ICML 2026; arXiv 2509.23782v4)

Slug: `c6-kappa-2509.23782`. Source: local `papers/c6-kappa-2509.23782/paper.txt` and page images only.

## In short

Two linear probes on the same residual state (one for the correct answer, one for the model's own choice) disagree on MCQs, most on TruthfulQA, BBQ and reasoning tasks. Their weight spans are moderately separated mid-depth, then jump to near-orthogonal within about one layer, where the model's choice crystallizes. KAPPA writes the knowledge coordinates into the prediction subspace (a closed-form minimal-norm edit) and lifts Llama-8B TruthfulQA from 56.7 to 73.5 (probe ceiling 76.3), with much smaller gains for Qwen. For us it is the ready-made subspace toolkit (angles, CKA, random controls, W_U comparison, injection test) for 'not routed'. Caveats: late-layer CKA is inflated by anisotropy, AGR mixes probe errors into the gap, and KAPPA hurts when the probe is worse than the model (Qwen3-14B GSM8k).

**Relevance.** Highly relevant, and already on our known-relevant list. It is the setup behind idea 4 (per-item knows-vs-acts), and it gives the subspace-geometry toolkit we want for the group angle, at 7-14B with a 32B check. It is less directly about our question than it looks: its 'gap' is mostly MCQ-format and abstention failure, measured per benchmark, not per item under social pressure. Its causal evidence (KAPPA) shows controllability of the prediction subspace, not how the model naturally routes information. Its most reusable finding for us is the abrupt depth jump in knowledge-prediction angle at the layer where the decision crystallizes, together with the warning that late-layer CKA and angle values need per-layer random calibration.

**Reuse verdict.** adapt: copy the paired knowledge and prediction probes at the answer position, the layer-wise principal angle and orthogonal CKA with random controls, the W_U comparison and the KAPPA injection, using the public repo. Adapt them by working per item on flipped vs resisted items, adding probe-vs-probe ceilings and per-layer random controls, replacing AGR with a 'probe right and model wrong' item filter, and applying it to user pushback and competing-option prompts rather than MCQ format failures.

## Setup

- **data**: 11 MCQ benchmarks. Reasoning: GSM8k (the MC version from Li et al. 2024), BBH-Algorithmic, BBH-NLP. Knowledge: MMLU Humanities, MMLU Social Sciences, MMLU STEM, ARC-Challenge, PubMedQA. Truthfulness and bias: TruthfulQA with 4 sampled choices, BBQ-Age, BBQ-Religion. Each question appears under several option orders to neutralize position bias. The prompt format is picked from 32 combinations (4 instructions x 2 answer formats x 4 option-symbol sets) to give the most balanced predictions on validation. Split sizes are in Table 8 (for example TruthfulQA 2208/1104/2208, BBQ-Age 8832/4416/8832). The paper does not say whether splits are disjoint at the question level.
- **models**: Main: Llama 3.1 8B Instruct and Qwen2.5 7B Instruct. Extra (TruthfulQA and GSM8k only): Mistral 7B Instruct v0.3, Qwen3 4B and Qwen3 14B. Qwen3 32B appears for TruthfulQA only (Table 13). All are instruct variants.
- **positions layers**: Residual stream at the final token of the answer prefix (for example 'The correct answer is (') at every layer. KAPPA is applied 'at each decoding step from the last token of the prompt to all subsequently generated tokens'. KAPPA layers: among layers whose prediction probe exceeds 85% validation accuracy, take the top-n consecutive layers by knowledge-probe validation accuracy (n = 1, 3, 6). In the figures these are late layers, roughly L17-29 of 32 for Llama and L19-27 of 28 for Qwen (approx, read from Figures 8-9).
- **probes**: Two k-class linear probes per layer. The knowledge probe is trained on ground-truth labels; the prediction probe is trained on the model's greedy-decoded option. Batch 128, learning rate 2e-4, best validation checkpoint over 1000 epochs. The subspace is the span of the mean-centered probe weight rows, rank r <= k-1.
- **metrics**: ACC; AGR (whether argmax of the knowledge probe equals argmax of the model); KLD = KL(p_model || p_knowledge-probe); mean principal angle between subspaces; orthogonal linear CKA of activations projected onto each subspace.
- **baselines**: Base model. CAA: difference-in-means steering vector between correct-option and incorrect-option continuations, best single layer on validation, fixed multiplier 1.0. DoLA: dynamic, best layer bucket on validation, alpha = 0. KP: the knowledge probe's own accuracy, used as a ceiling.
- **controls**: Matched random-subspace baselines (principal angle about 88.7 deg over 100 draws; CKA <= 0.2 over 32 draws). Comparison of the prediction subspace with the W_U subspace of the option tokens. Distractor resampling over 10 seeds with one-sided t-tests. Probe training-data subsampling (10-80%, 3 seeds). Cross-dataset transfer. Correct-to-incorrect flips reported alongside gains (Figure 14).
- **KAPPA hyperparameters**: alpha = 30, beta = 0 for 1-layer interventions; alpha = 10, beta = 0 for multi-layer. Table 13 also reports alpha = 80.

## Key results (quoted)

- Main claim: models fail MCQs while the correct answer is linearly decodable.  
  > "LLMs often fail on multiple-choice questions (MCQs) even if they encode correct answers in their hidden representations, revealing a misalignment between internal knowledge and output behavior." (p1)
- Probe position and design: residual stream at the last token of the answer prefix, every layer.  
  > "Concretely, at each layer l, we first extract the residual stream activation hl (x) ∈ Rd at the final token of the answer" (p3)
- The prediction probe is trained on the model's own greedy choice.  
  > "pairing activations with the model’s greedy-decoded predictions ỹ ∈ {1, . . . , k}. We then train a k-class linear classifier on each dataset" (p3)
- The knowledge advantage appears after the middle layers.  
  > "reveals that this advantage emerges after the middle layers, where knowledge probe accuracy increases sharply." (p4)
- The model's own output is even more decodable than the correct answer.  
  > "Prediction probes achieve higher accuracy than knowledge probes, often exceeding 90%, indicating that the model’s eventual output is also strongly represented in hidden states." (p4)
- BBQ example of a plausible override (hasty answering instead of 'undetermined').  
  > "Llama 3.1 8B Instruct chooses this option in only 13–27% of cases (accuracy: 59.9–67.6%), while knowledge probes achieve 38–51% (accuracy: 78.1–89.1%)." (p4)
- Hypothesized mechanism for truthfulness and bias: override by a preference for definite answers. This is not tested mechanistically.  
  > "we hypothesize that models override their encoded knowledge with a preference for hasty answering, providing definite answers even when abstention is appropriate" (p4)
- Subspace angles rise to near-random in late layers.  
  > "Across layers, the mean principal angle rises from 60◦ –80◦ in the middle layers to nearly 90◦ in the later layers, approaching the random baseline" (p5)
- Interpretation: both signals coexist but are routed along different directions.  
  > "these reveal that the knowledge and prediction signals coexist in the same residual stream while being routed along geometrically distinct directions." (p5)
- At the benchmark level, the angle tracks the gap (n = 8 benchmarks).  
  > "Llama 3.1 8B (ρ = 0.976, p = 0.001), and Qwen 2.5 7B shows a similar positive trend (ρ = 0.619, p = 0.106" (p6)
- KAPPA is a closed-form, minimal-norm edit confined to the prediction subspace.  
  > "This update minimally modifies the hidden state within the prediction subspace, leaving orthogonal components unchanged, while aligning its prediction coordinates with the corresponding knowledge coordinates." (p6)
- KAPPA is input-specific, unlike fixed steering vectors.  
  > "Unlike typical steering methods that apply a fixed direction across inputs, KAPPA dynamically computes an input-specific minimal adjustment based on the knowledge and prediction coordinates." (p6)
- KAPPA raises agreement by up to 22.2 points.  
  > "KAPPA consistently improves over base models and prior inference-time methods by significantly increasing agreement between the knowledge probe and the model’s final predictions (up to 22.2% points over base)." (p7)
- Llama accuracy gains are large on TruthfulQA and BBQ.  
  > "for LLaMA 3.1 8B, KAPPA achieves improvements of 8.9–16.9 points on TruthfulQA and BBQ, while delivering consistent gains of 1.5–5.0 points in reasoning benchmarks." (p7)
- The gap narrows with scale but persists.  
  > "Although larger-scale models tend to exhibit a relatively narrower gap compared to smaller counterparts (AGR: 76% for Qwen3 14B vs. 60% for Qwen3 4B), the misalignment remains substantial." (p7)
- KAPPA is not a direct token-logit swap: the prediction subspace stays at 65-70 deg from W_U at the KAPPA layers.  
  > "KAPPA intervention layers, with mean principal angles around 65◦ –70◦ . This indicates that KAPPA does not primarily operate by directly modifying the logits of the answer-symbol tokens." (p8)
- Subspaces are mostly task-specific.  
  > "many cross-dataset transfers show limited transferability, indicating that different datasets and tasks have geometrically distinct knowledge and prediction subspaces." (p8)
- Small transfer to free-form generation.  
  > "applying KAPPA improves the model accuracy from 41.7% to 44.2%." (p9)
- At the decision layers, knowledge and prediction spans are about as misaligned as random subspaces.  
  > "the directions encoding which option is correct become nearly as misaligned with the directions encoding which option will be selected as two random subspaces." (p20)
- The gap persists at 32B.  
  > "a non-trivial gap between base prediction and knowledge probe accuracy remains even for Qwen3-32B (9.6%)." (p28)
- Within one family, knowledge decodability differs less across scale than behavior does.  
  > "Comparing Qwen3-4B and Qwen3-32B, the difference in knowledge probe accuracy is relatively modest (∼10% points), whereas the difference in base prediction accuracy is much larger (∼23% points)." (p28)
- Works in low-data regimes (small margins for Qwen).  
  > "KAPPA consistently improves over the base model even when trained with only 10% of the data" (p29)

## Figures and tables

Numbers marked approx were read off plots; others come from printed tables or labels.

### Figure 1 (p2)

- **Plotted**: Schematic. (a) A sunset question answered correctly in free form ('summer') but wrongly as an MCQ ('(A) spring'). (b) A residual hidden state h at layer l is read by a knowledge probe and a prediction probe, giving two coordinate systems; KAPPA moves h so its prediction coordinates match its knowledge coordinates, and the answer becomes (C).
- **Takeaway**: This defines the framing: two linear readouts of the same hidden state can disagree, and KAPPA rewrites the action coordinates to match the decodable correct answer.
- **Numbers**: None.
- **Text matches figure?** yes

### Figure 2 (p3)

- **Plotted**: Llama 3.1 8B Instruct on TruthfulQA: layer-wise accuracy of the knowledge probe (blue) and the prediction probe (orange), with base accuracy dashed at 0.57.
- **Takeaway**: The correct answer becomes linearly decodable well above the model's own accuracy from mid-depth on, and the prediction probe becomes near-perfect a few layers later.
- **Numbers**: Approx, read from figure. Knowledge probe: about 0.26 at L0, rising to a peak of about 0.78 at L13-14, then about 0.75-0.77 to L31. Prediction probe: about 0.28 at L0, crossing the knowledge probe at about L13, about 0.85 at L14-16, about 0.97 from L17 and about 0.99 from L22. Base 0.57 (legend).
- **Text matches figure?** partly: the caption says the knowledge probe 'increases steadily and saturates at higher layers'. In the figure it peaks around L13-14 and then plateaus or dips slightly. The sharp prediction-probe transition at about L17 is accurately described.

### Table 1 (p4)

- **Plotted**: Per benchmark and model: base ACC, knowledge-probe gain over base (Delta ACC), AGR and KLD.
- **Takeaway**: The gap is large on TruthfulQA and BBQ-Age, moderate on reasoning, and small on knowledge benchmarks, where the probe can even be below the model.
- **Numbers**: From table. TruthfulQA: Qwen 58.8 / +21.3 / 61.8 / 1.01 and Llama 56.7 / +19.6 / 62.1 / 0.63. BBQ-Age: Qwen 83.2 / +9.3 and Llama 59.9 / +29.3 / 59.2 / 0.59. GSM8k: Qwen 47.8 / +2.5 / 60.3 and Llama 32.6 / +4.1 / 53.7. Negative Delta ACC: Llama MMLU Social Sciences -0.6, Qwen PubMedQA -0.3.
- **Text matches figure?** yes. A caveat the text does not draw out: on GSM8k, AGR is low (53.7) while the probe gains only +4.1, so most of the 'gap' there is the probe being wrong too, not unused knowledge.

### Figure 3 (p5)

- **Plotted**: Qwen2.5 7B Instruct, per item: knowledge-probe logit (x) against prediction-probe logit (y) for option A, at one layer per benchmark (GSM8k L27, TruthfulQA L19, MMLU-STEM L23, ARC L24), with Pearson r.
- **Takeaway**: On small-gap benchmarks the two coordinates lie on a tight diagonal. On large-gap benchmarks the cloud is diffuse (TruthfulQA) or looks like two groups (GSM8k: a dense low/low blob plus a vertical band where the knowledge logit is high but the prediction logit spans its whole range).
- **Numbers**: As printed: r = 0.6790 (GSM8k), 0.6118 (TruthfulQA), 0.9656 (MMLU-STEM), 0.9859 (ARC-Challenge). Approx, read from figure: GSM8k prediction logits span about -180 to +150 while knowledge logits span about -40 to +40.
- **Text matches figure?** yes

### Table 2 (p7)

- **Plotted**: ACC, AGR and KLD for Base, CAA, DoLA, KAPPA(1), KAPPA(3), KAPPA(6) and the KP ceiling on 6 large-gap benchmarks, for Qwen2.5 7B and Llama 3.1 8B.
- **Takeaway**: KAPPA raises agreement and accuracy, much more for Llama than for Qwen. Qwen stays far below the probe ceiling on TruthfulQA.
- **Numbers**: From table. Llama TruthfulQA ACC 56.7 to 73.5 with KAPPA(6) (KP 76.3; CAA 62.3). Llama BBQ-Age 59.9 to 76.8 (KP 89.1). Llama GSM8k ACC 32.6 to 36.6 and AGR 53.7 to 75.9 (KP 36.6). Llama BBH-Algo AGR 62.1 to 83.1 with KAPPA(1). Qwen TruthfulQA 58.8 to 64.1 (KP 80.1). Qwen BBQ-Religion 79.6 to 80.5 (KP 80.3).
- **Text matches figure?** partly: 'up to 22.2% points' matches Llama GSM8k AGR (75.9 - 53.7), and '8.9-16.9 points on TruthfulQA and BBQ' matches the Llama rows. However, 'gains of 1.5-5.0 points in reasoning' does not match the Llama rows, which give 3.4-5.0, and the smallest reasoning gain in either model is +1.8 (Qwen GSM8k).

### Figure 4 (p7)

- **Plotted**: TruthfulQA: improvement of AGR over base (percentage points) as alpha goes from 0 to 20 (six values), for beta = 0, 2, 4, 6. Qwen2.5 7B on the left, Llama 3.1 8B on the right.
- **Takeaway**: Both knobs increase agreement. For Llama, beta >= 2 saturates at about +21 points. For Qwen the whole sweep moves AGR by only about 2 points.
- **Numbers**: Approx, read from figure. Qwen: beta=0 from 0 to about 1.6 at alpha=20; beta=6 from about 0.6 to about 2.2. Llama: beta=0 from 0 to about 12.5; beta=2 from about 14.5 to about 20.8; beta=4 and beta=6 about 20.5-21.5 throughout.
- **Text matches figure?** yes. Not discussed in the text: the effect is about 10 times smaller for Qwen.

### Table 3 (p8)

- **Plotted**: TruthfulQA ACC, AGR and KLD for 5 models (Mistral v0.3 7B, Llama-3.1 8B, Qwen 2.5 7B, Qwen3 4B, Qwen3 14B) under Base, CAA, DoLA, KAPPA(1/3/6) and KP.
- **Takeaway**: The gap is present in all five. KAPPA helps every model, but CAA beats KAPPA(1) in ACC on Mistral.
- **Numbers**: From table. Mistral 40.7 to 58.3 with KAPPA(6) (CAA 52.8, KAPPA(1) 51.0, KP 69.5). Qwen3 4B 56.5 to 61.4 (KP 78.7). Qwen3 14B 71.6 to 77.7 (KP 85.8). AGR base: Qwen3 14B 76.0, Qwen3 4B 60.0.
- **Text matches figure?** yes

### Figure 5 (p8)

- **Plotted**: Llama 3.1 8B: change in AGR over base when probes and subspaces trained on the row dataset are applied to the column dataset (6 x 6 heatmap).
- **Takeaway**: Subspaces are mostly task-specific. Off-diagonal transfer is often harmful and only occasionally helpful within a category.
- **Numbers**: From figure cell labels. Diagonal: GSM8k 20.15, BBH-Algo 21.00, BBH-NLP 12.43, TruthfulQA 16.67, BBQ-Age 15.83, BBQ-Religion 14.51; these equal the KAPPA(1) AGR gains in Table 2. Off-diagonal: BBH-NLP to GSM8k +4.81; TruthfulQA to BBQ-Age +5.72; TruthfulQA to BBH-Algo -12.02; BBQ-Religion to BBH-Algo -11.88; GSM8k to BBH-Algo -8.25.
- **Text matches figure?** yes

### Table 4 (p9)

- **Plotted**: Free-form generation accuracy of Qwen2.5 7B under Base and KAPPA(1), using MCQ-trained subspaces, judged by GPT-4o-mini.
- **Takeaway**: Transfer to free-form is small: +2.5 on TruthfulQA, flat on BBQ-Age, -0.9 on GSM8k. The base model also solves GSM8k free-form far better (91.6) than as an MCQ (47.8 in Table 1), so the MCQ format itself causes much of the failure.
- **Numbers**: From table: TruthfulQA 41.7 to 44.2; BBQ-Age 89.7 to 89.9; GSM8k 91.6 to 90.7.
- **Text matches figure?** yes

### Figure 6 (p9)

- **Plotted**: For subspaces trained on GSM8k, BBH-Algorithmic or BBH-NLP: the share (%) of items that flip from incorrect to correct, broken down by BBH subtask. Top panel: BBH-Algorithmic subtasks. Bottom panel: BBH-NLP subtasks.
- **Takeaway**: Gains land on subtasks that need similar skills (logic-trained subspaces help logic-like NLP subtasks). These are shares of gains only, not net effects.
- **Numbers**: Approx, read from figure. BBH-NLP panel: GSM8k-trained gains about 21% Colored Objects and about 22% Date Understanding; BBH-NLP-trained gains about 25% Movie Recommendation and about 23% Ruin Names. BBH-Algo panel: Temporal Sequences gets about 2% of GSM8k-trained gains.
- **Text matches figure?** yes. The model is not named in the caption; Figure 14 says Llama-3.1 8B.

### Table 5 (p15)

- **Plotted**: The four task-instruction templates.
- **Takeaway**: This is part of the 32-way prompt-format search used to pick the most balanced format.
- **Numbers**: 4 instructions (from table).
- **Text matches figure?** yes

### Table 6 (p15)

- **Plotted**: Two answer formats: 'The correct answer is (' and 'Answer: ('.
- **Takeaway**: The probe position is the last token of this answer prefix.
- **Numbers**: 2 formats (from table).
- **Text matches figure?** yes

### Table 7 (p15)

- **Plotted**: Four option-symbol sets: (A)-(D), (i)-(iv), (1)-(4), and random letters (Q),(K),(E),(U).
- **Takeaway**: Symbols vary, which matters for the claim that KAPPA is not a token-logit swap.
- **Numbers**: 4 symbol sets (from table).
- **Text matches figure?** yes

### Table 8 (p16)

- **Plotted**: k and train/validation/test sizes per dataset.
- **Takeaway**: Sizes count option-order copies. BBQ-Age's total (22080) equals 3680 x 6 option orders, which fits question-level splits, but disjointness is never stated.
- **Numbers**: From table: GSM8k k=4, 8960/2992/10552; BBH-Algorithmic 4800/2400/4800; BBH-NLP 4464/2232/4472; MMLU Humanities 15000/7536/15104; MMLU Social Sciences 9800/4936/9880; MMLU STEM 10040/5064/10120; ARC k=3, 6714/1794/7032; PubMedQA k=3, 2400/600/3000; TruthfulQA k=4, 2208/1104/2208; BBQ-Age k=3, 8832/4416/8832; BBQ-Religion k=3, 2880/1440/2880.
- **Text matches figure?** yes

### Table 9 (p16)

- **Plotted**: BBH subtasks used. Algorithmic: geometric shapes, logical deduction, temporal sequences, tracking shuffled objects, reasoning about colored objects. NLP: date understanding, penguins in a table, movie recommendation, ruin names, salient translation error detection.
- **Takeaway**: Only subtasks with 4 or more options are kept.
- **Numbers**: 5 + 5 subtasks (from table).
- **Text matches figure?** yes

### Table 10 (p17)

- **Plotted**: MMLU subtasks per category (Humanities 13, Social Sciences 11, STEM 18).
- **Takeaway**: Bookkeeping only.
- **Numbers**: From table (counted): 13 / 11 / 18 subtasks.
- **Text matches figure?** yes

### Figure 7 (p18)

- **Plotted**: Llama 3.1 8B: layer-wise knowledge-probe (blue) and prediction-probe (orange) accuracy on GSM8k-MC, BBH-NLP, BBH-Algorithmic, TruthfulQA, BBQ-Religion and BBQ-Age, with base accuracy dashed.
- **Takeaway**: On every benchmark the prediction probe saturates near 0.96-0.99 by L17-18, while the knowledge probe plateaus at a modest level above base. The knowledge probe stays high in late layers, so the correct answer is not erased at the answer position.
- **Numbers**: Approx, read from figure. GSM8k-MC: knowledge probe plateau about 0.36-0.37 from L13 against base 0.33. BBH-NLP: about 0.63-0.64 against 0.60. BBH-Algo: about 0.50-0.51 against 0.45. TruthfulQA: as in Figure 2. BBQ-Religion: jump to about 0.81 at L13, then about 0.75-0.77, against 0.68. BBQ-Age: knowledge probe about 0.5 from L3 and about 0.82-0.85 after L18; prediction probe about 0.15 until about L10. The BBQ-Age panel has no y-tick labels.
- **Text matches figure?** partly: 'consistently exceeds baseline' holds only from mid-depth on; in early layers the knowledge probe is below base in several panels. In the BBQ-Age panel the dashed line labelled 'base acc = 0.60' sits at roughly the 0.43 height of the neighbouring panel, and an early prediction probe at about 0.15 would be below the 3-way chance level if the axis ran 0-1. The panel's axis is unclear.

### Figure 8 (p21)

- **Plotted**: Six panels (Llama and Qwen x BBH-Algo, GSM8k, TruthfulQA), each showing layer-wise mean principal angle between the knowledge and prediction subspaces (blue) and between the prediction subspace and the option-token W_U subspace (green), with the random baseline dashed at about 88.7 deg. Bands mark the KAPPA layers.
- **Takeaway**: The knowledge-prediction angle is moderate in early and middle layers and then jumps within about one layer to near-random (84-88 deg), at the same depth where the prediction probe saturates. This is a depth 'fork point'. The prediction subspace swings toward W_U (about 66-75 deg) only after that jump. KAPPA layers are all after the jump.
- **Numbers**: Approx, read from figure. Llama BBH-Algo: about 60-78 deg in L0-16, about 84 at L17, about 87-88 from L22. Llama TruthfulQA: about 50-66 in L0-15 (minimum about 50 near L8-10), about 87 from L17. Qwen TruthfulQA: about 50-60 mid-depth, about 87 from L19. Qwen GSM8k: about 56-65, then about 82-87 from L19. Llama GSM8k: noisy at 65-88. Prediction vs W_U: about 88 early, about 66-75 at the KAPPA layers. KAPPA top-6 bands: Llama about L17-22 (BBH-Algo), L23-28 (GSM8k), L24-29 (TruthfulQA); Qwen about L19-27.
- **Text matches figure?** partly: near-orthogonality in late layers matches. But '60-80 deg in middle layers' understates TruthfulQA, where angles drop to about 50 deg. The rise is an abrupt one-layer jump rather than a gradual climb. And Section 5.1 calls the KAPPA layers 'intermediate', while the figure and Appendix B.4 show late layers.

### Figure 9 (p22)

- **Plotted**: Same six panels: orthogonal linear CKA between knowledge-projected and prediction-projected activations (blue), plus two controls, knowledge vs random subspace (green) and prediction vs random subspace (purple), with KAPPA layer bands.
- **Takeaway**: In late layers the random-subspace control for the prediction projection rises to about 0.3-0.5. On TruthfulQA it equals or exceeds the knowledge-prediction CKA. Much of the intermediate late-layer CKA can therefore come from residual-stream anisotropy rather than shared structure.
- **Numbers**: Approx, read from figure, at the KAPPA layers. Knowledge vs prediction: Llama BBH-Algo about 0.53-0.58; Qwen BBH-Algo about 0.55-0.6; Llama GSM8k about 0.5-0.6; Qwen GSM8k about 0.55-0.8; Llama TruthfulQA about 0.40-0.43; Qwen TruthfulQA about 0.31-0.41. Prediction vs random: Llama TruthfulQA about 0.42-0.47; Qwen TruthfulQA about 0.28-0.48; Llama BBH-Algo about 0.3-0.37; Qwen GSM8k about 0.37-0.5. Knowledge vs random: about 0.1-0.25.
- **Text matches figure?** no for TruthfulQA, partly elsewhere: Appendix B.4 says CKA is 'clearly above the random control (<= 0.2)'. That holds against the knowledge-vs-random control, but the prediction-vs-random control reaches about 0.4-0.48 in late layers, matching or beating the knowledge-prediction CKA on TruthfulQA in both models.

### Figure 10 (p24)

- **Plotted**: One point per benchmark (8): mean knowledge-prediction principal angle at the best knowledge-probe layer (x) against the gap 1 - AGR (y), with a regression line and CI band. Llama on top, Qwen below.
- **Takeaway**: At the benchmark level, a larger angle goes with a larger gap for Llama. For Qwen the trend is positive but not significant, and most large-gap points bunch at 86-88 deg, close to the 88.7 deg random baseline.
- **Numbers**: As printed: Llama rho = 0.976 (p = 0.001); Qwen rho = 0.619 (p = 0.106). Approx, read from figure. Llama: ARC (about 44 deg, 0.02), MMLU-STEM (about 62, 0.14), BBQ-Religion (about 77, 0.21), BBH-NLP (about 85, 0.26), BBQ-Age/BBH-Algo/TruthfulQA (about 86-88, 0.38-0.42), GSM8k (about 88, 0.48). Qwen: ARC (72, 0.02), BBQ-Religion (about 81, 0.02), MMLU-STEM (about 82, 0.10), BBH-NLP/TruthfulQA/BBH-Algo/GSM8k (about 87-88, 0.30-0.40), BBQ-Age (about 88, 0.14).
- **Text matches figure?** yes

### Table 11 (p27)

- **Plotted**: KAPPA on the knowledge benchmarks (MMLU Humanities, Social Sciences, STEM; PubMedQA) for Qwen and Llama.
- **Takeaway**: The changes are small and sometimes harmful.
- **Numbers**: From table. Llama MMLU STEM AGR 91.12 to 90.56 / 86.30 / 84.58 for KAPPA 1/3/6. Llama MMLU Social Sciences ACC 74.04 to 70.49 and AGR 90.55 to 85.98 with KAPPA(6). Qwen MMLU Humanities ACC 59.91 to 61.92 (KP 62.47).
- **Text matches figure?** partly: Appendix E.1 says KAPPA 'consistently improves agreement (AGR)', but Llama MMLU STEM AGR falls in all three KAPPA settings.

### Table 12 (p28)

- **Plotted**: GSM8k ACC, AGR and KLD for 5 models under all methods.
- **Takeaway**: AGR always rises. But when the knowledge probe is worse than the model (Qwen3 14B, and marginally Qwen3 4B), aligning with it lowers accuracy.
- **Numbers**: From table. Qwen3 14B: base 63.5, KAPPA(1) 53.0, KAPPA(3) 61.3, KAPPA(6) 60.8, KP 56.6, CAA 64.3. Qwen3 4B: KP 46.7 against base 47.1. Mistral: AGR 58.3 to 83.8 with KAPPA(3) while ACC goes only 31.7 to 33.6 (KP 34.0).
- **Text matches figure?** partly: the text's 'consistently increases AGR' is true, but it leaves out the 10.5-point accuracy loss on Qwen3 14B.

### Table 13 (p28)

- **Plotted**: TruthfulQA across 6 models with KAPPA(1) at alpha = 30 and alpha = 80.
- **Takeaway**: The gap persists up to 32B, and the larger alpha helps everywhere in ACC.
- **Numbers**: From table. Qwen3 32B: 79.5 to 84.0 (alpha = 80), KP 89.1. Mistral: 40.7 to 58.7 (KP 69.5). Llama: 56.7 to 70.3 (KP 76.3). Qwen2.5 7B: 58.8 to 64.2 (KP 80.1). Qwen3 4B: 56.5 to 61.4 (KP 78.7). Qwen3 14B: 71.6 to 75.5 (KP 85.8).
- **Text matches figure?** yes: 9.6 (32B), 28.8 (Mistral) and 19.6 (Llama) all follow from the table.

### Table 14 (p29)

- **Plotted**: Mean and SD of accuracy over 10 distractor-resampled test sets (BBH-algo, TruthfulQA).
- **Takeaway**: The gains are stable across distractors, except Qwen BBH-algo KAPPA(1), which is about +0.2.
- **Numbers**: From table. Qwen BBH-algo: 51.67 (1.281), 51.90 (1.428), 53.85 (1.850). Qwen TruthfulQA: 59.93 to 61.74 / 64.71. Llama BBH-algo: 44.55 to 48.95 / 49.20. Llama TruthfulQA: 56.38 (1.530) to 67.36 (2.053) / 73.73 (2.529).
- **Text matches figure?** yes

### Table 15 (p29)

- **Plotted**: One-sided t-test p-values of KAPPA against base over the 10 distractor seeds.
- **Takeaway**: 7 of 8 settings are significant; Qwen BBH-algo KAPPA(1) is not.
- **Numbers**: From table: Qwen BBH-algo KAPPA(1) p = 0.105; all others < 0.0001 or smaller.
- **Text matches figure?** yes

### Figure 11 (p30)

- **Plotted**: KAPPA(1) accuracy against probe training-data fraction (base, 10-100%) on TruthfulQA, BBH-Algo and GSM8k for Llama (red) and Qwen (blue).
- **Takeaway**: KAPPA beats base even at 10% of the data, but for Qwen the gains are tiny at every fraction.
- **Numbers**: Approx, read from figure; the 100% values are printed. Llama TruthfulQA: 56.7 base, about 58.3 at 10%, 67.75 at 100%. Qwen TruthfulQA: 58.8, about 59.8, 60.64. Llama BBH-Algo: 45.1, about 46.2, 49.29. Qwen BBH-Algo: 51.0, about 51.3, 51.53. Llama GSM8k: 32.6, about 33.0, 34.86. Qwen GSM8k: 47.8, about 48.3, 49.61.
- **Text matches figure?** yes

### Figure 12 (p30)

- **Plotted**: Knowledge-probe accuracy against training-data fraction (10-100%).
- **Takeaway**: The knowledge probe is data-hungry. At 10%, the TruthfulQA probe is barely above base accuracy.
- **Numbers**: Approx, read from figure; the 100% values are printed. TruthfulQA: Qwen about 65 to 80.07, Llama about 59 to 76.27. BBH-Algo: Qwen about 51 to 55.44, Llama about 46 to 50.69. GSM8k: Qwen about 47 to 50.33, Llama about 33 to 36.63.
- **Text matches figure?** yes

### Figure 13 (p30)

- **Plotted**: Prediction-probe accuracy against training-data fraction.
- **Takeaway**: The model's own choice is easy to decode even from little data.
- **Numbers**: Approx, read from figure; the 100% values are printed. TruthfulQA: Llama about 94.6 to 95.65, Qwen about 87 to 88.81. BBH-Algo: Qwen about 94.6 to 96.31, Llama about 90.6 to 93.42. GSM8k: Qwen about 96.1 to 97.81, Llama about 92.3 to 94.24.
- **Text matches figure?** yes: 'around or above 90%' fits; Qwen TruthfulQA is about 87-89.

### Figure 14 (p31)

- **Plotted**: Llama-3.1 8B, BBH subtasks: shares of incorrect-to-correct gains (up) and correct-to-incorrect losses (down) for subspaces trained on GSM8k, BBH-Algo or BBH-NLP.
- **Takeaway**: Transferred subspaces also break many items that were right. Because the bars are shares, net effects per subtask cannot be read off.
- **Numbers**: Approx, read from figure. Loss shares about -10% to -28%: for example Salient Translation about -25 / -22 / -28, Geometric Shapes about -24 / -20 / -23, Temporal Sequences GSM8k-trained gain about 2% against loss about -19%.
- **Text matches figure?** yes

## Methods we could use

### Paired linear probes at the same position: a knowledge probe (ground truth) and a prediction probe (model's greedy option), at every layer

- **Measures**: Where in depth the correct answer becomes decodable, where the model's own choice becomes decodable, and per item whether the two readouts disagree (AGR, KLD).
- **Fate it could diagnose**: It sets up every fate: it defines the target items (knowledge probe right, model wrong). On its own it separates 'erased' (the knowledge probe drops in late layers; not seen here) from the others.
- **Causal follow-up**: KAPPA-style injection into the prediction subspace (tests 'not routed'). Projection or ablation of the knowledge subspace, which they do not do, would test whether that subspace is causally used (bypass vs mediated).
- **Data and compute**: One forward pass per item with all-layer activations at one token; k-class linear probes (they train up to 1000 epochs on about 2k-15k items). Cheap at 9B-27B for about 400-2000 items, but the knowledge probe needs many labeled items (Figure 12 shows it is data-hungry).
- **Code**: Yes: Repo: https://github.com/holi-lab/KAPPA (p1 footnote).

### Per-item scatter of knowledge-probe logit against prediction-probe logit for one option, with Pearson r

- **Measures**: Whether the two coordinates co-vary item by item, shown as a point cloud.
- **Fate it could diagnose**: Readout/threshold failure: points lie on the diagonal but are shifted. Override or not routed: a vertical spread where knowledge is high but prediction is anywhere. Transformed (sign flip): an anti-diagonal cluster. Discrete states vs manifold: GSM8k shows two clusters.
- **Causal follow-up**: Pick items in the off-diagonal region for patching; compare flipped vs resisted items matched on knowledge logit.
- **Data and compute**: Free once the probes exist.
- **Code**: Yes (repo).

### Mean principal angle between mean-centered probe-weight spans (rank <= k-1), layer by layer, against a matched random baseline

- **Measures**: The span-level (Grassmannian) separation between the knowledge and prediction subspaces, independent of data.
- **Fate it could diagnose**: Not routed or override (the subspaces diverge where the decision forms). It also localizes a depth 'fork point': a one-layer jump from about 50-67 deg to about 84-88 deg where the prediction probe saturates.
- **Causal follow-up**: Patch or inject at layers just before and just after the jump to test whether the knowledge signal is still causally usable before the fork.
- **Data and compute**: Negligible (an SVD of d x k matrices).
- **Code**: Yes (repo).

### Orthogonal linear CKA of activations projected onto the two spans, with knowledge-vs-random and prediction-vs-random controls

- **Measures**: Whether the two projections order and separate the same examples (example-level shared structure).
- **Fate it could diagnose**: Transformed vs erased in principle (shared structure that survives a change of basis). In practice the late-layer anisotropy inflates the prediction-vs-random CKA to about 0.45, so it needs per-layer random controls.
- **Causal follow-up**: Little directly. Use it as a descriptive check alongside Procrustes or RSA.
- **Data and compute**: Negligible.
- **Code**: Yes (repo).

### Angle between the prediction subspace and the W_U subspace of the option tokens

- **Measures**: How output-facing (logit-lens-like) the prediction subspace is at each depth.
- **Fate it could diagnose**: Readout/threshold failure (a late, near-W_U problem) vs upstream routing failure. It tells whether a fix acts on token logits or on more abstract features.
- **Causal follow-up**: Compare edits inside the W_U span with edits inside the prediction span.
- **Data and compute**: Negligible.
- **Code**: Yes (repo).

### KAPPA: closed-form minimal-norm edit h' = h + W_pred (W_pred^T W_pred)^-1 (alpha W~_know^T h~ + beta sign(.) - W~_pred^T h~), applied at 1-6 layers and every decoding step

- **Measures**: Whether writing the decodable correct-answer coordinates into the action subspace changes behavior. This is causal controllability of the prediction subspace.
- **Fate it could diagnose**: Not routed: injection works, and the action subspace is live and steerable. It rules out 'erased' at that layer for items it fixes. It does not show endogenous mediation, and needing alpha = 30 hints at a scale or threshold component.
- **Causal follow-up**: A Stage B fix and recheck. Compare a KAPPA-style alignment fix with plain DPO, then recheck whether the angle and CKA geometry and the fate mix change.
- **Data and compute**: Probes per task; inference-time cost of one small matrix product per layer per token.
- **Code**: Yes (repo).

### Cross-dataset subspace transfer matrix

- **Measures**: Whether knowledge and prediction subspaces fitted on one task work on another.
- **Fate it could diagnose**: H1 (pressures give different fate mixes): fit on pushback, apply to the tempting-option condition and vice versa. Poor transfer would mean pressure-specific geometry.
- **Causal follow-up**: The transfer test is itself causal (an intervention).
- **Data and compute**: Probes per condition; cheap.
- **Code**: Yes (repo).

### Benchmark-level Spearman correlation between angle and 1 - AGR

- **Measures**: Whether tasks with more misalignment have larger gaps.
- **Fate it could diagnose**: Weak: n = 8 benchmarks, and it is confounded by probe quality and layer choice.
- **Causal follow-up**: None.
- **Data and compute**: Negligible.
- **Code**: Yes (repo).

## Controls worth copying

- Matched-rank random-subspace baselines for principal angles and CKA, computed per layer. Report both the knowledge-vs-random and the prediction-vs-random controls, because the second exposes late-layer anisotropy.
- Comparison with the W_U subspace of the answer tokens, to rule out a plain token-logit swap.
- Selecting the prompt format for balanced option predictions and permuting option orders, to remove position bias before probing.
- Distractor resampling (10 seeds) with one-sided t-tests.
- Probe training-data subsampling (10-80%, 3 seeds) for both probes.
- Reporting correct-to-incorrect flips alongside gains for any intervention (Figure 14).
- Our additions, not in the paper: (a) the angle and CKA between two knowledge probes trained on disjoint halves or seeds, as a ceiling for 'aligned'; (b) condition fate analysis on items where the knowledge probe is right and the model wrong, instead of using AGR; (c) the logit margin as a probe-free baseline.

## Pitfalls

- Late-layer CKA is inflated by anisotropy. The prediction-vs-random control reaches about 0.4-0.48 (read from Figure 9) and equals or beats the knowledge-prediction CKA on TruthfulQA, contradicting the stated random level of about 0.2.  
  > "at the KAPPA intervention layers, the orthogonal CKA sits in an intermediate range (about 0.4–0.8), clearly above the random control (≲ 0.2)" (p20)
- Principal angles between low-rank probe spans have almost no headroom near 90 deg in about 4000-d space, so 'near-random' is the default and says little unless calibrated.  
  > "The baseline lies close to fully orthogonal because, in a ∼4000-dimensional residual stream, two low-rank (r ≤ k − 1) subspaces drawn at random are nearly orthogonal by default." (p19)
- Spans and correlations can disagree, so one metric alone misleads.  
  > "even when two subspaces are nearly orthogonal, their projections can remain correlated due to anisotropy in the residual stream." (p19)
- AGR and 1 - AGR mix 'unused knowledge' with 'the probe is wrong'. On GSM8k (Llama, KP 36.6 against base 32.6) most disagreement is probe error, so the angle-gap correlation may partly track probe quality. The paper concedes low probe accuracy limits gains on knowledge tasks.  
  > "This is expected, as the performance of the knowledge probe itself is comparatively low on these benchmarks, limiting the potential upper bound for accuracy improvements." (p27)
- When the probe is worse than the model, KAPPA hurts. On Qwen3 14B GSM8k, ACC falls from 63.5 to 53.0 with KAPPA(1) (KP 56.6), from Table 12. The text reports only that AGR rises.  
  > "Across all model families, KAPPA consistently increases AGR relative to the base model, demonstrating its effectiveness in mitigating the knowledge-prediction gap." (p27)
- The strength is large (alpha = 30 single-layer, up to 80), so KAPPA is strong steering toward the probe's argmax rather than a gentle 'alignment'. Baselines are not tuned equally: CAA uses a fixed multiplier of 1.0.  
  > "we use consistent settings across all configurations: α = 30, β = 0 for single-layer interventions and α = 10, β = 0 for multi-layer interventions." (p15)
- CAA baseline strength is fixed rather than swept, which favors KAPPA in the comparison.  
  > "with a fixed multiplier of 1.0 for the intervention, following Rimsky et al. (2024)." (p27)
- The main text calls the intervention layers 'intermediate', but the selected layers sit after the angle jump, in late layers (Figures 8-9, Appendix B.4).  
  > "We apply KAPPA primarily at intermediate layers, where hidden states encode rich internal knowledge while still exerting substantial influence on downstream predictions." (p6)
- Each question appears under several option orders, and question-level disjointness of the train/val/test splits is not stated, so the knowledge probe could memorize question content.  
  > "Each question is evaluated under multiple option orders. This procedure neutralizes positional biases while preserving exact class balance across the entire dataset." (p16)
- No test of endogenous mediation: the knowledge subspace is never ablated or patched, and KAPPA injects an external supervised probe's answer. The causal evidence is controllability of the prediction subspace only.  
  (not stated in the paper; see p6)
- The gap studied is mostly an MCQ-format and abstention failure (free-form GSM8k 91.6 against MCQ 47.8 for Qwen), not acting against knowledge under social pressure.  
  (not stated in the paper; see p9)

## Limitations and future work (authors' words)

- Limitations (p3): "While richer knowledge might be encoded in non-linear forms, we focus on linear accessibility because it represents the most interpretable and"
  - Note: The paper has no dedicated Limitations section. This sentence (Section 3) and the conclusion's list of extensions are the closest it comes.
- Future work (p9): "Several important extensions remain, including more reliably extracting knowledge from hidden states, aligning nonlinearly encoded knowledge, evaluating broader prompting settings, and extending to higher-dimensional interventions."

## Bearing on the six fates

| Fate | How | Strength |
|---|---|---|
| 3 not routed | The correct answer is decodable (knowledge probe), the action subspace is causally live (writing into it moves the output), and the two spans are near-orthogonal exactly where the decision forms. Together these fit 'not routed': injection works, but the model does not carry the knowledge there by itself. The caveat is that the injection uses an external supervised probe and the knowledge subspace is never ablated, so this is controllability, not proof of missing endogenous routing. | moderate |
| 4 override | The authors hypothesize that a 'hasty answering' preference overrides knowledge on BBQ (picking 'undetermined' only 13-27% of the time while the probe implies 38-51%). Not tested mechanistically; there is no competing-signal direction or head analysis. | weak |
| 1 readout/threshold failure | On small-gap tasks the per-item coordinates lie on a tight diagonal (r 0.97-0.99). KAPPA needs alpha = 30-80, and beta (a push toward extremes) alone gives about +21 AGR on Llama, which hints that part of the gap is amplitude or threshold. The paper does not separate threshold from routing. | moderate |
| 6 erased | Against erasure at the answer position: the knowledge probe stays on its plateau through the last layers on every benchmark (Figure 7). | moderate |
| 2 bypass | Not tested. Whether the knowledge subspace is causally inert would need ablation or patching of that subspace. | weak |
| 5 transformed | Not tested. The scatter (Figure 3) and CKA could reveal sign-flipped or rotated encodings, but there is no Procrustes or sign analysis, and near-orthogonal spans are equally consistent with re-encoding. | weak |

## Bearing on the group / subspace angle

This is the closest worked example of the subspace angle in our setting. It treats each probe's weight span as a subspace, tracks principal angles across depth against a random baseline, adds orthogonal CKA, compares with the W_U option subspace, and shows per-item point clouds. Three lessons for us. (1) The angle between knowledge and prediction spans does not drift: it jumps within about one layer from about 50-67 deg to about 84-88 deg, at the depth where the prediction probe saturates (read from Figures 2, 7 and 8). That is a depth 'fork point' to look for, and the token-level analogue is what we want for flipped vs resisted items. (2) Calibration is everything. Rank-(k-1) spans in about 4000-d are near-orthogonal by default, and late-layer anisotropy pushes random-subspace CKA to about 0.45, so every group metric needs per-layer random controls and a probe-vs-probe (seed or split) ceiling. (3) GSM8k's knowledge-vs-prediction cloud looks like two clusters, a hint of discrete states rather than a smooth manifold. Missing for us: per-item geometry (all angles are per benchmark), flipped vs resisted comparisons matched on confidence, token paths, attention heads, and Procrustes tests for 'transformed'.

## Where in the architecture

Residual stream at the last token of the answer prefix. The knowledge signal becomes decodable after the middle layers (about L10-14 of 32 for Llama, read from Figures 2 and 7). The prediction probe saturates at about L17 (Llama) and about L19 (Qwen, of 28). At that same depth the knowledge-prediction angle jumps to near-random, and the prediction subspace only then rotates toward W_U (about 66-75 deg at the KAPPA layers). These depth relations are consistent across Llama 3.1 8B and Qwen2.5 7B on three benchmarks, at about 55-65% relative depth for the jump (my arithmetic: 17/32 and 19/28). The gap itself is shown across Mistral 7B, Llama 8B, Qwen2.5 7B and Qwen3 4B/14B/32B. There is no component-level (attention head or MLP) analysis, no seeds, and no checkpoints.

## Actionable fix or monitor

Inference-time fix: KAPPA writes the decodable answer coordinates into the prediction subspace with an input-specific minimal-norm edit at a few late layers. It works when a supervised knowledge probe beats the model, needs labels per task (transfer is weak), and can hurt when the probe is worse than the model (Qwen3 14B GSM8k -10.5 ACC). Monitor: per-item KLD or disagreement between the knowledge probe and the model output as a cheap flag for 'acting against decodable knowledge'. For our Stage B, a KAPPA-style alignment is a natural fate-matched fix for 'not routed' items, to compare against plain DPO, with Figure 14-style loss accounting.

## Sections read

Abstract, 1 Introduction, 2 Related Work, 3 Measuring Knowledge-Prediction Gap in MCQs, 3.1 Linear Probing Analysis, 4 Geometric Interpretation of the Gap, 4.1 Visualizing Hidden State Activations, 4.2 Quantifying Subspace Misalignment, 4.3 Connecting the Knowledge-Prediction Gap to Subspace Misalignment, 5 Bridging the Knowledge-Prediction Gap, 5.1 Knowledge-Aligned Prediction Adjustment, 5.2 Experiments, 6 Analysis, 6.1 Comparing Prediction Subspace with Logit Space, 6.2 Transfer Across Datasets, 6.3 Transfer to Free-Form Generation, 7 Conclusion, Impact Statement, Acknowledgments, References, A Detailed Setup for the Knowledge-Prediction Gap, A.1 Evaluation Pipeline, A.2 Prompt Formats and Example, A.3 Data Statistics and Construction, A.3.1 Dataset Sizes and Subtasks, A.3.2 Data Construction Details, A.4 Additional Linear Probing Results, B Detailed Setup for the Geometric Analysis, B.1 Subspace Construction, B.2 Subspace Alignment Metrics, B.3 Random Baselines, B.4 Layer-wise Misalignment Results, B.5 Connecting the Knowledge-Prediction Gap to Subspace Misalignment, B.6 Comparing Prediction Subspace with Logit Space, C Derivation of KAPPA, C.1 Closed-form Update Derivation, C.2 Interpretations of the Update, C.3 Extended Alignment, D Baseline Details, E Additional Results, E.1 KAPPA on MCQ Benchmarks, E.2 Effect of Model Scale, E.3 Sensitivity Analysis, E.4 Extended Results for Cross-Dataset Transfer
