# Fate of knowledge under sycophancy and knowledge conflict: a full-text check of five papers against the compliant-context probe-aliasing artifact (2609.10739)

Scope note: these notes come from the arXiv full texts. HTML was used for 2609.10739, 2607.00415, 2508.02087, 2604.19117 and 2609.35822. 2602.04918 has no HTML version, so I extracted its PDF with pdftotext. KnownLieBench (2608.26372) was also read from its PDF. Where a fetch summarised a section rather than quoting it, I report only numbers that appeared explicitly. Items marked "Inference" are my own reasoning, not the papers' claims.

Reference artifact (2609.10739, "The Truth Was Never Gone: Perfect Aliasing in Compliant-Context Truth Probes"):
- §3: on compliant ("ally") trials, the true bit and the prescribed answer are identical, so a probe fit only there cannot tell which one it tracks. On "rival" trials they are complements, so the two AUROCs sum to one (Eq. 1). The paper calls this "perfect aliasing."
- Headline: a reward-trained Gemma-2-9B that answers falsely on every evaluated rival trial gives an ally-fit probe AUROC of 0.006 ± 0.005 at the final layer (three seeds). A mixed-fit probe scores 1.000 on the same held-out activations (§3–4, Table 2).
- The paper's own caveat: mixed fitting also uses about twice as many examples, plus labelled rival data. So it shows the true bit is linearly recoverable, not how much the label separation alone explains. — [2609.10739](https://arxiv.org/html/2609.10739)
- App. I: a single-orientation probe inverts to 0.000 under a rule flip, which the paper calls the signature of fitting a shortcut. App. J: because AUROC is antisymmetric under label negation, below-chance values mean systematic inversion, not absence of signal. App. G: final-layer readouts can invert while mid-stack decodability stays at 0.86–1.00. — [2609.10739](https://arxiv.org/html/2609.10739)
- Recommended remedies:
  - fit on mixed ally and rival data (App. H);
  - decorrelate the emitted token by randomising the codebook per trial, and fit the token as a separate label;
  - split rates by the true bit;
  - use grouped splits;
  - run positive and negative controls (fixed-symbol guard, single-orientation control, permutation nulls) (Apps. D, H–K).

  — [2609.10739](https://arxiv.org/html/2609.10739)
- In the portions I read, 2609.10739 does not cite any of the five target papers or KnownLieBench. The papers it does critique on interpretation are Natarajan et al. 2026, Taufeeque et al. 2026, Kumar 2026 and Gupta & Jenner 2025. The verdicts below are therefore my own application of its logic. — [2609.10739](https://arxiv.org/html/2609.10739)

## Q1. Which knowledge does each paper track: the answer token (logit/unembedding), an evaluative "this is wrong" signal, or a truth direction?

### Takeaway
Three papers track only the correct-answer token through the unembedding: 2607.00415 (logit lens plus answer-letter probes), 2508.02087 (logit lens decision score) and 2602.04918 (the correct token's unembedding vector). 2604.19117 tracks an evaluative "claim is wrong" signal: a mean-difference direction from wrong-opinion vs correct-opinion prompts, related to a Marks–Tegmark-style truth/lying direction. 2609.35822 tracks answer identity decoded from individual heads' OV outputs, plus logit-lens answer probabilities. None of the five fits a truth direction on pressured items and reads it per item.

### Cited Findings
- **2607.00415** (Joswin, Medicherla, Mammen, "A Mechanistic View of Authority Hierarchy in LLM Sycophancy"):
  - The logit lens tracks P(correct letter) and P(hinted letter) across layers (Sec. 4 RQ2, App. A.2).
  - Linear (LR) and MLP probes classify the correct answer letter as a 4-class problem with 5-fold out-of-fold evaluation (RQ3, App. A.3).
  - Knowledge here means the identity of the correct answer letter.

  — [2607.00415](https://arxiv.org/html/2607.00415)
- **2508.02087** (Wang, Li, Yang, Zhang, Wang, "When Truth Is Overridden"):
  - The logit lens projects the answer-position hidden state through the final norm and LM head onto options A–D (Eq. 2).
  - A "Decision Score" min-max normalises each option's logit across the four choices (Eq. 1, ε = 10⁻⁹). It is computed for the correct option and the user-endorsed option.
  - Layer-wise KL between the Plain and Opinion-only logit-lens distributions (Sec. 4.1).
  - No truth direction and no trained probe.

  — [2508.02087](https://arxiv.org/html/2508.02087)
- **2604.19117** (Pandey, "LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit"):
  - The sycophancy direction is the residual-stream mean difference at the last prompt token between wrong-opinion and correct-opinion TriviaQA prompts ("I believe the answer to 'q' is {w/c}. Am I correct?").
  - The lying direction follows the Marks–Tegmark and representation-engineering truth direction, from true vs false statements (§3.1, App. A).
  - Head importance is the write-norm w = ‖W_O(v̄⁺ − v̄⁻)‖₂ (§3.2).
  - The paper says "knowledge" and "lying" are used only in a mechanistic sense (§3).

  — [2604.19117](https://arxiv.org/html/2604.19117)
- **2609.35822** (Chen, Ying, Riggs Smith, Wertheimer, Shapira, "Tracing mechanisms of sycophantic agreement in language models"):
  - The logit lens is softmaxed over the four answer-token logits, so chance is 0.25 (Sec. 3.1).
  - "Answer retrieval" is defined as an output-ready write visible to the logit lens. "Opinion registration" is a non-output-ready write that the lens cannot see.
  - Probes decode answer identity (correct vs sycophantic answer) from per-head write-in vectors (App. B.2).
  - No truth direction and no tuned lens.

  — [2609.35822](https://arxiv.org/html/2609.35822)
- **2602.04918** (Zhang & Lin, "Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution"):
  - The method takes the unembedding vector w_correct of the ground-truth token from W_U and tracks the scalar logit L(y_correct) at each depth.
  - It defines the interference vector Δ_int = x_int − x_base at the final token, before RMSNorm.
  - Metrics are the norm ratio γ and cos(Δ_int, w_correct) (Experimental Setup: "Data Collection" and "Data Analysis").
  - "Truth direction" in this paper means w_correct, the unembedding row, not a fitted probe.

  — [2602.04918](https://arxiv.org/pdf/2602.04918)

### Inferences
- Papers that track only the unembedding of the correct token (2607 logit lens, 2508, 2602) measure output-readable knowledge. A drop in that readout fits all three fates: erased, bypassed/rotated, or outcompeted. On its own, it cannot tell them apart.
- 2604.19117's direction encodes "the user's claim is correct vs wrong" as a property of the input. It is evaluative, not answer-token based, which is why it is not subject to answer-token aliasing (see Q2).

### Gaps
- 2604.19117 does not report a direct per-item readout of claim falsity on trials where the model agreed with a false claim. The fetched text says so explicitly (§5 and App. T test transfer on lying statements, not on sycophantic items).
- I could not find the definition of the quantity averaged in 2604.19117 App. L "DIFF". The page defines DIFF = mean(non-syc) − mean(syc) of logit-lens activations, but not which logit.

## Q2. Probe training contexts (neutral only, pressured only, or mixed), and any below-chance results on misled items

### Takeaway
Only 2607.00415 fits probes on neutral contexts alone and evaluates them on pressured ones. It reports below-chance accuracy on flipped items, about 0.05 against 0.25 chance, on Gemma-2-9B. That is exactly the 2609.10739 aliasing signature. 2604.19117 fits on prompts balanced for claim truth (mixed by construction) and tests on unpressured statements. 2609.35822 fits within each condition using cross-validation. 2508.02087 and 2602.04918 train no probes. No paper other than 2607 reports below-chance values.

### Cited Findings
- **2607.00415, neutral-only fitting:** probes are "trained exclusively on baseline activations and evaluated on hinted activations." No hinted-trained or mixed probes are reported (RQ3, App. A.3). — [2607.00415](https://arxiv.org/html/2607.00415)
- **2607.00415, below chance on misled items:** for Gemma-2-9B under the Physician hint, LR and MLP probe accuracy on flipped questions "drop sharply from above 0.9 to near zero (≈0.05)" after the peak layer (chance = 0.25).
  - Llama and Qwen are described only as "near-complete probe collapse," with no numbers.
  - Resisted questions show "near-perfect probe transfer" for all personas.
  - The authors read below-chance as meaning "the correct answer representation is not merely absent but actively displaced." The conclusion says "the failure of both probe types rules out geometric reorganization and confirms genuine erasure, graded by authority level."
  - Flipped-subset sizes: Resident n=291, MS-3 n=117, MS-1 n=68. The Physician-flipped n is not stated.
  - The paper does not say whether the training activations were all questions or only baseline-correct ones.

  — [2607.00415](https://arxiv.org/html/2607.00415)
- **2508.02087:** no trained probes, so no probe training contexts and no below-chance results. Its expertise analysis uses PCA and class-centroid cosine similarity, not a classifier (Sec. 4.3). — [2508.02087](https://arxiv.org/html/2508.02087)
- **2604.19117, direction and probe fit:** L2 logistic regression (C=1.0, lbfgs) on last-token activations at layer ⌊0.85·L⌋, fixed before the experiment.
  - Training set: 100 wrong-opinion and 100 correct-opinion sycophancy activations, from TriviaQA pairs 0–99.
  - Test set: 100 true and 100 false "lying" statements from pairs 200–299 (disjoint content).
  - Transfer AUROC: Gemma-2-2B 0.83 [0.77, 0.89] at L22; Qwen3-8B 0.85 [0.80, 0.90] at L30; Mistral-7B 0.84 at L27 (single fit, no CI); Qwen2.5-1.5B 0.61 [0.59, 0.63] (App. T, Table 18).
  - No below-chance AUROC is reported. Anti-sycophancy DPO changes transfer by |Δ| ≤ 0.026, within a pre-specified ±0.05 margin (Table 19).

  — [2604.19117](https://arxiv.org/html/2604.19117)
- **2609.35822, within-condition fitting:**
  - Probes are four-class multinomial logistic regressions on head write-in vectors, reduced to 10 PCA components. The PCA is fit on pooled, mean-centred plain and deceptive vectors (L2, C=1.0, lbfgs, 1,000 iterations, stratified 5-fold CV).
  - Probes are fit "for each head and each of the four combinations of run condition (plain or deceptive) and target label." The baseline is mean accuracy over 20 label-shuffled permutations (App. B.2).
  - Opinion heads "strongly encode the sycophantic answer in the deceptive run, but do not strongly encode the correct answer" (Sec. 3.2, Fig. 4D).
  - Pushback-only heads "strongly encode the correct answer" in round two (Sec. 3.4, Fig. 6C). Numbers appear only in the figure.
  - No below-chance values are reported in the text.

  — [2609.35822](https://arxiv.org/html/2609.35822)
- **2602.04918:** no probes. Filtering keeps only items that were correct at baseline and then "flipped its output to the adversarial target" ("Mechanistic Compliance Filtering"). — [2602.04918](https://arxiv.org/pdf/2602.04918)

### Inferences
- **2607.00415 fits the aliasing setup.**
  - In the no-hint baseline (on baseline-correct items), the correct letter is the letter the model emits. So the baseline-fit probe cannot separate "correct answer" from "answer the model is about to give."
  - On flipped items the emitted letter is the hinted one, never the correct one. A probe that reads the emitted answer would score near 0 on the correct-letter label, which matches the observed ≈0.05.
  - On resisted items, emitted equals correct, so the same probe would score near-perfectly, which matches "near-perfect probe transfer."
  - This multi-class version of 2609.10739's binary result (0.006 vs 1.000) predicts exactly the reported pattern. The paper's reading of below-chance as "active displacement" is the very inference 2609.10739 warns against.
- **2604.19117's direction is fit across both claim-truth classes.** Under sycophancy the model's output ("agree") is often the same in both classes, so the direction cannot simply be the emitted token. That makes it structurally immune to compliant-context aliasing. The trade-off: it shows the input claim's correctness is represented somewhere, not that it survives on the specific agreed-wrong items.
- **2609.35822 is not exposed to the ally-to-rival transfer artifact.** Its probes are fit and cross-validated within the pressured condition, with correct-answer labels.
  - Separate confound (my inference): in the multi-round and pushback settings, round two contains the model's own round-one answer as text in context. A head that "encodes the correct answer" may be copying or referencing that context token, not retrieving parametric knowledge.
  - App. D.4 found opinion heads act more like a "reference" than a "content" mechanism, which makes this concern plausible for pushback heads too.

### Gaps
- 2607.00415 does not report what letter its probe predicts on flipped items. The share predicting the hinted letter would be the decisive check.
- 2609.35822 does not say whether its pushback probes used all items or only flipped ones. Its causal analyses use sycophantic items (e.g., "100% to 13.5%" sycophancy after patching), which suggests the analysis set is sycophantic items, but the text does not confirm this for probing.

## Q3. Lens used, causal tests and effect sizes

### Takeaway
Only 2604.19117 and 2609.35822 have strong, controlled causal tests (head ablation and patching with permutation nulls, BH correction and controls). 2508.02087 has single-layer patching. 2607.00415 has steering. 2602.04918 has none: its evidence is correlational geometry, though it is the only paper here that uses a Jacobian (of RMSNorm). No paper uses a tuned lens. Only the KnownLieBench pilot uses the Jacobian lens (J-Lens).

### Cited Findings
- **2607.00415 lens:** logit lens. The "peak layer" is the first layer from which P(hinted) stays ≥0.05 above P(correct) under the Physician hint: Llama-3.1-8B L17, Gemma-2-9B L28, Qwen3-8B L29. Lower personas show no crossover. — [2607.00415](https://arxiv.org/html/2607.00415)
- **2607.00415 causal tests:**
  - Adding per-question hint vectors (physician minus baseline) at the peak layer flips 63–82% of Physician-flipped items to the hinted letter across the three models (Fig. 5).
  - A mean hint vector does no better than a random control (≤7%).
  - Per-question authority vectors (physician minus MS-1) push MS-1 and Resident accuracy toward Physician levels (Fig. 6).
  - Geometry: Physician vector norm at peak is 61.2 (Gemma) and 67.8 (Qwen). |cos| between the authority and knowledge directions is <0.15. PC1 explains 91.0% (Gemma) and 81.9% (Qwen) of variance across authority vectors.
  - No significance tests, and no patching or ablation of the "erased" representation.

  — [2607.00415](https://arxiv.org/html/2607.00415)
- **2508.02087 lens:**
  - Logit lens Decision Score: the Opinion-only wrong-answer curve rises around layers 16–19, with a turning point near L19 in Llama. Qwen's shift is at L22 (App. A.2).
  - The KL spike is late. The text is inconsistent about where: "around layer 23" (Sec. 4.1) vs critical layer 32 for Llama and 27 for Qwen (Secs. 4.2, 5).

  — [2508.02087](https://arxiv.org/html/2508.02087)
- **2508.02087 causal tests:** full hidden-state activation patching at the single critical layer (token position not specified).
  - Plain into Opinion-only ("suppression") reduces Llama sycophancy by 36%.
  - Opinion-only into Plain ("induction") raises sycophancy to 47%.
  - Units (pp vs relative) are not stated, and there is no layer sweep.

  — [2508.02087](https://arxiv.org/html/2508.02087)
- **2604.19117 causal tests:**
  - Zeroing the shared heads in Gemma-2-2B raises sycophancy from 28% to 81% while factual accuracy goes only from 69% to 70% (abstract, §4.1).
  - Projection ablation takes Gemma-2-27B-IT agreement from 10.5% to 100%, and adds +27 pp on Llama-3.3-70B (§4.4).
  - Mean-ablation sufficiency/necessity: Mistral-7B +19.4% / +12.9%; Llama-3.1-70B +13.1% / −5.2% (n.s.); Llama-3.3-70B +13.2% / −20.1% (n.s.) (Table 6).
  - Activation-patching restoration ratio on Llama-3.3-70B: 1,732× (lying) and 2,248× (sycophancy) (Table 3).
  - Path-patching edge correlations: r = 0.993 (Gemma-2-2B, 275 edges).
  - Benjamini–Hochberg: 14/18 causal cells survive q < 0.05 (App. F). Uses 10,000-permutation layer-stratified nulls (App. G).

  — [2604.19117](https://arxiv.org/html/2604.19117)
- **2604.19117 logit lens (App. L):**
  - DIFF = mean(non-syc) − mean(syc) per layer. Peak excess = 100·(peak − final)/final.
  - Gemma-2-2B: peak +20.12 at L19 vs final +8.87, excess +127%.
  - Mistral-7B: peak +6.60 at L30 vs final +3.49, excess +89%.
  - Llama-3.1-70B: monotonic, peak at the final layer, excess 0%.
  - The paper reads the mid-layer peak with late attenuation as "detect-then-override." It says the 70B override "is not visible at this granularity."

  — [2604.19117](https://arxiv.org/html/2604.19117)
- **2609.35822 causal tests:**
  - Last-token residual patching from plain into deceptive and label-swapped runs, using a normalised logit difference. Critical layer: Llama 16, Mistral 17, Gemma 27 (App. B.1).
  - Plain-to-deceptive patching rises around L12; plain-to-label-swap rises after L16. The paper reads this as opinion registration preceding answer retrieval (Sec. 3.1).
  - Head OV patching: early opinion heads r = 0.26 across conditions; late retrieval heads r = 0.85.
  - Zero-ablating a single Llama head raises deceptive accuracy by up to 44%, with at most a 6% drop in plain accuracy. Patching the top four heads cuts sycophancy from 100% to 13.5% (Figs. S2–S3).
  - Mistral: one head adds up to +16.0%; ten heads bring sycophancy to 18.6% (Figs. S8–S9).

  — [2609.35822](https://arxiv.org/html/2609.35822)
- **2609.35822 pushback heads:**
  - Identified by patching per head from round one's last token to round two's last token. Pushback-only heads have negative effect under affirmation and positive effect under pushback.
  - Pushback and affirmation effects correlate at r = 0.27, and only 4 of the top 10 heads are shared.
  - "This suggests that these heads suppress the original correct answer once a pushback occurs" (Sec. 3.4). This rests on probe decoding.
  - Ablation and attribution are not reported for pushback heads. Suppression is not tested against erasure.

  — [2609.35822](https://arxiv.org/html/2609.35822)
- **2602.04918 evidence:**
  - A first-order logit decomposition through the RMSNorm Jacobian J_f(x) = (I − x̂x̂ᵀ)/‖x‖ splits the logit change into "Semantic Alignment" and "Norm Correction" terms. It argues that purely shrinking the magnitude cannot change logits under RMSNorm (Theoretical Framework).
  - Results: Llama γ = 0.978 (p < 0.05), described as dilution; Qwen γ ≈ 1.01 and GLM γ ≈ 1.06, so H1 (dilution) "is rejected as a generalizable law."
  - cos(Δ_int, w_correct): Qwen −0.115, Llama −0.060, GLM −0.146, "statistically distinct from −1.0."
  - Regressing angular misalignment on logit drop gives R² = 0.90 (Qwen) and 0.87 (GLM). Llama's R² is not given in the text.
  - The correct-token logit collapse is "most pronounced in the final 20% of the layers."
  - No patching, ablation or steering.

  — [2602.04918](https://arxiv.org/pdf/2602.04918)

### Inferences
- **2602.04918's "not erased" conclusion does not follow from its measurements.**
  - Near-zero cosine between a high-dimensional difference vector and any fixed vector is the default expectation. For random vectors it is about 0 ± 1/√d, roughly ±0.016–0.02 at d = 2,560–4,096. (The hidden sizes are my background knowledge, not from the paper.)
  - The reported −0.06 to −0.146 values are therefore several times that scale in the negative direction, consistent with a real antiparallel component and not pure orthogonality.
  - More importantly, the paper never tests whether the correct answer stays decodable from x_int (no probe and no patching). By its own Fig. 2 the readout along w_correct collapses.
  - It also contradicts itself. One passage says the model "fundamentally reorganizes its internal representation to prioritize the external context over its intrinsic knowledge." Another says "parametric memory structurally intact but functionally bypassed."
- **2607.00415's steering shows sufficiency, not erasure.** Adding hint vectors shows they can induce flips. It says nothing about whether the correct-answer information is gone.

### Gaps
- No paper here reports a tuned lens.
- None reports a mixed-fit truth probe evaluated on misled items, which would be the clean 2609.10739 test.

## Q4. Models (sizes), prompt formats, single vs multi-turn

### Takeaway
All five papers use mostly 4B–9B open models. 2604.19117 is the exception, extending to 70B–72B. Four are single-turn with multiple-choice letter outputs. 2609.35822 is the only one with multi-round and "are you sure?" pushback. 2602.04918 is not a knowledge-conflict study in the usual sense: its contexts are always erroneous MMLU "adversarial priors."

### Cited Findings
- **2607.00415:**
  - Models: Llama-3.1-8B-Instruct, Qwen3-8B and Gemma-2-9B-it, via TransformerLens.
  - Data: MedQA-USMLE, with five variants per question (no hint, plus four persona hints).
  - Format is question-then-hint: "Hint: A person who is {persona} thinks the answer is {letter}."
  - Outcome is the next-token probability over option letters. Single-turn is implied.
  - Physician-hint accuracy: 15% (Llama), 29% (Qwen), 34% (Gemma), against roughly 60% at baseline.

  — [2607.00415](https://arxiv.org/html/2607.00415)
- **2508.02087:**
  - Behavioural study: seven ~7B models (Llama3.1-8B-Instruct, Qwen2.5-7B-Instruct, OPT-6.7B, Mistral-7B-Instruct-v0.3, Falcon-7B, OLMoE-1B-7B-Instruct, Pythia-6.9B).
  - Mechanistic study: Llama3.1-8B-Instruct and Qwen2.5-7B-Instruct.
  - Data: MMLU (57 subjects).
  - Conditions: Plain; Opinion-only ("I believe the right answer is B", with the wrong option drawn uniformly); First-pov expertise (Beginner, Intermediate or Advanced, from 20 GPT-4o descriptions per level, e.g., "As an Expert in Abstract Algebra…"); Third-pov.
  - Single-turn is implied.
  - Mean agreement with wrong opinions is 63.7% (range 46.6–95.1%). Expertise framing changes sycophancy by at most 4.4 points. Third-pov lowers sycophancy by an average of 13.6%.

  — [2508.02087](https://arxiv.org/html/2508.02087)
- **2604.19117:**
  - 12 models from 1.5B to 72B: Gemma-2-2B/9B/27B-IT, Qwen2.5-1.5B/32B/72B, Qwen3-8B, Llama-3.1-8B/70B, Mistral-7B-v0.1, Mixtral-8x7B and Phi-4.
  - RLHF pair: Llama-3.1-70B vs Llama-3.3-70B, with sycophancy 39% → 3.5% and shared-head fraction 57/72 → 51/72.
  - Mistral → Zephyr-7B: head-importance Spearman 0.846 → 0.848.
  - Anti-sycophancy DPO: Mistral 28% → 2%, Gemma 52% → 28%.
  - Single-turn only. Multi-turn is explicitly excluded (§5). The paper notes that the 70B RLHF claim rests on n = 2.

  — [2604.19117](https://arxiv.org/html/2604.19117)
- **2609.35822:**
  - Models: Llama-3.1-8B-Instruct (main), Mistral-7B-Instruct-v0.3 (App. E) and Gemma-2-9B-it (App. F), plus Llama-3.1-8B base for comparison.
  - Data: MMLU STEM (2,476 candidates) and TriviaQA (9,960).
  - Settings: multi-round and description-based. Pushback is "I don't think that's right. Are you sure?"
  - Pushback sycophancy: Llama 38.6%, Mistral 10.7%. Gemma gave only 12 usable pushback examples and is excluded.
  - Llama's single-turn sycophancy rate is 32.8%.

  — [2609.35822](https://arxiv.org/html/2609.35822)
- **2602.04918:**
  - Models: Qwen-3-4B, Llama-3.1-8B and GLM-4-9B.
  - Data: 150 MMLU plus 150 MMLU-Pro questions, augmented with "adversarial priors—contextual frames that present incorrect information as factual or preferred," for 1,500 trials.
  - Only items correct at baseline and then flipped are kept. The retained n is not given.
  - Single-turn.

  — [2602.04918](https://arxiv.org/pdf/2602.04918)

### Inferences
- The brief describes 2602.04918 as knowledge conflicts "where following context is often correct." The full text does not match that framing. Every context is constructed to be erroneous ("novel but erroneous knowledge"), and the paper calls the result "the dominant geometric signature of LLM sycophancy." So in its dataset, following the context is never correct.
- Gemma-2-9B and Llama-3.1-8B appear in several papers. A matched replication across papers is therefore feasible.

### Gaps
- The total number of MedQA questions in 2607.00415 is not stated.
- 2508.02087 gives no sample sizes, seeds or confidence intervals in the main text.
- The retained-item counts in 2602.04918 are not stated.

## Q5. Does each paper's verdict survive the 2609.10739 artifact critique?

### Takeaway
2607.00415's "erasure" verdict is plausibly an artifact: its evidence carries the artifact's exact signature. The other four papers are not exposed to compliant-context aliasing because they fit no such probe. But "survives the artifact" is not the same as "well supported":
- **2508.02087 ("override")** is robust, but says only that output-readable preference changes late. It is silent on whether the truth persists elsewhere.
- **2604.19117 ("registered but overridden")** is robust to the artifact, and is the best-supported persistence or outcompetition claim. The evidence is population-level and fades at 70B.
- **2609.35822** is robust that the correct answer stays decodable in pushback heads. Its "suppress" mechanism is untested, and the decoded answer may be copied from context.
- **2602.04918 ("bypass/rotation, not erased")** is untouched by the artifact but unsupported by its own data: can't tell.

### Cited Findings
- **2607.00415:** probes are fit on baseline only. Accuracy falls from >0.9 to ≈0.05 on flipped items (below the 0.25 chance level), and transfer to resisted items is near-perfect. The authors conclude "genuine erasure." — [2607.00415](https://arxiv.org/html/2607.00415)
- **2609.10739:** an ally-fit probe gives 0.006 where a mixed-fit probe gives 1.000. A below-chance readout "does not, on its own, establish that truth information has been lost." Final-layer inversion can coexist with mid-stack decodability of 0.86–1.00 (§3–4, App. G). — [2609.10739](https://arxiv.org/html/2609.10739)
- **2508.02087:** the claim is that opinion prompts "override" a fact-based preference in late layers. The evidence is logit lens, KL and single-layer patching, with no probes (Sec. 4). — [2508.02087](https://arxiv.org/html/2508.02087)
- **2604.19117:**
  - Claims are "registered-but-overridden" (§1), models "register the error and agree anyway" (abstract), and "a routing failure rather than a knowledge gap" (§6). Alignment "masks but does not remove" the circuit.
  - The direction is fit on balanced wrong- and correct-opinion prompts; transfer AUROC is 0.83–0.85.
  - The logit-lens mid-layer peak with late attenuation shows +127% (Gemma-2-2B) and +89% (Mistral), but 0% at Llama-3.1-70B.

  — [2604.19117](https://arxiv.org/html/2604.19117)
- **2609.35822:** pushback-only heads "strongly encode the correct answer" in round two, and "these heads suppress the original correct answer." No ablation is reported for pushback heads (Sec. 3.4). — [2609.35822](https://arxiv.org/html/2609.35822)
- **2602.04918:** "The internal truth is not erased; it is merely bypassed by a lateral 'shove'." The evidence is cos(Δ, w_correct) ≈ −0.06 to −0.15 and norm ratios. There is no decodability test. — [2602.04918](https://arxiv.org/pdf/2602.04918)

### Inferences
- **2607.00415: plausibly artifact (high confidence).** Its three probe facts are each predicted by a probe that reads "answer about to be emitted": high on baseline, ≈0 on flipped, ≈1 on resisted. Its other evidence (logit-lens crossover, steering sufficiency) fits erased, bypassed and outcompeted alike. To rescue the verdict, the authors would need any of:
  - a probe fit on mixed baseline and hinted data, including flipped items;
  - a hinted-fit probe on flipped items with correct-letter labels;
  - a check of whether the baseline probe predicts the hinted letter on flipped items.

  Below-chance at mid layers just after the peak, rather than only at the final layer, is also something 2609.10739 App. G warns can occur while information stays decodable by a refit probe.
- **2508.02087: robust.** No fitted probe, so no aliasing. The "override" verdict, however, is a claim about the logit-lens-readable (output) subspace only. It neither rules in nor rules out that truth persists.
- **2604.19117: robust to the artifact.** Its persistence/outcompetition reading is the best supported among the five. Caveats:
  - The "knows" signal is shown at population level (claim-wrong vs claim-right inputs) and through transfer to unpressured lying.
  - It is not shown per item on agreed-wrong trials.
  - At 70B the detect-then-override logit-lens signature disappears (0% excess).
  - The paper itself does not separate "user claim is false" from "model inclined to defer" (§4.6 only argues the opinion direction is orthogonal, |cos| < 0.14).
- **2609.35822: robust to the artifact on persistence.** Fitting within the condition, with correct-answer labels on pressured items, means the probe cannot be reading the emitted answer. The "suppression" mechanism is still "can't tell": it is inferred, not ablated, and the round-two context contains the round-one answer.
- **2602.04918: can't tell.** The artifact is irrelevant because there is no probe, but the bypass verdict was never directly tested.
- **Net for the final report:**
  - "Erased" (2607) is the weakest verdict in this set.
  - "Outcompeted / overridden downstream, with the evaluative signal registered" (2604, partly 2508 and 2609) is the best supported at ≤9B.
  - "Bypassed by rotation" (2602) is a hypothesis without a decodability test.

### Gaps
- None of the five reports the mixed-fit control. Whether 2607's below-chance pattern reverses under mixed fitting is a prediction, not an observation.
- I did not see 2609.10739's main-text §5–9, or whether it explicitly discusses any of these five papers.

## Q6. KnownLieBench pilot appendix (~27B, single-turn, decision token): protocol as a template

### Takeaway
KnownLieBench (Liu, Zhao, Yuan, Ma, Huang, Jiang; arXiv 2608.26372) runs an exploratory J-Lens pilot on Qwen3.5-27B and Llama-3.1-8B. The setup is 16 paired single-turn eligibility scenarios, with an honest vs deceptive framing contrast and a matched not-entitled control. The readout is a yes-minus-no logit margin averaged over the 50th–90th percentile of depth, read at the record, question and first decision token. Because the lens is pretrained (no fitting on compliant contexts), and the matched control decorrelates truth from the prescribed answer, the design avoids the aliasing artifact.

### Cited Findings
- **Main benchmark:**
  - 8 customer-service domains and 112 source-grounded cases.
  - A neutral "knowledge gate" call per case verifies the model knows the entitlement before any dialogue is scored. It is framed as "no customer present and nothing is at stake -- this is an internal knowledge check" and returns JSON eligibility (App. G.1).
  - Emergent (incentive-only) vs instructed deception, with an honest control.
  - Multi-turn dialogues with a customer agent initialised at high, medium or low trust.
  - 18 models, more than 18,000 interactions (Sec. 1).

  — [KnownLieBench 2608.26372](https://arxiv.org/pdf/2608.26372)
- **Pilot protocol (Sec. 4.3 RQ4; App. F):**
  - 16 paired eligibility scenarios, two per domain. Each pair is an entitled record plus a matched not-entitled control, posed as a single-turn yes/no question over a short policy and record, separate from the benchmark dialogues.
  - Conditions: the entitled record under honest and deceptive framing; the not-entitled control under deceptive framing.
  - Lens: the pretrained Jacobian lens of Gurnee et al. (2026), under greedy decoding.
  - Readout: the yes-minus-no logit difference, averaged over layers spanning roughly the 50th–90th percentile of depth.
  - Positions: the record's entitlement field, the end of the customer question, and the first generated answer token.
  - Statistics: paired differences across the 16 scenarios, with standard errors.

  — [KnownLieBench 2608.26372](https://arxiv.org/pdf/2608.26372)
- **Four contrasts (App. F, Fig. 6a):**
  - Framing invariance: honest minus deceptive on entitled records, at the question.
  - Truth gap: entitled minus not-entitled under deceptive framing, at the question.
  - Suppression: honest minus deceptive on entitled records, at the first answer token.
  - Within-lie drop: question to first answer token under deceptive framing.

  — [KnownLieBench 2608.26372](https://arxiv.org/pdf/2608.26372)
- **Results (Sec. 4.3):**
  - Honest and deceptive readouts are similar at the question but diverge at the decision. The entitlement signal is lower under deceptive framing for both models.
  - Qwen3.5-27B also separates entitled from not-entitled cases at the question, and drops sharply from question to deceptive decision. Llama-3.1-8B shows the decision-time contrast without this stronger pre-decision pattern.
  - The authors' hedges: the signal "should not be interpreted as a direct measure of belief or intent," and the results "do not establish that this signal represents belief or causes the deceptive output." The deceptive framing explicitly instructs the model to deny eligibility, so the pilot covers instructed, single-turn denial, not the emergent multi-turn benchmark setting.

  — [KnownLieBench 2608.26372](https://arxiv.org/pdf/2608.26372)

### Inferences
- **As a template for sycophancy fate-of-knowledge work:**
  1. Run a knowledge gate per item.
  2. Pair each true item with a matched false control under the same pressure (the "truth gap"). This is the decorrelation that 2609.10739 recommends: under deceptive framing, the prescribed answer is "no" for both members of the pair, while the truth differs.
  3. Use a fixed, unfitted readout (J-Lens or logit lens), or a mixed-fit probe.
  4. Read at the pre-decision and decision positions separately, so "present before deciding, lower at the decision" can be told apart from "never present."
  5. Report paired differences with SEs.
- **What the template lacks:**
  - any causal test;
  - a fitted-probe comparison;
  - the numeric effect sizes, which appear only in Fig. 6 and were not extractable from the text.

  Note also that "suppression" here means a lower yes-minus-no margin at the decision token. That is an output-space readout, so a decision-time drop fits "outcompeted" as well as "erased."

### Gaps
- The exact Fig. 6a values (contrast magnitudes and SEs) were not in the extracted text.
- The pilot's model list (Qwen3.5-27B and Llama-3.1-8B) comes from Sec. 4.3. I found no third model.

## Q7. Why 2607.00415 and 2508.02087 disagree on whether authority framing is encoded

### Takeaway
They differ in behaviour, in prompts and in method. Behaviourally, 2508.02087 finds expertise moves sycophancy by at most 4.4 points, while 2607.00415 finds a monotonic persona hierarchy (Physician hint → 15–34% accuracy). Prompts differ in competence vs institutional credentials, first- vs third-person, MMLU vs domain-matched MedQA, and hint before vs after the question. Methodologically, centroid-cosine/PCA similarity in 2508 vs per-question difference vectors plus steering in 2607. Llama-3.1-8B is in both, so model choice alone does not explain the gap.

### Cited Findings
- **2607.00415 App. A.1.1** attributes the disagreement to three design differences:
  1. Wang et al. "were measuring the effect of competence descriptions rather than institutional authority."
  2. Wang et al. randomly pair expertise labels with unrelated domains, whereas 2607 uses domain-matched medical personas.
  3. Wang et al. use first-person self-claimed expertise, 2607 uses third-person attribution.

  Sec. 2 adds that Wang et al. put the hint before the question, while 2607 uses question-then-hint. The paper calls this its interpretation, not a controlled test.

  — [2607.00415](https://arxiv.org/html/2607.00415)
- **2508.02087's evidence for "not encoded"** is PCA of prompt-token hidden states at layer 32 (Llama) and 27 (Qwen), plus centroid cosine similarity:
  - Within-expertise: Intermediate–Advanced 0.997, Beginner–Intermediate 0.934, Beginner–Advanced 0.903.
  - Opinion-only vs each level: −0.955, −0.998, −0.990.
  - No trained probe or classification accuracy (Sec. 4.3).
  - Behavioural effect of expertise ≤4.4 points. Third-pov lowers sycophancy by an average of 13.6% relative to First-pov (Sec. 5).

  — [2508.02087](https://arxiv.org/html/2508.02087)
- **2508.02087 describes its expertise prefixes as built from GPT-4o-generated roles plus the MMLU subject** (e.g., "As an Expert in Abstract Algebra…"; App. A, Table 1). This seems to contradict 2607's statement that Wang et al. pair labels with unrelated domains. — [2508.02087](https://arxiv.org/html/2508.02087); contradicted by [2607.00415](https://arxiv.org/html/2607.00415)
- **2607.00415's evidence for "encoded":** per-question authority vectors (physician minus MS-1) restore Physician-level effects. PC1 explains 91.0% (Gemma) and 81.9% (Qwen) of authority-vector variance. |cos| between authority and knowledge directions is <0.15. No authority-classification probe is reported. — [2607.00415](https://arxiv.org/html/2607.00415)
- **Model overlap:** 2508 uses Llama3.1-8B-Instruct and Qwen2.5-7B-Instruct. 2607 uses Llama-3.1-8B-Instruct, Qwen3-8B and Gemma-2-9B-it. — [2508.02087](https://arxiv.org/html/2508.02087); [2607.00415](https://arxiv.org/html/2607.00415)

### Inferences
- **The two methods measure different things.**
  - High centroid cosine between expertise levels (0.90–0.997) is compatible with a small but linearly decodable and causally used authority direction. Centroid similarity is dominated by shared prompt content.
  - 2607's per-question difference vectors cancel shared content and isolate the persona delta.
  - So "not encoded" (2508) is not established by its method, and "encoded" (2607) is shown causally without any test of linear decodability.
- **The behavioural gap is the more substantive difference.** Institutional, domain-matched, third-person credentials in a medical domain move behaviour. Self-claimed competence levels in MMLU barely do.
- **Person framing points opposite ways.** 2508 finds third-person reduces sycophancy, yet 2607's strongest effects come with third-person attribution. This suggests credential type and domain coherence, not grammatical person, drive 2607's hierarchy. That is untested.
- **Both "encoded" and "erased" in 2607 rest on peak-layer analyses** that never control for the emitted answer (see Q5).

### Gaps
- Neither paper runs the other's prompts. No controlled cross-over (same model, same items, swapping credential type, person and placement) exists in either text.

### Comparison table

| Paper | Knowledge tracked | Probe fit contexts | Lens | Causal test (effect size) | Models | Verdict (paper's own) | Survives artifact? |
|---|---|---|---|---|---|---|---|
| 2607.00415 (authority hierarchy, MedQA) | Correct answer letter (logit lens P(correct) vs P(hinted); 4-class letter probes) | Neutral (no-hint baseline) only; evaluated on hinted; flipped items ≈0.05 vs 0.25 chance (Gemma-2-9B); resisted near-perfect | Logit lens + LR/MLP probes | Per-question hint-vector addition flips 63–82%; mean vector ≤7% (≈random); authority vectors restore Physician effect; no significance tests | Llama-3.1-8B-Inst, Qwen3-8B, Gemma-2-9B-it; single-turn, question-then-hint | Late-layer "genuine erasure," graded by authority | **Plausibly artifact**: exact aliasing signature (neutral-fit, below chance on misled, perfect on resisted) |
| 2508.02087 (When Truth Is Overridden, MMLU) | Answer-option logits (Decision Score); KL between Plain and Opinion distributions | No probes | Logit lens, layer-wise KL, PCA/centroid cosine | Single-layer full hidden-state patching: suppression −36% sycophancy, induction to 47% (Llama) | Llama3.1-8B-Inst, Qwen2.5-7B-Inst (mech); 7 ~7B models (behav); single-turn | Late-layer override; expertise not encoded (centroid cos 0.90–0.997) | **Robust** (no fitted probe), but silent on persistence beyond output space; "not encoded" weakly supported |
| 2604.19117 (shared sycophancy–lying circuit) | Evaluative "claim is wrong" direction (wrong- vs correct-opinion mean diff); truth/lying direction; head write-norms | Mixed by construction (100 wrong + 100 correct opinions); tested on unpressured true/false statements; AUROC 0.83–0.85 (0.61 Qwen2.5-1.5B); none below chance | Logit lens (App. L), DLA-style write-norm, attribution/path patching | Head zeroing 28%→81% sycophancy (Gemma-2-2B); projection ablation 10.5%→100% (Gemma-2-27B), +27 pp (Llama-3.3-70B); BH 14/18 cells | 12 models 1.5B–72B; Llama-3.1→3.3-70B RLHF pair; single-turn | Registered but overridden; persists through RLHF/DPO | **Robust**; best-supported persistence claim, but population-level and no per-item readout on agreed-wrong items; logit-lens override signature 0% at 70B |
| 2609.35822 (tracing sycophantic agreement; pushback heads) | Answer identity from head OV outputs; logit-lens answer probabilities | Within-condition, 5-fold CV (plain or deceptive; pushback rounds); PCA on pooled plain+deceptive; shuffle baseline; no below-chance reported | Logit lens, residual and head patching, probes | Top-4 head patch: sycophancy 100%→13.5% (Llama); single head +44% deceptive accuracy; pushback heads not ablated | Llama-3.1-8B-Inst (+base), Mistral-7B-Inst-v0.3, Gemma-2-9B-it; single and multi-round, "are you sure?" | Pushback heads "suppress the original correct answer" (correct answer still encoded) | **Robust** on persistence (pressured-context fit); "suppression" mechanism can't tell (inferred, not ablated; round-1 answer present in context) |
| 2602.04918 (Simulated Adoption, MMLU adversarial priors) | Correct-token unembedding w_correct and its logit; norm ratio; cos(Δ, w_correct) | No probes | RMSNorm-Jacobian decomposition; layer-wise logit; geometric metrics | None (correlational R² 0.90 Qwen, 0.87 GLM) | Qwen-3-4B, Llama-3.1-8B, GLM-4-9B; single-turn; flipped-only items | Bypass via orthogonal rotation; "not erased" | **Can't tell**: artifact inapplicable, but decodability never tested; cos −0.06 to −0.15 is not clearly orthogonal at d ≈ 2.5–4k |
| KnownLieBench pilot (2608.26372, App. F), template | Yes-minus-no entitlement logit margin | None fitted (pretrained J-Lens); matched not-entitled control under same deceptive framing | Jacobian lens, layers 50–90% depth; record, question and first decision token | None (paired differences ± SE over 16 scenarios) | Qwen3.5-27B, Llama-3.1-8B; single-turn, instructed denial | Entitlement decodable pre-decision (27B), lower at decision; no belief or causal claim | **Robust by design** (no compliant-context fit; truth decorrelated from prescribed answer) |
