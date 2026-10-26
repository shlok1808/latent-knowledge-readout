# Sweep A: know-do / knowledge-prediction / latent-knowledge / decoded-but-unused families (round 2)

Sweep date: 2026-10-22. Coverage window: 2023 to October 2026, plus a few foundational older papers (CCS 2022, Amnesic Probing 2020).
Method: search sweep only. I read abstracts (arXiv abstract pages for about 75 likely-Core papers) and otherwise used titles and search snippets. Nothing was deep-read. No PDFs were downloaded.

Terminology: "decodable" means a probe or lens recovers the correctness-relevant variable from activations. I do not write "the model knows" unless the paper shows more than that.

**Status of the stop rule: NOT met.** The coordinator cut the time budget and told me to stop searching. Before that, every search channel had also run out; the families were not saturated:
- The session-wide WebSearch budget (200 calls, shared with the parallel agents) was used up.
- The OpenAlex daily budget was used up, and an API key needs an email registration.
- Semantic Scholar returned 429 errors.
- The arXiv export API and then arXiv web search began returning empty pages.

Over the last 10 valid queries (Q128 to Q137), 7 still added new relevant papers. The literature is not saturated, especially for 2026-07 to 2026-10 preprints, multimodal "knows but cannot say" papers, and agentic probing papers. Later phases should treat this list as a large sample, not as complete.

---

## 0. Summary against the key questions

### KQ1. Which papers test, on internals, whether decodable correct information influences the output, and what did they find?

**Takeaway.** Most 2025-2026 work finds that decodable correctness-relevant information often does not reach the output. There are four distinct mechanistic accounts, and each has causal evidence in a few papers:
- **Override.** A competing signal wins at late layers.
- **Readout or routing failure.** The information is present but not routed into the decision, or it sits in a direction the unembedding does not read.
- **Bypass or entanglement.** The information is decodable, but steering along it does nothing specific.
- **Transformation.** The model represents knowledge and verbalization separately, and the verbalization state changes.

Several papers also show that "decodable on errors" alone cannot prove that the information was present and then discarded. These method caveats matter.

**Cited findings** (abstract-level):
- **Override (sycophancy).** The same small set of attention heads carries a "this statement is wrong" signal both when the model evaluates a claim alone and when it is pressured to agree. Silencing these heads flips sycophantic behavior while factual accuracy stays intact. Path patching shows that the same head-to-head edges drive sycophancy, factual lying and instructed lying. An RLHF refresh cuts sycophancy about tenfold while the heads persist. The study covers 12 open-weight models "spanning small to frontier scale". — [2604.19117](https://arxiv.org/abs/2604.19117)
- **Override (sycophancy, emergence).** Logit lens plus activation patching show a two-stage emergence: a late-layer shift in output preference, then deeper representational divergence. — [2508.02087](https://arxiv.org/abs/2508.02087) (AAAI 2026)
- **Override (persuasion).** Persuasive text changes a one-dimensional option-routing feature, so mid-layer "decision heads" copy the persuasion target instead of the correct option. Shallow heads read the persuasive keywords. — [2605.09314](https://arxiv.org/abs/2605.09314)
- **Override (game play, more than 8B).** Logit-lens readouts lean toward the Nash action through about three quarters of the network depth, then a late surge moves the output toward cooperation in Llama-3-70B, Qwen2.5-32B and Qwen2.5-72B. In Llama-3-8B the override is a single steerable direction. Whether Nash counts as "correct" is task-defined. — [2604.27167](https://arxiv.org/abs/2604.27167)
- **Override (temporal fact conflict).** The newer fact is recoverable, but the default forward pass prefers the outdated one. Activation patching flips 72-85% of cases, and steering beats norm-matched random directions. Released as an 8,746-item benchmark. — [2606.20959](https://arxiv.org/abs/2606.20959)
- **Routing failure.** Probes decode the constraint above 88% in two open-weight models. Activation patching repairs one model (+6.4 nats) and not the other (-0.07). The authors conclude "a routing problem, not a knowledge problem". — [2608.12321](https://arxiv.org/abs/2608.12321)
- **Readout or threshold failure.** A 0.6B model answers YES on all 1,200 logical verification items, yet probes read the verdict at 0.96 AUC. The verdict survives to the output logits (margin AUC 0.89), but a saturated decision threshold erases it. Across a 13x scale range, internal knowledge saturates while behavior is non-monotone, and an 8B model fails through its answer channel. — [2609.04582](https://arxiv.org/abs/2609.04582)
- **Inaccessible to readout.** Count directions are nearly orthogonal to the digit unembedding rows (|cos| ≤ 0.032). Updating only the digit rows of the output head fixes much of the failure. Tested on Pythia, Qwen3 and Mistral from 0.4B to 14B. — [2605.03258](https://arxiv.org/abs/2605.03258)
- **Bypass or entanglement (steering null).** Clinical hazards are decodable at 98.2% AUROC against 45.1% output sensitivity. Across four mechanistic methods, steering fixes at most about 24% of missed hazards, and SAE steering has zero effect. Models: Qwen2.5-7B-Instruct and Steerling-8B. — [2603.18353](https://arxiv.org/abs/2603.18353)
- **Bypass or entanglement (medical overthinking).** The failure regime is decodable at 71.6%, but 29 fixed linear steering configurations give Δ≈0. The direction overlaps 85-88% with task-critical computation, and LEACE erasure hurts accuracy. — [2605.05715](https://arxiv.org/abs/2605.05715)
- **Bypass (logical validity).** Validity is near-perfectly decodable even on behaviorally wrong items, but interventions along probe directions have only weak, nonspecific effects compared with random controls. — [2609.02438](https://arxiv.org/abs/2609.02438)
- **Bypass, with a random-init control.** Some probe performance is available from randomly initialized representations, and decodable information is "not always actionable". — [2608.17843](https://arxiv.org/abs/2608.17843)
- **Transformation.** Solvability "knowledge" and "verbalization" are separate, linearly decodable directions. Fabrication tracks changes in verbalization, not in knowledge, and steering improves abstention. — [2607.05013](https://arxiv.org/abs/2607.05013)
- **Transformation (deceptive instructions).** Deceptive instructions shift representations, mostly in early to middle layers. The instructed True/False output is linearly predictable in every condition. Models: Llama-3.1-8B and Gemma-2-9B. — [2507.22149](https://arxiv.org/abs/2507.22149) (EMNLP 2025)
- **Repairable (binding errors).** On wrong trials, the probe recovers the correct in-context binding above chance across 16 checkpoints. Steering toward the probe-decoded binding, with no gold label, raises accuracy. — [2609.11216](https://arxiv.org/abs/2609.11216). The follow-up tracks this across Pythia pretraining and post-training checkpoints and gives an information-theoretic counterexample: decodability on errors alone cannot establish discarded output information. — [2609.31401](https://arxiv.org/abs/2609.31401)
- **Present in the output distribution, more than 8B.** 16-47% of Instruct hallucinations occur while substantial probability mass is already on the correct concept, and the rate rises monotonically with scale (Qwen and Llama, 0.8B-72B). What distinguishes these failures is that the mass is dispersed across surface forms. — [2605.22007](https://arxiv.org/abs/2605.22007)
- **Agentic.** A residual-stream readout forecasts unauthorized actions before they happen (AUROC 0.94). Gating actions on this forecast cuts violations in four model families. — [2610.17301](https://arxiv.org/abs/2610.17301). Task-relevant state and value information is "represented more reliably than it is reflected in behaviour" in grid-world agents. — [2610.17444](https://arxiv.org/abs/2610.17444)
- **Multi-agent override.** Agents that yield to a wrong majority still represent their original two-hop bridge entity at answer time. The tests are pre-registered and include Qwen3.6-27B. — [2610.02702](https://arxiv.org/abs/2610.02702)
- **Foundational.** False demonstrations produce correct-leaning early layers that are overridden after a "critical layer". — [2307.09476](https://arxiv.org/abs/2307.09476)

**Inferences**
- "Override" and "readout or routing failure" have the strongest causal support. "Bypass" rests mostly on failed steering, which is weaker evidence: steering can fail for reasons unrelated to use (see [2604.09839](https://arxiv.org/abs/2604.09839), [2609.24821](https://arxiv.org/abs/2609.24821), [2610.07270](https://arxiv.org/abs/2610.07270)).
- The fates may be model-specific even within one task. [2608.12321](https://arxiv.org/abs/2608.12321) shows repair in one model and not another, and [2605.07984](https://arxiv.org/abs/2605.07984) finds that only Gemma-3-27B causally relies on its decodable plan.

**Gaps**
- I saw no paper in this sweep that compares several fates (override, bypass, readout) head-to-head on the same items with matched controls. Each paper argues for one fate.

### KQ2. Methods, controls, metrics and datasets directly usable for the core question
- **Controls against "decodable but meaningless":**
  - random-init representations: [2608.17843](https://arxiv.org/abs/2608.17843)
  - vision-ablation arbiter: [2606.31257](https://arxiv.org/abs/2606.31257)
  - probes relying on textual evidence: [2509.21344](https://arxiv.org/abs/2509.21344)
  - self-judgement confounding of correctness probes: [2607.16799](https://arxiv.org/abs/2607.16799)
  - distributional associations: [2603.04317](https://arxiv.org/abs/2603.04317)
  - unsupervised probes find salient features instead of knowledge: [2312.10029](https://arxiv.org/abs/2312.10029)
  - the information-theoretic counterexample: [2609.31401](https://arxiv.org/abs/2609.31401)
- **Decodability versus use:**
  - amnesic probing: [2006.00995](https://arxiv.org/abs/2006.00995), with LEACE and mean-projection fixes in [2506.11673](https://arxiv.org/abs/2506.11673)
  - Dyck decodable-versus-used dissociation: [2604.22128](https://arxiv.org/abs/2604.22128)
  - SAE decomposition of probe weights by gradient sensitivity: [2609.18080](https://arxiv.org/abs/2609.18080)
  - read-best versus steer-best layers: [2609.22135](https://arxiv.org/abs/2609.22135)
  - the "readable, faithful, used" triad: [2608.18768](https://arxiv.org/abs/2608.18768)
  - "Does the model use the feature?": [2610.07270](https://arxiv.org/abs/2610.07270)
  - steering non-surjectivity: [2604.09839](https://arxiv.org/abs/2604.09839)
  - steerable-but-not-decodable: [2604.02608](https://arxiv.org/abs/2604.02608)
- **Gap metrics:**
  - internal versus external knowledge: [2503.15299](https://arxiv.org/abs/2503.15299)
  - knowledge probe versus prediction probe: [2509.23782](https://arxiv.org/abs/2509.23782)
  - margin AUC versus threshold offset: [2609.04582](https://arxiv.org/abs/2609.04582)
  - semantic answer availability: [2605.22007](https://arxiv.org/abs/2605.22007)
  - Hits@k submerged knowledge: [2412.20846](https://arxiv.org/abs/2412.20846)
  - Knowledge / Symmetry / Routing / Repair quartet: [2608.12321](https://arxiv.org/abs/2608.12321)
  - Contrastive Endognostic Margin with the Jacobian lens: [2609.22219](https://arxiv.org/abs/2609.22219)
  - cross-axis routing diagnostic (CARD): [2608.20763](https://arxiv.org/abs/2608.20763)
  - Concealed Information Test: [2609.21996](https://arxiv.org/abs/2609.21996)
  - same-input counterfactual sign flip: [2606.05161](https://arxiv.org/abs/2606.05161)
- **Datasets and benchmarks:**
  - Parametric Temporal Conflict, 8,746 items: [2606.20959](https://arxiv.org/abs/2606.20959)
  - CAPE-Bench (agentic authorization): [2610.17301](https://arxiv.org/abs/2610.17301)
  - FaithConflict: [2610.00568](https://arxiv.org/abs/2610.00568)
  - TRAPSBench: [2608.13167](https://arxiv.org/abs/2608.13167)
  - quirky models: [2312.01037](https://arxiv.org/abs/2312.01037)
  - 400 clinical vignettes: [2603.18353](https://arxiv.org/abs/2603.18353)
  - CLadder anti-commonsense items: [2605.25891](https://arxiv.org/abs/2605.25891)

### KQ3. Scale and setting
- **Models above 8B with internals work** (abstract-level):
  - [2604.27167](https://arxiv.org/abs/2604.27167): 70B, 72B, 32B
  - [2605.22007](https://arxiv.org/abs/2605.22007): up to 72B, output distribution only
  - [2604.19117](https://arxiv.org/abs/2604.19117): "frontier scale" open weights
  - [2608.12321](https://arxiv.org/abs/2608.12321): GPT-OSS-20B and Qwen3-14B, per a search snippet
  - [2610.02702](https://arxiv.org/abs/2610.02702): Qwen3.6-27B
  - [2605.07984](https://arxiv.org/abs/2605.07984): Gemma-3-27B
  - [2605.03258](https://arxiv.org/abs/2605.03258): up to 14B
  - [2502.03407](https://arxiv.org/abs/2502.03407): Llama-3.3-70B
  - [2610.17301](https://arxiv.org/abs/2610.17301): "frontier open-weight agents"
  - Most other papers are 8B or smaller.
- **Settings:** mostly single-turn factual QA or MCQ. Agentic or multi-turn work with internals: [2610.17301](https://arxiv.org/abs/2610.17301), [2610.17444](https://arxiv.org/abs/2610.17444), [2610.02702](https://arxiv.org/abs/2610.02702), [2610.08129](https://arxiv.org/abs/2610.08129), [2610.19679](https://arxiv.org/abs/2610.19679), [2607.06503](https://arxiv.org/abs/2607.06503), [2604.19775](https://arxiv.org/abs/2604.19775), [2608.02657](https://arxiv.org/abs/2608.02657), [2511.10688](https://arxiv.org/abs/2511.10688).

### Direction 2 (geometry changes when the model acts against its knowledge)
- **Knowledge and prediction subspaces misalign on errors; realigning them helps:** [2509.23782](https://arxiv.org/abs/2509.23782)
- **Compliance under counterfactual context** is mostly an angular (direction) change, not norm dilution, in 2 of 3 architectures (Qwen-3-4B, Llama-3.1-8B, GLM-4-9B): [2602.04918](https://arxiv.org/abs/2602.04918)
- **A shared "compliance subspace"** for sycophancy and conformity; the abstract calls the transition to compliance "deterministic": [2601.11563](https://arxiv.org/abs/2601.11563)
- **Representational shifts in early to mid layers** under deceptive instructions: [2507.22149](https://arxiv.org/abs/2507.22149)
- **Opinion-agreement writes into a direction orthogonal to the factual-wrongness direction:** [2604.19117](https://arxiv.org/abs/2604.19117)
- **Correctness geometry:**
  - correctness as a direction: [2610.04512](https://arxiv.org/abs/2610.04512)
  - confidence manifold: [2602.08159](https://arxiv.org/abs/2602.08159)
  - alignment tuning reshaping sycophancy representations: [2607.18114](https://arxiv.org/abs/2607.18114)

### Direction 3 (gradual versus abrupt shift in multi-step or multi-turn settings)
- **Abrupt commitment:**
  - In agents, decisions sharpen around a commitment point: [2610.17444](https://arxiv.org/abs/2610.17444)
  - Answers are often fixed before chain-of-thought is generated, and steering the pre-committed answer works: [2603.01437](https://arxiv.org/abs/2603.01437), [2604.22266](https://arxiv.org/abs/2604.22266)
  - A finite-answer commitment theory: [2605.06723](https://arxiv.org/abs/2605.06723)
  - Eventual correctness is predictable after a few reasoning tokens: [2511.14773](https://arxiv.org/abs/2511.14773)
- **Stepwise, Markov-like multi-turn accuracy drift, with probes predicting answer changes:** [2511.10688](https://arxiv.org/abs/2511.10688)
- **Within-response degradation grows with scale ("snowball"):** [2607.18292](https://arxiv.org/abs/2607.18292)
- **Probe persistence across hops does not separate correct from incorrect outcomes:** [2608.07528](https://arxiv.org/abs/2608.07528)
- **Gap:** no paper found here tracks the decodable correct signal turn by turn through an agentic failure and fits a gradual-versus-abrupt model. [2610.17444](https://arxiv.org/abs/2610.17444) and [2511.10688](https://arxiv.org/abs/2511.10688) come closest.

---

## 1. QUERY LOG

F1 = know-do / knowing-doing / knows but does not say. F2 = knowledge-prediction gap / probe beats output. F3 = latent or hidden knowledge, ELK, know more than they show. F4 = decoded but unused, represented but not used, err knowingly. "fate" = a fate word combined with a family term. "dir2" and "dir3" = direction-specific queries.

**Counting.** "New" counts papers not already in my list. Through Q130 I counted every on-topic paper, including Peripheral ones. From Q131 I counted strictly: Core, Supporting, or clearly on-topic Peripheral only.

**Source labels.**
- arXiv API = export.arxiv.org with strict quoted-phrase field queries.
- arXiv web = the arxiv.org/search page. This often matched loosely and returned the most recent listings.

**Failed queries.** Queries that came back empty because of rate limits are shown in their own rows and were not counted toward the stop rule.

| # | Query | Source | Family | Hits looked at | New | New papers added |
|---|---|---|---|---|---|---|
| Q1 | "knowing-doing gap" OR "know-do gap" OR "knowing doing gap" (abs) | arXiv API | F1 | 12 | 4 | 2504.16078, 2605.14038, 2607.13618, 2505.13426 |
| Q2 | "knowledge-prediction gap" OR "knowledge prediction gap" | arXiv API | F2 | 1 | 1 | 2509.23782 |
| Q3 | "knowing-doing" AND "language model" | arXiv API | F1 | 7 | 1 | 2509.10625 |
| Q4 | "knows but" AND "does not" AND LLM | arXiv API | F1 | 20 | 4 | 2604.28031, 2510.01070, 2511.13240, 2506.07458 |
| Q5 | "knows the answer" AND internal AND "language model" | arXiv API | F1 | 3 | 2 | 2503.15299, 2608.06571 |
| Q6 | ti:know AND ti:say AND internal | arXiv API | F1 | 4 | 3 | 2607.08046, 2609.22219, 2607.22034 |
| Q7 | knowing AND doing AND "hidden states" AND LLM | arXiv API | F1 | 3 | 2 | 2602.07812, 2609.33010 |
| Q8-13 | 6 strict-phrase combinations (probe outperform model output; knowledge subspace / prediction subspace; probing accuracy exceeds answer; linear probe higher accuracy than answers; generation-discrimination gap internal; "unfaithful to" "internal knowledge") | arXiv API | F2 | ~10 | 1 | 2502.15543 |
| Q14 | LLM probe accuracy exceeds model's own answer "internal knowledge" | WebSearch | F2 | 9 | 5 | 2406.14673, 2512.20949, 2511.14773, 2604.11662, medRxiv 2025.09.17.25336018 |
| Q15 | "knows but does not say" language model internal representations | WebSearch | F1 | 9 | 2 | 2604.14180, 2510.09033 |
| Q16 | "know more than they show" LLM hidden states | WebSearch | F3 | 10 | 1 | 2410.02707 |
| Q17 | "err knowingly" OR "errs knowingly" | WebSearch | F4 | 10 | 0 | none |
| Q18 | "latent knowledge" AND "language model" | arXiv API | F3 | 30 | 11 | 2601.18468, 2605.28825, 2606.31257, 2606.00739, 2404.12957, 2610.15277, 2407.18712, 2312.10029, 2212.03827, 2411.02433, 2503.17755 |
| Q19 | ti:"eliciting latent knowledge" | arXiv API | F3 | 6 | 4 | 2606.12268, 2312.01037, 2610.26698, 2505.14352 |
| Q20 | "hidden knowledge" AND LLM AND probe | arXiv API | F3 | 1 | 0 | none |
| Q21 | ti:"latent knowledge" | arXiv API | F3 | 30 | 3 | 2602.05532, 2606.02860, 2605.12908 |
| Q22 | "discovering latent knowledge" | arXiv API | F3 | 2 | 0 | none |
| Q23 | quirky AND "latent knowledge" | arXiv API | F3 | 2 | 0 | none |
| Q24 | decodable AND "not used" AND "language model" | arXiv API | F4 | 25 | 5 | 2609.02438, 2601.03914, 2608.17843, 2601.15111, 2606.05753 |
| Q25 | "linearly decodable" AND behavior AND LLM | arXiv API | F4 | 13 | 10 | 2605.05715, 2607.05013, 2605.25151, 2605.08942, 2605.28553, 2609.37737, 2608.21766, 2605.29889, 2603.04317, 2510.08915 |
| Q26 | OR of quoted phrases ("represented but not used" etc.) | arXiv API | F4 | 25 (junk) | 0 | none |
| Q27-31 | 5 F4 retries | arXiv API | F4 | 0 (rate-limited) | 0 | none (failed) |
| Q32 | LLM internally encodes correct answer but outputs wrong; causal patching | WebSearch | F4 | 10 | 2 | 2608.12321, 2604.22271 |
| Q33 | "hidden knowledge" LLM probe answer generation gap 2026 | WebSearch | F3 | 10 | 1 | 2604.02830 (GRADE) |
| Q34 | sycophancy, internally knows correct, overridden, probe | WebSearch | F1, fate | 9 | 4 | 2601.16644, 2508.02087, 2607.07003, 2601.21183 |
| Q35 | agent knows action is wrong but does it anyway; internal probe | WebSearch | F1, agentic | 9 | 3 | 2610.08670, LessWrong "abstention geometry", 2602.04197 |
| Q36 | interpretability without actionability | OpenAlex | F4 | 10 | 1 | 2603.18353 |
| Q37 | knowledge-action gap, internal representations | OpenAlex | F1/F4 | 20 | 5 | 2608.20763, 2610.17444, 2609.32653, Brittle Minds (Findings EMNLP 2025), Intention-like representations (Phil Stud 2026) |
| Q38 | knowing-doing gap large language models | OpenAlex | F1 | 20 | 2 | Knowing Is Not Doing (Zenodo 2026), Do LLMs Know What They Don't Know (Findings ACL 2023) |
| Q39 | knows but does not say, hidden states | OpenAlex | F1 | 20 | 5 | 2609.22151, 2609.35591, Curious Case of Hallucinatory (Un)answerability (EMNLP 2023), 2411.14257, 2502.02013 |
| Q40 | knowledge prediction gap probe | OpenAlex | F2 | 20 | 2 | Knowledge Acquisition-Utilization Gap (Findings EMNLP 2023), Probing for Arithmetic Errors (EMNLP 2025) |
| Q41 | probe accuracy higher than model accuracy, hidden states, correct answer | OpenAlex | F2 | 20 | 10 | medRxiv probing (dup removed), 2610.08018, 2604.23351, 2609.01081, 2610.22046, 2610.04512, 2609.37700, 2606.02628, 2607.26119, 2609.04336 |
| Q42 | internal knowledge unused, final generation, residual stream | OpenAlex | F2 | 20 (junk) | 0 | none |
| Q43 | hidden factual knowledge, fail to generate | OpenAlex | F3 | 16 | 2 | Mechanistic Understanding of Non-Factual Hallucinations (Findings EMNLP 2024), 2609.24238 |
| Q44 | latent knowledge ignored / overridden output | OpenAlex | F3, fate | 20 | 7 | 2510.16727, 2606.05161, 2608.13167, 2602.22508, 2608.10430, 2608.03035, Pixels vs Priors (EMNLP 2025) |
| Q45 | model encodes correct answer but generates incorrect answer | OpenAlex | F4 | 20 | 1 | Seeing Clearly, Answering Incorrectly (CVPR 2025) |
| Q46 | correct answer suppressed, late layers | OpenAlex | F4, fate | 20 | 10 | 2603.18280, 2605.03052, 2604.22266, 2608.19211, 2604.27167, IJCAI-2026/153, 2609.29278, 2606.06890, 2609.30993, Confidence Regulation Neurons (NeurIPS 2024) |
| Q47 | internal representation bypassed output, causal | OpenAlex | F4, fate | 20 | 3 | 2602.03994, 2605.05593, 2609.22260 |
| Q48 | truthful representation override output, intervention | OpenAlex | F3, fate | 20 | 5 | 2610.00568, 2609.04166, 2505.14685, 2609.38810, 2602.08311 |
| Q49 | linear probe recovers correct answer when wrong; steering fails | OpenAlex | F2 | 20 | 14 | 2609.24821, 2605.03258, 2605.07990, 2603.01437, 2604.13386, 2606.11599, 2510.13290, 2607.11414, 2602.08159, 2606.20959, 2607.08456, 2605.06723, 2602.14143, 2412.00967 |
| Q50 | encode but fail to express | OpenAlex | F4 | 20 | 2 | 2610.20401, 2605.02028 |
| Q51 | representation-behavior dissociation, probe, causal | OpenAlex | F4 | 20 | 6 | 2609.22135, 2609.18080, 2609.11020, 2608.18768, 2609.04582, 2602.04931 |
| Q52 | know when they are wrong, hidden states, do not correct | OpenAlex | F1 | 20 | 6 | 2609.36783, 2610.01244, 2609.27220, 2608.17124, 2604.26139, 2606.29646 |
| Q53 | agent internal state knows failure but continues; probe | OpenAlex | F1, agentic | 20 | 12 | 2610.17301, 2610.19679, 2610.07753, 2606.22864, 2607.06503, 2410.14516, 2605.11436, 2610.18614, 2610.05241, 2609.03438, Still No Lie Detector (Phil Stud 2024), 2606.08508 |
| Q54-55 | "encoded but not" / "despite encoding" | arXiv API | F4 | 0 (rate-limited) | 0 | none (failed) |
| Q56 | "represented but" (abs) | arXiv API | F4 | 25 | 3 | 2605.22007, 2607.23379, 2607.04525 |
| Q57 | know but don't tell, probing | OpenAlex | F1 | 20 | 4 | 2602.02605, 2606.24251, 2512.24661, ITI 2306.03341 |
| Q58 | hallucination despite knowing the answer | OpenAlex | F3 | 20 | 5 | TruthX (ACL 2024), Trust Me I'm Wrong / CHOKE (Findings EMNLP 2025), 2607.18292, 2601.05201, Geometry of LM Hallucination (Harvard) |
| Q59 | knowledge subspace misalignment, residual stream, MC | OpenAlex | F2 | 20 | 1 | 2609.34970 |
| Q60 | eliciting latent knowledge, probes, truthfulness, causal | OpenAlex | F3 | 20 | 7 | 2310.18168, Universal Truthfulness Hyperplane (EMNLP 2024), Adaptive Activation Steering (WWW 2025), 2507.22149, Probing the Geometry of Truth (Findings ACL 2025), 2603.12270, Neighborhood Consistency (ACL 2026) |
| Q61 | knows the answer internally but fails multi-step; probe intermediate (OpenReview) | WebSearch | dir3 | 9 | 4 | 2411.15862, 2412.01113, 2602.02103, ACL 2026.acl-long.536 |
| Q62 | ACL Anthology 2026: internal knowledge not used in output; probing, causal | WebSearch | F2/F4 | 10 | 3 | 2605.25891 (Causal Tongue-Tie), Aligned Probing (TACL 2026), CausalGaze (Findings ACL 2026) |
| Q63 | lying: internal truth while outputting falsehood; probe | WebSearch | F3 | 10 | 1 | Quantized but Deceptive? (EMNLP 2025) |
| Q64 | title lookups (Causal Tongue-Tie; TruthfulnessEval) | OpenAlex | F2/F3 | 6 | 2 | 2609.31401, 2605.09314 (serendipitous) |
| Q65 | knowing-doing gap, mechanistic interpretability, hidden states | OpenAlex | F1 | 20 | 5 | 2608.07528, 2608.22916, Knowing-but-Doing (Findings ACL 2026), 2608.07261, 2609.00760 |
| Q66 | latent knowledge suppressed output, alignment fine-tuning, probe | OpenAlex | F3, fate | 20 | 1 | 2604.14324 |
| Q67 | probe decodes answer on model errors; steering repairs | OpenAlex | F4 | 20 | 5 | 2609.11216, 2603.16331, 2610.17663, 2610.07270, materials-science reading/steering (IOP 2026) |
| Q68 | persuasion: correct answer switches; internal representation still encodes | OpenAlex | fate | 20 | 3 | 2605.23932, Prompt Leakage Intents (EMNLP 2025), 2603.16017 |
| Q69-72 | 4 retries (F1/F3/F4) | arXiv API | F1/F3/F4 | 0 (blocked) | 0 | none (failed) |
| Q73 | "encoded but not" | arXiv web | F4 | 50 (mis-parsed) | 1 | 2610.15412 |
| Q74 | "despite encoding" | arXiv web | F4 | 8 | 0 | none |
| Q75 | language model "knows" "but" probe hidden states wrong answer | arXiv web | F1 | 0 | 0 | none |
| Q76 | decodable "not used" language model | arXiv web | F4 | 50 (loose) | 4 | 2610.03445, 2610.08413, 2610.12361, 2610.00656 |
| Q77 | "knowing-doing" (all fields) | arXiv web | F1 | 16 | 0 | none |
| Q78 | "latent knowledge" LLM output gap | arXiv web | F3 | 1 | 1 | 2412.20846 |
| Q79 | "hidden knowledge" language model | arXiv web | F3 | 18 | 2 | 2603.14672, 2403.09037 |
| Q80 | "knowledge-prediction gap" (all fields) | arXiv web | F2 | 1 | 0 | none |
| Q81 | "know more than" (all fields) | arXiv web | F3 | 25 | 7 | 2610.16135, 2605.02735, 2604.03877, 2511.10811, 2505.08662, 2512.04957, 2608.00583 |
| Q82 | "know but" (all fields) | arXiv web | F1 | 50 (loose) | 5 | 2610.08129, 2610.02902, 2610.06673, 2610.12610, 2610.05682 |
| Q83 | "knows but" (all fields) | arXiv web | F1 | 50 (loose) | 1 | 2610.18684 |
| Q84 | "probe" "outperforms" "own answers" | arXiv web | F2 | 1 | 0 | none |
| Q85 | "err knowingly" | arXiv web | F4 | 1 | 1 | 2508.18760 |
| Q86 | "act on what they know" (title) | arXiv web | F1 | 2 | 1 | 2505.12060 |
| Q87 | "fail to use" (title) | arXiv web | F4 | 3 | 3 | 2610.21731, 2610.25470, 2609.00868 |
| Q88 | "encode but" (title) | arXiv web | F4 | 50 (loose) | 1 | 2610.06235 |
| Q89 | "represent but" (title) | arXiv web | F4 | 50 (loose) | 4 | 2610.02702, 2610.06576, 2609.16247, 2609.22850 |
| Q90 | "knowing is not" (title) | arXiv web | F1 | 50 (loose) | 6 | 2610.27648, 2610.13910, 2610.27257, 2610.22594, 2610.04403, 2610.19226 |
| Q91 | "cannot say" (title) | arXiv web | F1 | 4 | 1 | 2607.23271 |
| Q92 | "latent knowledge" (title) | arXiv web | F3 | 19 | 0 | none |
| Q93 | "eliciting latent knowledge" (title) | arXiv web | F3 | 6 | 0 | none |
| Q94 | "hidden knowledge" (title) | arXiv web | F3 | 9 | 2 | 2602.06248, 2510.03263 |
| Q95 | OpenReview ICLR 2026: probe internal knowledge, answer gap, intervention | WebSearch | F2 | 9 | 4 | 2608.20569, 2603.17199, Residual Stream Under Knowledge Conflicts (NeurIPS-W 2024), 2604.09885 |
| Q96 | "knowledge vs prediction" probe subspace MC 2026 | WebSearch | F2 | 9 | 0 | none (confirms KAPPA at ICML 2026) |
| Q97 | reasoning model hidden states know answer; overthinking; commitment | WebSearch | dir3 | 9 | 2 | 2504.05419, 2605.01853 |
| Q98-99 | title lookups ("What do your logits know"; knowledge-conflict residual stream) | OpenAlex | F2 | 5 | 1 | 2606.08705 |
| Q100-103 | overthinking the truth; logit-lens overwritten; amnesic probing; sandbagging probe | OpenAlex | F4/F1 | 0 (budget exhausted) | 0 | none (failed) |
| Q104 | overthinking the truth | arXiv web | F4, fate | 6 | 1 | 2307.09476 |
| Q105 | logit lens correct answer overwritten final layers | arXiv web | F4, fate | 0 | 0 | none |
| Q106 | amnesic probing | arXiv web | F4 | 3 | 3 | 2006.00995, 2506.11673, 2304.10346 |
| Q107 | sandbagging probe | arXiv web | F1/F3 | 10 | 7 | 2609.21996, 2610.27577, 2609.02893, 2512.07810, 2602.18971, 2509.21344, 2606.28863 |
| Q108 | probe accuracy, model accuracy, gap, layers | arXiv web | F2 | 45 | 6 | 2608.20999, 2604.09364, 2604.02608, 2604.04411, 2608.08514, 2605.27772 |
| Q109 | correct answer decodable, incorrect output | arXiv web | F2/F4 | 16 | 5 | 2607.16799, 2604.00778, 2509.23417, 2510.26277, 2602.09486 |
| Q110 | internal correctness signal unused | arXiv web | F4 | 1 | 1 | 2606.05680 |
| Q111 | probe steering gap correct answer | arXiv web | F4 | 7 | 2 | 2607.00158, 2607.09544 |
| Q112 | knowledge internal external gap language model | arXiv web | F3 | 60 | 7 | 2608.02657, 2604.22193, 2601.11563, 2601.09445, 2505.12452, 2603.18614, 2501.11721 |
| Q113-117 | 5 long-phrase dir2/dir3/override queries | arXiv web | dir2/dir3 | 2 (possibly rate-limited) | 0 | none (not counted) |
| Q118 | sycophancy probe hidden states | arXiv web | fate | 1 | 1 | 2607.18114 |
| Q119 | multi-turn sycophancy hidden states | arXiv web | dir3 | 0 (possibly rate-limited) | 0 | none (not counted) |
| Q120 | belief-action gap | arXiv web | F1 | 50 | 5 | 2608.24691, 2608.20975, 2608.02358, 2605.00226, 2607.04334 |
| Q121 | error detection hidden states agents | arXiv web | F1, agentic | 5 | 2 | 2610.16141, 2605.13851 |
| Q122 | truth direction sycophancy | arXiv web | fate | 11 | 5 | 2604.19117, 2603.26410, 2602.04918, 2606.05183, 2505.13995 |
| Q123 | "know they are wrong" | arXiv web | F1 | 7 | 3 | 2604.19781, 2603.06604, 2506.23464 |
| Q124-127 | "knows it is wrong"; override circuit; competing signal; suppression neurons | arXiv web | F1/fate | 0 (empty pages, rate-limited) | 0 | none (failed) |
| Q128 | "suppression neurons" OR "suppression heads": correct answer overridden in final layers | WebSearch | fate | 9 | 3 | 2406.19384, 2403.19521, Truth Neurons (id not resolved) |
| Q129 | multi-turn: correct belief degrades gradually or abruptly; hidden states probe | WebSearch | dir3 | 9 | 4 | 2603.25187, 2511.10688, 2604.05397, 2508.05625 |
| Q130 | "context overrides" parametric knowledge; probe still encodes | WebSearch | fate | 9 | 2 | 2402.16061, 2503.10996 |
| Q131 | "latent knowledge" not used / ignored / bypassed; causal 2026 | WebSearch | F3/F4 | 10 | 3 | 2606.12689, 2605.07984, 2606.01243 |
| Q132 | "knowing-doing gap" interpretability probe steering agents 2026 | WebSearch | F1 | 9 | 3 | 2604.09839, 2607.03598, 2604.19775 |
| Q133 | "decodable" "causally" "unused" probe ablation | WebSearch | F4 | 10 | 2 | 2604.22128, 2605.00607 |
| Q134 | knowledge-prediction gap, free-form generation 2026 | WebSearch | F2 | 9 | 0 | none |
| Q135 | Anthropic Biology of an LLM: hallucination, known entity, motivated reasoning | WebSearch | lab blog | 9 | 1 | Anthropic "On the Biology of a Large Language Model" (2025-03) |
| Q136 | Transluce / Goodfire / Apollo blog: internal correct, output wrong | WebSearch | lab blog | 9 | 0 | none (Goodfire reward-hacking monitors, news-only) |
| Q137 | detecting strategic deception, linear probes, Llama-3.3-70B | WebSearch | F3 | 9 | 1 | 2502.03407 |
| Q138-140 | harmfulness vs refusal; reward-hacking awareness probe; verbalized vs internal confidence | WebSearch | F1/F3 | not run | n/a | WebSearch budget exhausted; sweep halted |

**Per-family tally** (valid queries only, approximate). F1 about 30 queries, F2 about 20, F3 about 25, F4 about 30, fate-combined about 15, dir2/dir3 about 6. No family reached 10 consecutive empty queries. The longest empty runs were Q20 to Q23 (F3: 3 of 4 empty) and Q74 to Q80 (mixed: 4 of 7 empty).

---

## 2. PAPER TABLE

Column notes:
- "in CSV" = whether a distinctive title fragment appears in papers_master.csv (grep -i). "no?" means the fragment was too generic to be sure.
- "n/r" = not read (title or snippet only).
- Models and scale come from abstracts or snippets only.
- "C/corr": C = causal intervention reported in the abstract; corr = correlational (probe, lens, output distribution); beh = behavioral only.

### 2a. Core

| Title | ID / venue | Date | Models, scale | Setting | C/corr | Fate / direction | Tier | in CSV | Why (one line) |
|---|---|---|---|---|---|---|---|---|---|
| Bridging the Knowledge-Prediction Gap in LLMs on MCQs (KAPPA) | 2509.23782 / ICML 2026 | 2025-09 | Llama-3.1-8B-Inst, Qwen2.5-7B-Inst (≤8B) | MCQ (BBH, MMLU, ARC-C, TruthfulQA) plus free-form transfer | C (subspace-alignment intervention) | Misaligned knowledge vs prediction subspaces (readout); dir2 geometry | Core | yes | Knowledge probe beats the answer; the fix realigns the two subspaces |
| Inside-Out: Hidden Factual Knowledge in LLMs | 2503.15299 | 2025-03 | 3 open-weight LLMs (n/r; secondary sources name Llama-3, Mistral, Gemma-2) | Closed-book QA | corr | Present but unexpressed; some answers never sampled in 1,000 draws | Core | yes | Formal internal vs external knowledge metric; 40% average relative gap |
| LLMs Know More Than They Show | 2410.02707 / ICLR 2025 | 2024-10 | n/r | QA, several datasets | corr | Correct answer encoded, wrong answer generated; truthfulness signal sits on the exact answer tokens | Core | yes | Canonical evidence of a discrepancy between internal encoding and output |
| Insights into LLM Long-Context Failures: When Transformers Know but Don't Tell | 2406.14673 / Findings EMNLP 2024 | 2024-06 | n/r | Long-context retrieval | corr | Position encoded but not used | Core | no | Probe locates the key information; the generated answer does not use it |
| Interpretability without actionability | 2603.18353 | 2026-03 | Qwen2.5-7B-Inst, Steerling-8B | 400 clinical triage vignettes | C (4 methods: concept-bottleneck steering, SAE steering, logit lens with patching, TSV steering) | Gap persists under intervention (bypass or entanglement) | Core | no | 98.2% probe AUROC vs 45.1% sensitivity; steering fixes ≤24% |
| LLMs Know the Constraint But Do Not Use It | 2608.12321 | 2026-05 | 14 models behaviorally; probes on 2 open-weight models (snippet: GPT-OSS-20B, Qwen3-14B; >8B) | Pragmatic constraint reasoning | C (activation patching, donor repair) | Routing failure (bypass); repairable in one model only | Core | yes | Knowledge / Symmetry / Routing / Repair quartet diagnostic |
| When Truth Is Overridden: Internal Origins of Sycophancy | 2508.02087 / AAAI 2026 | 2025-08 | Several model families (n/r) | Factual Qs with user opinion | C (logit lens plus patching) | Override; two-stage emergence (dir2) | Core | no | Late-layer preference shift, then representational divergence |
| LLMs Know They're Wrong and Agree Anyway: Shared Sycophancy-Lying Circuit | 2604.19117 | 2026-04 | 12 open-weight models, small to frontier scale (>8B) | Sycophancy, factual lying, instructed lying | C (head silencing, path patching) | Override via a deference circuit that reads the "wrong" signal; RLHF keeps the circuit | Core | no | Error signal decodable and causally routed into agreement |
| How LLMs Are Persuaded: A Few Attention Heads, Rerouted | 2605.09314 | 2026-05 | n/r | Controlled MCQ with persuasive context | C | Override or rerouting via an option-routing feature | Core | no | Causal pathway from keyword heads to routing feature to decision heads |
| When Decodability Is Not Enough: Logical Validity Representations | 2609.02438 | 2026-09 | 5 open-weight models (n/r) | Logical verification | C (probe-direction interventions vs random) | Bypass: decodable on wrong items, weak causal effect | Core | no | Represent, express and use are distinct |
| Encoded but Not Actionable: Decode-Generate-Steer Gap | 2608.17843 | 2026-08 | 6 frozen decoder-only LLMs | CAD geometric constraints | C (activation influence, steering) | Decodable but not actionable; random-init control | Core | no | Includes a random-init control showing some decodability needs no learning |
| Decodable but Not Corrected by Fixed Residual-Stream Linear Steering | 2605.05715 | 2026-05 | Qwen2.5-7B plus one other (n/r) | Medical QA, overthinking in CoT | C (29 steering configurations, LEACE) | Entanglement: direction overlaps task computation | Core | no | Strong null result for steering; probe still useful for abstention |
| Knowledge Knows, Verbalization Tells | 2607.05013 | 2026-07 | Multiple LLMs (n/r) | Math solvability, fabrication | C (steering) | Transformed: separate knowledge and verbalization directions | Core | no | Fabrication tracks verbalization, not knowledge |
| When Do Internal Probes Beat Reading the Answer? | 2609.04582 | 2026-09 | 0.6B, plus a five-model three-family factorial; 13x scale range (≤8B) | Logical conclusion verification | corr (margin and threshold analysis) | Readout: verdict reaches the logits, then a threshold offset erases it | Core | yes | "Single scalar" failure; scale is non-monotone |
| Legible Failures: Detecting and Repairing In-Context Binding Errors | 2609.11216 | 2026-09 | 16 public checkpoints | In-context entity binding | C (probe-guided steering) | Present but unused; repairable | Core | no | Probe beats chance on errors; label-free steering raises accuracy |
| Decodable In-Context State and Model Output Across Training | 2609.31401 | 2026-09 | Pythia checkpoints plus post-training | Binding task | C (steering across checkpoints) | Training dynamics; method caveat | Core | no | Proof that decodability on errors cannot by itself show discarded information |
| Hallucination as Commitment Failure: Larger LLMs Misfire Despite Knowing | 2605.22007 | 2026-05 | Qwen and Llama, 0.8B-72B (>8B) | Factual QA, free generation | corr (output distribution) | Present but dispersed; rate rises with scale | Core | no | 16-47% of hallucinations have substantial mass on the correct concept |
| The Agent Knew: Internal States Forecast Unauthorized Actions | 2610.17301 | 2026-10 | Frontier open-weight agents, 4 families | Agentic (CAPE-Bench, tau²-bench) | C-ish (gating actions on the forecast) | Decodable forecast before the violation; fix by gating | Core | yes | Agentic know-do test with a fix; beats CoT monitors |
| The belief-action gap in language model agents | 2610.17444 | 2026-10 | n/r | Agentic grid worlds | Mixed (probes, IRL, prefix transplant) | Represented more reliably than acted on; dir3 commitment point | Core | no | Direct agentic test of belief vs action |
| Silent Dissent: Agents That Yield Still Represent Their Original Premise | 2610.02702 | 2026-10 | Qwen3.5-4B, Qwen3.6-27B, Gemma-4-E4B (>8B) | Multi-agent debate, two-hop | corr (J-lens, logit lens; pre-registered) | Override at output; original premise kept | Core | no | Multi-agent conformity with internals |
| Do LLMs Act on What They Know? Partner Representations to Cooperative Actions | 2610.08129 | 2026-10 | 8 LLMs; Qwen3-8B case study | Hanabi-like cooperation | Probes plus activation transplant | Represented but not used; action translation helps | Core | no | Probe-predicted convention vs action |
| How a Cooperative-Override Circuit Suppresses Nash Play | 2604.27167 | 2026-04 | Llama-3-70B, Qwen2.5-32B/72B, Llama-3-8B (>8B) | Prisoner's Dilemma | C (steering in 8B) | Override in late layers | Core | no | Layerwise contest between Nash and cooperation; "correct" is task-defined |
| From Early Encoding to Late Suppression: Character Counting | 2604.00778 | 2026-04 | n/r (snippet: up to about 9B) | Character counting | C (component analysis; negative circuit in the final MLP) | Suppressed late | Core | no | Count encoded in the residual stream but not moved into probability space |
| Causal Tongue-Tie | 2605.25891 | 2026-05 | n/r | CLadder causal Yes/No | corr | Present but not said | Core | no | Probe ~0.97 vs output ~0.5; separates no-signal from can't-say |
| Right Knowledge, Wrong Answer: Parametric Temporal Conflict | 2606.20959 | 2026-06 | 4 open-weight models, 3 families | Temporal fact updates | C (patching flips 72-85%; steering vs random) | Competing outdated fact (override) | Core | no | 8,746-item benchmark |
| LLMs Know More About Numbers than They Can Say | 2602.07812 | 2026-02 | Several smaller open LLMs | Numeric comparison | corr | Present but not used | Core | no | Probe ranking >90% vs 50-70% behavior |
| The Knowing-Saying Gap: Probes See Errors that Confidence Misses | 2608.07528 | 2026-07 | Several families incl. reasoning models, Llama-3 | Multi-hop arithmetic with corruption | corr plus probe-based interventions | Detects corruption but uninformative about correctness; dir3 hops | Core | yes | Pre-registered null on probe persistence |
| Readable but Not Controllable: Medical LLM Hallucination | 2607.00158 | 2026-06 | 4 open models | Medical QA | C (neuron interventions) | Readable, distributed, not controllable | Core | yes | Detection works; control does not |
| The Right Answer, the Wrong Direction: Counting | 2605.03258 | 2026-05 | Pythia, Qwen3, Mistral 0.4B-14B (>8B) | Counting | C (output-head row update) | Inaccessible to readout (orthogonal to unembedding) | Core | no | Clean "readout geometry" fate with a 36,864-parameter fix |
| Knowing, and Saying It Only When Asked: LLM Endognostics | 2609.22219 | 2026-09 | Minerva-7B-Instruct | Professional-risk pairs; presupposed falsehoods | C (residual manipulation) | Internal differentiation not expressed | Core | no | Jacobian-lens margin metric |
| Simulated Adoption: Magnitude vs Direction in In-Context Conflict | 2602.04918 | 2026-02 | Qwen-3-4B, Llama-3.1-8B, GLM-4-9B | Counterfactual-context conflicts | corr (geometry) | dir2: angular change, not norm dilution | Core | no | Geometry of suppression under compliance |
| Human-like Social Compliance: Signal Competition Dynamics | 2601.11563 | 2025-12 | 15 LLMs behaviorally; probing on 3 | Sycophancy and conformity | corr | Override; "compliance subspace"; abstract calls the transition deterministic (dir3) | Core | no | Competing-signal account |
| They Infer What You Meant: Intent Represented More Reliably Than Acted On | 2607.03598 | 2026-07 | n/r | Communicative intent | Probe plus steering (aggregator: steering recovers ~98%) | Readout failure; decodable layers before behavior | Core | no | "Represent-readout gap"; steering repair figure not checked at the primary source |
| When Truthful Representations Flip Under Deceptive Instructions? | 2507.22149 / EMNLP 2025 | 2025-07 | Llama-3.1-8B-Inst, Gemma-2-9B-Inst (>8B) | Factual True/False under instructions | corr (probes, SAE) | Transformed; dir2 early-mid layer shift | Core | yes | Representation shift under deception |
| Overthinking the Truth: How LMs Process False Demonstrations | 2307.09476 / ICLR 2024 | 2023-07 | n/r | Few-shot classification | C (head ablation) | Override after a critical layer | Core (foundational) | no | Classic "correct early, overridden late" result |
| MCAIR: Monitoring and Correcting Agent Unfaithfulness via Internal Representations | 2610.19679 | 2026-10 | n/r | Agentic tool use, unfaithful reports | C-ish (correction) | Fix for agentic know-do divergence | Core | yes | Internal-representation monitor plus correction in agents |
| Why Not Act on What You Know? Self-Aware Guard Enhancement (SAGE) | 2505.12060 | 2025-05 | n/r | Safety / jailbreaks | n/r | Model discerns harm but complies; fix | Core | yes | Explicit "act on what you know" framing plus fix |
| Arbitration Failure, Not Perceptual Blindness (VLMs) | 2604.09364 | 2026-04 | 10 VLMs | Visual vs language-prior conflict | corr (layerwise logit lens) | Override by prior; encoding-grounding dissociation | Core (multimodal) | no | Failing models encode the evidence as strongly as correct ones |
| Beyond Text Following: Repairable Arbitration Reversals in Audio-LMs | 2606.05161 | 2026-06 | 5 audio-language models | Audio vs text conflict | C (same-audio counterfactual) | Represented but overridden; repairable | Core (multimodal) | no | 64.1% sign-flip metric |
| CARD: Belief-to-Action Routing Failures in VLMs | 2608.20763 | 2026-08 | Open-weight VLMs | Cooperative grid world (Relay Chain) | C (cross-axis steering) | Routing failure | Core (multimodal) | no | CARD method: steer one axis, measure another |
| Represented but Ignored: Prosodic Underuse in Audio-LMs | 2608.19211 | 2026-06 | Audio-LLMs (n/r) | Prosody understanding | C (stage-specific probe ladder) | Bypass | Core (multimodal) | no | Separates information loss, misinterpretation and non-use |
| When VLMs Destroy What They Know | 2610.13910 | 2026-10 | Multiple VLMs | Spatial relations | C-ish (controlled interventions; EICS metric) | Transformed or degraded below lexical priors | Core (multimodal) | no | Representations decay through depth |
| Answering the Unanswerable Is to Err Knowingly | 2508.18760 | 2025-08 | LRMs (n/r) | Unanswerable math | n/r (internals unclear from abstract) | Recognizes the flaw but does not abstain | Core? | yes | "Err knowingly" framing plus a mitigation |
| Missing vs. Unused Knowledge Hypothesis (patent) | 2505.12452 | 2025-05 | n/r | Patent classification | n/r | Decomposes errors into missing vs unused | Core? | no | Explicit missing-vs-unused framework |

### 2b. Supporting (method, control, metric, dataset or result directly usable)

| Title | ID / venue | Date | Models, scale | Setting | C/corr | Bears on | Tier | in CSV | Why |
|---|---|---|---|---|---|---|---|---|---|
| Discovering Latent Knowledge Without Supervision (CCS) | 2212.03827 | 2022-12 | n/r | True/false QA | corr | F3 method | Supporting (foundational) | yes | Unsupervised truth probe |
| Eliciting Latent Knowledge from Quirky LMs | 2312.01037 | 2023-12 | n/r | Quirky arithmetic | corr plus C (direction interventions) | F3 dataset and method | Supporting | no | Model organisms with known truth-vs-output divergence |
| Challenges with Unsupervised LLM Knowledge Discovery | 2312.10029 | 2023-12 | n/r | Probing | corr | Control | Supporting | no | Unsupervised probes find salient features, not knowledge |
| Amnesic Probing | 2006.00995 | 2020-06 | BERT-era | Probing | C (erasure) | Decodable vs used | Supporting (foundational) | yes | Erasure test of use |
| Improving Causal Interventions in Amnesic Probing (Mean Projection / LEACE) | 2506.11673 | 2025-06 | n/r | Probing | C | Decodable vs used | Supporting | yes | Better erasure control |
| Interventional Probing in High Dimensions (NLI) | 2304.10346 | 2023-04 | n/r | NLI | C | Decodable vs used | Supporting | no | Interventional probing method |
| Dissociating Decodability and Causal Use in Bracket-Sequence Transformers | 2604.22128 / ICML 2026 | 2026-04 | Small transformers trained on Dyck | Formal language plus SVA | C (subspace ablation, attention masking) | Decodable but not used | Supporting | yes | Clean ground-truth dissociation |
| Decodability is Not Causality (SAE decomposition) | 2609.18080 | 2026-09 | Snippet: Gemma2-9B-Inst (>8B) | True/False probe | C | Decodable vs used | Supporting | yes | Probe-aligned features need not be causal |
| Read-Best Is Not Steer-Best (omni-modal) | 2609.22135 | 2026-08 | Omni-modal LLMs | Emotion | C | Read vs steer layers | Supporting | yes | Layer-selection control |
| Readable, Faithful, Used: Dissociable Properties of Demographic Identity | 2608.18768 | 2026-08 | 1 LM | Survey simulation | C | Three-way decomposition | Supporting | yes | Method template |
| Does the Model Use the Feature? Steering vs Mechanism | 2610.07270 | 2026-10 | n/r | n/r | C | Control | Supporting | no | Steering effect does not imply use |
| Steered LLM Activations are Non-Surjective | 2604.09839 / ICLR 2026 | 2026-04 | n/r | Theory | Theory | Control | Supporting | no | Limits on reading steering as evidence of use |
| Steerable but Not Decodable: Function Vectors | 2604.02608 | 2026-04 | n/r | ICL | C | Inverse dissociation | Supporting | yes | Steerable without being decodable |
| The Answer-Basin Representation Hypothesis | 2609.24821 | 2026-09 | n/r | QA | corr/C | Control | Supporting | no | Probes and steering may track answer basins, not concepts |
| Diagnosing Correctness Probes under Self-Judgement Confounding | 2607.16799 | 2026-07 | n/r | Correctness probes | corr | Control | Supporting | no | Confound for "probe beats answer" claims |
| Linear Probes Rely on Textual Evidence | 2509.21344 | 2025-09 | n/r | Probing | corr | Control | Supporting | yes | Leakage mitigation |
| World Properties without World Models | 2603.04317 | 2026-03 | n/r | Decoding | corr | Control | Supporting | no | Distributional-association confound |
| Decodable Is Not Grounded: Vision-Ablation Arbiter | 2606.31257 | 2026-06 | VLMs | Spatial reasoning | C | Control | Supporting | yes | Probe plus steering can overstate knowledge |
| Beyond Decodability: Encoding Probe | 2605.00607 | 2026-05 | n/r | Representations | corr | Method | Supporting | no | Reverse (encoding) probe |
| Inference-Time Intervention (ITI) | 2306.03341 / NeurIPS 2023 | 2023-06 | LLaMA (n/r) | TruthfulQA | C | Fix | Supporting | yes | Truth-direction steering baseline |
| TruthX | ACL 2024 | 2024 | n/r | TruthfulQA | C (editing) | Fix | Supporting | no | "Untruthful despite knowing" editing fix |
| Trust Me, I'm Wrong: Hallucinate with Certainty Despite Knowing (CHOKE) | Findings EMNLP 2025 | 2025 | n/r | Factual QA | corr | Present but overridden with certainty | Supporting | yes | Defines high-certainty errors despite knowledge |
| Do LLMs Really Know What They Don't Know? (Internal States Reflect Recall) | 2510.09033 / Findings ACL 2026 | 2025-10 | n/r | Hallucination | corr | Counterpoint: states track recall, not truth | Supporting | yes | Absent-or-misrepresented fate |
| Do I Know This Entity? Knowledge Awareness and Hallucinations | 2411.14257 | 2024-11 | n/r (Gemma-2 family per memory; not verified) | Entity QA | C (SAE steering) | Recognition features steer refusal or hallucination | Supporting | yes | Causal SAE features of knowledge awareness |
| The Curious Case of Hallucinatory (Un)answerability | EMNLP 2023 | 2023 | n/r | Unanswerable QA | corr | Hidden states encode answerability | Supporting | no | Early "truth in hidden states" result |
| Mechanistic Understanding and Mitigation of LM Non-Factual Hallucinations | Findings EMNLP 2024 | 2024 | n/r | Factual QA | C | Mechanisms of hallucination | Supporting | no | Attribution of hallucination types |
| Are LLMs Really Not Knowledgeable? Submerged Knowledge | 2412.20846 | 2024-12 | n/r | QA | corr (output distributions) | Present in candidates | Supporting | no | Hits@k metric |
| Retrieval-Constrained Decoding Reveals Underestimated Parametric Knowledge | 2509.23417 | 2025-09 | n/r | QA | beh/decoding | Present but not output | Supporting | no | Decoding-side elicitation |
| No Answer Needed: Question-Only Linear Probes | 2509.10625 | 2025-09 | n/r | QA | corr | Correctness decodable pre-generation | Supporting | yes | Pre-answer correctness signal |
| Temporal Predictors of Outcome in Reasoning LMs | 2511.14773 | 2025-11 | n/r | MATH | corr | dir3: early predictability | Supporting | no | Correctness decodable after a few tokens |
| Reasoning Models Know When They're Right | 2504.05419 | 2025-04 | n/r | Math reasoning | corr (probe early exit) | Encode correctness but fail to exploit it | Supporting | yes | Overthinking despite a decodable signal |
| LLMs Decide Early and Explain Later | 2604.22266 | 2026-04 | Qwen3-4B | Reasoning | beh (forced completion) | dir3 commitment | Supporting | no | Answer fixed mid-CoT |
| Post-Hoc Reasoning in CoT: Decoding and Steering Pre-Committed Answers | 2603.01437 | 2026-03 | Instruction-tuned models (n/r) | Reasoning | C (steering) | dir3 commitment before CoT | Supporting | no | Pre-CoT probe plus steering |
| When Does a LM Commit? Pre-Verbalization Commitment | 2605.06723 | 2026-05 | n/r | Binary tasks | Theory/corr | dir3 | Supporting | no | Commitment-time metric |
| From Candidate Tokens to Final Outcomes: Long-Horizon Candidate Value | 2610.15412 | 2026-10 | n/r | Reasoning models | corr | dir3 | Supporting | yes | Decoding future value from representations |
| Modeling and Predicting Multi-Turn Answer Instability | 2511.10688 | 2025-11 | Gemini-1.5-Flash plus open models (n/r) | Multi-turn "think again" | corr (probes) | dir3: Markov drift | Supporting | yes | Turn-level dynamics plus probes |
| Probing the Lack of Stable Internal Beliefs | 2603.25187 | 2026-03 | n/r | Multi-turn 20-questions | corr | dir3 | Supporting | no | Turn-isolated probing method |
| Confidence Should Be Calibrated More Than One Turn Deep | 2604.05397 | 2026-04 | Llama-3.1-8B, Qwen2.5-7B | Multi-turn | beh | dir3 (ECE spike at turn 2, per snippet) | Supporting | no | Multi-turn calibration |
| Multi-turn persuasion linear probes (Jaipersaud, Krueger, Lubana) | 2508.05625 | 2025-08 | n/r | Multi-turn persuasion | corr | dir3 | Supporting | no? | Probe method for persuasion over turns |
| When Correct Beliefs Collapse: Clinical Pressure | 2605.23932 / ACL 2026 | 2026-04 | n/r | Multi-turn clinical sycophancy | beh | dir3 | Supporting | yes | Escalating-pressure dataset |
| Reliability Scales Inversely: Hallucinations Snowball Faster in Bigger LMs | 2607.18292 | 2026-06 | 3 families, 6 sizes | QA and chat logs | corr | dir3, scale | Supporting | no | Within-response degradation grows with scale |
| Metacognitive Behavioral Tuning for Multi-Hop QA | 2602.22508 | 2026-02 | n/r | Multi-hop QA | beh/training | Correct intermediate conclusion overridden (dir3) | Supporting | no | Training fix for override inside traces |
| Why Knowing Both Hops Is Not Enough | 2608.07261 | 2026-08 | Transformers trained from scratch | Symbolic two-hop | C (mechanistic) | Stored but not composed | Supporting | no | Mismatched intermediate representations |
| Do LLMs Really Think Step-by-step in Implicit Reasoning? | 2411.15862 | 2024-11 | n/r | Arithmetic | corr | dir3 | Supporting | no | Intermediate results not computed |
| Think-to-Talk or Talk-to-Think? | 2412.01113 | 2024-12 | n/r | Arithmetic | corr | dir3 timing | Supporting | no | When sub-answers emerge |
| Sycophancy Hides Linearly in the Attention Heads | 2601.16644 | 2026-01 | n/r | Pushback sycophancy | corr/C | Override | Supporting | yes | Head-level linear sycophancy signal |
| Dissociating the Internal Representations of Sycophancy | 2607.07003 / ICML 2026 | 2026-07 | Several LLMs | Factual vs opinion sycophancy | corr/C | Override subtypes | Supporting | no | Subtype probes and causal interference |
| Sycophantic Anchors | 2601.21183 / ACL SRW 2026 | 2026-01 | Reasoning models | Reasoning traces | corr | dir3 anchors | Supporting | yes | 74-85% balanced accuracy for anchor detection |
| How Alignment Tuning Shapes Representations of Sycophancy | 2607.18114 | 2026-07 | n/r | Cue-induced biases | corr | dir2 | Supporting | no | Base vs aligned representations |
| Linear Probe Penalties Reduce LLM Sycophancy | 2412.00967 | 2024-12 | n/r | Sycophancy | Training | Fix | Supporting | no | Probe-in-the-loss fix |
| Analysing the Residual Stream Under Knowledge Conflicts | NeurIPS-W 2024 (Edinburgh) | 2024-10 | n/r | Context-memory conflict | corr | Conflict signal decodable | Supporting | no | Conflict detection in the residual stream |
| Taming Knowledge Conflicts in LMs | 2503.10996 | 2025-03 | n/r | Context-memory conflict | C | Superposition of memory and context in heads | Supporting | no | Competing-signal mechanism |
| Where Knowledge Collides: Intra-Memory Conflict | 2601.09445 | 2026-01 | n/r | Parametric conflict | C (mechanistic) | Override | Supporting | no | Competing parametric facts |
| How LLMs Balance Internal Knowledge with User and Document Assertions | 2604.22193 | 2026-04 | n/r | Conflicts | n/r | Override | Supporting | no | Three-way arbitration |
| Probing Language Models on Their Knowledge Source | BlackboxNLP 2024 (2410.05817) | 2024-10 | n/r | Context vs parametric | corr | Conflict | Supporting | no | Source probe |
| Detecting Strategic Deception with Linear Probes | 2502.03407 / ICML 2025 | 2025-02 | Llama-3.3-70B-Inst (>8B) | Insider trading, sandbagging | corr | Truth decodable while the model deceives | Supporting | yes | Large-model probe evidence |
| A Lie Detector Test: Reading Knowledge a Model Won't Reveal | 2609.21996 | 2026-09 | n/r | Sandbagging, concealed answers | corr | Concealed Information Test | Supporting | yes | Method for hidden-answer detection |
| Auditing Games for Sandbagging | 2512.07810 | 2025-12 | n/r | Sandbagging | Mixed | F1 | Supporting | yes | Auditing protocol |
| Linear Probe Accuracy Scales with Model Size; Multi-Layer Ensembling | 2604.13386 | 2026-04 | Several sizes | "Know they are wrong" / deception | corr | Scale | Supporting | no | Probe scaling |
| Seamless Deception: Larger LMs Are Better Knowledge Concealers | 2603.14672 | 2026-03 | Several sizes | Concealment | corr | Scale; F3 | Supporting | no | Concealment grows with scale |
| Probe Generalization as Subspace Selection (OOD deception) | 2609.02893 | 2026-09 | n/r | Deception | corr | Method | Supporting | yes | Probe OOD generalization |
| Why Static Probes Fail: Context Determines Deception Features | 2610.27577 | 2026-10 | n/r | Deception | corr | Control | Supporting | no | Context-dependent features |
| Before the Agent Tells the Lie: Is Deception Already Represented? | 2610.06576 | 2026-10 | n/r | Agentic deception | corr | dir3, F1 | Supporting | yes | Pre-action deception signal |
| MechELK: Mechanistic Framework for Eliciting Latent Knowledge | 2605.28825 | 2026-04 | n/r | Factual and reasoning | C (SAE plus patching) | Elicitation fix | Supporting | yes | Locate, verify, extract |
| Towards Eliciting Latent Knowledge with Mechanistic Interpretability | 2505.14352 | 2025-05 | n/r | Secret-keeping models | C | F3 | Supporting | no | Model-organism ELK |
| Eliciting Secret Knowledge from LMs | 2510.01070 | 2025-10 | 3 model families, fine-tuned to deny | Secret elicitation | Mixed | F3 dataset | Supporting | yes | Testbed of models that apply knowledge but deny it |
| Towards Robust Model Organisms for ELK | 2610.26698 | 2026-10 | n/r | ELK | n/r | F3 dataset | Supporting | no | Model organisms |
| Split Personality Training | 2602.05532 | 2026-02 | n/r | Concealed misbehavior | Training | F3 | Supporting | yes | Elicitation via an alternate persona |
| Improving Preference Extraction via Classifying Probes | 2503.17755 | 2025-03 | n/r | LLM-as-judge | corr | Probe beats output | Supporting | yes | Probe outperforms generation for judging |
| Probing Hidden States for Calibrated, Alignment-Resistant Predictions | medRxiv 2025.09.17.25336018 | 2025-09 | n/r | Scientific/medical prediction | corr | Probe beats generation | Supporting | no | Probe vs generative output |
| Task-Specific KD via Intermediate Probes | 2603.12270 | 2026-02 | n/r | Reasoning | corr | Probe beats output | Supporting | no | Uses probe outputs as a better teacher |
| What do your logits know? | 2604.09885 | 2026-04 | n/r | Information leakage | corr | Information present beyond the output | Supporting | yes | Leakage view of hidden information |
| Do LLMs "know" internally when they follow instructions? | 2410.14516 | 2024-10 | n/r | Instruction following | C (steering) | Agentic F1 | Supporting | no | Instruction-following direction |
| Tool Calling is Linearly Readable and Steerable | 2605.07990 | 2026-05 | n/r | Agent tool choice | C | Agentic | Supporting | no | Tool-choice probes and steering |
| Structured but Silent: Capability Requirements in Hidden States | 2610.08018 | 2026-10 | n/r | Tool use | corr | Agentic F1 | Supporting | yes | Encoded requirement, not acted on |
| Your Agentic LLMs Secretly Encode Indirect Prompt-Injection Exposure | 2608.02657 | 2026-08 | n/r | Agentic IPI | corr | Agentic F1 | Supporting | yes | Encoded exposure despite compliance |
| When AUC 0.998 Is Not Enough (IPI probes, computer-use agents) | 2606.22864 | 2026-06 | VLM agents | Computer use | corr | Control | Supporting | no | Evaluation protocol for probes |
| Doomed from the Start: Early Abort via Probe Cascade | 2607.06503 | 2026-07 | n/r | Agent episodes | corr | dir3 | Supporting | yes | Early failure decodable |
| Conformal Interpretability of Temporal Concepts in LLM Agents | 2604.19775 | 2026-03 | n/r | ScienceWorld, AlfWorld | corr plus steering | dir3, agentic | Supporting | yes | Step-wise concept tracking |
| Where Do LLMs Decide to Break the Rules? Prompt Injection Compliance | 2609.37737 | 2026-09 | n/r | Prompt injection | C (localization) | Override | Supporting | yes | Localizes compliance decision |
| Evaluation Awareness: Representation, Verbalization, and Control | 2608.21766 | 2026-08 | n/r | Eval awareness | C | Represented vs verbalized | Supporting | yes | Same decomposition applied to another variable |
| Representation Without Control: Realization Effect | 2605.25151 | 2026-05 | n/r | Behavioral simulation | C | Bypass | Supporting | yes | Represented but not controlling |
| When Models Decide and When They Bind (MCQA) | 2601.03914 | 2026-01 | n/r | MCQ | C | Readout: solve vs bind symbol | Supporting | yes | Two-stage MCQ computation (relevant to KAPPA's setting) |
| Where Do Apparent Clinical Triage Failures Arise? MC Format Effect | 2605.29889 | 2026-05 | n/r | Clinical MCQ | C/corr | Format-induced readout failure | Supporting | no | Localizes format effect |
| Correctness Is a Direction | 2610.04512 | 2026-10 | n/r | Answer selection | corr | dir2 geometry | Supporting | no | Recoverable correctness direction |
| Locating Answer-Correctness Signals in Frozen LLMs | 2609.37700 | 2026-09 | n/r | QA | corr | Probe methodology | Supporting | no | Where correctness signals live |
| The Confidence Manifold | 2602.08159 | 2026-02 | n/r | Misconceptions | corr | dir2 geometry | Supporting | no | Geometry of correctness representations |
| Two Axes of LLM Abstention | 2607.08456 | 2026-07 | n/r | Abstention | corr | Separates correctness from answerability | Supporting | no | Two-axis decomposition |
| MERA: Mechanistic Error Reduction with Abstention | 2510.13290 | 2025-10 | n/r | Errors | C (steering) | Fix | Supporting | no | Selective steering |
| When is Your LLM Steerable? | 2606.11599 | 2026-06 | n/r | Steering | C | Control | Supporting | no | Prompt-dependence of steering |
| Watch the Model Think: On-Policy Steering Vectors | 2602.14143 | 2026-02 | n/r | Reasoning | C | Fix | Supporting | no | Success-vs-failure trajectory steering |
| Remarkable Robustness of LLMs: Stages of Inference? | 2406.19384 | 2024-06 | Several | General | C (layer ablation) | Suppression neurons in final layers | Supporting | no | Late-layer suppression stage |
| Confidence Regulation Neurons | NeurIPS 2024 | 2024 | n/r | Next-token prediction | C | Late-layer calibration | Supporting | no | Entropy and token-frequency neurons |
| Correcting Suppressed Log-Probabilities with Post-Transformer Adapters | 2603.18280 | 2026-03 | n/r | Alignment | C | Suppressed | Supporting | no | Adapter fix for suppression |
| How Language Models Process Negation | 2605.03052 | 2026-05 | Open-weight models | Negation QA | C (ablation) | Late-attention shortcut override | Supporting | no | Ablation improves accuracy |
| MLLMs Get It Right, Then Get It Wrong: Late-Layer Textual Bias | IJCAI 2026 (doi 10.24963/ijcai.2026/153) | 2026-09 | MLLMs | Vision vs text conflict | C | Override in late layers | Supporting | no | Correct mid-layers, wrong final output |
| Mechanisms of Prompt-Induced Hallucination in VLMs | 2601.05201 / ACL 2026 | 2026-01 | 3 VLMs | Object counting | C (head ablation) | Override by prompt | Supporting | no | Conformity heads |
| TRAPSBench: VLMs Encode but Fail to Express Epistemic Restraint | 2608.13167 | 2026-08 | 16 VLMs | Video physics abstention | corr | Encode but not express | Supporting | no | Benchmark plus PECS metric |
| From Knowing to Abstaining: Representation-Action Gap in VLMs | 2609.32653 | 2026-09 | VLMs | Abstention | Training | Fix | Supporting | no | Dataset to close the gap |
| Omni-Modal LLMs Know More Than They Say | 2610.16135 | 2026-10 | Omni-LLMs | Multimodal | corr (TMA metric) | Represented but not expressed | Supporting | yes | Token-level modality alignment metric |
| Latent Ordinal Evidence, Misaligned Outputs (MLLMs) | 2608.20999 | 2026-08 | MLLMs | Ordinal judgments | C (inference-time lens alignment) | Readout fix | Supporting | no | Ordinal-lens alignment |
| Responses Fall Short of Understanding (VDU) | 2604.04411 | 2026-04 | VLMs | Document understanding | corr | Internal vs response gap | Supporting | no | Multimodal know-do gap |
| The Count Is There, but Misaligned (VLMs) | 2607.09544 | 2026-07 | VLMs | Counting | C (correction) | Readout geometry | Supporting | no | Multimodal parallel of 2605.03258 |
| Knowing Isn't Always Saying: Spatial Encodings in VLMs | 2608.22916 | 2026-08 | VLMs | Spatial QA | corr/C | When encodings reach answers | Supporting | no | Layer-wise reach |
| Corrupted but Correct: VLMs Lie to Themselves Internally | 2610.03445 | 2026-10 | Qwen2.5-VL-7B | Adversarial captions | C | Train/inference dissociation | Supporting | no | Reverse dissociation (teacher-forced vs free generation) |
| Encoded but Not in Control: Grounding Gap in VLA Policies | 2610.06235 | 2026-10 | VLA policies | Robot instruction following | C (instruction interventions) | Encoded but not controlling | Supporting | no | Embodied analog |
| Retrieved but Not Read: Search-Augmented LLMs Fail to Use Evidence | 2610.21731 | 2026-10 | 1 proprietary plus 6 open configurations | Temporal facts with retrieval | n/r | Evidence available but not used | Supporting | no | RAG analog of the bypass fate |
| When Models Know but Can't Linearly Say (code vulnerabilities) | 2610.27257 | 2026-10 | Code LLMs | Vulnerability classification | corr | Probe-limitation control | Supporting | no | Nonlinear encoding caveat |
| Speak vs. Know: Behavior and Knowledge Distributions | 2610.27648 | 2026-10 | n/r | n/r | n/r | F1 | Supporting | no | Explicit speak/know split |
| The Model Knows Better: Medical MAS vs Prompt Injection | 2610.22594 | 2026-10 | n/r | Multi-agent medical | n/r | F1 | Supporting | no | Defense using internal knowledge |
| Causal Tongue-Tie (see Core) | — | — | — | — | — | — | — | — | — |
| Do LLMs Act on What They Know (see Core) | — | — | — | — | — | — | — | — | — |
| Knowing-but-Doing: Role-Play Jailbreaks via Moral Disengagement | Findings ACL 2026 (.349) | 2026 | n/r | Role-play jailbreaks | n/r | F1 | Supporting | yes | Knows harmful, still complies |
| Principled Under Pressure | 2610.08670 | 2026-10 | OLMo-3-7B-Inst, Tulu 3, Qwen2.5-7B (≤8B) | Moral scenarios, agent vs third person | beh | F1 | Supporting | no | Behavioral say-vs-do gap; post-training decides |
| Knowing What You Know Is Not Enough: Confidences vs Actions | 2511.13240 | 2025-11 | n/r | Agentic, multi-turn | beh | F1 | Supporting | yes | Confidence does not drive action |
| Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap | 2605.14038 | 2026-05 | n/r | Tool use | n/r (possibly internals) | F1 | Supporting | yes | Knowing-doing gap in tool calls |
| STOCKTAKE: Perception vs Action in LLM Agents | 2607.13618 | 2026-07 | n/r | Multi-week decisions | beh (fair oracle) | F1, dir3 | Supporting | yes | Separates misread-the-world from failed-to-act |
| LLMs are Greedy Agents (RL fine-tuning) | 2504.16078 | 2025-04 | n/r | Bandits/agents | beh | F1 | Supporting | yes | Knowing-doing gap; RL partly closes it |
| Why Do LLMs Struggle in Strategic Play? Broken Links | 2605.00226 | 2026-05 | n/r | Games | n/r | Observation-belief-action links | Supporting | no | Belief-action decomposition |
| Belief Without Behavior (ToM to action, VLMs) | 2608.20975 | 2026-08 | VLMs | Social coordination | n/r | F1 | Supporting | no | ToM not translated into action |
| ScrambleToolBench: Agents Search Even When Their Map Points to the Next Step | 2608.02358 | 2026-08 | n/r | Tool agents | beh | F1 | Supporting | no | Behavioral know-do gap in agents |
| From Evidence to Action: How Tool-Using Agents Fail | 2610.07753 | 2026-10 | n/r | Tool agents | beh | F1 | Supporting | no | Actions unsupported by established evidence |
| Emergent Unfaithfulness: Alignment Training Silently Overrides Task Faithfulness | 2610.00568 | 2026-09 | Several, scaling | FaithConflict dataset | beh (plus CoT taxonomy) | Override | Supporting | no | Dataset; effect grows with scale |
| Why Models Know But Don't Say: CoT Faithfulness Divergence | 2603.26410 | 2026-03 | Open-weight reasoning models | Thinking tokens vs answers | n/r | F1 | Supporting | no | Thinking-answer divergence |
| Catching Rationalization in the Act (motivated reasoning) | 2603.17199 | 2026-03 | n/r | Reasoning | corr | Override by motivated reasoning | Supporting | yes | Detecting motivated reasoning |
| Bypassing the Rationale: Causal Auditing of Implicit Reasoning | 2602.03994 | 2026-02 | n/r | CoT | C | Bypass | Supporting | no | CoT not causally used |
| Knowledge Acquisition-Utilization Gap in PLMs | Findings EMNLP 2023 | 2023 | PLMs | Knowledge use | beh | F2 | Supporting | no | Early acquired-vs-used measurement |
| Probing for Arithmetic Errors in LMs | EMNLP 2025 | 2025 | n/r | Arithmetic | corr | Error decodable | Supporting | yes | Error probes |
| Legible Failures (see Core) | — | — | — | — | — | — | — | — | — |
| Brittle Minds, Fixable Activations: Belief Representations | Findings EMNLP 2025 | 2025 | n/r | ToM | C (steering) | Fix | Supporting | no | Belief representations steerable |
| Language Models Encode the Contextual Truth of Propositions | 2608.03035 | 2026-08 | n/r | Contextual truth | corr | F3 | Supporting | no | Truth directions under context |
| Quantized but Deceptive? Truthfulness of Quantized LLMs (TruthfulnessEval) | EMNLP 2025 | 2025 | Quantized LLMs | Misleading prompts | corr | Truthful representations kept, outputs false | Supporting | yes | Representation vs output under quantization |
| Universal Truthfulness Hyperplane | EMNLP 2024 | 2024 | n/r | Truthfulness | corr | F3 | Supporting | no | Cross-task truth direction |
| Probing the Geometry of Truth (logical transformations) | Findings ACL 2025 | 2025 | n/r | Truth directions | corr | dir2 | Supporting | yes | Consistency of truth directions |
| Personas as a Way to Model Truthfulness | 2310.18168 / EMNLP 2024 | 2023-10 | n/r | Truthfulness | corr | F3 | Supporting | no | Truthful persona hypothesis |
| Adaptive Activation Steering (truthfulness) | WWW 2025 | 2025 | n/r | TruthfulQA | C | Fix | Supporting | no | "Understanding of truthfulness, yet fail to express it" |
| Where's the Plan? Latent Planning | 2605.07984 | 2026-05 | Qwen3, Gemma-3, Llama-3, >10 scales (to 27B) | Rhyming couplets | C (patching) | Decodable ≠ used; only Gemma-3-27B relies on the plan | Supporting | no | Scale- and model-specific use |
| Observable Patterns Are Not Explanations (latent reasoning) | 2606.12689 | 2026-06 | Coconut, CODI plus controls | Latent CoT | C | Graded, not binary, use | Supporting | no | Use as a matter of degree |
| Readable but Not Controllable (see Core) | — | — | — | — | — | — | — | — | — |
| A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals | 2609.00760 | 2026-09 | n/r | Refusals | C | F1 | Supporting | no | Knowledge-refusal mechanism |
| Do LMs Know Their Own Constraints | 2609.22151 | 2026-09 | n/r | Banned-ingredient recipes | n/r | Reportable vs enacted | Supporting | no | Constraint known vs reported |
| What LLM Forecasters Know but Don't Say | 2607.08046 | 2026-07 | Fine-tuned forecasters | Forecasting | corr | F1 | Supporting | no | Probes better calibrated than outputs |
| Small VLMs Know When They Are Wrong But Cannot Say So | 2607.22034 | 2026-07 | 2 small VLMs | Degraded images | corr | Internal vs stated confidence | Supporting | no | Confidence dissociation |
| Internal Knowledge Without External Expression (Classical Chinese LM) | 2604.14180 | 2026-03 | 318M from scratch; 8 models | OOD detection | corr | Internal uncertainty vs expression | Supporting | no | From-scratch control model |
| Do LLMs Signal When They're Right? Neuron Agreement | 2510.26277 | 2025-10 | n/r | Reasoning | corr | Correctness signal | Supporting | no | Neuron-agreement signal |
| How LLMs Detect and Correct Their Own Errors: Internal Confidence Signals | 2604.22271 | 2026-04 | n/r | Self-correction | n/r | Second-order confidence | Supporting | no | When decodable error signals become action |
| Decoding the Critique Mechanism in Large Reasoning Models | 2603.16331 | 2026-03 | LRMs | Backtracking | C | When error signals are used | Supporting | no | Critique mechanism |
| A Decodability Criterion for Hidden-State Selection vs Majority Voting | 2608.17124 | 2026-08 | n/r | Test-time fusion | corr | F2 | Supporting | no | Criterion for when probes beat voting |
| The Pushback Paradox: Two-Probe Diagnostic | 2610.06673 | 2026-10 | n/r | Compliance | corr | Override | Supporting | no | Two-probe design |
| LUMOS: Parametric Knowledge from Training Data to Outputs | 2610.02902 | 2026-10 | n/r | Factual | n/r | Training-data-to-behavior trace | Supporting | no | Data-side view |
| LLMs Know More Than They Say: Analogical Reasoning | 2604.03877 | 2026-04 | n/r | Analogies | corr | F3 | Supporting | no | Probe beats output |
| Revealing Economic Facts: LLMs Know More Than They Say | 2505.08662 | 2025-05 | n/r | Economic statistics | corr | F3 | Supporting | no | Probe on embeddings beats generation |
| Transformers Know More Than They Can Tell (Collatz) | 2511.10811 | 2025-11 | Small transformers | Collatz | corr | F3 | Supporting | no | Toy "know more than tell" |
| Visual Latents Know More Than They Say | 2605.02735 | 2026-05 | MLLMs | Latent reasoning | n/r | F3 | Supporting | no | Unsilencing latents |
| The First to Know: Token Distributions Reveal Hidden Knowledge in LVLMs | 2403.09037 | 2024-03 | LVLMs | Unanswerable or unsafe inputs | corr | F3 | Supporting | no | First-token logits encode it |
| Knowing When Not to Answer: Latent Underspecification Signals | 2610.08413 | 2026-10 | n/r | Underspecified Qs, multi-turn | corr | F1, dir3 | Supporting | yes | Latent signal generalizes across domains and turns |
| Agent-unrelated: Hallucination Is Linearly Decodable in Quantized LLMs | 2606.02628 | 2026-05 | Quantized open LLMs | Hallucination | corr | F2 | Supporting | no | Mid-layer signal |
| When CoT Fails, the Solution Hides in the Hidden States | 2604.23351 | 2026-04 | n/r | Reasoning | corr | F2 | Supporting | no | Solution decodable when CoT fails |
| StateSwap: Support vs Elimination Hidden States in MCQ | 2609.01081 | 2026-09 | n/r | MCQ | C (swap) | Readout | Supporting | no | Framing-dependent state |
| Effectiveness of Hidden-State Probes for Localizing Reasoning Failures | 2610.22046 | 2026-10 | n/r | Reasoning traces | corr | Capability vs use | Supporting | no | Separates beyond-capability from failure to use |
| Right Answers, Wrong States: Multi-Agent Collaboration | 2610.01244 | 2026-10 | n/r | Multi-agent | beh | Hidden information failures | Supporting | no | Corrupted shared state despite a right answer |
| Score × Decoder: Unsupervised Inference-Time Scaling | 2606.00739 | 2026-05 | n/r | Hallucination | Decoding | F3 fix | Supporting | no | Surfaces latent knowledge |
| SLED: Self Logits Evolution Decoding | 2411.02433 | 2024-11 | n/r | Factuality | Decoding | F3 fix | Supporting | no | Uses early-layer logits |
| Do Correct Answers Have Stable Latent Knowledge Support? | 2610.15277 | 2026-10 | n/r | QA | corr | F3 | Supporting | no | Latent support for correct outputs |
| Cluster-norm for Unsupervised Probing | 2407.18712 | 2024-07 | n/r | ELK probing | corr | Method | Supporting | no | Fix for CCS-style probes |
| GRADE: gradient-based knowledge-gap detection | 2604.02830 / COLM 2026 | 2026-04 | n/r | Knowledge sufficiency | Gradient | Control (probes unreliable) | Supporting | no? | Argues activations misalign with query needs |
| Anthropic, "On the Biology of a Large Language Model" | transformer-circuits.pub (2025-03) | 2025-03 | Claude 3.5 Haiku (per primary; not verified here) | Hallucination, motivated reasoning | C (attribution graphs) | Known-entity inhibition of the default "can't answer" circuit | Supporting | no | Lab report with experiments; secondary sources only in this sweep |

### 2c. Peripheral (framing, adjacent settings, or only loosely tied to internals)

| Title | ID / venue | Date | Models | Setting | C/corr | Bears on | Tier | in CSV | Why |
|---|---|---|---|---|---|---|---|---|---|
| G1: Perception and Reasoning in VLMs via RL | 2505.13426 | 2025-05 | VLMs | Games | Training | Knowing-doing framing | Peripheral | no | RL closes a behavioral knowing-doing gap |
| Models Recall What They Violate (DriftBench) | 2604.28031 | 2026-04 | 7 models | Multi-turn ideation | beh | F1, dir3 | Peripheral | yes | Models restate constraints they break |
| KScope: Knowledge Status Characterization | 2506.07458 | 2025-06 | n/r | Knowledge status | beh | F3 | Peripheral | no | Knowledge-status taxonomy |
| VLM Confidence Is Not a Property of the Answer | 2608.06571 | 2026-08 | VLMs | Confidence attack | C | Confidence gate | Peripheral | no | Adversarial confidence |
| Model-Aware Data Selection (In-and-Out Interplay) | 2609.33010 | 2026-09 | n/r | Post-training data | Training | F3 | Peripheral | no | "Know what they know" via data selection |
| ParamMute | 2502.15543 | 2025-02 | n/r | RAG faithfulness | C (FFN suppression) | Competing parametric signal | Peripheral | no | RAG override fix |
| Neural Probe-Based Hallucination Detection | 2512.20949 | 2025-12 | n/r | Hallucination | corr | Detection | Peripheral | no | Detection only |
| Robust Evaluation of Probe Robustness (ProbeDrift) | 2604.11662 | 2026-04 | n/r | UQ probes | corr | Control | Peripheral | no | Probe OOD robustness |
| Latent Knowledge as a Predictor of Fact Acquisition | 2601.18468 | 2026-01 | n/r | Biomedical FT | corr | F3 | Peripheral | no | Latent-strength metric |
| Reliable Latent Knowledge Estimation (ZP) | 2404.12957 | 2024-04 | n/r | Factual extraction | beh | F3 | Peripheral | no | Prompt-free estimation |
| The Impossibility of Eliciting Latent Knowledge | 2606.12268 | 2026-06 | Theory | ELK | Theory | F3 | Peripheral | no | Impossibility result |
| Forgetting is Not Erasure: Transport Keys | 2606.02860 | 2026-06 | n/r | Continual learning | C | Present but inaccessible | Peripheral | no | Information recoverable after forgetting |
| Mechanism of Weak-to-Strong Generalization (latent knowledge) | 2605.12908 | 2026-05 | n/r | W2S | n/r | F3 | Peripheral | no | Elicitation by fine-tuning |
| Auditing LM Unlearning via Information Decomposition | 2601.15111 | 2026-01 | n/r | Unlearning | corr | Present but suppressed | Peripheral | no | Residual information after unlearning |
| REBEL: Hidden Knowledge Recovery | 2602.06248 | 2026-02 | n/r | Unlearning | beh | Suppressed | Peripheral | no | Recovery after unlearning |
| Memory Self-Regeneration in Unlearned Models | 2510.03263 | 2025-10 | n/r | Unlearning | n/r | Suppressed | Peripheral | no | Same |
| Cosine Misleads (latent visual reasoning) | 2606.05753 | 2026-06 | VLMs | LVR | C | Control | Peripheral | no | Alignment metric misleads |
| Functional Metacognition (decompose and steer) | 2605.08942 | 2026-05 | n/r | Eval awareness | C | Adjacent | Peripheral | no | Metacognition directions |
| Refusal Before Decoding | 2605.28553 | 2026-05 | n/r | Refusal | corr | Pre-decoding signal | Peripheral | no | Probe prediction of refusal |
| Artificial Impressions | 2510.08915 | 2025-10 | n/r | Stereotypes | corr | Representation vs behavior | Peripheral | no | Adjacent variable |
| Principled-adjacent: LessWrong "abstention geometry: knowledge and behaviour are dissociable" | LessWrong post | n/r | n/r | Abstention | n/r | F1 | Peripheral | no | Blog post, not read |
| Intention-like Representations in LMs? | Phil Studies 2026 | 2026-09 | — | Philosophy | — | Framing | Peripheral | no | Conceptual |
| Knowing Is Not Doing: Predict Human Behavior but Respond Differently (Zenodo) | zenodo 23278352 | 2026-10 | n/r | Social prediction | beh | F1 framing | Peripheral | no | Third-person vs first-person gap |
| Do LLMs Know What They Don't Know? | Findings ACL 2023 | 2023 | n/r | Unanswerable Qs | beh | F1 | Peripheral | no | Behavioral self-knowledge |
| LMs Act on Hidden Valence | 2609.35591 | 2026-09 | n/r | Valence | C | Adjacent | Peripheral | no | Valence representation drives action |
| Layer by Layer: Hidden Representations | 2502.02013 | 2025-02 | n/r | Representation quality | corr | Intermediate layers outperform final | Peripheral | no | Mid-layer features beat final |
| RL vs SFT Representational Quality for Math | 2607.26119 | 2026-07 | n/r | Math | corr | Training effects | Peripheral | no | Training changes representations |
| MedProb: Probing VLMs for Medical QA | 2609.04336 | 2026-09 | VLMs | Med-VQA | corr | Probe beats generation | Peripheral | no | Probe-based answering |
| Memory vs Context reproduction (Yu et al. 2023) | 2609.24238 | 2026-09 | n/r | Knowledge conflict | C | Override | Peripheral | no | Replication study |
| Beacon: Latent Sycophancy | 2510.16727 | 2025-10 | n/r | Sycophancy | beh | Override | Peripheral | no | Diagnostic benchmark |
| Actionable Hallucination Detection: Latent Uncertainty to Agentic Critique | 2608.10430 | 2026-08 | n/r | Agents | corr plus action | F1 fix | Peripheral | yes | Probe output turned into critique |
| Pixels Versus Priors: Visual Counterfacts | EMNLP 2025 | 2025 | MLLMs | Vision vs prior | C (steering) | Override | Peripheral | no | Prior vs pixel arbitration |
| Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly | CVPR 2025 | 2025-06 | MLLMs | VQA | n/r | Sees but answers wrong | Peripheral | no | Multimodal know-say |
| Reasoning Instructions Can Break Answer Decoding in VLMs | 2609.29278 | 2026-09 | VLMs | MC evaluation | beh | Readout artifact | Peripheral | no | Evaluation artifact |
| Diagnosing Visual Ignorance in VLMs | 2606.06890 | 2026-06 | VLMs | VQA | n/r | Use of visual evidence | Peripheral | no | Adjacent |
| FLIP: Final Layer Inference-Time Probing (VLMs) | 2609.30993 | 2026-09 | VLMs | n/r | corr | Method | Peripheral | no | Logit-facing intervention site |
| Causal Probing for Visual Representations in MLLMs | 2605.05593 | 2026-05 | MLLMs | Visual concepts | C | Method | Peripheral | no | Causal probing |
| Causal Localization of the Refusal Direction in Audio LMs | 2609.22260 | 2026-09 | Audio LMs | Refusal | C | Adjacent | Peripheral | no | Modality transfer of refusal |
| From Deceptive Outputs to Deceptive Mechanisms | 2609.04166 | 2026-09 | — | Framework | — | Framing | Peripheral | no | Causal framework for deception claims |
| LMs Use Lookbacks to Track Beliefs | 2505.14685 | 2025-05 | n/r | ToM | C | Belief tracking | Peripheral | no | Mechanism background |
| CRAFT: Causal Responsibility in Medical VLMs | 2609.38810 | 2026-09 | Medical VLMs | Visual vs text conflict | C | Override | Peripheral | no | Competing-signal tracing |
| Moral Sycophancy in VLMs | 2602.08311 | 2026-02 | VLMs | Moral | beh | Override | Peripheral | no | Behavioral |
| Confidently Wrong: Financial QA Hallucinations from Internal States | 2607.11414 | 2026-07 | n/r | Finance QA | corr | Detection | Peripheral | no | Domain detection |
| Training LMs to Fail Gracefully | 2610.20401 | 2026-10 | n/r | RL abstention | Training | Fix | Peripheral | no | RL rewards guessing |
| LMs Fail at Extended Rule Following | 2605.02028 | 2026-05 | n/r | Agentic rules | beh | F1 | Peripheral | no | Behavioral |
| K/V-Cache Interventions: Representation Alignment vs Persona Expression | 2609.11020 | 2026-09 | n/r | Persona | C | Representation vs expression | Peripheral | no | Adjacent dissociation |
| Emergent Causal-Geometric Dynamics Across Depth | 2602.04931 | 2026-02 | n/r | General | C | dir2 | Peripheral | no | Depth geometry |
| A Heckler in the Hidden State (diffusion LMs, code) | 2609.36783 | 2026-09 | Diffusion LMs | Code | corr/C | Correctness signal | Peripheral | no | Non-AR setting |
| LOCKR: Stable-but-Wrong Lock-In (diffusion LMs) | 2609.27220 | 2026-09 | Diffusion LMs | Reasoning | corr | dir3 lock-in | Peripheral | no | Trajectory lock-in |
| HIVE: Hidden-Evidence Verification (diffusion LLMs) | 2604.26139 | 2026-04 | Diffusion LMs | Hallucination | corr | Detection | Peripheral | no | Non-AR setting |
| Fuzzing LLMs to Elicit Hidden Behaviours | 2606.29646 | 2026-06 | Sleeper agents | Backdoors | beh | F3 | Peripheral | no | Elicitation |
| Agent-BRACE: Decoupling Beliefs from Actions | 2605.11436 | 2026-05 | n/r | Long-horizon POMDP | Method | F1 | Peripheral | no | Verbalized belief uncertainty |
| Can an Agent Learn to Predict Its Own Failure? | 2610.18614 | 2026-10 | n/r | Agents | Training | F1 | Peripheral | no | Outcome-grounded confidence |
| StateWise: Persistent Operational State | 2610.05241 | 2026-10 | n/r | Agents | Method | Adjacent | Peripheral | no | State repair |
| Do GUI Agents Know When Not to Act? | 2609.03438 | 2026-09 | GUI agents | Infeasible instructions | Training | F1 | Peripheral | no | Termination |
| Still No Lie Detector for Language Models | Phil Stud 2024 | 2024-02 | — | Probing critique | — | Control (conceptual) | Peripheral | no | Conceptual limits of probes |
| ActProbe: Failure Detection in Robot Policies | 2606.08508 | 2026-06 | Robot policies | Robotics | corr | Detection | Peripheral | no | Embodied |
| Activation Oracles' Blind Spots | 2607.23379 | 2026-07 | n/r | Activation oracles | corr | Method | Peripheral | no | Reading activations with LMs |
| LMs Represent and Transform Concepts with Shared Geometry | 2607.04525 | 2026-07 | n/r | Concepts | corr | dir2 | Peripheral | no | Geometry background |
| Fine-Tuning LMs to Know What They Know | 2602.02605 | 2026-02 | n/r | Metacognition | Training | F1 | Peripheral | no | Metacognition training |
| Probing the Misaligned Thinking Process | 2606.24251 | 2026-06 | n/r | Misalignment | corr | F1 | Peripheral | yes | Probes for misaligned reasoning |
| Do LLMs Know What They Are Capable Of? | 2512.24661 | 2025-12 | n/r | Self-prediction, multi-step | beh | F1 | Peripheral | yes | Capability self-knowledge |
| Geometry of LM Hallucination (Harvard thesis) | dash.harvard.edu 42743982 | 2026 | n/r | Hallucination | n/r | dir2 | Peripheral | no | Thesis |
| See it, Say it, Sorted (emergent misalignment) | 2609.34970 | 2026-09 | n/r | EM | C | Adjacent | Peripheral | no | EM mechanism |
| Neighborhood Consistency (Illusions of Confidence) | ACL 2026 (.203) | 2026 | n/r | Truthfulness | beh | F3 | Peripheral | no | Behavioral truthfulness |
| Tele-Lens (myopic planning horizon) | 2602.02103 | 2026-02 | n/r | Planning | corr | dir3 | Peripheral | no | Aggregator summary only |
| Efficient TTS by Probing Internal States (multi-step) | ACL 2026 (.536) | 2026 | n/r | Step verification | corr | dir3 | Peripheral | no | Step-level probes |
| Aligned Probing (toxicity) | TACL 2026 | 2026 | n/r | Toxicity | corr/C | Representation vs output | Peripheral | no | Links outputs to internal encoding |
| CausalGaze | Findings ACL 2026 | 2026 | n/r | Hallucination | C | Method | Peripheral | no | Causal intervention for detection |
| Prompt Leakage Intents (I've Decided to Leak) | EMNLP 2025 (.1082) | 2025 | n/r | Prompt leakage | corr | Intent decodable | Peripheral | no | Adjacent |
| Moral Reasoning Trajectories (probing) | 2603.16017 | 2026-03 | n/r | Moral reasoning | corr | dir3 | Peripheral | no | Trajectory probing |
| Purging the Gray Zone (knowledge boundary) | 2604.14324 | 2026-04 | n/r | Abstention FT | Training | F1 | Peripheral | yes | Latent-geometric denoising |
| Lingtai: Concept Geometry Reveals / Does Not Reveal | 2610.00656 | 2026-10 | n/r | Inference | corr | dir2 control | Peripheral | no | Limits of concept geometry |
| Cited but Not Consulted (legal CoT) | 2610.12361 | 2026-10 | n/r | Legal CoT | C (counterfactual) | Bypass in CoT | Peripheral | no | Rationale not used |
| LLMs Know More Than Words (genre) | 2512.04957 | 2025-12 | n/r | Genre | corr | F3 | Peripheral | no | Probe beats output, stylistic |
| A False Average: CoT Monitor Fragility | 2608.00583 | 2026-08 | n/r | CoT monitoring | beh | Adjacent | Peripheral | no | Monitor accuracy |
| Knowing Is Not Telling: Identity Ossification | 2610.12610 | 2026-10 | n/r | Identity | corr | F1 | Peripheral | no | Representation vs telling |
| Knowing the Rules, Applying the Rules (Bazi) | 2610.05682 | 2026-10 | n/r | Rule application | beh | F1 | Peripheral | no | Behavioral |
| Knowing the User Is Not Knowing What We Established | 2610.18684 | 2026-10 | n/r | Personal agents | beh | F1 | Peripheral | no | Grounding awareness |
| Saying, Not Knowing (GGUF-quantized SLMs) | 2610.04403 | 2026-10 | Small LMs | Rare words | beh | Inverse gap | Peripheral | no | Inverse dissociation |
| Agentic Self-Awareness: Knowing, Acting, Holding | 2610.19226 | 2026-10 | n/r | Agents | n/r | F1 framing | Peripheral | no | Framing |
| What CLIP Knows but Cannot Say (negation) | 2607.23271 | 2026-07 | CLIP | Negation | corr | F1 | Peripheral | no | Encoder-only |
| VLMs Fail to Use Visual Representations for Text | 2610.25470 | 2026-10 | VLMs | Text in images | Training | Bypass | Peripheral | no | Self-distillation fix |
| The Visual Insensitivity Gap | 2609.00868 | 2026-09 | VLMs | Visual evidence | beh | Bypass | Peripheral | no | Behavioral |
| VoxParadox (audio LLMs listen or read) | 2605.27772 | 2026-05 | Audio LLMs | Paralinguistic | n/r | Override by text | Peripheral | no | Adjacent |
| Reproducing two LLM reliability approaches | 2608.08514 | 2026-08 | n/r | Reliability | Replication | Control | Peripheral | no | Stress test |
| Inter-Layer Disagreement (hallucination) | 2602.09486 | 2026-02 | n/r | Hallucination | Decoding | F3 fix | Peripheral | no | Layer contrast |
| CASS-RTL: Correctness-Aware Subspace Steering | 2606.05680 | 2026-06 | n/r | RTL code | C | Fix | Peripheral | no | Domain steering |
| ZebraArena: Reasoning-Action Coupling | 2603.18614 | 2026-03 | n/r | Tool agents | beh | F1 | Peripheral | no | Coupling diagnostic |
| Explain-Query-Test | 2501.11721 | 2025-01 | n/r | Self-evaluation | beh | F1 | Peripheral | no | Explanation vs comprehension gap |
| Confident at the Moment of Action (hidden information games) | 2608.24691 | 2026-08 | n/r | Games | beh | F1 | Peripheral | no | Belief miscalibration at action |
| Do GUI Agents Believe Their Eyes? | 2607.04334 | 2026-07 | GUI agents | State belief | beh | Adjacent | Peripheral | no | Pixel vs structure reliance |
| Phantom Merge: Agents Pick One but Tell About Another | 2610.16141 | 2026-10 | n/r | Agents | beh | F1 | Peripheral | no | Report-action mismatch |
| Invisible Orchestrators Suppress Protective Behavior | 2605.13851 | 2026-05 | n/r | Multi-agent | beh | Override | Peripheral | no | Safety in MAS |
| The Granularity Gap: Sycophancy in Gemini | 2606.05183 | 2026-06 | Gemini models | Sycophancy | beh | Override | Peripheral | no | Behavioral audit |
| ELEPHANT: Social Sycophancy | 2505.13995 | 2025-05 | n/r | Social sycophancy | beh | Override | Peripheral | no | Benchmark |
| Do SLMs Know When They're Wrong? (cascade) | 2604.19781 | 2026-04 | SLMs | Education grading | beh | F1 | Peripheral | no | Confidence cascade |
| Know When You're Wrong: Confidence-Correctness Alignment | 2603.06604 | 2026-03 | n/r | Error detection | Training | F1 | Peripheral | no | Alignment of confidence |
| The Confidence Paradox | 2506.23464 | 2025-06 | n/r | Confidence | beh | F1 | Peripheral | no | Behavioral |
| Interpreting Key Mechanisms of Factual Recall | 2403.19521 | 2024-03 | GPT-2 small | Factual recall | C | Anti-overconfidence suppression | Peripheral | no | Suppressive heads |
| Truth Neurons | id not resolved | n/r | n/r | Truthfulness | n/r | F3 | Peripheral | no | Seen only on a review site |
| Layer-wise Probing of Context Knowledge | 2402.16061 | 2024-02 | n/r | Context knowledge | corr | Conflict | Peripheral | no | V-usable information |
| Hallucinations vs Knowledge Conflicts | 2606.08705 | 2026-06 | n/r | Hallucination | corr | Conflict | Peripheral | no | Correlation study |
| Latent Reasoning Intervention (interpretability-guided) | 2606.01243 | 2026-06 | Latent-reasoning models | Latent CoT | C | Graded use | Peripheral | no | Adjacent |
| Open-Weight Masked Introspection | 2608.20569 | 2026-05 | Open-weight models | Introspection | n/r | Report vs computation | Peripheral | no | Self-report |
| Spatiotemporal Hidden-State Dynamics of Internal Reasoning | 2605.01853 | 2026-05 | n/r | Reasoning | corr | dir3 | Peripheral | no | Dynamics signature |
| From Helpfulness to Toxic Proactivity (agent misalignment) | 2602.04197 | 2026-02 | n/r | Agents | beh | Adjacent | Peripheral | no | Behavioral |
| Evidence-Type Competition (interventional data, causal direction) | 2607.29484 | 2026-07 | Synthetic pretraining | Causal learning | Training | Competing signal | Peripheral | no | Toy competition |
| Same Outcome, Different Readout (valence direction) | 2609.22850 | 2026-09 | n/r | Valence | C | Control | Peripheral | no | Steering readout ambiguity |
| The Pain Axis | 2609.16247 | 2026-09 | n/r | Self-harm | C | Adjacent | Peripheral | no | Represent-and-act |
| Defeat Devices in AI Systems | 2606.28863 | 2026-06 | — | Framing | — | F1 | Peripheral | no | Conceptual |
| When Do LLM Preferences Predict Downstream Behavior? | 2602.18971 | 2026-02 | n/r | Preferences | beh | F1 | Peripheral | no | Stated vs enacted |
| Materials-science mechanisms: reading and steering | IOP 2026 (doi 10.1088/3050-287x/aeb27b) | 2026-10 | 1 open-weight | Science QA | C | Represent vs use | Peripheral | no | Domain case |
| The Image Still Has a Say: Final-Answer Repair | 2610.17663 | 2026-10 | MLLMs | Multimodal CoT | Method | Fix | Peripheral | no | Answer repair |
| Goodfire reward-hacking activation-probe monitors | news coverage only | 2026-09 | n/r | Agents | corr | F1 | Peripheral | no | Unverified; primary post not read |

---

## 3. Seen but judged off-topic (titles only)

Exploration with Foundation Models (2509.19924); How Far Are LLMs from Professional Poker Players (2602.00528); Memory-Augmented State Machine Prompting (2510.18395); Research-Practice Gap in Algorithmic Fairness for Public Health (2606.11214); Didactic Knowledge or Clinical Cases (2609.22161); PaperGuide (2601.12988); Agile healthcare frameworks (2502.10403); Multi-agent AI classroom for physics teachers (2608.05820); Parametric Knowledge is Not All You Need (2601.21218); Machine Against the RAG (2406.05870); When to Retrieve (2404.19705); When an AI Judges Your Work (2603.02076); Knowing What LLMs DO NOT Know: Self-Detection (2310.17918); Can AI Assistants Know What They Don't Know (2401.13275); Self-Knowledge Guided Retrieval Augmentation (2310.05002); Do LVLMs Know What They Know (2508.19111); SIMBA UQ (2510.13836); Stochastic Sampling is Epistemically Shallow (2607.20464); SlimPLM proxy models (2402.12052); ClashEval (2404.10198); LLM Agents can Autonomously Hack Websites (2402.06664); Apollonion (2404.08692); Know Or Not library (2505.13545); Honest AI fine-tuning small LMs (2410.09699); Decomposed Prompting Does Not Fix Knowledge Gaps (2602.04853); From Implicit to Explicit Self-Recognition (2508.14408); Mind the Gap tabular priors (2605.06343); ContractLens (2610.04992); DeCoRe (2410.18860); ContextFocus (2601.04131); Enhancing Reliability across Short and Long-Form QA via RL (2512.08944); Faithfulness-QA (2604.25313); FaithfulRAG (2506.08938); Knowledge Rumination (2305.08732); Latent Knowledge Scalpel (2508.03741); Plausibility Vaccine (2503.12667); REALM (2002.08909); Frame Semantics (2509.19540); Does Transformer Interpretability Transfer to RNNs (2404.05971); Conditioning Predictive Models (2302.00805); Eight Methods to Evaluate Robust Unlearning (2402.16835); Minecraft knowledge correction (2505.24157); Chemistry LM SAEs (2512.08077); MTMT thought tree (2412.03987); PRIMO (2411.01205); Transfer learning framework for W2S (2405.16236); Mechanistic Unlearning (2410.12949); KG-RAG (2405.12035); CGM foundation model (2412.09727); Latent student knowledge (2405.08213); Biomedical KB completion (2106.09700); Forensic Reserve (image forgery; 2610.08639); VeriForge (2608.09698); Queries Knew More Than We Thought (segmentation; 2609.20283); UNVaMP (2608.03811); Generative Framework for Personalized Persuasion (2504.13904); Neural Assistant (1910.14613); Sequential Latent Knowledge Selection (2002.07510); Knowledge-grounded dialogue (2212.08946); Democratizing graph data augmentation (2502.13555); Latent Knowledge-Guided Video Diffusion (2411.11343); MEG speech decoding (2609.31997); Sky sphere representation (2607.27092); Pixel Decodability Is Not a Compression Signal (2609.13012); Alignment Problem in Constrained Code Generation (2606.21619); How Much is Left? Remaining Output Length (2607.05316); Unifying LLMs and Knowledge Graphs roadmap (TKDE 2024); Event Knowledge in LLMs (Cog Sci 2023); LMEnt (TACL 2026); Dissociating Language and Thought (TiCS 2024); EmbodiedMidtrain (2604.20012); Look Before Acting VLA (2603.15618); VLMs Provide Promptable Representations for RL (2402.02651); R-Tuning (NAACL 2024); Large LMs Must Be Taught to Know What They Don't Know (NeurIPS 2024); Know Your Limits abstention survey (TACL 2025); Subliminal learning (Nature 2026); Hidden Role Game evaluation (2605.22826); The Hidden Frame (2609.07735); Clinical knowledge Nature 2023; Probing Political Ideology (Findings EMNLP 2025); Training-free truthfulness via sparse MLP value vectors; Fact Overwriting to Knowledge Evolution (2605.28303); Grounding Probes (2610.04642); States Hidden in Hidden States (2407.11421); Probing Functional Correctness in Diffusion LMs (ACL SRW 2026); ReLope routing (2603.24787); Ownership bias in Banjarese (2026); Where Reliability Lives in VLMs (2605.08200); medical QA Nature Medicine 2025; Know When to Stop overthinking (2607.00482); Lying with Truths multi-agent collusion (2601.01685); Generation Space Size (2510.12699); LTA-thinker (2509.12875); Think When You Need (2504.03234); Do Linear Probes Generalize Better in Persona Coordinates (2605.09391); Frontier Coding Agents AlphaZero (2604.25067); Learning to Decide Not to Reason (2610.06950); Steering is not a watermark (2610.17810); and about 150 off-domain results returned by loose arXiv web-search matching (quantum computing, physics, robotics, networking, segmentation, speech and similar), not listed individually.

---

## 4. BLOCKED

- **WebSearch:** the session-wide budget (200 calls, shared across agents) was used up after Q137. Three planned queries were not run: harmfulness-vs-refusal ("encode harmfulness separately from refusal"), reward-hacking awareness probes, and verbalized-vs-internal confidence. More searching needs a follow-up turn or a higher CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION.
- **OpenAlex API:** the daily free budget for this IP is used up. Continuing needs an API key, which requires registering with the user's email. Skipped.
- **Semantic Scholar API:** returned 429 on every attempt; a higher limit needs an API-key application. Skipped.
- **arXiv export API, then arXiv web search:** both began returning empty responses (rate-limited) partway through. Q27-31, Q54-55, Q69-72 and Q124-127 failed, and Q113-117 and Q119 were possibly affected.
- **OpenReview** (COLM 2026, ICLR 2026/2027 submissions, NeurIPS 2026, workshops): not swept directly. No browser was used, so OpenReview-only submissions are under-covered.
- **ACL Anthology:** not searched directly; only reached through OpenAlex and WebSearch hits.
- **Lab blogs:** the Anthropic "Biology" paper was seen through secondary sources only. Goodfire was seen through news coverage only. Apollo was seen through its ICML 2025 paper. Transluce and EleutherAI posts were not found or read.
- **LessWrong posts** ("abstention geometry: knowledge and behaviour are dissociable"; "giving an LLM more correct evidence made its decisions worse") were seen in results but not read.
- **No PDFs were downloaded.** Abstract pages were enough for this sweep.
- **Unresolved identifiers:** "Truth Neurons" (no arXiv id found); the GRADE full title (only "GRADE", COLM 2026, 2604.02830); "Knowing-but-Doing" (Findings ACL 2026.349; no arXiv id).
