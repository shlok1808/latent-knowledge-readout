# Direction 1b deep-read: override, transformation and erasure

Skill used: `deep-research` (researcher reference). Method-neutral wording: "decodable" means a probe or lens reads it; "causal" means an intervention changes behavior; "endogenous mediation" means evidence that the model naturally routes through the component. Sources were read through arxiv.org/html with WebFetch (a summarizing fetch, so numbers should be spot-checked against the PDF before citation). No PDFs were downloaded.

## Per-paper extractions

### P1. LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit (2604.19117v4, Pandey, Georgia Tech; sweep lists an ICML 2026 listing, not verified)

### Takeaway
The strongest causal evidence in the strand for **override**: a few shared attention heads write a "statement is wrong" direction in sycophancy, factual lying and instructed lying, on 12 models from 1.5B to 72B. Silencing them sharply raises sycophancy (Gemma-2-2B 28% to 81%) while accuracy holds. RLHF and DPO lower sycophancy but leave the heads in place. This is causal controllability, not endogenous mediation: the paper's own single-channel mediation test at 72B is a null.

### Cited Findings
- (1) Models: Gemma-2-2B/9B/27B, Qwen2.5-1.5B/32B/72B, Qwen3-8B, Llama-3.1-8B/70B, Llama-3.3-70B (RLHF refresh), Mistral-7B, Mixtral-8x7B, Phi-4 14B. Ten checkpoints are above 8B. Task: TriviaQA (400 pairs, split into halves), 300 opinion pairs, instructed-lying templates on 7 models, NaturalQuestions cross-check. — [arXiv 2604.19117](https://arxiv.org/html/2604.19117)
- (5) Protocol: head importance is the L2 norm of the head's output-write difference at the **last prompt token** (a write-norm form of DLA). Task directions are mean residual differences over N=200 pairs, as in Arditi et al. Overlap is top-K with K=ceil(sqrt N), with hypergeometric and layer-stratified permutation nulls. Interventions: projection ablation, mean ablation, head zeroing, per-head patching (8B and below only), shared-set patching (32B and above), IOI-style path patching (Gemma-2-2B, Phi-4), head-to-unembed direct effects (Llama-70B). — [2604.19117](https://arxiv.org/html/2604.19117)
- (5) Controls worth copying: write-norm-matched random heads; a sentiment/topic control task (4-7x overlap vs 12-25x for correctness tasks); a factual-QA replacement task; Benjamini-Hochberg over an 18-cell causal grid (14/18 survive); a sham-DPO control on neutral pairs; split-half reliability (r=0.87); a K-sensitivity sweep; 2,000-resample bootstrap CIs. — [2604.19117](https://arxiv.org/html/2604.19117)
- (6) Override, causal: zeroing the shared heads in Gemma-2-2B moves sycophancy from 28% to 81% while factual accuracy goes from 69% to 70%. Projection ablation in Gemma-2-27B moves agreement from 10.5% to 100%. Path patching gives r=0.993 across the 275-edge sycophancy-vs-factual circuit. Restoration ratios range from 3.6x to 2,248x across 13 model-task cells. — [2604.19117](https://arxiv.org/html/2604.19117)
- (6) Fixes leave the circuit: Llama-3.1-70B to 3.3-70B drops sycophancy from 39% to 3.5%, the shared-head count barely moves (57/72 to 51/72), and the projection-ablation effect **grows** (+10.5 to +27 pp). Mistral-7B to Zephyr keeps head ranking at Spearman 0.85 while sycophancy rises 3.6x. Anti-sycophancy DPO (LoRA r=16, beta=0.1, 1,000 pairs) cuts sycophancy 93% on Mistral and 46% on Gemma, with probe AUROC unchanged within ±0.05. The base Qwen2.5-1.5B already shows the overlap (10.5x chance). — [2604.19117](https://arxiv.org/html/2604.19117)
- (6) Decodability while agreeing is **indirect**: sycophancy probes transfer to lying at AUROC 0.83-0.85. The fetch found no direct read of the "is wrong" state at the step where the model emits agreement. The headline wording is "registered-but-overridden". — [2604.19117](https://arxiv.org/html/2604.19117)
- (6) Nulls the paper reports: the Qwen2.5-72B MLP single-channel mediation test (rho=-0.21, p=0.43); Llama-3.1-70B mean ablation of 57 heads gives +3.5 pp, indistinguishable from random; Qwen3-8B is ceiling-bound at 100% sycophancy. — [2604.19117](https://arxiv.org/html/2604.19117)
- (6) Opinion agreement uses the same head positions but writes a direction orthogonal to factual wrongness (|cos|<0.14, vs 0.43-0.81 for sycophancy-lying). The behavioral effect of zeroing these heads is sign-inconsistent across models. — [2604.19117](https://arxiv.org/html/2604.19117)
- (3) Stated gaps: the attribution is first-order, validated per head only at 8B and below. The 70B sufficiency-necessity asymmetry rests on n=2 models from one family. Single-turn only. Per-model cosine varies (0.43-0.81). The Qwen2.5-1.5B probe AUROC floor is 0.61; a deployed monitor would need about 0.9. — [2604.19117](https://arxiv.org/html/2604.19117)
- (4) Next steps, quoted: "extending the controlled DPO to a second ≥70B family is the natural next step"; "replication beyond Gemma, Qwen, and Llama remains the single most informative follow-up". — [2604.19117](https://arxiv.org/html/2604.19117). No follow-up found in the time box (no search spent).
- (2) To check: Genadi et al. EACL 2026, "Sycophancy hides linearly in the attention heads"; Ying et al. 2026 (truthfulness spectrum, limited cross-task overlap); Vennemeyer et al. 2025 ("sycophancy is not one thing"); Chen et al. ICML 2024 (pinpoint tuning); Li et al. ICLR 2025; Campbell et al. 2023; Halawi et al. ICLR 2024; McGrath et al. (Hydra effect). — [2604.19117](https://arxiv.org/html/2604.19117)

### Inferences
- Where the competing signal lives here: a small set of attention heads (about 1% of heads at 70B) that write into the residual at the last prompt token. Because the same heads carry the correctness signal, "override" here may be closer to **gating of a shared correctness channel** than to a separate competing signal outweighing an intact one. The paper does not separate those two.
- Reuse verdict: **copy** the controls (norm-matched random heads, sentiment control task, BH over a causal grid, sham-DPO). **Adapt** the claim: the necessity evidence weakens at 70B, and the decodability-at-action-time measurement is missing. That missing read is exactly a bypass-vs-override experiment.

### Gaps
- Dataset licenses not stated. The ICML 2026 acceptance was not verified. Characters 100,000 onward of the HTML were not read (later appendices).

---

### P2. A Mechanistic View of Authority Hierarchy in LLM Sycophancy (2607.00415v1; Joswin, Medicherla, Mammen; arXiv preprint, ICML template, no venue stated on the page; the sweep's "ICML 2026 MI workshop" was not confirmed)

### Takeaway
Claims **erasure**: on Physician-flipped questions, probes trained on baseline activations fall from above 0.9 to about 0.05 (below the 0.25 chance level) after a late "peak layer". Below-chance accuracy is better read as **transformation** (the hinted letter now sits in the probe's subspace) or as distribution shift than as erasure. No probe controls are reported, so the erasure claim is weak and likely partly a probe artifact.

### Cited Findings
- (1) Models: Llama-3.1-8B-Instruct, Qwen3-8B, Gemma-2-9B-it (the only one above 8B), via TransformerLens. Data: MedQA-USMLE; size and license not given on the page. Four personas (MS-1, MS-3, Chief Resident, Board-Certified Physician) all suggest the same wrong letter; the hint is placed **after** the choices ("Hint: A person who is {persona} thinks the answer is {letter}"). Scoring uses next-token letter probabilities only. — [arXiv 2607.00415](https://arxiv.org/html/2607.00415)
- (5) Protocol: logit lens on P(correct) and P(hinted); the peak layer is the first layer from which P(hinted) stays at least 0.05 above P(correct) under the Physician hint (L17 Llama, L28 Gemma, L29 Qwen3). Linear and MLP probes are trained on **baseline** activations (4-way, 5-fold out-of-fold) and applied to hinted activations. The probe token position is not stated; RQ4 vectors use the last token. — [2607.00415](https://arxiv.org/html/2607.00415)
- (6) Erasure evidence: in Gemma-2-9B, probe accuracy drops from >0.9 to ~0.05 after the peak layer on flipped questions. On resisted questions, transfer stays near perfect. Erasure is reported as larger for higher authority. Subsets: Chief Resident n=291, MS-3 n=117, MS-1 n=68; the Physician n is not given. — [2607.00415](https://arxiv.org/html/2607.00415)
- (6) Intervention: adding **per-question** hint vectors at the peak layer flips 63-82% of answers. The **mean** hint vector does no better than random (≤7%). The mean authority vector has negligible effect. Authors: the signal is "tightly entangled with question-specific content". — [2607.00415](https://arxiv.org/html/2607.00415)
- (3) Missing controls: no control tasks, no selectivity, no shuffled labels, no probe-capacity report. The probe is conditioned on outcome (flipped subset). The claim that failure of both probes "rules out geometric reorganization" is stronger than the evidence. The appendix says "Qwen-2-8B" where the text says Qwen3-8B. — [2607.00415](https://arxiv.org/html/2607.00415)
- (4) No future-work section. — [2607.00415](https://arxiv.org/html/2607.00415)
- (2) To check: Mammen et al. 2026 (authority bias across expertise levels); Choi et al. 2026 (authority in multi-agent evaluation); Li et al. 2025 (authority bias in RAG); Poonia & Jain 2025 (persona-driven reasoning via patching); Chen et al. 2025 (persona vectors). — [2607.00415](https://arxiv.org/html/2607.00415)

### Inferences
- A 4-way probe at 0.05 accuracy is systematically predicting a specific wrong class. Information about the correct answer could still be present in another subspace; the test only shows the baseline-trained readout now points at the hinted letter. That fits "transformed" (the same decision variable now carries the hinted letter) better than "erased".
- Test that would settle it: train a probe **on hinted activations** (with labels = correct answer, held-out questions) and add a shuffled-label control. If that probe stays well above chance, the information is present but moved; if it falls to chance, erasure has real support.
- Reuse verdict: **adapt** the persona design (third-person, domain-matched authority is a useful manipulation). **Avoid** its erasure inference as is.

### Gaps
- Physician-subset size, MedQA sample size, license and probe token position were not on the page.

---

### P3. When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in LLMs (2508.02087v4; Wang et al.; cited by 2607.00415 as Wang et al. 2026, AAAI; the arXiv page lists no venue)

### Takeaway
A two-stage account on about 7B models: a **late-layer shift in output preference** (Llama around L16-19, Qwen L22), then **representational divergence** in the final layers (KL spike). Activation patching at one critical layer moves sycophancy both ways. It never tests whether the correct answer stays decodable, so it cannot tell override from transformation. Expertise framing changes sycophancy by at most 4.4%.

### Cited Findings
- (1) Behavior on 7 models around 7B (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, OPT-6.7B, Mistral-7B-v0.3, Falcon-7B, OLMoE-1B-7B, Pythia-6.9B). Mechanistic analysis on Llama-3.1-8B and Qwen2.5-7B only; nothing above 8B. Data: MMLU, 57 subjects; sample count and license not given. The user's stated answer is always one of the three wrong options. — [arXiv 2508.02087](https://arxiv.org/html/2508.02087)
- (5) Conditions: Plain; Opinion-only ("I believe the right answer is B"); first-person expertise at three levels (GPT-4o-generated prefixes); third-person Advanced only. Metrics: agreement rate, accuracy, an independent error rate (wrong answers matching neither the user nor the truth). — [2508.02087](https://arxiv.org/html/2508.02087)
- (5)(6) Logit lens at the answer position, a normalized Decision Score per option, layer-wise KL between Plain and Opinion-only. Patching Plain into Opinion-only cuts Llama sycophancy by 36%; patching Opinion-only into Plain raises it to 47%. Token positions are not stated. — [2508.02087](https://arxiv.org/html/2508.02087)
- (6) Opinion-only agreement averages 63.7% (46.6%-95.1%). Expertise shifts it by ≤4.4%. First-person framing gives about 13.6 points more sycophancy than third-person. — [2508.02087](https://arxiv.org/html/2508.02087)
- (6) The paper says the Opinion-only condition "never establishes a strong preference for the correct answer" in the lens. That is a lens readout through the unembedding, not a probe. — [2508.02087](https://arxiv.org/html/2508.02087)
- (3) Gaps: the KL peak is stated at L23 in one place and the critical patching layers at L32 (Llama) and L27 (Qwen) in another, unreconciled. Patching at a single layer. No significance tests. No limitations or future-work section. Authority "not encoded" is inferred from PCA overlap (cosine 0.90-0.997 between expertise levels), not from a behavioral test. — [2508.02087](https://arxiv.org/html/2508.02087)
- (2) To check: Panickssery et al. 2023 (CAA sycophancy steering); Yu et al. 2023 (heads for memory-context conflict); Sicilia et al. 2024; Fanous et al. 2025 (SycEval). — [2508.02087](https://arxiv.org/html/2508.02087)

### Inferences
- **Disagreement P2 vs P3 (authority).** P3 finds authority barely matters (≤4.4%) and is not encoded; P2 finds authority-graded flips and erasure. P2 names the reasons: competence levels vs institutional roles, random vs domain-matched pairing, first-person self-claim vs third-person attribution. Add: hint before the question (P3) vs after the choices (P2), MMLU vs MedQA, and P3's PCA test vs P2's probes. P3's own first- vs third-person gap (13.6 points) shows framing alone moves the effect a lot, so the two results are compatible with "authority matters only when socially anchored and in-domain". Neither paper tests that directly.
- **Disagreement P2/P3 vs P1 (fate).** P1 claims override with the correctness signal intact (indirect). P3 claims the late layers diverge, and P2 claims erasure. Different methods: P1 reads head write directions trained across tasks; P2 trains probes on unpressured activations; P3 uses the logit lens, which only sees what the unembedding reads. A lens can miss information that a probe would find, and a baseline-trained probe can miss information that moved.
- Reuse verdict: **adapt** the Plain vs Opinion-only design and the independent error rate. **Avoid** single-layer patching without position control.

### Gaps
- Venue not on the arXiv page (AAAI 2026 per 2607.00415's citation, unverified).

---

### P4. When Truthful Representations Flip Under Deceptive Instructions? (2507.22149v4; the sweep lists EMNLP 2025, but the arXiv v4 page lists no venue)

### Takeaway
On Llama-3.1-8B and Gemma-2-9B, a deceptive system prompt makes the models follow the lie. Probes at the final token before True/False read the **instructed** (flipped) label in every condition. SAE features shift most in early-to-mid layers (Llama L10-16). The paper does **not** test whether the factual truth value stays decodable under deception, so this is evidence for **transformation of the output-facing variable**, with no direct test of whether the truth signal survives. Correlational only.

### Cited Findings
- (1) Main: Llama-3.1-8B-Instruct, Gemma-2-9B-Instruct. Behavior only: Llama-3.1-70B, Gemma-2-2B, Mistral-7B, Qwen2.5-7B, Qwen2.5-14B. — [arXiv 2507.22149](https://arxiv.org/html/2507.22149)
- (5) Data: curated Logical-Bench (cities 1,496; sp_en_trans 354; element_symb 186; animal_class 164; inventors 406; facts 561; larger_than and smaller_than 1,980 each, plus negated, conjunction and disjunction variants), common_claim_true_false (4,450), counterfact_true_false (31,960). License not stated; release promised. Probes: logistic regression and TTPD at the final token before the response, every layer, 5k/1k/5k splits, leave-one-topic-pair-out, per-layer z-scoring. SAEs: Gemma Scope 16k, Llama Scope 128k (the text calls it "Llama-3.1-8B-Base" in one place). — [2507.22149](https://arxiv.org/html/2507.22149)
- (6) Llama-3.1-8B accuracy under deception is roughly 10-36% on several curated sets. Probe peak about L14 (Llama) and L21 (Gemma). Deceptive-vs-truthful shifts are sharpest at L10-16 (Llama); truthful-vs-neutral stays flat (cosine >0.95, overlap >0.80). Some SAE features show "polarity flips" (one at L16 active under truthful prompts, reversed at L32). — [2507.22149](https://arxiv.org/html/2507.22149)
- (3) Stated limits: English declaratives, linear probes, **no causal interventions**, binary labels. — [2507.22149](https://arxiv.org/html/2507.22149)
- (4) Quoted: "future work should extend to graded truth scales and multilingual settings." Not checked for follow-ups. — [2507.22149](https://arxiv.org/html/2507.22149)

### Inferences
- **Disagreement with P5 (Localizing Lying).** P4 puts the shift in early-to-mid layers of 8-9B models; P5 puts the honest-liar flip at L19-23 of 80 in Llama-2-70B, which is also early-to-mid by depth fraction (24-29%). So the two agree on relative depth; they differ only in absolute layer index and scale.
- Reuse verdict: **copy** the dataset suite and the neutral/truthful/deceptive triad. **Adapt**: add the missing read (probe trained on truthful prompts for factual truth, applied to deceptive prompts, with a sign-flip check), as P5 did.

---

### P5. Localizing Lying in Llama (2311.15131v1; Campbell, Ren, Guo; "Preprint. Under review."; the sweep lists a NeurIPS 2023 workshop, not confirmed on the PDF)

### Takeaway
On **Llama-2-70b-chat**, per-head probes trained on honest prompts transfer to liar prompts with high accuracy in early-mid layers, then fall to about 10% after around L23 with **anti-parallel** coefficients. That pattern means the truth value is still linearly present but sign-flipped: **transformation (negation)**, not erasure. Patching honest activations into layers 19-23 restores 63% correct from a 2% base; 46 heads (0.9% of heads) reach 83%. This is causal controllability; the paper does not test endogenous mediation.

### Cited Findings
- (1)(5) Data: Azaria & Mitchell true/false splits (Cities, Inventions, Chemical Elements, Animals, Companies, Scientific Facts), filtered to items Llama-2-7b-chat answers correctly with p>0.85. Prompts: 10 system prompts (honest and lying), 6 user prompts, 3 assistant prefixes. Prefixes ("Oh that's an easy one! The statement is definitely") were needed to overcome a token bias toward "False". Best liar prompts reach under 5% correct; honest prompts above 95%. Smaller Llama-2-7b/13b-chat lie poorly under the same prompts. — [arXiv 2311.15131 PDF](https://arxiv.org/pdf/2311.15131)
- (5) Probes: logistic regression on each head's z activation (dim 128) at the last position. Patching: z activations from the honest run into the liar run over the **last 25 positions**, 5 contiguous layers at a time; head selection by leave-one-out over 50 Scientific Facts items. — [2311.15131](https://arxiv.org/pdf/2311.15131)
- (6) Transfer accuracy and cosine similarity are high before about L23 and become low (down to 10%) and anti-parallel after it. The authors: "parallel representations before layer 23 and anti-parallel representations thereafter", read as the model "first estimating the truth-value of the statement and then negating this representation around layers 19-23". — [2311.15131](https://arxiv.org/pdf/2311.15131)
- (6) Generalization (Table 2): patched liars recover >50% on most prompt-split pairs (e.g. 98.1% Facts for prompt 6fiii) and patching works in reverse (patched honest falls toward liar). Some combinations generalize poorly. — [2311.15131](https://arxiv.org/pdf/2311.15131)
- (4) Future work, quoted: "The observed representation flip could be a 'truth' bit, 'intent' bit, or could be related to more general behavior such as answer tokens or some inscrutable abstraction." They call for work on how the model uses the truth representation "along with the system prompt to decide whether or not to respond truthfully." 2604.19117 is a partial follow-up (shared heads across lying and sycophancy). — [2311.15131](https://arxiv.org/pdf/2311.15131)
- (3) Gaps: toy one-token setting; heads chosen on 50 items and one prompt; no random-head control reported in the main text; patching 46 heads beats patching all heads (64% vs 83%), which the authors attribute to inhibitory heads or noise. — [2311.15131](https://arxiv.org/pdf/2311.15131)

### Inferences
- The anti-parallel flip is the cleanest public evidence for **transformed** at 70B: a probe with its sign flipped would still read truth after L23. A fair test of "erased" (P2) must check for a sign-flipped or rotated readout before calling a below-chance probe "erasure". P2's below-chance result on a 4-way task is the multi-class analogue of this flip.
- Reuse verdict: **copy** the honest-to-liar transfer matrix with coefficient cosines, and the bidirectional patching. **Adapt**: add random-head controls and a held-out prompt for head selection.

---

### P6. Overthinking the Truth: Understanding How Language Models Process False Demonstrations (2307.09476v3; Halawi, Denain, Steinhardt; ICLR 2024)

### Takeaway
The foundational **late override** result. With false few-shot labels, logit-lens accuracy matches the correct-demo condition until a "critical layer", then drops; early-exiting beats the full model. Five late "false induction heads" (1% of heads) attend to and copy false labels; zeroing them cuts the correct-vs-false gap by 38.9% on average, with a random-head control. Models range up to GPT-NeoX-20B and Pythia-12B.

### Cited Findings
- (1) Models: GPT-J-6B, GPT2-XL-1.5B, GPT-NeoX-20B, Pythia 410M/2.8B/6.9B/12B, Llama2-7B, plus instruction-tuned GPT2-XL, GPT-J and GPT-NeoX-20B (11 total; NeoX-20B and Pythia-12B above 8B). 14 classification datasets (SST-2, Poem Sentiment, Financial Phrasebank, Ethos, TweetEval-Hate/-Atheism/-Feminist, MQP, MRPC, SICK, RTE, AGNews, TREC, DBpedia) plus a toy "Unnatural" set. 1,000 sequences per model-dataset, k from 0 to 40 demos. — [arXiv 2307.09476 PDF](https://arxiv.org/pdf/2307.09476)
- (5) Metric: **calibrated** accuracy (correct label above its dataset-median probability), to reduce prompt sensitivity. Critical layer = the layer where the gap first reaches half its final value. Permuted score checks that errors copy the permuted label rather than being random. — [2307.09476](https://arxiv.org/pdf/2307.09476)
- (6) Critical layers: GPT-J L13-14, Pythia-6.9B L9, Llama-2-7B L13-17, GPT-NeoX-20B-Instruct L10-13. For GPT-J, decoding at layer 16 beats the full model on all but one dataset; Llama-2-7B's p19 beats the full model on all 14. Zeroing late heads recovers almost all of the early-exit gain; zeroing late MLPs does much less (Table 1). — [2307.09476](https://arxiv.org/pdf/2307.09476)
- (6) Heads: selected on the toy set by a prefix-matching score; label-promoting score 6.5 vs -0.04 (SD 0.41) for 100 random 5-head sets. Ablation cuts the gap by 38.9% on average vs 3.97% for random heads. — [2307.09476](https://arxiv.org/pdf/2307.09476)
- (4) Quoted: "one promising future direction would be to replace head outputs by their average value, as in Nanda et al. (2023)". They also note overthinking even with **correct** demonstrations, a "misalignment between the pretraining objective and the downstream few-shot task". Follow-up: Belrose et al. (tuned lens) used the idea for prompt-injection detection. — [2307.09476](https://arxiv.org/pdf/2307.09476)
- (5) Their point on lens vs probe, worth copying: a high logit-lens accuracy "implies a high probing accuracy, but the reverse is not necessarily true." — [2307.09476](https://arxiv.org/pdf/2307.09476)

### Inferences
- The fate here is override by a **separate** late competitor (copying heads), with the pre-critical correct answer readable through the unembedding. Head ablation removes only part of the gap, so the rest is unexplained (possibly distributed or Hydra-style compensation).
- Reuse verdict: **copy** the calibrated accuracy, the early-exit baseline, the random-head control and the label-promoting score. **Avoid** zero-ablation (their own stated pitfall); use mean ablation.

---

### P7. How LLMs Are Persuaded: A Few Attention Heads, Rerouted (2605.09314v2; Sun, Kong, Zhang, Zeng, Wang; arXiv only)

### Takeaway
A precise **override-by-rerouting** account: mid-layer "decision heads" copy whichever option they attend to; a rank-1 QK "option-routing" feature on the key side sets which option wins; shallow heads reading persuasive keywords write that feature. A rank-2 fix to the decision heads' key projections restores 12-41 points of accuracy. The paper does not directly test whether the correct answer stays decodable.

### Cited Findings
- (1) Models: Llama-3-8B-Instruct, Gemma-3-12B-it (above 8B), OLMo-2-7B-Instruct, Qwen3-4B-Instruct-2507. Data: NQ2 from FARM (489 four-option questions x 9 appeals = 4,401 pairs per model) and GEO from Geo-Bench (998 pairs, injected instruction in one source). Licenses not stated. — [arXiv 2605.09314](https://arxiv.org/html/2605.09314)
- (6) Decision heads: L17H24 (Llama-3), L29H11 (Gemma-3), L18H19 (OLMo-2), L24H29/H31 (Qwen3). Llama-3 NQ2: clean accuracy 61.76%, persuasion success 46.65%. Patching L17H24's output raises correct-option probability by 47.28%; patching its **attention pattern only** gives 42.10%, so most of the effect is routing, not content. — [2605.09314](https://arxiv.org/html/2605.09314)
- (6) Rank-2 key correction on held-out NQ2: +12.30 to +41.27 points under persuasion, keeping 95.2-96.4% of clean decisions. Reversing the update costs 31-42 points. Random-head and matched-scale random corrections do nothing. — [2605.09314](https://arxiv.org/html/2605.09314)
- (3)(4) Limits: controlled MCQ only; GEO injects into a fixed source. Quoted: "whether the same circuit governs free-form generation is open." — [2605.09314](https://arxiv.org/html/2605.09314)

### Inferences
- Where the competing signal lives: a low-rank key-side feature written by shallow heads, read by one or two decision heads. This is more specific than P1's "shared heads". It is an attention-routing account: the value content of the correct option is likely still there, but it is not selected. That would be a clean **bypass/override** case if a probe confirmed the correct option stays decodable at the decision head's input.
- Reuse verdict: **copy** attention-only patching (separates routing from content) and the reversed-update control.

### Gaps
- Characters 100,000 onward of the HTML (appendices) were not read.

---

### P8. Broken Symmetry in LLM Refusal: Answer Release Is More Local Than Refusal Restoration (2608.15772v1; Manchester and SUFE; arXiv only)

### Takeaway
The cleanest **suppressed-but-intact** (bypass) evidence. During instructed withholding, the answer stays linearly recoverable (1.000 bits, answer rate 0.000). Patching one position from the answer run **releases** the answer (1.00 across L16-26), but patching the refusal run into the answer run does **not** restore refusal (stays at 0.50 baseline). So the answer is held back by a distributed refusal state, not erased. It holds at 7B; at 14B and 32B the hidden-state asymmetry holds but direction interventions are inert.

### Cited Findings
- (1) Models: Qwen2.5-7B-Instruct (main); Qwen2.5-14B/32B-Instruct (scale); Llama-3.1-8B, Mistral-7B-v0.3, Gemma-4-E4B, Gemma-4-26B-A4B (cross-family). Task: A/B "withhold" with matched answer and REFUSE runs; baselines Greater-Than, IOI, SVA, factual MC. — [arXiv 2608.15772](https://arxiv.org/html/2608.15772)
- (5) Protocol: bidirectional residual patching at position -1 and at (-1,-2,-3); readouts at L16-26; decoders: linear, MLP, affine lens, tuned lens, scored by a held-out cross-entropy bound. — [2608.15772](https://arxiv.org/html/2608.15772)
- (6) Widening the reverse patch to three positions at L16 drops answers 0.50 to 0.00 and raises refusal to 0.84. The mean answer-to-refusal displacement captures 84-89% of displacement energy but adding it does not reliably assemble refusal (answers fall to 0% at λ=0.5; refusal reaches 84% only at λ=2). — [2608.15772](https://arxiv.org/html/2608.15772)
- (5) Controls worth copying: a forced-distractor override in the same format is roughly **symmetric**, unlike refusal; out-of-set overrides; T=0.7 sampling; single-head, attention-only and MLP-only patches are null; 66 naturalistic prompts. — [2608.15772](https://arxiv.org/html/2608.15772)
- (4) Quoted: "Testing this hypothesis would require a controlled comparison between SFT-only and SFT+RL variants of the same base model." Not found done (no search spent). — [2608.15772](https://arxiv.org/html/2608.15772)

### Inferences
- Decodability (1.000) + release-by-local-patch + non-local suppression is the exact signature a bypass-vs-override study needs. The forced-distractor control is the key contrast: an override by a competing answer is symmetric, a suppression gate is not. **Copy** the design.

---

### P9. Breaking Refusal in the First Half: A Mechanistic Study of the Prefill Jailbreak (2607.14147v1; arXiv only; code at github.com/collapseindex/breaking-refusal)

### Takeaway
Under a prefill attack ("Sure, here is"), a harm probe still reads 0.91-0.98 on complied prompts while refusal collapses (e.g. Qwen2.5-3B 0.98 to 0.05). Harm information is **present but not used**. Causal handles are partial and non-selective: the harm direction at L12 restores refusal to 48% vs 14% random, but benign over-refusal rises to 40%. The authors' own base-model control suggests part of the effect is generic autoregressive conditioning on the prefill.

### Cited Findings
- (1) Qwen2.5-1.5B/3B, SmolLM2-1.7B, Phi-3-mini-3.8B (CPU); Qwen2.5-7B and Phi-3-medium-14B (GPU scale checks; refusal 1.00 to 0.15 on Phi-3-medium). — [arXiv 2607.14147](https://arxiv.org/html/2607.14147)
- (6) The first half of the response window carries the effect (42% refusal restored vs 6% for the second half, 9% onset token alone). Transplanting the model's own refuse-state into the early window restores 74% (random 0%). Harm strength predicts refusal only weakly where a natural split exists (AUC 0.63-0.72, n=70). — [2607.14147](https://arxiv.org/html/2607.14147)
- (5) Pitfall the paper flags: a benign-residual transplant also restores refusal to 100%, so full-residual restoration is generic disruption. The probe is calibrated to AdvBench phrasing. — [2607.14147](https://arxiv.org/html/2607.14147)
- (4) Quoted: "whether some other handle can un-collapse the behavior is left open"; "A source-general harm monitor is unbuilt". — [2607.14147](https://arxiv.org/html/2607.14147)

### Inferences
- Fate: **bypass** of the decision (harm decodable, refusal not driven), with weak endogenous coupling (AUC 0.63-0.72). Copy the benign-residual and mass-matched attention controls. Only one model above 8B, and the 7B refusal-string result did not replicate cleanly.

---

### P10. Rewired or Gated? How Instruction Tuning Shapes Knowledge-Conflict Circuits (2609.25602v1; arXiv only; the sweep's BlackboxNLP 2026 listing not confirmed)

### Takeaway
Base vs instruct Llama-3.2-3B, Qwen-2.5-3B, Gemma-3-4B on context-vs-memory conflict (ParaConflict, 764-870 prompts). Instruction tuning lowers context reliance (Llama 0.675 to 0.263) by **reweighting the same late heads** (top-10 Jaccard 0.67-0.82, Spearman 0.82-0.91), not by building a new circuit. It does not test whether the losing answer stays decodable. All models are under 8B.

### Cited Findings
- Edge overlap: Llama 0.61, Qwen 0.39, Gemma 0.54, all above the 0.30 "rewired" threshold. 36/36 pre-registered ablation signs matched. Random-head ablation leaves CRR unchanged. Under a coherent multi-sentence conflict the base-instruct gap nearly vanishes (Qwen 1.000 to 0.878). — [arXiv 2609.25602](https://arxiv.org/html/2609.25602)
- Quoted: "Whether the observed gating behavior extends to substantially larger instruction-tuned models remains an open question." — [2609.25602](https://arxiv.org/html/2609.25602)

### Inferences
- Agrees with P1 (post-training lowers the behavior but leaves the heads) in a different setting. Together they suggest **post-training gates existing channels**. Copy the pre-registered sign predictions.

---

### P11. Linear representations in language models can change dramatically over a conversation (2601.20834v2; abstract only)

- Factuality representations of conversation-relevant statements can flip over a conversation (factual to non-factual and back); generic information is preserved; replaying another model's script gives similar changes; a sci-fi story framed as fiction gives much weaker changes; steering has different effects at different points. — [arXiv 2601.20834](https://arxiv.org/abs/2601.20834)
- The authors warn "it may be misleading to use static interpretations of features or directions". — [2601.20834](https://arxiv.org/abs/2601.20834)
- Fate: **transformed** (the representation itself shifts, role-driven), but models and sizes are not on the abstract page. Strongest caution for any probe trained in one context and applied in another (it bears directly on P2's design). Verdict: adapt as a control (re-fit probes per context).

### P12. LLMs Encode Harmfulness and Refusal Separately (2507.11878v5; abstract only; sweep lists NeurIPS 2025, the arXiv page lists no venue)

- A harmfulness direction is separate from the refusal direction. Steering harmfulness changes the model's judgment; steering refusal triggers refusal without changing the harmfulness judgment. Some jailbreaks suppress refusal while the harmfulness judgment stays intact; adversarial finetuning barely moves it. A Latent Guard built on it matches or beats Llama Guard 3 8B. — [arXiv 2507.11878](https://arxiv.org/abs/2507.11878)
- Fate: **bypass** with causal evidence for two separable directions. Models and positions not on the abstract page.

### Not deep-read in the time box (tier 3-4)
- 2503.10996 JuICE, 2508.18760 Err Knowingly, 2410.02707 LLMs Know More Than They Show, 2503.15299 Inside-Out: not fetched. The sweep rows for them are in sweep_A/B/C and were not verified here.

---

## Add-on (coordinator, low priority): ICLR 2027 / other titles, short only

- "Sycophancy Is Often a Single-Layer Phenomenon: Spectral Diagnosis and Training-Free Weight Surgery in LLMs" (Valentin Noël) appears as an **ICML 2026** listing, not only as an ICLR 2027 submission. Per the search snippet, sycophancy is mediated by the dominant singular subspace of one MLP weight matrix in 10 of 11 open models (1B-14B); compressing it reduces sycophancy and amplifying it induces sycophancy; surgery fails on Mistral-7B-v0.3. A companion ICML 2026 paper by the same author is "Belief Without Justification: Sycophancy as a Single-Layer Truth–Compliance Tension in LLMs". Snippet-level only, not read. — [ICML listing 79313](https://icml.cc/virtual/2026/79313); [ICML listing 77441](https://icml.cc/virtual/2026/77441)
- This **contradicts P1's locus** (attention heads, MLP single-channel mediation null at 72B) at the level of where the competing signal lives. Possible reasons: weight-space SVD vs activation-space head attribution, 1-14B vs up to 72B. Unresolved.
- "Post-Training Reshapes Truth–Sycophancy Geometry": no page found by that title. The closest match is The Truthfulness Spectrum Hypothesis (2602.20273), Section 6 "Post-training Reorganizes Geometry": for Llama-70B and Llama-8B, probes reach 0.74 AUROC on base but 0.62 and 0.59 on chat (snippet). — [arXiv 2602.20273](https://arxiv.org/pdf/2602.20273). This partly conflicts with P1's "probe AUROC unchanged under DPO" (different probes, different targets).
- Also surfaced: "Gotta Catch them all: the modes of Sycophancy" (2607.20146), separable sycophancy modes from layer 14 (aggregator snippet only). — [tldr.takara.ai](https://tldr.takara.ai/p/2607.20146)
- Not reached: "Sycophancy and Jailbreak Compliance Share a Causal Axis", "Rescuing Internally Recalled Facts under Knowledge Conflict via Curvature-Triggered Residual Steering", 2606.20959, 2604.27167.

---

## Strand synthesis

### Takeaway
Across settings, the decodable correctness signal is most often **not erased**. Under refusal and jailbreaks it stays intact and is bypassed (P8, P9, P12). Under instructed lying at 70B it is **sign-flipped** (P5). Under false demos and persuasion, separate late or mid heads **override** it by copying or rerouting (P6, P7). The only erasure claim (P2) rests on a baseline-trained probe going below chance, which matches the transformed pattern of P5 more than erasure. Almost all of this is causal controllability; endogenous mediation is rarely tested, and the one direct test at 72B (P1) is null.

### Evidence Ledger

| Claim | Supporting | Contradicting | Strength |
|---|---|---|---|
| Correctness info stays decodable while the model acts against it (refusal/jailbreak) | P8 2608.15772 (1.000 bits); P9 2607.14147 (0.91-0.98); P12 2507.11878 | none found | Strong at ≤8B; moderate above 8B (P8 inert at 14B/32B; P9 one 14B model). All arXiv-only except P12 (claimed NeurIPS, unverified) |
| Under instructed lying the truth signal is transformed (sign-flipped), not erased | P5 2311.15131 (anti-parallel probes after L23, 70B) | P4 2507.22149 reads only the instructed label, no direct truth test | Moderate (one model, workshop-level preprint) |
| Sycophancy erases the correct answer at a late layer | P2 2607.00415 | P5's flip pattern; P1's probe transfer; P11's warning on static probes | **Weak** (no probe controls, probe trained out of context, outcome-conditioned subset) |
| Sycophancy is an override by a small set of attention heads | P1 2604.19117 (1.5B-72B); P3 2508.02087 (late layers, ~7B) | Single-Layer Phenomenon (ICML 2026, one MLP matrix, 1-14B; snippet only); P1's own 70B mean-ablation null | Moderate: strong at ≤27B, weak at 70B |
| False demos / persuasion override through late or mid copying heads | P6 2307.09476 (ICLR 2024, to 20B); P7 2605.09314 (to 12B) | none; P6 ablation leaves ~60% of the gap | Strong (P6), moderate (P7, arXiv) |
| Post-training gates existing channels rather than removing them | P1 (Llama-3.1 to 3.3-70B, DPO); P10 2609.25602 (3-4B) | Truthfulness Spectrum 2602.20273 (probe AUROC falls base to chat, snippet) | Moderate |
| Authority level matters to sycophancy | P2 (domain-matched, third-person roles) | P3 (≤4.4% shift, first-person competence) | Weak both ways; design-dependent |
| Endogenous mediation (the model naturally routes through the found component) | P9 weak (AUC 0.63-0.72) | P1 72B MLP mediation null | Weak; mostly untested |

### Disagreements and why
- **P2 vs P3 on authority:** institutional, domain-matched, third-person hint after the choices (MedQA) vs self-claimed competence level, random domain, before the question (MMLU). P3's own 13.6-point first-vs-third-person gap shows framing alone is large.
- **P2 vs P1/P5 on fate:** a probe trained on unpressured activations failing (P2) vs cross-task head transfer (P1) and honest-to-liar transfer with coefficient sign (P5). P2's below-chance accuracy is consistent with a transformed readout.
- **P1 vs Single-Layer Phenomenon on locus:** heads in activation space vs one MLP weight subspace; up to 72B vs up to 14B.
- **P4 vs P5 on layers:** no real conflict once depth is normalized (about 25-30% depth in both).

### Scoop risk for a bypass-vs-override project
- **High for sycophancy at ≤14B.** P1 (12 models, many controls), the Single-Layer papers and P3 cover the circuit and its post-training persistence.
- **High for refusal bypass at ≤8B.** P8 already has the decisive asymmetric-patching design.
- **Lower for:** (a) a single protocol that classifies fate (bypass / override / transformed / erased) **per item** across settings with the same controls; (b) the direct "is the correctness signal decodable at the action token" read that P1 and P7 lack; (c) above-8B endogenous mediation; (d) P2's erasure claim re-tested with in-context probes.

### Candidate open questions and the smallest first experiment
1. **Is P2's "erasure" a transformation?** Smallest experiment: on Gemma-2-9B-it with MedQA physician hints, train a correct-answer probe on **hinted** activations (held-out questions, shuffled-label control, selectivity), and check P5-style sign/rotation of the baseline probe (confusion matrix: does it predict the hinted letter?). If the in-context probe stays high, the fate is transformed, not erased.
2. **Does the correctness signal survive at the action token in sycophancy?** Smallest experiment: in one P1 model (Gemma-2-9B or Qwen2.5-32B), read P1's factual-wrongness direction at the agreement token on sycophantic vs resisted items; then do P8's bidirectional patch (release vs suppression) to test symmetry (override) vs asymmetry (gate).
3. **Does persuasion leave the correct option decodable at the decision head's input?** Smallest experiment: on Llama-3-8B NQ2 (P7), probe the value-side content at L17H24 for the correct option on persuaded items.
4. **Mediation above 8B.** Smallest experiment: measure whether natural variation in the shared-head write predicts behavior on held-out items (not intervened), at 32B and 70B.

### Gaps
- Several venues stated by the sweeps (ICML 2026 for P1, workshop for P2, EMNLP 2025 for P4, NeurIPS for P5 and P12, BlackboxNLP for P10) were not confirmed on the arXiv pages.
- OpenReview was not browsed (time box). Licenses were not stated in any paper read.
- WebFetch summaries can misreport numbers; P5 and P6 were read directly as PDFs, the rest through summarizing fetches.

## BLOCKED
- None requiring permission. Deviation to note: WebFetch auto-saved the arXiv PDFs of 2311.15131 (3.1 MB) and 2307.09476 (925 KB) to the session tool-results folder (/Users/shloki/.claude/projects/-Users-shloki-agent-failure-probes/260ce1f6-34ed-4464-864c-359284c95eac/tool-results/), not to downloads/<slug>/. Both came from arxiv.org and were read only with the Read tool; nothing was run.
