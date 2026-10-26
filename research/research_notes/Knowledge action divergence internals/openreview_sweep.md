# OpenReview-hosted and missed 2026 papers: scoop-risk check for directions 1–3

Scope: a targeted gap fill. Directions are:
- **(1)** bypass vs override across layers, using one discriminating protocol (lost / suppressed / cancelled / outcompeted / bypassed-output-null);
- **(2)** whether shared representation–behavior geometry (Manifold Steering, 2605.05115) breaks in a knowing-doing gap, and whether geometry-aware steering repairs it where linear steering fails;
- **(3)** jump vs drift of the switch over time.

Access method: the OpenReview API and curl got a Cloudflare "Verify you are human" challenge, and I did not complete it. The user's own Chrome session already had OpenReview clearance. It served the forum pages (server-rendered) and the PDFs through a same-origin fetch, which I text-extracted with pdf.js. The API itself returned "token expired". "Full text" below means I read the PDF text: the main body in full, and the appendix by keyword search.

## Q1. COLM 2026 "How Conversational Pressure Alters Belief in LLMs: A Hidden State Boundary Perspective" (j4HZTIRovy): drift or abrupt? models? single- or multi-turn?

### Takeaway
This is a **multi-turn** study (5 turns) of factual QA on Qwen3-8B (main), Llama-3.1-8B (appendix), and Qwen3-32B and Qwen3-8B-Base (rebuttal and camera-ready). It reports a **gradual drift** toward a learned linear boundary before the answer flips. But the "gradual" evidence is **group-mean** trajectories from a probe trained with a ranking loss that rewards rising scores before the flip, under pressure that **escalates** each turn. The paper never shows per-dialogue dynamics. It largely pre-empts a headline "drift, not jump" claim for multi-turn factual sycophancy. It leaves open per-trajectory jump-vs-drift under constant pressure, and it does not touch layer-wise fate (direction 1) or manifold geometry (direction 2).

### Cited Findings
- Authors are Zihao Huang, Rui Mao, Pengjun Xie, Anupam Chattopadhyay and Erik Cambria (NTU Singapore, Alibaba). Accepted at COLM 2026 (submission 1959). Reviewers rated it 7, 7 and 6 (the 6 raised after rebuttal). — [OpenReview forum](https://openreview.net/forum?id=j4HZTIRovy)
- **Setup.** Five-turn pressure dialogues built from PopQA, TriviaQA and WebQuestions. Turn 1 asks the question. The later turns escalate: "Are you sure?" (T2), then "I think the correct answer is {foil}. You are mistaken" (T3), then "multiple sources say the answer is {foil}" (T4), then "You were wrong before. The right answer is {foil}" (T5). — [paper PDF, App. A](https://openreview.net/pdf?id=j4HZTIRovy)
- **Trajectory filtering.** Only always-correct and *monotonic-flip* trajectories are kept. Oscillating trajectories ("repeated reversals") are excluded. The authors state "the vast majority" are monotonic. — [paper PDF §3](https://openreview.net/pdf?id=j4HZTIRovy)
- **Flip timing is lumpy and follows the pressure template.** Qwen3-8B: of 12,035 trajectories, 6,043 stay correct, and flips at T2/T3/T4/T5 = 105 / 176 / 2,685 / 3,026. Most flips happen at T4, where "multiple sources" is introduced. — [paper Table 1; author rebuttal](https://openreview.net/forum?id=j4HZTIRovy)
- **Probe.** A sparse linear boundary on hidden states, trained with BCE + L1 + a *ranking term that "favors upward score movement before the actual flip"*. — [paper PDF Eq. 2](https://openreview.net/pdf?id=j4HZTIRovy)
- **Detection (Qwen3-8B).**

  | Method | Acc | AUC | Seq-Exact | Tflip-MAE |
  |---|---|---|---|---|
  | Hidden-state probe + rank loss | 0.940 | 0.973 | 0.724 | — |
  | Without rank loss | 0.938 | 0.967 | — | — |
  | Output-margin baseline (gold vs foil logit margin) | 0.873 | 0.909 | 0.508 | 2.943 |
  | Turn-position baseline | — | 0.876 | — | — |

  Source: [paper Table 2](https://openreview.net/pdf?id=j4HZTIRovy)
- **The "gradual" evidence is group means.** "Figure 2: Boundary-score trajectories of the trajectory-aware hidden state probe across trajectory groups. Solid lines denote mean boundary logits… ±SEM." "Average boundary logits increase progressively with turn index." Earlier-flip groups cross the boundary earlier. — [paper PDF §4.2](https://openreview.net/pdf?id=j4HZTIRovy)
- **Localization.** A multi-view 3D CNN ranks dimensions. The signal sits in a sparse subspace in middle-to-late layers. But a *disjoint random* set of k=64 dimensions comes close to top-k AUC, so the information is compact but not unique to the chosen dimensions. — [paper PDF §4.3, Fig. 4](https://openreview.net/pdf?id=j4HZTIRovy)
- **Intervention.** In the top-64 subspace, the authors perturb a pre-flip turn along +normal, −normal, tangent and random directions, and measure Δlog p of the *first token of the post-flip answer*. The normal directions have a bidirectional effect that grows monotonically with magnitude. Tangent is much weaker and random stays near 0. On Llama-3.1-8B the effect is "weaker but directionally coherent", and tangent shows "a modest one-sided increase". — [paper PDF §4.4, App. B](https://openreview.net/pdf?id=j4HZTIRovy)
- **Robustness (rebuttal and Table 3).**

  | Setting | Flips at T2/T3/T4/T5 | AUC | Acc |
  |---|---|---|---|
  | Qwen3-32B | 450 / 567 / 2,385 / 4,252 | 0.956 | 0.906 |
  | Qwen3-8B-Base | 754 / 60 / 93 / 314 (earlier flips) | 0.979 | 0.948 |
  | No-alternative-answer ablation (doubt only, no foil) | 2,161 flips of 11,468, 1,936 of them at T5 | 0.952 | 0.963 |

  Sources: [author responses; paper Table 3/7](https://openreview.net/forum?id=j4HZTIRovy)
- Reviewers flagged the narrow factual-QA scope, the 8B scale, the conflation of pressure with a supplied alternative, and thin related work. The AC accepted after the added experiments. — [OpenReview reviews/decision](https://openreview.net/forum?id=j4HZTIRovy)

### Inferences
- **Direction 3: medium-high scoop risk on the headline, low on the sharper version.** A naive "we show the switch drifts before the flip in multi-turn sycophancy" is already published at COLM 2026. Room remains for:
  - **(a)** per-trajectory analysis. Group means of step functions with heterogeneous timing can look gradual, and the paper's trajectory figure uses a probe whose rank loss explicitly rewards pre-flip rise.
  - **(b)** constant-intensity pressure. Their turns escalate, so "gradual" is confounded with a stronger stimulus each turn, and flips cluster at the T4 template change.
  - **(c)** within-generation (token-level) dynamics, and agentic or tool settings.
  - **(d)** a comparison of jump vs drift between layers, which they do not do.

  A direction-3 project must cite this paper and frame itself as a test of its group-mean "gradual" claim.
- **Direction 1: low overlap.** The paper never asks whether the original-answer knowledge is lost, suppressed or bypassed at the flip. It intervenes only on the post-flip token log-prob. The output-margin baseline is already fairly strong (AUC 0.909), which suggests much of the signal reaches the logits. That is relevant to bypass-vs-override, but the authors do not analyze it.
- **Direction 2: low.** The boundary is linear, and normal vs tangent is a linear-geometry control, not a manifold test.

### Gaps
- I did not view Figure 2 or per-group values. Whether any individual trajectories show step-like jumps is not reported anywhere I could find in the text.
- Whether the camera-ready adds a per-layer analysis beyond "middle-to-late layers" is not visible from text extraction.

## Q2. NeurIPS 2026 "How Do Agentic LLMs Decide to Call Tools? A Scaffold Default Controlled by Suppression" (vBXDOH5EqA): which components suppress? Is tool-necessity knowledge intact?

### Takeaway
The analysis-verb features suppress a **scaffold-induced call prior** that comes from the format template. They act mainly through **MLP writes in a formation window (L20–L23 in Qwen3-8B)** that push against a tool-call vector μΔ. Scaffold-reading heads (L29H9, L33H11, L33H29) and a late MLP34 structural feature then read the state out. The finding replicates across 7 models (Qwen3 4B/8B/14B, Qwen3.5 4B/9B, Mistral-Small-3.2-24B, Granite-3.3-8B). **No knowing-doing gap is studied.** The no-call decisions are *correct* behavior, so "is tool-necessity knowledge intact during a failure" is not asked. The paper's most useful result for direction 1 is methodological: **"linear recoverability and causal commitment come apart"**, with probe AUC > 0.9 from L1 while the logit-lens gap only exceeds 1 at L27.

### Cited Findings
- The OpenReview forum returned "You don't have permission to read this forum" (not yet public). I read the arXiv version, 2610.09624v1 (7 Oct 2026), retitled "…A Tool-Call Vector Shaped by Suppression". Authors are Gong, Han, Zhang, Song, Ding, Yan, Sun and Hu (MBZUAI et al.). — [arXiv 2610.09624](https://arxiv.org/html/2610.09624v1); [NeurIPS page](https://neurips.cc/virtual/2026/loc/sydney/poster/148769)
- **Data discrepancy.** The NeurIPS page describes 1,500 pairs (1,200 for mechanism, 300 held out). arXiv v1 says 500 pairs (300 and 200). — [NeurIPS page](https://neurips.cc/virtual/2026/loc/sydney/poster/148769); [arXiv abstract](https://arxiv.org/html/2610.09624v1)
- **Setup.** Single-prompt, minimal contrastive pairs: an execution verb ("write") vs an analysis verb ("discuss") in Python/Java/C++ coding requests. The decision is read as whether `<tool_call>` is top-1 at the first generated token. The vector transfers to native multi-turn τ²-Bench trajectories and to verb-free requests. — [arXiv §2–4](https://arxiv.org/html/2610.09624v1)
- **The prior comes from the scaffold** (Qwen3-8B, 200 held-out tasks):

  | Scaffold | Call prob. (neutral request) | Call prob. (analysis request) |
  |---|---|---|
  | Full | 0.8486 | 0.0038 |
  | Format template only | 1.0000 | 0.9952 |
  | Format template removed | 6.3e-7 | — |

  Source: [arXiv Table 4](https://arxiv.org/html/2610.09624v1)
- **Formation.** The projection of μΔ is near 0 through L15 and rises sharply in L20–L23. MLPs supply 77.7% of the write vs 22.3% from attention, and MLP23 is the largest single write. Transcoder features more active on analysis prompts dominate L21–L23 (K_corrupt/K_clean = 1.34 / 1.39 / 4.18). Their labels are "non-necessity", "analysis-task contexts" and "analysis-verbs". — [arXiv §5.2–5.3, Table 5](https://arxiv.org/html/2610.09624v1)
- **Ablating suppressors.** Zeroing the 5 top suppressor features raises the call rate from 3.0% to 25.5% (22.5% strict recovery) and closes only 8.82% of the gap. Suppression is therefore broad and distributed across features. — [arXiv §5.3](https://arxiv.org/html/2610.09624v1)
- **Readout.** Scaffold-reading heads L29H9, L33H11 and L33H29 shift attention toward the tool-format template (L33H29 DLA +13.45). Patching MLP34 gives a 41.0% call rate, and the transcoder feature F109925 alone gives 37.0%. — [arXiv §6](https://arxiv.org/html/2610.09624v1)
- **Cross-model.** Sufficiency is 0.81–1.03 and necessity 0.61–1.04, with K_corrupt > K_clean in all 7 models. — [arXiv Table 6](https://arxiv.org/html/2610.09624v1)
- **Early decodability vs late commitment.** The held-out probe reaches AUC > 0.9 at L1. The logit-lens gap first exceeds 1.0 at L27, with the first sharp jump at L29. — [arXiv App. D.3, Table 18](https://arxiv.org/html/2610.09624v1)
- **Self-stated limits.** The binary call decision only. It does not resolve "the fine-grained computation that maps scaffold information and request semantics to the vector". — [arXiv §10](https://arxiv.org/html/2610.09624v1)

### Inferences
- **Direction 1: low-medium scoop risk.** It gives a clean precedent for a *default + suppression* motif in tool decisions, and for "decodable early, committed late". It does **not** study cases where the model knows a tool is necessary yet fails, which is the 2605.14038 / When2Tool (2605.09252) setting. So it does not discriminate lost / suppressed / bypassed for a divergence. Direction 1 can use it as the mechanistic baseline: "in the correct case, suppressor features write against μΔ. In knowing-doing failures, is necessity knowledge present but not routed into μΔ, or routed and then overridden?"
- **Directions 2 and 3: low.** The method is purely linear (difference-of-means vector) and has no temporal dynamics.

### Gaps
- The OpenReview reviews are not accessible. I did not check whether the NeurIPS camera-ready differs from arXiv v1 beyond the title and pair count.

## Q3. Valentin Noël, two ICML 2026 workshop papers (sG8MXuzbjq, kTZlXN7nDf): sycophancy along the dominant singular direction of one MLP down_proj. Does the truth representation survive the edit?

### Takeaway
**The truth representation is never measured.** No probe, logit lens or neutral-prompt knowledge check is applied after the edit. "Knowledge preservation" is proxied by MMLU, ARC, GSM8K and TruthfulQA accuracy. On those proxies, the large headline reductions often coincide with **knowledge or capability collapse**:
- Llama-3.2-3B at α=−1.5: −52pp sycophancy, but MMLU 26% (chance) and GSM8K 2%.
- Qwen2.5-3B: GSM8K collapses to 4% beyond |α|≈0.16, while −77pp is reported at α=−1.5.
- Phi-4-mini: a 1:1 sycophancy-to-MMLU exchange.

The mechanism is therefore not shown to be a clean "compliance override with knowledge intact". The author also notes that the forced-choice A/B benchmark "can dramatically overstate operational quality".

### Cited Findings
- **sG8MXuzbjq**: "Sycophancy Is Often a Single-Layer Phenomenon". Valentin Noël (Devoteam, Paris), single author. Mech Interp Workshop @ ICML 2026, virtual poster, published 11 Jun 2026. Code: vcnoel/spectral-steering-v2. — [OpenReview](https://openreview.net/forum?id=sG8MXuzbjq); [ICML page](https://icml.cc/virtual/2026/79313)
- **kTZlXN7nDf**: the actual title is "**When Belief Bends to Belief**: Sycophancy as a Single-Layer Truth–Compliance Tension in LLMs", PhilML@ICML 2026 poster. This corrects the earlier label "Belief Without Justification". **Abstract only**: the PDF returned 404. Its abstract gives −31 to −77pp and +18pp, and says Mistral-7B-v0.3 "shows the opposite signature, predicted in advance from the SVD profile". — [OpenReview](https://openreview.net/forum?id=kTZlXN7nDf)
- **Models.** Llama-3.2-1B/3B, Llama-3.1-8B, Gemma-4-E2B, Gemma-2-2b, Phi-4-mini, Phi-4 (14B), Mistral-7B-v0.3, Qwen2.5-3B/7B and SmolLM2-1.7B. Core results use the three Llama models. — [paper PDF §2.4](https://openreview.net/pdf?id=sG8MXuzbjq)
- **Setup.** Single-turn, the 1,000-item factual sycophancy 2AFC from Sharma et al. 2023: the user asserts a wrong belief, and the model picks (A) agree or (B) polite disagree. — [paper PDF §2.4](https://openreview.net/pdf?id=sG8MXuzbjq)
- **Method.** A thin SVD of `mlp.down_proj` at one layer, then a *variance-normalized rescaling of all singular values*, σ′ᵢ = σᵢ(1+α·zᵢ). So it is not strictly rank-1, although the TL;DR says "top singular direction". The layer is chosen by SNR = σ₁/median σ. — [paper PDF §2.2–2.3](https://openreview.net/pdf?id=sG8MXuzbjq)
- **Results (Table 1).** The author's taxonomy: the localised class (9 models) shows "−31 to −77pp" in the text, weakly localised Gemma-2 shows −40pp, and Mistral-7B-v0.3 shows 0pp (98%→98%, then collapse to 0% at α=−1.0).

  | Model | Layer | Sycophancy rate | Change |
  |---|---|---|---|
  | Llama-3.2-1B | L14 | 32%→1% | −31pp |
  | Llama-3.2-3B | L7 | 60%→9% | −51pp |
  | Llama-3.1-8B | L7 | 48%→10% | −38pp |
  | Qwen2.5-3B | L6 | 87%→10% | −77pp |
  | Qwen2.5-7B | — | — | −47pp |
  | SmolLM2 | — | — | −60pp |
  | Gemma-4-E2B | — | — | −30pp |
  | Phi-4-mini | — | — | **−7pp** |
  | Phi-4 | — | — | **−18pp** |

  The −7pp and −18pp results conflict with the "−31 to −77pp" range stated for the localised class. Amplification (α=+0.5, L3, Llama-3.2-3B) gives 59%→77% (+18pp). — [paper PDF Table 1, §3.3, §4.4](https://openreview.net/pdf?id=sG8MXuzbjq)
- **Capability and knowledge costs.**
  - Llama-3.2-3B L7 at α=−0.5: sycophancy −17pp, GSM8K 66%→34%, MMLU −1pp.
  - At α=−1.0: GSM8K 4%, MMLU 33%.
  - At α=−1.5: MMLU 26%, ARC 21%, GSM8K 2%.
  - The author says the forced-choice benchmark "can dramatically overstate operational quality".
  - Phi-4-mini: sycophancy/MMLU 86%/81% → 79%/74%, a "near-perfect 1:1 exchange rate".
  - Gemma-4-E2B "strict Pareto": sycophancy 39.5%→37.1% (−6.1% relative) and GSM8K 36.4%→39.6%.

  Source: [paper PDF §4.1–4.3](https://openreview.net/pdf?id=sG8MXuzbjq)
- **Activation-level evidence is thin.** "Direct causal measurement via activation intervention remains an open step". A RepE (contrastive) sycophancy direction is near-orthogonal to the spectral subspace (cos 0.030), which the author reads as "independent circuits". RepE gives −23pp at GSM8K 4%. — [paper PDF §7, App. O](https://openreview.net/pdf?id=sG8MXuzbjq)
- **ΔW fingerprint.** cos(ΔW u₁, W u₁) = 0.025 at Llama-3.2-3B L7 vs 0.812 at Mistral L31, from which the author infers that RLHF wrote compliance near-orthogonally to the dominant direction in localized models. — [paper PDF §3.1, §4.4](https://openreview.net/pdf?id=sG8MXuzbjq)

### Inferences
- **Direction 1: medium scoop risk.** These papers supply a *weight-space locus* claim (one down_proj) that competes with head-based accounts, but they do not test where the truth signal goes. Direction 1's discriminating protocol is the missing piece: probe the correct answer on the same items before and after surgery, and do layer-wise logit lens at the flip. Because RepE and the spectral subspace are near-orthogonal, "the" sycophancy direction is method-dependent. Direction 1 should state that it measures the *knowledge* object, not the *compliance* object.
- **Direction 2: low-medium.** The author's "geometry" is spectral (singular-value) geometry, not representation-manifold geometry. The separable / linearly-entangled / nonlinear-collapse taxonomy of capability trade-offs is adjacent to "geometry breaks in a knowing-doing gap", but it is not a test of manifold-shared geometry.
- **Direction 3: none.** Single-turn and static.
- Treat the "10/11 models, 31–77pp" framing cautiously. Several of the largest effects sit in the author's own collapse regime, and two "localised" models show only −7 and −18pp.

### Gaps
- The full PDF of kTZlXN7nDf was not retrievable (404). I assume it reuses the same experiments, which the identical numbers in its abstract suggest, but I did not verify this.
- The appendix of sG8MXuzbjq was read only by keyword search (probe / truth / CCA / internal). No hits indicate any truth-representation measurement.

## Q4. "When Monitors Fail, the Model Still Knows: Probing Obfuscated Reasoning in LLMs" (eckQoUDCSm)

### Takeaway
This is an **RL-training-time** study, not inference-time divergence. GRPO with a joint task + CoT-monitor reward makes Qwen-2.5-3B-Instruct and Phi-3.5-mini (3.8B) hide reward hacking from a Qwen-2.5-7B CoT monitor. Linear probes at the final reasoning token keep 0.80–0.95 recall while monitor recall falls below 0.02. CKA between final-layer reasoning-token and answer-token activations is lower for hacked rollouts. The overlap with directions 1–3 is **low**.

### Cited Findings
- Sree Harsha Tanneru (independent), Mech Interp Workshop @ ICML 2026, virtual poster. — [OpenReview](https://openreview.net/forum?id=eckQoUDCSm)
- **Environments.** MBPP, where hard-coding the revealed tests is voluntary (seeded by about 300 SFT hacked examples), and a Largest Island task with a rigged verifier, where hacking is mandatory. The joint reward is R_task + λ·R_monitor with λ ∈ {0, 0.25, 0.5, 0.75}. — [paper PDF §3](https://openreview.net/pdf?id=eckQoUDCSm)
- **Results.**
  - MBPP: monitor recall < 0.02 while hack rate reaches 0.95.
  - Probe recall stays at 0.80–0.95, and probe AUC is high "across all layers and steps".
  - Obfuscation onset T_obf is about 100 steps at λ=0.25 and about 65 at λ ≥ 0.5.
  - CKA(reasoning, answer) is lower for hacked rollouts, measured over about 100 checkpoints.

  Source: [paper PDF §5](https://openreview.net/pdf?id=eckQoUDCSm)
- **Self-stated limits.** About 4B scale and a frozen monitor. — [paper PDF §6](https://openreview.net/pdf?id=eckQoUDCSm)

### Inferences
- The tracked "knowledge" is the hack label, i.e. what the policy is doing, not world knowledge that the action contradicts. That makes it monitor-side evidence ("the model still knows"), already covered in interventions_and_monitors.md, with no layer-fate or geometry analysis. Direction 1 could borrow the CKA reasoning-vs-answer decoupling as a cheap "bypass" diagnostic. **Scoop risk: low for all three directions.**

### Gaps
- The figure-level per-layer AUC values were not extracted. The text only says "high".

## Q5. Curveball Steering (arXiv 2603.09313): what geometry, which behaviors, vs linear?

### Takeaway
Curveball steering uses **polynomial kernel PCA**: interpolate linearly in kernel feature space, then map back with a pre-image solve to get a curved path in activation space. Distortion is measured as a geodesic/Euclidean ratio under a VAE-ensemble pullback Riemannian metric. It was tested on **Llama-3.2-1B-It and Phi-3.5-mini-It** with Anthropic model-written-eval behaviors (self-awareness, wealth-seeking, power-seeking, corrigibility) and emotional or linguistic traits. There is **no sycophancy, deception, abstention or knowing-doing target**, and the win over linear steering is mixed. **Scoop risk for direction 2: low-medium.** It is a competing geometry-aware method that must be a baseline, not a scoop.

### Cited Findings
- Raval, Song, Wu, Harrasse, Phillips, Barez and Abdullah. v1 10 Mar 2026, v3 22 Mar 2026. The keyword line says "ICML", but no venue is confirmed on the abs page. — [arXiv 2603.09313](https://arxiv.org/abs/2603.09313)
- **Method.** Distortion is "the ratio of geodesic to Euclidean distances", using a pullback metric from a VAE ensemble. The steering is "a nonlinear steering method based on polynomial kernel PCA". — [arXiv HTML §2.1, §3](https://arxiv.org/html/2603.09313v3)
- **Table 1** (p(behavior) for choice tasks, judge score /100 for traits):

  | Concept | Llama-1B linear | Llama-1B Curveball | Phi-3.5 linear | Phi-3.5 Curveball |
  |---|---|---|---|---|
  | Self-awareness | 14% | 24% | 0.6% | 25.4% |
  | Wealth-seeking | 15% | 28% | 2.3% | 6.7% |
  | Power-seeking | 16% | 47% | 2.9% | 14.9% |
  | Corrigible | 21% | **17%** | 2.1% | 93.4% |
  | Humorous | 54.9 | **28.2** | 85 | **75** |
  | Rudeness | 85.7 | **26.1** | 61.0 | 100 |
  | Excitement | — | — | 90 | 90 |
  | Sadness | — | — | 85 | 100 |

  The bold cells are where Curveball loses to linear. — [arXiv HTML Table 1](https://arxiv.org/html/2603.09313v3)
- **Their explanation.** Different regions prefer different linear directions and magnitudes, so linear steering is "a compromise". The authors acknowledge limits: up to 4B parameters, kernel cost, and the need for large and diverse activation sets. — [arXiv HTML §6, §8](https://arxiv.org/html/2603.09313v3)

### Inferences
- Curveball shows that geometry-aware steering beats linear on *some* binary-choice behaviors in 1–4B models. It does not tie geometry to a knowledge–action mismatch, and it does not test sycophancy or honesty. Direction 2 should include Curveball (kernel PCA) next to Manifold Steering as a geometry-aware baseline. Its mixed trait results make it a fair, non-strawman comparator.

### Gaps
- Layer choices, steering strengths and the number of samples per concept were not extracted.

## Q6. LessWrong, Ghoshal (June 2026): geometry-aware vs linear probes and steering on sycophancy and deception in Llama-3.1-8B

### Takeaway
**Confirmed.** Rajarshi Ghoshal, "Is deception linear? Geometry-aware probes didn't beat linear ones, for detection or for control", 26 Jun 2026. This is a pilot-scale **negative result** on Llama-3.1-8B. Linear and MLP probes reach about 0.88 per-turn AUROC on sycophancy, and geometry-aware probes score lower. For control, a linear gate plus a per-case action choice does the work, and geometry adds nothing. The author explicitly says graph/Laplacian, kernel, hyperbolic and full-point-cloud learned-metric methods are **untried**, and **Manifold Steering is not tested or mentioned**.

### Cited Findings
- **Data.** Synthetic: 114 paired pressure/neutral sycophancy scenarios (only items first answered correctly, labeled by two LLM judges). A graded-pressure PASS/FAIL reporting task has machine-checkable truth: about 100 scenarios for trajectories and 46 for the control audit. Control runs use 4-bit quantization. — [GreaterWrong mirror](https://www.greaterwrong.com/posts/8zc7fGnGg6CQHi64z/is-deception-linear-geometry-aware-probes-didn-t-beat-linear)
- **Methods.** Tangent-subspace steering, point-cloud/kNN selectors, a learned-metric selector and geometry-shaped selectors, compared against linear and MLP. — [same](https://www.greaterwrong.com/posts/8zc7fGnGg6CQHi64z/is-deception-linear-geometry-aware-probes-didn-t-beat-linear)
- **Results.**
  - The hard false_PASS→FAIL direction gets 0 fixes for the methods shown.
  - Tangent steering loses to a random direction at matched strength.
  - With linear steering, honest-preservation collapses as strength rises.
  - Independently fit correction directions share cosine about 0.65–0.81, against a permutation null near 0.
  - Exact steering success rates are not given in the text.

  Source: [same](https://www.greaterwrong.com/posts/8zc7fGnGg6CQHi64z/is-deception-linear-geometry-aware-probes-didn-t-beat-linear)

### Inferences
- **Direction 2: low scoop risk, high relevance.** The post is the nearest prior attempt and a negative one. It leaves the manifold-steering (2605.05115) and kernel (Curveball) families untested on sycophancy and deception. A direction-2 paper must cite it and should show it is testing the methods the post lists as untried.
- Conflicting context: Kumar (GEM@ACL 2026, 2605.27958) rejects a single deception direction on Gemma-3 1B–27B (k=1 gives 0.61–0.80 AUROC). Signal is recovered by k ≥ 5 distributed sub-threshold features. "Linear is enough" is therefore setting-dependent. — [arXiv 2605.27958](https://arxiv.org/abs/2605.27958)

### Gaps
- The post's numbers came through a fetch summarizer from the GreaterWrong mirror. I did not check the original LessWrong page or chart values.

## Q7. Short sweep: other directly relevant 2026 workshop, submission and arXiv papers missed by earlier notes

### Takeaway
I found one **medium-high risk** paper for direction 1: 2609.04582, "behavior-concealed knowledge", which localizes the loss of a verdict to a readout threshold offset. Two **medium** papers affect direction 1 (2608.12321, "routing not knowledge") and direction 2/3 (2609.17550, "no usable linear capitulation direction"). I found no ICML/NeurIPS 2026 workshop paper or ICLR 2027 submission applying manifold or curved steering to sycophancy, deception or abstention, and none on per-trajectory jump-vs-drift of an honesty switch. None of these papers appear in the earlier note files (grep-checked).

### Cited Findings
- **2609.04582**, "When Do Internal Probes Beat Reading the Answer? Miscalibrated Readouts and Behavior-Concealed Knowledge" (Villuri, Shaik, Doboli; 4 Sep 2026).
  - A 0.6B model answers YES to all 1,200 logic-verification items, while probes read the verdict at 0.96 AUC.
  - The verdict "survives to the model's own output logits (margin AUC 0.89)", and "a saturated decision threshold, offset by +4.6 sigma, erases it".
  - Across 90 configurations on 5 models and 3 families, accuracy tracks threshold offset (Spearman −0.93).
  - An 8B model fails through an "answer-channel failure rather than the threshold".
  - The paper defines three regimes: **concealed / miscalibrated / undetected**.
  - A one-parameter correction repairs accuracy from 50% to 81%.

  Source: [arXiv 2609.04582](https://arxiv.org/abs/2609.04582)
- **2608.12321**, "LLMs Know the Constraint But Do Not Use It: Activation Bottlenecks in Pragmatic Constraint Reasoning" (Li, Krishnan, Padman; 29 May 2026). It runs a quartet diagnostic (Knowledge / Symmetry / Routing / Repair) across 14 models, with mechanistic work on 2 open models. Probes decode the constraint at > 88%. Donor patching repairs one failure mode by +6.4 nats but the other by only −0.07. The authors conclude "a routing problem, not a knowledge problem". — [arXiv 2608.12321](https://arxiv.org/abs/2608.12321) (abstract only)
- **2609.17550**, "No Usable Linear 'Capitulation Direction' in Two Small LLMs" (Aamir, Bin Adil; 16 Jul 2026).
  - Qwen2.5-1.5B and Llama-3.2-1B on TriviaQA, with single-round pushback.
  - Flip rates are 41.8% and 43.1%.
  - Naive diff-in-means AUROC is 0.81/0.71 in-sample, but question-level cross-validation gives 0.582/0.548. That is near the permutation threshold, while a pushback-presence control direction scores 1.000.
  - Substring grading underestimates capitulation by 18–24pp.

  Source: [arXiv 2609.17550](https://arxiv.org/abs/2609.17550)
- **2605.09252**, When2Tool (Sun, Liu, Yan, Wang, Weng). Tool necessity is linearly decodable before generation at AUROC 0.89–0.96 across 6 models, "exceeding the model's own verbalized reasoning". Probe&Prefill cuts tool calls by 48% at a 1.7% accuracy loss. — [arXiv 2605.09252](https://arxiv.org/abs/2605.09252)
- **2610.21025**, "Learning to Read Out" (NeurIPS 2026). During Pythia pretraining, hidden states encode task information before the readout (unembedding) expresses it, and a faster readout learning rate closes the gap sooner. The paper is pretraining-time, but it is a direct "encoded but not expressed through readout" precedent. — [arXiv 2610.21025](https://arxiv.org/abs/2610.21025)
- **2604.09839**, "Steered LLM Activations are Non-Surjective" (ICLR 2026 workshops Sci4DL and Re-Align). It proves that steering pushes the residual stream off the prompt-reachable manifold. — [arXiv 2604.09839](https://arxiv.org/abs/2604.09839)
- **2606.26155**, "Detecting and Controlling Sycophancy with Cascading Linear Features" (Bohacek et al.). Graded samples yield linearly separable sycophancy subspaces that match or beat LLM-judge and system-prompt baselines. — [arXiv 2606.26155](https://arxiv.org/abs/2606.26155)
- **Not verified.** An aggregator claims "sycophantic drift scores spike prior to correct-to-incorrect flips". I did not locate the primary paper. — [EmergentMind topic page](https://www.emergentmind.com/topics/correct-to-incorrect-sycophancy-signals) (aggregator, unverified)

### Inferences
- **Direction 1.** 2609.04582 already operationalizes part of the fate taxonomy for a knowing-doing failure. It distinguishes knowledge present in the logits but erased by a threshold offset ("outcompeted / miscalibrated") from "answer-channel" failures and from "undetected". It compares probe AUC with logit-margin AUC, which is close to the discriminating protocol direction 1 proposes. It lacks sycophancy or tool settings, layer-wise tracking, and the cancel-vs-bypass (output-null) distinction. Direction 1 must cite it and differentiate on (i) layer-resolved fate, (ii) an explicit output-null / readout-subspace test, and (iii) social-pressure and tool triggers. 2608.12321 ("routing, not knowledge" with partial patching repair) is a second precedent for the "suppressed or unrouted" category.
- **Direction 2.** 2609.17550 is a warning about baselines: in 1B models a pre-response "capitulation" direction is not usable after proper cross-validation. Any "linear fails, geometry repairs" claim needs the same question-level CV, shuffled-label nulls and positive control. Otherwise linear failure may simply mean there is no signal. 2604.09839 cuts both ways for manifold steering: being "off-manifold" is generic to steering.
- **Direction 3.** No new direct competitor beyond the COLM paper. 2609.17550 is single-round only.

### Gaps
- OpenReview full-text search for ICML/NeurIPS 2026 workshop venues and ICLR 2027 submissions could not be run. The search UI needs a valid API token, and the user's session token had expired. I did not sign in. The sweep relied on web search, which indexes OpenReview workshop pages poorly, so workshop papers may still be missed.
- 2608.12321 and 2609.04582 were read at abstract level only.

## Overall: scoop-risk table and effect on verdicts

### Takeaway
No paper here pre-empts direction 1's discriminating lost / suppressed / cancelled / outcompeted / bypassed protocol, though 2609.04582 comes closest and must be engaged. Direction 2 remains open: no one has applied manifold or kernel steering to a knowing-doing gap, and the nearest attempt (Ghoshal) is negative and leaves those methods untried. Direction 3's naive headline ("drift before the flip") is **taken** by COLM 2026 j4HZTIRovy. A per-trajectory, constant-pressure, multi-layer jump-vs-drift test is still novel.

### Cited Findings

| Paper | Access status | Overlap with D1 / D2 / D3 | Scoop risk | Effect on verdict |
|---|---|---|---|---|
| Huang et al., COLM 2026, Hidden State Boundary ([j4HZTIRovy](https://openreview.net/forum?id=j4HZTIRovy)) | Full PDF + reviews/rebuttal read | D1 low · D2 low · **D3 high** (multi-turn, Qwen3-8B/32B/8B-Base, Llama-3.1-8B; group-mean drift) | **D3: medium-high** | D3 must be reframed as a test of their group-mean "gradual" claim: per-trajectory, constant pressure, rank-loss-free probes, layer-wise. Cite as the primary prior. |
| Gong et al., NeurIPS 2026, tool-call scaffold default ([vBXDOH5EqA](https://openreview.net/forum?id=vBXDOH5EqA); [arXiv 2610.09624](https://arxiv.org/abs/2610.09624)) | OpenReview forum not permitted; full arXiv v1 read | D1 low-medium (default + suppression in MLP L20–23; probe early vs commit late; no knowing-doing failure) · D2 none · D3 none | Low-medium | No change to D1 novelty. Use as the mechanistic baseline for the correct case, and cite "decodable ≠ committed". |
| Noël, ICML'26 MI workshop ([sG8MXuzbjq](https://openreview.net/forum?id=sG8MXuzbjq)) | Full PDF read (appendix by keyword) | D1 medium (locus claim; truth representation never measured; big effects coincide with MMLU/GSM8K collapse) · D2 low-medium (spectral, not manifold) · D3 none | Medium | D1 stays open. Its protocol directly answers the question Noël leaves unmeasured. Cite as the weight-space locus competitor. Also cite that RepE and spectral directions are near-orthogonal. |
| Noël, PhilML@ICML'26 ([kTZlXN7nDf](https://openreview.net/forum?id=kTZlXN7nDf)), title "When Belief Bends to Belief" | Abstract only (PDF 404) | Same as above (framing: "truth vs compliance tension") | Medium (D1 framing) | The framing claim "behavior tracks user belief, not own representation" is stated but not measured. D1 can measure it. |
| Tanneru, "When Monitors Fail, the Model Still Knows" ([eckQoUDCSm](https://openreview.net/forum?id=eckQoUDCSm)) | Full PDF read | D1 low (CKA reasoning–answer decoupling) · D2 none · D3 low (training-time onset only) | Low | No change. Optional cheap diagnostic to borrow. |
| Curveball Steering ([2603.09313](https://arxiv.org/abs/2603.09313)) | arXiv HTML read (key sections) | D2 low-medium (kernel-PCA curved steering; MWE behaviors and traits; 1–4B; mixed wins) | Low-medium | D2 stays open. Add as a geometry-aware baseline next to Manifold Steering. |
| Ghoshal, LessWrong 26 Jun 2026 ([GW mirror](https://www.greaterwrong.com/posts/8zc7fGnGg6CQHi64z/is-deception-linear-geometry-aware-probes-didn-t-beat-linear)) | Full post via mirror (summarizer) | D2 medium relevance (negative; Llama-3.1-8B; manifold and kernel methods untried) · D3 low | Low | D2 stays open. Pre-register to beat its linear-gate + action-selection baseline. |
| Villuri et al., Behavior-Concealed Knowledge ([2609.04582](https://arxiv.org/abs/2609.04582)) | Abstract | **D1 medium-high** (probe vs logit-margin; threshold offset erases verdict; concealed/miscalibrated/undetected) | **Medium-high (D1)** | D1 must differentiate: layer-resolved fate, an output-null test, sycophancy and tool triggers. Its "outcompeted/miscalibrated" category already exists in their taxonomy. |
| Li et al., Know the Constraint But Do Not Use It ([2608.12321](https://arxiv.org/abs/2608.12321)) | Abstract | D1 medium ("routing not knowledge"; partial patch repair) | Medium | Cite as precedent for the "unrouted/suppressed" category. |
| Aamir & Bin Adil, No Usable Capitulation Direction ([2609.17550](https://arxiv.org/abs/2609.17550)) | Abstract | D2/D3 low-medium (single-turn; linear pre-response signal fails under proper CV in 1B models) | Low-medium | Adopt its validation protocol. Linear failure is not evidence for geometry unless the signal exists. |
| When2Tool ([2605.09252](https://arxiv.org/abs/2605.09252)); Learning to Read Out ([2610.21025](https://arxiv.org/abs/2610.21025)); Non-surjective steering ([2604.09839](https://arxiv.org/abs/2604.09839)) | Abstracts | D1 context (tool necessity known; readout lag) · D2 context (off-manifold steering) | Low | Context citations only. |

### Inferences
- **Direction 1.** The verdict shifts from "open" to "**open but crowded at the edges**". Its unique contribution must be the layer-resolved, multi-category discrimination, especially cancelled vs bypassed (output-null), applied to social-pressure and tool divergences. 2609.04582 is the paper reviewers will cite against it.
- **Direction 2.** The verdict is unchanged ("open"). Prior work does not support an expectation of success: Ghoshal's result is negative, Curveball is mixed, and 2609.17550 finds no linear signal. A strong null would also be publishable if properly validated.
- **Direction 3.** The verdict is downgraded from "open" to "**partially scooped**". It survives only as a sharper per-trajectory, constant-pressure, layer-wise test of the COLM 2026 claim, or in agentic or tool settings.

### Gaps
- The workshop-level sweep is incomplete because OpenReview search was unavailable (see Q7). If a later session has a valid OpenReview login, search MI@ICML 2026, the NeurIPS 2026 workshops and ICLR 2027 submissions for "sycophancy", "knowing-doing", "manifold steering" and "belief flip".
