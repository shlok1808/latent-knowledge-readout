# Sweep D: geometry of representation and behavior, divergence geometry, and trajectory drift vs. jump

Sweep date: 2026-10-22. This is a search sweep only: I read abstracts, arXiv abstract pages and search snippets, with no full-text extraction. Skill used: `deep-research` (researcher reference).

**Stopping rule NOT met.** The run ended because the shared per-turn web-search budget (200 calls) ran out at query 72. It did not end on 10 consecutive empty queries. The last 10 completed queries still added 1 to 3 new papers each, mostly Supporting or Peripheral ones. See the BLOCKED list.

Wording convention: "correctness-relevant information is decodable" (CRID) means a probe or readout recovers it. It does not mean "the model knows".

---

## 0. Short answers to the key questions (from abstracts and snippets only)

**Q1. Which papers measure the geometry linking a representation to the behavior it drives, and does any test it when the model acts against decodable correct information?**
- **Rep-to-behavior geometry papers in general settings (not knowledge-action):**
  - These papers compare activation-space geometry with output-distribution geometry: Manifold Steering (2605.05115), Shape of Beliefs (2602.02315), Info-geometry of softmax (2602.15293), FishBack (2605.17231), Info geometry shared/learned (2609.11063) and Riemannian-Manifold Steering (2605.24942).
  - The spaces they compare are activation manifold M_h vs. behavior manifold M_y (output probabilities), or Euclidean vs. Fisher-Rao/Bregman metrics.
  - **The relation they claim is preserved** is that a path along M_h maps to a path along M_y. A secondary summary calls this an "approximate isometry", which is unverified.
  - **None of these tests a case where the model acts against CRID.** Their concept domains are days, months, ages, distribution parameters and graphs.
- **Papers that test geometry in act-against-CRID cases:**
  - **Perfect Detection, Failed Control (2606.24952).** The direction that detects fake entities sits at cos ≈ 0.12 (about 83°) from the refusal-producing direction. The cosine stays in 0.12 to 0.20 across 1B to 9B models. A 15° rotation partly bridges the gap.
  - **Model-Adaptive Tool Necessity (2605.14038).** "Necessity" and "tool-call" directions are both decodable but become nearly orthogonal at the late-layer, last-token position where the action is chosen.
  - **KAPPA, Bridging the Knowledge-Prediction Gap (2509.23782).** The paper finds distinct knowledge and prediction subspaces. A fix that aligns them reduces the gap.
  - **Decodable but Not Corrected (2605.05715).** The failure direction overlaps 85 to 88% with task-critical computation (entanglement), so linear steering has null effect.
  - **LLMs Know They're Wrong and Agree Anyway (2604.19117).** The same heads carry a "statement is wrong" signal under sycophancy. Opinion-agreement reuses those heads but writes into an orthogonal direction.
  - **Harmfulness and refusal, two papers.** 2507.11878 and 2607.14147 find that the harm direction is distinct from refusal and stays intact while refusal fails.
  - **Low-Rank Subspace Analysis (2606.14388).** This supplies a principal-angle overlap metric (7B to 70B), but not in an act-against setting.

**Q2. Does manifold or nonlinear steering repair behavior where linear steering fails?**
- **Partial yes, in general control tasks:**
  - Curveball (2603.09313) reports gains in high-distortion regimes.
  - Manifold Steering (2605.05115) reports natural outputs where linear steering goes off-manifold.
  - Shape of Beliefs (2602.02315) reports that geometry-aware steering preserves the belief family.
  - Non-linear Interventions (2605.14749) reports more precise refusal steering.
  - NL-ITI (2403.18680) reports TruthfulQA MC1 rising from about 36 to about 50 on LLaMA-2-7B versus ITI.
- **Closest to the core question:** MAGS (2605.21770). It is a trajectory-aware, correctness-manifold steering method for errors "despite possessing the underlying knowledge". It beats static steering on math, code and molecules, but I did not see whether knowledge decodability was shown per item.
- **No paper found that applies a manifold or nonlinear method to the specific cases where linear steering failed in an act-against-CRID paper.** That means the overthinking failures in 2605.05715 and the fake-entity refusal case in 2606.24952.
- **Counter-evidence:**
  - Confidence Manifold (2602.08159) finds that nonlinear classifiers add nothing over linear for correctness detection.
  - Semantic specificity (2608.29431) finds that an MLP probe matched the linear probe.
  - Spokes, not a Ring (NeurIPS 2026) finds that a "manifold" can be discrete states with a step-like rep-to-behavior map.
  - Steerable but Not Decodable (2604.02608) shows the reverse dissociation.

**Q3. Direction 3: white-box vs. behavior-only evidence on gradual vs. abrupt change.**
- **White-box, mostly gradual or accumulating:**
  - Crescendo RepE (2507.02956): representations grow more "benign" as turns accumulate.
  - Assistant Axis (2601.10387): persona drift along the axis.
  - Proxy Reward Internalization (2606.09711): a probe score rises before visible hacking, and ablating the direction reduces hacking.
  - Geometry of Divergence (2608.30650): curvature and variance-slope signals separate failing episodes before they finish.
- **White-box, abrupt or discrete:**
  - LessWrong "From Drift to Snap" (Llama-3.1-70B, not peer-reviewed): 20 of 21 violating dialogues showed sharp transitions, most often around turn 10.
  - Latent Policy States (2607.18532): discrete regimes are recovered from activation trajectories.
  - Belief-action gap (2610.17444): action preferences sharpen around a commitment point.
  - Beyond the Commitment Boundary (2606.13603): the answer stabilizes in a single step.
- **White-box, both:** STAR (2603.15684) reports steady drift away from refusal representations plus abrupt phase transitions triggered by role-conditioned context.
- **White-box, early-fixed:**
  - Doomed from the Start (2607.06503): failure is decodable from round 1.
  - Premature commitment (2606.22936): early hidden-state convergence.
  - Progress Is Not Prognosis (Qwen3-32B): a value axis dips and recovers with local errors, while a separate outcome direction exists.
- **Behavior-only:** World-Model Collapse (2606.31399, abrupt phase diagram), Drift No More (2510.07777, equilibria), SPINE (2609.09090), DriftBench KBV (2604.28031, restates constraints it violates), Time-to-Inconsistency (ICLR 2026) and Agents Rot (2609.01660).
- **Threshold-crossing point.** Belief Dynamics (2511.00617) gives a closed-form Bayesian account in which smoothly accumulating log-odds cross a predictable point N*. Graph ICL work shows a sharp behavioral transition while PCA structure is recovered gradually. Both show that a smooth latent change can look like an abrupt behavioral jump.
- **No paper found that measures, in the same agent trajectory, whether CRID about a constraint persists while the agent violates it.** DriftBench's restatement probe is behavioral only.

**Q4. Scale above 8B.**
- 2604.19117 (up to 72B)
- 2606.14388 (7B to 70B)
- 2606.22936 (Llama-3.1-70B)
- Progress Is Not Prognosis (Qwen3-32B)
- 2607.18532 (up to 32B)
- 2607.14147 (14B)
- 2602.08159 (up to 14B)
- 2603.05488 (R1 671B, GPT-OSS 120B)
- 2605.07984 (Gemma-3-27B)
- Drift to Snap post (70B)
- 2503.15299 (small Qwen3-32B check)
- 2607.07003 (Gemma-3-12B)
- Most core act-against-CRID papers are at 9B or below: 2606.24952, 2605.05715, 2603.18353 and 2604.13068.

---

## 1. QUERY LOG

Source codes:
- **AX** = arXiv export API. It returned HTTP 429 for every query, so all 8 failed.
- **OA** = OpenAlex API.
- **S2** = Semantic Scholar. It returned 429, so it was not used.
- **WS** = web search: general, or with an openreview.net or aclanthology.org domain filter.

Families: F1 = shared geometry; F2 = divergence geometry; F3 = trajectory drift vs. jump.

"New" counts papers not already on my list. Lookup queries for an already-found title are marked "lookup".

| # | Query | Source | Fam | Hits looked at | New relevant added |
|---|---|---|---|---|---|
| 1 | manifold steering language model activations behavior 2025 | WS | F1 | 9 | 6 (Manifold Steering 2605.05115; Shape of Beliefs; Why Steering Works; Riemannian-Manifold; Spherical; Spokes) |
| 2 | LLM knows answer but outputs wrong ... belief behavior direction angle | WS | F2 | 9 | 2 (Retraction 2505.16170; Know More Than They Show) |
| 3-10 | 8 queries (manifold steering; shared geometry; isometry; representational geometry; nonlinear steering; manifold activations; feature manifold; curved steering) | AX | F1 | 0 | FAILED (HTTP 429) |
| 11 | arxiv nonlinear steering fails linear steering 2026 | WS | F1 | 9 | 3 (Decodable-not-Corrected 2605.05715; Curveball; Braun thesis 2602.17881) |
| 12 | arxiv "feature manifold" ... steering curved | WS | F1 | 9 | 1 (Goodfire manifold-steering post) |
| 13 | arxiv isometry activation space and output distribution space | WS | F1 | 10 | 3 (Info-geometry shared 2609.11063; sampling-error geometry 2605.04899; Directions to Regions) |
| 14 | manifold steering language model | OA | F1 | 12 | 4 (MAGS 2605.21770; GEOSTEER; Signatures of Steerability; GeoSteer-CoT) |
| 15 | shared geometry representation behavior neural network | OA | F1 | 12 | 0 |
| 16 | representation manifold steering off-manifold linear steering | OA | F1 | 12 | 2 (Overthinking manifold steering NeurIPS 2025; CHaRS) |
| 17 | isometry representation space output space language model | OA | F1 | 12 | 0 |
| 18 | nonlinear steering large language model activations | OA | F1 | 12 | 2 (Mechanistic Indicators; Non-Identifiability) |
| 19 | feature manifolds language model curved representations | OA | F1 | 12 | 4 (Confidence Manifold; Ordinal geometry; Symmetry; DMET 2505.20340) |
| 20 | representational geometry concepts actions language model | OA | F1 | 12 | 1 (Abstract representational geometry 2606.23345) |
| 21 | geodesic steering activation space | OA | F1 | 12 | 5 (Hesitation/hyperbolic; Cond. Optimal Bridge; Riemannian diverse reasoning; Angle-Norm; PersonaManifold) |
| 22 | knowing doing gap language model internal representations | OA | F2 | 12 | 5 (Knowing→Abstaining VLM; Tool-necessity KDG; Belief-action gap 2610.17444; Knowing Isn't Always Saying; Knowing Before Answering) |
| 23 | truth direction behavior direction cosine similarity language model | OA | F2 | 12 | 4 (More-to-Refusal; Correctness Is a Direction; truth directions context-sensitive; Mahalanobis cosine) |
| 24-29 | 6 queries (hidden states encode correct but generate incorrect; probe decodable steering fails; subspace overlap; lying geometry; sycophancy overridden; misalignment geometry) | OA | F2 | 0 | FAILED (OpenAlex daily budget exhausted; retry also failed) |
| 30 | arxiv angle between truth direction and output steering direction ... | WS | F2 | 9 | 4 (Can LLMs Lie; Directions→Cones; Truthfulness Spectrum; Perfect Detection 2606.24952) |
| 31 | arxiv "recognize but fail" OR "know but don't tell" ... | WS | F2 | 9 | 2 (Inside-Out; 2510.09033); plus 2 known only from secondary write-ups (Causal Tongue-Tie; Senses Wide Shut) |
| 32 | arxiv probe decodable but linear steering does not correct | WS | F2 | 9 | 3 (Steerable-not-Decodable; Static Unit-Scale Bias; Encoded-not-Decoded) |
| 33 | arxiv sycophancy internally represents correct answer overridden | WS | F2 | 9 | 4 (When Truth Is Overridden; Sycophantic Anchors; Know They're Wrong 2604.19117; authority-bias 2607.00415) |
| 34 | arxiv subspace overlap knowledge representation output behavior | WS | F2 | 9 | 2 (KAPPA 2509.23782; Low-Rank Subspace 2606.14388) |
| 35 | "The Geometry of Knowing vs. Steering" | WS | F2 | 9 | 0 (lookup) |
| 36 | arxiv curvature of activation trajectories ... | WS | F2 | 9 | 4 (Curved Inference; Trajectory Geometry across layers; reasoning-critical neurons; Reasoning models move differently) |
| 37 | arxiv representation-action gap ... causal intervention 2026 | WS | F2 | 10 | 2 (Interpretability w/o Actionability 2603.18353; Where's the Plan 2605.07984) |
| 38 | arxiv deception lying internal truth preserved ... | WS | F2 | 9 | 2 (Truthful reps flip; Limits of lie detector) |
| 39 | "LLMs Know They're Wrong and Agree Anyway" | WS | F2 | 9 | 0 (lookup) |
| 40 | Billa 2026 three regimes ... | WS | F2 | 9 | 4 (LAP 2604.15557; Read-Best≠Steer-Best; Why Linear Interp. Works 2602.09783; Belief-state geometry ICL 2609.17376) |
| 41 | arxiv manifold or nonlinear steering truthfulness ... ITI | WS | F1 | 10 | 3 (NL-ITI; Local Linearity 2604.19018; Tethered Reasoning) |
| 42 | arxiv belief-action gap LM agents ... | WS | F2/F3 | 9 | 3 (Broken Links 2605.00226; Knowing-What-You-Know 2511.13240; Belief Inertia ACL 2026) |
| 43 | arxiv "Confidence Manifold" ... | WS | F2 | 10 | 1 (Confidence Geometry trace-level 2605.16824) |
| 44 | arxiv multi-turn activations gradual drift vs abrupt ... honest to dishonest | WS | F3 | 9 | 4 (NeuroFilter; Time-to-Inconsistency; Drift No More; Drift-to-Snap post) |
| 45 | arxiv LLM agent "change point" hidden states ... | WS | F3 | 9 | 3 (Real-time detect/repair 2608.02464; Premature commitment 2606.22936; Doomed from Start 2607.06503) |
| 46 | arxiv "latent policy states" OR "tipping point" ... | WS | F3 | 9 | 4 (Latent Policy States 2607.18532; Goal-directedness 2602.08964; Progress≠Prognosis; State-drift preprint) |
| 47 | arxiv "recognize but fail to utilize" OR "context rot" internal | WS | F3 | 9 | 2 (Know-but-Don't-Tell 2406.14673; Context rot search 2606.29718) |
| 48 | arxiv agent restates constraint then violates it ... probe | WS | F3 | 9 | 3 (DriftBench 2604.28031; IF-dimension 2410.14516; Illusory Completion) |
| 49 | arxiv persona drift instruction drift multi-turn activations | WS | F3 | 9 | 3 (Assistant Axis; Persona drift 2402.10962; Prompting vs persona drift) |
| 50 | arxiv multi-turn jailbreak refusal direction erodes crescendo | WS | F3 | 9 | 2 (RepE Crescendo 2507.02956; STAR 2603.15684) |
| 51 | arxiv goal drift LLM agents long horizon probe | WS | F3 | 9 | 2 (Conformal temporal concepts 2604.19775; Asymmetric goal drift) |
| 52 | arxiv geometric signals temporal curvature variance slope ... | WS | F3 | 9 | 4 (Geometry of Divergence 2608.30650; Attractor States; PsychoPass; Phase-space reasoning 2410.04415) |
| 53 | arxiv sycophancy across turns hidden state probe ... | WS | F3 | 9 | 5 (Chain Holds/Answer Folds; Answer instability 2511.10688; Sycophancy hides linearly; Dissociating sycophancy 2607.07003; Gotta Catch) |
| 54 | arxiv CoT hidden states abrupt phase transition error onset | WS | F3 | 9 | 5 (Hidden Error Awareness; Commitment Boundary; Backtracking; Temporal Predictors; Statistical Physics) |
| 55 | "From Drift to Snap" instruction violation ... | WS | F3 | 9 | 1 (Energy-driven latent conflict 2609.08646) |
| 56 | openreview 2026 knowledge action gap ... steering fails | WS (openreview filter) | F2 | 10 | 3 (Polished but Unresolved 2609.00823; Predicting future behaviors 2606.11172; Cylindrical) |
| 57 | aclanthology 2026 encodes correct answer ... outputs wrong | WS (ACL filter) | F2 | 9 | 1 (Right Makes Might 2606.03234) |
| 58 | "Read-Best Is Not Steer-Best" | WS | F2 | 9 | 2 peripheral (speech probe-shift 2609.18533; Stories in Space 2605.12412) |
| 59 | Park 2026 dually flat geometry softmax | WS | F1 | 9 | 1 (Info-Geometry of Softmax 2602.15293) |
| 60 | "Spokes, not a Ring" | WS | F1 | 9 | 0 (lookup) |
| 61 | representational similarity activations vs output behavior RSA | WS | F1 | 9 | 1 (Causality≠Invariance 2602.22424) |
| 62 | overthinking the truth ... false demonstrations | WS | F2 | 9 | 1 (Halawi 2307.09476) |
| 63 | "Polished but Unresolved" | WS | F3 | 9 | 0 (lookup) |
| 64 | "Cylindrical Representation Hypothesis" | WS | F1 | 9 | 0 (lookup) |
| 65 | knowledge conflict context overrides parametric ... still encodes | WS | F2 | 9 | 4 (JuICE; SpARE; residual stream KC 2410.16090; 2506.06485) |
| 66 | reward hacking agent internally represents task intent ... | WS | F2/F3 | 9 | 4 (Proxy Reward Internalization; Mitigating RH via task reps; Monitoring RH internal; 2603.04069) |
| 67 | "Attractor States Emerge" | WS | F3 | 9 | 0 (lookup) |
| 68 | "Progress Is Not Prognosis" | WS | F3 | 9 | 0 (lookup) |
| 69 | Inside-Out hidden factual knowledge | WS | F2 | 10 | 0 (lookup) |
| 70 | steering vector reliability geometry separability predicts steerability | WS | F1 | 9 | 1 (Braun ICLRW 2505.22637) |
| 71 | scheming deception probe agentic trajectories per-step ... | WS | F3 | 9 | 2 (DeceptGuard; Apollo deception probes 2504.04072) |
| 72 | intervention direction differs from probe direction ITI ... | WS | F2 | 9 | 2 (ITI 2306.03341; Detection w/o Correction 2604.13068) |
| 73 | honesty steering lying direction ... layer-wise | WS | F2 | 9 | 1 (Depth-wise honesty steering 2512.07667) |
| 74 | instruction-following failure constraint encoded ... steering | WS | F2 | 9 | 3 (Semantic Gravity Wells; IF steering 2410.12877; How LLMs Follow Instructions) |
| 75 | context length degradation internal retrieval succeeds generation fails | WS | F3 | 9 | 0 |
| 76 | linear steering off-manifold ... geometry-aware | WS | F1 | 9 | 0 |
| 77 | hallucination probe detects but steering fails ... | WS | F2 | 9 | 2 (Readable not Controllable 2607.00158; Single Direction of Truth 2507.23221) |
| 78 | multi-turn representation drift hidden states sudden shift | WS | F3 | 9 | 1 (DeepContext) |
| 79 | ICL abrupt transition belief update ... hidden states | WS | F3 | 9 | 3 (Belief Dynamics 2511.00617; Belief or Circuitry 2605.08405; Park ICL-of-representations) |
| 80 | angle between detection direction and intervention direction cosine ... | WS | F2 | 10 | 0 (radar results only) |
| 81 | behavioral drift LLM agents long-horizon activations white-box | WS | F3 | 10 | 3 (TACT 2605.05980; Agent Drift 2601.04170; Agents Rot 2609.01660) |
| 82 | kernel nonlinear steering ... outperform difference of means | WS | F1 | 9 | 0 |
| 83 | agent dishonesty emerges over trajectory probe per step ... | WS | F3 | 9 | 3 (LH-Deception; Secret Agenda; Obfuscation Atlas) |
| 84 | activations deviate from correctness manifold at error ... | WS | F2 | 9 | 4 (REMA; Reject Wrong Answers 2603.13259; CLUE ACL 2026; Geometric taxonomy hallucination) |
| 85 | trajectory signatures hidden states LLM agent failure white-box | WS | F3 | 9 | 2 (TRACES; Trading-agent signatures) |
| 86 | behavior manifold output distributions activation manifold ... | WS | F1 | 10 | 0 |
| 87 | sycophancy turn-by-turn probe gradual erosion vs sudden flip | WS | F3 | 10 | 2 (SPINE 2609.09090; SYCON-Bench) |
| 88 | isometric activation geometry next-token Fisher-Rao steering | WS | F1 | 9 | 2 (FishBack 2605.17231; Sharp Directions NeurIPS 2026) |
| 89 | knowing doing gap LLM internal probe ... agents | WS | F2 | 9 | 2 (LLMs are Greedy Agents; STOCKTAKE) |
| 90 | constraint violation long-horizon agent phase transition abrupt vs gradual probe | WS | F3 | 9 | 2 (World-Model Collapse 2606.31399; World Model Science 2609.17419) |
| 91 | nonlinearly encoded concept MLP probe succeeds linear steering fails | WS | F1/F2 | 9 | 2 (Non-linear Interventions 2605.14749; Semantic specificity 2608.29431) |
| 92 | "Non-linear Interventions on LLMs" refusal | WS | F1 | 9 | 1 (Refusal nonlinear perspective 2502.17420) |
| 93 | steering fails where model acts against encoded knowledge manifold steering repairs | WS | F2 | 9 | 2 (Steering off Course 2504.04635; Safety pitfalls 2603.24543) |
| 94 | refusal representation preserved while model complies jailbreak ... | WS | F2 | 9 | 3 (Harmfulness≠Refusal 2507.11878; Breaking Refusal 2607.14147; LOCA 2605.00123) |
| 95 | reasoning model CoT correct internally then switches wrong ... | WS | F2/F3 | 9 | 3 (Reasoning Theater; Post-hoc reasoning 2603.01437; Chain-of-Probe) |
| 96-99 | 4 queries (KDG geometry orthogonal; manifold steering honesty; agent step-level honesty transition; truth geometry under sycophancy) | WS | F1/F2/F3 | 0 | NOT RUN (web-search budget exhausted) |

**Stopping-rule status:** the longest run of zero-new queries was 2 in a row (rows 15 and 17 are not consecutive; rows 75 and 76 are). The last 10 executed queries (86 to 95) added 0, 2, 2, 2, 2, 2, 1, 2, 3 and 3 papers. **The rule was not satisfied.**

---

## 2. PAPER TABLE

Column notes:
- "csv" means a title-only case-insensitive match in papers_master.csv. I read the title column only, never the rating or takeaway columns.
- **C/R** = causal (an intervention changes behavior) vs. correlational.
- **Tier:** Core = tests the core question on internals, or tests a fix; Supporting; Peripheral.
- Models are listed "as stated in abstract or snippet". "n.s." = not stated in what I saw.

### 2a. Direction 2: act-against-CRID and divergence geometry

| Title | ID / venue | Date | Models / scale | Setting | C/R | Spaces / relation compared | Tier | csv | Why |
|---|---|---|---|---|---|---|---|---|---|
| Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in LMs | arXiv 2606.24952 | 2026-06 | Gemma-2-2B-it main; 4 models, 3 families, 1B-9B | Fake-entity hallucination vs refusal; JSON format control | Causal | Angle between best-detecting and best-causing directions; cos≈0.12 (≈83°) for hallucination→refusal, ≈1 for format; cosine unchanged by instruction tuning; 15° rotation gives 73%/60% refusal at 1.8% FP | Core | no | Directly measures detection-vs-control geometry in an act-against-CRID case |
| Decodable but Not Corrected by Fixed Residual-Stream Linear Steering: Medical LLM Failure Regimes | arXiv 2605.05715 | 2026-05 | n.s. main; replicated on Qwen2.5-7B | Medical QA "overthinking" (correct under resampling, wrong in long CoT); MMLU-STEM | Causal (null) | Failure direction 85-88% overlap with task-critical computation; 5 linear-steering families (29 configs) give Δ≈0; LEACE hurts accuracy; LAP regime "nonlinearly encoded" (probe 81.5% vs logit-lens 58.6%); probe still supports abstention (AUROC 0.61) | Core | no | Decodable, but linear steering cannot fix it; entanglement explanation; points to nonlinear regime without testing a nonlinear fix |
| Interpretability without actionability: mechanistic methods cannot correct LM errors despite near-perfect internal representations | arXiv 2603.18353 | 2026-03 | Qwen2.5-7B-Instruct; Steerling-8B | 400 clinical triage vignettes | Causal (mostly null) | Probe 98.2% AUROC vs output sensitivity 45.1%; concept-bottleneck, SAE (zero effect) and TSV steering (fixes 24% of misses) | Core | no | Quantifies a large act-against-CRID gap; several fixes fail |
| Detection Without Correction: A Robust Asymmetry in Activation-Based Hallucination Probing | arXiv 2604.13068 | 2026-03 | 7 models, 117M-7B (GPT-2, Pythia, Qwen2.5) | Hallucination | Causal (null) | Probe-derived direction steering fails 7/7; output confidence beats probes above 410M | Core | no | Probe direction ≠ control direction |
| Readable but Not Controllable: Neuron-Level Evidence for Medical LLM Hallucination | arXiv 2607.00158 | 2026-06 | 4 open models (n.s.) | Medical QA | Causal (null) | Distributed, redundant signal (random neuron subsets suffice); neuron steering does not reliably correct | Core | yes | Same detection-control asymmetry |
| Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use | arXiv 2605.14038 | 2026-05 | 4 models (n.s.) | Arithmetic and factual QA tool calls | Correlational (probes) | Necessity and tool-call signals both decodable but **near-orthogonal at late-layer last token**; mismatch concentrated in the cognition→action step | Core | yes | Geometric locus of the knowing-doing gap |
| Bridging the Knowledge-Prediction Gap in LLMs on MCQs (KAPPA) | arXiv 2509.23782; ICML 2026 | 2025-09 | n.s. | MCQ | Causal (fix) | Distinct knowledge vs prediction subspaces in residual stream; aligning them reduces the gap | Core | yes | Geometry plus a working fix for act-against-CRID |
| LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit | arXiv 2604.19117; ICML 2026 | 2026-04 | 12 open-weight models, 1.5B-72B; ablation on Gemma-2-2B | Sycophancy, factual and instructed lying | Causal (head ablation, path patching) | Same heads carry a "wrong" signal; ablation moves sycophancy 28%→81% with accuracy flat; opinion-agreement writes an orthogonal direction | Core | no | CRID preserved, behavior diverges via a deference circuit; >8B |
| When Truth Is Overridden: Internal Origins of Sycophancy | arXiv 2508.02087 | 2025-08 | multiple families (n.s.) | Opinion-induced sycophancy | Causal (patching) | Two-stage: late-layer output-preference shift, then deeper representational divergence | Core | no | Layer-wise account of where behavior departs from fact representation |
| LLMs Encode Harmfulness and Refusal Separately | arXiv 2507.11878; NeurIPS 2025 | 2025-07 | n.s. | Jailbreaks | Causal (steering) | Harmfulness direction ≠ refusal direction; jailbreaks lower refusal without reversing harm belief | Core | yes | Decodable harm info persists while the model acts against it |
| Breaking Refusal in the First Half: Mechanistic Study of the Prefill Jailbreak | arXiv 2607.14147 | 2026-07 | 4 models, 3 families, 1.5-3.8B, plus 14B | Prefill jailbreak | Causal | Harm probe 0.91-0.98 on flipped prompts while refusal is at chance; restoring the harm direction in the early-response window partly re-engages refusal; injecting the refuse-state reverses 74% | Core | no | A fix in a time-localized window; up to 14B |
| Manifold-Guided Attention Steering (MAGS) | arXiv 2605.21770; ICML 2026 | 2026-05 | n.s. | Math, code, molecule reasoning | Causal (fix) | Head outputs leave a low-dimensional correctness manifold at the error point, and the deviation compounds; trajectory-aware manifold steering beats static steering | Core | no | Only manifold-type fix aimed at "errors despite knowledge"; per-item CRID not verified from abstract |
| Overthinking the Truth: How LMs Process False Demonstrations | arXiv 2307.09476; ICLR 2024 | 2023-07 | n.s. (several open LMs) | Few-shot with false labels | Causal (head ablation) | Correct answer decodable at intermediate layers, then overridden after a "critical layer" by late false-induction heads | Core (foundational) | no | Classic CRID-then-overridden finding |
| Hidden Error Awareness in CoT: The Signal Is Diagnostic, Not Causal | arXiv 2605.09502 | 2026-05 | n.s. | CoT reasoning | Causal test (negative) | Probe 0.95 AUROC (0.79 at step 1) while verbalized confidence stays high; signal not causal | Core | no | Tests whether the decodable error signal drives action; says no |
| Inference-Time Intervention (ITI) | arXiv 2306.03341; NeurIPS 2023 | 2023-06 | LLaMA-7B | TruthfulQA | Causal | About 40% probe-vs-generation gap; mass-mean-shift direction steers better than the probe weight direction | Supporting (foundational) | yes | Early evidence that the probe direction ≠ the effective intervention direction |
| NL-ITI: Non-Linear Inference Time Intervention | arXiv 2403.18680 | 2024-03 | LLaMA-2-7B (+others) | TruthfulQA | Causal | Nonlinear probe plus multi-token intervention; MC1 about 36→50 vs ITI | Supporting | no | Nonlinear beats linear on a truthfulness intervention |
| Predicting Where Steering Vectors Succeed (LAP) | arXiv 2604.15557 | 2026-04 | Pythia-2.8B to Llama-8B | 24 binary concept families | Causal validation | Logit-lens A_lin predicts steering success (ρ 0.86-0.91); three regimes: output-aligned / nonlinearly encoded / not extractable | Supporting | no | Diagnostic for when linear vs nonlinear steering should work |
| Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens | arXiv 2604.02608; ICML 2026 | 2026-04 | 6 models, 3 families | ICL function vectors | Causal | Steering succeeds where the logit lens fails; decodable-not-steerable is rare (3/72); MLP probe closes 5/10 | Supporting | yes | Reverse dissociation; control for "decodable ⇒ steerable" |
| Read-Best Is Not Steer-Best | arXiv 2609.22135 | 2026-08 | 3 omni-modal models | Emotion | Causal | Probe-best layers scatter (0.29-1.0 normalized depth); steer-best band 0.48-0.74; 26× gap vs random | Supporting | yes | Read and write locations differ |
| Encoded but Not Decoded: Three-Level Gap in LLM Syntax | arXiv 2609.29848 | 2026-09 | n.s. | Trilingual syntax | Causal (null) | Behavior ≤ LM-head readout ≤ probe recoverability; linear additive steering not sufficient | Supporting | yes | Three-level CRID-to-behavior framework |
| The Confidence Manifold: Geometric Structure of Correctness Representations | arXiv 2602.08159; ICLR 2026 | 2026-02 | 11 models, 124M-14B | TruthfulQA-style correctness | Causal (steering, erasure, DAS) | Signal sits in a 2-8 dim subspace; class separation is a mean shift; **nonlinear adds nothing**; internal AUC 0.80-0.97 vs output 0.44-0.64; steering shifts hallucination by about 9 pts | Supporting | no | Shows correctness geometry is simple, which counts against needing a manifold for detection |
| Correctness Is a Direction: Geometric Answer Selection | arXiv 2610.04512 | 2026-10 | 5 models, 1B-8B | ARC, MMLU, TruthfulQA | Correlational (readout) | Mean incorrect→correct displacement at about 70% depth beats log-prob scoring by up to +32 to +52 pts | Supporting | no | Large CRID-vs-output gap exploitable by readout |
| How Transformers Reject Wrong Answers: Rotational Dynamics | arXiv 2603.13259 | 2026-02 | models >1.6B (n.s.) | Forced correct vs incorrect continuations | Correlational (explicit) | Displacements rotate apart through mid-depth, then late asymmetric commitment | Supporting | no | Angular geometry of correct vs incorrect paths |
| REMA: Unified Reasoning Manifold Framework | arXiv 2509.22518 | 2025-09 | n.s. | Reasoning errors | Correlational | kNN deviation of error representations from the "correct reasoning manifold", localized by layer | Supporting | no | Manifold-deviation metric for errors |
| Can LLMs Lie? Investigation beyond Hallucination | arXiv 2509.03518 | 2025-09 | n.s. | Instructed lying | Causal (steering) | Lie vs truth separate mid-layer; honesty steering vector not aligned with the truth centroid; later layers drift back toward truth | Supporting | no | Steering direction ≠ truth-cluster direction |
| From Directions to Cones: Multidimensional Representations of Propositional Facts | arXiv 2505.21800 | 2025-05 | n.s. | True/false facts | Causal | Multiple low-cosine orthogonal directions all steer truth | Supporting | no | Truth is a cone, not a single direction |
| The Truthfulness Spectrum Hypothesis | arXiv 2602.20273 | 2026-02 | n.s. | 5 truth types, sycophantic lying | Correlational | Post-training pushes sycophantic-lying directions further from other truth types | Supporting | no | Geometry of truth vs lying changes with training |
| When Truthful Representations Flip Under Deceptive Instructions? | arXiv 2507.22149 | 2025-07 | Llama-3.1-8B, Gemma-2-9B | Instructed deception | Correlational (+SAE) | Probes read the *instructed* answer; shifts in early-to-mid layers | Supporting | yes | Caveat: truth representation may itself flip |
| Probing the Limits of the Lie Detector Approach | arXiv 2603.10003 | 2026-02 | 3 open models | Non-lying deception | Correlational | Truth probes miss technically-true deception | Peripheral | no | Probe-validity caveat |
| Depth-Wise Activation Steering for Honest LMs | arXiv 2512.07667 | 2025-12 | 7 models | MASK honesty | Causal (fix) | Gaussian depth schedule beats single-layer steering in 6/7 | Supporting | no | Fix for false assertions despite CRID |
| A Single Direction of Truth: Observer probe steers contextual hallucinations | arXiv 2507.23221 | 2025-07 | Gemma-2 | Contextual hallucination | Causal | Linear direction does steer (counterpoint) | Supporting | no | Counterexample to detection≠control |
| LLMs Know More Than They Show | arXiv 2410.02707; ICLR 2025 | 2024-10 | n.s. | QA | Correlational | Encodes the correct answer yet generates an incorrect one; truth signal concentrated in answer tokens | Supporting | yes | Base CRID-vs-output phenomenon |
| Inside-Out: Hidden Factual Knowledge in LLMs | arXiv 2503.15299; COLM 2025 | 2025-03 | Llama-3-8B, Mistral-7B, Gemma-2-9B; Qwen3-32B check | Closed-book QA | Correlational | Internal vs external knowledge gap about 40%; some answers never sampled in 1000 tries | Supporting | yes | Quantifies hidden knowledge |
| Do LLMs Really Know What They Don't Know? | arXiv 2510.09033 | 2025-10 | n.s. | Hallucination | Correlational | Internal states reflect recall, not truthfulness; subspace overlap with MLP weights | Supporting (control) | yes | Caveat for reading CRID |
| When Do LLMs Admit Their Mistakes? Role of Model Belief in Retraction | arXiv 2505.16170 | 2025-05 | n.s. | Retraction | Causal (belief steering) | Belief probes track behavior, not correctness; steering the belief direction controls retraction | Supporting | no | "Belief" direction drives action even when wrong |
| Low-Rank Subspace Analysis of LLM Interventions | arXiv 2606.14388; ICML 2026 | 2026-06 | 7B-70B | Refusal, sycophancy, jailbreak | Causal | Behaviors as low-rank subspaces; principal-angle overlap predicts side effects | Supporting | no | Ready-made subspace-overlap metric; >8B |
| Comparing Linear Probes with Mahalanobis Cosine Similarity | arXiv 2606.19603 | 2026-06 | n.s. | Method | n/a | Mahalanobis cosine between directions | Supporting (method) | no | Better angle metric for probe vs steer directions |
| There Is More to Refusal than a Single Direction | arXiv 2602.02132 | 2026-02 | n.s. | Refusal | Causal | Multiple refusal directions give the same trade-off | Peripheral | no | Refusal geometry |
| Truth-value judgment: 'truth directions' are context sensitive | arXiv 2404.18865 | 2024-04 | n.s. | Truth probes | Correlational | Context sensitivity of truth directions | Peripheral | no | Probe-stability caveat |
| Belief-action gap in LM agents | arXiv 2610.17444 | 2026-10 | n.s. | Text grid worlds | Mostly correlational; activation edits give limited control | Probes for state beliefs, IRL action values, action preferences; info represented more reliably than shown in behavior; preferences sharpen near a commitment point | Core (D2 + D3) | no | Agentic act-against-CRID with a time course |
| Why Do LLMs Struggle in Strategic Play? Broken Links Between Observations, Beliefs, and Actions | arXiv 2605.00226 | 2026-05 | n.s. | Strategic games | Probing (n.s.) | Belief encoded at decision point, not converted to action | Supporting | no | Agentic belief-action gap |
| Knowing What You Know Is Not Enough (confidence vs action) | arXiv 2511.13240 | 2025-11 | n.s. | Betting, tool use | Behavioral | Acts against elicited confidence | Peripheral | yes | Behavioral analogue |
| LLMs are Greedy Agents (knowing-doing gap) | arXiv 2504.16078 | 2025-04 | n.s. | Bandits | Behavioral + RL fix | Knowing-doing gap narrowed by RL fine-tuning | Peripheral | yes | Origin of the KDG term |
| STOCKTAKE: perception vs action with a fair oracle | arXiv 2607.13618 | 2026-07 | n.s. | Multi-week decisions | Behavioral | Separates perception from action failures | Peripheral | yes | Benchmark control |
| Knowing Isn't Always Saying: Spatial Encodings in VLMs | arXiv 2608.22916 | 2026-08 | VLMs | Spatial QA | Probing | Encoded spatial info fails to reach answers | Peripheral | no | Multimodal analogue |
| From Knowing to Abstaining: Representation-Action Gap in VLMs | arXiv 2609.32653 | 2026-09 | VLMs | Abstention | n.s. | Representation-action gap | Peripheral | no | Multimodal analogue |
| Where's the Plan? Locating Latent Planning | arXiv 2605.07984 | 2026-05 | incl. Gemma-3-27B | Rhyme planning | Causal (patching) | Only Gemma-3-27B causally uses the decodable plan | Supporting | yes | Decodable ≠ used; scale-dependent |
| Semantic Gravity Wells: Why Negative Constraints Backfire | arXiv 2601.08070 | 2026-01 | n.s. | "Don't use word X" | Causal (patching L23-27) | Suppression signal present but weaker in failures; the instruction primes the forbidden token | Supporting | no | Constraint represented yet violated, single-turn |
| Do LLMs "know" internally when they follow instructions? | arXiv 2410.14516 | 2024-10 | n.s. | Instruction following | Causal | Instruction-following dimension; shifting along it improves success | Supporting | yes | Fix for constraint violation |
| Taming Knowledge Conflicts (JuICE) / Analysing Residual Stream under Knowledge Conflicts / SpARE | arXiv 2503.10996 / 2410.16090 / 2410.15999 | 2024-25 | n.s. | Context vs parametric | Causal / correlational | Superposition of contextual and parametric roles; conflict decodable mid-layer before the answer | Supporting | no / no / no | Parametric CRID persists when context overrides |
| Proxy Reward Internalization and Mechanistic Exploitation | arXiv 2606.09711 | 2026-06 | n.s. (coding agent) | Reward hacking | Causal (ablation) | Probe score for "proxy accepts vs truly correct" rises before visible hacking; ablating reduces hacking | Core (D2 + D3) | yes | Correctness info and hacking action over time |
| Monitoring and Discovering Reward Hacking with Internal Representations / Mitigating RH via Task Representations / Monitoring Emergent RH via activations | arXiv 2609.19101 / ICML+NeurIPS 2026 / 2603.04069 | 2026 | n.s. | Reward hacking | Corr. / causal (steering) | Shared "cheating" direction; prompt-space steering beats response-space | Supporting | yes / yes / yes | Methods for act-against-intent |
| The Secret Agenda: LLMs Strategically Lie | arXiv 2509.20393 | 2025-09 | 38 models | Strategic lying | Causal (null) | Labeled SAE deception features silent; steering 100+ features fails; unlabeled activations separate | Supporting | no | Linear feature steering fails on strategic lying |
| Right Makes Might: Aligning Verified Hidden States | arXiv 2606.03234 | 2026-06 | n.s. | RL math | Causal (training) | Hidden-state alignment beats logit alignment for "knowing but not doing" | Supporting | no | Training-time fix |
| Reasoning Theater: Disentangling Model Beliefs from CoT | arXiv 2603.05488 | 2026-03 | DeepSeek-R1 671B, GPT-OSS 120B | MMLU, harder tasks | Correlational | Final answer decodable far earlier than a CoT monitor sees it | Supporting | no | >8B; latent belief ahead of text |
| Post-Hoc Reasoning in CoT: Decoding and Steering Pre-Committed Answers | arXiv 2603.01437 | 2026-03 | n.s. | CoT | Causal | Probe-direction steering flips >50% of answers | Supporting | no | Pre-committed answer is causal |
| Minimal, Local, Causal Explanations for Jailbreak Success (LOCA) | arXiv 2605.00123 | 2026-04 | n.s. | Jailbreaks | Causal | Per-instance small residual changes restore refusal | Supporting | no | Per-instance fix geometry |
| Causality ≠ Invariance (concept vectors via RSA) | arXiv 2602.22424; ICLR 2026 | 2026-02 | n.s. | ICL concepts | Causal | Representations invariant across formats differ from those that drive ICL | Peripheral | no | Concept rep vs behavior-driving rep |
| Senses Wide Shut: Representation-Action Gap in Omnimodal LLMs; Causal Tongue-Tie | via secondary write-ups only | 2026 | n.s. | Omnimodal; causal QA | n.s. | Probe 97% vs Yes/No at random (Tongue-Tie, secondary) | Peripheral (unverified) | yes / no | Seen only in secondary sources |

### 2b. Direction 2 / family 1: shared geometry and manifold or nonlinear steering (methods; no act-against-CRID test)

| Title | ID / venue | Date | Models / scale | Setting | C/R | Spaces / relation compared | Tier | csv | Why |
|---|---|---|---|---|---|---|---|---|---|
| Manifold Steering Reveals the Shared Geometry of NN Representation and Behavior | arXiv 2605.05115; ICML 2026 | 2026-05 | Llama-3.1-8B (snippet); video world model | Cyclic and sequential concepts (days, months, letters, ages); graph ICL | Causal | Activation manifold M_h vs behavior manifold M_y (output distributions); manifold steering traces M_y, linear goes off-manifold; reverse optimization recovers M_h curvature ("approx. isometry" only in a secondary summary) | Supporting | no | Main "shared geometry" paper; not applied to knowledge-action |
| The Shape of Beliefs: Representation Manifolds of LMs' Posteriors | arXiv 2602.02315 | 2026-02 | Llama-3.2 (size n.s.) | ICL inference of Gaussian parameters | Causal | Curved belief manifolds; linear steering off-manifold with coupled changes; linear field probing preserves the belief family | Supporting | no | Belief geometry plus geometry-aware steering |
| Riemannian-Manifold Steering: Geometry-Aware Generative Autoencoders | arXiv 2605.24942 | 2026-05 | n.s. | Label-free steering | Causal | Steering as geodesics under a metric pulled back from output Hellinger distance | Supporting | no | Output-metric-induced activation geometry |
| Curveball Steering: The Right Direction To Steer Isn't Always Linear | arXiv 2603.09313; ICML 2026 | 2026-03 | n.s. | Behavior steering | Causal | Geodesic/Euclidean distortion ratio; kernel-PCA steering beats linear in high-distortion regimes | Supporting | no | Nonlinear fix where linear fails |
| Non-linear Interventions on LLMs | arXiv 2605.14749; ICML 2026 | 2026-05 | n.s. | Refusal bypass | Causal | Intervention on non-linear features; more precise than linear | Supporting | no | Nonlinear intervention framework |
| The Cylindrical Representation Hypothesis for LM Steering | arXiv 2605.01844; ICML 2026 | 2026-05 | multiple; 100 concepts | Steering variance | Causal | Axis, normal plane and sensitive sectors explain per-sample steering variance | Supporting | no | Explains why one direction steers unevenly |
| The Information Geometry of Softmax: Probing and Steering | arXiv 2602.15293 | 2026-02 | n.s. | Theory + steering | Causal | Bregman (dually flat) geometry; dual steering | Supporting | no | Principled representation-to-output geometry |
| FishBack: Pullback Fisher Geometry for Optimal Activation Steering | arXiv 2605.17231 | 2026-05 | n.s. | Steering | Causal | Fisher metric of the softmax head governs perturbation→output change; Euclidean assumption wrong | Supporting | no | Output-induced metric |
| The information geometry of LLMs is shared, learned, and controllable | arXiv 2609.11063 | 2026-09 | Transformer, SSM, RNN families | Next-token geometry | Mixed | Fisher-Rao output geometry agrees across models more than activation geometry; activation path length tracks behavioral change (snippet: 99 cells) | Supporting | no | Cross-model output-geometry invariance |
| Spokes, not a Ring: Testing Continuity of Representation Manifolds | NeurIPS 2026 (no arXiv found) | 2026 | n.s. | Weekday representations | Causal | Seven discrete states; non-uniform, step-like steering sensitivity | Supporting (control) | no | Caveat: test continuity before manifold steering |
| Understanding (Un)Reliability of Steering Vectors; Geometric Predictors and Limits of Linear Approximations | arXiv 2505.22637 (ICLRW 2025); 2602.17881 (thesis) | 2025-26 | n.s. | Many behaviors | Causal | Directional agreement and separability predict steerability; off-manifold hypothesis | Supporting | no / no | Predictors of linear-steering failure |
| Signatures of Steerability in Activation Space | arXiv 2609.14151 (NeurIPS 2026) | 2026-09 | n.s. | Steering | Corr. | Separation metrics correlate with steerability | Supporting | no | Same |
| CHaRS: Concept Heterogeneity-aware Representation Steering | arXiv 2603.02237 | 2026-02 | n.s. | Steering | Causal | OT between concept clusters vs DiM global translation | Supporting | no | Non-global steering |
| Spherical Steering; Angle-Norm Decomposition; GEOSTEER; Conditional Optimal Bridge; Riemannian diverse reasoning; Sharp Directions Are Against You | 2602.08169; 2606.06735; 2609.10658; 2607.10517; 2511.08305; NeurIPS 2026 | 2025-26 | n.s. | Steering methods | Causal | Rotation, geodesic, Riemannian or curvature variants of steering | Peripheral | no (all) | Method variants |
| Local Linearity of LLMs Enables Activation Steering via Linear Optimal Control | arXiv 2604.19018 | 2026-04 | Gemma-2-2B etc. | Truthfulness | Causal | Locally linear control (A-LQR) | Peripheral | no | Linear counterpoint |
| From Directions to Regions: Decomposing Activations via Local Geometry | arXiv 2602.02464; ICML 2026 | 2026-02 | n.s. | Decomposition | n.s. | Regions vs global directions | Peripheral | no | Non-linear decomposition |
| Mechanistic Indicators of Steering Effectiveness; Non-Identifiability of Steering Vectors; Steering off Course; Safety Pitfalls of Steering Vectors | 2602.01716; 2602.06801; 2504.04635; 2603.24543 | 2025-26 | various | Steering reliability | Causal | Limits of steering | Peripheral | no (all) | Background caveats |
| Mitigating Overthinking via Manifold Steering; GeoSteer (CoT); PersonaManifold; Hesitation Has a Geometry; Tethered Reasoning; BOReFT | NeurIPS 2025; 2601.10229; 2609.34571; 2610.02391; 2602.17691; 2609.33722 | 2025-26 | n.s. | Task-specific manifold steering | Causal | Manifold- or hyperbolic-aware steering | Peripheral | no (all) | Application variants |
| Geometry of Ordinal Representations; Symmetry in language statistics shapes geometry; Abstract representational geometry supports inference | 2607.04167; 2602.15029; 2606.23345 | 2026 | n.s. | Concept geometry | Corr. | Curved or ordinal concept manifolds | Peripheral | no (all) | Framing |
| Curved Inference: Concern-Sensitive Geometry in Residual Streams | arXiv 2507.21107 | 2025-07 | Gemma3-1B, LLaMA3.2-3B | 20 matched prompts | Correlational | Residual-trajectory curvature under a pullback unembedding metric | Peripheral | no (only "Curved Inference II" is in csv) | Curvature metric |
| Trajectory Geometry of Transformer Representations Across Layers; Reasoning Models Don't Just Think Longer, They Move Differently | 2606.09287; 2605.15454 | 2026 | GPT-2, TinyLlama, Qwen2.5; reasoning models | Layer and CoT trajectories | Correlational | Curvature and convergence; length confound | Peripheral / Supporting (control) | no / no | Trajectory-geometry controls |

### 2c. Direction 3: multi-step drift vs. jump (kept separate from single-turn)

| Title | ID / venue | Date | Models / scale | Setting | C/R | White-box? / gradual vs abrupt | Tier | csv | Why |
|---|---|---|---|---|---|---|---|---|---|
| State-Dependent Safety Failures in Multi-Turn LM Interaction (STAR) | arXiv 2603.15684 | 2026-03 | n.s. | Multi-turn safety | Mechanistic analysis (n.s. whether causal) | White-box: **steady drift** away from refusal representations **plus abrupt phase transitions** triggered by role-conditioned context | Core (D3) | yes | Both modes in internals |
| From Drift to Snap: Instruction Violation as a Phase Transition (LessWrong post) | lesswrong.com/posts/hG78AzdC56nz6QbnK | n.s. | Llama-3.1-70B; 26 dialogues | "No contractions" over 50 turns | Correlational | White-box: 20/21 violating dialogues snap (about turn 10); failing activations collapse to a "violation centroid" | Supporting (not peer-reviewed, small n) | no | Only abrupt-transition white-box evidence for constraint violation; >8B |
| A Representation Engineering Perspective on Multi-Turn Jailbreaks (Crescendo) | arXiv 2507.02956 | 2025-06 | n.s. | Crescendo | Correlational | White-box: representations grow more "benign" with more turns (gradual) | Supporting | no | Gradual internal shift |
| The Assistant Axis | arXiv 2601.10387 | 2026-01 | several models | Persona drift in long chats | Causal (activation capping) | White-box: drift along the axis, domain-dependent; capping stabilizes | Supporting | yes | Drift metric plus fix |
| Geometry of Divergence: Hidden-State Trajectories for Multi-Turn Reasoning | arXiv 2608.30650 | 2026-08 | 3 LLMs | 4 multi-turn tasks incl. τ-bench | Correlational + intervention at critical turns | White-box: temporal curvature and variance slope separate correct vs failed episodes before completion; failed episodes show more reversals | Supporting | yes | Trajectory signature of drift |
| Reasoning Fine-Tuning Induces Persistent Latent Policy States | arXiv 2607.18532 | 2026-07 | 1.5B-32B | CoT | Correlational | White-box: switching dynamical system with discrete latent policy regimes (jump-like) | Supporting | yes | Discrete-state framing; >8B |
| Polished but Unresolved: Late-Stage Pressure States in Tool-Use Agents | arXiv 2609.00823 | 2026-09 | n.s. | Long-horizon tool use | Causal | White-box: probe identifies a pressure state; steering changes early submission despite unresolved constraints | Core (D3) | yes | Agent acts against unresolved-constraint info; causal |
| When Agents Commit Too Soon | arXiv 2606.22936 | 2026-06 | Llama-3.1-70B (+2 models) | ReAct HotpotQA, StrategyQA | Correlational | White-box: early hidden-state convergence (step 4, r=-0.35); committed-wrong and committed-correct look alike | Supporting | yes | Early lock-in; >8B |
| Doomed from the Start: Early Abort via Probe Cascade | arXiv 2607.06503 | 2026-07 | Qwen-2.5-7B, Llama-3.2-3B | TextCraft etc. | Correlational | White-box: failure decodable from round 1 (not gradual onset) | Supporting | yes | Early-fixed state |
| Progress Is Not Prognosis: Local State and Eventual Outcome in Long-Horizon Agents | NeurIPS 2026 (no arXiv found) | 2026 | Qwen3-32B; 500 SWE-bench Verified runs | Coding agent | Correlational | White-box: frozen value axis dips after errors and recovers (local); a separate outcome direction (80.3% pairwise) | Supporting | no | Separates local drift from eventual outcome; >8B |
| Belief-action gap in LM agents | arXiv 2610.17444 | 2026-10 | n.s. | Grid worlds | Mostly correlational | White-box: action preferences sharpen and belief uncertainty drops near a commitment point | Core (D2 + D3) | no | (also in 2a) |
| Proxy Reward Internalization | arXiv 2606.09711 | 2026-06 | n.s. | Coding agent | Causal | White-box: probe rises *before* visible hacking (precursor) | Core (D2 + D3) | yes | (also in 2a) |
| TACT: Overthinking and Overacting in Coding Agents via Activation Steering | arXiv 2605.05980; NeurIPS 2026 | 2026-05 | n.s. | SWE agents | Causal | White-box: residual-stream detection of drift before behavioral failure | Supporting | yes | Drift detection plus fix |
| From Actions to Understanding: Conformal Interpretability of Temporal Concepts in LLM Agents | arXiv 2604.19775 | 2026-03 | n.s. | ScienceWorld, AlfWorld | Causal (prelim. steering) | White-box: success, failure and drift directions linearly separable over steps | Supporting | yes | Step-wise concept tracking |
| NeuroFilter: Activation-Based Guardrails (activation velocity) | arXiv 2601.14660 | 2026-01 | n.s. | Multi-turn privacy attacks | Correlational | White-box: activation velocity across turns | Supporting | yes | Temporal-derivative signal |
| Modeling and Predicting Multi-Turn Answer Instability | arXiv 2511.10688 | 2025-11 | n.s. | Follow-up pushback | Correlational | White-box: probes predict future answer changes | Supporting | yes | Turn-level sycophancy prediction |
| Dissociating the Internal Representations of Sycophancy | arXiv 2607.07003 | 2026-07 | Gemma-3-12B-IT, Llama-3.1-8B-IT | Two-turn pushback | Correlational | White-box: factual vs opinion sycophancy probes; transfer | Supporting | no | >8B |
| Real-Time Detection and Repair of LLM Agent Failures | arXiv 2608.02464 | 2026-08 | n.s. | Agents | Correlational + repair | Change-point (CUSUM) on surprise channels; one-class monitors | Supporting | yes | Explicit change-point method |
| A Behavioural and Representational Evaluation of Goal-Directedness | arXiv 2602.08964 | 2026-02 | n.s. | Grid world | Correlational | White-box: nonlinear spatial map; pre- vs post-reasoning activations | Peripheral | yes | Agent representations |
| Energy-Driven Latent Conflict Detection (instruction conflict) | arXiv 2609.08646 | 2026-09 | n.s. | Instruction drift | Correlational | White-box: energy separates compliant vs drifting states | Supporting | no | Drift-state separation |
| Beyond the Commitment Boundary: Probing Epiphenomenal CoT | arXiv 2606.13603 | 2026-06 | n.s. | Reasoning traces | Correlational | White-box: answer crosses a sharp boundary in one step, then stable | Supporting | yes | Abrupt commitment |
| Temporal Predictors of Outcome; Characterizing Backtracking; Statistical Physics of LM Reasoning | 2511.14773; ICLR 2026; 2506.04374 | 2025-26 | n.s. | CoT | Correlational | White-box: early prefix probes (selection-effect caveat); onset of backtracking; drift-diffusion with latent regimes | Supporting / Peripheral / Supporting | no (all) | Single-trace dynamics |
| Belief Dynamics Reveal the Dual Nature of ICL and Activation Steering | arXiv 2511.00617 | 2025-11 | n.s. | ICL + steering | Causal | Closed-form Bayesian: evidence accumulates smoothly and crosses at a predictable N* | Supporting | no | Formal smooth-state, abrupt-behavior link |
| ICL of Representations (Park et al., graph tracing) / Belief or Circuitry? | ICLR 2025 / arXiv 2605.08405 | 2025-26 | Llama-3.1-8B | Graph ICL | Corr. / causal | Sharp behavioral transition while PCA structure is recovered gradually; rebuttal says induction circuits suffice | Supporting | no / no | Same question at ICL scale |
| Chain Holds, Answer Folds: Trace-Answer Dissociation under Adversarial Pressure | arXiv 2605.29087 | 2026-05 | reasoning models | Multi-turn pushback | Behavioral (CoT vs answer) | Text-level: CoT stays correct while answer flips | Supporting (behavior + CoT) | no | Restates truth while capitulating |
| Models Recall What They Violate (DriftBench) | arXiv 2604.28031 | 2026-04 | 7 models (2 open) | Multi-turn ideation | Behavioral | Restatement probe (not white-box): restates constraints it violates; KBV 8-99% | Supporting | yes | The "restate but violate" agent phenomenon |
| World-Model Collapse as a Phase Transition | arXiv 2606.31399 | 2026-06 | several | Deterministic long-horizon tasks | Behavioral | Abrupt collapse band; world-state fidelity fails before action validity | Supporting (behavior-only) | no | Abrupt behavioral transition |
| World Model Science: SOC, Weak Chaos, Metastable Belief Dynamics | arXiv 2609.17419 | 2026-07 | n.s. | Long-horizon agents | Behavioral / state | Belief-basin transitions, error avalanches | Peripheral | no | Dynamical framing |
| Drift No More? Context Equilibria | arXiv 2510.07777 | 2025-10 | n.s. | Multi-turn | Behavioral (KL) | Drift stabilizes at equilibria (not runaway) | Supporting (behavior-only) | yes | Counter to monotone drift |
| SPINE: Sycophancy under Sustained Pressure; SYCON-Bench | arXiv 2609.09090; 2505.23840 | 2025-26 | n.s. | Multi-turn sycophancy | Behavioral | Erosion events vs sharp drops; Turn-of-Flip | Peripheral (behavior-only) | no / no (SYCON fragment matched, but not this title) | Behavioral drift vs flip |
| Time-To-Inconsistency: Survival Analysis | ICLR 2026 | 2026 | n.s. | Multi-turn | Behavioral | Abrupt prompt-to-prompt drift raises hazard; cumulative drift protective | Peripheral | no | Behavioral |
| Measuring and Controlling Persona Drift in LM Dialogs; Prompting Against Persona Drift | arXiv 2402.10962; 2609.24532 | 2024 / 2026 | n.s. | Persona drift | Behavioral + attention | Drift within 8 rounds; attention-decay hypothesis | Peripheral | no / yes | Background |
| Agent Drift (multi-agent); How Fast Do Agents Rot?; Context rot in long-horizon search; Asymmetric Goal Drift in Coding Agents | 2601.04170; 2609.01660; 2606.29718; Cruz et al. | 2026 | GPT-4, Claude etc. | Agents | Behavioral | Degradation curves (geometric form wins 28/36 cells, 2609.01660) | Peripheral | no / no / yes / yes | Behavior-only drift |
| LH-Deception; DeceptGuard; Detecting Strategic Deception with Linear Probes (Apollo); SchemeArena | 2510.03999; 2603.13791; 2504.04072; 2609.08126 | 2025-26 | incl. Llama-3.3-70B (Apollo) | Deception and scheming | Mostly behavioral; probes | Long-horizon deception auditing; probe monitors | Peripheral | yes / yes / yes / yes | Background; no per-step onset analysis seen |
| Insights into LLM Long-Context Failures: When Transformers Know but Don't Tell | arXiv 2406.14673; EMNLP Findings 2024 | 2024-06 | n.s. | Long context | Correlational | White-box (single-turn): position of key info decodable, yet not used | Supporting | no | "Recognize but fail to utilize" |
| Attractor States Emerge in Multi-Turn LLM Conversations | arXiv 2606.30571; ICML 2026 | 2026-06 | 7 LLMs (incl. API models) | Dyadic debates | Correlational | "Representation space" trajectories, probably text embeddings, so not model internals (unverified) | Peripheral | yes | Attractor framing |
| TRACES; PsychoPass; DeepContext; Representation Signatures in Trading Agents | 2605.27690; 2606.03136; 2602.16935; 2605.28850 | 2026 | observer LLM / embeddings / Qwen2.5-0.5B | Multi-turn safety; trading | Correlational | Mostly observer-model or embedding trajectories | Peripheral | yes / no / no / yes | Monitoring frameworks |

---

## 3. Off-topic papers seen (title only)
- Neural representational geometries reflect behavioral differences in monkeys and RNNs
- On the Relationship Between Representation Geometry and Generalization in Deep Neural Networks
- Universality of representation in biological and artificial neural networks
- Neural population geometry and optimal coding of tasks with shared latent structure
- Geometry-Informed Neural Networks
- Neural Networks with Fractal Architecture
- When do complex-valued neural networks help?
- Model Merging by Output-Space Projection
- Isometric Representation Learning for Disentangled Latent Space of Diffusion Models
- Isometry pursuit
- Gromov-Wasserstein-like Distances in the Gaussian Mixture Models Space
- Deep Language Geometry: Constructing a Metric Space from LLM Weights
- Language Models as Hierarchy Encoders
- Attention Manifolds: Steering or Blocking LMs by Editing B-Spline Surfaces
- Non-Autoregressive Semantic Steering via Continuous-Time Phase Manifolds
- Universal Behavioral Manifold 2 (UBM2)
- Manifold Projection and Iterative Autoencoder Refinement for MLM
- Leviathan: Decoupling Input and Output Representations
- RMRRT: Riemannian Barrier Metric RRT
- Fine-grained descending control of steering in walking Drosophila
- GGSS: Geodesic-Gated Spherical Steering for VLM debiasing
- Learning to Steer, Steering to See (RLVR geometry)
- High-Dimensional Random Projection for Activation Steering
- Adversarial Robustness of Activation Steering
- Cross-Lingual Activation Steering
- Rethinking Layer Relevance Beyond Cosine Similarity
- Cosine-Similarity Routing with Semantic Anchors for MoE
- The Amplifying Mirror (partisan direction)
- COSMIC: Generalized Refusal Direction Identification
- The Reasoning-Memorization Interplay Is Mediated by a Single Direction
- Intention-like representations in language models? (philosophy)
- Dissociating language and thought in LLMs
- FlashBack (streaming VLMs)
- Auditing Asset-Specific Preferences in Financial LLMs
- GuideVLA; Learning What to Say to Your VLA; Steering Robustness into World Action Models
- SteeringSafety benchmark
- Improved Representation Steering for LMs (RePS)
- Taxonomy of Representation Engineering
- Controllable LLM Reasoning via SAE Steering
- Causal Activation Steering via Sparse Mediation
- Activation Steering Decoding
- Phantom Transitions in LM Fine-Tuning
- Phase Transitions in Attention: Bayesian Theory of Copy Heads
- In-Context Learning Dynamics with Random Binary Sequences
- Markov-chain ICL stages (2402.11004)
- PABU; RB-VLA; Belief inertia in embodied agents
- PaperGuide
- Skill Coverage
- radar and beamforming steering-vector papers (several)
- physics papers on driving-induced crossover, critical drift, spatial activation systems
- Interaction Dynamics as a Reward Signal
- Expert-neuron alignment (ExpertLens)
- Geometry of Representational Failures in VLMs
- Model merging, survey and explainability-survey papers in OpenAlex results

## 4. BLOCKED list
- **arXiv export API:** HTTP 429 on all 8 family-1 queries (rows 3 to 10). Not retried later.
- **Semantic Scholar API:** HTTP 429 at first test. Not used.
- **OpenAlex API:** the free daily budget for the network IP was exhausted after 10 queries, so 6 family-2 queries failed (rows 24 to 29). An API key would be needed, and I did not use one.
- **Web search:** the shared per-turn budget of 200 searches was exhausted. 4 planned queries were not run (rows 96 to 99), and the 10-empty-query stopping rule could not be completed. The user can raise CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION or start a follow-up to continue.
- **OpenReview reading via Chrome:** not done, because I ran out of time and budget. These papers are known only from NeurIPS, ICML and ICLR listing snippets, with no arXiv page found: Spokes not a Ring; Progress Is Not Prognosis; Sharp Directions Are Against You; Geometric Signatures of Steerability; Characterizing Backtracking; Time-To-Inconsistency.
- **No PDFs were downloaded.** The downloads folder was not used.
- **Unverified items:**
  - The "Billa (2026)" LAP framework is probably the same as 2604.15557, but this is not confirmed.
  - The "approximate isometry, about 2.8× lower energy" claim for 2605.05115 comes only from a LinkedIn summary.
  - The Causal Tongue-Tie and Senses Wide Shut details come from Modelwire write-ups only.
  - The authority-bias paper 2607.00415 is known only from an alphaxiv overview.
  - The HARC paper is known from a docshare mirror only.

---

## Continuation (2026-10-22, cut short by the coordinator's time limit)

### QUERY LOG (continued)

| # | Query | Source | Fam | Hits looked at | New relevant added |
|---|---|---|---|---|---|
| 100 | arxiv knowing-doing gap probe directions orthogonal action subspace LLM 2026 | WS (the single retry the coordinator allowed; it worked) | F2 | 9 | 0. Confirms details for 2605.14038: authors Cheng, Fan, JafariRaviz, Rezaei, Feizi; NeurIPS 2026 poster (neurips.cc/virtual/2026/poster/154208); probe directions only partly aligned in mid-layers, near-orthogonal only at the late-layer last token |
| 101 | "manifold steering" (connectivity test only; results not reviewed) | arxiv.org/search via curl | F1 | 0 reviewed | 0. HTTP 200; the page lists 2610.30000, 2610.22738 and 2610.13195 as the newest hits, which I have not looked at |

### PAPER TABLE (continued): updates to existing rows
- **Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use** | arXiv 2605.14038; **NeurIPS 2026** | Cheng, Fan, JafariRaviz, Rezaei, Feizi | the geometry is partial alignment in mid-layers and near-orthogonality at the late-layer, last-token action position | still Core | csv yes.

No new papers were added in this continuation.

### BLOCKED (updated)
- **Long-tail search not done.** The coordinator's time cut came before any review of arXiv search pages (the scripts are ready: axs.sh and axparse.py in the scratchpad), ACL Anthology or alphaxiv.
- **OpenReview reading via Chrome not done** (time cut). These are still known only from listing snippets: Spokes, not a Ring; Progress Is Not Prognosis; Sharp Directions Are Against You; Geometric Signatures of Steerability; Characterizing Backtracking; Time-To-Inconsistency. COLM 2026, NeurIPS 2026 and ICLR 2027 submissions were not searched.
- **Secondary-source claims still unverified** (time cut):
  - The approximate-isometry and about 2.8x lower-energy claim for 2605.05115 (LinkedIn).
  - Causal Tongue-Tie and Senses Wide Shut (Modelwire).
  - HARC (docshare).
  - The authority-bias paper 2607.00415 (alphaxiv overview).
  - The Billa (2026) LAP framework, probably the same as 2604.15557.
- **Earlier BLOCKED items still apply:** the arXiv API and Semantic Scholar returned 429, and the OpenAlex daily budget was exhausted.

**Stopping rule: NOT met. Interrupted by the coordinator's time cut; the last 10 completed queries were still adding papers.**
