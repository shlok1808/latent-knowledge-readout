# Scout pass 2: group-geometry angle (2026-10-25)

Triage from abstracts only. Budget: 12 WebSearch queries, plus arXiv API lookups by id (to resolve titles, not to search).
"Seen r1/r2" means an earlier sweep listed the paper, but it is not in the new-angle reading set.

## The user's paper: 2608.27510 (already in the set as u5)
How Do Linear Probes Emerge? Concept-Targeted Attribution (Palit, Draye, Zhang, Schölkopf, Jin; 27 Aug 2026).
- **Verdict: relevant, medium-high, for fate 2 (bypass).**
- It builds attribution graphs that target a probe score instead of a next-token logit.
- It reports a double dissociation. Ablating features that only the probe graph uses lowers the probe score and almost never changes the token. Ablating features that only the logit graph uses flips the token every time.
- Models: Gemma-2-2B and Llama-3.2-1B.
- This is "decodable but not used", shown at the feature level. A probe-vs-logit graph overlap per item could serve as a bypass check.
- Limits: small models; it needs transcoders; it traces one probe direction rather than groups of activations. It also reports a shared feature core across prompts and layers, which is a group-level claim.

## Step 1: citation chase (cites.py)
13 arXiv ids are cited by 2 or more of our papers and are not in our set:

| id | title | judgement |
|---|---|---|
| 2310.01405 | Representation Engineering | foundational, known |
| 2407.21783 | Llama 3 Herd | model report, skip |
| 2308.10248 | Activation Addition steering | foundational, known |
| 2505.09388 | Qwen3 Technical Report | model report, skip |
| 2408.00118 | Gemma 2 | model report, skip |
| 2301.04709 | Causal Abstraction | foundational, known |
| 2110.14168 | GSM8K | dataset, skip |
| 2311.03658 | Linear Representation Hypothesis (causal inner product) | **ranked #13** (methods) |
| 2207.05221 | LMs (Mostly) Know What They Know | known |
| 2502.00873 | LMs Use Trigonometry to Do Addition | off-topic, skip |
| 2601.04480 | When Models Manipulate Manifolds (counting / linebreaking) | **ranked #10** |
| 2505.18235 | Origins of Representation Manifolds | near-miss |
| 2310.06824 | Geometry of Truth | known (seen r2) |

## Step 3: PASS-2 ranked list (15)

| # | Paper | Why it matters for the group angle | Priority | Scoop risk |
|---|---|---|---|---|
| 1 | Rewriting or Reweighting? A Geometric Account in LMs, [2608.01835](https://arxiv.org/abs/2608.01835) (Aug 2026; seen r1) | Builds low-dimensional "behavioral manifold" charts for sycophancy in two spaces: activations, and how much the model routes information through them. Tracks the charts through post-training: SFT rewrites the geometry, reward optimization keeps the chart and reweights it. The charts partly align across architectures. This is close to Stage B's "removed or only hidden?" question, asked with group geometry. | high | **HIGH** (Stage B + geometry) |
| 2 | Signal Competition: Unifying Sycophancy and Conformity, [2601.11563](https://arxiv.org/abs/2601.11563) (seen r2) | Claims that sycophancy and conformity share one "compliance subspace", with a linear boundary where a social signal suppresses a calibration signal. This is the opposite of H1 (different pressures give different geometry) and an "override" framing. We need to cite and test against it. | high | medium (H1) |
| 3 | Gotta Catch Them All: the Modes of Sycophancy, [2607.20146](https://arxiv.org/abs/2607.20146) (Jul 2026; seen r2) | 948 social-pressure items. The three modes give near-identical outputs but are perfectly linearly separable from layer 14, emerge at different stages and use distinct attention circuits. This supports H1 and conflicts with #2. | high | medium (H1) |
| 4 | Sycophancy Hides Linearly in the Attention Heads, Genadi et al., [EACL 2026](https://preview.aclanthology.org/credits/2026.eacl-long.324) (seen r2) | A sparse set of middle-layer attention heads attend to tokens where the user expresses doubt; steering those heads works best. These are the "heads that read the user's claim" components. | high | low-medium (no fates) |
| 5 | A Low-Rank Subspace Analysis of LLM Interventions, [2606.14388](https://arxiv.org/abs/2606.14388) (Torr group; seen r2) | Treats behaviours (including sycophancy) as low-rank subspaces at 7B-70B. Measures overlap by principal angles and each subspace's angle to a "decision subspace". Interventions spill over more when overlap is high. This gives a ready metric for bypass vs routed (the angle from the correct-answer subspace to the decision subspace) and for Stage B side effects. | high | medium |
| 6 | Spokes, not a Ring: Testing Continuity of Representation Manifolds, [NeurIPS 2026](https://neurips.cc/virtual/2026/157355) (no arXiv found; seen r2 as listing only) | The weekday "ring" is seven discrete states with a step-like map from representation to behaviour. Gives sensitivity and occupancy tests to run before calling anything a manifold. This is the control for "manifold vs discrete states". | high | low |
| 7 | Polymorphism Is Rotation, [2605.24577](https://arxiv.org/abs/2605.24577) (May 2026; new) | Independently trained seeds differ by a random rotation; one Procrustes fit restores the alignment, and the same holds across checkpoints. Steering vectors can invert when misaligned. Implication: a fixed probe can read "erased" when the information was only rotated. This motivates the Procrustes test for transformed vs erased and cross-seed claims. Only Pythia-70m scale. | medium | low |
| 8 | Steerable but Not Decodable, [2604.02608](https://arxiv.org/abs/2604.02608) (seen r2) | Function-vector steering works where the logit lens cannot decode the answer, across 6 models from 3 families. The reverse case (decodable but not steerable) is nearly empty. Keeps decodability and controllability separate, with some nonlinear encoding ("transformed"). | medium | low |
| 9 | A Geometric Perspective on Next-Token Prediction: Three Emerging Phases, [2605.09011](https://arxiv.org/abs/2605.09011) (new) | Tracks a "predictive readout subspace" across layers on the Grassmannian (rise, plateau, descent). This gives a reference path to compare the correct-answer subspace against, to find where routing to the output fails. | medium | low |
| 10 | When Models Manipulate Manifolds: the Geometry of a Counting Task, [2601.04480](https://arxiv.org/abs/2601.04480) (cited by 2 of ours) | Attention heads twist count manifolds, and an orthogonal arrangement creates a linear decision boundary. A worked example of "transformed" information (information re-encoded on its way to a decision) with a known component type. | medium | low |
| 11 | Testing the Limits of Truth Directions in LLMs, [2604.03754](https://arxiv.org/abs/2604.03754) (new) | Truth directions depend on the layer, the task type and the instruction. Relevant to H3 (probes trained on normal behaviour) and to probe-direction stability: one fixed direction is fragile. | medium | low |
| 12 | The Information Geometry of LLMs Is Shared, Learned, and Controllable, [2609.11063](https://arxiv.org/abs/2609.11063) (seen r2) | Output (Fisher-Rao) geometry agrees across architectures more than activation geometry, and activation path length tracks behaviour change. For cross-model claims, compare in output-geometry coordinates, not only activation CKA. | medium | low |
| 13 | The Linear Representation Hypothesis and the Geometry of LLMs, [2311.03658](https://arxiv.org/abs/2311.03658) (cited by 2 of ours) | Its causal inner product: Euclidean angles and principal angles in the residual stream can mislead without it. A methods prerequisite for the subspace-angle plans. | medium (methods) | none |
| 14 | Are LMs Aware of the Road Not Taken?, [2511.04527](https://arxiv.org/abs/2511.04527) (new) | Hidden states predict token-level forking or uncertainty, and steerability peaks at those change points. A method for finding the "fork point" where flipped and resisted items split. Multi-step CoT setting. | low-medium | low |
| 15 | Layer-wise Representation Dynamics, [2605.12714](https://arxiv.org/abs/2605.12714) (new) | Grassmann speed and curvature tooling across 31 models. A method source only (embedding tasks). | low | none |

Near-misses (not ranked):
- 2508.02087 When Truth Is Overridden: two-stage sycophancy; already deep-read in r2.
- 2607.07003 Dissociating sycophancy subtypes: models differ in factual vs opinion geometry; seen r2.
- 2602.02315 Shape of Beliefs: curved belief manifolds, linear steering goes off-manifold.
- 2505.18235 Origins of Representation Manifolds.
- 2603.03308 Old Habits Die Hard: multi-turn geometric trap (direction 3).
- 2606.10304 MIRAGE: a direction that flips polarity (an example of a sign flip).
- 2609.17376 belief-state geometry in-context.
- 2412.07961 Forking Paths (black-box).

Gap check: no search found a paper that compares flipped vs resisted items as point clouds matched on prior confidence. None found Procrustes-based erased-vs-rotated tests after fine-tuning at 9B+, or RSA/CKA across pressure types. These remain open on abstract-level evidence; #1, #2 and #3 are the closest.

## BLOCKED
- OpenReview (paper folder `u1-openreview-ZP7VzpXmKR`): api.openreview.net and api2.openreview.net both return 403 "Challenge verification required" (a bot check). I did not bypass it. The `paper.pdf` in that folder is an HTML 403 page, not a PDF. Its title is unconfirmed; it may be Spokes (#6), but that is not verified.
- Spokes, not a Ring: only the NeurIPS virtual listing was found, with no arXiv version. Read it in Chrome on OpenReview, read-only, as approved before.
- Genadi et al. (EACL 2026): seen through the ACL Anthology preview and search snippets only; no arXiv id found.
