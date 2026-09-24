# Results — does the single-step probe recipe transfer to multi-step agent tasks?

Status: **interim (2026-09-24).** Phase 2 and the depth-1 stage-2 replication are complete for Qwen3-1.7B on TextCraft. Phase 3 last-step version done (§4); still to come: Phase 3 averaged-over-range version, depth ≥ 2 subset, the MC-value label, Phase 4 and the 7B model. Every number below is within-task AUC unless marked otherwise, with task-grouped 5-fold cross-fitting (no task in both train and test).

## Headline finding for Prof. Lin

**For predicting the final outcome of a multi-step episode, the standard single-step probe recipe adds almost nothing over surface text features. It works very well for predicting whether the immediate action is valid.** The one exception is a narrow window: depth-1 tasks at round 3, where the probe beats text by about 0.10. That window replicates the stage-2 result and holds across most layers, but the advantage is gone one or two rounds either side.

We treat this as a finding, not a failure. It is the split Prof. Lin asked us to characterise ("what's wrong if we directly treat it as a single-step task?"). The recipe is: train a linear probe on the residual state at the end of step t, with a label.
- It succeeds when the label is a property of the step the model just wrote.
- For the long-horizon outcome, it matches an outside observer reading the transcript, except in a one-round window.

That round-specificity is itself early evidence for Lin's stage hypothesis. The Phase 3 transfer matrices test it directly.

## What is new here vs prior work

- **Ruan et al. (2607.06503), Fig. 3,** already shows round-by-round probe separability for TextCraft: round-1 AUC 0.86 for Qwen-2.5-7B and 0.81 for Qwen3-1.7B. They train one probe per round and never transfer across rounds. **Our round-by-round results confirm that baseline** (pooled round-1 0.791).
- **PAIR (2605.17877)** shows hidden-state probes degrade under an injected erroneous prefix, while attention features stay robust. They don't test cross-stage transfer either (their only transfer is cross-domain, GTA → HotpotQA), and they report pooled AUROC with no text baseline.
- **Our contribution:** the within-task decomposition, and **the stage-transfer matrix**: train on stage A, test on stage B, with a same-episode control, on naturally occurring trajectories. Details in ASSUMPTIONS A-22, A-23.

## Hypothesis

From the 2026-09-17 meeting (transcript 21:52–39:10): a probe trained on one stage of a trajectory performs well on that stage and degrades on others, like CEB's motivation figure (arXiv 2607.12397, Fig. 1). If so, stages carry different internal patterns, which would motivate a stage-robust method.

**What would falsify it:**
- In the **same-episode** transfer matrix (primary analysis), off-diagonal cells within the diagonal's confidence interval, for every label.
- A single probe trained on all stages matching each per-stage probe on its own stage.

## Setup

- **Model and env:** Qwen3-1.7B on AgentGym TextCraft, recipe depths 1–3.
  - 533 tasks × 8 rollouts = **4,264 episodes**: depth 1 = 1,000, depth 2 = 2,328, depth 3 = 936.
  - Success rate: 76.6% at depth 1 (stage-2 HF run: 77.5%), 9.7% at depth 2, 1.4% at depth 3.
  - Mixed-outcome tasks: 49 at depth 1, 92 at depth 2, 8 at depth 3.
- **Sampling:** vLLM, temp 0.8, top-p 0.95, max 20 rounds.
- **Activations:** replayed through HF over the saved tokens; every layer, last token of each round's generation; 69,782 rows.
- **Labels (1 = bad outcome):**
  - FINAL: the episode fails.
  - IMMEDIATE-validity: the env rejects this round's action.
  - IMMEDIATE-progress: this round's action added no item on the goal's recipe tree.
  - MC value (depth 2 only; on hold).
- **Baselines:** TF-IDF (char 3–5-grams) plus 5 surface counts, trained on the visible text up to and including round t's response. Round t's observation is excluded, because the activation hasn't seen it and it would leak the IMMEDIATE labels.

## Strategies

- **Inputs:** S1 = last-token state at round t; S2 = mean of rounds 1..t.
- **Probes:** L2 logistic (C = 1) and difference of means.
- **Grid:** every layer (0 = embeddings, 1–28 = blocks) × t ∈ {1, 2, 3, 5, 8}. That's 580 cells per label, so every "best" number is an exploratory maximum over that grid.

## Results

### 1. FINAL outcome: probe ≈ text at every round

S1, L2 logistic, best of 29 layers per round, versus TF-IDF+surface at the same round:

| round t | probe | text | probe − text |
|---|---|---|---|
| 1 | 0.585 | 0.569 | +0.016 |
| 2 | 0.627 | 0.605 | +0.022 |
| 3 | 0.639 | 0.639 | 0.000 |
| 5 | 0.745 | 0.732 | +0.013 |
| 8 | 0.819 | 0.825 | −0.006 |

- **The gap stays within ±0.02, even after picking the best layer.** Mean-pooling (S2) moves it to +0.018 at t = 8. Difference-of-means is worse everywhere.
- **The layer curve is broad, not peaked:** 0.75–0.82 across layers 9–28 at t = 8.
- **Both curves rise with t, but a large part of that is survivorship.** The failure share among alive episodes goes 0.76 (t = 1) → 0.85 (t = 3) → 0.95 (t = 8), and mixed-outcome tasks fall from 149 to 108. This is why the same-episode control is the primary Phase 3 analysis.

### 2. IMMEDIATE validity and progress: probe ≫ text

| label | best probe | text | embeddings only (layer 0) |
|---|---|---|---|
| action invalid, t = 1 | 0.970 (L16) | 0.826 | 0.537 |
| action invalid, t = 3 | 0.990 (L15) | 0.840 | 0.875 |
| no recipe progress, t = 5 | 0.982 (L16) | 0.779 | ≈ 0.81 |

- **The margin is +0.13 to +0.20 over text.**
- **At t = 1 it is genuinely computed:** embeddings are at chance and layer 16 reaches 0.97.
- **From t ≥ 3 much of it is lexical:** the embedding of the emitted token alone reaches 0.84–0.88. Mid-layer computation adds about 0.11 on top.

### 3. Stage-2 replication (depth 1 only): replicates, but only at round 3

Depth-1 episodes only (1,000 episodes, 125 tasks), action-token site as in stage-2, full layer × round grid:

| round t | best probe (layer) | text | layers 1–28 beating text by > 0.05 | mixed tasks |
|---|---|---|---|---|
| 1 | 0.620 (L13) | 0.605 | 0% | 49 |
| 2 | 0.722 (L25) | 0.694 | 0% | 49 |
| **3** | **0.834 (L18)** | **0.717** | **71%** | 44 |
| 5 | 0.705 (L17) | 0.655 | 4% | 31 |
| 8 | 0.660 (L16) | 0.689 | 0% | 21 |

- **The pre-registered cell (round 3, layer 12) scores 0.815 [0.75, 0.87]** within-task (pooled 0.856), versus 0.717 text. Stage-2 had 0.801 vs 0.725, so **the stage-2 result replicates** on fresh rollouts with a new seed.
- **At round 3 it is not a single-cell fluke:** 71% of layers beat text by more than 0.05.
- **It is specific to round 3.** At rounds 1, 2, 5 and 8 the probe ties or loses to text; at round 8 it is 0.03 below at the best layer and far below at most layers.
- **Why round 3 is plausible:** at depth 1 the round-3 population is exactly the episodes that did *not* succeed in their first two actions. Round 3 is where the model is recovering from, or repeating, a failed first attempt, which is where internal state could carry information the transcript doesn't yet show. This is an interpretation, not a tested claim.
- **In the pooled depth-1–3 analysis (§1) the window disappears.** There, depth-2/3 episodes, which are mostly long failures, dominate every round.

### 4. Stage transfer (Phase 3), last-step version: the final-outcome probe is stage-specific; the immediate-validity probe mostly transfers

**Setup.** Each round range is represented by its last step: early 1–2 → round 2, mid 3–5 → round 5, late 6–10 → round 10. A probe is trained on one representative round and tested on the others (task-grouped folds shared across stages). The **same-episode** (cohort) version uses only episodes lasting ≥ 10 rounds (3,373 episodes, 82 mixed-outcome tasks), so every stage is scored on the same episodes. The naive version uses whoever is alive at each round. Layers {13, 15, 16, 28}: the top 3 from Phase 2 plus the last layer. All depths. The averaged-over-range version (every round in the range, episode-weighted) is still running and will be added alongside this.

**FINAL outcome, same-episode, layer 16** (within-task AUC [95% CI]; rows = trained on, columns = tested on):

| train ↓ / test → | early (r2) | mid (r5) | late (r10) |
|---|---|---|---|
| early (r2) | 0.537 [0.47, 0.60] | 0.479 [0.41, 0.55] | 0.480 [0.42, 0.55] |
| mid (r5) | 0.530 | 0.589 [0.52, 0.66] | 0.653 [0.59, 0.72] |
| late (r10) | **0.425 [0.37, 0.49]** | 0.549 | **0.766 [0.71, 0.82]** |
| pooled (all 3) | 0.525 | 0.620 | 0.749 |

- **A late-stage probe is strong on its own stage (0.77–0.80 across the 4 layers) and fails on the same episodes' early states.** At layers 15 and 16 it scores significantly *below* chance there (0.403 [0.34, 0.47]; 0.425 [0.37, 0.49]). The failure direction it learned at round 10 is anti-aligned with how failing and succeeding episodes differ at round 2. That is a sign flip, not just a weaker signal.
- **Early states carry little final-outcome signal of their own** (diagonal 0.54–0.55; lower CIs 0.47–0.48). So "early doesn't transfer forward" is partly "there is little to transfer".
- **Mid → late degrades by ~0.1:** 0.63–0.71 vs late's own 0.76–0.80. The CIs overlap.
- **A single probe pooled over all three stages roughly matches the per-stage probes on each stage** (L16: 0.525 / 0.620 / 0.749 vs diagonal 0.537 / 0.589 / 0.766). So this matrix does *not* show that per-stage probes beat one pooled probe. Both versions of Lin's hypothesis need stating separately: (a) a probe trained at one stage fails at another — **supported, strongly for late → early**; (b) a pooled probe can't cover all stages — **not supported here**.
- **The naive matrix overstates transfer.** L16 naive early → mid 0.643 and early → late 0.619, vs 0.479 / 0.480 in the same-episode version. Without the cohort control, population shift (task difficulty carried across rounds) makes early probes look like they transfer.
- **Calibration drifts only mildly:** ECE 0.03–0.04 on the diagonal, 0.05–0.08 off it (largest for early → mid/late and late → early).

**IMMEDIATE validity (action rejected), same-episode, layer 16:** diagonal 0.980 / 0.986 / 0.977; worst off-diagonal early → late 0.923 and late → early 0.933. **The probe that works mostly transfers:** a drop of ≤ 0.06 at mid-network. The last layer transfers worse (L28 early → late 0.832 vs 0.971 on its own stage). No-progress behaves the same way (L16 worst off-diagonal 0.857, late → early).

**Reading.** Across stages, the recipe splits along the same line as in §1–2. The immediate-step probe is accurate *and* stage-robust at mid layers. The final-outcome probe is stage-specific: strongest late, weak early, and reversed when a late probe reads early states. That is the pattern Lin predicted, for the long-horizon label.

**Caveats.**
- The same-episode population is episodes lasting ≥ 10 rounds: mostly failures plus slow successes (82 mixed tasks).
- The layers were chosen from Phase 2 on the same data. That can inflate absolute AUCs but not the matrix shape.
- 4 layers × 3 labels × 2 populations × 12 cells were computed, so single cells should not be over-read. The late → early reversal holds at 3 of 4 layers (L13: 0.522, n.s.).

## Open questions

1. Does the IMMEDIATE-label probe, the one that works, stay stable across stages? This is the version of Lin's hypothesis that can still come out either way.
2. Why is the depth-1 advantage confined to round 3? Does it track the recover-or-repeat decision after a failed first attempt?
3. Is "probe ≈ text for final outcome" specific to a 1.7B model that mostly fails on hard tasks, or does it hold at Qwen2.5-7B (Ruan's strongest TextCraft setting)?
4. Does an MC-value label, which varies within an episode, recover a hidden-state advantage that the binary final label can't show?
