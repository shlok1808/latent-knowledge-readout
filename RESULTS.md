# Results — does the single-step probe recipe transfer to multi-step agent tasks?

Status: **interim (2026-09-24).** Phase 2 and the depth-1 stage-2 replication are complete for Qwen3-1.7B on TextCraft. Still to come: Phase 3 (stage transfer, naive and same-episode), the MC-value label, Phase 4 and the 7B model. Every number below is within-task AUC unless marked otherwise, with task-grouped 5-fold cross-fitting (no task in both train and test).

## Headline finding for Prof. Lin

**For predicting the final outcome of a multi-step episode, the standard single-step probe recipe adds almost nothing over surface text features. It works very well for predicting whether the immediate action is valid.** The one exception is a narrow window: depth-1 tasks at round 3, where the probe beats text by about 0.10. That window replicates the stage-2 result and holds across most layers, but the advantage is gone one or two rounds either side.

We treat this as a finding, not a failure. It is the split Prof. Lin asked us to characterise ("what's wrong if we directly treat it as a single-step task?"). The recipe is: train a linear probe on the residual state at the end of step t, with a label.
- It succeeds when the label is a property of the step the model just wrote.
- For the long-horizon outcome, it matches an outside observer reading the transcript, except in a one-round window.

That round-specificity is itself early evidence for Lin's stage hypothesis. The Phase 3 transfer matrices test it directly.

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

### 4. Stage transfer (Phase 3): running

Coming, side by side:
- naive (alive-at-stage) vs same-episode transfer matrices
- round stages (early / mid / late) and milestone stages (k-th craft)
- per-stage probes vs one pooled probe
- AUC and ECE per test round

## Open questions

1. Does the IMMEDIATE-label probe, the one that works, stay stable across stages? This is the version of Lin's hypothesis that can still come out either way.
2. Why is the depth-1 advantage confined to round 3? Does it track the recover-or-repeat decision after a failed first attempt?
3. Is "probe ≈ text for final outcome" specific to a 1.7B model that mostly fails on hard tasks, or does it hold at Qwen2.5-7B (Ruan's strongest TextCraft setting)?
4. Does an MC-value label, which varies within an episode, recover a hidden-state advantage that the binary final label can't show?
