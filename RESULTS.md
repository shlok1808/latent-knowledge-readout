# Results — does the single-step probe recipe transfer to multi-step agent tasks?

Status: **interim (2026-09-24).** Phase 2 and the depth-1 stage-2 replication are complete for Qwen3-1.7B on TextCraft. Phase 3 done for all three schemes, all depths and depth ≥ 2, at layers 12/13/15/16/28 (§4); still to come: the MC-value label, Phase 4 and the 7B model. Every number below is within-task AUC unless marked otherwise, with task-grouped 5-fold cross-fitting (no task in both train and test).

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

## Pre-registration for ALFWorld and τ²-bench (written 2026-09-24 ~4:55 PM ET, before any probe result for either env)

Written after the TextCraft results (§1–§4), before extraction had finished for ALFWorld (Qwen2.5-7B, 1,600 episodes) or τ²-bench (Qwen3-8B; pilot 160 episodes, full 1,312). Nothing below may be edited after those results arrive. Deviations go in a dated "Deviations" note.

**Questions** (from Lin's hypothesis, sharpened by TextCraft):
- **Q1, final outcome, stage transfer:** does a probe trained at one stage fail at another, on the same episodes?
- **Q2, immediate step, stage transfer:** does the step-level probe stay stable across stages? τ² is the first env with a real step label: the tool call matches a reference-trajectory action (A-24).
- **Q3, probe vs text:** does the hidden state beat TF-IDF+surface on the final outcome anywhere?

**Predictions** (from TextCraft):
- **P1:** same-episode, last-step scheme: late → early < late → late by ≥ 0.10 within-task AUC. Late → early's CI includes 0.5 or lies below it.
- **P2:** the IMMEDIATE-label probe transfers: worst off-diagonal within 0.07 of the diagonal at mid layers.
- **P3:** final-outcome probe ≈ text: |probe − text| ≤ 0.03 within-task at most rounds; no advantage broad across layers.

**What falsifies them** (each env separately):
- **P1 is falsified** if late → early is within the late → late CI at *both* pre-specified layers. That would mean no stage gap for the final outcome in that env.
- **P2 is falsified** if the worst off-diagonal drops > 0.10 below the diagonal at both pre-specified layers. That would mean the step-level probe is also stage-specific.
- **P3 is falsified** if the probe beats text by > 0.05 at ≥ 2 rounds with > 50% of layers.

**Fixed in advance:**
- **Layers:** 12 and 28 of 28 (Qwen2.5-7B, ALFWorld: 12 = stage-2's layer, 28 = last, as Ruan used). Proportional depth for Qwen3-8B (36 layers): **16 and 36**. Selected layers get reported too, but only as exploratory.
- **Stage schemes:**
  - (a) last step of round ranges: early 1–2 → r2, mid 3–5 → r5, late 6–10 → r10
  - (b) range-averaged
  - (c) milestones: ALFWorld = k-th state change ("You pick up / put / clean / heat / cool …"); τ² = k-th tool call matching a reference action
  - (d) τ² only, conversation phase: *identify* (before the first successful user-lookup tool call) → *lookup* (read-only tools) → *act* (first write tool call onward) → *close* (final 2 turns). Analysed alongside (a)–(c) per Shlok.
- **Population:** the same-episode cohort is primary (episodes present in every compared stage); naive reported beside it.
- **Metric:** within-task pair-weighted AUC with task-cluster bootstrap CIs; pooled AUC and ECE secondary. Cells with < 15 mixed tasks are flagged and not used for the P1–P3 verdicts.

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

### 4. Stage transfer (Phase 3): the final-outcome probe does not transfer across stages; the immediate-validity probe does

**Setup.** Train a probe on one stage, test it on the others. Task-grouped folds are shared across stages. Three stage schemes:
- **Last step of each range** (early 1–2 → round 2, mid 3–5 → round 5, late 6–10 → round 10).
- **Range-averaged** (every round in the range, each episode weighted equally).
- **Milestones** (the round of the k-th non-terminal craft).

Two populations:
- **Same-episode (primary):** only episodes lasting ≥ 10 rounds, so every stage is scored on the same episodes. That's 3,373 episodes, 114 successes, 82 mixed-outcome tasks: depth 1 = 252 episodes / 15 mixed tasks, depth 2 = 2,187 / 59, depth 3 = 934 / 8.
- **Naive:** whoever is alive at each stage.

Layers: 12 is the stage-2 best layer (pre-specified, independent rollouts), 28 is Ruan's layer (pre-specified), and 13 / 15 / 16 were selected from Phase 2 on these data.

**FINAL outcome, same-episode, last-step scheme: layer 12 vs layer 16** (within-task AUC; rows = trained on, columns = tested on):

| train ↓ / test → | L12 early | L12 mid | L12 late | | L16 early | L16 mid | L16 late |
|---|---|---|---|---|---|---|---|
| early (r2) | 0.562 | 0.475 | 0.490 | | 0.537 | 0.479 | 0.480 |
| mid (r5) | 0.510 | 0.576 | 0.623 | | 0.530 | 0.589 | 0.653 |
| late (r10) | 0.508 | 0.547 | **0.775** | | **0.425** | 0.549 | **0.766** |
| pooled | 0.534 | 0.552 | 0.729 | | 0.525 | 0.620 | 0.749 |

**Late-trained probe, same-episode, last-step, every layer tested** (late → early [95% CI] / late → late):

| layer | how chosen | all depths (82 mixed tasks) | depth ≥ 2 only (67 mixed tasks) |
|---|---|---|---|
| 12 | stage-2 best (independent) | 0.508 [0.44, 0.57] / 0.775 | 0.538 [0.46, 0.61] / 0.814 |
| 13 | Phase 2 (same data) | 0.522 [0.45, 0.59] / 0.761 | 0.529 [0.45, 0.61] / 0.764 |
| 15 | Phase 2 (same data) | **0.403 [0.34, 0.47]** / 0.776 | **0.398 [0.33, 0.47]** / 0.767 |
| 16 | Phase 2 (same data) | **0.425 [0.37, 0.49]** / 0.766 | 0.447 [0.37, 0.52] / 0.762 |
| 28 | Ruan's (pre-specified) | 0.435 [0.37, 0.51] / 0.802 | 0.451 [0.37, 0.53] / 0.801 |

**What holds, and what doesn't:**
- **Robust, at every layer, both populations, all schemes with enough power: the final-outcome probe is stage-specific.** A late-trained probe reads the late stage well (0.76–0.81) and is at chance or worse on the same episodes' early states (0.40–0.54). An early-trained probe is weak everywhere (~0.48–0.56). Mid → late is 0.62–0.79, below late's own diagonal at most layers. This is Lin's hypothesis (a), a probe trained at one stage fails at another, **supported at the two pre-specified layers (12 and 28), not only at the selected ones**.
- **The below-chance reversal is layer-dependent.** It is significant at L15 in both populations and at L16 across all depths. It is borderline at L28 and absent at L12 and L13. **We should not headline it.** The robust claim is "no transfer", not "sign flip".
- **Excluding depth 1 changes nothing qualitatively** (right-hand column). Depth 1 was 7.5% of the cohort's episodes and 18% of its mixed tasks.
- **Range-averaged scheme** (same-episode, L12 / L16): late → late 0.893 / 0.884, early → late 0.472 / 0.632, late → early 0.505 / 0.500. Same shape, with a stronger late diagonal because it averages 5 late rounds.
- **A single probe pooled over all stages ≈ the per-stage probes** in every scheme (e.g. range-averaged L12: pooled → late 0.897 vs late → late 0.893). Lin's hypothesis (b), that one probe can't cover all stages, is **not supported**.
- **The naive matrix overstates transfer** (L16 last-step early → mid 0.64 vs 0.48 same-episode). Population shift makes early probes look transferable.
- **The milestone scheme is underpowered:** only 8 mixed tasks survive to milestone 3 in the same-episode population, and the AUCs swing 0.14–0.71. No conclusion from it at 1.7B.
- **IMMEDIATE validity/progress probes transfer:** ≤ 0.06 drop at mid layers, more at L28.

**Circularity caveat (layer selection).** Layers 13 / 15 / 16 were selected by Phase 2 on the same 4,264 episodes that Phase 3 analyses. That inflates their absolute AUCs, and **the reversal appears mainly at those selected layers**. Layers 12 (from the independent stage-2 run) and 28 (fixed in advance) were not selected on these data, and both show the robust pattern (late → late 0.78–0.80, late → early 0.44–0.51). **Held-out layer selection before this goes to Lin:** not needed for the robust claim, which the two pre-specified layers already carry. It *would* be needed to claim the below-chance reversal. It's cheap (select layers on half the tasks, evaluate the matrix on the other half, ~10 min of CPU) and worth running if the reversal is going into the story. Otherwise, report the reversal as layer-dependent and exploratory.

## Open questions

1. Is the late → early below-chance reversal real? It's layer-dependent (L15/L16) and needs a held-out layer selection before it can be claimed.
2. Why is the depth-1 advantage confined to round 3? Does it track the recover-or-repeat decision after a failed first attempt?
3. Is "probe ≈ text for final outcome" specific to a 1.7B model that mostly fails on hard tasks, or does it hold at Qwen2.5-7B (Ruan's strongest TextCraft setting)?
4. Does an MC-value label, which varies within an episode, recover a hidden-state advantage that the binary final label can't show?
