# Assumptions — multistep-gap run

Every definition not specified by Shlok, with reason and alternatives. Status tags:
**[confirmed]** checked against code/data/paper (where stated), **[unconfirmed]** could not verify (where I looked), **[pending user]** awaiting an answer.

## Sources and the meeting

**A-1 Transcript is the source of truth; discrepancies with the summary.** [confirmed — read full transcript PDF, 33 pp.]
- Speaker diarisation flips in the first minute: at 00:04 "Speaker A" says "Hi, Professor". From ~00:26 onward B = Shlok, A = Prof. Lin, as stated. Only the first ~20 s are mislabelled.
- Summary "8% accuracy": transcript 11:49, Prof. Lin asks "the probe is getting 8% accuracy?" — a transcription error for ~0.8 AUC, as Shlok said.
- Summary "on harder benchmarks probe performance drops and often aligns with TF-IDF": never run. Transcript 28:51 only says the parent paper had a harder benchmark where *they* did not get good results. Treated as unknown.
- Summary "Minecraft" = TextCraft, "Critique Bank"/"CUB" = CEB (arXiv 2607.12397, Zeng … Lu Lin), "Hira" = Hera (2605.24598), "1,000 rounds" = 1,000 episodes.
- Summary says TF-IDF ≈ 0.6 and probe 0.7–0.8. In the transcript Shlok corrects himself mid-sentence (06:37–06:45: "TF-IDF was scoring 0.7, and I got 0.8"). The log is authoritative: at round 3, text+surface 0.725 vs layer-12 probe 0.801 within-task (`RESEARCH_LOG.md`, showdown). At round 1, layer 28: 0.547 (CUDA) / 0.585 (MPS) within-task.
- Summary's "Probes … fail to generalize across different rounds" is stated as a finding; in the transcript it is Prof. Lin's *hypothesis* (38:59: "that's one assumption I'm having, but I don't know whether that's true"). Treated as hypothesis.
- Transcript 36:08–38:59 adds a detail the summary drops: the comparison is per-stage probes vs **one probe trained on all steps**, and the desired end state is a single probe that handles the shift. Phase 3 includes the pooled probe for this reason.
- Transcript 32:07–32:49: "stage" may be the exact step or "the step at the end of the milestone". Both definitions are used (A-9).

**A-2 CEB's motivation figure is the template for the main figure.** [confirmed — CEB PDF, Fig. 1] x-axis = within-trajectory step depth 1–10, y-axis = ECE (panel a) and AUC (panel b), one line per confidence method (PPL, ME, PRO, P(True), Verbalization, CoT, CEB) on Mind2Web/GPT-5.4. Labels are step-level productivity. So besides AUC I also report **ECE** per test stage; calibration drift is a plausible failure mode that AUC cannot see.

## Data

**A-3 Task identity and ordering** use the existing canonical recipe order (`AGENTS.md`). Depth histogram under canonical order: {1:125, 2:291, 3:117, 4:11}. [confirmed — `TextCraft().fingerprint()` on the Mac]

**A-4 Sampling settings match stage-2**: temperature 0.8, top-p 0.95, top-k off, max_new_tokens 256, thinking off, AgentGym TextCraft prompt. [confirmed — `configs/stage2.json`]

**A-5 max_rounds = 20 for every depth.** [confirmed by pilot] The pilot's success lengths are not bunched at the cap: depth 2 spans 6–18 rounds (11 successes), depth 3 spans 10–15 (4 successes). So the cap stays at 20 for comparability with stage-2 and Ruan. Alternative: 30 rounds for depth ≥ 3. That would add some successes but change the failure definition between depths.

**A-5b Depth 4 is excluded.** [confirmed by pilot] 0/80 successes on 10 of the 11 depth-4 tasks, so it has no mixed outcomes (the brief: "4 if mixed outcomes exist"). Full run = depths 1–3, 533 tasks × 8 = 4264 episodes.

**A-6 Generation backend** [confirmed by Shlok]: vLLM for generation (per-request seed = the same `generation_seed + data_idx*100000 + attempt*100 + round` formula), HF replay for activations (unchanged `Actor.replay` logic, generalised to all layers). Reason: HF batch-1 decode ran 3.75 s/round. Depth 2–4 episodes are longer, and MC labels (A-8) need about 10× more generation. Replay is a separate forward pass over the fixed saved tokens, so vLLM/HF numeric differences cannot contaminate the activations. Check: a depth-1 vLLM pilot must reproduce stage-2's success rate (77.5%) within sampling error. Alternative: keep HF generate (slower, identical to prior run).

**A-6b vLLM validation.** [confirmed on the pilot] On the same 10 depth-1 tasks, stage-2 (HF generate) succeeded 44/80 and vLLM 48/80. Per-task counts: 0/0, 8/8, 1/0, 0/4, 8/8, 3/3, 4/3, 8/8, 8/8, 4/6. The full depth-1 comparison is in STATUS once the run finishes.

**A-7 Probe token site.** The brief says "last-token hidden state of the model's output". Primary site = the last content token of round t's generation. Stage-2/Ruan used the final token of the *action span*. Both are stored per round (they usually differ only by a newline/EOS). Also stored: all layers (embedding output + every decoder block, residual stream).

## Labels

**A-8 Labels.** The env has no per-step reward, so (b) and (c) are proxies.
- (a) FINAL: episode reward (1 iff goal obtained within max_rounds).
- (b) IMMEDIATE-validity: env accepted round t's action (`validity == executable`, i.e. observation starts with Got/Crafted/Inventory). Error, impossible, unparseable and unclassified = 0. Caveat: an `inventory` command counts as valid but makes no progress.
- (c) IMMEDIATE-progress: round t's action *added* an item (Got/Crafted) that is on the goal's recipe tree. The tree is the union over all recipes reachable from the goal down to base items, goal included. Alternative: count only the recipe path the agent actually used; that is circular, since it needs hindsight.
- (d) *Proposal* MC-VALUE: V(τ_t) = mean final reward of N continuations sampled from the same policy starting after round t's observation (Xiong et al. 2024 IPR, as used by 2604.19775). This is the quantity a "probe predicting final reward at step t" is implicitly estimating, and unlike (a) it varies across stages within one episode. Binarised for AUC at V ≥ 0.5, or used continuously (Spearman). Cost ≈ N × remaining rounds per prefix. [confirmed by Shlok: N=8]. **Scope: depth-2 and depth-3 episodes only** (Shlok, 2026-09-23). Depth-1 episodes end by round 2–3, so the stage question barely exists there. MC-label results are therefore depth ≥ 2 by construction, even in the 'all depths' output.
- All labels are attached to the activation at the end of round t's generation, before round t's observation is seen. So (b)/(c) ask "will the action I just wrote succeed/progress", and (a)/(d) ask "will this episode succeed".

**A-9 Stages.**
- Round-based: early = rounds 1–2, mid = 3–5, late = 6+ (as specified).
- Milestone-based (depth ≥ 2): stage k = the round at which the k-th *successful craft* (observation starts with "Crafted") happens. The probe reads the activation at the end of that round's generation. Episodes that never reach milestone k are absent from stage k.

**A-10 Population per round/stage** = episodes alive at that round (Ruan's convention; finished episodes contribute nothing after their last round). [confirmed matches prior pipeline]

**A-11 Metrics.** Pooled AUC (task-grouped cross-fitting) and within-task AUC (pair-weighted over failed-vs-successful rollouts of the same task, mixed-outcome tasks only, same as `analysis.py`), both with task-cluster bootstrap 95% CIs. Plus ECE (10 bins) for the stage-transfer figure.

**A-12 Population-shift control (added; not in brief).** [confirmed need — stage-2 data] At depth 1, successes end at round 2–3 while every failure runs to round 20. The failure share among alive episodes is 22% at t=1, 40% at t=3, 79% at t=6. Mixed tasks fall 57 → 28 (t=6) → 11 (t=10). A transfer matrix built on alive-at-t populations therefore changes *who is being scored* as well as *which stage the activation comes from*. The degradation Prof. Lin hypothesises could appear purely from the population change. Control (**primary analysis**, per Shlok): repeat the transfer matrix on a **fixed cohort**, episodes alive through the last compared stage, so train/test stages differ only in time, not in population. Within-task AUC is also reported, as it removes the task-mix part of the shift.

**A-13 Rule 6 "<15 mixed-outcome tasks" is applied per phase dataset** (e.g. all depth-2 episodes), not per (stage, round) cell. Cells under 15 mixed tasks are still computed but flagged "underpowered" and excluded from conclusions. [confirmed by Shlok]

**A-15 Baseline text excludes round t's observation.** The TF-IDF/surface baselines at round t see the task prompt, responses 1..t, and observations 1..t−1. The activation at round t has not seen observation t, and for IMMEDIATE labels observation t *is* the label ("Could not find …"). The stage-2 showdown's text baseline included observation t. That was fair for FINAL at round 3, but it would leak the answer here. Surface features: current response and prompt length, prior invalid count, prior impossible count, distinct prior actions. There are no logprobs, because vLLM generation doesn't keep them (stage-2's round-1 5-feature version used action logprob).

**A-16 Label polarity.** Every binary label is coded 1 = bad outcome (fail / invalid / no progress / MC value < 0.5). AUC is symmetric; this only fixes the sign of diff-of-means scores.

**A-17 Milestones exclude the terminal craft.** Crafting the goal ends the episode with success. At that round the action text alone determines FINAL, so it is not a "stage" of an unfinished trajectory. Milestone k = k-th craft that is not the goal.

**A-18 Stage evaluation weights.** When a stage contributes several rounds per episode (e.g. a failing episode has up to 15 "late" rounds), each row gets weight 1/(rows of that episode in the stage), so every episode counts once. This applies to pooled AUC, within-task pairs and ECE alike. Training uses all rows unweighted.

**A-19 Cohort definitions (primary control).** Round stages: episodes with ≥ 6 rounds, so every member has early, mid and late rows. Milestone stages: episodes reaching milestone 3. If that leaves < 15 tasks, the cohort is episodes reaching milestone 2 and m3 is dropped from the cohort matrix. Per-round curves: episodes with ≥ t_max = 10 rounds, evaluated at t = 1..10. The cohort is itself a selected population (long episodes: most failures and the slow successes). So cohort results answer "does the same trajectory become harder to read with a probe trained elsewhere", not "what happens to a typical episode".

**A-20 Folds.** One task → fold assignment per run: 5 folds, tasks shuffled within each depth with seed 42 and dealt round-robin. It is shared by every Phase 2 cell, every Phase 3 matrix entry and every baseline. So all transfer comparisons use exactly the same train/test tasks.

**A-21 Phase 4 labels (HTTP envs).** Validity: the env's own invalid-action message: ALFWorld "Nothing happens.", ScienceWorld "No known action matches that input.", BabyAI "The action is not recognized" [confirmed by smoke test on the box]. Progress: ScienceWorld uses the **native** per-step score (score increased this step); that is a real reward, not a proxy, satisfying criterion 2. **BabyAI has no native step signal** [confirmed in code, 2026-09-24]: the AgentGym server never passes `obs_to_reward`, so the reward is MiniGrid's terminal-only completion reward. An earlier version of this entry said BabyAI's progress was native; that was wrong. For BabyAI, progress is a **proxy**, like ALFWorld's: the observation reports a newly carried object ("You are carrying …" changes) or a door that changed state. ALFWorld has none, so progress is a **proxy**: the observation reports a state change ("You pick up / put / clean / heat / cool / turn on / slice"). Milestone k = k-th progress event that does not end the episode, mirroring TextCraft's k-th craft. Success: ALFWorld `won`, ScienceWorld score = 100, BabyAI done with "The task is completed" (the server only sets done when the discounted completion reward exceeds 0.5; the original rule `reward ≥ 1.0` mislabelled every success, fixed and relabelled 2026-09-24). max_rounds 30 and context 16,384 follow the AgentGym protocol used in the HF model cards cited in DATASET_CHOICE.md.

## Related work and framing

**A-22 PAIR (arXiv 2605.17877, Kim, In, Park, Lee, Park; v2 2026-07-31) is related work, not a test of our question.** [confirmed: abstract and full HTML text read 2026-09-24]
- **What they find:** "hidden-state probes degrade severely under prefix contamination tracking coherence with the (possibly corrupted) prefix rather than grounded correctness, while attention-based features remain robust to contamination but underperform on clean prefixes." They combine the two into a step-level reward model (PAIR) for GRPO.
- **Setup:** Qwen2.5-7B-Instruct on GTA (229 samples, 2–8 turns) and ToolBench (1,000 samples, ≥ 3 turns); HotpotQA for transfer. Contamination = one turn replaced with an injected error (reasoning error, tool misuse, observation misinterpretation, stale memory), all other turns unchanged.
- **What they don't test:** cross-stage transfer. There is no experiment that trains a probe on one stage, step index or trajectory position and tests it on another. Their only transfer test is cross-*domain* (train on GTA, apply to HotpotQA). They report pooled AUROC only: no within-task split, no text/TF-IDF baseline.
- **Relation to us:** their result is a mechanism by which a hidden-state probe could fail across stages (later states encode coherence with an earlier, possibly wrong, prefix). **Train on stage A, test on stage B, under a same-episode control, is our specific contribution.** Their contamination is injected; ours is the policy's own naturally occurring trajectory.

**A-23 Our round-by-round results confirm Ruan et al.'s baseline; the stage-transfer matrix is the new result.** [confirmed: Ruan et al. 2607.06503v2 HTML, Fig. 3 and §Per-Round Failure Scorers, read 2026-09-24]
- **What Ruan's Fig. 3 shows:** "Cross-fitted post-generation probe AUC for predicting eventual episode failure at each gate round in the full 2×3 matrix", rounds 1–6, TextCraft and WebShop × {Qwen-2.5-7B, Llama-3.2-3B, Qwen3-1.7B}. TextCraft is already separable at round 1 (AUC 0.86 for Qwen-2.5-7B, 0.81 for Qwen3-1.7B).
- **Ruan trains one probe per round** ("A per-round probe f_r … is trained to predict eventual failure 1−y from episodes alive at round r") and never tests a probe trained at one round on another round.
- **Framing:** our pooled round-by-round numbers (stage-2 round 1: 0.791 vs their 0.81; mstep Phase 2 at each t) **confirm their baseline**. It is not our contribution.
- **What we add beyond it:**
  1. The within-task decomposition, which shows the pooled number is mostly task difficulty, and that the final-outcome probe ≈ text except in a depth-1 / round-3 window (RESULTS §1, §3).
  2. The stage-transfer matrix (train on stage A → test on stage B, naive vs same-episode, round-range and last-step and milestone stages). **That matrix is the new result.**
- The 7B comparison against their 0.86 waits for the Qwen2.5-7B full run (the pilots so far measure only outcome rates).

## Analysis hygiene

**A-14 Grid-search luck.** Every "best" cell is reported with the number of cells searched, the runner-up, and whether the same layer wins at a neighbouring round. Phase 3's layer subset is fixed *before* looking at Phase 3 results: the best 3 layers from Phase 2 by mean within-task AUC across t, plus Ruan's layer 28 (or proportional depth for 8B). Selection uses the same episodes Phase 3 then analyses. This biases the chosen layers' *absolute* AUC upward, but not the *shape* of the transfer matrix, which is the quantity of interest. Ruan's layer is always included as a non-selected reference.

## Comparison with arXiv 2604.19775 (conformal interpretability) [confirmed — read v2 PDF in full]

| | 2604.19775 | This run |
|---|---|---|
| Model | Llama-2-7B, **SFT'd** on expert ReAct trajectories | Qwen3-1.7B / 8B, off-the-shelf |
| Envs | ScienceWorld, ALFWorld | TextCraft + one of those |
| Labels | MC step value (IPR, N continuations), binarised by conformal p-values (ε=0.1); ambiguous steps get **no label** and are dropped | Final + validity/progress proxies + MC value (N=8) |
| Probe | One probe per (layer, t), last token of trajectory up to t | Same, plus cross-stage transfer and pooled probe |
| Metric | Accuracy, F1 | AUC (pooled + within-task), ECE |
| Within-task vs pooled | **Not reported**; no task-difficulty control, no text baseline | Both; TF-IDF + surface baselines |
| Cross-stage transfer | **Not tested**: each P_t is trained and tested at the same t | Core experiment |
| Test size | Small (accuracies step in 1/12–1/20 units ⇒ ~12–20 test items per t) | Hundreds per stage |

Their conformal labelling discards ambiguous middle-value steps, which inflates separability by construction. Their ALFWorld accuracies (60–95%, non-monotone across t) are compatible with the stage-dependence hypothesis, but they never test transfer, so the paper is neither support nor refutation. Their MC step-value label is the most useful idea to borrow (A-8d).
