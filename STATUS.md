# Status — multistep-gap run

## 2026-09-24 22:42 UTC (6:42 PM ET) — Shlok leaving; τ² full finishing unattended

- **ALFWorld:** DONE. Box self-terminated after its W&B upload. Data pulled from the W&B artifact `ms-alfworld-q25-7b-data` into `runs/ms-alfworld-q25-7b` (1,600 episodes) and `results/multistep/alfworld_qwen2.5-7b`. Pre-registered verdicts are in RESULTS §5: **P1 supported, P2 falsified for action-validity (early→later), P3 supported.**
- **τ² full (H100 `afp-tau2-full`):** Phase 3 (52 jobs, layers 16/19/21/28/36; schemes round, laststep, milestone, phase) finishing ~6:50–6:55 PM ET. Then figures → W&B → PIPELINE COMPLETE → `finish` uploads artifact **`ms-tau2-qwen3-8b-data`** → 15-min grace → **self-terminates ~7:15 PM ET.** No Mac needed.
- **NEXT SESSION, first steps:**
  1. Confirm no Lambda instances remain (`scripts/lambda.sh` / API list). If `afp-tau2-full` is still up, check `logs/runner_status.md` on it, pull, then terminate.
  2. Download the artifact `agentic-interp/agent-failure-probes-multistep/ms-tau2-qwen3-8b-data:latest` (wandb.Api().artifact(...).download) into `runs/ms-tau2-qwen3-8b` and `results/multistep/tau2_qwen3-8b/ms-tau2-qwen3-8b`, the same way as ALFWorld.
  3. Score P1–P3 for τ² at the pre-registered layers **16 and 36** (same code as the ALFWorld verdicts), plus the conversation-phase scheme (d). Write RESULTS §6.
  4. Update the RESULTS headline for Lin across all three envs.

### Blocked on me
- Nothing. (MC hold on TextCraft depth 2 is still undecided; the TextCraft box is long gone, so re-extraction would be needed.)

### Changed
- RESULTS §5 (ALFWorld verdicts); ALFWorld data + results on the Mac.

### Found
- P1 and P3 replicate on ALFWorld/Qwen2.5-7B. P2 breaks for action-validity (early-trained probes don't transfer forward; possibly few positives at round 2).

## 2026-09-24 21:58 UTC (5:58 PM ET) — τ² full collected; pre-registration gaps fixed; boxes self-finishing

- **τ² full** (H100 `afp-tau2-full`): 1,312 conversations. Retail 16.2% success, 47/114 mixed; airline 18.2%, 9/50 mixed (56 mixed total). Retail's full τ² reward incl. the local LLM judge is 13.6%, below our deterministic 16.2% (A-25). One-pass extraction ≈ 25 turns/s (incremental prompts, verified prefix 53/53). Analysis starts ~6:00 PM ET, with the phase scheme + prereg layers 16/36 + last-step added.
- **ALFWorld** (A100): Phase 3 (auto layers 10/14/23/28) finishing; **pre-registered Phase 3X on layers 12 and 28** running in parallel (tmux `prereg`).
- **τ² pilot** A100: pulled (160/160) and terminated at 5:16 PM ET. Pilot results are underpowered (8 mixed tasks). Valid-action probe 0.83–0.84 vs text 0.59 at the only powered cell.
- **Self-finish on both boxes** (`scripts/finish_selfterm.sh`, tmux `finish`): when their pipeline (and, for ALFWorld, the prereg 3X marker) completes, they upload episodes/results/logs to the private W&B project, wait 15 min, then terminate themselves. The Mac watchers were stopped so they can't terminate a box before its pre-registered analysis finishes.
- **Lessons recorded:** probes on GPU aren't faster at these sizes (validated equal, 3–9× slower); analysis jobs must default to all cores (fixed in the runners).

### Blocked on me
- Nothing. Results expected: ALFWorld ~6:15–6:30 PM ET; τ² full ~6:45–7:00 PM ET.

### Changed
- `mstep/dataset.py` (τ² phases), `mstep/probes.py` (phase + laststep in Phase 3, prereg layers, GPU path off by default), `mstep/figures.py`, `mstep/rollout.py` (incremental prompts, step_workers), `mstep/extract.py` (--episode-pass), `scripts/finish_selfterm.sh`, `scripts/pull_and_terminate.sh`, `scripts/setup_tau2_env.sh`, `scripts/tau2_summary.py`, runners.

### Found
- τ² full has enough mixed tasks (56) for within-task and stage analysis.

## 2026-09-24 20:15 UTC (4:15 PM ET) — ALFWorld collected; τ² pilot running

- **ALFWorld @ Qwen2.5-7B** (A100 `afp-alfworld`, 129.213.18.165): 1,600/1,600 episodes. **35.1% success, 132/200 mixed tasks**, successes spread over rounds 3–30. Extraction started 4:04 PM ET (~30 min), then Phases 2–3 + W&B automatically (`scripts/run_env.sh`, tmux `alf`). Speed fixes: 8 parallel servers; parallel env creation (serial creation cost ~12 min per 320-episode chunk).
- **τ²-bench pilot** (A100 `afp-tau2`, us-west-2, 129.146.61.181): adapter `mstep/tau2env.py` + `ServerEngine`; smoke test passed (airline task solved, DB 1.0 / COMMUNICATE 1.0). Pilot 20 tasks × 8 launched 4:10 PM ET (`scripts/run_tau2.sh`, tmux `tau`). Decisions: ASSUMPTIONS A-24–A-26.
- **H100:** SSH never came up, even after a restart; terminated by Shlok. A duplicate A100 I had launched was terminated immediately (empty).
- **W&B:** private project https://wandb.ai/agentic-interp/agent-failure-probes-multistep (team `agentic-interp`, whose only member is Shlok). TextCraft results backfilled; new runs log automatically.

### Blocked on me
- Nothing now. After the τ² pilot: decide its full run (gate ≥ 6/20 mixed).

### Changed
- `mstep/tau2env.py`, `mstep/wb.py`, `mstep/rollout.py` (ServerEngine, parallel env creation, N-server sharding, W&B), `mstep/httpenv.py` (port override), `mstep/dataset.py` (τ² labels), `scripts/run_tau2.sh`, `scripts/run_env.sh`, `scripts/env_servers.sh`, configs.

### Found
- ALFWorld @ 7B is a much richer testbed for stages than TextCraft: 66% of tasks mixed vs 28%, with successes throughout the episode.
- Qwen3-8B on τ² hallucinates tool names and arguments (e.g. `get_user_id_by_email` instead of `find_user_id_by_email`): a realistic failure mode.

## 2026-09-24 ~18:35 UTC (2:35 PM ET) — Box 1 terminated; everything pulled

- **Pulled to the Mac and verified against the box before terminating:**
  - `runs/` for all 9 mstep runs (5,645 files; episode counts match: tc 4,264, tc pilot 320, 7B tc pilot 320, 7B sciworld/babyai/alfworld 160 each, 1.7B sciworld/babyai/alfworld 80 each), including `rounds.pkl` and `acts/{index,extraction}.json`.
  - `logs/` (14 files) plus `logs/box_home/` (env-server and preflight logs).
  - `results/multistep/` (94 files).
- **Not pulled:** `runs/ms-tc-1p7b/acts/` (39 GB of activations). They are rebuildable from the saved tokens with `mstep.extract`, about 1 h on an A100.
- `afp-multistep-2` terminated at Shlok's request; the GH200 helper had already terminated itself. **No Lambda instances running.** Total ≈ 5.8 box-hours this run (+2.2 earlier).
- **To resume:** new box → `scripts/preflight_box.sh` + `preflight_box2.sh` → rsync `runs/` up → re-extract (`mstep.extract`) only if MC or new probes are needed.

### Blocked on me
- Phase 4 env (recommend ALFWorld @ 7B), MC on depth 2 (yes/no), held-out layer selection (only if the reversal is claimed).

### Changed
- Nothing in code.

### Found
- Nothing new.

## 2026-09-24 18:25 UTC (2:25 PM ET) — All runs done; results written

- Everything finished by 1:23 PM ET. The GH200 helper terminated itself (verified: only afp-multistep-2 remains). Box 1 has been idle on the MC hold since then. Box-hours 5.7.
- **Corrected Phase 4 gate (7B):** ScienceWorld 4/20 (fail), ALFWorld 14/20, BabyAI 16/20, TextCraft d2–3 14/40. Recommendation: ALFWorld @ 7B (DATASET_CHOICE.md).
- **Phase 3 (RESULTS §4):** the final-outcome probe is stage-specific at every layer, including the pre-specified L12 and L28, and with depth 1 excluded. The below-chance late → early reversal is layer-dependent (L15, L16), so it is not headlined. A pooled probe ≈ per-stage probes. The milestone scheme is underpowered (8 mixed tasks).

### Blocked on me
- MC release (box idling at $1.99/h); Phase 4 env (recommend ALFWorld @ 7B); held-out layer selection (only needed if the reversal is claimed); terminate box 1 if nothing else runs.

### Changed
- RESULTS §4 rewritten with L12 vs L16, depth ≥ 2, and the circularity caveat; DATASET_CHOICE 7B table.

### Found
- The stage-specific final-outcome probe holds at pre-specified layers; the reversal does not.

## 2026-09-24 16:57 UTC (12:57 PM ET) — CPU helper box running depth >= 2 analyses; Shlok away for lunch

- **Helper box:** `afp-cpu-2` (GH200, 64 cores, ARM, us-east-3, 192.222.57.216, $2.29/h), id in `.lambda_instance_id_cpu2`. It runs `scripts/cpu2_run.sh`: depth ≥ 2 Phase 3 (range-averaged + milestone + per-round curves) and Phase 3X (layers 12, 13, 15, 16, 28 × all three schemes) in parallel, 30 workers each. Started 12:54 PM ET.
- **Box 1 tmux `to2`** (`scripts/cpu2_orchestrate.sh`) stopped box 1's depth ≥ 2 jobs (`fast` lane B, `d2x`), holds depth_ge2's lock so tc2 won't redo them, and copied the data. When the helper finishes, it pulls the results back, writes DONE_all, draws the figures, then has the helper **terminate itself**. The Lambda key is stored only on the helper, mode 600, and dies with it.
- **Still on box 1:** tc2 (all-depths Phase 3, averaged), `l12` (layers 12 vs 16), `p7b2` (resuming the crashed ScienceWorld/BabyAI 7B pilots → CORRECTED gate summary). MC on hold.
- **ALFWorld 7B pilot:** 66/160 success, **14/20 mixed** → passes the Phase 4 gate (1.7B: 0/80).
- ETA: everything by ~1:30–2:00 PM ET. Box 1 then idles on the MC hold and bills until Shlok terminates it or releases MC.

### Blocked on me
- MC release; Phase 4 env choice (ALFWorld passes at 7B; ScienceWorld/BabyAI pending the reruns); terminate box 1 when done.

### Changed
- `scripts/cpu2_run.sh`, `scripts/cpu2_orchestrate.sh`; box-1 key `~/.ssh/box2_ed25519` (box 1 → helper only).

### Found
- ALFWorld becomes viable at 7B (14/20 mixed).

## 2026-09-24 16:15 UTC (12:15 PM ET) — Phase 3 last-step results (all depths)

`phase3_transfer_laststep.csv` (done 12:06 PM ET). Headline, same-episode, FINAL:
- late → late 0.77–0.80; late → early 0.40–0.52, significantly below chance at L15/L16 (sign flip).
- early states are weak on their own (0.54).
- mid → late drops ~0.1.
- a pooled probe ≈ per-stage probes.
- the naive matrix overstates early → later transfer (0.64/0.62 vs 0.48/0.48 same-episode).

Immediate validity/progress: ≤ 0.06 transfer drop at L16, larger at L28. Written into RESULTS §4.

### Blocked on me
- MC release; Phase 4 / 7B env choice after the corrected gate summary.

### Changed
- RESULTS §4 (last-step version).

### Found
- Lin's hypothesis (a) — a probe trained at one stage fails at another — is supported for the final-outcome label, strongest late → early, where it reverses sign. Hypothesis (b) — one pooled probe can't cover all stages — is not supported. The immediate-validity probe is stage-robust at mid layers.

## 2026-09-24 15:57 UTC — Last-step Phase 3 variant + parallel analysis lanes

- **Last-step stage scheme (Shlok-approved boundaries):** early 1–2 → round 2, mid 3–5 → round 5, late 6–10 → round 10 (late capped at 10: an open-ended range has no last step, and an episode's final round leaks the outcome). Same-episode cohort = episodes with ≥ 10 rounds. `mstep/probes.py --phase 3L` → `phase3_transfer_laststep.csv`; same layers as Phase 3. Unit-checked on a toy table. **Both versions are kept:** the average-over-range version (tc2, running since 15:45) and the last-step version.
- **Speed-ups:**
  - tmux `fast`: lane A runs 3L on all depths now, then 3L on depth ≥ 2. Lane B runs depth ≥ 2 Phase 2+3 now, instead of after tc2's all-depths Phase 3. All 30 cores are busy (load ~32).
  - `probes.py` got a lock and a `DONE_<phase>` marker, so tc2 skips depth ≥ 2 when it reaches that step.
  - `logs/gpu_util` = 0.85, so MC takes the whole GPU once released. Don't release it while a 7B pilot is running.
- **7B pilots:** the ScienceWorld (1/160 episodes) and BabyAI (97/160) pilots crashed on a keep-alive bug: uvicorn drops idle connections after 5 s, and a 7B generation round takes longer. Fixed in `httpenv.py` with a fresh connection per request and no retry, since a retried step could run twice. ALFWorld started after the fix. `p7b2` resumes the crashed pilots and writes a CORRECTED PHASE 4 GATE SUMMARY. The first summary's "ScienceWorld 0/20 mixed" is invalid.

### Blocked on me
- MC release; Phase 4 env and 7B full-run env after the corrected gate summary.

### Changed
- `mstep/probes.py` (laststep scheme, `--phase 3L`, lock/DONE), `mstep/figures.py` (last-step heatmaps), `mstep/httpenv.py` (Connection: close), `scripts/run_fast.sh`, `scripts/pilots_7b_rerun.sh`.

### Found
- 7B TextCraft pilot: depth 2 has 27.5% success and 11/20 mixed tasks (1.7B: 9.7%); depth 3 has 3.1% and 3/20.

## 2026-09-24 15:45 UTC — Depth-1 replication: stage-2 result holds, but only at round 3

`scripts/run_depth1_replication.sh` (tmux d1repl): depth 1 only, action-token site, Phase 2, full grid. Results in `results/multistep/textcraft_qwen3-1.7b/depth_eq1_action/`.

- **The pre-registered cell (round 3, layer 12) scores 0.815 [0.75, 0.87] within-task (pooled 0.856), vs text 0.717.** Stage-2 had 0.801 vs 0.725, so it replicates.
- **At round 3 the advantage is broad:** 71% of layers 1–28 beat text by more than 0.05 (best 0.834 at L18).
- **It is round-specific:** the probe ties or loses to text at t = 1, 2, 5 and 8 (0%, 0%, 4%, 0% of layers beat text by more than 0.05).
- RESULTS.md headline revised: the recipe ≈ text for the final outcome, except in a depth-1 / round-3 window, and that round-specificity is itself early stage evidence.

### Blocked on me
- MC release; Phase 4 env and 7B full-run env (after the 7B pilots).

### Changed
- `mstep/probes.py`: `depth==1` subset (commit 7c61aa4). RESULTS.md written (interim).

### Found
- Stage-2 layer-12 / round-3 result replicates (0.815 vs 0.801), is broad across layers at round 3, and is absent at every other round.

## 2026-09-24 15:30 UTC — FIRST REAL PHASE 2 RESULTS (1.7B, TextCraft depths 1-3, 4,264 episodes)

Found on the box: the 2026-09-23 22:11-22:18 `probes_nomc` run **completed Phase 2 on the full dataset** before that box was terminated (config.json: 69,782 rows, 4,264 episodes, 533 tasks, 29 layer slots). Phase 3 had only reached `phase3_layers.json`. Files pulled to the Mac. These are the first non-synthetic probe numbers of the run. MC labels absent (hold), so 3 labels x 580 cells each.

### FINAL label (does this episode fail): the probe never beats text

Within-task AUC, S1 last-token, L2 logistic, best layer per round, vs TF-IDF+surface at the same round:

| round t | probe (best of 29 layers) | TF-IDF+surface | probe - text |
|---|---|---|---|
| 1 | 0.585 (L13) | 0.569 | +0.016 |
| 2 | 0.627 (L15) | 0.605 | +0.022 |
| 3 | 0.639 (L20) | 0.639 | 0.000 |
| 5 | 0.745 (L15/16) | 0.732 | +0.013 |
| 8 | 0.819 (L16) | 0.825 | **-0.006** |

S2 (mean-pool rounds 1..t) is slightly better than S1: 0.660 at t=3, 0.843 at t=8 — its t=8 edge over text is +0.018. Diff-of-means is uniformly worse than logistic.

**Not yet comparable to the stage-2 headline, so no replication claim either way.** Stage-2 (depth 1 only, 57 mixed tasks, round 3, layer 12) had 0.801 within-task vs 0.725 text. These numbers pool depths 1-3 (149 mixed tasks, most of them failing depth-2/3 episodes), so the population differs. The text baseline also differs: the stage-2 one *included* round t's observation, this one excludes it (A-15). The pooled-depth probe is far below 0.801 at t=3 (0.639), but a depth-1-only rerun is the fair test and has not been run. What *is* established: across the pooled population, the probe is within ±0.02 of text at every round. **580 cells were searched per label**, so 0.819 is an exploratory maximum.

The layer curve is flat-to-broad (0.75-0.82 across layers 9-28 at t=8), not the sharp mid-network peak the stage-2 note described.

### IMMEDIATE labels: internals beat text by a wide margin

| label | best within-task AUC | TF-IDF+surface | layer 0 (embeddings only) |
|---|---|---|---|
| action invalid (t=3) | **0.990** (L15) | 0.840 | 0.875 |
| no recipe progress (t=5) | **0.982** (L16) | 0.779 | ~0.81 |
| action invalid (t=1) | 0.970 (L16) | 0.826 | 0.537 |

The model's residual stream at the end of its own action almost perfectly encodes whether the environment will accept that action (+0.15 over text). **Caveat:** at t>=3 the *embedding* layer alone reaches 0.84-0.88, so much of this is lexical (which item string was just emitted); the depth-16 gain over layer 0 is ~0.11. At t=1, layer 0 is at chance (0.537) and layer 16 is 0.970, so there the signal is genuinely computed, not lexical.

### The shape of the answer to Lin's question

Sharpening: the naive single-step recipe is *excellent* for an immediate-step label and *no better than TF-IDF* for the long-horizon final-outcome label, at every round we measured. The stage-transfer matrices (Phase 3) will say whether the immediate-label probe is also stage-stable, which is the version of the hypothesis that can still come out either way.

### Survivorship (why the cohort control is primary)

`population_by_round.csv`: failure share among alive episodes 0.764 (t=1) -> 0.845 (t=3) -> 0.948 (t=8); mixed tasks 149 -> 108. The FINAL-label rise from 0.585 to 0.819 is therefore partly a population change, not only more evidence per episode. Cohort membership share (episodes surviving to t=10) is 0.79-0.98, so the cohort matrix will be well-powered here.

### Box state
Extraction on the new box was at 56k/69,782 rows at 15:24 (ETA ~15:32). `tc2` will then re-run dataset/probes/figures, reproducing the above **and** adding Phase 3 (both transfer matrices, both stage schemes, layers [13, 15, 16, 28]) plus the figures. `p7b` follows with the 7B pilots. MC still on hold. Box-hours 2.78; watchdog alive.

### Blocked on me
- Release MC (hold) - note the FINAL-label result weakens the case for MC: if the hidden state does not beat text on final reward, an MC-value label mainly re-measures the same quantity with a softer target. Still worth it as the label Lin's framing implies, but it is now the most droppable 6-10 box-hours.
- Phase 4 env + 7B full-run env after the pilot summary.

### Changed
- Pulled `results/multistep/` from the box to the Mac; no code changes.

### Found
- FINAL-label probe ~= text baseline (±0.02) at every round on the pooled depth-1-3 population. Whether stage-2's depth-1 layer-12/round-3 advantage replicates needs a depth-1-only rerun (not yet done).
- IMMEDIATE-label probe >> text (0.99 vs 0.84), partly lexical at later rounds, genuinely computed at t=1.
- Survivorship confirmed large enough to matter; cohort control well-powered.

## 2026-09-24 15:10 UTC — 7B pilot plan v2 (Shlok)

- TextCraft 7B pilot = **20 tasks per depth** (depths 2, 3) × 8. ScienceWorld 20 × 8.
- If the 7B ScienceWorld pilot has **< 6/20 mixed tasks**, `p7b` automatically pilots BabyAI and ALFWorld at 7B (20 × 8 each; ALFWorld with per-chunk server restarts). The last line is a `PHASE 4 GATE SUMMARY` covering all envs.
- How it was applied: the old `p7b` was only a waiting loop (no job launched, verified with pgrep), so it was stopped; `scripts/pilots_7b.sh` was replaced and `p7b` relaunched at 15:09. Nothing else on the box was touched. `tc2` is still extracting (30k/70k rows); MC hold unchanged. Commit f49bfce.

### Blocked on me
- Release MC; Phase 4 env and 7B full-run env after the pilot summary.

### Changed
- `scripts/pilots_7b.sh` (v2), `configs/ms_tc_q25_7b.json` (tasks_per_depth 20), new `configs/ms_{babyai,alfworld}_q25_7b.json`.

### Found
- Nothing new.

## 2026-09-24 15:12 UTC — Plan change (Shlok): Qwen2.5-7B, MC hold, depth-2 MC; handoff written

**Plan change, applied on the box with no session needed:**
- Phase 5 model: **Qwen2.5-7B-Instruct** @ a09a3545 (28 layers, d=3584, 4 KV heads, 15.2 GB bf16), replacing Qwen3-8B.
- **MC gate:** `mstep.rollout mc` reads `runs/ms-tc-1p7b/mc_scope.json`. It is now `{"min_depth": 2, "max_depth": 2, "hold": true}`, so tc2's MC step parks before loading any model. Gate tested on a scratch copy: it parked, with no GPU process started. Release = set `"hold": false`. MC scope is **depth 2 only** (depth 3 has 8 mixed tasks).
- `tc2` itself was **not** edited or restarted (bash reads running scripts incrementally). After extraction it starts MC (parks) and, in parallel, the no-MC analysis = 1.7B Phases 2–3 on CPU.
- New tmux `p7b` (`scripts/pilots_7b.sh`): after the 1.7B extraction finishes and no other collector runs, it checks the 7B fit, then runs pilots: TextCraft depths 2–3 (20 tasks: 10/depth × 8) and ScienceWorld (20 × 8), vLLM util 0.85. **These pilots decide Phase 4.**
- 1.7B ScienceWorld pilot finished (comparison only): **1/80 success, 1/10 mixed**, 73 round-limit and 6 env-done failures.
- Watchdog cap 40 unchanged (offset 2.2 h from the earlier boxes).

**State:** extraction 26k/70k rows at 15:07 (ETA ~15:25). Box-hours 2.53 at 15:05.

### Blocked on me
- Release MC (hold) after the 7B pilots; choose the Phase 4 env and the 7B full-run env (HANDOFF §5).

### Changed
- `mstep/rollout.py` (MC hold gate + max_depth), `configs/ms_tc_q25_7b.json`, `configs/ms_sciworld_q25_7b.json`, `scripts/pilots_7b.sh`; HANDOFF.md; CLAUDE.md first line; box: `mc_scope.json` (previous kept as `mc_scope.prev.json`).

### Found
- ScienceWorld fails the mixed-outcome gate at 1.7B (1/10), consistent with the published 0.12/100 mean score.

## 2026-09-24 — Episode reconciliation (before any launch)

`runs/ms-tc-1p7b` on the Mac: **4,264 episode files = 533 tasks × 8**. Depth 1: 1,000 (125×8), depth 2: 2,328 (291×8), depth 3: 936 (117×8). All carry the manifest hash; 0 duplicates, 0 unknown tasks, every task has exactly 8 attempts; `progress.jsonl` has 4,264 unique lines.

**The "2,281" in the 21:10 and 22:15 entries** was the count at the 21:04 pause. File mtimes (preserved by rsync) split 2,281 before the 21:05 resume and 1,983 after, and 2,281 + 1,983 = 4,264. The resumed collection finished before the final pull. The 22:15 entry's "all 4,264" is correct; the 21:10 entry's 2,281 is a snapshot, not the final count.

| Depth | Success | Mixed tasks | All-fail tasks |
|---|---|---|---|
| 1 | 76.6% (stage-2 HF: 77.5% → vLLM validated at full scale, A-6b) | 49/125 | 9 |
| 2 | 9.7% | 92/291 | 199 |
| 3 | 1.4% | **8/117** | 109 |

Depth 3 alone is under the 15-mixed-task floor, but depth ≥ 2 has 100 mixed tasks, so the phase dataset passes rule 6 (A-13). **Flag for Shlok:** MC on depth 3 costs ~30% of the MC budget but adds almost no within-task signal (in all-fail tasks V ≈ 0 everywhere). It is kept, per instruction; dropping it is the first thing to cut if the budget gets tight.

## 2026-09-23 ~22:15 UTC — Box terminated (Shlok home); state for next session

**Pulled to the Mac first:** `runs/ms-tc-1p7b` (all 4,264 episodes, 1.2 GB), `runs/ms-{alfworld,babyai}-1p7b-pilot` (80 each), `runs/ms-tc-1p7b-pilot`, `logs/`, `results/`. The partial activations (`acts/`) were *not* pulled: extraction was mid-way, and they're rebuildable from the saved tokens. Instance `afp-multistep` terminated. **Box-hours used: ~2.2 (≈ $4.40).**

**Where each piece stands**
- TextCraft 1.7B Phase 1: collection complete (4,264/4,264). Still to do: extract (≈ 1 h), then MC (depth 2–3) in parallel with analysis. Relaunch: new box → `scripts/preflight_box.sh` + vLLM cu129 fixes (see 20:25 entry) → push `runs/ms-tc-1p7b` back up → `bash scripts/run_tc2.sh ...`. Collection resumes as a no-op.
- Phase 4 pilots:
  - **ALFWorld 0/80.** Looks genuine: the model paraphrases commands (`put mug 2 in fridge 1` vs the required `put mug 2 in/on fridge 1`), so 37% of actions are invalid. The published 19.4% used temp 0.4 vs our 0.8.
  - **BabyAI: 33/80 are real successes, mislabelled as 0.** BUG: the env reward is step-discounted (0.84–0.99 at "The task is completed"), and `httpenv` required ≥ 1.0. Fix: success = done and "The task is completed" in the observation. Then relabel the pilot episodes offline (no rerun needed) and recount mixed tasks.
  - **ScienceWorld: not run.** The server hit `Too many open files` / connection resets when creating 80 envs at once. Fix: `ulimit -n 65536` in `env_servers.sh` start, smaller chunks (≈ 40), `restart_server_every_chunk`.
- DATASET_CHOICE.md: fill in the pilot columns after the two fixes; BabyAI is currently the front-runner.

### Blocked on me
- Nothing. Next session needs a new Lambda box (same key `afp-lambda-key`).

### Changed
- Nothing in code since 5a799c0. Data now local under `runs/` (gitignored).

### Found
- Base Qwen3-1.7B at temp 0.8 cannot use ALFWorld's exact action strings; BabyAI gives plenty of mixed outcomes.

## 2026-09-23 21:10 UTC — Paused, resumed, pilot fixes

- 21:04: Shlok asked to kill the box. I stopped all jobs and pulled the episodes to the Mac (`runs/ms-tc-1p7b` 2,281 episodes, `runs/ms-tc-1p7b-pilot`, `logs/`; 630 MB). The box was **not** terminated. Shlok then chose to keep it running and will stop it himself later.
- 21:05: relaunched as `tc2` (`scripts/run_tc2.sh`, resumes collection: 1,983 episodes left) plus `envpilots`. The handoff session is no longer needed; tc17 is gone.
- The Phase 4 pilots failed at first for two reasons, both fixed (commit 5a799c0):
  1. `httpenv.conversation_start` evaluated *every* prompt in the client file, and the FUNCTION_CALLING prompts are f-strings. It now evaluates only the first two (ReAct). This path had not been covered by the earlier smoke test.
  2. A vLLM startup race: the pilot engine initialised at the same moment as tc2's, and vLLM's memory profiling counted the other process's allocations, leaving "no memory for cache blocks". The pilots now wait 180 s plus free memory ≥ 20 GB, and retry 3×.
- Collection now runs at gpu_util 0.50 (the `logs/gpu_util` override applies to every engine start), so it is somewhat slower than before, and the pilots share the GPU.

### Blocked on me
- Nothing. Shlok will terminate the box himself (`scripts/lambda.sh terminate` from the Mac, or the Lambda dashboard). Watchdog cap is still 40 box-hours.

### Changed
- `mstep/httpenv.py` (prompt extraction), `scripts/run_env_pilots.sh` (delay + retries).

### Found
- Nothing new scientifically.

## 2026-09-23 21:05 UTC — Budget re-plan (Shlok: MC on depth 2–3 only)

**MC scope:** `runs/ms-tc-1p7b/mc_scope.json` = `{"min_depth": 2}`, read by `mstep.rollout mc`. That's 2,328 depth-2 + 936 depth-3 episodes; depth-1 rows get no MC label (ASSUMPTIONS A-8d).

**Measured so far:** collection runs at 39.7 rounds/s with 1,088 concurrent episodes. It is prefill-bound: KV cache ≈ 270k tokens is far below 1,088 × ~1.5k-token histories, so every round re-prefills. Prompt construction is only 1.4 ms/state (profiled). MC should run faster than that rate, because the 8 samples of a prefix share its whole history (prefix-cache hits). So the MC figure below is an upper bound, to be re-measured after the first MC chunk.

**Scheduling fix:** v1 ran the multi-hour CPU analysis *before* MC, which would leave the GPU idle for hours of billed time. `scripts/handoff_tc.sh` (tmux `handoff`) swaps tc17 for `scripts/run_tc2.sh` (tmux `tc2`) as soon as collection finishes. v2 runs extract, then MC ‖ no-MC analysis in parallel, then with-MC analysis. All steps resume, so nothing is lost.

**Projection (box-hours = instance wall-clock; cap 40):**

| Block | Worst case | Likely | Notes |
|---|---|---|---|
| Used so far | 0.9 | 0.9 | incl. first instance |
| TextCraft 1.7B: finish collect + extract | 1.3 | 1.0 | extraction ≈ 72k rows × ~1.7k tokens |
| TextCraft MC (depth 2–3) ‖ no-MC analysis | 15 | 6–8 | upper bound = 2.2M continuation rounds at the collection rate |
| TextCraft with-MC analysis (CPU) | 4 | 3 | 2 probe passes (all depths, depth ≥ 2) |
| Phase 4 pilots + full Phases 1–3 (1.7B) | 2 | 1 | overlapped with TextCraft MC on the shared GPU (0.40 / 0.50); no MC for Phase 4 unless budget allows |
| Qwen3-8B Phases 1–3, one env, no MC | 7.5 | 6 | needs the whole GPU, so it runs after TextCraft MC; MC at 8B ≈ 3× the 1.7B cost, which doesn't fit |
| **Total** | **≈ 31** | **≈ 19–21** | |

**Decision:** the projection fits under 40 box-hours, so nothing is dropped now. Standing rule (Shlok's priority order): after the first MC chunk, and at every later checkpoint, re-project. If the projected total exceeds 38 h, drop from the bottom: Qwen3-8B first, then Phase 4's full run (keeping its pilot), never the TextCraft Phase 3 same-episode control. Log what was dropped and why here. Pre-emptively dropped: MC labels for Qwen3-8B and (unless budget remains) for the Phase 4 env, because each would cost ≥ 10 box-hours.

**ALFWorld server restarts:** built in before any full run. `restart_server_every_chunk: true` in `configs/ms_alfworld_1p7b.json`. `mstep.rollout` calls `scripts/env_servers.sh restart alfworld` after every 256-episode chunk (collect and MC), and episodes resume from disk.

### Blocked on me
- Nothing.

### Changed
- `mstep/rollout.py`: MC scope file, server-restart hook. `scripts/env_servers.sh`: `restart <env>`. New `scripts/run_tc2.sh`, `scripts/handoff_tc.sh`, `scripts/mc_cost.py`. `pitch.*` and `AGENTS.md` untouched.

### Found
- The rollout engine is prefill-bound at high concurrency, not CPU-bound. Depth-2 episodes average ~19 rounds (22,758 rounds / 1,176 episodes), so almost all run to the cap.

## 2026-09-23 ~21:00 UTC — Parallel validation done; Phase 4 pilots queued

**Ran**
- Analysis code (dataset → probes → figures) validated end to end on the box against synthetic activations over the 320 real pilot episodes, for all depths and depth ≥ 2. Output: 2,320 Phase 2 cells, naive + cohort transfer matrices for both stage schemes, 30 PNGs. One cosmetic figure bug fixed. Expect each full-scale `probes` call to take ≥ 1 h of CPU (70k rows × d=2048).
- `mstep/httpenv.py` validated on ALFWorld, ScienceWorld and BabyAI (9 tasks, invalid-action detection, native scores, 64-way concurrency OK; RAM fine).
- Phase 4 pilots queued: tmux `envpilots` (`scripts/run_env_pilots.sh`: ALFWorld, BabyAI, ScienceWorld, 10 tasks × 8 each). It starts automatically when ≥ 20 GB of GPU is free. It uses vLLM at 0.40, and TextCraft MC will use 0.50 via `logs/gpu_util`, so the two share the A100.
- TextCraft full run (`tc17`): 1,088/4,264 episodes collected at last check.
- Local commits `d57c191`, plus the Phase 4 commit. Nothing pushed.

**Known issues:** the ALFWorld server has no `/close`, so envs leak. That's fine for the pilot; the full run needs server restarts between chunks. The env servers are the preflight nohup processes, not supervised.

**GPU hours used:** ~0.9 box-hours.

### Blocked on me
- Nothing.

### Changed
- `mstep/httpenv.py`, `configs/ms_{alfworld,sciworld,babyai}_1p7b.json`, `scripts/run_env_pilots.sh`, `scripts/env_servers.sh`, `scripts/httpenv_smoke.py`, `DATASET_CHOICE.md` (draft), GPU-share override in `mstep/rollout.py`.

### Found
- Published ScienceWorld evidence (0.12/100 for base 1.7B) makes ALFWorld or BabyAI the likely Phase 4 env; the pilot decides (DATASET_CHOICE.md).

## 2026-09-23 20:47 UTC — Phase 1 full TextCraft run launched (unattended)

**Ran**
- Pilot `runs/ms-tc-1p7b-pilot` (on box): 10 tasks/depth × 8. Collection took 111 s for 320 episodes.
  - Success: depth 1 60% (4/10 mixed), depth 2 14% (5/10 mixed), depth 3 5% (2/10 mixed), depth 4 0%.
  - vLLM check: the same 10 depth-1 tasks succeeded 48/80 vs 44/80 under stage-2 HF, and per-task counts match closely (A-6b).
  - Depth 4 dropped (A-5b); max_rounds stays 20 (A-5).
- Pilot MC was stopped before finishing. It was much slower than estimated: a failing 20-round episode needs ~650 continuation rounds. Full-scale MC ≈ 8 h. That is within the 40 h cap but above my earlier 3–5 h estimate.
- **Full run launched**: tmux `tc17` on the box, `scripts/run_tc.sh configs/ms_tc_1p7b.json runs/ms-tc-1p7b results/multistep/textcraft_qwen3-1.7b`. 533 tasks × 8 = 4264 episodes.
  - Order: collect → extract (all layers, 2 sites) → dataset/probes/figures without MC (all depths + depth≥2) → MC → dataset/probes/figures with MC.
  - GPU steps are fatal on error; analysis steps are soft (logged, pipeline continues). Progress: `logs/runner_status.md`, detail: `logs/ms-tc-1p7b.log`.
- Budget watchdog: tmux `watchdog`, kills GPU jobs at 39.5 box-hours (`logs/budget.txt`).
- Local commit `d57c191` on `multistep-gap` (code only; meeting-derived docs are gitignored). Nothing pushed.

**Untested**: `mstep.dataset`, `mstep.probes`, `mstep.figures` have not run on real data yet. They first run inside `tc17` after extraction. If they fail, the GPU steps still complete; fix the code and re-run `analyze` by hand. `mstep/httpenv.py` (Phase 4 envs) is written but not yet run against the servers.

**GPU hours used:** ~0.7 box-hours at launch.

**Next (per TASKS.md):** check `tc17` → validate analysis outputs → Phase 4 candidate pilots (start env servers from `~/envs/<name>`, configs with AgentEval test indices, context_limit 16384, max_rounds 30) → DATASET_CHOICE.md.

### Blocked on me
- Nothing.

### Changed
- `mstep/` package (rollout, extract, dataset, metrics, probes, figures, httpenv), `scripts/run_tc.sh`, `scripts/budget_watchdog.sh`, `scripts/ms_summary.py`, `configs/ms_tc_1p7b.json` (depths 1–3).
- `.gitignore`: meeting-derived docs, logs, results/multistep.

### Found
- Harder depths do give mixed outcomes, with much longer successes (depth 2: 6–18 rounds). So depth ≥ 2 is where the survivorship confound is weakest and the stage question is best posed.
- Base Qwen3-1.7B essentially cannot do depth 4 (0/80).

## 2026-09-23 — Preflight answers (all items resolved)

| # | Item | Answer |
|---|---|---|
| SSH | passphrase key | keep using unencrypted `lambda_key` (`afp-lambda-key`) |
| 1 | GPU cap | **40 box-hours** (all instance wall-clock) — stop and log if exceeded |
| 2 | MC value label | **yes**, N=8, prefixes at rounds {1,2,3,5,8} + milestones |
| 3 | vLLM generation | **yes**, validated against stage-2 depth-1 success rate |
| 4 | Publishing | **nothing pushed anywhere**. Local commits only; nothing leaves the box except the end-of-run pull to the Mac. No GitHub repo created |
| 5 | Teardown | pull episodes/tables/figures to the Mac, then terminate |
| 6 | Rule 6 scope | per phase dataset; cells <15 mixed tasks flagged underpowered |
| + | Task set | all 544 tasks × 8 rollouts |
| + | Same-episode (cohort) stage control | **primary analysis** in RESULTS.md, side by side with the naive transfer matrix |


## 2026-09-23 20:25 UTC — Phase -1 preflight done, waiting for answers

**Ran**
- Read the transcript (source of truth), the summary, arXiv 2604.19775 (full), and CEB Fig. 1. Discrepancies are in ASSUMPTIONS A-1, the paper comparison at the end of ASSUMPTIONS.
- Lambda box `afp-multistep` (A100-SXM4-40GB, us-east-1, $1.99/h, 497 GB disk, 430 GB free), id in `.lambda_instance_id`.
- Box preflight (`scripts/preflight_box.sh`, `preflight_box2.sh`, logs `~/preflight*.log` on the box):

| Item | Result |
|---|---|
| Main venv (py3.13, torch 2.11 cu128) | OK, 23/23 tests pass on CUDA |
| Qwen3-1.7B @ 70d244cc (stage-2 pin) | downloaded |
| Qwen3-8B @ b968826d | downloaded; HF bf16 gen + 8k-token all-layer forward = **18.2 GiB peak** → fits |
| vLLM 0.30.0 (cu129 build; cu130 default needs newer driver than the box's 12.8) | OK. **9.2k tok/s** on 1.7B, per-request seeds reproduce exactly |
| ScienceWorld (AgentGym server, py3.10, OpenJDK 17) | OK, reset + step |
| ALFWorld 0.3.3 (+2.3 GB data) | OK, reset + step, 200 eval games present |
| BabyAI (minigrid<2.5) | OK, reset + step |
| HF token (read) | valid; neither model gated |
| GitHub | `gh` logged in as shlok1808 with repo scope; **repo is public** |

**Failed / fixed along the way**
- `~/.ssh/lambda_final` is passphrase-protected and no agent held it. I registered your existing unencrypted `~/.ssh/lambda_key.pub` on Lambda as `afp-lambda-key` and relaunched. The first instance was terminated empty after ~5 min.
- vLLM default wheel (cu130) refused the 12.8 driver. Fixed by using the cu129 build + matching torchvision, removing torchcodec, and installing ninja.

**GPU hours used:** ~0.2 (instance wall-clock ~15 min so far, including the first instance).

**Phase 0 (done on Mac, no GPU)**
- TextCraft reward is sparse: `reward = 1` only when the goal item is crafted/got (`vendor/.../agentenv_textcraft/environment.py` step, lines 60/77). Everything else 0.
- Layers from `config.json`: Qwen3-1.7B 28 (d=2048), Qwen3-8B 36 (d=4096).
- Depth histogram (canonical order): {1:125, 2:291, 3:117, 4:11}.
- Disk, last-token residual at every layer (incl. embeddings), fp16, 2 token sites/round: 1.7B 29×2048×2×2 = 232 KB/round, 8B 37×4096×2×2 = 606 KB/round. All 544 tasks × 8 × ~15 rounds ≈ 65k rounds → **15 GB (1.7B), 40 GB (8B)**. Fits the box (430 GB free). Decision: **store every layer**. The Mac has only 32 GB free, so activations stay on the box; only tables/figures/episodes come back.
- Survivorship in stage-2 depth-1 data: failure share among alive episodes is 22% (t=1), 40% (t=3), 79% (t=6). Mixed tasks 57 → 28 (t=6) → 11 (t=10). This motivates the population-shift control (ASSUMPTIONS A-12).
- External evidence for Phase 4: on AgentGym's protocol, base Qwen3-1.7B has ScienceWorld mean score **0.12/100** and ALFWorld success **19.4%** (HF model cards by SeanWang0027, temp 0.4, 30 rounds). ScienceWorld likely fails criterion 3 at 1.7B despite winning criterion 2. The pilot will check this directly.

**Next:** Shlok's answers → then Phase 1 code (`mstep/`) and the depth pilot.

### Blocked on me
- Nothing (answers received 2026-09-23, table at top).

### Changed
- New branch `multistep-gap` (from `stage2-determinism`'s HEAD `67c145d`; your uncommitted `writeup/pitch.*` edits carried over untouched).
- Added `CLAUDE.md`, `TASKS.md`, `STATUS.md`, `ASSUMPTIONS.md`, `scripts/lambda.sh`, `scripts/preflight_box*.sh`; `.gitignore` += instance state.
- Outside the repo: Lambda SSH key `afp-lambda-key` registered; instance `afp-multistep` running.

### Found
- Depth-1 survivorship makes naive late-stage transfer uninterpretable without a cohort control.
- 2604.19775 never tests cross-step transfer, reports no within-task or text baseline, and drops ambiguous labels via conformal filtering. It neither supports nor refutes Prof. Lin's hypothesis.
- ScienceWorld is probably not viable for base Qwen3-1.7B (0.12/100 mean score).
