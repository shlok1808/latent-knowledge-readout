# HANDOFF — multistep-gap run (written 2026-09-24 15:10 UTC)

Read this, then `TASKS.md` (checklist), `STATUS.md` (newest entry first), `ASSUMPTIONS.md` (every definition), `DATASET_CHOICE.md` (Phase 4 env). Everything important is in those files, not in any chat history.

## 1. What we are testing and why

Prof. Lu Lin's hypothesis (meeting 2026-09-17; transcript 21:52–39:10 is the source of truth; Speaker A = Lin, B = Shlok): the standard **single-step probe recipe does not transfer to multi-step agentic tasks**. A probe trained on one stage of a trajectory performs well on that stage and degrades on others, like the motivation figure in her group's CEB paper (arXiv 2607.12397, Fig. 1: AUC/ECE vs within-trajectory step). If true, different stages have different internal patterns, which motivates a stage-robust method later. "Stage" is tested two ways: round index (early 1–2 / mid 3–5 / late 6+) and milestones (k-th successful non-terminal craft/sub-goal). **We want evidence for or against the gap, not high AUC. A clean negative result is fine.**

Main threat, and why the **same-episode (cohort) control is a PRIMARY analysis**: at shallow TextCraft depths, successes finish early and failures run to the round cap. So "late stage" populations are mostly failures (depth 1: 79% failures by round 6). A naive transfer matrix changes *who is scored* as well as *when*. The cohort version trains and tests on the same episodes across stages (ASSUMPTIONS A-12, A-19). It is reported side by side with the naive matrix.

Prior baseline (stage-2, depth 1, round 1, layer 28): pooled AUC 0.791 (replicates Ruan et al. 2607.06503), within-task 0.547; a layer/round sweep found 0.801 at layer 12 / round 3. Lin called layer/round picking "heuristic tuning, not a contribution".

## 2. Box state right now

Lambda instance **`afp-multistep-2`** (A100-SXM4-40GB, us-east-1, 129.213.24.153), id in `.lambda_instance_id`. Access: `scripts/lambda.sh {ip,ssh,sync,pull,status,terminate}` (keys in `~/.kyc-secrets`, SSH key `~/.ssh/lambda_key` = Lambda key `afp-lambda-key`). Repo on box: `~/agent-failure-probes`. **Box-hours used: 2.53 at 15:05 UTC** (this box ~0.3 + previous boxes 2.2). Watchdog cap 40.

| tmux session | What it is | State at handoff |
|---|---|---|
| `tc2` | `scripts/run_tc2.sh` for TextCraft/Qwen3-1.7B (`runs/ms-tc-1p7b`, 4,264 episodes, depths 1–3) | extracting all-layer activations (26k/70k rows, ETA ~15:25). Next it starts the MC step in the background, which will **park on HOLD** (see below), and in the foreground runs the no-MC analysis: `mstep.dataset` → `mstep.probes` (Phases 2+3, all depths, then depth ≥ 2) → `mstep.figures`. That's CPU, likely several hours. After MC is released and done: with-MC analysis. |
| `p7b` | `scripts/pilots_7b.sh` | waits for (a) the 1.7B extraction to finish and (b) no other collector running. Then it checks the Qwen2.5-7B-Instruct fit (config: 28 layers, d=3584, 15.2 GB bf16) and runs pilots at vLLM util 0.85: TextCraft **20 tasks per depth** (depth 2 and 3) × 8, then ScienceWorld 20 × 8. **If ScienceWorld has < 6/20 mixed tasks it automatically also pilots BabyAI and ALFWorld (20 × 8 each).** It ends with one `PHASE 4 GATE SUMMARY` line in `logs/runner_status.md` (v2 relaunched 15:09 UTC). |
| `watchdog` | `scripts/budget_watchdog.sh 40 2.2` | kills GPU jobs at 39.5 box-hours; `logs/budget.txt` |
| `envsrv_sciworld` | ScienceWorld server with `ulimit -n 65536` (port 36001) | up. ALFWorld (36002) and BabyAI (36003) servers are unsupervised preflight processes |
| `envpilots`, `preflight` | finished; idle shells | the 1.7B ScienceWorld pilot finished: 1/80 success, 1/10 mixed |

**MC hold:** `runs/ms-tc-1p7b/mc_scope.json` = `{"min_depth": 2, "max_depth": 2, "hold": true, ...}`. `mstep.rollout mc` checks this file before loading any model and waits while `hold` is true (no GPU memory used). **To release** (only after Shlok decides): set `"hold": false` in that file on the box. MC then proceeds on depth 2 only, and `tc2` continues to the with-MC analysis on its own.

**Where things land (on the box):** runner milestones in `logs/runner_status.md`; step logs in `logs/<run>.log`, `logs/pilots_7b.log`, `logs/env_pilots.log`. Results go to `results/multistep/textcraft_qwen3-1.7b/{all_depths,depth_ge2}/` (CSVs: `phase2_cells.csv`, `phase2_baselines.csv`, `phase3_transfer.csv`, `phase3_round_curves.csv`, `population_by_round.csv`, `phase3_layers.json`; PNGs `fig_*.png`). 7B pilots: `runs/ms-{tc,sciworld}-q25-7b-pilot`.

Mac copies (pulled 2026-09-23): all 4,264 TextCraft episodes, the ALFWorld/BabyAI/1.7B pilots and the logs are under `runs/` and `logs/`. Activations live only on the box (rebuildable, ~1 h).

## 3. Standing instructions (from Shlok)

- The repo `shlok1808/agent-failure-probes` was made **private** on 2026-09-24 and the `multistep-gap` branch (code, docs, result CSVs/figures) is pushed there. It is still Lin's unpublished direction: never push anywhere public and never change the repo's visibility. No `Co-Authored-By` lines.
- **MC: depth 2 only**, and do not start MC until Shlok decides after the 7B pilots (hold file above).
- **Priority if budget is tight:** TextCraft Phase 3 with the same-episode control > Phase 4 env (Phases 1–3) > Qwen2.5-7B full run. Drop from the bottom and log it in STATUS.
- **Bigger model = Qwen2.5-7B-Instruct** (replaces Qwen3-8B; Ruan's strongest TextCraft setting). **The 7B pilots decide Phase 4**; the 1.7B ScienceWorld pilot is only a comparison point.
- **ALFWorld needs automatic server restarts** if ever used (`restart_server_every_chunk` in its config; `scripts/env_servers.sh restart alfworld`). Already built.
- **Leave `writeup/pitch.*` and `AGENTS.md` untouched** (uncommitted user edits).
- **No destructive actions without asking:** no deleting data or prior results, no terminating the box, no force-push, nothing outside the repo. Prior results `runs/stage2-002`, `results/stage2-002` are read-only.
- Every unrequested definition goes in ASSUMPTIONS.md. Every STATUS update ends with **Blocked on me / Changed / Found**. Task-grouped CV everywhere; report pooled AND within-task AUC for every result. Flag grid-search luck.
- Run long jobs in named tmux sessions; never `pkill -f` a pattern that also matches your own ssh command line. Never edit a shell script a running tmux session is executing; add a gate file or a new script instead.

## 4. First thing the next session should run

```bash
cd ~/agent-failure-probes
scripts/lambda.sh status                      # expect: active
scripts/lambda.sh ssh 'cd ~/agent-failure-probes; tmux ls; cat logs/budget.txt; tail -15 logs/runner_status.md; cat runs/ms-tc-1p7b/mc_scope.json; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```
Healthy = tmux has `tc2`, `p7b` (until its pilots finish) and `watchdog`; box_hours < 40; no `FAILED` lines except the known soft ones you can explain; mc_scope still shows `"hold": true` unless Shlok released it. Then re-arm a monitor on new lines of `logs/runner_status.md` (a 60 s ssh poll that prints new lines; see how the previous session did it in STATUS).

## 5. Open decisions waiting on Shlok

1. **Release MC?** After seeing the 7B pilots (and ideally the 1.7B no-MC Phase 3 results), set `"hold": false`. Depth-2 MC upper bound ≈ 1.55M continuation rounds, roughly 6–10 box-hours (STATUS 2026-09-23 21:05).
2. **Phase 4 env,** decided on the 7B pilots: ScienceWorld if it passes the mixed-outcome gate (≥ 3/10 → with 20 tasks, ≥ 6/20 mixed), else BabyAI (1.7B: 41% success, 8/10 mixed, but no native step reward and few real sub-goals; see DATASET_CHOICE.md). If ScienceWorld fails at 7B, the BabyAI and ALFWorld 7B pilots run automatically (Shlok 2026-09-24); report all three.
3. **7B full run:** which env (TextCraft depth ≥ 2 vs the Phase 4 env) and whether it gets MC, given the budget.
4. (Resolved) The 7B TextCraft pilot is 20 tasks per depth, per Shlok.
