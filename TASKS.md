# Tasks — multistep-gap run

Source of truth for what's next. `[x]` done, `[ ]` open, `[~]` in progress, `[-]` skipped (reason in STATUS.md).

## Phase -1 — Preflight
- [x] Read transcript (source of truth, 21:52–39:10) and summary; log discrepancies
- [x] Read arXiv 2604.19775 (conformal interpretability) and CEB 2607.12397 Fig. 1
- [x] Lambda API key + HF token found (`~/.kyc-secrets`), both verified; Qwen3-1.7B/8B not gated
- [x] SSH: `lambda_final` is passphrase-protected → registered existing unencrypted `lambda_key.pub` as `afp-lambda-key`
- [x] Launch A100-40GB box, sync repo
- [x] Box preflight: main venv, tests, both models downloaded, 8B fit test, sciworld/alfworld/babyai servers smoke-tested
- [x] vLLM 0.30 (cu129) venv + smoke test: 9.2k tok/s, seeds reproducible
- [x] Ask user all blockers in one message; record answers in STATUS.md

## Phase 0 — Verify (no GPU)
- [x] TextCraft reward is sparse: 1 only when goal item is crafted/got (`environment.py` step)
- [x] Layers from config: Qwen3-1.7B 28 (d=2048), Qwen3-8B 36 (d=4096)
- [x] Disk estimate: 15 GB (1.7B) / 40 GB (8B) all layers → store every layer, keep on box
- [x] Depth histogram {1:125, 2:291, 3:117, 4:11}

## Phase 1 — Data (Qwen3-1.7B, TextCraft)
- [x] `mstep/` collection code (vLLM generation, HF replay extraction), resumable, seeded
- [x] Label code: FINAL, step validity, subgoal progress, MC value
- [x] Pilot ~10 tasks per depth (collect done; MC stopped as too slow to wait on; analysis untested → validated on full run)
- [~] Full collection depths 1–3 (depth 4: 0/80 in pilot → dropped). Running: tmux tc17
- [x] Analysis code validated on real data (Phase 2 CSVs complete, 69,782 rows)
- [x] MC scope restricted to depth 2–3 (mc_scope.json); budget re-plan in STATUS
- [x] Now running as tmux `tc2` (MC parallel to analysis); handoff no longer needed
- [ ] Re-project budget after first MC chunk; apply drop order if > 38 h
- [x] MC scope → depth 2 ONLY (depth 3: 8 mixed tasks) and HOLD gate (mc_scope.json) — Shlok 2026-09-24
- [ ] Shlok decides: release MC hold after 7B pilots
- [~] 1.7B extraction (tc2) → then no-MC Phases 2–3 on CPU automatically (tc2)
- [~] Qwen2.5-7B pilots (p7b v2, auto after extraction): TextCraft 20 tasks/depth (d2, d3) ×8, ScienceWorld 20×8; if ScienceWorld < 6/20 mixed → BabyAI + ALFWorld 20×8 each → decides Phase 4; report all
- [ ] Extraction: last-token states per round, layers per disk decision

- [x] Box terminated 2026-09-23 22:15 UTC; all episodes pulled to Mac (acts not pulled; re-extract)
- [x] New box afp-multistep-2, preflight one-pass OK, episodes uploaded, run_tc2.sh resumed (tc2)
- [x] BabyAI success rule fixed; pilot relabelled: 33/80, 8/10 mixed; no native step reward (A-21)
- [x] ScienceWorld fd limit fixed; 1.7B pilot: 1/80 success, 1/10 mixed (comparison point only)

## Phase 2 — Probe strategies — DONE on 1.7B TextCraft (see STATUS 2026-09-24 15:30)
- [x] S1 last-token at t∈{1,2,3,5,8}; S2 mean-pool rounds 1..t
- [x] L2 logistic + diff-of-means; layer sweep (29 layer slots, 580 cells/label)
- [x] Baselines: TF-IDF+logistic on rounds 1..t; surface features
- [x] Pooled + within-task AUC for every cell; CIs
- [ ] Re-run with MC label once hold released
- [x] Depth-1-only Phase 2 rerun DONE: L12/t3 0.815 vs text 0.717 (replicates stage-2); round-3-only (tmux d1repl, site=action) → results/multistep/textcraft_qwen3-1.7b/depth_eq1_action; result line lands in logs/runner_status.md

## Phase 3 — Stage generalization
- [ ] Round-based stages (early 1–2, mid 3–5, late 6+) transfer matrix
- [ ] Milestone stages (end of k-th successful craft; depth ≥2)
- [ ] Pooled-over-stages probe vs per-stage probes
- [ ] PRIMARY: same-episode cohort transfer matrix, side by side with naive (ASSUMPTIONS A-12)
- [~] Last-step variant (early→r2, mid→r5, late→r10; cohort ≥10 rounds) running in tmux `fast` alongside the averaged version
- [ ] Main figure: AUC vs test stage, FINAL vs IMMEDIATE panels

## Phase 4 — Second environment
- [x] Env servers up (preflight nohup processes on 36001-3; `scripts/env_servers.sh` keeps them); httpenv smoke-tested on 9 tasks + 64-way concurrency; configs `configs/ms_{alfworld,sciworld,babyai}_1p7b.json` (AgentEval test indices, 30 rounds, 16k ctx)
- [~] DATASET_CHOICE.md drafted with prior evidence; pilot columns pending
- [x] Launched `scripts/run_env_pilots.sh` in tmux `envpilots` (waits for ≥20 GB free GPU) (shares GPU: pilots MSTEP_GPU_UTIL=0.40, TextCraft MC reads logs/gpu_util=0.50)
- [x] ALFWorld server restarts every chunk (config flag + env_servers.sh restart)
- [ ] Pilot chosen env ~10 tasks; fallback if criterion 3 fails or >1 GPU-h debugging
- [ ] Phases 1–3 on it

## Phase 5 — Qwen2.5-7B-Instruct (replaces Qwen3-8B, Shlok 2026-09-24)
- [x] Config: 28 layers, d=3584, 15.2 GB bf16; runtime fit check in pilots_7b.sh
- [ ] Choose env for the 7B full run (Shlok decision), justify in STATUS.md
- [ ] Phases 1–3; compare transfer-matrix shape

- [x] HANDOFF.md written (2026-09-24 15:10)

- [x] 2026-09-24 2:35 PM ET: box terminated after full pull (acts not pulled; rebuildable)
- [ ] DECIDE: Phase 4 env (ALFWorld @ 7B recommended), MC depth 2, held-out layer selection

## Deliverables
- [ ] results/ CSV + PNG by env/model
- [~] RESULTS.md interim written (Phase 2 + headline framing for Lin); fill §3 replication, §4 Phase 3
