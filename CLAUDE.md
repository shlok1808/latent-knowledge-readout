# sycophancy-shared-trace: Shlok's personal project

Created 2026-11-10. This is Shlok's own side project, separate from the advisor work. It continues the
sycophancy / useful-updating line that Dr. Lin decided not to pursue on 2026-11-09. The advisor project
(know-do gap) lives in `~/latent-knowledge-readout` on its own branches. Keep the two apart.

## Start here in a new session
1. Read this file, then the ledger at the top of `RESEARCH_LOG.md`.
2. For experiment, result or paper work, load the `research-rules` skill (research plugin) and follow it.
   Machinery (laptop / Lambda / git / HF): `~/claude-mods/research/SETUP.md`.
3. The full record of what was run is `results/meta_instruct/EXPERIMENT_NOTES.md` (sections 5f-5h matter most).
4. Ask Shlok what he wants to do. Don't launch GPU runs until he says go.

## How this folder relates to the old one
- Same git repo (github.com/shlok1808/latent-knowledge-readout, private), checked out here as a **git worktree**
  on branch `personal/shared-trace`, cut from `sycophancy-eval-id8yhs` at 8242f54 (all Gate 1, Gate 2, E2, E3 work).
- Never merge this branch into `main` or the advisor branches without asking. The branch is local only until
  Shlok OKs the first push. Lambda needs it pushed: `scripts/lambda_up.sh <ip> personal/shared-trace`.
- Raw rows and logs are NOT copied (2.6 GB, git-ignored). They stay in `~/latent-knowledge-readout/runs/` and
  `~/latent-knowledge-readout/logs/`. Read them from there; new runs go to this folder's `runs/`.
- `research/lin_deck_2026-11-09/` is a local copy of the 9 Nov deck (slides.md, speaker_notes.md, figures). Git-ignored.
- `LAB_NOTEBOOK.md` is a local copy of the lab notebook up to 2026-11-10 (git-ignored).

## The question
Orgad et al. 2610.28325 showed on Llama-3.1-8B-Instruct that pruning the weights behind sycophancy (B-TAP) also cuts
useful updating (following a reliable source that is right). So caving and useful listening share weights.

Shlok's idea, the "shared trace across training stages": when does that shared support appear? Prune at each
post-training stage of the same base model and ask:
1. Do caving and useful updating rise together through training?
2. When does the shared trace appear (SFT, DPO, RL)?
3. Are the same weights picked at each stage (overlap)?
4. Can we cut caving without cutting useful listening (separate them)?

Checkpoints (no training of our own): `meta-llama/Llama-3.1-8B` (base), `allenai/Llama-3.1-Tulu-3-8B-SFT`, `-DPO`,
`allenai/Llama-3.1-Tulu-3-8B` (final RL), `meta-llama/Llama-3.1-8B-Instruct` (Orgad's model, the sanity gate).
Tulu also has 70B stages.

## Dr. Lin's objection (2026-11-09), which any pitch has to answer
- Caving and useful updating are "fundamentally coupled", both forms of context-following.
- Separating them "may not be feasible or clearly motivated".
- Whether a model should follow the user depends on correctness, so the problem is open-ended.
- Possible answer (sealed lab envelope #35, Claude's proposal; Shlok's take not given yet, so it is still sealed).

## Where things stand (2026-11-10)
- **Gate 1 PASSED** (2026-11-05): unpruned Meta Instruct matches the paper's Table D1 "Original" in 8/8 cells within 20%
  (user + 12 text sources) under bare-letter reading ("A"/" A" only). All-spellings reading: 6/8. Rule: our own
  findings must hold under both readings. EXPERIMENT_NOTES 5e.
- **Gate 2 NOT passed** (2026-11-05): B-TAP as written gets 4/8 (bare letters), 7/8 (all spellings). Pruning breaks the
  one-letter answer format ("The correct answer is (C) ...") because every caving target was a bare letter;
  the paper doesn't report this. Random control: no change. EXPERIMENT_NOTES 5f, 5g.
- **Format fix** (2026-11-09): the paper's method + 10x more protection (p = 5e-4, 0.05% per matrix) keeps the
  format (replies opening with "(" 93% vs original 96%, paper-style 74%) but costs 3.3-4.6 pp written accuracy
  (pre-set limit was 3 pp). Under it, user-claim log-odds ratios pruned/original: 0.84 wrong claims, 0.83 right
  claims; random 1.00. Exact D1 Pruned numbers: 2/8. EXPERIMENT_NOTES 5h.
- **Separation:** in no pruning arm do harmful and useful updating come apart for the same speaker (difference
  -0.06 to -0.01). The "subtract useful updating" arm spares reliable sources, but only because its objective was
  built with the speaker split (caving = user wordings, subtracted set = 12 text sources). That is not evidence
  about correctness-based separation.
- **The decisive test is still missing:** a within-speaker contrastive arm (user caving MINUS user correct
  assertions), plus a check that eval wording doesn't overlap pruning wording ("I believe the answer is {option}" is
  pruning wording 1).
- Training stages: not started. Base Llama puts ~2% of its probability on answer letters in the plain format
  (it continues the dialogue), so base needs a different format (few-shot or an answer prefix) first.
  `results/base/letter_check.json`.
- Open in the lab tool: bet #26 (Gate 2) needs settling together; envelope #35 needs Shlok's take.

## Candidate next steps (Shlok decides, one at a time)
1. Answer Lin's objection with data already on disk (no GPU). See envelope #35.
2. Within-speaker contrastive pruning arm (one H100, about 2 h). The real separation test.
3. Training stages: fix the base format, then Step 1 (behaviour per stage), then Step 2 (prune per stage).

## Related work to check before claiming anything new
- 2610.22937: sycophancy heads traced across Tulu-3-8B / OLMo-2-7B stages (closest; multiple choice, ≤8B).
- 2610.37788: behavioural, across training stages; capitulation appears only after post-training.
- 2611.07071 (Goodfire, mechanistic shaping): reliance vs encoding; never tests useful updating. Note: `papers/2611.07071.md`.
- 2604.09544: B-TAP origin; code at github.com/orgadhadas/harm_pruning (no license file).
- Paper notes in `papers/`. B-TAP in plain words: `papers/btap_explained.md`.

## Code map
- `knowledge_readout/syco/`: items, prompts (Table C3 templates), scorer, metrics, preflight, pruning (B-TAP,
  margin loss), pruning_data, reply_probe. Contract: `CONTRACT.md`. Plan and pre-registered rules: `PRUNING_PLAN.md`.
- `scripts/`: `run_sycophancy_eval.py` (main eval), `run_btap.py`, `run_gate2.sh`, `run_e2.sh`, `run_e2_evals.sh`,
  `analyze_e2.py`, `run_e3.py`, checks.
- Configs: `configs/syco_checkpoints.json`, `configs/syco_templates.json`. Tests: `pytest -q` in `.venv`
  (if uv rebuilt it, `uv sync --extra dev` first).
- Measurement in plain words: flip rate = did the answer change the way the claim pushed; log-odds shift = how much
  belief in the pushed answer moved. Harmful (initially right) and useful (initially wrong) are never pooled.

## Working rules
- Lab mode: seal Claude's proposal or bet before Shlok gives his; one question at a time; short replies; park tangents.
  This is Shlok's project: don't present Claude's ideas as his.
- Plain names in chat ("the format fix", "the contrastive arm"), not codes like E2-D.
- Launch and terminate Lambda GPUs yourself (key in `~/.kyc-secrets`, never print it); always terminate.
  Speed-check batch / memory before every GPU run.
- Never delete, force-push or publish without asking.
- The old multi-step probe project (`~/Archive/agent-failure-probes-old-project/`, branch `archive/stage-transfer-v1`)
  is off limits unless Shlok asks.
- Slides: Colloquium.
