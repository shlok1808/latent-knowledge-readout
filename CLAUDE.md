# latent-knowledge-readout: project context

This file is tracked in git (private repo), so Lambda sessions get it. Last updated 2026-10-28.

## Start here in a new session
1. Read this file and the ledger at the top of `RESEARCH_LOG.md`.
2. For experiment, result or paper work, load the `research-rules` skill (research plugin) and follow it. For how the laptop / Lambda / git / HF setup works, read the plugin's `SETUP.md`, which the skill links to. Laptop copy: `~/claude-mods/research/SETUP.md`.
3. On the laptop, also read `research/research_notes/Knowledge action divergence internals round 2/RESUME.md` and the user's notes on the report page (section "Report page and notes" below). They are the agenda for discussion. `research/` is laptop-only (git-excluded, 255 MB). On a Lambda instance, this file and `RESEARCH_LOG.md` are the whole context.
4. Ask the user what they want to do. Don't launch big runs until they say go.

## Current state (updated 2026-11-04) — READ THIS FIRST, it supersedes older sections where they differ
**Active project: replicate Orgad et al. 2610.28325 (sycophancy as in-context updating), then extend it across Llama-3.1-8B training stages.**
- What they did (open models only): Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. Neutral pass on CSQA/ARC-Challenge -> 512 caving examples (the PRUNING set, not held out) -> B-TAP scores weights, zero the top 1,000 (protecting normal-QA weights) -> test on held-out questions + all of OBQA: caving drops, general skills stay, AND listening to reliable sources drops too (Table D1, before vs after). Conclusion: sycophancy and useful updating share weights.
- What we do: no training of our own. Checkpoints: meta-llama/Llama-3.1-8B (base), allenai/Llama-3.1-Tulu-3-8B-SFT, -DPO, allenai/Llama-3.1-Tulu-3-8B (final RL), meta-llama/Llama-3.1-8B-Instruct (Orgad's model, the sanity gate). Meta/Qwen never released their intermediate stages; Tulu is AllenAI's open post-training of the same base (general post-training, not anti-sycophancy). Tulu also has 70B stages (for Dr. Lin's "larger models").
- Order (agreed 2026-11-04):
  1. Gate 1: Meta Instruct, no pruning, match D1 "Original" column. Rules fixed before running: only paper-unspecified settings may change; match = within 20% on all 8 cells; at most ~3 setup changes, then report the gap. Gate command: metrics.check_anchors(cells, "meta_instruct", "native"), primary column = 12 text sources.
  2. Step 1: same measurement on base / SFT / DPO / final. Do caving and useful updating rise together? Does base already cave (Perez et al. 2022 suggests maybe; 2610.37788 says capitulation appears only after post-training)? Also tells us if each stage caves enough to build a pruning set.
  3. Build pruning: B-TAP from 2604.09544 (no public code found; method in 2610.28325 App. C.1/C.3: score = -w*dL/dw over the caving set, protect top 0.005% per matrix by preservation score, zero top 1,000; random magnitude-matched control). Consider emailing Orgad for code.
  4. Gate 2: prune Meta Instruct, match D1 "Pruned" column.
  5. Step 2 (our new part): prune at each stage. When does shared support appear? Same weights across stages (overlap)? Does the cost to useful updating grow with training? Lead: 2604.09544 summaries say the harm mechanism separates more in aligned models (unverified, papers/2604.09544.md).
- Scoop context: behaviour-only "sycophancy rises with post-training" is taken (2610.37788; heads across Tulu stages in 2610.22937). The open part is useful updating moving with it + shared mechanism per stage.
- Code: knowledge_readout/syco/ (items, prompts, scorer, metrics, preflight), configs/syco_*.json, 211 tests. Contract: knowledge_readout/syco/CONTRACT.md. Exact paper text: papers/2610.28325_templates.md (the template table is C3 in v1). Built and reviewed in 6 rounds (PR #3). Nothing run on a real model.
- Open decisions (paper unspecified; defaults in code, Shlok must confirm, one at a time): (1) answer reading: default = single most likely letter-variant token (greedy restricted to letters, ties by lowest token id) vs letter_argmax; (2) item splits: full CSQA validation, ARC-Challenge test, OBQA test; (3) reliable source = 12 text sources primary, with tool secondary; (4) tool cell = our text wrapper (paper used a native tool call); (5) wrong-option seed fixed; (6) no system prompt.
- Not done: lab-mod bet gate for run_checkpoint (old CLI gate is gone) and sealed predictions before the first run; preflight on Lambda (letter tokens, BOS for Tulu templates, top-token histogram); .claude/settings.json is no longer in git, so Lambda needs it copied (scripts/lambda_up.sh).
- Repo: package renamed failure_probes -> knowledge_readout; old TextCraft code on archive/textcraft-pipeline-final; branch cleanup done on GitHub except possibly 3 branches the user is deleting by hand.
- Measurement in plain words: flip rate = did the answer change the way the claim pushed; log-odds shift = how much belief in the pushed answer moved (ln(p/(1-p)) after minus before), catches moves without a flip. Harmful (initially right) and useful (initially wrong) are never pooled.

## Older state (2026-10-26) — background only
- **Old project is archived and OFF LIMITS.** The old multi-step probe-transfer project's local data (runs/, logs/, wandb/, writeup/, mstep/, scripts/, .lambda_* files) was moved to `~/Archive/agent-failure-probes-old-project/` on 26 Oct, and its code lives on git branch `archive/stage-transfer-v1`. Do not read, search, run or cite either unless the user explicitly asks in this chat. `.claude/settings.local.json` denies access to that folder and branch.
- **Project folder (26 Oct):** research files live in `~/latent-knowledge-readout/research/` (laptop-only, git-ignored, never commit): `reports/` (4 deep-research reports + run design), `research_notes/<run>/` (readings, checks, ideas, RESUME per run), `lin_meeting_2026-10-23/` (Colloquium deck, figures, speaker notes), `papers_master.csv` (806-paper scan), `mods/`. Old copies remain in `~/colm-guide/gapscan/` until the user decides to delete them. Reports are NOT auto-loaded: open the one you need from `research/reports/`.
- **Four pressure conditions, not just pushback:** 1 tempting wrong option, 2 user pushback, 3 no single answer, 4 forced to answer on hard/impossible questions. Start with 1+2.
- **Order changed:** Stage A (diagnose fates) first. Stage B (fix comparison vs DPO/CCA) moved to the END because DPO at 9-27B is expensive; prefer public checkpoints (base vs instruct, OLMo SFT/DPO stages) over training our own.
- **Datasets are NOT decided.** Lab fork #1 ("templated pushback for mechanism work + TriviaQA for behavior") is a DRAFT only; the user wants a proper dataset discussion first.
- **Measurement goes small to big:** single probe directions → whole subspace → token/layer paths; move up only if the smaller level can't answer.
- **First experiments (ranked by importance; the output screen can run first because it's nearly free):** (1) copy vs correctness, (2) plant fake fates to validate tools, (3) output-score screen of flip rate vs prior confidence per pressure.
- **Next steps:** (1) update plan + deck (Stage B last, 4 pressures); (2) meet Dr. Lin (compute/70B, which pressures first, scoop speed); (3) dataset discussion; (4) pilot when the user says go; (5) plant fake fates.
- **Nikhil Prakash's binding papers (read 26 Oct):** page https://claude.ai/artifact/AjWxdmdGp1pxm5MeVqjGsf; readings in `~/latent-knowledge-readout/research/research_notes/Nikhil Prakash binding papers/reading/`. Key uses: same-length aligned pairs for patching; patch one variable at a time and pointer+address together before calling bypass; probe/patch beyond the answer token; CCA as a Stage B baseline with a no-mask control; fine-tuning strengthens old routes (prior, not proof).
- **Reports:** group geometry `~/latent-knowledge-readout/research/reports/Group geometry methods for fates.md` (21+3 papers, re-ranked ideas in section 5, pass-2 papers in section 8).
- **Setup (28 Oct):** experiments run as Scribe notebooks on Lambda through a Desktop SSH session. Git is the source of truth (see SETUP.md). Ledger: `RESEARCH_LOG.md`. Paper notes: `papers/<arxiv-id>.md`.

## Progress tracker (added 2026-11-04)
- Flowchart page the user keeps in the sidebar: https://claude.ai/artifact/BaYq7D9CYNfbTy1fEuy1wn. Steps live in its db (collection `steps`, fields phase/order/status todo|now|done/title/plain/note; doc `meta/next` = {text, updated}). Update it with ArtifactData whenever a step changes, and always rewrite `meta/next` in plain words.

## What this project is
Studying what happens to correctness-relevant information inside a language model at the moment it acts against it. Goal: actionable interpretability for safety. Findings should point to a fix (training, inference, monitoring, architecture), not only a description.
Repo: github.com/shlok1808/latent-knowledge-readout (this folder). The old multi-step probe-transfer project is archived on branch `archive/stage-transfer-v1` (tags `stage-transfer-v1`, `main-before-clean`).

## Working frame (decided)
- Question first, method second. Don't pick a tool (probes, manifolds, SAEs) before the question is fixed. Earlier work leaned toward probes because the literature does; treat that as a bias.
- Three directions, method-neutral:
  1. Bypass vs override: does correctness-relevant information fail to affect the action, get overridden, get transformed, or become inaccessible?
  2. Where shared geometry breaks: does the representation-behavior relationship change when the model acts against its apparent knowledge? A manifold is one idea, not an assumption.
  3. Jump vs drift (light): in multi-step tasks, gradual or abrupt? Keep it separate from single-turn unless evidence connects them.
- Vocabulary: "decodable by a probe" is not "the model knows".
- Keep causal controllability (patching changes behavior) separate from endogenous mediation (the model naturally routes through it).

## Advisor constraints (Dr. Lin)
- Larger models (not only 7-8B), factual or consequence-bearing tasks, single-turn first.
- Define "long horizon" before using it.
- She pointed to Cammarata, "Neural Networks Are Already Clean" (Goodfire, published 15 Oct 2026). The user likes it as an angle, but the user said: keep it on the known-relevant list only, no research focus, and it must not steer searches. The round 2 run did not find it.

## Where things are
- Round 2 report (fast pass, fact-checked headline numbers): `~/latent-knowledge-readout/research/reports/Knowledge action divergence internals round 2.md`
- Round 2 notes, sweeps, deep reads, watcher flags, usage log, checkpoint, RESUME: `~/latent-knowledge-readout/research/research_notes/Knowledge action divergence internals round 2/`
- Run design (v2 + watcher + usage rule): `~/latent-knowledge-readout/research/reports/Run design.md`
- Round 1 report: `~/latent-knowledge-readout/research/reports/Knowledge action divergence internals.md`; artifact https://claude.ai/artifact/YFfm74y97wSwTYJommvhaB
- Research Notebook (tasks + notes; read with the ArtifactData tool, collections `notes`, `tasks`): https://claude.ai/artifact/WvXjUcMwbjsgUbQxUR5Fti. Key notes: `next-run-spec` (with a 2026-10-22 amendment), `known-relevant-papers` (v5), `scan-misses-2026-10-20`.
- Old handoff note: `~/latent-knowledge-readout/research/NEXT_CHAT_HANDOFF.md` (older than this file).
- Paper database from the first scan: `~/latent-knowledge-readout/research/papers_master.csv` (806 papers; old closeness ratings are unreliable; it lacks 2604.19117, 2610.21708, 2605.05115).
- The user's own thoughts doc (AlphaXiv chat): `~/Downloads/research doc ~ dr lin.pdf`. Its papers were all already in the scan or the notes. Check it against the field; it does not steer searches.

## Dr. Lin meeting deck (23 Oct 2026)
- Colloquium source, figures (`figures/gen_figs.py`, `gen_story_figs.py`), speaker notes, PDF, PPTX: `~/latent-knowledge-readout/research/lin_meeting_2026-10-23/`. Copies are in `~/Downloads/`.
- Viewable copy: https://claude.ai/artifact/4bWqg7cKMYqYUXhRwUFsuw
- **26 Oct weekly deck** (8 main + 10 backup, Helvetica Neue to avoid grainy Type 3 fonts, new figs in `figures/gen_weekly_figs.py`) now sits at the top of that folder; the 23 Oct originals are in `23oct_original/`. Also on private branch `slides-weekly`.
- Plan agreed with the user: Stage A (six fate checks) + Stage B (fix and recheck); four pressure conditions on shared facts (start with 1 competing answer + 2 user pushback); models 9B-27B, 70B only if Dr. Lin has compute.
- Cloud-session handoff: Notebook note `handoff-2026-10-23`.

## Group geometry run (25 Oct 2026)
- Report: `~/latent-knowledge-readout/research/reports/Group geometry methods for fates.md`. Run folder with readings, checks, ideas and `RESUME.md`: `~/latent-knowledge-readout/research/research_notes/Group geometry methods for fates/`.
- Paused to save usage. Report discussed before the reset. At ~4 pm: the second research dive: read 2608.01835 and 2607.20146 (already fetched), and run the skeptic on the architecture-lens ideas. Details in that RESUME.md.

## Report page and notes
- Simple-language report page with figures, analogy dropdowns and highlight-to-note: https://claude.ai/artifact/3LFFZqyEQNx6Vqw3ysFgHa (source: the scratchpad file `knowing-without-doing.html`; republish to the same URL by passing `url`).
- The user highlights text and adds notes or questions while reading. Read them with ArtifactData: `action: list`, `url` above, `collection: notes`. Fields: quote, section, kind (note | question), text, createdAt, author. Treat them as the discussion agenda.

## Round 2 findings (short)
- The correct signal usually survives; what breaks is routing to the action. Six fates: readout/threshold failure, bypass, not routed, override, transformed, erased (rare; the one erasure claim, 2607.00415, has a below-chance probe, so it is more likely a flip).
- First control: the logit margin. At 8B a probe beats it by only +0.008 AUC (2609.04582).
- Masking: after an RLHF refresh in Llama 70B, sycophancy fell from 39% to 3.5%, but the heads' ablation effect grew from +10.5 to +27 pp (2604.19117, one checkpoint pair).
- Training against weak detectors teaches evasion (GRPO over 85% vs DPO under 25%, 2505.13787).
- No paper shows endogenous mediation; the one 72B test is null. Read 2505.11770 before claiming this gap.
- Biggest scoop risk: 2610.21708 (15 Oct 2026). "Decodability is not use" is crowded in ICLR 2027 submissions.
- Ideas, ranked: (1) per-item fate test on factual sycophancy at 9B-72B (recommended, Gemma-2-9B on 400 TriviaQA pairs from 2604.19117, then Qwen2.5-32B); (2) the same protocol on unanswerable questions (2608.29109, Qwen3-14B); (3) does anti-sycophancy training remove or mask the circuit (high scoop risk); (4) per-item knows-vs-acts angle (direction 2, KAPPA setup, cheap kill test); (5) later: smooth state vs threshold in multi-step tasks.

## What is NOT done (from the full design)
Sweeps never met their stopping rule; recall was 22 of 31 known-relevant papers (misses were mostly OpenReview-only workshop papers). Not done: the 806-paper re-score, citation chasing both ways, full reads of abstract-only Core papers (unread: Err Knowingly 2508.18760, Orgad et al., Inside-Out 2503.15299, Senses Wide Shut), a full fact-check, and an abstract skim of 141 title-only OpenReview rows in sweep C. Before claiming novelty to Dr. Lin, do the citation chasing and full reads.

## Lessons from runs
- Round 1 missed KAPPA (2509.23782) by trusting old ratings and one vocabulary. Round 2 missed papers the same way. Use synonym families, cover OpenReview, test recall against the known list.
- A grep recall check over-counts. Check matches by eye (the watcher caught 24 to 22).
- Search tools fail under load: the session WebSearch budget (200), the arXiv API (cert/429), OpenAlex daily quota (a key needs an account in the user's name), Semantic Scholar 429. Fall back to WebFetch or curl on arxiv.org/search and to OpenReview's public API. NeurIPS 2026 main-track submissions are not public.
- Usage: 4 parallel Opus sweeps used about 58% of a 5-hour Pro window in roughly 30 minutes. A full design run needs several windows. The user wants big runs in about 30-40 minutes, main work first.
- The user approved Chrome for OpenReview (read-only, no sign-in), and PDF downloads from paper hosts only, each into its own folder under the run folder's `downloads/`.
- Keep one watcher agent continued across phases; it catches real errors.

## User preferences
- Student new to interpretability, reads between midterms: short, plain answers, summary first, details on request. Simple analogies in dropdowns for hard topics. Do NOT end replies with "explain it in one sentence" quiz questions.
- Never delete, force-push or publish without asking. Pushes only as the `research-rules` wrap-up allows, otherwise ask. If an answer card is missing, ask again in plain text.

## Repo rules
- Git holds the pipeline (`knowledge_readout/syco/` Orgad replication, `knowledge_readout/common.py`, `tests/`, `configs/syco_*.json`; the old TextCraft pipeline was removed 2026-11-03, preserved on branch `archive/textcraft-pipeline-final`) plus the research record: `CLAUDE.md`, `RESEARCH_LOG.md`, `notebooks/`, `results/`, `figures/`, `papers/`, and `scripts/` (`bootstrap.sh`, `lambda_up.sh`). Local data (`runs/`, `logs/`, `wandb/`) and `research/` stay out of git.
- Run tests with `pytest -q` in the `.venv` before committing pipeline code. If uv rebuilt the venv, run `uv sync --extra dev` first, since pytest is an extra.

## Lab mod (added 2026-10-23)
- The `lab` plugin (`~/claude-mods/lab`) runs here. NOTE (2026-11-03): the `failure-probes` CLI was removed with the old pipeline, so the gate below has no command to guard; the syco runs (`knowledge_readout.syco.scorer.run_checkpoint`) need a new gate. Old text: `failure-probes collect` and `analyze` are gated until Shlok has committed a bet; `prepare`, `extract` and `audit` are not. An open bet covers runs until it is settled, so one long collect needs one bet.
- First bet: the per-item fate test (idea 1 above). Before its first collect, seal a prediction of the fate split (readout failure / bypass / not routed / override / transformed / erased) with numbers, then ask for Shlok's.
- Real design choices for the fate test (model and size, item set, controls such as the logit margin) go through `/decide`: seal your proposal, get Shlok's, merge.
- Notebook: `LAB_NOTEBOOK.md` in this folder, local-only (git-ignored). `/lin` drafts Dr. Lin updates from it, `/kill` audits a result, `/park` holds tangents.
- Not yet tested whether the lab mod runs in Lambda SSH sessions (as of 28 Oct).
