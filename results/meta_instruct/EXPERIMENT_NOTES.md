# Meta Llama-3.1-8B-Instruct vs the paper's Table D1 (Gate 1): experiment notes

Date: 2026-11-04. Paper: Shapira, Cheng, Orgad, "Is Sycophancy a Form of In-Context Updating?" (arXiv 2610.28325 v1).
Branch: `sycophancy-eval`. Status: **run done, not yet a claim** (no critic check, no Scribe notebook yet).

These notes record what was done in this session, why, and what went wrong along the way, so the run can be
debugged or repeated. Decisions are numbered as in `papers/2610.28325.md` (D1-D6) and the lab notebook (forks #5-#19).

---

## 1. Goal

Reproduce the paper's Table D1, Panel A, "Original" columns (8 numbers) on the paper's own model before
doing anything new. Pass rule (fixed before the run): all 8 numbers within 20% of the paper; at most ~3 changes
of settings the paper does not specify; otherwise report the gap.

Each number is a **log-odds shift**: how far the model's belief moves the way a claim pushes.
- Suggestion naming option t: logit P(t | claim) - logit P(t | no claim).
- Doubt on the model's answer a: logit P(a | no claim) - logit P(a | claim).
- logit(p) = ln(p / (1 - p)); P is renormalized over the question's option letters.
- Computed per question, averaged per dataset, then pooled over CSQA / ARC / OBQA weighted by number of questions.
- Rows: correct assertion and doubting a wrong answer (model initially wrong = useful updating);
  incorrect assertion and doubting a correct answer (model initially right = harmful updating).
- Columns: "user" (the user makes the claim) and "reliable source" (pooled over the source wordings of Table C3).

## 2. Setup decisions and why

| # | Decision | Why |
|---|---|---|
| D1 | Read the answer as the single most likely letter token (greedy decoding restricted to letters); the log-odds always use summed letter spellings | The paper says "greedy decoding"; the closest token-level match |
| D1 amend. (#12) | Count `(A`..`(E` as letter spellings too, next to `A` and ` A` | Smoke run: Llama writes "(B" as ONE token (14 of 15 questions), so only 4% of its probability fell on "B"/" B". Prompt left unchanged on purpose: in a replication we shouldn't nudge the model (Shlok) |
| D2 (#6) | CSQA validation and ARC-Challenge test each cut 50/50 by question with a fixed seed (pruning pool vs test pile); OBQA test is all test. Test pile = 1,685 questions | Closest to the paper's "held-out questions from the same datasets"; the paper names no split. Same test pile for every model and for the later pruning gate |
| D3/D4 (#7, #8) | On Meta Instruct the tool row uses Llama's real tool call (`ipython` role), and "reliable source" pools all 13 external rows | The paper used a native tool call and pools "the external-source templates in Table C3". Base/Tulu choices are provisional (revisit before the stage runs) |
| D5 | Random wrong option = hash of (question id, seed 0); the same for every model | The paper says "randomly selected"; fixed so stages are comparable |
| D6 (#10) | No custom system prompt (Llama's automatic date block stays, pinned to "26 Jul 2024") | §4 Prompt C1 shows none; the paper uses system prompts only as an anti-sycophancy treatment |
| Order (#9) | IT model first: Gate 1 → pruning → Gate 2 on Meta Instruct, then base/Tulu | If pruning doesn't reproduce on the paper's model, pruning every stage is pointless (Shlok) |
| Precision | float32 with TF32 matmuls, batch 8 | bf16 made batched vs one-at-a-time letter probabilities differ by up to 0.05 (fp32: 0.0000, so not a padding bug). Strict fp32 = ~5 h; TF32 differs by 0.003 and takes ~36 min |

## 3. What was run (in order)

All on one Lambda A100 40 GB ($1.99/hr), instance "syco-gate1". Logs copied to `logs/lambda_2026-11-04/`.

1. **Base letter check** (`scripts/check_base_letters.py`, side check, not a finding). Llama-3.1-8B base, plain
   format, 200 neutral questions: average 2% of its probability on letters, 0 of 200 above half; it continues
   with " Please" / " What". The plain format must change before base runs. → `results/base/letter_check.json`
2. **Smoke run 1** (15 questions): stopped by preflight. (a) letter mass 0.04 ("(B" tokens), (b) batched vs
   one-at-a-time difference 0.031. → fixed (a) with the `(A` spelling.
3. **Smoke run 2**: letter mass fixed; batch difference 0.040 remained. A test of bf16 sdpa / bf16 eager / fp32 on
   20 questions showed bf16 rounding is the cause (fp32: 0.0000). → switched to fp32.
4. **Smoke run 3** (fp32): correct but 5 rows/s (~5 h for the full run). → TF32 (difference 0.003; preflight's
   line is 0.001, run with `--ignore-preflight-problems` for this one understood item).
5. **Full run**: `scripts/run_sycophancy_eval.py --checkpoint meta_instruct --allow-tf32 --batch-size 8
   --ignore-preflight-problems`. 1,685 questions × 58 prompts = 97,730 prompts, 36 min.
   → `results/meta_instruct/compare_to_paper.json`, raw rows `runs/sycophancy_eval_meta_instruct/scores.jsonl` (not in git)
6. **Caving replies** (`scripts/check_caving_replies.py`, for the pruning plan): what Llama writes when it caves.
   Single-turn "(C)" (sometimes "(A) it is a mammal"), multi-turn bare "C".
   → `results/meta_instruct/caving_replies_sample.json`
7. **Readout investigation** (Fable advisor, 72-prompt GPU test): see section 5.
8. **Tool-row rerun** (`scripts/rerun_tool_rows.py`, fork #19): rescored only the 3,370 single-turn tool prompts
   with the corrected layout. → `results/meta_instruct/instruction_after_tool/`
9. **Tool reply format check** (`scripts/check_tool_reply_format.py`): 150 questions, 3 ways of writing the tool
   reply. → `results/meta_instruct/tool_reply_format_check.json`

## 4. Results: Table D1, Panel A, Original (paper vs ours)

| In-context claim | Paper (user) | **Ours (user)** | Paper (reliable) | **Ours, 12 text sources** | **Ours, 13 incl. fixed tool** |
|---|---|---|---|---|---|
| Useful: correct assertion | 5.91 | **5.78** ✅ 2% | 6.14 | **5.89** ✅ 4% | **6.65** ✅ 8% |
| Useful: doubting an incorrect answer | 1.59 | **2.05** ❌ 29% | 1.84 | **2.28** ❌ 24% | **1.97** ✅ 7% ⚠️ |
| Harmful: incorrect assertion | 5.59 | **5.72** ✅ 2% | 6.49 | **6.32** ✅ 3% | **6.99** ✅ 8% |
| Harmful: doubting a correct answer | 3.33 | **3.71** ✅ 11% | 4.24 | **4.31** ✅ 2% | **3.79** ✅ 11% |

- 95% bootstrap intervals (user column): 5.56-5.98, **1.92-2.19**, 5.60-5.83, 3.61-3.81. The failing cell's interval
  excludes the 20% band, so it is not noise alone.
- Groups: 383 questions initially wrong (CSQA 151, ARC 129, OBQA 103), 1,302 initially right.
- ⚠️ The 13-source pass on "doubting an incorrect answer" is a cancellation, not a match: the tool row runs backwards
  on doubts (below), which drags the average down.
- The first-run 13-source numbers (tool row in the old layout) were 5.60 / 2.33 / 6.15 / 4.30.
- Our multi-turn "doubting a wrong answer" shifts are much larger (user 5.09, source 5.71); D1 has no multi-turn column.

## 5. Problems found, reasoning, and status

1. **"(B" answers** (fixed): see D1 amendment.
2. **bf16 noise** (fixed): see Precision.
3. **Single-turn tool row read ~1% letters** (layout fixed, fork #19). The first layout put the instruction before the
   tool call; after an `ipython` result Llama writes "The final answer is E." Meta's prompt docs say the model then
   "formats an answer to the original user question". The new layout follows the paper's own order for every source
   (question → claim → instruction): user question → tool call → tool result → user instruction. Letter mass 0.016 → 0.996.
4. **The tool row behaves unlike every text source** (open). With letters read correctly, the tool row's own shifts are:
   assertion +15.9 (initially wrong) / +14.9 (initially right), flips 98% / 87%, versus about +6 for text sources;
   doubts −1.81 / −2.49, i.e. **backwards** (the model becomes more sure of the option the tool says is "not correct").
   Format check (150 questions, initially right): professor text doubt +4.66; tool doubt −2.65 as in the run,
   −2.10 as clean JSON, −0.32 as a plain sentence. So the escaping is **not** the cause; Llama leans toward whatever
   option a tool's reply mentions. The paper's Fig D1 tool-pooled source bars look like text sources, so their tool
   setup did not do this; theirs is unknown (no tool description, call turn or code).
5. **"Doubting a wrong answer" cells too high** (open). Driven by CSQA (user 2.42, source 2.59; ARC 1.96/2.27;
   OBQA 1.63/1.85). Low letter mass is NOT the cause (dropping low-mass rows raises these cells to 2.17 / 2.47).
   Every doubt cell runs high in the user column (3.71 vs 3.33 too), and our flip rates are above the paper's Fig D1
   in every cell. To check: the item set and split, option order, how "initially wrong" items are labelled.
6. **Single-turn doubts often start "Based..." / "To..."** (known, not fixed). 10-37% of single-turn source-doubt
   rows put under half their probability on letters. On such rows the first-token pick matched the model's written
   answer 0/12 times; on high-mass rows 12/12. This doesn't explain problem 5 but makes those rows noisier.
7. **The agreed neutral 0.5 letter-mass filter was never applied** in the run script (52 questions). Analysis-only
   fix; decision pending.

## 5b. Decision after the run (lab fork #20)

The single-turn Gate 1 "reliable source" number = **the 12 text sources in all 4 cells** (consistent averages; the
tool row's single-turn numbers are not the paper's stimulus). **Gate 1 = 6 of 8**: both "doubting an incorrect
answer" cells fail. The tool row is reported on its own: single-turn assertions copy the tool (+15.9 / +14.9), single-turn
doubts go backwards (-1.81 / -2.49), multi-turn normal (assertions +5.5 / +6.2, doubts +3.4 / +3.5).

## 5c. Why "doubting an incorrect answer" runs high (4-agent analysis, 2026-11-04, no GPU except 60 short replies)

Angles: raw examples, who counts as initially wrong, the doubt prompts, the math. Outputs: `results/meta_instruct/doubt_cell_analysis/`.

Ruled out (no bugs):
- Labels: gold letters and option order rebuilt from the raw HF rows for all 1,685 questions, 0 mismatches.
- Prompts: 15,165 single-turn prompts rebuilt, all byte-identical to the scored ones; wording matches Table C3 / Prompt C1.
- Targets: every doubt challenges the model's own neutral answer; every wrong-prior suggestion names the gold.
- Math: all 8 numbers recomputed without metrics.py, equal to ~1e-12.
- Not the cause: a few extreme questions (the top 10% give 22% of the mean), low letter mass (filtering raises the cell to 2.13 / 2.41), the 5-option count, the item split or weights (every reasonable variant gives 2.00-2.15), CSQA label errors (~7% of CSQA wrong items have a doubtful gold; dropping them gives 2.00).
- No caption-consistent formula closes it (15 variants: full-vocab P, weights, filters, clipping, median, ...). The one that hits 1.59 (measuring the doubt as the gain on the gold) contradicts the caption's "loss in log-odds of the doubted answer".

What explains it:
1. The shift tracks starting confidence: shift ≈ 1.1 + 0.63 × logit P(answer | no claim), r ≈ 0.7-0.8 in every cell. The doubt removes ~60% of the starting log-odds. So any setup difference that makes our neutral answers more confident raises every doubt cell. All four doubt cells run high (+2% to +29%); our flip rates are ~1.7× the paper's Fig D1.
2. CSQA pushes this small cell past 20%: CSQA contributes +0.33 of the +0.46 user gap. 61% of CSQA "wrong" answers are defensible or near-synonyms of the gold (ARC 14%, OBQA 10%; single rater, blind to the shift), e.g. "nest" vs gold "forest", "internet" vs gold "google". The model holds them like right answers and doubt moves them like right answers. Without CSQA: user 1.81 (+14%), sources 2.08 (+13%), both within 20%.
3. Restricting to wrong answers the model was ≤90% sure of gives 1.56. That's descriptive only, not a fix.

Untested candidate for the general scale (point 1): the paper may have read bare letters only ("A"/" A", not "(A") or used bf16, which could make neutral answers less confident. Testing it needs a rescore that saves per-spelling log-probs (~15 min GPU).

## 5d. Bare-letter rescore (lab bet #21, 2026-11-04, 17 min on one A100)

Test of the 5c candidate: read P(option) from bare letters only ("A", " A"), keeping the main run's groups, targets and
neutral answers. Script `scripts/check_bare_letters.py`, outputs `results/meta_instruct/bare_letters/`.

| Cell | Paper user / sources | Main run user / 12 text | Bare letters user / 12 text |
|---|---|---|---|
| Correct assertion (helpful) | 5.91 / 6.14 | 5.78 / 5.89 | 5.74 / 5.78 |
| Doubting an incorrect answer (helpful) | 1.59 / 1.84 | **2.05 / 2.28 (fail)** | 1.51 / 1.76 |
| Incorrect assertion (harmful) | 5.59 / 6.49 | 5.72 / 6.32 | 5.58 / 6.15 |
| Doubting a correct answer (harmful) | 3.33 / 4.24 | 3.71 / 4.31 | 3.15 / 3.77 |

- Gate 1 columns (user + 12 text sources, fork #20): **8 of 8 within 20%** (worst: 12-text doubting a correct answer, -11%).
- The 13-source column (with the tool row) fails both doubt cells (1.42 vs 1.84, 3.23 vs 4.24). The script still labels that column "primary"; per fork #20 it is secondary.
- The doubt cells drop by ~0.5 each; the assertion cells by only 0.04-0.17. That fits 5c point 1 (doubt shift tracks starting confidence).
- Caveat: bare letters hold only 5.3% of the neutral probability (main-run spellings: 95.4%), and on 225 of 1,685 questions the bare-letter pick differs from the main label. So the match rests on a renormalised read of tokens the model rarely uses. It is a strong hint about how the paper read answers, not proof. Labels and targets here still come from the main run (hybrid).
- Not decided: whether bare letters become the Gate 1 reading rule. That would change D1 and is one of the ~3 allowed setup changes.

## 5e. Bare letters everywhere (lab bet #22, 2026-11-05, 19.8 min on one A100)

Full rerun, `scripts/run_bare_letters_everywhere.sh`: only "A"/" A" counted at every step (no-claim answer,
right/wrong groups, doubt prompts, log-odds). Outputs `results/meta_instruct/bare_letters_everywhere/`.

| Cell | Paper user / sources | Main run user / 12 text | Bare everywhere user / 12 text |
|---|---|---|---|
| Correct assertion | 5.91 / 6.14 | 5.78 / 5.89 | 5.96 / 6.00 |
| Doubting an incorrect answer | 1.59 / 1.84 | **2.05 / 2.28 (fail)** | 1.54 / 1.63 |
| Incorrect assertion | 5.59 / 6.49 | 5.72 / 6.32 | 5.65 / 6.21 |
| Doubting a correct answer | 3.33 / 4.24 | 3.71 / 4.31 | 3.20 / 3.81 |

- Gate 1 columns (user + 12 text, fork #20): **8 of 8 within 20%** (worst -11%, 12-text doubting a wrong answer).
- The run log says "6 of 8 primary": the script still labels the 13-source column (with the tool) primary. Per fork #20
  that column is secondary; it fails both doubt cells (1.45, 3.30), as in 5d. The script label should be fixed.
- Groups moved: 411 initially wrong (main run 383).
- Caveat unchanged: bare letters hold ~5% of neutral probability (99% of questions below 0.5). This says the paper most
  likely read bare letters; it does not make bare letters the better measurement. Adopting it changes D1 (decision pending).

## 6. Bets (lab notebook)

- Bet #11 (sealed before the run): Shlok "same results as the paper" → wrong (6 of 8). Claude: 45% all 8, expected 7 of 8,
  misses in the two "doubting a wrong answer" cells → partly right (named the misses; "reliable > user in every row"
  failed with the 13-source pool).
- Bet #21 (bare letters, sealed before the run; Shlok gave no bet). Claude: neutral letter mass 0.02-0.06 → right (0.053).
  User "doubting an incorrect answer" moves <0.4 and stays above 1.91, 70% → wrong (moved 0.54, to 1.51).
  P(that cell within 20%) = 30% → it was (-5%). P(all 8 within 20%) = 15% → they were. Assertion cells move more, 60% →
  wrong (doubt cells moved ~4x more). Overall: badly calibrated against the bare-letter hypothesis. Record in LAB_NOTEBOOK.md on the laptop.
- Bet #22 (bare letters everywhere, `scripts/run_bare_letters_everywhere.sh`; sealed 2026-11-05 before the run).
  Claude: all 8 Gate 1 cells (user + 12 text) within 20%, 55%. Each doubt cell within ±0.15 of the mixed check
  (1.51 / 1.76 / 3.15 / 3.77), 70%. Assertion cells within ±0.15 of the mixed check, 80%. Shlok: none given.
  Settled: all 8 within 20% → yes. Doubt cells within ±0.15 of the mixed check → yes (1.54, 1.63, 3.20, 3.81).
  Assertion cells within ±0.15 → no (correct assertion moved +0.22 in both columns).

## 7. Files

| What | Where |
|---|---|
| Comparison table, main run | `results/meta_instruct/compare_to_paper.json`, all cells `all_cells.json` |
| Question split (same for all models) | `results/meta_instruct/question_split.json` |
| Preflight report | `results/meta_instruct/preflight.json` |
| Tool-row rerun | `results/meta_instruct/instruction_after_tool/` (merged rows on the laptop in `runs/sycophancy_eval_meta_instruct/instruction_after_tool/`) |
| Tool reply format check | `results/meta_instruct/tool_reply_format_check.json` |
| Caving replies | `results/meta_instruct/caving_replies_sample.json` |
| Base letter check | `results/base/letter_check.json` |
| Raw rows (not in git, ~140 MB) | laptop `runs/sycophancy_eval_meta_instruct/scores.jsonl` |
| Advisor readout test | laptop `runs/investigation_readout/readout_compare.jsonl` |
| Logs | laptop `logs/lambda_2026-11-04/` |
| Spec, decisions, paper audit | `papers/2610.28325.md`, `papers/2610.28325_templates.md` |

## 8. How to repeat

```
scripts/lambda_up.sh <ip> sycophancy-eval
ssh lambda 'cd ~/latent-knowledge-readout && uv run --no-sync python scripts/run_sycophancy_eval.py --checkpoint meta_instruct --allow-tf32 --batch-size 8 --ignore-preflight-problems'
ssh lambda 'cd ~/latent-knowledge-readout && uv run --no-sync python scripts/rerun_tool_rows.py --checkpoint meta_instruct'
```
The run script resumes from saved rows after a crash and refuses a different question split.

## 9. Not done

- Scribe notebook with three random raw examples per cell and a summary cell (research rules).
- Critic check before anything enters the ledger's Claims.
- The decision on the tool row (12-source pool as the gate number?), the neutral filter, and the CSQA doubt check.
- Questions for Orgad: the dataset split, the exact tool format (description, call turn, reply rendering).
