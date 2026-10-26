# Update check: deck brought from 23 Oct to 26 Oct state

Branch `slides-update` (from `origin/research-sync`). Deck went from 12 to 16 slides.
"Brief" below = the 26 Oct state the user gave in the task (session background). Many 26 Oct decisions exist only there, not in any file under `research/`; those are marked **brief only**.

G = `research/reports/Group geometry methods for fates.md`
NP = `research/research_notes/Nikhil Prakash binding papers/`
R2 = `research/reports/Knowledge action divergence internals round 2.md`

## 1. Changes, slide by slide

| New # | Slide | Before → after | Source |
|---|---|---|---|
| 1 | Title | date "23 October 2026" → "23 October 2026 (updated 26 October)" (also frontmatter) | — |
| 2–6 | Problem, readable vs used, six fates, evidence map, two lessons | unchanged | — |
| 7 | Three predictions | H2 row labelled "(tested last)"; footnote adds that H1 is contested (2601.11563 one shared direction vs 2607.20146 separable modes) | order: brief only; H1 conflict: G:197, G §8 (line ~214) |
| 8 | Our study (fig_plan) | Stage B box "fix and recheck: apply the fix (and plain DPO)…" → "Stage B, last: … public checkpoints first (base vs instruct, OLMo SFT → DPO); own DPO / CCA only if needed". Footnote: models "Gemma-2-9B, Gemma-2-27B, Qwen2.5-14B" → "first Gemma-2-9B, then Qwen2.5-32B"; Stage B moved to end and why; SAE/ICA line moved to slide 9 | brief only (order, public checkpoints, Qwen2.5-32B). OLMo checkpoints exist: `research_notes/Knowledge action divergence internals/dynamics_and_training.md:69`. DPO cost: G:159 |
| 9 | **NEW** How we measure: small to big | one direction → subspace → token/layer paths; move up only if needed. Footnote: decodable ≠ knows; causal controllability ≠ endogenous mediation | brief only (ladder); vocabulary: brief + NP/nikhil-thesis-proposal-2026-10-26.md:34 |
| 10 | **NEW** Lessons from Nikhil Prakash | same-length pairs; one variable at a time; pointer + address together before "bypass"; beyond the answer token; CCA baseline + no-mask control; fine-tuning strengthens routes (prior) | NP/reading/2505.14685.md:20, :22, :16 and :23 (pointer patch); 2603.22278.md:26–27; 2512.16914.md:24, :32; 2402.14811.md:22 |
| 11 | Why it's worth the time | "one-day pilot checks this first" → "output-score screen checks this first"; scoop adds "decodability is not use is crowded in ICLR 2027 submissions"; cost line adds own DPO ≈ 40–70 GPU-h at 9B, so Stage B last | G:109 (screen), G:132 ("crowded"), G:159 (40–70 GPU-h). ICLR 2027 framing: **brief only** (G:132 says "crowded" without naming ICLR 2027) |
| 12 | First step: one-day pilot → **First experiments, ranked** | 4-step pilot replaced by table: 1 copy vs correctness (1–2 h), 2 plant fake fates (< 1 h), 3 output-score screen (≈ 0 h); #3 can run first | G:91–98, :100–107, :109–116; kill rules G:95, :104; ranking + "3 first": brief |
| 13 | **Split out** Still open: datasets, and my questions | Datasets marked undecided, draft named. Questions 4 → 3: compute/70B; which pressures; speed vs scoop. Dropped old Q3 "add instructed-deception later?" (not in the brief's question list) | brief only |
| 14 | Backup: geometry test | unchanged | — |
| 15 | **NEW** Backup: eight ranked ideas | table of G §5 ideas 1–8 with cost and scoop | G:87–163 |
| 16 | Backup: trust | adds "never met its stopping rule"; "no citation chasing yet, several core papers unread, so we claim no novelty" | R2:310, R2:286, R2:17 |

Other files:
- `figures/gen_story_figs.py`: Stage B text changed; font fallback list now includes Nimbus Sans / Liberation Sans / FreeSans (Helvetica Neue isn't installed in the cloud box, and DejaVu fallback made text overlap). Only `fig_plan` was regenerated (PNG at 220 dpi + SVG); other figures untouched.
- `speaker_notes.md`: rewritten for slides 8–16, small edits to 1 and 7.
- `slides.html`, `slides.pdf`, `slides/slide-01..16.png` rebuilt.
- `.pptx`: **not regenerated**. It was made with python-pptx (file metadata), but I couldn't confirm whether via `colloquium export --pptx` or a custom script, so it is now out of date (still 23 Oct content).

## 2. Missing items (in sources since 23 Oct, not in the deck)
1. Idea 3's second half, the margin-scaled threshold push, and its "blocked correct updates" item set (G:109–116). Only the free screen is in the deck.
2. Stage B design rules from G §8: compare fix arms only at matched flip-rate drop or matched KL, and report a base-vs-base noise floor (G §8, line ~220).
3. Stage A control ladder (logit margin, P(True), two-centroid detector, full probe) and "train detectors on the model's own pressured answers, not teacher-forced" (G §6, ~line 179–186). Teacher-forced → own answers drops AUROC 0.957 → 0.688 (G:93).
4. Nikhil's earliest-patch failure diagnosis (pair a wrong item with a matched correct one, patch variables in order) is framed in NP as *our extension*, not in his paper (2505.14685.md:22). Not on the slide.
5. 2505.14685 shows 7–8B models mostly fail the binding task, so the mechanism may not exist in our 9B range (2505.14685.md caveats). Relevant risk, not in deck.
6. OpenReview ZP7VzpXmKR still blocked by the bot check (`research_notes/Group geometry methods for fates/RESUME.md`).
7. Geometry report idea 5 (intervention ladder) has the highest remaining scoop risk (G §8). Only in backup table.

## 3. Deck vs sources: possible contradictions
- Slide 8/brief say "OLMo SFT/DPO stages"; G:159 says "OLMo-2-13B SFT/DPO/Instruct checkpoints (availability unverified)", while the earlier notes describe OLMo-3 base/SFT/DPO/RLVR. Which OLMo is meant isn't settled. Slide says just "OLMo".
- Brief: "Gemma-2-9B, then Qwen2.5-32B". Old deck said Gemma-2-27B and Qwen2.5-14B; G mostly says "9B, about 3x at 27B". Next model after 9B isn't consistent across files.
- Slide 11 "Stage A about 1 GPU-day per model per condition" is carried over from the 23 Oct deck; I did not find its source in this pass (G's per-idea costs are hours).
- Slide 5 evidence map and slide 6 numbers unchanged and not re-checked.
- `NEXT_CHAT_HANDOFF.md` is dated 22 Oct and stale (doesn't mention Stage B order, four pressures, Nikhil). Not edited (outside the deck).

## 4. Could not do or verify
- Could not trace to a file: four-pressure start order, Stage B-last decision, small-to-big measurement ladder, experiment ranking, "datasets not decided", the three questions, ICLR 2027 crowding, Qwen2.5-32B. All from the brief only.
- Slide 10 "patch pointer and address together before calling it bypass" is only partly traced: NP/reading/2505.14685.md:16 quotes that an address/pointer mismatch blocks the lookup, and :23 describes the pointer patch, but the explicit rule comes from the brief. Also :23 says the pointer test only applies when both candidates are in context (e.g. the model's own earlier answer in the transcript); the slide doesn't say that.
- Did not re-verify any paper numbers against the PDFs; numbers were copied from the reports and notes.
- `Run design.md` is the deep-research run design (22 Oct), not the experiment design; nothing in it needed to change the deck.
- No web check this pass (time limit).
- Capture needed a wrapper: `/usr/local/bin/chromium` → Playwright Chromium with `--no-sandbox` (root in container), plus `apt install ghostscript`.

## 5. Visual check
All 16 PNGs viewed. Fixed: Stage B text overflowing its box in fig_plan (shortened); wrong font in fig_plan (fallback list); experiments-table cost column wrapping (shorter header + no-wrap); eight-ideas title wrapping onto two lines so the footnote overlapped the table (shorter title). Final pass: no overflow, overlap, broken images or placeholders. Slides 9 and 10 are text-only with empty space at the bottom (readable, not cramped). Fig_plan's Stage B line is the smallest text on slide 8 but legible.
