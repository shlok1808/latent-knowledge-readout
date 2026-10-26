# Run design: deep research, round 2 (DRAFT v2, not started)

Drafted 2026-10-22 from the `next-run-spec` note (v3), revised the same day. Nothing here runs until the user says "go". Skill: `deep-research`, Opus for every agent.

## 0. Question and frame
Core question (method-neutral): when correctness-relevant information is represented while a model acts against it, what role does that information play in producing the action? Candidate fates, not a complete list: absent or misrepresented; present but not causally used (bypass); outweighed by a competing signal (override); transformed into another decision-relevant state; inaccessible to the decision.

- Direction 1, bypass vs override: in depth.
- Direction 2, where shared geometry breaks: in depth. A manifold is one idea, not an assumption. First say what spaces are compared and what relation should be preserved.
- Direction 3, jump vs drift: light. Separate from single-turn unless evidence connects them. Keep agent papers that bear on the core question.
- Goal: actionable. Every finding is tied to a fix or to a reason there is none.
- Left out: defining long horizon, lab-by-lab surveys, a broad dataset hunt beyond what the shortlisted questions need.
- Say "correctness-relevant information is decodable", not "the model knows", unless a paper shows more.

## 1. Run shape
- One single run, no time limit. Take as long as the work needs.
- 27 agents in total including the watcher, hard ceiling 27 (user's limit, 2026-10-22).
- Effort: sweep researchers, re-score scorers, deep readers, fixes researchers: medium. Reviewer and writer: high. Fact-checkers: medium. Opus for every agent.
- Phases are ordered so the most valuable parts finish first (Core papers, directions 1 and 2, fixes). The reset window is 2-3 hours, so the cheap-to-lose bulk work (re-scoring 806 papers) comes after.
- The only planned stop is the checkpoint after the sweeps and recall test (phase 3).

## 2. Setup
1. Seed list = the Known-Relevant list (section 8). It is also the recall test.
2. Access: OpenReview through the user's Chrome, read-only, no sign-in. The user answered "yes" earlier; the latest message left a "[yes / no]" placeholder unfilled, so I treat the earlier yes as standing. If still blocked, download the PDF into its own new folder (one folder per download), stating filename and source, and read it from there.
3. Semantic Scholar or OpenAlex for citation chasing, with retries. If no key works, say so; do not silently skip chasing.
4. Workspace: `research_notes/Knowledge action divergence internals round 2/`, one file per phase, so a pause loses nothing.

## 3. Phases (in order)
| # | Phase | What | Agents (effort) | Gate |
|---|---|---|---|---|
| 1 | Usage check, then search sweeps | Synonym families run separately: know-do; knowledge-prediction; representation-action; latent/hidden knowledge; decoded but unused; readout. Also: override / competing signals; null space / bypass; transformed / flipped representation; inaccessible; fixes. Every query logged with hits and new-paper count. | 4 (medium) | Query log saved |
| 2 | Recall test | Test the sweep results against the Known-Relevant list before any deep reading. Missing items mean the search is broken: fix and rerun phase 1. | 1 (medium) | All found, or the misses explained |
| 3 | **STOP: checkpoint** | I show what was found (new papers, recall result, query counts, what the sweeps missed) and you redirect. Nothing continues until you say go. | none | Your go |
| 4 | Usage check, then Core deep read, directions 1 and 2 first | Per paper: the extractions in sections 5 and 6. Related-work extraction is mandatory and feeds phase 7. Direction 3 gets one light reader. | 4 (medium): 2 on direction 1, 1 on direction 2, 1 on direction 3 | One table row per Core paper |
| 5 | Usage check, then fixes strand | Training recipes that open or close the gap, honesty/consistency training, training against detectors (when it teaches hiding), whether fixes generalize, capability cost. | 2 (medium) | Each fix: tried by whom, worked or not, side effect |
| 6 | Usage check, then citation chasing both ways | From every anchor and every Core paper; every unfollowed related-work citation goes on the to-check list. | 2 (medium) | List empty, or holding only clearly off-topic papers |
| 7 | Usage check, then re-score all 806 papers | Old `closeness` ignored entirely. One Opus scorer at medium effort per paper, batched about 115 papers per agent. Scorers see title, abstract and our question only. An arbiter (high not needed; medium) only on borderline scores (2 vs 3). Papers the old CSV lacks are added. Newly surfaced 3-4 papers get one late deep reader. | 7 scorers + 1 arbiter + 1 late reader (medium) | Score histogram reported; every Known-Relevant paper scores 3+ or is explained |
| 8 | Usage check, then reviewer, before the writer | Checks in section 7. | 1 (high) | Sign-off or an applied fix list |
| 9 | Usage check, then writer | Outputs in section 9. | 1 (high) | |
| 10 | Usage check, then fact-check | Every claim, number and quote checked against the source after writing. | 2 (medium) | Corrections applied; changes listed |

Total: 4+1+4+2+2+7+1+1+1+1+2 = 26 agents, plus 1 watcher = 27 (ceiling 27).

Watcher (added 2026-10-22): one Opus agent at medium effort, continued after every phase. It compares each phase's files to that phase's gate, and flags drift from the brief, repetition, stalls, recall misses, and steps that were skipped or blocked without a note. One line per flag in `watcher.md` in the run folder: phase, agent, problem, suggested fix. It only flags: it spawns nothing and changes no other file.

Usage (revised 2026-10-22): no pause before the user's reset. After the reset, stop the run around 95% used, save phase files and `RESUME.md`, and resume when the user says go.

## 4. Usage rule
- Check usage before every phase (the session usage tool).
- Never stop mid-phase.
- If usage is about 85% or more at a phase boundary: save the phase files, tell you, and wait while you use your reset; then continue from the saved state.
- Never drop depth, switch off Opus, or skip the reviewer or fact-check to save usage.

## 5. Re-score rubric (phase 7)
- 4: tests the core question directly on internals (the role the represented information plays in the action), or tests a fix for it.
- 3: gives a method, control, dataset or result that bears directly on one direction.
- 2: related setting (behavior-only evidence, agent failure with no white-box part) that frames a direction.
- 1: same words, different question.
- 0: off-topic.
Borderline = scores that sit on the 2/3 line.

## 6. Per-paper extraction (every close or partly useful paper)
1. What it did: models, setting, task, scale.
2. Related-work section: every cited paper that bears on the question, each added to the to-check list.
3. Gaps: what it did not do.
4. What the authors suggest next (quote limitations and future work), and whether anyone has since done it (forward citations).
5. For partly useful papers: what we can borrow (method, metric, dataset, pitfall).

Methodology and evidence harvest (Core and Supporting):
- Protocol: data, prompts, token positions read, layers, lenses, baselines, controls.
- Controls worth copying: aliasing control, question-level cross-validation, matched twins, lexical baselines, permutation nulls.
- Metrics and how they were validated; datasets with size, license and base rates.
- Pitfalls the authors hit or warn about.
- Evidence for or against each direction, with the paper's own caveats.
- Reuse verdict: copy / adapt / avoid, with reason.
- Checks from the user's doc (they test paper claims, they do not steer search): causal controllability versus endogenous mediation; whether "not used" rests on one probe direction only or rules out redundant and nonlinear routes; whether a gradual state can cross a decision threshold abruptly.

## 7. Search hygiene and reviewer checks
- Stop searching only when 10 consecutive queries across at least 3 families find nothing new.
- Reviewer flags any paper cited by 2+ anchors, or citing 2+ anchors, that is not in the notes.
- Reviewer checks for the old probe bias: evidence is reported by method-neutral fate first, then by method.
- Reviewer checks every "nobody has tested" claim with a targeted search before it is written.
- Reviewer checks that workshop papers and abstract-only papers are labelled in the evidence strength.

## 8. Known-relevant list (the recall test)
KAPPA 2509.23782 · Where Do LLMs Decide to Break the Rules? 2609.37737 · When Truthful Representations Flip Under Deceptive Instructions (EMNLP 2025) · Decoded but Unused (OpenReview da9ylT2doy) · Readout-Mediator Angle 2605.29126 · Answering the Unanswerable Is to Err Knowingly 2508.18760 · LLMs Know More Than They Show (Orgad et al.) · Inside-Out 2503.15299 · Few-Shot Demonstrations Elicit Use of In-Context World Representations 2609.24352 · Senses Wide Shut 2605.13737 · Matryoshka Attribution 2609.25518 · anchors 2605.14038, 2607.05013, 2610.17301, 2605.05115, 2607.00415, 2604.19117, 2602.04918, 2610.21708, 2609.04582, 2608.12321.

From the user's doc: Oracle-State Intervention (aU7tabEHsu) · Lost in Distraction (IrfhtJkhmV) · ContextRotBench (DiZ67nkfay) · Models Recall What They Violate (9kbTg0tKKr) · Reasoning Fine-Tuning Induces Persistent Latent Policy States (H0MwQr9XMW) · A Mechanistic View of Authority Hierarchy in LLM Sycophancy (arXiv id to find) · geometry papers 2607.18532, Trajectory Signatures of Deception, Curved Inference II 2608.24037, Geometry of Divergence 2608.30650 · round-2 reading 2607.00415, 2609.10739, 2604.19117.

Cammarata, "Neural Networks Are Already Clean" (Goodfire, published 15 October 2026): on the list only so the run finds it. No researcher and no phase for it, and it must not steer the searches.

## 9. Outputs
1. Per-paper table, plus the to-check list (ends empty or off-topic).
2. Methods Playbook: what to copy, in order of use.
3. Evidence Ledger: claim -> supporting and contradicting papers -> strength.
4. Actionability map for direction 1, per fate (fails to affect / overridden / transformed / inaccessible): the fix it implies (training-time: data, objectives, post-training recipe; inference-time: steering, gating, readout repair; monitoring; architecture), what has been tried, whether it worked, costs (capability loss, learning to hide).
5. Shortlist of 3-5 research questions, each with: why it is open, the actionable result if it works, the smallest first experiment, null-result risk, scoop risk.
6. Kill criteria per direction.
7. Updated report in the round-1 style, short summary first, details below.
