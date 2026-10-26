# Handoff for the next chat (written 2026-10-22)

Read this first. Then ask the user what they want to do; do NOT launch anything big until they say so.

## The project in one line
Studying what happens to what a language model knows at the moment it acts against it, with the goal of actionable interpretability for safety. Repo: github.com/shlok1808/latent-knowledge-readout (local folder: ~/latent-knowledge-readout; old project archived on branch `archive/stage-transfer-v1`, tags `stage-transfer-v1`, `main-before-clean`).

## Where everything is
- Full report (fact-checked, 12 sections): https://claude.ai/artifact/YFfm74y97wSwTYJommvhaB ; source: ~/latent-knowledge-readout/research/reports/Knowledge action divergence internals.md
- Research Notebook (tickable tasks + notes inbox, saved in a database): https://claude.ai/artifact/WvXjUcMwbjsgUbQxUR5Fti . Read with the ArtifactData tool (collections `notes`, `tasks`). Important notes there: `next-run-spec` (the full run design), `known-relevant-papers`, `scan-misses-2026-10-20`, `kappa-2509-23782`.
- Research notes behind the report: ~/latent-knowledge-readout/research/research_notes/Knowledge action divergence internals/ (9 files incl. _review.md)
- Paper database from the first scan: ~/latent-knowledge-readout/research/papers_master.csv (806 papers)
- Sidebar pane mods (backed up): ~/latent-knowledge-readout/research/mods/ (candidate-directions, research-skills). They load only in the session that has hot-reloading enabled; to keep them, reload the plugin-authoring skill and install them.
- Page-building scripts: report HTML is built from the markdown with a script that lived in the old scratchpad; rebuild if needed (mistune, wraps tables, adds comment buttons).

## The three directions (user's wording; method-neutral, "just ideas")
1. Bypass vs override: does correctness-relevant information fail to affect the action, get overridden by competing signals, get transformed, or become inaccessible to the decision?
2. Where shared geometry breaks: does the relationship between representation and behavior change when a model acts against its apparent knowledge? A manifold is one possible idea, not an assumption.
3. Jump vs drift (later, light depth): gradual or abrupt change in multi-step tasks.

## Decisions and lessons (important)
- The user does NOT want to pick a method (probes, manifolds, anything) before the research question is fixed. Question first, method second. My earlier reports leaned toward probes because the literature does; treat that as a bias.
- Goal: actionable. Findings should point to a fix (training, inference, monitoring, architecture), not only a description.
- Dr. Lin (advisor) wants: larger models (not 7-8B), factual/consequence-bearing tasks, single-turn first, define "long horizon" before using it. Her note: the user should check the Goodfire essay (published 15 Oct 2026) "Neural Networks Are Already Clean" (Cammarata); the user likes it as a possible angle.
- The first deep-research run MISSED KAPPA (2509.23782) because researchers trusted the scan's low closeness ratings and one vocabulary. Fixes are in the spec: re-score all 806 papers ignoring old ratings, synonym families, finish citation chasing, test the run against the known-relevant list before writing, cross-citation check, stopping rule.
- Papers the user has read: KAPPA, 2610.21708, Readout-Mediator Angle (2605.29126), and the four geometry papers (2607.18532, Trajectory Signatures of Deception, Curved Inference II 2608.24037, Geometry of Divergence 2608.30650). Round 2 reading (not done): 2607.00415, 2609.10739, 2604.19117.
- Verdicts so far (see report): erasure claim (2607.00415) is plausibly a probe artifact; the knowledge usually survives and is overridden downstream; direction 1 partially done with scoop risks (KAPPA, 2609.04582, KnownLieBench pilot); direction 2 open but high null risk; direction 3 partly scooped by a COLM 2026 paper.

## Next step
1. The user will send a doc of their own thoughts (from an AlphaXiv chat). Check it against the field; it does NOT steer the searches.
2. Draft the full run design from the `next-run-spec` note and get approval. Open question: may a researcher use the user's Chrome to pass OpenReview's bot check (last time it did, read-only, no sign-in)? Scope: directions 1 and 2 in depth, direction 3 light.
3. Only after the user says "go": run `deep-research` (Opus for every agent, reviewer before the writer, fact-check after, check usage between phases; pause and resume if the limit runs low; user said never let the limit hurt quality).
4. Use `creative-thinking-for-research` for brainstorming the question, not for the literature run.
5. The user wants Colloquium for any slides (see global CLAUDE.md).

## Working preferences
- Short plain answers; the user is a student new to interpretability and reads between midterms. Give summaries first, details on request.
- After any big process, keep a task list and end with a recap. Name the skill used at the start of research work.
- Never delete or push without asking. Answer cards sometimes do not reach me: if an answer is missing, ask again in plain text.
- Artifacts: only keep what is needed. The Brief was removed from the links; the user must delete it themselves (the app only lets them confirm deletes).
