# Project context (read this first in every new session)

Last updated 2026-10-23. Sources: the handoff note in the Knowing-Doing Notebook and the user's highlight notes on the report page.

## Where things live

| What | Link |
|---|---|
| Notebook (checklist + notes inbox, notes in db collection `notes`) | https://claude.ai/artifact/WvXjUcMwbjsgUbQxUR5Fti |
| Round 2 report, user's highlight notes in db collection `notes` | https://claude.ai/artifact/3LFFZqyEQNx6Vqw3ysFgHa |
| Deck for Dr. Lin, 23 Oct (slide PNGs + PDF) | https://claude.ai/artifact/4bWqg7cKMYqYUXhRwUFsuw |
| Round 1 report / brief | https://claude.ai/artifact/YFfm74y97wSwTYJommvhaB, https://claude.ai/artifact/N16sXwwXJijaAWdC899YiH |

Read the db notes with the ArtifactData tool. Laptop-only (not reachable from cloud): `~/colm-guide` (deck source in `gapscan/lin_meeting_2026-10-23/`, run design in `gapscan/reports/Run design.md`), the local CLAUDE.md.

## Project

What happens to correctness-relevant information inside an LLM when it acts against it. Goal: actionable interpretability for safety (findings should point to a fix).

## Findings so far (round 2 fast pass, ~70 papers, recall 22/31)

- The correct signal usually survives; routing to the action breaks.
- Six fates: readout/threshold, bypass, not routed, override, transformed, erased (rare).
- Control: logit margin. A probe adds only +0.008 AUC at 8B (2609.04582).
- Masking: RLHF on Llama-70B cut sycophancy 39% to 3.5%, but head-ablation effect grew +10.5 to +27 pp (2604.19117).
- Natural route (endogenous mediation) is essentially untested; the one 72B test is null.
- Scoop risk: 2610.21708 (15 Oct). Medium risk: KAPPA 2509.23782.

## Agreed plan

- Stage A: sort each failure into a fate with 6 checks, cheapest first (output scores; bypass both-way push + DAS angle; injection + twin-input natural use; competitor ablation vs random; old probe below chance; fresh-probe erasure).
- Four pressure conditions on shared facts: 1 competing answer, 2 user pushback, 3 no single answer, 4 forced to answer vs allowed "I don't know". Start with 1+2.
- Stage B: apply the fix each fate implies plus plain DPO, rerun Stage A: removed or only hidden?
- Add-ons: SAE / ICA Lens (2606.11722) features alongside probes; per-item knowing-vs-acting angle as kill test for the geometry direction.
- Later: instructed-deception condition; multi-step after single-turn.
- Models 9B-27B (~30B affordable); 70B only if Dr. Lin has compute. First run: Gemma-2-9B, 400 TriviaQA pushback pairs from 2604.19117, ~1 GPU-day.
- Dr. Lin wants: larger models, factual tasks, single-turn first.

## Next

1. Get Dr. Lin's answers: compute, start with 1+2?, deception condition?, move fast vs 2610.21708?
2. Before claiming novelty: citation chasing both ways, full reads of abstract-only core papers, read 2505.11770.
3. Seal a fate-split prediction before the first data collection.

## User's open questions from the report (22 Oct)

1. Has anyone tested this on deceptive examples, or tasks that are hard for the model?
2. Why the six-fate sorting test as the best next step?
3. Why does everyone use probes? Wouldn't SAE features or an ICA lens be better for finding something hidden?
4. Has anyone actually found the "agree with the user" signal?
5. If steering a direction changes behavior, why does it matter whether the unmodified model uses it?
6. KAPPA: steering helped and its subspace differs from the unembedding, so it steered representations, not logits?
7. Unanswerable-but-answered: bypass or not routed? Fold into idea 1.
8. Build a manifold of override situations?
9. Fold the RLHF masking test into ideas 1-2 as the actionable step.

## Working preferences

Short, scannable replies; next action first; one question at a time. Slides are made in Colloquium. Never push, delete or publish without asking.
