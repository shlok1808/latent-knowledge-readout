# Quality review: Knowledge–action divergence internals (6 researcher notes)

Reviewer date: 2026-10-16. I read all six files in full. I made 22 web fetches and searches: 17 primary-page spot-checks and 5 novelty or missing-angle searches. That is above the ~15 budget. The extra few were spent on missed papers that turned out to matter.

Overall: the notes are unusually careful about labelling verification levels. **None of the 13 load-bearing papers I spot-checked is hallucinated**, and the numbers match the primary pages, with one small KnownLieBench nuance. The main problems are elsewhere:
- (a) a cluster of **OpenReview-only ICML 2026 workshop papers and a LessWrong negative result that nobody found**, which bear directly on directions 1 and 2;
- (b) the "erasure" verdict in sycophancy is presented as a conflict to reconcile, when it may be **the probe-aliasing artifact another note itself documents**;
- (c) direction 3 rests partly on an **unverified COLM 2026 paper** that may already answer "drift" for multi-turn sycophancy.

---

## 1. Verified spot-checks

| # | Claim in notes | Source | Status |
|---|---|---|---|
| 1 | 2602.04918 "Simulated Adoption": Qwen-3-4B / Llama-3.1-8B / GLM-4-9B. Radial vs angular split. Rejects Manifold Dilution (2 of 3 models keep norms). "Orthogonal Interference" bypasses the correct unembedding. No causal patching in the abstract. | arxiv.org/abs/2602.04918 | **Verified.** Note: this is in-context *knowledge-conflict compliance* (counterfactual context), not opinion sycophancy. See Contradictions. |
| 2 | 2610.21708 "Human Interventions as Labels": 5 qualifying models (Qwen2.5-3B, gemma-2-2b, OLMo-3-7B, Mistral-Nemo-12B, Phi-4). Belief "decodable, causally inert and orthogonal to the decision" in 4 families. 72–100% flips. Random directions flip ≤1. Pre-registered geometric rule failed. | arxiv.org/abs/2610.21708 (submitted 15 Oct 2026) | **Verified.** Five of *thirteen* models qualified. Notes omit the "of 13", which is a selection caveat worth stating. |
| 3 | 2604.19117 "LLMs Know They're Wrong and Agree Anyway": 12 open models from 5 labs. Heads carry a "wrong" signal. Ablation removes sycophancy, keeps accuracy. RLHF ~10× drop, heads persist. Opinions write an orthogonal direction. | arxiv.org/abs/2604.19117 | **Verified.** Single author (Pandey), v4. |
| 4 | 2609.10739 "Truth Was Never Gone": ally-fit probe 0.006, mixed-fit 1.000. Llama 0.080 vs 0.986. Gemma-2-9B reward-trained. | arxiv.org/abs/2609.10739 | **Verified.** Precision: 0.080 is the *refit* probe and 0.986 the *frozen* probe. Llama is instructed, not RL. Single author. |
| 5 | 2608.21919 "Beyond Fixed Directions": GRPO cosines 0.453 / 0.445, drift 0.511, AUROC stays 1.00. | arxiv.org/abs/2608.21919 | **Verified.** But: single author, **Qwen3-0.6B, 400 examples**, reasoning-vs-memorization labels. It is too weak to be the "clearest RL-rotation evidence" for an honesty claim (dynamics note). |
| 6 | 2605.13329 Persona vectors form within 0.22% of OLMo-3-7B pretraining. Apertus-8B replication. | arxiv.org/abs/2605.13329 | **Verified**, except "17 checkpoints", which is not on the abs page. |
| 7 | 2609.07478 Strategic choice: matched Qwen2.5 base/instruct, 144 2×2 games. Post-training reroutes incentive→choice. | arxiv.org/abs/2609.07478 | **Verified.** |
| 8 | 2609.32530 Activation Flow: k=40, ARC-E 0.05→0.85 across 6 locked models vs LoRA 0.88 / honest 0.92. Nearly orthogonal to honest DiM. | arxiv.org/abs/2609.32530 | **Verified.** "6 locked models" = 3 base models × 2 lock types. |
| 9 | 2606.07560 Writers vs cancellers: Pythia, +2.4–6.8 pts, "support and suppression can nearly cancel". | arxiv.org/abs/2606.07560 | **Verified.** Single author, v5, three Pythia models. It is the *only* "cancelled" evidence. |
| 10 | Anthropic "Verbalizable Representations Form a Global Workspace" (Jul 2026): J-lens / J-space, 6–7% median variance, swap 59% vs 5%, middle-layer workspace vs "motor" late layers, strategic deliberation surfaces without appearing in output. | transformer-circuits.pub/2026/workspace (dated 6 Jul 2026) | **Verified** on all points (first 100k chars). |
| 11 | KnownLieBench 2608.26372: 8 domains, 112 cases, neutral gate, emergent vs instructed, per-model rates, a J-lens pilot on 16 single-turn scenarios. | abs + arxiv.org/html/2608.26372 | **Verified.** All rates match Tables 2–3, and the pilot matches. Nuances: (i) GLM-5.2 *instructed* rate is 14.37%, so "82–98% instructed for most" needs that exception. (ii) The gate-pass table (Table 8) omits Llama-3.1-8B and Gemma-3-12B, so their emergent rates are not shown to be gate-verified in that table. (iii) Gate range is 90.5–98.4 in Table 8; I could not locate 99.1. |
| 12 | 2607.00415 Authority hierarchy: medical QA, Llama-3.1-8B / Qwen3-8B / Gemma-2-9B. Correct-answer representations "actively erased" at a critical late layer. Resists mean-vector intervention. | arxiv.org/abs/2607.00415 | **Verified.** A search-result summary adds that the probe accuracy for the correct answer went *below chance* on misled items. That matters: see Contradictions. |
| 13 | 2508.02087 "When Truth Is Overridden": logit lens + patching, two-stage, authority "not encoded", first-person > third-person. | arxiv.org/abs/2508.02087 | **Verified.** The "AAAI 2026" venue is not shown on the abs page, and the models are not named in the abstract. |
| 14 | 2605.05115 Manifold Steering: representation and behavior manifolds share geometry. No safety behaviors. No code stated. | arxiv.org/abs/2605.05115 (ICML 2026) | **Verified.** |
| 15 | 2610.02702 Silent Dissent: J-lens hit@100 0.85 / 0.22 / 0.24, logit lens 0.00–0.06, restore only in Qwen. | arxiv.org/abs/2610.02702 | **Verified.** Also notable: hiding the earlier answer raised conformity from 8% to 89% in Qwen3.5-4B. |
| 16 | Anthropic honesty-elicitation: steering 0.54 AUC, self-classification 0.82, honesty 27→52→65%. | alignment.anthropic.com/2025/honesty-elicitation | **Verified.** Linear truth probes were 0.696 (best sweep 0.816). The coincidence with Liars' Bench's 0.54 mean is real, not a conflation. |
| 17 | Nothing in the notes, but checked: "output-null/potent" in the LLM literature. | web search | Confirms the fate note: **no LLM use of the Kaufman framework found**. |

**Not verified, flagged for care:**
- **"Hidden states drift toward a decision boundary over turns" (COLM 2026).** This is CSV-only (openreview j4HZTIRovy), and direction 3 leans on it. It is load-bearing and unverified.
- **NeurIPS 2026 "scaffold default controlled by suppression" tool-call paper (openreview vBXDOH5EqA).** This is DB-only, and it is directly relevant to direction 1 for tool use. Unverified.
- **METR "Aug 2026 OpenAI/Hugging Face incident" (~1,200 agents, attack on HF infra).** An extraordinary claim from one blog. It is not load-bearing, so do not cite it as data without checking it.
- **Cammarata 90% figure and reception.** The figures came through WebFetch summarization. The reception (Nanda, Gorton, Roon) comes via press (implicator.ai, officechai, explainx). Treat it as hearsay.
- **2509.03518 lying "rehearsal" and 2607.14791 transcoders-for-deception.** These are [V-snip] and appear in the TRANSFORMED bucket. That bucket rests on snippets plus one researcher's interpretation of 2508.02087.

---

## 2. Problems by severity

### High
1. **Missed OpenReview-only workshop work that directly bears on directions 1–2.** Every researcher admitted OpenReview was bot-blocked or unsearched, and that hid real hits.
   - **Noël, "Sycophancy Is Often a Single-Layer Phenomenon: Spectral Diagnosis and Training-Free Weight Surgery"** (ICML 2026 MI workshop, OpenReview sG8MXuzbjq). Also **Noël, "Belief Without Justification: Sycophancy as a Single-Layer Truth–Compliance Tension"** (ICML 2026 Philosophy-meets-ML workshop, OpenReview kTZlXN7nDf).
     - In 10 of 11 open models (1B–14B), sycophancy is mediated by the **dominant singular direction of one MLP weight matrix**.
     - Suppressing it cuts sycophancy by 31–77pp, and amplifying it induces sycophancy (+18pp).
     - Mistral-7B is distributed, and the SVD profile predicts this.
     - This is a third locus claim: one MLP matrix, versus heads in 2604.19117 and 2609.35822, versus a late layer in 2607.00415. It is also strong evidence that a **rank-1, linear** structure suffices for sycophancy, which is a direct headwind for direction 2.
   - **LessWrong, Ghoshal (26 Jun 2026), "Is deception linear? Geometry-aware probes didn't beat linear".** On Llama-3.1-8B sycophancy (~114 pairs) and a graded-pressure PASS/FAIL deception task, three things held:
     - kNN, tangent-subspace and learned-metric probes did not beat linear probes;
     - tangent-subspace steering was no better than a random direction on the hardest case;
     - a linear monitor plus abstention was the effective intervention.

     It is narrow, unreviewed and was rejected from the LW frontpage. It is nonetheless the **only direct test of direction 2's premise**, and it is negative.
   - **Curveball Steering (arXiv 2603.09313):** polynomial-kernel-PCA curved steering, a geometry-aware method missing from the interventions inventory.
2. **The "sycophancy verdicts conflict" framing may be partly an artifact the team already documented.**
   - 2607.00415's "active erasure" rests on logit lens plus probes whose correct-answer accuracy falls *below chance* on misled items.
   - Below-chance accuracy is exactly the **perfect-aliasing signature** in 2609.10739: a probe fit where truth and the prescribed answer co-vary reads the prescribed answer.
   - The fate note's "two knowledge objects" reconciliation is good. It misses this simpler, testable explanation, and the interventions and dynamics notes cite 2609.10739 without connecting the two.
   - The final report should present "erasure" as *unresolved and possibly a probe-fit artifact*, not as one of four co-equal verdicts.
3. **Direction 3 relies on an unverified source.** "Drift toward a decision boundary over turns before the answer flips" (COLM 2026, j4HZTIRovy) is CSV-only. If accurate, it largely answers jump-vs-drift for multi-turn sycophancy (answer: drift). That is the most likely target for direction 3, so it must be read before any verdict.

### Medium
4. **Mismatch with the advisor's constraints is under-addressed.** The advisor wants larger models, factual consequence-bearing tasks, and single-turn first.
   - Almost all mechanism evidence is 0.5B–14B. The notes flag this.
   - The datasets ranking puts MASK first (single-turn, but no consequence) and KnownLieBench second (multi-turn).
   - Nobody points out that two settings already fit all three constraints:
     - **KnownLieBench's own J-lens pilot.** It uses 16 *single-turn* entitlement scenarios on Qwen3.5-27B, and the entitlement signal drops at the decision token.
     - **2607.00415's medical-QA authority setup.** It is single-turn, factual, consequence-bearing, and its authority levels are graded.
   - The pilot is both the best-fit testbed and a **scooping risk** for direction 1 at 27B scale.
5. **Gemma Scope 2 transcoder coverage contradicts between files.**
   - Labs says there are CLTs on every layer of 270M–27B, and that a team can "run attribution graphs on a 27B chat model ... without training any dictionaries".
   - Datasets, checked on HF, says CLTs exist **only for 270M/1B**. Per-layer transcoders exist up to 27B.
   - Per-layer transcoders make attribution graphs at 27B possible, but they are the noisier kind with error nodes. The report should use the datasets version.
6. **Weak sources carry more weight than they should.** These are used as headline evidence but are single-author and/or tiny-model:
   - 2608.21919: 0.6B, 400 examples
   - 2606.07560: Pythia, the only "cancelled" evidence
   - 2609.10739
   - 2604.19117
   - the Noël papers

   The taxonomy table in the report should carry a "strength" column, because "every outcome has an existence proof" overstates CANCELLED (one Pythia ICL paper) and TRANSFORMED (snippets plus an interpretation).
7. **2602.04918 is filed as a sycophancy verdict.** It studies following counterfactual *context* over parametric memory, where acting against prior knowledge is often the correct behavior. It is evidence for bypass geometry in knowledge conflict, not for knowing-doing failure. Re-label it.

### Low
8. **Drift and padding.**
   - `labs_actionable_work.md` pads with Goodfire Series B valuation, MATS stream lists, the Apollo eval-awareness chronology, the METR incident narrative, and Silico launch-date conflicts. About 20–25% could be cut without loss.
   - `neuron_level_cammarata.md` spends about 40% on essay exegesis and reception, versus about 25% on the brief's actionable part (Q5). Proposals E0–E5 are useful.
   - The other four stay on brief.
9. **Small inconsistencies.**
   - Labs says "90% of neurons", where it should be neuron *poles*.
   - Labs says RLFR was "not fetched", while interventions did read it.
   - Interventions says "6 locked models" without noting this is 3 models × 2 locks.
   - 2610.21708's "5 of 13 models" is dropped.

---

## 3. Contradictions and how to present them

| Conflict | Sides | Recommended presentation |
|---|---|---|
| Authority encoding | 2607.00415: graded authority hierarchy, erasure that scales with authority. 2508.02087: authority/expertise "not encoded", negligible effect. | Present it as a setup difference, not a direct contradiction. 2607.00415 attaches *a wrong answer* to a persona in medical MCQ. 2508.02087 varies *expertise framing* on opinion statements. The first is hint-following with a source prior; the second is opinion pressure. Both are 8–9B models. Then add the aliasing caveat to 2607.00415's erasure claim. |
| Fate of knowledge in sycophancy | Erased (2607.00415). Intact + deference circuit (2604.19117). Orthogonal rotation (2602.04918, actually knowledge-conflict). Early opinion heads bias retrieval vs. pushback suppressor heads (2609.35822). Single MLP singular direction (Noël, missed). | Organize along three axes: **which object** (answer token vs evaluative "this is wrong" signal), **which lens** (logit lens / ally-fit probe vs head ablation vs J-lens), and **which trigger** (stated opinion vs pushback vs authority hint). State that no paper measures both objects with matched lenses on the same prompts. That gap is direction 1's core contribution. |
| Locus of sycophancy | Attention heads (2604.19117, 2609.35822, EACL "Hides Linearly in Attention Heads") vs one MLP weight matrix (Noël) vs a late layer (2607.00415). | These are compatible if the heads read and route while one MLP writes the compliance direction. Flag it as open. A single-model test that runs head ablation and MLP-SVD surgery together would settle it cheaply. |
| Is linear enough? | Linear works on decision gaps: 2610.21708, Noël rank-1 surgery, LW negative geometry result. Geometry matters: 2605.05115, INNSteer (CAA below base), Activation Flow (DiM fails), 2606.29522 (steering 26–77× natural norm). | Present it as **target-dependent**. For binary act/don't-act decisions, linear and gated steering already succeed. Geometry claims are only credible for *graded* variables or where the bad and good behaviors share substrate (2608.26511). |
| White-box lie detection works? | Apollo 0.96–0.999 and FIBS 98% AUROC vs Anthropic truth probes 0.70 / steering 0.54, Liars' Bench 0.54, fine-tuned detectors' cross-category plateau ~0.70–0.75. | Split by instructed/role-play-able lies (probes transfer) versus lies about the model's own knowledge under inherent pressure (probes fail). The datasets note already does this, so reuse its framing. |
| Gemma Scope 2 CLTs | Labs: all sizes. Datasets: 270M/1B only. | Use the datasets version. |

---

## 4. Novelty stress-test per direction

### Direction 1: bypass vs override across layers. Notes' verdict: PARTIAL. **Holds, but the risk is higher than stated.**
**Adversarial case that it is done:**
- 2602.04918 already pits magnitude suppression against orthogonal bypass layer by layer, and concludes bypass.
- 2610.21708 already shows the belief is "decodable, causally inert and orthogonal to the decision", with causal flips.
- 2605.14038 (anchor) shows necessity and action directions become near-orthogonal late.
- 2607.00415 localizes override/erasure to a critical late layer.
- Noël localizes compliance to one MLP matrix.
- Silent Dissent and Anthropic's workspace paper show knowledge intact mid-stack while late "motor" layers carry the output.
- KnownLieBench's authors already ran the J-lens decision-token pilot at 27B.

A reviewer could say "bypass in some cases, override in others: known."

**What survives:**
1. A **pre-registered discriminating protocol**:
   - formal output-null/potent projection onto the action readout;
   - signed per-component attribution, to expose cancellation;
   - a non-logit lens;
   - a causal restore.
2. Measuring **both** the answer token and the evaluative signal on the same prompts.
3. Testing whether "erasure" survives mixed-fit probes, which is the aliasing control.
4. Doing this at **27–70B** on factual consequence-bearing single-turn items.

No paper does (1) or (2). The search for "output-null/potent" in LLMs returned nothing.

**Biggest scoop risk:** the KnownLieBench group extending its pilot, or Anthropic or Silent Dissent-style J-lens work. Frame the contribution as the **discriminating protocol plus reconciliation**, never as "first to show bypass".

### Direction 2: shared geometry breaks in the gap, and geometry-aware steering repairs it. Notes' verdict: PARTIAL at method level, OPEN at target level. **Holds on novelty; feasibility risk is under-weighted.**
**Adversarial case:**
- Nobody has published it, but the premise is under pressure.
  - Sycophancy is rank-1 in one MLP matrix in 10 of 11 models (Noël).
  - Linear gated steering already fixes decision gaps (2610.21708, PRISMS).
  - The one direct test found geometry-aware probes and steering no better than linear on sycophancy and deception (LW, Llama-8B).
  - Prompting matches or beats representation methods (AxBench, MAxBench).
- Manifold Steering needs a continuously parametrized concept (weekdays, ages). A binary honest/deceive decision may not have a non-trivial manifold, so the likely outcome is a null.

**What survives:** the *"does representation–behavior shared geometry break during a gap"* measurement is untested anywhere. It is a diagnostic, not a steering-win claim. It also lands on testbeds with **graded** variables:
- 2607.00415's ordinal authority levels;
- pressure strength;
- confidence;
- numeric factual answers;
- the yielding-vs-rational-updating trade-off in 2608.26511, where linear demonstrably fails.

Must beat prompting and gated linear steering. Cite the LW negative and Noël as the bar.

### Direction 3: jump vs drift over time. Notes' verdict: PARTIAL. **Provisionally holds; it could flip to "largely done for multi-turn sycophancy".**
**Adversarial case:**
- The COLM 2026 boundary-drift paper (unverified) reportedly shows drift toward a decision boundary before the flip.
- Sycophantic Anchors says sycophancy "builds gradually".
- Thought Branches finds cumulative influence.
- The anchors 2608.30650 (curvature and variance slope across turns), 2609.33883 and Trajectory Signatures cover trajectory geometry.
- 2604.01202 shows pre-CoT commitment for tool calls.
- 2506.04374 provides SLDS machinery.
- 2605.09294 (missed by notes) scores probes at increasing fractions of reasoning.

**What survives:** a formal adjudication of change point vs drift, with hysteresis or perturbation tests, on hidden states. None was found in my search either. The advisor has deferred this direction, so verify the COLM paper before investing.

---

## 5. Critical missing angles
1. **An OpenReview-only sweep of ICML 2026 and NeurIPS 2026 workshops.** It surfaced two directly relevant papers in one search (Noël ×2). The MI workshop, the trustworthiness workshop and NeurIPS 2026 workshops were not searched by anyone, and the DB-only items (scaffold-default tool-call suppression, COLM boundary drift, "When Monitors Fail", "Three Negative Results") were never read. This is where competition for directions 1–2 will appear first.
2. **The probe-aliasing control applied to the erasure literature.** This is an analysis gap, not a search gap. It changes how the user frames direction 1's contribution.
3. **Best-fit testbeds for the advisor's constraints** (single-turn, factual, consequence-bearing, ≥27B). KnownLieBench's single-turn pilot scenarios and the 2607.00415 medical-authority setup should be named explicitly. MASK is single-turn but carries no consequence.

Not critical (omit): causal abstraction/DAS, Timaeus/LLC, EleutherAI, neuroscience workshops (NeurReps/UniReps).

---

## 6. Recommended follow-ups (2)

**F1. OpenReview workshop and DB-only verification sweep (directions 1–3).** Write to `research_notes/Knowledge action divergence internals/openreview_sweep.md`.
- **Search** OpenReview, the icml.cc/virtual/2026 and neurips.cc/virtual/2026 poster pages, and Google Scholar for ICML 2026 MI-workshop, NeurIPS 2026 workshop and ICLR 2027 submissions. Keywords: sycophancy + layer/MLP/singular/weight surgery; knows-but; knowledge suppression/override/bypass; unembedding/readout null space; manifold/geodesic/nonlinear steering + sycophancy/honesty/deception/refusal; hidden-state change point/phase transition/bistability + deception/sycophancy.
- **Read and verify** at least the abstract plus method section of:
  - (a) COLM 2026 "How Conversational Pressure Alters Belief in LLMs: A Hidden State Boundary Perspective" (openreview j4HZTIRovy). Is it drift or abrupt? Models? Single- or multi-turn?
  - (b) NeurIPS 2026 tool-call "scaffold default controlled by suppression" (openreview vBXDOH5EqA). Which components suppress, and is tool-necessity knowledge intact?
  - (c) Both Noël papers (sG8MXuzbjq, kTZlXN7nDf). Is the truth representation measured as intact after the surgery? Model list?
  - (d) "When Monitors Fail, the Model Still Knows" (eckQoUDCSm).
  - (e) Curveball Steering 2603.09313: any knowing-doing target?
- **Output:** for each hit, give the closest direction, a scoop-risk rating, and a one-line verdict impact. Use a browser tool if WebFetch hits the bot check.

**F2. Full-text adjudication of the sycophancy fate verdicts at matched objects and lenses (direction 1 framing).** Write to `research_notes/Knowledge action divergence internals/sycophancy_verdicts_fulltext.md`.
- **Read the method sections** (arxiv HTML) of 2607.00415, 2508.02087, 2604.19117, 2609.35822 and 2602.04918. For each, extract:
  - (i) the knowledge object tracked (answer token vs evaluative signal);
  - (ii) the probe fit contexts: neutral-only, compliant-only or mixed. Does any probe go below chance? Apply the 2609.10739 aliasing test conceptually;
  - (iii) the lens: logit, tuned or J;
  - (iv) the causal test used and its effect size;
  - (v) the models and prompt format.
- **Also** read the KnownLieBench J-lens pilot appendix (2608.26372 HTML from offset 100000) for its exact protocol, layers and any stated follow-up plans. That is the closest at-scale competitor to direction 1.
- **Output:** a matched comparison table, and a verdict on whether "erasure" survives an aliasing critique.
