# Check: c8-represented-used-2610.05292 (Li, Zheng, Wu, Chen, "Erased, Rerouted, or Rescaled?", arXiv 2610.05292v1)

Verdict: NEEDS FIXES (minor to moderate; no fabricated numbers or quotes; the problems are two statements that go past the figures/text and several "strength" labels and fix claims that rest on toy-model results only)

## Quote check
- `quote_check.py`: 26 quotes, 0 failed.
- I also spot-checked the page numbers of all 16 `key_results` quotes plus the pitfall quotes, `limitations_quote` (p9) and `future_work_quote` (p9) against the page markers in `paper.txt`. All sit on the page the reading gives (e.g. 20-77 degree rotation p13, ridge R2 0.139 p11, interleaved-split quote p19, "recomputes part of an exchanged component" p16).

## Figures vs manifest
- Manifest lists 20 items (Fig 1-16, Tables 1-4). The reading has 20 figure/table entries. None missing. Page PNGs exist for every figure page (p02-p06, p08, p14-p22).
- Sections: the reading's `sections_read` covers every section in the paper (Abstract, 1-7, Limitations, App A.1-A.4, B, C, D, D.1, E). Nothing skipped.
- I opened Fig 2 (p5), Fig 3 and 4 (p6), Fig 5 (p8), Table 3 and Fig 6 (p14), Fig 7-9 (p15), Fig 11 (p17), Fig 14 (p20), Fig 16 (p22). The numbers the reading gives match the plots within reading error: Fig 2 coordinates (SFT A~0.13/U~0.15; GRPO b=0 ~0.07/0.55; KL-RL b=0 ~0.017/0.2; weight-decay SFT ~0.0024/0.06), Table 3 (all rows identical to the printed ranges), Fig 5 panel values (gamma 0.45, 16.0, 9.40, 0.895), Fig 6 (GRPO b=0 above base use in the entangled transformer), Fig 8 (policy/NTP 0.15/0.56 SFT, ~0.53/0.75 GRPO, ~0.8 KL-RL), Fig 11 II (GRU slow-nuisance R_inv ~0.42 to ~1.0, gamma ~0.27 to ~0.95), Fig 14 II (blocks 20-27 reaches ~20, LoRA ~16-17), Table 4.
- The reading's own flag on Fig 8 (text says "GRPO and DPO near 0.9", plotted GRPO b=0 twin gap is ~0.53/0.75; ~0.9 is the transfer-share panel) is correct.

## Errors (reading says / paper says / evidence)

1. **Fig 16 takeaway and architecture_location say decodability is unchanged "at every relative depth". The figure shows an early-depth gap.**
   - Reading says (figures, Fig 16 takeaway): "Base and post-trained decodability curves nearly coincide at every depth". architecture_location: "belief-state decodability is nearly unchanged at every relative depth".
   - Paper says (p7): "peak R2 changes by at most 0.05 for the belief state and, with three exceptions in the smallest model, by at most 0.06 for the invisible component". The claim is about peak R2, not every depth.
   - Evidence: Fig 16 II (Qwen3-4B), belief-state R2 at relative depth ~0.12 is ~0.35 post-trained vs ~0.57 base, a gap of ~0.2. The reading's own numbers field records this exact dip, so the takeaway contradicts the reading's own data. Fig 16 I (1.7B) invisible component also differs by ~0.1 at the last layers (~0.5 post vs ~0.38 base) and at depth ~0.45. Fix: "peak decodability unchanged (<=0.05); early layers (relative depth <~0.2) can differ, e.g. Qwen3-4B belief R2 0.35 vs 0.57; three exceptions in the invisible component in the 1.7B model". This matters because "WHERE" (relative depth) is the project's goal, and the one depth-resolved LM figure is not "no change anywhere".

2. **architecture_location attributes the head-only control and "written in the residual stream" to the Qwen3 experiment. Both come from the toy transformers.**
   - Reading says: "under controlled GRPO the collapsed or amplified log-odds are written in the residual stream (head-only training does not collapse)".
   - Paper says (App D.1, p18): "An untied head with a trainable trunk collapses like the tied head ... and the collapsed log-odds are written in the residual stream. Training only the head does not collapse within 5,000 steps". This is the small 4-layer, width-128 transformer in the main world. The Qwen3 section (App E) has no head-only or untied-head control. What Qwen3 shows is Fig 14 I: the class component of the representation at the marker grows 31-110x while the mean stays at 0.80-0.95 of the second moment, plus blocks 20-27 alone reproduce the amplification (gamma 19.9 vs 16.0 for LoRA, p22).
   - Fix: split the sentence: toy = head-only/untied controls show the shift is in the shared representation; Qwen3 = representation at the marker changes (class component up) and training only the last 8 of 28 blocks (relative depth ~0.71-1.0) is enough, but no head-only control was run. Also say that blocks 20-27 amplified "faster than adapters" (paper, p22), which is a parameterization difference, not "like LoRA".

3. **actionable_fix generalises a toy-only direction and a toy beta threshold.**
   - Reading says: "unanchored RLVR (beta=0, common in current recipes) or SFT on fixed labels reroutes" and "keep a KL anchor (beta of a few hundredths or more)".
   - Paper says: the collapse/rerouting of unanchored GRPO is in 4-layer toy transformers (48/48 runs, Table 4). In Qwen3-1.7B/4B-Base, unanchored GRPO does the opposite: "amplified ten to twenty times and no run collapses" (p7), and Limitations (p9) says small transformers and LMs "leave the anchor identity in opposite directions" with no explanation. The beta window (0.003, 0.02] is the toy GRPO threshold. In Qwen3, beta=0.04 still amplifies 6-9x at first and returns to gamma~1 only after 4,000-10,000 steps (p20); beta>=0.3 returns within 1,000 steps (p7).
   - Also: no Qwen3 run reports U (twin-exchange use). LM evidence is R (>=1.0), A, gamma. So "reroutes" has no LM evidence at all; the only LM "use" evidence is that within-class log-odds change.
   - Fix: write "toy: unanchored RL and SFT reroute (U falls to 0.15-0.67); Qwen3: unanchored GRPO amplifies within-class log-odds (gamma 10-20), U not reported; anchor restores gamma~1 at beta>=0.3 in 1k steps, at beta=0.04 only after thousands". The bearing on our H2 is still real (an anchor is the lever) but the rate and direction are not transferable.

4. **bears_on_fates strengths are too high where the evidence is toy-only or a single paragraph.**
   - 2 bypass, "strong": all U numbers (0.15-0.22 SFT, 0.53-0.67 GRPO b=0) are from 4-layer width-128 transformers and 2-layer GRUs on HMM streams. No released-LM or Qwen3 result shows a decision that stopped using decodable information. Strong for the method, none for LLMs. Fix: "strong for measurement design; no LLM instance".
   - 5 transformed, "strong": the evidence is one paragraph in App D (p13) with no figure or table. Numbers: kernel image rotates 20-77 degrees at beta=0.04; base-fitted interchange transfers 0.78-0.88, refit 0.94-0.97; frozen probe "fails" (no value given). It is toy, anchored runs, where nothing failed (U and R intact). So it shows frozen probes misread re-encoding as erasure, which supports the H3 mechanism, but it is not an example of a transformed failure. Fix: "moderate; method only; benign re-encoding in toy anchored runs".
   - "Grassmannian distance" (bears_on_group_angle) and "principal-angle" wording are the reader's gloss. The paper never uses principal angles, Grassmannian, Procrustes, CKA or RSA (grep: zero hits). It only says the image "rotates by 20 to 77 degrees" and does not say how the angle is computed (max principal angle, mean, per-direction). Mark as "not defined in paper; ours to specify".
   - 3 not routed, "moderate": the GRU filter-use test patches the twin's component and finds later predictions barely change (0.01 of base). Under the project's definitions, that is "decodable but causally inert" = fate 2 bypass; fate 3 needs "injecting works, the model never carries it". The revival test (accessibility to a new reader) is a third thing and is not a "not routed" case either. Fix: re-label as 2 (inside-computation bypass) and treat revival as "low-variance access cost", not a fate.

5. **Slope/intercept regression (methods[4]) is mapped to correct-vs-pushed answers without saying the paper only defines it for tied (equally rewarded) tokens.**
   - Reading: "Per-item log-odds (correct vs pushed answer) with and without pressure ... 4 override (slope kept, large global shift) vs 2 bypass/collapse (slope near 0)".
   - Paper (p4, App C p13): gamma and the intercept regress the policy's WITHIN-class log-odds among equally rewarded tokens on the reference's. Proposition 2(i) (slope 1 under any beta) holds only for tied actions; for tokens of different reward the log-odds shift by dq/beta, so the slope-1 identity does not apply to correct-vs-wrong.
   - Fix: add "defined for equally rewarded alternatives only; for correct vs pushed answers the expected relation is unspecified, so this is our adaptation (e.g. across paraphrases of the same wrong answer, or among equally wrong distractors)".

6. **actionable_fix: "SFT-style fixes are expected to hide (reroute) rather than remove" has no support in the paper and may be inverted.**
   - In the paper SFT is the cause of rerouting of reward-null information (the nuisance). In our setting the user's pushback is the signal we want the decision to ignore, and the correct answer is the reward-relevant variable. SFT toward the correct answer could legitimately make the decision ignore the pushback while leaving it decodable. Whether our "hidden vs removed" worry maps depends on which variable is "reward-null", which the reading does not discuss. Mark as our hypothesis, not a paper result.
   - "plain DPO behaves like anchored RL here": true in the paper only for idealised DPO (pairs from a Bradley-Terry model on the exact reward, optimum = the KL tilt, p12). Real preference pairs (correct vs sycophantic) are not shown.

7. **Fig 3 numbers differ slightly from the text.** Reading: "R_inv at ordinary positions stays 0.25-0.75". Paper (p16): 0.15 to 0.85 decodable (0.53-0.65 after SFT), filter use 0.01 to 0.09. Minor; use the text range. Same for Fig 4 I: reading says Qwen RankMe ~0.5-0.8; text gives falls of 16 to 61% (RankMe 0.39-0.84).

## Omissions
- **The exact-ground-truth premise for A, U and twins.** A is the variance explained by regression on the true belief coordinates; twins match the class marginal to 1e-9 (App C). Neither exists for facts under pushback. The reading gives "cheap at 9B-27B" for R/U/A but only the reuse_verdict hints at the identification problem. Add one line: R is portable (refit linear probes), U needs hand-built twins that cannot isolate a "kernel", A needs a known subspace.
- **Proposition 1 has a stated limit.** The paper says normalization layers leave only rotations as exact symmetries, so allocation "is still a well-defined property of the trained solution" (p3). bears_on_group_angle uses the invertible-map warning without this caveat; in real LMs with RMSNorm the arbitrary-G argument is weaker (rotations, which preserve CKA and principal angles, are the exact symmetries). Keep the warning (refit probes plus interchange) but say it is strongest for variance-share/effective-rank, weaker for rotation-invariant comparisons like linear CKA.
- **A model family was excluded for failing to converge in context** (App E, p20: "A base model of another family did not converge to the Bayes predictor in context and was excluded before any comparison"). architecture_location says "one family only" but not that a second family was dropped. This bears on "stable across models".
- **Two pre-registered directional predictions were overturned** (App B, p12: unanchored PG sharpens tied outputs; recurrent retains more than transformer). Useful for how much to trust theory-driven fate predictions.
- **NTP exit also falls after SFT** (0.56 of base, with a wide error bar in Fig 8). The two-exit takeaway "one head using it and the other not" is stronger than "0.15 vs 0.56".
- **Next-token loss in Qwen3**: "restores ordinary positions and leaves the amplification at the marker in place" (p22). The reading states the transformer result (no restoration) only in the toy sense; add that the LM result is the same.
- Qwen3 controlled GRPO uses seed 0 for the headline panels (Fig 14 I, 5 III); the "consistent across post-training seeds" statement in architecture_location is a toy result (48 runs, 3 pretrained models). Say so.

## What the reading gets right (keep)
- Table 3, Fig 2, Fig 5, Fig 6, Fig 7-9, Fig 11 numbers and all 26 quotes.
- The pitfall that the paper's own LM probe for the invisible component uses ridge, which its own App C says reads allocation as loss (p11 vs p19). This is a real internal tension and is worded correctly.
- The reuse_verdict ("adapt the measurement design, do not carry over findings") and relevance_to_project are the right scope. The five frozen-vs-refit, gauge-control, twin-exchange, two-exit and revival designs are accurately described and reusable.
- The overall caveat that this is not a pushback/flip study and that the LM evidence is synthetic HMM streams at 1.7-4B.

## Fix list (for whoever edits the reading)
1. Fig 16 takeaway and architecture_location: "peak" decodability, early-depth gap (4B belief 0.35 vs 0.57 at depth ~0.12), 1.7B invisible exceptions.
2. architecture_location: move head-only/untied/residual-stream claim to the toy; add that Qwen3 has no head-only control; add "faster than LoRA" for blocks 20-27 (relative depth ~0.71-1.0).
3. actionable_fix: separate toy (reroute) from Qwen3 (amplify, U not measured); give Qwen3 beta timings; mark SFT-hides and DPO-equivalence as our hypotheses.
4. bears_on_fates: bypass "strong" to method-only, transformed "strong" to "moderate, benign re-encoding, one paragraph", move filter-use from 3 to 2, drop "Grassmannian".
5. methods[4]: add the tied-token restriction on gamma.
6. Add the omissions above (ground-truth premise, normalization-layer symmetry caveat, excluded family, overturned predictions).

Verdict: NEEDS FIXES. Core extraction is reliable; apply fixes 1-4 before this reading feeds the synthesis.
