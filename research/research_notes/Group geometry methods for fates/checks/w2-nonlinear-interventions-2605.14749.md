# Check: w2-nonlinear-interventions-2605.14749 (Kim, "Non-linear Interventions on Large Language Models")

Verdict: OK (accurate on every number, quote and figure; a few over-generous strength labels and small omissions listed below, none blocking)

## Quote check
- `quote_check.py`: 15 quotes, 0 failed (first 120 normalised characters only).
- Stricter pass (my own script, scratchpad `chk/fullcheck.py`): every quote matches the full text and sits on the page the reading gives. 15/15 OK, including `limitations_quote` and `future_work_quote` (p4).

## Figures vs manifest
- Manifest lists 6: Figure 1, Table 3, Table 1, Table 2, Figure 2, Table 4. The reading covers all six. None missing. Page PNGs exist for p03, p04, p08.
- I opened Fig 1 and Fig 2 at 300 dpi.
  - Fig 1 matches the reading. Llama: none ~0.01, DIM ablation ~0.64 / ActAdd ~0.70, RDO ablation ~0.66 / ActAdd ~0.83, non-linear ~0.81. Qwen: none ~0.07, DIM ~0.74 / ~0.75, RDO ablation ~0.66, non-linear ~0.76. Qwen RDO ActAdd is absent, as footnote 1 says.
  - Fig 2 matches. Llama: layers 0-30 plotted, <=0.27 for 0-9, ~0.80 at 10, 0.70-0.77 for 11-15, 0.44-0.56 for 16-18, <=0.10 from 19. Qwen: layers 0-26 plotted, 0.10-0.39 for 0-7, 0.50-0.755 for 8-19, ~0.44 at 20, 0.03-0.24 for 21-26.
- Tables 1-4 numbers match the text layer exactly. The reading's own divisions (per-site L2: Llama ActAdd 3.17-3.18, Qwen DIM ActAdd ~41.8) are correct (1816.2/572, 1964.5/619, 24676.3/591) and are labelled "not in paper".
- Setup details match the paper: layer 10 / token -3 (Llama), layer 14 / token -4 (Qwen), zero-indexed block outputs, k = 1, 2 i-ResNet blocks, hidden width 128, Lipschitz 0.6, up to 30 fixed-point steps (tol 1e-5), D+ and D- 2,000 each, 313 StrongREJECT prompts. "tau and |S| not reported" and "no seeds / CIs / capability check" are true of the text.

## Errors (reading says / paper says / evidence)
No factual errors found. Items below are label strength and wording.

1. **Bypass strength "moderate" conflicts with the reading's own caveat.**
   - Reading: `bears_on_fates` 2 bypass "moderate", from the Table 3 pattern (linear 0.010 / 0.072 vs non-linear 0.813 / 0.760). The reading also says the linear control "may be under-tuned" (its value equals no intervention, edit size unreported) and that there is no shuffled-label or random-pair control for the i-ResNet.
   - Paper: Table 3 text only says the drop "suggest[s]" non-linearity contributes (p4). A rank-1 single-site single-token linear edit is also a far weaker intervention than DIM ActAdd (same direction at ~570-620 positions), and the i-ResNet has much more capacity. Recommend "weak to moderate" and keep the expressivity confound in the strength note.

2. **"Early and late layers do not work" (architecture_location) is slightly stronger than the figure.** Fig 2: Qwen layers 3-5 reach ~0.36-0.39 and layer 20 ~0.44; Llama layers 7-8 ~0.25-0.27. The paper says "largely ineffective" (p4). Suggest "largely ineffective".

3. **Qwen sweep point vs Table 3.** At the chosen Qwen layer 14 the sweep reads ~0.74; Table 3 says 0.760. The Qwen sweep peak is at layers 15-17 (~0.755), so the chosen Qwen layer is not the maximum. The reading says "the Llama main site (layer 10) is the best layer" (true, ~0.80) but does not note the Qwen case or the small mismatch (could be retraining variance; unreported). Minor.

4. **Pitfall 3 quote is cut mid-sentence** ("... finds an effective feature map at"). The full sentence continues "any layer; rather, it captures a non-linear feature governing refusal/compliance" (p4). Verbatim as far as it goes; consider extending.

## Omissions
- **Per-site L2 contrast is given only against ActAdd.** Against the ablation baselines the per-site norm is far smaller than the non-linear edit: Llama DIM ablation 2898.7/54,778 = 0.053, RDO ablation 0.031; Qwen DIM ablation 0.83, RDO ablation 0.29, vs non-linear 3.5 and 28.3. So the non-linear edit is roughly 30-100x larger per site than an ablation edit. This strengthens the reading's point that "two orders of magnitude smaller" comes from site count, so worth adding.
- **The paper's own "more precisely" sentence is not quoted.** The reading's pitfall about it has an empty quote field. Paper, p3: "This empirically suggests that our non-linear intervention captures the feature inside the model more precisely than the linear baselines do." The abstract repeats "steers the model more precisely". Add as a quoted overclaim.
- **Train/eval overlap not stated.** The paper does not say whether the 313 StrongREJECT prompts are held out from the 2,000 SALAD-Bench D- prompts. Not flagged by the reading.
- **Motivating citations** (Engels et al. 2025 circular structure, Gurnee et al. 2025 counting manifolds) are in the abstract and intro but absent from the reading. Low priority.

## Fate labels, architecture_location, actionable_fix (summary)
- Fates 1 and 6 "none" (correct: not addressed), 4 "weak", 5 "weak", 3 "moderate", 2 "moderate" (see item 1). The statement "no subspace tracking, principal angles or Grassmannian paths" is correct.
- architecture_location: block-output residual stream, one prompt token near the end of the chat template (Llama L10 of 32, token -3; Qwen L14 of 28, token -4), band ~10-18 / ~8-19: correct against Fig 2 and Table 4 (see item 2 for wording).
- actionable_fix is clearly framed as a proposed defence ("Untested for capability side effects"); the paper itself uses the method as a refusal-bypass attack. Fine.
- Controllability vs endogenous mediation: kept separate ("controllability by a flexible map is not endogenous mediation"). Good.
- Vocabulary: no "knows". Good.
