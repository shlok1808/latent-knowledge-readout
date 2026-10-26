# Curved Inference II: Sleeper Agent Geometry (Manson, arXiv 2608.24037)

**Verdict: avoid as evidence. Adapt only the sublayer-resolution trajectory sampling.** It is less relevant than the title suggests.

## In one paragraph
The paper samples the unnormalised residual stream after every attention and MLP sublayer ("double resolution") and computes one number per response: semantic surface area A', which is path length plus curvature, measured under the logit-space metric G = UᵀU. It then shows that A' differs between response classes labelled by LLM judges (transparent vs evasive) in Gemma3-1b and LLaMA3.2-3b, on one "should I disclose my prior recommendation?" scenario with five variants. It trains no probe, runs no causal test and uses no matched pairs. Several groups have 2-5 items.

## Setup
- **Data:** one scenario family, five variants (honest, strategic, persuasive, deceptive, malicious). The variants differ only in hidden "Thought:" lines and the final visible statement. 100 samples per variant per model, so 500 per model.
- **Labels:** Gemini 1.5 Flash, Claude 3.5 Sonnet and GPT-4o Mini see only the visible response and rate transparency (high/medium/low) and response type (explanatory/evasive/manipulative). The paper uses two label sets: majority vote, and unanimous agreement (LLaMA kept 201/500, Gemma 293/500).
- **Models:** Gemma3-1b and LLaMA3.2-3b. Table 1 gives 1.3B and 2.3B parameters, but the text says 1B and 3B, so the table is inconsistent.
- **Read positions:** all tokens and all layers, at both sublayer boundaries (2L+1 points per token). How tokens are pooled into one A' per response is not described. The limitations section mentions "normalised by response length", but the methods never explain it.
- **Metric:** A' = Σ (salience + γ·curvature), with γ = 1.0. Salience is ‖Δx‖ under G = UᵀU.
- **Baselines:** none. No probe, no length-only baseline, no shuffled labels, no held-out prediction.
- **Controls:** vocabulary-matched variants, labels taken from observed outputs rather than designed intent, within-model comparisons only, bootstrap CIs, and the unanimous-label subset.

## Figures and tables (8 of 8 covered)

| ID | Page | What it shows | Takeaway | Matches text? |
|---|---|---|---|---|
| Fig 1 | 3 | Schematic 3D "Residual Trajectory Manifold": 50 tokens × 28 layers, drawn as lines and then filled in as a surface | Concept picture only, no data | Yes |
| Table 1 | 4 | Model specs: Gemma3-1b 1.3B, 26 layers, d = 2,048; LLaMA3.2-3b 2.3B, 28 layers, d = 3,072 | Two tiny RoPE models | Partly: the text says 1B and 3B |
| Table 2 | 11 | LLaMA KW p-values, full vs unanimous labels; d = 2.15 / – / 4.15 / 1.02 / 2.22 | Significant for every strategy, but groups are very unbalanced | Partly: the text mixes full-consensus η² with unanimous p-values |
| Table 3 | 11 | Gemma: same comparison. Honest goes from 0.555/0.310 to 0.048; deceptive is a single class | Weak or untestable in Gemma | Partly: the text's p = 0.138 and 0.794 appear in no table |
| Fig 2 | 12 | Dashboard: A' bars and violins by transparency, donut of group sizes, an **empty placeholder panel** | **The image is LLaMA deceptive (N = 47: low 1,276 with n = 40; medium 2,971 with n = 5; high 2,396 with n = 2)** | **No: the caption says Gemma N = 61. The images look swapped with Fig 3** |
| Fig 3 | 13 | Single-group histogram, mean 9,769, median 8,847, σ = 4,488, one outlier near 34,000 (read from figure) | **The image is Gemma deceptive (N = 61, one class)** | **No: the caption says LLaMA N = 47. One panel also has wrong axis labels** |
| Fig 4 | 14 | One bar, Gemma deceptive, 9,769, CI about 8,650-10,950 (read from figure) | No separation is possible with one class, yet the title says "separates" | Partly |
| Fig 5 | 15 | LLaMA deceptive bars and violins by group | Low < high < medium, with medium n = 5 and high n = 2; the order is not monotone | Yes, but the caption omits the n's |

## Key results (with quotes)
- "geometric structure reliably predicts semantic classification, with statistically significant differences in surface area observed across five prompt strategies and two model families." (p1)
- LLaMA deceptive: "Sample composition shows predominant low transparency responses (85.1%, n=40), with smaller medium (10.6%, n=5) and high transparency groups (4.3%, n=2)." (p14). The KW test gives p < 0.001 and η² = 0.448 (p14).
- Gemma "strengthening": honest goes from p = 0.555 to 0.048 (p11). This rests on "Medium transparency (n=3)" against 60 low-transparency responses (p32).
- "Surface area correlations with salience and curvature contributions exceeded r > 0.99 across all analyses" (p16). So A' is in effect path length.
- A 6.7× difference in absolute A' between the models; the only shared pattern is explanatory > evasive (p16).
- Judge agreement is low. For Gemma transparency, "Krippendorff’s Alpha: 0.309 (Poor)" (p27).

## Methods we could use
1. **Double-resolution sampling (post-attention and post-MLP).** This lets us credit a divergence between flipped and resisted items to an attention or MLP sublayer. That matters for override (attention reading the user's claim) vs transformation (MLP). It is cheap: TransformerLens `hook_resid_mid` and `hook_resid_post`. Copy this.
2. **A' (path length + curvature).** Weak. It is one scalar per response and is not tied to the correct-answer direction, so it cannot separate transformed, erased and override. It is also confounded by norm growth and token count.
3. **Pullback metric G = UᵀU.** This is our extension, not something the paper tests. At each layer, compare a probe direction's Euclidean norm with its G-norm. A signal that is decodable but has a small G-norm sits outside what the unembedding reads, which makes it a bypass or not-routed candidate. It is only a rough lens, because middle layers are not aligned with the unembedding.
4. **Label-precision sensitivity (majority vs unanimous judges).** Use this only if we label hedged or flipped answers with LLM judges.

## Controls worth copying
- Vocabulary-matched variants. For us: a pushback turn vs a neutral follow-up, matched in length.
- Group items by observed behaviour (flipped or not), not by the condition applied.
- Compare geometry within each model only, and compare models through standardised effects.
- Show group n next to every p-value.

## Pitfalls
- It claims to go "beyond probes" with no probe baseline: "even in naturalistic contexts where linear probes would likely fail" (p2).
- No multiple-comparison correction, by choice (p29).
- Unanimous filtering keeps only the easiest items (admitted, p19).
- The length confound is handled by an undescribed normalisation (p19).
- The "40% reduction" and "60% reduction" labels are swapped (p10).
- Judges see only the output, so the geometry is correlated with output style, not with internal intent (p27).
- It can be evaded: "highly sophisticated deceptive systems might learn to maintain geometric consistency" (p19).
- Not stated by the paper: the Fig 2 and Fig 3 images are swapped, one panel is a placeholder, and one plot has wrong axis labels.

## Limitations and future work (quoted)
- Limitations (p19): "We evaluate only two small open-weight decoder-only models (Gemma3-1b and LLaMA3.2-3b), using a small number of carefully constructed prompt variants."
- Future work (p20): "combining geometric methods with more targeted techniques (such as causal patching around curvature spikes, or hybrid approaches that fuse probes, symbolic tools, and trajectory) based signals."

## Bearing on our fates
- **Transformed:** weak. A high-curvature turn is consistent with re-encoding, but nothing ties it to the correct-answer information.
- **Bypass / not routed:** weak. No probe and no patching. The G-norm idea above is ours.
- **Override:** weak. The setup has a competing goal (hidden thoughts pushing toward concealment), but no competing signal is measured.
- **Readout failure / erased:** not addressed.

## Bearing on the group-geometry angle
Mostly a warning. Collapsing the token × layer cloud to one scalar throws away the direction information we need (principal angles, Procrustes, CKA). Any path metric must control for residual-norm growth and token count, and must be compared against a probe and the logit margin. Two things are worth keeping. First, sublayer-resolution sampling for finding fork points. Second, reporting both normalised and unnormalised residual paths. Groups of 2-5 items give large-looking d and η² values, so our flipped vs resisted sets must be matched and sized first.

## Where in the architecture
Nothing localised: everything is summed over layers and tokens. The claims that attention bends trajectories and MLPs amplify them (Appendix A) are conceptual. Across two models only the direction of the effect agrees. There are no seeds or checkpoints.

## Actionable fix
Monitoring only (real-time A' with thresholds set per model), and it is not validated for prediction. No fix is implied.

## Relevance
Low. The paper does not study knowledge-action divergence, uses models of 1-3B, and has serious method and reporting problems. For us the one useful thing is the post-attention/post-MLP sampling for locating where flipped and resisted trajectories diverge.
