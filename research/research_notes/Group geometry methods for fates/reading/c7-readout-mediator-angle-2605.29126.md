# When and How Long? The Readout-Mediator Angle in Temporal Reasoning

Fadnavis, Kanakaraj, Wyss (Bioscope AI). arXiv 2605.29126v1, 27 May 2026. Slug: `c7-readout-mediator-angle-2605.29126`.

## In one paragraph
A sin/cos probe decodes day-of-year from Gemma 2 2B at R^2 ~0.99, yet projecting out its 2-D subspace barely changes duration answers (~0.6 pp). A 4-D subspace trained to *maximise* the correct-answer NLL when projected out, at the same layer (L1), drives accuracy from 42% to 0%. The mean principal angle between the two subspaces (~88 deg) is at the Haar-random null. The authors trace a circuit (QK-mediated boundary heads at L11-12, late MLPs L18-25) and argue that probe-based safety monitors can be blind. **For us:** a useful methods-and-pitfalls source for fate 2 (bypass), H3 and subspace geometry. It is weak evidence about fates: no pressure, no flips, 1.5-9B models, the circuit only at 2B, and the "mediator" looks like an early structural/format subspace.

## Sections read
All of them: Abstract; 1-7 (incl. Algorithm 1, Figs 1-5, Table 1, Fig 6 text, conclusion/future work); references; supplements S1-S54 (DAS details, gradient probe, angle tables, QK-twist scan, AP list, SAE controls, Pythia emergence, topology, transplant, related work, spectrum, clinical benchmark S12-S17/S19/S21, collateral S18, MC check S20, proofs S22, pseudocode S23, anisotropy S24, CIs S25, norms S26, layer sweep S27, effective rank S28, loss sensitivity S29, cross-set S30, splits S31, NDM S32, open directions S33, cross-task S34, TFA S35, token dynamics S36, offset origin S37, proposed perturbation test S38, non-linear probes S39, alternative targets S40, erasure baselines S41, matched budget S42, energy tracking S43, per-direction ablation S44, MLP vs attention S45, cascading S46, MLP SAEs S47-S48, steering/transcoders S49, SAE dissociation S50, OV S51, flow graph S52, monitoring battery S53, notation S54).

## Setup
- **Task:** "How many days between X and Y?"; correct = within +-20%. Set-A 3,650 prompts (365 DOY x 10 templates) for probes; Set-F n=332 for evaluation (DAS 232 train / 100 eval).
- **Models:** Gemma 2 2B (all circuit work), Qwen 2.5 1.5B, Qwen 2.5 7B, Gemma 2 9B, Pythia 1.4B checkpoints.
- **Layer:** L* = peak probe R^2 layer: L1 (Gemma 2B), L0 (Qwen 1.5B), L8 (Qwen 7B), L3 (Gemma 9B). Hook on `resid_post`, last position.
- **The "DAS" here:** a QR-parametrised k x d basis trained 400 steps to maximise NLL under the projection-out hook. This is an ablation-maximising search, not interchange-intervention DAS.
- **Controls:** 25 Haar-random matched-rank ablations (20 at 7B/9B); analytic and Monte Carlo angle nulls; whitened coordinates; splits/CV/bootstrap/template holdout; layer sweep; seeds and a k-sweep; off-task collateral; norm removed; untrained-checkpoint probe.
- **Clean accuracy:** 42-51%. The factual task (capitals) was dropped because the 2B model scored 0%.

## Figures and tables (all 47 manifest entries; values "approx" are read off plots)
- **Fig 1 (p2):** schematic of the probe vs the mediator (-0.6% vs 0%, 88 deg). *Partly matches*: the schematic says "Jun 27th" for June 22nd and "+-10 days not found" where the text says +-7.
- **Fig 2 (p5):** (A) clean 42.0, probe 41.0, random 42.0, DAS 0.0. (B) DAS-probe 88.3/87.9/86.7 deg at k=2/4/6, DAS-PCA ~86.6-86.7. (C) rho 190x/1050x/262x. (D) DAS drops 44/42/51/45 pp across the four models. *Partly*: the null band is flat at 88.3 deg for every k, though the null is 87.6/87.1 at k=4/6. The probe drop shows as 1.0 pp, not 0.6.
- **Fig 3 (p6):** QK-twist heads, offsets at +-30/+-61 in both families, super-linear dose-response (Gemma 10/24 heads ~17 pp), attribution-patching overlap. *No*: panel D shows IoU = recall = 1.0 at K=24, but S5 reports 0.14/0.25. Panel A puts boundary heads above L15, but S4 says there are none.
- **Fig 4 (p7):** (A) DAS energy 26.4x null at L1, ~3x at L18, ~6.5x at L22, ~2x at L25; probe 0-4x. (B) MLP writes: probe peaks at L19 (4.3x), DAS at L20 (3.2x). (C, D) TFA-Pred is 82.7-83.7 deg from DAS, the probe 88.5. *Partly*: the probe exceeds DAS at L21, L23 and L24 (3.0x vs 2.0x), so it is not a clean "inversion". The null line is for 4-vs-4 dims, but the TFA components are 10-dim.
- **Fig 5 (p7):** circuit cartoon; #12499 logits month/Month/months; #2309 months/weeks/days. (D) transcoder DAS alignment rises from 0.975x to ~1.15x null. *Partly*: the gradient is real but tiny.
- **Fig 7 (p17):** Pythia probe R^2 0.956 at init rising to 0.998; boundary heads 133 -> ~44-62; circularness 0.002 -> 0.07. *Yes*.
- **Table 1 (p8):** temporal 87.9 deg / 0.6 / 42.0 / 1050x; spatial 88.4 / -6.0 / 20.0 / 20.8x; arithmetic 88.1 / 0.0 / 68.0. *Yes*, but the listed null (88.3) is the k=2 value for k=4 rows.
- **Table 2 (p13):** additive DAS drops of 42-51 pp; random drops 0.20/0.04/0/0 pp; ratios up to >500,000x. *Yes*, but there is no probe column at 7B/9B.
- **Tables 3-4 (p14):** gradient subspace 85.3 deg to DAS (null 87.6), 88.9 to the probe. *Yes*.
- **Table 5 (p15):** gradient ablation 1/6/12 pp at k=2/4/6, rho 5x/150x/75x. *Partly*: the same tool is ~21x in S11 and 19.1x in Fig 17.
- **Fig 8 (p19):** clean 39.5, probe 38.9, gradient 36.0, SAE 28.0, DAS 0. *Partly*: the numbers differ from the S11 table on the same page.
- **Fig 9 (p21):** n=75 clinical set, AUROC 0.86, AUPRC 0.99 at 96% prevalence. *Yes*, but only ~3 answers are correct.
- **Fig 10 (p22):** pooled r=+0.34; by tier A +0.69, B +0.06, C +0.31; r saturates near k=6. *Partly*: the caption for (C) describes a different panel.
- **Fig 11 (p23):** tier AUROCs 0.58/0.59/0.62; reliability bins 0.96, 0.85, 0.91, 0.95, 0.99. *No*: the text claims "monotonically increasing".
- **Fig 12 (p28):** probe-DAS sum cos^2 sits inside the +-2 sigma random band in raw and whitened coordinates. *Yes*.
- **Fig 13 (p29):** probe-vs-DAS cos^2 across layers 0.0006-0.003, probe self-similarity is layer-specific. *No*: the caption describes angles at three k with a null line, which the plot does not show. The text gives the k=4 null as 89.6 deg (wrong).
- **Fig 14 (p30):** Qwen 1.5B drops 37/44/33/35 pp at k=2/4/8/12; basis cosines 0.33-0.67. *Yes*.
- **Table 6 (p34):** TFA-Pred is 7.1-7.6x null with DAS and 2.7-3.0x with the probe; Novel is 2.0-2.5x and 1.8-2.2x. *Yes*. Note that the probe is also above null against the predictable component.
- **Fig 15 (p35):** Grassmannian MDS; probe 88.5 deg, at the upper tail of the random histogram (88.0 +- 0.3). *Partly* (null-dimension mismatch).
- **Fig 16 (p35):** bars as in Table 6. *Yes*.
- **Fig 17 (p36):** month features; rho DAS 1050, PCA 15, gradient 19.1, probe ~0. *Partly*: panel C labels the 7x bar "Novel->DAS", contradicting the caption and Table 6.
- **Fig 18 (p37):** per-token mediator energy 2.9-13%, lowest at "between", high at delimiters/"is". *Partly*: the three prompts have nearly identical profiles (template-driven), and the similarity maps are mostly uniform.
- **Fig 19 (p38):** chance of 2 shared offset modes p=0.020. *Yes*.
- **Fig 20 (p40):** DAS energy 4.6% at L1, trough 0.55% at L18, 1.15% at L22; probe at null. *Partly*: not monotone, and the caption colours are wrong.
- **Fig 21 (p41):** single directions 0.40/0.18/0.08/0.51 dNLL vs 68.77 for the full ablation. *Yes*.
- **Fig 22 (p41):** MLP zero-ablation 0.43/0.40/0.22/0.13/0.12/0.15/0.34/-0.02 at L18-25; attention often negative. *Yes*. The largest effects are at L18-19.
- **Fig 23 (p42):** early heads 0.45, late heads 0.02, all 0.49. *Yes*.
- **Table 7 (p44):** DAS projection +69.1 vs steering about +-0.04. *Yes*, but the two intervention types differ.
- **Table 8 (p44):** dictionaries span 5-9% of the DAS subspace; Jaccard <=0.010. *Yes*.
- **Figs 24, 29, 30, 31 (p45, 48-50):** NeuronPedia screenshots. *Cannot verify* (unreadable).
- **Figs 25-28 (p46-47):** G(k)=1; coverage 5-9%; group steering and frozen-attention effects at noise level (|dNLL| < 0.02). *Yes*.
- **Fig 32 (p51):** OV alignment, boundary heads ~1.2x the others, barely above null. *Yes*.
- **Fig 33 (p52):** attribution flow graph, qualitative. *Yes*.
- **Table 9 (p52):** self-graded "PASS" battery (71 d / 5.7 d; MI 0.000; 2650x vs -6.5x; monitor 66% / 88.4 deg).
- **Table 10 (p52):** seven probe targets within 2.8 deg of null. *Partly*: circular DOY is 87.6 here vs 87.9-88.3 elsewhere.
- **Fig 34 (p53):** one panel of probe angles. *Partly*: the caption describes three panels.
- **Fig 35 (p54):** MI(probe; DOY) ~3.99 nats, MI(probe; DAS energy) ~0. *Yes*, but DAS "energy" is a scalar norm that carries no date information at all.
- **Fig 36 (p54):** rho PCA 14.7, gradient 19.1, probe -6.5, DAS 2651. Random rho spreads from about -10 to +7.
- **Fig 37 (p55):** confidence-monitor probe at 88.4 deg (null 88.8). *Yes*.
- **Fig 38 (p55):** probe shift under DAS ablation 16.65 d vs random ~1-15 d. *No*: the caption says per-DOY shifts are "mostly below 3 days", but the plot shows most at 10-30 days.
- **Fig 6 (p9)** is not in the manifest and has no page image, so it is used from the text only.

## Key results (quotes checked against paper.txt)
- Probe ablation is harmless and DAS ablation catastrophic: "drops duration accuracy by only 0.6 pp—statistically identical to a random control. DAS ablation at the same layer and same k drops accuracy from 42% to 0%" (p5).
- Angle at null (p1). The bootstrap shows the probe is very slightly but significantly closer than null, by 0.5 deg (p32).
- Additive drops are 42-51 pp at all scales (p5). The angles at 7B/9B are only "saved with checkpoint pickles" (p15).
- Effective rank 4 with seed-stable bases (p5); 59x "cooperation" in NLL (p40).
- Task specificity 12.6x vs 240 off-task prompts (p5), but JS shift is ~10x random on unrelated prompts (p23).
- Routing bottleneck at L11-12, distributed and QK-mediated (p6); MLP writes switch at L19/L20 (p6); attribution patching peaks at L24, not L1 (p30).
- A gradient subspace leans toward the mediator, 85.3 vs 87.6 deg (p3). PCA removes 18.5 pp despite an 86.8 deg angle (p19).
- An untrained Pythia already reaches probe R^2 0.956 (p8).
- Cross-model Procrustes transplant fails (2.9% rescue) (p17).
- The clinical delta(x) monitor ties the entropy baseline (p24); clinical accuracy is 3-10% (p22).

## Methods we could use
1. **Sum cos^2 / principal angles with exact Haar nulls** (Monte Carlo plus whitened). Use: fate 2 and H1 subspace comparisons. Compute is CPU only.
2. **25-draw matched-rank random-ablation envelope with additive drops.** This is a required control for every ablation in Stage A/B.
3. **Ablation-maximising subspace search with a k-sweep and seed CCA.** It finds where the correct answer is load-bearing, at 4-42 min per model. Pair it with format and off-task checks.
4. **Gradient subspace** (dNLL/dh SVD): a 1-minute screen of whether a readout direction is output-sensitive (fate 1 vs 2).
5. **Subspace energy across layers and tokens:** an erased-vs-present screen (fates 6, 3). Compare at matched token positions.
6. **Grassmannian MDS against a random-frame cloud:** compare subspaces across pressures (H1).
7. **QK head scan with a permutation null and BH correction:** adapt to heads that prefer the user's claim vs the model's own earlier answer (fates 3/4).
8. **Per-direction / combination ablation:** is the signal one direction or a joint subspace (matters for H3)?
9. **Probe-shift-under-ablation:** a direct H3 test, but measure the model's output change too.
10. **Untrained-checkpoint probe floor:** guards against "decodable = known".

## Controls worth copying
Additive drops with random envelopes (no ratios near zero); sum cos^2 vs k1*k2/d with matched dimensions and a Monte Carlo null; whitening; strict splits, CV, bootstrap and template holdout for angles; full layer sweep; off-task JS/top-1 and norm-removed checks; seed CCA and k-sweep; simple monitor baselines (entropy, top-1 prob, length); non-linear probe families.

## Pitfalls (paper quote where stated; otherwise marked as reader's note)
- The "mediator" is an NLL-maximising ablation subspace (p55), and its full ablation breaks integer output (p16). That is causal controllability, not proven mediation of the duration variable.
- The L1 subspace looks structural, token-like and positional (p36, p14; Fig 18 profiles are identical across prompts).
- At k << d everything is near 90 deg (p4). Mean angle is insensitive (PCA case, p19), and orthogonality does not imply inertness (p4).
- rho is unstable at a noise-floor denominator (p19). Anisotropy makes the null 14x wider (p27).
- 7B/9B probe results are not in the paper (p15). Low clean accuracy; factual task excluded (p33).
- The adversarial "71-day shift" is measured in DAS coordinates, not in outputs (p53).
- Reader's note: the probe decodes an input variable while the DAS search targets the output loss, so their difference is less surprising than framed.
- Internal inconsistencies: Fig 3D vs S5 (p16); boundary-head layers (p16 vs p42); the "monotone" reliability claim (p21); the Fig 38 caption (p55); null 89.6 deg (p29); delta(x) "from the mediator subspace" (p9) vs the PCA in Algorithm 4; Fig 17C labels; CCA >0.94 (S1) vs >0.85 (S28).
- Reader's note: NLL saturation (~69 nats) inflates the "super-additivity", and the steering vs projection comparisons are mismatched.

**Limitations (closest admission, p23):** "the DAS basis does induce measurable distributional shift beyond a random k=4 ablation on non-date prompts (JS ∼10× the random null) — these four directions are not literally inert for the rest of the model". There is no limitations section.

**Future work (p9):** "Whether the readout–mediator angle can guide the design of causally grounded monitors—e.g. probes regularized toward the DAS subspace or mediator-aligned circuits as oversight targets—is the most immediate next step".

## Bearing on our fates
| Fate | How | Strength |
|---|---|---|
| 2 bypass | Template: decodable-but-inert probe vs a load-bearing subspace, with angles and random envelopes (caveats above) | moderate |
| 1 readout/threshold | Gradient subspace check only | weak |
| 3 not routed | Energy tracking, head scans and cascading ablation show how to follow a signal; the injection test was not run | moderate |
| 4 override | No competing signal studied | weak |
| 5 transformed | Cross-model Procrustes failed; suggests a within-model test | weak |
| 6 erased | Energy-above-null is the kind of test needed; energy is not content | weak |
| H3 | Mock monitor probe and probe-shift tests, built weakly | moderate |

## Group-geometry angle
**Helps:** a full subspace toolkit: principal angles and sum cos^2 with exact nulls, Grassmannian MDS, k-sweeps and seed CCA, energy along layers and tokens. It transfers to tracking the correct-answer subspace and to comparing pressures.

**Warns:**
- At k << d almost every pair sits near 90 deg, so compare sum cos^2 and match subspace dimensions to the null.
- Activations are anisotropic.
- Token-level subspace energy follows template position, so compare flipped vs resisted items at matched positions.
- An optimised causal subspace may be an "output format" subspace.
- Do Procrustes tests within one model.

## Where in the architecture
Gemma 2 2B only:
- optimised subspace at L1 (~4% depth);
- QK-mediated head bottleneck at L11-12 (~45% depth);
- late MLPs L18-25, strongest at L18-19 and L24;
- attribution patching peaks at L24 (92% depth).

The selected layers are early in every model because they are chosen by probe R^2. Offset modes recur in Gemma 2B and Qwen 1.5B (population level). Coordinates do not transfer across models. Pythia: head structure and circularness emerge between 10^3 and 5x10^4 steps, while the probe works from init.

## Actionable fix
Never ship a probe monitor without its sum cos^2 to a causal subspace, an additive ablation drop and a random envelope. The authors propose (untested) probes regularised toward the causal subspace. Their delta(x) monitor is weak (AUROC 0.63, tied with entropy). For us: every probe-based fate call in Stage A needs an ablation of an optimised subspace with random, format and off-task controls. In Stage B, check that a probe target is causally load-bearing before training against it.

## Verdict
**Adapt.** Copy the measurement and control kit. Avoid four things: treating an early NLL-maximising ablation subspace as "the mediator", ratio metrics, the scalar-energy MI test, and measuring a mechanism shift in the subspace's own coordinates.

## Relevance (honest)
Methodologically relevant to fate 2 and H3, and a good source of subspace-geometry controls. It does not study acting against known information: no pressure, no flips, 1.5-9B models, low accuracy. Its headline is weaker than framed, and several figures contradict the text. **Less relevant than its title suggests.**
