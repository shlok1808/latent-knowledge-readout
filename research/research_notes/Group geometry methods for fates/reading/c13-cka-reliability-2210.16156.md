# Reliability of CKA as a Similarity Measure in Deep Learning

Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky. arXiv 2210.16156v2 (Nov 2022). Slug: `c13-cka-reliability-2210.16156`

## In one line
Linear CKA can be pushed toward 0 by moving *part* of a point cloud, even when the classes stay separable. It can also be trained to any target map while the model's behaviour stays the same. It is also blind to rotations.

## Relevance to our project (honest)
- **What it is:** a methods warning for the group angle, not evidence about LLMs. Every experiment uses small vision models on CIFAR10, with no language models, facts or pressure.
- **Why it matters:** its subset-translation theorem fits our setting closely. If only the **flipped** items get shifted along an "agree with user" direction, CKA between neutral and pressured activations drops, while the correct answer stays decodable.
- **What to change:** don't let CKA/RSA make the transformed-vs-erased call. Lead with Procrustes plus probe transfer.

## Sections read (25)
Abstract; 1 Introduction; 2 Background on CKA and Related Work; 3 CKA Sensitivity to Subset Translation (Thm 1, Cor 2-4, nonlinear extension); 4 Experiments and Results; 4.1 Early Layer Results; 4.2 Practical Implications of Theoretical Results; 4.3 Explicitly Optimizing the CKA Map (incl. Wider networks, Analysis of Modified Representations); 5 Discussion and Conclusion; Acknowledgements; References (skimmed); App. A Experimental Details; A.1 Minibatch CKA; A.2 Network Architecture; A.3 Training Details; A.4 CKA Map Loss Balance (Alg. 1); App. B Additional Results; B.1 Invertible Linear Transformations; B.2 Wider Networks; B.3 RBF Kernel; B.4 Logistic Loss; B.5 ViT; B.6 Early Layers; B.7 OOD Evaluation; App. C Proofs.

## Setup
- **Data:**
  - synthetic two-cube data (10K + 10K points in 1K dimensions, linearly separable);
  - CIFAR10;
  - Patch Camelyon (CKA maps under a data shift);
  - CIFAR-10C (OOD accuracy).
- **Models (vision only):**
  - a 9-layer CNN, trained as generalised, memorised (random labels), or left randomly initialised;
  - ResNet-34 at widths x1/x2/x4/x5;
  - VGG-style nets at widths x1/x2/x4/x8/x10;
  - a small ViT (8 blocks, 12 heads, hidden size 256).
- **Layers:** full layer-by-layer CKA maps. Layers 1-2 for the early-layer comparison. The last hidden layer for the SVM translation test and the PCA.
- **Baselines:** the original maps and accuracies (CNN 85.9%, ResNet-34 x1 93.7%, ViT 88.7%), and the published conclusions of Kornblith 2019 and Nguyen 2021/2022.
- **Controls:**
  - distillation (or cross-entropy) keeps outputs fixed during manipulation;
  - test-set reporting and an OOD check;
  - linear probes next to CKA;
  - same vs different seeds; 5 seeds and 10 random instantiations;
  - several RBF bandwidths;
  - an unbiased minibatch HSIC estimator.

## The theorem in plain terms
- **Setup:** take a cloud X. Keep a subset S where it is, move the rest a distance c along a direction v, and let c grow.
- **Result:** linear CKA between the original and the moved copy tends to Γ(ρ) · ‖mean of S‖² / E‖x‖² · √(participation ratio), where Γ(ρ) = ρ/(1−ρ) and ρ is the fraction left in place.
- **Why the limit is small:** in practice every factor is small. With weight decay, representation norms are small (p4).
- **The catch:** the next layer ignores this shift if its weights are orthogonal to v. So the shift can be **behaviourally inert**.

## Figures (16 listed in the manifest, 16 covered, plus Fig 3 and Fig 9 noted; every provided page image viewed)
"From labels" means numbers printed on the figure. "Approx" means read off the plot.

| Fig | Page | What is plotted | Takeaway | Numbers | Text matches? |
|---|---|---|---|---|---|
| 1 | 4 | Schematic of subset translation, single outlier, separability-preserving shift | Defines the transform | none | yes |
| 2 | 6 | CKA vs layer, generalised vs random / vs memorised CNN (5 seeds) | Early layers score high CKA despite unrelated features | approx: G-vs-M 1.0, 0.85, ... down to 0.03 at L8; G-vs-R 0.67, 0.52, ... 0.17 at L8 | partly: high for G-vs-M, moderate for G-vs-R |
| 3 (merged into the manifest's Fig 2 entry) | 6 | Layer 1-2 conv filters of the three nets | Filters look completely different | images | yes |
| 4 | 6 | (a) synthetic CKA vs translation distance; (b) CIFAR10 last layer: class or single-point shift that keeps SVM hyperplanes | CKA collapses under subset shifts; one point is enough if moved far | approx: (a) about 0.2 by distance 10, plateau 0.1-0.15, RBF 0.2·median stays about 1.0; (b) whole class plateaus near 0.18, single point about 0.15 at 2300 and 0.02 at 5000; SVM about 91% | partly: text says (b) "drop to 0", but the plateau is about 0.18 |
| 5 | 7 | 8x8 CKA maps, original vs maximise first/last, maximise all, minimise all (linear, RBF) | Map set by training, accuracy kept | from labels: 85.9 → linear 85.4/85.5/85.4; RBF 85.5/85.4/84.3 | yes |
| 6 | 7 | "Comical" target maps and the maps reached | Arbitrary shapes are reachable | from labels: 85.3-85.7% vs 85.9% | partly: fidelity varies |
| 7 | 8 | ResNet-34 wide maps optimised to look thin | Block structure removable | from labels: 93.9/93.7/93.7 → 93.3/94.1/93.2; target 93.7 | yes |
| 8 | 8 | PCA before/after "maximise all" | The optimiser pushed a subset of one class out along a line (emergent subset translation) | approx: class-9 points stretch to PC1 about 150 | yes |
| 10 | 17 | Fig 7 maps tested on Patch Camelyon | Manipulation survives a data shift | none | yes |
| 11 | 17 | VGG widths, original vs optimised | Works on a second architecture | from labels: 88.6/90.9/91.9/92.3 → 88.4/90.7/91.5/92.0; target 85.9 | partly: the originals already look target-like |
| 12 | 18 | Fig 5 at RBF σ = 0.2/0.4/0.8 × median | Informative RBF CKA is just as manipulable; at σ 0.2 the map is all about 1 | from labels: σ0.4 minimise-all 81.7% | yes |
| 13 | 19 | Cross-entropy instead of distillation | Distillation isn't needed | from labels: 85.5/85.6/85.4 | yes |
| 14 | 19 | ViT maps manipulated | Transformers too | from labels: 88.7 → 88.3/88.5/85.2 | partly: minimise costs 3.5 points |
| Table 1 | 19 | ViT architecture | Tiny vision transformer | 8 blocks, 12 heads, 256 hidden, 200 epochs | yes |
| 15 | 20 | Early-layer CKA bars, same/different seed | High early CKA regardless of training | approx: G-vs-M 0.97/0.91 (same seed), G-vs-R 0.57-0.69 at L1 | partly |
| 16 | 20 | Linear-probe accuracy, layers 1-2 | High CKA (G-vs-M) yet probe accuracy 73% vs 11% | approx: G about 71/73%, M about 10-11%, R about 10% | yes (random-init at chance is unexplained) |
| 17 | 21 | CIFAR-10C accuracy per corruption | Manipulated nets behave the same OOD | from legend: averages 66.0 / 65.7 / 66.0 / 65.5 | yes |
| 9 (not in manifest; no page image) | 16 | CKA vs invertible random linear map | Information-preserving map → CKA about 0 | text only | not checked |

## Key results (quotes from paper.txt)
- Headline: "the CKA value can be easily manipulated without substantial changes to the functional behaviour of the models, and call for caution when leveraging activation alignment metrics." (p1)
- The theorem: "any set of internal neural representations X (e.g., from hidden layers of a network) can be manipulated with simple transformations (translations of a subset" (p4)
- The shift is inert downstream: "it is also easy to eliminate or ignore in a single layer transformation, as long as the weight vectors associated with the neurons in the subsequent layer" (p4)
- Separability is kept: "translating one of the subsets in a direction that preserves the linear separability of the representations and the size of the margins while simultaneously decreasing the CKA" (p5)
- On real activations: "The CKA values quickly drop to 0, despite the existence of a linear classifier that can classify both sets of representations into the correct classes with > 90% accuracy." (p7)
- One point is enough: "the fact that the position of a single point out of tens of thousands can so drastically influence the CKA value raises doubts about CKA’s reliability" (p7)
- High CKA is not usefulness: "considerably high CKA similarity values for early layers, does not necessarily translate to more useful, or similar, captured features." (p6)
- The map can be set: "In cases (1) and (2), the network performance is barely hindered by the manipulations of its CKA map." (p8)
- Emergent mechanism: "We observe that a a single class has a very noticeable set of points that are translated in a particular direction, away from the general set of classes." (p9)
- OOD is unchanged: "We observe that overall the average performance of these models on this out-of-distribution data remains effectively unchanged." (p21)
- Their advice: "Comparison methods that are straightforward to interpret or which are linked to well understood and simple theoretical properties such as solving the orthogonal Procrustes problem" (p9)

## Methods we could use
1. **Linear and RBF CKA (minibatch, unbiased HSIC).** Descriptive companion for H1 only. It cannot separate transformed from unchanged, or erased from shifted. Cheap.
2. **Subset-translation and outlier stress tests.** Inject a known shift on the flipped items, orthogonal to the correctness probe, and see which of our measures react. Do this before trusting any fate call. Trivial cost.
3. **SVM-hyperplane-preserving check.** Is the decision geometry (hyperplane and margins) shared even when a global score says the sets differ? This separates "changed along an inert direction" (bypass or override) from "information lost". Cheap.
4. **Adversarial map optimisation (Eq. 3).** Shows a geometry monitor can be gamed. A small LoRA version could test whether our fate monitor survives training against it (H3 / Stage B). Our estimate for a 9B model: one LoRA run. The paper does not test this.
5. **PCA of what changed.** Is a difference driven by a few items or by a coherent group shift?
6. **Probe next to CKA.** Decodability is the right test for "erased", and CKA is not.

## Controls worth copying
- Run synthetic ground-truth stress tests on our pipeline first.
- Always report probe accuracy next to CKA/RSA.
- Check behaviour OOD after any intervention.
- Use several seeds and random instantiations. Drop RBF bandwidths where the map is uniformly about 1.
- Use PCA to explain any similarity change.
- Use several measures, including Procrustes, each tied to a stated notion of similarity.

## Pitfalls
- Blind to rotations: "it is invariant to orthogonal transformations such as permutations, rotations and reflections and to isotropic scaling." (p3)
- Can happen without anyone noticing: "these transformations can naturally occur in ANNs, as discussed in Sec. 3, and one wouldn’t necessarily know that they have occurred." (p7)
- Misses low-variance directions: "Ding et al. (2021) demonstrated that CKA lacks sensitivity to the removal of low variance principal components from the analyzed representations even when this removal significantly decreases probing accuracy." (p3)
- A few points dominate: "the previously observed high CKA similarity between representations of later layers in large capacity models (so-called block structure) is actually caused by a few dominant data points" (p3)
- Not a metric: "Williams et al. (2021) discussed how CKA does not respect the triangle inequality, which makes it problematic to use CKA values as a similarity measure in downstream analysis tasks." (p3)
- Small RBF bandwidth: "RBF CKA with σ = 0.2 × median is significantly less informative than RBF CKA with higher values of σ." (p6)
- Invertible maps: "we observe that even for small values of µ and σ, CKA drops to 0" (p16)
- One number: "Such measures are trying to condense a large amount of rich geometrical information from many high-dimensional representations into a single scalar" (p9)
- Our models' training isn't controlled: "especially when the procedure used to produce the model is not known, consistent, or controlled." (p9)
- Reader notes (not stated in the paper):
  - Fig 4b's "drop to 0" is really a plateau of about 0.18.
  - Fig 16's random-init probe at chance is unexplained.
  - Fig 11's originals already look like the target.

## Limitations and future work
- Limitation (p9): "Indeed, some of the problematic transformations we identify are not necessarily encountered in many applications."
- Also (p9): "It is not our intention to cast doubts over linear CKA as a representation similarity measure in deep learning, or over recent work using this method."
- Also (p5): "it is outside the scope of our work to theoretically analyze nonlinear kernel CKA"
- Future work (p9): "Significant work is still required to understand these methods in the context of deep learning representation comparison, what notion of similarity each of them concretely measures"

## Bears on fates
- **Transformed (strong):** CKA cannot see a whole-set rotation or reflection, and a low CKA can arise with no re-encoding at all. Use Procrustes plus probe transfer.
- **Erased (strong):** low CKA is not erasure. Classes stay more than 90% separable while CKA approaches 0. Show erasure with probes and causal tests.
- **Override (moderate, our inference):** an "agree" push on only the flipped items is a subset translation, so CKA would drop with correctness intact. Recompute after centring flipped and resisted items separately; if the drop disappears, it was a group shift.
- **Bypass (moderate):** a shift orthogonal to the next layer's weights is inert. Geometry differences need patching before they count as mechanism.
- **Readout and not routed (weak):** nothing direct.

## Bears on the group angle
1. **Robustness for H1:** CKA/RSA scores across pressures need bootstrap or leave-one-item-out ranges, plus PCA outlier checks.
2. **Transformed vs erased:** this call should rest on Procrustes (with and without translation), principal angles between class-mean subspaces, and probe transfer, not on CKA.
3. **Track the mean shift separately:** flipped vs resisted clouds may differ mostly by one group's mean shift. Principal angles on centred data ignore it, while CKA over-reacts to it. Track the mean-shift vector and the subspace as separate things.
4. **Monitoring:** a geometry score is not a safe Stage B pass/fail monitor, because it can be optimised with behaviour fixed.

## Where in the architecture
- **No LLM or component claims.**
- **"Where" claims from CKA maps are fragile:** in vision layer maps, both early-vs-late similarity and wide-net block structure can be created or erased without changing behaviour. A claim like "neutral and pressured representations diverge at layer L" needs more than a CKA map.
- **Early layers:** between networks with unrelated features, layers 1-2 still score about 0.5-1.0 CKA (approx, Figs 2 and 15).

## Actionable fix or monitor
- **No model fix.**
- **Monitor rule:** never judge "fix removed vs hid the failure" with CKA or any single-scalar similarity. In the CNN, training moved maps to targets with at most 0.5 accuracy points of change (linear), and up to 1.6 points (RBF).
- **What to use instead:** patching, probes and behaviour.
- **If a geometry monitor is kept:** combine measures and stress-test it with subset shifts and outliers.

## Reuse verdict
**Adapt.**
- Copy: the stress tests, and the multiple-measures + Procrustes + probes + PCA recipe.
- Keep: linear CKA as a descriptive companion only.
- Don't use CKA alone: for transformed vs erased, or as a Stage B monitor.
