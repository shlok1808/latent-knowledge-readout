# LLM Circuit Analyses Are Consistent Across Training and Scale

Tigges, Hanna, Yu, Biderman. NeurIPS 2024. arXiv 2407.10827v2. Slug: `c12-tigges-circuits-across-training-2407.10827`

## In one line
A function (for example "copy the right name") stays put at the level of a *class of heads*. The individual heads that do it come and go during training, and the model makes up for any it loses.

## Relevance to our project (honest)
Moderately relevant, and less than the title suggests for the group-geometry angle. This is a circuits paper on Pythia pretraining checkpoints (circuits up to 2.8B) with toy tasks. It does not look at activation geometry or at models acting against decodable knowledge. What it gives us:
- a **Stage B warning**: roles move between heads while behaviour stays the same, and self-repair can hide a removal;
- **class-level metrics** for comparing circuits across pressures (H1) and before vs after a fix.

## Sections read (26)
Abstract; 1 Introduction; 2 Methods; 2.1 Circuits; 2.2 Circuit Finding; 2.3 Models; 2.4 Tasks; 3 Circuit Formation; 3.1 Behavioral Evaluation; 3.2 Component Emergence; 4 Algorithmic Stability...; 4.1 Model Behavior and Circuit Components Post-Formation; 4.2 Circuit Algorithm Stability Over Training; 5 Graph-Level Circuit Analysis; 6 Discussion (Implications; Quantization Model; Limitations and Future Work); 7 Related Work; Acknowledgements; References (skimmed); App. A Analysis of Task Circuits; A.1 IOI Circuit & Algorithmic Criteria; App. B Other Circuit-Analysis Methods; App. C Size/Similarity/Change Rate; App. D Component Metrics; App. E Task-Dependent Learning Ceilings; App. F Compute; App. G Licenses.

## Setup
- **Data:** four toy tasks, each with clean and corrupted minimal pairs:
  - IOI: 70 examples, logit diff.
  - Gendered Pronoun: 70, logit diff.
  - Greater-Than: 200, prob diff.
  - SVA: 200, prob diff.
  - App. E also re-plots SciQ, PIQA, WinoGrande and ARC-Easy evals.
- **Models:** Pythia only.
  - Circuits for 70M, 160M, 410M, 1.4B and 2.8B. Behaviour only for 6.9B and 12B (left out of circuit finding for compute).
  - 154 checkpoints each, about 300B tokens. Pretraining only, no fine-tuning.
- **Where read:** every attention head and MLP is a node, and behaviour is read at the final-token logits. The algorithm analysis uses Pythia-160m at every post-formation checkpoint and the larger models at about 1/3 of checkpoints.
- **Baselines:** whole-model performance (a circuit must recover at least 80%). For path patching, the denominator is the effect of all upstream heads. CSPA uses a mean-ablation baseline.
- **Controls:**
  - corrupted counterfactual prompts and a faithfulness test;
  - a head counts only if its functional score is above 10% AND ablating it hurts;
  - an extra positional-vs-token test for S2-inhibition heads;
  - early, poor-performance checkpoints are treated as not meaningful.
- **Compute:** 8 A40s over two months, about 0.25 utilisation (App. F).

## Figures (9 listed, 9 covered; each page image viewed)
Numbers marked "approx" were read off the plots.

| Fig | Page | What is plotted | Takeaway | Numbers | Text matches? |
|---|---|---|---|---|---|
| 1 | 4 | Task behaviour vs tokens, 70m-12b | Same onset (a few 1e9 tokens) across sizes; all but 70m end at similar level | approx: IOI 0 to 3-4 logit diff between 3e9 and 1e10 (70m about 0.5-1); GT dips to about -0.4, then 0.9-1.0 | partly: the 6.9b/12b-like-160m claim can't be checked by eye (pastel colours) |
| 2 | 6 | Normalised class scores (succession, induction, CSPA, copy) vs tokens | Components appear when the task appears; induction and successor scores decline later | approx: induction jumps to 0.75-1.0 by about 2-3e9; name movers rise over 3e9-8e9; successor peaks about 0.9 near 1e11, then about 0.5 | yes; peaks are 1.0 by construction (per-model normalisation) |
| 3 | 7 | Pythia-160m single heads over training | Heads gain and lose roles; others take over | approx: name mover (4,6) drops from about 0.95 to 0.1 between about 3e10 and 1e11; successor (5,9) peaks about 0.8, ends about 0.45; CSPA raw scale 0-0.0065 | partly: which colour is (11,0) is uncertain |
| 4 | 8 | A: 160m IOI circuit. B-D: class effect shares vs tokens, 160m-2.8b | Class shares stay flat after formation in all sizes | approx: B 68-92%; C 45-95%; D 40-100% (mostly 55-90) | yes ("generally above 50%") |
| 5 | 9 | EWMA node Jaccard vs tokens | Mostly 0.6-0.9 with sharp dips in small models | approx: 70m dips to about 0.33-0.35; 160m falls to about 0.35 late in Gendered Pronoun | partly: 410m-2.8b overlap |
| 6 | 23 | Circuit node count vs step | Bigger models have bigger circuits; size is noisy | approx: IOI 2.8b 110-185 nodes vs 70m 0-55; some checkpoints about 0 nodes. Text: r = 0.72 / 0.9 / 0.6 | partly: the 0-node checkpoints are not discussed |
| 7 | 23 | Node Jaccard with the final circuit | Fluctuates at every checkpoint, trends upward | approx: mid-training 0.2-0.7; ends at 1.0 | yes (the end value of 1.0 holds by definition) |
| 8 | 24 | Edge-weighted Jaccard with the final circuit | Smooth climb toward the final circuit | approx: 0.1-0.35 early, 0.6-0.85 by about 2e11 | partly ("remains high at end" holds by construction) |
| 9 | 26 | ARC-E / SciQ / PIQA / WinoGrande accuracy vs step | Same onset step across sizes; final accuracy still scales | approx: SciQ 0.2 to 0.6-0.75 at steps 1000-2000; final 0.6 (70M) to 0.9 | partly: WinoGrande onset is late and only for large models |

## Key results (quotes from paper.txt)
- Emergence lines up across scale: "We find that task abilities and the functional components that support them emerge consistently at similar token counts across scale." (p1)
- Head turnover: "the name-mover head (4,6) suddenly stops exhibiting this behavior at 3 × 1010 tokens, having acquired it after 4 × 109 tokens." (p7). This means 3x10^10 and 4x10^9.
- Takeover: "Pythia-160m’s main successor head (5,9) loses its successor behavior towards the end of training; however, (11,0) exhibits more successor behavior at precisely that time." (p7)
- The algorithm is stable: "Notably, in no model or metric are there dramatic shifts in algorithm corresponding to functional component shifts within the circuit." (p8)
- The shares are high: "all scores are relatively high, generally above 50%; the core solvers of the algorithm, copy suppression and name-mover heads, have scores above 70%." (p8)
- Scale changes the details: "the Pythia-160m circuit does not involve duplicate-token heads, while others do. So small differences exist amid big-picture similarity." (p9)
- Size correlation: "The Pearson correlation is r =0.72 for IOI and SVA, 0.9 for Greater-Than, 0.6 for Gender Pronoun." (p9)
- Stability vs scale: "larger models tend to form more stable circuits (with both higher average values and fewer sharp fluctuations); EWMA-Jaccard is more volatile for Pythia-70m/160m." (p9)
- Roles get shared, not specialised: "Increases in model size do not necessarily help heads become more specialized in their roles; rather, in these circuits more heads will take on the same roles." (p23)
- The same head type can have the opposite sign: "In the original IOI circuit, copy suppression heads hurt performance, downweighting the correct name. In contrast, we find (Appendix D) that they contribute positively to the Pythia IOI circuit" (p5)

## Methods we could use
1. **EAP-IG minimal faithful circuit (at least 80%).** Tests whether the component where the correct answer is decodable sits inside the circuit that makes the flip. Fates: bypass and not routed; it also locates where an override enters. Follow-up: patching. Cost: a fixed number of forward and backward passes. Code: not stated (the method is from Hanna et al.).
2. **Class-level path-patching share.** The share of the deciding heads' input that comes from one class of heads (for us: "own earlier answer" vs "user claim"). Fates: not routed, override. It is already causal.
3. **Functional head scores (copy score, CSPA, induction, etc.).** Adapt the copy score to the user's claimed answer vs the model's own answer to label heads. Cheap, weight-based.
4. **Node and edge Jaccard (raw, EWMA, vs final).** Circuit overlap across pressures (H1) or before vs after a fix (Stage B). Free once you have circuits.
5. **Summed class score over time.** Did total "user-claim copying" fall after the fix, or move to other heads?

## Controls worth copying
- Minimal pairs that flip the expected answer.
- Logit diff, not prob diff, when there is a single right/wrong pair (it shows negative heads).
- Count a head only if its functional score is high AND ablating it hurts.
- A faithfulness threshold found by binary search.
- Normalise by all upstream effect.
- Re-find the circuit per model and checkpoint.
- Drop poor-performance cases.

## Pitfalls
- Completeness can't be verified: "However, ensuring that a circuit is entirely complete, containing all relevant model nodes and edges, is challenging, and no definitive method of verifying this has emerged." (p3)
- Probes can be unfaithful: "This stands in contrast to probing [3], which only offers layer-representationlevel explanations, and can be unfaithful, capturing features unused by the model [21]." (p3)
- Self-repair can mask removal: "It seems plausible that self-repair [46, 58] contributes to this behavioral stability, but we leave the question of the exact “load-balancing” mechanism to future work." (p7)
- Size is noisy: "We also find a high degree of variability—circuit sizes can remain stable or fluctuate significantly, with no clear pattern based on the model or the task." (p9)
- A declining score is not declining importance: "this could happen if the number of heads exhibiting successor behavior decreases, while the impact of said heads (on the model’s logits) increases." (p6)
- Harder tasks may differ: "we stress that these algorithmic similarities might not hold for more complex tasks, for which a greater variety of algorithms could exist." (p9)
- Reader note (not stated in the paper): the similarity-to-final curves end at 1.0 by definition. Figure 6 shows checkpoints with circuits of about 0 nodes.

## Limitations and future work
- Limitations (p10): "Our analysis was limited to a narrow range of tasks feasible for small models. This limits in turn the scope of the claims that we can make."
- Also (p10): "Our analysis also studied models only from one model family, Pythia. It is thus not possible to tell if our results are limited"
- Also (p2): "Though we do not study fine-tuning or other post-training techniques"
- Future work (p10): "future work would do well to explore more complex phenomena, such as the self-repair and load-balancing mechanisms of LLMs, which ensure consistent task performance despite component fluctuations."

## Bears on fates
- **Stage B removed vs hidden (strong):** re-checking the pre-fix heads can mislead, because the function can move to other heads. Re-find the circuit on the fixed model.
- **Override (moderate):** IOI is a worked example of competing head classes on one logit diff, and the class share measures which one wins.
- **Not routed / bypass (moderate):** path patching through receivers measures whether a signal reaches the deciding heads.
- **Transformed (weak):** the only hints are implementation drift and a sign flip of role across families.
- **Readout and erased (weak):** not addressed.

## Bears on the group angle
- It supports groups at the *component* level: a function is stable as a class of heads while single heads change.
- Define head groups by functional score per model and checkpoint, and compare classes, not indices.
- Jaccard is a cheap set-level companion to RSA/CKA for H1.
- It says nothing about activation geometry, subspaces, Grassmannian paths, manifolds or token paths.

## Where in the architecture
- **Component types:** heads carry IOI and Greater-Than. SVA and Gendered Pronoun are MLP-dominant.
- **Depth:** in 160m's final IOI circuit, induction heads sit at layers 4-5, S-inhibition at 6-7, and name movers and copy suppression at 8-10, the last third if the model has 12 layers (the paper does not state the layer count).
- **Over training the location moves:** a layer-4 name mover is lost, and the successor role goes from layer 5 to layer 11.
- **Across sizes (one family):** the same classes appear, with more heads per role. Families, seeds and fine-tuning are not tested.

## Actionable fix or monitor
- Re-run circuit discovery and class scoring on every fixed checkpoint.
- Don't rely on single-head ablation as the fix, because self-repair compensates.
- Re-validate any head-based monitor after each update.
- The within-family transfer supports building on Gemma-2-9B and checking on 27B. There is no evidence across families or after RLHF.

## Reuse verdict
**Adapt.**
- Copy: the class-level path-patching share, the two-criterion head inclusion, and Jaccard overlap.
- Don't: trust EAP-IG alone, or assume the stability carries over to post-training or harder tasks.
