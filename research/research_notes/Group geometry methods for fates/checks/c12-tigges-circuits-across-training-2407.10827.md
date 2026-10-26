# Check: c12-tigges-circuits-across-training-2407.10827 (Tigges, Hanna, Yu, Biderman, "LLM Circuit Analyses Are Consistent Across Training and Scale", NeurIPS 2024)

Verdict: NEEDS FIXES (minor to moderate; no fabricated quotes or numbers; the problems are over-strong labels, a few figure misreads, and missing caveats and leads). Not a REDO: the reading covers every section, every figure and the appendices.

## Quote check
- `quote_check.py`: 27 quotes, 0 failed (it only tests the first 120 normalised characters).
- Stricter pass (my own script in the scratchpad): all 27 quotes match the full text, and each sits on the page the reading gives (the form-feed page split of paper.txt). 27/27 OK, including `limitations_quote`, `future_work_quote` (p10) and the footnote quote (p2).

## Figures vs manifest
- Manifest lists 9 figures (Fig 1-9), no tables. The reading has 9 figure entries. None missing. PNGs exist for p04, p06, p07, p08, p09, p23 (Fig 6 and 7), p24, p26. I opened all of them, and re-rendered p07, p08, p23 and p24 at 250 dpi to measure Fig 3, 4, 6, 7, 8.
- Skipped sections: none. `sections_read` covers the body, Appendix A-G and the references (skimmed).
- Most numbers agree with the plots within reading error (Fig 1 onsets and the 70m exception, Fig 2 windows, Fig 3 induction jump near 1.5e9, Fig 4B range, Fig 5 dips, Fig 9 accuracies).

## Errors (reading says / paper says / evidence)

1. **Stage B label "strong" is over-strong (bears_on_fates[0]), and pitfalls[3] glosses its quote too far.**
   - Reading says: strong; "after DPO or a fate-matched fix, re-checking the heads found before the fix can falsely show 'removed' when the function moved". Pitfall gloss: "Self-repair/load balancing can mask the effect of removing a component."
   - Paper says: it does not study fine-tuning or post-training (p2, footnote 1). Self-repair is only "plausible" and left to future work (p7). Nothing is ablated or fixed in the study; the head turnover is natural pretraining drift, mainly in Pythia-160m on toy tasks. The one ablation-based self-repair statement is the Fig 4A legend ("Negative heads (*) perform self-repair by reducing their negativity upon ablation of NMHs", p8), inherited from the original IOI work, and the reading does not mention it.
   - Fix: downgrade to "moderate (analogy from pretraining drift)". Reword the pitfall to "heads change role over pretraining while behaviour holds; the authors conjecture self-repair". Keep the advice as our design rule. Keep endogenous drift (this paper) separate from intervention-time compensation (hydra effect, self-repair papers), as the project rules require.

2. **actionable_fix (4) overreaches: "supports developing on Gemma-2-9B and checking on 27B".**
   - Paper says: circuits for Pythia 70M-2.8B only, toy tasks, one family with one data and architecture recipe. The authors warn results may not hold for other families or complex tasks (p9, p10). The algorithm-stability result is IOI only.
   - Evidence in Fig 4C: the 2.8B share of effect through S-inhibition heads is about 43-50% for roughly the second half of the plotted range, so the "same algorithm" claim is already weaker at the largest size they tested.
   - Fix: say "no evidence for 9B to 27B, different family; motivation only, test it".
   - Also actionable_fix (3) ("monitor needs re-validation after each training update") is our inference from pretraining drift. Label it.

3. **architecture_location: Greater-Than is called attention-head carried.**
   - Reading says: "attention heads carry IOI and Greater-Than ... SVA and Gendered Pronoun are MLP-dominant".
   - Paper says: it studies only the attention-head components of Greater-Than and notes that Greater-Than work found task-relevant MLPs and neurons that it leaves out (footnote 3, p5).
   - Fix: "IOI: attention heads. Greater-Than: successor and induction heads were scored, MLPs not analysed. SVA and Gendered Pronoun: MLP-dominant (stated, not shown)".

4. **architecture_location: "the successor role moves from layer 5 to layer 11" is overstated, and the Fig 3 successor numbers are off.**
   - Reading (Fig 3 entry): (5,9) "peaks at about 0.8 near 1e11 tokens". Plot (measured): peak 0.79 at about 4e10; 0.67 at 1e11; 0.61 at 2e11; 0.45 at 3e11. Head (11,0) stays near 0.03 until about 2e11 and reaches only about 0.35 at the end. So the "takeover" is partial, late, one head, and only the last third of the decline coincides with it.
   - Score type: the successor score is weight-based (OV x unembedding after MLP0), so it measures capability, not use. The paper has no ablation of (5,9) or (11,0).
   - Fix: "(11,0) gains about 0.3 in successor score over the last ~1e11 tokens while (5,9) loses about 0.35 over ~3e11; partial, one head, Greater-Than only."

5. **Fig 7 and Fig 8 takeaways: "all curves reach 1.0 at the final checkpoint" is false for one series.**
   - In Fig 7 and 8 (Gendered Pronoun), only 4 of 5 models rise to 1.0. The 1.4b series is a flat line at 0.0 that stops near 1.5e11 tokens. Fig 6 shows why: the 1.4b Gendered-Pronoun circuit is at 0 nodes for the last stretch of training, so the final circuit is empty and similarity-to-final is 0 at every checkpoint.
   - Fig 6 also shows 0-node stretches elsewhere: Gendered Pronoun 2.8b at two points near 3.5e4 and 5.5e4 steps; SVA 1.4b at 0 around 8e3-1.2e4 steps with no data after about 1.5e4; SVA 410m flickering between about 0 and 90 nodes late. The reading's "isolated checkpoints" understates this: the circuit finder fails for whole stretches on the MLP-dominant tasks.
   - Why it matters: factual recall is probably MLP-heavy, and the reading's `reuse_verdict` leans on EAP-IG. Fix: add "sustained empty circuits on MLP-dominant tasks; one similarity series degenerate" to pitfalls and to the Fig 6-8 entries.

6. **Fig 4: "stays roughly flat" and "a few points just below 50" are generous.**
   - Plot: B is flat (about 70-90). In C the 160m share falls from about 96 to about 72 over training, and 2.8b falls from about 68 to about 43-50, sitting at or below 50 for about half its points after 1e11. In D the 2.8b series swings 45-82 and 160m dips to about 40 near 5e10.
   - Paper's "generally above 50%" is true for three of four sizes. Fix: "flat in B; C declines with training and with size; 2.8b is at about 45-50 in C and D late".

7. **Figs 2 and 3 "name mover" curves are copy score only, and the two-criterion head rule is not what Fig 2/3 use.**
   - Reading (methods[2]): "the paper requires both a score > 10% and a harmful ablation effect before counting a head".
   - Paper: that two-part rule is Section 4.2 / Fig 4 (p7). Figs 2 and 3 use functional scores only (Fig 2 sums them over EAP-IG circuit heads). Copy score is OV-only ("simulates what will happen if the head attended perfectly", App D), so it does not show the head attends to names or matters.
   - Evidence: the Fig 3 legend lists (4,6), (4,8), (4,10), (4,11) and (5,0), which Fig 4A lists among the final induction heads (4.11, 4.6, 5.0), not name movers (8.10, 8.2, 9.4, 10.7). So "name-mover head (4,6) loses its role" (key_results[2], architecture_location "an early name mover in layer 4") is a copy-score statement for an early head that is an induction head in the final circuit; it is probably a role shift, not a name-mover loss. The paper does not say this.
   - Fix: say "copy-score heads", note the (4,6) overlap, and scope the two-criterion rule to Fig 4.

8. **Circuit size vs model size (key_results[8], Fig 6 takeaway, key_results[10]).**
   - The correlations (r = 0.6-0.9, p9) are over 5 model sizes (70M-2.8B), so n = 5 with no p-values (r = 0.72 at n = 5 is about p = 0.17). Fig 6 is not monotone: late in IOI the 410m circuit is about the size of 2.8b (~120-130 nodes), and in Gendered Pronoun 1.4b is above 2.8b for most of training.
   - "More heads take on the same roles" (App C, p23) is the authors' reading of node counts alone (nodes include MLPs). No per-head role analysis supports it.
   - Fix: add "n = 5, interpretation untested" to both entries.

9. **Pitfall that the reading proposes for reuse: summed, normalised class scores (methods[4], "Stage B removed vs hidden").**
   - Fig 2 sums scores over the heads in the found circuit, and Fig 6 shows circuit size swinging between 0 and about 185 nodes. A falling sum can reflect circuit churn, not weaker function. Per-model normalisation (peak = 1 by construction) would also erase a before/after fix effect.
   - The reading flags the normalisation under pitfalls but still proposes the metric for Stage B without saying "use raw scores over a fixed head set". Fix: add that.

## Omissions
- **Leads the paper points to that the reading never lists** (relevant to "get the best ideas from all these papers"):
  - Prakash et al., "Fine-tuning enhances existing mechanisms" (ICLR 2024, ref 56): circuits in 7B models after fine-tuning. This is the closest published evidence for Stage B (does a fix create new circuitry or amplify the old one). Cited on p1.
  - McGrath et al., hydra effect (ref 46) and Rushing and Nanda, self-repair (ref 58): the direct sources for the "masking" worry.
  - Hanna et al., "Have faith in faithfulness" (ref 33): the EAP-IG faithfulness method behind the 80% rule.
  - Zhang and Nanda, patching best practices (ref 75): why logit difference is preferred.
  - Elazar et al., amnesic probing (ref 21): the "decodable is not used" citation inside the probing pitfall.
- **Appendix gap not flagged.** Section 2.4 (p3) says the Greater-Than and IOI circuits are verified in Appendix A, and p3-5 say the SVA and Gendered-Pronoun circuits are checked as MLP-dominant. The appendix contains only A.1 (IOI criteria) and a generic path-patching description (B). No verification result is shown for Greater-Than, SVA or Gendered Pronoun, so the Greater-Than component claims (successor and induction heads) rest on scores of in-circuit heads alone.
- **Emergence is correlational.** The paper says "we can be reasonably sure" the heads cause the performance emergence (p6) from timing alone; nothing is intervened on during training. Add to pitfalls.
- **No seeds, error bars or tests.** One training run per size. The reading mentions seeds only in architecture_location; add to pitfalls.
- **No layer indices for 410M-2.8B.** The paper gives layer.head indices only for Pythia-160m, so "relative depth stable across models" (our goal) has no evidence here, only class types and effect shares. Say so in architecture_location.
- **Fig 5 definition is unclear.** The y-axis is labelled "Weighted Jaccard Similarity" for node sets, and the text defines x_t as similarity to "all previous checkpoints" while the recursion uses the previous one. The reading presents it as a plain EWMA of node-set Jaccard.

## Fate labels, architecture_location, actionable_fix (summary)
- Fate labels: Stage B "strong" -> moderate or weak (item 1). Fate 4 override "moderate": the name-mover versus negative-name-mover competition is within-circuit calibration on one correct answer, not two answer sources, so treat it as a method analogy. Fates 2/3 "moderate": method transfer only, since no probe or decodability is measured. Fates 5, 1, 6 "weak" are right.
- architecture_location: IOI depth for 160m is correct (answer-writing heads at layers 8-10 of 12; the 12 layers are not stated in the paper but are correct for Pythia-160m). Fix items 3, 4, 7 and the no-depth-for-larger-models omission.
- actionable_fix: (1) class-level shares and (2) "do not rely on single-head ablation" are reasonable as our rules; label them as inference. Fix items 1 and 2.
- reuse_verdict "adapt" and relevance "moderate" are reasonable. `bears_on_group_angle` is accurate that the paper is graph-level only (no subspaces, Grassmannian paths, Procrustes, CKA or token paths).
- Vocabulary: uses "decodable", no "knows". Good.

## Priority fixes for whoever edits the reading
1. Items 1 and 2 (strength labels, Gemma scale claim).
2. Items 3, 4, 7 (what is actually carried by attention heads, the successor "move", copy-score identity of Fig 2/3 heads).
3. Item 5 (empty circuits; degenerate similarity series) and item 9 (use raw scores for Stage B).
4. Add the leads (refs 56, 46, 58, 33) and the Fig 4A self-repair legend.
