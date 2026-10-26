# SubTrack your Grad: Gradient Subspace Tracking for Memory and Time Efficient Full-Parameter LLM Training (arXiv 2502.01586v1; the slug says SubTrack++, but the local file is the v1 SubTrack-Grad paper)

`u2-subtrackpp-2502.01586`

**Bottom line:** An optimizer paper, not an interpretability paper. Borrow only the Grassmannian geodesic and principal-angle math for measuring how a subspace moves across layers or tokens, and heed its warning that single-snapshot SVD subspaces are noisy and jump. Its rank-1 smoothing would hide real jumps, so do not use it as a jump detector.

**Relevance to project:** Low, lower than the title suggests. This is an optimizer-efficiency paper about gradient subspaces during training; it never looks at activations, knowledge, behaviour under pressure, or interpretability, and says nothing about architecture location. Its only value for us is the Grassmannian subspace math and the warning that single-snapshot SVD subspaces are noisy and jump. Note: the local file is v1 ('SubTrack your Grad'), not the SubTrack++ version the slug names, so later additions in ++ are not covered.

**Reuse verdict:** adapt (math only): borrow the Grassmannian geodesic, tangent and principal-angle formulas (about 20 lines) as a subspace-motion measure across layers or tokens with a noise floor. Avoid the optimizer itself for diagnosis, and do not rely on its rank-1 smoothing, which hides jumps.

## Setup

- **task type:** Optimizer efficiency. No activations, no interpretability, no knowledge/behaviour analysis. The object being tracked is the low-rank subspace of each weight matrix's GRADIENT during training.
- **data:** Fine-tuning: GLUE (CoLA, STS-B, MRPC, RTE, SST-2, MNLI, QNLI, QQP) and SuperGLUE (BoolQ, CB, COPA, WIC, WSC, AXg). Pre-training: C4 for 10K iterations.
- **models with sizes:** RoBERTa-Base (GLUE, ranks 4 and 8), RoBERTa-Large (SuperGLUE, rank 8); Llama-style architectures of 60M, 130M, 350M, 1B and 3B parameters trained from scratch (ranks 128/256/256/512/512; Table 14 gives hidden 512-4096, layers 8-32). Parameter counts of RoBERTa models are not stated in the paper.
- **positions layers read:** Not applicable: every weight matrix's m x n gradient is projected onto a rank-r subspace (left singular vectors if m <= n, else right). No per-layer reporting.
- **method:** Initialise subspace by SVD of the first gradient; every k steps solve least squares A = argmin ||S A - G||, residual R = G - S A (in the orthogonal complement of S), tangent vector -2 R A^T, keep its rank-1 approximation, and step along the Grassmannian geodesic with step size eta (Eq. 6), which keeps S orthonormal.
- **baselines:** Full-Rank Adam, GaLore (periodic SVD of the gradient), BAdam (block coordinate descent).
- **controls:** All shared hyperparameters equal across methods; wall-time measured over a fixed number of subspace updates (5 for fine-tuning, 10 for pre-training) with evaluation steps excluded; NVIDIA A100 40GB; update-interval sweep 50-500 (Fig. 4, Table 15); toy 2D Ackley-function comparison (Fig. 3). No seeds or error bars reported.

## Figures and tables (20 of 20 in manifest)

### Figure 1 (p. 1)
- **What is plotted:** Schematic: a curved Grassmannian surface with two subspaces (green 'Subspace one', blue 'Subspace two'); projected gradients G~1..G~t lie on subspace one, G~t+1..G~t+k on subspace two; a red tangent vector grad-F at S1..St and a dashed geodesic leading to the new point St+1..St+k.
- **Takeaway:** Between updates the gradient is projected onto a fixed subspace; at an update the subspace moves along a geodesic in the direction of the tangent of the estimation-error loss.
- **Numbers:** none
- **Text matches figure:** yes: matches Section 3 and Algorithm 1.

### Table 1 (p. 6)
- **What is plotted:** GLUE scores for RoBERTa-Base fine-tuned 30 epochs: Full-Rank, BAdam, GaLore r=4, SubTrack-Grad r=4, GaLore r=8, 'SubTrack(Last)-Grad' r=8.
- **Takeaway:** SubTrack-Grad is within about 0.1-0.3 average points of GaLore; both are slightly below full-rank; BAdam is lowest, mainly from CoLA and RTE.
- **Numbers:** Avg: Full-Rank 85.86; BAdam 83.10; GaLore r=4 85.18; SubTrack r=4 85.11; GaLore r=8 84.67; SubTrack(Last) r=8 84.91. CoLA: Full 62.57, BAdam 54.44, GaLore r=4 60.34, SubTrack r=4 61.07. RTE: Full 77.98, BAdam 68.59.
- **Text matches figure:** partly: the text says 'superior or comparable'; at r=4 SubTrack is marginally below GaLore on average (85.11 vs 85.18), so 'comparable' holds, 'superior' does not. The r=8 row is labelled 'SubTrack(Last)-Grad' and the variant is never explained.

### Table 2 (p. 6)
- **What is plotted:** SuperGLUE scores for RoBERTa-Large, rank 8, 30 epochs: Full-Rank, GaLore, BAdam, SubTrack-Grad on BoolQ, CB, COPA, WIC, WSC, AXg.
- **Takeaway:** SubTrack-Grad has the best average; BAdam collapses on CB, COPA and AXg.
- **Numbers:** Avg: Full-Rank 80.64; GaLore 81.54; BAdam 62.93; SubTrack 81.95. CB: BAdam 53.28 vs others 88.85-90.33. AXg: GaLore and SubTrack 100.00, Full-Rank 96.30, BAdam 51.85.
- **Text matches figure:** partly: numbers match the text's claim; the caption lists AXb (Matthews correlation) but the table has no AXb column.

### Figure 2 (p. 7)
- **What is plotted:** Bar charts of wall-time in seconds: (a) Llama 60M-3B pre-training; (b) RoBERTa-Large on SuperGLUE tasks; (c) RoBERTa-Base on GLUE tasks; bars for GaLore, SubTrack-Grad, Full-Rank, BAdam.
- **Takeaway:** GaLore's overhead grows with model size (tallest bar at 3B); SubTrack-Grad sits between full-rank and GaLore everywhere.
- **Numbers:** (a) 3B: GaLore about 2700 s, SubTrack about 1400 s, Full-Rank about 1050 s, BAdam about 1000 s (approx, read from figure; exact values in Table 6: 2715.5 / 1385.7 / 1055.9 / 1004.1). (b) BoolQ about 1400-1550 s for all but BAdam about 800 s (approx, read from figure).
- **Text matches figure:** partly: the caption says panel (a) shows 'the memory and wall-time', but the panel only plots wall-time (memory is only in Table 7). Bar heights agree with Tables 6, 8, 10.

### Table 3 (p. 7)
- **What is plotted:** Evaluation loss on C4 after 10K pre-training iterations for Llama 60M-3B: Full-Rank, GaLore, BAdam, SubTrack-Grad.
- **Takeaway:** SubTrack-Grad is slightly worse than GaLore at every size except a tie at 1B, and both low-rank methods are far worse than full-rank at 1B and 3B; BAdam is worst.
- **Numbers:** Avg: Full 3.84; GaLore 4.87; BAdam 7.39; SubTrack 4.97. 1B: Full 4.61, GaLore 6.53, SubTrack 6.53. 3B: Full 4.52, GaLore 6.57, SubTrack 6.59. 60M: Full 3.41, GaLore 4.02, SubTrack 4.13.
- **Text matches figure:** partly: the abstract's 'better or on-par results compared to GaLore' does not hold for pre-training loss (SubTrack avg 4.97 vs GaLore 4.87). Full-rank loss is higher at 1B/3B (4.61/4.52) than at 60M (3.41), consistent with the short 10K-step budget and the lr 1e-5 / batch 8 used for 1B/3B in Table 14 (my reading; the authors only say the setup was for comparison).

### Table 4 (p. 7)
- **What is plotted:** Optimizer-state memory and subspace-update time complexity for an m x n gradient, rank r.
- **Takeaway:** SubTrack keeps GaLore's memory (mr + 2nr) but replaces the O(nm^2) SVD with O(mnr) operations.
- **Numbers:** Adam memory 2mn; GaLore mr+2nr, update O(nm^2); SubTrack-Grad mr+2nr, update O(mnr).
- **Text matches figure:** yes.

### Figure 3 (p. 8)
- **What is plotted:** Optimization paths of Adam+SubTrack-Grad (a, d) and Adam+GaLore (b, e) on the 2D Ackley function, 100 steps, update interval 10, scale factor 1 (top) and 3 (bottom); 3D overlays (c, f).
- **Takeaway:** At scale 3 GaLore's path shows visible loops/zig-zags near the minimum while SubTrack's path is smooth; at scale 1 the two paths look almost the same, with GaLore's end marker sitting slightly off the centre.
- **Numbers:** none (single toy function, single start point; read from figure).
- **Text matches figure:** partly: the scale-3 jumps are visible; the scale-1 difference ('can not reach the global minimum') is small by eye. It is one 2D toy example, so it illustrates rather than tests the noise argument.

### Figure 4 (p. 8)
- **What is plotted:** Wall-time (s) vs subspace update interval (50-500) for GaLore and SubTrack-Grad: RoBERTa-Base rank 8 on CoLA (10 epochs) and Llama-60M rank 128 on C4 (2500 iterations).
- **Takeaway:** GaLore's wall-time rises steeply as updates get more frequent; SubTrack-Grad stays nearly flat.
- **Numbers:** RoBERTa-Base: GaLore about 1040 s at interval 50 down to about 460 s at 500; SubTrack about 417 s to about 408 s. Llama-60M: GaLore about 758 s at 50 to about 679 s at 500; SubTrack about 696 s at 50 to about 676 s at 500 (approx, read from figure). The RoBERTa values agree with Table 15 (1038.79 / 458.79 and 417.07 / 408.22).
- **Text matches figure:** yes for RoBERTa; for Llama-60M SubTrack also rises by about 20 s at interval 50 (approx, read from figure), so 'minimal overhead' is relative to GaLore.

### Table 5 (p. 15)
- **What is plotted:** Per-step time complexity of the SubTrack-Grad subspace update.
- **Takeaway:** The O(mnr) matrix products dominate; no SVD of the full gradient.
- **Numbers:** Cost function O(mr^2); residual O(mrn); partial derivative O(mnr); tangent vector O(m^2 r); rank-1 approximation O(mr^2); update rule O(mr^2); overall O(mnr).
- **Text matches figure:** yes (minor: Appendix C calls the update rule Eq. (7), but the geodesic update is Eq. (6)).

### Table 6 (p. 15)
- **What is plotted:** Wall-time (s) for 2000 pre-training iterations (10 subspace updates) of Llama 60M-3B, with average % change vs Full-Rank and vs GaLore.
- **Takeaway:** GaLore's cost explodes at 1B-3B; SubTrack-Grad cuts it by about half at 3B.
- **Numbers:** 3B: Full 1055.9, BAdam 1004.1, GaLore 2715.5, SubTrack 1385.7. Avg: Full 997.3; BAdam 811.0 (-18.68%); GaLore 1535.2 (+53.94%); SubTrack 1127.7 (+13.08% vs FR, -26.54% vs GaLore).
- **Text matches figure:** yes: 2715.5/1055.9 = 2.57 (+157%), 1385.7/1055.9 = 1.31 (+31%), (2715.5-1385.7)/2715.5 = 49%, matching the abstract. Full-rank wall-time is lower at 1B/3B than at 350M because batch size drops from 64 to 8 (Table 14).

### Table 7 (p. 16)
- **What is plotted:** Peak memory (units not stated; A100 40GB) for Llama 60M-3B pre-training.
- **Takeaway:** SubTrack-Grad uses the same memory as GaLore; both save about 10% vs full-rank on average, about 25% at 3B.
- **Numbers:** 3B: Full 34.92, BAdam 15.25, GaLore 26.03, SubTrack 26.16. Avg: Full 24.92; BAdam 15.88 (-36.28%); GaLore 22.30 (-10.51%); SubTrack 22.33 (-10.39%, +0.13% vs GaLore).
- **Text matches figure:** yes.

### Table 8 (p. 16)
- **What is plotted:** Wall-time (s) for RoBERTa-Base GLUE fine-tuning up to 5 subspace updates, ranks 4 and 8.
- **Takeaway:** SubTrack-Grad is about 16% faster than GaLore on average.
- **Numbers:** Avg: Full 144.1; BAdam 113.5 (-21.24%); GaLore r=4 210.7 (+46.22%); SubTrack r=4 176.3 (+22.48%, -16.23 vs GaLore); GaLore r=8 210.7; SubTrack r=8 176.6 (+22.35, -16.33%).
- **Text matches figure:** partly: the abstract's 'up to 20.57% on GLUE tasks (15% average reduction)' cannot be reproduced from this table. My per-task calculation gives the largest reduction on SST-2 r=4 (185.0 to 141.2 = 23.7%) and CoLA (23.3%), and the table's own average is 16.2-16.3%.

### Table 9 (p. 17)
- **What is plotted:** Peak memory for RoBERTa-Base GLUE fine-tuning, ranks 4 and 8.
- **Takeaway:** Same memory as GaLore; about 10.8% below full-rank.
- **Numbers:** Avg: Full 6.21; BAdam 2.43 (-60.87%); GaLore r=4 5.57 (-10.31%); SubTrack r=4 5.54 (-10.79%, -0.54%); r=8 same averages.
- **Text matches figure:** yes.

### Table 10 (p. 17)
- **What is plotted:** Wall-time (s) for RoBERTa-Large SuperGLUE fine-tuning up to 5 subspace updates.
- **Takeaway:** Largest gain on small tasks where the update cost dominates (COPA, AXg).
- **Numbers:** Avg: Full 355.3; BAdam 212.25 (-40.26%); GaLore 523.4 (+47.31%); SubTrack 404.62 (+13.88%, -22.69% vs GaLore). COPA: GaLore 235.3 vs SubTrack 82.2.
- **Text matches figure:** yes: COPA (235.3-82.2)/235.3 = 65% and the 22.69% average match 'up to 65% ... 22% average'.

### Table 11 (p. 18)
- **What is plotted:** Peak memory for RoBERTa-Large SuperGLUE fine-tuning.
- **Takeaway:** Same memory as GaLore.
- **Numbers:** Avg: Full 11.37; BAdam 5.17 (-54.52%); GaLore 9.24 (-18.73%); SubTrack 9.16 (-19.44%, -0.87% vs GaLore).
- **Text matches figure:** yes.

### Table 12 (p. 18)
- **What is plotted:** Hyperparameters for RoBERTa-Base GLUE: batch size, learning rate, epochs, max length; SubTrack step size per task; update interval; alpha; BAdam settings.
- **Takeaway:** SubTrack's step size is tuned per task over four orders of magnitude, a real tuning burden.
- **Numbers:** Batch 16 (CoLA 32), lr 1e-5 (MRPC, CoLA 3e-5), 30 epochs, max length 512, update interval 500, alpha 4 (r=4) / 2 (r=8). Step size r=4: MNLI 0.1, SST-2 0.001, MRPC 5.0, CoLA 5.0, QNLI 0.01, QQP 1.0, RTE 8.0, STS-B 10.0; r=8: 0.1, 0.1, 5.0, 13.0, 0.1, 1.0, 5.0, 10.0. BAdam block switch 100, random.
- **Text matches figure:** yes (Appendix E says these match GaLore's settings with interval 500).

### Table 13 (p. 18)
- **What is plotted:** Hyperparameters for RoBERTa-Large SuperGLUE.
- **Takeaway:** Per-task step size (1 to 50) and update interval (100 to 500) again tuned per task.
- **Numbers:** Batch 16, 30 epochs, lr 1e-5, max length 512, rank 8, alpha 4. Step size / interval: BoolQ 0.1/500, CB 1.0/100, COPA 50.0/100, WIC 50.0/500, WSC 1/250, AXg 1.0/100. BAdam block switch 100, 50, 50, 100, 50, 50.
- **Text matches figure:** yes (values read from the page image; the extracted text is scrambled).

### Table 14 (p. 19)
- **What is plotted:** Llama architectures and pre-training hyperparameters.
- **Takeaway:** The 1B and 3B runs use lr 1e-5 and batch 8 for only 10K iterations, so they are far from converged.
- **Numbers:** Hidden 512/768/1024/2048/4096; intermediate 1376/2048/2736/5461/11008; heads 8/12/16/24/32; layers 8/12/24/32/32; lr 1e-3 (60M-350M), 1e-5 (1B, 3B); batch 128/128/64/8/8; 10K iterations; grad accumulation 2; clipping 1.0; warmup 1000; scale 0.25; bfloat16; rank 128/256/256/512/512; interval 200; SubTrack step size 10.
- **Text matches figure:** yes.

### Table 15 (p. 19)
- **What is plotted:** Wall-time and CoLA performance (Matthews correlation) for GaLore and SubTrack-Grad as the update interval shrinks from 500 to 50 (RoBERTa-Base).
- **Takeaway:** GaLore's time more than doubles with frequent updates; SubTrack's rises about 2%; performance is similar.
- **Numbers:** GaLore time 458.79 (500) to 1038.79 (50), performance 0.5358 to 0.5463; SubTrack time 408.22 to 417.07, performance 0.5332 to 0.5673.
- **Text matches figure:** yes. Note these CoLA scores (about 0.53-0.57) are lower than Table 1 (58-62) because this run is 10 epochs (Fig. 4 caption) vs 30.

### Figure 5 (p. 19)
- **What is plotted:** Training loss vs wall-time for Llama-3B on C4 and RoBERTa-Large on BoolQ; Full-Rank, GaLore, SubTrackGrad.
- **Takeaway:** Llama-3B: all three fall from about 40 to about 6-7 almost immediately and plateau around 5-6; GaLore's run stretches to about 12,500 s vs about 6,000 s for full-rank and SubTrack. BoolQ: full-rank drops fastest (about 0.1 by about 3,000 s); GaLore and SubTrack are slower, with SubTrack slightly above GaLore most of the run; all end near 20,000-21,000 s (approx, read from figure).
- **Numbers:** approx, read from figure (see takeaway).
- **Text matches figure:** partly: 'convergence pattern comparable' holds against GaLore, but on BoolQ both low-rank methods converge visibly slower than full-rank, and the 3B panel's y-range (0-45) hides differences in the plateau.

## Key results

- Wall-time savings vs GaLore in fine-tuning.
  > "SubTrack-Grad reduces wall-time by up to 20.57% on GLUE tasks (15% average reduction) and up to 65% on SuperGLUE tasks (22% average reduction) compared to GaLore." (p. 1)
- At 3B pre-training, GaLore's overhead is +157% vs full-rank and SubTrack's +31% (consistent with Table 6).
  > "Notably, for a 3B parameter model, GaLore incurred a substantial 157% increase in wall-time compared to full-rank training, whereas SubTrack-Grad exhibited a 31% increase" (p. 1)
- Task performance is comparable to GaLore despite rank-1 subspace updates (true for fine-tuning; pre-training loss is slightly worse, Table 3).
  > "Despite limiting updates to rank-1 modifications of the previous subspace, SubTrack-Grad achieves superior or comparable performance compared to GaLore." (p. 6)
- Optimizer-state memory equals GaLore's (Table 7: +0.13%).
  > "Additionally, the memory required for storing optimizer states in SubTrack-Grad, is equivalent to GaLore's." (p. 7)
- Frequent subspace updates are cheap for SubTrack but expensive for GaLore (Fig. 4, Table 15).
  > "GaLore's runtime significantly increases with more frequent subspace updates, whereas SubTrack-Grad maintains minimal overhead." (p. 7)
- Single-snapshot SVD re-estimation causes optimization jumps on a toy function (Fig. 3).
  > "These jumps prevent GaLore from reaching the global minimum within 100 steps when the scale factor is set to 1." (p. 7)
- Partial (block) updates hurt performance, more so at larger scale.
  > "its inability to update all the parameters simultaneously has caused performance loss, and the loss gets greater as the model size increases." (p. 6)

## Methods

### Grassmannian subspace tracking with a rank-1 geodesic step (SubTrack-Grad update, Eq. 3-6, Algorithm 1)
- **what it measures:** How the dominant rank-r subspace of a sequence of matrices moves: fit the new matrix by least squares in the current subspace, take the residual in the orthogonal complement, form the tangent vector, and step along the geodesic. Quote: 'The tangent vector ∇F provides the direction for adjusting the subspace by accounting for the error lying in the orthogonal complement.' The size of that tangent or of the step is a rotation-speed signal.
- **which fate it could diagnose:** Only if adapted from gradients to activations: 5 transformed (the correct-answer subspace rotates between layers, tokens, or pressure conditions but keeps its energy) vs 6 erased (energy in the subspace and its rotated neighbourhood collapses into the residual). Also direction 3 (jump vs drift) via the per-step rotation profile.
- **causal followup it allows:** None directly. It can pick candidate layers or tokens (largest rotation) where patching or ablation should be tried.
- **data and compute needed:** Cheap: O(mnr) per update (Table 5). For us, stacked activations (items x d_model) per layer or token for a few hundred items; no GPU beyond activation extraction.
- **code available:** Not stated in the paper (no repository link in the text).

### SVD-based subspace (GaLore baseline): re-estimate the subspace from scratch by SVD of one matrix
- **what it measures:** The top-r singular subspace of a single snapshot.
- **which fate it could diagnose:** Same as above, but the paper argues it is noise-sensitive and produces jumps; it is a cautionary baseline for us.
- **causal followup it allows:** None.
- **data and compute needed:** O(nm^2) per SVD (Table 4).
- **code available:** GaLore is a separate public method; not provided by this paper.

### Memory-efficient full-parameter training via projected optimizer states
- **what it measures:** Not a measurement: it reduces Adam state memory from 2mn to mr + 2nr per matrix.
- **which fate it could diagnose:** None. Relevant only to Stage B cost (full-parameter DPO or fate-matched fine-tuning on 9B-27B).
- **causal followup it allows:** Allows full-parameter rather than LoRA fine-tuning under a memory limit, so a 'fix' can touch all weights.
- **data and compute needed:** Tested only up to 3B pre-training (10K iterations) and RoBERTa fine-tuning on an A100 40GB; never on DPO or 7B+ instruction models.
- **code available:** Not stated in the paper.

### Toy optimization-path visualization on the Ackley function
- **what it measures:** Qualitative smoothness of optimization paths.
- **which fate it could diagnose:** None.
- **causal followup it allows:** None.
- **data and compute needed:** Trivial.
- **code available:** Not stated.

## Controls worth copying

- When comparing two subspace-estimation procedures, fix the number of update events (here exactly 5 or 10 subspace updates) so costs are comparable, and exclude evaluation steps from timing.
- Sweep the update frequency (interval 50-500, Table 15) and report both cost and outcome, rather than one setting.
- Keep all shared hyperparameters identical across methods.
- For our use: compare a smoothed (tracked) subspace with an unsmoothed per-layer SVD subspace; if they disagree on where a jump is, the jump may be noise.

## Pitfalls

- A subspace estimated by SVD of a single matrix is noise-sensitive; for us, subspaces from a few hundred items per layer will jitter, and adjacent-layer 'jumps' may be noise.
  > "Moreover, applying SVD to a single gradient matrix is susceptible to data noise (Vaswani et al., 2018)" (p. 2)
- Not all layers have a stable subspace (cited claim, not shown here); one subspace per model is not safe.
  > "This is problematic because not all layers' gradients converge to a stable subspace early in training (Jaiswal et al., 2024)." (p. 2)
- Higher-rank geodesic steps were unstable, so updates were restricted to rank-1. For diagnosis this design hides abrupt multi-dimensional changes: a tracker built to change slowly cannot detect a 'jump'.
  > "caused convergence issues in some cases. Therefore, we restricted updates to rank-1 in this paper" (p. 9)
- Full-rank SVD of a tangent with tiny singular values hurts convergence.
  > "using the full-rank SVD of the tangent vector from (5), can hinder convergence if the singular values of the tangent vector become very small." (p. 7)
- The convergence proof assumes the subspace never changes, so it gives no guarantee for the tracking itself.
  > "This is where the constant subspace assumption becomes necessary." (p. 11)
- Step size trades tracking against stability; a decaying step size stops tracking.
  > "However, a decreasing step size can impair the ability to accurately track and adapt to subspace changes." (p. 5)
- Short pre-training budget: models are not converged, so pre-training comparisons are weak.
  > "this setup was sufficient for comparing methods rather than creating perfectly optimized models." (p. 16)
- Not stated by authors: the step size is tuned per task from 0.001 to 50 (Tables 12-13), the abstract's 'up to 20.57% on GLUE' is not reproducible from Table 8 (largest per-task reduction about 23.7%), no seeds or error bars are given, and pre-training loss is slightly worse than GaLore (Table 3).

## Limitations and future work (authors)

- **Limitations:** "In some cases, the extent of changes in the subspace may require updates of ranks greater than 1; the pre-training of Llama-based architectures is a clear example of this need." (p. 7)
- **Future work:** "In future work, we plan to explore increasing the rank of updates without compromising convergence." (p. 9)

## Bearing on the six fates

| Fate | How | Strength |
|---|---|---|
| 5 transformed | The geodesic/tangent machinery (or plain principal angles between per-layer subspaces) could measure how far the correct-answer subspace rotates between layers, tokens, or pushback vs no-pushback. A small rotation that keeps the energy fits 'transformed'. This is my extrapolation; the paper only tracks gradients. | weak |
| 6 erased | The residual R = G - S A (energy outside the tracked subspace) is a natural erasure measure: erasure means projected energy collapses with no nearby rotated subspace to absorb it. Extrapolation; not tested. | weak |
| Stage B fixes (H2, any fate) | Projected-optimizer training could make full-parameter fine-tuning (plain DPO or fate-matched) on 9B-27B fit in memory, but the paper tests nothing above 3B and no preference training. | weak |

## Bearing on the group / subspace / token-path angle

Helps: a worked, cheap recipe for following a rank-r subspace through a sequence of matrices on the Grassmannian. It fits the current subspace by least squares, takes the residual in the orthogonal complement, builds a tangent vector, and steps along a geodesic that keeps the basis orthonormal ('This update rule preserves the orthonormality of St+1 , ensuring it remains on the Grassmannian.', p. 4). For us the sequence index would be layer or token, and the matrix would be stacked activations or flipped-minus-resisted differences. The per-step rotation gives a profile across depth, and a spike is a candidate fork point; this maps onto direction 3 (jump vs drift). Warnings: (1) single-snapshot SVD subspaces are noisy and jump, so we need bootstrap or label-permutation noise floors before calling a jump real; (2) SubTrack-style smoothing with the previous subspace and rank-1 steps deliberately suppresses abrupt change, so it would bias a jump-vs-drift test toward 'drift'. For diagnosis, prefer unsmoothed principal angles between independently estimated subspaces plus a noise floor, and use tracking only for denoising. Nothing in the paper touches activations.

## Architecture location

Nothing about where information lives. The method is applied independently to each weight matrix's gradient; there is no per-layer, per-component, or depth analysis of subspace behaviour. The only layer-level statement is cited from Jaiswal et al. (2024), not shown. Efficiency results cover Llama 60M-3B and RoBERTa-Base/Large, single runs with no seeds reported, so there is nothing on consistency across models or seeds beyond the efficiency trend.

## Actionable fix or monitor

No fix for any fate follows from this paper. Indirect uses: (a) a monitor, i.e. a Grassmannian-distance or principal-angle profile of the correct-answer subspace across layers, to localise where it rotates or collapses before running patching; (b) for Stage B, GaLore/SubTrack-style projected optimizer states could allow full-parameter DPO on 9B-27B under a memory limit (optimizer memory mr+2nr vs 2mn, Table 4). This is untested for DPO or for models above 3B, and LoRA-DPO is the cheaper standard. The authors also speculate that low-rank optimization 'may also serve as a regularizer' (p. 9), which could change what a fine-tuning fix does compared with full fine-tuning.

## Sections read

Abstract, Figure 1 (page 1), 1. Introduction (incl. contributions list), 2. Related Works, 3. SubTrack-Grad: Tracking the Gradient Subspace (incl. Eq. 1-6, Algorithm 1), 4. Theoretical Analysis (Def. 4.1 L-continuity, Thm 4.2 Convergence, Def. 4.3 Exponential Map, Def. 4.4 Stiefel Manifold, Def. 4.5 Grassmann Manifold, Thm 4.6 Grassmann Exponential), 5. Experiments and Results: Fine-Tuning Experiments, 5. Experiments and Results: Pre-Training Experiments, 5. Experiments and Results: Time and Space Complexity, 5. Experiments and Results: Reducing Optimization Jumps, 5. Experiments and Results: Runtime Consistency, 6. Discussion and Conclusion, References, Appendix A. Convergence of SubTrack-Grad (proof of Thm 4.2), Appendix B. Grassmann Exponential (proof of Thm 4.6, Def. B.1 Orthogonal Group), Appendix C. Time Complexity Analysis, Appendix D. Memory and Time Comparison, Appendix E. Fine-Tuning RoBERTa, Appendix F. Pre-Training LLama-Based Architectures, Appendix G. Time and Performance Consistency, Appendix H. Convergence vs. Wall-Time
