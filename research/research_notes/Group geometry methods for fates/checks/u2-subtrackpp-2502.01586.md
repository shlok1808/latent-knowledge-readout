# Check: u2-subtrackpp-2502.01586 (SubTrack your Grad, arXiv 2502.01586v1)

Verdict: **OK** (four small optional edits listed below; no misreading of a main result)

## 1. Quote check
- `python3 quote_check.py u2-subtrackpp-2502.01586`: 16 quotes, 0 failed.
- That script only tests the first 120 normalised characters and does not test page numbers, so I also checked (a) the full text of every quote and (b) that each quote sits on the page the reading gives, using per-page `pdftotext` of `paper.pdf`. All 16 full quotes are present and all 16 page numbers are correct.

## 2. Coverage checks
- Figure/table entries: manifest lists 20 (Fig 1-5, Tables 1-15); the reading has all 20.
- `sections_read` covers every section and Appendices A-H. Nothing skipped.
- I opened p06, p07, p08, p18, p19 and compared the reading's figure/table entries with the images and with `paper.txt`.
- Recomputed from the paper's tables: Table 1/2/3/6 averages, Table 6 ratios (2715.5/1055.9 = 2.57, 1385.7/1055.9 = 1.31, 49% cut vs GaLore), Table 8 per-task reductions (largest 23.7% SST-2 r=4, 23.3% CoLA, mean 16.2-16.3%, so the abstract's "up to 20.57% / 15%" really is not reproducible), Table 10 (COPA 65.1%, mean of averages 22.7%). All the reading's numbers and its "cannot reproduce 20.57%" flag are correct.
- Tables 12 and 13 hyperparameters (the extracted text is scrambled) match the page image p18, including per-task step sizes 0.001-50.
- Fig 3, Fig 4 (both panels) and Fig 5 (both panels) descriptions match the images. The Fig 2 caption/panel mismatch ("memory and wall-time" but only wall-time plotted), the Table 2 caption listing AXb with no AXb column, the unexplained "SubTrack(Last)-Grad" row (appears only in Table 1; grep confirms), the Appendix C "Eq. (7)" vs Eq. (6) slip, and "no seeds, no code link" are all true.
- Fate labels (5 transformed, 6 erased, Stage B fixes) are all rated "weak" and described as extrapolation. That matches the paper, which never touches activations or behaviour. No wrong labels.

## 3. Errors (what the reading says / what the paper says)
1. **Table 1 text-vs-figure note is one-sided (minor).**
   - Reading: "at r=4 SubTrack is marginally below GaLore (85.11 vs 85.18), so 'comparable' holds, 'superior' does not."
   - Paper (p6, Tables 1 and 2): r=8 row SubTrack(Last)-Grad 84.91 vs GaLore 84.67; SuperGLUE SubTrack 81.95 vs GaLore 81.54. So "superior or comparable" (p6: "achieves superior or comparable performance compared to GaLore") is literally true on 2 of 3 comparisons. The caveats are that the r=8 row is a different, unexplained variant, there are no seeds, and all gaps are under 0.5 point.
   - Suggested wording: "marginally below at r=4, marginally above at r=8 (a different variant) and on SuperGLUE; all within 0.5 point, single runs."
2. **Principal-angle formulas credited to the paper (minor).**
   - Reading (`reuse_verdict`): "borrow the Grassmannian geodesic, tangent and principal-angle formulas (about 20 lines)."
   - Paper: no hit for "principal" anywhere. It gives only the tangent vector (Eq. 5), the geodesic step (Eq. 6) and the Grassmann exponential (Thm 4.6, p4-5, App. B). Principal angles are the reader's own addition, which is a fine idea, but it should be labelled as such. The "about 20 lines" figure is also the reader's estimate.
3. **Table 15 cause stated as fact (nit).**
   - Reading: Table 15 CoLA scores (0.53-0.57) are lower than Table 1 (58-62) "because this run is 10 epochs."
   - Paper: the Fig 4 caption (p8) says 10 epochs; nothing says that is why the scores differ. It is a sensible inference (interval 500 is the Table 1 setting) and should be marked "my inference".
   - Also, the takeaway "performance is similar" hides that SubTrack at interval 50 is 0.5673 vs GaLore 0.5463, the largest gap in the table, in a single run with no seeds.

## 4. Important omissions
1. **The authors' "faster convergence" claim is not flagged as unsupported.**
   - Paper p1-2: "thereby accelerating convergence"; "reducing abrupt shifts for faster convergence."
   - Evidence in the paper: Fig 5 (p19) shows only "comparable" and on BoolQ SubTrack's loss is above GaLore's for most of the run; Table 3 (p7) is slightly worse than GaLore; the only "fewer jumps" evidence is one 2D Ackley toy path (Fig 3).
   - The reading has every one of these facts but never connects them to the contribution bullet. This matters because our project would borrow "tracking removes jumps", and that has no support beyond one toy example.
2. **Table 14's "3B" model is internally inconsistent (unresolved).**
   - Paper (p19, Table 14): the 3B column is hidden 4096, intermediate 11008, 32 heads, 32 layers.
   - By arithmetic, those dimensions give about 6.5-6.7B parameters (the standard Llama-7B shape), not 3B. Full-rank bf16 Adam on that shape needs about 54 GB for weights, gradients and states, but Table 7 (p16) reports a 34.92 GB peak on a 40 GB A100.
   - Either the label or the config in the table is wrong, and I cannot resolve it offline. The reading repeats "tested only up to 3B" with no hedge. It only matters for the Stage B feasibility remark (9B-27B), so suggest "labelled 3B; Table 14 dims look like a 7B config; unresolved."
3. **Fairness of the GaLore comparison (optional).** SubTrack has an extra per-task tuned knob (step size 0.001-50, Tables 12-13, p18). GaLore has no equivalent. The reading calls this a "tuning burden" but not a bias in the "comparable to GaLore" claim.

## 5. Verdict
**OK.** Main results, numbers, figures, quotes and fate labels are accurate and appropriately hedged. Optional edits: error 1, error 2, omission 1, omission 2. If the slug's intent is SubTrack++ (later version), the local v1 file does not cover it. The reading already says so, and re-fetching the latest version is a separate step that needs the web (not allowed in this check).
