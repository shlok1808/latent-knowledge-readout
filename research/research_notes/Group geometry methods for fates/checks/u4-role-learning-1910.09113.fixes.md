# Fixes: u4-role-learning-1910.09113

Applied to reading/u4-role-learning-1910.09113.json and .md. quote_check.py: 19 quotes, 0 failed. Pages of the new quotes were checked with per-page pdftotext.

## Changes
1. key_results[2] (error 3): "built only on training data, predicts ROLE's roles on held-out data" + A.4 quote (p14) -> "matches ROLE's roles on 98.7% of test sequences (p. 6); the 20,642/20,910 count in A.4 pools train and test; checks the hand algorithm against ROLE's roles, not ROLE itself; the paper does not say whether Table 1 is train, test or pooled" + the Sec 5.2 quote "Though the algorithm was created ... 98.7% of sequences." (p6)
2. setup.evaluation_and_controls (error 3): added "(98.7% match on test sequences, p. 6; reported count pools train and test; validates the algorithm against ROLE's roles, not ROLE itself)" (p6, p14)
3. methods[5].what_it_measures (error 3): "on unseen data (98.7%)" -> "98.7% of test sequences; the 20,642 of 20,910 count pools train and test" (p6, p14)
4. controls_worth_copying[2] (error 3): "(98.7% match)" -> notes that it checks the hand rules against ROLE's roles and ROLE is not reported on held-out encodings separately (p5-6, p14)
5. key_results[3] (error 1): "one role marks 'and', another marks 'after' sequences, which sets output order" -> "roles 30 and 17 are exact markers of 'and'/'after'; the authors hypothesise the decoder uses them to fix output structure and order, not causally tested (footnote 2 excludes and/after from surgery)" (p6, p7)
6. bears_on_fates "4 override" (follows error 1): "set the global output plan and order" -> "mark the global structure; the authors hypothesise (untested) the decoder uses them...; the paper excludes these tokens from surgery" (p6, p7)
7. bears_on_fates "2 bypass" and "5 transformed" strength (error 2): "moderate" -> "weak to moderate: validation-logic transfer only"
8. architecture_location (error 4): "2015-2017 sentence encoders" -> "2013-2017" (SST is Socher et al. 2013, p7)
9. Table 2 text_matches_figure (error 5): "yes" -> "partly": SST is also within the same order of magnitude as BOW (5.58e-3 vs 9.38e-3); ROLE's margin over BOW at most about 1.7x; discrete ROLE worse than BOW on InferSent (5.76e-4 vs 4.34e-4), so discrete structure is shown only for SCAN (p7, p8)
10. methods[0].causal_followup_it_allows (omission 1): added that surgery roles are computed by the hand-extracted A.5 algorithm, not by ROLE directly (p7)
11. methods[2].data_and_compute_needed (omission 1): added the need for an explicit role-assignment rule set (47-rule A.5 algorithm), the main transfer obstacle for factual-pushback prompts (p7)
12. pitfalls (omission 1): new entry "Surgery depends on the hand-extracted role algorithm", quote "The roles in these constituents are determined by the algorithm of Appendix A.5." (p7)
13. pitfalls (omission 2): new reader-side entry "no negative control for ROLE itself" (flexible bidirectional-LSTM, context-dependent roles; never fit to a known non-compositional target), quote "The role assignment module used a bidirectional 2-layer LSTM" (p13)
14. controls_worth_copying (omission 2): new item: fit the same learned role scheme to a random or shuffled-target network and run the same functional test; BOW is a weak control for a flexible role assigner
15. pitfalls last entry (omission 3, optional): added "the A.6 '4 times simpler' claim rests on a self-described crude estimate (200 vs 47 lines)" (p16)

## Checker points rejected or corrected
- Error 3, partly rejected: the checker says p6 gives only "equally successful" with no test-only number. In fact p6 (Sec 5.2) says the algorithm matches ROLE's roles "for 98.7% of sequences" on test sequences. So a held-out figure does exist. I kept the held-out framing, cited the p6 sentence, and added the pooled-count and "checks the algorithm, not ROLE" caveats, which are correct.
- Omission 2 page: the checker places the bidirectional 2-layer LSTM statement (A.4) on p14. Per-page pdftotext puts it on p13, so I cited p13.
