# Check: u4-role-learning-1910.09113 (ROLE, Soulos, McCoy, Linzen, Smolensky; arXiv 1910.09113v3)

Verdict: **NEEDS FIXES** (minor; three one-line rewordings plus two additions; no redo)

## 1. Quote check
- `python3 quote_check.py u4-role-learning-1910.09113`: 17 quotes, 0 failed.
- That script only tests the first 120 normalised characters and does not test page numbers, so I also checked the full text of every quote and its page, using per-page `pdftotext` of `paper.pdf`. All 17 full quotes are present and all 17 page numbers are correct.

## 2. Coverage checks
- Figure/table entries: manifest lists 8 (Fig 1-5, Tables 1-3); the reading has all 8.
- `sections_read` covers every section, the acknowledgments, references and Appendix A.1-A.8.
- I opened p01, p04, p05, p06, p08 and compared them with the entries. Numbers in Table 1, Table 2 (all four models, all columns), Table 3 and the Fig 5 bars (about 99, 94, 89, 86, 84, 83, 85, 86, 85) match.
- I re-derived the Fig 5 left-panel roles by hand from the A.5 rules (run 11, left 36, twice 8, after 43, jump 10, opposite 17, right 4, thrice 46). They match, as the reading says.
- Figure 1's role table (run = 3 in "jump and run twice") really does conflict with the legend and A.5 (first word after "and" = 11). The reading's flag is correct.
- Also correct: the dangling "A.8.1" and "Sec. 6.1" references (A.6 point 3 and 4, p16), the duplicated "Elements of the command before and" heading in A.5, and the 47 conditionals, 16 of 50 roles, 20,910 sequences and 151,688 words figures.

## 3. Errors (what the reading says / what the paper says)
1. **`key_results[3]` states an untested hypothesis as a result (moderate; right quote, stronger conclusion).**
   - Reading: "role 30 marks 'and', another marks 'after' sequences, which sets output order."
   - Paper p6 (the quoted sentence): role 30 is used "for, and only for, the filler and, while role 17 is used in and only in sequences containing after." That is an association between roles and fillers. "Sets output order" comes from the authors' interpretation ("the decoder can use these roles to tell which basic structure is in play", p6; "can tell the decoder whether the input order ... corresponds to their output order", A.6, p16). The paper never tests it: footnote 2 (p7) excludes `and` and `after` from the surgery because they reshuffle roles. The reading lists that exclusion under pitfalls but not here.
   - Suggested wording: "roles 30 and 17 are exact markers of and/after; the authors hypothesise the decoder uses them to fix output order, but this is not causally tested."
2. **`bears_on_fates` strength "moderate" for 2 bypass and 5 transformed is generous (moderate; label).**
   - The paper has no case of information that is present but unused or re-bound, no pressure condition, one GRU bottleneck (hidden size 100), and a synthetic grammar. Its positive causal result is that edits to bound constituents change the output as predicted for 83-86% after 4 or more edits (Fig 5, p8), for substitutions that do not reshuffle roles.
   - The reading itself says relevance is "moderate to low" and "untested beyond RNN bottlenecks and synthetic grammars." Suggest "weak to moderate: validation-logic transfer only" so a downstream reader does not rank this above papers that test the fate directly.
3. **"Held-out" framing of the 98.7% match is stronger than the paper supports (minor).**
   - Reading (`key_results[2]`, `controls_worth_copying`): the algorithm "built only on training data, predicts ROLE's roles on held-out data (98.7%)."
   - Paper p14 (A.4/A.5): "on both datasets, it exactly matches the roles chosen by the Role Learner for 98.7% of sequences (20,642 out of 20,910)." 20,910 is the whole SCAN set (A.6, p16), so the quoted count pools training and test. A test-only number is not given; p6 says only that it is "equally successful" on test sequences.
   - The paper also never says whether ROLE itself was fit on test-set encodings (Sec 5.1, p6: encodings are extracted "for each example in the training and test sets. These are the encodings we attempt to approximate"), nor whether the Table 1 substitution accuracies are train, test or pooled. The held-out check is of the hand algorithm against ROLE's own roles, not of ROLE.
4. **Wrong date range (nit).**
   - Reading (`architecture_location`): "four 2015-2017 sentence encoders."
   - Paper's own citations (Sec 6, p7): SST model = Socher et al. 2013, Skip-thought = 2015, SPINN = 2016, InferSent = 2017. So 2013-2017.
5. **Table 2 "text matches figure: yes" should be "partly" (nit).**
   - Paper p7: "for each of the sentence embedding models tested except for SST, performance is within the same order of magnitude as structure-free BOW."
   - Table 2 (p8): SST is 5.58e-3 vs BOW 9.38e-3, which is also the same order of magnitude. ROLE's margin over BOW is at most 1.7x anywhere (InferSent 1.07x; Skip-thought BOW wins; SST 1.68x; SPINN 1.27x). The authors' phrasing is loose.
   - Also, discrete ROLE is worse than BOW on InferSent (5.76e-4 vs 4.34e-4). So the evidence that structure is discrete exists only for SCAN.
   - The reading's direction of takeaway ("no evidence of compositional structure in these encoders") is right and matches the quoted "Although not definitive".

## 4. Important omissions
1. **Surgery depends on the hand-written role algorithm (matters for transfer).**
   - Paper p7 (Sec 5.3): "The roles in these constituents are determined by the algorithm of Appendix A.5. If changing a word leads other roles in the sequence to change (according to the algorithm), we update the encoding with those new roles as well."
   - So the causal result needs (a) ROLE's learned W, F, R and (b) a 47-rule, 98.7%-accurate role-assignment algorithm. The reading says surgery uses "the learned W, filler and role vectors" and "needs matched counterfactual fillers", but not that the roles come from the extracted rule set. For factual-pushback prompts there is no such rule set, which is the real obstacle (beyond the single-vector issue the reading already notes).
2. **No negative control for ROLE itself (reader-side critique, not in the paper).**
   - ROLE's role assigner is a bidirectional 2-layer LSTM over the whole input (A.4, p14), and A.5 shows roles depending on global context (e.g. `after` gets 17 or 43 depending on other roles). It is therefore a flexible sequence model, not a fixed role scheme.
   - The paper never fits ROLE to a target known to be non-compositional (e.g. a random or scrambled encoder) and checks functional replacement. The sentence-model arm is MSE-only.
   - The reading copies the "structure-free BOW baseline" as the control, but BOW is a weak negative control for a learned, context-dependent role scheme. The strongest evidence for non-trivial structure in the paper is the combination of 16 of 50 roles used, the 47-rule algorithm, and surgery. This should be added to `pitfalls` and `controls_worth_copying` (a shuffled-target or random-network fit with the same functional test).
3. **Authors' "4 times simpler" claim (low stakes).** A.6 point 4 (p16) rests on a crude estimate (200 lines vs 47). Not flagged, but it does not affect the project.

## 5. Verdict
**NEEDS FIXES (minor).** The reading is careful, hedged, and its numbers, figures, quotes, pages, methods and "adapt, do not run as is" verdict are accurate. Fix errors 1 to 3 (error 4 and 5 are optional) and add omission 1 and 2 to `pitfalls`/`controls_worth_copying`. No redo needed.
