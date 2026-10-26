# Check: c9-lagged-coupling-2609.01048 (Xun, Lagged Coupling, arXiv 2609.01048)

Verdict: **NEEDS FIXES** (minor to moderate; no REDO needed). Numbers, tables and figure readings are accurate. Problems are 3 repeated overclaims, 1 over-strong fate label, a missing caveat on the headroom metric the reading tells us to copy, and some small errors.

## Quote check
`quote_check.py`: 25 quotes, 0 failed. It only matches the first 120 characters and never checks pages, so I checked pages by hand against the page footers in paper.txt and `pdftotext`. 24 of 25 pages are right; 1 is wrong (E5).

## Figures and tables
All 9 manifest items are in the reading (Tables 1-4, Figures 1-5), plus B1, C1, C2. I opened p03, p04, p06, p07, p08, p09, p10, p11. The reading says p15 (B1, C1, C2) was read from text only, so I rendered p15 myself. Table B1 matches Figure 4 cell by cell, the five bold cells are right, and C1 and C2 match the text. No missing figures. Figure 1 intervals, Figure 3 series, Figure 4 cells and Figure 5 growth factors and noise floors in the reading all match the paper. The reading's own catches are correct: the Figure 5 caption miscounts 2.8B, the Figure 4 colorbar says "family median", the Figure 3 "gap widens with scale" claim rests on two sizes, and the 2.8B point sits on the edge of its CI. The reading's 12B arithmetic (about 0.0023, about 14x below the floor) is right: 0.00108 x 2.16 = 0.00233, and 0.0328 / 0.00233 = 14.1.
Skipped sections: none. The reading lists every section, appendices included.

## Errors (reading says / paper says / quote + page)

**E1 (moderate). key_results[2]: "Behavioral readability develops ... later at larger scale." Table C2 does not show this.**
- Paper, Table C2 (p15): 6.9B is ahead of 2.8B at 16k-64k (0.775/0.875/0.877 vs 0.687/0.791/0.857). At 143k the order is 2.8B 0.968 > 12B 0.909 > 6.9B 0.896 > 1B 0.711.
- The gain from 1k to 143k is 2.8B +0.434, 12B +0.385, 6.9B +0.327. That is not monotone in size, and the C2 caption "gradient steepens with scale" is false for 2.8B.
- The quoted sentence ends "crossing 0.9 only at the final checkpoint" (p7), but 6.9B ends at 0.896, below 0.9.
- Only 12B lagging 6.9B holds up. The reading flagged the "gap widens" caption but kept the "later at larger scales" claim as a key result.
- Fix: write "paper claims later onset at larger scale; Table C2 supports this only for 12B vs 6.9B, and 2.8B ends highest".

**E2 (moderate). key_results[3] "The ordering is always read-before-write" and key_results[4] "Scale does not close the lag."**
- Paper, p8: "(11 of 11 pattern-1 units; no instance of the inverted, write-before-read pattern was observed)". p5 notation: "Pattern-1 unit: a unit whose milestones activate in the canonical order". The definition is circular.
- Only 11 of 192 units are classified, and the paper never says how the other units fall.
- Probe AUROC is at ceiling at the first checkpoint (left-censored), so "internal first" is built in. Causal is null in 43/48 cells, so "causal last" is trivially true.
- "Always" is too strong. Write "in the 11 units classified pattern-1".
- For "scale does not close the lag", the paper's CI is [-0.722, +0.438] (p8). A CI that crosses zero means no detectable change, not scale-invariance.
- The paper is inconsistent here. It calls the n=6 onset regressions underpowered (p12) but treats this crossing-zero CI as evidence for "scale-invariant" latency (p8). The regression n is not stated.
- The reading's Figure 3 note ("only shows that the lag does not shrink") is closer to right than the key_results line. Fix the key_results line to match.

**E3 (moderate). bears_on_fates[fate 2 bypass].strength = "strong".**
- The reading's own reuse_verdict says "do not lean on its claims as evidence about our setting". The reading also says the 6.9B and 12B nulls are below the noise floor and the probes may be trivially saturated.
- "Strong" is only true as a method template (Track 1 high + matched-null Track 3 null), not as evidence.
- Fix: "strong as method template, weak as evidence (base Pythia, no pressure, nulls unresolved at 6.9B and 12B)", or downgrade to moderate.

**E4 (minor). setup.positions_layers: "Answer position."**
- Paper (p4, §3.2) puts the answer position only on the logit z_t(x). The position of the probe a_l(x) and of the edit is never stated.
- Site l is never defined as layer or sublayer.
- Config summary (p15) says "best site per unit, frozen after dev". The site is picked per unit on dev, a mild selection effect. The reading says "pre-registered best site" without that nuance.

**E5 (minor). Wrong page.** The code-release quote ("Code, configurations, manifests, and the audit ledger will be released with the camera-ready.") is on **p13**, not p12. `pdftotext -f 13` confirms it. Affected: `pitfalls[6].page = 12` and `methods[0].code_available "(p12)"`.

**E6 (minor). bears_on_fates[fate 4]: "training-time analogue".**
- Paper (§3.1 p4, §4.6 p10): the bias arm aggregates 24 size x family cells, 6 sizes x 4 families. It is a scale analogue, not a training-time one.
- The lower CI bound is +0.00062 (p10), "excluding zero by a hair". The reading calls this "moderate" without noting how fragile it is.

**E7 (minor). Saturation stated as fact.**
- relevance_to_project says "trivially saturated probes". The paper reports no control-task, random-init or embedding-layer probe baseline, so this is only a suspicion. The pitfalls entry words it correctly ("strongly suggests"); the relevance field should match.
- "AUROC >= 0.990 at every size" (Figure 2 labels, Table 4, the reading's Table 4 note) holds for dev only. Test minimum is 0.982 (p7 quote, p15). The reading kept the test minimum in the key_results quote but not in the Figure 2 and Table 4 notes.

## Omissions
**O1 (important, because the reading says to copy this).** Headroom C is an uncentered RMS of the residual component along w-hat (p5). It can grow with residual norm or mean offset, and there is no C baseline along random directions.
- The edit is a fixed unit-norm push (lambda x 1, §3.2 p5; lambda grid unreported). If C grows 57x, c = e(1)/C shrinks by construction. Pythia is pre-LN, so later LayerNorms rescale the push as well. This is my analysis, not the paper's.
- Add to the reading's "methods" and "controls_worth_copying": compute C for matched random directions, use a centered variance or class-separation measure, and scale edit size to a fraction of the residual norm. Otherwise "occupied but unused" cannot be called bypass.

**O2.** The reading computed the below-resolution point only at 12B. The same logic applies at 2.8B and 6.9B: implied e(1) is about 1e-3 at 143k (|c| 0.07 to 0.11% times C of roughly 1 to 2 in Fig 5a, approximate), against floors of 0.0118 and 0.0286. So the reported "pinned write-in" is below the instrument floor at all three large sizes.

**O3.** No multiple-comparison handling:
- 5 of 48 cells have |z| >= 2, versus about 2.2 expected if z were N(0,1). The Figure 4 z is a median of four family z's (colorbar, p9), so its null is not N(0,1) anyway.
- "Backfire is real" (p12) is not quantified.
- Near-misses 1B 1k -1.81 and 6.9B 1k -1.88 (Table B1) are also negative, which supports the sign pattern but was not mentioned.

**O4.** 2.8B 1k and 2k, two of the four backfire cells, fall in the step range 0-6,300 that Appendix A (p14) says is corrupted in the public gen-5 mirror. The paper says it used the verified pinned revision (p14-15). Add a one-line caveat that these cells depend on the lineage fix.

**O5.** The paper's re-wiring-window hypothesis (p11) says backfire cells "cluster near behavioral transitions". Table C2 does not show that. 2.8B behavioral AUROC is flat at 1k-4k (0.534/0.517/0.519), and the 12B 8k "transition" is a dip (0.632 to 0.561). The reading is right not to adopt it. If it is added, label it as unsupported.

**O6.** Fork A and C are defined mechanically in §3.3 (p5): A = CI entirely below zero, C = entirely above. The reading says they are "never defined in substance", which is fine. It could add that the "3 A" tally includes the degenerate 2.8B interval, whose upper bound is -5.8e-12 with the point estimate on the edge (Figure 1, p7). That makes the tally 2 clear A, 1 degenerate A, 3 INDETERMINATE.

## Best ideas the reading could carry forward (per the user's "get the best ideas" request)
1. **Per-track clocks.** Do not fit one onset. Measure decodable / margin-readable / causal onsets separately. For our group angle, the same idea applies along the token path and across layers, to find the fork point. The reading only ties the development index D to direction 3 weakly.
2. **Split-half replication of any single significant cell.** The paper's 12B 8k pulse is +2.49 on test and +2.04 on dev (p9, p15). Use the same dev/test replication for any single-cell fate claim.
3. **Report every cell.** The full grid with boxed |z| >= 2 (Figure 4) is a good reporting norm for a fate-by-condition table.
4. **Pipeline-assembled claim text.** The bias-arm claim text is assembled by the pipeline (p4 footnote, p10). This fits the lab mod's sealed bets.
5. **Monitoring-vs-steering table (Table 4).** Reuse it for Stage B. After a fix, recheck Track 1 (decodability) and Track 3 (causal write-in) separately to tell "removed" from "hidden". The reading has this already.

## What the reading gets right (no change needed)
The reading catches the missing controls, the below-floor large-scale nulls, the unreported layer, items, lambda grid and target variable, the family-median caveat, the single-direction-only scope, the Figure 1 oddities, the lack of code, and the "less relevant than the title suggests" call. The actionable_fix and architecture_location fields are appropriately hedged. No fate is claimed on evidence the paper does not give, except the "strong" label in E3.

## Suggested edits (the reading file was not changed)
- Fix E1 to E3 and E5 to E7 in place.
- Add O1 and O2 to `methods[2]` (headroom), `controls_worth_copying` and `pitfalls`.
- Add O3 and O4 to `pitfalls`.
