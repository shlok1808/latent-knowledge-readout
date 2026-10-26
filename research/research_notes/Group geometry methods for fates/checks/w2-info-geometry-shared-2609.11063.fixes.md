# Fixes: w2-info-geometry-shared-2609.11063

All checker points verified against paper.txt; all accepted. quote_check.py: 31 quotes, 0 failed (was 29; 2 pitfall quotes added).

## Changes (old -> new, page)
- E1 methods[2].what_it_measures: "cost = teacher-forced sequence KL on reference continuations" -> cost is on continuations of the same 4 reference prompts that define the metric; preservation in-sample, only target effect held out (min 0.0128 nats); no held-out disturbance number (p12, p78)
- E1 methods[2].causal_followup: added "measure disturbance on held-out prompts ourselves (the paper does not)" (p78)
- E1 controls "Separate donor, reference and target prompts so a fix is tested on prompts it never saw" -> target effect held out only; disturbance measured in-sample on reference prompts (p12, p78)
- E1 bears_on_fates Stage B: "2.9-6.3x less reference-sequence KL" -> same number, plus "measured in-sample on the reference prompts ... no held-out disturbance number; effects 0.013-0.21 nats, no flip or accuracy outcome; honest-over-flattering, not factual pushback" (p12, p78, p81)
- E1 actionable_fix: added "the paper shows lower disturbance only on that reference set itself, so we must measure disturbance on separate held-out prompts" (p78)
- E1 new pitfall with quote "Reference prompts also define the metric, so the readout evaluates continuation disturbance on that reference set." (p78)
- E2 bears_on_group_angle (3): "evidence that the metric matters more than linearity of a single step" -> both arms relinearised, Fisher advantage persists; no relinearised-vs-single-step comparison; no manifold steering tested, only cited (ref 14) (p11, p12, p23, p77, p2)
- E3 relevance_to_project: "warns that Euclidean activation geometry, including Euclidean principal angles, is coordinate-dependent" -> paper shows it for distances, inner products, cosines; principal angles marked as our inference (p1, p49, p83)
- E3 bears_on_group_angle (1): "hence Euclidean principal angles" -> marked as our inference; paper uses principal angles only for read-out subspaces; added that Prop. 1 is abstract (norm layers not discussed) and that G needs a stated layer (p49, p17)
- E4 methods[4]: "z < 1" -> "z <= 1"; added necessary-condition-only, zero violations over 45,315 flips, LAMBADA not independent validation (p72)
- E4 methods[4].which_fate: removed "flips with large m ... point to override or transformation"; now: z is a function of two output distributions, cannot separate threshold from override (p72)
- E4 methods[5]: replaced "near-zero alignment = non-mediation; strong negative alignment = override" with caveats: paper's update is training-loss step (80 Pythia-70M examples); flips imply negative alignment by definition; different contexts need a convention; weak (p73)
- E4 bears_on_fates fate 1: strength "moderate" -> "weak-moderate"; text rewritten as necessary condition (p72)
- E4 bears_on_fates fate 3: strength "moderate (method)" -> "moderate (injection test), weak (alignment)" (p73)
- E5 key_results[8].claim: "Output flips ... are forced by margin vs distribution distance" -> flip requires distance >= half the margin (necessary, not forcing); about training-time acquisition, not pushback (p72)
- E6 key_results[11].claim: "onset of movement along the answer margin (gradual drift vs commitment)" -> "onset of unsigned movement along the target-alternative margin direction", synthetic languages, training time; gloss marked ours (p8)
- E7 key_results[10].claim: "does not determine use" -> "does not determine how fast an objective is learned"; added 0.6811 and 0.541; "use"/mediation marked as our inference (p73)
- E8 pitfalls[1]: "our 9B-27B runs in bf16 would be unreliable" -> absolute cosines 0.056-0.542, Pythia-410M/1.4B two contexts each, kappa 1e5-1e7; 9B-27B marked as extrapolation (p79)
- E8/E9 methods[0].data_and_compute: "Full GPU solves fit only up to 2.8B on a 32 GB RTX 5000 Ada; 9B-27B needs 80 GB GPUs or ... Woodbury" -> paper does not say which card ran 2.8B; 80 GB marked our inference; Woodbury still needs |S| VJPs; 85 iterations = exact-arithmetic bound; bf16 numbers added (p25, p78, p79, p88)
- E10 Supp. Table 1 page: 54 -> 55 (p55)
- E10 architecture_location: "default at about 45% depth, final prompt position" -> middle layer round(0.45 x depth), final position stated for reusable control (p16, p23, p77)
- E10 architecture_location: "most answer-relevant probability motion" -> "largest answer-motion value (0.59) among the bands", defined per p18; output-geometry modes, not an architecture location (p39, p18)
- E11 bears_on_fates fate 4: "showing competing objectives interact through the metric" -> "moved sycophancy in the opposite direction (the paper gives no mechanism; ...)" (p36, p82)
- Omission 1 key_results[4].claim: added effect sizes (min 0.0128 nats, up to about 0.21), measured not matched (p78, p12)
- Omission 1 methods[2].which_fate: added small effects, no flip/accuracy outcome (p78, p81)
- Omission 1 new pitfall with quote "the association reverses sign beyond that regime" (erosion predictor) (p77)
- Omission 2 key_results[5].claim: added per-prompt frontiers on 12 prompts, honest-over-flattering not factual pushback, no flip/accuracy outcome (p11, p81); same caveat in key_results[4]
- Omission 3 reuse_verdict: added FishBack 2605.17231 (8B) and Park et al. 2602.15293 as precedents to read (p2)
- Omission 4 methods[0].which_fate: "near-zero Fisher sensitivity at the action position is output-inert by construction" -> same-position, first-order only; earlier-token directions could act via attention (p17)
- Omission 5: 0.6811 / 0.541 added (see E7)
- Omission 6 bears_on_group_angle (4): added "curvature of subspace paths" to not-covered list
- Extra (found while checking, not in the checker): methods[1] and controls now note that Supp. Note 14 says localised objectives use on-prompt off-target KL while the Supp. Fig. 2 caption says held-out neutral prompts (p81, p82)
- reading .md: mirrored every change above (summary line, Setup layers, Fig 5 row, key results 1/5/7/8/10, methods, controls, pitfalls, fates table, group angle, where/fix, verdict, relevance)

## Checker points rejected
- None. Partial note: E3's points on norm layers and the layer choice for G are the checker's own observations, not paper claims; added as our caveats, not attributed to the paper. Omission 5's Supp. Table 1 range and code-on-request were already in the reading; no change.
