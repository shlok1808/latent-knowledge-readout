# Check: w2-layerwise-grassmann-dynamics-2605.12714

Paper: Jiang, Yang, Tam (HKUST), "Layer-wise Representation Dynamics: An Empirical Investigation Across Embedders and Base LLMs" (arXiv 2605.12714). Reading checked: `reading/w2-layerwise-grassmann-dynamics-2605.12714.json` (and the .md, which agrees with it).

## Verdict: NEEDS FIXES (minor)
Quotes, page refs and almost every table number are right. One key_results claim overstates what the paper tested ("best label-free predictor"), the family-level "decoders bend late" evidence needs its per-model spread, and one claim about base LLMs should cite Table D.6 instead of a weaker sentence. No REDO.

## Quote result
- `quote_check.py`: 20 quotes, 0 failed.
- Stricter check (full-string match plus page number from `pdftotext` per page): 20/20 match in full, 20/20 on the claimed page.

## Figures checked
The manifest lists 6 items (Tables 1-4, Figs 1-2), all in the reading with the right pages. I opened p05, p06, p07, p08, p09. Table 1 (0.56/0.89/0.78; +0.015/+0.144/+0.121), Table 2 (all 27 values), Table 3 (all 15 values), Fig 1 (rho +0.85/+0.72/+0.68; two d_{0,L} clusters about 13-20 and 26-30), Table 4 (all 40 values) and Fig 2 match.
The reading marked 9 appendix items "not checked visually" because they were not in the manifest. I rendered and checked them from the PDF: Table B.1 layer counts (6-24 encoders, 28-42 decoders, 12-48 base LLMs); Fig C.1 and C.2 (p.19, 20); Table C.3 (p.20-21; every number quoted in the reading is correct); Table D.4 (p.21-22; d_{0,L} min +0.12 SprintDuplicate, max +0.88 NFCorpus, all positive; GFMI +0.04 and -0.07); Fig D.4 (p.24); Table D.5 (p.24; matches); Table D.6 / Fig D.5 (p.25-26; range -96.6% to +64.9% confirmed). Not opened: Fig D.3 (p.23, a heatmap of the Table D.4 values) and Table B.2 (task list). I also recomputed the MMLU-6 Spearman from Table D.5: +0.64 / +0.46 / +0.64, with p = 0.17 / 0.35 / 0.17.

## Errors (reading says / paper says)
1. **"Best label-free predictor" overclaims.** (medium)
   - Reading says: key_results[2] "End-to-end subspace displacement is the best label-free predictor of embedder quality."
   - Paper says (p.1 and p.7): d_{0,L} is "the strongest" of the three LRD scores. p.9 limitations: "We omit head-to-head label-free or transferability baselines" (our reading's own setup entry says the same). Correct claim: strongest of the three LRD scores, no external baseline tested.
2. **Family means hide the spread behind the "decoders bend late" finding.** (medium-minor)
   - Reading says: Table 1 takeaway "encoder embedders peak mid-depth and stay flat"; bears_on_group_angle uses 0.89 (decoder embedders) and 0.78 (base LLMs).
   - Paper says: Table C.3 (p.20) and Fig C.1 (p.19): encoder peaks run 0.08 to 0.99 (E5 about 0.08-0.11, Snowflake about 0.99; p.19 text "more dispersed"), so 0.56 is a mean of a wide spread, not a mid-depth peak. The decoder-embedder 0.89 is 12 models of which 8 are Mistral-7B fine-tunes with near-identical peaks (0.867-0.868 for 8 of them), so not 12 independent points. Base LLMs: Llama-3.2-1B 0.924, Gemma-2-2B 0.901, GPT-2 XL 0.893, Mistral-7B 0.856, Falcon-7B 0.744, GPT-2 0.380. The "late bend" holds for 5 of 6 base LLMs; GPT-2 is the counterexample (the reading does name it).
   - The architecture_location line "last ~10-25% of depth" is fine for the five late-peaking base LLMs; say n = 5 of 6 and that the decoder-embedder number is Mistral-driven.
3. **"On the base LLM none of the rules clearly beats random" is not what the paper text says.** (minor)
   - Reading says: key_results[11] claim, supported by the quote "On LLMs the three rules are harder to separate from Random." (p.9).
   - Paper says only that (plural "LLMs", no per-model numbers). The stronger statement is true in Table D.6 (p.26, Mistral-7B-v0.1): my recomputed mean over its 3 tasks is Random -12.7 / +8.9 / +8.3 / -9.1 vs GFMI -11.8 / -20.6 / +2.3 / -10.6 at 5/10/15/20%; Frenet is best at 5% (+9.8), NRS best at 20% (+0.2). But the cells run from -83% to +65%, so n = 3 is noise. Cite Table D.6, not the sentence.
4. **Table 2 flags one exception, but there are more.** (minor, low stakes) Paper claim (p.6): "retrieval has higher mean speed and GFMI scale." Table 2: encoder GFMI is highest for Other (109.8 > RET 98.1), and base-LLM speed is highest for Other (8.873 > RET 8.024). The reading only flags encoder curvature (Other 0.142 > CLS 0.137).
5. **key_results[6] "Task type shifts trajectory shape"** drops the caveat the reading's own Table 2 entry gives: the differences are in the third decimal (0.137 vs 0.132) and smaller than within-family spread (decoder embedder curvature 0.103-0.150 across models, Table C.3).
6. **Table C.3 entry lists the wrong sixth row.** (minor) It lists Qwen3-Embedding-8B (a decoder embedder) and omits GPT-2 XL (0.893 / 0.076 / 0.255 / 6.273 / 381.4). All listed numbers are correct. The family means in Table 1 reproduce from the per-model rows (base LLM 0.783 and +0.121).
7. **"Agrees with 2605.09011" is only qualitative.** Different objects (example-cloud PCA subspace on MTEB inputs vs lens-weight subspace on web text) and different statistics (curvature peak vs RSS decline). Say "directionally consistent".

## Omissions
- **Pruning budgets collapse for small models.** Table D.6: BGE-base (12 layers, 6 prunable after protecting 6) gives identical numbers at 5% and 10% and identical numbers at 15% and 20% (e.g. Random on AmazonPolarity -2.0, -2.0, -5.4, -5.4). The "light budget" statements (Frenet best at 5%, Random best at 10%) partly reflect removing one layer in small models. Add to pitfalls.
- **Pruning panel identity is missing from the setup.** 5 models (BGE-base-en-v1.5, E5-large-v2, Nemotron-Embed-1B, LLM2Vec-Mistral-7B, Mistral-7B-v0.1) x 3 tasks (AmazonPolarity, FiQA2018, STSBenchmark); Nemotron-Embed-1B and LLM2Vec-Mistral-7B are not in the 31-model Table B.1; one base LLM only (p.8 footnote 1, p.9 limitations).
- **Pooling for base LLMs is unspecified** ("official or model-recommended pooling when available", p.5); the reading notes pooling matters but not that no rule is given for base LLMs.
- The MMLU-6 rho values are not significant (p about 0.17 / 0.35 / 0.17); "directional" is right, add the p-values.
- Menger curvature 4A/(abc) divides by the product of the three distances, so it is amplified wherever speed is tiny; not flagged, relevant if we reuse it on small between-condition distances.

## High-stakes items, status
- Key results: all 14 quotes exact; claims fine except items 1 and 3 above.
- Controls worth copying: accurate (random with 3 seeds, last-k, mean and median, fixed r, per-task significance counts, protected boundary layers).
- Fate labels: all "weak" or "none" and hedged; fair, since the paper is population-level on MTEB with no failures.
- architecture_location: correct against Table 1 and Table C.3 (peaks 0.74-0.92 for five base LLMs; GPT-2 0.38), with the spread caveat in item 2.
- actionable_fix: correctly "none for safety"; the three proposals are labelled ours and untested.
- Skipped sections: none; sections_read covers all appendices.
- Minor fact checks that pass: ref [30] is a 6G networks paper (arXiv 2311.09047); no code release stated; compute (4x V100, about 300 GPU-hours); k_NRS = 20, k_GFMI = 30, 20 percentile points over [5, 95], 95% variance rule; authors and venue.
