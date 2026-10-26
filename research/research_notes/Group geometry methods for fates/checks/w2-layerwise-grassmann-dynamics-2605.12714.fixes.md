# Fixes: w2-layerwise-grassmann-dynamics-2605.12714

All checker points verified against paper.txt and `pdftotext -layout` of pp. 25-26 (Table D.6). quote_check.py: 20 quotes, 0 failed.

## Changes (old -> new, page)
- key_results[2] claim: "best label-free predictor of embedder quality" -> "strongest of the paper's three LRD scores; no external label-free or transferability baseline tested (stated omission)" (p.1, p.7, p.9).
- Table 1 takeaway: "encoder embedders peak mid-depth and stay flat" -> family means only; encoder peaks 0.08 (E5) to 0.99 (Snowflake); decoder-embedder 0.89 driven by 8 Mistral-7B fine-tunes at 0.867-0.868; late peak holds for 5 of 6 base LLMs (GPT-2 0.38) (p.5, p.19-21 Table C.3).
- Table 1 text_matches_figure: "yes" -> "yes (means reproduce from Table C.3), with the spread caveat" (p.5).
- key_results[4] claim: added "family means; 5 of 6 base LLMs late, GPT-2 0.38; decoder-embedder mean Mistral-driven" (p.5, p.20-21).
- key_results[11] claim: "On the base LLM none of the rules clearly beats random" (supported by the p.9 sentence) -> text says only "harder to separate"; the stronger statement rests on Table D.6 (Mistral-7B-v0.1, 3 tasks: Random -12.7/+8.9/+8.3/-9.1 vs GFMI -11.8/-20.6/+2.3/-10.6 at 5-20%; cells -83% to +65%; n = 3 is noise) (p.9, p.26; means recomputed and confirmed).
- key_results[6] claim: "Task type shifts trajectory shape." -> "only slightly": third-decimal curvature differences, smaller than within-family spread (decoder-embedder curvature 0.103-0.150), with exceptions (p.6, p.20).
- Table 2 text_matches_figure: added the two further exceptions (encoder GFMI Other 109.8 > RET 98.1; base-LLM speed Other 8.873 > RET 8.024) (p.6).
- Table 3 numbers: MMLU-6 rho now carry "not significant, p about 0.17 / 0.35 / 0.17 (recomputed from Table D.5)" (p.7, p.24; p-values from the checker's recomputation, not re-run here).
- Table C.3 entry: sixth row "Qwen3-Embedding-8B 0.916 / ..." (a decoder embedder) -> "GPT-2 XL 0.893 / 0.076 / 0.255 / 6.273 / 381.4"; added E5 and Snowflake peaks and decoder-embedder curvature range; takeaway notes the Mistral cluster; status "not checked visually" -> "checked by the checker on rendered pages" (p.20-21).
- Table D.6 entry numbers: "about -97% to about +65%; row alignment uncertain" -> "-96.6% to +64.9%", plus the Mistral-7B 3-task means and the BGE-base budget collapse (p.25-26).
- setup.baselines: added the pruning panel (5 models x 3 tasks = 15 cells; one base LLM; Nemotron-Embed-1B and LLM2Vec-Mistral-7B not in Table B.1) (p.8 footnote 1).
- setup.data: pooling "official or model-recommended" -> adds "when available; no rule given for models without one (e.g. base LLMs)" (p.5).
- New pitfalls: (a) pruning budgets collapse for small models (BGE-base identical at 5/10% and 15/20%, e.g. Random AmazonPolarity -2.0, -2.0, -5.4, -5.4) (p.26); (b) Menger curvature 4A/(abc) is amplified when speed is tiny (p.3-4, our reading); (c) family means hide spread and non-independence (Mistral fine-tunes) (p.20-21).
- bears_on_group_angle: "agrees with 2605.09011" -> "directionally consistent ... qualitative only" (different objects and statistics); added n = 5 of 6 and Mistral-driven caveat.
- architecture_location: "per-model peaks 0.74-0.92 for the base LLMs other than GPT-2" -> "for 5 of 6 base LLMs ...; decoder-embedder mean driven by 8 Mistral-7B fine-tunes"; base-LLM pruning sentence now cites Table D.6 (3 tasks, noisy).
- .md: same changes mirrored (one-paragraph summary, Setup baselines/pooling, Table 1/2/3 rows, appendix rows' check status, key-result caveat after the p.9 quote, three new pitfalls, group angle and Where wording).

## Rejected checker points
- None rejected. Left as is: the JSON "not checked visually" status on Table B.1, B.2, Fig C.1, C.2, D.4 entries; the checker verified B.1, C.1, C.2, D.4 numbers but the reading's values were already correct, so no content change was needed.
