# Checkpoint after sweeps and recall test (phase 3, written 2026-10-22, run continued without stopping as instructed)

## Plan change
At the user's request (finish in 30-40 minutes, main work first), the run was compressed: sweeps were cut short, then a deep read with 4 readers (direction 1 bypass/readout, direction 1 override/transformed, direction 2 + light direction 3, fixes), then a writer with reviewer checks, then a quick fact-check. **Dropped:** the 806-paper re-score, citation chasing as its own phase, and the full fact-check phase.

## Sweeps (phase 1)
| Sweep | Families | Table rows (incl. headers) | Stopping rule met? | Main blocker |
|---|---|---|---|---|
| A | know-do, knowledge-prediction, latent/hidden knowledge, decoded but unused | 448 (~45 Core, ~130 Supporting, ~120 Peripheral) | No (7 of last 10 queries still added papers) | WebSearch budget, OpenAlex quota, S2 429, arXiv empty pages |
| B | representation-action, readout, null space/bypass, inaccessible, transformed | 215 (~130 papers, 29 Core) | No | WebSearch per-turn cap, arXiv rate limit, OpenAlex needs a key (user's email) |
| C | override, settings, fixes, generalization/cost | 500 (~190 + 141 OpenReview title-only rows) | No (OpenReview still adding) | arXiv API; NeurIPS 2026 main track not public on OpenReview |
| D | geometry, divergence geometry, trajectory (dir 3) | 213 | No (budget ran out) | WebSearch budget; OpenReview not read |

Query counts: A ~140, B ~35+, C ~56 OpenAlex + 73 web + 22 OpenReview, D ~95+1.

## Recall test (phase 2): 22 of 32 found (corrected after the watcher's check; the grep first said 24)
The grep over-counted two items: "Decoded but Unused" (da9ylT2doy) matched only a family label in sweep A, and "Curved Inference II" (2608.24037) matched only part I. Both are misses; reader 1a has Decoded but Unused and reader 2/3 has Curved Inference II. Open ID conflict: sweep D gives "Latent Policy States" the id 2607.18532, which the known list holds as a separate paper.
The phase-3 stop was skipped on the user's instruction (unattended run).
Missed by all four sweeps:
- The Readout-Mediator Angle 2605.29126 (assigned to deep reader 1a)
- Few-Shot Demonstrations Elicit Use of In-Context World Representations 2609.24352 (reader 1a)
- Matryoshka Attribution 2609.25518 (reader 1a)
- Oracle-State Intervention (aU7tabEHsu), Lost in Distraction (IrfhtJkhmV), ContextRotBench (DiZ67nkfay) (reader 2/3)
- 2609.10739 (reader 2/3)
- Cammarata, Neural Networks Are Already Clean (list only, not researched)

Pattern of the misses: workshop and agent-benchmark papers on OpenReview (not on arXiv), and methods papers whose titles use none of our vocabulary. Same failure mode as the KAPPA miss in round 1: the vocabulary is still too narrow, and OpenReview coverage is weak.

## New and notable from the sweeps (abstract level)
- Authority-hierarchy sycophancy paper = 2607.00415.
- 2604.19117 is not in papers_master.csv (hand-checked).
- ICLR 2027 public submissions are a large pool round 1 never covered (titles only so far), e.g. "The Expression Gap", "Lagged Coupling", "Post-Training Reshapes Truth-Sycophancy Geometry", "Harm Recognition to Refusal Gap", "Drop the Act: Probe-Filtered RL", "Refusal Falls off a Cliff".
- Method caveat: 2609.31401 argues decodability on errors alone cannot show information was discarded.

## Not done in phase 1
OpenReview NeurIPS 2026 main track (not public); ACL Anthology 2026 search; lab blogs; abstract skim of 141 title-only OpenReview rows; duplicate resolution between sweeps.
