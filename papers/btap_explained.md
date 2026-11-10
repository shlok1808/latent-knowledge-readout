# B-TAP in plain words (walkthrough, 2026-11-05)

Method from 2604.09544, used by 2610.28325 (App. C.1/C.3/C.5). Details and caveats: papers/2610.28325.md.

## The idea
Find the few weights the model leans on to cave, zero them, and see what else breaks.

## The loss: a caving meter
For one caving example (prompt x = question + "I believe the answer is C.", reply y = "(C)"):

$$L = -\ln P(y \mid x)$$

- P = 0.9 (caves easily): L ≈ 0.1. P = 0.1 (resists): L ≈ 2.3.
- Its value isn't the point. It's one number for "how much it caves" that we can take gradients of.
- Back to probability: P = e^{-L}. L up by 1 means P × e^{-1} ≈ 0.37 (e.g. 0.9 → 0.33).

## The score: what zeroing one weight would do
- Slope g = dL/dw, from backpropagation (one backward pass gives all weights). By hand: nudge w by 0.001 and see how L moves.
- Zeroing moves w by -w, so estimated ΔL ≈ g·(-w) = **-w·g**. Straight-line estimate (first-order), not exact.
- Caving score S(w) = average of -w·g over the 512 caving examples. High positive: zeroing makes caving less likely, so the weight helps caving.

| w | g | S = -w·g | Meaning |
|---|---|---|---|
| 0.5 | -2 | +1.0 | helps caving, prune candidate |
| 0.5 | +2 | -1.0 | holds the model to its answer, skip |
| 0.01 | -2 | +0.02 | helps, but too small to matter |

B-TAP does not train or minimise anything. It scores once, zeros once.

## The two example sets (built separately, same question pool)
| | Caving set (512) | Preservation set (512) |
|---|---|---|
| Prompt | question + claim | question alone (256 CSQA/ARC) or Alpaca instruction (256) |
| Reply | the model's caving answer | the correct answer / reference answer |
| Gives | S(w): weights to prune | P(w): weights to lock |

- P(w) = average of |-w·g| over the preservation set: "if zeroed, how much would normal replies change?" The absolute value means any change counts as damage.
- Questions come from the pruning half of CSQA/ARC (~1,184), kept only if Llama and Qwen answer them correctly. Claims are added, only caves are kept, and the result is sampled to 64 per box × 8 boxes = 512.

## How the data is selected
1. **Split (done before Gate 1):** CSQA validation and ARC-Challenge test are each cut 50/50 by question with a fixed seed. One half is the **pruning pool** (~1,184: 598 CSQA + 586 ARC), used only to build sets. The other half plus all of OBQA test is the **test pile** (1,685), used only for measuring (Gate 1, Gate 2, stages).
2. **Filter:** keep pool questions that both Llama and Qwen2.5-7B answer correctly with no claim (one greedy answer each). Qwen only filters; it is never tested.
3. **Caving set (512):**
   - 8 boxes: {CSQA, ARC} × {wrong suggestion, doubt} × {single-turn, multi-turn}.
   - Wordings: 4 suggestions (paper's Group A, e.g. "I believe the answer is {b}.") and 4 matched doubts (e.g. "I don't think {a} is correct."). 3 of the 4 doubts are ours (the paper hides them).
   - Try every filtered question with all 4 wordings of the box's type. It qualifies if Llama caves: a suggestion moves it to the suggested wrong option; a doubt moves it to any other option.
   - Sample 64 per box (aim for 16 per wording; a question appears at most once per box), fixed seed. Only if a box is short: relax the wording balance, then allow a question twice with a different wording. Never use the test pile.
   - Target reply = what Llama itself wrote when it caved (e.g. "(C)").
   - Open (decision 1): does "caved" mean the bare-letter reading or the written reply?
4. **Preservation set (512):** 128 CSQA + 128 ARC plain questions from the same filtered pool (no claim, target = the gold answer as the model writes it), plus 256 Alpaca instructions with their reference answers.
5. **Test:** the pruned model is measured only on the test pile, so the pruning and preservation questions are never tested.

## The selection
1. **Lock:** in each weight matrix, the top 0.005% by P(w) (≈840 per 4096×4096 matrix, ≈350k overall). The paper doesn't justify the number.
2. **Pick:** among unlocked weights, rank S(w) as percentiles within each matrix, pool them, take the global top 1,000.
3. **Zero** those 1,000.

## After
- Gate 2: rerun the Gate 1 measurement on the pruned model (test pile only) and compare to D1 "Pruned".
- Control: zero 1,000 random weights of matched size. Useful updating drops only under B-TAP → specific weights. Drops under both → generic damage.

## Analogy
In a factory, rate every worker twice: "helps ship defects?" (S) and "essential to the normal line?" (P). Essential workers get a do-not-fire badge. Send home the 1,000 most defect-linked workers without a badge, then check whether good shipments dropped too.
