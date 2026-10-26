# Speaker notes: Knowing Without Doing (Dr. Lin, 23 Oct 2026)

## 1. Title
- The question: when a model acts against what it knows, can we tell from its internals which fix will work?
- "I'll go problem, what's known, what's new, the plan, why it's worth it, and a first step."

## 2. The model knows Canberra, then says Sydney
- Walk the example: correct in turn 1, flips under pushback, while a probe still reads "Canberra" inside.
- Why it matters: this is sycophancy seen from the inside, and it's what white-box lie detectors try to catch.
- It's real at scale: unanswerable questions are recognized internally (AUC 0.939, up to 70B), yet answered anyway.

## 3. Readable is not the same as used
- Left: a probe reads a variable at R² = 0.981. Removing that direction costs 0.6 points, the same as a random direction. The subspace the model really uses (found by DAS) drops accuracy from 42% to 0%.
- Right: after fine-tuning, a fact stays readable (0.98) but is barely used (0.15). "Used" is measured without any probe: does the decision change when only that fact changes?
- So what: a probe result alone proves nothing about the cause. Every claim we make gets a causal test.
- Caveat if asked: the angle alone isn't proof; that's why we push the signal both ways and swap inputs.

## 4. Six fates, each pointing to a different fix
- In plain words:
  - **readout:** the answer reaches the output, but the cut-off is wrong;
  - **bypass:** it sits in a side channel the decision never reads;
  - **not routed:** inserting it would fix things, but nothing carries it there;
  - **override:** a stronger signal wins;
  - **transformed:** still there but rewritten, e.g. flipped;
  - **erased:** gone.
- Why each matters: each one needs a different fix (calibration, gating, training the route, tuning the competitor, refitting monitors, better data).
- So what: the label is only worth having if it predicts which fix works. That's prediction H2.
- Dark band = studied before, dashed = barely. Even the studied ones were each one task, mostly in small models.

## 5. Today: each paper tests one fate, with one method
- How to read it: each row is a paper, the dot is the fate it tested, the label is its biggest model.
- The real problem isn't just "nobody did all six". It's that the method seems to decide the answer. On sycophancy, three papers disagree (erased, override, two-stage), each with a different method.
- So what: running every check on the same items tells us who's right, or that each is right about different items.
- Almost nobody tests the natural route (the 72B test is null).

## 6. Two lessons from that work shape our design
- **Lesson 1:** at 8B a probe adds only +0.008 AUC over the plain output scores. So check 1 is always the output scores; maybe nothing is hidden.
- **Lesson 2:** RLHF cut sycophancy from 39% to 3.5%, but the responsible heads matter more afterwards (+10.5 → +27 pp). A fix can hide the problem, so Stage B reruns every check after the fix. Caveat: that's one public checkpoint pair.

## 7. What's new: three predictions that can fail
- **H1 (mechanism):** pushback and a tempting wrong option produce different fate mixes. If true, "sycophancy" isn't one thing and needs different fixes.
- **H2 (fixes):** a fate-matched fix beats plain DPO on held-out items. If false, we learn diagnosis isn't needed, which is still useful.
- **H3 (monitors):** probes trained on normal behavior miss transformed and bypass failures. If true, lie detectors need in-context checks.
- Key line: every outcome tells labs something they can act on.

## 8. Our study: sort each failure, then test the fix
- Same facts under four kinds of pressure, so differences come from the pressure, not the topic:
  1. tempting wrong option;
  2. user pushback;
  3. no single answer;
  4. forced to answer.

  Start with 1 and 2.
- Stage A (six checks, cheapest first) answers H1 and H3. Stage B (fix, then recheck) answers H2.
- Tools: probes plus SAE / ICA Lens features, so we can find signals we didn't guess.
- Models 9B–27B; 70B only with her compute.

## 9. Why it's worth the time
- **Who uses it:** lie-detector builders (H3), auditors (if fixes only hide the problem), training teams (some fixes teach hiding: GRPO with a weak detector, over 85% evasion), and scale questions.
- **Risks, each with an answer:**
  - mostly readout failure → the pilot checks it first;
  - scoop → 2610.21708 is about agents, so we stay on factual pressure and fixes;
  - H2 false → still a result.
- **Cost:** about 1 GPU-day per model per condition.

## 10. First step: a one-day pilot, and my questions
- The pilot: about 200 pushback items on Gemma-2-9B. Measure the flip rate, then compare output-score AUC with probe AUC at the flip.
- Go if the probe clearly beats the output scores (there's a hidden signal to sort); pivot if not.
- Then ask the four questions, starting with compute.

## 11. Backup: a cheap geometry test
- Each signal is a direction. The angle between "knowing" and "acting" says how much knowledge can drive the answer.
- KAPPA trains one probe on the true answer and one on the model's pick; they sit 60–80° apart. Nudging them together helps (TruthfulQA 56.7 → 67.8–73.5).
- Open: is the angle wider on failing items? If not, we drop direction 2.

## 12. Backup: how much to trust this review
- It's a fast pass: 22 of 31 must-find papers found, headline numbers fact-checked.
- Still to do: citation chasing, full reads, and 2505.11770.
