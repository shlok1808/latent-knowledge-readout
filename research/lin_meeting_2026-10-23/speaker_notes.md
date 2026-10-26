# Speaker notes: Knowing Without Doing (Dr. Lin, 23 Oct 2026, updated 26 Oct)

## 1. Title
- The question: when a model acts against what it knows, can we tell from its internals which fix will work?
- "I'll go problem, what's known, what's new, the plan, how we measure, why it's worth it, the first experiments, and my questions."

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
- **H2 (fixes):** a fate-matched fix beats plain DPO on held-out items. If false, we learn diagnosis isn't needed, which is still useful. We test it last (Stage B).
- **H3 (monitors):** probes trained on normal behavior miss transformed and bypass failures. If true, lie detectors need in-context checks.
- Key line: every outcome tells labs something they can act on.
- If asked about H1: two papers disagree. 2601.11563 claims one shared compliance direction; 2607.20146 finds separable sycophancy modes in Gemma-2-9B. So H1 is open.

## 8. Our study: sort each failure, then test the fix
- Same facts under four kinds of pressure, so differences come from the pressure, not the topic:
  1. tempting wrong option;
  2. user pushback;
  3. no single answer;
  4. forced to answer hard or impossible questions.

  Start with 1 and 2.
- Stage A (sort into the six fates, cheapest check first) comes first and answers H1 and H3.
- What changed since Friday: Stage B (fix, then recheck, vs DPO and CCA) moved to the end. Running DPO ourselves at 9–27B is expensive, so we start with public checkpoints: base vs instruct, and OLMo's SFT and DPO stages. We train our own only if those can't answer it.
- Models 9B–27B: first Gemma-2-9B on TriviaQA pairs, then Qwen2.5-32B. 70B only with her compute.

## 9. How we measure: start small, go bigger only if needed
- Three levels: one probe direction, then a whole subspace, then paths through tokens and layers.
- Analogy: check one wire first, then the whole cable, then trace the full circuit. Only open the wall if the wire check can't answer.
- Vocabulary: "decodable by a probe" is not "the model knows". A patch that fixes the output shows we *can* steer it (causal controllability); it does not show the model uses that route by itself (endogenous mediation). Keep those separate.

## 10. Lessons from Nikhil Prakash's binding papers
- I read four of his papers on 26 Oct, after his thesis proposal talk.
- Same-length pairs: our pushback twin changes the token count, so positions don't line up for patching. Template both versions to the same length.
- One variable at a time: an early full-state patch copies everything and "fixes" trivially.
- Before calling something bypass, patch pointer and address together; a mismatch between them can block the lookup even when the route is used.
- Probe and patch beyond the answer token (earlier answer, pushback span, last token). In his VLM paper, patching only the object tokens did not flip the answer; adding the surrounding strips did. For text this is a hypothesis.
- CCA is a Stage B baseline. Its no-mask version tied or beat the masked one, so we must run that control.
- Fine-tuning tends to strengthen existing routes: a prior for what DPO does, not proof (his task had no pressure).

## 11. Why it's worth the time
- **Who uses it:** lie-detector builders (H3), auditors (if fixes only hide the problem), training teams (some fixes teach hiding: GRPO with a weak detector, over 85% evasion), and scale questions.
- **Risks, each with an answer:**
  - mostly readout failure → the output-score screen checks it first;
  - scoop → 2610.21708 is about agents, and "decodability is not use" is crowded in ICLR 2027 submissions, so we stay on factual pressure, per-item fates and fixes;
  - H2 false → still a result.
- **Cost:** Stage A about 1 GPU-day per model per condition. Our own DPO is about 40–70 GPU-h at 9B, which is why Stage B goes last.

## 12. First experiments, ranked by importance
- **1. Copy vs correctness:** after pushback, "the probe still reads Canberra" might only be the model's own earlier answer sitting in the chat. Cross "earlier answer right/wrong" with "fact true/false" and see if a truth probe still works where they disagree. Below 0.65 AUC, Stage A's probe premise fails.
- **2. Plant fake fates:** write known fakes (bypass, rotation, sign flip, offset, erasure) into real 9B activations and see which tools call them right. Any tool that miscalls over 20% is barred.
- **3. Output-score screen:** flip rate vs prior confidence, per pressure. Nearly free, so it can run first. It replaces Friday's "one-day pilot".
- Analogy for 2: test the smoke detector with a match before trusting it in a real fire.

## 13. Still open: datasets, and my questions
- Datasets are NOT decided. "Templated pushback for mechanism work + TriviaQA for behavior" is a draft. I'd like a proper dataset discussion before anything runs.
- Questions, starting with compute:
  1. stay at 30B or below, or is 70B possible?
  2. which pressures first (my plan: tempting wrong option + pushback)?
  3. how fast to move, given 2610.21708 (15 Oct) and a crowded ICLR 2027 field?

## 14. Backup: a cheap geometry test
- Each signal is a direction. The angle between "knowing" and "acting" says how much knowledge can drive the answer.
- KAPPA trains one probe on the true answer and one on the model's pick; they sit 60–80° apart. Nudging them together helps (TruthfulQA 56.7 → 67.8–73.5).
- Open: is the angle wider on failing items? If not, we drop the geometry direction.

## 15. Backup: the eight ranked ideas
- From the 25 Oct geometry report, section 5, after two skeptic passes. Ideas 1–3 are the first experiments.
- Ideas 1–2 come before any fate call. Idea 5 (intervention ladder) has the highest scoop risk. Idea 8 (Stage B report card) is the costliest and runs last.

## 16. Backup: how much to trust this review
- It's a fast pass: 22 of 31 must-find papers found, headline numbers fact-checked. The search never met its stopping rule.
- No citation chasing yet, several core papers unread, so no novelty claims.
- Still to do: citation chasing, full reads, and 2505.11770.
