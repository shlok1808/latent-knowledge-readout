---
title: "Knowing Without Doing"
author: "Shlok"
date: "23 October 2026"
fonts:
  heading: "Inter"
  body: "Inter"
custom_css: |
  :root { --colloquium-accent: #2B8A9E; --colloquium-heading: #1F2430; --colloquium-text: #1F2430; }
  .slide h2 { font-weight: 700; letter-spacing: -0.01em; font-size: 44px; }
  .slide--title h2 { font-size: 30px; font-weight: 500; color: #4B5563; margin-top: 10px; }
  .slide .slide-content img { display: block; margin: 0 auto; max-height: 470px; object-fit: contain; }
  .slide.img-short .slide-content img { max-height: 380px; }
  .slide.img-mid .slide-content img { max-height: 330px; }
  .slide.img-tall .slide-content img { max-height: 490px; }
  .slide.img-pair .slide-content img { max-height: 330px; }
  .slide.img-map .slide-content img { max-height: 410px; }
  .slide .slide-content li { margin-bottom: 0.35em; }
  .slide .lead { font-size: 22px; color: #4B5563; margin: -6px 0 10px; }
  .slide .so { font-size: 22px; color: #1F2430; margin-top: 14px; border-left: 4px solid #2B8A9E; padding: 6px 14px; background: #F1F8FA; border-radius: 4px; }
  .slide .cap { font-size: 18px; color: #374151; margin-top: 8px; line-height: 1.4; }
  .slide .q { background: #E3F4F8; border-left: 4px solid #2B8A9E; padding: 14px 18px; border-radius: 6px; }
  .slide table { font-size: 19px; border-collapse: collapse; width: 100%; }
  .slide th { text-align: left; color: #4B5563; font-weight: 600; border-bottom: 2px solid #D1D5DB; padding: 8px 10px; }
  .slide td { border-bottom: 1px solid #E5E7EB; padding: 10px; vertical-align: top; }
---

# Knowing Without Doing

## When a model acts against what it knows, can we tell from the inside which fix will work?

Shlok · meeting with Dr. Lin · 23 October 2026

---

## The model knows Canberra, then says Sydney

<!-- class: img-short -->

<p class="lead">Goal: find out <i>how</i> the right answer fails to reach the output, and which fix each failure needs.</p>

![](figures/fig_problem.svg)

<!-- footnote: This is sycophancy seen from the inside, and it is what white-box lie detectors try to catch. Not a toy effect: unanswerable questions are recognized internally (AUC 0.939, 11 models up to 70B) yet answered anyway [2608.29109]. -->

---

## Readable is not the same as used

<!-- class: img-mid -->

![](figures/fig_used.svg)

<p class="so">→ So we never stop at "a probe can read it". Every claim gets a causal test: push the signal both ways, remove it, and swap inputs that differ only in that fact.</p>

---

## Six fates, each pointing to a different fix

<!-- class: img-tall -->

![](figures/fig_fates.svg)

<!-- footnote: Dark band = studied before; dashed = barely studied. → A fate label is only worth having if it predicts which fix works. That is what we test (prediction H2). -->

---

## Today: each paper tests one fate, with one method

<!-- class: img-map -->

<p class="lead">Three sycophancy papers disagree ("erased", "override", "two-stage"), each with a different method.</p>

![](figures/fig_evidence_map.svg)

<!-- footnote: → Running every check on the same items tells us who is right, or whether each is right about different items. "Natural route" = does the unmodified model actually use the signal? -->

---

## Two lessons from that work shape our design

<!-- size: small -->
<!-- class: img-pair -->
<!-- columns: 2 -->

**1. Rule out the boring explanation first.** At 8B, output scores alone almost match a trained probe.

![](figures/fig_margin.svg)

<p class="cap">→ Check 1 of our study is always the output scores.</p>

|||

**2. Check whether a fix removed the problem.** RLHF cut sycophancy ~10×, but the heads behind it matter more.

![](figures/fig_masking.svg)

<p class="cap">→ Stage B reruns every check after the fix.</p>

<!-- footnote: Left: 2609.04582 (synthetic logic task, up to 8B). Right: 2604.19117, one pair of public 70B checkpoints. -->

---

## What's new: three predictions that can fail

| | Prediction | If true | If false |
|---|---|---|---|
| **H1 · Mechanism** | Pushback and a tempting wrong option produce *different* fate mixes | "Sycophancy" is not one thing; each pressure needs its own fix | One shared mechanism; one fix may cover both |
| **H2 · Fixes** | A fix matched to the fate beats plain DPO on held-out items, with less damage | Diagnose first, then fix | Skip the diagnosis; generic fixes are enough |
| **H3 · Monitors** | Probes trained on normal behavior miss "transformed" and "bypass" failures | Lie detectors need in-context checks | Today's monitors are safer than feared |

<!-- footnote: Every outcome tells labs something they can act on, including the "if false" column. -->

---

## Our study: sort each failure, then test the fix

<!-- class: img-tall -->

![](figures/fig_plan.svg)

<!-- footnote: Stage A tests H1 and H3; Stage B tests H2. Models 9B–27B (Gemma-2-9B, Gemma-2-27B, Qwen2.5-14B); 70B only if compute allows. Probes plus SAE / ICA Lens features, so we are not limited to signals we guessed. Examples 3 and 4 are illustrations. -->

---

## Why it's worth the time

<!-- size: small -->
<!-- columns: 2 -->

**If it works, who uses it**

- **Lie detectors:** shows *when* white-box monitors miss the failure (H3)
- **Audits:** if fixes only hide the problem, behavior tests can't certify a model (Stage B)
- **Training:** flags fixes that teach hiding (weak detector + GRPO: >85% evasion; DPO: <25%)
- **Scale:** tests whether small-model mechanisms hold in larger models

|||

**Risks, and what we do about them**

- **Most flips are plain readout failures** → the project shrinks to calibration. The one-day pilot checks this first.
- **Scoop:** 2610.21708 (15 Oct) covers agent ask-vs-act → we stay on factual pressure and on fixes.
- **No difference between fixes (H2 false)** → still a result: diagnosis isn't needed.

**Cost:** about 1 GPU-day per model per condition, at 9B–27B.

---

## First step: a one-day pilot, and my questions

<!-- columns: 3/2 -->

**Pilot: is there a hidden signal worth sorting?**

1. Gemma-2-9B, about 200 TriviaQA pushback items
2. How often does the model flip?
3. At the flip, compare output-score AUC with probe AUC
4. **Go** if the probe clearly beats the output scores; **pivot** if not

|||

<div class="q">

**Questions for you**

1. Compute: stay at 30B or below, or is 70B possible?
2. Start with pushback + tempting wrong option only?
3. Add an instructed-deception condition later?
4. Given 2610.21708, how fast should we move?

</div>

<!-- footnote: In one line: find how the right answer fails to reach the action, then test whether monitors catch it and whether fixes remove it or only hide it. -->

---

## Backup: a cheap geometry test

<!-- size: small -->
<!-- class: img-pair -->
<!-- columns: 2 -->

**What the angle means**

![](figures/fig_angle_explainer.svg)

<p class="cap">KAPPA (Park et al.): in multiple-choice QA, train one probe on the true answer and one on the answer the model picks, then measure the angle between them.</p>

|||

**What papers measure**

![](figures/fig_angles.svg)

<p class="cap">Nudging the "pick" direction toward the "true" one helps (TruthfulQA 56.7 → 67.8–73.5). <b>Open:</b> is the angle wider on the items where the model fails?</p>

<!-- footnote: If the angle does not change on failing items, we drop the geometry direction. -->

---

## Backup: how much to trust this review

- Fast literature pass: about 70 papers, many read from abstracts only
- Found 22 of 31 papers we knew we had to find; the misses were mostly OpenReview-only workshop papers
- Headline numbers fact-checked against the papers; the rest still marked "to verify"
- Before claiming novelty: chase citations both ways and fully read the abstract-only core papers, including 2505.11770 (closest prior on the natural-route gap)

<!-- footnote: Full report with every source: reports/Knowledge action divergence internals round 2.md -->
