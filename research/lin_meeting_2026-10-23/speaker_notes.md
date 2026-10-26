# Speaker notes: Knowing Without Doing (weekly update, 26 Oct 2026)

Main deck is 8 slides (about 10 minutes). Everything else is backup, for questions.

## 1. Title
- Weekly update. Same question as last time; this week is about the plan and what runs first.

## 2. Recap: the model knows Canberra, then says Sydney
- Turn 1 it answers right. Turn 2 the user pushes back and it gives in.
- A probe can still read "Canberra" inside. So the knowledge is there; something stops it reaching the answer.

## 3. Since last week
- Pressure is broader than pushback: a tempting wrong option, pushback, no single answer, forced to answer. We start with the first two.
- Order changed: sort failures first (Stage A), test fixes last (Stage B), because training our own fixes is the expensive part.
- The geometry review gave 8 ranked ideas; the first three are cheap (under 3 GPU-hours on a 9B model).
- Datasets are not decided; I want to decide them with you.

## 4. The idea: does knowing stop steering?
- Sketch, not data. Two arrows per item: "knowing" (a probe trained on the true answer) and "acting" (a probe trained on the answer the model gives).
- When the model holds the right answer, we expect the arrows to point roughly the same way.
- Prediction: when it flips under pressure, the angle opens up. The knowledge is still there but no longer steers.
- KAPPA measured this angle on average (60–80° in middle layers). Open question for us: is it wider on flipped items? If the angle is the same on both, we drop this idea.

## 5. Plan: sort first, fix last
- Same facts under four pressures; start with 1 + 2.
- Stage A: sort each failure into one of the six fates (backup slide has them). Measure small to big: one direction first, then a subspace, then paths through tokens and layers, only moving up when needed.
- Stage B last: does each fate need its own fix? Use public checkpoint pairs first (base vs instruct, OLMo SFT vs DPO), because our own DPO is about 40–70 GPU-hours at 9B.
- Models: Gemma-2-9B first, then Qwen2.5-32B; 70B only with your compute.

## 6. First run: a nearly free screen
- Sketch, not data. Uses only output scores, so it costs almost nothing.
- Blue: only unsure items flip. Then the story is calibration and the project shrinks; good to know early.
- Orange: confident items flip too. Then something inside overrides what the model knows, and Stage A is worth it.
- Then two quick checks before trusting any fate label: copy vs correctness (does the probe read the fact or just the earlier answer in the chat?) and planting fake fates to test our tools.

## 7. Still open: which data?
- Templates give same-length pairs, which patching needs, but they are artificial.
- TriviaQA is natural but lengths vary.
- My draft: templates for mechanism, TriviaQA for behavior. I'd like your view.

## 8. Questions for you
- Compute: 30B and below, or is 70B possible?
- Start with tempting wrong option + pushback?
- Speed: 2610.21708 (15 Oct) is close on agent ask-vs-act, and the "decodability is not use" angle is crowded. How fast should we move?

## Backup slides
- Six fates; readable vs used; evidence map; two lessons (output-score control, masking after RLHF); three predictions H1–H3; what papers measure for the angle; first experiments with kill rules; the eight ranked ideas; why it's worth the time and risks; how much to trust the literature review.
