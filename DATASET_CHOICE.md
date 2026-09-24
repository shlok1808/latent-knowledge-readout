# Phase 4 environment choice

Status: **Decision moved to the Qwen2.5-7B pilots** (Shlok 2026-09-24). The 1.7B pilots below are comparison points only. ALFWorld and BabyAI pilots done. Every pilot: 10 tasks × 8 rollouts, Qwen3-1.7B, temp 0.8 / top-p 0.95 (same as TextCraft), max_rounds 30, AgentEval test split, AgentGym prompts and parsing.

## Criteria (brief's priority order)
1. Multi-step, with identifiable milestones/subgoals.
2. Real per-step or subgoal reward, so IMMEDIATE labels aren't a proxy.
3. Mixed outcomes (success and failure on the same task) for Qwen3-1.7B. **Hard gate**: ≥ 3/10 mixed in the pilot (≈ ≥ 15 on the full split).
4. Text-only, runs in the existing AgentGym setup without heavy dependencies.

## Scored comparison

| | ScienceWorld | ALFWorld | BabyAI |
|---|---|---|---|
| **1 Milestones** | Yes: tasks decompose into scored sub-goals | Implicit (find → take → clean/heat/cool → place); env doesn't signal them. State-change messages ("You pick up …") are observable | **Weak.** Only 1 of the 18 test-split levels has explicit sub-goals (`GoToSeqS5R2`, 5/90 tasks). 3 more are implicitly multi-step (`GoToImpUnlock`, `SynthLoc` PutNext cases, `PickupAbove`: ≤ 15/90 tasks). The other 14 levels (70/90 tasks) are single GoTo/Pickup instructions, where a "stage" is only navigation time |
| **2 Native step reward** | **Yes**: score 0–100 updated per step [confirmed in code] | **No**: `reward = info["won"]` only [confirmed: `env_wrapper.py:98`] | **No.** [confirmed in code] The AgentGym server builds `BabyAI(game_name, seed)` without `obs_to_reward`, so the sub-goal reward in `update_reward` is dead code on the server path. `reward` is plain MiniGrid: 0 every step, `1 − 0.9·steps/max_steps` on completion. *My earlier draft scored this "partial". That was wrong.* |
| **3 Pilot success / mixed** (1.7B) | **1/80, 1/10 mixed** (73 round-limit, 6 env-done); published base-1.7B mean score 0.12/100 | **0/80, 0/10 mixed.** Genuine, not an adapter bug: the model paraphrases commands (`put mug 2 in fridge 1` vs the required `put mug 2 in/on fridge 1`), 37% of actions are "Nothing happens.", all 80 hit the 30-round cap | **33/80 (41%), 8/10 mixed** after the success-rule fix (below). Success lengths 3–18 rounds, so late stages still contain successes |
| **4 Setup** | Works after the fd-limit fix (`ulimit -n 65536`, 40-episode chunks, per-chunk restarts) | OK; server leaks envs (no `/close`), handled by per-chunk restarts | OK, lightest |
| Test split | 200 variations | 200 games | 90 tasks (18 levels × 5) |

**BabyAI success bug (fixed 2026-09-24):** `httpenv` required reward ≥ 1.0, but MiniGrid's completion reward is step-discounted (0.84–0.99). Now success = done and "The task is completed". `done` is only set when the reward is > 0.5, so done ⇒ success. The pilot was relabelled offline by `scripts/relabel_babyai.py`; old values are kept in each episode under `relabel_2026_09_24`.

Per-task BabyAI pilot outcomes (successes / 8):

| Level | Task idx | Successes |
|---|---|---|
| GoToLocalS8N7 | 1804 | 1 |
| GoToLocalS8N7 | 1924 | 4 |
| GoToRedBlueBall | 2088 | 3 |
| PickupAbove | 2260 | 0 |
| GoToObjDoor | 2850 | 4 |
| PickupDistDebug | 3539 | 4 |
| GoToSeqS5R2 | 3847 | 2 |
| PickupLoc | 4058 | 2 |
| GoToObjMazeS7 | 4685 | 5 |
| GoToImpUnlock | 7726 | 8 |

## Where I looked for what I couldn't confirm
- BabyAI base-model success rate: no published number found (HF model cards, AgentGym-RL 2509.08755 search results). The pilot replaces it.
- BabyAI level definitions: MiniGrid v2.4.0 source (`minigrid/envs/babyai/{goto,pickup,other,synth}.py`, `core/levelgen.py`). The installed minigrid is `<2.5` per AgentGym's pyproject; I did not confirm the exact installed minor version.
- ALFWorld's published 19.4% used temp 0.4. I did not test whether temp 0.4 would give mixed outcomes, because the brief fixes sampling settings across environments.

## Decision logic (applied to the Qwen2.5-7B pilots, which decide)
- If ScienceWorld passes the criterion-3 gate, it wins: it is the only candidate that satisfies criterion 2, and it has real milestones.
- If ScienceWorld fails the gate (the prior expectation), **BabyAI** is the only remaining candidate that passes criterion 3. It then enters Phase 4 with these stated weaknesses:
  - IMMEDIATE labels are proxies (validity = "The action is not recognized"; progress = observation-derived, e.g. an object picked up or a door opened).
  - Milestone-stage analysis can only be run on the multi-step levels, which are ≤ 20/90 tasks, and will probably be underpowered.
  - Round-based stages still apply to every level.
- ALFWorld is out at 1.7B (criterion 3). It could be revisited for Qwen2.5-7B.
