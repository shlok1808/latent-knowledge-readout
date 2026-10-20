"""τ²-bench (sierra-research/tau2-bench) behind the mstep Episode interface.

The agent is OUR policy (Qwen via the vLLM server, exact token ids recorded for replay);
τ²'s own AgentGymEnv runs the conversation: tool execution against the domain DB, the
user simulator (an LLM, served locally by the same vLLM server through litellm, thinking
off), and τ²'s evaluator for the final reward (DB end state + required communications).

Agent prompt mirrors τ²'s LLMAgent: SYSTEM_PROMPT(AGENT_INSTRUCTION, domain policy) with
the domain tools passed through the chat template. One action per turn: a tool call
(first <tool_call> block) or a message to the user.

Per-round fields beyond the uniform ones (action, observation, reward, done, validity):
  expected_action_match: the tool call equals one of the task's ground-truth actions
      (same name and same arguments) -- a REAL per-step label, not a text proxy;
  action_kind: "tool" | "text"; chat: the new messages to append to the agent's history.
"""
import json
import re

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_tasks(domains_splits, pilot_per_domain=None):
    """[(domain, split)] -> frozen task dicts, including the ground-truth action list."""
    from tau2.registry import registry
    from failure_probes.common import digest
    out, idx = [], 0
    for domain, split in domains_splits:
        tasks = registry.get_tasks_loader(domain)(split)
        if pilot_per_domain:
            step = max(1, len(tasks) // pilot_per_domain)
            tasks = tasks[::step][:pilot_per_domain]
        for t in tasks:
            ec = t.evaluation_criteria
            exp = [{"name": a.name, "arguments": a.arguments} for a in (ec.actions or [])] if ec else []
            task = {"task_id": f"tau2_{domain}_{t.id}", "domain": domain, "tau_task_id": str(t.id), "split": split,
                    "data_idx": idx, "recipe_depth": 0, "expected_actions": exp,
                    "reward_basis": [str(r) for r in (ec.reward_basis or [])] if ec else []}
            task["task_hash"] = digest(task)
            out.append(task)
            idx += 1
    return out


_PATCHED = False


def _local_judge(cfg):
    """Point τ²'s NL-assertion judge (default GPT-4.1, an external API) at the local server.
    Its verdict is recorded but NOT used for the primary success label (A-25)."""
    global _PATCHED
    if _PATCHED:
        return
    import tau2.evaluator.evaluator_nl_assertions as nl
    nl.DEFAULT_LLM_NL_ASSERTIONS = cfg["user_sim"]["llm"]
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(cfg["user_sim"]["args"])
    _PATCHED = True


DETERMINISTIC = ("DB", "COMMUNICATE")


def _breakdown(info):
    try:
        rb = json.loads(info.get("reward_info") or "{}").get("reward_breakdown") or {}
    except Exception:
        rb = {}
    return {str(k).split(".")[-1]: float(v) for k, v in rb.items()}


def _canon(args):
    return json.dumps(args or {}, sort_keys=True, default=str)


class Episode:
    parallel_step = True  # each step blocks on the user simulator: step episodes concurrently

    def __init__(self, task, cfg):
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
        from tau2.gym.gym_agent import AgentGymEnv
        u = cfg["user_sim"]
        _local_judge(cfg)
        self.env = AgentGymEnv(domain=task["domain"], task_id=task["tau_task_id"], max_steps=cfg.get("tau2_max_steps", 200),
                               user_llm=u["llm"], user_llm_args=dict(u["args"]))
        self.env.reset()
        # reset() can return before the user simulator's opening message lands in the agent's
        # observation (race inside AgentGymEnv); without this wait the agent acts on an empty
        # conversation and hallucinates tool arguments (smoke test 2026-09-24).
        import time as _t
        from tau2.data_model.message import UserMessage
        deadline = _t.time() + 120
        while not any(isinstance(m, UserMessage) for m in (self.env._agent.observation or [])):
            if self.env._simulation_done.is_set() or _t.time() > deadline:
                raise RuntimeError(f"no opening user message for {task['task_id']}")
            _t.sleep(0.05)
        self.tools = [t.openai_schema for t in self.env._get_tools()]
        system = SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=self.env._get_policy())
        first = self._chat(self.env._agent.observation)
        self.initial_messages = [{"role": "system", "content": system}] + first
        self.initial_observation = "\n".join(m["content"] for m in first)
        self.expected = {(a["name"], _canon(a["arguments"])) for a in task["expected_actions"]}
        self.final_reward = 0.0

    def _chat(self, messages):
        """New user/tool messages only: the gym's observation can repeat messages the agent
        already saw (the opening user turn came back after the first tool call, 2026-09-24)."""
        from tau2.data_model.message import ToolMessage, UserMessage
        seen = self.__dict__.setdefault("_seen", set())
        out = []
        for m in messages or []:
            key = (type(m).__name__, getattr(m, "timestamp", None), str(getattr(m, "content", "")),
                   getattr(m, "id", None))
            if key in seen:
                continue
            seen.add(key)
            if isinstance(m, ToolMessage):
                out.append({"role": "tool", "content": str(m.content), "_error": bool(getattr(m, "error", False))})
            elif isinstance(m, UserMessage):
                out.append({"role": "user", "content": str(m.content or "")})
        return out

    def step(self, response):
        text = THINK.sub("", response).strip()
        calls = TOOL_CALL.findall(text)
        call, kind, parse_error = None, "text", None
        if calls:
            kind = "tool"
            try:
                call = json.loads(calls[0])
                action = json.dumps({"name": call["name"], "arguments": call.get("arguments") or {}})
            except Exception as e:  # malformed tool JSON: τ² sees the raw text and replies
                parse_error, action = repr(e), text
        else:
            action = text or "(empty)"
        obs, reward, done, _, info = self.env.step(action)
        chat = self._chat(self.env._agent.observation)
        tool_error = any(m.get("_error") for m in chat) or (obs or "").startswith("Invalid action")
        for m in chat:
            m.pop("_error", None)
        match = bool(call and (call.get("name"), _canon(call.get("arguments"))) in self.expected)
        breakdown = _breakdown(info) if done else {}
        det = [breakdown[k] for k in DETERMINISTIC if k in breakdown]
        success = bool(done) and (all(v >= 1.0 for v in det) if det else reward >= 1.0)
        if done:
            self.final_reward = float(reward)
        return {"action": action, "action_kind": kind, "n_tool_calls": len(calls), "parse_error": parse_error,
                "observation": "\n".join(f'{m["role"]}: {m["content"]}' for m in chat) or (obs or ""),
                "chat": chat, "reward": int(success), "tau2_reward": float(reward), "reward_breakdown": breakdown,
                "done": bool(done), "validity": "invalid" if (parse_error or tool_error) else "executable",
                "expected_action_match": int(match)}

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
