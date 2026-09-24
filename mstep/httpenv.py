"""AgentGym HTTP environments (ALFWorld, ScienceWorld, BabyAI) behind the same
Episode interface as mstep.tc, mirroring each AgentGym client exactly: same
conversation start, same first observation, same action parsing, same next-turn
message (vendor/AgentGym/agentenv/agentenv/envs/{alfworld,sciworld,babyai}.py).

Each env server runs in its own venv (~/envs/<name>) on a local port; see
scripts/env_servers.sh.

Uniform per-round fields: action, observation, reward (1 iff this step completed
the task), done, validity (executable / invalid), native_score (env's own
progress signal: ScienceWorld score 0-100; BabyAI's is terminal-only (MiniGrid
completion reward, 0 before success), so it is NOT a step signal; ALFWorld has none
-> None), score_delta.
"""
import ast
import re

import requests

from failure_probes.common import ROOT

CLIENTS = ROOT / "vendor/AgentGym/agentenv/agentenv/envs"
PORTS = {"alfworld": 36002, "sciworld": 36001, "babyai": 36003}
# AgentEval test-split indices (HF AgentGym/AgentEval *_test.json).
INVALID = {"alfworld": ("Nothing happens.",), "sciworld": ("No known action matches that input.",),
           "babyai": ("The action is not recognized",)}


def conversation_start(env):
    """First two ConversationMessage literals of the client's (REACT) conversation start."""
    fname = {"alfworld": "alfworld.py", "sciworld": "sciworld.py", "babyai": "babyai.py"}[env]
    tree = ast.parse((CLIENTS / fname).read_text())
    msgs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("conversation_start", "conversation_start_dict")
                                                for t in node.targets):
            calls = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)
                     and getattr(n.func, "id", None) == "ConversationMessage"]
            msgs = [ast.literal_eval(c.args[0]) for c in calls[:2]]  # REACT is listed first; later ones may be f-strings
            break
    if len(msgs) != 2:
        raise ValueError(f"could not extract conversation start for {env}")
    return [{"role": "user" if m["from"] == "human" else "assistant", "content": m["value"]} for m in msgs]


def parse_react(text):
    """AgentGym BaseAdapter.parse_react: text after the last 'Action:'."""
    parts = text.rsplit("Action:", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def parse_babyai(text):
    matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", text, re.DOTALL)
    if len(matches) > 1:
        return None
    action = matches[-1] if matches else ""
    action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
    return " ".join(action.split()).strip()


class Episode:
    def __init__(self, env, task, host="127.0.0.1"):
        self.env, self.base = env, f"http://{host}:{PORTS[env]}"
        self.session = requests.Session()
        self.id = self._post("create", {}, with_id=False)["id"]
        idx = task["data_idx"]
        if env == "alfworld":
            r = self._post("reset", {"game": idx, "world_type": "Text"})
            self.initial_observation = f"{r['observation']}\nAVAILABLE ACTIONS: {','.join(r['available_actions'])}"
            self.score = 0.0
        elif env == "sciworld":
            r = self._post("reset", {"data_idx": idx})
            self.initial_observation = r["task_description"] + "\n" + r["observation"]
            self.score = 0.0
        else:
            r = self._post("reset", {"data_idx": idx})
            self.initial_observation = r["observation"]
            self.score = float(r.get("reward", 0.0))
        self.last_observation = self.initial_observation

    def _post(self, path, data, with_id=True):
        if with_id:
            data = dict(data, id=self.id)
        res = self.session.post(f"{self.base}/{path}", json=data, timeout=300)
        res.raise_for_status()
        out = res.json()
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"{self.env} {path}: {out['error']}")
        return out

    def step(self, response):
        env = self.env
        text = response[:-5] if response.endswith("</s>") else response
        if env == "babyai":
            action = parse_babyai(text)
            if action is None:
                obs = "Error: Only one 'Action' is allowed per response. Please adjust your response."
                return self._rec("", obs, False, invalid=True, native=self.score)
        else:
            action = parse_react(text)
        r = self._post("step", {"action": action})
        obs, done = r["observation"], bool(r["done"])
        if env == "alfworld":
            success, native = float(r["reward"]) >= 1.0, None
        elif env == "sciworld":
            native = float(r["score"])
            success = native >= 100
        else:
            # AgentGym's BabyAI server never passes obs_to_reward, so `reward` is plain MiniGrid:
            # 0 on every step, 1 - 0.9*steps/max_steps on completion, and `done` is only set when
            # that reward exceeds 0.5 (environment.py update()). So done == success, and there is
            # NO per-step or subgoal reward (A-21).
            native = float(r["reward"])
            success = done and "The task is completed" in obs
        invalid = obs.strip().startswith(INVALID[env])
        return self._rec(action, obs, done, invalid, native, success)

    def _rec(self, action, obs, done, invalid, native, success=False):
        prev = self.score
        if native is not None:
            self.score = native
        self.last_observation = obs
        return {"action": action, "observation": obs, "reward": int(success), "done": bool(done or success),
                "validity": "invalid" if invalid else "executable", "native_score": native,
                "score_delta": (None if native is None else native - (prev or 0.0))}

    def close(self):
        try:
            self._post("close", {})
        except Exception:
            pass
        self.session.close()
