"""TextCraft: frozen tasks, a lightweight per-episode env, and per-round labels.

Reuses the stage-2 mechanics unchanged (canonical recipe order, upstream step
logic, AgentGym action parsing), so episodes here are directly comparable to
runs/stage2-002.
"""
import contextlib
import io
from copy import deepcopy

from failure_probes.environment import TextCraft, parse_action, upstream_conversation, validity


class Episode:
    """One env instance sharing the (read-only) crafting tree."""

    def __init__(self, tc, task):
        self.env = tc.env_cls(tc.tree, task["commands"], task["goal"])
        observation, _ = self.env.reset(commands=task["commands"], goal=task["goal"])
        if observation != task["initial_observation"]:
            raise ValueError("Frozen task reset changed its prompt")
        self.initial_observation = observation

    def step(self, response):
        action, span, error = parse_action(response)
        before = deepcopy(self.env.inventory)
        if error:
            observation, reward, done = error, 0, False
        else:
            with contextlib.redirect_stdout(io.StringIO()):  # upstream prints "Wrong Item Count"
                observation, reward, done, _, _ = self.env.step(action)
        return {"action": action, "action_char_span": span, "observation": observation,
                "reward": reward, "done": done, "validity": validity(observation, error),
                "inventory_before": before, "inventory_after": deepcopy(self.env.inventory)}


def load(depths, environment_seed=42):
    """Frozen task dicts for every task whose min recipe depth is in `depths`."""
    tc = TextCraft()
    tasks = []
    for idx in tc.tasks_by_depth(depths):
        task = tc.freeze_task(idx, environment_seed)
        task["recipe_tree"] = recipe_tree(tc, task["goal"])
        tasks.append(task)
    return tc, tasks


def recipe_tree(tc, goal):
    """Names (item ids or tags) of the goal and every ingredient reachable below it."""
    with contextlib.redirect_stdout(io.StringIO()):  # upstream prints cycle notices
        recipes = tc.tree.traverse_recipe_tree(goal, set())
    names = {goal}
    for recipe in recipes:
        names.add(recipe.output_item.item_tag.name)
        for item in recipe.input_items:
            names.add(item.item_tag.name)
    return sorted(str(n) for n in names)


def round_labels(task, rec, tree_tags):
    """IMMEDIATE proxies for one executed round (ASSUMPTIONS A-8 b/c)."""
    tree = set(task["recipe_tree"])
    gained = [k for k, v in rec["inventory_after"].items() if v > rec["inventory_before"].get(k, 0)]
    on_tree = any(k in tree or tree_tags.get(k) in tree for k in gained)
    return {"valid": int(rec["validity"] == "executable"),
            "progress": int(rec["observation"].startswith(("Got ", "Crafted ")) and on_tree),
            "crafted": int(rec["observation"].startswith("Crafted "))}


def conversation_start():
    return upstream_conversation()
