"""vLLM rollout engine for TextCraft: main episodes and Monte Carlo continuations.

Runs in the vLLM venv (~/envs/vllm). All active episodes advance one round per
`llm.generate` call, so thousands run concurrently. Every request carries its own
seed: main rounds use the stage-2 formula
`generation_seed + data_idx*100000 + attempt*100 + round`; MC rounds use a digest
of (episode, prefix round, sample, round). vLLM does not promise bit-identical
logits across different batch compositions, so a resumed chunk is a fresh valid
sample from the same policy, not a byte replay (ASSUMPTIONS A-6).

Resumable: episodes/MC results are written per episode as they finish, chunks
skip anything already on disk.

  python -m mstep.rollout collect --config configs/ms_tc_1p7b.json --run runs/ms-tc-1p7b [--pilot]
  python -m mstep.rollout mc      --run runs/ms-tc-1p7b
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from failure_probes.actor import locate_action_tokens
from failure_probes.common import digest, read_json, write_json, append_jsonl
from mstep import tc as tcmod


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mc_seed(episode_id, t, j, rnd):
    return int(hashlib.sha256(f"mc|{episode_id}|t{t}|s{j}|r{rnd}".encode()).hexdigest()[:8], 16)


class Engine:
    def __init__(self, cfg):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
        # Override lets two vLLM jobs share the A100 (e.g. TextCraft MC + a Phase 4 pilot):
        # env MSTEP_GPU_UTIL, else logs/gpu_util (read at engine start), else config.
        override = os.environ.get("MSTEP_GPU_UTIL")
        util_file = Path(__file__).resolve().parents[1] / "logs" / "gpu_util"
        if override is None and util_file.exists():
            override = util_file.read_text().strip()
        util = float(override) if override else cfg.get("gpu_memory_utilization", 0.88)
        log(f"vLLM gpu_memory_utilization={util}")
        self.llm = LLM(cfg["model"], revision=cfg["model_revision"], dtype="bfloat16",
                       gpu_memory_utilization=util,
                       max_model_len=cfg["context_limit"], enable_prefix_caching=True, seed=0)
        self.special = set(self.tok.all_special_ids)

    def params(self, seed):
        from vllm import SamplingParams
        c = self.cfg
        return SamplingParams(temperature=c["temperature"], top_p=c["top_p"], top_k=-1,
                              max_tokens=c["max_new_tokens"], seed=seed, repetition_penalty=1.0)

    def prompt_ids(self, messages):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(text, add_special_tokens=False).input_ids

    def run(self, states, keep_tokens, on_finish):
        """Advance every state until done/round cap. `state` keys: episode, messages,
        rounds, seed_fn(round_index)->int, max_rounds."""
        from vllm.inputs import TokensPrompt
        active = [s for s in states if not s.get("finished")]
        while active:
            prompts, params, live = [], [], []
            for s in active:
                ids = self.prompt_ids(s["messages"])
                if len(ids) + self.cfg["max_new_tokens"] > self.cfg["context_limit"]:
                    s["termination"] = "context_limit"
                    self._finish(s, on_finish)
                    continue
                s["_ids"] = ids
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(self.params(s["seed_fn"](len(s["rounds"]))))
                live.append(s)
            if not live:
                break
            outs = self.llm.generate(prompts, params, use_tqdm=False)
            decoded = []
            for s, out in zip(live, outs):
                gen = list(out.outputs[0].token_ids)
                content = [i for i, t in enumerate(gen) if t not in self.special]
                last = content[-1] if content else len(gen) - 1
                raw = self.tok.decode(gen[:last + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                response = self.tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                _, span, _ = tcmod.parse_action(raw)
                try:
                    positions = locate_action_tokens(self.tok, gen, span)
                except ValueError:
                    positions = []
                decoded.append((s, gen, last, raw, response, positions))
            # Env steps: remote (HTTP) envs are stepped concurrently, in-process ones serially.
            if live and getattr(live[0]["episode"], "session", None) is not None:
                with ThreadPoolExecutor(max_workers=32) as pool:
                    recs = list(pool.map(lambda d: d[0]["episode"].step(d[4]), decoded))
            else:
                recs = [d[0]["episode"].step(d[4]) for d in decoded]
            for (s, gen, last, raw, response, positions), rec in zip(decoded, recs):
                rnd = len(s["rounds"]) + 1
                rec.update({"round": rnd, "seed": s["seed_fn"](rnd - 1), "response": response,
                            "raw_response": raw, "last_content_token": last,
                            "action_token_indices": positions,
                            "hit_token_limit": len(gen) == self.cfg["max_new_tokens"] and gen[-1] not in self.special})
                if keep_tokens:
                    rec["prompt_token_ids"] = s["_ids"]
                    rec["generated_token_ids"] = gen
                else:
                    for k in ("inventory_before", "inventory_after", "action_char_span", "raw_response"):
                        rec.pop(k, None)
                s["rounds"].append(rec)
                s["messages"] += [{"role": "assistant", "content": response},
                                  {"role": "user", "content": rec["observation"]}]
                if rec["done"]:
                    s["termination"] = "success" if rec["reward"] == 1 else "env_done"
                    self._finish(s, on_finish)
                elif len(s["rounds"]) >= s["max_rounds"]:
                    s["termination"] = "round_limit"
                    self._finish(s, on_finish)
            active = [s for s in active if not s.get("finished")]

    @staticmethod
    def _finish(s, on_finish):
        s["finished"] = True
        s["success"] = bool(s["rounds"] and s["rounds"][-1].get("reward") == 1)
        on_finish(s)
        if hasattr(s["episode"], "close"):
            s["episode"].close()
        # Free the heavy parts; states for finished MC samples stay in memory until the chunk ends.
        s["messages"] = None
        s["episode"] = None


def select(tasks, cfg, pilot):
    if not pilot:
        return tasks
    per = cfg["pilot"]["tasks_per_depth"]
    picked = []
    for depth in cfg["depths"]:
        pool = [t for t in tasks if t["recipe_depth"] == depth]
        stride = max(1, len(pool) // per)
        picked += pool[::stride][:per]
    return picked


def restart_server(cfg):
    """Between chunks, restart an env server that leaks envs (ALFWorld has no /close)."""
    if cfg.get("restart_server_every_chunk"):
        import subprocess
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(["bash", str(root / "scripts/env_servers.sh"), "restart", cfg["env"]],
                             capture_output=True, text=True)
        log(f"server restart {cfg['env']}: {out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-200:]}")


def episode_factory(cfg, manifest):
    if cfg["env"] == "textcraft":
        tc = tcmod.TextCraft()
        if tc.fingerprint() != manifest["recipe_order"]:
            raise ValueError("Crafting tree differs from prepare time")
        return lambda task: tcmod.Episode(tc, task)
    from mstep import httpenv
    return lambda task: httpenv.Episode(cfg["env"], task)


def prepare(cfg, run, pilot):
    run = Path(run)
    if (run / "manifest.json").exists():
        manifest = read_json(run / "manifest.json")
        if manifest["config"] != cfg or manifest["pilot"] != pilot:
            raise ValueError("Run exists with a different config; use a new run directory")
        return manifest
    if cfg["env"] == "textcraft":
        tc, tasks = tcmod.load(cfg["depths"], cfg["environment_seed"])
        tasks = select(tasks, cfg, pilot)
        manifest = {"config": cfg, "pilot": pilot, "conversation_start": tcmod.conversation_start(),
                    "recipe_order": tc.fingerprint(), "tasks": tasks,
                    "item_id_to_tag": {str(k): str(v) for k, v in tc.tree.item_id_to_tag.items()},
                    "python": sys.version}
    else:
        from mstep import httpenv
        idxs = cfg["task_indices"]
        if pilot:
            n = cfg["pilot"]["tasks"]
            idxs = idxs[::max(1, len(idxs) // n)][:n]
        tasks = [{"task_id": f'{cfg["env"]}_{i}', "data_idx": i, "recipe_depth": 0, "task_hash": digest([cfg["env"], i])}
                 for i in idxs]
        manifest = {"config": cfg, "pilot": pilot, "conversation_start": httpenv.conversation_start(cfg["env"]),
                    "tasks": tasks, "python": sys.version}
    manifest["manifest_hash"] = digest(manifest)
    write_json(run / "manifest.json", manifest)
    log(f"prepared {len(tasks)} {cfg['env']} tasks -> {run}")
    return manifest


def collect(cfg_path, run, pilot):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Run with PYTHONHASHSEED=0 (upstream prompt construction depends on set order)")
    cfg = read_json(cfg_path)
    run = Path(run)
    manifest = prepare(cfg, run, pilot)
    make = episode_factory(cfg, manifest)
    engine = Engine(cfg)
    A = cfg["attempts"]
    todo = [(task, a) for task in manifest["tasks"] for a in range(A)
            if not (run / "episodes" / f'{task["task_id"]}_attempt_{a:02d}.json').exists()]
    log(f"{len(todo)} episodes to collect")
    started = time.time()
    chunk = cfg.get("chunk_episodes", 1088)
    for c0 in range(0, len(todo), chunk):
        states = []
        for task, a in todo[c0:c0 + chunk]:
            ep = make(task)
            seed0 = cfg["generation_seed"] + task["data_idx"] * 100000 + a * 100
            states.append({"task": task, "attempt": a, "episode": ep, "rounds": [],
                           "messages": list(manifest["conversation_start"]) + [{"role": "user", "content": ep.initial_observation}],
                           "seed_fn": (lambda s0: (lambda r: s0 + r))(seed0), "max_rounds": cfg["max_rounds"]})

        def done(s):
            eid = f'{s["task"]["task_id"]}_attempt_{s["attempt"]:02d}'
            record = {"episode_id": eid, "task_id": s["task"]["task_id"], "task_hash": s["task"]["task_hash"],
                      "recipe_depth": s["task"]["recipe_depth"], "manifest_hash": manifest["manifest_hash"],
                      "attempt": s["attempt"], "rounds": s["rounds"], "success": s["success"],
                      "initial_observation": s["episode"].initial_observation,
                      "termination": s["termination"],
                      "total_generated_tokens": sum(len(r["generated_token_ids"]) for r in s["rounds"])}
            write_json(run / "episodes" / f"{eid}.json", record)
            append_jsonl(run / "progress.jsonl", {"episode_id": eid, "depth": record["recipe_depth"],
                                                  "success": s["success"], "rounds": len(s["rounds"]),
                                                  "seconds": round(time.time() - started, 1)})

        engine.run(states, keep_tokens=True, on_finish=done)
        restart_server(cfg)
        log(f"chunk {c0 // chunk + 1}: {min(c0 + chunk, len(todo))}/{len(todo)} episodes, {time.time() - started:.0f}s")
    log("collect done")


def mc_prefixes(episode, cfg):
    """Rounds t at which V(tau_t) is estimated: fixed rounds + milestone rounds (A-8d)."""
    n = len(episode["rounds"])
    ts = {t for t in cfg["mc"]["rounds"] if t <= n}
    if cfg["mc"].get("milestones"):
        ts |= {r["round"] for r in episode["rounds"] if r["observation"].startswith("Crafted ")}
    return sorted(ts)


def mc(run):
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Run with PYTHONHASHSEED=0")
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    cfg = manifest["config"]
    N = cfg["mc"]["n"]
    make = episode_factory(cfg, manifest)
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    paths = sorted((run / "episodes").glob("*.json"))
    # Optional scope restriction recorded next to the run (e.g. {"min_depth": 2}); written
    # after launch so the running runner script never has to be edited.
    scope = read_json(run / "mc_scope.json") if (run / "mc_scope.json").exists() else {}
    if "min_depth" in scope:
        keep = {t["task_id"] for t in manifest["tasks"] if t.get("recipe_depth", 0) >= scope["min_depth"]}
        paths = [p for p in paths if p.stem.rsplit("_attempt_", 1)[0] in keep]
    todo = [p for p in paths if not (run / "mc" / p.name).exists()]
    log(f"MC: {len(todo)} of {len(paths)} in-scope episodes need values (N={N}, scope={scope or 'all'})")
    engine = Engine(cfg)
    started = time.time()
    chunk = cfg["mc"].get("chunk_episodes", 256)
    for c0 in range(0, len(todo), chunk):
        states, results = [], {}
        for p in todo[c0:c0 + chunk]:
            e = read_json(p)
            eid, task = e["episode_id"], tasks[e["task_id"]]
            results[eid] = {"episode_id": eid, "n": N, "values": {}, "samples": {}}
            for t in mc_prefixes(e, cfg):
                prefix = e["rounds"][:t]
                if prefix[-1]["done"] or t == cfg["max_rounds"]:
                    # Terminal prefix: the value is the observed outcome, no sampling needed.
                    results[eid]["values"][str(t)] = float(prefix[-1]["reward"] == 1)
                    continue
                results[eid]["samples"][str(t)] = []
                for j in range(N):
                    ep = make(task)
                    msgs = list(manifest["conversation_start"]) + [{"role": "user", "content": ep.initial_observation}]
                    for r in prefix:
                        again = ep.step(r["response"])
                        if again["observation"] != r["observation"]:
                            if hasattr(ep, "close"):
                                ep.close()
                            raise ValueError(f"{eid}: prefix replay diverged at round {r['round']}")
                        msgs += [{"role": "assistant", "content": r["response"]}, {"role": "user", "content": r["observation"]}]
                    states.append({"eid": eid, "t": t, "j": j, "episode": ep, "messages": msgs,
                                   "rounds": [dict(round=i + 1) for i in range(t)],  # placeholders keep round numbering
                                   "seed_fn": (lambda e_, t_, j_: (lambda r: mc_seed(e_, t_, j_, r)))(eid, t, j),
                                   "max_rounds": cfg["max_rounds"]})

        def done(s):
            cont = s["rounds"][s["t"]:]
            results[s["eid"]]["samples"][str(s["t"])].append(
                {"j": s["j"], "success": s["success"], "len": len(cont), "termination": s["termination"],
                 "actions": [r["action"] for r in cont]})

        engine.run(states, keep_tokens=False, on_finish=done)
        restart_server(cfg)
        for eid, res in results.items():
            for t, samples in res["samples"].items():
                if len(samples) != N:
                    raise ValueError(f"{eid} t={t}: {len(samples)} of {N} MC samples")
                res["values"][t] = sum(x["success"] for x in samples) / N
            write_json(run / "mc" / f"{eid}.json", res)
        log(f"MC chunk {c0 // chunk + 1}: {min(c0 + chunk, len(todo))}/{len(todo)} episodes, "
            f"{len(states)} continuations, {time.time() - started:.0f}s")
    log("mc done")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--config", required=True); c.add_argument("--run", required=True)
    c.add_argument("--pilot", action="store_true")
    m = sub.add_parser("mc"); m.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.config, a.run, a.pilot)
    else:
        mc(a.run)


if __name__ == "__main__":
    main()
