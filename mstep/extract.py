"""Replay saved tokens through HF and store last-token residuals at every layer.

Runs in the main venv. One row per (episode, round). Two token sites per row
(ASSUMPTIONS A-7): `last` = last content token of the generation (primary),
`action` = final token of the parsed Action span (stage-2/Ruan site; falls back
to `last` when no action parsed). Layer 0 = embedding output, layer i = output
of decoder block i (pre final RMSNorm), matching stage-2's hook convention.

Output (runs/<run>/acts/): `{site}.npy` float16 memmap [rows, n_layers+1, d],
`index.json` row -> (episode_id, round), `extraction.json` metadata.
Resumable: rows are filled in episode order and a progress marker records the
last completed episode.

  python -m mstep.extract --run runs/ms-tc-1p7b [--token-budget 32768]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from failure_probes.common import read_json, write_json

SITES = ("last", "action")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_index(run):
    rows = []
    for p in sorted((run / "episodes").glob("*.json")):
        e = read_json(p)
        for r in e["rounds"]:
            rows.append({"episode_id": e["episode_id"], "round": r["round"],
                         "len": len(r["prompt_token_ids"]) + len(r["generated_token_ids"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--episode-pass", action="store_true",
                    help="one forward pass per episode when each turn's prompt is an exact prefix of the next")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = read_json(run / "manifest.json")["config"]
    out = run / "acts"
    out.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["model_revision"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    n_layers, d = len(model.model.layers), model.config.hidden_size

    if (out / "index.json").exists():
        index = read_json(out / "index.json")
    else:
        index = build_index(run)
        write_json(out / "index.json", index)
    N = len(index)
    mode = "r+" if (out / "last.npy").exists() else "w+"
    arrays = {s: np.lib.format.open_memmap(out / f"{s}.npy", mode=mode, dtype=np.float16, shape=(N, n_layers + 1, d))
              for s in SITES}
    done_path = out / "done_rows.npy"
    done = np.load(done_path) if done_path.exists() else np.zeros(N, dtype=bool)
    log(f"{N} rows, {int(done.sum())} already extracted, layers={n_layers}, d={d}")

    # Hooks write the gathered positions of the current batch into `grab`.
    grab, positions = {}, {}
    def hook(layer):
        def fn(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            grab[layer] = h[positions["b"], positions["p"]].float().cpu()  # [2B, d]
        return fn
    handles = [model.model.embed_tokens.register_forward_hook(hook(0))]
    handles += [blk.register_forward_hook(hook(i + 1)) for i, blk in enumerate(model.model.layers)]

    started, n_done = time.time(), 0
    if a.episode_pass:
        # Group pending rows by episode; an episode qualifies if every turn's prompt+generation
        # is an exact prefix of the next turn's prompt (incremental prompts, rollout.py).
        by_ep = {}
        for i in range(N):
            if not done[i]:
                by_ep.setdefault(index[i]["episode_id"], []).append(i)
        jobs, fallback = [], 0
        for eid, rows in by_ep.items():
            e = read_json(run / "episodes" / f"{eid}.json")
            rr = [e["rounds"][index[i]["round"] - 1] for i in rows]
            seqs = [r["prompt_token_ids"] + r["generated_token_ids"] for r in rr]
            if all(b["prompt_token_ids"][:len(sa)] == sa for sa, b in zip(seqs, rr[1:])):
                pos = []
                for r in rr:
                    base = len(r["prompt_token_ids"])
                    pos.append((base + r["last_content_token"],
                                base + (r["action_token_indices"][-1] if r["action_token_indices"] else r["last_content_token"])))
                jobs.append((len(seqs[-1]), seqs[-1], rows, pos))
            else:
                fallback += len(rows)
        log(f"episode-pass: {len(jobs)} episodes in one pass each; {fallback} rows fall back to per-turn replay")
        jobs.sort(key=lambda j: j[0])
        b0 = 0
        while b0 < len(jobs):
            b1 = b0 + 1
            while b1 < len(jobs) and (b1 - b0 + 1) * jobs[b1][0] <= a.token_budget:
                b1 += 1
            batch = jobs[b0:b1]
            L = max(j[0] for j in batch)
            ids = torch.zeros((len(batch), L), dtype=torch.long); mask = torch.zeros((len(batch), L), dtype=torch.long)
            bi, pi, rows_all = [], [], []
            for k, (n, seq, rows, pos) in enumerate(batch):
                ids[k, :n] = torch.tensor(seq); mask[k, :n] = 1
                for row, (pl, pa) in zip(rows, pos):
                    bi += [k, k]; pi += [pl, pa]; rows_all.append(row)
            positions["b"] = torch.tensor(bi, device="cuda"); positions["p"] = torch.tensor(pi, device="cuda")
            with torch.inference_mode():
                model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
            stacked = torch.stack([grab[l] for l in range(n_layers + 1)], dim=1)
            if not torch.isfinite(stacked).all():
                raise ValueError("non-finite hidden state")
            stacked = stacked.to(torch.float16).numpy()
            rows_np = np.array(rows_all)
            arrays["last"][rows_np] = stacked[0::2]; arrays["action"][rows_np] = stacked[1::2]
            done[rows_np] = True
            n_done += len(rows_all); b0 = b1
            if (b0 % 50) < len(batch):
                for arr in arrays.values():
                    arr.flush()
                np.save(done_path, done)
                log(f"episode-pass: {b0}/{len(jobs)} episodes, {int(done.sum())}/{N} rows, {n_done / (time.time() - started):.1f} rows/s")
        for arr in arrays.values():
            arr.flush()
        np.save(done_path, done)

    # Length-sorted batches under a token budget; episodes are read lazily per batch.
    order = sorted((i for i in range(N) if not done[i]), key=lambda i: index[i]["len"])
    cache = {}
    def record(i):
        eid, rnd = index[i]["episode_id"], index[i]["round"]
        if eid not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[eid] = read_json(run / "episodes" / f"{eid}.json")
        return cache[eid]["rounds"][rnd - 1]

    b0 = 0
    while b0 < len(order):
        L = index[order[b0]]["len"]
        b1 = b0 + 1
        while b1 < len(order) and (b1 - b0 + 1) * index[order[b1]]["len"] <= a.token_budget:
            b1 += 1
        batch = order[b0:b1]
        L = max(index[i]["len"] for i in batch)
        ids = torch.full((len(batch), L), 0, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        bi, pi = [], []
        for k, i in enumerate(batch):
            r = record(i)
            seq = r["prompt_token_ids"] + r["generated_token_ids"]
            ids[k, :len(seq)] = torch.tensor(seq)
            mask[k, :len(seq)] = 1
            base = len(r["prompt_token_ids"])
            last = base + r["last_content_token"]
            act = base + (r["action_token_indices"][-1] if r["action_token_indices"] else r["last_content_token"])
            bi += [k, k]; pi += [last, act]
        positions["b"] = torch.tensor(bi, device="cuda"); positions["p"] = torch.tensor(pi, device="cuda")
        with torch.inference_mode():
            # Right padding + causal attention: real positions never attend to padding.
            model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        stacked = torch.stack([grab[l] for l in range(n_layers + 1)], dim=1)  # [2B, L+1, d]
        if not torch.isfinite(stacked).all():
            raise ValueError("non-finite hidden state")
        stacked = stacked.to(torch.float16).numpy()
        rows = np.array(batch)
        arrays["last"][rows] = stacked[0::2]
        arrays["action"][rows] = stacked[1::2]
        done[rows] = True
        n_done += len(batch)
        b0 = b1
        if n_done % 2000 < len(batch):
            for arr in arrays.values():
                arr.flush()
            np.save(done_path, done)
            rate = n_done / (time.time() - started)
            log(f"{int(done.sum())}/{N} rows, {rate:.1f} rows/s, eta {(len(order) - n_done) / rate / 60:.0f} min")
    for arr in arrays.values():
        arr.flush()
    np.save(done_path, done)
    for h in handles:
        h.remove()

    # Replay determinism check on a few rows: batch-1 recompute vs stored.
    check = []
    for i in list(range(0, N, max(1, N // 5)))[:5]:
        r = record(i)
        seq = torch.tensor([r["prompt_token_ids"] + r["generated_token_ids"]]).cuda()
        pos = len(r["prompt_token_ids"]) + r["last_content_token"]
        with torch.inference_mode():
            hs = model.model(input_ids=seq, use_cache=False, output_hidden_states=True).hidden_states
        # hidden_states[-1] is post-norm in HF; compare pre-norm layers only.
        ref = torch.stack([h[0, pos].float().cpu() for h in hs[:-1]]).numpy()
        got = arrays["last"][i, :n_layers].astype(np.float32)
        check.append(float(np.max(np.abs(ref - got) / (np.abs(ref).max(axis=1, keepdims=True) + 1e-6))))
    write_json(out / "extraction.json", {"rows": N, "n_layers": n_layers, "hidden_size": d, "sites": list(SITES),
                                         "dtype_stored": "float16", "model": cfg["model"],
                                         "model_revision": cfg["model_revision"],
                                         "layer_convention": "0=embeddings, i=decoder block i output (pre final norm)",
                                         "batch_vs_single_max_rel_err": check,
                                         "all_rows_done": bool(done.all())})
    log(f"extraction done; batch-vs-single max rel err {max(check):.4f}")


if __name__ == "__main__":
    main()
