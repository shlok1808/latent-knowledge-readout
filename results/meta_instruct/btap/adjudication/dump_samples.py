"""Dump seeded random samples of UNPARSED replies per file for hand labelling."""
import json, random, sys
from pathlib import Path

ROOT = Path("/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/written_flips")
OUT = Path("/private/tmp/claude-501/-Users-shloki--claude-spares-spare-7Qip7k/ab0e02ee-90de-4560-9187-a85503a906ac/scratchpad/adjudicate")
items = json.load(open(OUT / "items.json"))

FILES = ["original", "pruned_seed0", "random_seed0", "original_reliable12", "pruned_seed0_reliable12"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150

for f in FILES:
    d = json.load(open(ROOT / f"{f}.json"))
    un = [(i, r) for i, r in enumerate(d["rows"]) if r["read"] is None]
    rng = random.Random(0)
    samp = un if len(un) <= N else rng.sample(un, N)
    with open(OUT / f"sample_{f}.txt", "w") as fh:
        for i, r in samp:
            it = items[r["item_id"]]
            opts = " | ".join(f"{l}:{t}" for l, t in it["options"])
            fh.write(f"### idx={i} cell={r['cell']} initial={r['initial']} target={r['target']} gold={it['answer']} letters={it['letters']}\n")
            fh.write(f"OPTS: {opts}\n")
            fh.write("REPLY: " + r["reply"].replace("\n", "\\n") + "\n\n")
    print(f, "unparsed", len(un), "sampled", len(samp))
