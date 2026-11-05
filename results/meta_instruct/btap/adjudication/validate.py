"""Validate the better parser against the hand labels; print disagreements on the F/N/U decision."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from better_parser import parse_and_decide, decide
from hand_labels import LABELS

ROOT = Path("/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/written_flips")
items = json.load(open(Path(__file__).parent / "items.json"))
VERBOSE = "-v" in sys.argv

tot = agree = 0
conf = {}
for f, labs in LABELS.items():
    rows = json.load(open(ROOT / f"{f}.json"))["rows"]
    fa = 0
    for idx, (lab, cat) in labs.items():
        r = rows[idx]
        p, d = parse_and_decide(r, items[r["item_id"]])
        if lab in "ABCDE":
            hd = decide(r["cell"], r["initial"], r["target"], {"letter": lab, "rejects": set(), "accepts_claim": False, "rejects_claim": False})
        elif lab == "N":
            hd = "N"
        else:
            hd = "U"
        ok = d == hd
        conf[(hd, d)] = conf.get((hd, d), 0) + 1
        tot += 1; agree += ok; fa += ok
        if not ok and VERBOSE:
            print(f"[{f} idx={idx}] hand={lab}/{cat} hand_dec={hd} | parser={p['letter']}({p['how']}) rej={sorted(p['rejects'])} rc={p['rejects_claim']} ac={p['accepts_claim']} dec={d} | {r['cell']} init={r['initial']} tgt={r['target']}")
            print("    " + r["reply"][:200].replace("\n", "\\n"))
    print(f"== {f}: decision agreement {fa}/{len(labs)} = {fa/len(labs):.3f}")
print(f"TOTAL decision (F/N/U) agreement {agree}/{tot} = {agree/tot:.3f}")
print("confusion (hand, parser):", dict(sorted(conf.items())))
