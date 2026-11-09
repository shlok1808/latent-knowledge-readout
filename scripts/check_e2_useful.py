#!/usr/bin/env python
"""Post-check of results/meta_instruct/btap/e2/useful_margin.jsonl (code review, 2026-11-09): the margin loss reads the
FIRST reply position, so keep only useful-updating rows whose written reply opens with the gold letter ("(X", "X");
and require 64 rows in each kind x turn cell (warn otherwise). Rewrites the file in place, keeps a .raw copy."""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

p = Path("results/meta_instruct/btap/e2/useful_margin.jsonl")
rows = [json.loads(l) for l in open(p)]
shutil.copy(p, p.with_suffix(".raw.jsonl"))
def opens_with_gold(r):
    t = r["reply_text"].lstrip()
    return t.startswith("(" + r["gold"]) or (t[:1] == r["gold"] and t[1:2] in ("", ")", ".", ":", "<"))
keep = [r for r in rows if opens_with_gold(r)]
cells = Counter(f"{r['kind']}/{r['turn']}" for r in keep)
with open(p, "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
print(f"useful rows {len(rows)} -> {len(keep)} (dropped {len(rows) - len(keep)} not opening with the gold letter); cells {dict(cells)}")
short = {c: n for c, n in cells.items() if n < 64}
for c in ("suggestion/single", "suggestion/multi", "doubt/single", "doubt/multi"):
    if cells.get(c, 0) < 64:
        print(f"WARNING: cell {c} has {cells.get(c, 0)} rows (< 64)")
sys.exit(0)
