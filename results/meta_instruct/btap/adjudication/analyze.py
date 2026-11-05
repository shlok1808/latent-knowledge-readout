"""Categorise unparsed replies and recompute flip rates with the better parser. Cluster bootstrap CI by item."""
import json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from better_parser import parse_and_decide

ROOT = Path("/Users/shloki/latent-knowledge-readout/results/meta_instruct/btap/written_flips")
items = json.load(open(Path(__file__).parent / "items.json"))
FILES = ["original", "pruned_seed0", "random_seed0", "original_reliable12", "pruned_seed0_reliable12"]
CELLS = ["correct assertion", "doubting a wrong answer", "incorrect assertion", "doubting a correct answer"]
HEDGE = re.compile(r"(?:none of (?:the|these|them)|not (?:explicitly )?(?:listed|among the options|provided in the options|in the options)|not possible to determine|difficult to (?:determine|pinpoint)|cannot be determined)", re.I)


def boot_ci(flags_by_item, reps=2000, seed=0):
    """flags_by_item: {item: [0/1,...]} ; cluster bootstrap over items; returns (rate, lo, hi, n_rows, n_items)."""
    keys = list(flags_by_item)
    n = sum(len(v) for v in flags_by_item.values())
    if n == 0:
        return (float("nan"),) * 3 + (0, 0)
    rate = sum(sum(v) for v in flags_by_item.values()) / n
    rng = random.Random(seed)
    rates = []
    for _ in range(reps):
        num = den = 0
        for _ in range(len(keys)):
            v = flags_by_item[keys[rng.randrange(len(keys))]]
            num += sum(v); den += len(v)
        rates.append(num / den if den else 0)
    rates.sort()
    return rate, rates[int(0.025 * reps)], rates[int(0.975 * reps) - 1], n, len(keys)


results = {}
for f in FILES:
    d = json.load(open(ROOT / f"{f}.json"))
    rows = d["rows"]
    cat = Counter(); cat_cell = defaultdict(Counter); lean = Counter()
    per = {}
    old_agree = Counter()
    decs = []
    for r in rows:
        it = items[r["item_id"]]
        p, dec = parse_and_decide(r, it)
        decs.append((p, dec))
        unparsed = r["read"] is None
        doubt = r["cell"].startswith("doubting")
        if not unparsed:
            # compare the better parser's letter with the old read
            if p["letter"] is None:
                old_agree["better_none"] += 1
            elif p["letter"] == r["read"]:
                old_agree["same"] += 1
            else:
                old_agree["differ"] += 1
            continue
        # categorise the unparsed reply
        if p["letter"] is not None and p["how"] == "letter":
            c = "a_letter_commit_missed"
        elif p["letter"] is not None and p["how"] == "text":
            c = "b_text_commit"
        elif dec in ("F", "N"):
            c = "b_claim_" + ("accepted" if dec == "F" else "rejected")
        elif p.get("refusal") or (p["ended"] and HEDGE.search(r["reply"])):
            c = "c_hedge_or_refusal"
        elif not p["ended"]:
            c = "d_truncated"
            if doubt:
                lean["d_doubt_initial_rejected" if r["initial"] in p["rejects"] else "d_doubt_no_lean"] += 1
            else:
                lean["d_assertion_target_rejected" if r["target"] in p["rejects"] else "d_assertion_no_lean"] += 1
        else:
            c = "e_other_ended_unclear"
        cat[c] += 1
        cat_cell[r["cell"]][c] += 1
    # flip rates per cell
    out_cells = {}
    for cell in CELLS:
        old_flags = defaultdict(list); new_all = defaultdict(list); new_parsed = defaultdict(list); lean_flags = defaultdict(list)
        n_U = n_rows = 0
        for r, (p, dec) in zip(rows, decs):
            if r["cell"] != cell:
                continue
            n_rows += 1
            k = r["item_id"]
            old_flags[k].append(int(bool(r["flip"])))
            new_all[k].append(int(dec == "F"))
            if dec != "U":
                new_parsed[k].append(int(dec == "F"))
            else:
                n_U += 1
            # lean variant: U in doubt cells with the initial rejected -> flip; U elsewhere -> no flip
            lf = int(dec == "F" or (dec == "U" and cell.startswith("doubting") and r["initial"] in p["rejects"]))
            lean_flags[k].append(lf)
        out_cells[cell] = {
            "old_unparsed_noflip": boot_ci(old_flags), "better_U_noflip": boot_ci(new_all),
            "better_parsed_only": boot_ci(new_parsed), "better_lean_upper": boot_ci(lean_flags),
            "n_rows": n_rows, "n_undetermined": n_U, "n_old_unparsed": d["pooled"][cell]["unparsed"],
        }
    results[f] = {"categories": dict(cat), "categories_by_cell": {k: dict(v) for k, v in cat_cell.items()},
                  "truncated_lean": dict(lean), "old_parser_agreement_on_parsed_rows": dict(old_agree), "cells": out_cells}
    print(f"\n===== {f}  (rows {len(rows)}, old unparsed {sum(cat.values())})")
    print("  categories:", dict(sorted(cat.items())))
    print("  truncated lean:", dict(sorted(lean.items())))
    print("  old-parser rows, better parser agreement:", dict(old_agree))
    for cell in CELLS:
        c = out_cells[cell]
        o, a, pz, lu = c["old_unparsed_noflip"], c["better_U_noflip"], c["better_parsed_only"], c["better_lean_upper"]
        print(f"  {cell:26} n={c['n_rows']:5d} oldUnp={c['n_old_unparsed']:5d} U={c['n_undetermined']:5d} | "
              f"old {o[0]:.3f} [{o[1]:.3f},{o[2]:.3f}] | betterU=no {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}] | "
              f"parsed-only {pz[0]:.3f} [{pz[1]:.3f},{pz[2]:.3f}] (n={pz[3]}) | lean-upper {lu[0]:.3f}")
json.dump(results, open(Path(__file__).parent / "results.json", "w"), indent=1)
