#!/usr/bin/env python3
"""Usage: quote_check.py <slug>  -> checks every quote in reading/<slug>.json against papers/<slug>/paper.txt"""
import json, os, re, sys
R = os.path.dirname(os.path.abspath(__file__))
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower().replace("ﬁ","fi").replace("ﬂ","fl"))
slug = sys.argv[1]
txt = norm(open(os.path.join(R, "papers", slug, "paper.txt"), encoding="utf-8", errors="ignore").read())
data = json.load(open(os.path.join(R, "reading", slug + ".json")))
quotes = []
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("quote", "limitations_quote", "future_work_quote") and isinstance(v, str) and v.strip() and not v.strip().lower().startswith("none stated"):
                quotes.append((path + "/" + k, v))
            else: walk(v, path + "/" + k)
    elif isinstance(o, list):
        for i, v in enumerate(o): walk(v, f"{path}[{i}]")
walk(data)
bad = [(p, q) for p, q in quotes if norm(q)[:120] not in txt]
print(json.dumps({"slug": slug, "n_quotes": len(quotes), "n_failed": len(bad), "failed": [{"where": p, "quote": q[:200]} for p, q in bad]}, indent=1))
