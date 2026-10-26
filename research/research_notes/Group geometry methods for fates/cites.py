#!/usr/bin/env python3
"""Count arXiv ids cited across our papers' reference lists; list those cited by >=2 papers and not in our set."""
import os, re, json, collections
R = os.path.dirname(os.path.abspath(__file__))
ours, cnt = set(), collections.Counter()
for slug in sorted(os.listdir(os.path.join(R, "papers"))):
    m = re.search(r"(\d{4}\.\d{4,5})", slug)
    if m: ours.add(m.group(1))
for slug in sorted(os.listdir(os.path.join(R, "papers"))):
    f = os.path.join(R, "papers", slug, "paper.txt")
    if not os.path.exists(f): continue
    t = open(f, encoding="utf-8", errors="ignore").read()
    i = t.rfind("References"); refs = t[i:] if i > 0 else t
    for a in set(re.findall(r"(?:arXiv[:\s]*|abs/|pdf/)(\d{4}\.\d{4,5})", refs)): cnt[a] += 1
out = [{"arxiv": a, "cited_by_n_of_ours": n} for a, n in cnt.most_common() if n >= 2 and a not in ours]
print(json.dumps(out, indent=1))
