#!/usr/bin/env python3
"""Fetch full texts, render figure pages, build per-paper manifests. Read-only on downloads."""
import json, os, re, subprocess, sys
R = os.path.dirname(os.path.abspath(__file__))
P = [
 ("u1-openreview-ZP7VzpXmKR", "https://openreview.net/pdf?id=ZP7VzpXmKR"),
 ("u2-subtrackpp-2502.01586", "https://arxiv.org/pdf/2502.01586v1"),
 ("u3-geometry-of-divergence-2608.30650", "https://arxiv.org/pdf/2608.30650"),
 ("u4-role-learning-1910.09113", "https://arxiv.org/pdf/1910.09113"),
 ("c5-manifold-steering-2605.05115", "https://arxiv.org/pdf/2605.05115"),
 ("c6-kappa-2509.23782", "https://arxiv.org/pdf/2509.23782"),
 ("c7-readout-mediator-angle-2605.29126", "https://arxiv.org/pdf/2605.29126"),
 ("c8-represented-used-2610.05292", "https://arxiv.org/pdf/2610.05292"),
 ("c9-lagged-coupling-2609.01048", "https://arxiv.org/pdf/2609.01048"),
 ("c10-curved-inference-ii-2608.24037", "https://arxiv.org/pdf/2608.24037"),
 ("c11-trajectory-signatures-deception-acl2026", "https://aclanthology.org/2026.acl-long.1582.pdf"),
 ("c12-tigges-circuits-across-training-2407.10827", "https://arxiv.org/pdf/2407.10827"),
 ("c13-cka-reliability-2210.16156", "https://arxiv.org/pdf/2210.16156"),
 ("c14-mags-2605.21770", "https://arxiv.org/pdf/2605.21770"),
 ("c15-confidence-manifold-2602.08159", "https://arxiv.org/pdf/2602.08159"),
 ("u5-user-2608.27510", "https://arxiv.org/pdf/2608.27510"),
 ("w2-readout-subspace-grassmann-2605.09011", "https://arxiv.org/pdf/2605.09011"),
 ("w2-layerwise-grassmann-dynamics-2605.12714", "https://arxiv.org/pdf/2605.12714"),
 ("w2-shape-of-beliefs-2602.02315", "https://arxiv.org/pdf/2602.02315"),
 ("w2-info-geometry-shared-2609.11063", "https://arxiv.org/pdf/2609.11063"),
 ("w2-fishback-2605.17231", "https://arxiv.org/pdf/2605.17231"),
 ("w2-nonlinear-interventions-2605.14749", "https://arxiv.org/pdf/2605.14749"),
 ("p2-sycophancy-manifold-posttraining-2608.01835", "https://arxiv.org/pdf/2608.01835"),
 ("p2-shared-compliance-subspace-2601.11563", "https://arxiv.org/pdf/2601.11563"),
 ("p2-separable-sycophancy-modes-2607.20146", "https://arxiv.org/pdf/2607.20146"),
]
cap_re = re.compile(r"^\s*(Figure|Fig\.|Table)\s*(\d+)\s*[:.]", re.M)
summary = []
for slug, url in P:
    d = os.path.join(R, "papers", slug); os.makedirs(d, exist_ok=True)
    pdf = os.path.join(d, "paper.pdf")
    if not os.path.exists(pdf) or os.path.getsize(pdf) < 10000:
        subprocess.run(["curl", "-sL", "--max-time", "90", "-A", "Mozilla/5.0", "-o", pdf, url])
    ok = os.path.exists(pdf) and open(pdf, "rb").read(5) == b"%PDF-"
    rec = {"slug": slug, "url": url, "pdf_ok": ok}
    if ok:
        subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, os.path.join(d, "paper.txt")])
        txt = open(os.path.join(d, "paper.txt"), encoding="utf-8", errors="ignore").read()
        pages = txt.split("\f")
        rec["pages"] = len([p for p in pages if p.strip()])
        rec["chars"] = len(txt)
        rec["title_guess"] = " ".join(txt.strip().splitlines()[:3])[:160]
        caps, figpages = [], set()
        for i, pg in enumerate(pages, 1):
            for m in cap_re.finditer(pg):
                kind = "Table" if m.group(1) == "Table" else "Figure"
                key = f"{kind} {m.group(2)}"
                if key not in [c["id"] for c in caps]:
                    line = pg[m.start():m.start() + 200].replace("\n", " ")
                    caps.append({"id": key, "page": i, "caption_start": line})
                    figpages.add(i)
        rec["figures_tables"] = caps
        rec["n_figures_tables"] = len(caps)
        imgd = os.path.join(d, "pages"); os.makedirs(imgd, exist_ok=True)
        for pg in sorted(figpages):
            out = os.path.join(imgd, f"p{pg:02d}")
            if not os.path.exists(out + ".png"):
                subprocess.run(["pdftoppm", "-f", str(pg), "-l", str(pg), "-r", "110", "-png", "-singlefile", pdf, out])
        heads = [l.strip() for l in txt.splitlines() if re.match(r"^\s*(\d+(\.\d+)*|[A-Z](\.\d+)*)\s+[A-Z][A-Za-z].{2,80}$", l) and len(l.strip()) < 90]
        rec["section_heading_candidates"] = heads[:80]
        rec["looks_abstract_only"] = rec["chars"] < 15000
        has_refs = re.search(r"\n\s*References\s*\n", txt) is not None
        rec["has_references"] = has_refs
    json.dump(rec, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    summary.append(rec)
json.dump(summary, open(os.path.join(R, "papers", "_manifest_all.json"), "w"), indent=1)
for r in summary:
    print(f'{r["slug"][:48]:48s} ok={r["pdf_ok"]} pages={r.get("pages")} chars={r.get("chars")} figs={r.get("n_figures_tables")} short={r.get("looks_abstract_only")} | {r.get("title_guess","")[:70]}')
