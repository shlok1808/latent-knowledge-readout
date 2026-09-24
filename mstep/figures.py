"""Figures for one results directory (PNG, light surface).

  python -m mstep.figures --out results/multistep/textcraft_qwen3-1.7b

Main figure (the CEB-style motivation plot): within-task AUC vs test round, one
line per training strategy, columns = labels, rows = naive vs same-episode cohort.
Transfer heatmaps: train stage x test stage, naive and cohort side by side.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Reference palette (dataviz skill), fixed categorical order, light mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("auc", ["#c0392b", "#f0efec", "#1c5cab"])

LABEL_NAMES = {"final_bad": "FINAL (episode fails)", "mc_bad": "MC value < 0.5",
               "valid_bad": "IMMEDIATE: action invalid", "noprogress_bad": "IMMEDIATE: no recipe progress"}
STRATEGIES = [("train_early", "trained early (r1-2)", "-", "o"), ("train_mid", "trained mid (r3-5)", "-", "s"),
              ("train_late", "trained late (r6+)", "-", "^"), ("train_pooled", "trained all rounds", "-", "D"),
              ("train_same_round", "trained same round", ":", "v"), ("text_pooled", "TF-IDF+surface, all rounds", "--", "x")]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def round_curves(out, layer, metric="within_auc", fname=None):
    cur = pd.read_csv(out / "phase3_round_curves.csv")
    cur = cur[((cur.layer == layer) & (cur.probe == "logreg")) | (cur.strategy == "text_pooled")]
    labels = [l for l in LABEL_NAMES if l in set(cur.label)]
    fig, axes = plt.subplots(2, len(labels), figsize=(3.3 * len(labels), 6.2), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for i, pop in enumerate(("naive", "cohort")):
        for j, lab in enumerate(labels):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0.5, color=AXIS, linewidth=1)
            for k, (key, name, ls, mk) in enumerate(STRATEGIES):
                d = cur[(cur.population == pop) & (cur.label == lab) & (cur.strategy == key)].sort_values("t")
                if d.empty:
                    continue
                tt = d.t.to_numpy()
                y = d[metric].to_numpy(dtype=float)
                weak = d["underpowered"].to_numpy(dtype=bool)
                ax.plot(d.t, y, color=SERIES[k], linestyle=ls, linewidth=2, label=name)
                ax.scatter(tt[~weak], y[~weak], color=SERIES[k], marker=mk, s=22, zorder=3, edgecolor=SURFACE, linewidth=0.8)
                ax.scatter(tt[weak], y[weak], facecolor=SURFACE if mk != "x" else SERIES[k], edgecolor=SERIES[k], marker=mk, s=22, zorder=3)
            if i == 0:
                ax.set_title(LABEL_NAMES[lab], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(("naive: alive at t" if pop == "naive" else "same episodes (cohort)") +
                              f"\n{metric.replace('_', ' ')}", fontsize=9, color=INK)
            if i == 1:
                ax.set_xlabel("test round t", fontsize=9, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(f"Probe AUC by test round, layer {layer} (open markers: <15 mixed tasks)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / (fname or f"fig_round_curves_L{layer}_{metric}.png"), dpi=160)
    plt.close(fig)


def transfer_heatmaps(out, layer, scheme, metric="within_auc", src="phase3_transfer.csv"):
    mat = pd.read_csv(out / src)
    mat = mat[(mat.layer == layer) & (mat.probe == "logreg") & (mat.scheme == scheme)]
    if mat.empty:
        return
    labels = [l for l in LABEL_NAMES if l in set(mat.label)]
    fig, axes = plt.subplots(len(labels), 2, figsize=(7.4, 2.5 * len(labels)), facecolor=SURFACE, squeeze=False)
    for i, lab in enumerate(labels):
        for j, pop in enumerate(("naive", "cohort")):
            ax = axes[i, j]
            d = mat[(mat.label == lab) & (mat.population == pop)]
            tests = [s for s in ("early", "mid", "late", "early@2", "mid@5", "late@10", "m1", "m2", "m3") if s in set(d.test)]
            trains = tests + ["pooled"]
            grid = np.full((len(trains), len(tests)), np.nan)
            weak = np.zeros_like(grid, bool)
            for a, tr in enumerate(trains):
                for b, te in enumerate(tests):
                    c = d[(d.train == tr) & (d.test == te)]
                    if len(c):
                        grid[a, b] = c[metric].iloc[0]
                        weak[a, b] = bool(c["underpowered"].iloc[0])
            ax.imshow(grid, cmap=DIVERGING, vmin=0.3, vmax=0.7, aspect="auto")
            for a in range(len(trains)):
                for b in range(len(tests)):
                    if np.isfinite(grid[a, b]):
                        ax.text(b, a, f"{grid[a, b]:.2f}" + ("*" if weak[a, b] else ""), ha="center", va="center",
                                fontsize=8, color=INK)
            ax.set_xticks(range(len(tests)), tests, fontsize=8, color=INK2)
            ax.set_yticks(range(len(trains)), trains, fontsize=8, color=INK2)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title("naive (alive in stage)" if pop == "naive" else "same episodes (cohort)", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{LABEL_NAMES[lab]}\ntrain stage", fontsize=8, color=INK)
            if i == len(labels) - 1:
                ax.set_xlabel("test stage", fontsize=8, color=INK)
    fig.suptitle(f"{scheme} stages, layer {layer}, {metric.replace('_', ' ')} (* <15 mixed tasks; colour centred at 0.5)",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"fig_transfer_{scheme}_L{layer}_{metric}.png", dpi=160)
    plt.close(fig)


def layer_sweep(out):
    cells = pd.read_csv(out / "phase2_cells.csv")
    base = pd.read_csv(out / "phase2_baselines.csv")
    labels = [l for l in LABEL_NAMES if l in set(cells.label)]
    ts = sorted(cells.t.unique())
    fig, axes = plt.subplots(len(labels), 2, figsize=(9, 2.6 * len(labels)), sharex=True, facecolor=SURFACE, squeeze=False)
    for i, lab in enumerate(labels):
        for j, inp in enumerate(("S1", "S2")):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0.5, color=AXIS, linewidth=1)
            for k, t in enumerate(ts):
                d = cells[(cells.label == lab) & (cells.input == inp) & (cells.probe == "logreg") & (cells.t == t)].sort_values("layer")
                if d.empty:
                    continue
                ax.plot(d.layer, d.within_auc, color=SERIES[k], linewidth=2, label=f"t={t}")
                b = base[(base.label == lab) & (base.t == t) & (base.baseline == "tfidf_surface")]
                if len(b):
                    ax.axhline(b.within_auc.iloc[0], color=SERIES[k], linestyle="--", linewidth=1)
            if i == 0:
                ax.set_title(("S1 last token at round t" if inp == "S1" else "S2 mean-pool rounds 1..t"), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{LABEL_NAMES[lab]}\nwithin-task AUC", fontsize=8, color=INK)
            if i == len(labels) - 1:
                ax.set_xlabel("layer (0 = embeddings)", fontsize=8, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(ts), fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle("Phase 2 layer sweep, L2 logistic (dashed: TF-IDF+surface at same t)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "fig_phase2_layer_sweep.png", dpi=160)
    plt.close(fig)


def population(out):
    p = pd.read_csv(out / "population_by_round.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), facecolor=SURFACE)
    for ax, col, name in ((axes[0], "fail_share", "failure share among alive episodes"),
                          (axes[1], "mixed_tasks", "mixed-outcome tasks alive")):
        style(ax)
        ax.plot(p.t, p[col], color=SERIES[0], linewidth=2, marker="o", markersize=4)
        ax.set_title(name, fontsize=9, color=INK)
        ax.set_xlabel("round t", fontsize=8, color=INK)
    axes[1].axhline(15, color=SERIES[7], linestyle="--", linewidth=1)
    axes[1].text(p.t.max(), 15, " 15-task floor", color=INK2, fontsize=7, va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(out / "fig_population_by_round.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    layers = json.loads((out / "phase3_layers.json").read_text())["layers"] if (out / "phase3_layers.json").exists() else []
    if (out / "phase2_cells.csv").exists():
        layer_sweep(out)
    if (out / "population_by_round.csv").exists():
        population(out)
    for layer in layers:
        if (out / "phase3_round_curves.csv").exists():
            round_curves(out, layer)
            round_curves(out, layer, metric="pooled_auc")
            round_curves(out, layer, metric="ece")
        if (out / "phase3_transfer.csv").exists():
            if (out / "phase3_transfer_laststep.csv").exists():
                transfer_heatmaps(out, layer, "laststep", src="phase3_transfer_laststep.csv")
                transfer_heatmaps(out, layer, "laststep", metric="pooled_auc", src="phase3_transfer_laststep.csv")
            for scheme in ("round", "milestone"):
                transfer_heatmaps(out, layer, scheme)
                transfer_heatmaps(out, layer, scheme, metric="pooled_auc")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
