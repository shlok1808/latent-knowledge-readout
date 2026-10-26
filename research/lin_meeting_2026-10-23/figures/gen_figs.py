#!/usr/bin/env python3
"""Figures for the Dr. Lin meeting deck (2026-10-23).

Every number comes from the round 2 report:
~/colm-guide/gapscan/reports/Knowledge action divergence internals round 2.md
(fact-check log at its end). Run: .venv/bin/python gen_figs.py
"""
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA3AF", "axes.labelcolor": "#1a1a2e",
    "xtick.color": "#4B5563", "ytick.color": "#4B5563",
    "savefig.dpi": 220, "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

INK = "#1a1a2e"
MUTED = "#6B7280"
LINE = "#D1D5DB"
NAVY = "#0f3460"
# Okabe-Ito, one colour per fate
FATE = {
    "Readout": "#E69F00",
    "Bypass": "#56B4E9",
    "Not routed": "#0072B2",
    "Override": "#D55E00",
    "Transformed": "#009E73",
    "Erased": "#8C8C8C",
}


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), transparent=False, facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}.svg"), facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- fig 1: six fates
def fig_fates():
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#F3F4F6", ec=LINE, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2 + 2.2, title, ha="center", va="center",
                fontsize=16, weight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                fontsize=12.5, color=MUTED)

    def arrow(p, q, color=MUTED, ls="-", lw=1.8, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                                     color=color, lw=lw, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    box(2, 17, 18, 12, "Question", "“Capital of Australia?”")
    box(29, 13, 32, 20, "Middle layers", "right answer is decodable", fc="#EEF3FB")
    box(72, 15, 22, 16, "Decision point", "late layers", fc="white", ec=INK, lw=1.6)
    box(104, 17, 19, 12, "Answer", "“Sydney” ✗")
    arrow((21, 23), (28.4, 23))
    arrow((94.6, 23), (103.4, 23))
    arrow((61.6, 23), (71.4, 23), color=FATE["Bypass"], ls=(0, (4, 3)), lw=2.2)

    def tag(x, y, n, name, desc, ha="left"):
        c = FATE[name]
        ax.add_patch(plt.Circle((x, y), 1.6, color=c, zorder=5))
        ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=11,
                weight="bold", color="white", zorder=6)
        dx = 2.6 if ha == "left" else -2.6
        ax.text(x + dx, y + 1.0, name, ha=ha, va="center", fontsize=13.5,
                weight="bold", color=c)
        ax.text(x + dx, y - 2.1, desc, ha=ha, va="center", fontsize=12, color=INK)

    # inside the middle-layers box: transformed + erased
    tag(33, 8.5, 5, "Transformed", "re-encoded (e.g. sign-flipped)")
    tag(33, 2.0, 6, "Erased", "really gone (rare)")
    # bypass above the dashed link
    tag(57, 42, 2, "Bypass", "readable, but pushing it does nothing")
    # not routed: curve under
    arrow((60, 15), (76, 14.4), color=FATE["Not routed"], ls=(0, (1.5, 2.5)), lw=2.0, rad=0.35)
    tag(66, 3.0, 3, "Not routed", "works if injected, never carried there")
    # override from above into decision
    arrow((90, 41), (85, 31.4), color=FATE["Override"], lw=2.4)
    tag(92, 42, 4, "Override", "a competing signal wins")
    tag(104, 8.5, 1, "Readout", "scores lean right; cut-off wrong")
    save(fig, "fig_fates")


# ---------------------------------------------------------------- fig 2: evidence map
def fig_evidence_map():
    cols = ["Readout", "Bypass", "Not routed", "Override", "Transformed", "Erased", "Natural\nroute"]
    rows = [  # (paper, task, fate column, largest model, depth)
        ("2609.04582", "logic verification", "Readout", "8B"),
        ("2605.29126", "date variable", "Bypass", "9B"),
        ("2610.21708", "agent ask vs act", "Bypass", "14B"),
        ("2605.05715", "medical QA", "Bypass", "8B"),
        ("2608.29109", "unanswerable questions", "Not routed", "70B"),
        ("2509.23782", "multiple-choice QA (KAPPA)", "Not routed", "14B"),
        ("2607.14147", "jailbreak refusal", "Not routed", "14B"),
        ("2307.09476", "false few-shot labels", "Override", "20B"),
        ("2605.09314", "persuasion", "Override", "12B"),
        ("2609.37737", "prompt injection", "Override", "32B"),
        ("2604.19117", "factual sycophancy", "Override", "72B"),
        ("2311.15131", "instructed lying", "Transformed", "70B"),
        ("2607.00415", "authority sycophancy", "Erased", "9B"),
    ]
    natural = {"2604.19117": ("×", "null at 72B"), "2607.14147": ("~", "weak")}
    n = len(rows)
    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(n + 0.9, -1.1)
    ax.axis("off")
    for j, c in enumerate(cols):
        ax.text(j, -0.75, c, ha="center", va="bottom", fontsize=13, weight="bold",
                color=FATE.get(c, INK))
    for i, (pid, task, fate, size) in enumerate(rows):
        ax.axhline(i + 0.5, color="#EEF0F3", lw=1, zorder=0)
        ax.text(-0.62, i - 0.13, task, ha="right", va="center", fontsize=13, color=INK)
        ax.text(-0.62, i + 0.27, pid, ha="right", va="center", fontsize=9.5, color=MUTED,
                family="DejaVu Sans Mono")
        j = cols.index(fate)
        ax.scatter([j], [i], s=230, color=FATE[fate], zorder=3, edgecolor="white", lw=1.5)
        ax.text(j + 0.2, i, size, ha="left", va="center", fontsize=11, color=MUTED)
        if pid == "2607.00415":
            ax.text(j - 0.2, i, "contested", ha="right", va="center", fontsize=10.5,
                    color=MUTED, style="italic")
        if pid in natural:
            mark, why = natural[pid]
            ax.text(6, i, mark, ha="center", va="center", fontsize=18, color=INK, weight="bold")
            ax.text(6.22, i, why, ha="left", va="center", fontsize=10.5, color=MUTED)
    # this project row
    y = n + 0.35
    ax.add_patch(FancyBboxPatch((-0.42, y - 0.42), len(cols) - 0.16, 0.84,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#FFF4EC", ec="#D55E00", lw=1.6, zorder=1))
    ax.text(-0.62, y, "This project", ha="right", va="center", fontsize=14,
            weight="bold", color="#D55E00")
    for j, c in enumerate(cols):
        col = FATE.get(c, INK)
        ax.scatter([j], [y], s=150, color=col, zorder=3, edgecolor="white", lw=1.2)
    ax.text(len(cols) / 2 - 0.5, y + 0.75, "all checks on the same items  ·  9B–27B (70B if compute allows)",
            ha="center", va="top", fontsize=12, color="#D55E00")
    save(fig, "fig_evidence_map")


# ---------------------------------------------------------------- fig 3: margin check
def fig_margin():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    data = [("Qwen3-0.6B", 0.887, 0.964), ("Qwen3-8B", 0.980, 0.989)]
    for i, (name, m, p) in enumerate(data):
        y = 1 - i
        ax.plot([m, p], [y, y], color=LINE, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([m], [y], s=190, color="#E69F00", zorder=3, edgecolor="white", lw=1.5)
        ax.scatter([p], [y], s=190, color=NAVY, zorder=3, edgecolor="white", lw=1.5)
        ax.text(m, y + 0.2, f"{m:.3f}", ha="right" if i else "center", va="bottom",
                fontsize=12.5, color="#B07500")
        ax.text(p, y + 0.2, f"{p:.3f}", ha="left" if i else "center", va="bottom",
                fontsize=12.5, color=NAVY)
        gap = "+0.076" if i == 0 else "+0.008"
        ax.text((m + p) / 2 if i == 0 else 0.9555, y - 0.3, f"probe adds {gap}",
                ha="center", va="top", fontsize=13, color=INK, weight="bold")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([d[0] for d in data], fontsize=14)
    ax.set_ylim(-0.75, 1.65)
    ax.set_xlim(0.85, 1.0)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("AUC: separating right from wrong items")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EEF0F3")
    ax.set_axisbelow(True)
    ax.scatter([], [], s=120, color="#E69F00", label="output scores only (no training)")
    ax.scatter([], [], s=120, color=NAVY, label="trained probe")
    ax.legend(loc="upper left", frameon=False, fontsize=12.5, bbox_to_anchor=(0, 1.08),
              ncol=2, handletextpad=0.2, columnspacing=1.0)
    save(fig, "fig_margin")


# ---------------------------------------------------------------- fig 4: masking
def fig_masking():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), gridspec_kw={"wspace": 0.45})
    panels = [
        ("Sycophancy rate", ["3.1-70B", "3.3-70B"], [39, 3.5], ["39%", "3.5%"], 45),
        ("Effect of removing\nthe shared heads", ["before", "after"], [10.5, 27],
         ["+10.5 pp", "+27 pp"], 31),
    ]
    for ax, (title, labels, vals, txt, top) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=["#B9C2CF", "#D55E00"], width=0.58)
        for b, t in zip(bars, txt):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.025, t,
                    ha="center", va="bottom", fontsize=14, weight="bold", color=INK)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=14.5, weight="bold", color=INK, pad=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=0, labelsize=13)
    fig.text(0.5, -0.04, "Llama-3.1-70B $\\rightarrow$ Llama-3.3-70B (RLHF refresh), from 2604.19117",
             ha="center", fontsize=11.5, color=MUTED)
    save(fig, "fig_masking")


# ---------------------------------------------------------------- fig 5: angles
def fig_angles():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.35, 1.25)
    ax.set_ylim(-0.12, 1.12)
    # reference arc
    th = [math.radians(a) for a in range(0, 91)]
    ax.plot([math.cos(t) * 0.25 for t in th], [math.sin(t) * 0.25 for t in th], color=LINE, lw=1)
    ax.annotate("", xy=(1.0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=3, color=NAVY))
    ax.text(1.02, -0.02, "“knowing”\ndirection", ha="left", va="top", fontsize=13,
            color=NAVY, weight="bold")
    items = [
        (70, "KAPPA: knowledge vs prediction", "60–80°, mid layers", "#009E73"),
        (math.degrees(math.acos(0.16)), "fake entity: detect vs refuse", "cos 0.12–0.20", "#E69F00"),
        (math.degrees(math.acos(0.087)), "unanswerable: recognize vs refuse", "cos 0.087, up to 70B", "#D55E00"),
        (88, "probe vs causal direction", "≈88° = random", "#56B4E9"),
    ]
    for k, (deg, name, val, c) in enumerate(items):
        t = math.radians(deg)
        r = 0.95
        ax.annotate("", xy=(r * math.cos(t), r * math.sin(t)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=c))
    labels_y = [0.98, 0.80, 0.62, 0.44]
    for (deg, name, val, c), ly in zip(items, labels_y):
        ax.text(0.34, ly, name, ha="left", va="center", fontsize=12.5, color=c, weight="bold")
        ax.text(0.34, ly - 0.075, val, ha="left", va="center", fontsize=11.5, color=MUTED)
    ax.text(-0.02, 1.08, "“acting” directions", ha="left", va="center", fontsize=13,
            color=INK, weight="bold")
    ax.text(0.18, 0.06, "0° would mean\nknowing = acting", ha="left", va="bottom",
            fontsize=10.5, color=MUTED)
    save(fig, "fig_angles")


# ---------------------------------------------------------------- fig 6: plan
def fig_plan():
    fig, ax = plt.subplots(figsize=(12.6, 4.9))
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 49)
    ax.axis("off")

    def card(x, y, w, h, head, body, ec, fc="white", hc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=1.5))
        ax.text(x + 1.6, y + h - 2.6, head, ha="left", va="center", fontsize=13.5,
                weight="bold", color=hc or ec)
        ax.text(x + 1.6, y + h - 6.4, body, ha="left", va="top", fontsize=11.8, color=INK,
                linespacing=1.35)

    ax.text(1, 47, "Same facts, four kinds of pressure", fontsize=15, weight="bold", color=INK)
    conds = [
        ("1  Competing answer", "a tempting wrong option\nin the question itself"),
        ("2  User pushback", "“No, I'm sure it's Sydney”"),
        ("3  No single answer", "is “this is uncertain”\ndecodable while it answers?"),
        ("4  Forced to answer", "“you must answer” vs\nallowed to say “I don't know”"),
    ]
    for k, (h, b) in enumerate(conds):
        x = 1 + (k % 2) * 27
        y = 25 - (k // 2) * 18
        card(x, y, 25, 15.5, h, b, ec=NAVY if k < 2 else "#9AA3AF", hc=NAVY if k < 2 else MUTED)
    ax.text(27.5, 3.2, "start with 1 + 2; then 3 + 4", ha="center", fontsize=11.5, color=MUTED)

    ax.add_patch(FancyArrowPatch((55.5, 24), (61.5, 24), arrowstyle="-|>", mutation_scale=22,
                                 color=MUTED, lw=2))
    ax.text(63, 47, "Stage A: sort each failure, cheapest check first", fontsize=15,
            weight="bold", color=INK)
    steps = [("Readout", "output scores already right?", "recalibrate"),
             ("Bypass", "push both ways: any effect?", "gate the action"),
             ("Not routed", "inject works, natural use ≈ 0?", "realign / train route"),
             ("Override", "silence competitor: fixed?", "tune those heads"),
             ("Transformed", "old probe below chance?", "refit monitors"),
             ("Erased", "fresh probe at chance?", "data / retrieval")]
    for k, (f, q, fix) in enumerate(steps):
        y = 40 - k * 6.4
        ax.add_patch(plt.Circle((64.8, y), 1.5, color=FATE[f]))
        ax.text(64.8, y - 0.05, str(k + 1), ha="center", va="center", fontsize=10.5,
                color="white", weight="bold")
        ax.text(67.5, y, f, ha="left", va="center", fontsize=13, weight="bold", color=FATE[f])
        ax.text(80.5, y, q, ha="left", va="center", fontsize=12, color=INK)
        ax.text(110, y, "$\\rightarrow$ " + fix, ha="left", va="center", fontsize=12, color=MUTED)
    ax.add_patch(FancyBboxPatch((63, 0.6), 62, 4.6, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="#FFF4EC", ec="#D55E00", lw=1.4))
    ax.text(64.5, 2.9, "Stage B: apply the fix (and plain DPO), rerun Stage A $\\rightarrow$ removed or only hidden?",
            ha="left", va="center", fontsize=12.3, color="#B44A00", weight="bold")
    save(fig, "fig_plan")


# ---------------------------------------------------------------- fig 7: decodable vs used
def fig_used():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), gridspec_kw={"wspace": 0.55})
    ax = axes[0]
    labels = ["remove the probe's\ndirection", "remove the subspace\nthe model uses (DAS)"]
    vals = [0.6, 42]
    bars = ax.barh(labels, vals, color=["#B9C2CF", "#D55E00"], height=0.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 50)
    ax.set_xlabel("accuracy drop (points)")
    for b, t in zip(bars, ["\u22120.6 (same as a random direction)", "\u221242 (42% to 0%)"]):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, t, va="center", fontsize=12.5,
                color=INK, weight="bold")
    ax.set_title("Probe reads it almost perfectly (R\u00b2 = 0.981)\nbut the model doesn't use that direction",
                 fontsize=13.5, weight="bold", color=INK, loc="left", pad=10)
    ax.text(0, -0.32, "2605.29126 (Gemma-2-2B, a date variable)", transform=ax.transAxes,
            fontsize=10.5, color=MUTED)
    ax = axes[1]
    bars = ax.barh(["decodable (R)", "used (U)"], [0.98, 0.15], color=[NAVY, "#D55E00"], height=0.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("score (0 to 1)")
    for b, t in zip(bars, ["0.98", "0.15"]):
        ax.text(b.get_width() + 0.02, b.get_y() + b.get_height() / 2, t, va="center", fontsize=13,
                color=INK, weight="bold")
    ax.set_title("After fine-tuning: still readable,\nbarely used",
                 fontsize=13.5, weight="bold", color=INK, loc="left", pad=10)
    ax.text(0, -0.32, "2610.05292 (Qwen \u22644B): 'used' = does the decision change when only that fact changes?",
            transform=ax.transAxes, fontsize=10.5, color=MUTED)
    for a in axes:
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0, labelsize=13)
        a.grid(axis="x", color="#EEF0F3")
        a.set_axisbelow(True)
    save(fig, "fig_used")


if __name__ == "__main__":
    # explanatory figures and the angle figure live in gen_story_figs.py
    fig_evidence_map()
    fig_margin()
    fig_masking()
    fig_used()
    print("done")
