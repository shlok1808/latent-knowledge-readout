#!/usr/bin/env python3
"""Main-deck figures for the weekly update (26 Oct). Drawn at slide size so text
stays >= 15 pt on a 1280x720 slide. Sketches are labelled as sketches, not data.
Run: .venv/bin/python gen_weekly_figs.py
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

from gen_story_figs import (CYAN, FATE, GRAY_FILL, INK, MUTED, PEACH, PEACH_EC, TEAL, arrow,
                            canvas, rbox, save)

NAVY = "#0f3460"     # knowing
ORANGE = "#D55E00"   # acting / the interesting case
BLUE = "#0072B2"     # the boring case


def tag(ax, x, y, text="sketch, not data"):
    ax.text(x, y, text, ha="right", va="top", fontsize=13, color=MUTED, style="italic")


# ------------------------------------------------------------------ angle story
def fig_angle_story():
    """Held vs flipped: does the angle between knowing and acting open up?"""
    fig, ax = canvas(12.0, 5.0, 120, 50)

    def panel(x0, title, deg, verdict):
        ax.text(x0 + 25, 47, title, ha="center", va="center", fontsize=19, weight="bold", color=INK)
        ox, oy, L = x0 + 6, 12, 30
        arrow(ax, (ox, oy), (ox + L, oy), color=NAVY, lw=4, ms=26)
        ax.text(ox + L + 1.5, oy, "knowing", ha="left", va="center", fontsize=17, color=NAVY,
                weight="bold")
        t = math.radians(deg)
        tip = (ox + L * math.cos(t), oy + L * math.sin(t))
        arrow(ax, (ox, oy), tip, color=ORANGE, lw=4, ms=26)
        ax.text(tip[0] + 1.5, tip[1] + 0.5, "acting", ha="left", va="center", fontsize=17,
                color=ORANGE, weight="bold")
        ax.add_patch(Arc((ox, oy), 16, 16, theta1=0, theta2=deg, color=INK, lw=2))
        mid = math.radians(deg / 2)
        ax.text(ox + (13 if deg < 30 else 10.5) * math.cos(mid), oy + (13 if deg < 30 else 10.5) * math.sin(mid) + (1.2 if deg < 30 else 0), "θ", fontsize=20,
                color=INK, style="italic", ha="center", va="center")
        ax.text(x0 + 25, 4, verdict, ha="center", va="center", fontsize=16, color=INK)

    panel(2, "Model holds the right answer", 18, "small θ: what it knows steers what it says")
    ax.plot([60, 60], [6, 46], color="#D8DCE3", lw=1.5)
    panel(64, "Model flips under pressure", 72, "wide θ? it still knows, but that stops steering")
    tag(ax, 119, 1.2)
    save(fig, "fig_angle_story")


# ------------------------------------------------------------------ simple plan
def fig_plan_simple():
    fig, ax = canvas(12.0, 4.3, 120, 43)

    # pressures
    rbox(ax, 1, 2, 36, 39, fc="white", ec="#D8DCE3", r=1.6)
    ax.text(19, 37.5, "Same facts, 4 pressures", ha="center", va="center", fontsize=17,
            weight="bold", color=INK)
    chips = [("1  tempting wrong option", True), ("2  user pushback", True),
             ("3  no single answer", False), ("4  forced to answer", False)]
    for k, (txt, now) in enumerate(chips):
        y = 27.5 - k * 6.6
        rbox(ax, 3.5, y - 2.6, 31, 5.2, fc=CYAN if now else GRAY_FILL,
             ec=TEAL if now else "#C9CED6", ls=(0, (3, 2)), r=1.2)
        ax.text(5.5, y, txt, va="center", fontsize=14, color=INK if now else MUTED,
                weight="bold" if now else "normal")
    ax.text(19, 32.8, "start with 1 + 2", ha="center", va="center", fontsize=14, color=TEAL,
            style="italic")

    arrow(ax, (38.5, 21.5), (44.5, 21.5), lw=2.6, ms=24)

    # stage A
    rbox(ax, 46, 2, 36, 39, fc=CYAN, ec=TEAL, r=1.6)
    ax.text(64, 37, "Stage A · now", ha="center", va="center", fontsize=19, weight="bold",
            color=TEAL)
    ax.text(64, 29.5, "Sort each failure\ninto one of six fates", ha="center", va="center",
            fontsize=16, color=INK, linespacing=1.35)
    for k, c in enumerate(FATE.values()):
        ax.add_patch(Circle((53.5 + k * 4.2, 21.3), 1.5, color=c))
    ax.text(64, 10.5, "measure small $\\rightarrow$ big:\none direction, then subspace,\nthen paths", ha="center",
            va="center", fontsize=14, color=MUTED, linespacing=1.35)

    arrow(ax, (83.5, 21.5), (89.5, 21.5), lw=2.6, ms=24)

    # stage B
    rbox(ax, 91, 2, 28, 39, fc=PEACH, ec=PEACH_EC, r=1.6)
    ax.text(105, 37, "Stage B · last", ha="center", va="center", fontsize=19, weight="bold",
            color="#9A4A12")
    ax.text(105, 26, "Test fixes:\ndoes each fate\nneed its own?", ha="center", va="center",
            fontsize=16, color=INK, linespacing=1.35)
    ax.text(105, 11.5, "public checkpoints\nfirst (cheaper)", ha="center", va="center",
            fontsize=14, color=MUTED, linespacing=1.35)
    save(fig, "fig_plan_simple")


# ------------------------------------------------------------------ free screen
def fig_screen():
    """How to read the nearly free screen: flip rate vs prior confidence."""
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    x = np.linspace(0, 1, 200)
    boring = 0.85 * (1 - x) ** 3
    deep = boring + 0.33 * (1 - (1 - x) ** 3)
    ax.plot(x, boring, color=BLUE, lw=3.5, solid_capstyle="round")
    ax.plot(x, deep, color=ORANGE, lw=3.5, solid_capstyle="round")
    ax.text(0.53, 0.16, "only unsure items flip\n$\\rightarrow$ a calibration problem (cheap fix)",
            fontsize=16, color=INK, va="bottom")
    ax.plot([0.505, 0.505], [0.17, 0.25], color=BLUE, lw=5, solid_capstyle="round")
    ax.text(0.53, 0.47, "sure items flip too\n$\\rightarrow$ something inside overrides it",
            fontsize=16, color=INK, va="bottom")
    ax.plot([0.505, 0.505], [0.48, 0.56], color=ORANGE, lw=5, solid_capstyle="round")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.92)
    ax.set_xticks([0.03, 0.97], ["unsure", "very sure"], fontsize=15, color=MUTED)
    ax.set_yticks([0.03, 0.85], ["rarely", "often"], fontsize=15, color=MUTED)
    ax.tick_params(length=0, pad=8)
    ax.set_xlabel("How sure the model was before the pressure", fontsize=16, color=INK, labelpad=8)
    ax.set_ylabel("How often it flips", fontsize=16, color=INK, labelpad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9CA3AF")
    ax.text(1.0, 0.92, "sketch, not data", ha="right", va="top", fontsize=13, color=MUTED,
            style="italic", transform=ax.transData)
    save(fig, "fig_screen")


if __name__ == "__main__":
    fig_angle_story()
    fig_plan_simple()
    fig_screen()
    print("done")
