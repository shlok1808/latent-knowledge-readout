#!/usr/bin/env python3
"""Explanatory figures in the safety-paper explainer style (robot icons, dashed
pastel boxes, real dialog text). Numbers come from the round 2 report.
Run: .venv/bin/python gen_story_figs.py
"""
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Wedge

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "DejaVu Sans"],
    "savefig.dpi": 220, "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
})

INK = "#1F2430"
MUTED = "#6B7280"
TEAL = "#2B8A9E"        # model, correct signal
CYAN = "#E3F4F8"
PEACH = "#FCE5D0"
PEACH_EC = "#E3A877"
RED = "#D64545"
GREEN = "#2E9E5B"
GRAY_FILL = "#F3F4F6"
FATE = {"Readout": "#E69F00", "Bypass": "#56B4E9", "Not routed": "#0072B2",
        "Override": "#D55E00", "Transformed": "#009E73", "Erased": "#8C8C8C"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}.svg"), facecolor="white")
    plt.close(fig)


def canvas(w, h, W, H):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc="white", ec=MUTED, ls="-", lw=1.4, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, ls=ls, lw=lw, zorder=z))


def pill(ax, cx, cy, text, w, h=3.2, fc=CYAN, ec=TEAL, tc=TEAL, ls=(0, (3, 2)), fs=13,
         rot=0, alpha=1.0, weight="bold"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=fc, ec=ec, ls=ls, lw=1.6, zorder=3, alpha=alpha))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight,
            rotation=rot, zorder=4, alpha=alpha)


def arrow(ax, p, q, color=MUTED, lw=1.8, ls="-", rad=0.0, ms=14, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=z))


def robot(ax, x, y, s=1.0, color=TEAL, mood="flat"):
    """Simple line-art robot head centred at (x, y)."""
    w, h = 6.0 * s, 4.8 * s
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={1.1 * s}",
                                fc="white", ec=color, lw=2.6 * s ** 0.5, zorder=5))
    for sx in (-1, 1):  # ears
        ax.add_patch(FancyBboxPatch((x + sx * (w / 2) - (0.0 if sx > 0 else 0.7 * s), y - 0.9 * s),
                                    0.7 * s, 1.8 * s, boxstyle="round,pad=0,rounding_size=0.25",
                                    fc=color, ec=color, zorder=4))
    ax.plot([x, x], [y + h / 2, y + h / 2 + 0.9 * s], color=color, lw=2.4 * s ** 0.5, zorder=4)
    ax.add_patch(Circle((x, y + h / 2 + 1.15 * s), 0.45 * s, fc="white", ec=color,
                        lw=2.2 * s ** 0.5, zorder=5))
    for sx in (-1, 1):  # eyes
        ax.add_patch(Circle((x + sx * 1.3 * s, y + 0.55 * s), 0.5 * s, fc=color, ec=color, zorder=6))
    if mood == "flat":
        ax.plot([x - 1.2 * s, x + 1.2 * s], [y - 1.0 * s] * 2, color=color, lw=2.4 * s ** 0.5, zorder=6)
    else:  # smile or frown arc
        theta1, theta2, cy = (200, 340, y - 0.3 * s) if mood == "happy" else (20, 160, y - 1.8 * s)
        ax.add_patch(Arc((x, cy), 2.6 * s, 1.8 * s, theta1=theta1, theta2=theta2, color=color,
                         lw=2.4 * s ** 0.5, zorder=6))


def mark(ax, x, y, ok, r=1.5):
    c = GREEN if ok else RED
    ax.add_patch(Circle((x, y), r, fc="white", ec=c, lw=2.2, zorder=5))
    if ok:
        ax.plot([x - 0.7 * r, x - 0.15 * r, x + 0.75 * r], [y, y - 0.55 * r, y + 0.6 * r], color=c,
                lw=2.6, zorder=6, solid_capstyle="round")
    else:
        for d in (1, -1):
            ax.plot([x - 0.55 * r, x + 0.55 * r], [y - d * 0.55 * r, y + d * 0.55 * r], color=c,
                    lw=2.6, zorder=6, solid_capstyle="round")


# ------------------------------------------------------------------ problem
def fig_problem():
    fig, ax = canvas(12.5, 5.6, 125, 56)

    def row(y, label, user, mood, out, ok, inside=None):
        ax.text(2, y + 8.5, label, fontsize=14, weight="bold", color=INK)
        rbox(ax, 2, y - 3.5, 38, 9, fc=CYAN, ec=TEAL, ls=(0, (3, 2)))
        ax.text(4, y + 1, user, fontsize=13, color=INK, va="center", linespacing=1.3)
        arrow(ax, (41, y + 1), (47, y + 1), lw=2.2, ms=18)
        robot(ax, 54, y + 1, s=1.25, mood=mood)
        if inside:
            pill(ax, 54, y - 6.6, inside, w=27, h=3.4, fs=12)
            ax.text(54, y - 10.3, "read out of the model's internals", ha="center", fontsize=10.5,
                    color=MUTED, style="italic")
        ax.plot([62, 72], [y + 1, y + 1], color=MUTED, lw=1.6, ls=(0, (1.5, 2.5)))
        rbox(ax, 73, y - 2.2, 34, 6.4, fc=PEACH, ec=PEACH_EC)
        ax.text(90, y + 1, out, ha="center", va="center", fontsize=13.5, color=INK)
        mark(ax, 112, y + 1, ok, r=2.0)

    ax.text(55, 54.5, "", fontsize=1)
    row(40, "Turn 1", "What is the capital of Australia?", "happy", "“Canberra.”", True)
    ax.plot([2, 123], [29.5, 29.5], color="#C9CED6", lw=1.2)
    row(16, "Turn 2: the user pushes back", "“No, I'm pretty sure it's Sydney.”",
        "sad", "“You're right, it's Sydney.”", False,
        inside="probe still reads: Canberra")
    ax.text(62.5, 1.2, "The right answer is still decodable inside. What stops it from reaching the output?",
            ha="center", fontsize=14, style="italic", color=INK)
    save(fig, "fig_problem")


# ------------------------------------------------------------------ six fates
def fig_fates():
    fig, ax = canvas(13.2, 8.6, 132, 86)
    W, H = 42, 41
    spots = [(1, 44), (45, 44), (89, 44), (1, 1), (45, 1), (89, 1)]
    names = [("Readout", "Readout failure"), ("Bypass", "Bypass"), ("Not routed", "Not routed"),
             ("Override", "Override"), ("Transformed", "Transformed"), ("Erased", "Erased")]
    notes = ["scores lean right; the cut-off is wrong",
             "push it either way: nothing changes",
             "injected, it works; never carried there",
             "a competing signal wins",
             "still there, re-encoded (sign-flipped)",
             "gone, even for a fresh probe"]
    status = [  # (studied before?, what was shown, why chase it again)
        (True, "logic task, ≤8B", "untested on sycophancy or above 8B"),
        (True, "dates, agent asking, ≤14B", "rarely for factual correctness"),
        (True, "unanswerable questions, to 70B", "natural use never measured"),
        (True, "sycophancy heads, ≤27B", "unclear at 70B; not read as it agrees"),
        (False, "once: instructed lying, Llama-2-70B", "untested under user pressure"),
        (False, "one contested claim, no controls", "needs a controlled test"),
    ]
    for k, ((x0, y0), (key, title), note, (done, shown, again)) in enumerate(
            zip(spots, names, notes, status)):
        c = FATE[key]
        rbox(ax, x0, y0, W, H, fc="#FBFBFC", ec="#D8DCE3", lw=1.2, r=1.6)
        ax.text(x0 + 2, y0 + H - 3, f"{chr(97 + k)})  {title}", fontsize=14, weight="bold",
                color=c, va="center")
        b = y0 + 11          # base of the old panel drawing
        ax.text(x0 + 2, b + 2.6, note, fontsize=11.5, color=INK, style="italic", va="center")
        # status band
        if done:
            rbox(ax, x0 + 1.2, y0 + 5.0, W - 2.4, 4.4, fc=INK, ec=INK, r=0.9, z=2)
            ax.text(x0 + 2.4, y0 + 7.2, "STUDIED", fontsize=9.5, weight="bold", color="white",
                    va="center", zorder=3)
            ax.text(x0 + 11.5, y0 + 7.2, shown, fontsize=10, color="white", va="center", zorder=3)
        else:
            rbox(ax, x0 + 1.2, y0 + 5.0, W - 2.4, 4.4, fc="white", ec="#9AA3AF", ls=(0, (3, 2)),
                 r=0.9, z=2)
            ax.text(x0 + 2.4, y0 + 7.2, "BARELY", fontsize=9.5, weight="bold", color=MUTED,
                    va="center", zorder=3)
            ax.text(x0 + 10.6, y0 + 7.2, shown, fontsize=9.8, color=INK, va="center", zorder=3)
        ax.text(x0 + 2.4, y0 + 2.4, "why redo: " + again, fontsize=10, color=c,
                weight="bold", va="center")
        # inside-the-model region
        rbox(ax, x0 + 2, b + 6.5, 26, 17, fc=CYAN, ec=TEAL, ls=(0, (3, 2)), lw=1.1, r=1.2)
        ax.text(x0 + 3, b + 21.6, "inside the model", fontsize=9.5, color=TEAL, style="italic",
                va="center")
        rx, ry = x0 + 22.5, b + 13.5
        robot(ax, rx, ry, s=0.7, mood="sad")
        pill(ax, x0 + 36.2, ry, "Sydney", w=10, h=3.4, fc=PEACH, ec=PEACH_EC, tc=INK,
             ls="-", fs=11.5, weight="normal")
        arrow(ax, (rx + 2.5, ry), (x0 + 31, ry), lw=1.5, ms=11)
        mark(ax, x0 + 36.2, ry + 4.4, False, r=1.1)
        sx, sy = x0 + 9.5, b + 17.0
        if key == "Readout":
            pill(ax, sx, sy, "Canberra", w=11, h=3.0, fs=11)
            arrow(ax, (sx + 5.6, sy), (rx - 2.3, ry + 0.8), color=TEAL, lw=1.6, ms=11)
            for j, (lab, val) in enumerate([("Canberra", 0.48), ("Sydney", 0.52)]):
                yy = b + 11.0 - j * 2.6
                ax.text(x0 + 3.2, yy, lab, fontsize=8.5, color=MUTED, va="center")
                ax.add_patch(FancyBboxPatch((x0 + 10.5, yy - 0.7), 10 * val, 1.4,
                             boxstyle="square,pad=0", fc=TEAL if j == 0 else PEACH_EC, ec="none"))
            ax.plot([x0 + 15.5, x0 + 15.5], [b + 7.2, b + 12.4], color=RED, lw=1.4)
            ax.text(x0 + 15.5, b + 12.8, "cut-off", fontsize=8.5, color=RED, ha="center")
        elif key == "Bypass":
            pill(ax, sx, sy, "Canberra", w=11, h=3.0, fs=11)
            ax.plot([sx + 5.6, sx + 8.0], [sy, sy - 0.8], color=TEAL, lw=1.6, ls=(0, (2, 2)))
            ax.text(sx + 8.6, sy - 1.0, r"$\emptyset$", fontsize=13, color=TEAL, va="center")
            pill(ax, x0 + 10.5, b + 9.8, "other features", w=13, h=2.8, fc="white", ec=MUTED,
                 tc=MUTED, fs=9.5, weight="normal")
            arrow(ax, (x0 + 17.1, b + 9.8), (rx - 2.3, ry - 1.0), color=MUTED, lw=1.6, ms=11)
        elif key == "Not routed":
            pill(ax, sx, sy, "Canberra", w=11, h=3.0, fs=11)
            ax.plot([sx + 5.6, sx + 7.6], [sy, sy - 0.6], color=TEAL, lw=1.6)
            ax.plot([sx + 9.6, rx - 2.4], [sy - 1.2, ry + 0.6], color=TEAL, lw=1.6, ls=(0, (1, 2)))
            ax.text(sx + 8.4, sy + 1.4, "no wire", fontsize=8.5, color=TEAL, ha="center")
            ax.text(x0 + 11, b + 9.6, "inject it $\\rightarrow$ Canberra", fontsize=9.5,
                    color=FATE[key], ha="center", weight="bold")
        elif key == "Override":
            pill(ax, sx, sy, "Canberra", w=11, h=3.0, fs=11)
            arrow(ax, (sx + 5.6, sy), (rx - 2.3, ry + 0.8), color=TEAL, lw=1.0, ms=9)
            pill(ax, x0 + 10.5, b + 9.8, "agree with user", w=14, h=3.0, fc="#FDE3DC",
                 ec=FATE[key], tc=FATE[key], fs=10, ls="-")
            arrow(ax, (x0 + 17.6, b + 9.8), (rx - 2.3, ry - 1.0), color=FATE[key], lw=3.2, ms=15)
        elif key == "Transformed":
            pill(ax, sx, sy, "Canberra", w=11, h=3.0, fs=11, rot=180)
            arrow(ax, (sx + 5.6, sy), (rx - 2.3, ry + 0.8), color=TEAL, lw=1.6, ms=11)
            ax.text(x0 + 11, b + 10.2, "old probe reads it\nas the wrong answer", fontsize=9,
                    color=FATE[key], ha="center", va="center", weight="bold")
        else:
            pill(ax, sx, sy, "Canberra", w=11, h=3.0, fs=11, fc="white", ec="#BFC4CC",
                 tc="#BFC4CC", alpha=0.9)
            ax.plot([sx - 5, sx + 5], [sy, sy], color=FATE[key], lw=1.6)
            ax.text(x0 + 11, b + 10.2, "nothing left to read", fontsize=9.5, color=FATE[key],
                    ha="center", weight="bold")
    save(fig, "fig_fates")


def fig_angle_explainer():
    """What 'the angle between knowing and acting' means, before any numbers."""
    fig, ax = canvas(6.4, 5.0, 1.30, 1.02)
    ox, oy = 0.08, 0.10
    arrow(ax, (ox, oy), (ox + 0.85, oy), color="#0f3460", lw=3.2, ms=22)
    ax.text(ox + 0.86, oy - 0.04, "knowing", ha="left", va="top", fontsize=14, color="#0f3460",
            weight="bold")
    ax.text(ox + 0.86, oy - 0.115, "probe trained on the\ncorrect answer", ha="left", va="top",
            fontsize=10.5, color=MUTED)
    t = math.radians(62)
    arrow(ax, (ox, oy), (ox + 0.8 * math.cos(t), oy + 0.8 * math.sin(t)), color="#D55E00",
          lw=3.2, ms=22)
    ax.text(ox + 0.8 * math.cos(t) + 0.03, oy + 0.8 * math.sin(t), "acting", ha="left",
            va="center", fontsize=14, color="#D55E00", weight="bold")
    ax.text(ox + 0.8 * math.cos(t) + 0.03, oy + 0.8 * math.sin(t) - 0.09,
            "direction that sets the\nanswer the model gives", ha="left", va="top", fontsize=10.5,
            color=MUTED)
    ax.add_patch(Arc((ox, oy), 0.34, 0.34, theta1=0, theta2=62, color=INK, lw=1.6))
    ax.text(ox + 0.22, oy + 0.09, "θ", fontsize=17, color=INK, style="italic")
    rbox(ax, 0.02, 0.86, 1.24, 0.14, fc=GRAY_FILL, ec="#D8DCE3", r=0.03)
    ax.text(0.05, 0.93, "θ ≈ 0°: knowing drives the answer     θ ≈ 90°: unrelated",
            fontsize=11.5, color=INK, va="center")
    save(fig, "fig_angle_explainer")


def fig_angles():
    fig, ax = canvas(7.4, 5.0, 1.62, 1.07)
    ox, oy = 0.06, 0.06
    ax.add_patch(Wedge((ox, oy), 0.92, 60, 80, fc="#009E73", alpha=0.13, ec="none"))
    ax.add_patch(Arc((ox, oy), 0.46, 0.46, theta1=0, theta2=90, color="#C9CED6", lw=1))
    arrow(ax, (ox, oy), (ox + 0.98, oy), color="#0f3460", lw=3, ms=20)
    ax.text(ox + 1.0, oy - 0.02, "knowing", ha="left", va="top", fontsize=13, color="#0f3460",
            weight="bold")
    items = [
        (88.0, "probe vs causal direction", "≈ 88°, same as a random pair", "#56B4E9"),
        (math.degrees(math.acos(0.087)), "unanswerable: recognize vs refuse",
         "≈ 85° (cos 0.087), 11 models up to 70B", "#D55E00"),
        (math.degrees(math.acos(0.16)), "fake entity: detect vs refuse",
         "≈ 78–83° (cos 0.12–0.20)", "#E69F00"),
        (70.0, "KAPPA: knowledge vs chosen answer", "60–80° in middle layers (shaded)", "#009E73"),
    ]
    for deg, _, _, c in items:
        t = math.radians(deg)
        arrow(ax, (ox, oy), (ox + 0.9 * math.cos(t), oy + 0.9 * math.sin(t)), color=c, lw=2.4, ms=16)
    for k, (deg, name, val, c) in enumerate(items):
        ly = 0.96 - k * 0.19
        ax.text(0.5, ly, name, fontsize=12.5, color=c, weight="bold", va="center")
        ax.text(0.5, ly - 0.065, val, fontsize=11, color=MUTED, va="center")
    ax.text(ox + 0.2, oy + 0.05, "0° would mean\nknowing = acting", fontsize=10, color=MUTED)
    save(fig, "fig_angles")


# ------------------------------------------------------------------ plan
def fig_plan():
    fig, ax = canvas(15.0, 6.0, 150, 60)
    ax.text(1, 57, "Same facts, four kinds of pressure", fontsize=15, weight="bold", color=INK)
    conds = [
        ("1  Competing answer", "“Capital of Australia?\n (a) Sydney  (b) Canberra”", True),
        ("2  User pushback", "“No, I'm pretty sure\n it's Sydney.”", True),
        ("3  No single answer", "“Which is Australia's\n most important city?”", False),
        ("4  Forced to answer", "“You must answer: who is\n the mayor of Canberra?”", False),
    ]
    for k, (head, quote, first) in enumerate(conds):
        x = 1 + (k % 2) * 30
        y = 29 - (k // 2) * 25
        ec = TEAL if first else "#AEB5BF"
        rbox(ax, x, y, 28, 22, fc=CYAN if first else GRAY_FILL, ec=ec, ls=(0, (3, 2)), r=1.4)
        ax.text(x + 1.6, y + 19.2, head, fontsize=13.5, weight="bold", color=TEAL if first else MUTED,
                va="center")
        ax.text(x + 1.6, y + 14.6, quote, fontsize=11.5, color=INK, va="top", linespacing=1.35)
        robot(ax, x + 22.5, y + 4.8, s=0.62, color=ec, mood="flat")
    ax.text(31, 1.0, "start with 1 + 2 on shared facts, then add 3 + 4", ha="center", fontsize=11.5,
            color=MUTED, style="italic")
    arrow(ax, (61.5, 30), (67, 30), lw=2.2, ms=20)

    ax.text(68, 57, "Stage A: sort each failure, cheapest check first", fontsize=15, weight="bold",
            color=INK)
    steps = [("Readout", "output scores already lean right?", "recalibrate"),
             ("Bypass", "push it both ways: any effect?", "gate the action"),
             ("Not routed", "injecting works, natural use ≈ 0?", "realign / train the route"),
             ("Override", "silencing a competitor fixes it?", "tune those heads"),
             ("Transformed", "old probe below chance?", "refit monitors in context"),
             ("Erased", "fresh probe at chance?", "better data / retrieval")]
    ax.text(122, 52.2, "fix it points to", fontsize=10.5, color=MUTED, style="italic")
    for k, (f, q, fix) in enumerate(steps):
        y = 48 - k * 6.6
        c = FATE[f]
        ax.add_patch(Circle((70, y), 1.7, color=c, zorder=3))
        ax.text(70, y - 0.05, str(k + 1), ha="center", va="center", fontsize=11, color="white",
                weight="bold", zorder=4)
        ax.text(73, y, f, fontsize=13, weight="bold", color=c, va="center")
        ax.text(89, y, q, fontsize=11.5, color=INK, va="center")
        ax.text(122, y, fix, fontsize=11, color=MUTED, va="center")
        if k < len(steps) - 1:
            ax.plot([70, 70], [y - 1.9, y - 4.7], color="#C9CED6", lw=1.4)
    rbox(ax, 67.5, 2.0, 81, 7.4, fc=PEACH, ec=PEACH_EC, r=1.4)
    ax.text(69.5, 7.0, "Stage B: fix and recheck", fontsize=13.5, weight="bold", color="#9A4A12",
            va="center")
    ax.text(69.5, 3.9, "apply the fix (and plain DPO), rerun Stage A: problem removed, or only hidden?",
            fontsize=11.3, color=INK, va="center")
    save(fig, "fig_plan")


if __name__ == "__main__":
    fig_problem()
    fig_fates()
    fig_angle_explainer()
    fig_angles()
    fig_plan()
    print("done")
