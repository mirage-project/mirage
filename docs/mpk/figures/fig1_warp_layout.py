"""Figure 1: Warp role layout for mirage v2 persistent kernel.

8 warps × 32 threads per SM. Each warp has a fixed role.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis("off")

# Color palette — distinct, not over-saturated
COLOR_CONSUMER = "#4C72B0"   # blue
COLOR_LOADER = "#55A868"     # green
COLOR_LAUNCHER = "#C44E52"   # red
COLOR_STORER = "#8172B2"     # purple
COLOR_CONTROLLER = "#CCB974" # yellow-brown
COLOR_TEXT_DARK = "#222222"
COLOR_TEXT_LIGHT = "#ffffff"
COLOR_FRAME = "#3a3a3a"

# Outer SM frame
sm = FancyBboxPatch(
    (0.3, 0.4), 12.4, 4.2,
    boxstyle="round,pad=0.02,rounding_size=0.15",
    linewidth=2, edgecolor=COLOR_FRAME, facecolor="#f7f7f7"
)
ax.add_patch(sm)
ax.text(0.55, 4.30, "SM (block) — 8 warps × 32 threads = 256",
        fontsize=11, color=COLOR_FRAME, weight="bold")
ax.text(0.55, 3.95, "launch_bounds(256, 1)",
        fontsize=9, color=COLOR_FRAME, family="monospace")

# Warp boxes
warps = [
    # (idx, label_top, label_role, label_sub, color, text_color)
    (0, "W0", "Consumer", "register-heavy compute", COLOR_CONSUMER, COLOR_TEXT_LIGHT),
    (1, "W1", "Consumer", "register-heavy compute", COLOR_CONSUMER, COLOR_TEXT_LIGHT),
    (2, "W2", "Consumer", "register-heavy compute", COLOR_CONSUMER, COLOR_TEXT_LIGHT),
    (3, "W3", "Consumer", "register-heavy compute", COLOR_CONSUMER, COLOR_TEXT_LIGHT),
    (4, "W4", "Loader",   "TMA load",                COLOR_LOADER,   COLOR_TEXT_LIGHT),
    (5, "W5", "Launcher", "tensor core (MMA)",       COLOR_LAUNCHER, COLOR_TEXT_LIGHT),
    (6, "W6", "Storer",   "TMA store",               COLOR_STORER,   COLOR_TEXT_LIGHT),
    (7, "W7", "Controller","dep wait + metadata",    COLOR_CONTROLLER, COLOR_TEXT_DARK),
]

box_w = 1.45
box_h = 2.3
x0 = 0.65
y0 = 1.0
gap = 0.05

for i, (idx, top, role, sub, color, tcolor) in enumerate(warps):
    x = x0 + i * (box_w + gap)
    b = FancyBboxPatch(
        (x, y0), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, edgecolor="#222222", facecolor=color
    )
    ax.add_patch(b)
    cx = x + box_w / 2
    ax.text(cx, y0 + box_h - 0.35, top, ha="center", va="center",
            fontsize=14, weight="bold", color=tcolor)
    ax.text(cx, y0 + box_h - 0.85, role, ha="center", va="center",
            fontsize=11, weight="bold", color=tcolor)
    ax.text(cx, y0 + box_h - 1.30, sub, ha="center", va="center",
            fontsize=8.5, color=tcolor, style="italic")
    # thread range below
    t_lo = idx * 32
    t_hi = t_lo + 31
    ax.text(cx, y0 + 0.30, f"threads {t_lo}–{t_hi}",
            ha="center", va="center", fontsize=8, color=tcolor,
            family="monospace")

# Bracket showing consumer group
bracket_y = 0.78
ax.plot([0.65, 0.65, 0.65 + 4 * (box_w + gap) - gap, 0.65 + 4 * (box_w + gap) - gap],
        [bracket_y, bracket_y - 0.15, bracket_y - 0.15, bracket_y],
        color="#222222", linewidth=1.2)
ax.text(0.65 + 2 * (box_w + gap) - gap / 2, bracket_y - 0.40,
        "4 consumer warps = 128 threads",
        ha="center", va="top", fontsize=9, color="#222222", style="italic")

plt.tight_layout()
out = "/home/xinhaoc/mirage-refactor/docs/mpk/figures/fig1_warp_layout.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
