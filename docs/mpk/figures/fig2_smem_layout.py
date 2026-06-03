"""Figure 2: SMEM page layout for the linear task.

14 physical pages × 16 KB. W regions own 2 pages each (not packable).
A regions pack 3 per page (sub-page packable). Scratch piggybacks.
Each page is labeled with its release_stage.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

fig, ax = plt.subplots(figsize=(17, 5.0))
ax.set_xlim(0, 17)
ax.set_ylim(0, 5.5)
ax.axis("off")

def stage_color(s, base="W"):
    if base == "W":
        return plt.cm.Blues(0.35 + s * 0.10)
    return plt.cm.Oranges(0.35 + s * 0.10)

COLOR_FRAME = "#3a3a3a"
COLOR_TEXT_DARK = "#222222"
COLOR_TEXT_LIGHT = "#ffffff"

# Title
ax.text(0.3, 5.20,
        "SMEM physical pages (linear task) — 14 pages × 16 KB = 224 KB",
        fontsize=12, weight="bold", color=COLOR_FRAME)
ax.text(0.3, 4.85,
        "Page assignment fixed at task registration.  "
        "release_stage[p] = the max stage whose region sits on page p.",
        fontsize=9, color=COLOR_FRAME, style="italic")

cell_w = 1.15
cell_h = 2.6
y0 = 1.3
x0 = 0.3
gap = 0.04

def draw_page(idx, fill, label_top, sub_lines, release_stage, text_color):
    x = x0 + idx * (cell_w + gap)
    b = FancyBboxPatch(
        (x, y0), cell_w, cell_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor="#222222", facecolor=fill
    )
    ax.add_patch(b)
    cx = x + cell_w / 2
    # page index above
    ax.text(cx, y0 + cell_h + 0.13, f"p{idx}",
            ha="center", va="bottom", fontsize=10, weight="bold",
            color=COLOR_FRAME)
    # main label
    ax.text(cx, y0 + cell_h - 0.35, label_top, ha="center", va="center",
            fontsize=13, weight="bold", color=text_color)
    # sub-lines (packed contents)
    for i, line in enumerate(sub_lines):
        ax.text(cx, y0 + cell_h - 0.85 - i * 0.32, line,
                ha="center", va="center", fontsize=8,
                color=text_color, family="monospace")
    # release_stage below the cell, compact
    ax.text(cx, y0 - 0.25, f"▶ stage {release_stage}",
            ha="center", va="top", fontsize=9,
            color=COLOR_FRAME, weight="bold")

# W pages — paired by stage
for stage in range(6):
    for off in (0, 1):
        idx = 2 * stage + off
        col = stage_color(stage, "W")
        text_col = COLOR_TEXT_DARK if stage < 2 else COLOR_TEXT_LIGHT
        draw_page(
            idx,
            fill=col,
            label_top=f"W_{stage}",
            sub_lines=["32 KB", "(1 of 2 pages)"] if off == 0 else ["32 KB", "(2 of 2 pages)"],
            release_stage=stage,
            text_color=text_col
        )

# Bracket annotation for each W pair
for stage in range(6):
    x_left = x0 + (2 * stage) * (cell_w + gap)
    x_right = x_left + 2 * cell_w + gap
    by = y0 + cell_h + 0.42
    ax.plot([x_left + 0.05, x_left + 0.05, x_right - 0.05, x_right - 0.05],
            [by - 0.06, by, by, by - 0.06],
            color="#555555", linewidth=0.9)

# A pages: 2 packed pages
a_packing = {
    12: [("A_0", "4 KB"), ("A_1", "4 KB"), ("A_2", "4 KB"), ("·", "free")],
    13: [("A_3", "4 KB"), ("A_4", "4 KB"), ("A_5", "4 KB"), ("scratch", "16 B")],
}
a_release = {12: 2, 13: 5}

for page in (12, 13):
    sub = [f"{n}  ({sz})" for n, sz in a_packing[page]]
    draw_page(
        page,
        fill=stage_color(a_release[page], "A"),
        label_top="A pack" + (" + S" if page == 13 else ""),
        sub_lines=sub,
        release_stage=a_release[page],
        text_color=COLOR_TEXT_LIGHT
    )

# Legend
legend_y = 0.50
ax.text(0.3, legend_y + 0.15, "Legend",
        fontsize=10, weight="bold", color=COLOR_FRAME)
ax.add_patch(Rectangle((0.3, legend_y - 0.35), 0.35, 0.22,
                       facecolor=stage_color(3, "W"), edgecolor="#222222"))
ax.text(0.75, legend_y - 0.24,
        "W_K   32 KB, not packable, 2 dedicated pages per stage",
        fontsize=9, color=COLOR_FRAME, va="center")
ax.add_patch(Rectangle((8.0, legend_y - 0.35), 0.35, 0.22,
                       facecolor=stage_color(3, "A"), edgecolor="#222222"))
ax.text(8.45, legend_y - 0.24,
        "A_K   4 KB, packable, 3 A's per page (+ scratch on p13)",
        fontsize=9, color=COLOR_FRAME, va="center")

plt.tight_layout()
out = "/home/xinhaoc/mirage-refactor/docs/mpk/figures/fig2_smem_layout.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
