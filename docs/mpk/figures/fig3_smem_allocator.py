"""Figure 3: SMEM allocator — pipeline-aware page placement.

The corrected allocation algorithm:

  Goal: B's loader starts at task-B kickoff and never blocks on A.

  Rule 1:  Place B's earliest-stage regions on FREE pages (not touched by A).
           B starts TMA into them immediately.

  Rule 2:  Place B's later-stage regions on pages that A RELEASES, matched
           by stage order:
             B.stage_K  uses pages that A releases at stage (K - shift)
           where `shift` = number of B-stages that fit on free pages.

  Effect:  B's pipeline runs in lockstep with A's release pipeline. By the
           time B reaches stage 1, A's stage 0 has already released the
           pages B needs. No idle waiting.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 9.5))
ax.set_xlim(-0.5, 18)
ax.set_ylim(-1.8, 12.0)
ax.axis("off")

COLOR_FRAME = "#3a3a3a"
COLOR_TEXT_DARK = "#222222"
COLOR_A = "#4C72B0"
COLOR_B = "#C44E52"
COLOR_FRESH = "#55A868"     # green for pages B starts on (free, no wait)
COLOR_HANDOFF = "#d97706"   # orange for pages A → B handoff

# ── Title ──────────────────────────────────────────────────────────────
ax.text(-0.5, 11.45,
        "SMEM allocator — pipeline-aware page placement",
        fontsize=14, weight="bold", color=COLOR_FRAME)
ax.text(-0.5, 10.95,
        "Task A: 2 stages, 4 pages.   Task B: 3 stages, 6 pages.   Pool: 8 pages.",
        fontsize=10, color=COLOR_FRAME, style="italic")
ax.text(-0.5, 10.55,
        "Goal: B starts on FREE pages (immediate, no wait); B's later stages reuse pages A releases in matching stage order.",
        fontsize=10, color=COLOR_FRAME, style="italic")

# ── Columns ────────────────────────────────────────────────────────────
LEFT_X = 0.0
LEFT_W = 3.4
PAGE_X = 5.5
PAGE_W = 5.0
RIGHT_X = 13.0
RIGHT_W = 4.0

ax.text(LEFT_X + LEFT_W / 2, 9.55,
        "Task A regions",
        ha="center", fontsize=11, weight="bold", color=COLOR_A)
ax.text(PAGE_X + PAGE_W / 2, 9.55,
        "8-page SMEM pool",
        ha="center", fontsize=11, weight="bold", color=COLOR_FRAME)
ax.text(RIGHT_X + RIGHT_W / 2, 9.55,
        "Task B regions",
        ha="center", fontsize=11, weight="bold", color=COLOR_B)

# ── Page grid (8 pages) ────────────────────────────────────────────────
PAGE_BOX_H = 0.85
PAGE_GAP = 0.18
PAGE_Y_TOP = 9.0

def page_y(pidx):
    return PAGE_Y_TOP - pidx * (PAGE_BOX_H + PAGE_GAP)

# Optimal allocation: use ALL fresh pages first for B's earliest stages,
# then fall back to A's earliest-released pages for B's remaining stages.
#   p0, p1 : A's stage 0 → B's stage 2 (handoff at earliest A release)
#   p2, p3 : A only — B does not use; codegen loader prefix arrives them
#   p4, p5 : B's stage 0 (fresh)
#   p6, p7 : B's stage 1 (fresh — was unused in previous wrong version)
pages = [
    (0, "A: W_0 (s0)  →  B: W_2 (s2)",  "handoff", "rel@s0 in A,  claim@s2 in B"),
    (1, "A: A_0 (s0)  →  B: A_2 (s2)",  "handoff", "rel@s0 in A,  claim@s2 in B"),
    (2, "A: W_1 (s1)  →  (B unused)",   "aonly",   "released, B prefix arrives"),
    (3, "A: A_1 (s1)  →  (B unused)",   "aonly",   "released, B prefix arrives"),
    (4, "B: W_0 (s0)  ←  fresh",        "fresh",   "no wait — B starts here"),
    (5, "B: A_0 (s0)  ←  fresh",        "fresh",   "no wait — B starts here"),
    (6, "B: W_1 (s1)  ←  fresh",        "fresh",   "no wait — B uses fresh next"),
    (7, "B: A_1 (s1)  ←  fresh",        "fresh",   "no wait — B uses fresh next"),
]

KIND_FILL = {
    "handoff": "#fde6cc",   # orange-tinted: shared, handoff
    "fresh":   "#d6ecd9",   # green-tinted: B starts here
    "aonly":   "#e8e8e8",   # gray: A uses, B does not
    "free":    "#ffffff",
}
KIND_EDGE = {
    "handoff": COLOR_HANDOFF,
    "fresh":   COLOR_FRESH,
    "aonly":   "#888888",
    "free":    "#cccccc",
}

for (pidx, content, kind, note) in pages:
    y = page_y(pidx)
    ax.add_patch(Rectangle((PAGE_X, y), PAGE_W, PAGE_BOX_H,
                           facecolor=KIND_FILL[kind],
                           edgecolor=KIND_EDGE[kind], linewidth=1.8))
    ax.text(PAGE_X + 0.18, y + PAGE_BOX_H / 2,
            f"p{pidx}",
            ha="left", va="center", fontsize=10.5, weight="bold",
            color=COLOR_TEXT_DARK, family="monospace")
    # single centered label — no overlap with a second line
    ax.text(PAGE_X + 0.95, y + PAGE_BOX_H / 2,
            content,
            ha="left", va="center", fontsize=10, weight="bold",
            color=COLOR_TEXT_DARK)

# ── Task A regions ─────────────────────────────────────────────────────
def stage_color_w(s): return plt.cm.Blues(0.42 + s * 0.18)
def stage_color_a(s): return plt.cm.Oranges(0.42 + s * 0.18)

A_regions = [
    ("W_0", 0, stage_color_w(0)),
    ("A_0", 0, stage_color_a(0)),
    ("W_1", 1, stage_color_w(1)),
    ("A_1", 1, stage_color_a(1)),
]
REG_H = 0.85
REG_GAP = 0.18
A_TOP = 9.0
for i, (name, stage, color) in enumerate(A_regions):
    y = A_TOP - i * (REG_H + REG_GAP)
    ax.add_patch(Rectangle((LEFT_X, y), LEFT_W, REG_H,
                           facecolor=color, alpha=0.88,
                           edgecolor="#222222", linewidth=0.8))
    text_color = COLOR_TEXT_DARK if (name.startswith("W") and stage == 0) else "white"
    ax.text(LEFT_X + 0.2, y + REG_H / 2, name,
            ha="left", va="center", fontsize=11, weight="bold",
            color=text_color)
    ax.text(LEFT_X + LEFT_W - 0.2, y + REG_H / 2, f"stage {stage}",
            ha="right", va="center", fontsize=9,
            color=text_color, family="monospace")

# ── Task B regions ─────────────────────────────────────────────────────
B_regions = [
    ("W_0", 0, stage_color_w(0)),
    ("A_0", 0, stage_color_a(0)),
    ("W_1", 1, stage_color_w(1)),
    ("A_1", 1, stage_color_a(1)),
    ("W_2", 2, stage_color_w(2)),
    ("A_2", 2, stage_color_a(2)),
]
B_TOP = 9.0
for i, (name, stage, color) in enumerate(B_regions):
    y = B_TOP - i * (REG_H + REG_GAP)
    ax.add_patch(Rectangle((RIGHT_X, y), RIGHT_W, REG_H,
                           facecolor=color, alpha=0.88,
                           edgecolor="#222222", linewidth=0.8))
    text_color = COLOR_TEXT_DARK if (name.startswith("W") and stage == 0) else "white"
    ax.text(RIGHT_X + 0.2, y + REG_H / 2, name,
            ha="left", va="center", fontsize=11, weight="bold",
            color=text_color)
    ax.text(RIGHT_X + RIGHT_W - 0.2, y + REG_H / 2, f"stage {stage}",
            ha="right", va="center", fontsize=9,
            color=text_color, family="monospace")

# ── Arrows ─────────────────────────────────────────────────────────────
# Task A: region i → page i
for i in range(4):
    src = (LEFT_X + LEFT_W, A_TOP - i * (REG_H + REG_GAP) + REG_H / 2)
    dst = (PAGE_X, page_y(i) + PAGE_BOX_H / 2)
    ax.add_patch(FancyArrowPatch(src, dst,
                                  arrowstyle="->", mutation_scale=11,
                                  color=COLOR_A, linewidth=1.4, alpha=0.85,
                                  connectionstyle="arc3,rad=-0.05"))

# Task B mapping (corrected — use all fresh pages first):
#   W_0(s0) → p4    A_0(s0) → p5    (fresh, B starts here)
#   W_1(s1) → p6    A_1(s1) → p7    (fresh, no wait either)
#   W_2(s2) → p0    A_2(s2) → p1    (handoff from A's stage 0 release)
B_to_page = [4, 5, 6, 7, 0, 1]
for i, pidx in enumerate(B_to_page):
    src = (RIGHT_X, B_TOP - i * (REG_H + REG_GAP) + REG_H / 2)
    dst = (PAGE_X + PAGE_W, page_y(pidx) + PAGE_BOX_H / 2)
    ax.add_patch(FancyArrowPatch(src, dst,
                                  arrowstyle="->", mutation_scale=11,
                                  color=COLOR_B, linewidth=1.4, alpha=0.85,
                                  connectionstyle="arc3,rad=0.05"))

# ── Bottom: rule annotations ───────────────────────────────────────────
ax.text(-0.5, -0.10,
        "Allocator rules:",
        fontsize=11, weight="bold", color=COLOR_FRAME)

ax.add_patch(Rectangle((-0.5, -0.75), 0.5, 0.32,
                       facecolor=KIND_FILL["fresh"],
                       edgecolor=KIND_EDGE["fresh"], linewidth=1.5))
ax.text(0.2, -0.59,
        "Rule 1 — fill all fresh pages first  (p4, p5, p6, p7):",
        fontsize=10, weight="bold", va="center", color=COLOR_FRESH)
ax.text(0.2, -0.95,
        "          Pages A never touches.  Use them for B's EARLIEST stages — TMA starts at task-B kickoff, zero wait.",
        fontsize=9.5, va="center", color=COLOR_FRAME)

ax.add_patch(Rectangle((-0.5, -1.45), 0.5, 0.32,
                       facecolor=KIND_FILL["handoff"],
                       edgecolor=KIND_EDGE["handoff"], linewidth=1.5))
ax.text(0.2, -1.29,
        "Rule 2 — only after fresh pages run out, fall back to A's earliest-released pages  (p0, p1):",
        fontsize=10, weight="bold", va="center", color=COLOR_HANDOFF)
ax.text(0.2, -1.65,
        "          B's remaining late stages use A's stage-0 releases — earliest available, shortest wait.",
        fontsize=9.5, va="center", color=COLOR_FRAME)

plt.tight_layout()
out = "/home/xinhaoc/mirage-refactor/docs/mpk/figures/fig3_smem_allocator.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
