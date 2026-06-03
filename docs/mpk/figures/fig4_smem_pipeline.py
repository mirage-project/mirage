"""Figure 4: SMEM page pipeline — v1 baseline vs v2 page allocator.

Same time axis, same TOTAL SMEM size in both panels. The difference
is granularity:
  v1: one monolithic block. Whole SMEM transitions A → idle → B together.
  v2: 8 independent pages. Each transitions on its own release_stage.

The two panels have matching widths AND matching SMEM-strip heights,
so the visual area each task occupies is directly comparable.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(18, 13.5))
ax.set_xlim(-2.0, 32.0)
ax.set_ylim(-2.5, 22.5)
ax.axis("off")

COLOR_FRAME = "#3a3a3a"
COLOR_A = "#4C72B0"
COLOR_B = "#C44E52"
COLOR_FREE = "#e8e8e8"
COLOR_FRESH = "#55A868"
COLOR_HANDOFF = "#d97706"
COLOR_CONTROLLER = "#8172B2"
COLOR_TEXT_DARK = "#222222"

T_START, T_END = 0.0, 30.0

# Both panels run the same workloads.
#   Task A duration = 13 units of work.   Task B duration = 13 units of work.
# v1 schedules them strictly serially (with a 1-unit hard-barrier gap).
# v2 overlaps them — B starts mid-A on fresh pages.
# Total time:  v1 = 27 units,  v2 = 19 units.  v2 saves 8 units (~30%).

T_A_START = 1.0
T_A_LOADER_PREFIX = 1.5
T_A_STAGE0_REL = 10.0   # mid-tail of A
T_A_STAGE1_REL = 13.0
T_A_END = 14.0          # A duration = 13

# v1 schedule: A → gap → B (each task uses whole SMEM, serial)
T_V1_GAP_END = 15.0     # 1-unit gap
T_V1_B_END   = 28.0     # B starts at 15, duration 13

# v2 schedule: B kicks off at t=7 (loader wakes), B duration 13 → ends at 20
T_V2_B_START = 7.0
T_V2_B_END   = 20.0

# Geometry — page grid (v2)
page_h = 0.78
page_gap = 0.18
N_PAGES = 8
PAGE_GRID_HEIGHT = N_PAGES * (page_h + page_gap) - page_gap  # = 7.50

# ════════════════════════════════════════════════════════════════════════
# TOP PANEL — v1 baseline
# ════════════════════════════════════════════════════════════════════════
V1_PANEL_TITLE_Y = 20.5
V1_GRID_Y0 = 12.0
V1_GRID_TOP = V1_GRID_Y0 + PAGE_GRID_HEIGHT  # equal-height strip to v2 grid

ax.text(-2.0, V1_PANEL_TITLE_Y,
        "v1 baseline — SMEM is one monolithic block",
        fontsize=12.5, weight="bold", color=COLOR_FRAME)

# Time axis for v1 panel
V1_TIMELINE_Y = V1_PANEL_TITLE_Y - 0.7
ax.annotate("", xy=(T_END, V1_TIMELINE_Y), xytext=(T_START, V1_TIMELINE_Y),
            arrowprops=dict(arrowstyle="->", lw=1.5, color=COLOR_FRAME))
ax.text(T_END + 0.2, V1_TIMELINE_Y, "time", fontsize=10, va="center",
        color=COLOR_FRAME)

# Left-side label for the v1 strip
ax.text(T_START - 0.15, V1_GRID_Y0 + PAGE_GRID_HEIGHT / 2,
        "SMEM\n(treated as one\nmonolithic block)",
        ha="right", va="center", fontsize=10, weight="bold",
        color=COLOR_FRAME)

# Same 8-row grid as v2. In v1 each task RESERVES the whole SMEM but
# only USES the rows its data actually fills. The unused rows are
# "wasted" — reserved by the task, unavailable to any other task.
# A uses 4 rows of data (matching v2's p0..p3); B uses 6 rows of data
# (matching v2's p0,p1,p4..p7).
A_USED_ROWS = 4
B_USED_ROWS = 6

for p in range(N_PAGES):
    row_y = V1_GRID_Y0 + (N_PAGES - 1 - p) * (page_h + page_gap)
    # row background
    ax.add_patch(Rectangle((T_START, row_y), T_END - T_START, page_h,
                           facecolor="#f6f6f6", edgecolor="#cccccc",
                           linewidth=0.4))
    # Task A interval — only the top A_USED_ROWS rows are filled
    a_used = (p < A_USED_ROWS)
    a_color = COLOR_A if a_used else "#cbd5e6"   # light blue = reserved but unused
    a_alpha = 0.85 if a_used else 0.65
    ax.add_patch(Rectangle((T_A_START, row_y), T_A_END - T_A_START, page_h,
                           facecolor=a_color, alpha=a_alpha,
                           edgecolor="#222222", linewidth=0.4,
                           hatch=None if a_used else "///"))
    # idle gap — all rows are idle
    ax.add_patch(Rectangle((T_A_END, row_y), T_V1_GAP_END - T_A_END, page_h,
                           facecolor=COLOR_FREE, alpha=0.85,
                           edgecolor="#222222", linewidth=0.4))
    # Task B interval — only the top B_USED_ROWS rows are filled
    b_used = (p < B_USED_ROWS)
    b_color = COLOR_B if b_used else "#e9b9bb"   # light red = reserved but unused
    b_alpha = 0.85 if b_used else 0.65
    ax.add_patch(Rectangle((T_V1_GAP_END, row_y), T_V1_B_END - T_V1_GAP_END, page_h,
                           facecolor=b_color, alpha=b_alpha,
                           edgecolor="#222222", linewidth=0.4,
                           hatch=None if b_used else "///"))

# Centered labels for the USED portion of each task
v1_a_mid_y = V1_GRID_Y0 + (N_PAGES - A_USED_ROWS / 2) * (page_h + page_gap) - page_gap / 2 - page_h / 2
v1_b_mid_y = V1_GRID_Y0 + (N_PAGES - B_USED_ROWS / 2) * (page_h + page_gap) - page_gap / 2 - page_h / 2
v1_mid_y = V1_GRID_Y0 + PAGE_GRID_HEIGHT / 2

ax.text((T_A_START + T_A_END) / 2, v1_a_mid_y,
        f"Task A\n(uses {A_USED_ROWS} rows of data)",
        ha="center", va="center", fontsize=10, weight="bold",
        color="white")
ax.text((T_A_END + T_V1_GAP_END) / 2, v1_mid_y,
        "idle",
        ha="center", va="center", fontsize=10, weight="bold",
        color="#cc0000")
ax.text((T_V1_GAP_END + T_V1_B_END) / 2, v1_b_mid_y,
        f"Task B\n(uses {B_USED_ROWS} rows of data)",
        ha="center", va="center", fontsize=10, weight="bold",
        color="white")

# Annotate wasted rows
v1_a_waste_mid_y = V1_GRID_Y0 + (N_PAGES - A_USED_ROWS) / 2 * (page_h + page_gap) - page_gap / 2
v1_b_waste_mid_y = V1_GRID_Y0 + (N_PAGES - B_USED_ROWS) / 2 * (page_h + page_gap) - page_gap / 2
ax.text((T_A_START + T_A_END) / 2, v1_a_waste_mid_y,
        "wasted SMEM\n(reserved by A, no data)",
        ha="center", va="center", fontsize=8, weight="bold",
        color="#666666", style="italic")
ax.text((T_V1_GAP_END + T_V1_B_END) / 2, v1_b_waste_mid_y,
        "wasted",
        ha="center", va="center", fontsize=8, weight="bold",
        color="#666666", style="italic")

# Hard-barrier visual cue: short red dashed lines bracketing the idle gap.
# Text annotation moved to v1 subtitle (above the strip) so it doesn't collide
# with the v2 panel title.
ax.text(-2.0, V1_PANEL_TITLE_Y - 0.50,
        "Hard barrier between tasks — B cannot claim any page until A fully exits.",
        fontsize=10, color="#cc0000", style="italic")
v1_grid_top = V1_GRID_Y0 + PAGE_GRID_HEIGHT
ax.plot([T_A_END, T_A_END],
        [V1_GRID_Y0, v1_grid_top],
        color="#cc0000", linewidth=1.6, linestyle="--", alpha=0.7)
ax.plot([T_V1_GAP_END, T_V1_GAP_END],
        [V1_GRID_Y0, v1_grid_top],
        color="#cc0000", linewidth=1.6, linestyle="--", alpha=0.7)

# ════════════════════════════════════════════════════════════════════════
# BOTTOM PANEL — v2 allocator
# ════════════════════════════════════════════════════════════════════════
V2_PANEL_TITLE_Y = 11.2
V2_GRID_Y0 = 1.0

ax.text(-2.0, V2_PANEL_TITLE_Y,
        "v2 page allocator — 8-page pool, per-page handoff, tasks overlap",
        fontsize=12.5, weight="bold", color=COLOR_FRAME)

V2_TIMELINE_Y = V2_PANEL_TITLE_Y - 0.95
ax.annotate("", xy=(T_END, V2_TIMELINE_Y), xytext=(T_START, V2_TIMELINE_Y),
            arrowprops=dict(arrowstyle="->", lw=1.5, color=COLOR_FRAME))
ax.text(T_END + 0.2, V2_TIMELINE_Y, "time", fontsize=10, va="center",
        color=COLOR_FRAME)

# Task bands
def task_band(x_start, x_end, color, label, y, h=0.55):
    ax.add_patch(Rectangle((x_start, y), x_end - x_start, h,
                           facecolor=color, alpha=0.30,
                           edgecolor=COLOR_FRAME, linewidth=0.9))
    ax.text((x_start + x_end) / 2, y + h / 2, label,
            ha="center", va="center", fontsize=10, weight="bold",
            color=COLOR_FRAME)
task_band(T_A_START, T_A_END, COLOR_A, "Task A", V2_TIMELINE_Y + 0.30)
task_band(T_V2_B_START, T_V2_B_END, COLOR_B, "Task B", V2_TIMELINE_Y + 0.30)

# Controller row (above pages, below task bands)
CTRL_Y = V2_GRID_Y0 + PAGE_GRID_HEIGHT + 0.20
CTRL_H = 0.5
ax.add_patch(Rectangle((T_START, CTRL_Y), T_END - T_START, CTRL_H,
                       facecolor="#f0eef5", edgecolor="#aaa",
                       linewidth=0.5))
ax.text(T_START - 0.15, CTRL_Y + CTRL_H / 2,
        "Controller (W7)",
        ha="right", va="center", fontsize=9, weight="bold",
        color=COLOR_CONTROLLER, family="monospace")

def ctrl_event(x, label):
    ax.plot([x, x], [CTRL_Y - 0.05, CTRL_Y + CTRL_H + 0.05],
            color=COLOR_CONTROLLER, linewidth=2.0)
    ax.text(x, CTRL_Y + CTRL_H + 0.10, label,
            ha="center", va="bottom", fontsize=8.5, weight="bold",
            color=COLOR_CONTROLLER)
ctrl_event(T_A_START - 0.3, "publish A")
ctrl_event(T_V2_B_START - 0.5, "publish B")

# v2 page rows
ax.text(T_START - 0.15, V2_GRID_Y0 + PAGE_GRID_HEIGHT / 2,
        "SMEM pages\n(8-page pool)",
        ha="right", va="center", fontsize=10, weight="bold",
        color=COLOR_FRAME)

def ownership_intervals(p):
    if p in (0, 1):
        return [
            (T_A_START, T_A_STAGE0_REL, "A", COLOR_A),
            (T_A_STAGE0_REL, T_V2_B_END, "B", COLOR_B),
        ]
    if p in (2, 3):
        return [
            (T_A_START, T_A_STAGE1_REL, "A", COLOR_A),
            (T_A_STAGE1_REL, T_V2_B_END, "free", COLOR_FREE),
        ]
    return [
        (T_A_START, T_A_LOADER_PREFIX, "A-prefix-released", "#9fb8d8"),
        (T_A_LOADER_PREFIX, T_V2_B_START, "free", COLOR_FREE),
        (T_V2_B_START, T_V2_B_END, "B", COLOR_B),
    ]

for p in range(N_PAGES):
    row_y = V2_GRID_Y0 + (N_PAGES - 1 - p) * (page_h + page_gap)
    ax.add_patch(Rectangle((T_START, row_y), T_END - T_START, page_h,
                           facecolor="#f6f6f6", edgecolor="#cccccc",
                           linewidth=0.4))
    ax.text(T_START + 0.05, row_y + page_h / 2, f"p{p}",
            ha="left", va="center", fontsize=8.5, weight="bold",
            color="#888", family="monospace")
    for (xs, xe, owner, color) in ownership_intervals(p):
        ax.add_patch(Rectangle((xs, row_y), xe - xs, page_h,
                               facecolor=color, alpha=0.85,
                               edgecolor="#222222", linewidth=0.4))
        if xe - xs > 1.5 and owner in ("A", "B"):
            ax.text((xs + xe) / 2, row_y + page_h / 2, owner,
                    ha="center", va="center", fontsize=9.5, weight="bold",
                    color="white")
        elif xe - xs > 2.5 and owner == "free":
            ax.text((xs + xe) / 2, row_y + page_h / 2, "free",
                    ha="center", va="center", fontsize=8.5,
                    color="#666666", style="italic")

# v2 event markers below the grid
def mark_event(x, label, color, dy):
    top = V2_GRID_Y0 + PAGE_GRID_HEIGHT + 0.05
    bot = V2_GRID_Y0 - 0.05
    ax.plot([x, x], [bot, top], color=color, linewidth=1.0,
            linestyle="--", alpha=0.65)
    ax.text(x, bot - dy, label, ha="center", va="top",
            fontsize=8.5, weight="bold", color=color)

mark_event(T_A_LOADER_PREFIX, "A prefix: release p4..p7", COLOR_FRESH, dy=0.30)
mark_event(T_V2_B_START,      "B starts: claim p4..p7",   COLOR_FRESH, dy=0.85)
mark_event(T_A_STAGE0_REL,    "A s0: release p0,p1",      COLOR_HANDOFF, dy=0.30)
mark_event(T_A_STAGE1_REL,    "A s1: release p2,p3",      COLOR_HANDOFF, dy=0.85)

# ── Finish-time markers on BOTH panels ─────────────────────────────────
# v1 finish line at T_V1_B_END
v1_grid_top = V1_GRID_Y0 + PAGE_GRID_HEIGHT
ax.plot([T_V1_B_END, T_V1_B_END],
        [V1_GRID_Y0 - 0.15, v1_grid_top + 0.15],
        color=COLOR_FRAME, linewidth=2.2)
ax.text(T_V1_B_END + 0.15, v1_grid_top + 0.25,
        "v1 done",
        ha="left", va="bottom", fontsize=10, weight="bold",
        color=COLOR_FRAME)

# v2 finish line at T_V2_B_END
v2_grid_top = V2_GRID_Y0 + PAGE_GRID_HEIGHT
ax.plot([T_V2_B_END, T_V2_B_END],
        [V2_GRID_Y0 - 0.15, v2_grid_top + 0.15],
        color=COLOR_FRAME, linewidth=2.2)
ax.text(T_V2_B_END + 0.15, v2_grid_top + 0.25,
        "v2 done",
        ha="left", va="bottom", fontsize=10, weight="bold",
        color=COLOR_FRAME)

# Time-savings arrow at the very top — connecting v2 done to v1 done
SAVINGS_Y = v1_grid_top + 1.4
ax.annotate("",
            xy=(T_V1_B_END, SAVINGS_Y),
            xytext=(T_V2_B_END, SAVINGS_Y),
            arrowprops=dict(arrowstyle="<->", color="#55A868",
                            lw=2.0))
ax.text((T_V2_B_END + T_V1_B_END) / 2, SAVINGS_Y + 0.18,
        f"v2 saves {T_V1_B_END - T_V2_B_END:.0f} units (~{(T_V1_B_END - T_V2_B_END) / (T_V1_B_END - T_A_START) * 100:.0f}%)",
        ha="center", va="bottom", fontsize=10, weight="bold",
        color="#55A868")

# ── Overlap callout — green tinted band, label in legend ──────────────
overlap_x0 = T_V2_B_START
overlap_x1 = T_A_END
ax.add_patch(Rectangle((overlap_x0, V2_GRID_Y0 - 0.05),
                       overlap_x1 - overlap_x0,
                       PAGE_GRID_HEIGHT + 0.10,
                       facecolor="#55A868", alpha=0.10,
                       edgecolor="#55A868", linewidth=1.8, linestyle="--"))

# ── Legend ─────────────────────────────────────────────────────────────
LEG_Y = -1.4
items = [
    (COLOR_A, "owned by A"),
    (COLOR_B, "owned by B"),
    (COLOR_FREE, "free (awaiting claim)"),
    ("#9fb8d8", "released by A's loader prefix (v2)"),
    ("#cbd5e6", "reserved but unused (v1 waste)"),
    (COLOR_CONTROLLER, "controller publish event"),
    ("#55A868", "cross-task overlap (v2 only)"),
]
xx = -2.0
for color, name in items:
    ax.add_patch(Rectangle((xx, LEG_Y), 0.45, 0.30,
                           facecolor=color, alpha=0.85,
                           edgecolor="#222222"))
    ax.text(xx + 0.55, LEG_Y + 0.15, name,
            fontsize=9, va="center", color=COLOR_FRAME)
    xx += 6.0

# Future-opt callout
ax.text(-2.0, LEG_Y - 0.6,
        "Future opt: the controller could pre-issue TMA descriptors for B's fresh pages while still publishing the task,"
        " removing the loader-wake latency.",
        fontsize=9, style="italic", color=COLOR_CONTROLLER)

plt.tight_layout()
out = "/home/xinhaoc/mirage-refactor/docs/mpk/figures/fig4_smem_pipeline.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
