"""Figure 5: Role-function coordination for one linear task instance.

Bigger fonts, more whitespace, fewer items per lane. Shows the actual
wait order: cross-task page wait FIRST (codegen prefix), then dep-wait,
then the MMA cycle (where mma_mbar is the loader's refill gate — but
only blocks from iter 6 onwards; first 6 iters pass through via init
parity).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 16))
ax.set_xlim(-1.0, 22)
ax.set_ylim(-1.0, 28)
ax.axis("off")

# Colors
COLOR_FRAME = "#3a3a3a"
COLOR_CONSUMER = "#4C72B0"
COLOR_LOADER = "#55A868"
COLOR_LAUNCHER = "#C44E52"
COLOR_CONTROLLER = "#CCB974"
COLOR_TEXT_DARK = "#222222"
COLOR_WAIT = "#dceaf6"
COLOR_ARRIVE = "#fde6cc"
COLOR_PAGE = "#a8d5ba"      # page-related (cross-task)
COLOR_INLINE_DEP = "#3a8a4d"

# Title
ax.text(-1.0, 27.4,
        "Role-function coordination — one linear task instance",
        fontsize=15, weight="bold", color=COLOR_FRAME)
ax.text(-1.0, 26.85,
        "WAIT (blue) and ARRIVE (orange) shown explicitly. Time flows top to bottom. "
        "The 32-iter MMA loop is shown collapsed once.",
        fontsize=11, color=COLOR_FRAME, style="italic")

# Swimlanes
LANES = [
    ("Controller W7",         1.7,  COLOR_CONTROLLER, COLOR_TEXT_DARK),
    ("Loader W4",             6.3,  COLOR_LOADER,     "white"),
    ("Launcher W5",          11.7,  COLOR_LAUNCHER,   "white"),
    ("Consumer W0–W3",       17.0,  COLOR_CONSUMER,   "white"),
]
LANE_TOP = 25.4
LANE_BOT = 1.0
LANE_W = 4.0

for name, x, color, text_color in LANES:
    ax.add_patch(FancyBboxPatch(
        (x - LANE_W / 2, LANE_TOP), LANE_W, 0.85,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=color, alpha=0.95,
        edgecolor="#222222", linewidth=1.0))
    ax.text(x, LANE_TOP + 0.42, name,
            ha="center", va="center", fontsize=12, weight="bold",
            color=text_color)
    ax.add_patch(Rectangle((x - LANE_W / 2, LANE_BOT),
                           LANE_W, LANE_TOP - LANE_BOT,
                           facecolor=color, alpha=0.05,
                           edgecolor="#cccccc", linewidth=0.4))

def lane_x(idx):
    return LANES[idx][1]

def op_box(lane_idx, y, label, h=0.9, w=None, kind=None,
           text_color=None, subtitle=None, fill=None, edge=None):
    x = lane_x(lane_idx)
    lane_color = LANES[lane_idx][2]
    if kind == "wait":
        fill = fill or COLOR_WAIT
        text_color = text_color or COLOR_TEXT_DARK
        edge = edge or "#2a4a6a"
    elif kind == "arrive":
        fill = fill or COLOR_ARRIVE
        text_color = text_color or COLOR_TEXT_DARK
        edge = edge or "#9a5a1a"
    elif kind == "page":
        fill = fill or COLOR_PAGE
        text_color = text_color or COLOR_TEXT_DARK
        edge = edge or "#2a6a4a"
    elif kind == "inline_dep":
        fill = fill or COLOR_INLINE_DEP
        text_color = text_color or "white"
        edge = edge or "#1a4a2a"
    else:
        fill = fill or lane_color
        text_color = text_color or ("white" if lane_idx != 0 else COLOR_TEXT_DARK)
        edge = edge or lane_color
    if w is None:
        w = LANE_W - 0.4
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=fill, edgecolor=edge, linewidth=1.6))
    if subtitle:
        ax.text(x, y + 0.15, label,
                ha="center", va="center", fontsize=10.5, weight="bold",
                color=text_color)
        ax.text(x, y - 0.22, subtitle,
                ha="center", va="center", fontsize=8.5,
                color=text_color, family="monospace", style="italic")
    else:
        ax.text(x, y, label,
                ha="center", va="center", fontsize=10.5, weight="bold",
                color=text_color)

def mbar_arrow(src_lane, src_y, dst_lane, dst_y, label=None,
               color=COLOR_FRAME, rad=0.0, linewidth=1.5,
               label_dy=0.20):
    src_x = lane_x(src_lane) + (LANE_W / 2 if dst_lane > src_lane else -LANE_W / 2)
    dst_x = lane_x(dst_lane) - (LANE_W / 2 if dst_lane > src_lane else -LANE_W / 2)
    ax.add_patch(FancyArrowPatch(
        (src_x, src_y), (dst_x, dst_y),
        arrowstyle="->", mutation_scale=16,
        color=color, linewidth=linewidth,
        connectionstyle=f"arc3,rad={rad}"
    ))
    if label:
        mid_x = (src_x + dst_x) / 2
        mid_y = (src_y + dst_y) / 2 + label_dy
        ax.text(mid_x, mid_y, label,
                ha="center", va="center", fontsize=9, weight="bold",
                color=color, family="monospace",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor=color, linewidth=0.7))

# ════════════════════════════════════════════════════════════════════
#  warp_loop wrapper — start
# ════════════════════════════════════════════════════════════════════
y = 24.5
op_box(0, y, "publish + init_sem",
       subtitle="(controller)", h=0.85)
op_box(1, y, "WAIT", kind="wait",
       subtitle="task_metadata_ready", h=0.85)
op_box(2, y, "WAIT", kind="wait",
       subtitle="task_metadata_ready", h=0.85)
op_box(3, y, "WAIT", kind="wait",
       subtitle="task_metadata_ready", h=0.85)
mbar_arrow(0, y, 1, y, "task_metadata_ready\n(warp_loop)",
           rad=-0.04)
mbar_arrow(0, y, 2, y, "", rad=-0.02)
mbar_arrow(0, y, 3, y, "", rad=0.02)

# ════════════════════════════════════════════════════════════════════
#  Cross-task page wait (loader codegen prefix) + dep wait
# ════════════════════════════════════════════════════════════════════
y = 22.7
ax.text(11, y + 0.5,
        "── cross-task setup ──",
        ha="center", va="center", fontsize=11, weight="bold",
        color=COLOR_FRAME)

y = 21.7
# Loader: WAIT page_finished (codegen prefix, lane-parallel × 14 pages)
op_box(1, y, "WAIT page_finished[0..13]", kind="page",
       subtitle="codegen prefix, lane-parallel\n(cross-task page handoff)",
       h=1.2)

# Consumer T0: wait upstream producer event
op_box(3, y, "T0: wait producer event", kind="wait",
       subtitle="spin on GMEM event counter\nset by an upstream SM's task",
       h=1.2)

# Launcher: wait SEM_DEP_READY
op_box(2, y, "WAIT", kind="wait",
       subtitle="SEM_DEP_READY",
       h=1.2)

# Arrow consumer T0 → launcher's SEM_DEP_READY
mbar_arrow(3, y, 2, y, "SEM_DEP_READY",
           color=COLOR_CONSUMER, rad=0.0)

y = 20.2
# Consumer T0: ARRIVE SEM_DEP_READY (after its spin)
# (Conceptually shown inside the T0 box above; just add a small label)

# ════════════════════════════════════════════════════════════════════
#  TMEM alloc / read
# ════════════════════════════════════════════════════════════════════
y = 19.8
op_box(2, y, "alloc TMEM", kind="arrive",
       subtitle="tcgen05.alloc → tmem_ready", h=0.9)
op_box(3, y, "read taddr", kind="wait",
       subtitle="WAIT tmem_ready", h=0.9)
mbar_arrow(2, y, 3, y, "tmem_ready", rad=0.0)

# ════════════════════════════════════════════════════════════════════
#  MMA loop (collapsed view of one cycle)
# ════════════════════════════════════════════════════════════════════
LOOP_Y_TOP = 18.5
LOOP_Y_BOT = 9.3

ax.plot([-0.5, -0.5], [LOOP_Y_BOT, LOOP_Y_TOP],
        color=COLOR_FRAME, linewidth=2.8)
ax.plot([-0.5, -0.2], [LOOP_Y_TOP, LOOP_Y_TOP],
        color=COLOR_FRAME, linewidth=2.8)
ax.plot([-0.5, -0.2], [LOOP_Y_BOT, LOOP_Y_BOT],
        color=COLOR_FRAME, linewidth=2.8)
ax.text(-0.85, (LOOP_Y_TOP + LOOP_Y_BOT) / 2,
        "MMA loop\n× 32 iters\nstage K = i % 6",
        rotation=90, ha="center", va="center", fontsize=10.5,
        weight="bold", color=COLOR_FRAME)

ax.text(11, 18.1, "── MMA pipeline cycle (one iter shown) ──",
        ha="center", va="center", fontsize=11, weight="bold",
        color=COLOR_FRAME)

y = 17.2
# Loader: WAIT mma_mbar[K] (refill back-edge)
op_box(1, y, "WAIT mma_mbar[K]", kind="wait",
       subtitle="refill back-edge\n(no-op on iter 0..5)",
       h=1.1)

y = 15.7
# Loader: TMA W
op_box(1, y, "TMA W + ARRIVE", kind="arrive",
       subtitle="W_tma_mbar[K]", h=1.0)
op_box(2, y, "WAIT", kind="wait",
       subtitle="W_tma_mbar[K]", h=1.0)
mbar_arrow(1, y, 2, y, "W_tma_mbar[K]", rad=0.0)

y = 14.1
# Loader: inline dep wait (first iter only)
op_box(1, y, "wait producer event INLINE", kind="inline_dep",
       subtitle="own GMEM-event spin,\nfirst iter only", h=1.0)

y = 12.6
# Loader: TMA A
op_box(1, y, "TMA A + ARRIVE", kind="arrive",
       subtitle="A_tma_mbar[K]", h=1.0)
op_box(2, y, "WAIT", kind="wait",
       subtitle="A_tma_mbar[K]", h=1.0)
mbar_arrow(1, y, 2, y, "A_tma_mbar[K]", rad=0.0)

y = 11.0
# Launcher: MMA + commit (arrives mma_mbar[K])
op_box(2, y, "MMA + ARRIVE", kind="arrive",
       subtitle="tcgen05.mma + commit → mma_mbar[K]",
       h=1.1)

# Back-edge from launcher MMA → loader's WAIT mma_mbar at top of loop
mbar_arrow(2, y, 1, 17.2, "mma_mbar[K]  back-edge",
           color=COLOR_LAUNCHER, rad=-0.45, label_dy=0)

# Per-stage page release IN the launcher (Phase 5 final design):
# after tcgen05.commit, if this was stage K's last fire, arrive
# page_finished[p] for the pages stage K owned. Enables cross-task
# weight prefetch.
y = 10.0
op_box(2, y, "if stage K last fire:\nRELEASE page_finished[K]",
       kind="arrive",
       fill="#a8d5ba", edge="#2a6a4a",
       subtitle="(in launcher — final Phase 5 design)",
       h=0.95)

# ════════════════════════════════════════════════════════════════════
#  After MMA loop tail
# ════════════════════════════════════════════════════════════════════
ax.text(11, 8.9, "── end of MMA loop ──",
        ha="center", va="center", fontsize=11, weight="bold",
        color=COLOR_FRAME)

y = 8.0
op_box(2, y, "tcgen05.commit", kind="arrive",
       subtitle="mainloop_mbar", h=0.9)
op_box(3, y, "WAIT mainloop", kind="wait",
       subtitle="mainloop_mbar", h=0.9)
mbar_arrow(2, y, 3, y, "mainloop_mbar", rad=0.0)

y = 6.7
op_box(3, y, "tcgen05.ld + epilogue",
       subtitle="TMEM → reg → HBM", h=0.9)

y = 5.4
op_box(3, y, "ARRIVE × 128", kind="arrive",
       subtitle="consumer_done", h=0.9)
op_box(2, y, "WAIT consumer_done", kind="wait",
       subtitle="(blocks dealloc until\nTMEM is no longer read)", h=0.9)
mbar_arrow(3, y, 2, y, "consumer_done", rad=0.0)

y = 4.0
op_box(2, y, "tcgen05.dealloc",
       subtitle="(only safe after\nconsumer_done)", h=0.9)

# ════════════════════════════════════════════════════════════════════
#  warp_loop wrapper — end
# ════════════════════════════════════════════════════════════════════
ax.text(11, 3.0,
        "▲ task body returns → warp_loop arrives task_done ▲",
        ha="center", va="center", fontsize=10, weight="bold",
        style="italic", color="#888888")

y = 2.3
op_box(1, y, "ARRIVE", kind="arrive",
       subtitle="task_done", h=0.7)
op_box(2, y, "ARRIVE", kind="arrive",
       subtitle="task_done", h=0.7)
op_box(3, y, "ARRIVE × 4 warps", kind="arrive",
       subtitle="task_done", h=0.7)
op_box(0, y, "WAIT × 7", kind="wait",
       subtitle="task_done\n(count = NUM_ROLE_WARPS)", h=0.7)
mbar_arrow(1, y, 0, y, "", rad=-0.06)
mbar_arrow(2, y, 0, y, "", rad=-0.03)
mbar_arrow(3, y, 0, y, "", rad=0.0)

y = 1.3
op_box(0, y, "trigger events,\nrecycle slot", h=0.7)

# Legend
LEG_Y = 0.3
items = [
    (COLOR_WAIT,        "#2a4a6a",  "WAIT mbar"),
    (COLOR_ARRIVE,      "#9a5a1a",  "ARRIVE mbar"),
    ("#a8d5ba",         "#2a6a4a",  "cross-task page op (wait or release)"),
    (COLOR_INLINE_DEP,  "#1a4a2a",  "loader inline upstream-event wait"),
]
xx = -1.0
for fill, edge, name in items:
    ax.add_patch(Rectangle((xx, LEG_Y), 0.55, 0.30,
                           facecolor=fill, edgecolor=edge, linewidth=1.0))
    ax.text(xx + 0.7, LEG_Y + 0.15, name,
            fontsize=10, va="center", color=COLOR_FRAME)
    xx += 5.5

plt.tight_layout()
out = "/home/xinhaoc/mirage-refactor/docs/mpk/figures/fig5_task_split.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
