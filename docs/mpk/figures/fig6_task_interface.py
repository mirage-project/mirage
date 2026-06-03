"""Figure 6: Task interface — how to declare a new v2 task.

A new task is registered with TWO pieces of metadata plus 5 role-function
bodies. This figure shows the API surface, where each piece flows, and
what the runtime does with it.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 12.5))
ax.set_xlim(-0.5, 22)
ax.set_ylim(-0.5, 18)
ax.axis("off")

COLOR_FRAME = "#3a3a3a"
COLOR_SPEC = "#4C72B0"        # blue — declarative spec
COLOR_BODY = "#C44E52"        # red — role function bodies
COLOR_CODEGEN = "#CCB974"     # yellow — codegen / planner
COLOR_RUNTIME = "#55A868"     # green — runtime output
COLOR_TEXT_DARK = "#222222"
COLOR_CODE_BG = "#f5f5f5"

# Title
ax.text(-0.5, 17.5,
        "Task interface — declaring a new v2 task",
        fontsize=15, weight="bold", color=COLOR_FRAME)
ax.text(-0.5, 16.95,
        "Three inputs from the user. The framework generates the dispatch & init code.",
        fontsize=11, color=COLOR_FRAME, style="italic")

# ════════════════════════════════════════════════════════════════════
# LEFT: User inputs (3 pieces)
# ════════════════════════════════════════════════════════════════════
USER_X = 0.0
USER_W = 10.0

ax.text(USER_X, 16.0,
        "USER INPUT",
        fontsize=11, weight="bold", color=COLOR_SPEC)

# ── (1) TaskSmemInfo ─────────────────────────────────────────────────
y_top = 15.5
y_bot = 11.0
ax.add_patch(FancyBboxPatch(
    (USER_X, y_bot), USER_W, y_top - y_bot,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor=COLOR_CODE_BG, edgecolor=COLOR_SPEC, linewidth=1.5))
ax.text(USER_X + 0.2, y_top - 0.30,
        "(1)  TaskSmemInfo  — declarative SMEM region spec",
        fontsize=11, weight="bold", color=COLOR_SPEC)

spec_lines = [
    "TaskSmemInfo info {",
    "    .size      = total_bytes;",
    "    .alignment = 1024;",
    "    .regions   = {",
    "        // 6 W regions, 32 KB each, NOT packable",
    "        {\"W_0\", 32KB, can_pack=false, release_stage=0},",
    "        ... (W_1..W_5)",
    "        // 6 A regions, 4 KB each, packable",
    "        {\"A_0\",  4KB, can_pack=true,  release_stage=0},",
    "        ... (A_1..A_5)",
    "        {\"scratch\", 16B, can_pack=true, release_stage=5},",
    "    };",
    "};",
]
for i, ln in enumerate(spec_lines):
    ax.text(USER_X + 0.3, y_top - 0.75 - i * 0.28, ln,
            fontsize=8.5, family="monospace", color=COLOR_TEXT_DARK)

# ── (2) TaskRoleVariantCode ──────────────────────────────────────────
y_top = 10.5
y_bot = 7.5
ax.add_patch(FancyBboxPatch(
    (USER_X, y_bot), USER_W, y_top - y_bot,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor=COLOR_CODE_BG, edgecolor=COLOR_BODY, linewidth=1.5))
ax.text(USER_X + 0.2, y_top - 0.30,
        "(2)  TaskRoleVariantCode  — role bodies + codegen flags",
        fontsize=11, weight="bold", color=COLOR_BODY)

role_lines = [
    "TaskRoleVariantCode code {",
    "    .init_semaphores = \"mbar_init(...) x 24 op-sems\";",
    "    .loader          = \"linear_loader_task<...>(...)\";",
    "    .launcher        = \"linear_launcher_task<...>(...)\";",
    "    .consumer        = \"linear_consumer_task<...>(...)\";",
    "    .storer          = \"\";     // empty for linear",
    "    .auto_loader_page_lifecycle = true;",
    "    .auto_consumer_finish       = false;  // launcher releases pages",
    "};",
]
for i, ln in enumerate(role_lines):
    ax.text(USER_X + 0.3, y_top - 0.75 - i * 0.28, ln,
            fontsize=8.5, family="monospace", color=COLOR_TEXT_DARK)

# ── (3) Role function bodies ─────────────────────────────────────────
y_top = 7.0
y_bot = 4.0
ax.add_patch(FancyBboxPatch(
    (USER_X, y_bot), USER_W, y_top - y_bot,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor=COLOR_CODE_BG, edgecolor=COLOR_BODY, linewidth=1.5))
ax.text(USER_X + 0.2, y_top - 0.30,
        "(3)  Role function bodies  — __device__ functions referenced above",
        fontsize=11, weight="bold", color=COLOR_BODY)

body_lines = [
    "__device__ void linear_loader_task(TaskDesc*, ...) {",
    "    // TMA load W, dep wait inline, TMA load A",
    "}",
    "__device__ void linear_launcher_task(TaskDesc*, ...) {",
    "    // tcgen05.alloc + MMA loop + per-stage release + dealloc",
    "}",
    "__device__ void linear_consumer_task(TaskDesc*, ...) {",
    "    // wait tmem_ready, tcgen05.ld, epilogue store, arrive cons_done",
    "}",
]
for i, ln in enumerate(body_lines):
    ax.text(USER_X + 0.3, y_top - 0.65 - i * 0.30, ln,
            fontsize=8.5, family="monospace", color=COLOR_TEXT_DARK)

# ── Registration call ────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (USER_X, 2.6), USER_W, 1.0,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor="#fff0d6", edgecolor=COLOR_CODEGEN, linewidth=1.5))
ax.text(USER_X + USER_W / 2, 3.1,
        "register_v2_task_role_variant(TASK_LINEAR_SM100_V2, variant, code);",
        ha="center", va="center", fontsize=10, weight="bold",
        family="monospace", color=COLOR_TEXT_DARK)

# ════════════════════════════════════════════════════════════════════
# CENTER: Arrow showing flow into codegen/planner
# ════════════════════════════════════════════════════════════════════
CENTER_X = 11.5
ax.annotate("", xy=(CENTER_X + 1.5, 9.0), xytext=(CENTER_X - 0.5, 9.0),
            arrowprops=dict(arrowstyle="->", lw=2.5, color=COLOR_CODEGEN))
ax.text(CENTER_X + 0.5, 10.0,
        "codegen +",
        ha="center", va="center", fontsize=11, weight="bold",
        color=COLOR_CODEGEN)
ax.text(CENTER_X + 0.5, 9.55,
        "planner",
        ha="center", va="center", fontsize=11, weight="bold",
        color=COLOR_CODEGEN)
ax.text(CENTER_X + 0.5, 8.5,
        "(build time)",
        ha="center", va="center", fontsize=8.5, style="italic",
        color=COLOR_CODEGEN)

# ════════════════════════════════════════════════════════════════════
# RIGHT: What the framework generates
# ════════════════════════════════════════════════════════════════════
GEN_X = 13.5
GEN_W = 8.0

ax.text(GEN_X, 16.0,
        "FRAMEWORK OUTPUT",
        fontsize=11, weight="bold", color=COLOR_RUNTIME)

# Output (1): physical page assignment
y_top = 15.5
y_bot = 13.5
ax.add_patch(FancyBboxPatch(
    (GEN_X, y_bot), GEN_W, y_top - y_bot,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor=COLOR_CODE_BG, edgecolor=COLOR_RUNTIME, linewidth=1.5))
ax.text(GEN_X + 0.2, y_top - 0.30,
        "(A)  Physical page assignment (from TaskSmemInfo)",
        fontsize=10.5, weight="bold", color=COLOR_RUNTIME)
out1_lines = [
    "Each region → SmemPageRegionDesc {",
    "    physical_page_start, page_count, byte_offset",
    "}",
    "stored in TaskDesc.smem_regions[]",
]
for i, ln in enumerate(out1_lines):
    ax.text(GEN_X + 0.3, y_top - 0.65 - i * 0.30, ln,
            fontsize=8.5, family="monospace", color=COLOR_TEXT_DARK)

# Output (2): per-warp dispatch functions
y_top = 13.0
y_bot = 10.5
ax.add_patch(FancyBboxPatch(
    (GEN_X, y_bot), GEN_W, y_top - y_bot,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor=COLOR_CODE_BG, edgecolor=COLOR_RUNTIME, linewidth=1.5))
ax.text(GEN_X + 0.2, y_top - 0.30,
        "(B)  Per-warp dispatch (from TaskRoleVariantCode)",
        fontsize=10.5, weight="bold", color=COLOR_RUNTIME)
out2_lines = [
    "_execute_loader_task_v2(task, ...)    { case 246: ... }",
    "_execute_launcher_task_v2(task, ...)  { case 246: ... }",
    "_execute_consumer_task_v2(task, ...)  { case 246: ... }",
    "_execute_storer_task_v2(task, ...)    { case 246: <empty> }",
    "_execute_init_semaphores_v2(...)      { case 246: ... }",
]
for i, ln in enumerate(out2_lines):
    ax.text(GEN_X + 0.3, y_top - 0.70 - i * 0.32, ln,
            fontsize=8.5, family="monospace", color=COLOR_TEXT_DARK)

# Output (3): codegen-wrapped role bodies
y_top = 10.0
y_bot = 7.0
ax.add_patch(FancyBboxPatch(
    (GEN_X, y_bot), GEN_W, y_top - y_bot,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor=COLOR_CODE_BG, edgecolor=COLOR_RUNTIME, linewidth=1.5))
ax.text(GEN_X + 0.2, y_top - 0.30,
        "(C)  Auto-wrapped role bodies (from flags)",
        fontsize=10.5, weight="bold", color=COLOR_RUNTIME)
out3_lines = [
    "loader body wrapped with:  if (auto_loader_page_lifecycle)",
    "    WAIT page_finished[0..13];  release pages task doesn't use",
    "    (cross-task page protocol prefix)",
    "",
    "consumer body wrapped with:  if (auto_consumer_finish)",
    "    arrive page_finished[p] for pages this task uses",
    "    (linear's flag = false → no suffix; launcher handles)",
]
for i, ln in enumerate(out3_lines):
    ax.text(GEN_X + 0.3, y_top - 0.65 - i * 0.30, ln,
            fontsize=8.5, family="monospace", color=COLOR_TEXT_DARK)

# Output (4): kernel persistent loop
y_top = 6.5
y_bot = 3.5
ax.add_patch(FancyBboxPatch(
    (GEN_X, y_bot), GEN_W, y_top - y_bot,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor=COLOR_CODE_BG, edgecolor=COLOR_RUNTIME, linewidth=1.5))
ax.text(GEN_X + 0.2, y_top - 0.30,
        "(D)  Runtime dispatch in worker_v2_kernel",
        fontsize=10.5, weight="bold", color=COLOR_RUNTIME)
out4_lines = [
    "worker_v2_kernel<<<dim3(num_SMs), dim3(256), ...>>>:",
    "    if (warp_id < 4)              consumer_warp_loop(...)",
    "    else if (warp_id == 4)        loader_warp_loop(...)",
    "    else if (warp_id == 5)        launcher_warp_loop(...)",
    "    else if (warp_id == 6)        storer_warp_loop(...)",
    "    else if (warp_id == 7)        controller_warp_loop(...)",
    "                                  → dispatches to (B) above",
]
for i, ln in enumerate(out4_lines):
    ax.text(GEN_X + 0.3, y_top - 0.65 - i * 0.30, ln,
            fontsize=8.5, family="monospace", color=COLOR_TEXT_DARK)

# ════════════════════════════════════════════════════════════════════
# Footer: code locations
# ════════════════════════════════════════════════════════════════════
ax.text(-0.5, 2.0,
        "Where each piece lives in the codebase:",
        fontsize=11, weight="bold", color=COLOR_FRAME)
ax.text(-0.5, 1.55,
        "  (1) TaskSmemInfo                : include/mirage/persistent_kernel/tasks/blackwell_v2/<task>_spec.h    (e.g. linear_sm100_v2_spec.h)",
        fontsize=9, color=COLOR_FRAME, family="monospace")
ax.text(-0.5, 1.15,
        "  (2) TaskRoleVariantCode wiring  : src/kernel/task_register.cc   (e.g. register_linear_sm100_v2_task)",
        fontsize=9, color=COLOR_FRAME, family="monospace")
ax.text(-0.5, 0.75,
        "  (3) Role function bodies        : include/mirage/persistent_kernel/tasks/blackwell_v2/<task>_sm100.cuh",
        fontsize=9, color=COLOR_FRAME, family="monospace")
ax.text(-0.5, 0.35,
        "  (A,B,C,D) framework output      : src/kernel/v2_role_codegen.cc + include/mirage/persistent_kernel/runtime_v2.cuh",
        fontsize=9, color=COLOR_FRAME, family="monospace")

plt.tight_layout()
out = "/home/xinhaoc/mirage-refactor/docs/mpk/figures/fig6_task_interface.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"saved {out}")
