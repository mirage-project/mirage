"""Schedule files -> layer.cu -> built, loaded and given the tensors of every GPU.

A schedule file (schedule_gpu<g>.json, written by compiler.compile_plan) has:
  nodes                per graph node: name, grid, params, counter, buf, inputs (the node writing each input: its grid, counter,
                       buf, params; null: graph input)
  all_tasks            per task: node, pos (grid position), deps (the task ids it waits for)
  worker_task_queues   per SM, the task ids in the order the SM runs them
  info                 cluster_size (CTAs per cluster of the launch), gpu

layer.cu = five parts, each written by one function below:
  header_code     the kernel-wide sizes (tokens per step, GPUs, SMs: config.cuh), the exchange region's layout (from the layers:
                  StaticMegakernel.exchange_layout), StaticTask {node, x, y, z} (one entry of an SM's list), StaticNode /
                  StaticParams / StaticBufSlots / StaticMapSlots (a node's grid, counter, output buffer slot, params, own buffer
                  slots and map slots as template constants), the structs the host code receives; static_megakernel/core.cuh and
                  the used task types' files (tasks/<name>.cuh)
  smem_code       the launch's dynamic shared memory: the most any node's task type needs (its smem_<name>)
  task_tables     per GPU, the entries of every SM's list one after another, and where each SM's list begins
  kernel_code     layer_kernel: kernel_begin, a loop over this SM's entries with one `case` per node that calls the node's task
                  function static_mk::run_<name> with its node, params, buffer and map slots and its inputs' producers as template
                  arguments, kernel_end (core.cuh)
  host_code       HOST_CODE below: init (also the per-SM task tables on each GPU), one launch on every GPU, wait, the kernel's
                  instrumentation buffers by name (read by a test harness: static_harness.py), Python; the buffers and maps are
                  static_megakernel/host.cuh's, from the layers' slot args (host_slot_args)

  write_schedule(path, nodes, all_tasks, queues, info)      checks the lists cannot deadlock, writes the file
  generate_code(pk, schedules) -> str                       layer.cu
  compile_static(pk, schedule_paths, gpu_tensors, out_dir, extra_flags) -> StaticKernel
"""
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sysconfig
from typing import Dict, List, Optional

from . import gpu_backend


# ---------------- schedule file ----------------
def check_deadlock_free(all_tasks: List[dict], queues: List[List[int]], groups: List[List[int]] = ()) -> None:
    """Raise unless every task can run.
      1. every task is in exactly one SM's list
      2. put an edge a -> b for "b waits for a": b depends on a (deps), or a is right before b in an SM's list (the SM runs b
         only after a). The tasks of a concurrent group (they wait for each other while they run) count as one: they start only
         when every one of them can start. Remove, again and again, the tasks nothing waits on any more (a topological sort).
         If some tasks never get removed, they wait for each other in a circle: that plan would hang on the GPU."""
    num_tasks = len(all_tasks)
    rep_of = list(range(num_tasks))                  # a group's tasks -> its first task
    for group in groups:
        for t in group:
            rep_of[t] = group[0]
    seen = [0] * num_tasks
    for sm_list in queues:
        for t in sm_list:
            seen[t] += 1
    wrong = [t for t in range(num_tasks) if seen[t] != 1]
    if wrong:
        raise ValueError(f"tasks {wrong[:8]} are in {[seen[t] for t in wrong[:8]]} lists, must be in 1")

    edges = [(d, t) for t in range(num_tasks) for d in all_tasks[t]["deps"]]
    edges += [(sm_list[i - 1], sm_list[i]) for sm_list in queues for i in range(1, len(sm_list))]
    edges = {(rep_of[a], rep_of[b]) for a, b in edges if rep_of[a] != rep_of[b]}
    vertices = set(rep_of)
    successors: Dict[int, List[int]] = {}
    waiting_for = {v: 0 for v in vertices}
    for a, b in edges:
        successors.setdefault(a, []).append(b)
        waiting_for[b] += 1
    ready = [v for v in vertices if waiting_for[v] == 0]
    done = 0
    while ready:
        v = ready.pop()
        done += 1
        for s in successors.get(v, []):
            waiting_for[s] -= 1
            if waiting_for[s] == 0:
                ready.append(s)
    if done != len(vertices):
        stuck = [t for t in range(num_tasks) if waiting_for[rep_of[t]] > 0]
        per_node: Dict[int, int] = {}
        for t in stuck:
            per_node[all_tasks[t]["node"]] = per_node.get(all_tasks[t]["node"], 0) + 1
        first = sorted(per_node)[:6]
        # one circle among the stuck tasks (a walk backwards along "waits for" edges until a task repeats): each entry
        # (node, position, its SM list) and how it waits for the next (dep: a dependency, list: the task before it on its SM)
        sm_of = {t: i for i, sm_list in enumerate(queues) for t in sm_list}
        preds: Dict[int, List[tuple]] = {}
        for d_, t_ in [(d, t) for t in range(num_tasks) for d in all_tasks[t]["deps"]]:
            preds.setdefault(rep_of[t_], []).append((rep_of[d_], "dep"))
        for sm_list in queues:
            for i in range(1, len(sm_list)):
                preds.setdefault(rep_of[sm_list[i]], []).append((rep_of[sm_list[i - 1]], "list"))
        v, path, seen_at = rep_of[stuck[0]], [], {}
        while v not in seen_at:
            seen_at[v] = len(path)
            nxt = next(((p, how) for p, how in preds.get(v, []) if waiting_for[p] > 0 and p != v), None)
            if nxt is None:
                break
            path.append((v, nxt[1]))
            v = nxt[0]
        circle = path[seen_at.get(v, 0):][:12]
        desc = [(all_tasks[t]["node"], tuple(all_tasks[t]["pos"]), sm_of.get(t), how) for t, how in circle]
        raise ValueError(f"the task lists can deadlock: {len(stuck)} of {num_tasks} tasks wait for each other in a circle "
                         f"(stuck tasks per node, the lowest nodes: {[(n, per_node[n]) for n in first]}; a circle "
                         f"(node, pos, SM, waits for the next by): {desc})")


def write_schedule(path: str, nodes: Dict[str, dict], all_tasks: List[dict], queues: List[List[int]], info: dict,
                   groups: List[List[int]] = ()) -> None:
    check_deadlock_free(all_tasks, queues, groups)
    doc = {"nodes": nodes, "all_tasks": all_tasks, "worker_task_queues": queues, "info": info}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def code_tag() -> str:
    """The version of the kernel templates a cost record measured: the md5 of the backend's include/mirage/<template_dir>/*.cuh and
    tasks/*.cuh (gpu_backend; first 12 hex digits). the cost records carry it ("code"); search.Costs(code=code_tag()) uses only the records of these templates,
    so times measured with other kernel code are not mixed in."""
    from ..kernel import get_key_paths
    _, include_path, _ = get_key_paths()
    h = hashlib.md5()
    root = os.path.join(include_path, "mirage", gpu_backend.template_dir())
    for path in sorted(glob.glob(os.path.join(root, "*.cuh")) + glob.glob(os.path.join(root, "tasks", "*.cuh"))):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


# ---------------- code generation ----------------
THREADS = 256   # one CTA of 256 threads per SM (runtime.cuh: the warp roles)
MAX_TASKS_PER_SM = 64   # entries of an SM's task table (config.cuh): its list, then an end entry
# the device's includes per backend (gpu_backend): the HIP templates map the CUDA runtime names the host code uses (cuda_names.h)
BACKEND_INCLUDES = {"cuda": ["#include <cuda.h>", "#include <cuda_runtime.h>"],
                    "hip": ["#include <hip/hip_runtime.h>", '#include "mirage/static_megakernel_hip/cuda_names.h"']}
# the timing build's clock per backend: ns (globaltimer; s_memrealtime ticks x gpu_backend.ns_per_tick)
STATIC_NOW = {"cuda": "asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t));",
              "hip": "t = wall_clock64() * STATIC_NS_PER_TICK;"}


def waves() -> int:
    """Warps (CUDA) or wavefronts (HIP) of a CTA."""
    return THREADS // gpu_backend.wave()


def stage_slot() -> int:
    """Timing build, per list entry: start, each warp's end, then 2 stage stamps (KernelLocals::stage_stamps)."""
    return 1 + waves()


def time_slots() -> int:
    return stage_slot() + 2


def config_checks(pk, used: List[str]) -> List[str]:
    """static_asserts that the Python copies of the kernel's sizes (here and in static_megakernel.py) are config.cuh's and the
    used task types'."""
    from .static_megakernel import MAX_BUFS, MAX_MAPS, MAX_SUM_PARTS, NODE_COUNTER_LINES, RING_STAGES, SF_CHUNK
    checks = [f"static_mk::MAX_TASKS_PER_SM == {MAX_TASKS_PER_SM}", f"static_mk::MAX_BUFS == {pk.buf_slots()}",
              f"static_mk::MAX_MAPS == {pk.map_slots()}", f"static_mk::NODE_COUNTER_LINES == {pk.counter_lines()}",
              f"static_mk::SMAX == {RING_STAGES}", f"static_mk::SF_CHUNK == {SF_CHUNK}"]
    if "sum_send" in used:
        checks.append(f"static_mk::MAX_SUM_PARTS == {MAX_SUM_PARTS}")
    return [f"static_assert({' && '.join(checks)}, \"the Python copies of the kernel's sizes (static_schedule.py, "
            'static_megakernel.py)");']


def header_code(pk, nodes: Dict[str, dict], num_gpus: int, num_lists: int, profile: bool = False) -> List[str]:
    set_bytes, hello, offsets, sizes = pk.exchange_layout()
    used = sorted({n["name"] for n in nodes.values()})
    tdir = gpu_backend.template_dir()
    return [f"// layer.cu -- generated by mirage.mpk.static_schedule.compile_static from the graph and {num_gpus} "
            f"schedule.json files"] + (["#define STATIC_TIMING_BUILD 1   // timing hooks (e.g. GEMM stage stamps)"] if profile else []) + [
            *BACKEND_INCLUDES[gpu_backend.backend()], "#include <initializer_list>", "#include <map>",
            "#include <string>", "#include <vector>", "",
            "// the kernel-wide sizes (config.cuh): tokens per step and GPUs (the graph's: StaticMegakernel.tokens, num_gpus), SMs (the",
            "// schedules' task lists per GPU)",
            f"#define STATIC_TOKENS {pk.tokens}", f"#define STATIC_GPUS {num_gpus}", f"#define STATIC_SMS {num_lists}", "",
            "// the exchange region's layout (static_megakernel.py exchange_layout): one set's bytes, the start barrier's slots, per",
            "// buffer slot its offset and bytes (an exchange buffer; else 0)",
            f"#define STATIC_MAX_BUFS {pk.buf_slots()}", f"#define STATIC_MAX_MAPS {pk.map_slots()}",
            f"#define STATIC_COUNTER_LINES {pk.counter_lines()}",
            f"#define STATIC_EXCHANGE_SET {set_bytes}", f"#define STATIC_EXCHANGE_HELLO {hello}",
            f"#define STATIC_EXCHANGE_OFFSETS {{{', '.join(map(str, offsets))}}}",
            f"#define STATIC_EXCHANGE_BYTES {{{', '.join(map(str, sizes))}}}", "",
            "struct StaticTask { int node, x, y, z; };   // one entry of an SM's list: the graph node and the task's grid position",
            "// a node: its grid, its first counter (-1: none) and its output buffer slot (-1: none); as an input's producer also its",
            "// params (v, then a 0)",
            "template <int X, int Y, int Z, int C = -1, int B = -1, int... P> struct StaticNode {",
            "  static constexpr int x = X, y = Y, z = Z, counter = C, buf = B;",
            "  static constexpr int v[sizeof...(P) + 1] = {P..., 0};",
            "};",
            "template <int... P> struct StaticParams { static constexpr int v[sizeof...(P) + 1] = {P..., 0}; };   // a node's params",
            "template <int... S> struct StaticBufSlots { static constexpr int v[sizeof...(S) + 1] = {S..., -1}; };   // a node's buffer slots (G::buf)",
            "template <int... S> struct StaticMapSlots { static constexpr int v[sizeof...(S) + 1] = {S..., -1}; };   // a node's map slots (Maps::m)",
            "",
            "// what the host code sees of one GPU / of all GPUs",
            "struct StaticGpuView {",
            "  int gpu;                                  // CUDA device index",
            "  std::map<std::string, void *> tensors;    // the graph's tensors on this GPU, by name",
            "};",
            "struct StaticContext { int num_gpus = 0, num_lists = 0; std::vector<StaticGpuView> gpus; };",
            f'#include "mirage/{tdir}/core.cuh"'] + \
        [f'#include "mirage/{tdir}/tasks/{name}.cuh"' for name in used] + config_checks(pk, used) + [""]


def task_tables(schedules: List[dict]) -> List[str]:
    """Per GPU g two C++ arrays:
      static_tasks_gpu<g>       the entries of SM 0's list, then SM 1's, ...: {node, x, y, z}
      static_list_begin_gpu<g>  SM k's entries are [begin[k], begin[k + 1])
    Example: SM 0 = [topk_route task 0, moe_experts task 0, allreduce_send task 0] -> {31, 0, 0, 0}, {36, 0, 0, 0}, {37, 0, 0, 0}; begin = {0, 3, ..}.
    static_init copies them into a per-SM table on each GPU (max_tasks entries per SM, the rest node = -1).
    Each list ends with an end entry node = -2 - (rank + 1024 E) (the loop stops at any node < 0): the SM's rank among the E SMs
    that hold no task of the graph's last node (1023: it holds one), for core.cuh kernel_end (the arena's resets)."""
    lines = []
    for g, schedule in enumerate(schedules):
        entries, begin = [], [0]
        last = max(int(k) for k in schedule["nodes"])
        holds = [any(schedule["all_tasks"][t]["node"] == last for t in sm_list) for sm_list in schedule["worker_task_queues"]]
        n_free = sum(1 for h in holds if not h)
        if n_free == 0:   # (every SM holds the last node, e.g. MoE's residual_add: all take slices)
            holds = [False] * len(holds)
            n_free = len(holds)
        rank = 0
        for sm_list, h in zip(schedule["worker_task_queues"], holds):
            for t in sm_list:
                task = schedule["all_tasks"][t]
                entries.append("{%d, %d, %d, %d}" % (task["node"], *task["pos"]))
            entries.append("{%d, 0, 0, 0}" % (-2 - ((1023 if h else rank) + 1024 * n_free)))
            rank += 0 if h else 1
            begin.append(len(entries))
        lines.append(f"static const StaticTask static_tasks_gpu{g}[] = {{{', '.join(entries)}}};")
        # (core.cuh STATIC_RESET_FIRST_NODE N: the next launch's exchange resets on the CUs whose list starts with node N; with
        # none on a GPU nobody would reset them)
        firsts = [schedule["all_tasks"][sm_list[0]]["node"] if sm_list else -1 for sm_list in schedule["worker_task_queues"]]
        lines += ["#ifdef STATIC_RESET_FIRST_NODE",
                  f"static_assert(static_mk::count_of(STATIC_RESET_FIRST_NODE, {{{', '.join(map(str, firsts))}}}) > 0, "
                  f"\"STATIC_RESET_FIRST_NODE: no CU of GPU {g} starts with that node\");",
                  "#endif"]
        lines.append(f"static const int static_list_begin_gpu{g}[] = {{{', '.join(map(str, begin))}}};")
    return lines


def task_args(node: dict) -> str:
    """A node's template arguments: the node (StaticNode: grid, counter, buffer slot), its params, its own buffer slots and map slots, and per input
    the node that writes it (with its params; StaticNode<0, 0, 0> for a graph input)."""
    def snode(grid, counter, buf, params=()):
        return "StaticNode<%s>" % ", ".join(str(v) for v in list(grid) + [counter, buf] + list(params))
    args = [snode(node["grid"], node["counter"], node["buf"]), "StaticParams<%s>" % ", ".join(str(p) for p in node["params"]),
            "StaticBufSlots<%s>" % ", ".join(str(v) for v in node["buf_slots"]),
            "StaticMapSlots<%s>" % ", ".join(str(v) for v in node["map_slots"])]
    args += [snode(i["grid"], i["counter"], i["buf"], i["params"]) if i else "StaticNode<0, 0, 0>" for i in node["inputs"]]
    return ", ".join(args)


def task_call(node: dict) -> str:
    """One node's call in the kernel loop: static_mk::run_<name> with its template arguments (task_args).
    static_mk::run_topk_route<StaticNode<8, 1, 1, -1, 2>, StaticParams<896, 16>, StaticBufSlots<1>, StaticMapSlots<>, StaticNode<7, 14, 1, -1, 0, 0, 7168, 896, 0>, StaticNode<0, 0, 0>>(maps, g, L, tk)"""
    return f"static_mk::run_{node['name']}<{task_args(node)}>(maps, g, L, tk)"


def smem_code(nodes: Dict[str, dict]) -> List[str]:
    """STATIC_SMEM_BYTES: the launch's dynamic shared memory, for the most any node's task type needs from the 1024-aligned base
    (static_mk::smem_<name> with the node's template arguments; config.cuh smem_launch_bytes adds the alignment)."""
    needs = [f"static_mk::smem_{node['name']}<{task_args(node)}>()" for _, node in sorted(nodes.items(), key=lambda kv: int(kv[0]))]
    return ["// the launch's dynamic shared memory: the most any node's task type needs (smem_<name>)",
            "constexpr int static_smem_need(std::initializer_list<int> v) { int m = 0; for (int x : v) m = x > m ? x : m; return m; }",
            "#ifdef STATIC_DRD_HOLD",   # (core.cuh STATIC_DRD_HOLD: the held recurrence state, 64 KB at STATIC_DRD_HOLD_OFF)
            "constexpr int STATIC_SMEM_BYTES = static_mk::smem_launch_bytes(static_smem_need({%s, STATIC_DRD_HOLD_OFF + 65536}));" % ", ".join(needs),
            "#else",
            "constexpr int STATIC_SMEM_BYTES = static_mk::smem_launch_bytes(static_smem_need({%s}));" % ", ".join(needs),
            "#endif",
            f'static_assert(STATIC_SMEM_BYTES <= {gpu_backend.smem_limit_kb()} * 1024, "the dynamic shared memory of one SM");', ""]


def noinline_tasks() -> set:
    """MPK_NOINLINE_TASKS (comma-separated task type names, e.g. moe_experts): those nodes' tasks as functions of their own
    (__noinline__), so the kernel's register allocation does not span them (a fused attention + MoE kernel: one function
    holding every task type spilled; MI355X 10-13)."""
    return {x for x in os.environ.get("MPK_NOINLINE_TASKS", "").split(",") if x}


def noinline_code(nodes: Dict[str, dict]) -> List[str]:
    """the wrappers of the noinline tasks: static_task_<idx>(maps, g, sm, sm_id[, stage_stamps], tk[, warm])"""
    out = []
    for idx, node in sorted(nodes.items(), key=lambda kv: int(kv[0])):
        if node["name"] not in noinline_tasks():
            continue
        warm = ", bool warm" if node["dry"] else ""
        call = task_call(node)[:-1] + (", warm)" if node["dry"] else ")")
        out += [f"__device__ __noinline__ void static_task_{idx}(static_mk::Maps const &maps, static_mk::G const &g, char *sm, int sm_id,",
                f"                                                  unsigned long long *stage_stamps, StaticTask const tk{warm}) {{",
                "  static_mk::KernelLocals L;",
                "  L.sm = sm;",
                "  L.sm_id = sm_id;",
                "#ifdef STATIC_TIMING_BUILD",
                "  L.stage_stamps = stage_stamps;",
                "#else",
                "  (void)stage_stamps;",
                "#endif",
                f"  {call};",
                "}"]
    return out


def case_code(idx: str, node: dict) -> str:
    """One node's case in the kernel loop. A node whose task type has a dry pass (its layer's `dry`): the task function gets
    `warm` last: false for the node's first task on this SM (its code is not in the instruction cache yet: the task first runs
    its own code on stand-in inputs, no global stores, while its inputs are on their way; the same instructions as the real pass),
    true for the later ones (kernel_code keeps one flag per such node). The flag is a register of the kernel loop, so deciding
    costs almost nothing (checking whether the inputs have already landed would cost loads and a CTA barrier per task).
      case 29: static_mk::run_topk_route<...>(maps, g, L, tk, warm_29); warm_29 = true; break;"""
    run = task_call(node)
    if node["name"] in noinline_tasks():
        run = f"static_task_{idx}(maps, g, L.sm, L.sm_id, STATIC_STAGE_STAMPS_OF(L), tk)"
    if not node["dry"]:
        return f"      case {idx}: {run}; break;"
    return f"      case {idx}: {run[:-1]}, warm_{idx}); warm_{idx} = true; break;"


def kernel_code(nodes: Dict[str, dict], profile: bool = False) -> List[str]:
    """layer_kernel:
        __global__ void __launch_bounds__(256, 1) layer_kernel(const __grid_constant__ static_mk::Maps maps, static_mk::G g, StaticTask const *static_tasks) {
          static_mk::KernelLocals L; static_mk::kernel_begin(g, L);
          StaticTask const *my = static_tasks + blockIdx.x * (static_mk::MAX_TASKS_PER_SM);   // this SM's list
          for (int ti = 0; ti < static_mk::MAX_TASKS_PER_SM; ti++) {
            StaticTask const tk = my[ti];
            if (tk.node < 0) break;
            switch (tk.node) {
              case 26: static_mk::run_gemm_tile<StaticNode<7, 14, 1, -1, 0>, StaticParams<0, 7168, 896, 0>, ...>(maps, g, L, tk); break;
              ...                                                 // one case per node (task_call)
            }
          }
          static_mk::kernel_end(g, L);
        }
    profile: the timing build: one more kernel parameter, static_times, and per list entry ti thread 0 writes the task's start and
    lane 0 of every warp writes when its warp left the task (globaltimer ns; plain stores). Without profile the kernel is unchanged."""
    lines = ["", "// ---- the kernel ----",
             "#ifdef STATIC_TIMING_BUILD",
             "#define STATIC_STAGE_STAMPS_OF(L) (L).stage_stamps",
             "#else",
             "#define STATIC_STAGE_STAMPS_OF(L) nullptr",
             "#endif"] + noinline_code(nodes)
    if profile:
        b = gpu_backend.backend()
        lines += ([f"#define STATIC_NS_PER_TICK {gpu_backend.ns_per_tick(b)}"] if b == "hip" else []) + [
                  f"__device__ __forceinline__ unsigned long long static_now() {{ unsigned long long t; {STATIC_NOW[b]} return t; }}",
                  f"#define STATIC_TIME(ti) (static_times + ((size_t)blockIdx.x * (static_mk::MAX_TASKS_PER_SM) + (ti)) * {time_slots()})"]
    # STATIC_EARLY_L2 (core.cuh early_l2): a listed node's task -> its weight lines (gemm_tile nodes; others none)
    lines += ["#ifdef STATIC_EARLY_L2",
              "__device__ __forceinline__ static_mk::L2Span static_l2_span(static_mk::Maps const &maps, int node, int x, int y) {",
              "  switch (node) {"]
    lines += [f"    case {idx}: return static_mk::gemm_tile_l2span<{task_args(node)}>(maps, x, y);"
              for idx, node in sorted(nodes.items(), key=lambda kv: int(kv[0])) if node["name"] == "gemm_tile"]
    lines += ["  }", "  return static_mk::L2Span{nullptr, 0, 0, 0u};", "}", "#endif"]
    extra = ", unsigned long long *static_times" if profile else ""
    lines += [f"__global__ void __launch_bounds__({THREADS}, 1) layer_kernel(const __grid_constant__ static_mk::Maps maps, static_mk::G g, "
              f"StaticTask const *static_tasks{extra}) {{",
              "#ifdef STATIC_TASK_PREFETCH",
              "  // STATIC_TASK_PREFETCH: this SM's first list entry loaded before kernel_begin (its wait for the previous kernel / the go), the",
              "  // list's other lines touched by the first lanes (into the L2): not the previous kernel's data; the first task then starts",
              "  // without a memory round trip after the go (MI355X 10-13, TP8 timing build: go -> first task start 2.2-4.4 us)",
              "  StaticTask const tk0 = static_tasks[blockIdx.x * (static_mk::MAX_TASKS_PER_SM)];",
              "  int const tl_touch = (int)threadIdx.x * 4 < static_mk::MAX_TASKS_PER_SM",
              "                       ? static_tasks[blockIdx.x * (static_mk::MAX_TASKS_PER_SM) + threadIdx.x * 4].node : 0;",
              "#endif",
              "  static_mk::KernelLocals L;",
              "#ifdef STATIC_RESET_FIRST_NODE",   # (core.cuh kernel_begin: the next launch's exchange resets on the CUs starting with that node)
              "  L.first_node = tk0.node; L.tasks = reinterpret_cast<int const *>(static_tasks);",
              "#endif",
              "  static_mk::kernel_begin(g, L);",
              "#ifdef STATIC_EARLY_L2",   # (core.cuh early_l2: the listed nodes' weight lines into the L2 by the CUs starting with that node)
              "  if (L.first_node == STATIC_RESET_FIRST_NODE) {",
              "    static_mk::early_l2(L, [&](int n_, int x_, int y_) { return static_l2_span(maps, n_, x_, y_); });",
              "  }",
              "#endif",
              "  StaticTask const *my = static_tasks + blockIdx.x * (static_mk::MAX_TASKS_PER_SM);",
              "#ifdef STATIC_TASK_PREFETCH",
              "  if (tl_touch == -12345) g.stamps[0] = 0;   // (the touch loads stay: never true, a list holds nodes >= -1)",
              "#endif"]
    lines += [f"  bool warm_{idx} = false;   // a task of node {idx} ran on this SM: its code is in the instruction cache"
              for idx, node in sorted(nodes.items(), key=lambda kv: int(kv[0])) if node["dry"]]
    # MPK_SPLIT_LOOP=S (a one-kernel layer of two parts, e.g. attention + MoE, every CU's tasks of nodes < S before its
    # others): two task loops, the first over the nodes < S, the second over the rest, so the first part's tasks are not
    # register-allocated together with the second part's (MI355X 10-16: in the fused KDA + MoE kernel moe_experts' ~500
    # VGPRs made the loop-carried values spill to scratch; the KDA part ran 67 vs 45 us)
    split = os.environ.get("MPK_SPLIT_LOOP")
    # (several split points "S1,S2,..": one loop per range of nodes, e.g. two MoE + attention layer pairs in one kernel)
    splits = sorted(int(x) for x in split.split(",")) if split else []
    split = splits[0] if len(splits) == 1 else None
    loop_start = len(lines)
    hip = gpu_backend.backend() == "hip"
    lines += [
             "  for (int ti = 0; ti < static_mk::MAX_TASKS_PER_SM; ti++) {",
             "#if defined(STATIC_TASK_SLOAD) && defined(STATIC_TASK_PREFETCH)",
             "    // STATIC_TASK_SLOAD: the entry by scalar loads (s_load, lgkmcnt) through a constant-space pointer - the host writes the",
             "    // list before the launch only. The vector loads (3 x global_load_dword, each waited with vmcnt(0)) waited for every store",
             "    // the previous task issued, its pushes to the other GPUs too (MI355X dec399, T 64: 4-7 us from R send's end to",
             "    // sum_rmsnorm's start on the same CU, 2.5-4 us after latent_up / moe_agg1)",
             "    StaticTask tk = tk0;",
             "    if (ti != 0) {   // (one 16-B vector: a struct cannot be assigned from another address space)",
             "      typedef int st4_ __attribute__((ext_vector_type(4)));",
             "      st4_ const v_ = ((__attribute__((address_space(4))) st4_ const *)my)[ti];",
             "      tk = StaticTask{v_.x, v_.y, v_.z, v_.w};",
             "    }",
             "#elif defined(STATIC_TASK_PREFETCH)",
             "    StaticTask const tk = ti == 0 ? tk0 : my[ti];",
             "#else",
             "    StaticTask const tk = my[ti];",
             "#endif",
             "    if (tk.node < 0) {"] + (
             ["      L.end_code = tk.node;   // (core.cuh kernel_end: the CU's share of the arena's resets)"] if hip else []) + [
             "      break;",
             "    }",
             "#ifdef STATIC_SKIP_TASKS",   # (a probe: every task skipped - the kernel's fixed cost: launch, kernel_begin, kernel_end)
             "    continue;",
             "#endif"]
    if profile:
        lines.append("    if (threadIdx.x == 0) STATIC_TIME(ti)[0] = static_now();")
        lines.append(f"    L.stage_stamps = STATIC_TIME(ti) + {stage_slot()};   // e.g. GEMM tasks: [0] first load issued, [1] first stage landed")
    lines += [
        "#if defined(STATIC_GEMM_L2WARM) || defined(STATIC_DRD_L2NEXT) || defined(STATIC_DRD_HOLD)",   # (gemm_tile.cuh / delta_rule_decode.cuh: the next task's data / node)
        "    L.nxt = ti + 1 < static_mk::MAX_TASKS_PER_SM ? reinterpret_cast<int const *>(my + ti + 1) : nullptr;",
        "#endif",
        "#ifdef STATIC_ARG_RELOAD",
        "    // STATIC_ARG_RELOAD: g and maps through constant-space pointers the compiler cannot prove the same each round: a task",
        "    // loads the few buffer pointers / map fields it uses (s_load, K$) instead of every one staying in registers for the",
        "    // whole kernel (MI355X 10-15: the fused KDA + MoE build had 584 SGPR / 99 VGPR spills, none in the KDA-only build)",
        "    // (the kernel argument segment itself: maps at 0, g after it; &g of a by-value argument is a stack copy)",
        "    { __attribute__((address_space(4))) char const *ka_ = (__attribute__((address_space(4))) char const *)__builtin_amdgcn_kernarg_segment_ptr();",
        "      constexpr size_t goff_ = (sizeof(static_mk::Maps) + alignof(static_mk::G) - 1) / alignof(static_mk::G) * alignof(static_mk::G);",
        "      __attribute__((address_space(4))) static_mk::G const *gp_ = (__attribute__((address_space(4))) static_mk::G const *)(ka_ + goff_);",
        "      __attribute__((address_space(4))) static_mk::Maps const *mp_ = (__attribute__((address_space(4))) static_mk::Maps const *)ka_;",
        "      asm volatile(\"\" : \"+s\"(gp_), \"+s\"(mp_));",
        "      static_mk::G const &g_ = *(static_mk::G const *)gp_; static_mk::Maps const &maps_ = *(static_mk::Maps const *)mp_;",
        "#define STATIC_G_ g_",
        "#define STATIC_MAPS_ maps_",
        "#else",
        "    {",
        "#define STATIC_G_ g",
        "#define STATIC_MAPS_ maps",
        "#endif",
        "    switch (tk.node) {"]
    head = lines[loop_start:]
    del lines[loop_start:]
    tail = ["    }", "    }", "#undef STATIC_G_", "#undef STATIC_MAPS_",
            "#ifdef STATIC_GEMM_L2WARM", "    L.prev_node = tk.node;", "#endif",
            "#ifdef STATIC_GEMM_ACT_KEEP", "    L.keep_node = tk.node;", "    L.keep_y = tk.y;", "#endif",
            "#ifdef STATIC_NODE_END_STAMPS",
            "    if (threadIdx.x == 0) {",
            "#pragma unroll",
            "      for (int ne_ = 0; ne_ < static_mk::NODE_END_N; ne_++)",
            "        if (static_mk::NODE_END_LIST[ne_] == tk.node) g.stamps[L.sm_id * static_mk::NSTAMP + 8 + ne_] = static_mk::gtime();",
            "    }",
            "#endif"]
    if profile:
        w = gpu_backend.wave()
        tail.append(f"    if ((threadIdx.x & {w - 1}) == 0) STATIC_TIME(ti)[1 + threadIdx.x / {w}] = static_now();   // this warp left the task")
    tail.append("  }")
    def cases(keep):
        return [case_code(idx, node).replace("(maps, g, L", "(STATIC_MAPS_, STATIC_G_, L")
                for idx, node in sorted(nodes.items(), key=lambda kv: int(kv[0])) if keep(int(idx))]
    if len(splits) > 1:
        loop_line = "  for (int ti = 0; ti < static_mk::MAX_TASKS_PER_SM; ti++) {"
        assert head.count(loop_line) == 1 and head.count("    if (tk.node < 0) {") == 1
        pre = [("  int ti = 0;   // (MPK_SPLIT_LOOP: one task loop per node range %s)" % splits if x == loop_line else x)
               for x in head[:head.index(loop_line) + 1]]
        rest = head[head.index(loop_line) + 1:]
        brk = rest.index("    if (tk.node < 0) {")
        bounds = [None] + splits + [None]
        lines += pre
        for lo, hi in zip(bounds, bounds[1:]):
            body = rest if hi is None else rest[:brk] + [f"    if (tk.node >= {hi}) break;   // (the CU's first task of the next range)"] + rest[brk:]
            keep = (lambda i, lo=lo, hi=hi: (lo is None or i >= lo) and (hi is None or i < hi))
            lines += ["  for (; ti < static_mk::MAX_TASKS_PER_SM; ti++) {"] + body + cases(keep) + tail
    elif split is None:
        lines += head + cases(lambda i: True) + tail
    else:
        loop_line = "  for (int ti = 0; ti < static_mk::MAX_TASKS_PER_SM; ti++) {"
        assert head.count(loop_line) == 1 and head.count("    if (tk.node < 0) {") == 1
        at = head.index("    if (tk.node < 0) {")
        first = [("  int ti = 0;   // (MPK_SPLIT_LOOP: two task loops, nodes < %d first)" % split if x == loop_line else x)
                 for x in head[:head.index(loop_line) + 1]]
        first += ["  for (; ti < static_mk::MAX_TASKS_PER_SM; ti++) {"]
        rest = head[head.index(loop_line) + 1:]
        brk = rest.index("    if (tk.node < 0) {")
        loop1 = rest[:brk] + [f"    if (tk.node >= {split}) break;   // (the CU's first task of the second part: the second loop)"] + rest[brk:]
        lines += first + loop1 + cases(lambda i: i < split) + tail
        lines += ["  for (; ti < static_mk::MAX_TASKS_PER_SM; ti++) {"] + rest + cases(lambda i: i >= split) + tail
    lines += ["  static_mk::kernel_end(g, L);", "}", ""]
    return lines


def launch_code(cs: int) -> str:
    """The kernel launch in static_launch. cs = the schedules' info "cluster_size" (compiler.compile_plan decides; 1: the plain
    launch; 2: pairs of CTAs 2k, 2k + 1 that share their shared memory (DSMEM), for sum_rmsnorm's on-chip swap; all 148 CTAs stay
    resident)."""
    if cs != 1 and not gpu_backend.has_clusters():
        raise ValueError(f"cluster size {cs}: the {gpu_backend.backend()} backend has no CTA clusters (compile with the plain launch)")
    if cs == 1:
        return "layer_kernel<<<static_mk::NSM, @THREADS@, (int)(@SMEM@), s.stream>>>(@LAUNCH_ARGS@, s.tasks STATIC_TIMES_ARG);"
    return "\n    ".join([
        "{",
        "  cudaLaunchConfig_t cfg = {};",
        "  cfg.gridDim = dim3(static_mk::NSM);",
        "  cfg.blockDim = dim3(@THREADS@);",
        "  cfg.dynamicSmemBytes = (size_t)(@SMEM@);",
        "  cfg.stream = s.stream;",
        "  cudaLaunchAttribute at[1];",
        "  at[0].id = cudaLaunchAttributeClusterDimension;",
        f"  at[0].val.clusterDim.x = {cs};",
        "  at[0].val.clusterDim.y = 1;",
        "  at[0].val.clusterDim.z = 1;",
        "  cfg.attrs = at;",
        "  cfg.numAttrs = 1;",
        "  ST_CK(cudaLaunchKernelEx(&cfg, layer_kernel, @LAUNCH_ARGS@, (StaticTask const *)s.tasks STATIC_TIMES_ARG));",
        "}"])


def cluster_size(schedules: List[dict]) -> int:
    """The CTAs per cluster the schedules were compiled for (compiler.compile_plan; the same on every GPU)."""
    sizes = {int(s["info"]["cluster_size"]) for s in schedules}
    if len(sizes) != 1:
        raise ValueError(f"the GPUs' schedules have different cluster sizes {sorted(sizes)}")
    return sizes.pop()


def host_code(pk, schedules: List[dict], profile: bool = False) -> List[str]:
    """HOST_CODE with every @NAME@ replaced: the slot args (StaticMegakernel.host_slot_args, e.g. {"buf.0": "out router_logits
    401408 255 1"}), the number of GPUs and of SMs (lists)."""
    args = pk.host_slot_args()
    args_code = ", ".join('{"%s", "%s"}' % (k, v) for k, v in args.items())
    num_gpus = len(schedules)
    fill = {
        "@LAUNCH@": launch_code(cluster_size(schedules)),     # first: its text has the other @NAME@s in it
        "@INIT@": f"static_host_init(g_ctx, std::map<std::string, std::string>{{{args_code}}});",
        "@LAUNCH_ARGS@": "static_host::g_gpus[gpu].maps, static_host::g_gpus[gpu].g",
        "@THREADS@": str(THREADS), "@SMEM@": "STATIC_SMEM_BYTES",
        "@TASKS@": ", ".join(f"static_tasks_gpu{g}" for g in range(num_gpus)),
        "@MAX_TASKS@": "static_mk::MAX_TASKS_PER_SM",
        "@BEGINS@": ", ".join(f"static_list_begin_gpu{g}" for g in range(num_gpus)),
        "@PROFILE@": "1" if profile else "0", "@SLOTS@": str(time_slots()), "@WARPS@": str(waves()),
        "@TEMPLATES@": gpu_backend.template_dir(),
    }
    code = HOST_CODE
    for placeholder, value in fill.items():
        code = code.replace(placeholder, value)
    return [code]


def generate_code(pk, schedules: List[dict], profile: bool = False) -> str:
    """layer.cu for pk's graph and one schedule per GPU. All GPUs run the same kernel, so their nodes (grids, params) and number
    of SM lists must be the same; only their task tables differ. profile: the timing build (kernel_code)."""
    nodes = schedules[0]["nodes"]
    if len(schedules) != pk.num_gpus:
        raise ValueError(f"{len(schedules)} schedules, the graph is divided over {pk.num_gpus} GPUs (StaticMegakernel.num_gpus)")
    for g, schedule in enumerate(schedules):
        if schedule["nodes"] != nodes:
            raise ValueError(f"schedule of GPU {g}: nodes differ from GPU 0's (one kernel for every GPU)")
        if len(schedule["worker_task_queues"]) != len(schedules[0]["worker_task_queues"]):
            raise ValueError("every GPU needs the same number of task lists")
    lines = header_code(pk, nodes, len(schedules), len(schedules[0]["worker_task_queues"]), profile)
    lines += smem_code(nodes)
    lines += task_tables(schedules)
    lines += kernel_code(nodes, profile)
    lines += host_code(pk, schedules, profile)
    return "\n".join(lines)


HOST_CODE = r"""
// ---- host: all GPUs of this process ----
#include <Python.h>
#include <atomic>
#include <thread>
#include "mirage/@TEMPLATES@/host.cuh"
#define ST_CK(x)                                                                                    \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) {                                                                        \
      fprintf(stderr, "layer: %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_));                  \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)
// STATIC_SOLO (HIP host.cuh): built for GPUS GPUs, run on one (GPU 0's lists)
#ifdef STATIC_SOLO
#define STATIC_RUN_GPUS 1
#else
#define STATIC_RUN_GPUS static_mk::GPUS
#endif
static StaticTask const *const static_tasks_of[static_mk::GPUS] = {@TASKS@};
static int const *const static_list_begin[static_mk::GPUS] = {@BEGINS@};
struct StaticGpu {
  cudaStream_t stream = nullptr;
  cudaStream_t stream2 = nullptr;             // STATIC_EARLY: the layer kernel's stream
  cudaEvent_t start = nullptr;                // recorded right before the kernel (static_elapsed)
  cudaEvent_t end = nullptr;                  // recorded after the kernel (static_wait, static_done, static_elapsed)
  StaticTask *tasks = nullptr;                // the per-SM task table on the GPU
  unsigned long long *times = nullptr;        // timing build: the per-task times
};
#define STATIC_PROFILE @PROFILE@
#if STATIC_PROFILE
#define STATIC_TIMES_ARG , s.times
#else
#define STATIC_TIMES_ARG
#endif
static size_t const static_times_count = (size_t)static_mk::NSM * (@MAX_TASKS@) * @SLOTS@;   // timing build: per SM, per list entry
static std::vector<StaticGpu> g_st;
static StaticContext g_ctx;
static unsigned long long g_launches = 0;
#ifdef STATIC_LAUNCH_GATE
// STATIC_LAUNCH_GATE (HIP): every GPU's start event and kernel wait on its stream for one host-written word (a stream wait on
// pinned host memory), released by one store after all are queued: the GPUs start within the command processors' poll of that
// word, not ~5-8 us apart as host threads launching them (MI355X 10-11); in a server the previous layer's exchanges line the
// GPUs up the same way. The start event is recorded after the wait: e2e counts from the release.
static unsigned *g_gate = nullptr;
#endif
#if defined(STATIC_LINEUP) && defined(__HIP_PLATFORM_AMD__)
// STATIC_LINEUP (HIP, several GPUs): before each launch a one-wave kernel per GPU lines the GPUs up
// (it writes the launch number into this GPU's hello slot on every GPU, then waits until every slot has it), and the layer
// kernel follows it on the stream; the start event is recorded between the two, so e2e counts from the line-up's end, as the
// B300 harness's GPU line-up kernel before the layer (test_plugin --prev-us)
__global__ void static_lineup_kernel(static_mk::G g) {
  if (threadIdx.x != 0) return;
  using namespace static_mk;
  unsigned const gen = g.gen;
  size_t const set = (size_t)(gen & 1u) * EXCHANGE_SET;
  for (int r = 0; r < GPUS; r++) {
    st16_sys(g.peer[r] + EXCHANGE_HELLO + set + (size_t)g.rank * 16, make_uint4(gen, gen, gen, gen));
  }
  for (int r = 0; r < GPUS; r++) {
    while (ld16_sys(g.rv + EXCHANGE_HELLO + set + (size_t)r * 16).x < gen) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
#ifdef STATIC_EARLY_RES
  // STATIC_EARLY_RES (core.cuh): wait until every GPU's layer kernel is resident (its CU 0's slot; at most ~200 us, so a
  // dispatch that cannot place it never hangs here), 2 us more for its other workgroups, then let this GPU's layer go
  long long const t0 = gtime();
  for (int r = 0; r < GPUS; r++) {
    while (ld16_sys(g.rv + EXCHANGE_HELLO + 1024 + set + (size_t)r * 16).x < gen && gtime() - t0 < 200000) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  // STATIC_EARLY_PREV_US (us, default 2): how long after every layer kernel is resident the go comes (with
  // STATIC_EARLY_PREWAIT: the stand-in previous kernel's length its pre-wait resets may overlap, B300's --prev-us 5)
#ifndef STATIC_EARLY_PREV_US
#define STATIC_EARLY_PREV_US 2
#endif
  long long const t1 = gtime();
  while (gtime() - t1 < STATIC_EARLY_PREV_US * 1000) {
    __builtin_amdgcn_s_sleep(8);
  }
  st16_sys(g.rv + EXCHANGE_HELLO + 2048 + set, make_uint4(gen, gen, gen, gen));
#endif
}
#endif

// per GPU: a stream, two events, the kernel's shared memory size, the per-SM task table on the GPU (max_tasks entries per SM:
// the SM's list, then node = -1); the view the host code gets (its tensors); then static_host_init
static void static_init(std::vector<std::map<std::string, void *>> const &tensors) {
  int const n = STATIC_RUN_GPUS;
  if ((int)tensors.size() != n) {
    fprintf(stderr, "layer: built for %d GPUs, got tensors for %zu\n", n, tensors.size());
    exit(1);
  }
  g_st.assign(n, StaticGpu());
  g_ctx.num_gpus = n;
  g_ctx.num_lists = static_mk::NSM;
  g_ctx.gpus.assign(n, StaticGpuView());
  for (int g = 0; g < n; g++) {
    ST_CK(cudaSetDevice(g));
    ST_CK(cudaStreamCreate(&g_st[g].stream));
    ST_CK(cudaStreamCreate(&g_st[g].stream2));
    ST_CK(cudaEventCreate(&g_st[g].start));
    ST_CK(cudaEventCreate(&g_st[g].end));
    ST_CK(cudaFuncSetAttribute(layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(@SMEM@)));
    int const max_tasks = (int)(@MAX_TASKS@);
    std::vector<StaticTask> table((size_t)static_mk::NSM * max_tasks, StaticTask{-1, 0, 0, 0});
    for (int sm = 0; sm < static_mk::NSM; sm++) {
      int const b = static_list_begin[g][sm], n = static_list_begin[g][sm + 1] - b;
      if (n > max_tasks - 1) {
        fprintf(stderr, "layer: GPU %d SM %d: %d tasks, at most %d\n", g, sm, n, max_tasks - 1);
        exit(1);
      }
      for (int i = 0; i < n; i++) table[(size_t)sm * max_tasks + i] = static_tasks_of[g][b + i];
    }
    ST_CK(cudaMalloc(&g_st[g].tasks, table.size() * sizeof(StaticTask)));
    ST_CK(cudaMemcpy(g_st[g].tasks, table.data(), table.size() * sizeof(StaticTask), cudaMemcpyHostToDevice));
    StaticGpuView &v = g_ctx.gpus[g];
    v.gpu = g;
    v.tensors = tensors[g];
  }
  @INIT@
#ifdef STATIC_LAUNCH_GATE
  ST_CK(hipHostMalloc((void **)&g_gate, 64, hipHostMallocMapped | hipHostMallocPortable));
  *(volatile unsigned *)g_gate = 0;
#endif
  // timing build: the times buffer is allocated after the layer's buffers, so those keep the addresses of the normal build (the
  // buffers' addresses change the layer's time)
  if (STATIC_PROFILE)
    for (int g = 0; g < n; g++) {
      ST_CK(cudaSetDevice(g));
      ST_CK(cudaMalloc(&g_st[g].times, static_times_count * sizeof(unsigned long long)));
    }
}

// GPU gpu's launch: (serving mode: the flush and line-up, static_host_prelude) its start event, the kernel, its end event (on
// its stream)
static size_t g_flush_bytes = 0;   // (static_launch's flush_bytes, for the launching threads)
static void static_launch_one(int gpu) {
  StaticGpu &s = g_st[gpu];
  ST_CK(cudaSetDevice(gpu));
#ifdef STATIC_RESET_IN_KERNEL
  static_host_prelude(gpu, s.stream, g_flush_bytes, (unsigned)(g_launches + 1));
#endif
  ST_CK(cudaEventRecord(s.start, s.stream));
  @LAUNCH@
  ST_CK(cudaGetLastError());
  ST_CK(cudaEventRecord(s.end, s.stream));
}

// STATIC_EARLY (HIP, with STATIC_LINEUP): the layer kernel on the GPU's second stream, launched before the line-up kernel on
// the first stream ends, so its workgroups are resident when the line-up ends (as PDL on B300: the next kernel launched
// before the previous one ends; with STATIC_EARLY_PREWAIT + STATIC_EARLY_RES it posts resident and waits for the line-up's
// go, core.cuh kernel_begin); its end event on the second stream: e2e counts from the line-up's end to the layer's end
static void static_launch_early(int gpu) {
  StaticGpu &s0 = g_st[gpu];
  StaticGpu s = s0;
  s.stream = s0.stream2;
  ST_CK(cudaSetDevice(gpu));
  @LAUNCH@
  ST_CK(cudaGetLastError());
  ST_CK(cudaEventRecord(s0.end, s0.stream2));
}

// more than one GPU: one host thread per GPU launches it, all released at once (one thread launching the GPUs one after the
// other starts them ~8 us apart on MI355X 10-11, and the first ones wait for the last at the kernel's start barrier). The
// threads wait for a launch number (yielding), launch, count themselves done; static_finalize ends them.
static std::vector<std::thread> g_pool;
static std::atomic<unsigned long long> g_go{0};
static std::atomic<int> g_pool_done{0};
static std::atomic<bool> g_pool_quit{false};
static void static_pool_start() {
  int const n = (int)g_st.size();
  g_pool_quit = false;
  unsigned long long const go0 = g_go.load(std::memory_order_acquire);   // read before the threads exist: a thread that
                                                                         // starts after the first go still sees it
  for (int gpu = 0; gpu < n; gpu++) {
    g_pool.emplace_back([gpu, go0]() {
      unsigned long long seen = go0;
      for (;;) {
        unsigned long long v;
        while ((v = g_go.load(std::memory_order_acquire)) == seen) {
          if (g_pool_quit.load(std::memory_order_acquire)) return;
          std::this_thread::yield();
        }
        seen = v;
        static_launch_one(gpu);
        g_pool_done.fetch_add(1, std::memory_order_acq_rel);
      }
    });
  }
}

// one launch on every GPU: per GPU the resets on its stream; wait for every GPU (so all GPUs start together); then per
// GPU: event, kernel, event (one host thread per GPU when there are several). Serving mode (STATIC_RESET_IN_KERNEL: the kernel
// resets its own buffers): no host step and no host wait; per GPU on its stream the flush of flush_bytes (> 0; L2 emptied) and
// the GPUs' line-up (host.cuh static_host_prelude), then event, kernel, event (the events time the launch as a server sees it:
// static_elapsed)
static void static_launch(size_t flush_bytes) {
  g_flush_bytes = flush_bytes;
  int const n = (int)g_st.size();
  // a probe (environment STATIC_COLD_RESETS_MB = N): N MB of scratch written after the host's resets, so the reset buffers are
  // cold when the layer starts, as the rest of its data after the harness's flush
  static size_t const cold = getenv("STATIC_COLD_RESETS_MB") ? (size_t)atoi(getenv("STATIC_COLD_RESETS_MB")) << 20 : 0;
  static std::vector<void *> cold_buf(n, nullptr);
  for (int g = 0; g < n; g++) {
    StaticGpu &s = g_st[g];
    ST_CK(cudaSetDevice(g));
    static_host_reset(g, s.stream, g_launches);
    if (STATIC_PROFILE) ST_CK(cudaMemsetAsync(s.times, 0, static_times_count * sizeof(unsigned long long), s.stream));
    if (cold) {
      if (!cold_buf[g]) ST_CK(cudaMalloc(&cold_buf[g], cold));
      ST_CK(cudaMemsetAsync(cold_buf[g], (int)(g_launches & 0x7F), cold, s.stream));
    }
  }
#ifndef STATIC_RESET_IN_KERNEL
  (void)flush_bytes;
  for (int g = 0; g < n; g++) {
    ST_CK(cudaSetDevice(g));
    ST_CK(cudaStreamSynchronize(g_st[g].stream));
  }
#endif
#if defined(STATIC_LINEUP) && defined(STATIC_EARLY) && defined(__HIP_PLATFORM_AMD__)
  if (n > 1) {
    for (int g = 0; g < n; g++) {
      ST_CK(cudaSetDevice(g));
      static_lineup_kernel<<<1, 64, 0, g_st[g].stream>>>(static_host::g_gpus[g].g);
      ST_CK(cudaEventRecord(g_st[g].start, g_st[g].stream));
      static_launch_early(g);
    }
    g_launches++;
    return;
  }
#endif
#if defined(STATIC_LINEUP) && defined(__HIP_PLATFORM_AMD__)
  if (n > 1) {
    for (int g = 0; g < n; g++) {
      ST_CK(cudaSetDevice(g));
      static_lineup_kernel<<<1, 64, 0, g_st[g].stream>>>(static_host::g_gpus[g].g);
      static_launch_one(g);
    }
    g_launches++;
    return;
  }
#endif
#ifdef STATIC_LAUNCH_GATE
  if (n > 1) {
    unsigned const go = (unsigned)(g_launches + 1);
    for (int g = 0; g < n; g++) {
      ST_CK(cudaSetDevice(g));
      ST_CK(hipStreamWaitValue32(g_st[g].stream, g_gate, go, hipStreamWaitValueGte, 0xFFFFFFFFu));
      static_launch_one(g);
    }
    __atomic_store_n(g_gate, go, __ATOMIC_RELEASE);
    g_launches++;
    return;
  }
#endif
  if (n == 1) {
    static_launch_one(0);
  } else {
    if (g_pool.empty()) static_pool_start();
    g_pool_done.store(0, std::memory_order_release);
    g_go.fetch_add(1, std::memory_order_acq_rel);
    while (g_pool_done.load(std::memory_order_acquire) < n) std::this_thread::yield();
  }
  g_launches++;
}

// per GPU the time between the last launch's two events, in microseconds (after static_wait)
static cudaError_t static_elapsed(std::vector<float> &us) {
  us.assign(g_st.size(), 0.f);
  for (size_t g = 0; g < g_st.size(); g++) {
    ST_CK(cudaSetDevice((int)g));
    float ms = 0.f;
    cudaError_t const e = cudaEventElapsedTime(&ms, g_st[g].start, g_st[g].end);
    if (e != cudaSuccess) return e;
    us[g] = ms * 1000.f;
  }
  return cudaSuccess;
}

// wait for every GPU's end event
static cudaError_t static_wait() {
  for (size_t g = 0; g < g_st.size(); g++) {
    ST_CK(cudaSetDevice((int)g));
    cudaError_t const e = cudaEventSynchronize(g_st[g].end);
    if (e != cudaSuccess) return e;
#ifdef __HIP_PLATFORM_AMD__
    static_host_par_mirror((int)g, g_launches - 1);   // (after the end event: not in the timed window)
#endif
  }
  return cudaSuccess;
}

// whether every GPU's end event has passed (no wait)
static cudaError_t static_done(bool *done) {
  *done = true;
  for (size_t g = 0; g < g_st.size(); g++) {
    ST_CK(cudaSetDevice((int)g));
    cudaError_t const e = cudaEventQuery(g_st[g].end);
    if (e == cudaErrorNotReady) {
      *done = false;
    } else if (e != cudaSuccess) {
      return e;
    }
  }
#ifdef __HIP_PLATFORM_AMD__
  if (*done) { // (static_wait's copy, for the polled wait)
    for (size_t g = 0; g < g_st.size(); g++) {
      ST_CK(cudaSetDevice((int)g));
      static_host_par_mirror((int)g, g_launches - 1);
    }
  }
#endif
  return cudaSuccess;
}

// GPU g's instrumentation buffer `name` (the kernel's: host.cuh static_host_buffer; the timing build's "task_times"), copied
// on a non-blocking stream (readable while a launch has not finished)
static cudaError_t static_read(int g, std::string const &name, std::vector<char> &out) {
  void const *p = nullptr;
  size_t bytes = 0;
  if (name == "task_times" && STATIC_PROFILE) {
    p = g_st[g].times;
    bytes = static_times_count * sizeof(unsigned long long);
  } else {
    static_host_buffer(g, name, p, bytes);
  }
  if (!p) return cudaErrorInvalidValue;
  out.resize(bytes);
  ST_CK(cudaSetDevice(g));
  cudaStream_t nb;
  cudaError_t e = cudaStreamCreateWithFlags(&nb, cudaStreamNonBlocking);
  if (e != cudaSuccess) return e;
  e = cudaMemcpyAsync(out.data(), p, bytes, cudaMemcpyDeviceToHost, nb);
  if (e == cudaSuccess) e = cudaStreamSynchronize(nb);
  cudaStreamDestroy(nb);
  return e;
}

static void static_finalize() {
  g_pool_quit = true;
  for (std::thread &t : g_pool) t.join();
  g_pool.clear();
  static_host_finalize(g_ctx);
  for (size_t g = 0; g < g_st.size(); g++) {
    StaticGpu &s = g_st[g];
    cudaSetDevice((int)g);
    if (s.tasks) cudaFree(s.tasks);
    if (s.times) cudaFree(s.times);
    cudaEventDestroy(s.start);
    cudaEventDestroy(s.end);
    cudaStreamDestroy(s.stream);
  }
  g_st.clear();
}

// ---- Python binding: module __mirage_static_launcher ----
static PyObject *py_init(PyObject *self, PyObject *args) {
  PyObject *py_names, *py_ptrs;
  if (!PyArg_ParseTuple(args, "OO", &py_names, &py_ptrs)) return NULL;
  if (!PyList_Check(py_names) || !PyList_Check(py_ptrs) || PyList_Size(py_names) != PyList_Size(py_ptrs)) {
    PyErr_SetString(PyExc_TypeError, "names / pointers: one list per GPU");
    return NULL;
  }
  std::vector<std::map<std::string, void *>> tensors(PyList_Size(py_names));
  for (Py_ssize_t g = 0; g < PyList_Size(py_names); g++) {
    PyObject *nl = PyList_GetItem(py_names, g), *pl = PyList_GetItem(py_ptrs, g);
    if (!PyList_Check(nl) || !PyList_Check(pl) || PyList_Size(nl) != PyList_Size(pl)) {
      PyErr_SetString(PyExc_TypeError, "bad tensor lists");
      return NULL;
    }
    for (Py_ssize_t i = 0; i < PyList_Size(nl); i++) {
      const char *s = PyUnicode_AsUTF8(PyList_GetItem(nl, i));
      if (!s) return NULL;
      tensors[g][s] = PyLong_AsVoidPtr(PyList_GetItem(pl, i));
    }
  }
  if (PyErr_Occurred()) return NULL;
  static_init(tensors);
  Py_RETURN_NONE;
}
static PyObject *py_launch(PyObject *self, PyObject *args) {
  unsigned long long flush_bytes = 0;
  if (!PyArg_ParseTuple(args, "|K", &flush_bytes)) return NULL;
  static_launch((size_t)flush_bytes);
  Py_RETURN_NONE;
}
static PyObject *py_elapsed(PyObject *self, PyObject *args) {
  std::vector<float> us;
  cudaError_t const err = static_elapsed(us);
  if (err != cudaSuccess) { PyErr_Format(PyExc_RuntimeError, "event times not readable: %s", cudaGetErrorString(err)); return NULL; }
  PyObject *out = PyList_New((Py_ssize_t)us.size());
  for (size_t g = 0; g < us.size(); g++) PyList_SetItem(out, (Py_ssize_t)g, PyFloat_FromDouble(us[g]));
  return out;
}
static PyObject *py_wait(PyObject *self, PyObject *args) {
  cudaError_t err;
  Py_BEGIN_ALLOW_THREADS
  err = static_wait();
  Py_END_ALLOW_THREADS
  if (err != cudaSuccess) { PyErr_Format(PyExc_RuntimeError, "layer kernel failed: %s", cudaGetErrorString(err)); return NULL; }
  Py_RETURN_NONE;
}
static PyObject *py_done(PyObject *self, PyObject *args) {
  bool done = false;
  cudaError_t const err = static_done(&done);
  if (err != cudaSuccess) { PyErr_Format(PyExc_RuntimeError, "layer kernel failed: %s", cudaGetErrorString(err)); return NULL; }
  return PyBool_FromLong(done);
}
static PyObject *py_read(PyObject *self, PyObject *args) {
  int g;
  const char *name;
  if (!PyArg_ParseTuple(args, "is", &g, &name)) return NULL;
  if (g < 0 || g >= (int)g_st.size()) { PyErr_Format(PyExc_IndexError, "GPU %d: the layer runs on %zu", g, g_st.size()); return NULL; }
  std::vector<char> h;
  cudaError_t const err = static_read(g, name, h);
  if (err == cudaErrorInvalidValue) { PyErr_Format(PyExc_KeyError, "no instrumentation buffer %s", name); return NULL; }
  if (err != cudaSuccess) { PyErr_Format(PyExc_RuntimeError, "%s of GPU %d not readable: %s", name, g, cudaGetErrorString(err)); return NULL; }
  return PyBytes_FromStringAndSize(h.data(), (Py_ssize_t)h.size());
}
// the sizes a test harness needs to read the instrumentation buffers (config.cuh, and the timing build's per-entry slots)
static PyObject *py_layout(PyObject *self, PyObject *args) {
  using namespace static_mk;
#ifdef STATIC_RESET_IN_KERNEL
  PyObject *const serving = Py_True;
#else
  PyObject *const serving = Py_False;
#endif
  return Py_BuildValue("{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:O,s:i,s:i,s:i,s:O}",
                       "num_gpus", static_mk::GPUS, "num_sms", static_mk::NSM,
                       "counter_lines", NODE_COUNTER_LINES, "counters_per_line", NCNT / NODE_COUNTER_LINES,
                       "num_stamps", NSTAMP, "stamp_start", STAMP_START, "stamp_task0", STAMP_TASK0, "stamp_task1", STAMP_TASK1,
                       "stamp_end", STAMP_END, "profile", STATIC_PROFILE ? Py_True : Py_False,
                       "max_tasks", (int)(@MAX_TASKS@), "time_slots", @SLOTS@, "warps", @WARPS@, "serving", serving);
}
static PyObject *py_finalize(PyObject *self, PyObject *args) { static_finalize(); Py_RETURN_NONE; }
static PyMethodDef StaticMethods[] = {
  {"init_func", py_init, METH_VARARGS, "the graph's tensors of every GPU (names, pointers): the buffers and maps"},
  {"launch_func", py_launch, METH_VARARGS, "one launch on every GPU ([flush bytes]: serving mode, L2 emptied on the stream first)"},
  {"elapsed_func", py_elapsed, METH_NOARGS, "per GPU the microseconds between the events around the last launch's kernel"},
  {"wait_func", py_wait, METH_NOARGS, "wait for every GPU"},
  {"done_func", py_done, METH_NOARGS, "whether every GPU has finished the last launch (no wait)"},
  {"read_func", py_read, METH_VARARGS, "(gpu, name): an instrumentation buffer's bytes (also while a launch has not finished)"},
  {"layout_func", py_layout, METH_NOARGS, "the sizes of the instrumentation buffers"},
  {"finalize_func", py_finalize, METH_NOARGS, "free everything"},
  {NULL, NULL, 0, NULL}};
static struct PyModuleDef StaticModuleDef = {PyModuleDef_HEAD_INIT, "__mirage_static_launcher", NULL, -1, StaticMethods, NULL, NULL, NULL, NULL};
PyMODINIT_FUNC PyInit___mirage_static_launcher(void) { return PyModule_Create(&StaticModuleDef); }
"""


class StaticKernel:
    """The built layer, driving len(schedule_paths) GPUs from this process: launch, wait, and the kernel's instrumentation buffers
    for a test harness (static_harness.Harness: L2 emptied, timeout, spans, task times)."""

    def __init__(self, module, num_gpus: int, cu_path: str, so_path: str):
        self.module = module
        self.num_gpus = num_gpus
        self.cu_path = cu_path
        self.so_path = so_path
        self._finalized = False

    def launch(self, flush_bytes: int = 0):
        """One launch on every GPU: per GPU the resets, then (every GPU's resets done) the kernel. Serving mode (layout()
        "serving": built with -DSTATIC_RESET_IN_KERNEL): the kernel resets its own buffers, no host step or wait; flush_bytes > 0:
        L2 emptied on each GPU's stream first, then the GPUs lined up (more than one), then the kernel."""
        self.module.launch_func(int(flush_bytes))

    def elapsed_us(self) -> List[float]:
        """Per GPU the microseconds between the events right before and after the last launch's kernel (after wait). In serving
        mode with flush_bytes > 0 the launch is queued while the flush runs, so this is the kernel as a server sees it."""
        return self.module.elapsed_func()

    def wait(self):
        """Wait for every GPU; RuntimeError if a kernel failed."""
        self.module.wait_func()

    def done(self) -> bool:
        """Whether every GPU has finished the last launch (no wait); RuntimeError if a kernel failed."""
        return self.module.done_func()

    def read(self, gpu: int, name: str) -> bytes:
        """GPU gpu's instrumentation buffer `name` (layout() gives the sizes), also while a launch has not finished:
        "counters"       uint32 [counter_lines][counters_per_line], the nodes' task counters
        "stamps"         int64 [num_sms][num_stamps], globaltimer ns per SM (stamp_start, stamp_task0, stamp_task1, stamp_end)
        "start_barrier"  int64 [1], globaltimer ns of the last SM past the start barrier
        "task_times"     timing build: uint64 [num_sms][max_tasks][time_slots] per list entry: start, each warp's end, 2 stage
                         stamps"""
        return self.module.read_func(int(gpu), name)

    def layout(self) -> dict:
        """The sizes of the instrumentation buffers (read)."""
        return self.module.layout_func()

    def finalize(self):
        if not self._finalized:
            self.module.finalize_func()
            self._finalized = True


def compile_command(cu_path: str, so_path: str, extra_flags: Optional[List[str]] = None, arch: Optional[str] = None) -> List[str]:
    """-O3 -std=c++17, the GPU's own arch, plus what a Python extension needs (-shared, -fPIC, the Python headers). CUDA: nvcc,
    -lcuda; HIP (gpu_backend): hipcc --offload-arch (arch: e.g. gfx950; default the GPU's)."""
    from ..kernel import get_key_paths
    _, include_path, _ = get_key_paths()
    scheme = sysconfig.get_default_scheme() if hasattr(sysconfig, "get_default_scheme") else sysconfig._get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    python_include = sysconfig.get_paths(scheme=scheme)["include"]
    if gpu_backend.backend() == "hip":
        return ([shutil.which("hipcc") or "hipcc", "-O3", "-std=c++17", f"--offload-arch={arch or gpu_backend.offload_arch()}",
                 f"-I{include_path}", f"-I{python_include}", "-shared", "-fPIC", cu_path, "-o", so_path]
                + list(extra_flags or []))
    if arch is None:
        import torch
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"{major}{minor}a" if major >= 9 else f"{major}{minor}"
    return ([shutil.which("nvcc"), "-O3", "-std=c++17", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
             f"-I{include_path}", f"-I{python_include}",
             "-shared", "-Xcompiler=-fPIC", "-lcuda", cu_path, "-o", so_path]
            + list(extra_flags or []))


def compile_static(pk, schedule_paths: List[str], gpu_tensors: Optional[List[dict]] = None, out_dir: Optional[str] = None,
                   extra_flags: Optional[List[str]] = None, profile: bool = False, solo: bool = False) -> StaticKernel:
    """  1. read the schedule files; write <out_dir>/layer.cu = generate_code
       2. nvcc / hipcc it into a Python extension (compile_command) and load it
       3. init_func: every GPU's tensors (gpu_tensors[g] = {name: torch tensor on GPU g}) -> static_init -> static_host_init
    profile: the timing build (static_harness.Harness.task_times). solo (HIP): built for every GPU's schedule, run on one
    (GPU 0's lists; gpu_tensors: GPU 0's dict only); the kernel skips its waits for other GPUs' data (STATIC_SOLO)."""
    num_gpus = 1 if solo else len(schedule_paths)
    if solo:
        assert gpu_backend.backend() == "hip", "solo: the HIP templates only"
        extra_flags = list(extra_flags or []) + ["-DSTATIC_SOLO"]
    gpu_tensors = gpu_tensors or [dict(pk._model_tensors)]
    assert len(gpu_tensors) == num_gpus, "one tensor dict per GPU"
    out_dir = out_dir or os.path.dirname(os.path.abspath(schedule_paths[0]))
    os.makedirs(out_dir, exist_ok=True)

    schedules = []
    for path in schedule_paths:
        with open(path) as f:
            schedules.append(json.load(f))
    cu_path = os.path.join(out_dir, "layer.cu")
    so_path = os.path.join(out_dir, "layer" + sysconfig.get_config_var("EXT_SUFFIX"))
    code = generate_code(pk, schedules, profile)
    with open(cu_path, "w") as f:
        f.write(code)

    command = compile_command(cu_path, so_path, extra_flags)
    # the build is reused when the code, the command and every header under the kernel's include directories are the same as
    # for the library already in out_dir (a GPU run then skips the ~6-10 s compile)
    import hashlib
    h = hashlib.sha256((code + "\0" + " ".join(command)).encode())
    inc = os.path.join(os.environ.get("MIRAGE_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "include",
                       "mirage")
    for root, _, files in sorted(os.walk(inc)):
        for name in sorted(files):
            if name.endswith((".cuh", ".h")):
                with open(os.path.join(root, name), "rb") as f:
                    h.update(name.encode() + f.read())
    key, key_path = h.hexdigest(), os.path.join(out_dir, "build.key")
    if os.path.exists(so_path) and os.path.exists(key_path) and open(key_path).read() == key:
        print("building the layer: unchanged, reusing", so_path, flush=True)
    else:
        print("building the layer:", " ".join(command), flush=True)
        subprocess.check_call(command)
        with open(key_path, "w") as f:
            f.write(key)

    spec = importlib.util.spec_from_file_location("__mirage_static_launcher", so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [list(tensors.keys()) for tensors in gpu_tensors]
    pointers = [[t.data_ptr() for t in tensors.values()] for tensors in gpu_tensors]
    module.init_func(names, pointers)
    return StaticKernel(module, num_gpus, cu_path, so_path)
