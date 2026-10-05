"""Schedule files -> layer.cu -> built, loaded and given the tensors of every GPU.

A schedule file (schedule_gpu<g>.json, written by compiler.compile_plan) has:
  nodes                per graph node: name, grid, params, counter, buf, inputs (the node writing each input: its grid, counter,
                       buf, params; null: graph input)
  all_tasks            per task: node, pos (grid position), deps (the task ids it waits for)
  worker_task_queues   per SM, the task ids in the order the SM runs them
  concurrent_groups    task ids that must run at the same time (they wait for each other while they run)

layer.cu = five parts, each written by one function below:
  header_code     the kernel-wide sizes (tokens per step, GPUs, SMs: config.cuh), the exchange region's layout (from the layers:
                  StaticMegakernel.exchange_layout), StaticTask {node, x, y, z} (one entry of an SM's list), StaticNode /
                  StaticParams / StaticSlots (a node's grid, counter, output buffer slot, params and own slots as template
                  constants), the structs the host code receives; static_megakernel/core.cuh and the used task types' files
                  (tasks/<name>.cuh)
  smem_code       the launch's dynamic shared memory: the most any node's task type needs (its smem_<name>)
  task_tables     per GPU, the entries of every SM's list one after another, and where each SM's list begins
  kernel_code     layer_kernel: kernel_begin, a loop over this SM's entries with one `case` per node that calls the node's task
                  function static_mk::run_<name> with its node, params, slots and its inputs' producers as template arguments,
                  kernel_end (core.cuh)
  host_code       HOST_CODE below: init (also the per-SM task tables on each GPU), one launch on every GPU, wait, timing, Python;
                  the buffers and maps are static_megakernel/host.cuh's, from the layers' slot args (host_slot_args)

  write_schedule(path, nodes, all_tasks, queues, info)      checks the lists cannot deadlock, writes the file
  generate_code(pk, schedules) -> str                       layer.cu
  compile_static(pk, schedule_paths, gpu_tensors, out_dir, extra_flags, code=None) -> StaticKernel
"""
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sysconfig
from typing import Dict, List, Optional, Tuple


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
        raise ValueError(f"the task lists can deadlock: {len(stuck)} of {num_tasks} tasks wait for each other in a circle")


def write_schedule(path: str, nodes: Dict[str, dict], all_tasks: List[dict], queues: List[List[int]], info: dict,
                   groups: List[List[int]] = ()) -> None:
    check_deadlock_free(all_tasks, queues, groups)
    doc = {"nodes": nodes, "all_tasks": all_tasks, "worker_task_queues": queues, "info": info, "concurrent_groups": list(groups)}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def code_tag() -> str:
    """The version of the kernel templates a cost record measured: the md5 of include/mirage/static_megakernel/*.cuh and tasks/*.cuh (first 12
    hex digits). the cost records carry it ("code"); search.Costs(code=code_tag()) uses only the records of these templates,
    so times measured with other kernel code are not mixed in."""
    from ..kernel import get_key_paths
    _, include_path, _ = get_key_paths()
    h = hashlib.md5()
    root = os.path.join(include_path, "mirage", "static_megakernel")
    for path in sorted(glob.glob(os.path.join(root, "*.cuh")) + glob.glob(os.path.join(root, "tasks", "*.cuh"))):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


# ---------------- code generation ----------------
THREADS = 256   # one CTA of 256 threads per SM (runtime.cuh: the warp roles)


def header_code(pk, nodes: Dict[str, dict], num_gpus: int, num_lists: int, profile: bool = False) -> List[str]:
    set_bytes, hello, offsets, sizes = pk.exchange_layout()
    used = sorted({n["name"] for n in nodes.values()})
    return [f"// layer.cu -- generated by mirage.mpk.static_schedule.compile_static from the graph and {num_gpus} "
            f"schedule.json files"] + (["#define STATIC_TIMING_BUILD 1   // timing hooks (e.g. GEMM stage stamps)"] if profile else []) + [
            "#include <cuda.h>", "#include <cuda_runtime.h>", "#include <chrono>", "#include <initializer_list>", "#include <map>",
            "#include <string>", "#include <vector>", "",
            "// the kernel-wide sizes (config.cuh): tokens per step and GPUs (the graph's: StaticMegakernel.tokens, num_gpus), SMs (the",
            "// schedules' task lists per GPU)",
            f"#define STATIC_TOKENS {pk.tokens}", f"#define STATIC_GPUS {num_gpus}", f"#define STATIC_SMS {num_lists}", "",
            "// the exchange region's layout (static_megakernel.py exchange_layout): one set's bytes, the start barrier's slots, per",
            "// buffer slot its offset and bytes (an exchange buffer; else 0)",
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
            "template <int... S> struct StaticSlots { static constexpr int v[sizeof...(S) + 1] = {S..., -1}; };   // a node's own slots",
            "",
            "// what the host code sees of one GPU / of all GPUs",
            "struct StaticGpuView {",
            "  int gpu;                                  // CUDA device index",
            "  std::map<std::string, void *> tensors;    // the graph's tensors on this GPU, by name",
            "  cudaStream_t stream;",
            "};",
            "struct StaticContext { int num_gpus = 0, num_lists = 0; std::vector<StaticGpuView> gpus; };",
            '#include "mirage/static_megakernel/core.cuh"'] + \
        [f'#include "mirage/static_megakernel/tasks/{name}.cuh"' for name in used] + [""]


def task_tables(schedules: List[dict]) -> List[str]:
    """Per GPU g two C++ arrays:
      static_tasks_gpu<g>       the entries of SM 0's list, then SM 1's, ...: {node, x, y, z}
      static_list_begin_gpu<g>  SM k's entries are [begin[k], begin[k + 1])
    Example: SM 0 = [topk_route task 0, moe_experts task 0, allreduce_send task 0] -> {31, 0, 0, 0}, {36, 0, 0, 0}, {37, 0, 0, 0}; begin = {0, 3, ..}.
    static_init copies them into a per-SM table on each GPU (max_tasks entries per SM, the rest node = -1)."""
    lines = []
    for g, schedule in enumerate(schedules):
        entries, begin = [], [0]
        for sm_list in schedule["worker_task_queues"]:
            for t in sm_list:
                task = schedule["all_tasks"][t]
                entries.append("{%d, %d, %d, %d}" % (task["node"], *task["pos"]))
            begin.append(len(entries))
        lines.append(f"static const StaticTask static_tasks_gpu{g}[] = {{{', '.join(entries)}}};")
        lines.append(f"static const int static_list_begin_gpu{g}[] = {{{', '.join(map(str, begin))}}};")
    return lines


def task_args(node: dict) -> str:
    """A node's template arguments: the node (StaticNode: grid, counter, buffer slot), its params, its own slots, and per input
    the node that writes it (with its params; StaticNode<0, 0, 0> for a graph input)."""
    def snode(grid, counter, buf, params=()):
        return "StaticNode<%s>" % ", ".join(str(v) for v in list(grid) + [counter, buf] + list(params))
    args = [snode(node["grid"], node["counter"], node["buf"]), "StaticParams<%s>" % ", ".join(str(p) for p in node["params"]),
            "StaticSlots<%s>" % ", ".join(str(v) for v in node.get("slots", []))]
    args += [snode(i["grid"], i["counter"], i["buf"], i["params"]) if i else "StaticNode<0, 0, 0>" for i in node["inputs"]]
    return ", ".join(args)


def task_call(node: dict) -> str:
    """One node's call in the kernel loop: static_mk::run_<name> with its template arguments (task_args).
    static_mk::run_topk_route<StaticNode<8, 1, 1, -1, 1>, StaticParams<896, 16>, StaticSlots<11>, StaticNode<7, 14, 1, -1, 0, 0, 7168, 896, 0, 2, 0>, StaticNode<0, 0, 0>, ...>(maps, g, L, tk)"""
    return f"static_mk::run_{node['name']}<{task_args(node)}>(maps, g, L, tk)"


def smem_code(nodes: Dict[str, dict]) -> List[str]:
    """STATIC_SMEM_BYTES: the launch's dynamic shared memory, for the most any node's task type needs from the 1024-aligned base
    (static_mk::smem_<name> with the node's template arguments; config.cuh smem_launch_bytes adds the alignment)."""
    needs = [f"static_mk::smem_{node['name']}<{task_args(node)}>()" for _, node in sorted(nodes.items(), key=lambda kv: int(kv[0]))]
    return ["// the launch's dynamic shared memory: the most any node's task type needs (smem_<name>)",
            "constexpr int static_smem_need(std::initializer_list<int> v) { int m = 0; for (int x : v) m = x > m ? x : m; return m; }",
            "constexpr int STATIC_SMEM_BYTES = static_mk::smem_launch_bytes(static_smem_need({%s}));" % ", ".join(needs),
            'static_assert(STATIC_SMEM_BYTES <= 227 * 1024, "the dynamic shared memory of one SM");', ""]


def case_code(idx: str, node: dict) -> str:
    """One node's case in the kernel loop. A node whose task type has a dry pass (its layer's `dry`): the task function gets
    `warm` last: false for the node's first task on this SM (its code is not in the instruction cache yet: the task first runs
    its own code on stand-in inputs, no global stores, while its inputs are on their way; the same instructions as the real pass),
    true for the later ones (kernel_code keeps one flag per such node). The flag is a register of the kernel loop, so deciding
    costs almost nothing (checking whether the inputs have already landed would cost loads and a CTA barrier per task).
      case 29: static_mk::run_topk_route<...>(maps, g, L, tk, warm_29); warm_29 = true; break;"""
    run = task_call(node)
    if not node.get("dry"):
        return f"      case {idx}: {run}; break;"
    return f"      case {idx}: {run[:-1]}, warm_{idx}); warm_{idx} = true; break;"


def kernel_code(nodes: Dict[str, dict], profile: bool = False) -> List[str]:
    """layer_kernel:
        __global__ void __launch_bounds__(256, 1) layer_kernel(const __grid_constant__ static_mk::Maps maps, static_mk::G g, StaticTask const *static_tasks) {
          static_mk::KernelLocals L; static_mk::kernel_begin(maps, g, L);
          StaticTask const *my = static_tasks + blockIdx.x * (static_mk::MAX_TASKS_PER_SM);   // this SM's list
          for (int ti = 0; ti < static_mk::MAX_TASKS_PER_SM; ti++) {
            StaticTask const tk = my[ti];
            if (tk.node < 0) break;
            switch (tk.node) {
              case 30: static_mk::run_gemm_tile<StaticNode<7, 14, 1, -1, 0>, StaticParams<0, 7168, 896, 0, 2>, ...>(maps, g, L, tk); break;
              ...                                                 // one case per node (task_call)
            }
          }
          static_mk::kernel_end(g, L);
        }
    profile: the timing build: one more kernel parameter, static_times, and per list entry ti thread 0 writes the task's start and
    lane 0 of every warp writes when its warp left the task (globaltimer ns; plain stores). Without profile the kernel is unchanged."""
    slots = 1 + THREADS // 32 + 2   # start, each warp's end, then the 2 stage stamps (KernelLocals::stage_stamps)
    lines = ["", "// ---- the kernel ----"]
    if profile:
        lines += ["__device__ __forceinline__ unsigned long long static_now() { unsigned long long t; "
                  "asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t; }",
                  f"#define STATIC_TIME(ti) (static_times + ((size_t)blockIdx.x * (static_mk::MAX_TASKS_PER_SM) + (ti)) * {slots})"]
    extra = ", unsigned long long *static_times" if profile else ""
    lines += [f"__global__ void __launch_bounds__({THREADS}, 1) layer_kernel(const __grid_constant__ static_mk::Maps maps, static_mk::G g, "
              f"StaticTask const *static_tasks{extra}) {{",
              "  static_mk::KernelLocals L; static_mk::kernel_begin(maps, g, L);",
              "  StaticTask const *my = static_tasks + blockIdx.x * (static_mk::MAX_TASKS_PER_SM);"]
    lines += [f"  bool warm_{idx} = false;   // a task of node {idx} ran on this SM: its code is in the instruction cache"
              for idx, node in sorted(nodes.items(), key=lambda kv: int(kv[0])) if node.get("dry")]
    lines += [
             "  for (int ti = 0; ti < static_mk::MAX_TASKS_PER_SM; ti++) {",
             "    StaticTask const tk = my[ti];",
             "    if (tk.node < 0) break;"]
    if profile:
        lines.append("    if (threadIdx.x == 0) STATIC_TIME(ti)[0] = static_now();")
        lines.append(f"    L.stage_stamps = STATIC_TIME(ti) + {1 + THREADS // 32};   // e.g. GEMM tasks: [0] first load issued, [1] first stage landed")
    lines.append("    switch (tk.node) {")
    for idx, node in sorted(nodes.items(), key=lambda kv: int(kv[0])):
        lines.append(case_code(idx, node))
    lines.append("    }")
    if profile:
        lines.append("    if ((threadIdx.x & 31) == 0) STATIC_TIME(ti)[1 + threadIdx.x / 32] = static_now();   // this warp left the task")
    lines += ["  }", "  static_mk::kernel_end(g, L);", "}", ""]
    return lines


def launch_code(cs: int) -> str:
    """The kernel launch in static_launch. cs = the schedules' info "cluster_size" (compiler.compile_plan decides; 1: the plain
    launch; 2: pairs of CTAs 2k, 2k + 1 that share their shared memory (DSMEM), for sum_rmsnorm's on-chip swap; all 148 CTAs stay
    resident)."""
    if cs == 1:
        return "layer_kernel<<<@NLISTS@, @THREADS@, (int)(@SMEM@), s.stream>>>(@LAUNCH_ARGS@, s.tasks STATIC_TIMES_ARG);"
    return "\n    ".join([
        "{",
        "  cudaLaunchConfig_t cfg = {};",
        "  cfg.gridDim = dim3(@NLISTS@);",
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
    sizes = {int(s["info"].get("cluster_size", 1)) for s in schedules}
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
        "@NUM_GPUS@": str(num_gpus),
        "@INIT@": f"static_host_init(g_ctx, std::map<std::string, std::string>{{{args_code}}});",
        "@LAUNCH_ARGS@": "static_host::g_gpus[gpu].maps, static_host::g_gpus[gpu].g",
        "@THREADS@": str(THREADS), "@SMEM@": "STATIC_SMEM_BYTES",
        "@TASKS@": ", ".join(f"static_tasks_gpu{g}" for g in range(num_gpus)),
        "@MAX_TASKS@": "static_mk::MAX_TASKS_PER_SM",
        "@BEGINS@": ", ".join(f"static_list_begin_gpu{g}" for g in range(num_gpus)),
        "@NLISTS@": str(len(schedules[0]["worker_task_queues"])),
        "@PROFILE@": "1" if profile else "0", "@SLOTS@": str(1 + THREADS // 32 + 2), "@WARPS@": str(THREADS // 32),
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
#include "mirage/static_megakernel/host.cuh"
#define ST_CK(x)                                                                                    \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) {                                                                        \
      fprintf(stderr, "layer: %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_));                  \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)
static StaticTask const *const static_tasks_of[@NUM_GPUS@] = {@TASKS@};
static int const *const static_list_begin[@NUM_GPUS@] = {@BEGINS@};
struct StaticGpu {
  cudaStream_t stream = nullptr;
  cudaEvent_t e0 = nullptr, e1 = nullptr;     // the kernel's start and end
  void *flush = nullptr;                      // read before a launch to empty L2 (flush_bytes, + 16 B: static_l2_read's sink)
  size_t flush_bytes = 0;
  StaticTask *tasks = nullptr;                // the per-SM task table on the GPU
  unsigned long long *times = nullptr;        // timing build: the per-task times
};
#define STATIC_PROFILE @PROFILE@
#if STATIC_PROFILE
#define STATIC_TIMES_ARG , s.times
#else
#define STATIC_TIMES_ARG
#endif
static size_t const static_times_count = (size_t)@NLISTS@ * (@MAX_TASKS@) * @SLOTS@;   // timing build: per SM, per list entry
static std::vector<StaticGpu> g_st;
static StaticContext g_ctx;
// empties L2 by reading flush_bytes (not writing: a write leaves L2 full of dirty lines, and the layer's reads would then also pay
// their write-back to memory, which a layer in a real model does not see); the xor only keeps the loads
__global__ void static_l2_read(uint4 const *p, size_t n, unsigned *sink) {
  unsigned a = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 const v = __ldcg(p + i);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0x9e3779b9u) *sink = a;
}
static unsigned long long g_launches = 0;

// per GPU: a stream, two events, the kernel's shared memory size, the per-SM task table on the GPU (max_tasks entries per SM:
// the SM's list, then node = -1); the view the host code gets (its tensors); then static_host_init
static void static_init(std::vector<std::map<std::string, void *>> const &tensors) {
  int const n = @NUM_GPUS@;
  if ((int)tensors.size() != n) {
    fprintf(stderr, "layer: built for %d GPUs, got tensors for %zu\n", n, tensors.size());
    exit(1);
  }
  g_st.assign(n, StaticGpu());
  g_ctx.num_gpus = n;
  g_ctx.num_lists = @NLISTS@;
  g_ctx.gpus.assign(n, StaticGpuView());
  for (int g = 0; g < n; g++) {
    ST_CK(cudaSetDevice(g));
    ST_CK(cudaStreamCreate(&g_st[g].stream));
    ST_CK(cudaEventCreate(&g_st[g].e0));
    ST_CK(cudaEventCreate(&g_st[g].e1));
    ST_CK(cudaFuncSetAttribute(layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(@SMEM@)));
    int const max_tasks = (int)(@MAX_TASKS@);
    std::vector<StaticTask> table((size_t)@NLISTS@ * max_tasks, StaticTask{-1, 0, 0, 0});
    for (int sm = 0; sm < @NLISTS@; sm++) {
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
    v.stream = g_st[g].stream;
  }
  @INIT@
  // timing build: the times buffer is allocated after the layer's buffers, so those keep the addresses of the normal build (the
  // buffers' addresses change the layer's time)
  if (STATIC_PROFILE)
    for (int g = 0; g < n; g++) {
      ST_CK(cudaSetDevice(g));
      ST_CK(cudaMalloc(&g_st[g].times, static_times_count * sizeof(unsigned long long)));
    }
}

// one launch on every GPU: per GPU the L2 read (flush_bytes > 0) and the resets on its stream; wait for every GPU (so
// all GPUs start together); then per GPU: event, kernel, event
static void static_launch(size_t flush_bytes) {
  int const n = (int)g_st.size();
  for (int g = 0; g < n; g++) {
    StaticGpu &s = g_st[g];
    ST_CK(cudaSetDevice(g));
    if (flush_bytes > 0) {
      if (s.flush_bytes < flush_bytes) {   // + 16 B at the end: static_l2_read's sink
        if (s.flush) ST_CK(cudaFree(s.flush));
        ST_CK(cudaMalloc(&s.flush, flush_bytes + 16));
        ST_CK(cudaMemset(s.flush, 0, flush_bytes + 16));
        s.flush_bytes = flush_bytes;
      }
      static_l2_read<<<1184, 256, 0, s.stream>>>((uint4 const *)s.flush, flush_bytes / 16, (unsigned *)((char *)s.flush + flush_bytes));
    }
    static_host_reset(g_ctx, g, s.stream, g_launches);
    if (STATIC_PROFILE) ST_CK(cudaMemsetAsync(s.times, 0, static_times_count * sizeof(unsigned long long), s.stream));
  }
  for (int g = 0; g < n; g++) {
    ST_CK(cudaSetDevice(g));
    ST_CK(cudaStreamSynchronize(g_st[g].stream));
  }
  for (int gpu = 0; gpu < n; gpu++) {
    StaticGpu &s = g_st[gpu];
    ST_CK(cudaSetDevice(gpu));
    ST_CK(cudaEventRecord(s.e0, s.stream));
    @LAUNCH@
    ST_CK(cudaGetLastError());
    ST_CK(cudaEventRecord(s.e1, s.stream));
  }
  g_launches++;
}

// wait for every GPU's end event; timeout_s > 0: give up after that long (*timed_out = true)
static cudaError_t static_wait(double timeout_s, bool *timed_out) {
  auto const t0 = std::chrono::steady_clock::now();
  *timed_out = false;
  for (size_t g = 0; g < g_st.size(); g++) {
    ST_CK(cudaSetDevice((int)g));
    for (;;) {
      cudaError_t e = cudaEventQuery(g_st[g].e1);
      if (e == cudaSuccess) break;
      if (e != cudaErrorNotReady) return e;
      double const waited_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (timeout_s > 0 && waited_s > timeout_s) {
        *timed_out = true;
        return cudaSuccess;
      }
    }
  }
  return cudaSuccess;
}

static void static_finalize() {
  static_host_finalize(g_ctx);
  for (size_t g = 0; g < g_st.size(); g++) {
    StaticGpu &s = g_st[g];
    cudaSetDevice((int)g);
    if (s.flush) cudaFree(s.flush);
    if (s.tasks) cudaFree(s.tasks);
    if (s.times) cudaFree(s.times);
    cudaEventDestroy(s.e0);
    cudaEventDestroy(s.e1);
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
  unsigned long long flush = 0;
  if (!PyArg_ParseTuple(args, "|K", &flush)) return NULL;
  static_launch((size_t)flush);
  Py_RETURN_NONE;
}
static PyObject *py_wait(PyObject *self, PyObject *args) {
  double timeout_s = 0.0;
  if (!PyArg_ParseTuple(args, "|d", &timeout_s)) return NULL;
  cudaError_t err; bool timed_out = false;
  Py_BEGIN_ALLOW_THREADS
  err = static_wait(timeout_s, &timed_out);
  Py_END_ALLOW_THREADS
  if (timed_out) { PyErr_Format(PyExc_TimeoutError, "layer kernel not finished after %d s", (int)timeout_s); return NULL; }
  if (err != cudaSuccess) { PyErr_Format(PyExc_RuntimeError, "layer kernel failed: %s", cudaGetErrorString(err)); return NULL; }
  Py_RETURN_NONE;
}
// per GPU (start barrier passed, end) of the last launch, globaltimer ns
static PyObject *py_timing(PyObject *self, PyObject *args) {
  PyObject *out = PyList_New((Py_ssize_t)g_st.size());
  for (size_t g = 0; g < g_st.size(); g++) {
    long long bar = 0, end = 0;
    static_host_timing(g_ctx, (int)g, bar, end);
    PyList_SetItem(out, (Py_ssize_t)g, Py_BuildValue("(LL)", bar, end));
  }
  return out;
}
// timing build: per GPU, per SM, per list entry (start, the last warp's end, the 2 stage stamps) globaltimer ns of the last launch
// (0: no task there / no stamp)
static PyObject *py_task_times(PyObject *self, PyObject *args) {
  if (!STATIC_PROFILE) { PyErr_SetString(PyExc_RuntimeError, "not a timing build (compile_static(profile=True))"); return NULL; }
  std::vector<unsigned long long> h(static_times_count);
  PyObject *out = PyList_New((Py_ssize_t)g_st.size());
  for (size_t g = 0; g < g_st.size(); g++) {
    ST_CK(cudaSetDevice((int)g)); ST_CK(cudaMemcpy(h.data(), g_st[g].times, h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    PyObject *sms = PyList_New(@NLISTS@);
    for (int sm = 0; sm < @NLISTS@; sm++) {
      PyObject *entries = PyList_New(@MAX_TASKS@);
      for (int ti = 0; ti < (int)(@MAX_TASKS@); ti++) {
        unsigned long long const *t = h.data() + ((size_t)sm * (@MAX_TASKS@) + ti) * @SLOTS@;
        unsigned long long end = 0; for (int w = 1; w <= @WARPS@; w++) end = t[w] > end ? t[w] : end;
        PyList_SetItem(entries, ti, Py_BuildValue("(KKKK)", t[0], end, t[@WARPS@ + 1], t[@WARPS@ + 2]));
      }
      PyList_SetItem(sms, sm, entries);
    }
    PyList_SetItem(out, (Py_ssize_t)g, sms);
  }
  return out;
}
static PyObject *py_report(PyObject *self, PyObject *args) { static_host_report(g_ctx); fflush(stdout); fflush(stderr); Py_RETURN_NONE; }
static PyObject *py_finalize(PyObject *self, PyObject *args) { static_finalize(); Py_RETURN_NONE; }
static PyMethodDef StaticMethods[] = {
  {"init_func", py_init, METH_VARARGS, "the graph's tensors of every GPU (names, pointers): the buffers and maps"},
  {"launch_func", py_launch, METH_VARARGS, "one launch on every GPU (optional bytes read first to empty L2)"},
  {"wait_func", py_wait, METH_VARARGS, "wait for every GPU (optional timeout in seconds)"},
  {"timing_func", py_timing, METH_NOARGS, "per GPU (start barrier passed, end) globaltimer ns of the last launch"},
  {"task_times_func", py_task_times, METH_NOARGS, "timing build: per GPU, per SM, per list entry (start, end) ns of the last launch"},
  {"report_func", py_report, METH_NOARGS, "the counters and stamps (also for a launch that has not finished)"},
  {"finalize_func", py_finalize, METH_NOARGS, "free everything"},
  {NULL, NULL, 0, NULL}};
static struct PyModuleDef StaticModuleDef = {PyModuleDef_HEAD_INIT, "__mirage_static_launcher", NULL, -1, StaticMethods, NULL, NULL, NULL, NULL};
PyMODINIT_FUNC PyInit___mirage_static_launcher(void) { return PyModule_Create(&StaticModuleDef); }
"""


class StaticKernel:
    """The built layer, driving len(schedule_paths) GPUs from this process."""

    def __init__(self, module, num_gpus: int, cu_path: str, so_path: str):
        self.module = module
        self.num_gpus = num_gpus
        self.cu_path = cu_path
        self.so_path = so_path
        self._finalized = False

    def launch(self, l2_flush_bytes: int = 0):
        """One launch on every GPU. l2_flush_bytes > 0: read that many bytes on each GPU first (empties L2, leaves it clean)."""
        self.module.launch_func(int(l2_flush_bytes))

    def wait(self, timeout_s: float = 0.0):
        """Wait for every GPU; timeout_s > 0 raises TimeoutError after that long (the kernel is then still running)."""
        self.module.wait_func(float(timeout_s))

    def timing(self) -> List[Tuple[int, int]]:
        """Per GPU (start barrier passed, end) of the last launch, globaltimer ns."""
        return self.module.timing_func()

    def span_us(self) -> float:
        """Max over GPUs of (end - start barrier passed) of the last launch, in microseconds."""
        return max(end - start for start, end in self.timing()) / 1e3

    def task_times(self) -> List[List[List[Tuple[int, int]]]]:
        """Timing build: [gpu][sm][list entry] = (start, the last warp's end, stage stamp 0, stage stamp 1) of the last launch,
        globaltimer ns (0: no task / no stamp). MoE GEMM tasks: stage 0 = first load issued, 1 = first stage landed."""
        return self.module.task_times_func()

    def report(self):
        self.module.report_func()

    def finalize(self):
        if not self._finalized:
            self.module.finalize_func()
            self._finalized = True


def compile_command(cu_path: str, so_path: str, extra_flags: Optional[List[str]] = None) -> List[str]:
    """-O3 -std=c++17, the GPU's own arch, -lcuda, plus what a Python extension
    needs (-shared, -fPIC, the Python headers)."""
    import torch
    from ..kernel import get_key_paths
    _, include_path, _ = get_key_paths()
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"{major}{minor}a" if major >= 9 else f"{major}{minor}"
    scheme = sysconfig.get_default_scheme() if hasattr(sysconfig, "get_default_scheme") else sysconfig._get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    python_include = sysconfig.get_paths(scheme=scheme)["include"]
    return ([shutil.which("nvcc"), "-O3", "-std=c++17", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
             f"-I{include_path}", f"-I{os.path.join(include_path, 'mirage/persistent_kernel')}", f"-I{python_include}",
             "-shared", "-Xcompiler=-fPIC", "-lcuda", cu_path, "-o", so_path]
            + list(extra_flags or []))


def compile_static(pk, schedule_paths: List[str], gpu_tensors: Optional[List[dict]] = None, out_dir: Optional[str] = None,
                   extra_flags: Optional[List[str]] = None, code: Optional[str] = None, profile: bool = False) -> StaticKernel:
    """  1. read the schedule files; write <out_dir>/layer.cu = generate_code (or `code`: a layer.cu given by the caller)
       2. nvcc it into a Python extension (compile_command) and load it
       3. init_func: every GPU's tensors (gpu_tensors[g] = {name: torch tensor on GPU g}) -> static_init -> static_host_init
    profile: the timing build (StaticKernel.task_times)."""
    num_gpus = len(schedule_paths)
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
    with open(cu_path, "w") as f:
        f.write(code if code is not None else generate_code(pk, schedules, profile))

    command = compile_command(cu_path, so_path, extra_flags)
    print("building the layer:", " ".join(command), flush=True)
    subprocess.check_call(command)

    spec = importlib.util.spec_from_file_location("__mirage_static_launcher", so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [list(tensors.keys()) for tensors in gpu_tensors]
    pointers = [[t.data_ptr() for t in tensors.values()] for tensors in gpu_tensors]
    module.init_func(names, pointers)
    return StaticKernel(module, num_gpus, cu_path, so_path)
