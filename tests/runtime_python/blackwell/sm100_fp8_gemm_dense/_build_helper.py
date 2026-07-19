"""Shared on-demand `setup.py build_ext` helper for sm100_fp8_gemm_dense tests.

Each of the 5 tests in this directory wants the same .so available before
import. Centralise the rebuild check so changes to the wrapper or sm100
headers only have to invalidate the cached .so in one place.

The dense f32-block quantizers + reference GEMM now live in
`pytorch_reference.py` (the canonical home). They are re-exported here for
back-compat so any caller doing `from _build_helper import reference_gemm`
keeps working.
"""
import os
import subprocess
import sys

# Re-export the canonical references for back-compat.
from pytorch_reference import (  # noqa: F401,E402
    quantize_a_f32scale,
    quantize_b_f32scale,
    reference_gemm,
    cosine_sim,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SO = "runtime_kernel_blackwell_fp8_gemm_dense"


# ===========================================================================
# SHAPE REGISTRY (the #2 fidelity fix, 2026-06-16): faithfully gate EVERY dense
# decode projection at its CORRECT (K, N), not just qkv_a.
#
# WHY: the faithful harness was hardcoded to qkv_a (K=7168, N=2176). When an
# agent optimized o_proj (K=2048, N=7168) there was NO faithful gate at that
# shape, so the ferret/KDA bridge silently fell back to the STANDALONE green-ctx
# bench -- which MIS-RANKS this family (the whole reason the faithful harness
# exists). The registry parametrizes the test/eval by a named shape.
#
# Each entry is (K, N) for the dense decode FP8 GEMM C[M,N] = A[M,K] @ B[N,K].T,
# at the PRODUCTION TP=8 (world_size=8, num_local_q_heads = 128/8 = 16) config.
# DERIVED from python/mirage/mpk/models/deepseek_v3/builder.py + demo.py (weight
# is stored [N, K]; the SHARD_RULES dim is noted per row). NamedTuple (not a bare
# (K,N) tuple) so a swapped positional never silently invalidates the gate
# (reviewer + Codex 2026-06-16: tests call run_case(N, K, ...) -- different order
# from the registry's (K, N) -- so we hand back NAMED K/N, never a raw tuple).
#
#   qkv_a  : qkv_a_proj.weight = (2176, 7168), REPLICATED (fused q_a+kv_a, full
#            hidden in) -> K=7168 N=2176.   (the ferret finen target)
#   o_proj : o_proj_original.weight = (7168, H*128=16384) full, SHARD dim=1 /8 ->
#            (7168, 2048) -> K=2048 N=7168.  (the ferret gemv_m1 target; the
#            workspace4 candidate's NCH=4 is hardcoded for K=2048 == THIS shape.)
#   q_b    : q_b_nope.weight = (128*128, 1536) full, SHARD dim=0 /8 ->
#            (2048, 1536) -> K=1536 N=2048.
#   q_b_pe : q_b_pe.weight = (128*64, 1536) full, SHARD dim=0 /8 ->
#            (1024, 1536) -> K=1536 N=1024.
#   kv_b   : kv_b_k/kv_b_v.weight = (128*128, 512) full, SHARD dim=0 /8 ->
#            (2048, 512) -> K=512 N=2048.   (the chunked-prefill kv-up shape)
# ===========================================================================
import collections  # noqa: E402

DenseShape = collections.namedtuple("DenseShape", ["name", "K", "N"])

SHAPE_REGISTRY = {
    "qkv_a":  DenseShape("qkv_a",  7168, 2176),
    "o_proj": DenseShape("o_proj", 2048, 7168),
    "q_b":    DenseShape("q_b",    1536, 2048),
    "q_b_pe": DenseShape("q_b_pe", 1536, 1024),
    "kv_b":   DenseShape("kv_b",    512, 2048),
    # Shared-expert MoE dense GEMMs at TP8 decode (M=1). MOE_INTERMEDIATE_SIZE=2048,
    # shared_moe_intermediate = 2048//world_size = 256 at TP8 (builder.py:187).
    #   shared_gate_up: fused [2*256, 7168] col-parallel -> K=7168 N=512.
    #   shared_down:    [7168, 256]        row-parallel -> K=256  N=7168.
    # The remaining unexploited dense front (the ~11x roofline-gap gate_up); same
    # M=1 dense FP8 GEMM family as the 5 above => gemv_m1/finen-able.
    "shared_gate_up": DenseShape("shared_gate_up", 7168, 512),
    "shared_down":    DenseShape("shared_down",     256, 7168),
    # Routing-gate GEMV (router): hidden(7168) -> num_routed_experts(256), M=1.
    # vLLM ref 3us (target <=2.5). Production uses SPLITK_LINEAR (~6.75us); the
    # CUDA-core router GEMV (#180, ~4.48us) was deleted for a TP8 crash (#215) —
    # a TP8-safe GEMV <=2.5us is the open lever. (gemv_m1 KIND @ --shape router.)
    "router":         DenseShape("router",          7168, 256),
}

# The production B200 persistent worker count. The faithful slowCTA is ALWAYS
# measured at this grid (grid.x = num_workers). DSv3 decode TP8 = 136.
PRODUCTION_NUM_WORKERS = 136


def resolve_shape(shape=None, K=None, N=None):
    """Return a DenseShape (named K/N), from a registry name OR explicit K/N.

    * resolve_shape("o_proj")        -> SHAPE_REGISTRY["o_proj"]
    * resolve_shape(K=2048, N=7168)  -> DenseShape("K2048_N7168", 2048, 7168)
    * resolve_shape() / unset        -> the qkv_a default (existing usage
                                        unbroken; additive).

    Reads MPK_TEST_SHAPE if `shape` is None so a test driver can be parametrized
    by env without a CLI plumb. An unknown name raises (loud), never silently
    falls back to qkv_a -- a typo'd shape must FAIL, not mis-gate.
    """
    if K is not None and N is not None:
        return DenseShape(f"K{int(K)}_N{int(N)}", int(K), int(N))
    if shape is None:
        shape = os.environ.get("MPK_TEST_SHAPE", "qkv_a")
    if shape not in SHAPE_REGISTRY:
        raise ValueError(
            f"unknown dense shape {shape!r}; known: "
            f"{sorted(SHAPE_REGISTRY)} (or pass explicit K=,N=). Refusing to "
            f"silently fall back to qkv_a -- a typo'd shape must fail, not "
            f"mis-gate the kernel.")
    return SHAPE_REGISTRY[shape]


def production_num_workers(gpu: int = 0) -> int:
    """The PINNED production grid = num_workers (136 on B200), with a loud
    sanity check.

    The faithful per-task slowCTA is DEFINED as the slowest-CTA body when the
    megakernel launches the production persistent worker grid (grid.x =
    num_workers). On a B200 that is `get_configurations_from_gpu(0)[0]` = 136.
    We read it from the GPU (so it is correct on any SM count) BUT log + warn if
    it is not the expected 136, so a result line is never silently measured at a
    different grid. MPK_TEST_NUM_WORKERS remains the ONLY explicit override (an
    opt-in sweep; NOT for a verdict-grade number).
    """
    import mirage
    workers, _ = mirage.get_configurations_from_gpu(gpu)
    if workers != PRODUCTION_NUM_WORKERS:
        print(f"  [production_num_workers] WARNING: GPU reports {workers} "
              f"workers, expected {PRODUCTION_NUM_WORKERS} (B200 decode TP8). "
              f"The faithful slowCTA will be measured at grid={workers}; cite "
              f"the grid explicitly.", flush=True)
    return workers


def ensure_extension_built(so_name: str = _DEFAULT_SO, force: bool = False) -> None:
    """Build the C++/CUDA extension in this test dir if the .so is missing.

    Args:
        so_name: base name of the extension (no .cpython-...so suffix).
        force: remove any pre-built artifacts and rebuild from scratch.
    """
    import glob
    so_glob = os.path.join(THIS_DIR, f"{so_name}.cpython-*.so")
    if force:
        build_dir = os.path.join(THIS_DIR, "build")
        if os.path.isdir(build_dir):
            import shutil
            shutil.rmtree(build_dir)
        for so in glob.glob(so_glob):
            os.remove(so)
    if not glob.glob(so_glob):
        print(f"Building C++ extension: {so_name}", flush=True)
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=THIS_DIR)


# ---------------------------------------------------------------------------
# Worker-constrained measurement helpers (opt-in; default behaviour unchanged).
#
# WHY: the DSv3 decode megakernel launches only 136 persistent worker CTAs (1
# SM each), and parallel paths shrink a task's REAL SM budget further. A grid
# (e.g. fine-N's CTA count) tuned assuming all 136/148 SMs are free can NULL
# in-MPK. These helpers let a test sweep the worker count N IN the real MPK
# kernel and read latency-vs-N, alongside the existing cos check, so grid-size
# design is done under the real available-SM budget.
# ---------------------------------------------------------------------------

def resolve_num_workers(default_num_workers: int,
                        default_num_schedulers: int = None):
    """Return (num_workers, num_schedulers), honouring MPK_TEST_NUM_WORKERS=N.

    Default (env unset / invalid) -> (`default_num_workers`,
    `default_num_schedulers`) unchanged, so behaviour is byte-identical. When
    set to a positive N, the megakernel launches a grid of N persistent workers
    and the task's grid_dim.x is N -> the kernel runs at grid=N.

    SCHEDULER SCALING (reviewer fix 2026-06-15): schedulers are carved as
    SEPARATE SMs (grid.x = num_workers + num_schedulers/per_sm), and
    `max_worker_per_scheduler = num_workers//min_schedulers + 1` is baked into
    the .so. If we shrink workers but keep the GPU-default scheduler count, the
    worker:scheduler ratio (and the fixed scheduler-SM overhead in the timed
    e2e) shifts with N and CONFOUNDS the latency-vs-N reading. So when N is
    overridden we scale schedulers proportionally to hold worker:scheduler
    roughly constant. Pass `default_num_schedulers` (the GPU value) to enable
    this; if omitted, schedulers are left to the caller.

    Even so, the timed quantity (see ``timed_kernel_run``) is whole-megakernel
    e2e and still carries a fixed scheduler/dispatch floor — read latency-vs-N
    as a SENSITIVITY signal (does this tile blow up as SMs shrink?), not as an
    isolated per-task-body time.
    """
    env = os.environ.get("MPK_TEST_NUM_WORKERS")
    if not env:
        return default_num_workers, default_num_schedulers
    try:
        n = int(env)
    except ValueError:
        print(f"  [MPK_TEST_NUM_WORKERS] ignoring non-int value {env!r}",
              flush=True)
        return default_num_workers, default_num_schedulers
    if n <= 0:
        print(f"  [MPK_TEST_NUM_WORKERS] ignoring non-positive {n}", flush=True)
        return default_num_workers, default_num_schedulers
    sched = default_num_schedulers
    if n != default_num_workers and default_num_schedulers:
        # Hold worker:scheduler ratio ~constant (>=1 scheduler).
        sched = max(1, round(default_num_schedulers * n / default_num_workers))
        print(f"  [MPK_TEST_NUM_WORKERS] overriding num_workers "
              f"{default_num_workers} -> {n}, num_schedulers "
              f"{default_num_schedulers} -> {sched} (ratio held)", flush=True)
    elif n != default_num_workers:
        print(f"  [MPK_TEST_NUM_WORKERS] overriding num_workers "
              f"{default_num_workers} -> {n}", flush=True)
    return n, sched


def timing_iters() -> int:
    """Number of timed iterations from MPK_TEST_TIMING_ITERS (default 0 = off).

    0 (unset) -> no timing; the test runs the kernel once for correctness only,
    byte-identical to before. A positive K -> the caller times K launches and
    reports a median latency (see ``timed_kernel_run``).
    """
    env = os.environ.get("MPK_TEST_TIMING_ITERS")
    if not env:
        return 0
    try:
        k = int(env)
    except ValueError:
        return 0
    return max(k, 0)


def timed_kernel_run(pk, iters: int, warmup: int = 5, label: str = ""):
    """Time `iters` launches of an already-compiled PersistentKernel via CUDA
    events and print + return the median latency in microseconds.

    Returns None if iters <= 0 (caller should fall back to a single plain pk()
    call for the correctness-only path). The kernel is launched warmup+iters
    times; the per-launch elapsed is measured with torch.cuda.Event around each
    pk() so the number is the megakernel's end-to-end wall for this task graph
    at the current (possibly overridden) worker count.
    """
    import statistics

    import torch

    if iters <= 0:
        return None
    # Warmup (also primes any JIT/driver caches so the first timed iter isn't an
    # outlier).
    for _ in range(max(warmup, 0)):
        pk()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        pk()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    times_ms.sort()
    med_us = statistics.median(times_ms) * 1e3
    tag = f" [{label}]" if label else ""
    # NOTE: this is whole-megakernel e2e (scheduler + dispatch + the task), not
    # an isolated task-body time -> use as a SENSITIVITY signal vs N, and for an
    # absolute per-kernel number prefer the ferret standalone bench (PART A).
    print(f"  TIMING{tag}: median={med_us:.2f} us over {iters} iters "
          f"(min={times_ms[0]*1e3:.2f} max={times_ms[-1]*1e3:.2f} us) "
          f"[megakernel e2e]", flush=True)
    return med_us


# ===========================================================================
# FAITHFUL per-task in-MPK latency (the #1 fidelity fix, 2026-06-16).
#
# WHY THIS EXISTS: a standalone kernel bench (ferret / cpp_examples) and the
# whole-megakernel e2e (``timed_kernel_run`` above) both MISLEAD the decode
# campaign:
#   * standalone gives the kernel a dedicated grid on an empty GPU -> a fine-N
#     tile that looks 1.7x faster standalone can NULL in-MPK (the fine-N e2e
#     NULL, 2026-06-16) because in-MPK the kernel runs at the PRODUCTION grid
#     (grid.x = num_workers = 136 persistent worker CTAs, idle CTAs early-
#     return) with the real per-task SETUP/dispatch overhead.
#   * whole-megakernel e2e buries the ~65us fixed scheduler/launch floor on top
#     of the task -> the task signal is swamped.
#
# THE FAITHFUL METRIC: run the REAL single-task megakernel at the production
# grid/worker config, turn the persistent profiler ON (same begin/end CTA
# events the perfetto trace uses), and extract THIS task's per-instance span
# from the profiler buffer:
#   slowCTA_us = max over CTAs of (end-begin)   -- the slowest CTA's BODY
#                (per-instance critical-path cost; for a co-started single-task
#                 dispatch this ~= the call's compute latency)
#   wall_us    = max(end) - min(begin)          -- the call's TRUE latency
#                (includes the multi-wave dispatch stagger across the worker
#                 grid; the thing standalone misses)
# (Identical definitions to scratch/per_position_grid.py, the perfetto-trace
# auditor -- so this number is comparable to the in-MPK profiler numbers.)
#
# Under MPK_TEST_MODE the scheduler runs the task graph EXACTLY ONCE per launch
# (prepare_next_batch finalizes the lone request immediately -- see
# persistent_kernel.cuh ``if (true)`` under MPK_TEST_MODE), so one profiled
# launch == one dispatch instance (event_no=0). We therefore take a MEDIAN over
# ``iters`` separate profiled launches. Each launch is run via ``launch_func``
# directly (NOT PersistentKernel.__call__, which writes a ~4.8s tg4perfetto
# trace per call) and we dump the profiler buffer ourselves with the cheap
# export_to_csv.
#
# SCOPING (be honest -- this is a v1): this captures the per-task in-MPK SETUP +
# real grid/worker dispatch overhead at the production worker count -- the thing
# the standalone bench misses. It is still a SINGLE-TASK megakernel: there is NO
# concurrent co-resident contention (the real decode layer runs ~20 other tasks
# that contend for SMs / L2 / HBM bandwidth alongside this GEMM). Adding a
# co-residency harness (e.g. a representative decode task-graph slice, or a
# parallel filler task) is the v2 refinement.
# ===========================================================================

# Mirror per_position_grid.py: events the auditor excludes (scheduler /
# bookkeeping). For a single-task graph only the task's own events + the
# scheduler events appear; we filter the GEMM task by name in the caller, so
# this set is a belt-and-suspenders guard.
_PROFILER_SKIP_NAMES = {
    "TASK_SCHD_EVENTS", "TASK_SCHD_TASKS", "TASK_SCHD_PREPARE_BATCH",
    "TASK_GET_EVENT", "TASK_GET_NEXT_TASK", "TASK_NVSHMEM_GLOBAL_ARGMAX",
    "TASK_BEGIN_TASK_GRAPH", "TASK_SM100_TASK_END", "TASK_SM100_TASK_BEGIN",
    "TASK_ARGMAX_PARTIAL_SM100",
}

_U32 = 1 << 32


def _span_ns(begin: int, end: int) -> int:
    """Duration of one begin/end pair, modulo 2^32 (the profiler stores raw
    32-bit %globaltimer_lo). A negative/huge wrap (begin>end across the 2^32
    boundary) -> 0, matching per_position_grid.py's ``d()``."""
    x = (end - begin) % _U32
    return x if x < (1 << 31) else 0


def make_profiler_tensor():
    """Allocate the on-device profiler buffer (uint64 entries).

    MUST be created BEFORE PersistentKernel.compile() and passed as the
    ``profiler_tensor`` init param: compile() keys ``-DMIRAGE_ENABLE_PROFILER``
    (and the per-task PROFILER_EVENT_START/END emission) off it being non-None.

    A MPK_TEST_MODE single-task single-iteration graph writes only
    ~2*num_workers + a few scheduler entries (~300 for 136 workers), so we use a
    SMALL default (32768) rather than the demo's 6000*128 = 768K. This matters
    for SPEED: ``export_to_csv`` walks the WHOLE buffer in pure Python every
    launch, so a 768K buffer makes a 40-launch median take minutes; 32K keeps it
    snappy while leaving >100x headroom. Override via MPK_TEST_PROFILER_ENTRIES
    (e.g. raise it if a future multi-iteration test overflows -- the CSV export
    warns loudly on overflow).
    """
    import torch
    entries = int(os.environ.get("MPK_TEST_PROFILER_ENTRIES", 32768))
    return torch.zeros(entries, dtype=torch.uint64, device="cuda")


def _parse_task_span_from_csv(csv_path: str, task_name: str):
    """Read one profiled-launch CSV and return (n_ctas, slowCTA_us, wall_us) for
    the named task's single dispatch instance, or None if it's absent.

    The CSV (profiler_persistent.export_to_csv) has one row per begin/end pair:
    task_type_name, block_idx, group_idx, event_no, begin_ts, end_ts,
    duration_ns. For a MPK_TEST_MODE single-task graph the task is dispatched
    once across the active worker CTAs (idle CTAs early-return and emit nothing
    OR emit a near-zero-duration body); each active CTA = one row at event_no=0.
    """
    import csv as _csv
    rows = []
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            nm = r.get("task_type_name", "")
            if nm != task_name:
                continue
            try:
                b = int(r["begin_ts"]); e = int(r["end_ts"])
                bi = int(r["block_idx"])
            except (KeyError, ValueError):
                continue
            rows.append((b, e, bi))
    if not rows:
        return None
    # event_no is not split out here: MPK_TEST_MODE runs the graph once/launch,
    # so all rows belong to the single dispatch instance. (If a future config
    # runs multiple iters/launch, group by event_no first.)
    n_ctas = len({bi for _, _, bi in rows})
    slow_us = max(_span_ns(b, e) for b, e, _ in rows) / 1000.0
    wall_us = ((max(e for _, e, _ in rows) - min(b for b, _, _ in rows))
               % _U32) / 1000.0
    return n_ctas, slow_us, wall_us


def _launch_only(pk):
    """Run the megakernel ONCE without PersistentKernel.__call__'s per-call
    trace export. The default __call__ writes a perfetto trace on EVERY launch
    when profiling is on -- tg4perfetto serialization is ~4.8s/call and (a)
    makes a K-iteration median take minutes, (b) pollutes any wall-clock e2e
    reading. We bypass it and read the profiler buffer ourselves (cheap CSV),
    so the only GPU work timed is the megakernel itself."""
    import torch
    stream = torch.cuda.current_stream()
    pk.launch_func(int(stream.cuda_stream))


# ===========================================================================
# GPU-EXCLUSIVITY pre/post check (2026-06-16, MEASUREMENT INTEGRITY).
#
# WHY: the faithful slowCTA@136 is only a TRUSTED number on an EXCLUSIVE card.
# A co-resident compute process (a leftover test, a classmate's job) contends
# for SMs / L2 / HBM bandwidth and inflates the wall (and -- if MPS is ever on,
# or simply via the scheduler -- can perturb the slowCTA body). The user flagged
# that up to 3 leftover test procs squatted on the measurement card for ~8h; a
# faithful number must therefore SELF-CERTIFY exclusivity, and FAIL-CLOSED (same
# as cos<floor / wall-regress / suspect-CTA) if the card is contended.
#
# This is the SINGLE chokepoint: every faithful measurement (finen test, gemv
# test, the ferret faithful_eval bridge, the KDA testmode_correctness bridge)
# routes its timing through ``profiled_per_task_latency`` below, so guarding HERE
# covers all of them. The bridges ALSO call this at their subprocess boundary
# (belt-and-suspenders), but THIS in-process check is the authoritative one: it
# runs in the very process that owns the megakernel, so "this measurement's own
# PID (+ its children)" is unambiguous.
# ===========================================================================

def _own_pid_tree():
    """Set of PIDs that ARE this measurement: this process + every descendant.

    The megakernel runs IN-PROCESS (``pk.launch_func`` on this python's CUDA
    context), so os.getpid() is the measuring PID. But torch/NCCL/JIT can fork
    helper children that ALSO show up as compute apps on the card, so we include
    the whole descendant tree -- otherwise we'd false-positive on our own helper.
    Pure-/proc walk (no psutil dependency); silently tolerant of races (a child
    that exits mid-walk just drops out).
    """
    import os as _os
    me = _os.getpid()
    own = {me}
    # Build child-map from /proc/<pid>/stat field #4 (ppid). Cheap, no deps.
    children = {}
    try:
        for d in _os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as f:
                    parts = f.read().split()
                # stat: pid (comm...) state ppid ...  -> ppid is the field AFTER
                # the (possibly space-containing) comm. Find the ')' then +2.
                rparen = None
                with open(f"/proc/{d}/stat") as f2:
                    raw = f2.read()
                rparen = raw.rfind(")")
                ppid = int(raw[rparen + 1:].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            children.setdefault(ppid, []).append(int(d))
    except OSError:
        return own
    # BFS from me down the tree.
    stack = [me]
    while stack:
        p = stack.pop()
        for c in children.get(p, []):
            if c not in own:
                own.add(c)
                stack.append(c)
    return own


def _resolve_physical_gpu_index():
    """The PHYSICAL nvidia-smi index of the card this process is computing on.

    ``nvidia-smi -i <N>`` is a PHYSICAL index that IGNORES CUDA_VISIBLE_DEVICES,
    but the bridges set CUDA_VISIBLE_DEVICES=<gpu> so torch's logical device 0 IS
    that physical card. So:
      * CUDA_VISIBLE_DEVICES set to a single integer (the bridge contract) ->
        that integer is the physical index (logical 0 == physical CVD).
      * otherwise, match the CURRENT torch device's UUID against
        ``nvidia-smi --query-gpu=index,uuid`` (robust to any remap / MIG).
    Returns an int physical index, or None if it cannot be resolved (caller then
    SKIPS the check loudly rather than fail-closing on a harness limitation).
    """
    import os as _os
    cvd = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    toks = [t for t in cvd.replace(",", " ").split() if t != ""]
    if len(toks) == 1 and toks[0].isdigit():
        # Single visible card: torch logical 0 == this physical index.
        return int(toks[0])
    # Fall back to UUID match for the current torch device.
    try:
        import subprocess as _sp

        import torch as _torch
        dev = _torch.cuda.current_device()
        # torch >= 2.x exposes the device UUID.
        uuid = None
        try:
            uuid = str(_torch.cuda.get_device_properties(dev).uuid)
        except Exception:
            uuid = None
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True)
        for line in out.splitlines():
            idx_s, _, uid = line.partition(",")
            idx_s, uid = idx_s.strip(), uid.strip()
            if uuid and uuid in uid:
                return int(idx_s)
    except Exception:
        pass
    return None


def gpu_exclusivity_check(phase: str = "", physical_gpu: int = None):
    """Query the card for FOREIGN compute processes and return a verdict dict.

    Returns {exclusive: bool, gpu: int|None, foreign_pids: [(pid, mem)], note}.

    ``exclusive`` is True iff EVERY compute process on the physical card belongs
    to this measurement's own PID tree (``_own_pid_tree``). A card we cannot
    resolve / query (no nvidia-smi, CVD not a single int + UUID match failed) is
    reported exclusive=None (UNKNOWN) so the caller can decide -- we do NOT
    fail-closed on a harness limitation, only on a CONFIRMED foreign process.
    """
    import subprocess as _sp
    gpu = physical_gpu if physical_gpu is not None else \
        _resolve_physical_gpu_index()
    tag = f" [{phase}]" if phase else ""
    if gpu is None:
        print(f"  [gpu_exclusivity{tag}] WARNING: could not resolve the physical "
              f"GPU index (CUDA_VISIBLE_DEVICES not a single int and UUID match "
              f"failed) -- exclusivity UNKNOWN, not verified.", flush=True)
        return {"exclusive": None, "gpu": None, "foreign_pids": [],
                "note": "unresolved_gpu_index"}
    try:
        out = _sp.check_output(
            ["nvidia-smi", "-i", str(gpu),
             "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True, stderr=_sp.STDOUT)
    except Exception as e:  # nvidia-smi missing / errored
        print(f"  [gpu_exclusivity{tag}] WARNING: nvidia-smi query failed on GPU "
              f"{gpu} ({e}) -- exclusivity UNKNOWN, not verified.", flush=True)
        return {"exclusive": None, "gpu": gpu, "foreign_pids": [],
                "note": "nvidia_smi_failed"}
    own = _own_pid_tree()
    foreign = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, mem_s = line.partition(",")
        pid_s = pid_s.strip()
        if not pid_s.isdigit():
            continue
        pid = int(pid_s)
        if pid in own:
            continue
        foreign.append((pid, mem_s.strip()))
    exclusive = (len(foreign) == 0)
    if not exclusive:
        # The LOUD line the task asks for; identical phrasing both phases.
        print(f"  [gpu_exclusivity{tag}] GPU NOT EXCLUSIVE: other PIDs="
              f"{[p for p, _ in foreign]} (mem={[m for _, m in foreign]}) on "
              f"physical GPU {gpu} -> FAITHFUL MEASUREMENT INVALID", flush=True)
    return {"exclusive": exclusive, "gpu": gpu, "foreign_pids": foreign,
            "note": "ok" if exclusive else "foreign_compute_procs"}


def profiled_per_task_latency(pk, task_name: str, iters: int,
                              warmup: int = 5, label: str = "",
                              grid: int = None, tile_ctas: int = None):
    """Run ``iters`` profiled launches of an already-compiled PersistentKernel
    (its profiler_tensor MUST be set) and report the MEDIAN faithful per-task
    in-MPK latency for ``task_name``.

    ``grid``: the LAUNCHED persistent worker grid (= pk.num_workers); printed as
    ``grid=<n>`` so the worker count behind the slowCTA is NEVER ambiguous
    (defaults to pk.num_workers if None). ``tile_ctas``: optional analytic count
    of output-tile-owning CTAs (ceil(N/BN) for this kernel) -- printed alongside
    the PROFILED CTA count so the reader can see how many of the launched
    workers actually owned a tile vs early-returned (reviewer + Codex 2026-06-16:
    the profiler wraps the whole task call, so early-return dense workers STILL
    emit a short row -> the profiled-CTA count is NOT the active-tile count).

    Returns a dict {profiled_ctas, grid, tile_ctas, slowCTA_us, wall_us, e2e_us}
    of medians, or None if iters<=0 or the task never appears in the profiler
    buffer.

    Under MPK_TEST_MODE the megakernel runs the task graph ONCE per launch and
    the profiler buffer is rewritten in place each launch (worker write ptrs
    reset at PROFILER_INIT) -> one launch == one dispatch instance. We launch
    the megakernel directly (``_launch_only``, no perfetto export), bracket it
    with CUDA events for the true whole-megakernel e2e, then dump THIS launch's
    profiler buffer to a CSV and parse the task's per-instance span. The median
    over ``iters`` launches is the reported faithful per-task latency.
    """
    import statistics

    import torch

    from mirage.mpk.profiler_persistent import export_to_csv

    if iters <= 0:
        return None
    if grid is None:
        grid = getattr(pk, "num_workers", None)
    csv_path = (pk.trace_name if getattr(pk, "trace_name", "") else
                f"mirage_{pk.mpi_rank}") + "._faithful.csv"

    # GPU-EXCLUSIVITY pre-check (2026-06-16): a faithful slowCTA is only trusted
    # on an EXCLUSIVE card. If ANY foreign compute process shares this physical
    # GPU, FAIL-CLOSED with the loud line + a clearly-invalid 0.0 verdict (same
    # fail-closed contract as cos<floor / suspect-CTA): _tp(0.0)=0 -> ratio 0 <
    # target -> the gate BLOCKS. exclusive=None (UNKNOWN, harness could not
    # resolve/query the card) does NOT fail-close -- only a CONFIRMED foreign proc
    # does -- but is surfaced as gpu_exclusive=unknown so the number is not
    # silently self-certified.
    excl_pre = gpu_exclusivity_check(phase="pre")
    if excl_pre["exclusive"] is False:
        print(f"  in-MPK per-task{(' [' + label + ']') if label else ''}: "
              f"slowCTA=0.00 us  wall=0.00 us  grid={grid}  "
              f"gpu_exclusive=no  -> FAITHFUL MEASUREMENT INVALID "
              f"(foreign PIDs {[p for p, _ in excl_pre['foreign_pids']]} on GPU "
              f"{excl_pre['gpu']}; refusing to produce a trusted number on a "
              f"contended card)", flush=True)
        return {
            "profiled_ctas": 0, "n_ctas": 0, "grid": grid,
            "tile_ctas": tile_ctas, "cta_constant": True,
            "slowCTA_us": 0.0, "wall_us": 0.0, "e2e_us": 0.0,
            "gpu_exclusive": False, "gpu": excl_pre["gpu"],
            "foreign_pids": excl_pre["foreign_pids"],
            "invalid_reason": "gpu_not_exclusive",
        }

    for _ in range(max(warmup, 0)):
        _launch_only(pk)
    torch.cuda.synchronize()

    # NOTE: do NOT zero the profiler buffer between launches. The persistent
    # megakernel launches on its own stream; a torch .zero_() on the current
    # stream RACES the kernel's profiler writes (observed: it clobbers the GEMM
    # events, leaving only a scheduler entry). It is also unnecessary: under
    # MPK_TEST_MODE every launch runs the graph once and rewrites the SAME
    # slots with the SAME number of events (worker write-ptrs reset at
    # PROFILER_INIT), so successive launches overwrite cleanly in place.
    #
    # The no-zero correctness HINGES on the per-launch CTA count being constant
    # (a launch that drops an event would leave a stale slot from a prior launch
    # and silently corrupt max(end)/min(begin)). Both the ablation-reviewer and
    # Codex flagged this as the latent measurement bug, so we GUARD it: collect
    # the per-launch distinct-CTA count and warn loudly (and report the spread)
    # if it ever varies. (Empirically constant at 133 over 40 launches for
    # qkv_a M=1 -- the 3-vs-136 gap is the scheduler-reserved workers that never
    # get this task dispatched; deterministic, not a dropped-event hazard.)
    slow_samples, wall_samples, e2e_ms, n_ctas_seen = [], [], [], []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _launch_only(pk)   # megakernel only (scheduler + dispatch + the task)
        end.record()
        torch.cuda.synchronize()
        e2e_ms.append(start.elapsed_time(end))
        # Dump THIS launch's profiler buffer (cheap pure-Python CSV) and parse.
        export_to_csv(pk.profiler_tensor, csv_path)
        parsed = _parse_task_span_from_csv(csv_path, task_name)
        if parsed is None:
            continue
        n_ctas, slow_us, wall_us = parsed
        n_ctas_seen.append(n_ctas)
        slow_samples.append(slow_us)
        wall_samples.append(wall_us)

    # GPU-EXCLUSIVITY post-check (2026-06-16): a foreign process could have
    # LANDED on the card mid-measurement (the ~8h-squatter scenario in reverse).
    # Re-query at the END; if exclusivity broke during timing, the number is
    # contaminated -> FAIL-CLOSED exactly like the pre-check.
    #
    # HONEST SCOPE (reviewer + Codex 2026-06-16): pre+post are ENDPOINT samples.
    # They catch a foreign proc present at start, present at end, or that landed
    # mid-run and STAYED (the common contention cases). They CANNOT prove the
    # absence of a fully-transient proc that both starts AND exits entirely
    # inside the timing window. For a hard guarantee you'd need CUDA
    # EXCLUSIVE_PROCESS compute-mode or a cgroup/scheduler reservation; this is
    # the best practical in-harness guard, not a proof of zero overlap.
    excl_post = gpu_exclusivity_check(phase="post")
    if excl_post["exclusive"] is False:
        print(f"  in-MPK per-task{(' [' + label + ']') if label else ''}: "
              f"slowCTA=0.00 us  wall=0.00 us  grid={grid}  "
              f"gpu_exclusive=no  -> FAITHFUL MEASUREMENT INVALID "
              f"(exclusivity BROKE mid-measurement: foreign PIDs "
              f"{[p for p, _ in excl_post['foreign_pids']]} on GPU "
              f"{excl_post['gpu']} appeared during timing)", flush=True)
        return {
            "profiled_ctas": 0, "n_ctas": 0, "grid": grid,
            "tile_ctas": tile_ctas, "cta_constant": True,
            "slowCTA_us": 0.0, "wall_us": 0.0, "e2e_us": 0.0,
            "gpu_exclusive": False, "gpu": excl_post["gpu"],
            "foreign_pids": excl_post["foreign_pids"],
            "invalid_reason": "gpu_exclusivity_broke_mid_measurement",
        }
    # exclusive==True both ends -> certified; None (UNKNOWN) at either -> unknown.
    gpu_exclusive = (True if (excl_pre["exclusive"] and excl_post["exclusive"])
                     else (None if (excl_pre["exclusive"] is None
                                    or excl_post["exclusive"] is None)
                           else False))

    if not slow_samples:
        print(f"  [profiled_per_task] task {task_name!r} not found in profiler "
              f"buffer ({csv_path}) -- did profiling compile in? "
              f"(profiler_tensor set before compile()?)", flush=True)
        return None

    # GUARD (reviewer+Codex A2): a varying CTA count across launches means the
    # no-zero buffer reuse may be leaking stale events into the parse -> the
    # median is untrustworthy. Constant count == clean overwrite.
    cta_constant = (min(n_ctas_seen) == max(n_ctas_seen))
    if not cta_constant:
        print(f"  [profiled_per_task] WARNING: per-launch CTA count for "
              f"{task_name!r} VARIED across launches "
              f"(min={min(n_ctas_seen)} max={max(n_ctas_seen)}) -- the no-zero "
              f"profiler-buffer reuse may be leaking stale events; treat the "
              f"latency below as SUSPECT. (Expected constant under MPK_TEST_MODE "
              f"single-iter; a varying count indicates dropped/extra events.)",
              flush=True)

    res = {
        # n_ctas kept as a back-compat alias for callers that still read it;
        # profiled_ctas is the correctly-named field (it is the count of CTAs
        # that emitted a profiler row -- which, because the profiler wraps the
        # whole task call, INCLUDES early-return workers, so it is NOT the
        # active-tile count). tile_ctas is the analytic active-tile owner count.
        "profiled_ctas": max(n_ctas_seen),
        "n_ctas": max(n_ctas_seen),
        "grid": grid,
        "tile_ctas": tile_ctas,
        "cta_constant": cta_constant,
        "slowCTA_us": statistics.median(slow_samples),
        "wall_us": statistics.median(wall_samples),
        "e2e_us": statistics.median(e2e_ms) * 1e3,
        # MEASUREMENT INTEGRITY: every trusted number self-certifies exclusivity.
        "gpu_exclusive": gpu_exclusive,
        "gpu": excl_post["gpu"] if excl_post["gpu"] is not None
        else excl_pre["gpu"],
    }
    tag = f" [{label}]" if label else ""
    grid_s = f"grid={grid}" if grid is not None else "grid=?"
    tile_s = (f" tile_ctas={tile_ctas}" if tile_ctas is not None else "")
    excl_s = ("yes" if gpu_exclusive is True
              else ("unknown" if gpu_exclusive is None else "no"))
    print(f"  in-MPK per-task{tag}: slowCTA={res['slowCTA_us']:.2f} us  "
          f"wall={res['wall_us']:.2f} us  {grid_s}  "
          f"profiled_ctas={res['profiled_ctas']}{tile_s}"
          f"{'' if cta_constant else '(VARIED!)'}  "
          f"gpu_exclusive={excl_s}  "
          f"(megakernel e2e={res['e2e_us']:.2f} us; n={len(slow_samples)} "
          f"profiled launches)", flush=True)
    print(f"    slowCTA = slowest single-CTA BODY (per-instance compute);  "
          f"wall = max(end)-min(begin) incl. dispatch stagger.  [REAL MPK "
          f"execution path, LAUNCHED {grid_s}; profiled_ctas counts CTAs that "
          f"emitted a profiler row (early-return workers included, so >= "
          f"tile_ctas); SINGLE-task: no co-resident contention]",
          flush=True)
    return res
