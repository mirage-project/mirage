"""Let the superoptimizer choose an MPK task's schedule.

Today every generated MPK task is a threadblock graph a person wrote by hand
(persistent_kernel.py's generated_* layers). The transpiler generates the
CUDA, but the *schedule* it lowers -- grid_dim, block_dim, forloop_range, and
each input's input_map and forloop_dim -- is a set of literals someone chose.
Choosing schedules is exactly what search() does, and its interface already
fits: it takes a muGraph (what to compute) and returns candidates carrying a
threadblock graph (how to compute it), which is precisely what
`register_task(bgraph, "generated")` accepts.

The unit matters. Searching a whole model is intractable, and searching every
torch.compile partition means paying a ~20-minute search per partition. But
tasks are REUSED: one SwiGLU task type serves all 28 Qwen3 layers. Searching
once per task *type* is a handful of searches, cached.

    spec    = TaskSpec("swiglu", build_fn, [TensorSpec(...), ...])
    scheds  = search_task_schedules(spec, grid_dim=(1, 1, 1))
    register_searched_task(pk, scheds[i], inputs=[...], output=od)

SELECTION IS NOT PART OF THIS MODULE, and search() does not do it either:
search enumerates schedules and verifies they compute the same thing (a
fingerprint check), but never measures performance. superoptimize() does
measure, by timing each candidate as a STANDALONE kernel -- which is the
wrong number for a task that will run inside a megakernel, on one persistent
worker CTA, with cold L2 and no launch of its own.

Even per-task in-MPK latency is the wrong objective. Measured on Qwen3-0.6B:
a silu_mul schedule that was 1.20x faster per task (8800 -> 7360 ns) left
the model 2.5% SLOWER end to end, because silu_mul is the cheapest of the
three MLP tasks and the matmuls that dominate were untouched. A task can get
faster while the model does not.

So a candidate is ranked by building the whole model with it and measuring
that model's throughput. experiments/searched_tasks/rank_by_model.py drives
it; Schedule.to_dict/from_dict exist because each candidate needs its own
process (one megakernel, one CUDA context).
"""
from __future__ import annotations

import json
import logging
import os

import mirage as mi
from mirage.core import CyTBGraph
from mirage.kernel import KNGraph, search
from mirage.threadblock import TBGraph

# MPK launches its workers at WORKER_NUM_THREADS
# (persistent_kernel/tasks/common/worker_config.h), and every worker thread
# runs the task body. The transpiler bakes NUM_THREADS into named-barrier
# widths, so a body compiled for any other width traps in wg_sync. Search must
# therefore be constrained to this block size: left at its default,
# get_block_dim_cand only ever proposes {128,1,1} and nothing it finds would
# be runnable.
MPK_BLOCK_DIM = (256, 1, 1)

log = logging.getLogger(__name__)

# Winning schedules, keyed by task type AND exact operand shapes, because a
# schedule is only known good for what it was measured on: fl=4 beat the
# hand-written fl=16 for (8,1024)@(1024,3072) at batch 8, which says nothing
# about another shape. hidden_size // 64 remains a perfectly reasonable rule;
# it just is not optimal there. A miss therefore has to leave the caller's
# existing behaviour completely alone.
SCHEDULE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "searched_schedules.json")

# Set MPK_SEARCHED_SCHEDULES=0 to ignore the cache entirely and use the
# hand-written schedules, e.g. to A/B a regression against it.
def _cache_enabled():
    return os.environ.get("MPK_SEARCHED_SCHEDULES", "1") != "0"


def cache_key(task_name, input_dims, grid_dim, block_dim=MPK_BLOCK_DIM):
    shapes = ";".join("x".join(str(d) for d in dims) for dims in input_dims)
    return (f"{task_name}|{shapes}|grid={'x'.join(str(d) for d in grid_dim)}"
            f"|block={'x'.join(str(d) for d in block_dim)}")


def _load_cache():
    if not os.path.exists(SCHEDULE_CACHE_PATH):
        return {}
    try:
        with open(SCHEDULE_CACHE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        log.warning("mpk: could not read %s (%s); using hand-written "
                    "schedules", SCHEDULE_CACHE_PATH, e)
        return {}


def lookup_schedule(task_name, input_dims, grid_dim, block_dim=MPK_BLOCK_DIM):
    """A measured-best schedule for exactly this task and shape, or None."""
    if not _cache_enabled():
        return None
    entry = _load_cache().get(cache_key(task_name, input_dims, grid_dim,
                                         block_dim))
    if entry is None:
        return None
    return Schedule.from_dict(entry["schedule"])


def store_schedule(task_name, input_dims, grid_dim, sched, provenance,
                    block_dim=MPK_BLOCK_DIM, path=None):
    """Record a winner. `provenance` says what measurement justified it --
    without it an entry is an unfalsifiable claim that some schedule is
    better."""
    path = path or SCHEDULE_CACHE_PATH
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            cache = json.load(f)
    cache[cache_key(task_name, input_dims, grid_dim, block_dim)] = {
        "schedule": sched.to_dict(),
        "provenance": provenance,
    }
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


class TaskSearchError(Exception):
    """No schedule search returned is usable as an MPK task."""


class TensorSpec:
    """One muGraph input: what the task consumes."""

    def __init__(self, dims, dtype=None, strides=None):
        self.dims = tuple(dims)
        self.dtype = dtype if dtype is not None else mi.bfloat16
        # Row-major unless stated. A weight consumed as (K, N) is declared
        # that way rather than transposed at runtime, matching how the
        # hand-written generated_* layers take already-transposed weights.
        self.strides = tuple(strides) if strides is not None else _row_major(dims)


def _row_major(dims):
    strides, acc = [], 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


class TaskSpec:
    """WHAT one task computes, independent of how.

    build(kn, inputs) -> the single output DTensor. Written the way a person
    thinks about the task ("two matmuls, silu one, multiply"), with no tiling
    or loop decisions in it -- those are what search supplies.
    """

    def __init__(self, name, build, inputs):
        self.name = name
        self.build = build
        self.inputs = list(inputs)


class Schedule:
    """A discovered implementation of a TaskSpec, replayable into a TBGraph."""

    def __init__(self, name, grid_dim, block_dim, forloop_range,
                 reduction_dimx, ops, num_inputs):
        self.name = name
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.forloop_range = forloop_range
        self.reduction_dimx = reduction_dimx
        self.ops = ops                # bgraph operator dicts, topological
        self.num_inputs = num_inputs

    def describe(self):
        kinds = [o["op_type"].replace("tb_", "").replace("_op", "")
                 for o in self.ops]
        return (f"{self.name}: grid={self.grid_dim} block={self.block_dim} "
                f"forloop_range={self.forloop_range} ops={kinds}")

    # Ranking a schedule means building a whole model with it and measuring
    # that model, which has to happen in a separate process (one megakernel,
    # one CUDA context, per candidate). So a schedule has to survive a trip
    # through JSON.
    def to_dict(self):
        return {
            "name": self.name,
            "grid_dim": list(self.grid_dim),
            "block_dim": list(self.block_dim),
            "forloop_range": self.forloop_range,
            "reduction_dimx": self.reduction_dimx,
            "ops": self.ops,
            "num_inputs": self.num_inputs,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            grid_dim=tuple(d["grid_dim"]),
            block_dim=tuple(d["block_dim"]),
            forloop_range=d["forloop_range"],
            reduction_dimx=d["reduction_dimx"],
            ops=d["ops"],
            num_inputs=d["num_inputs"],
        )


def _build_spec_graph(spec):
    kn = mi.new_kernel_graph()
    tensors = [kn.new_input(dims=s.dims, strides=s.strides, dtype=s.dtype)
               for s in spec.inputs]
    out = spec.build(kn, tensors)
    kn.mark_output(out)
    return kn


_WIDE_INPUTS_NOTE = """Why a >3-input task spec is not practical yet.

Three separate things had to be true, and only two are fixed.

1. search's default caps. max_num_threadblock_graph_inputs is 3 and
   max_num_kernel_graph_op is 5 -- and the latter counts KN_INPUT_OPs, so a
   3-input spec fits EXACTLY (3 inputs + one customized op + one output) and a
   4-input one cannot fit at all. Both are now settable from Python
   (max_tb_graph_inputs / max_kn_graph_ops on mirage.core.search).

2. dim_strategy's get_customized_input_cand_idx hardcoded which input
   combinations a customized op may consume: {0,1,2} at exactly three inputs,
   and otherwise ONLY THE LAST TWO. So with four inputs no candidate could
   ever consume them all, whatever the caps said. Measured on a+b+c+d, the
   simplest 4-input spec there is: 161 graphs, ZERO fully fused. It now also
   offers the all-inputs combination when the configured cap allows.

3. The search space -- which turned out to be FINE. An earlier version of this
   note claimed four inputs would not converge, citing 4.75 million states in
   631s on a+b+c+d. That blowup was caused by raising max_tb_graph_ops to 24
   while probing. At the DEFAULT threadblock op limit of 9, the full attention
   core searches in 1.7 SECONDS. The threadblock op cap, not the input count,
   is what governs cost, so leave it alone unless a body genuinely needs more.
   Pinning imaps to the batch dim helps further and is what a per-instance MPK
   task wants anyway.

So a >3-input spec works. exp(Q@K^T + mask) @ V searches, registers and is
numerically correct (rel 2.9e-3), returning a body identical to the
hand-written generated_attention_layer. CHAINING was never a barrier either.

forloop_range used to limit it -- a chained matmul was silently wrong past the
first loop iteration -- but that was a backend defect, not a scheduling one,
and it is fixed: transpiler_tb_blackwell.cc derived the N_LOOP flag as
`forloop_dim == 1`, which names the N dim only for a 2-D operand and inverted
for a batched one. The attention core now runs correctly at forloop_range 2
and 4."""


def search_task_schedules(spec, grid_dim=None, forloop_range=None,
                           verbose=False, max_ops=None, wide_inputs=False):
    """Every distinct MPK-usable implementation of `spec` that search found.

    Returned UNRANKED. search() enumerates and verifies equivalence; it never
    measures performance, so ordering here carries no information about speed.
    Ranking is the caller's job and must be done on whole-model throughput --
    a task measured alone can improve while the model does not (measured: a
    silu_mul schedule 1.20x faster per task left Qwen3-0.6B 2.5% slower).

    grid_dim / forloop_range narrow the space when the caller already knows
    the task granularity it wants (MPK expands grid_dim into one task
    instance per cell, so it is a scheduling decision, not just a perf one).

    A spec with more than THREE inputs needs wide_inputs=True; see
    _WIDE_INPUTS_NOTE for what that lifts. max_ops raises the threadblock op
    limit (default 9), but raising it is what makes a wide search expensive --
    the attention core needs 10 threadblock ops and searches in under two
    seconds at the default, so leave max_ops alone unless a body needs it.
    """
    if len(spec.inputs) > 3 and not wide_inputs:
        raise TaskSearchError(
            f"{spec.name!r} has {len(spec.inputs)} inputs; search fuses at "
            f"most 3 by default and would return nothing at all. Pass "
            f"wide_inputs=True to lift the caps; see "
            f"task_search._WIDE_INPUTS_NOTE.")
    kn = _build_spec_graph(spec)
    required = _required_ops(kn)

    griddims = [tuple(grid_dim)] if grid_dim is not None else None
    franges = [int(forloop_range)] if forloop_range is not None else None

    cygraphs = search(
        kn.cygraph, backend="cuda", imaps=None, omaps=None,
        griddims=griddims,
        blockdims=[MPK_BLOCK_DIM],   # mandatory -- see MPK_BLOCK_DIM
        fmaps=None, franges=franges,
        previous_checkpoint=None, verbose=verbose,
        default_config=None, is_formal_verified=False,
        max_tb_graph_ops=(int(max_ops) if max_ops else -1),
        # Both caps stay at their defaults unless the caller opted in, so
        # every <=3-input spec searches exactly as it did before.
        max_tb_graph_inputs=(len(spec.inputs) if wide_inputs else -1),
        max_kn_graph_ops=(len(spec.inputs) + 4 if wide_inputs else -1),
    )
    candidates = [KNGraph(g) for g in cygraphs]

    scheds, rejected = [], []
    seen = set()
    for cand in candidates:
        try:
            sched = _as_schedule(spec, cand, required)
        except TaskSearchError as e:
            rejected.append(str(e))
            continue
        # search emits the same schedule many times over (it explores
        # orderings that lower identically); ranking each duplicate would
        # multiply the cost of the model runs for nothing.
        key = (sched.grid_dim, sched.block_dim, sched.forloop_range,
               tuple((o["op_type"],
                      tuple(_imap(o["input_map"])) if "input_map" in o else (),
                      o.get("forloop_dim", -2)) for o in sched.ops))
        if key in seen:
            continue
        seen.add(key)
        scheds.append(sched)

    if not scheds:
        raise TaskSearchError(
            f"search returned {len(candidates)} candidate(s) for {spec.name!r}, "
            f"none usable as an MPK task. Reasons: "
            f"{sorted(set(rejected))[:5]}"
        )
    return scheds


def search_task_schedule(spec, **kw):
    """The first MPK-usable schedule.

    Note this does NOT rank: search() itself only enumerates and verifies
    equivalence (a fingerprint check), it never measures performance, and
    this returns whichever valid candidate came first. Use
    search_task_schedules() plus a whole-model objective to actually choose.
    """
    return search_task_schedules(spec, **kw)[0]


# The tcgen05 F16/BF16 MMA atom consumes a 32-byte K slice, so 16 elements at
# 16-bit. A K tile below that is not a whole atom.
MMA_K_ATOM = 16

# ---------------------------------------------------------------------------
# What tile a group is given, which is the other half of a schedule: search
# enumerates bodies, these decide the shape the body is asked for. They live
# beside the guards below because they encode the same hardware limits -- a
# legal MMA M tile, a whole number of MMA K-atoms, a TMA box that fits.
#
# None of them RANK. A wider tile buys a bigger MMA and costs parallelism, and
# which way that lands is whole-model throughput, so they enumerate and the
# caller measures.
# ---------------------------------------------------------------------------

MATMUL_N_TILES = (64, 128)


def default_grid(group: Group, n_tile: int = 64) -> tuple[int, int, int]:
    """One `n_tile`-wide column of the output per threadblock.

    64 is what every hand-written generated_* layer uses
    (grid_dim=(N // 64, 1, 1)), and is the default here. It is not obviously
    the best: the hand-written linear_sm100 hardcodes MMA_M = 128, so at 64
    a generated task issues an MMA covering half the output columns per
    instruction. Which wins is a measurement -- see grid_candidates.
    """
    n = group.output.dims[-1]
    if n % n_tile:
        raise ValueError(
            f"group {group.tag!r} output last dim {n} is not a multiple of "
            f"{n_tile}; pass an explicit grid for it")
    return (n // n_tile, 1, 1)


def _has_matmul(graph: ModelGraph, group: Group) -> bool:
    """Does this group contain a matmul ANYWHERE, not just as its first node?

    A partition is free to put the elementwise ops before the matmul --
    silu | mul | down fuses as one group whose first node is a silu -- and such
    a group still needs a K loop. Testing only node 0 gave it forloop_range=1,
    which asks search for a single 3072-wide K step: over the 256 a TMA box
    allows, so search returns nothing and the group has no fallback.
    """
    return any(graph.nodes[i].op == "matmul" for i in group.nodes)


def has_rms_norm(graph: ModelGraph, group: Group) -> bool:
    return any(graph.nodes[i].op == "rms_norm" for i in group.nodes)


def rows_grid_candidates(group: Group) -> list[tuple]:
    """How many rows one block of a reducing group takes: 1, 2, 4, ... all.

    default_grid would give each block an `n_tile`-wide slice of the OUTPUT's
    last dim, but for a norm that dim is the one being reduced over -- a
    64-wide slice asks each block to normalise a sixteenth of a row, which is
    a different computation. Measured at Qwen3's norm shape ((8, 1024) *
    (1, 1024)): grid (16, 1, 1) returns only candidates that leave the weight
    multiply at kernel level, which task_search rejects, while (8, 1, 1) and
    (1, 1, 1) each return one usable schedule. So split rows, not columns.

    How MANY rows is the reduction's parallelism knob, and it pulls two ways.
    tb::ReductionKernel gives each OUTPUT a warp, and a group that reduces
    over its last dim has one output per row -- so a block holding one row
    keeps one warp of eight busy, and a block holding eight keeps all eight.
    But rows per block is also blocks per task: (8,1,1) is 8 blocks of the
    148 workers, (1,1,1) is one.

    Which way that lands is whole-model throughput, so enumerate rather than
    pick -- the same rule MATMUL_K_TILES follows. Widest split first, so
    cands[0] stays the one-row-per-block default.
    """
    rows = group.output.dims[0]
    return [(rows // r, 1, 1) for r in (1, 2, 4, 8, 16) if rows % r == 0]


def grid_candidates(graph: ModelGraph, group: Group) -> list[tuple]:
    """Every grid this group could legally take, cands[0] the default.

    Two kinds of group have a choice. A group that REDUCES over its last dim
    splits rows -- see rows_grid_candidates. A matmul splits the output's last
    dim, where the N tile becomes the MMA's M under swapAB, so the candidates
    are exactly MATMUL_N_TILES filtered to those that divide N. Everything
    else has one legal grid: the tile is over the output and 64 is simply the
    elementwise width.

    Ranking is NOT done here. A wider tile buys a bigger MMA and costs
    parallelism -- N=3072 gives 48 blocks at 64 and 24 at 128 -- and which way
    that lands is whole-model throughput, the same rule rank_partitions.py
    applies to partitions.
    """
    if has_rms_norm(graph, group):
        return rows_grid_candidates(group)
    if not _has_matmul(graph, group):
        return [default_grid(group)]
    n = group.output.dims[-1]
    return [(n // t, 1, 1) for t in MATMUL_N_TILES if n % t == 0]


# The K tile must be a whole number of MMA K-atoms, and no TMA box side may
# exceed MAX_TMA_BOX. Within that, these are the tiles worth trying: 64 is what
# every hand-written generated_* layer uses, and wider trades smem per stage
# against loop iterations.
MATMUL_K_TILES = (64, 128, 256)


def forloop_candidates(graph, group) -> list[int]:
    """Every forloop_range this group could legally take, fewest steps last.

    Mirage's search DOES explore franges on its own; the problem is that it
    picks any value that verifies, and verifying says nothing about speed --
    for gate/up (K=1024) it chose 64, a K tile of 16, the minimum atom. So
    enumerate the sensible tiles here and let measurement choose, rather than
    replacing search's bad policy with a fixed one.

    Only a matmul group has a choice: elsewhere the forloop runs over the
    output and 1 is right.
    """
    if not _has_matmul(graph, group):
        return [1]
    k = group.external_inputs[0].dims[-1]
    return [k // t for t in MATMUL_K_TILES
            if t % MMA_K_ATOM == 0 and k % t == 0 and k // t >= 1]


def default_forloop(graph, group, k_tile: int = 64) -> int:
    """How many K steps a matmul group should take -- i.e. a K tile of 64.

    Left free, search picks a forloop_range that verifies, and verifying says
    nothing about speed: for the gate/up projections (K=1024) it chose 64,
    a K tile of 16 -- the minimum bf16 MMA K-atom -- and those tasks measured
    24.5 us/call against 3.9-7.1 us for the hand-written ones. Every
    generated_* layer uses K // 64, so ask for that.

    Only for a group that CONTAINS a matmul -- wherever in the group it sits.
    Elsewhere the forloop is over the output, and 1 is right.
    """
    if not _has_matmul(graph, group):
        return 1
    k = group.external_inputs[0].dims[-1]
    return max(1, k // k_tile)


def _check_matmul_tiles(ops):
    """Reject matmul tiles a 1-SM tcgen05 MMA cannot issue.

    Mirrors the guards in transpiler_tb_blackwell.cc.

    M: the tile must be 64 or 128. Anything else is computed as C^T = B^T * A^T
    (swapAB), which puts that dimension into N -- legal at any multiple of 8 up
    to 256 -- and brings the other operand's extent into M, where the 64/128
    rule then applies to *it*. This is why a batched matmul needs its N split
    across the grid: with the whole N in one threadblock, swapAB makes mma_m =
    N, and an N of 256 is not 64 or 128.

    K: the tile must be a whole number of MMA K-atoms. A K-splitting
    forloop_range reaches a sub-atom K easily (K=128 at forloop_range=64
    leaves 2), and CUTLASS then divides by zero building the MMA tiler.

    Search proposes tilings freely and knows none of this, so checking here
    lets the next candidate be tried rather than failing at registration.
    """
    for o in ops:
        if o["op_type"] != "tb_matmul_op":
            continue
        a, b = o["input_tensors"][0]["dim"], o["input_tensors"][1]["dim"]
        m, k, n = a[-2], a[-1], b[-1]
        swap_ab = m not in (64, 128)
        mma_m, mma_n = (n, m) if swap_ab else (m, n)
        if mma_m not in (64, 128) or mma_n == 0 or mma_n % 8 or mma_n > 256:
            raise TaskSearchError(
                f"matmul tile m={m} n={n} -> mma_m={mma_m} mma_n={mma_n} is "
                f"not a legal 1-SM tcgen05 shape (transpiler error 3)")
        if k % MMA_K_ATOM:
            raise TaskSearchError(
                f"matmul K tile {k} is not a multiple of the {MMA_K_ATOM}-"
                f"element MMA K-atom (transpiler error 3)")


def _check_accum_operands(ops):
    """A forloop accumulator must accumulate a COMPUTED value, not a raw input.

    The accumulator is what turns a per-iteration value into a real
    shared-memory tensor, which is why every hand-written generated_* layer
    accumulates a matmul (or an elementwise chain) and never an input tile
    straight off tb_input_op.

    Search proposes the degenerate form anyway: on silu_mul it drew
    [input, input, accum, accum, silu, mul, output], accumulating the two
    INPUTS. At forloop_range == 1 that is an identity, so equivalence checking
    has no reason to reject it -- and the resulting task registers, compiles,
    and HANGS the megakernel (measured: two hours, no token).

    This is the same failure family _check_matmul_operands documents for a
    matmul operand. Note what it deliberately does NOT reject: accumulating a
    matmul and applying activations AFTER, feeding the output directly, is the
    legitimate fused-MLP shape (gate+up+SwiGLU in one task).
    """
    from_input = {
        t["guid"]
        for o in ops if o["op_type"] == "tb_input_op"
        for t in o.get("output_tensors", [])
    }
    for o in ops:
        if not o["op_type"].startswith("tb_forloop_accum"):
            continue
        for t in o.get("input_tensors", []):
            if t["guid"] in from_input:
                raise TaskSearchError(
                    f"{o['op_type']} accumulates a task input directly rather "
                    f"than a computed value; such a task hangs the megakernel")


def _check_matmul_consumers(ops):
    """Reject schedules the Blackwell task-body emitter cannot express.

    A matmul's result lives in TMEM and is only written out by
    write_tC_to_sC, which applies `exp` and nothing else. So anything else
    consuming a matmul directly gets fused into that epilogue and would
    silently vanish -- the transpiler refuses it (error 6), and a chained
    matmul likewise (error 4). The accumulator is what makes the result a
    real shared-memory tensor, which is why every hand-written generated_*
    layer reads `forloop_accum(matmul(...))` and applies activations after.

    Search does not know this, and will happily propose fusing silu into the
    matmul. Encoding it here lets the next candidate be tried instead of
    failing at registration.
    """
    matmul_out = {
        t["guid"]
        for o in ops if o["op_type"] == "tb_matmul_op"
        for t in o["output_tensors"]
    }
    if not matmul_out:
        return
    # exp is what write_tC_to_sC applies. add is here because the attention
    # core is exp(matmul(Q,K^T) + mask) -- the mask add consumes a matmul
    # result directly and the Blackwell backend supports that chain
    # explicitly (it is the whole compiled-attention path). Anything outside
    # this set is refused rather than silently dropped in the epilogue; if
    # the backend turns out to accept more, the model build is the authority
    # and a wrongly-rejected candidate only costs coverage, not correctness.
    allowed = ("tb_forloop_accum", "tb_exp_op", "tb_add_op")
    for o in ops:
        if o["op_type"] in ("tb_matmul_op", "tb_input_op"):
            continue
        if not any(t["guid"] in matmul_out for t in o.get("input_tensors", [])):
            continue
        if not o["op_type"].startswith(allowed):
            raise TaskSearchError(
                f"{o['op_type']} consumes a matmul result directly; the "
                f"Blackwell epilogue only supports exp (transpiler error 6)")


# Kernel-graph op types that carry no computation: everything else in a
# candidate's kernel graph must live inside the one customized op.
_KN_STRUCTURAL_OPS = ("kn_input_op", "kn_output_op", "kn_customized_op")


def _check_matmul_operands(ops):
    """Reject a matmul whose operand comes out of a forloop accumulator.

    The accumulator materializes its result in the layout the loop writes,
    which is not one the Blackwell matmul can read as an A or B operand -- the
    transpiler returns CUDA_T_LAYOUT_ERROR (2).

    Search proposes this: on exp(Q@K^T + mask) it produced a candidate that
    accumulates all three INPUTS before the matmul. At forloop_range == 1 that
    accumulation is an identity, so the schedule buys nothing and only costs a
    copy in a layout that then fails. The sibling candidates that accumulate
    AFTER the matmul, feeding an elementwise op, register and are correct.
    """
    accum_out = {
        t["guid"]
        for o in ops if o["op_type"].startswith("tb_forloop_accum")
        for t in o["output_tensors"]
    }
    if not accum_out:
        return
    for o in ops:
        if o["op_type"] != "tb_matmul_op":
            continue
        if any(t["guid"] in accum_out for t in o["input_tensors"]):
            raise TaskSearchError(
                "a matmul operand is a forloop accumulator's output; that "
                "layout is not a legal MMA operand (transpiler error 2)")


# cp.async.bulk.tensor encodes each box dimension in 8 bits, so no TMA tile
# side may exceed 256 elements.
MAX_TMA_BOX = 256


def _check_tma_box(ops):
    """Reject a TMA-loaded operand whose tile is wider than a TMA box.

    A forloop-split operand consumed by a matmul is loaded by TMA, and the
    descriptor is built on the HOST: an oversized box trips
    `smem_box_shape[1] <= (1 << 8)` inside cute::make_tma_copy_desc and ABORTS
    the process after the megakernel has already compiled. Measured on Qwen3's
    o projection, (8,2048)@(2048,1024) at forloop_range 4: K tile 512.

    Search has no model of this, and it is not a returnable error code, so it
    has to be caught here.
    """
    fdim = {t["guid"]: o["forloop_dim"]
            for o in ops if o["op_type"] == "tb_input_op"
            for t in o["output_tensors"]}
    for o in ops:
        if o["op_type"] != "tb_matmul_op":
            continue
        for operand, role in zip(o["input_tensors"], ("A", "B")):
            if fdim.get(operand["guid"], -1) == -1:
                continue                      # not forloop-split, so not TMA
            for extent in operand["dim"][-2:]:
                if extent > MAX_TMA_BOX:
                    raise TaskSearchError(
                        f"matmul operand {role} tile {operand['dim'][-2:]} "
                        f"exceeds the {MAX_TMA_BOX}-element TMA box limit")


def _check_batched_matmul_forloop(ops):
    """Reject a BATCHED matmul whose operand is forloop-split on K.

    A batched matmul works at forloop_range == 1 (verified: 8x8x128 @
    8x128x256 matches torch.bmm to rel 2.5e-3). Splitting K makes the operand
    TMA-pipelined -- is_pipelined_input is is_chunked_input && forloop_dim !=
    -1 && a matmul consumer -- and the Blackwell TMA path cannot build a
    descriptor for a K-split operand that also carries batch dims. Two
    distinct failures were measured, neither of them a returnable error code:

      grid=(8,4,1) fl=4: a device-side static_assert in cute::tma_partition
                         (size<0>(stensor) != size<0>(gtensor)), i.e. an nvcc
                         template cascade;
      grid=(8,1,1) fl=2: a HOST-side assert, "Majorness of smem doesn't match
                         majorness of gmem", which aborts the process.

    The second is why this is a hard reject rather than a documented caveat.

    Scoped to the A operand, and only when A is loaded from global memory.
    Both measured failures K-split A: a plain batched matmul has to split BOTH
    operands on K together for the accumulation to mean anything, so A and B
    move as a pair and A is the one to test.

    Attention is the case this must NOT reject, and it is exactly the case
    where A is not a gmem load. generated_attention_layer forloop-splits K^T
    and V on the sequence dim -- V's split IS its K dim -- but the second
    matmul's A operand is exp(...), recomputed inside the loop rather than
    loaded, so nothing streams a K-split A through TMA. That schedule is
    verified correct, and an earlier, broader version of this check rejected
    it.
    """
    from_input = {t["guid"]: o["forloop_dim"]
                  for o in ops if o["op_type"] == "tb_input_op"
                  for t in o["output_tensors"]}
    for o in ops:
        if o["op_type"] != "tb_matmul_op":
            continue
        a = o["input_tensors"][0]
        nd = len(a["dim"])
        if nd <= 2:            # not batched; K-splitting is supported
            continue
        # A is (..., m, k), so its K is the last dim.
        if from_input.get(a["guid"], -1) == nd - 1:
            raise TaskSearchError(
                f"batched matmul operand A (dims {a['dim']}) is a gmem load "
                f"forloop-split on its K dim; the Blackwell TMA path cannot "
                f"build that descriptor")


# Kernel-level ops that cannot be synthesized from the others, mapped to the
# threadblock op a correct task body must contain. Binary arithmetic is left
# out on purpose: search legitimately reassociates and refactors it, so
# requiring it would reject correct rewrites.
_IRREDUCIBLE_OPS = {
    "kn_matmul_op": "tb_matmul_op",
    "kn_exp_op": "tb_exp_op",
    "kn_silu_op": "tb_silu_op",
    "kn_gelu_op": "tb_gelu_op",
    "kn_relu_op": "tb_relu_op",
    "kn_sqrt_op": "tb_sqrt_op",
    "kn_square_op": "tb_square_op",
    "kn_rms_norm_op": "tb_rms_norm_op",
}


def _required_ops(spec_graph):
    """Which threadblock ops any correct body for this spec must contain."""
    return {
        _IRREDUCIBLE_OPS[op["op_type"]]
        for op in spec_graph.cygraph.get_graph_structure()
        if op["op_type"] in _IRREDUCIBLE_OPS
    }


def _check_computes_the_spec(ops, required):
    """Reject a candidate that silently drops part of the spec.

    search() verifies equivalence with PROBABILISTIC fingerprints, and that
    check false-accepts. Measured on exp(Q@K^T + mask): roughly one draw in
    six returns a candidate whose body is matmul + add + accumulate with the
    exp GONE -- not moved to kernel level, where the leaked-op check would
    catch it, simply absent. Registering it yields a task that computes the
    wrong thing with nothing to report it.

    An exp cannot be built out of matmul and add, so requiring the spec's
    irreducible ops to survive into the body costs nothing correct and closes
    that hole. Presence, not count: search may legitimately merge two matmuls
    into one, and rejecting that would cost real coverage.
    """
    present = {o["op_type"] for o in ops}
    missing = sorted(required - present)
    if missing:
        raise TaskSearchError(
            f"candidate does not compute the spec: {missing} missing from the "
            f"task body (search's probabilistic verifier accepted it anyway)")


def _as_schedule(spec, cand, required=frozenset()):
    """Validate one candidate against MPK's task conventions."""
    structure = cand.cygraph.get_graph_structure()
    cops = [op for op in structure if op["op_type"] == "kn_customized_op"]
    if len(cops) != 1:
        raise TaskSearchError(
            f"a task is one fused op; candidate has {len(cops)}")

    # ...and ALL of the task must be inside it. search() is free to leave part
    # of the computation as a plain kernel-level op beside the customized one
    # -- for exp(matmul(q,k) + mask) it routinely emits a bgraph computing only
    # matmul+add with a separate kn_exp_op after it. That kernel graph is
    # equivalent to the spec, but MPK registers ONLY the customized op as the
    # task body, so the leftover op would be silently dropped and the task
    # would compute the wrong thing with nothing to report it.
    leaked = [op["op_type"] for op in structure
              if op["op_type"] not in _KN_STRUCTURAL_OPS]
    if leaked:
        raise TaskSearchError(
            f"{len(leaked)} op(s) sit outside the fused task and would be "
            f"dropped on registration: {leaked}")

    bgraph = cops[0]["bgraph"]
    live = cand.cygraph.get_customized_op_bgraphs()
    if len(live) != 1:
        raise TaskSearchError("could not resolve the candidate's bgraph")
    block_dim = live[0].block_dim
    bd = (block_dim["x"], block_dim["y"], block_dim["z"])
    if bd != MPK_BLOCK_DIM:
        raise TaskSearchError(f"block_dim {bd} != {MPK_BLOCK_DIM}")

    ops = bgraph["operators"]
    n_in = sum(1 for o in ops if o["op_type"] == "tb_input_op")
    n_out = sum(1 for o in ops if o["op_type"] == "tb_output_op")
    if n_in != len(spec.inputs):
        raise TaskSearchError(
            f"expected {len(spec.inputs)} inputs, candidate declares {n_in}")
    if n_out != 1:
        raise TaskSearchError(f"expected 1 output, candidate has {n_out}")
    # create_customized_op segfaults if the graph output is not accumulated,
    # even at forloop_range == 1.
    if not any(o["op_type"].startswith("tb_forloop_accum") for o in ops):
        raise TaskSearchError("no forloop_accum on the output")
    _check_accum_operands(ops)

    _check_computes_the_spec(ops, required)
    _check_matmul_consumers(ops)
    _check_matmul_operands(ops)
    _check_matmul_tiles(ops)
    _check_batched_matmul_forloop(ops)
    _check_tma_box(ops)

    gd = bgraph["grid_dim"]
    return Schedule(
        name=spec.name,
        grid_dim=(gd["x"], gd["y"], gd["z"]),
        block_dim=bd,
        forloop_range=bgraph["forloop_range"],
        reduction_dimx=64,
        ops=ops,
        num_inputs=len(spec.inputs),
    )


# tb op_type -> how to replay it on a TBGraph. Unary and binary are separated
# because replay has to know how many operand STensors to look up.
_UNARY = {
    "tb_exp_op": "exp", "tb_silu_op": "silu", "tb_gelu_op": "gelu",
    "tb_relu_op": "relu", "tb_square_op": "square", "tb_sqrt_op": "sqrt",
}
_BINARY = {
    "tb_add_op": "add", "tb_mul_op": "mul", "tb_div_op": "div",
    "tb_sub_op": "sub",
}
_ACCUM = {
    "tb_forloop_accum_no_red_op": None,
    "tb_forloop_accum_red_ld_sum_op": "sum",
    "tb_forloop_accum_red_ld_mean_op": "mean",
    "tb_forloop_accum_red_ld_rms_op": "rms",
    "tb_forloop_accum_redtox_ld_sum_op": "sum_todimx",
}


def _imap(entry):
    return (entry["x"], entry["y"], entry["z"])


def register_searched_task(pk, sched, inputs, output, pipeline_stages=None,
                           pad_mma_n=None):
    """Replay a discovered schedule as an MPK task and register it.

    The discovered bgraph belongs to search's own KNGraph, and it uses a real
    TB_OUTPUT_OP for its result. MPK reads a task's I/O only from TB_INPUT_OPs
    (annotated_graph.cc's split_bgraph_ops), positionally -- reads first, then
    writes -- so the output has to be declared as a trailing new_input as
    well. Rather than mutate search's graph, replay its operators into a fresh
    TBGraph on `pk`'s kernel graph, inserting that declaration.
    """
    out_ops = [o for o in sched.ops if o["op_type"] == "tb_output_op"]
    if len(out_ops) != 1:
        raise TaskSearchError("schedule must have exactly one TB_OUTPUT_OP")
    output_map = _imap(out_ops[0]["output_map"])

    tb_graph = TBGraph(CyTBGraph(sched.grid_dim, sched.block_dim,
                                  sched.forloop_range, sched.reduction_dimx))
    stensors = {}          # discovered stensor guid -> replayed STensor
    next_input = 0

    for op in sched.ops:
        kind = op["op_type"]

        if kind == "tb_input_op":
            if next_input >= len(inputs):
                raise TaskSearchError(
                    f"schedule reads {next_input + 1} inputs, {len(inputs)} given")
            st = tb_graph.new_input(inputs[next_input], _imap(op["input_map"]),
                                    op["forloop_dim"], True)
            stensors[op["output_tensors"][0]["guid"]] = st
            next_input += 1
            if next_input == len(inputs):
                # Declare the write immediately after the last read, the order
                # every hand-written generated_* layer uses. split_bgraph_ops
                # only looks at TB_INPUT_OPs and splits them positionally, so
                # this must be the last one.
                tb_graph.new_input(output, output_map, -1, True)
            continue

        if kind == "tb_output_op":
            continue

        src = [stensors[t["guid"]] for t in op["input_tensors"]]
        if kind in _UNARY:
            res = getattr(tb_graph, _UNARY[kind])(src[0])
        elif kind in _BINARY:
            res = getattr(tb_graph, _BINARY[kind])(src[0], src[1])
        elif kind == "tb_matmul_op":
            res = tb_graph.matmul(src[0], src[1])
        elif kind in _ACCUM:
            res = tb_graph.forloop_accum(src[0], _ACCUM[kind])
        elif kind == "tb_rms_norm_op":
            res = tb_graph.rms_norm(src[0])
        else:
            raise TaskSearchError(f"cannot replay threadblock op {kind!r}")
        stensors[op["output_tensors"][0]["guid"]] = res

    # The store itself, reading whatever the TB_OUTPUT_OP consumed.
    result = stensors[out_ops[0]["input_tensors"][0]["guid"]]
    tb_graph.new_output(result, output_map, -1)

    pk.kn_graph.customized(list(inputs) + [output], tb_graph)
    # params: [pipeline_stages] or [pipeline_stages, pad_mma_n]. Positional,
    # so the stage count has to be present to pass the second.
    params = None
    if pipeline_stages or pad_mma_n is not None:
        params = [pipeline_stages or 2]
        if pad_mma_n is not None:
            params.append(1 if pad_mma_n else 0)
    pk.kn_graph.register_task(tb_graph, "generated", params)
    return tb_graph
