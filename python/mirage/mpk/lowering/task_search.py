"""Let the superoptimizer choose an MPK task's schedule."""
from __future__ import annotations

import json
import logging
import os

import mirage as mi
from mirage.core import CyTBGraph
from mirage.kernel import KNGraph, search
from mirage.threadblock import TBGraph

MPK_BLOCK_DIM = (256, 1, 1)

log = logging.getLogger(__name__)

SCHEDULE_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "searched_schedules.json")

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
        self.strides = tuple(strides) if strides is not None else _row_major(dims)


def _row_major(dims):
    strides, acc = [], 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


class TaskSpec:
    """WHAT one task computes, independent of how."""

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


def search_task_schedules(spec, grid_dim=None, forloop_range=None,
                           verbose=False, max_ops=None, wide_inputs=False):
    """Every distinct MPK-usable implementation of `spec` that search found."""
    if len(spec.inputs) > 3 and not wide_inputs:
        raise TaskSearchError(
            f"{spec.name!r} has {len(spec.inputs)} inputs; search fuses at "
            f"most 3 by default and would return nothing at all. Pass "
            f"wide_inputs=True to lift the caps.")
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
        max_tb_graph_inputs=(len(spec.inputs) if wide_inputs else -1),
        max_kn_graph_ops=(len(spec.inputs) + 8 if wide_inputs else -1),
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
    """The first MPK-usable schedule."""
    return search_task_schedules(spec, **kw)[0]


MMA_K_ATOM = 16


MATMUL_N_TILES = (64, 128)


def default_grid(group: Group, n_tile: int = 64) -> tuple[int, int, int]:
    """One `n_tile`-wide column of the output per threadblock."""
    n = group.output.dims[-1]
    if n % n_tile:
        raise ValueError(
            f"group {group.tag!r} output last dim {n} is not a multiple of "
            f"{n_tile}; pass an explicit grid for it")
    return (n // n_tile, 1, 1)


def _has_matmul(graph: ModelGraph, group: Group) -> bool:
    """Does this group contain a matmul ANYWHERE, not just as its first node?"""
    return any(graph.nodes[i].op == "matmul" for i in group.nodes)


def has_rms_norm(graph: ModelGraph, group: Group) -> bool:
    return any(graph.nodes[i].op == "rms_norm" for i in group.nodes)


def rows_grid_candidates(group: Group) -> list[tuple]:
    """How many rows one block of a reducing group takes: 1, 2, 4, ... all."""
    rows = group.output.dims[0]
    return [(rows // r, 1, 1) for r in (1, 2, 4, 8, 16) if rows % r == 0]


def batched_grid_candidates(group) -> list[tuple]:
    """A 3-D group splits its BATCH dim, one block per batch element.

    default_grid slices the output's last dim, which for a batched matmul is
    the head dim -- that leaves the whole batch in one threadblock. The
    attention core is (fold, 8, hd) @ ... with fold = kv_heads * requests, and
    one block per fold element is what the hand-written core used.
    """
    batch, n = group.output.dims[0], group.output.dims[-1]
    return [(batch, n // t, 1) for t in MATMUL_N_TILES if n % t == 0] or \
           [(batch, 1, 1)]


def grid_candidates(graph: ModelGraph, group: Group) -> list[tuple]:
    """Every grid this group could legally take, cands[0] the default."""
    if has_rms_norm(graph, group):
        return rows_grid_candidates(group)
    if len(group.output.dims) == 3:
        return batched_grid_candidates(group)
    if not _has_matmul(graph, group):
        return [default_grid(group)]
    n = group.output.dims[-1]
    return [(n // t, 1, 1) for t in MATMUL_N_TILES if n % t == 0]


MATMUL_K_TILES = (64, 128, 256)


def forloop_candidates(graph, group) -> list[int]:
    """Every forloop_range this group could legally take, fewest steps last."""
    if not _has_matmul(graph, group):
        return [1]
    k = group.external_inputs[0].dims[-1]
    return [k // t for t in MATMUL_K_TILES
            if t % MMA_K_ATOM == 0 and k % t == 0 and k // t >= 1]


def knobs_from_env(graph):
    """The per-group schedule knobs, as env-selected candidates.

    Returns (grid_for, forloop_for, stages_for) -- the three callables lower()
    takes. Each knob is enumerated above and ranked by nothing here: which
    value wins is whole-model throughput, so these exist to be swept.

      MPK_MATMUL_N_TILE  64 (default) | 128   the MMA's M under swapAB
      MPK_MATMUL_K_TILE  64 | 128 (default) | 256
      MPK_MATMUL_STAGES  2..8, 4 (default)
      MPK_NORM_ROWS      rows per block for a group that reduces its last dim

    A group the requested value does not divide keeps cands[0], the default
    for its kind, rather than failing the whole lowering.
    """
    import os

    def _one(name, default, allowed=None, lo=None, hi=None):
        v = int(os.environ.get(name, default))
        if allowed is not None and v not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {v}")
        if lo is not None and not lo <= v <= hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}], got {v}")
        return v

    n_tile = _one("MPK_MATMUL_N_TILE", "64", MATMUL_N_TILES)
    k_tile = _one("MPK_MATMUL_K_TILE", "128", MATMUL_K_TILES)
    stages = _one("MPK_MATMUL_STAGES", "4", lo=2, hi=8)
    norm_rows = _one("MPK_NORM_ROWS", "1", lo=1, hi=1024)

    def grid_for(group):
        cands = grid_candidates(graph, group)
        if len(group.output.dims) == 3:
            return cands[0]                      # batched: split the batch dim
        if has_rms_norm(graph, group):
            want = (max(1, group.output.dims[0] // norm_rows), 1, 1)
        else:
            want = (group.output.dims[-1] // n_tile, 1, 1)
        return want if want in cands else cands[0]

    def forloop_for(group):
        want = default_forloop(graph, group, k_tile)
        return (want if want in forloop_candidates(graph, group)
                else default_forloop(graph, group))

    return grid_for, forloop_for, lambda group: stages


def default_forloop(graph, group, k_tile: int = 64) -> int:
    """How many K steps a matmul group should take -- i.e. a K tile of 64."""
    if not _has_matmul(graph, group):
        return 1
    k = group.external_inputs[0].dims[-1]
    return max(1, k // k_tile)


def _check_matmul_tiles(ops):
    """Reject matmul tiles a 1-SM tcgen05 MMA cannot issue."""
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
    """A forloop accumulator must accumulate a COMPUTED value, not a raw input."""
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
    """Reject schedules the Blackwell task-body emitter cannot express."""
    matmul_out = {
        t["guid"]
        for o in ops if o["op_type"] == "tb_matmul_op"
        for t in o["output_tensors"]
    }
    if not matmul_out:
        return
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


_KN_STRUCTURAL_OPS = ("kn_input_op", "kn_output_op", "kn_customized_op")


def _check_matmul_operands(ops):
    """Reject a matmul whose operand comes out of a forloop accumulator."""
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


MAX_TMA_BOX = 256


def _check_tma_box(ops):
    """Reject a TMA-loaded operand whose tile is wider than a TMA box."""
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
    """Reject a BATCHED matmul whose operand is forloop-split on K."""
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
    """Reject a candidate that silently drops part of the spec."""
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
    """Replay a discovered schedule as an MPK task and register it."""
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
    params = None
    if pipeline_stages or pad_mma_n is not None:
        params = [pipeline_stages or 2]
        if pad_mma_n is not None:
            params.append(1 if pad_mma_n else 0)
    pk.kn_graph.register_task(tb_graph, "generated", params)
    return tb_graph
