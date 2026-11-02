"""StaticMegakernel: a PersistentKernel whose graph is built into one generated kernel with a static per-SM task schedule
(compiler.py -> static_schedule.py), instead of MPK's runtime.

Its layers (one graph node each, as MPK's layers) keep what compiler.py and the code generator need about the node: its slices,
its changeable grid axes, and what its task type's kernel template reads and writes, declared here, inside the layer, not by the
caller: its output buffer (in this GPU's memory, or in the cross-GPU exchange region), scratch buffers, graph tensors it reads by
pointer, tensor maps, counters. Each buffer gets a buffer slot (an index into G::buf), each tensor map a map slot (an index into
Maps::m); the generated host code (static_megakernel/host.cuh) allocates or binds them, the task function reads them by slot
(its StaticNode, StaticBufSlots and StaticMapSlots template arguments). The kernel templates are in
include/mirage/static_megakernel/: core.cuh (the layer state, the kernel's start and end), host.cuh, tasks/<name>.cuh (one task
type each: its body and run_<name>). Each layer adds its node with a default grid (no K split); compiler.compile_plan regrids
to the plan. The sizes come from the graph: each layer reads them from its tensors and puts those its kernel template needs in
the node's params; the kernel-wide ones (tokens per step, GPUs) are the StaticMegakernel's (tokens, num_gpus), which the
generated layer defines (config.cuh).
"""
import math
import os
import struct

import torch

from ..core import *
from ..kernel import TBGraph
from .persistent_kernel import PersistentKernel

# the kernel's limits (config.cuh MAX_BUFS, MAX_MAPS, NODE_COUNTER_LINES, SMAX = the ring's stages; core.cuh Combine)
MAX_BUFS, MAX_MAPS, NODE_COUNTER_LINES, RING_STAGES = 24, 32, 16, 5
# HIP: a graph with more buffer slots than MAX_BUFS gets HIP_MAX_BUFS (the build's STATIC_MAX_BUFS, config.cuh): G grows only for
# such graphs (the MoE / KDA builds keep their argument layout)
HIP_MAX_BUFS = int(os.environ.get("MPK_HIP_MAX_BUFS", "64"))
HIP_MAX_MAPS, HIP_COUNTER_LINES = int(os.environ.get("MPK_HIP_MAX_MAPS", "64")), int(os.environ.get("MPK_HIP_COUNTER_LINES", "64"))   # (the same for maps and counter lines: STATIC_MAX_MAPS, STATIC_COUNTER_LINES)


def _hip() -> bool:
    from . import gpu_backend
    return gpu_backend.backend() == "hip"
# the task types Mirage registers by name (src/kernel/graph.cc); any other task type registers as "static:<name>" (build-only)
REGISTERED_TASKS = {"gemm_tile", "topk_route", "sum_quant_send", "situ_and_mul", "moe_experts", "allreduce_send", "sum_rmsnorm",
                    "sum_gpus", "sum_send", "residual_add"}
COMBINE = {"slots": 0, "add": 1, "store": 2, "push": 3, "finish": 4, "finish_quant": 5,   # push, finish*: HIP only
           "push_all": 6,
           "local": 7,      # HIP: (below)
           "push_rs": 8,    # HIP: as push_all (every GPU all rows, a partial), each row block only to the GPU owning its rows
                            # ([GPUS][N / GPUS / R][T][R] bf16 there; attn_residual_layer(rs=True) adds the GPUs')
           "finish_rs": 9}  # HIP: K parts in a scratch, a row block's last part adds them and sends it as push_rs does
           # HIP: a polled input, one K part, bf16 [T][N] into this GPU's 0xFF-prefilled buffer (MLA's q_b)   # HIP: every row's partial (one K part) to every GPU ([GPUS][T][N] bf16; the reader adds the GPUs')
SF_CHUNK = 512   # one MX scale chunk (config.cuh): 128 rows x 4 B
MAX_SUM_PARTS = 14   # the most K parts sum_send adds (tasks/sum_send.cuh)


def zq_chunk(tokens: int) -> int:
    """tasks/sum_quant_send.cuh zq_chunk: one 128-column scale chunk of z_q on HIP, 16 B per token, at least SF_CHUNK."""
    return max(SF_CHUNK, 16 * tokens)


def zq_bytes(tokens: int, width: int) -> int:
    """tasks/sum_quant_send.cuh zq_bytes: z_q [tokens][width] e4m3, then its scale chunks (one per 128 columns)."""
    return tokens * width + width // 128 * zq_chunk(tokens)


def zq32_bytes(tokens: int, width: int) -> int:
    """tasks/group_quant.cuh zq32_bytes: z_q [tokens][width] e4m3, then its fp32 scales [width / 128][tokens]."""
    return tokens * width + (width // 128) * tokens * 4


def float_bits(v: float) -> int:
    """An fp32 value as the int of its bits (a param; the kernel template reads it with __int_as_float)."""
    return struct.unpack("<i", struct.pack("<f", v))[0]


def shuffle_gemm_weight(w: torch.Tensor) -> torch.Tensor:
    """A [N, K] bf16 weight in gemm_tile's shuffled layout (HIP, gemm_weight_layout = "shuffled"), same shape and dtype: tile-major
    [N / 16][K / 32][64 lanes][8 bf16], lane l of row tile r, K step k = row 16 r + l % 16, columns 32 k + 8 (l / 16) .. + 7 (the
    MFMA 16x16x32 A fragment), so a wave's fragment load is 1 KB contiguous."""
    N, K = w.shape
    assert N % 16 == 0 and K % 32 == 0, (N, K)
    t = w.reshape(N // 16, 16, K // 32, 4, 8)          # [tile][row in tile][k step][g][8]
    t = t.permute(0, 2, 3, 1, 4)                         # [tile][k step][g][row in tile][8]: lane = g * 16 + row
    return t.contiguous().reshape(N, K)


class StaticMegakernel(PersistentKernel):

    def __init__(self, num_gpus: int = 1, num_sms: int = None):
        """num_gpus: the GPUs the layer is divided over (a rows-split node's GPUs, an exchange buffer's slots; config.cuh GPUS).
        num_sms: the SMs of one GPU, one task list each (config.cuh NSM); None: the current device's.
        PersistentKernel.__init__ takes MPK's serving configuration (mode, workers, schedulers, batch limits, paging, meta
        tensors); the static path reads none of it (no runtime scheduler: every SM runs its own task list), so it gets MPK's
        defaults, with test_mode allocating the meta tensors it checks."""
        super().__init__(**PersistentKernel.get_default_init_parameters(), test_mode=True)
        self.num_gpus = num_gpus
        self.num_sms = num_sms
        self.tokens = None        # tokens per step: the first dimension of the layers' token-major tensors (_tokens; config.cuh T)
        self._static_nodes = {}   # graph index -> what compiler.py needs about a node (_add_node)
        self._bufs = []           # G::buf slot -> {"kind": out | exchange | scratch | tensor, ...} (_add_node, _scratch, _tensor_slot)
        self._maps = []           # Maps::m slot -> (kind, source, dims) (_map)
        self._counter_lines = 0   # counter lines given out (_add_node)

    def _num_sms(self):
        if self.num_sms is None:
            self.num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        return self.num_sms

    def _tokens(self, n: int) -> int:
        """Tokens per step: the first layer's tensors set it, every other layer's must agree. Returns it."""
        if self.tokens is None:
            self.tokens = n
        if n != self.tokens:
            raise ValueError(f"a layer's tensors have {n} tokens, the graph's earlier layers {self.tokens}")
        return n

    def _dtypes(self, layer: str, **tensors) -> None:
        """tensors: argument name = (tensor, dtype). Raises unless each tensor has the dtype the task type's kernel reads or writes
        it as."""
        for name, (t, want) in tensors.items():
            if t.dtype != want:
                raise ValueError(f"{layer}: {name} is {t.dtype}, its kernel uses {want}")

    def _add_node(self, name: str, grid_dim: tuple, block_dim: tuple, tensors: list, input_maps: list, params: list,
                  changeable_grid_dims: dict = None, rows_split_over_gpus: bool = False, one_task_per_sm: bool = False,
                  concurrent_along_axis: int = None, cluster_pair_params: list = None, out_buffer=None, exchange: bool = False,
                  counters: int = 0, buf_slots: tuple = (), map_slots: tuple = (), dry: bool = False,
                  cost_inputs: tuple = (), grid_choices: list = None):
        """Add one node, as MPK's layers do (customized + register_task), and keep for compiler.py what the graph does not hold:
          input_maps              input_maps[i] = which dimension of tensors[i] each grid axis cuts (MPK's input_map,
                                  TBGraph.new_input: (1, -1, -1) = grid x cuts dim 1); a task's slices follow from the input
                                  maps and the grid, for any grid
          changeable_grid_dims    {grid axis: unit or (unit, min units, max units)}: the grid axes the compiler may change;
                                  one task gets a whole number of units of each dimension that axis cuts (gemm_tile: {1: 128},
                                  K in whole 128-column pieces), optionally between min and max units; the other axes keep
                                  grid_dim
          rows_split_over_gpus    grid axis x is divided among the GPUs: each GPU computes its part, in blocks of grid_dim's x
                                  whatever the grid becomes (compiler.owned_positions)
          one_task_per_sm         grid x = the number of SMs; task x runs on SM x (the plan puts it in SM x's list)
          concurrent_along_axis   tasks that differ only in this grid axis exchange data while they run: they must run at the
                                  same time, on different SMs
          cluster_pair_params     when such a group has 2 tasks, the compiler may put them on the 2 CTAs of one cluster (CTAs 2k,
                                  2k + 1; the kernel then launched in clusters of 2) and give the node these params (the task body
                                  that swaps through shared memory); compiler.compile_plan decides
          out_buffer              the node's output lives in a buffer slot (G::buf; StaticNode::buf): a function node -> (bytes,
                                  reset byte before each launch or -1, re-armed by its reader), evaluated with the node's grid
                                  when the layer is built (host_slot_args); its consumers read it through the slot. Re-armed by
                                  its reader: the one task that reads each value writes 0xFF back after use (REARM), so a
                                  STATIC_RESET_IN_KERNEL host fills it only once
          exchange                the output buffer is in the cross-GPU exchange region (every GPU's copy at the same offset,
                                  two sets; written with multicast stores): the data a task sends to every GPU
          counters                the node's counters (u32; the first is StaticNode::counter; one 128-B line per 32, since
                                  counters in one line contend): its tasks count there, its consumers wait on them
          buf_slots               the buffer slots (G::buf) the task function reads besides its inputs' (StaticBufSlots), in the
                                  order its kernel template reads them: _scratch, _tensor_slot or another node's buffer
          map_slots               the map slots (Maps::m) the task function reads (StaticMapSlots), in that order: _map
          dry                     the task type has a dry pass (run_<name> takes `warm`; static_schedule.case_code)
          cost_inputs             the inputs whose producer's grid changes the task's time (topk_route: the router's K parts, which
                                  it adds); their producers' grids are part of its cost key (search.cost_key), the others not
          grid_choices            the grids the compiler may give a node whose tasks split no tensor by the grid (allreduce_send:
                                  slices of a buffer it reads by pointer), instead of changeable_grid_dims"""
        from .static_schedule import THREADS
        if tuple(block_dim) != (THREADS, 1, 1):   # one kernel for every node: its CTAs have THREADS threads (the warp roles)
            raise ValueError(f"{name}: block_dim {tuple(block_dim)}; every task runs on the kernel's ({THREADS}, 1, 1) threads")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        for t, m in zip(tensors, input_maps):
            tb_graph.new_input(t, m, -1, True)
        self.kn_graph.customized(tensors, tb_graph)
        self.kn_graph.register_task(tb_graph, name if name in REGISTERED_TASKS else "static:" + name, params)
        idx = self.kn_graph.get_num_operators() - 1
        for b in self._bufs:   # the scratch buffers the layer method made for this node (_scratch, before this call)
            if b["kind"] == "scratch" and "node" not in b:
                b["node"] = idx
        buf = -1
        if out_buffer is not None:
            buf = self._new_buf({"kind": "exchange" if exchange else "out", "node": idx, "fill": out_buffer})
        counter = -1
        if counters:
            lines = (counters + 31) // 32
            limit = HIP_COUNTER_LINES if _hip() else NODE_COUNTER_LINES
            assert self._counter_lines + lines <= limit, f"more than {limit} counter lines (config.cuh)"
            counter = 32 * self._counter_lines; self._counter_lines += lines
        self._static_nodes[idx] = dict(
            name=name, input_maps=list(input_maps), changeable_grid_dims=dict(changeable_grid_dims or {}),
            rows_split_over_gpus=rows_split_over_gpus, gpu_split_blocks=grid_dim[0], one_task_per_sm=one_task_per_sm,
            concurrent_along_axis=concurrent_along_axis, cluster_pair_params=cluster_pair_params,
            buf=buf, out_buffer=out_buffer, exchange=exchange, counter=counter, buf_slots=list(buf_slots), map_slots=list(map_slots),
            dry=dry,
            cost_inputs=list(cost_inputs), grid_choices=[tuple(g) for g in grid_choices] if grid_choices else None)

    # ---- buffer slots (G::buf) and map slots (Maps::m) ----
    def _new_buf(self, entry: dict) -> int:
        from . import gpu_backend
        limit = HIP_MAX_BUFS if gpu_backend.backend() == "hip" else MAX_BUFS
        assert len(self._bufs) < limit, f"more than {limit} buffer slots (config.cuh MAX_BUFS)"
        self._bufs.append(entry)
        return len(self._bufs) - 1

    def _scratch(self, name: str, nbytes, reset: int = -1, by_reader: bool = False) -> int:
        """A buffer only the node's tasks use (allocated by the host code; reset before each launch to the byte `reset`, -1: none;
        by_reader: its readers write `reset` back after use, so with STATIC_RESET_IN_KERNEL the host fills it once), made by a
        layer method before its _add_node, which gives it the node. nbytes: an int, or a function of the node as compiled (0: no
        buffer). Returns its buffer slot."""
        size = nbytes if callable(nbytes) else (lambda n: nbytes)
        # (a reset buffer: whole 16-B pieces, as the in-kernel resets take them; e.g. attn_residual's partials at T = 1)
        padded = lambda n: -(-size(n) // 16) * 16 if reset >= 0 else size(n)
        return self._new_buf({"kind": "scratch", "name": name, "fill": lambda n: (padded(n), reset, by_reader)})

    def _tensor_slot(self, tensor: DTensor) -> int:
        """The buffer slot of a graph tensor the task reads or writes by pointer (the same tensor again: the same slot)."""
        for i, b in enumerate(self._bufs):
            if b["kind"] == "tensor" and b["guid"] == tensor.guid:
                return i
        nbytes = 1
        for d in range(tensor.num_dims):
            nbytes *= tensor.dim(d)
        return self._new_buf({"kind": "tensor", "guid": tensor.guid, "fill": lambda n: (nbytes, -1, False)})

    def _map(self, kind: str, source: tuple, dims: tuple) -> int:
        """A tensor map slot (Maps::m; the same map again: the same slot). source: ("t", tensor) a graph tensor or a node's output
        by its tensor; ("b", buffer slot, byte offset); ("x", exchange buffer slot, byte offset, set) in the exchange region's set.
        kind and dims as host.cuh reads them: bf16 (rows, K, box rows), wblk (MXFP4 pieces, box pieces), wblk8 (e4m3 pieces, box
        pieces), sf (chunks, box chunks), act8 (K bytes, rows), act8kt (K bytes, rows, K tiles). Returns the slot."""
        key = (kind, (source[0], source[1].guid) + tuple(source[2:]) if source[0] == "t" else tuple(source), tuple(dims))
        for i, m in enumerate(self._maps):
            if m[3] == key:
                return i
        assert len(self._maps) < (HIP_MAX_MAPS if _hip() else MAX_MAPS), "more tensor maps than config.cuh MAX_MAPS"
        self._maps.append((kind, source, tuple(dims), key))
        return len(self._maps) - 1

    def _gemm_map(self, tensor: DTensor, box_rows: tuple) -> int:
        """The map slots of a bf16 [rows, K] tensor, one per box row count in box_rows (consecutive slots; the same tensor and
        boxes again: the same slots). Returns the first."""
        rows, K = tensor.dim(0), tensor.dim(1)
        for i in range(len(self._maps) - len(box_rows) + 1):
            if all(self._maps[i + j][3] == ("bf16", ("t", tensor.guid), (rows, K, b)) for j, b in enumerate(box_rows)):
                return i
        assert len(self._maps) + len(box_rows) <= (HIP_MAX_MAPS if _hip() else MAX_MAPS), "more tensor maps than config.cuh MAX_MAPS"
        first = len(self._maps)
        for b in box_rows:
            self._maps.append(("bf16", ("t", tensor), (rows, K, b), ("bf16", ("t", tensor.guid), (rows, K, b))))
        return first

    def buf_slots(self) -> int:
        """G::buf's size for this graph (config.cuh MAX_BUFS; HIP: STATIC_MAX_BUFS: 32 or 64 when MAX_BUFS is too few, e.g. a
        fused attention + MoE layer)"""
        n = len(self._bufs)
        return MAX_BUFS if n <= MAX_BUFS else (32 if n <= 32 else HIP_MAX_BUFS)

    def map_slots(self) -> int:
        """Maps::m's size (config.cuh MAX_MAPS; HIP: STATIC_MAX_MAPS)"""
        return MAX_MAPS if len(self._maps) <= MAX_MAPS else HIP_MAX_MAPS

    def counter_lines(self) -> int:
        """the counter lines (config.cuh NODE_COUNTER_LINES; HIP: STATIC_COUNTER_LINES)"""
        return NODE_COUNTER_LINES if self._counter_lines <= NODE_COUNTER_LINES else HIP_COUNTER_LINES

    def exchange_layout(self):
        """The exchange region's layout, from the exchange buffers: the ones below 256 KB (smallest first, then by slot), the start
        barrier's slots, the larger ones (the same order); each 4096-aligned. Returns (one set's bytes, the start barrier's offset in
        a set, per buffer slot its offset, per buffer slot its bytes). Keep this order: the buffers' offsets change the layer's time
        (laid out in slot order instead, the same kernel is slower)."""
        from .compiler import graph_nodes
        nodes = {n.graph_idx: n for n in graph_nodes(self)}
        offsets, sizes, end = [0] * self.buf_slots(), [0] * self.buf_slots(), 0
        align = lambda v: (v + 4095) // 4096 * 4096
        for i, b in enumerate(self._bufs):
            if b["kind"] == "exchange":
                sizes[i] = b["fill"](nodes[b["node"]])[0]
        order = sorted((i for i in range(len(self._bufs)) if sizes[i]), key=lambda i: (sizes[i], i))
        hello = None
        for i in order:
            if hello is None and sizes[i] >= 256 * 1024:
                hello, end = end, align(end + self.num_gpus * 16)
            offsets[i] = end
            end = align(end + sizes[i])
        if hello is None:
            hello, end = end, align(end + self.num_gpus * 16)
        return end, hello, offsets, sizes

    def host_slot_args(self) -> dict:
        """The host code's slot args (host.cuh): "buf.<slot>" -> "<kind> <name> <bytes> <reset> <by reader>" (the node outputs with
        their nodes' grids now), "map.<slot>" -> "<kind> <source> <dims...>"."""
        from .compiler import graph_nodes
        nodes = {n.graph_idx: n for n in graph_nodes(self)}
        names = self._tensor_names   # guid -> the graph tensor's name
        args = {}
        for slot, b in enumerate(self._bufs):
            n = nodes.get(b.get("node"))
            nbytes, reset, by_reader = b["fill"](n)
            name = {"out": lambda: names[n.outputs[0].guid], "exchange": lambda: names[n.outputs[0].guid],
                     "scratch": lambda: b["name"], "tensor": lambda: names[b["guid"]]}[b["kind"]]()
            args[f"buf.{slot}"] = f"{b['kind']} {name} {nbytes} {reset} {int(by_reader)}"
        # the host binds a node's output buffer to the graph tensor of the same name when it is given one (a layer output the
        # caller reads, e.g. "hidden"): two different tensors with one name would share memory
        by_name = {}
        for slot, b in enumerate(self._bufs):
            if b["kind"] in ("out", "tensor"):
                key = args[f"buf.{slot}"].split()[1]
                guid = nodes[b["node"]].outputs[0].guid if b["kind"] == "out" else b["guid"]
                if by_name.setdefault(key, guid) != guid:
                    raise ValueError(f"host_slot_args: two graph tensors named {key!r} (a node output and another tensor): "
                                     f"the host would bind them to one buffer; rename one")
        set_bytes = self.exchange_layout()[0]
        for slot, (kind, source, dims, _) in enumerate(self._maps):
            if source[0] == "t":
                src = f"t:{names[source[1].guid]}"
            elif source[0] == "b":
                src = f"b:{source[1]}+{source[2]}"
            else:   # an exchange buffer: its set 0 copy is buffer slot source[1]; set k at + k sets
                src = f"b:{source[1]}+{source[2] + source[3] * set_bytes}"
            args[f"map.{slot}"] = " ".join([kind, src] + [str(d) for d in dims])
        return args

    def gemm_tile_layer(
        self,
        input: DTensor,     # [T, K] bf16
        weight: DTensor,    # [N, K] bf16
        output: DTensor,    # [T, N] fp32 ("add": int64 fixed point; its buffer holds what `combine` says)
        combine: str = "slots",   # how the K parts are combined (gemm_tile.cuh Combine): "slots" (one 0xFF-prefilled slot per
                                  # K part, the reader adds them), "add" (fixed-point atomic add, a counter per 128-row block),
                                  # "store" (plain stores, one K part, a counter of tasks)
        rows_split_over_gpus: bool = False,   # each GPU computes its part of the weight's rows (latent_down, latent_up)
        input_polled: bool = False,   # the input is a node's 0xFF-prefilled bf16 buffer, polled per 16 B straight into the ring's
                                      # stages (each task starts on its columns as soon as they are written, without a counter)
        block_dim: tuple = (256, 1, 1),
        norm_gamma: DTensor = None,   # HIP, "finish": RMSNorm(input) x gamma folded in (input: an allreduce_send's rows; each
        norm_eps: float = 1e-5,       # task adds the GPUs' rows x gamma, the finisher scales by rstd; gemm_tile.cuh gemm_norm_fill)
        hybrid: bool = False,         # HIP, a waited input, 128-row tasks: each wave's first row tile's weights copied into LDS, its
                                      # second's held in registers, all issued before the wait (gemm_tile.cuh gemm_wave HYB);
                                      # params[7] = 1
    ):
        """Task (x, y) = weight rows [R x, R x + R) times K part y. Default grid: R = 128 rows, K in 1 part (polled: the fewest K
        parts with at most RING_STAGES K tiles per task); the compiler may change grid x to R = 64 rows (the kernel template has a
        64-row tile, MMA M = 64) and grid y (K parts), in whole 128-column pieces, at least 2 per task ("store": one K part, 128 rows,
        no options; polled: 128 rows, at most RING_STAGES pieces per task, as the whole K part is written into the stages up front).
        The weight and the activation reach the task through tensor maps (map slots in Maps::m: the weight's 128- and 64-row boxes, the
        activation's; polled: the weight's 128-row box only), the output through the node's buffer slot. params = [mode, K, N,
        polled (0 / 1)]; map_slots = [weight map slot, activation map slot (-1: polled)]."""
        assert input.num_dims == 2 and weight.num_dims == 2 and weight.dim(1) == input.dim(1)
        assert output.num_dims == 2 and output.dim(1) == weight.dim(0) and weight.dim(0) % 128 == 0 and weight.dim(1) % 128 == 0
        mode, N, K, T, GPUS = COMBINE[combine], weight.dim(0), weight.dim(1), self._tokens(input.dim(0)), self.num_gpus
        self._dtypes("gemm_tile_layer", input=(input, bfloat16), weight=(weight, bfloat16),
                     output=(output, int64 if combine == "add" else bfloat16 if combine in ("push", "push_all", "local", "finish", "push_rs", "finish_rs")
                             else uint8 if combine == "finish_quant" else float32))
        if input_polled:
            from .compiler import graph_nodes, producer_of
            p = producer_of(input, graph_nodes(self))
            if p is None or p.info["buf"] < 0 or p.info["out_buffer"](p)[1] != 0xFF:
                raise ValueError(f"gemm_tile_layer: a polled input must be a node's 0xFF-prefilled output buffer; it comes from "
                                 f"{p.name if p else 'the graph inputs'}")
            assert combine in ("slots", "push", "push_all", "local", "finish", "push_rs"), "polled: the K parts in slots (or pushed, HIP)"
        exchange, buf_slots = False, ()
        if combine == "finish_quant":   # K parts in a scratch; per 128-row block the last part makes MXFP8 z_q and pushes it
            # (sum_quant_send's work, gemm_tile.cuh COMBINE_FINISH_Q); the output: z_q then its scale chunks, every GPU's
            assert not input_polled and rows_split_over_gpus and N % 128 == 0, "finish_quant: rows split over GPUs, 128-row tiles"
            out_buffer = lambda n: (zq_bytes(T, N), 0xFF, True)
            exchange = True
            buf_slots = (self._scratch("gemm_tile.finish_parts_q", lambda n: n.grid[1] * T * N * 4),)
        elif combine == "finish":   # K parts in a scratch; per row block the last part to count adds them, pushes bf16 like push
            assert input_polled and rows_split_over_gpus and N % (128 * GPUS) == 0, "finish: polled input, rows split over GPUs"
            out_buffer = lambda n: (GPUS * T * (N // GPUS) * 2, 0xFF, True)
            exchange = True
            # the K parts [P][T][N / GPUS] fp32, then (norm) the parts' sums of squares [N / GPUS / 16][P][T] x 16 B
            # (MPK_FINISH_FLAG=1, with the kernel flag STATIC_GEMM_FINISH_FLAG: 0xFF before each launch, the parts their own flags)
            buf_slots = (self._scratch("gemm_tile.finish_parts", lambda n: n.grid[1] * T * (N // GPUS) * 4
                                       + (n.grid[1] * T * 16 * (N // GPUS // 16) if norm_gamma is not None else 0),
                                       reset=0xFF if os.environ.get("MPK_FINISH_FLAG") == "1" else -1),)
            if norm_gamma is not None:
                assert norm_gamma.num_dims == 1 and norm_gamma.dim(0) == K, "norm: gamma [K]"
                self._dtypes("gemm_tile_layer", norm_gamma=(norm_gamma, bfloat16))
                buf_slots = buf_slots + (self._tensor_slot(norm_gamma),)
        elif norm_gamma is not None:
            raise ValueError("gemm_tile_layer: norm_gamma needs combine='finish'")
        elif combine == "push":   # one K part, bf16 [GPUS][T][N / GPUS] in the exchange region: this GPU's rows to every GPU
            assert input_polled and rows_split_over_gpus and N % (128 * GPUS) == 0, "push: polled input, rows split over GPUs"
            out_buffer = lambda n: (GPUS * T * (N // GPUS) * 2, 0xFF, True)
            exchange = True
        elif combine == "push_all":   # one K part, bf16 [GPUS][T][N] in the exchange region: this GPU's partial of every row
            # to every GPU (KDA's o_proj: each GPU's heads give a partial sum of all N rows; attn_residual adds the GPUs')
            assert not rows_split_over_gpus, "push_all: every GPU all rows (input polled, or counted: T > 16 decode)"
            out_buffer = lambda n: (GPUS * T * N * 2, 0xFF, True)
            exchange = True
        elif combine == "push_rs":   # one K part, every GPU all rows; row block r to the GPU owning it: [GPUS][T][N / GPUS] bf16
            assert not rows_split_over_gpus and N % (128 * GPUS) == 0, "push_rs: every GPU all rows, whole 128-row tiles per GPU"
            out_buffer = lambda n: (GPUS * T * (N // GPUS) * 2, 0xFF, True)
            exchange = True
        elif combine == "finish_rs":   # K parts [P][T][N] fp32 in a scratch; per row block the last part adds them, push_rs's output
            assert not input_polled and not rows_split_over_gpus and N % (128 * GPUS) == 0, \
                "finish_rs: a counted input, every GPU all rows, whole 128-row tiles per GPU"
            out_buffer = lambda n: (GPUS * T * (N // GPUS) * 2, 0xFF, True)
            exchange = True
            buf_slots = (self._scratch("gemm_tile.finish_rs_parts", lambda n: n.grid[1] * T * N * 4),)
        elif combine == "local":   # one K part, bf16 [T][N], this GPU's buffer, 0xFF before each launch (its reader polls it)
            assert input_polled and not rows_split_over_gpus, "local: polled input, all rows"
            out_buffer = lambda n: (T * N * 2, 0xFF, False)
        elif combine == "slots" and input_polled:   # this GPU's rows only: [K parts][T][N / GPUS] fp32 (gemm_tile.cuh, run_gemm_tile)
            assert rows_split_over_gpus and N % (128 * GPUS) == 0, "polled: the rows split over the GPUs, whole 128-row blocks"
            out_buffer = lambda n: (n.grid[1] * T * (N // GPUS) * 4, 0xFF, True)
        elif combine == "slots":   # [K parts][T][N] fp32, 0xFF (the reader polls the slots and re-arms them)
            out_buffer = lambda n: (n.grid[1] * T * N * 4, 0xFF, True)
        elif combine == "add":     # [T][N] int64 fixed point, 0 before each launch
            out_buffer = lambda n: (T * N * 8, 0, False)
        else:                      # [T][N] fp32, every value written each launch
            out_buffer = lambda n: (T * N * 4, -1, False)
        assert combine != "add" or N // 128 <= 32, "one counter per 128-row block, at most 32"
        tiles = K // 128
        # HIP (gfx950): a task of 128, 64, 32 or 16 weight rows (a wave's MFMA tile is 16 rows; with 32 / 16 the 4 waves split
        # the K part too), K parts of any whole number of 128-column pieces, in every combine mode
        from . import gpu_backend
        hip = gpu_backend.backend() == "hip"
        rows_hip = (16, 1, 16, (1, 2, 4, 8, 16)) if hybrid == 4 else (16, 1, 8, (1, 2, 4, 8))   # (hybrid 4: 256-row tasks too)
        if input_polled:
            parts = next(y for y in range(1, tiles + 1) if tiles % y == 0 and tiles // y <= RING_STAGES)
            params, map_slots = [mode, K, N, 1], [self._gemm_map(weight, (128,)), -1]
            changeable = {0: rows_hip, 1: (128, 1, None)} if hip else {1: (128, 2, RING_STAGES)}
            if combine == "finish":
                assert hip, "finish: HIP only"
            if combine in ("push", "push_all", "local"):
                assert hip, "push: HIP only"
                parts, changeable = 1, {0: rows_hip}
        else:
            parts = 1
            params, map_slots = [mode, K, N, 0], [self._gemm_map(weight, (128, 64)), self._gemm_map(input, (T,))]
            if hip:
                changeable = {0: rows_hip, 1: (128, 1, None)} if combine != "store" else {0: rows_hip}
            else:
                changeable = {0: (64, 1, 2), 1: (128, 2, None)} if combine != "store" else {}
            if combine == "finish_quant":
                assert hip, "finish_quant: HIP only"
                changeable = {1: (128, 1, None)}   # 128-row tasks: one z_q tile each
        if hip:   # params[4]: the weight's layout (gemm_tile.cuh GemmLayout): the layer's gemm_weight_layout
            layout = getattr(self, "gemm_weight_layout", "rowmajor")
            assert layout in ("rowmajor", "shuffled"), layout
            params = params + [1 if layout == "shuffled" else 0]
            if norm_gamma is not None:   # params[5], [6]: the RMSNorm folded in, its eps (fp32 bits)
                params = params + [1, float_bits(norm_eps)]
            if hybrid:   # params[7] (5, 6: no RMSNorm): 1, or 4 (gemm_wave HYB4: 256-row tasks, 3 of a wave's 4 tiles in registers)
                assert norm_gamma is None and not input_polled and combine == "slots", "hybrid: a plain waited GEMM"
                params = params + [0, 0, 4 if hybrid == 4 else 1]
        else:
            assert not hybrid, "hybrid: HIP only"
        self._add_node("gemm_tile", (N // 128, parts, 1), block_dim, [input, weight, output],
                       [(-1, 1, -1), (0, 1, -1), (1, -1, -1)], params,
                       # K: at least 2 tiles of 128 per task (the two MMA issuers take the stages in turn, runtime.cuh issue_job;
                       # with one stage issuer 1 has no MMA and its stale accumulator would be added)
                       changeable_grid_dims=changeable, rows_split_over_gpus=rows_split_over_gpus, out_buffer=out_buffer,
                       counters={"slots": 0, "add": N // 128, "store": 1, "push": 0, "push_all": 0, "push_rs": 0, "local": 0,
                                 "finish_rs": N // 64,   # finish_rs: one per row block of 64 rows or more
                                 "finish": N // GPUS // 16,   # finish: one per row block of 16 rows or more
                                 "finish_quant": N // 128}[combine],
                       map_slots=map_slots, buf_slots=buf_slots,
                       **({"exchange": True} if exchange else {}))

    def scaled_gemm_tile_layer(
        self,
        input: DTensor,     # [T, K] e4m3 bytes (uint8): a group_quant node's output (then its fp32 scales [K / 128][T])
        weight: DTensor,    # [N, K] e4m3 bytes (uint8), in 128 x 128 pieces [N / 128][K / 128] (fp8_fp32scale_weights)
        weight_scales: DTensor,   # the checkpoint's fp32 block scales: [N / 128, K / 128], or [N / 256, K / 128, 2] (gate_up)
        output: DTensor,    # [T, N] fp32 ("add": int64 fixed point; its buffer holds what `combine` says)
        combine: str = "add",   # as gemm_tile_layer: "slots", "add" or "store"
        block_dim: tuple = (256, 1, 1),
    ):
        """Declare an FP8 GEMM with fp32 scales (tasks/scaled_gemm_tile.cuh).

        Task (x, y) computes 128 weight rows and K part y.
        Each piece has its own accumulator. The epilogue applies weight and activation scales before it adds the partial sums.
        Grid y divides K into parts of whole 128-column tiles. The "store" mode requires one K part.
        The scale values come from the checkpoint. Their layout selects the scale for each weight row:
          BlockScales: [N / 128, K / 128], one scale per piece (pack_w2).
          GateUpScales: [N / 256, K / 128, 2], separate gate and up scales (pack_w13).
        A GateUpScales piece holds 64 gate rows, then 64 up rows.
        Tensor maps load weights and activations. Pointers load fp32 weight scales.
        Weight maps cover one or two pieces. Activation maps cover one or two K tiles, in both exchange sets.
        params = [mode, K, N, scale layout (0 BlockScales, 1 GateUpScales)].
        buf_slots = [weight scale buffer slot]. map_slots = [weight map slot, activation map slot].
        """
        from .compiler import graph_nodes, producer_of
        assert input.num_dims == 2 and weight.num_dims == 2 and weight.dim(1) == input.dim(1)
        mode, N, K, T = COMBINE[combine], weight.dim(0), weight.dim(1), self._tokens(input.dim(0))
        self._dtypes("scaled_gemm_tile_layer", input=(input, uint8), weight=(weight, uint8), weight_scales=(weight_scales, float32),
                     output=(output, int64 if combine == "add" else float32))
        assert output.num_dims == 2 and output.dim(0) == input.dim(0) and output.dim(1) == N and N % 128 == 0 and K % 128 == 0
        q = producer_of(input, graph_nodes(self))
        if q is None or q.name != "group_quant":
            raise ValueError("scaled_gemm_tile_layer: the input must be a group_quant node's output")
        if any(n.name == "allreduce_send" and n.info["buf_slots"][0] == q.info["buf"] for n in graph_nodes(self)):
            raise ValueError("scaled_gemm_tile_layer: an allreduce_send node already re-arms this input after moe_experts; add the "
                             "scaled_gemm_tile node before it (allreduce_send_layer re-arms z_q only when moe_experts is its only reader)")
        dims = tuple(weight_scales.dim(i) for i in range(weight_scales.num_dims))
        layouts = {(N // 128, K // 128): 0, (N // 256, K // 128, 2): 1}
        if dims not in layouts:
            raise ValueError(f"scaled_gemm_tile_layer: weight scales {dims}, expected one of {list(layouts)}")
        layout = layouts[dims]
        assert combine != "add" or N // 128 <= 32, "one counter per 128-row block, at most 32"
        out_buffer = {"slots": lambda n: (n.grid[1] * T * N * 4, 0xFF, True),   # as gemm_tile_layer (not polled)
                      "add": lambda n: (T * N * 8, 0, False),
                      "store": lambda n: (T * N * 4, -1, False)}[combine]
        pieces = N // 128 * (K // 128)
        wmap = self._map("wblk8", ("t", weight), (pieces, 1))
        assert self._map("wblk8", ("t", weight), (pieces, 2)) == wmap + 1
        amap = None
        for kind, dims in (("act8", (K, T)), ("act8kt", (K, T, 2))):   # set 0, then set 1; the same maps as moe_experts' z_q
            for s in (0, 1):
                m = self._map(kind, ("x", q.info["buf"], 0, s), dims)
                amap = m if amap is None else amap
                assert m == amap + (2 if kind == "act8kt" else 0) + s, "scaled_gemm_tile: the four activation maps in consecutive slots"
        self._add_node("scaled_gemm_tile", (N // 128, 1, 1), block_dim, [input, weight, weight_scales, output],
                       # the input whole: each task waits until all group_quant tasks are counted
                       [(-1, -1, -1), (0, 1, -1), (-1, -1, -1), (1, -1, -1)], [mode, K, N, layout],
                       changeable_grid_dims={1: (128, 1, None)} if combine != "store" else {}, out_buffer=out_buffer,
                       counters={"slots": 0, "add": N // 128, "store": 1}[combine],
                       buf_slots=[self._tensor_slot(weight_scales)], map_slots=[wmap, amap])

    def topk_route_layer(
        self,
        input: DTensor,     # router logits [T, E] fp32
        bias: DTensor,      # score correction bias [E] fp32
        output: DTensor,    # routing pairs [T, K] int64
        routed_scaling_factor: float = None,   # multiplies the routed weights (None: no scaling)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task t = token t: adds the router's partial sums of row t, top K (output.dim(1)) of the E experts. params = [E, K], or
        [E, K, the routed scaling factor as fp32 bits]."""
        T, E, K = self._tokens(input.dim(0)), input.dim(1), output.dim(1)
        self._dtypes("topk_route_layer", input=(input, float32), bias=(bias, float32), output=(output, int64))
        assert input.num_dims == 2 and bias.num_dims == 1 and bias.dim(0) == E and output.num_dims == 2 and output.dim(0) == T
        if routed_scaling_factor is not None and not (math.isfinite(routed_scaling_factor) and routed_scaling_factor > 0):
            # the kernel reads the fp32 bits 0 as "no scaling": 0.0 would not scale
            raise ValueError(f"topk_route_layer: routed_scaling_factor must be finite and > 0 (or None), not {routed_scaling_factor}")
        params = [E, K] if routed_scaling_factor is None else [E, K, float_bits(routed_scaling_factor)]
        self._add_node("topk_route", (output.dim(0), 1, 1), block_dim, [input, bias, output],
                       [(0, -1, -1), (-1, -1, -1), (0, -1, -1)], params,
                       # (MPK_BOUND_FRONT_READY=1, a harness bound only: never reset - with the same inputs every launch the
                       # experts find an earlier launch's pairs at once)
                       out_buffer=lambda n: (T * K * 8, 0x1FF if os.environ.get("MPK_BOUND_FRONT_READY") == "1" else 0xFF, False),   # the pairs, u64, polled by every moe_experts task
                       buf_slots=[self._tensor_slot(bias)], dry=True, cost_inputs=[0])   # adds the router's K parts

    def sum_quant_send_layer(
        self,
        input: DTensor,     # latent z [T, L] fp32
        output: DTensor,    # z_q [T, L] e4m3 bytes (uint8)
        block_dim: tuple = (256, 1, 1),
        skip: bool = False,   # a placeholder node (params [L, 1], its tasks return at once): latent_down makes z_q (finish_quant)
    ):
        """Task x = columns [128 x, 128 x + 128) of z: adds latent_down's partial sums, MXFP8, sends to every GPU. Divided among
        the GPUs like latent_down's rows. Its output (z_q, then its scale chunks) is in the exchange region: every GPU writes its
        columns into every GPU's copy; 0xFF before the first launch, re-armed by the node after moe_experts (allreduce_send).
        params = [L] (z's width)."""
        if skip:   # (the graph's node numbers kept for the plans and recipes that name them)
            T, L = self._tokens(output.dim(0)), output.dim(1)
            self._add_node("sum_quant_send", (L // 128, 1, 1), block_dim, [input, output], [(1, 0, -1), (1, 0, -1)], [L, 1],
                           rows_split_over_gpus=True, changeable_grid_dims={1: (1, 1, T)},
                           out_buffer=lambda n: (16, -1, False), cost_inputs=[0])
            return
        T, L = self._tokens(input.dim(0)), input.dim(1)
        self._dtypes("sum_quant_send_layer", input=(input, float32), output=(output, uint8))
        assert input.num_dims == 2 and output.num_dims == 2 and (output.dim(0), output.dim(1)) == (T, L) and L % 128 == 0
        # (grid y: token groups, tasks/sum_quant_send.cuh Y > 1; 1 unless a plan regrids it)
        self._add_node("sum_quant_send", (output.dim(1) // 128, 1, 1), block_dim, [input, output],
                       [(1, 0, -1), (1, 0, -1)], [L], rows_split_over_gpus=True, changeable_grid_dims={1: (1, 1, T)},
                       # (MPK_BOUND_FRONT_READY=1, a harness bound only: z_q never reset, as the pairs)
                       out_buffer=lambda n: (zq_bytes(T, L), 0x1FF if os.environ.get("MPK_BOUND_FRONT_READY") == "1" else 0xFF, True),
                       exchange=True, cost_inputs=[0])   # adds latent_down's K parts

    def group_quant_layer(
        self,
        input: DTensor,     # x [T, L] bf16: a graph input (each GPU has all of x), or a node's output (h_s of silu_and_mul)
        output: DTensor,    # z_q [T, L] e4m3 bytes (uint8)
        block_dim: tuple = (256, 1, 1),
    ):
        """Quantize one 128-column group per task, with a separate scale for each token.

        Use s = max(amax, 1e-10) / 448 and q = e4m3(x / s).
        Store z_q in this GPU's exchange region, followed by fp32 scales [L / 128][T].
        Before each launch, fill the output with 0xFF for scaled_moe_experts to poll.
        Each task increments the completion counter after all its stores.
        scaled_gemm_tile waits for all these tasks before its TMA loads.
        params = [L]. buf_slots = [x's buffer slot], or [-1] if a node produces x.
        """
        from .compiler import graph_nodes, producer_of
        T, L = self._tokens(input.dim(0)), input.dim(1)
        self._dtypes("group_quant_layer", input=(input, bfloat16), output=(output, uint8))
        assert input.num_dims == 2 and (output.dim(0), output.dim(1)) == (T, L) and L % 128 == 0
        p = producer_of(input, graph_nodes(self))
        if p is not None and (p.info["buf"] < 0 or p.info["counter"] < 0 or p.info["exchange"]):
            raise ValueError(f"group_quant_layer: x from {p.name}: it must be a node's output buffer on this GPU (not in the "
                             f"exchange region) whose node counts its tasks")
        self._add_node("group_quant", (L // 128, 1, 1), block_dim, [input, output],
                       # x from a node: each task waits until all its tasks are counted
                       [(1, -1, -1) if p is None else (-1, -1, -1), (1, -1, -1)], [L],
                       out_buffer=lambda n: (zq32_bytes(T, L), 0xFF, False), exchange=True,
                       counters=1, buf_slots=[-1 if p is not None else self._tensor_slot(input)])

    def situ_and_mul_layer(
        self,
        input: DTensor,     # shared gate_up [T, 2 * SHR] int64 fixed point (a gemm_tile node's output, combine "add")
        output: DTensor,    # h_s [T, SHR] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = h_s features [F x, F x + F), from the gate and up columns of those features in shared gate_up (blocks of 128
        columns: 64 gate, then their 64 up). Default F = 64 (12 tasks); the compiler may give F = 32, 64, 96 or 128 (24, 12, 8, 6
        tasks): grid x in units of 32 columns, at most 8 per task of the input (F = 128 reads 256 input columns)."""
        T, SHR = self._tokens(input.dim(0)), output.dim(1)
        self._dtypes("situ_and_mul_layer", input=(input, int64), output=(output, bfloat16))
        assert input.num_dims == 2 and output.num_dims == 2 and input.dim(1) == 2 * SHR and output.dim(0) == T and SHR % 64 == 0
        self._add_node("situ_and_mul", (output.dim(1) // 64, 1, 1), block_dim, [input, output],
                       [(1, -1, -1), (1, -1, -1)], [], changeable_grid_dims={0: (32, 1, 8)},
                       out_buffer=lambda n: (T * SHR * 2, -1, False), counters=1,   # h_s bf16; a count of tasks
                       cost_inputs=[0])   # waits for shared gate_up's K parts

    def silu_and_mul_layer(
        self,
        input: DTensor,     # shared gate_up [T, 2 * SHR] int64 fixed point (a gemm_tile or scaled_gemm_tile node, combine "add")
        output: DTensor,    # h_s [T, SHR] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Compute h_s = bf16(SiLU(gate) * up) for the shared expert.

        Task x writes features [F x, F x + F). Each input block holds 64 gate values, then 64 up values.
        The default F is 64. The compiler can select F = 32, 64, 96, or 128 when SHR is divisible by F.
        The grid uses units of 32 input columns, with at most 8 units per task.
        F = 128 reads 256 input columns. tasks/silu_and_mul.cuh implements this task.
        situ_and_mul_layer implements K3's different activation, SiTU.
        """
        T, SHR = self._tokens(input.dim(0)), output.dim(1)
        self._dtypes("silu_and_mul_layer", input=(input, int64), output=(output, bfloat16))
        assert input.num_dims == 2 and output.num_dims == 2 and input.dim(1) == 2 * SHR and output.dim(0) == T and SHR % 64 == 0
        self._add_node("silu_and_mul", (output.dim(1) // 64, 1, 1), block_dim, [input, output],
                       [(1, -1, -1), (1, -1, -1)], [], changeable_grid_dims={0: (32, 1, 8)},
                       out_buffer=lambda n: (T * SHR * 2, -1, False), counters=1,   # h_s bf16; a count of tasks
                       cost_inputs=[0])   # waits for shared gate_up's K parts

    def moe_experts_layer(
        self,
        z_q: DTensor, pairs: DTensor,        # z_q uint8 (sum_quant_send's output), the routing pairs int64 (topk_route's)
        w13: DTensor, w13_scales: DTensor,   # MXFP4 weights and their scales, uint8 (the scales read through the tensor maps only)
        w2: DTensor, w2_scales: DTensor,
        output: DTensor,    # R [T x K, L] fp32: this GPU's routed rows, one per (token, routed expert k); allreduce_send adds a
                            # token's K in order (shared down is its own gemm_tile node)
        w13_cu: bool = False,   # HIP, T <= 16: a W13 item per CU, its 4 row tiles over the 4 waves (few items: small T)
        router_logits: DTensor = None, bias: DTensor = None,   # HIP, T = 1, w13_cu: a local route (each CU makes the pairs
                                # itself from the router's K parts and the score bias, tasks/moe_experts.cuh RSPLIT)
        block_dim: tuple = (256, 1, 1),
    ):
        """One task per SM (task x on SM x); the SMs take the expert work from a queue at run time; the output rows are the
        node's buffer. Declares what its kernel template (tasks/moe_experts.cuh ExpertSlots) reads: h_q and its scales (scratch),
        the tensor maps of the expert weights and scales, of z_q (both sets of the exchange region) and of h_q; its counters: the
        SMs done, the queue's next entry, per expert slot (T x K of them) the W13 items done (one line, one line, T x K / 32 lines).
        Sizes: E experts and K per token (pairs, topk_route's params), L = z_q's width, IR = the expert intermediate on this GPU
        (w13 [E, 2 IR, L / 2] MXFP4 bytes, w2 [E, L, IR / 2]; their scales one byte per 32 values); params = [IR]."""
        T, K, L = self._tokens(pairs.dim(0)), pairs.dim(1), z_q.dim(1)
        E, IR = w13.dim(0), w13.dim(1) // 2
        NSLOT, KT_L, KT2 = T * K, L // 128, IR // 128
        self._dtypes("moe_experts_layer", z_q=(z_q, uint8), pairs=(pairs, int64), w13=(w13, uint8), w13_scales=(w13_scales, uint8),
                     w2=(w2, uint8), w2_scales=(w2_scales, uint8), output=(output, float32))
        assert w13.num_dims == 3 and (w13.dim(1), w13.dim(2)) == (2 * IR, L // 2) and IR % 128 == 0 and L % 128 == 0
        assert w2.num_dims == 3 and (w2.dim(0), w2.dim(1), w2.dim(2)) == (E, L, IR // 2)
        assert (w13_scales.dim(0), w13_scales.dim(1), w13_scales.dim(2)) == (E, 2 * IR, L // 32)
        assert (w2_scales.dim(0), w2_scales.dim(1), w2_scales.dim(2)) == \
            (E, L, 16 if os.environ.get("MPK_MOE_AITER") == "1" else IR // 32)   # (MPK_MOE_AITER: AITER's padded W2 scales)
        assert (output.dim(0), output.dim(1)) == (T * K, L)
        from .compiler import graph_nodes, producer_of
        zq = producer_of(z_q, graph_nodes(self))
        if zq is None or not zq.info["exchange"]:
            raise ValueError("moe_experts_layer: z_q must be a node's output in the exchange region (sum_quant_send_layer)")
        route = producer_of(pairs, graph_nodes(self))
        if route is None or route.name != "topk_route" or list(route.params) != [E, K]:
            raise ValueError(f"moe_experts_layer: the pairs must be a topk_route node's over the {E} experts of w13")
        nbytes = lambda t: t.dim(0) * t.dim(1) * t.dim(2)
        from . import gpu_backend
        hip = gpu_backend.backend() == "hip"
        # HIP: h_q and its scales 0xFF before each launch (W2 polls them: tasks/moe_experts.cuh expert_w2_hq; unused
        # columns are never read there)
        # (tasks/moe_experts_csr.cuh: h_q and its scales by pair, [T x K][IR] and [T x K][IR / 128][4]; T <= 16 too with
        # MPK_MOE_CSR=1: params[2] = 1)
        csr = hip and (T > 16 or os.environ.get("MPK_MOE_CSR") == "1")
        hq = self._scratch("moe_experts.h_q", T * K * IR if csr else NSLOT * T * IR, 0xFF if hip else -1)  # every row written before its counter lets W2 read it
        hsf = self._scratch("moe_experts.h_q_scales", T * K * KT2 * 4 if csr else NSLOT * KT2 * SF_CHUNK,
                            0xFF if hip else 0)   # (CUDA: unused bytes stay 0)
        # HIP: the work queues, per die one for the W13 items and one for the W2 items (16 counters, one 128-B line each, 0
        # before each launch; tasks/moe_experts.cuh)
        # HIP, one scratch buffer, 0 before each launch (tasks/moe_experts.cuh MISC_TRACE): the work queues (per die one for the
        # W13 items and one for the W2 items: 16 counters, one 128-B line each), then, the timing build, the item trace (per
        # wave, 4 per SM, 32 items x 16 B)
        # then 8 KB not used (a count per W13 item of the removed tail split), before the trace; then 32 counters 4 KB apart:
        # n_live, R's tiles done per 128-column block (core.cuh MOE_MISC_RBLK). moe_experts.w13_half: not used now (the
        # removed tail split's halves); kept so the buffer slots stay as the libraries were measured with
        queues = [self._scratch("moe_experts.misc", 16 * 128 + 8192 + self._num_sms() * 4 * 32 * 16 + 32 * 4096, 0),
                  self._scratch("moe_experts.w13_half", 16 if csr else NSLOT * (IR // 32) * 4 * 2 * 64 * 16)] \
            if gpu_backend.backend() == "hip" else []
        w13_pieces, w2_pieces = nbytes(w13) // 8192, nbytes(w2) // 8192              # MXFP4: 128 x 128 pieces of 8 KB, scales 512 B each
        maps = []
        for weight, scales, pieces in ((w13, w13_scales, w13_pieces), (w2, w2_scales, w2_pieces)):
            maps += [self._map("wblk", ("t", weight), (pieces, 1)), self._map("wblk", ("t", weight), (pieces, 2)),
                      self._map("sf", ("t", scales), (pieces, 1)), self._map("sf", ("t", scales), (pieces, 2))]
        for kind, offset, dims in (("act8", 0, (L, T)), ("act8kt", 0, (L, T, 2)), ("sf", T * L, (KT_L, KT_L))):
            first = self._map(kind, ("x", zq.info["buf"], offset, 0), dims)      # z_q in set 0, then set 1 (the next slot)
            assert self._map(kind, ("x", zq.info["buf"], offset, 1), dims) == first + 1
            maps.append(first)
        maps += [self._map("act8kt", ("b", hq, 0), (IR, T * K if csr else NSLOT * T, KT2)),
                 self._map("sf", ("b", hsf, 0), (NSLOT * KT2, KT2))]
        local = router_logits is not None
        if local:
            router = producer_of(router_logits, graph_nodes(self))
            assert hip and T == 1 and w13_cu and not csr and bias is not None and router is not None and \
                router.graph_idx == producer_of(route.inputs[0], graph_nodes(self)).graph_idx, \
                "moe_experts_layer: a local route: HIP, T = 1, " \
                "w13_cu, topk_route's own router logits and bias"
        tensors = [z_q, pairs, w13, w2, output]   # (a local route: the router's node reaches the kernel through
                                                  # compiler.node_table, the bias through buffer slot 4)
        self._add_node("moe_experts", (self._num_sms(), 1, 1), block_dim, tensors, [(-1, -1, -1)] * len(tensors),
                       [IR] + ([int(w13_cu), int(csr)] if (w13_cu or csr) else []) + ([1] if local else []),
                       one_task_per_sm=True,
                       out_buffer=lambda n: (T * K * L * 4, -1, False),
                       counters=32 * 2 + (0 if csr else NSLOT),   # (CHQ: per slot, unused on HIP; T > 16 none)
                       buf_slots=[hq, hsf] + queues + ([self._tensor_slot(bias)] if local else []), map_slots=maps)

    def scaled_moe_experts_layer(
        self,
        z_q: DTensor, pairs: DTensor,        # z_q uint8 (group_quant's output), the routing pairs int64 (topk_route's)
        w13: DTensor, w13_scales: DTensor,   # e4m3 weights (uint8) and their fp32 block scales (read by pointer)
        w2: DTensor, w2_scales: DTensor,
        output: DTensor,    # R [T x K, L] fp32: this GPU's routed rows, one per (token, routed expert k); allreduce_send adds a
                            # token's K in order
        block_dim: tuple = (256, 1, 1),
    ):
        """Declare the routed FP8 experts in tasks/scaled_moe_experts.cuh.

        Weights retain the checkpoint's e4m3 bytes and fp32 scales for 128 x 128 blocks.
        Activations x and h use e4m3, with an fp32 scale per token and 128 values. The expert activation is SiLU.
        One task runs on each SM. The SMs take expert work from a shared queue and write routed output rows.
        buf_slots: the scratch buffers for h_q, its scales, h in fp32 and group maxima, and the fp32 weight scales (by pointer).
        map_slots: the tensor maps of the weights, of z_q in both exchange sets, and of h_q (scaled_moe::Slots).
        Counters track completed SMs, the next queue entry, completed W13 items per expert slot, and completed items per h group.
        The first two counters each reserve 32 words. The other counters reserve NSLOT and NSLOT * IR / 128 words.
        E is the expert count. K is the experts per token. L is the hidden width.
        IR is this GPU's expert intermediate width: 128 or 256. GLM-5.3 therefore needs TP8 or TP16.
        Weight layouts from fp8_fp32scale_weights:
          w13: [E, 2 IR, L] e4m3 bytes; scales: [E, IR / 128, L / 128, 2] fp32 (gate, up).
          w2: [E, L, IR] e4m3 bytes; scales: [E, L / 128, IR / 128] fp32.
        z_q must come from group_quant_layer. params = [IR].
        """
        from .compiler import graph_nodes, producer_of
        T, K, L = self._tokens(pairs.dim(0)), pairs.dim(1), z_q.dim(1)
        E, IR = w13.dim(0), w13.dim(1) // 2
        NSLOT, KT_L, KT2 = T * K, L // 128, IR // 128
        self._dtypes("scaled_moe_experts_layer", z_q=(z_q, uint8), pairs=(pairs, int64), w13=(w13, uint8),
                     w13_scales=(w13_scales, float32), w2=(w2, uint8), w2_scales=(w2_scales, float32), output=(output, float32))
        dims = lambda t: tuple(t.dim(i) for i in range(t.num_dims))
        assert dims(w13) == (E, 2 * IR, L) and dims(w2) == (E, L, IR) and IR % 128 == 0 and L % 128 == 0 and KT2 <= 2
        assert dims(w13_scales) == (E, IR // 128, KT_L, 2) and dims(w2_scales) == (E, KT_L, KT2)
        assert (output.dim(0), output.dim(1)) == (T * K, L)
        if any(n.name in ("moe_experts", "scaled_moe_experts") for n in graph_nodes(self)):
            # the expert phase switch runs once per kernel (KernelLocals::ring_relaid): a second expert node would skip it
            raise ValueError("scaled_moe_experts_layer: the graph already has an expert node; one expert node per graph")
        zq = producer_of(z_q, graph_nodes(self))
        if zq is None or zq.name != "group_quant":
            raise ValueError("scaled_moe_experts_layer: z_q must be a group_quant node's output")
        route = producer_of(pairs, graph_nodes(self))
        if route is None or route.name != "topk_route" or list(route.params[:2]) != [E, K]:
            raise ValueError(f"scaled_moe_experts_layer: the pairs must be a topk_route node's over the {E} experts of w13")
        hq = self._scratch("scaled_moe_experts.h_q", NSLOT * T * IR)          # every row written before its counter lets W2 read it
        # the h_q scales, 0xFF (NaN) before each launch: the W2 epilogue polls the values themselves (data as the flag)
        hsf = self._scratch("scaled_moe_experts.h_q_scales", NSLOT * T * KT2 * 4, 0xFF)
        w13_pieces, w2_pieces = E * 2 * IR * L // 16384, E * L * IR // 16384   # one 128 x 128 e4m3 piece: 16 KB
        bufs = [hq, hsf, self._tensor_slot(w13_scales),
                self._scratch("scaled_moe_experts.h32", NSLOT * T * IR * 4 + NSLOT * (IR // 64) * T * 4),   # h (fp32), its amax
                self._tensor_slot(w2_scales)]
        maps = [self._map("wblk8", ("t", w13), (w13_pieces, 1)), self._map("wblk8", ("t", w13), (w13_pieces, 2)),
                self._map("wblk8", ("t", w2), (w2_pieces, 1)), self._map("wblk8", ("t", w2), (w2_pieces, 2))]
        for kind, dims_ in (("act8", (L, T)), ("act8kt", (L, T, 2))):
            first = self._map(kind, ("x", zq.info["buf"], 0, 0), dims_)       # z_q in set 0, then set 1 (the next slot)
            assert self._map(kind, ("x", zq.info["buf"], 0, 1), dims_) == first + 1
            maps.append(first)
        maps.append(self._map("act8kt", ("b", hq, 0), (IR, NSLOT * T, KT2)))
        tensors = [z_q, pairs, w13, w2, output]
        self._add_node("scaled_moe_experts", (self._num_sms(), 1, 1), block_dim, tensors, [(-1, -1, -1)] * len(tensors),
                       [IR], one_task_per_sm=True, out_buffer=lambda n: (T * K * L * 4, -1, False),
                       counters=32 * 2 + NSLOT + NSLOT * KT2, buf_slots=bufs, map_slots=maps)

    # ---- after moe_experts (tasks/allreduce_send.cuh, sum_rmsnorm.cuh, sum_gpus.cuh, sum_send.cuh, residual_add.cuh). The
    # graph tensors between them (the sent partial sums, sum_rmsnorm's Rn, sum_gpus's Ssum, latent_up's K parts, sum_send's o)
    # stand for the buffers the bodies use; their input maps give the dependencies.
    def allreduce_send_layer(
        self,
        input: DTensor,         # fp32 [T x G, W], a node's output buffer: G rows per token (added in order), e.g. this GPU's routed
                                # rows R [T x 16, LAT] (moe_experts, scaled_moe_experts) or S [T, H] (the shared down node)
        output: DTensor,        # what is sent: [T, W] bf16 (its buffer: [num_gpus][T][W] bf16 in the exchange region)
        num_tasks: int = None,  # slices of the partial sum; default: one per task of the producer (as its grid is now)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x: slice x of this GPU's partial sum (bf16) to every GPU, into this GPU's slot of the node's output (in the exchange
        region), once the input's producer is done (all its tasks, counted in its counter). params = [W, G]. When the producer is
        moe_experts or scaled_moe_experts, re-arm z_q here only if that producer is its sole reader.
        buf_slots = [z_q's buffer slot] enables re-arm; [-1] disables it. With other readers, reset z_q before each launch instead.
        Declare all other z_q readers before this call so the reader check sees them."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        if p is None or p.info["buf"] < 0 or p.info["counter"] < 0:
            raise ValueError(f"allreduce_send_layer: the input must be a node's output buffer whose node counts its tasks; it comes from "
                             f"{p.name if p else 'the graph inputs'}")
        T, W, GPUS = self._tokens(output.dim(0)), output.dim(1), self.num_gpus
        self._dtypes("allreduce_send_layer", input=(input, float32), output=(output, bfloat16))
        assert input.dim(0) % T == 0 and input.dim(1) == W and W % 8 == 0
        rearm = -1
        params, slots = [W, input.dim(0) // T], []
        if p.name in ("moe_experts", "scaled_moe_experts"):   # z_q: its first input; another reader (scaled_gemm_tile) can still read it
            zq = p.inputs[0]
            readers = [n for n in graph_nodes(self) if any(t.guid == zq.guid for t in n.inputs)]
            if len(readers) == 1:
                rearm = producer_of(zq, graph_nodes(self)).info["buf"]
            if os.environ.get("MPK_BOUND_FRONT_READY") == "1":   # (a harness bound only: z_q left as the last launch wrote it)
                rearm = -1
        if p.name == "moe_experts":
            from . import gpu_backend
            if gpu_backend.backend() == "hip":   # params[2] 1, buf_slots[1] moe's misc scratch (R in blocks)
                params, slots = params + [1], [p.info["buf_slots"][2]]
        tasks = num_tasks or p.grid[0] * p.grid[1] * p.grid[2]
        self._add_node("allreduce_send", (tasks, 1, 1), block_dim, [input, output],
                       [(-1, -1, -1)] * 2, params,
                       out_buffer=lambda n: (GPUS * T * W * 2, 0xFF, True), exchange=True, buf_slots=[rearm] + slots,
                       grid_choices=sorted({(s, 1, 1) for s in (8, 16, 32, 64, 128, tasks) if s <= T * W // 8}))

    def sum_rmsnorm_layer(
        self,
        input: DTensor,         # R as sent (allreduce_send's output, bf16)
        gamma: DTensor,         # RMSNorm weight [L] bf16
        output: DTensor,        # Rn [T, L] bf16
        recompute: bool = False,
        eps: float = 1e-5,
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y): part x of R's columns for token y (1, 2 or 4 parts: the compiler picks): add R over the GPUs, RMSNorm.
        The tasks of a token need the sum of squares of the whole row. Default: they run at the same time on different SMs and
        swap their sums through a scratch buffer, ss_part (params [0, eps]); with 2 parts the compiler may put the pair on one cluster,
        where they swap in shared memory (params [2, eps], compile_plan). recompute: each task adds the squares of the whole row
        itself (params [1, eps]; no swap, so no rule on where the tasks run). eps: as its fp32 bits (float_bits)."""
        T, L = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("sum_rmsnorm_layer", input=(input, bfloat16), gamma=(gamma, bfloat16), output=(output, bfloat16))
        assert gamma.dim(0) == L and L % 64 == 0 and L // 16 <= 256   # tasks/sum_rmsnorm.cuh rn_threads
        e = float_bits(eps)
        self._add_node("sum_rmsnorm", (4, T, 1), block_dim, [input, gamma, output],
                       [(-1, -1, -1), (-1, -1, -1), (1, 0, -1)], [1 if recompute else 0, e],
                       changeable_grid_dims={0: (L // 4, 1, 4)},   # whole quarters: 1, 2 or 4 tasks per token
                       concurrent_along_axis=None if recompute else 0, cluster_pair_params=None if recompute else [2, e],
                       out_buffer=lambda n: (T * L * 2, 0xFF, False),   # Rn bf16, polled by every latent_up task of its columns
                       buf_slots=[self._tensor_slot(gamma),   # ss_part: read in mode 0 only
                                  self._scratch("sum_rmsnorm.ss_part", lambda n: T * 4 * 4 if n.params[0] == 0 else 0, 0xFF)])

    def sum_gpus_layer(
        self,
        input: DTensor,         # S as sent (allreduce_send's output, bf16)
        output: DTensor,        # Ssum [T, H] bf16
        parts: int = 8,         # tasks per token, each H / parts columns
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y): part x of S for token y: add S over the GPUs."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("sum_gpus_layer", input=(input, bfloat16), output=(output, bfloat16))
        assert H % (8 * parts) == 0 and H // parts // 8 <= 256   # tasks/sum_gpus.cuh: one 8-column vector per thread of a part
        self._add_node("sum_gpus", (parts, T, 1), block_dim, [input, output], [(-1, -1, -1), (1, 0, -1)], [],
                       out_buffer=lambda n: (T * H * 2, 0xFF, True))   # Ssum bf16, polled and re-armed by residual_add

    def sum_send_layer(
        self,
        input: DTensor,         # [T, H] fp32: a gemm_tile node's output in slots (its K parts), its rows split over the GPUs
        output: DTensor,        # o [T, H] bf16: each GPU sends its own rows to every GPU
        slices: int = 14,       # tasks per row block, each 128 / slices of the block's 128 16-B vectors
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y): row block x (this GPU's blocks, as its producer's), slice y of `slices` of the block's vectors: adds the K
        parts, sends to every GPU."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        assert p is not None and p.name == "gemm_tile" and p.params[0] == COMBINE["slots"] and p.info["rows_split_over_gpus"], \
            "sum_send_layer: the input is a gemm_tile node's output in slots, its rows split over the GPUs"
        T, H = self._tokens(input.dim(0)), input.dim(1)
        self._dtypes("sum_send_layer", input=(input, float32), output=(output, bfloat16))
        assert (output.dim(0), output.dim(1)) == (T, H) and H % (128 * self.num_gpus) == 0
        # at most MAX_SUM_PARTS K parts: the producer's tasks take at least K / 128 / MAX_SUM_PARTS pieces of K
        spec = p.info["changeable_grid_dims"].get(1)
        if spec is not None:
            unit, lo, hi = (spec, 1, None) if isinstance(spec, int) else tuple(spec)
            p.info["changeable_grid_dims"][1] = (unit, max(lo, -(-(p.inputs[0].dim(1) // unit) // MAX_SUM_PARTS)), hi)
        assert p.grid[1] <= MAX_SUM_PARTS, f"sum_send_layer: the producer has {p.grid[1]} K parts, at most {MAX_SUM_PARTS}"
        assert 1 <= slices <= 128   # tasks/sum_send.cuh
        self._add_node("sum_send", (H // 128, slices, 1), block_dim, [input, output],
                       [(1, -1, -1), (1, -1, -1)], [], rows_split_over_gpus=True,
                       out_buffer=lambda n: (T * H * 2, 0xFF, True), exchange=True,   # [GPUS][T][H / GPUS] bf16
                       cost_inputs=[0])   # adds the GEMM's K parts

    # ---- the K3 KDA layer (HIP; tasks/short_conv_silu.cuh, delta_rule_recurrence.cuh, attn_residual.cuh). The serving
    # contract (SGLang's KDA target-verify): the conv window and the recurrent state are read from the framework's pools at
    # the slot, not written; the ring entries (k, v after the conv, the gate's decay, sigmoid(beta)) and the windows after
    # each token are written. info (int32): [bank rows, bank token stride, bank row stride (elements), ring length L, icw
    # steps, slot, icw row, 0]
    def short_conv_silu_layer(
        self,
        input: DTensor,         # [T, N] fp32: the projection, a gemm_tile node's K parts in slots
        conv_weight: DTensor,   # [NCONV, 4] fp32: the 4 taps per q / k / v channel (tap 3 = the newest input)
        conv_pool: DTensor,     # [slots x 3, NCONV] bf16: the framework's carried windows (oldest first), read at slot[0]
        output: DTensor,        # [T, N] bf16 (0xFF before each launch): the projection's row; then [T][NCONV] fp32 q / k / v
        icw: DTensor,           # [rows x steps x 3, NCONV] bf16: per token the window after it, at row icw_row[0]
        ring_k: DTensor,        # [slots x heads x L, 128] bf16: k after the conv + SiLU, per token
        ring_v: DTensor,        # the same for v
        info: DTensor,          # int32 [8]: [bank rows, bank strides (2), ring length, window steps, 0, 0, 0]
        slot: DTensor = None,   # int32 [>= 1]: the request's state slot (the framework's index tensor, read in the kernel)
        icw_row: DTensor = None,  # int32 [>= 1]: the request's window scratch row
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = projection columns [64 x, + 64): the K parts added; columns < NCONV through the 4-tap conv over the pool's
        window and the T new inputs, SiLU, out in fp32 (after the row); the other columns (g, f_a, beta) bf16 into the row.
        params = [NCONV]; buf_slots = conv weights, pool, icw, ring_k, ring_v, info, slot, icw_row; one count per task."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        assert p is not None and p.name == "gemm_tile" and p.params[0] == COMBINE["slots"], \
            "short_conv_silu_layer: the input is a gemm_tile node's K parts in slots"
        T, N, NCONV = self._tokens(input.dim(0)), input.dim(1), conv_weight.dim(0)
        self._dtypes("short_conv_silu_layer", input=(input, float32), conv_weight=(conv_weight, float32),
                     conv_pool=(conv_pool, bfloat16), output=(output, bfloat16), icw=(icw, bfloat16), ring_k=(ring_k, bfloat16),
                     ring_v=(ring_v, bfloat16), info=(info, int32), slot=(slot, int32), icw_row=(icw_row, int32))
        assert N % 64 == 0 and NCONV % 384 == 0 and NCONV <= N and conv_weight.dim(1) == 4 and conv_pool.dim(1) == NCONV
        self._add_node("short_conv_silu", (N // 64, 1, 1), block_dim, [input, conv_weight, output],
                       [(1, -1, -1), (-1, -1, -1), (1, -1, -1)], [NCONV],
                       out_buffer=lambda n: (T * N * 2 + T * NCONV * 4, 0xFF, False), counters=1,
                       buf_slots=[self._tensor_slot(t) for t in (conv_weight, conv_pool, icw, ring_k, ring_v, info, slot, icw_row)],
                       cost_inputs=[0])

    def delta_rule_recurrence_layer(
        self,
        rec_in: DTensor,        # short_conv_silu's output: [T, PROJ_N] bf16 row (g, f_a, beta) + [T][3 HD] fp32 q / k / v
        wfb: DTensor,           # [HD, 128] bf16: the gate's second projection (f_b = f_a . W_fb^T per head)
        prm: DTensor,           # [heads + HD + 128] fp32: A_log per head, dt_bias per head and k, the output norm's weight
        state_pool: DTensor,    # [slots x heads x 128 v, 128 k] fp32: the framework's state, read at slot[0], not written
        output: DTensor,        # y [T, HD] bf16 (0xFF before each launch: o_proj polls it)
        ring_g: DTensor,        # [slots x heads x L, 128] fp32: per token the gate's natural-log decay
        ring_beta: DTensor,     # [slots x heads, L] fp32: sigmoid(beta)
        info: DTensor,          # int32 [8]
        slot: DTensor = None,   # int32 [>= 1]: the request's state slot
        col_g: int = 4608, col_fa: int = 6144, col_beta: int = 6272,
        onorm_eps: float = 1e-5,
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = head x, the T tokens in order: f_b, the gate, beta; q, k L2-normalised; the gated delta rule on the head's
        128 x 128 state (from the pool); o_norm x weight x sigmoid(g) -> y. params = [PROJ_N, col_g, col_fa, col_beta, eps bits];
        buf_slots = wfb, prm, state_pool, ring_g, ring_beta, info, slot; one count per head."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(rec_in, graph_nodes(self))
        assert p is not None and p.name == "short_conv_silu", "delta_rule_recurrence_layer: the input is short_conv_silu's output"
        T, PROJ_N, HD = self._tokens(rec_in.dim(0)), rec_in.dim(1), wfb.dim(0)
        heads = HD // 128
        self._dtypes("delta_rule_recurrence_layer", rec_in=(rec_in, bfloat16), wfb=(wfb, bfloat16), prm=(prm, float32),
                     state_pool=(state_pool, float32), output=(output, bfloat16), ring_g=(ring_g, float32),
                     ring_beta=(ring_beta, float32), info=(info, int32), slot=(slot, int32))
        assert HD % 128 == 0 and wfb.dim(1) == 128 and prm.dim(0) == heads + HD + 128 and p.params[0] == 3 * HD
        assert (output.dim(0), output.dim(1)) == (T, HD) and state_pool.dim(1) == 128 and state_pool.dim(0) % (heads * 128) == 0
        assert col_g + HD <= PROJ_N and col_fa + 128 <= PROJ_N and col_beta + heads <= PROJ_N
        self._add_node("delta_rule_recurrence", (heads, 1, 1), block_dim, [rec_in, wfb, output],
                       [(-1, -1, -1), (0, -1, -1), (1, -1, -1)],
                       [PROJ_N, col_g, col_fa, col_beta, float_bits(onorm_eps)],
                       out_buffer=lambda n: (T * HD * 2, 0xFF, False), counters=1,
                       buf_slots=[self._tensor_slot(t) for t in (wfb, prm, state_pool, ring_g, ring_beta, info, slot)])

    # ---- the K3 KDA layer in plain decode (HIP; tasks/short_conv_decode.cuh, delta_rule_decode.cuh): T requests, one new token
    # each (row t = request t). The serving contract (SGLang's KDA decode, amd/notes/sglang_v0521_decode_contract.md B.3): the
    # conv window and the recurrent state of request t are read and written in place at slot[t]; slot[t] < 0 = a padded row
    # (no state read or written).
    def short_conv_decode_layer(
        self,
        input: DTensor,         # [T, N] fp32: the projection, a gemm_tile node's K parts in slots
        conv_weight: DTensor,   # [NCONV, 4] fp32: the 4 taps per q / k / v channel (tap 3 = the newest input)
        conv_pool: DTensor,     # [slots x 3, NCONV] bf16: the windows (oldest first), read and written at slot[t]
        output: DTensor,        # [T, N] bf16: the projection's row; then [T][NCONV] fp32 q / k / v
        slots: DTensor,         # int32 [T]: request t's state slot (< 0: a padded row)
        tokens_per_task: int = 0,   # grid y = T / tokens_per_task (0: all T; the compiler may regrid y: 1 .. 64 tokens)
        conv_only: bool = False,    # only the q / k / v columns (grid x = NCONV / 64): delta_rule_decode reads g, f_a and beta
                                    # from the projection's K parts itself (its proj input)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y) = projection columns [64 x, + 64) of the tokens [TS y, TS y + TS) (TS = T / grid y): the K parts added; columns < NCONV per request through the 4-tap conv over
        its window and the new input, SiLU, out in fp32 (after the row), the window moved on in place; the other columns (g, f_a,
        beta) bf16 into the row. params = [NCONV]; buf_slots = conv weights, pool, slots; one count per task."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        assert p is not None and p.name == "gemm_tile" and p.params[0] == COMBINE["slots"], \
            "short_conv_decode_layer: the input is a gemm_tile node's K parts in slots"
        T, N, NCONV = self._tokens(input.dim(0)), input.dim(1), conv_weight.dim(0)
        self._dtypes("short_conv_decode_layer", input=(input, float32), conv_weight=(conv_weight, float32),
                     conv_pool=(conv_pool, bfloat16), output=(output, bfloat16), slots=(slots, int32))
        assert N % 64 == 0 and NCONV % 384 == 0 and NCONV <= N and conv_weight.dim(1) == 4 and conv_pool.dim(1) == NCONV
        assert slots.dim(0) >= T
        tpt = tokens_per_task or T
        assert T % tpt == 0, "short_conv_decode_layer: tokens_per_task divides T"
        cut = (-1, 0, -1) if conv_only else (1, 0, -1)   # (conv_only: grid x covers only the first NCONV of the N columns)
        self._add_node("short_conv_decode", ((NCONV if conv_only else N) // 64, T // tpt, 1), block_dim, [input, conv_weight, output],
                       [cut, (-1, -1, -1), cut], [NCONV], changeable_grid_dims={1: (1, 1, 64)},
                       out_buffer=lambda n: (T * N * 2 + T * NCONV * 4, 0xFF, False), counters=1,
                       buf_slots=[self._tensor_slot(t) for t in (conv_weight, conv_pool, slots)],
                       cost_inputs=[0])

    def delta_rule_decode_layer(
        self,
        rec_in: DTensor,        # short_conv_decode's output: [T, PROJ_N] bf16 row (g, f_a, beta) + [T][3 HD] fp32 q / k / v
        wfb: DTensor,           # [HD, 128] bf16: the gate's second projection (f_b = f_a . W_fb^T per head)
        prm: DTensor,           # [heads + HD + 128] fp32: A_log per head, dt_bias per head and k, the output norm's weight
        state_pool: DTensor,    # [slots x heads x 128 v, 128 k] fp32: the states, read and written at slot[t]
        output: DTensor,        # y [T, HD] bf16 (0xFF before each launch: o_proj polls it)
        slots: DTensor,         # int32 [T]: request t's state slot (< 0: a padded row)
        requests_per_task: int = 1,
        col_g: int = 4608, col_fa: int = 6144, col_beta: int = 6272,
        pairs_per_task: int = 0,  # HIP, > 0 (no proj): task x = the (head, request) pairs P x .. + P - 1, head-major, every state
                                  # loaded at the start (delta_rule_decode.cuh FLAT); grid (heads T / P, 1, 1); params + [2, P]
        onorm_eps: float = 1e-5,
        proj: DTensor = None,   # the projection (a gemm_tile node's K parts in slots), with short_conv_decode conv_only: f_a,
                                # beta and g read from it (polled, the parts added in order, bf16, as short_conv_decode does)
        block_dim: tuple = (256, 1, 1),
        conv: tuple = None,     # (conv_weight [3 HD, 4] fp32, conv_pool [slots x 3, 3 HD] bf16), rec_in None: no conv node - each
                                # task makes its head's q / k / v from proj itself (delta_rule_decode.cuh CIR); params + [1, HD]
    ):
        """Task (x, y) = head x, requests R y .. + R - 1 (R = T / grid y, 1 to 16; the compiler may change grid y), one after
        another: f_b, the gate, beta; q, k L2-normalised; one gated delta-rule step on the request's 128 x 128 state, written
        back; o_norm x weight x sigmoid(g) -> y. params = [PROJ_N, col_g, col_fa, col_beta, eps bits]; buf_slots = wfb, prm,
        state_pool, slots; one count per task."""
        from .compiler import graph_nodes, producer_of
        if conv is not None:   # (CIR: no conv node)
            conv_weight, conv_pool = conv
            assert rec_in is None and proj is not None, "delta_rule_decode_layer conv: proj, no rec_in"
            pp = producer_of(proj, graph_nodes(self))
            assert pp is not None and pp.name == "gemm_tile" and pp.params[0] == COMBINE["slots"], \
                "delta_rule_decode_layer conv: proj = the projection's K parts in slots"
            T, PROJ_N, HD = self._tokens(proj.dim(0)), proj.dim(1), wfb.dim(0)   # (the row width, as rec_in's)
            heads = HD // 128
            self._dtypes("delta_rule_decode_layer", proj=(proj, float32), wfb=(wfb, bfloat16), prm=(prm, float32),
                         state_pool=(state_pool, float32), output=(output, bfloat16), slots=(slots, int32),
                         conv_weight=(conv_weight, float32), conv_pool=(conv_pool, bfloat16))
            assert conv_weight.dim(0) == 3 * HD and conv_weight.dim(1) == 4 and conv_pool.dim(1) == 3 * HD and pp.params[2] == PROJ_N
            assert HD % 128 == 0 and wfb.dim(1) == 128 and prm.dim(0) == heads + HD + 128
            assert (output.dim(0), output.dim(1)) == (T, HD) and state_pool.dim(1) == 128 and state_pool.dim(0) % (heads * 128) == 0
            assert col_g + HD <= PROJ_N and col_fa + 128 <= PROJ_N and col_beta + heads <= PROJ_N
            assert 1 <= requests_per_task <= 16 and T % requests_per_task == 0, "delta_rule_decode_layer: R divides T, at most 16"
            if pairs_per_task > 0:   # (FLAT + CIR: the pairs as FLAT, each pair's q / k / v made from proj; params + [3, P, HD])
                assert 1 <= pairs_per_task <= 4, "delta_rule_decode_layer pairs_per_task: 1 .. 4"
                self._add_node("delta_rule_decode", (-(-heads * T // pairs_per_task), 1, 1), block_dim, [proj, wfb, output],
                               [(-1, -1, -1), (0, -1, -1), (1, 0, -1)],
                               [PROJ_N, col_g, col_fa, col_beta, float_bits(onorm_eps), 3, pairs_per_task, HD],
                               out_buffer=lambda n: (T * HD * 2, 0xFF, False), counters=1,
                               buf_slots=[self._tensor_slot(t) for t in (wfb, prm, state_pool, slots, conv_weight, conv_pool)])
                return
            self._add_node("delta_rule_decode", (heads, T // requests_per_task, 1), block_dim, [proj, wfb, output],
                           [(-1, -1, -1), (0, -1, -1), (1, 0, -1)],
                           [PROJ_N, col_g, col_fa, col_beta, float_bits(onorm_eps), 1, HD],
                           changeable_grid_dims={1: (1, 1, 16)},
                           out_buffer=lambda n: (T * HD * 2, 0xFF, False), counters=1,
                           buf_slots=[self._tensor_slot(t) for t in (wfb, prm, state_pool, slots, conv_weight, conv_pool)])
            return
        p = producer_of(rec_in, graph_nodes(self))
        assert p is not None and p.name == "short_conv_decode", "delta_rule_decode_layer: the input is short_conv_decode's output"
        T, PROJ_N, HD = self._tokens(rec_in.dim(0)), rec_in.dim(1), wfb.dim(0)
        heads = HD // 128
        self._dtypes("delta_rule_decode_layer", rec_in=(rec_in, bfloat16), wfb=(wfb, bfloat16), prm=(prm, float32),
                     state_pool=(state_pool, float32), output=(output, bfloat16), slots=(slots, int32))
        assert HD % 128 == 0 and wfb.dim(1) == 128 and prm.dim(0) == heads + HD + 128 and p.params[0] == 3 * HD
        assert (output.dim(0), output.dim(1)) == (T, HD) and state_pool.dim(1) == 128 and state_pool.dim(0) % (heads * 128) == 0
        assert col_g + HD <= PROJ_N and col_fa + 128 <= PROJ_N and col_beta + heads <= PROJ_N
        assert 1 <= requests_per_task <= 16 and T % requests_per_task == 0, "delta_rule_decode_layer: R divides T, at most 16"
        if proj is not None:
            pp = producer_of(proj, graph_nodes(self))
            assert pp is not None and pp.name == "gemm_tile" and pp.params[0] == COMBINE["slots"] and proj.dim(1) == PROJ_N and \
                p.grid[0] * 64 == 3 * HD, "delta_rule_decode_layer: proj = the projection's K parts, the conv only q / k / v"
        if pairs_per_task > 0:
            assert proj is None and 1 <= pairs_per_task <= 4, "delta_rule_decode_layer pairs_per_task: 1 .. 4, no proj"
            self._add_node("delta_rule_decode", (-(-heads * T // pairs_per_task), 1, 1), block_dim, [rec_in, wfb, output],
                           [(-1, -1, -1), (0, -1, -1), (1, 0, -1)],
                           [PROJ_N, col_g, col_fa, col_beta, float_bits(onorm_eps), 2, pairs_per_task],
                           out_buffer=lambda n: (T * HD * 2, 0xFF, False), counters=1,
                           buf_slots=[self._tensor_slot(t) for t in (wfb, prm, state_pool, slots)])
            return
        self._add_node("delta_rule_decode", (heads, T // requests_per_task, 1), block_dim,
                       [rec_in] + ([proj] if proj is not None else []) + [wfb, output],
                       [(-1, -1, -1)] + ([(-1, -1, -1)] if proj is not None else []) + [(0, -1, -1), (1, 0, -1)],
                       [PROJ_N, col_g, col_fa, col_beta, float_bits(onorm_eps)],
                       changeable_grid_dims={1: (1, 1, 16)},
                       out_buffer=lambda n: (T * HD * 2, 0xFF, False), counters=1,
                       buf_slots=[self._tensor_slot(t) for t in (wfb, prm, state_pool, slots)])

    def attn_residual_layer(
        self,
        input: DTensor,         # the GPUs' partial sums: a push_all gemm_tile's output ([GPUS][T][H] bf16, exchange)
        bank: DTensor,          # [T, rows allocated, H] bf16: the earlier blocks' outputs (info[0] rows, strides info[1], [2])
        prefix: DTensor,        # [T, H] bf16
        q: DTensor, gain: DTensor, ow: DTensor,   # [H] bf16
        output: DTensor,        # hidden [T, H] bf16
        u_output: DTensor,      # [T, H] bf16: u = bf16(prefix + the summed partials), the next layer's prefix
        info: DTensor,          # int32 [8]
        eps: float = 1e-5,
        chunks: int = 0,        # tasks per token (0: H / 1024; HIP: 4 -> 1792-column chunks, 8 columns per thread)
        prefix_wait: tuple = None,   # HIP: (counter, need): prefix / bank rows written in this kernel (moe_agg1's count 1)
        block_dim: tuple = (256, 1, 1),
        rs: bool = False,       # HIP: the input a push_rs gemm_tile's output (each GPU its columns): a task per token on each GPU
                                # for its H / GPUS columns, the dots to every GPU and this GPU's u / out columns to every GPU through
                                # the node's exchange buffer, then the whole rows into output / u_output (attn_residual.cuh RS)
    ):
        """Task (x, y) = token x, columns [CW y, + CW), CW = H / chunks (SGLang ROCm's _agg_kernel + RMSNorm, mpk-kernels
        tests/kda_reference.py attn_residual): the GPUs' partials added in GPU order, a = bf16(that), u = bf16(prefix + a); for
        the bank rows and u: score = (c . src) / sqrt(mean(src^2) + eps), c = q gain (fp32); softmax; h = the weighted sum (fp32);
        out = bf16(h rsqrt(mean(h^2) + eps) ow). The H / 1024 tasks of a token run at the same time and add their partial sums
        (the bank rows' dots with c and pairwise dots before the partials land, then u's dots; h's sum of squares from those)
        through a scratch buffer and 2 T counters. params = [H, eps
        bits]; buf_slots = bank, prefix, q, gain, ow, u_output, info, the partials."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        assert p is not None and p.name == "gemm_tile" and p.params[0] in ((COMBINE["push_rs"], COMBINE["finish_rs"]) if rs else (COMBINE["push_all"],)), \
            "attn_residual_layer: the input is a push_all gemm_tile's output (rs: a push_rs one)"
        T, H = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("attn_residual_layer", input=(input, bfloat16), bank=(bank, bfloat16), prefix=(prefix, bfloat16),
                     q=(q, bfloat16), gain=(gain, bfloat16), ow=(ow, bfloat16), output=(output, bfloat16),
                     u_output=(u_output, bfloat16), info=(info, int32))
        assert bank.num_dims == 3 and bank.dim(2) == H and H % (8 * 256) == 0 or H % 8 == 0
        assert p.params[2] == H
        if rs:   # (attn_residual.cuh AR_PW = 170 floats per GPU's chunk; then u and out [T][H] bf16 each; 0xFF before each launch)
            GPUS = self.num_gpus
            assert H % (GPUS * 32) == 0 and H // GPUS // 4 <= 256, "attn_residual_layer rs: 4 columns per thread"
            from mirage.core import uint8
            xbuf = self.new_tensor(dims=(T, GPUS * 170 * 4 + 2 * H * 2), dtype=uint8, name="attn_residual.rs_exchange")
            self._add_node("attn_residual", (T, 1, 1), block_dim, [input, xbuf],
                           [(-1, -1, -1), (0, -1, -1)], [H, float_bits(eps)] + (list(prefix_wait) if prefix_wait else []),
                           counters=1, exchange=True,
                           out_buffer=lambda n: (T * GPUS * 170 * 4 + 2 * T * H * 2, 0xFF, True),
                           buf_slots=[self._tensor_slot(t) for t in (bank, prefix, q, gain, ow, u_output, info, output)])
            return
        NC = chunks or H // 1024
        CW = H // NC
        assert NC * CW == H and (NC == 1 and H % 1024 == 0 or CW == 1024 or (CW % 64 == 0 and CW // 8 <= 256 and 8 * (CW // 8) % 256 == 0)), \
            "attn_residual_layer: 1024-column chunks (4 per thread), up to 256 threads x 8 columns, or 1 (a task per token: TOK)"
        # (attn_residual.cuh AR_PW; 0xFF before each launch: the values are their own flags)
        part = self._scratch("attn_residual.parts", T * NC * (16 + 16 * 17 // 2 + 2 + 16) * 4, reset=0xFF)
        self._add_node("attn_residual", (T, NC, 1), block_dim, [input, output],
                       [(-1, -1, -1), (0, 1, -1)], [H, float_bits(eps)] + (list(prefix_wait) if prefix_wait else []),
                       concurrent_along_axis=1, counters=1 + 2 * T,
                       out_buffer=lambda n: (T * H * 2, -1, False),
                       buf_slots=[self._tensor_slot(t) for t in (bank, prefix, q, gain, ow, u_output, info)] + [part])

    # ---- the K3 MLA layer (HIP; tasks/mla_norm.cuh, mla_attn.cuh, mla_vgate.cuh; q_b: a "local" gemm_tile; o_proj and the
    # attention residual: KDA's). info (int32 [16]): [bank rows, bank token stride, bank row stride, seq_k (the request's KV
    # rows with this step's T), 0 x 4, the cache rows of this step's T tokens]
    def mla_norm_layer(
        self,
        latent: DTensor,        # [T, NA] fp32: the fused q_a / kv_a / k_pe projection, a gemm_tile node's K parts in slots
        w_q: DTensor,           # [QL] bf16: q_a_layernorm
        w_kv: DTensor,          # [KV] bf16: kv_a_layernorm
        kv_cache: DTensor,      # [rows, KV + RP] bf16: the framework's cache (this step's rows written at out_loc[t])
        out_loc: DTensor,       # int64 [>= T]: the rows of this step's tokens (SGLang's out_cache_loc)
        q_a: DTensor,           # [T, QL] bf16 (0xFF before each launch: q_b polls it)
        ql: int = 1536, kv: int = 512, rp: int = 64,
        eps_q: float = 1e-5, eps_kv: float = 1e-5,
        groups: int = 1,        # HIP: tasks per token (column ranges; their sums of squares exchanged through a scratch)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y) = token x, column range y of groups: the K parts added (bf16), q_a = rmsnorm(latent[:QL]) w_q -> q_a; the
        KV row [rmsnorm(latent[QL:QL + KV]) w_kv | latent[QL + KV:]] -> the cache. One count per task. params = [QL, KV, RP,
        eps_q, eps_kv bits]; buf_slots = w_q, w_kv, kv_cache, out_loc (, groups > 1: the partial sums [T][groups][2] fp32, 0xFF
        before each launch)."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(latent, graph_nodes(self))
        assert p is not None and p.name == "gemm_tile" and p.params[0] == COMBINE["slots"], "mla_norm_layer: K parts in slots"
        T = self._tokens(latent.dim(0))
        self._dtypes("mla_norm_layer", latent=(latent, float32), w_q=(w_q, bfloat16), w_kv=(w_kv, bfloat16),
                     kv_cache=(kv_cache, bfloat16), out_loc=(out_loc, int64), q_a=(q_a, bfloat16))
        assert latent.dim(1) >= ql + kv + rp and kv_cache.dim(1) == kv + rp and q_a.dim(1) == ql
        assert groups == 1 or _hip(), "mla_norm_layer: column groups on HIP only"
        slots = [self._tensor_slot(t) for t in (w_q, w_kv, kv_cache, out_loc)]
        if groups > 1:
            slots.append(self._scratch("mla_norm.psum", T * groups * 2 * 4, reset=0xFF))
        self._add_node("mla_norm", (T, groups, 1), block_dim, [latent, q_a], [(-1, -1, -1), (0, -1, -1)],
                       [ql, kv, rp, float_bits(eps_q), float_bits(eps_kv)],
                       out_buffer=lambda n: (T * ql * 2, 0xFF, False), counters=1,
                       concurrent_along_axis=1 if groups > 1 else None,
                       buf_slots=slots, cost_inputs=[0])

    def mla_absorb_layer(
        self,
        q: DTensor,             # [T, heads x (NOPE + RP)] bf16: q_b (a "local" gemm_tile's output, polled)
        w_kc: DTensor,          # [heads x KV, NOPE] bf16: kv_b_proj's k part per head, transposed ([h][l][n]: SGLang's memory)
        output: DTensor,        # the query [T, heads x (KV + RP)] bf16: q_abs, then q_pe per head (0xFF before each launch)
        heads: int = 12, nope: int = 128, rp: int = 64, kv: int = 512,
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y) = head x, latent rows [128 y, + 128): q_abs = bf16(q_nope . w_kc) by MFMA (w_kc in registers from the task's
        start); task y = 0 also copies q_pe. params = [heads, NOPE, RP, KV]; buf_slots = w_kc."""
        from .compiler import graph_nodes, producer_of
        pq = producer_of(q, graph_nodes(self))
        assert pq is not None and pq.name == "gemm_tile" and pq.params[0] == COMBINE["local"], "mla_absorb_layer: q from a local gemm"
        T = self._tokens(q.dim(0))
        self._dtypes("mla_absorb_layer", q=(q, bfloat16), w_kc=(w_kc, bfloat16), output=(output, bfloat16))
        assert q.dim(1) == heads * (nope + rp) and (w_kc.dim(0), w_kc.dim(1)) == (heads * kv, nope)
        assert output.dim(1) == heads * (kv + rp) and kv % 128 == 0
        self._add_node("mla_absorb", (heads, kv // 128, 1), block_dim, [q, output],
                       [(-1, -1, -1), (-1, -1, -1)], [heads, nope, rp, kv],
                       out_buffer=lambda n: (T * heads * (kv + rp) * 2, 0xFF, False),
                       buf_slots=[self._tensor_slot(w_kc)])

    def mla_attn_layer(
        self,
        query: DTensor,         # mla_absorb's output [T, heads x (KV + RP)] bf16 (polled)
        norm_out: DTensor,      # mla_norm's output (its counter: this step's KV rows written)
        kv_cache: DTensor,
        table: DTensor,         # int32 [requests, max rows]: SGLang's req_to_token (page size 1; info[3] = its row stride)
        info: DTensor,          # int32 [8]: [bank rows, bank strides (2), table row stride, 0 ...]
        seq_lens: DTensor,      # int64 [>= 1]: the request's rows before this step
        req_idx: DTensor,       # int64 [>= 1]: the request's row of the table (req_pool_indices)
        out_loc: DTensor,       # int64 [>= T]: this step's rows (out_cache_loc)
        output: DTensor,        # attn [T, heads x KV] bf16 (0xFF before each launch: mla_vgate polls it)
        heads: int = 12, rp: int = 64, kv: int = 512, page_size: int = 1, scale: float = 192 ** -0.5,
        kv_splits: int = 1, group: int = 1, block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y) = head x, KV split y: attention of the head's query over the request's KV rows (paged, causal within the
        step), online softmax in chunks of 64 rows (the first chunk's rows of earlier steps loaded as the task starts);
        kv_splits > 1: the chunks over that many splits per head (their partials in a scratch buffer, combined by split 0; one
        counter per head). group > 1 (HIP tasks/mla_attn_grp.cuh): task (x, y) = heads [group x, + group), the next chunk's rows
        in flight while one is computed, every split combines a share of its group's outputs (one counter per group; the
        plan must give each CU at most one of the node's tasks). params = [heads, KV, RP, PS, scale bits, group]; buf_slots =
        cache, table, info, seq_lens, req_idx, out_loc (, the partials)."""
        from .compiler import graph_nodes, producer_of
        pq, pn = producer_of(query, graph_nodes(self)), producer_of(norm_out, graph_nodes(self))
        assert pq is not None and pq.name == "mla_absorb", "mla_attn_layer: the query is mla_absorb's output"
        assert pn is not None and pn.name == "mla_norm", "mla_attn_layer: norm_out is mla_norm's output"
        T = self._tokens(query.dim(0))
        self._dtypes("mla_attn_layer", query=(query, bfloat16), kv_cache=(kv_cache, bfloat16),
                     table=(table, int32), info=(info, int32), seq_lens=(seq_lens, int64), req_idx=(req_idx, int64),
                     out_loc=(out_loc, int64), output=(output, bfloat16))
        assert query.dim(1) == heads * (kv + rp) and output.dim(1) == heads * kv
        assert group in (1, 2) and heads % group == 0, "mla_attn_layer: 1 or 2 heads per task (mla_attn_grp.cuh mla_g::HG)"
        assert group == 1 or _hip(), "mla_attn_layer: head groups on HIP only"
        slots = [self._tensor_slot(t) for t in (kv_cache, table, info, seq_lens, req_idx, out_loc)]
        if kv_splits > 1:
            slots.append(self._scratch("mla_attn.parts", heads * kv_splits * (T * kv + 2 * T) * 4))
        self._add_node("mla_attn", (heads // group, kv_splits, 1), block_dim, [query, norm_out, output],
                       [(-1, -1, -1), (-1, -1, -1), (1, -1, -1)], [heads, kv, rp, page_size, float_bits(scale), group],
                       concurrent_along_axis=1 if kv_splits > 1 else None, counters=heads // group if kv_splits > 1 else 0,
                       out_buffer=lambda n: (T * heads * kv * 2, 0xFF, False), buf_slots=slots)

    def mla_attn_decode_layer(
        self,
        query: DTensor,         # mla_absorb's output [T, heads x (KV + RP)] bf16 (polled)
        norm_out: DTensor,      # mla_norm's output (its counter: this step's KV rows written)
        kv_cache: DTensor,
        table: DTensor,         # int32 [requests, max rows]: SGLang's req_to_token (page size 1; info[3] = its row stride)
        info: DTensor,          # int32 [8]: [bank rows, bank strides (2), table row stride, 0 ...]
        seq_lens: DTensor,      # int64 [>= T]: request t's rows with this step's token
        req_idx: DTensor,       # int64 [>= T]: request t's row of the table (req_pool_indices)
        out_loc: DTensor,       # int64 [>= T]: request t's row of this step (out_cache_loc)
        output: DTensor,        # attn [T, heads x KV] bf16 (0xFF before each launch: mla_vgate polls it)
        heads: int = 12, rp: int = 64, kv: int = 512, page_size: int = 1, scale: float = 192 ** -0.5,
        kv_splits: int = 1, block_dim: tuple = (256, 1, 1),
    ):
        """Plain decode (tasks/mla_attn_decode.cuh): task (x, y) = request x, KV split y, every head of the request at once
        (the MFMA's columns): online softmax over the request's rows in chunks of 64; kv_splits > 1: partials in a scratch
        buffer, combined by split 0 (one counter per request). params = [heads, KV, RP, PS, scale bits]; buf_slots = cache,
        table, info, seq_lens, req_idx, out_loc (, the partials)."""
        from .compiler import graph_nodes, producer_of
        pq, pn = producer_of(query, graph_nodes(self)), producer_of(norm_out, graph_nodes(self))
        assert pq is not None and pq.name == "mla_absorb", "mla_attn_decode_layer: the query is mla_absorb's output"
        assert pn is not None and pn.name == "mla_norm", "mla_attn_decode_layer: norm_out is mla_norm's output"
        T = self._tokens(query.dim(0))
        self._dtypes("mla_attn_decode_layer", query=(query, bfloat16), kv_cache=(kv_cache, bfloat16),
                     table=(table, int32), info=(info, int32), seq_lens=(seq_lens, int64), req_idx=(req_idx, int64),
                     out_loc=(out_loc, int64), output=(output, bfloat16))
        assert query.dim(1) == heads * (kv + rp) and output.dim(1) == heads * kv and heads <= 16 and heads % 4 == 0
        slots = [self._tensor_slot(t) for t in (kv_cache, table, info, seq_lens, req_idx, out_loc)]
        # MPK_MLA_COMB_FLAG=1 (HIP): the partials their own flags (0xFF scratch, the combining split writes it back; no count)
        flag = kv_splits > 1 and os.environ.get("MPK_MLA_COMB_FLAG") == "1"
        # MPK_MLA_COMB_SPREAD=1 (HIP): every split writes its partial and counts; each combines its share of the output
        spread = kv_splits > 1 and not flag and os.environ.get("MPK_MLA_COMB_SPREAD") == "1"
        # MPK_MLA_COMB_VGATE=1 (HIP): every split writes its partial and counts; mla_vgate combines them for its head and tokens
        vgate = kv_splits > 1 and not flag and os.environ.get("MPK_MLA_COMB_VGATE") == "1"
        spread = spread and not vgate
        if kv_splits > 1:
            slots.append(self._scratch("mla_attn_decode.parts", T * kv_splits * (heads * kv + 2 * heads) * 4,
                                       **({"reset": 0xFF, "by_reader": True} if flag else {})))
        self._add_node("mla_attn_decode", (T, kv_splits, 1), block_dim, [query, norm_out, output],
                       [(-1, -1, -1), (-1, -1, -1), (0, -1, -1)], [heads, kv, rp, page_size, float_bits(scale)] + ([1] if flag else [2] if spread else [3] if vgate else []),
                       concurrent_along_axis=1 if kv_splits > 1 else None, counters=T if kv_splits > 1 else 0,
                       out_buffer=lambda n: (T * heads * kv * 2, 0xFF, False), buf_slots=slots)

    def mla_vgate_layer(
        self,
        attn: DTensor,          # mla_attn's (or mla_attn_decode's) output [T, heads x KV] bf16 (polled)
        gate_parts: DTensor,    # [T, heads x VD] fp32: g_proj, a gemm_tile node's K parts in slots (polled)
        w_vc: DTensor,          # [heads x VD, KV] bf16: kv_b_proj's v part per head, transposed ([h][vd][l]: SGLang's memory)
        output: DTensor,        # o_in [T, heads x VD] bf16 (0xFF before each launch: o_proj polls it)
        heads: int = 12, kv: int = 512, vd: int = 128, token_groups: int = 0,
        block_dim: tuple = (256, 1, 1),
        counted: bool = False,  # its tasks count (a counter): o_proj waits for the count and fills its activation by DMA
    ):
        """Task (x, y) = head x, tokens [T / G y, + T / G) (G = token_groups; 0: the fewest with at most 16 tokens each; the
        compiler may change it): v = attn . w_vc, o_in = bf16(v bf16(sigmoid(gate))). params = [heads, KV, VD]; buf_slots =
        w_vc."""
        from .compiler import graph_nodes, producer_of
        pa, pg = producer_of(attn, graph_nodes(self)), producer_of(gate_parts, graph_nodes(self))
        assert pa is not None and pa.name in ("mla_attn", "mla_attn_decode"), "mla_vgate_layer: attn is mla_attn's output"
        assert pg is not None and pg.name == "gemm_tile" and pg.params[0] == COMBINE["slots"], "mla_vgate_layer: g_proj in slots"
        T = self._tokens(attn.dim(0))
        self._dtypes("mla_vgate_layer", attn=(attn, bfloat16), gate_parts=(gate_parts, float32), w_vc=(w_vc, bfloat16),
                     output=(output, bfloat16))
        assert (w_vc.dim(0), w_vc.dim(1)) == (heads * vd, kv) and output.dim(1) == heads * vd and gate_parts.dim(1) == heads * vd
        G = token_groups or next(g for g in range(1, T + 1) if T % g == 0 and T // g <= 16)
        assert T % G == 0 and T // G <= 16, "mla_vgate_layer: at most 16 tokens per task"
        # (the attention in MPK_MLA_COMB_VGATE mode, params[5] = 3: this node combines its splits' partials - params[3] = its KV
        # splits, buffer slots 1, 2 = its seq_lens and partials)
        ra = self._static_nodes[pa.graph_idx]
        vc = pa.name == "mla_attn_decode" and len(pa.params) > 5 and pa.params[5] == 3
        extra_p = [pa.grid[1]] if vc else []
        extra_s = [ra["buf_slots"][3], ra["buf_slots"][6]] if vc else []
        self._add_node("mla_vgate", (heads, G, 1), block_dim, [attn, gate_parts, output],
                       [(-1, -1, -1), (-1, -1, -1), (1, 0, -1)], [heads, kv, vd] + extra_p,
                       changeable_grid_dims={1: (1, 1, 16)},
                       out_buffer=lambda n: (T * heads * vd * 2, 0xFF, False),
                       buf_slots=[self._tensor_slot(w_vc)] + extra_s, **({"counters": 1} if counted else {}))

    def residual_add_layer(
        self,
        addend: DTensor,        # S [T, H] bf16, sum_gpus's output buffer
        input: DTensor,         # o [T, H] bf16: this GPU's part, sent by sum_send (every GPU's part arrives through the exchange region),
                                # or a second sum_gpus node's output (GLM-5.3: the routed sum), or (HIP) latent_up's gemm_tile output
        residual: DTensor,      # [T, H] bf16 (a graph input)
        output: DTensor,        # y [T, H] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x of the SMs: slice x of y = bf16(bf16(input + addend) + residual). params = [H]."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("residual_add_layer", addend=(addend, bfloat16), input=(input, bfloat16), residual=(residual, bfloat16),
                     output=(output, bfloat16))
        assert (residual.dim(0), residual.dim(1)) == (T, H) and H % (8 * self.num_gpus) == 0
        from .compiler import graph_nodes, producer_of
        nodes = graph_nodes(self)
        a, o = producer_of(addend, nodes), producer_of(input, nodes)
        if a is None or a.name != "sum_gpus" or o is None or o.name not in ("sum_send", "sum_gpus", "gemm_tile"):
            raise ValueError("residual_add_layer: the addend from sum_gpus, the input from sum_send, sum_gpus or a gemm_tile "
                             "(HIP: latent_up's finish / push)")
        self._add_node("residual_add", (self._num_sms(), 1, 1), block_dim, [addend, input, residual, output],
                       [(-1, -1, -1)] * 4, [H], buf_slots=[self._tensor_slot(residual), self._tensor_slot(output)],
                       grid_choices=sorted({(n, 1, 1) for n in (32, 64, 128, self._num_sms())}))   # (slices by pointer)

    def moe_attn_residual_layer(
        self,
        addend: DTensor,        # S [T, H] bf16, sum_gpus's output buffer
        input: DTensor,         # o [T, H] bf16 (every GPU's part through the exchange region, sum_send's layout)
        residual: DTensor,      # [T, H] bf16 (a graph input)
        output: DTensor,        # y [T, H] bf16 (the MoE layer's output, the next layer's prefix)
        bank: DTensor,          # [T, rows allocated, H] bf16: the earlier blocks' outputs (info [0] rows, strides [1], [2])
        q: DTensor, gain: DTensor, ow: DTensor,   # [H] bf16: the next layer's self_attention_res_proj, _res_norm, input_layernorm
        hidden: DTensor,        # [T, H] bf16: the next layer's attention input (aggregation 1 + RMSNorm)
        info: DTensor,          # int32 [8]: bank rows, token stride, row stride, write (bank row info[0] = y: a block-write layer)
        eps: float = 1e-5,
        block_dim: tuple = (256, 1, 1),
    ):
        """residual_add with the next layer's aggregation 1 (tasks/moe_attn_residual.cuh): task (x, y) = token x, columns
        [1024 y, + 1024): y = bf16(bf16(input + addend) + residual), then attn_residual's aggregation of the bank rows and y
        (the 7 tasks of a token add their partial sums through a scratch buffer). params = [H, eps bits]."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("moe_attn_residual_layer", addend=(addend, bfloat16), input=(input, bfloat16),
                     residual=(residual, bfloat16), output=(output, bfloat16), bank=(bank, bfloat16), q=(q, bfloat16),
                     gain=(gain, bfloat16), ow=(ow, bfloat16), hidden=(hidden, bfloat16), info=(info, int32))
        assert (residual.dim(0), residual.dim(1)) == (T, H) and H % 1024 == 0 and H % (8 * self.num_gpus) == 0
        assert bank.num_dims == 3 and bank.dim(2) == H
        NC = H // 1024
        part = self._scratch("moe_attn_residual.parts", T * NC * (16 + 16 * 17 // 2 + 2 + 16) * 4, reset=0xFF)
        self._add_node("moe_attn_residual", (T, NC, 1), block_dim, [addend, input, residual, output],
                       [(-1, -1, -1)] * 4, [H, float_bits(eps)], concurrent_along_axis=1, counters=1 + 2 * T,
                       buf_slots=[self._tensor_slot(t) for t in (bank, residual, q, gain, ow, output, info)] + [part,
                                  self._tensor_slot(hidden)])

    def input_wait_layer(
        self,
        input: DTensor,         # [T, H] bf16 graph tensor, written by the kernel before this one (all-ones until written)
        output: DTensor,        # [T, H] bf16: the copy this kernel's tasks read (the node's buffer)
        rearm: bool = False,    # all-ones back into the input after the copy (the next launch polls it again)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = token x (tasks/input_wait.cuh): polls its input row until written, copies it, re-arms it; one count. For a
        layer kernel overlapped with the one before it (its readers wait for this node's count). params = [H, rearm]."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("input_wait_layer", input=(input, bfloat16), output=(output, bfloat16))
        assert (input.dim(0), input.dim(1)) == (T, H) and H % 8 == 0
        self._add_node("input_wait", (T, 1, 1), block_dim, [input, output], [(-1, -1, -1), (0, -1, -1)], [H, int(rearm)],
                       counters=1, out_buffer=lambda n: (T * H * 2, -1, False), buf_slots=[self._tensor_slot(input)])

    def moe_agg1_layer(
        self,
        addend: DTensor, input: DTensor, residual: DTensor, output: DTensor,   # as moe_attn_residual_layer
        bank: DTensor, q: DTensor, gain: DTensor, ow: DTensor, hidden: DTensor, info: DTensor,
        eps: float = 1e-5,
        posted: bool = False,   # HIP: hidden is the node's output, read by a later node of this kernel (a MoE + next attention
                                # kernel): device-scope stores, one counter (the T tasks), hidden in an output buffer
        attn_prefix: DTensor = None,   # (posted) [T, H] bf16: the next attention's prefix, y or zero after a bank row write
        bare: DTensor = None,   # (posted, with attn_prefix) [T, H] bf16: bf16(o + S), the MoE's output without the prefix
        mix: DTensor = None,    # (posted, with bare) [T, H] bf16: the pre-norm mixture h (SGLang's aggregate_stream, its dspark
                                # capture of the stream)
        block_dim: tuple = (256, 1, 1),
    ):
        """moe_attn_residual_layer's work with one task per token (tasks/moe_agg1.cuh): no exchange between CUs; at most 8
        bank rows. params = [H, eps bits]."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("moe_agg1_layer", addend=(addend, bfloat16), input=(input, bfloat16),
                     residual=(residual, bfloat16), output=(output, bfloat16), bank=(bank, bfloat16), q=(q, bfloat16),
                     gain=(gain, bfloat16), ow=(ow, bfloat16), hidden=(hidden, bfloat16), info=(info, int32))
        assert (residual.dim(0), residual.dim(1)) == (T, H) and H % 1024 == 0 and H % (8 * self.num_gpus) == 0
        assert bank.num_dims == 3 and bank.dim(2) == H
        slots = [self._tensor_slot(t) for t in (bank, residual, q, gain, ow, output, info, hidden)]
        if attn_prefix is not None:
            assert posted, "moe_agg1_layer: attn_prefix needs posted=True"
            slots.append(self._tensor_slot(attn_prefix))
        if bare is not None:
            assert attn_prefix is not None, "moe_agg1_layer: bare after attn_prefix"
            slots.append(self._tensor_slot(bare))
        if mix is not None:
            assert bare is not None, "moe_agg1_layer: mix after bare"
            self._dtypes("moe_agg1_layer", mix=(mix, bfloat16))
            slots.append(self._tensor_slot(mix))
        if posted:   # (the static task's output is its last tensor: hidden; y stays an input of the node)
            # (counter 0: hidden complete, the next attention's x; counter 1: y, the bank row, attn_prefix, bare too)
            self._add_node("moe_agg1", (T, 1, 1), block_dim, [addend, input, residual, output, hidden],
                           [(-1, -1, -1)] * 5, [H, float_bits(eps)], counters=2,
                           out_buffer=lambda n: (T * H * 2, -1, False), buf_slots=slots)
            return
        self._add_node("moe_agg1", (T, 1, 1), block_dim, [addend, input, residual, output],
                       [(-1, -1, -1)] * 4, [H, float_bits(eps)],
                       buf_slots=slots)

    # ---- the generated layer (static_schedule.py) ----
    def compile_static(self, schedule_paths, gpu_tensors=None, out_dir=None, extra_flags=None, profile=False, solo=False):
        """Build the graph as it is now, one schedule.json per GPU (static_schedule.compile_static); profile: the timing build
        (per-task start / end stamps, static_harness.Harness.task_times); solo: run GPU 0's share on one GPU (HIP only)."""
        from .static_schedule import compile_static
        return compile_static(self, schedule_paths, gpu_tensors, out_dir, extra_flags=extra_flags, profile=profile, solo=solo)
