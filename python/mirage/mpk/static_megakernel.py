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
import struct

import torch

from ..core import *
from ..kernel import TBGraph
from .persistent_kernel import PersistentKernel

# the kernel's limits (config.cuh MAX_BUFS, MAX_MAPS, NODE_COUNTER_LINES, SMAX = the ring's stages; core.cuh Combine)
MAX_BUFS, MAX_MAPS, NODE_COUNTER_LINES, RING_STAGES = 64, 32, 64, 5   # RING_STAGES: T <= 16's (ring_stages)


def ring_stages(tokens: int) -> int:
    """The GEMM weight ring's stages (config.cuh SMAX) at T tokens: 5, or 4 at T = 32 (the activation's T x 256 B a stage)."""
    return 5 if tokens <= 16 else 4
COMBINE = {"slots": 0, "add": 1, "store": 2, "store_bf16": 3}
SF_CHUNK = 512   # one MX scale chunk (config.cuh): 128 rows x 4 B
MAX_SUM_PARTS = 14   # the most K parts sum_send adds (tasks/sum_send.cuh)
HC_MIX_STRIDE = 32   # floats per (part, token) of hc_post_mix's mixes
# DSA constants (dsa_common.cuh, tasks/dsa_index_topk.cuh):
#   DSA_TOPK_BINS        the bins of a token histogram of the index scores (TOPK_BINS)
#   DSA_TOPK_REC         the ints of a chunk record of the top-k count node (TOPK_REC)
#   DSA_TOPK_CH          the positions of a top-k chunk (TOPK_CH)
#   DSA_INDEX_KEY_BYTES  the bytes of one key in the index cache: 128 e4m3 values and 4 scale bytes (index_page_bytes)
DSA_TOPK_BINS, DSA_TOPK_REC, DSA_TOPK_CH, DSA_INDEX_KEY_BYTES = 4096, 19, 1024, 132


def zq_bytes(tokens: int, width: int) -> int:
    """tasks/sum_quant_send.cuh zq_bytes: z_q [tokens][width] e4m3, then its scale chunks (one per 128 columns)."""
    return tokens * width + width // 128 * SF_CHUNK


def zq32_bytes(tokens: int, width: int) -> int:
    """tasks/group_quant.cuh zq32_bytes: z_q [tokens][width] e4m3, then its fp32 scales [width / 128][tokens]."""
    return tokens * width + (width // 128) * tokens * 4


def float_bits(v: float) -> int:
    """An fp32 value as the int of its bits (a param; the kernel template reads it with __int_as_float)."""
    return struct.unpack("<i", struct.pack("<f", v))[0]


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
        self._prefetch = []       # the weights kernel_begin prefetches into L2 (l2_prefetch_weights): (tensor, bytes)
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
                  cost_inputs: tuple = ()):
        """Add one node, as MPK's layers do (customized + register_task), and keep for compiler.py what the graph does not hold:
          input_maps              input_maps[i] = which dimension of tensors[i] each grid axis cuts (MPK's input_map,
                                  TBGraph.new_input: (1, -1, -1) = grid x cuts dim 1); a task's slices follow from the input
                                  maps and the grid, for any grid. The compiler gets the dependencies from these slices. A
                                  task must not wait at run time for a task outside them. If it does, a plan that
                                  passes check_deadlock_free can hang
          changeable_grid_dims    {grid axis: unit or (unit, min units, max units[, length])}: the grid axes the compiler may
                                  change; one task gets a whole number of units of each dimension that axis cuts (gemm_tile:
                                  {1: 128}, K in whole 128-column pieces), optionally between min and max units; length: what
                                  the axis splits when no input map uses it (sparse_mla over pool ids: its index slots); the
                                  other axes keep grid_dim
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
                                  it adds); their producers' grids are part of its cost key (search.cost_key), the others not"""
        from .static_schedule import THREADS
        if tuple(block_dim) != (THREADS, 1, 1):   # one kernel for every node: its CTAs have THREADS threads (the warp roles)
            raise ValueError(f"{name}: block_dim {tuple(block_dim)}; every task runs on the kernel's ({THREADS}, 1, 1) threads")
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        for t, m in zip(tensors, input_maps):
            tb_graph.new_input(t, m, -1, True)
        self.kn_graph.customized(tensors, tb_graph)
        self.kn_graph.register_task(tb_graph, name, params)
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
            assert self._counter_lines + lines <= NODE_COUNTER_LINES, f"more than {NODE_COUNTER_LINES} counter lines (config.cuh)"
            counter = 32 * self._counter_lines; self._counter_lines += lines
        self._static_nodes[idx] = dict(
            name=name, input_maps=list(input_maps), changeable_grid_dims=dict(changeable_grid_dims or {}),
            rows_split_over_gpus=rows_split_over_gpus, gpu_split_blocks=grid_dim[0], one_task_per_sm=one_task_per_sm,
            concurrent_along_axis=concurrent_along_axis, cluster_pair_params=cluster_pair_params,
            buf=buf, out_buffer=out_buffer, exchange=exchange, counter=counter, buf_slots=list(buf_slots), map_slots=list(map_slots),
            dry=dry,
            cost_inputs=list(cost_inputs))

    # ---- buffer slots (G::buf) and map slots (Maps::m) ----
    def _new_buf(self, entry: dict) -> int:
        assert len(self._bufs) < MAX_BUFS, f"more than {MAX_BUFS} buffer slots (config.cuh MAX_BUFS)"
        self._bufs.append(entry)
        return len(self._bufs) - 1

    def _scratch(self, name: str, nbytes, reset: int = -1) -> int:
        """A buffer only the node's tasks use (allocated by the host code; reset before each launch to the byte `reset`, -1: none),
        made by a layer method before its _add_node, which gives it the node. nbytes: an int, or a function of the node as compiled
        (0: no buffer). Returns its buffer slot."""
        size = nbytes if callable(nbytes) else (lambda n: nbytes)
        return self._new_buf({"kind": "scratch", "name": name, "fill": lambda n: (size(n), reset, False)})

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
        assert len(self._maps) < MAX_MAPS, f"more than {MAX_MAPS} tensor maps (config.cuh MAX_MAPS)"
        self._maps.append((kind, source, tuple(dims), key))
        return len(self._maps) - 1

    def _gemm_map(self, tensor: DTensor, box_rows: tuple) -> int:
        """The map slots of a bf16 [rows, K] tensor, one per box row count in box_rows (consecutive slots; the same tensor and
        boxes again: the same slots). Returns the first."""
        rows, K = tensor.dim(0), tensor.dim(1)
        for i in range(len(self._maps) - len(box_rows) + 1):
            if all(self._maps[i + j][3] == ("bf16", ("t", tensor.guid), (rows, K, b)) for j, b in enumerate(box_rows)):
                return i
        assert len(self._maps) + len(box_rows) <= MAX_MAPS, f"more than {MAX_MAPS} tensor maps (config.cuh MAX_MAPS)"
        first = len(self._maps)
        for b in box_rows:
            self._maps.append(("bf16", ("t", tensor), (rows, K, b), ("bf16", ("t", tensor.guid), (rows, K, b))))
        return first

    def exchange_layout(self):
        """The exchange region's layout, from the exchange buffers: the ones below 256 KB (smallest first, then by slot), the start
        barrier's slots, the larger ones (the same order); each 4096-aligned. Returns (one set's bytes, the start barrier's offset in
        a set, per buffer slot its offset, per buffer slot its bytes). Keep this order: the buffers' offsets change the layer's time
        (laid out in slot order instead, the same kernel is slower)."""
        from .compiler import graph_nodes
        nodes = {n.graph_idx: n for n in graph_nodes(self)}
        offsets, sizes, end = [0] * MAX_BUFS, [0] * MAX_BUFS, 0
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

    def l2_prefetch_weights(self, budget_mb: float = 64.0,
                            stop_at: tuple = ("kda_decode", "sparse_mla", "scaled_moe_experts")) -> int:
        """STATIC_L2_PREFETCH (core.cuh kernel_begin): the layer's first GEMMs' weights (gemm_tile's weight, scaled_gemm_tile's
        weight and its scales; graph inputs) in graph order, those before the first node of a kind in stop_at (the
        attention or expert core: the GEMMs after it run long after the start, and their weights in flight at the start
        would take bandwidth from the first GEMMs) and within budget_mb (the tensor that crosses it: its first bytes up to
        it, whole MB), prefetched into L2 by the CTAs that start before the previous kernel ends (PDL): they claim 1 MB
        batches while they wait at the start barrier, so the first GEMMs read L2 (in flight during the previous kernel's
        tail, they also slow it: a smaller budget trades the one for the other). Call once the graph is built. Returns the
        bytes listed."""
        from .compiler import graph_nodes, producer_of
        nodes = graph_nodes(self)
        size = {bfloat16: 2, float16: 2, float32: 4, int32: 4, int64: 8, uint8: 1}
        seen, total, self._prefetch = set(), 0, []
        for n in nodes:
            if n.name in stop_at:
                break
            ws = {"gemm_tile": n.inputs[1:2], "scaled_gemm_tile": n.inputs[1:3]}.get(n.name, [])
            for t in ws:
                if t.guid in seen or producer_of(t, nodes) is not None:
                    continue
                seen.add(t.guid)
                b = size[t.dtype]
                for d in range(t.num_dims):
                    b *= t.dim(d)
                if len(self._prefetch) == 8:   # core.cuh MAX_PF ranges
                    return total
                left = int(budget_mb * 2 ** 20) - total
                if b > left:   # the first whole MB of it that fit
                    if left >= 2 ** 20:
                        self._prefetch.append((t, left // 2 ** 20 * 2 ** 20))
                        total += left // 2 ** 20 * 2 ** 20
                    return total
                self._prefetch.append((t, b))
                total += b
        return total

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
        set_bytes = self.exchange_layout()[0]
        for slot, (kind, source, dims, _) in enumerate(self._maps):
            if source[0] == "t":
                src = f"t:{names[source[1].guid]}"
            elif source[0] == "b":
                src = f"b:{source[1]}+{source[2]}"
            else:   # an exchange buffer: its set 0 copy is buffer slot source[1]; set k at + k sets
                src = f"b:{source[1]}+{source[2] + source[3] * set_bytes}"
            args[f"map.{slot}"] = " ".join([kind, src] + [str(d) for d in dims])
        for i, (t, b) in enumerate(self._prefetch):   # STATIC_L2_PREFETCH: "pf.<i>" -> "t:<name> <bytes>"
            args[f"pf.{i}"] = f"t:{names[t.guid]} {b}"
        return args

    def gemm_tile_layer(
        self,
        input: DTensor,     # [T, K] bf16
        weight: DTensor,    # [N, K] bf16
        output: DTensor,    # [T, N] fp32 ("add": int64 fixed point; "store_bf16": bf16; its buffer holds what `combine` says)
        combine: str = "slots",   # how the K parts are combined (gemm_tile.cuh Combine): "slots" (one 0xFF-prefilled slot per
                                  # K part, the reader adds them), "add" (fixed-point atomic add, a counter per 128-row block),
                                  # "store" (plain stores, one K part, a counter of tasks), "store_bf16" (the same in bf16: a
                                  # layer's final output; the compiler may use 64-row tiles)
        rows_split_over_gpus: bool = False,   # each GPU computes its part of the weight's rows (latent_down, latent_up)
        input_polled: bool = False,   # the input is a node's 0xFF-prefilled bf16 buffer, polled per 16 B straight into the ring's
                                      # stages (each task starts on its columns as soon as they are written, without a counter)
        reader_rearms: bool = True,   # "slots": the one reader of each value re-arms it (REARM); False when several tasks read
                                      # the same values (kda_decode's f_a / g_a): the kernel or the host then refills the buffer
        block_dim: tuple = (256, 1, 1),
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
                     output=(output, {"add": int64, "store_bf16": bfloat16}.get(combine, float32)))
        if input_polled:
            from .compiler import graph_nodes, producer_of
            p = producer_of(input, graph_nodes(self))
            if p is None or p.info["buf"] < 0 or p.info["out_buffer"](p)[1] != 0xFF:
                raise ValueError(f"gemm_tile_layer: a polled input must be a node's 0xFF-prefilled output buffer; it comes from "
                                 f"{p.name if p else 'the graph inputs'}")
            assert combine == "slots", "polled: the K parts in slots"
        if combine == "slots" and input_polled:   # this GPU's rows only: [K parts][T][N / GPUS] fp32 (gemm_tile.cuh, run_gemm_tile)
            assert rows_split_over_gpus and N % (128 * GPUS) == 0, "polled: the rows split over the GPUs, whole 128-row blocks"
            out_buffer = lambda n: (n.grid[1] * T * (N // GPUS) * 4, 0xFF, True)
        elif combine == "slots":   # [K parts][T][N] fp32, 0xFF (the reader polls the slots and re-arms them)
            out_buffer = lambda n: (n.grid[1] * T * N * 4, 0xFF, reader_rearms)
        elif combine == "add":     # [T][N] int64 fixed point, 0 before each launch
            out_buffer = lambda n: (T * N * 8, 0, False)
        elif combine == "store_bf16":   # [T][N] bf16, every value written each launch
            out_buffer = lambda n: (T * N * 2, -1, False)
        else:                      # [T][N] fp32, every value written each launch
            out_buffer = lambda n: (T * N * 4, -1, False)
        assert combine != "add" or N // 128 <= 32, "one counter per 128-row block, at most 32"
        tiles = K // 128
        if input_polled:
            parts = next(y for y in range(1, tiles + 1) if tiles % y == 0 and tiles // y <= ring_stages(T))
            params, map_slots = [mode, K, N, 1], [self._gemm_map(weight, (128,)), -1]
            changeable = {1: (128, 2, ring_stages(T))}
        else:
            parts = 1
            params, map_slots = [mode, K, N, 0], [self._gemm_map(weight, (128, 64)), self._gemm_map(input, (T,))]
            changeable = {"store": {}, "store_bf16": {0: (64, 1, 2)}}.get(combine, {0: (64, 1, 2), 1: (128, 2, None)})
        self._add_node("gemm_tile", (N // 128, parts, 1), block_dim, [input, weight, output],
                       [(-1, 1, -1), (0, 1, -1), (1, -1, -1)], params,
                       # K: at least 2 tiles of 128 per task (the two MMA issuers take the stages in turn, runtime.cuh issue_job;
                       # with one stage issuer 1 has no MMA and its stale accumulator would be added)
                       changeable_grid_dims=changeable, rows_split_over_gpus=rows_split_over_gpus, out_buffer=out_buffer,
                       counters={"slots": 0, "add": N // 128, "store": 1, "store_bf16": 1}[combine], map_slots=map_slots)

    def scaled_gemm_tile_layer(
        self,
        input: DTensor,     # [T, K] e4m3 bytes (uint8): a group_quant node's output (then its fp32 scales [K / 128][T])
        weight: DTensor,    # [N, K] e4m3 bytes (uint8), in 128 x 128 pieces [N / 128][K / 128] (fp8_fp32scale_weights)
        weight_scales: DTensor,   # the checkpoint's fp32 block scales: [N / 128, K / 128], or [N / 256, K / 128, 2] (gate_up)
        output: DTensor,    # [T, N] fp32 ("add": int64 fixed point; "store_bf16": bf16; its buffer holds what `combine` says)
        combine: str = "add",   # as gemm_tile_layer: "slots", "add", "store" or "store_bf16" (a layer's final output)
        block_dim: tuple = (256, 1, 1),
    ):
        """Declare an FP8 GEMM with fp32 scales (tasks/scaled_gemm_tile.cuh).

        Task (x, y) computes 128 weight rows and K part y.
        Each piece has its own accumulator. The epilogue applies weight and activation scales before it adds the partial sums.
        Grid y divides K into parts of whole 128-column tiles. The "store" and "store_bf16" modes require one K part.
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
                     output=(output, {"add": int64, "store_bf16": bfloat16}.get(combine, float32)))
        assert output.num_dims == 2 and output.dim(0) == input.dim(0) and output.dim(1) == N and N % 128 == 0 and K % 128 == 0
        q = producer_of(input, graph_nodes(self))
        # The task reads the zq32 layout in the exchange region and waits for the counter of the producer. The producer is
        # a group_quant node, or a node with the same output (dsa_q_norm_quant).
        if q is None or not q.info["exchange"] or q.info["counter"] < 0 or q.info["out_buffer"](q)[0] != zq32_bytes(T, K):
            raise ValueError("scaled_gemm_tile_layer: the input must be the z_q output of a group_quant node (or of a node "
                             "with the same output: the zq32 layout in the exchange region, a counter of its tasks)")
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
                      "store": lambda n: (T * N * 4, -1, False),
                      "store_bf16": lambda n: (T * N * 2, -1, False)}[combine]
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
                       changeable_grid_dims={1: (128, 1, None)} if combine in ("slots", "add") else {},
                       out_buffer=out_buffer, counters={"slots": 0, "add": N // 128, "store": 1, "store_bf16": 1}[combine],
                       buf_slots=[self._tensor_slot(weight_scales)], map_slots=[wmap, amap])

    def topk_route_layer(
        self,
        input: DTensor,     # router logits [T, E] fp32
        bias: DTensor,      # score correction bias [E] fp32
        output: DTensor,    # routing pairs [T, K] int64
        block_dim: tuple = (256, 1, 1),
    ):
        """Task t = token t: adds the router's partial sums of row t, top K (output.dim(1)) of the E experts. params = [E, K]."""
        T, E, K = self._tokens(input.dim(0)), input.dim(1), output.dim(1)
        self._dtypes("topk_route_layer", input=(input, float32), bias=(bias, float32), output=(output, int64))
        assert input.num_dims == 2 and bias.num_dims == 1 and bias.dim(0) == E and output.num_dims == 2 and output.dim(0) == T
        self._add_node("topk_route", (output.dim(0), 1, 1), block_dim, [input, bias, output],
                       [(0, -1, -1), (-1, -1, -1), (0, -1, -1)], [E, K],
                       out_buffer=lambda n: (T * K * 8, 0xFF, False),   # the pairs, u64, polled by every moe_experts task
                       buf_slots=[self._tensor_slot(bias)], dry=True, cost_inputs=[0])   # adds the router's K parts

    def sum_quant_send_layer(
        self,
        input: DTensor,     # latent z [T, L] fp32
        output: DTensor,    # z_q [T, L] e4m3 bytes (uint8)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = columns [128 x, 128 x + 128) of z: adds latent_down's partial sums, MXFP8, sends to every GPU. Divided among
        the GPUs like latent_down's rows. Its output (z_q, then its scale chunks) is in the exchange region: every GPU writes its
        columns into every GPU's copy; 0xFF before the first launch, re-armed by the node after moe_experts (allreduce_send).
        params = [L] (z's width)."""
        T, L = self._tokens(input.dim(0)), input.dim(1)
        self._dtypes("sum_quant_send_layer", input=(input, float32), output=(output, uint8))
        assert input.num_dims == 2 and output.num_dims == 2 and (output.dim(0), output.dim(1)) == (T, L) and L % 128 == 0
        self._add_node("sum_quant_send", (output.dim(1) // 128, 1, 1), block_dim, [input, output],
                       [(1, -1, -1), (1, -1, -1)], [L], rows_split_over_gpus=True,
                       out_buffer=lambda n: (zq_bytes(T, L), 0xFF, True), exchange=True, cost_inputs=[0])   # adds latent_down's K parts

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

    def moe_experts_layer(
        self,
        z_q: DTensor, pairs: DTensor,        # z_q uint8 (sum_quant_send's output), the routing pairs int64 (topk_route's)
        w13: DTensor, w13_scales: DTensor,   # MXFP4 weights and their scales, uint8 (the scales read through the tensor maps only)
        w2: DTensor, w2_scales: DTensor,
        output: DTensor,    # R [T x K, L] fp32: this GPU's routed rows, one per (token, routed expert k); allreduce_send adds a
                            # token's K in order (shared down is its own gemm_tile node)
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
        assert (w2_scales.dim(0), w2_scales.dim(1), w2_scales.dim(2)) == (E, L, IR // 32)
        assert (output.dim(0), output.dim(1)) == (T * K, L)
        from .compiler import graph_nodes, producer_of
        zq = producer_of(z_q, graph_nodes(self))
        if zq is None or not zq.info["exchange"]:
            raise ValueError("moe_experts_layer: z_q must be a node's output in the exchange region (sum_quant_send_layer)")
        route = producer_of(pairs, graph_nodes(self))
        if route is None or route.name != "topk_route" or list(route.params) != [E, K]:
            raise ValueError(f"moe_experts_layer: the pairs must be a topk_route node's over the {E} experts of w13")
        nbytes = lambda t: t.dim(0) * t.dim(1) * t.dim(2)
        hq = self._scratch("moe_experts.h_q", NSLOT * T * IR)                        # every row written before its counter lets W2 read it
        hsf = self._scratch("moe_experts.h_q_scales", NSLOT * KT2 * SF_CHUNK, 0)     # the scale chunks' unused bytes stay 0
        w13_pieces, w2_pieces = nbytes(w13) // 8192, nbytes(w2) // 8192              # MXFP4: 128 x 128 pieces of 8 KB, scales 512 B each
        maps = []
        for weight, scales, pieces in ((w13, w13_scales, w13_pieces), (w2, w2_scales, w2_pieces)):
            maps += [self._map("wblk", ("t", weight), (pieces, 1)), self._map("wblk", ("t", weight), (pieces, 2)),
                      self._map("sf", ("t", scales), (pieces, 1)), self._map("sf", ("t", scales), (pieces, 2))]
        for kind, offset, dims in (("act8", 0, (L, T)), ("act8kt", 0, (L, T, 2)), ("sf", T * L, (KT_L, KT_L))):
            first = self._map(kind, ("x", zq.info["buf"], offset, 0), dims)      # z_q in set 0, then set 1 (the next slot)
            assert self._map(kind, ("x", zq.info["buf"], offset, 1), dims) == first + 1
            maps.append(first)
        maps += [self._map("act8kt", ("b", hq, 0), (IR, NSLOT * T, KT2)), self._map("sf", ("b", hsf, 0), (NSLOT * KT2, KT2))]
        tensors = [z_q, pairs, w13, w2, output]
        self._add_node("moe_experts", (self._num_sms(), 1, 1), block_dim, tensors, [(-1, -1, -1)] * len(tensors),
                       [IR], one_task_per_sm=True, out_buffer=lambda n: (T * K * L * 4, -1, False), counters=32 * 2 + NSLOT,
                       buf_slots=[hq, hsf], map_slots=maps)

    def allreduce_send_layer(
        self,
        input: DTensor,         # fp32 [T x G, W], a node's output buffer: G rows per token (added in order), e.g. this GPU's routed
                                # rows R [T x 16, LAT] (moe_experts, scaled_moe_experts) or S [T, H] (the shared down node)
        output: DTensor,        # what is sent: [T, W] bf16 (its buffer: [num_gpus][T][W] bf16 in the exchange region)
        num_tasks: int = None,  # slices of the partial sum; default: one per task of the producer (as its grid is now)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x: slice x of this GPU's partial sum (bf16) to every GPU, into this GPU's slot of the node's output (in the exchange
        region), once the input's producer is done (all its tasks, counted in its counter). params = [W, G, FX]. When the producer is
        moe_experts or scaled_moe_experts, re-arm z_q here only if that producer is its sole reader.
        buf_slots = [z_q's buffer slot] enables re-arm; [-1] disables it. With other readers, reset z_q before each launch instead.
        Declare all other z_q readers before this call so the reader check sees them."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        # a gemm_tile node's K parts in slots (no counter: each part is polled, tasks/allreduce_send.cuh)
        # (polled: rows of this GPU only, the same layout with one GPU)
        slots_src = (p is not None and p.name == "gemm_tile" and p.params[0] == COMBINE["slots"] and
                     (p.params[3] == 0 or self.num_gpus == 1))
        if p is None or p.info["buf"] < 0 or (p.info["counter"] < 0 and not slots_src):
            raise ValueError(f"allreduce_send_layer: the input must be a node's output buffer whose node counts its tasks, or a "
                             f"gemm_tile node's output in slots; it comes from {p.name if p else 'the graph inputs'}")
        T, W, GPUS = self._tokens(output.dim(0)), output.dim(1), self.num_gpus
        fx = input.dtype == int64   # a COMBINE_ADD GEMM's sums (its K parts): a task per 128-column block, after its parts
        if fx:
            if p.name not in ("gemm_tile", "scaled_gemm_tile") or p.params[0] != 1:
                raise ValueError("allreduce_send_layer: an int64 input is a COMBINE_ADD GEMM's output")
            assert W % 128 == 0 and input.dim(0) == T
            self._dtypes("allreduce_send_layer", output=(output, bfloat16))
            num_tasks = W // 128
        else:
            self._dtypes("allreduce_send_layer", input=(input, float32), output=(output, bfloat16))
        assert input.dim(0) % T == 0 and input.dim(1) == W and W % 8 == 0
        assert not slots_src or input.dim(0) == T, "allreduce_send_layer: a gemm_tile input has one row per token"
        rearm = -1
        if p.name in ("moe_experts", "scaled_moe_experts"):   # z_q: its first input; another reader (scaled_gemm_tile) can still read it
            zq = p.inputs[0]
            readers = [n for n in graph_nodes(self) if any(t.guid == zq.guid for t in n.inputs)]
            if len(readers) == 1:
                rearm = producer_of(zq, graph_nodes(self)).info["buf"]
        self._add_node("allreduce_send", (num_tasks or p.grid[0] * p.grid[1] * p.grid[2], 1, 1), block_dim, [input, output],
                       [(-1, -1, -1)] * 2, [W, input.dim(0) // T, int(fx)],
                       out_buffer=lambda n: (GPUS * T * W * 2, 0xFF, True), exchange=True, buf_slots=[rearm],
                       cost_inputs=[0] if slots_src else [])   # slots: adds the GEMM's K parts

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

    def residual_add_layer(
        self,
        addend: DTensor,        # S [T, H] bf16, sum_gpus's output buffer
        *,                      # the other arguments by keyword
        input: DTensor = None,  # o [T, H] bf16: this GPU's part, sent by sum_send (every GPU's part arrives through the exchange region),
                                # or a second sum_gpus node's output (GLM-5.3: the routed sum). None: one sum, as in an attention layer
        residual: DTensor,      # [T, H] bf16 (a graph input)
        output: DTensor,        # y [T, H] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x of the SMs: slice x of y = bf16(bf16(input + addend) + residual), or y = bf16(addend + residual) with no
        input. The residual is a graph input. params = [H]. buf_slots: [residual, output]."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        self._dtypes("residual_add_layer", addend=(addend, bfloat16), residual=(residual, bfloat16), output=(output, bfloat16),
                     **({"input": (input, bfloat16)} if input is not None else {}))
        assert (residual.dim(0), residual.dim(1)) == (T, H) and H % (8 * self.num_gpus) == 0
        from .compiler import graph_nodes, producer_of
        nodes = graph_nodes(self)
        a = producer_of(addend, nodes)
        if a is None or a.name != "sum_gpus":
            raise ValueError("residual_add_layer: the addend from sum_gpus")
        if input is not None:
            o = producer_of(input, nodes)
            if o is None or o.name not in ("sum_send", "sum_gpus"):
                raise ValueError("residual_add_layer: the input from sum_send or sum_gpus")
        if producer_of(residual, nodes) is not None:
            raise ValueError("residual_add_layer: the residual must be a graph input")
        tensors = [addend, residual, output] if input is None else [addend, input, residual, output]
        self._add_node("residual_add", (self._num_sms(), 1, 1), block_dim, tensors, [(-1, -1, -1)] * len(tensors), [H],
                       buf_slots=[self._tensor_slot(residual), self._tensor_slot(output)])

    def _producer(self, layer: str, tensor: DTensor, names: tuple, allow_none: bool = False):
        """Return the node that writes `tensor`. Raise ValueError if the name of that node is not in `names`. With
        allow_none, a graph input (no node writes it) gives None instead of an error."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(tensor, graph_nodes(self))
        if (p is None and not allow_none) or (p is not None and p.name not in names):
            raise ValueError(f"{layer}: the input must be the output of a {' or '.join(names)} node"
                             + (", or a graph input" if allow_none else ""))
        return p

    def _fx_producer(self, layer: str, tensor: DTensor):
        """Return the COMBINE_ADD GEMM node (gemm_tile or scaled_gemm_tile) that writes `tensor` (int64 fixed point [T, N])."""
        p = self._producer(layer, tensor, ("gemm_tile", "scaled_gemm_tile"))
        if p.params[0] != COMBINE["add"]:
            raise ValueError(f"{layer}: the input must be the output of a GEMM node with combine='add'")
        return p

    def _scratch_slot(self, node, name: str) -> int:
        """Return the buffer slot of the scratch buffer `name` of `node` (_scratch)."""
        for i, b in enumerate(self._bufs):
            if b["kind"] == "scratch" and b["name"] == name and b.get("node") == node.graph_idx:
                return i
        raise ValueError(f"node {node.graph_idx} ({node.name}) has no scratch buffer {name}")

    def hc_pre_layer(
        self,
        mix: DTensor,           # an hc_post_mix node's mixes
        residual: DTensor,      # [T, HC H] bf16, the same streams
        hc_small: DTensor,      # fp32 [>= NMIX + 3]: hc_*_base [NMIX], then hc_*_scale [3]
        gamma: DTensor,         # [H] bf16: the sublayer's input RMSNorm weight
        output: DTensor,        # x [T, H] bf16: the sublayer's input
        export: tuple,          # (h_post [T, HC], h_res [T, HC HC]) fp32 graph tensors: post and comb finished into them
        rms_eps: float = 1e-5, hc_eps: float = 1e-6, norm_eps: float = 1e-5, post_mult: float = 2.0,
        sinkhorn_iters: int = 20,
        block_dim: tuple = (256, 1, 1),
    ):
        """Tasks (t, 0) and (t, 1) of token t: pre / post / comb from the mixes, x = RMSNorm(sum_i pre_i R_i) gamma (SGLang's
        fused norm) by task (t, 0) into its buffer (a graph tensor output binds it), post and comb (Sinkhorn) finished by
        task (t, 1) into export's two graph tensors (SGLang's MHCState h_post, h_res: the sublayer's post runs after the
        layer, in a serving framework); its counter counts its 2 T tasks. Streams from hc_post_mix in this layer: the tasks
        wait for all of its tasks."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(mix, graph_nodes(self))
        if p is None or p.name != "hc_post_mix":
            raise ValueError("hc_pre_layer: mix must be an hc_post_mix node's mixes")
        HC, H, NMIX = p.params
        T = self._tokens(output.dim(0))
        assert hc_small.dim(0) >= NMIX + 3
        self._dtypes("hc_pre_layer", mix=(mix, float32), residual=(residual, bfloat16), hc_small=(hc_small, float32),
                     gamma=(gamma, bfloat16), output=(output, bfloat16))
        h_post, h_res = export
        self._dtypes("hc_pre_layer", h_post=(h_post, float32), h_res=(h_res, float32))
        assert (h_post.dim(0), h_post.dim(1), h_res.dim(0), h_res.dim(1)) == (T, HC, T, HC * HC)
        slots = [self._tensor_slot(t) for t in (residual, hc_small, gamma, h_post, h_res)]
        params = [H, float_bits(rms_eps), float_bits(hc_eps), float_bits(norm_eps), float_bits(post_mult), sinkhorn_iters]
        self._add_node("hc_pre", (T, 2, 1), block_dim, [mix, residual, output], [(0, -1, -1), (0, -1, -1), (0, -1, -1)],
                       params, out_buffer=lambda n: (T * H * 2, -1, False), counters=1,
                       cost_inputs=[0],   # adds the mixes' K parts
                       buf_slots=slots)

    def hc_post_layer(
        self,
        sent: DTensor,          # [T, H] bf16: an allreduce_send node's output (the sublayer output's partial sums)
        residual: DTensor,      # [T, HC H] bf16: the sublayer's input streams
        output: DTensor,        # [T, HC H] bf16: the new streams
        coef: tuple,            # (h_post [T, HC], h_res [T, HC HC]) fp32 graph tensors, post and comb finished (the sublayer's
                                # pre ran before the layer: SGLang's MHCState)
        parts: int = 4,         # tasks per token, each H / parts columns
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (q, t): columns [q C, q C + C) of token t: out = sum over the GPUs, R'_j = sum_i comb[i][j] R_i + post_j out.
        Counters: one per token, its Q tasks (an hc_pre of the next sublayer in this layer waits on them)."""
        from .compiler import graph_nodes, producer_of
        nodes = graph_nodes(self)
        s = producer_of(sent, nodes)
        if s is None or s.name != "allreduce_send":
            raise ValueError("hc_post_layer: sent from an allreduce_send node")
        T, H = self._tokens(sent.dim(0)), sent.dim(1)
        HC = residual.dim(1) // H
        self._dtypes("hc_post_layer", sent=(sent, bfloat16), residual=(residual, bfloat16), output=(output, bfloat16))
        h_post, h_res = coef
        self._dtypes("hc_post_layer", h_post=(h_post, float32), h_res=(h_res, float32))
        assert (h_post.dim(0), h_post.dim(1), h_res.dim(0), h_res.dim(1)) == (T, HC, T, HC * HC)
        slots = [self._tensor_slot(t) for t in (residual, output, h_post, h_res)]
        self._add_node("hc_post", (parts, T, 1), block_dim, [sent, residual, h_res, output],
                       [(1, 0, -1), (-1, 0, -1), (-1, 0, -1), (-1, 0, -1)], [H, HC], counters=T, buf_slots=slots)

    def hc_post_mix_layer(
        self,
        sent: DTensor,          # [T, H] bf16: an allreduce_send node's output (the sublayer output's partial sums)
        residual: DTensor,      # [T, HC H] bf16: the sublayer's input streams
        coef: tuple,            # (h_post [T, HC], h_res [T, HC HC]) fp32 graph tensors: the sublayer's, finished
        fn: DTensor,            # [NMIX, HC H] fp32: the next sublayer's hc_*_fn
        output: DTensor,        # [T, HC H] bf16: the new streams R'
        mix: DTensor,           # [T, HC_MIX_STRIDE] fp32 (a stand-in: the buffer holds [H / 32][T][HC_MIX_STRIDE]): the
                                # partial mixes for hc_pre
        block_dim: tuple = (256, 1, 1),
    ):
        """hc_post (MODE 1) and the next sublayer's partial mixes in one task type (tasks/hc_post_mix.cuh): task s = columns
        [32 s, 32 s + 32) of every stream and token: the sum over the GPUs, R' of those columns, their partial mixes with fn
        (hc_pre adds them). params = [HC, H, NMIX]. Counter: one, every task counts
        there."""
        from .compiler import graph_nodes, producer_of
        s = producer_of(sent, graph_nodes(self))
        if s is None or s.name != "allreduce_send":
            raise ValueError("hc_post_mix_layer: sent from an allreduce_send node")
        T, H, NMIX = self._tokens(sent.dim(0)), sent.dim(1), fn.dim(0)
        HC = residual.dim(1) // H
        h_post, h_res = coef
        assert NMIX == HC * (HC + 2) and fn.dim(1) == HC * H and H % 32 == 0
        assert (h_post.dim(0), h_post.dim(1), h_res.dim(0), h_res.dim(1)) == (T, HC, T, HC * HC)
        self._dtypes("hc_post_mix_layer", sent=(sent, bfloat16), residual=(residual, bfloat16), h_post=(h_post, float32),
                     h_res=(h_res, float32), fn=(fn, float32), output=(output, bfloat16), mix=(mix, float32))
        self._add_node("hc_post_mix", (H // 32, 1, 1), block_dim, [sent, residual, h_res, fn, mix, output],
                       [(1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)], [HC, H, NMIX],
                       out_buffer=lambda n: (n.grid[0] * T * HC_MIX_STRIDE * 4, 0xFF, False), counters=1,
                       buf_slots=[self._tensor_slot(residual), self._tensor_slot(output), self._tensor_slot(h_post),
                                  self._tensor_slot(h_res), self._tensor_slot(fn)])

    # ---- GLM-5.3-Flash KDA (tasks/kda_decode.cuh): a KDA sublayer is in_proj (a gemm_tile in slots) -> kda_decode ->
    # o_proj -> allreduce_send -> hc_post_mix -> hc_pre (the next sublayer's input).
    def kda_decode_layer(
        self,
        proj: DTensor,          # [T, NPROJ] fp32: the merged projection, a gemm_tile node's output in slots; columns
                                # [q HL D | k HL D | v HL D | b HL | f_a D | g_a D | pad]
        conv_weight: DTensor,   # [3 HL D, 4] fp32: the q | k | v short-conv weights of this GPU's heads
        f_b: DTensor,           # [HL D, D] bf16
        g_b: DTensor,           # [HL D, D] bf16
        small: DTensor,         # fp32 [>= HL D + HL + D]: dt_bias [HL D], A_log [HL], o_norm weight [D]
        conv_state: DTensor,    # [NR, 3 HL D 3] bf16: per state row, per channel the last 3 inputs (oldest first)
        rec_state: DTensor,     # [NR, HL D D] fp32: per state row the [HL][V][K] recurrent state
        rows: DTensor,          # [T] int32: the state row of each token's request
        output: DTensor,        # o [T, HL D] bf16 (after the gated o_norm)
        lower_bound: float = -5.0, norm_eps: float = 1e-5,
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (t, h): token t, local head h: short conv, gates, one delta-rule step on the state, gated RMSNorm
        (tasks/kda_decode.cuh). The states are updated in place. params = [HL, D, lower bound, o_norm eps]."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(proj, graph_nodes(self))
        if p is None or p.name != "gemm_tile" or p.params[0] != COMBINE["slots"] or p.params[3] != 0:
            raise ValueError("kda_decode_layer: proj must be a gemm_tile node's output in slots")
        T, HLD, D = self._tokens(output.dim(0)), output.dim(1), f_b.dim(1)
        HL = HLD // D
        # the kernel indexes these by the head count and head dim: other shapes would be read silently wrong
        assert (conv_weight.dim(0), conv_weight.dim(1)) == (3 * HLD, 4)
        assert (f_b.dim(0), g_b.dim(0), g_b.dim(1)) == (HLD, HLD, D)
        assert small.dim(0) >= HLD + HL + D
        assert conv_state.dim(1) == 3 * HLD * 3 and rec_state.dim(1) == HLD * D
        self._dtypes("kda_decode_layer", proj=(proj, float32), conv_weight=(conv_weight, float32), f_b=(f_b, bfloat16),
                     g_b=(g_b, bfloat16), small=(small, float32), conv_state=(conv_state, bfloat16),
                     rec_state=(rec_state, float32), rows=(rows, int32), output=(output, bfloat16))
        self._add_node("kda_decode", (T, HL, 1), block_dim, [proj, output], [(0, -1, -1), (0, 1, -1)],
                       [HL, D, float_bits(lower_bound), float_bits(norm_eps)],
                       out_buffer=lambda n: (T * HLD * 2, -1, False), counters=1,   # o; the o_proj GEMM waits for all tasks
                       cost_inputs=[0],   # adds the projection's K parts
                       buf_slots=[self._tensor_slot(t) for t in (conv_weight, f_b, g_b, small, conv_state, rec_state, rows)])

    def compile_static(self, schedule_paths, gpu_tensors=None, out_dir=None, extra_flags=None, profile=False, device=0,
                       rank=None, exchange=None):
        """Build the graph as it is now, one schedule.json per GPU (static_schedule.compile_static); profile: the timing build
        (per-task start / end stamps, static_harness.Harness.task_times); device: the CUDA device of GPU 0. rank / exchange:
        GPU `rank` alone in this process with the caller's exchange region (multicast address, local address), e.g. one
        TP rank of a serving framework (exchange_region_bytes)."""
        from .static_schedule import compile_static
        return compile_static(self, schedule_paths, gpu_tensors, out_dir, extra_flags=extra_flags, profile=profile,
                              device=device, rank=rank, exchange=exchange)

    def exchange_region_bytes(self) -> int:
        """The exchange region a rank-alone build needs from its caller (compile_static rank / exchange): both sets."""
        return 2 * self.exchange_layout()[0]
