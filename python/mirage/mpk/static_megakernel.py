"""StaticMegakernel: a PersistentKernel whose graph is built into one generated kernel with a static per-SM task schedule
(compiler.py -> static_schedule.py), instead of MPK's runtime.

Its layers (one graph node each, as MPK's layers) keep what compiler.py and the code generator need about the node: its slices,
its changeable grid axes, and what its task type's kernel template reads and writes, declared here, inside the layer, not by the
caller: its output buffer (in this GPU's memory, or in the cross-GPU exchange region), scratch buffers, graph tensors it reads by
pointer, tensor maps, counters. Each gets a slot; the generated host code (static_megakernel/host.cuh) allocates or binds them,
the task function reads them by slot (its StaticNode and StaticSlots template arguments). The kernel templates are in
include/mirage/static_megakernel/: core.cuh (the layer state, the kernel's start and end), host.cuh, tasks/<name>.cuh (one task
type each: its body and run_<name>). Every layer takes `globals` (the G struct) and, for the GEMM-side layers, `maps` (the
tensor maps) as extra inputs; in the built layer both are kernel parameters. Each layer adds its node with a default grid (no K
split); compiler.compile_plan regrids to the plan. The sizes come from the graph: each layer reads them from its tensors and puts
those its kernel template needs in the node's params; the kernel-wide ones (tokens per step, GPUs) are the StaticMegakernel's
(tokens, num_gpus), which the generated layer defines (config.cuh).
"""
import struct

import torch

from ..core import *
from ..kernel import TBGraph
from .persistent_kernel import PersistentKernel

# the slots the kernel has (config.cuh MAX_BUFS, MAX_MAPS, NODE_COUNTER_LINES, SMAX = the ring's stages; core.cuh Combine)
MAX_BUFS, MAX_MAPS, NODE_COUNTER_LINES, RING_STAGES = 24, 32, 16, 5
COMBINE = {"slots": 0, "add": 1, "store": 2}
SF_CHUNK = 512   # one MX scale chunk (config.cuh): 128 rows x 4 B


def zq_bytes(tokens: int, width: int) -> int:
    """tasks/sum_quant_send.cuh zq_bytes: z_q [tokens][width] e4m3, then its scale chunks (one per 128 columns)."""
    return tokens * width + width // 128 * SF_CHUNK


def float_bits(v: float) -> int:
    """An fp32 value as the int of its bits (a param; the kernel template reads it with __int_as_float)."""
    return struct.unpack("<i", struct.pack("<f", v))[0]


class StaticMegakernel(PersistentKernel):

    def __init__(self, *args, num_gpus: int = 1, **kwargs):
        """num_gpus: the GPUs the layer is divided over (a rows-split node's GPUs, an exchange buffer's slots; config.cuh GPUS)."""
        super().__init__(*args, **kwargs)
        self.num_gpus = num_gpus
        self.tokens = None        # tokens per step: the first dimension of the layers' token-major tensors (_tokens; config.cuh T)
        self._static_nodes = {}   # graph index -> what compiler.py needs about a node (_add_node)
        self._bufs = []           # G::buf slot -> {"kind": out | exchange | scratch | tensor, ...} (_add_node, _scratch, _tensor_slot)
        self._maps = []           # Maps::m slot -> (kind, source, dims) (_map)
        self._counter_lines = 0   # counter lines given out (_add_node)

    def _num_sms(self):
        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    def _tokens(self, n: int) -> int:
        """Tokens per step: the first layer's tensors set it, every other layer's must agree. Returns it."""
        if self.tokens is None:
            self.tokens = n
        if n != self.tokens:
            raise ValueError(f"a layer's tensors have {n} tokens, the graph's earlier layers {self.tokens}")
        return n

    def _add_node(self, name: str, grid_dim: tuple, block_dim: tuple, tensors: list, maps: list, params: list,
                  changeable_grid_dims: dict = None, rows_split_over_gpus: bool = False, one_task_per_sm: bool = False,
                  concurrent_along_axis: int = None, cluster_pair_params: list = None, out_buffer=None, exchange: bool = False,
                  counters: int = 0, slots: tuple = (), dry: bool = False, cost_inputs: tuple = ()):
        """Add one node, as MPK's layers do (customized + register_task), and keep for compiler.py what the graph does not hold:
          maps                    maps[i] = which dimension of tensors[i] each grid axis cuts (MPK's convention: (1, -1, -1) =
                                  grid x cuts dim 1); a task's slices follow from the maps and the grid, for any grid
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
          slots                   the task function's own slots (StaticSlots), in the order its kernel template reads them:
                                  buffer slots (_scratch, _tensor_slot, another node's buffer) and map slots (_map)
          dry                     the task type has a dry pass (run_<name> takes `warm`; static_schedule.case_code)
          cost_inputs             the inputs whose producer's grid changes the task's time (topk_route: the router's K parts, which
                                  it adds); their producers' grids are part of its cost key (search.cost_key), the others not"""
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        for t, m in zip(tensors, maps):
            tb_graph.new_input(t, m, -1, True)
        self.kn_graph.customized(tensors, tb_graph)
        self.kn_graph.register_task(tb_graph, name, params)
        idx = self.kn_graph.get_num_operators() - 1
        buf = -1
        if out_buffer is not None:
            buf = self._new_buf({"kind": "exchange" if exchange else "out", "node": idx, "fill": out_buffer})
        counter = -1
        if counters:
            lines = (counters + 31) // 32
            assert self._counter_lines + lines <= NODE_COUNTER_LINES, f"more than {NODE_COUNTER_LINES} counter lines (config.cuh)"
            counter = 32 * self._counter_lines; self._counter_lines += lines
        self._static_nodes[idx] = dict(
            name=name, maps=list(maps), changeable_grid_dims=dict(changeable_grid_dims or {}),
            rows_split_over_gpus=rows_split_over_gpus, gpu_split_blocks=grid_dim[0], one_task_per_sm=one_task_per_sm,
            concurrent_along_axis=concurrent_along_axis, cluster_pair_params=cluster_pair_params,
            buf=buf, out_buffer=out_buffer, exchange=exchange, counter=counter, slots=list(slots), dry=dry,
            cost_inputs=list(cost_inputs))

    # ---- slots ----
    def _new_buf(self, entry: dict) -> int:
        assert len(self._bufs) < MAX_BUFS, f"more than {MAX_BUFS} buffer slots (config.cuh MAX_BUFS)"
        self._bufs.append(entry)
        return len(self._bufs) - 1

    def _scratch(self, name: str, nbytes: int, reset: int = -1, by_reader: bool = False) -> int:
        """A buffer only the node's tasks use (allocated by the host code; reset before each launch to the byte `reset`, -1: none).
        Returns its buffer slot."""
        return self._new_buf({"kind": "scratch", "name": name, "fill": lambda n: (nbytes, reset, by_reader)})

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
        kind and dims as host.cuh reads them: bf16 (rows, K, box rows), wblk (pieces, box pieces), sf (chunks, box chunks), act8
        (K bytes, rows), act8kt (K bytes, rows, K tiles). Returns the slot."""
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

    def host_slot_args(self) -> dict:
        """The host code's slot args (host.cuh): "buf.<slot>" -> "<kind> <name> <bytes> <reset> <by reader>" (the node outputs with
        their nodes' grids now), "map.<slot>" -> "<kind> <source> <dims...>"."""
        from .compiler import graph_nodes
        nodes = {n.graph_idx: n for n in graph_nodes(self)}
        names = self._tensor_names   # guid -> the graph tensor's name
        _, _, ex_offsets, _ = self.exchange_layout()
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
        return args

    def gemm_tile_layer(
        self,
        input: DTensor,     # [T, K] bf16
        weight: DTensor,    # [N, K] bf16
        globals: DTensor,   # G struct (uint8)
        maps: DTensor,      # CUtensorMaps (uint8)
        output: DTensor,    # [T, N] fp32 (its buffer holds what `combine` says)
        combine: str = "slots",   # how the K parts are combined (gemm_tile.cuh Combine): "slots" (one 0xFF-prefilled slot per
                                  # K part, the reader adds them), "add" (fixed-point atomic add, a counter per 128-row block),
                                  # "store" (plain stores, one K part, a counter of tasks)
        rows_split_over_gpus: bool = False,   # each GPU computes its part of the weight's rows (latent_down, latent_up)
        input_polled: bool = False,   # the input is a node's 0xFF-prefilled bf16 buffer, polled per 16 B straight into the ring's
                                      # stages (each task starts on its columns as soon as they are written, without a counter)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y) = weight rows [R x, R x + R) times K part y. Default grid: R = 128 rows, K in 1 part (polled: the fewest K
        parts with at most RING_STAGES K tiles per task); the compiler may change grid x to R = 64 rows (the kernel template has a
        64-row tile, MMA M = 64) and grid y (K parts), in whole 128-column pieces, at least 2 per task ("store": one K part, 128 rows,
        no options; polled: 128 rows, at most RING_STAGES pieces per task, as the whole K part is written into the stages up front).
        The weight and the activation reach the task through tensor maps (Maps::gemm slots: the weight's 128- and 64-row boxes, the
        activation's; polled: the weight's 128-row box only), the output through the node's buffer slot. params = [mode, K, N,
        polled (0 / 1)]; slots = [weight map slot, activation map slot (-1: polled)]."""
        assert input.num_dims == 2 and weight.num_dims == 2 and weight.dim(1) == input.dim(1)
        assert output.num_dims == 2 and output.dim(1) == weight.dim(0) and weight.dim(0) % 128 == 0 and weight.dim(1) % 128 == 0
        mode, N, K, T, GPUS = COMBINE[combine], weight.dim(0), weight.dim(1), self._tokens(input.dim(0)), self.num_gpus
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
            out_buffer = lambda n: (n.grid[1] * T * N * 4, 0xFF, True)
        elif combine == "add":     # [T][N] int64 fixed point, 0 before each launch
            out_buffer = lambda n: (T * N * 8, 0, False)
        else:                      # [T][N] fp32, every value written each launch
            out_buffer = lambda n: (T * N * 4, -1, False)
        assert combine != "add" or N // 128 <= 32, "one counter per 128-row block, at most 32"
        tiles = K // 128
        if input_polled:
            parts = next(y for y in range(1, tiles + 1) if tiles % y == 0 and tiles // y <= RING_STAGES)
            params, slots = [mode, K, N, 1], [self._gemm_map(weight, (128,)), -1]
            changeable = {1: (128, 2, RING_STAGES)}
        else:
            parts = 1
            params, slots = [mode, K, N, 0], [self._gemm_map(weight, (128, 64)), self._gemm_map(input, (T,))]
            changeable = {0: (64, 1, 2), 1: (128, 2, None)} if combine != "store" else {}
        self._add_node("gemm_tile", (N // 128, parts, 1), block_dim, [input, weight, globals, maps, output],
                       [(-1, 1, -1), (0, 1, -1), (-1, -1, -1), (-1, -1, -1), (1, -1, -1)], params,
                       # K: at least 2 tiles of 128 per task (the two MMA issuers take the stages in turn, runtime.cuh issue_job;
                       # with one stage issuer 1 has no MMA and its stale accumulator would be added)
                       changeable_grid_dims=changeable, rows_split_over_gpus=rows_split_over_gpus, out_buffer=out_buffer,
                       counters={"slots": 0, "add": N // 128, "store": 1}[combine], slots=slots)

    def topk_route_layer(
        self,
        input: DTensor,     # router logits [T, E] fp32
        bias: DTensor,      # score correction bias [E] fp32
        globals: DTensor,
        output: DTensor,    # routing pairs [T, K]
        block_dim: tuple = (256, 1, 1),
    ):
        """Task t = token t: adds the router's partial sums of row t, top K (output.dim(1)) of the E experts. params = [E, K]."""
        T, E, K = self._tokens(input.dim(0)), input.dim(1), output.dim(1)
        assert input.num_dims == 2 and bias.num_dims == 1 and bias.dim(0) == E and output.num_dims == 2 and output.dim(0) == T
        self._add_node("topk_route", (output.dim(0), 1, 1), block_dim, [input, bias, globals, output],
                       [(0, -1, -1), (-1, -1, -1), (-1, -1, -1), (0, -1, -1)], [E, K],
                       out_buffer=lambda n: (T * K * 8, 0xFF, False),   # the pairs, u64, polled by every moe_experts task
                       slots=[self._tensor_slot(bias)], dry=True, cost_inputs=[0])   # adds the router's K parts

    def sum_quant_send_layer(
        self,
        input: DTensor,     # latent z [T, L] fp32
        globals: DTensor,
        output: DTensor,    # z_q [T, L] e4m3 bytes (uint8)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = columns [128 x, 128 x + 128) of z: adds latent_down's partial sums, MXFP8, sends to every GPU. Divided among
        the GPUs like latent_down's rows. Its output (z_q, then its scale chunks) is in the exchange region: every GPU writes its
        columns into every GPU's copy; 0xFF before the first launch, re-armed by the node after moe_experts (allreduce_send).
        params = [L] (z's width)."""
        T, L = self._tokens(input.dim(0)), input.dim(1)
        assert input.num_dims == 2 and output.num_dims == 2 and (output.dim(0), output.dim(1)) == (T, L) and L % 128 == 0
        self._add_node("sum_quant_send", (output.dim(1) // 128, 1, 1), block_dim, [input, globals, output],
                       [(1, -1, -1), (-1, -1, -1), (1, -1, -1)], [L], rows_split_over_gpus=True,
                       out_buffer=lambda n: (zq_bytes(T, L), 0xFF, True), exchange=True, cost_inputs=[0])   # adds latent_down's K parts

    def situ_and_mul_layer(
        self,
        input: DTensor,     # shared gate_up [T, 2 * SHR] fp32
        globals: DTensor,
        output: DTensor,    # h_s [T, SHR] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = h_s features [F x, F x + F), from the gate and up columns of those features in shared gate_up (blocks of 128
        columns: 64 gate, then their 64 up). Default F = 64 (12 tasks); the compiler may give F = 32, 64, 96 or 128 (24, 12, 8, 6
        tasks): grid x in units of 32 columns, at most 8 per task of the input (F = 128 reads 256 input columns)."""
        T, SHR = self._tokens(input.dim(0)), output.dim(1)
        assert input.num_dims == 2 and output.num_dims == 2 and input.dim(1) == 2 * SHR and output.dim(0) == T and SHR % 64 == 0
        self._add_node("situ_and_mul", (output.dim(1) // 64, 1, 1), block_dim, [input, globals, output],
                       [(1, -1, -1), (-1, -1, -1), (1, -1, -1)], [], changeable_grid_dims={0: (32, 1, 8)},
                       out_buffer=lambda n: (T * SHR * 2, -1, False), counters=1,   # h_s bf16; a count of tasks
                       cost_inputs=[0])   # waits for shared gate_up's K parts

    def moe_experts_layer(
        self,
        z_q: DTensor, pairs: DTensor, h_s: DTensor, w13: DTensor, w2: DTensor, globals: DTensor, maps: DTensor,
        output: tuple,      # (R [T x K, L] fp32,): this GPU's routed rows, one per (token, routed expert k); allreduce_send adds a
                            # token's K in order (shared down is its own gemm_tile node)
        w13_scales: DTensor = None, w2_scales: DTensor = None,   # read through the tensor maps only
        block_dim: tuple = (256, 1, 1),
    ):
        """One task per SM (task x on SM x); the SMs take the expert work from a queue at run time; the output rows are the
        node's buffer. Declares what its kernel template (tasks/moe_experts.cuh ExpertSlots) reads: h_q and its scales (scratch),
        the tensor maps of the expert weights and scales, of z_q (both sets of the exchange region) and of h_q; its counters: the
        SMs done, the queue's next entry, per expert slot (T x K of them) the W13 items done (one line, one line, T x K / 32 lines).
        Sizes: E experts and K per token (pairs, topk_route's params), L = z_q's width, IR = the expert intermediate on this GPU
        (w13 [E, 2 IR, L / 2] MXFP4 bytes, w2 [E, L, IR / 2]); params = [IR]."""
        T, K, L = self._tokens(pairs.dim(0)), pairs.dim(1), z_q.dim(1)
        E, IR = w13.dim(0), w13.dim(1) // 2
        NSLOT, KT_L, KT2 = T * K, L // 128, IR // 128
        assert w13.num_dims == 3 and (w13.dim(1), w13.dim(2)) == (2 * IR, L // 2) and IR % 128 == 0 and L % 128 == 0
        assert w2.num_dims == 3 and (w2.dim(0), w2.dim(1), w2.dim(2)) == (E, L, IR // 2)
        assert len(output) == 1 and (output[0].dim(0), output[0].dim(1)) == (T * K, L)
        assert w13_scales is not None and w2_scales is not None   # read through the tensor maps only (a node has at most 8 inputs)
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
        slots = [hq, hsf]
        for weight, scales, pieces in ((w13, w13_scales, w13_pieces), (w2, w2_scales, w2_pieces)):
            slots += [self._map("wblk", ("t", weight), (pieces, 1)), self._map("wblk", ("t", weight), (pieces, 2)),
                      self._map("sf", ("t", scales), (pieces, 1)), self._map("sf", ("t", scales), (pieces, 2))]
        for kind, offset, dims in (("act8", 0, (L, T)), ("act8kt", 0, (L, T, 2)), ("sf", T * L, (KT_L, KT_L))):
            first = self._map(kind, ("x", zq.info["buf"], offset, 0), dims)      # z_q in set 0, then set 1 (the next slot)
            assert self._map(kind, ("x", zq.info["buf"], offset, 1), dims) == first + 1
            slots.append(first)
        slots += [self._map("act8kt", ("b", hq, 0), (IR, NSLOT * T, KT2)), self._map("sf", ("b", hsf, 0), (NSLOT * KT2, KT2))]
        tensors = [z_q, pairs, h_s, w13, w2, globals, maps] + list(output)
        self._add_node("moe_experts", (self._num_sms(), 1, 1), block_dim, tensors, [(-1, -1, -1)] * len(tensors),
                       [IR], one_task_per_sm=True, out_buffer=lambda n: (T * K * L * 4, -1, False), counters=32 * 2 + NSLOT,
                       slots=slots)

    # ---- after moe_experts (tasks/allreduce_send.cuh, sum_rmsnorm.cuh, sum_gpus.cuh, sum_send.cuh, residual_add.cuh). The
    # graph tensors between them (the sent partial sums, sum_rmsnorm's Rn, sum_gpus's Ssum, latent_up's K parts, sum_send's o)
    # stand for the buffers the bodies use; their maps give the dependencies.
    def allreduce_send_layer(
        self,
        input: DTensor,         # fp32 [T x G, W], a node's output buffer: G rows per token (added in order), e.g. this GPU's routed
                                # rows R [T x 16, LAT] (moe_experts) or S [T, H] (the shared down node)
        globals: DTensor,
        output: DTensor,        # what is sent: [T, W] (its buffer: [num_gpus][T][W] bf16 in the exchange region)
        num_tasks: int = None,  # default: one per SM
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x: slice x of this GPU's partial sum (bf16) to every GPU, into this GPU's slot of the node's output (in the exchange
        region), once the input's producer is done (all its tasks, counted in its counter). params = [W, G]. When the producer is
        moe_experts, after which no SM reads z_q, the tasks also re-arm z_q (slots: [z_q's buffer slot], else [-1])."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        if p is None or p.info["buf"] < 0 or p.info["counter"] < 0:
            raise ValueError(f"allreduce_send_layer: the input must be a node's output buffer whose node counts its tasks; it comes from "
                             f"{p.name if p else 'the graph inputs'}")
        T, W, GPUS = self._tokens(output.dim(0)), output.dim(1), self.num_gpus
        assert input.dim(0) % T == 0 and input.dim(1) == W and W % 8 == 0
        rearm = -1
        if p.name == "moe_experts":   # z_q: moe_experts' first input
            rearm = producer_of(p.inputs[0], graph_nodes(self)).info["buf"]
        self._add_node("allreduce_send", (num_tasks or self._num_sms(), 1, 1), block_dim, [input, globals, output],
                       [(-1, -1, -1)] * 3, [W, input.dim(0) // T],
                       out_buffer=lambda n: (GPUS * T * W * 2, 0xFF, True), exchange=True, slots=[rearm])

    def sum_rmsnorm_layer(
        self,
        input: DTensor,         # [R | S] as sent
        gamma: DTensor,         # RMSNorm weight [L] bf16
        globals: DTensor,
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
        assert gamma.dim(0) == L and L % 64 == 0 and L // 16 <= 256   # tasks/sum_rmsnorm.cuh rn_threads
        e = float_bits(eps)
        self._add_node("sum_rmsnorm", (4, T, 1), block_dim, [input, gamma, globals, output],
                       [(-1, -1, -1), (-1, -1, -1), (-1, -1, -1), (1, 0, -1)], [1 if recompute else 0, e],
                       changeable_grid_dims={0: (L // 4, 1, 4)},   # whole quarters: 1, 2 or 4 tasks per token
                       concurrent_along_axis=None if recompute else 0, cluster_pair_params=None if recompute else [2, e],
                       out_buffer=lambda n: (T * L * 2, 0xFF, False),   # Rn bf16, polled by every latent_up task of its columns
                       slots=[self._tensor_slot(gamma), self._scratch("sum_rmsnorm.ss_part", T * 4 * 4, 0xFF)])   # the swap slots

    def sum_gpus_layer(
        self,
        input: DTensor,         # [R | S] as sent
        globals: DTensor,
        output: DTensor,        # Ssum [T, H] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y): eighth x of S for token y: add S over the GPUs."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        assert H % 64 == 0 and H // 64 <= 256   # tasks/sum_gpus.cuh: one 8-column vector per thread of each eighth
        self._add_node("sum_gpus", (8, T, 1), block_dim, [input, globals, output], [(-1, -1, -1), (-1, -1, -1), (1, 0, -1)], [],
                       out_buffer=lambda n: (T * H * 2, 0xFF, True))   # Ssum bf16, polled and re-armed by residual_add

    def sum_send_layer(
        self,
        input: DTensor,         # [T, H] fp32: a gemm_tile node's output in slots (its K parts), its rows split over the GPUs
        globals: DTensor,
        output: DTensor,        # o [T, H] bf16: each GPU sends its own rows to every GPU
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y): row block x (this GPU's blocks, as its producer's), slice y of 14 of the block's vectors: adds the K parts,
        sends to every GPU."""
        from .compiler import graph_nodes, producer_of
        p = producer_of(input, graph_nodes(self))
        assert p is not None and p.name == "gemm_tile" and p.params[0] == COMBINE["slots"] and p.info["rows_split_over_gpus"], \
            "sum_send_layer: the input is a gemm_tile node's output in slots, its rows split over the GPUs"
        T, H = self._tokens(input.dim(0)), input.dim(1)
        assert (output.dim(0), output.dim(1)) == (T, H) and H % (128 * self.num_gpus) == 0
        self._add_node("sum_send", (H // 128, 14, 1), block_dim, [input, globals, output],
                       [(1, -1, -1), (-1, -1, -1), (1, -1, -1)], [], rows_split_over_gpus=True,
                       out_buffer=lambda n: (T * H * 2, 0xFF, True), exchange=True,   # [GPUS][T][H / GPUS] bf16
                       cost_inputs=[0])   # adds the GEMM's K parts

    def residual_add_layer(
        self,
        addend: DTensor,        # S [T, H] bf16, sum_gpus's output buffer
        input: DTensor,         # o: this GPU's part, sent by sum_send (every GPU's part arrives through the exchange region)
        residual: DTensor,      # [T, H] bf16 (a graph input)
        globals: DTensor,
        output: DTensor,        # y [T, H] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x of the SMs: slice x of y = bf16(bf16(input + addend) + residual). params = [H]."""
        T, H = self._tokens(output.dim(0)), output.dim(1)
        assert (residual.dim(0), residual.dim(1)) == (T, H) and H % (8 * self.num_gpus) == 0
        self._add_node("residual_add", (self._num_sms(), 1, 1), block_dim, [addend, input, residual, globals, output],
                       [(-1, -1, -1)] * 5, [H], slots=[self._tensor_slot(residual), self._tensor_slot(output)])

    # ---- the generated layer (static_schedule.py) ----
    def compile_static(self, schedule_paths, gpu_tensors=None, out_dir=None, extra_flags=None, code=None, profile=False):
        """Build the graph as it is now, one schedule.json per GPU (static_schedule.compile_static); code: a layer.cu to build
        instead of the generated one; profile: the timing build (per-task start / end stamps, StaticKernel.task_times)."""
        from .static_schedule import compile_static
        return compile_static(self, schedule_paths, gpu_tensors, out_dir, extra_flags=extra_flags, code=code, profile=profile)
