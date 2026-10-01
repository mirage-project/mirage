"""StaticMegakernel: a PersistentKernel whose graph is built into one generated kernel with a static per-SM task schedule
(compiler.py -> static_schedule.py), instead of MPK's runtime.

It adds the layers of the K3 MoE layer (each adds one graph node, as MPK's layers do, and keeps what compiler.py needs about
it), the task family declaration those layers use (declare_host_state), and compile_static. The kernel code of these layers is
in include/mirage/static_megakernel/ (the task bodies, moe_kernel.cuh: the task functions the generated loop calls, moe_host.cuh:
the host code). Every layer takes the layer state `globals` (the G struct) and, for the GEMM-side layers, `maps` (the tensor
maps) as extra inputs; in the built layer both are kernel parameters. Each layer adds its node with a default grid (no K
split); compiler.compile_plan regrids to the plan.
"""
import torch

from ..core import *
from ..kernel import TBGraph
from .persistent_kernel import PersistentKernel

# the sizes the kernel bodies are written for (include/mirage/static_megakernel/config.cuh); each layer checks its tensors against
# them, so a different model fails here instead of in the kernel
T, H, LAT, NE, SHR = 8, 7168, 3584, 896, 768


class StaticMegakernel(PersistentKernel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._host_states = {}    # task family -> its pieces of the generated layer (declare_host_state)
        self._static_nodes = {}   # graph index -> what compiler.py needs about a node (_add_node)

    def _num_sms(self):
        return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

    def _add_node(self, name: str, grid_dim: tuple, block_dim: tuple, tensors: list, maps: list, params: list,
                  changeable_grid_dims: dict = None, rows_split_over_gpus: bool = False, one_task_per_sm_at_end: bool = False):
        """Add one node, as MPK's layers do (customized + register_task), and keep for compiler.py what the graph does not hold:
          maps                    maps[i] = which dimension of tensors[i] each grid axis cuts (MPK's convention: (1, -1, -1) =
                                  grid x cuts dim 1); a task's slices follow from the maps and the grid, for any grid
          changeable_grid_dims    {grid axis: unit}: the grid axes the compiler may change, and the number of elements of each
                                  dimension that axis cuts one task must get a multiple of (gemm_tile: {1: 128}, K in whole
                                  128-column pieces); the other axes keep grid_dim
          rows_split_over_gpus    grid axis x is divided among the GPUs: each GPU computes its part (compiler.owned_row_blocks)
          one_task_per_sm_at_end  one task on every SM (grid x = SM), after the SM's other tasks"""
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        for t, m in zip(tensors, maps):
            tb_graph.new_input(t, m, -1, True)
        self.kn_graph.customized(tensors, tb_graph)
        self.kn_graph.register_task(tb_graph, name, params)
        self._static_nodes[self.kn_graph.get_num_operators() - 1] = dict(
            name=name, maps=list(maps), changeable_grid_dims=dict(changeable_grid_dims or {}),
            rows_split_over_gpus=rows_split_over_gpus, one_task_per_sm_at_end=one_task_per_sm_at_end)

    def gemm_tile_layer(
        self,
        input: DTensor,     # [T, K] bf16
        weight: DTensor,    # [N, K] bf16
        globals: DTensor,   # G struct (uint8)
        maps: DTensor,      # CUtensorMaps (uint8)
        output: DTensor,    # [T, N] fp32; the K-split partial sums live in G
        kind: int,          # 0 router, 1 latent_down, 2 shared gate_up, 3 shared down: which weight and output the task uses
        block_dim: tuple = (256, 1, 1),
    ):
        """Task (x, y) = weight rows [128 x, 128 x + 128) times K part y. Default grid: K in 1 part; the compiler may change
        grid y (K parts), in whole 128-column pieces. params = [kind, K]."""
        assert input.num_dims == 2 and weight.num_dims == 2 and weight.dim(1) == input.dim(1) and input.dim(0) == T
        assert output.num_dims == 2 and output.dim(1) == weight.dim(0) and weight.dim(0) % 128 == 0 and weight.dim(1) % 128 == 0
        self._add_node("gemm_tile", (weight.dim(0) // 128, 1, 1), block_dim, [input, weight, globals, maps, output],
                       [(-1, 1, -1), (0, 1, -1), (-1, -1, -1), (-1, -1, -1), (1, -1, -1)], [kind, weight.dim(1)],
                       changeable_grid_dims={1: 128}, rows_split_over_gpus=(kind == 1))

    def route_layer(
        self,
        input: DTensor,     # router logits [T, E] fp32
        bias: DTensor,      # score correction bias [E] fp32
        globals: DTensor,
        output: DTensor,    # routing pairs [T, 16]
        block_dim: tuple = (256, 1, 1),
    ):
        """Task t = token t: adds the router's partial sums of row t, top-16."""
        assert input.num_dims == 2 and (input.dim(0), input.dim(1)) == (T, NE) and bias.num_dims == 1 and bias.dim(0) == NE
        assert output.num_dims == 2 and output.dim(1) == 16
        self._add_node("route", (output.dim(0), 1, 1), block_dim, [input, bias, globals, output],
                       [(0, -1, -1), (-1, -1, -1), (-1, -1, -1), (0, -1, -1)], [])

    def quant_layer(
        self,
        input: DTensor,     # latent z [T, L] fp32
        globals: DTensor,
        output: DTensor,    # z_q [T, L] e4m3 bytes (uint8)
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = columns [128 x, 128 x + 128) of z: adds latent_down's partial sums, MXFP8, sends to every GPU. Divided among
        the GPUs like latent_down's rows."""
        assert input.num_dims == 2 and (input.dim(0), input.dim(1)) == (T, LAT) and output.num_dims == 2 and output.dim(1) == LAT
        self._add_node("quant", (output.dim(1) // 128, 1, 1), block_dim, [input, globals, output],
                       [(1, -1, -1), (-1, -1, -1), (1, -1, -1)], [], rows_split_over_gpus=True)

    def sact_layer(
        self,
        input: DTensor,     # shared gate_up [T, 2 * SHR] fp32
        globals: DTensor,
        output: DTensor,    # h_s [T, SHR] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """Task x = h_s features [64 x, 64 x + 64), from shared gate_up columns [128 x, 128 x + 128): adds the partial sums."""
        assert input.num_dims == 2 and output.num_dims == 2 and (input.dim(0), input.dim(1)) == (T, 2 * SHR) and output.dim(1) == SHR
        self._add_node("sact", (output.dim(1) // 64, 1, 1), block_dim, [input, globals, output],
                       [(1, -1, -1), (-1, -1, -1), (1, -1, -1)], [])

    def expert_queue_layer(
        self,
        z_q: DTensor, pairs: DTensor, h_s: DTensor, w13: DTensor, w2: DTensor, globals: DTensor, maps: DTensor,
        output: tuple,      # (R [T, L] fp32, S [T, H] fp32) rank partials
        w13_scales: DTensor = None, w2_scales: DTensor = None, shared_down_weight: DTensor = None,   # read through the tensor maps only
        block_dim: tuple = (256, 1, 1),
    ):
        """One task per SM, after the SM's placed tasks; the SMs take the expert work from a queue at run time. Also declares the
        MoE task family: the pieces of the generated layer (static_schedule.py)."""
        assert len(output) == 2 and (output[0].dim(0), output[0].dim(1)) == (T, LAT) and (output[1].dim(0), output[1].dim(1)) == (T, H)
        assert w13_scales is not None and w2_scales is not None and shared_down_weight is not None
        # a task has at most 8 inputs: these three reach the bodies through the tensor maps, built by the family's host code
        extra = {"w13_scales": self._tensor_names[w13_scales.guid], "w2_scales": self._tensor_names[w2_scales.guid],
                 "shared_down_weight": self._tensor_names[shared_down_weight.guid]}
        from .moe import moe_host_args
        self.declare_host_state(
            "moe", header="mirage/static_megakernel/moe_host.cuh", args=lambda pk: moe_host_args(pk, extra),
            init="moe_host_init", reset="moe_host_reset", finalize="moe_host_finalize", report="moe_host_report",
            timing="moe_host_timing",
            phase_stamps="moe_host_phase_stamps", phase_names=["queue_start", "queue_end"],   # per SM (config.cuh STAMP_QUEUE_*)
            # the kernel's parameters: the tensor maps and the layer state G (by value), 256 threads,
            # SMEM_BYTES dynamic shared memory
            kernel_params="const __grid_constant__ static_mk::Maps maps, static_mk::G g",
            launch_args="moe_host::g_gpus[gpu].maps, moe_host::g_gpus[gpu].g",
            threads=256, dynamic_smem="static_mk::SMEM_BYTES", max_tasks="static_mk::MAX_TASKS_PER_SM",
            kernel_begin="static_mk::KernelLocals L; static_mk::kernel_begin(maps, g, L);",
            task_begin="",
            task_function="static_mk::run_{name}", task_args="maps, g, L, tk",   # e.g. static_mk::run_route<...>(maps, g, L, tk)
            kernel_end="static_mk::kernel_end(g, L);")
        tensors = [z_q, pairs, h_s, w13, w2, globals, maps] + list(output)
        self._add_node("expert_queue", (self._num_sms(), 1, 1), block_dim, tensors, [(-1, -1, -1)] * len(tensors), [],
                       one_task_per_sm_at_end=True)

    def tail_layer(
        self,
        r_partial: DTensor, s_partial: DTensor, prefix: DTensor, gamma: DTensor, w_up: DTensor, globals: DTensor, maps: DTensor,
        output: DTensor,    # y [T, H] bf16
        block_dim: tuple = (256, 1, 1),
    ):
        """One task per SM, after the SM's expert queue task: [R|S] all-reduce, RMSNorm, latent_up, y."""
        assert output.num_dims == 2 and (output.dim(0), output.dim(1)) == (T, H)
        tensors = [r_partial, s_partial, prefix, gamma, w_up, globals, maps, output]
        self._add_node("tail", (self._num_sms(), 1, 1), block_dim, tensors, [(-1, -1, -1)] * len(tensors), [],
                       one_task_per_sm_at_end=True)

    # ---- the generated layer (static_schedule.py) ----
    def declare_host_state(self, name: str, **decl):
        """A task family's pieces of the generated layer (static_schedule.generate_code): its header, its host functions (init,
        reset, finalize, report, timing, with their args: a dict or a function pk -> dict evaluated at build time), and its
        kernel pieces (kernel_params, launch_args, threads, dynamic_smem, max_tasks = entries per SM list, kernel_begin,
        task_begin, task_function = the name pattern of a node's task function, e.g. "static_mk::run_{name}", task_args,
        kernel_end). phase_stamps / phase_names: per-SM stamps the family's kernel writes anyway (used by timing tools, not the
        build). Declared once per family."""
        self._host_states.setdefault(name, dict(decl))

    def compile_static(self, schedule_paths, gpu_tensors=None, out_dir=None, extra_flags=None, code=None):
        """Build the graph as it is now, one schedule.json per GPU (static_schedule.compile_static); code: a layer.cu to build
        instead of the generated one."""
        from .static_schedule import compile_static
        return compile_static(self, schedule_paths, gpu_tensors, out_dir, extra_flags=extra_flags, code=code)
