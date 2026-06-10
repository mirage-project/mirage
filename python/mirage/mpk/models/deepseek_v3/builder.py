"""DeepSeek V3 model builder for Mirage MPK.

Architecture: 61 decoder layers with MLA attention and MoE MLP.
- Layers 0-2: Dense MLP (DeepseekV2MLP)
- Layers 3-60: MoE MLP (256 experts, top-8, + shared experts)
- MLA: 128 Q heads, 1 KV head after weight absorption, head_dim=576 (512+64)

Weight absorption: at load time, kv_b_proj is absorbed into q_b_proj so that
runtime only needs compressed KV cache [c_latent(512), k_pe(64)] = 576 dims.
"""

import math
import torch
from typing import Optional

from ..utils import grid_for_rmsnorm_linear_layer
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from . import tasks as dsv3_tasks
from ....core import (bfloat16, float8_e4m3, float32, uint32, int32, int64,
                       uint64)


# DeepSeek V3 architecture constants
HIDDEN_SIZE = 7168
NUM_LAYERS = 61
NUM_Q_HEADS = 128         # total Q heads
Q_LORA_RANK = 1536        # q_a_proj output dim
KV_LORA_RANK = 512        # c_latent dim
QK_NOPE_HEAD_DIM = 128    # per-head nope dim
QK_ROPE_HEAD_DIM = 64     # per-head rope dim
# QKV-a fused GEMM output width:
#   1536 (q_a) + 512 (c_latent) + 64 (k_pe) + 64 (pad to 128-row MMA tile) = 2176
QKV_A_FUSED_N = 2176
V_HEAD_DIM = 128          # per-head value dim (before absorption)
QK_HEAD_DIM_TOTAL = 576   # 512 latent + 64 rope (after absorption)
V_HEAD_DIM_TOTAL = 512    # latent dim only (after absorption)
INTERMEDIATE_SIZE = 18432       # Dense MLP intermediate (layers 0-2)
MOE_INTERMEDIATE_SIZE = 2048    # Per-expert intermediate (routed + shared)
NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_SHARED_EXPERTS = 1
FIRST_MOE_LAYER = 3
VOCAB_SIZE = 129280
RMS_NORM_EPS = 1e-6

# FP8 MoE group GEMM N-split helper. Picks grid_dim.y so the kernel's per-CTA
# N-slice (ORIG_OUTPUT_SIZE / Y) stays a multiple of MMA_M=128 and has at
# least one full tile. Without this each active-expert CTA serializes all
# m-tiles. Empirically Y=2 gives ~20% speedup on TP=2 MTP=2 stress (6.6 ms/tok
# vs 8.3 ms/tok baseline). Higher Y regressed in earlier tests, but those
# regressions now look like they were GPU-pair/contention artifacts, not the
# kernel itself — Y=2 is a conservative landing. Queue bumped to 8192 in
# 11f45fd so Y=2 has headroom.
_MOE_FP8_MMA_M = 128




# B34 (2026-05-15): multi-row-per-CTA grid for RMSNorm. Shrinks grid.x from
# mbt down to ~mbt//8 so each CTA processes ~8 rows via the kernel's existing
# batch_idx loop, gated by a runtime row_count_cap (active_rows - first_row)
# so decode iters don't overwrite inactive rows with stale-bf16 normalized
# output. Frees up worker slots that previously sat idle when active_rows < mbt.
# The threadblock partition asserts (mbt % grid.x == 0), so when mbt is not a
# multiple of the preferred rows-per-task we fall back to the largest divisor
# of mbt at or below it (typically aligned to powers of two in real configs).
# C2 (2026-05-16): rows-per-CTA tuning sweep.
#   8 → 4: -17 μs/layer (max 29 → 16 μs)
#   4 → 2: -10 μs/layer (max 16 → 9 μs)
#   2 → 1: -2  μs/layer (max  9 → 5 μs, tail-latency win)
# At rows=1, grid=128 = exactly num_workers, but RMSnorm and other tasks
# don't seem to contend severely. Empirically still positive.
_RMSNORM_ROWS_PER_TASK = 1


def _rmsnorm_grid(mbt: int) -> tuple:
    if mbt <= _RMSNORM_ROWS_PER_TASK:
        # Single-CTA covers the whole batch — kernel BATCH_SIZE = mbt.
        return (1, 1, 1)
    target = mbt // _RMSNORM_ROWS_PER_TASK
    # Snap to a divisor of mbt at or above `target` so grid_x divides mbt.
    # Walk up from target until we find a divisor; bounded by mbt itself.
    g = target
    while g <= mbt and mbt % g != 0:
        g += 1
    if g > mbt:
        g = mbt  # legacy 1-row-per-CTA fallback
    return (g, 1, 1)






def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float,
                              max_position_embeddings: int) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(low_rot: int,
                                high_rot: int,
                                dim: int,
                                base: float,
                                max_position_embeddings: int) -> tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(
            low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(
            high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _tensor_parallel_allreduce_grid(output_size: int) -> tuple[int, int, int]:
    """Pick a grid for the NVSHMEM tile allreduce that mirrors the producer's
    column-tile granularity.

    All current producers of an allreduce input — FP8 swapAB linear, BF16
    — partition the hidden dim
    into 128-wide column tiles (linear.grid.x = output_size // 128, or
    128-wide split). The legacy default of 1024-wide allreduce tiles
    therefore generated ~8x fewer tasks than the producing layer (7 vs 56
    for DSv3 hidden=7168), starving the persistent runtime of dispatchable
    work right after the matmul finished.

    Defaulting the allreduce tile to 128 keeps the partition aligned with
    the producer so each upstream task has a one-to-one downstream
    consumer.

    Empirical note (2026-05-11): a tile-size sweep on prefill-128 TP=4 EP=2
    showed TILE=128 / 512 / 1024 all within ~1.5ms (= run-to-run noise)
    end-to-end. Per-task barrier wallclock (~370μs in prefill, ~12μs in
    decode) is **independent of tile size** — barrier-dominated, not
    bandwidth-bound. So tile-size tuning here is not a productive lever;
    closing the AR vs vLLM gap (12μs vs 6μs in decode) requires kernel-
    level changes (e.g., single global barrier shared across all AR tasks).
    """
    if output_size % 128 != 0:
        raise ValueError(
            "Tensor-parallel all-reduce expects a 128-aligned output "
            f"dimension, got {output_size}")
    return (output_size // 128, 1, 1)


@register_model_builder("deepseek-v3", "DeepSeek-V3", "deepseek-ai/DeepSeek-V3")
class DeepSeekV3Builder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.num_workers = mpk.num_workers
        self.rank = mpk.mpi_rank
        self.ep_size = getattr(mpk, "ep_size", 1)
        assert self.ep_size >= 1
        assert self.world_size % self.ep_size == 0
        self.routed_tp_size = self.world_size // self.ep_size
        self.routed_tp_rank = self.rank % self.routed_tp_size
        self.ep_rank = self.rank // self.routed_tp_size
        assert NUM_EXPERTS % self.ep_size == 0
        self.num_local_experts = NUM_EXPERTS // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self._use_nvshmem = mpk.use_nvshmem  # True only if nvshmem is actually enabled
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        # Weight attach cache (per-name dedup of attach_input declarations).
        self._attach_cache = {}
        self.max_num_batched_tokens = mpk.max_num_batched_tokens
        self.ckv_kpe_cache = None
        self.position_embeddings = None
        self.rope_theta = 10000.0
        self.rope_parameters = None
        self.original_max_position_embeddings = 4096

        # DeepSeek V3 dimensions
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.num_q_heads = NUM_Q_HEADS
        self.num_local_q_heads = NUM_Q_HEADS // self.world_size
        self.qk_head_dim = QK_HEAD_DIM_TOTAL  # 576 after absorption
        self.v_head_dim = V_HEAD_DIM_TOTAL     # 512 after absorption
        self.q_lora_rank = Q_LORA_RANK
        self.kv_lora_rank = KV_LORA_RANK
        self.intermediate_size = INTERMEDIATE_SIZE // self.world_size
        # Routed experts are split over tensor-parallel ranks inside an
        # expert-parallel group; shared experts stay tensor-parallel over all ranks.
        self.shared_moe_intermediate_size = MOE_INTERMEDIATE_SIZE // self.world_size
        self.routed_moe_intermediate_size = MOE_INTERMEDIATE_SIZE // self.routed_tp_size
        # Kept for legacy shared-expert helper paths.
        self.moe_intermediate_size = self.shared_moe_intermediate_size

        # M_TOTAL for the new GEMM = NUM_LOCAL_EXPERTS * BM_PADDING. Both
        # are compile-time-ish (NUM_LOCAL_EXPERTS depends on ep_size at
        # __init__ time; BM_PADDING matches the new GEMM's largest BM tile
        # and must be a multiple of fp8_group_gemm's internal BM tile =128).
        self._moe_bm_padding = 128
        # Experts-per-CTA for the NEW-MoE moe_permute_sm100 task. EPC=4
        # shrinks the permute launch from (E_LOCAL,1,1) to (E_LOCAL/4,1,1)
        # so the decode permute fits in one SM wave instead of contending
        # with the shared-expert GEMM across ~3 waves (analyzer-found
        # ~40 μs/layer decode "valley"); TP4+TP8 validated. E_LOCAL
        # (= num_local_experts) must be divisible by EPC — asserted at the
        # moe_permute call site once num_local_experts is known.
        self._moe_permute_epc = 4
        # C1 (2026-05-16): fan the MLA KV-gather unified task across 8 CTAs
        # by striding seq_pos. The legacy 1-CTA gather was 121 μs/layer
        # (15% of layer wallclock) with 127 workers idle; with 8 splits each
        # CTA strides seq_pos by 8 → ~121/8 μs/layer.
        self._kv_gather_splits = 8


    def _decode_q_len(self) -> int:
        # Decode q_len is 1 (single-token decode; MTP removed 2026-06-10 —
        # the MTP rebuild on its own branch reintroduces spec_length here).
        return 1

    def build_from_model(self, model_name: str, model_path: str = None):
        raise NotImplementedError(
            "DeepSeek V3 is too large for direct HuggingFace loading. "
            "Use build_from_config() with pre-converted weights."
        )

    def build_from_config(self, model_config: MirageModelConfig, layer_indices: list = None):
        """Build from pre-processed config with absorbed weights.

        Args:
            layer_indices: If provided, only build these specific layer indices.
        """
        self.ckv_kpe_cache = model_config.k_cache  # [num_layers, num_pages, page_size, 576]
        self.position_embeddings = model_config.position_embeddings
        self.rope_theta = model_config.rope_theta or 10000.0
        self.rope_parameters = model_config.rope_parameters
        max_pos = model_config.max_position_embeddings
        if self.rope_parameters is not None:
            max_pos = self.rope_parameters.get(
                "original_max_position_embeddings", max_pos)
        self.original_max_position_embeddings = max_pos or 4096

        self.build_from_dict(
            model_config.state_dict,
            model_config.with_lm_head,
            layer_indices=layer_indices,
        )

    def _new_tp_partial(self, output, name):
        if output.dim(1) % 128 != 0:
            raise ValueError(
                "TP residual allreduce expects the output dimension to be "
                f"128-aligned, got {output.dim(1)}")
        return self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, output.dim(1)),
            dtype=bfloat16,
            name=name,
            io_category="nvshmem_tensor" if self._use_nvshmem else "cuda_tensor",
        )

    def _allreduce_residual(self, partial, output, residual, gate_mode: int = 0):
        self.mpk.allreduce_layer(
            input=partial,
            buffer=self.allreduce_buf,
            output=output,
            residual=residual,
            grid_dim=_tensor_parallel_allreduce_grid(output.dim(1)),
            block_dim=(128, 1, 1),
            gate_mode=gate_mode,
        )

    def _fp8_linear_grid_dim(self, weight, grid_dim):
        """Pick grid_x for the FP8 swapAB kernel.

        The kernel asserts `output_size_per_task % MMA_M=128 == 0` and
        iterates m_tile internally, so a single task can cover *any positive
        multiple* of 128 output cols — 128 is the minimum, not the only
        allowed shard width. (The earlier comment "validated for N=128 task
        shards" was conservative; the cause of hangs at small shards was
        smaller-than-128, not larger-than-128.)

        With that, pick grid_x as the largest divisor of `output_size // 128`
        that is <= num_workers, so the layer fits in a single worker wave
        with the most parallelism the kernel allows. For shapes where
        `output_size // 128 <= num_workers`, this is the original behavior
        (one MMA_M tile per task).
        """
        output_size = weight.dim(0)
        if output_size % 128 != 0:
            raise ValueError(
                "FP8 linear runtime currently requires a 128-aligned output "
                f"dimension, got {output_size}")
        max_n_tiles = output_size // 128
        if max_n_tiles <= self.num_workers:
            grid_x = max_n_tiles
        else:
            # Largest divisor of max_n_tiles that fits in one wave.
            best = 1
            i = 1
            while i * i <= max_n_tiles:
                if max_n_tiles % i == 0:
                    if i <= self.num_workers:
                        best = max(best, i)
                    other = max_n_tiles // i
                    if other <= self.num_workers:
                        best = max(best, other)
                i += 1
            # If max_n_tiles has no decent divisor ≤ num_workers (e.g. prime),
            # 1-task-per-layer is far worse than letting a small overflow
            # wave run. Fall back to max_n_tiles in that pathological case.
            if best * 4 < self.num_workers:
                grid_x = max_n_tiles
            else:
                grid_x = best
        return (grid_x, grid_dim[1], grid_dim[2])

    def _can_use_decode_fp8_linear(self, input_fp8, weight, output, grid_dim):
        if input_fp8.dim(0) > 16 or output.dim(0) > 16:
            return False
        if weight.dim(1) % 128 != 0:
            return False
        if output.dim(1) % grid_dim[0] != 0:
            return False
        return (output.dim(1) // grid_dim[0]) % 128 == 0

    def _fp8_buffers_for_reduction(self, reduction_size: int):
        mbt = self.max_num_batched_tokens
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size
        if not hasattr(self, '_fp8_bufs'):
            self._fp8_bufs = {}
        cache_key = reduction_size
        if cache_key not in self._fp8_bufs:
            fp8_buf = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=float8_e4m3,
                name=f"fp8_input_{reduction_size}_shared",
                io_category="cuda_tensor",
            )
            # Column-major UE8M0 scale stored as transposed row-major:
            # physical shape=[packed_k, aligned_batch], dtype=uint32.
            packed_k = (num_groups + 3) // 4
            aligned_batch = ((mbt + 3) // 4) * 4
            scale_buf = self.mpk.new_tensor(
                dims=(packed_k, aligned_batch), dtype=uint32,
                name=f"fp8_scale_{reduction_size}_shared",
                io_category="cuda_tensor",
            )
            self._fp8_bufs[cache_key] = (fp8_buf, scale_buf)
        return self._fp8_bufs[cache_key]

    def _fp8_mbt_buffers_for_reduction_f32scale(self, reduction_size: int):
        """Per-token-batch FP8 buffer + float32 scale (NEW kernel format).

        Used by `fp8_gemm_dense_smallm_sm100` / `fp8_gemm_dense_mediumm_sm100`,
        which take row-major float32 scales `[M, K/128]` instead of the
        packed UE8M0 column-major scales used by the older
        `linear_fp8_sm100` kernel. Cache shared across all FP8 GEMMs of the
        same reduction_size, like `_fp8_buffers_for_reduction`.
        """
        mbt = self.max_num_batched_tokens
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size
        if not hasattr(self, "_fp8_mbt_f32_bufs"):
            self._fp8_mbt_f32_bufs = {}
        cache_key = reduction_size
        if cache_key not in self._fp8_mbt_f32_bufs:
            fp8_buf = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=float8_e4m3,
                name=f"fp8_input_v2_{reduction_size}_shared",
                io_category="cuda_tensor",
            )
            scale_buf = self.mpk.new_tensor(
                dims=(mbt, num_groups), dtype=float32,
                name=f"fp8_scale_v2_{reduction_size}_shared",
                io_category="cuda_tensor",
            )
            self._fp8_mbt_f32_bufs[cache_key] = (fp8_buf, scale_buf)
        return self._fp8_mbt_f32_bufs[cache_key]

    def _fp8_dense_num_workers(self, output_size=None):
        """Number of persistent workers each fp8_gemm_dense_{smallm,mediumm} call
        is allowed to occupy.

        B26 (2026-05-15): default lowered from `self.num_workers` (128) to
        80 for DSv3 dual-dispatch (`_use_prefill=True`). On B200 with
        128 workers, the dense GEMM CTA waves frequently have <128 tiles
        worth of real work (qkv_a has 17 tiles, q_b ~48, O_proj ~56),
        so reducing the per-task occupancy from 128 to 80 frees ~48
        worker slots per dense wave to overlap with concurrent ROPE /
        rmsnorm / KV gather tasks. Measured: 412 → 402 us/MoE-layer
        (−2.4%) on the 19-layer TP=4 EP=2 mbt=128 decode iter.

        Per-call-site wave-collapse (2026-06-02): a blanket 80 still over-
        provisions every dense GEMM — each only needs
        `ceil(M_max/BN) * ceil(N/BN)` output tiles (BN=128 fixed in the
        kernel; the kernel strides `bidx = iter*num_workers + worker_idx`
        and idle CTAs `worker_idx >= total` early-return, so dropping
        num_workers to the exact single-wave tile count is BYTE-IDENTICAL —
        the tile→value mapping doesn't depend on num_workers). qkv_a
        (N=2176) needs ceil(2176/128)=17, others larger. Caller passes the
        GEMM's output dim via `output_size`; we return
        `min(80, max(ceil(M_max/128)*ceil(N/128), FLOOR))`. The ~63 idle
        CTAs at 80 wasted dispatch + mbarrier framing on the dispatch-bound
        decode chain and held SMs that could overlap concurrent
        ROPE/rmsnorm/KV-gather. Direct analog of the landed MOE_PERMUTE
        EPC=4 wave-collapse (128→32 CTAs, byte-identical).

        FLOOR=24: experiment_history B26 found num_workers < 16 crashes
        the NVSHMEM barrier; 24 is a safe margin and still collapses qkv_a
        (17→24 is below the old 80). (Sweep note: 64 crashes the nvshmem
        barrier, 80/96 both work but 80 is faster.)

        `_fp8_dense_kv_b_proj` keeps full `num_workers` independently — the
        runtime_m_mode=1 + large M path has tighter constraints and crashes
        at <128.

        Each task strides through output tiles internally, so lowering num_workers
        below the actual tile count just means each task does more iterations.
        For output 1536/128 = 12 tiles, num_workers >= 12 covers all tiles in
        one wave; <12 means each worker handles multiple tiles.
        """
        if not self._use_prefill:
            return self.num_workers
        base = min(80, self.num_workers)
        if output_size is None:
            return base
        # BN=128 fixed in fp8_gemm_dense_{smallm,mediumm}; single wave needs
        # ceil(M_max/128) * ceil(N/128) tiles. M_max = compile-time mbt
        # (runtime_m_ is capped to active_rows at exec, never larger).
        FLOOR = 24
        bn = 128
        m_tiles = (self.max_num_batched_tokens + bn - 1) // bn
        n_tiles = (output_size + bn - 1) // bn
        single_wave = m_tiles * n_tiles
        return min(base, max(single_wave, FLOOR))

    def _fp8_linear_v2(self, input_bf16, weight_fp8_raw, weight_scale_raw,
                       output, residual=None, gate_mode: int = 0,
                       input_row_stride: int = None,
                       input_col_offset: int = 0,
                       share_quantize_tag: str = None,
                       input_fp8_override=None,
                       input_scale_override=None,
                       no_wave_collapse: bool = False):
        """FP8 linear via the NEW dense-GEMM kernel (smallm/mediumm).

        Replaces the old `linear_fp8_sm100` path which has a row-coverage
        bug for batch>16 prefill (rows 1-15 of every output stay zero,
        propagating through attention and MLP). The new kernel was
        introduced for `_fp8_dense_kv_b_proj` and supports any M including
        mbt=128.

        Args
        ----
        input_bf16: (mbt, K) bf16 tensor — pre-quantize input.
        weight_fp8_raw: (N, K) fp8_e4m3 tensor — raw checkpoint weight.
        weight_scale_raw: (N/128, K/128) float32 tensor — checkpoint scale.
            Use `_attach_raw_fp8_weight` (NOT `_attach_fp8_weight`) for the
            weight side so the scale stays in raw float32 layout.
        output: (mbt, N) bf16 tensor.
        residual: optional (mbt, N) bf16. When provided AND world_size>1,
            we GEMM into a partial buffer then AllReduce + add residual.
            Without TP (world_size=1), we GEMM into a partial then add
            residual via elementwise_add (the new kernel has no fused
            residual epilogue).
        gate_mode: 0=always run, 1=prefill phase only, 2=decode phase only
            (mirrors the dense-GEMM `runtime_m_mode`).
        input_row_stride / input_col_offset: when reading a column slice of
            a wider input buffer (QKV-a fused path), specify the parent's
            row stride and the slice's start column. K (= weight.dim(1))
            tells the kernel how many cols to actually quantize per row.
            Defaults preserve legacy contiguous reads.
        share_quantize_tag: B24 (2026-05-15). Dual-dispatch GEMMs reading the
            same input slice (e.g., decode q_b + prefill q_b both reading
            q_a_out[..., :q_lora]) emit two quantize tasks that write
            identical bytes to the shared cached buffer. When both callers
            pass the same `share_quantize_tag`, the first call emits a
            single quantize with active_mode=0 (always run) and subsequent
            calls skip the quantize. Saves one ~5 us wave dispatch per
            shared input per layer.
        no_wave_collapse: when True, force the dense GEMM to keep the
            ORIGINAL (blanket-80 / full) num_workers instead of the
            per-call-site `ceil(N/128)` wave-collapse. Set by call sites
            whose output feeds a downstream `linear_fp8_bmm_*` GEMM — the
            BMM is templated on the per-head N-tile shape and the producing
            GEMM's grid/num_workers must stay at the value that path was
            validated against (the q_b_nope -> q_nope_fp8 -> BMM chain).
        """
        if weight_scale_raw is None:
            raise ValueError("FP8 linear v2 requires FP8 weight scale.")
        if input_bf16.num_dims != 2:
            raise ValueError("FP8 linear v2 expects 2D input.")
        # Output may be 2D (M, N) or 3D (M, H, D_per_head). Storage is
        # row-major contiguous either way; the kernel writes M*N bf16. The
        # 3D form is for the decode BMM path that wants H exposed
        # downstream without an extra reshape task.
        if output.num_dims not in (2, 3):
            raise ValueError("FP8 linear v2 expects 2D or 3D output.")
        if weight_fp8_raw.num_dims != 2 or weight_scale_raw.num_dims != 2:
            raise ValueError("FP8 linear v2 expects 2D weight + scale.")

        # Per-call-site dense GEMM worker count. Normally wave-collapsed to
        # `ceil(N/128)` (see `_fp8_dense_num_workers`); BMM-feeding sites pass
        # `no_wave_collapse=True` to keep the original (blanket-80 / full)
        # count their downstream BMM template was validated against.
        dense_nw = (self._fp8_dense_num_workers()
                    if no_wave_collapse
                    else self._fp8_dense_num_workers(weight_fp8_raw.dim(0)))

        reduction_size = weight_fp8_raw.dim(1)
        # B37 (2026-05-15): caller may pre-allocate the FP8 + scale buffers
        # and bypass the shared cache. Used by the fused
        # rmsnorm+quantize path to give the fused task a unique writer for
        # its FP8/scale outputs — otherwise the shared buffers carry
        # multiple-writer semantics across layers, the fused task becomes a
        # join-consumer, and `build_annotated_graph` flags the embedding
        # producer with case 3 (fork-producer + join-producer).
        if input_fp8_override is not None and input_scale_override is not None:
            input_fp8 = input_fp8_override
            input_scale = input_scale_override
        elif input_fp8_override is None and input_scale_override is None:
            input_fp8, input_scale = self._fp8_mbt_buffers_for_reduction_f32scale(
                reduction_size)
        else:
            raise ValueError(
                "input_fp8_override and input_scale_override must be both "
                "set or both None")
        # B24: emit quantize only on the FIRST call that supplies a given
        # share_quantize_tag this layer; subsequent calls reuse the buffer.
        # The shared-quantize variant uses active_mode=0 (always run) so
        # both decode and prefill iters see fresh data. The non-shared path
        # mirrors gate_mode into the quantize active_mode (existing
        # behavior).
        emit_quantize = True
        if share_quantize_tag is not None:
            already = getattr(self, "_fp8_quantize_emitted", set())
            if share_quantize_tag in already:
                emit_quantize = False
            else:
                already.add(share_quantize_tag)
                self._fp8_quantize_emitted = already

        if emit_quantize:
            active_mode = (
                0 if share_quantize_tag is not None
                else (2 if gate_mode == 1
                      else 3 if gate_mode == 2
                      else 0)
            )
            quantize_kwargs = {}
            if input_row_stride is not None or input_col_offset != 0:
                quantize_kwargs["hidden_size_override"] = reduction_size
                quantize_kwargs["input_stride_override"] = (
                    input_row_stride if input_row_stride is not None
                    else input_bf16.dim(1))
                quantize_kwargs["in_offset_elems"] = input_col_offset
            self.mpk.quantize_fp8_layer(
                input=input_bf16,
                output_fp8=input_fp8,
                output_scale=input_scale,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
                scale_ue8m0=False,
                active_mode=active_mode,
                **quantize_kwargs,
            )

        # B20 (2026-05-15): mirror gate_mode into the GEMM kernel itself so
        # the dual-dispatch O_proj branches early-exit the wave for the
        # wrong phase. Otherwise both prefill and decode O_proj GEMMs run
        # every iter (~30-50 μs each of wasted MMA wave on the unused
        # branch) — visible in perfetto as a 90 μs bubble after the
        # MLA attention path.
        gemm_runtime_m_mode = (2 if gate_mode == 1
                               else 3 if gate_mode == 2
                               else 0)

        if residual is None:
            dsv3_tasks.fp8_gemm_dense_layer(
                self.mpk,
                input_fp8=input_fp8,
                weight_fp8=weight_fp8_raw,
                input_scale=input_scale,
                weight_scale=weight_scale_raw,
                output=output,
                num_workers=dense_nw,
                runtime_m_mode=gemm_runtime_m_mode,
            )
            return

        if self.world_size > 1:
            idx = getattr(self, "_tp_residual_linear_idx", 0)
            self._tp_residual_linear_idx = idx + 1
            partial = self._new_tp_partial(output, f"tp_v2_residual_partial_{idx}")
            dsv3_tasks.fp8_gemm_dense_layer(
                self.mpk,
                input_fp8=input_fp8,
                weight_fp8=weight_fp8_raw,
                input_scale=input_scale,
                weight_scale=weight_scale_raw,
                output=partial,
                num_workers=dense_nw,
                runtime_m_mode=gemm_runtime_m_mode,
            )
            self._allreduce_residual(partial, output, residual,
                                     gate_mode=gate_mode)
            return

        # TP=1 path: dense GEMM into a partial buffer, then add residual.
        partial = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, weight_fp8_raw.dim(0)),
            dtype=bfloat16, name=f"fp8_v2_partial_{id(weight_fp8_raw)}",
            io_category="cuda_tensor",
        )
        dsv3_tasks.fp8_gemm_dense_layer(
            self.mpk,
            input_fp8=input_fp8,
            weight_fp8=weight_fp8_raw,
            input_scale=input_scale,
            weight_scale=weight_scale_raw,
            output=partial,
            num_workers=dense_nw,
            runtime_m_mode=gemm_runtime_m_mode,
        )
        self.mpk.elementwise_add_layer(
            input_a=partial,
            input_b=residual,
            output=output,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

    def _fp8_sequence_buffers_for_reduction(
        self, reduction_size: int, tag: str = "shared"
    ):
        # DEBUG 2026-05-10: pad row dim to multiple of 128 (chunked prefill
        # TMA box BN_BOX=128) to avoid OOB NaN-fill propagation in PV MMA.
        _raw_rows = (self.mpk.max_num_batched_requests
                     * self.mpk.max_seq_length)
        rows = ((_raw_rows + 127) // 128) * 128
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size
        if not hasattr(self, '_fp8_seq_bufs'):
            self._fp8_seq_bufs = {}
        cache_key = (rows, reduction_size, tag)
        if cache_key not in self._fp8_seq_bufs:
            fp8_buf = self.mpk.new_tensor(
                dims=(rows, reduction_size), dtype=float8_e4m3,
                name=f"fp8_seq_input_{reduction_size}_{tag}",
                io_category="cuda_tensor",
            )
            scale_buf = self.mpk.new_tensor(
                dims=(rows, num_groups), dtype=float32,
                name=f"fp8_seq_scale_{reduction_size}_{tag}",
                io_category="cuda_tensor",
            )
            self._fp8_seq_bufs[cache_key] = (fp8_buf, scale_buf)
        return self._fp8_seq_bufs[cache_key]


    def _fp8_linear(self, input_bf16, weight, weight_scale, output,
                     grid_dim, block_dim, residual=None, gate_mode: int = 0,
                     input_row_stride: int = None,
                     input_col_offset: int = 0,
                     share_quantize_tag: str = None,
                     input_fp8_override=None,
                     input_scale_override=None,
                     no_wave_collapse: bool = False):
        """Quantize BF16 input → FP8, then run FP8 GEMM.

        Now routes through the new `fp8_gemm_dense_smallm/mediumm_sm100`
        kernels (`_fp8_linear_v2`) instead of the older `linear_fp8_sm100`,
        which has a row-coverage bug for batch>16 prefill (rows 1-15 of
        every output stay zero). The `grid_dim`/`block_dim` arguments are
        accepted for API compatibility but ignored — the new kernel uses
        a persistent `(num_workers, 1, 1)` grid internally.
        """

        if weight_scale is None:
            # BF16 fallback is kept for fixtures or pre-converted weights that
            # intentionally arrive without FP8 scale metadata.
            if residual is not None:
                if self.world_size > 1:
                    idx = getattr(self, "_tp_residual_linear_idx", 0)
                    self._tp_residual_linear_idx = idx + 1
                    partial = self._new_tp_partial(output, f"tp_bf16_residual_partial_{idx}")
                    self.mpk.linear_layer(
                        input=input_bf16, weight=weight, output=partial,
                        grid_dim=grid_dim, block_dim=block_dim)
                    self._allreduce_residual(partial, output, residual,
                                             gate_mode=gate_mode)
                else:
                    self.mpk.linear_with_residual_layer(
                        input=input_bf16, weight=weight, residual=residual,
                        output=output, grid_dim=grid_dim, block_dim=block_dim)
            else:
                self.mpk.linear_layer(
                    input=input_bf16, weight=weight, output=output,
                    grid_dim=grid_dim, block_dim=block_dim)
            return

        # Route to the new dense FP8 GEMM kernel (smallm/mediumm).
        self._fp8_linear_v2(
            input_bf16=input_bf16,
            weight_fp8_raw=weight,
            weight_scale_raw=weight_scale,
            output=output,
            residual=residual,
            gate_mode=gate_mode,
            input_row_stride=input_row_stride,
            input_col_offset=input_col_offset,
            share_quantize_tag=share_quantize_tag,
            input_fp8_override=input_fp8_override,
            input_scale_override=input_scale_override,
            no_wave_collapse=no_wave_collapse,
        )

    def _fused_rmsnorm_quantize_qkv_a_tag(self, layer_idx: int) -> str:
        """B37: deterministic share_quantize_tag for the fused
        rmsnorm + qkv_a-quantize path. Pre-populated in
        `_fp8_quantize_emitted` so the qkv_a `_fp8_linear` call skips its
        internal quantize (we already wrote the FP8 + scale buffers in
        the fused task)."""
        return f"layer_{layer_idx}_qkv_a_fused_rmsnorm_quantize"

    def _emit_fused_rmsnorm_qkv_a_quantize(self,
                                            input_x: 'DTensor',
                                            w_norm: 'DTensor',
                                            layer_idx: int,
                                            reduction_size: int) -> str:
        """B37: write self.rmsnorm_out (BF16) AND the qkv_a-side FP8 input
        + float32 scale buffers in one fused task.

        Returns the share_quantize_tag the caller MUST forward to
        `_fp8_linear(share_quantize_tag=...)`.

        Buffer ownership (2026-05-15 case-3 fix): unlike the standalone
        quantize path (which uses the cross-layer SHARED FP8/scale
        buffers via `_fp8_mbt_buffers_for_reduction_f32scale`), the fused
        task here allocates **per-layer-unique** FP8 + scale buffers.

        Why: the fused task takes its FP8/scale outputs as store_in_dmem
        inputs in the task graph (the kernel's "outputs" are wired as
        input slots — MPK convention). If those buffers are shared across
        layers (multiple writers across the megakernel's task list), the
        annotated-graph builder sees the fused task as a join-consumer,
        and the producer of its `input` slot (embedding at layer 0) ends
        up as both a fork-producer (other consumers exist) and a
        join-producer (this consumer is a join-consumer) — case 3
        violation. With per-layer-unique buffers the fused task has a
        unique writer per buffer, breaks the join, and the AG accepts.

        Cost: extra `mbt * K + mbt * (K/128) * 4` bytes per layer (~14 KB
        for K=7168 at mbt=128). Trade-off vs the B23/B24 dedup pattern is
        accepted because qkv_a has no other quantize peer to dedup with
        (kv_b/q_b use different K).

        The per-layer buffers are stashed on `self._fused_qkv_a_bufs`
        keyed by layer_idx so the qkv_a `_fp8_linear` call site can pull
        them as `input_fp8_override` / `input_scale_override`.
        """
        mbt = self.max_num_batched_tokens
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size
        if not hasattr(self, "_fused_qkv_a_bufs"):
            self._fused_qkv_a_bufs = {}
        if not hasattr(self, "_fused_rmsnorm_out_per_layer"):
            self._fused_rmsnorm_out_per_layer = {}
        if layer_idx not in self._fused_qkv_a_bufs:
            input_fp8 = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=float8_e4m3,
                name=f"fused_qkv_a_fp8_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
            input_scale = self.mpk.new_tensor(
                dims=(mbt, num_groups), dtype=float32,
                name=f"fused_qkv_a_scale_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
            self._fused_qkv_a_bufs[layer_idx] = (input_fp8, input_scale)
        if layer_idx not in self._fused_rmsnorm_out_per_layer:
            # Per-layer-unique BF16 rmsnorm output buffer. The shared
            # `self.rmsnorm_out` is reused across all layers, so the AG
            # sees layer N's fused task reading from layer (N-1)'s write
            # via store_in_dmem — making the fused task a join-consumer
            # (case-3 trigger at embedding). With per-layer unique buffer,
            # only ONE writer per buffer.
            self._fused_rmsnorm_out_per_layer[layer_idx] = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=bfloat16,
                name=f"fused_rmsnorm_out_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
        input_fp8, input_scale = self._fused_qkv_a_bufs[layer_idx]
        rmsnorm_out_bf16 = self._fused_rmsnorm_out_per_layer[layer_idx]
        # Pre-populate the emitted-set so _fp8_linear_v2 will skip the
        # internal quantize call that would otherwise overwrite the
        # fused-task output bytes with identical-but-redundant work.
        already = getattr(self, "_fp8_quantize_emitted", set())
        tag = self._fused_rmsnorm_quantize_qkv_a_tag(layer_idx)
        already.add(tag)
        self._fp8_quantize_emitted = already

        # Float32 scale path (matches _fp8_mbt_buffers_for_reduction_f32scale
        # layout; dense GEMM consumes (M, K/128) f32 row-major scales).
        # `output_bf16` uses the per-layer-unique buffer (case-3 fix).
        # emit_bf16=False because nothing downstream reads the bf16 in fused
        # mode (qkv_a GEMM reads fp8/scale directly).
        dsv3_tasks.fused_rmsnorm_quantize_fp8_layer(
            self.mpk,
            input=input_x,
            weight=w_norm,
            output_bf16=rmsnorm_out_bf16,
            output_fp8=input_fp8,
            output_scale=input_scale,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
            scale_ue8m0=False,
            emit_bf16=False,
        )
        return tag

    def _fused_q_a_quantize_tag(self, layer_idx: int) -> str:
        """Deterministic tag for the q_a-layernorm + q_b-input-quantize
        fusion. Pre-populated in `_fp8_quantize_emitted` so downstream
        q_b `_fp8_linear` calls (BMM=1 and dual-dispatch prefill variants)
        skip their internal quantize and read the FP8/scale the fused
        task already wrote.
        """
        return f"layer_{layer_idx}_q_a_fused_quantize"

    def _emit_fused_q_a_rmsnorm_quantize(self,
                                          input_x: 'DTensor',
                                          w_norm: 'DTensor',
                                          layer_idx: int,
                                          reduction_size: int,
                                          in_offset_elems: int,
                                          out_offset_elems: int) -> tuple:
        """2026-05-28: fused q_a_layernorm + per-token-group FP8 quantize.

        Analogous to `_emit_fused_rmsnorm_qkv_a_quantize` (B37) but for the
        INNER q_a layernorm (after qkv_a GEMM, before q_b GEMMs). Reduces
        the chain hop count by collapsing the 2-task chain (rmsnorm_layer
        + q_b's internal quantize_fp8) into 1 fused task. Saves ~2-3μs/
        layer on the decode critical-path chain (the +1.4 + +1.6 + part
        of the +5.2μs gaps the analyzer flagged before q_b_GEMM).

        Returns `(input_fp8, input_scale, tag)` — the caller threads
        input_fp8/scale to q_b `_fp8_linear(..., input_fp8_override=...,
        input_scale_override=...)` calls. The tag is also pre-populated
        in `_fp8_quantize_emitted` so a `share_quantize_tag=tag` arg makes
        downstream callers skip their quantize emission.

        Buffer ownership (case-3 fix, B37 pattern):
          The fused task takes its FP8/scale outputs as `store_in_dmem`
          inputs in the task graph. Per-layer-unique buffers prevent the
          fused task from being a cross-layer join-consumer.

        emit_bf16=False: nothing downstream reads the rmsnormed BF16 q_a
        as BF16. The q_b GEMM reads the FP8 (via the share_quantize_tag
        + input_fp8_override threading); no other consumer of q_a_out's
        q_a slice exists (verified by grep at builder.py audit time).
        Skipping emit_bf16 saves an HBM round-trip too.

        scale_ue8m0=False (float32): matches the new dense GEMM family's
        expected scale layout (`_fp8_mbt_buffers_for_reduction_f32scale`).
        """
        mbt = self.max_num_batched_tokens
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size
        if not hasattr(self, "_fused_q_a_bufs"):
            self._fused_q_a_bufs = {}
        if layer_idx not in self._fused_q_a_bufs:
            input_fp8 = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=float8_e4m3,
                name=f"fused_q_a_fp8_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
            input_scale = self.mpk.new_tensor(
                dims=(mbt, num_groups), dtype=float32,
                name=f"fused_q_a_scale_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
            self._fused_q_a_bufs[layer_idx] = (input_fp8, input_scale)
        input_fp8, input_scale = self._fused_q_a_bufs[layer_idx]
        # Pre-populate the emitted-set so _fp8_linear_v2 will skip the
        # internal quantize call that would otherwise overwrite the
        # fused-task output bytes with redundant work.
        already = getattr(self, "_fp8_quantize_emitted", set())
        tag = self._fused_q_a_quantize_tag(layer_idx)
        already.add(tag)
        self._fp8_quantize_emitted = already

        # output_bf16: required argument but emit_bf16=False means the
        # kernel never writes to it (we still need to pass a valid tensor;
        # the input itself satisfies the dim assertions).
        dsv3_tasks.fused_rmsnorm_quantize_fp8_layer(
            self.mpk,
            input=input_x,
            weight=w_norm,
            output_bf16=input_x,   # placeholder; emit_bf16=False skips write
            output_fp8=input_fp8,
            output_scale=input_scale,
            grid_dim=_rmsnorm_grid(mbt),
            block_dim=(128, 1, 1),
            process_dim=reduction_size,
            in_offset_elems=in_offset_elems,
            out_offset_elems=out_offset_elems,
            scale_ue8m0=False,
            emit_bf16=False,
        )
        return input_fp8, input_scale, tag

    def _bmm_decode_q_path(self, state_dict, attn, layer_idx, qb_slice_kwargs,
                           qb_share_tag=None,
                           qb_input_fp8_ovr=None,
                           qb_input_scale_ovr=None):
        """Decode Q path: replaces the absorbed q_b_proj decode GEMM with a
        per-head BMM chain that loads the unabsorbed weights at runtime:

          rmsnorm_linear(q_a, q_b_nope)   → q_nope (mbt, H, 128)  bf16
          rmsnorm_linear(q_a, q_b_pe)     → q_pe   (mbt, H, 64)   bf16
          quantize_fp8(q_nope, UE8M0)     → q_nope_fp8, q_nope_scale
          linear_fp8_bmm(q_nope_fp8, kv_b_k_bmm) → q_nope_abs (mbt, H, 512)
          assemble_q_decode(q_nope_abs, q_pe) → q_nope_pe (mbt, H, 576)

        Win over the absorbed monolith (single (H*576, q_lora) FP8 GEMM):
        per-task weight load drops from (576, q_lora=1536) per head-tile to
        (128, q_lora) + (64, q_lora) + (512, 128) per head, materially less
        TMA traffic per CTA. The absorbed weight buffer (~6.8 GB across
        DSv3 layers) also goes away.
        """
        # Per-layer reusable 3D buffers. Sized to mbt × H_local.
        H_local = self.num_local_q_heads
        mbt = self.max_num_batched_tokens
        if not hasattr(self, "_bmm_decode_buffers"):
            self._bmm_decode_buffers = {}
            # bf16 output of the q_b_pe FP8 dense GEMM (3D so the BMM input
            # partition map can see H as an explicit dim).
            self._bmm_decode_buffers["q_pe_3d"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 64), dtype=bfloat16,
                name="q_pe_decode_3d", io_category="cuda_tensor")
            # FP8 q_nope + UE8M0 packed scale for BMM input. K=128 ≤ 512
            # so packed_K = 1 (one uint32 per row).
            self._bmm_decode_buffers["q_nope_fp8"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 128), dtype=float8_e4m3,
                name="q_nope_decode_fp8", io_category="cuda_tensor")
            self._bmm_decode_buffers["q_nope_scale"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 1), dtype=uint32,
                name="q_nope_decode_scale", io_category="cuda_tensor")
            # BMM output FUSE: instead of a separate (mbt, H, 512) buffer,
            # attach a slice view of q_nope_pe[:, :, :512] (parent is
            # (mbt, H, 576) torch tensor). The slice has strides (H*576,
            # 576, 1) which matches what BMM TMA needs to write each head's
            # 512 nope cols at the [h*576:h*576+512] slot of the wider
            # buffer. This eliminates the separate q_nope_abs allocation
            # and lets BMM output directly land in the per-head
            # [nope|pe] interleaved layout.
            q_nope_abs_view = self._q_nope_pe_torch[:, :, :512]
            self._bmm_decode_buffers["q_nope_abs"] = self.mpk.attach_input(
                q_nope_abs_view, name="q_nope_abs_view")
        q_pe_3d = self._bmm_decode_buffers["q_pe_3d"]
        q_nope_fp8 = self._bmm_decode_buffers["q_nope_fp8"]
        q_nope_scale = self._bmm_decode_buffers["q_nope_scale"]
        q_nope_abs = self._bmm_decode_buffers["q_nope_abs"]

        w_q_b_nope, s_q_b_nope = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_nope.weight",
            f"layer_{layer_idx}_q_b_nope_decode")
        w_q_b_pe, s_q_b_pe = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_pe.weight",
            f"layer_{layer_idx}_q_b_pe_decode")
        # 1) q_b_pe FIRST so its _fp8_linear emits the q_a input-side
        # quantize task (shared via qb_share_tag with q_b_nope's downstream
        # consumer): the input-quantize task must already be emitted before
        # the fused q_b_nope GEMM below, since the fused GEMM reads the same
        # q_a FP8 buffer and skips the redundant quantize via the share tag.
        self._fp8_linear(
            self.q_a_out, w_q_b_pe, s_q_b_pe, q_pe_3d,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_pe.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
            gate_mode=2 if self._use_prefill else 0,
            share_quantize_tag=qb_share_tag,
            input_fp8_override=qb_input_fp8_ovr,
            input_scale_override=qb_input_scale_ovr,
            **qb_slice_kwargs)
        # 2+3 fused, D1) q_b_nope FP8 dense GEMM with epilogue UE8M0 quantize
        # → q_nope_fp8 + q_nope_scale directly. Reads q_a's FP8 / scale
        # from the shared cache (the q_b_pe call above already emitted
        # the quantize). Replaces the (bf16 q_b_nope GEMM →
        # quantize_fp8) two-task chain with one task; saves ~9 μs/layer
        # on the BMM Q-up critical path.
        # 2026-05-28: the fused q_a layernorm+quantize task wrote the
        # FP8/scale into the per-layer buffers passed via qb_input_fp8_ovr /
        # qb_input_scale_ovr. Use those instead of the cache (which is empty
        # in fusion mode — the q_b_pe call above also bypassed the cache via
        # input_fp8_override).
        reduction_size = w_q_b_nope.dim(1)
        if qb_input_fp8_ovr is not None and qb_input_scale_ovr is not None:
            input_fp8_buf, input_scale_buf = (qb_input_fp8_ovr, qb_input_scale_ovr)
        else:
            input_fp8_buf, input_scale_buf = (
                self._fp8_mbt_buffers_for_reduction_f32scale(reduction_size))
        gemm_runtime_m_mode = 3 if self._use_prefill else 0
        dsv3_tasks.fp8_gemm_dense_layer(
            self.mpk,
            input_fp8=input_fp8_buf,
            weight_fp8=w_q_b_nope,
            input_scale=input_scale_buf,
            weight_scale=s_q_b_nope,
            fp8out=True,
            output_fp8=q_nope_fp8,
            output_scale=q_nope_scale,
            # BMM-feeding GEMM: q_nope_fp8 -> linear_fp8_bmm (kv_b_k). Keep
            # the ORIGINAL num_workers (no per-call-site wave-collapse) so
            # the downstream BMM template's validated grid is preserved.
            num_workers=self._fp8_dense_num_workers(),
            runtime_m_mode=gemm_runtime_m_mode,
        )
        # 4) BMM(q_nope_fp8, kv_b_k_bmm) → q_nope_abs (mbt, H, 512).
        # swapAB body (dense=False): UE8M0-packed scales, D_out shardable.
        w_kvk_bmm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_b_k_bmm.weight"],
            name=f"layer_{layer_idx}_kv_b_k_bmm")
        s_kvk_bmm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_b_k_bmm.weight_scale_ue8m0"],
            name=f"layer_{layer_idx}_kv_b_k_bmm_scale")
        dsv3_tasks.linear_fp8_bmm_layer(
            self.mpk,
            input_fp8=q_nope_fp8,
            input_scale=q_nope_scale,
            weight_fp8=w_kvk_bmm,
            weight_scale=s_kvk_bmm,
            output=q_nope_abs,
            grid_dim=(512 // 128, H_local, 1),  # (4, H, 1)
            block_dim=(256, 1, 1),
            dense=False,
        )
        # 5) Assemble (PE-only): BMM already wrote nope into q_nope_pe[:, :, :512]
        # via the q_nope_abs slice-view fuse, so the assemble step only needs
        # to write q_pe into the tail [512:576]. Half the per-CTA traffic.
        dsv3_tasks.assemble_q_decode_sm100_layer(
            self.mpk,
            q_nope_abs=q_nope_abs,
            q_pe=q_pe_3d,
            q_nope_pe=self.q_nope_pe,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
            pe_only=True,
        )

    def _bmm_decode_o_path(self, state_dict, attn, layer_idx, residual):
        """C9 (2026-05-16): post-attn decode BMM path.

        Replaces the load-time-absorbed decode o_proj (fused with W_UV)
        with runtime BMM + smaller linear:
          quantize(attn_out)           → attn_out_fp8 (mbt, H, 512) FP8
          BMM(attn_out_fp8, kv_b_v_bmm_dense) → attn_out_reduced (mbt, H, 128)
          fp8_linear_with_residual(attn_out_reduced, o_proj_original) → attn_proj_out

        The BMM goes through the DENSE block-scaled GEMM body (float32
        128-K-group scales — split-K-friendly, unlike swapAB's 512-K-packed
        UE8M0). The o_proj_original.weight is the SAME weight used by the
        prefill path (hidden × H*128, FP8). After BMM, decode + prefill both
        use the smaller unabsorbed o_proj.

        Gate: this entire path runs only on decode iters (Q_LEN<=8) via
        the FP8 linear's gate_mode=2 + BMM's MMA_N=16 decode constraint.

        Returns: None (writes attn_proj_out directly).
        """
        H_local = self.num_local_q_heads
        mbt = self.max_num_batched_tokens
        V_HEAD_DIM = 128  # post-attn V un-absorption dim per head
        KV_LORA = 512     # current attn_out per-head dim

        # nk = number of 128-K groups in the per-head reduction (= 512/128 = 4)
        # for the float32 activation-scale buffer.
        nk_o = (KV_LORA + 127) // 128
        if not hasattr(self, "_bmm_decode_o_buffers"):
            self._bmm_decode_o_buffers = {}
            # FP8 of attn_out. K=512.
            self._bmm_decode_o_buffers["attn_out_fp8"] = self.mpk.new_tensor(
                dims=(mbt, H_local, KV_LORA), dtype=float8_e4m3,
                name="attn_out_bmm_fp8", io_category="cuda_tensor")
            # float32 1x128-group activation scale [mbt, H, nk].
            self._bmm_decode_o_buffers["attn_out_scale_f32"] = self.mpk.new_tensor(
                dims=(mbt, H_local, nk_o), dtype=float32,
                name="attn_out_bmm_scale_f32", io_category="cuda_tensor")
            # BMM output: reduced attn (mbt, H, 128). Allocate as 2D so it
            # feeds directly into _fp8_linear without a reshape — BMM
            # wrapper accepts 2D or 3D output per its docstring.
            self._bmm_decode_o_buffers["attn_out_reduced"] = self.mpk.new_tensor(
                dims=(mbt, H_local * V_HEAD_DIM), dtype=bfloat16,
                name="attn_out_reduced_2d", io_category="cuda_tensor")

        attn_out_fp8 = self._bmm_decode_o_buffers["attn_out_fp8"]
        attn_out_scale_f32 = self._bmm_decode_o_buffers["attn_out_scale_f32"]
        attn_out_reduced = self._bmm_decode_o_buffers["attn_out_reduced"]

        active_mode_o = 3 if self._use_prefill else 0  # decode-only on dual-dispatch

        # Step 1: quantize attn_out BF16 → FP8 + float32 1x128-group scale
        # [mbt, H, nk]. Input self.attn_out is (mbt, H*KV_LORA) 2D; output
        # FP8 is 3D (mbt, H, KV_LORA), same byte layout; the float32 scale
        # is row-major [batch, num_groups] = [mbt*H, nk] which views as
        # [mbt, H, nk].
        self.mpk.quantize_fp8_layer(
            input=self.attn_out,
            output_fp8=attn_out_fp8,
            output_scale=attn_out_scale_f32,
            grid_dim=(1, mbt * H_local, 1),
            block_dim=(128, 1, 1),
            scale_ue8m0=False,
            active_mode=active_mode_o,
        )
        # Step 2: per-head BMM via the DENSE block-scaled GEMM body.
        # kv_b_v_bmm_dense prepared in demo.py: weight (H, 128, 512) FP8 +
        # float32 block scale (H, 1, nk).
        w_kvv_bmm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_b_v_bmm_dense.weight"],
            name=f"layer_{layer_idx}_kv_b_v_bmm_dense")
        s_kvv_bmm = self.mpk.attach_input(
            torch_tensor=state_dict[
                f"{attn}kv_b_v_bmm_dense.weight_scale_inv"],
            name=f"layer_{layer_idx}_kv_b_v_bmm_dense_scale")
        dsv3_tasks.linear_fp8_bmm_layer(
            self.mpk,
            input_fp8=attn_out_fp8,
            input_scale=attn_out_scale_f32,
            weight_fp8=w_kvv_bmm,
            weight_scale=s_kvv_bmm,
            output=attn_out_reduced,
            grid_dim=(1, H_local, 1),
            block_dim=(256, 1, 1),
            dense=True,
        )

        # Step 3: smaller o_proj linear with residual.
        # Use the o_proj_original.weight (FP8, hidden × H*128, saved by demo.py
        # before W_UV fusion). gate_mode=2 = decode-only.
        w_o_orig, s_o_orig = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj_original.weight",
            f"layer_{layer_idx}_o_proj_original_bmm")
        self._fp8_linear(
            attn_out_reduced,
            w_o_orig,
            s_o_orig,
            self.attn_proj_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
            block_dim=(128, 1, 1),
            residual=residual,
            gate_mode=2 if self._use_prefill else 0,
        )

    def _fp8_dense_kv_b_proj(
        self, ckv, weight, weight_scale, output, tag: str,
        shared_quantize_tag: str = None,
    ):
        if weight_scale is None:
            raise ValueError("kv_b prefill projection requires FP8 weight scale.")
        # B23 (2026-05-15): kv_b_k and kv_b_v both quantize the SAME ckv_sep
        # to FP8 with the SAME group_size — emitting two quantize tasks
        # writes the same bytes twice. When `shared_quantize_tag` is given
        # we reuse the buffer that the previous call already wrote, and
        # skip the quantize task entirely (only emit the GEMM). Caller
        # must pass the same `shared_quantize_tag` to both kv_b_k and
        # kv_b_v calls in the same layer; the FIRST call emits quantize,
        # the SECOND reuses.
        if shared_quantize_tag is not None:
            input_fp8, input_scale = self._fp8_sequence_buffers_for_reduction(
                self.kv_lora_rank, tag=shared_quantize_tag)
            already_quantized = getattr(self, "_kv_b_quantized_tags", set())
            if shared_quantize_tag not in already_quantized:
                self.mpk.quantize_fp8_layer(
                    input=ckv,
                    output_fp8=input_fp8,
                    output_scale=input_scale,
                    grid_dim=(input_fp8.dim(0), 1, 1),
                    block_dim=(128, 1, 1),
                    scale_ue8m0=False,
                    active_mode=1,
                )
                already_quantized.add(shared_quantize_tag)
                self._kv_b_quantized_tags = already_quantized
        else:
            input_fp8, input_scale = self._fp8_sequence_buffers_for_reduction(
                self.kv_lora_rank, tag=tag)
            self.mpk.quantize_fp8_layer(
                input=ckv,
                output_fp8=input_fp8,
                output_scale=input_scale,
                grid_dim=(input_fp8.dim(0), 1, 1),
                block_dim=(128, 1, 1),
                scale_ue8m0=False,
                active_mode=1,
            )
        # Chunked-prefill kv_b_k/v: keep at full self.num_workers (NOT the
        # wave-collapsed `_fp8_dense_num_workers`). The runtime_m_mode=1 +
        # larger M_total path has tighter constraints and crashes at low
        # num_workers (tested 48/64 both fail).
        dsv3_tasks.fp8_gemm_dense_layer(
            self.mpk,
            input_fp8=input_fp8,
            weight_fp8=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output=output,
            num_workers=self.num_workers,
            runtime_m_mode=1,
        )


    @staticmethod





    def _silu_mul_fp8_linear(self, silu_input, silu_bf16_output, weight,
                             weight_scale, output, silu_grid_dim,
                             linear_grid_dim, block_dim, residual=None):
        self.mpk.silu_mul_layer(
            input=silu_input,
            output=silu_bf16_output,
            grid_dim=silu_grid_dim,
            block_dim=(128, 1, 1),
        )
        self._fp8_linear(
            silu_bf16_output,
            weight,
            weight_scale,
            output,
            grid_dim=linear_grid_dim,
            block_dim=block_dim,
            residual=residual,
        )

    def _precompute_rope_embeddings(self):
        """Precompute vLLM/SGLang-aligned DeepSeek-V3 RoPE tables."""
        rope_dim = QK_ROPE_HEAD_DIM  # 64
        max_seq = self.mpk.max_seq_length
        half = rope_dim // 2
        rope_params = self.rope_parameters or {}
        rope_type = rope_params.get("rope_type", rope_params.get("type", "default"))
        factor = float(rope_params.get("factor", 1.0))
        base = float(self.rope_theta)

        pos_freqs = base ** (
            torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        if rope_type in ("yarn", "deepseek_yarn", "deepseek_llama_scaling"):
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (factor * pos_freqs)
            beta_fast = int(rope_params.get("beta_fast", 32))
            beta_slow = int(rope_params.get("beta_slow", 1))
            low, high = _yarn_find_correction_range(
                beta_fast,
                beta_slow,
                rope_dim,
                base,
                int(self.original_max_position_embeddings),
            )
            if low == high:
                high += 0.001
            ramp = torch.clamp(
                (torch.arange(half, dtype=torch.float32) - low) / (high - low),
                0,
                1,
            )
            extrapolation_factor = float(
                rope_params.get("extrapolation_factor", 1.0))
            inv_freq_mask = (1 - ramp) * extrapolation_factor
            freqs = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
            attn_factor = float(rope_params.get("attn_factor", 1.0))
            mscale = (
                _yarn_get_mscale(factor, float(rope_params.get("mscale", 1.0)))
                / _yarn_get_mscale(
                    factor, float(rope_params.get("mscale_all_dim", 0.0)))
                * attn_factor
            )
        else:
            freqs = 1.0 / pos_freqs
            mscale = 1.0

        positions = torch.arange(max_seq, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # [max_seq, half]
        # vLLM/SGLang run DeepSeek MLA RoPE with interleaved/GPT-J semantics
        # (is_neox_style=False): pair dims (0,1), (2,3), ... and use
        # repeat_interleave cos/sin. The local HF checkpoint first permutes the
        # interleaved tensor to half layout and then applies rotate_half; that
        # is mathematically equivalent for QK dot products, but the physical
        # tensor layout differs. Keep MPK aligned with vLLM/SGLang.
        # Keep PyTorch tensors alive on self — the persistent kernel stores
        # raw GPU pointers, so the tensors must not be garbage-collected.
        self._rope_cos_buf = (angles.cos() * mscale).repeat_interleave(
            2, dim=-1).to(dtype=torch.bfloat16, device="cuda")
        self._rope_sin_buf = (angles.sin() * mscale).repeat_interleave(
            2, dim=-1).to(dtype=torch.bfloat16, device="cuda")
        # Attach as DTensors
        self.cos_pos_embed = self.mpk.attach_input(
            torch_tensor=self._rope_cos_buf, name="rope_cos")
        self.sin_pos_embed = self.mpk.attach_input(
            torch_tensor=self._rope_sin_buf, name="rope_sin")

    def _new_intermediate_tensors(self):
        """Allocate intermediate computation buffers."""
        mbt = self.max_num_batched_tokens

        # MBT caps prefill chunk size. Decode/verify work is bounded by
        # MTP+1 tokens per request and should not force prefill kernels into
        # the graph. Prompt tails with q_len > 8 still use the prefill path
        # when MBT is a real chunk size; q_len <= 8 tails use decode kernels.
        self._use_prefill = mbt > 8
        # Direct-paged decode skips the dense KV gather copy. TP decode kernels
        # consume the runtime page table directly; TP1 still relies on physical
        # page order, so only enable that legacy shortcut for the single-request
        # demo path.
        # TP8 direct-paged decode currently hangs in the end-to-end DeepSeek
        # demo on V4, so keep TP8 on the dense gather path until that variant
        # is fixed and validated separately.
        direct_paged_tp_decode = self.world_size in (2, 4)
        direct_paged_tp1_decode = (
            self.world_size == 1
            and self.mpk.max_num_batched_requests == 1
            and self.mpk.total_num_requests == 1
        )
        self._direct_paged_decode_kv = (
            self.mpk.page_size == 128
            and (direct_paged_tp_decode or direct_paged_tp1_decode)
        )
        if self._use_prefill:
            print(f"  [MLA path] MBT={mbt} → MLA prefill + runtime-gated decode")
        else:
            print(f"  [MLA path] Q_LEN={mbt} → MLA decode / MTP decode")

        # RMSNorm output
        self.rmsnorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )

        # MLA projections — QKV-a fusion (landed 2026-05-13, made default).
        # Single qkv_a_out (mbt, QKV_A_FUSED_N) buffer; the fused FP8 GEMM
        # writes q_a + c_latent + k_pe in one task and downstream consumers
        # read their slice via (row_stride, offset) kernel params. The 3-GEMM
        # unfused path was removed (was +14.9% slower e2e at 19-layer scale).
        # Layout per row:
        #   cols [0    : 1536) = q_a_out      (q_lora_rank = 1536)
        #   cols [1536 : 2048) = c_latent_out (kv_lora_rank = 512)
        #   cols [2048 : 2112) = k_pe_out real (QK_ROPE_HEAD_DIM = 64)
        #   cols [2112 : 2176) = k_pe_out zero pad (= MMA_M tail)
        qkv_a_total = QKV_A_FUSED_N
        self.qkv_a_out = self.mpk.new_tensor(
            dims=(mbt, qkv_a_total),
            dtype=bfloat16, name="qkv_a_out", io_category="cuda_tensor",
        )
        # C20 (2026-05-17): each logical slot is a `mpk.narrow` view of
        # qkv_a_out. The view bakes the slot's byte offset into base_ptr
        # (via view_offset) and inherits the parent row stride into
        # view.stride[0]. Consumers read the slot from base_ptr (no extra
        # in_offset needed) and walk rows by view.stride[0]; task_register,
        # the FP8 TMA descriptor builder (tma.cuh) and annotated_graph's
        # 2D bbox window-overlap check were all updated in
        # P1/P2/B3/bbox commits to consume the view metadata uniformly.
        # The explicit *_offset / row_stride params remaining at callsites
        # become decorative — they encode 0 offset + parent row stride,
        # which matches what the view already supplies.
        self._qkv_a_row_stride = qkv_a_total
        self._qkv_a_q_offset = 0
        self._qkv_a_c_latent_offset = 0
        self._qkv_a_k_pe_offset = 0
        self.q_a_out = self.mpk.narrow(
            self.qkv_a_out, dim=1, start=0, length=self.q_lora_rank)
        self.q_a_out_buf = None
        # q_b output (after absorption): [batch, num_local_q_heads * qk_head_dim]
        self.q_nope_pe_buf = None
        # Allocated as a 3D torch tensor so we can attach slice views
        # (q_nope_pe[:, :, :512] for BMM output, q_nope_pe[:, :, 512:]
        # for q_pe) and have the decode BMM write per-head [nope|pe]
        # interleaved directly without an assemble task.
        import torch as _torch
        self._q_nope_pe_torch = _torch.zeros(
            mbt, self.num_local_q_heads, self.qk_head_dim,
            dtype=_torch.bfloat16, device="cuda")
        self.q_nope_pe = self.mpk.attach_input(
            self._q_nope_pe_torch, name="q_nope_pe")
        # Decode consumes absorbed [CKV, KPE] Q. Prefill consumes vLLM's
        # original per-head split Q: [nope(128), rope(64)].
        self.q_nope = self.mpk.new_tensor(
            dims=(mbt, self.num_local_q_heads * QK_NOPE_HEAD_DIM),
            dtype=bfloat16, name="q_nope", io_category="cuda_tensor",
        )
        self.q_pe = self.mpk.new_tensor(
            dims=(mbt, self.num_local_q_heads * QK_ROPE_HEAD_DIM),
            dtype=bfloat16, name="q_pe", io_category="cuda_tensor",
        )
        # kv_a outputs (c_latent + k_pe) are `mpk.narrow` views of the
        # fused qkv_a_out, mirroring q_a_out above. The view encodes the
        # slot's start offset; downstream consumers see the slot's slice
        # width via view.dim[1] and the parent row stride via
        # view.stride[0]. See C20 note above q_a_out.
        self.c_latent_out_buf = None
        self.c_latent_out = self.mpk.narrow(
            self.qkv_a_out, dim=1,
            start=self.q_lora_rank,
            length=self.kv_lora_rank)
        self.k_pe_out = self.mpk.narrow(
            self.qkv_a_out, dim=1,
            start=self.q_lora_rank + self.kv_lora_rank,
            length=QK_ROPE_HEAD_DIM)
        # Combined KV entry after layernorm: [batch, 576]
        self.kv_combined = self.mpk.new_tensor(
            dims=(mbt, self.qk_head_dim),  # [batch, 576]
            dtype=bfloat16,
            name="kv_combined",
            io_category="cuda_tensor",
        )
        # Decode can skip this copy on direct-paged paths. The buffer remains
        # allocated for legacy decode layouts and for unified prefill/decode
        # graphs whose prefill side also needs split contiguous KV views.
        self.contiguous_kv = self.mpk.new_tensor(
            dims=(self.mpk.max_num_batched_requests * self.mpk.max_seq_length,
                  self.qk_head_dim),
            dtype=bfloat16,
            name="contiguous_kv",
            io_category="cuda_tensor",
        )
        if self._use_prefill:
            # DEBUG 2026-05-10: pad row dim up to multiple of 128 (chunked
            # prefill TMA box BN_BOX=128) to avoid OOB NaN-fill in V/K SMEM
            # which propagates through hmma16 (0 * NaN = NaN in IEEE math).
            _raw_rows = (self.mpk.max_num_batched_requests
                         * self.mpk.max_seq_length)
            _kv_rows = ((_raw_rows + 127) // 128) * 128
            self.ckv_sep = self.mpk.new_tensor(
                dims=(_kv_rows, self.kv_lora_rank),
                dtype=bfloat16, name="ckv_sep", io_category="cuda_tensor",
            )
            self.kpe_sep = self.mpk.new_tensor(
                dims=(_kv_rows, QK_ROPE_HEAD_DIM),
                dtype=bfloat16, name="kpe_sep", io_category="cuda_tensor",
            )
            # `kpe_sep_v2` is the receiving tensor for an identity-copy
            # "phantom bridge" task inserted between the unified KV gather
            # and the chunked-prefill kernel. The bridge is NOT a real
            # compute step; it exists purely to legalize the MPK task
            # graph. See the identity_layer call after the gather for the
            # full rationale; the short version:
            #
            # Without the bridge, `mla_kv_gather_unified` simultaneously
            # plays two roles the runtime can't represent in one task:
            #
            #   * fork-producer: gather has multiple distinct downstream
            #     consumer layers (quantize_kv_b_k, quantize_kv_b_v,
            #     chunked_prefill).
            #   * join-producer: one of those consumers
            #     (`chunked_prefill`) is itself a join-consumer with 4
            #     distinct producers, so gather feeds a join event.
            #
            # A single MPK task has exactly one `trigger_event` slot, so a
            # layer that is both fork-producer and join-producer would
            # need to fire two distinct events. `annotated_graph.cc`
            # rejects that as case-3. Inserting the identity copy
            # (`kpe_sep → kpe_sep_v2`) turns the gather→chunked_prefill
            # edge into gather→identity→chunked_prefill, breaking the
            # fork+join overlap: the identity has a single producer (no
            # join) and a single consumer (no fork), and gather no longer
            # directly feeds the join-consumer.
            self.kpe_sep_v2 = self.mpk.new_tensor(
                dims=(_kv_rows, QK_ROPE_HEAD_DIM),
                dtype=bfloat16, name="kpe_sep_v2",
                io_category="cuda_tensor",
            )
            self.prefill_k_nope = self.mpk.new_tensor(
                dims=(_kv_rows,
                      self.num_local_q_heads * QK_NOPE_HEAD_DIM),
                dtype=bfloat16, name="prefill_k_nope", io_category="cuda_tensor",
            )
            self.prefill_v = self.mpk.new_tensor(
                dims=(_kv_rows, self.num_local_q_heads * V_HEAD_DIM),
                dtype=bfloat16, name="prefill_v", io_category="cuda_tensor",
            )
        else:
            self.ckv_sep = None
            self.kpe_sep = None
            self.kpe_sep_v2 = None
            self.prefill_k_nope = None
            self.prefill_v = None
        # MLA decode partial outputs (PR 651: bf16 for partials)
        # MLA kernel writes blocks at stride D_V*128 and LSE at stride 128.
        # TP kernels use split-K: each split handles one KV tile (128 tokens).
        # Buffer = mbr * num_groups * max_splits blocks.
        mbr = self.mpk.max_num_batched_requests
        if self.world_size > 1:
            max_splits = (self.mpk.max_seq_length + 127) // 128
            if self.world_size == 2:
                _qpg = min(2, mbt)
            elif self.world_size == 4:
                _qpg = min(4, mbt)
            else:  # TP=8
                _qpg = 2
            _q_for_groups = mbt + (mbt % 2) if self.world_size == 8 else mbt
            _num_groups = (_q_for_groups + _qpg - 1) // _qpg
            _partial_blocks = mbr * _num_groups * max_splits
        else:
            max_splits = (self.mpk.max_seq_length + 127) // 128
            _hpb = 128 // mbt
            while 128 % _hpb != 0:
                _hpb -= 1
            _num_groups = 128 // _hpb
            _partial_blocks = mbr * _num_groups * max_splits
        self.mla_partial_o = self.mpk.new_tensor(
            dims=(_partial_blocks, self.v_head_dim * 128),
            dtype=bfloat16,
            name="mla_partial_o",
            io_category="cuda_tensor",
        )
        self.mla_partial_lse = self.mpk.new_tensor(
            dims=(_partial_blocks, 128),
            dtype=float32,
            name="mla_partial_lse",
            io_category="cuda_tensor",
        )
        self.mla_max_splits = max_splits
        # Attention output: [batch, num_local_q_heads * v_head_dim_absorbed]
        # v_head_dim = 512 (kv_lora_rank, after absorption)
        self.attn_out_buf = None
        self.attn_out = self.mpk.new_tensor(
            dims=(mbt, self.num_local_q_heads * self.v_head_dim),
            dtype=bfloat16,
            name="attn_out",
            io_category="cuda_tensor",
        )
        # V un-absorption output: [batch, num_local_q_heads * v_head_dim_original]
        # v_head_dim_original = 128 (before absorption)
        V_HEAD_DIM_ORIG = 128
        self.attn_unabsorbed = self.mpk.new_tensor(
            dims=(mbt, self.num_local_q_heads * V_HEAD_DIM_ORIG),
            dtype=bfloat16,
            name="attn_unabsorbed",
            io_category="cuda_tensor",
        )
        # O projection output (same as hidden_size)
        # When TP > 1, this feeds into nvshmem allreduce and must be in symmetric memory.
        _attn_proj_io = "nvshmem_tensor" if self._use_nvshmem else "cuda_tensor"
        self.attn_proj_out_buf = None
        self.attn_proj_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="attn_proj_out",
            io_category=_attn_proj_io,
        )

        # MLP intermediates
        # Dense MLP: gate+up = 2 * intermediate_size
        self.mlp_mid = self.mpk.new_tensor(
            dims=(mbt, 2 * self.intermediate_size),
            dtype=bfloat16,
            name="mlp_mid",
            io_category="cuda_tensor",
        )
        self.silu_mul_out = self.mpk.new_tensor(
            dims=(mbt, self.intermediate_size),
            dtype=bfloat16,
            name="silu_mul_out",
            io_category="cuda_tensor",
        )
        # When TP > 1, this feeds into nvshmem allreduce and must be in symmetric memory.
        _mlp_out_io = "nvshmem_tensor" if self._use_nvshmem else "cuda_tensor"
        self.mlp_out_buf = None
        self.mlp_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="mlp_out",
            io_category=_mlp_out_io,
        )

        # AllReduce buffer
        if self.world_size > 1:
            _allreduce_io = "nvshmem_tensor" if self._use_nvshmem else "cuda_tensor"
            self.allreduce_buf = self.mpk.new_tensor(
                dims=(self.world_size, mbt, self.hidden_size),
                dtype=bfloat16,
                name="allreduce_buf",
                io_category=_allreduce_io,
            )
            self.allreduce_out = self.mpk.new_tensor(
                dims=(mbt, self.hidden_size),
                dtype=bfloat16,
                name="allreduce_out",
                io_category=_allreduce_io,
            )

        # Argmax
        self.argmax_part_value = self.mpk.new_tensor(
            dims=(mbt, self.mpk.num_workers),
            dtype=bfloat16,
            name="argmax_part_value",
            io_category="cuda_tensor",
        )
        self.argmax_part_index = self.mpk.new_tensor(
            dims=(mbt, self.mpk.num_workers),
            dtype=int64,
            name="argmax_part_index",
            io_category="cuda_tensor",
        )

    def _safe_attach(self, tensor, name):
        """Attach tensor. FP8 is now natively supported in core.pyx.
        Also keeps a reference to prevent GC from freeing the underlying memory.
        Sanitizes name for C++ codegen (dots → underscores).
        Uses _attach_cache to avoid re-declaring the same C++ variable (for
        repeated attaches of the same weight)."""
        if not hasattr(self, '_attached_tensors'):
            self._attached_tensors = []
        safe_name = name.replace('.', '_')
        if safe_name in self._attach_cache:
            return self._attach_cache[safe_name]
        self._attached_tensors.append(tensor)
        dtensor = self.mpk.attach_input(torch_tensor=tensor, name=safe_name)
        self._attach_cache[safe_name] = dtensor
        return dtensor

    @staticmethod

    @property

    def _attach_fp8_weight(self, state_dict, key, name):
        """Attach FP8 weight + raw float32 scale_inv (NEW kernel format),
        or BF16 weight as fallback.

        Previously this requantized the weight to UE8M0 scale for the old
        `linear_fp8_sm100` kernel. We now use the new dense GEMM kernels
        (`fp8_gemm_dense_smallm_sm100` / `_mediumm_sm100`) which take raw
        float32 block scales instead — same layout as
        `_attach_raw_fp8_weight`. Keeping the same function name avoids
        churn at the many existing call sites; the weight tuple `(w, s)`
        now matches `_attach_raw_fp8_weight`'s contract.
        """
        scale_key = f"{key}_scale_inv"
        if scale_key in state_dict:
            if state_dict[key].dtype != torch.float8_e4m3fn:
                raise TypeError(f"{key} must be torch.float8_e4m3fn when {scale_key} exists.")
            if state_dict[scale_key].dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError(f"{scale_key} must be a floating scale tensor.")
            w = self._safe_attach(state_dict[key], name)
            scale = state_dict[scale_key].to(torch.float32).contiguous()
            s = self._safe_attach(scale, f"{name}_scale")
        else:
            # BF16 fallback is used by reduced fixtures or explicitly
            # pre-converted weights that have no scale tensor.
            if state_dict[key].dtype != torch.bfloat16:
                raise TypeError(f"{key} without scale must be torch.bfloat16.")
            w = self._safe_attach(state_dict[key], name)
            s = None  # weight is already BF16 (post-dequant)
        return w, s

    @staticmethod
    def _float_to_ue8m0(t: torch.Tensor) -> torch.Tensor:
        """fp32 → UE8M0 (8-bit exponent only). Uses CEIL rounding to match
        the kernel-side `encode_ue8m0` in per_token_group_quantize_fp8.cuh
        (which uses `ceilf(log2f(scale))`). The standalone test uses
        torch.round — that's only consistent within the test because the
        test feeds pre-encoded SFA to the kernel without re-quantizing;
        production has the kernel re-encoding SFA at runtime so the Python
        weight-pack MUST use the same rounding convention as the kernel.
        """
        pos = torch.where(t > 0, t, torch.full_like(t, 1e-30))
        p2 = torch.pow(2.0, torch.ceil(torch.log2(pos)))
        bits = p2.view(torch.int32)
        ue = ((bits >> 23) & 0xFF).to(torch.uint8)
        ue = torch.where(t > 0, ue, torch.zeros_like(ue))
        return ue

    @staticmethod
    def _pack_moe_scale_ue8m0(scale_per_row: torch.Tensor) -> torch.Tensor:
        """[dim, nk] fp32 → [num_sf_k, dim] uint32 row-major, UE8M0-packed.

        Identical packing as test_wrapper.py::pack_sf — the new PR-674
        grouped FP8 GEMM's SFA/SFB TMA descriptors expect this transposed
        layout (gd=[dim, num_sf_k] with dim as the innermost axis).

        scale_per_row: per-output-row dequant scale (after repeat_interleave
        along the output dim). For W13 this is reshape from
        (E, 2*intermediate, K/128); pass it flattened as
        (E*2*intermediate, K/128).
        """
        dim, nk = scale_per_row.shape
        num_sf_k = (nk + 3) // 4
        ue = DeepSeekV3Builder._float_to_ue8m0(scale_per_row).to(torch.int64)
        out = torch.zeros(num_sf_k, dim, dtype=torch.int64,
                          device=scale_per_row.device)
        zero = torch.zeros(dim, num_sf_k, dtype=torch.int64,
                           device=scale_per_row.device)
        for j in range(4):
            ki = torch.arange(num_sf_k, device=scale_per_row.device) * 4 + j
            valid = ki < nk
            ue_col = torch.where(valid,
                                 ue[:, ki.clamp(max=nk - 1)],
                                 zero[:, 0:num_sf_k])
            out |= (ue_col.t() & 0xFF) << (j * 8)
        return out.to(torch.uint32).contiguous()

    def _attach_raw_fp8_weight(self, state_dict, key, name):
        """Attach checkpoint-style FP8 weight + float32 block scale.

        PR674's dense FP8 GEMM uses the original block scale layout
        [output/128, K/128], not the packed UE8M0 scale used by the small-B
        linear runtime.
        """
        scale_key = f"{key}_scale_inv"
        if scale_key not in state_dict:
            raise ValueError(f"{key} requires {scale_key} for FP8 dense GEMM.")
        if state_dict[key].dtype != torch.float8_e4m3fn:
            raise TypeError(f"{key} must be torch.float8_e4m3fn.")
        w = self._safe_attach(state_dict[key], name)
        s = self._safe_attach(
            state_dict[scale_key].float().contiguous(), f"{name}_scale")
        return w, s

    def _build_mla_attention_layer(self, layer_idx: int, state_dict: dict):
        """Build MLA attention for one decoder layer (FP8 weights)."""
        prefix = f"model.layers.{layer_idx}."
        attn = f"{prefix}self_attn."

        # One fused FP8 GEMM emits the full qkv_a_out (mbt, 2176): cols
        # [0:1536) = q_a, [1536:2048) = c_latent, [2048:2112) = k_pe,
        # [2112:2176) = zero pad. Demo builds qkv_a_proj.weight by
        # FP8-byte-concatenating q_a_proj + kv_a_proj_with_mqa (the FP8 block
        # boundaries already align at 128 rows so no requantize is needed).
        w_qkv_a, s_qkv_a = self._attach_fp8_weight(
            state_dict, f"{attn}qkv_a_proj.weight",
            f"layer_{layer_idx}_qkv_a_proj")
        if s_qkv_a is None:
            raise RuntimeError(
                "qkv_a_proj.weight + weight_scale_inv must be present in "
                "the state_dict. demo.py builds them at load time.")
        # B37: the fused rmsnorm+quantize task pre-populated the
        # share_quantize_tag in _fp8_quantize_emitted at the input_layernorm
        # call site, so _fp8_linear_v2's internal quantize is skipped.
        # Also pull the per-layer-unique FP8/scale buffers the fused task
        # wrote to (case-3 fix — see `_emit_fused_rmsnorm_qkv_a_quantize`
        # docstring).
        qkv_a_quantize_tag = self._fused_rmsnorm_quantize_qkv_a_tag(layer_idx)
        qkv_a_fp8_ovr, qkv_a_scale_ovr = self._fused_qkv_a_bufs[layer_idx]
        self._fp8_linear(
            self.rmsnorm_out, w_qkv_a, s_qkv_a, self.qkv_a_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv_a.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
            share_quantize_tag=qkv_a_quantize_tag,
            input_fp8_override=qkv_a_fp8_ovr,
            input_scale_override=qkv_a_scale_ovr)

        # Diagnostic (PRE-RMSnorm dump 2026-05-13): captures RAW qkv_a_out
        # immediately after the fused GEMM, before any consumer touches it.
        # This discriminates: if rows 1..71 are zero HERE, GEMM is the bug;
        # if rows 1..71 only become zero post-RMSnorm, RMSnorm is the bug.
        if (layer_idx == 0
                and getattr(self.mpk, "dump_layer0_intra_tensors", None) is not None
                and getattr(self, "_layer0_q_a_zero_pt", None) is not None):
            q_a_zero_dt = self.mpk.attach_input(
                torch_tensor=self._layer0_q_a_zero_pt, name="q_a_zero_local")
            q_a_dump_dt = self.mpk.attach_input(
                torch_tensor=self.mpk.dump_layer0_intra_tensors[1],
                name="q_a_dump_local")
            self.mpk.elementwise_add_layer(
                input_a=self.q_a_out, input_b=q_a_zero_dt,
                output=q_a_dump_dt,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

        # Step 2: q_a_layernorm (BF16 norm weight) — in-place RMSnorm of the
        # q_a slice [0:q_lora_rank) inside the fused qkv_a_out buffer, fused
        # with the downstream q_b input-quantize (saves ~2-3μs/layer on the
        # decode chain).
        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}q_a_layernorm.weight"],
            name=f"layer_{layer_idx}_q_a_layernorm")
        q_a_fused_fp8_ovr, q_a_fused_scale_ovr, q_a_fused_tag = (
            self._emit_fused_q_a_rmsnorm_quantize(
                input_x=self.q_a_out, w_norm=w_q_a_ln,
                layer_idx=layer_idx, reduction_size=self.q_lora_rank,
                in_offset_elems=self._qkv_a_q_offset,
                out_offset_elems=self._qkv_a_q_offset))

        # Step 3: q_b projections.
        # Decode uses absorbed q_b [H*(512+64)] to match the compressed cache.
        # Prefill uses vLLM's original split q_b [H*128] + [H*64].
        # QKV-a fused: q_a_out aliases qkv_a_out (mbt, 2176); pass the slice
        # row stride + offset so the FP8 quantize reads only q_a's 1536 cols.
        qb_slice_kwargs = dict(
            input_row_stride=self._qkv_a_row_stride,
            input_col_offset=self._qkv_a_q_offset)
        # B24 (2026-05-15): when dual-dispatch is active, decode q_b
        # and prefill q_b both quantize self.q_a_out with K=q_lora=1536.
        # Share the quantize task between them — saves one ~5 us wave
        # dispatch per layer on decode iters (the prefill quantize was
        # early-exiting on decode but still paying the dispatch cost).
        # 2026-05-28: the fused q_a layernorm already emitted the FP8/scale
        # into per-layer buffers via the fused task above; use that tag +
        # thread the per-layer buffers as input_fp8_override /
        # input_scale_override to all q_b GEMMs so they read from the
        # per-layer buf (NOT the shared cache; case-3).
        qb_share_tag = q_a_fused_tag
        qb_input_fp8_ovr = q_a_fused_fp8_ovr
        qb_input_scale_ovr = q_a_fused_scale_ovr
        # Decode Q path: runtime BMM-based absorption (five tasks instead of
        # one monolithic absorbed-q_b FP8 GEMM; each task loads smaller
        # per-head weights → less TMA traffic, better overlap potential).
        self._bmm_decode_q_path(state_dict, attn, layer_idx, qb_slice_kwargs,
                                qb_share_tag=qb_share_tag,
                                qb_input_fp8_ovr=qb_input_fp8_ovr,
                                qb_input_scale_ovr=qb_input_scale_ovr)
        if self._use_prefill:  # prefill-exclusive unabsorbed q_b GEMMs
            w_q_b_nope, s_q_b_nope = self._attach_fp8_weight(
                state_dict, f"{attn}q_b_nope.weight",
                f"layer_{layer_idx}_q_b_nope")
            self._fp8_linear(
                self.q_a_out, w_q_b_nope, s_q_b_nope, self.q_nope,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_nope.dim(0)),
                          1, 1),
                block_dim=(128, 1, 1),
                gate_mode=1,
                share_quantize_tag=qb_share_tag,
                input_fp8_override=qb_input_fp8_ovr,
                input_scale_override=qb_input_scale_ovr,
                **qb_slice_kwargs)
            w_q_b_pe, s_q_b_pe = self._attach_fp8_weight(
                state_dict, f"{attn}q_b_pe.weight",
                f"layer_{layer_idx}_q_b_pe")
            self._fp8_linear(
                self.q_a_out, w_q_b_pe, s_q_b_pe, self.q_pe,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_pe.dim(0)),
                          1, 1),
                block_dim=(128, 1, 1),
                gate_mode=1,
                share_quantize_tag=qb_share_tag,
                input_fp8_override=qb_input_fp8_ovr,
                input_scale_override=qb_input_scale_ovr,
                **qb_slice_kwargs)  # 2026-05-13: was missing — see scratch/fp8_dense_smallm_n2176_bug.md
        # Step 4: kv_a (c_latent + k_pe) is produced by the fused qkv_a GEMM
        # above; no separate kv_a_proj_with_mqa GEMMs are emitted.

        rope_q_grid = (
            self.mpk.max_num_batched_requests,
            self.num_local_q_heads,
            1,  # B35: TILE_Q==mbt -> 1 CTA per (req, head); kernel inner-loop covers all tokens
        )
        # B25 (2026-05-15): in dual-dispatch the fused (absorbed) ROPE_Q
        # only matters on decode iters and the split (unabsorbed) ROPE_Q
        # only matters on prefill iters. Add phase gates so the wrong-
        # phase ROPE returns immediately instead of rotating stale data.
        dsv3_tasks.deepseek_mla_rope_q_fused_layer(
            self.mpk,
            q_nope_pe=self.q_nope_pe,
            cos_pos_embed=self.cos_pos_embed,
            sin_pos_embed=self.sin_pos_embed,
            num_heads=self.num_local_q_heads,
            grid_dim=rope_q_grid,
            q_tile_size=self.max_num_batched_tokens,
            phase_gate=2 if self._use_prefill else 0,
        )
        if self._use_prefill:  # prefill-exclusive split (unabsorbed) ROPE-Q
            dsv3_tasks.deepseek_mla_rope_q_split_layer(
                self.mpk,
                q_pe=self.q_pe,
                cos_pos_embed=self.cos_pos_embed,
                sin_pos_embed=self.sin_pos_embed,
                num_heads=self.num_local_q_heads,
                grid_dim=rope_q_grid,
                q_tile_size=self.max_num_batched_tokens,
                qfused_mode=0,
                phase_gate=1,
            )
        # k_pe lives at cols [2048:2112) inside the 2176-wide qkv_a_out;
        # pass row stride + offset so the ROPE kernel rotates the right slice.
        dsv3_tasks.deepseek_mla_rope_k_layer(
            self.mpk,
            k_pe=self.k_pe_out,
            cos_pos_embed=self.cos_pos_embed,
            sin_pos_embed=self.sin_pos_embed,
            grid_dim=(
                self.mpk.max_num_batched_requests,
                1,
                1,  # B35: TILE_Q==mbt collapses grid.z to 1
            ),
            q_tile_size=self.max_num_batched_tokens,
            k_pe_row_stride=self._qkv_a_row_stride,
            k_pe_offset=self._qkv_a_k_pe_offset,
        )

        # Step 5: kv_a_layernorm on c_latent slice [1536:2048) of qkv_a_out.
        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_a_layernorm.weight"],
            name=f"layer_{layer_idx}_kv_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln,
            output=self.c_latent_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
            process_dim=self.kv_lora_rank,
            in_offset_elems=self._qkv_a_c_latent_offset,
            out_offset_elems=self._qkv_a_c_latent_offset)

        # Step 6: MLA attention (KV gather + unified prefill/decode + reduce).
        # When `_use_prefill` is True, register one MLA main task that chooses
        # prefill vs decode from runtime Q_LEN. The decode reduce stays
        # separate and keeps its Q_LEN gate.
        layer_cache = self.mpk.attach_input(
            torch_tensor=self.ckv_kpe_cache[layer_idx],
            name=f"layer_{layer_idx}_kv_cache")
        q_len_mla = self.max_num_batched_tokens
        decode_q_len_mla = self._decode_q_len()
        kv_len_max = self.mpk.max_seq_length
        kv_tiles_max = (kv_len_max + self.mpk.page_size - 1) // self.mpk.page_size
        # TP decode's direct-write path is only validated for one 128-token
        # KV tile; multi-tile configs keep the partial+reduce path.
        single_split_mla = kv_tiles_max <= 1
        mla_num_splits_override = 1 if single_split_mla else None
        mla_decode_out = self.attn_out if single_split_mla else self.mla_partial_o
        mla_decode_kv = (
            layer_cache if self._direct_paged_decode_kv else self.contiguous_kv
        )
        # c_latent and k_pe live at offsets 1536 / 2048 of the 2176-wide
        # qkv_a_out row. Pass row strides + offsets so the gather kernel
        # reads the right slice for each input.
        kv_gather_slice_kwargs = dict(
            c_latent_row_stride=self._qkv_a_row_stride,
            c_latent_offset_elems=self._qkv_a_c_latent_offset,
            k_pe_row_stride=self._qkv_a_row_stride,
            k_pe_offset_elems=self._qkv_a_k_pe_offset)
        if self._use_prefill:
            dsv3_tasks.mla_kv_gather_unified_layer(
                self.mpk,
                c_latent_new=self.c_latent_out,
                k_pe_new=self.k_pe_out,
                paged_cache=layer_cache,
                contiguous_kv=mla_decode_kv,
                ckv_sep=self.ckv_sep,
                kpe_sep=self.kpe_sep,
                mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
                grid_dim=(
                    self.mpk.max_num_batched_requests,
                    self._kv_gather_splits,
                    1),
                block_dim=(128, 1, 1),
                num_gather_splits=self._kv_gather_splits,
                **kv_gather_slice_kwargs,
            )
            # =================================================================
            # PHANTOM BRIDGE for chunked-prefill kpe_sep dependency tracking.
            #
            # Problem this fixes:
            #   `mla_kv_gather_unified` is registered as (4 inputs, 2 outputs)
            #   in graph.cc so its ckv_sep / kpe_sep writes get proper output
            #   edges in the annotated task graph. Without that, downstream
            #   consumers (kv_b_k / kv_b_v GEMMs, chunked_prefill) have no
            #   dependency edge from the gather and the megakernel scheduler
            #   is free to race them — in practice kv_b_k/v read zero ckv_sep
            #   and chunked_prefill emits all-zero attn_unabsorbed.
            #
            # Why the bridge is needed once the gather IS tracked:
            #   `chunked_prefill` is a join-consumer (4 distinct producers:
            #   RoPE_q, kv_b_k FP8 GEMM, gather, kv_b_v FP8 GEMM). Gather
            #   feeds it via kpe_sep AND is also a fork-producer to other
            #   consumers (the two quantize tasks). That makes gather both a
            #   `fork-producer` AND `is_join_producer` (one of its consumers
            #   is a join-consumer). MPK's `FullTaskDesc` has exactly one
            #   `trigger_event` slot, so a task literally cannot fire two
            #   distinct events (the fork event AND the downstream join
            #   event). `annotated_graph.cc` rejects this as case-3.
            #
            # The fix:
            #   Insert an identity copy `kpe_sep → kpe_sep_v2`, and have
            #   chunked_prefill read kpe_sep_v2 instead of kpe_sep. The
            #   gather now only directly feeds 1-producer-only tasks
            #   (the two quantize tasks plus this identity), so it stays
            #   fork-producer but is no longer is_join_producer. The
            #   identity itself is is_join_producer (its consumer
            #   chunked_prefill is a join-consumer) but is NOT a
            #   fork-producer (single consumer), so it's also case-3-safe.
            #
            #   In event terms: gather fires E1 (its fork event). E1
            #   launches quantize_kv_b_k, quantize_kv_b_v, AND the
            #   identity. The identity, when done, fires E2 (the join
            #   event for chunked_prefill). E2 also collects triggers from
            #   the other join-producers (kv_b_k/v GEMMs, RoPE_q). When
            #   E2's counter reaches num_triggers, chunked_prefill fires.
            #
            # NB on grid_dim: identity_layer's dim_maps partition the LAST
            # tensor dim across grid.x, and grid.x must DIVIDE the inner
            # dim. kpe_sep's inner dim is 64 (rope dim). grid.x=1 = no
            # partition; a single block does the full copy. The copy is
            # tiny (kv_rows * 64 bf16 < 64 KB).
            # =================================================================
            # P3 (2026-05-14 v3): bump grid from (1,1,1) to (8,1,1) so 8
            # CTAs share the kpe_sep → kpe_sep_v2 BF16 copy. Earlier two
            # attempts looked like correctness regressions but those were
            # baseline-noise misdiagnoses (decode-from-step-100 outputs
            # all-zero tokens regardless). With 64-wide rope dim ÷ 8 = 8
            # cols per CTA the partition divides cleanly. Single-CTA was
            # the 55 μs straggler in perfetto (user-flagged ID 63112).
            self.mpk.identity_layer(
                input=self.kpe_sep,
                output=self.kpe_sep_v2,
                grid_dim=(8, 1, 1),
                block_dim=(128, 1, 1),
                # B19 (2026-05-15): decode iter has Q_LEN=1 and chunked_prefill
                # early-returns via its own Q_LEN > 8 gate. The copy here is
                # wasted ~16 μs/layer in decode. Gate the body on Q_LEN > 8
                # so decode iters skip the copy entirely (kpe_sep_v2 keeps
                # stale data, harmless because chunked_prefill doesn't read
                # it on decode).
                gate_decode_q_len=True,
            )
        else:
            self.mpk.mla_kv_gather_layer(
                c_latent_new=self.c_latent_out,
                k_pe_new=self.k_pe_out,
                paged_cache=layer_cache,
                contiguous_kv=mla_decode_kv,
                mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
                grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
                **kv_gather_slice_kwargs,
            )
        if self._use_prefill:
            # Prefill-EXCLUSIVE attention work (kv_b_k/v dense GEMMs +
            # chunked_prefill); the decode MLA tasks below still register.
            w_kv_b_k, s_kv_b_k = self._attach_raw_fp8_weight(
                state_dict, f"{attn}kv_b_k.weight",
                f"layer_{layer_idx}_kv_b_k")
            w_kv_b_v, s_kv_b_v = self._attach_raw_fp8_weight(
                state_dict, f"{attn}kv_b_v.weight",
                f"layer_{layer_idx}_kv_b_v")
            # B23 (2026-05-15): share the FP8 quantize of ckv_sep between
            # kv_b_k and kv_b_v — both operate on the same input with the
            # same group_size, so emitting two quantize tasks duplicates
            # the bytes (and one extra ~5 μs dispatch wave on decode iters
            # where both early-exit). One quantize, two GEMM consumers.
            kv_b_shared_tag = f"layer_{layer_idx}_kv_b_shared"
            self._fp8_dense_kv_b_proj(
                self.ckv_sep, w_kv_b_k, s_kv_b_k, self.prefill_k_nope,
                tag=f"layer_{layer_idx}_kv_b_k",
                shared_quantize_tag=kv_b_shared_tag)
            self._fp8_dense_kv_b_proj(
                self.ckv_sep, w_kv_b_v, s_kv_b_v, self.prefill_v,
                tag=f"layer_{layer_idx}_kv_b_v",
                shared_quantize_tag=kv_b_shared_tag)
            dsv3_tasks.mla_prefill_tp8_chunked_layer(
                self.mpk,
                q_nope=self.q_nope,
                q_pe=self.q_pe,
                k_nope=self.prefill_k_nope,
                # k_rope comes from `kpe_sep_v2`, the phantom-bridged copy
                # of kpe_sep produced by the identity_layer above. This
                # breaks the gather→chunked_prefill direct edge that
                # would otherwise make gather a fork+join layer (case 3).
                k_rope=self.kpe_sep_v2,
                v=self.prefill_v,
                output=self.attn_unabsorbed,
                mla_params=(
                    self.num_local_q_heads,
                    q_len_mla,
                    kv_len_max,
                    0,
                ),
                grid_dim=(
                    self.num_local_q_heads,
                    (q_len_mla + 63) // 64,
                    self.mpk.max_num_batched_requests,
                ),
                block_dim=(128, 1, 1),
                qfused_mode=0,
            )
        # Decode MLA main + reduce. Registered in BOTH prefill-capable and
        # decode-only builds — the decode kernels' runtime Q_LEN gates skip
        # prefill iters. tp_size picks the per-rank head-count variant.
        dsv3_tasks.mla_mtp_decode_layer(
            self.mpk,
            self.q_nope_pe, mla_decode_kv,
            mla_decode_out, self.mla_partial_lse,
            decode_q_len_mla, kv_len_max,
            tp_size=self.world_size,
            num_splits_override=mla_num_splits_override)
        if not single_split_mla:
            dsv3_tasks.mla_mtp_reduce_layer(
                self.mpk,
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, decode_q_len_mla, kv_len_max,
                tp_size=self.world_size)

        # Step 7: O projection.
        # Decode uses the absorbed output projection produced during weight
        # conversion: [hidden, H*512]. Prefill uses PR674's unabsorbed attention
        # output and therefore must use the original projection:
        # [hidden, H*128]. Both are registered in the same graph and runtime
        # phase gates make exactly one branch write the residual output.
        self.attn_proj_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_attn_proj_fused",
            io_category="cuda_tensor",
        )
        if self._use_prefill:
            # Prefill-EXCLUSIVE unabsorbed o_proj (gate_mode=1).
            w_o_prefill, s_o_prefill = self._attach_fp8_weight(
                state_dict, f"{attn}o_proj_original.weight",
                f"layer_{layer_idx}_o_proj_original")
            self._fp8_linear(
                self.attn_unabsorbed,
                w_o_prefill,
                s_o_prefill,
                self.attn_proj_out,
                grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                block_dim=(128, 1, 1),
                residual=self.x,
                gate_mode=1,
            )
            # C9: decode o_proj via the post-attn BMM path:
            # quantize(attn_out) → BMM(kv_b_v_bmm_dense) → smaller
            # o_proj_original linear. Eliminates the H*512 fused GEMM in
            # favor of an H*128 unabsorbed GEMM + BMM. Decode-only via
            # gate_mode=2.
            self._bmm_decode_o_path(state_dict, attn, layer_idx, residual=self.x)
        else:
            # Pure decode (mbt<=8): same post-attn BMM path.
            self._bmm_decode_o_path(state_dict, attn, layer_idx, residual=self.x)

    def _build_dense_mlp(self, layer_idx: int, state_dict: dict):
        """Build dense MLP for layers 0-2 (FP8 weights)."""
        prefix = f"model.layers.{layer_idx}."

        w_gate_up, s_gate_up = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.gate_up_proj.weight",
            f"layer_{layer_idx}_gate_up_proj")
        # TP=2 workaround (2026-05-18): the fp8_gemm_dense_mediumm kernel
        # faults with cudaErrorLaunchFailure (719) at `mb_arrive_tx` when
        # invoked with the TP=2 gate_up shape (M=8, N=18432, K=7168). Other
        # sizes (N=2176 qkv_a, N=4096 q_b_pe, N=7168 o_proj, N=9216 TP=4
        # gate_up, N=36864 TP=1 gate_up) all run cleanly. The root cause is
        # in the kernel and likely requires kernel-team-level fix. As a
        # builder-side workaround, split the gate_up GEMM into two sub-calls
        # each with N=9216 (= TP=4's known-good size). The first sub-call
        # produces the local gate half; the second produces the local up
        # half. silu_mul reads them from disjoint output slots — layout is
        # preserved. See [[project-tp2-debug-session-20260518]].
        split_tp2_gate_up = (
            self.world_size == 2
            and w_gate_up.dim(0) % 2 == 0
        )
        if split_tp2_gate_up:
            N_full = w_gate_up.dim(0)
            N_half = N_full // 2
            # Each half-weight is [N_half, K]; scale half is [N_half/128, K/128].
            w_gate = self.mpk.narrow(w_gate_up, dim=0, start=0, length=N_half)
            w_up = self.mpk.narrow(
                w_gate_up, dim=0, start=N_half, length=N_half)
            s_gate = self.mpk.narrow(
                s_gate_up, dim=0, start=0, length=N_half // 128)
            s_up = self.mpk.narrow(
                s_gate_up, dim=0, start=N_half // 128,
                length=N_half // 128)
            mlp_mid_gate = self.mpk.narrow(
                self.mlp_mid, dim=1, start=0, length=N_half)
            mlp_mid_up = self.mpk.narrow(
                self.mlp_mid, dim=1, start=N_half, length=N_half)
            half_grid = grid_for_rmsnorm_linear_layer(N_half)
            self._fp8_linear(self.rmsnorm_out, w_gate, s_gate, mlp_mid_gate,
                             grid_dim=(half_grid, 1, 1),
                             block_dim=(128, 1, 1))
            self._fp8_linear(self.rmsnorm_out, w_up, s_up, mlp_mid_up,
                             grid_dim=(half_grid, 1, 1),
                             block_dim=(128, 1, 1))
        else:
            gate_up_grid = grid_for_rmsnorm_linear_layer(w_gate_up.dim(0))
            self._fp8_linear(self.rmsnorm_out, w_gate_up, s_gate_up, self.mlp_mid,
                             grid_dim=(gate_up_grid, 1, 1),
                             block_dim=(128, 1, 1))
        # silu_mul reads gate from first half + up from second half of each
        # input block. The interleave split is computed from the FULL (pre-shard)
        # gate_up dimension in demo.py weight prep. In TP>1, shard halves the
        # number of chunk pairs but keeps each pair's layout intact. So:
        #   silu_mul_grid = interleave_split / world_size
        # where interleave_split = grid_fn(FULL_gate_up_dim) // 2.
        full_gate_up_dim = 2 * self.intermediate_size * self.world_size
        interleave_split = grid_for_rmsnorm_linear_layer(full_gate_up_dim) // 2
        silu_mul_grid = interleave_split // self.world_size
        w_down, s_down = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.down_proj.weight",
            f"layer_{layer_idx}_down_proj")
        # Per-layer output to avoid aliasing self.x ↔ self.mlp_out.
        self.mlp_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_mlp_fused",
            io_category="cuda_tensor",
        )
        self._silu_mul_fp8_linear(
            self.mlp_mid,
            self.silu_mul_out,
            w_down,
            s_down,
            self.mlp_out,
            silu_grid_dim=(silu_mul_grid, 1, 1),
            linear_grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
            block_dim=(128, 1, 1),
            residual=self.x,
        )

    def _setup_new_moe_buffers(self):
        """Lazy-init the SHARED routed-MoE buffers + static m_indices
        buffer the first time _build_moe_mlp runs.

        All buffers are shared across MoE layers (single allocation per
        rank, reused per layer) to keep peak HBM bounded.
        """
        if getattr(self, "_new_moe_alloced", False):
            return
        bm = self._moe_bm_padding  # 128
        E = self.num_local_experts
        m_total = E * bm
        K = self.hidden_size
        N_w13 = 2 * self.routed_moe_intermediate_size
        N_w2 = K  # W2 maps intermediate → hidden
        K_intermediate = self.routed_moe_intermediate_size
        # Packed UE8M0 num_sf_k for K = hidden, K_intermediate = intermediate.
        nk_K = (K + 127) // 128
        nk_int = (K_intermediate + 127) // 128
        K_PACKED_K = (nk_K + 3) // 4
        K_PACKED_INT = (nk_int + 3) // 4
        topk = NUM_EXPERTS_PER_TOK

        # Static m_indices = [0,0,..0, 1,1,..1, ..., E-1,...,E-1] each repeated
        # BM_PADDING times. Lives in GPU memory for the entire megakernel run.
        # Keep a Python-side reference so the tensor isn't GC'd (MPK stores
        # raw pointers).
        self._new_moe_m_indices_tensor = (
            torch.arange(m_total, dtype=torch.int32, device="cuda") // bm
        ).contiguous()
        self.new_moe_m_indices_dt = self.mpk.attach_input(
            torch_tensor=self._new_moe_m_indices_tensor,
            name="new_moe_m_indices_static",
        )

        # Permuted input (W13's A): shared across layers.
        self.new_moe_permuted_in_fp8 = self.mpk.new_tensor(
            dims=(m_total, K), dtype=float8_e4m3,
            name="new_moe_permuted_in_fp8", io_category="cuda_tensor",
        )
        self.new_moe_permuted_in_scale = self.mpk.new_tensor(
            dims=(K_PACKED_K, m_total), dtype=uint32,
            name="new_moe_permuted_in_scale", io_category="cuda_tensor",
        )
        # NOTE: per-iter buffers (meta, permuted_fp8/scale, w13/silu/w2 outs)
        # are allocated PER LAYER inside `_build_moe_mlp`'s NEW path, not
        # shared globally. Sharing them across layers makes MPK's task-graph
        # dep tracker hit "case 3 — fork+join producer" errors because the
        # same buffer gets two distinct producer→consumer chains in
        # consecutive layers. Per-layer allocations keep each chain linear.
        self._new_moe_per_iter_alloced = False  # legacy flag, unused
        # All per-iter buffers (input quantized, permuted_*, meta) are
        # allocated PER LAYER inside `_build_moe_mlp`'s NEW path — sharing
        # them across layers breaks MPK's dep tracker (case-3 fork+join).
        # The only thing this setup does globally is the static m_indices
        # attach + the per-layer scale-pack cache (so we don't re-pack the
        # same weight scale every rebuild).
        self._new_moe_packed_scale_cache = {}
        self._new_moe_alloced = True

    def _pack_and_attach_moe_weight_scale(self, state_dict, key, name):
        """Pack `state_dict[key]` (per-block fp32 scale_inv) into the
        UE8M0-transposed layout the new fp8_group_gemm expects, and attach.

        Input shape: (E, N/128, K/128) fp32
        Pack steps:
          1) repeat-interleave dim=1 → (E, N, K/128)
          2) flatten → (E*N, K/128)
          3) pack_sf → (num_sf_k, E*N) uint32 UE8M0
        """
        cache = self._new_moe_packed_scale_cache
        if name in cache:
            return cache[name]
        raw = state_dict[key].to(torch.float32).clamp(min=1e-30)
        # (E, N/128, K/128) → (E, N, K/128)
        expanded = raw.repeat_interleave(128, dim=1).contiguous()
        E_, N_, NK_ = expanded.shape
        flat = expanded.reshape(E_ * N_, NK_).contiguous()
        packed = self._pack_moe_scale_ue8m0(flat).contiguous()
        dt = self._safe_attach(packed, name)
        cache[name] = dt
        return dt

    def _new_moe_dispatch_inline(self, layer_idx, prefix, state_dict, mbt,
                                  bm_pad, m_total, K, K_intermediate,
                                  K_PACKED_K, K_PACKED_INT,
                                  w13_scale_key, w_experts_w13,
                                  moe_topk_weights, moe_routing_indices,
                                  new_moe_input_fp8, new_moe_input_scale,
                                  new_moe_permuted_in_fp8,
                                  new_moe_permuted_in_scale,
                                  new_moe_meta,
                                  new_moe_w13_out, new_moe_silu_out,
                                  new_moe_silu_fp8,
                                  new_moe_silu_scale_Mfirst,
                                  new_moe_silu_scale, new_moe_w2_out):
        """Per-layer NEW MoE task dispatch. Tasks numbered 1..8 below."""
        # Pack W13 weight scale (always — needs to be attached for w13).
        s_w13_packed = self._pack_and_attach_moe_weight_scale(
            state_dict, w13_scale_key,
            f"layer_{layer_idx}_experts_w13_scale_ue8m0")
        # 1) Quantize MoE input with UE8M0-packed scale.
        self.mpk.quantize_fp8_layer(
            input=self.rmsnorm_out,
            output_fp8=new_moe_input_fp8,
            output_scale=new_moe_input_scale,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            scale_ue8m0=True,
        )
        # 2) Zero-init meta. The dummy=new_moe_input_fp8 trick chains the
        # tensor_init between the upstream quantize_fp8 (which writes
        # new_moe_input_fp8) and the downstream moe_permute_sm100 (which reads
        # new_moe_input_fp8). With those RAW edges in place, no WAW edge on
        # `new_moe_meta` is required to serialize this zero-fill against
        # moe_permute's write of the same buffer — the Q→T→P chain dominates.
        self.mpk.tensor_init_layer(
            target=new_moe_meta,
            dummy=new_moe_input_fp8,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            dummy_input_map=(-1, -1, -1),
            target_input_map=(-1, -1, -1),
        )
        # 3) Permute + scale-transpose.
        # E_PER_CTA: collapse the decode permute valley. E_LOCAL
        # (= num_local_experts = m_total // bm_pad) must divide evenly.
        _epc = self._moe_permute_epc
        _e_local = m_total // bm_pad
        assert _e_local % _epc == 0, (
            f"moe_permute e_per_cta ({_epc}) must divide num_local_experts "
            f"({_e_local})")
        dsv3_tasks.moe_permute_sm100_layer(
            self.mpk,
            input_fp8=new_moe_input_fp8,
            input_scale=new_moe_input_scale,
            topk_weights=moe_topk_weights,
            routing_indices=moe_routing_indices,
            permuted_fp8=new_moe_permuted_in_fp8,
            permuted_scale=new_moe_permuted_in_scale,
            meta=new_moe_meta,
            bm_padding=bm_pad,
            e_per_cta=_epc,
        )
        # 4) Group GEMM W13.
        dsv3_tasks.fp8_group_gemm_layer(
            self.mpk,
            a_fp8=new_moe_permuted_in_fp8,
            b_fp8=w_experts_w13,
            sfa_packed=new_moe_permuted_in_scale,
            sfb_packed=s_w13_packed,
            m_indices=self.new_moe_m_indices_dt,
            output=new_moe_w13_out,
            num_workers=self.mpk.num_workers,
            meta=new_moe_meta,
        )
        # 5) SiLU+MUL.
        # 2026-05-14 v3: grid=(num_workers, 1, 1) restored after disk-full
        # turned out to be the earlier crash root cause (not the grid).
        # The previous (1,1,1) was a 30 ms/call hot spot (single CTA on
        # 1 SM doing m_total*OUTPUT_SIZE ops). With input_map=(0,-1,-1)
        # the runtime partitions dim 0 across grid.x CTAs and the C++
        # register pulls num_active_tokens from per-CTA STensor shape
        # (= m_total / grid_dim.x rows per CTA). So each CTA processes
        # m_total/num_workers rows in-bounds, fully parallel across SMs.
        _silu_grid = min(self.mpk.num_workers, m_total)
        self.mpk.moe_silu_mul_layer(
            input=new_moe_w13_out,
            output=new_moe_silu_out,
            grid_dim=(_silu_grid, 1, 1),
            block_dim=(128, 1, 1),
            meta=new_moe_meta,
            # bm_padding = per-expert row count in the permuted buffer.
            # Wrapper combines with rows_per_cta (= input.dim(0)/grid.x)
            # to derive my_expert = bid.x / (bm_padding / rows_per_cta).
            bm_padding=bm_pad,
        )
        # 6) Quantize SiLU → UE8M0 directly into K-outermost layout.
        # C8 (2026-05-16): quantize_fp8 kernel writes UE8M0 packed scale at
        # offset `packed_idx * aligned_batch + batch_idx` which IS K-outer
        # row-major. The previous (m_total, K_PACKED) declaration was a
        # "shape lie" that required a separate transpose_scale task to
        # reconcile. By declaring the output as (K_PACKED, m_total) the
        # write pattern matches the declared shape, and the downstream W2
        # SFA TMA descriptor (which expects K-outer) reads correct bytes
        # directly — eliminating TRANSPOSE_SCALE (-19 μs/layer).
        num_local_experts = m_total // bm_pad
        self.mpk.quantize_fp8_layer(
            input=new_moe_silu_out,
            output_fp8=new_moe_silu_fp8,
            output_scale=new_moe_silu_scale,
            grid_dim=(m_total, 1, 1),
            block_dim=(128, 1, 1),
            scale_ue8m0=True,
            expert_active_meta=new_moe_meta,
            expert_active_e_local=num_local_experts,
            expert_active_bm_padding=bm_pad,
        )
        # C8: transpose_scale eliminated — quantize_fp8 writes directly
        # into the K-outer-declared new_moe_silu_scale.
        # 7+8) Pack W2 weight scale + attach W2 weight + Group GEMM W2.
        w2_scale_key_for_pack = f"{prefix}experts.w2.weight_scale_inv"
        w_experts_w2_new = self._safe_attach(
            state_dict[f"{prefix}experts.w2.weight"],
            f"layer_{layer_idx}_experts_w2")
        s_w2_packed = self._pack_and_attach_moe_weight_scale(
            state_dict, w2_scale_key_for_pack,
            f"layer_{layer_idx}_experts_w2_scale_ue8m0")
        dsv3_tasks.fp8_group_gemm_layer(
            self.mpk,
            a_fp8=new_moe_silu_fp8,
            b_fp8=w_experts_w2_new,
            sfa_packed=new_moe_silu_scale,
            sfb_packed=s_w2_packed,
            m_indices=self.new_moe_m_indices_dt,
            output=new_moe_w2_out,
            num_workers=self.mpk.num_workers,
            meta=new_moe_meta,
        )

    def _build_shared_expert(self, layer_idx: int, prefix: str, state_dict: dict):
        """Register shared-expert dense FP8 path. Returns ``shared_residual``
        which the routed-MoE finalize step (moe_unpermute / moe_mul_sum_add)
        adds to the per-token routed contribution.
        """
        shared_prefix = f"{prefix}shared_experts."

        # gate_proj + up_proj fused (FP8) — use _attach_fp8_weight for requantize
        shared_gate_w = state_dict[f"{shared_prefix}gate_proj.weight"]
        shared_up_w = state_dict[f"{shared_prefix}up_proj.weight"]
        gate_scale_key = f"{shared_prefix}gate_proj.weight_scale_inv"
        has_shared_scale = gate_scale_key in state_dict
        if shared_gate_w.shape[0] != self.moe_intermediate_size:
            pass  # shard mismatch warning removed
        from ..utils import shuffle_tensors as _shuffle_tensors
        out_dim_total = shared_gate_w.shape[0] + shared_up_w.shape[0]
        linear_grid = grid_for_rmsnorm_linear_layer(out_dim_total)
        scale_dim_0 = shared_gate_w.shape[0] // 128
        shared_split = min(linear_grid // 2, scale_dim_0)
        while shared_gate_w.shape[0] % shared_split != 0 or scale_dim_0 % shared_split != 0:
            shared_split -= 1
            if shared_split < 1:
                shared_split = 1; break
        fused_key = f"layer_{layer_idx}_shared_expert_gate_up"
        state_dict[f"{fused_key}.weight"] = _shuffle_tensors(
            [shared_gate_w, shared_up_w], split=shared_split, dim=0)
        if has_shared_scale:
            shared_gate_s = state_dict[gate_scale_key]
            shared_up_s = state_dict[f"{shared_prefix}up_proj.weight_scale_inv"]
            state_dict[f"{fused_key}.weight_scale_inv"] = _shuffle_tensors(
                [shared_gate_s, shared_up_s], split=shared_split, dim=0)
        w_shared_gate_up, s_shared_gate_up = self._attach_fp8_weight(
            state_dict, f"{fused_key}.weight",
            f"layer_{layer_idx}_shared_expert_gate_up")
        shared_mid = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, 2 * self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_mid",
            io_category="cuda_tensor",
        )
        self._fp8_linear(self.rmsnorm_out, w_shared_gate_up,
                         s_shared_gate_up, shared_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(
                             w_shared_gate_up.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        # silu_mul + down_proj
        shared_silu_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_silu",
            io_category="cuda_tensor",
        )
        w_shared_down, s_shared_down = self._attach_fp8_weight(
            state_dict, f"{shared_prefix}down_proj.weight",
            f"layer_{layer_idx}_shared_expert_down")
        _down_w = state_dict[f"{shared_prefix}down_proj.weight"]
        if _down_w.shape[1] != self.moe_intermediate_size:
            pass  # shard mismatch warning removed
        shared_residual = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_residual",
            io_category="cuda_tensor",
        )
        self._silu_mul_fp8_linear(
            shared_mid,
            shared_silu_out,
            w_shared_down,
            s_shared_down,
            shared_residual,
            silu_grid_dim=(shared_split, 1, 1),
            linear_grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
            residual=None,
        )
        return shared_residual

    def _build_moe_mlp(self, layer_idx: int, state_dict: dict):
        """Build MoE MLP for layers 3-60."""
        prefix = f"model.layers.{layer_idx}.mlp."

        # Router
        w_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}gate.weight"],
            name=f"layer_{layer_idx}_moe_gate",
        )

        # MoE routing tensors — topk_sigmoid outputs float32 weights and int32 indices/mask
        moe_topk_weights = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK),
            dtype=float32,
            name=f"layer_{layer_idx}_moe_topk_weights",
            io_category="cuda_tensor",
        )
        moe_routing_indices = self.mpk.new_tensor(
            dims=(self.num_local_experts, self.max_num_batched_tokens),
            dtype=int32,
            name=f"layer_{layer_idx}_moe_routing_indices",
            io_category="cuda_tensor",
        )
        moe_mask = self.mpk.new_tensor(
            dims=(self.num_local_experts + 1,),
            dtype=int32,
            name=f"layer_{layer_idx}_moe_mask",
            io_category="cuda_tensor",
        )

        # Router logits → topk routing
        router_logits = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, NUM_EXPERTS),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_router_logits",
            io_category="cuda_tensor",
        )
        router_grid = min(grid_for_rmsnorm_linear_layer(w_gate.dim(0)),
                          w_gate.dim(0) // 8)
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_gate,
            output=router_logits,
            grid_dim=(router_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        _moe_io = "nvshmem_tensor" if self._use_nvshmem else "cuda_tensor"
        moe_output = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_output",
            io_category=_moe_io,
        )

        # TopK sigmoid routing (DeepSeek V3: scoring_func=sigmoid)
        # e_score_correction_bias is added to sigmoid scores for routing selection
        bias_key = f"{prefix}gate.e_score_correction_bias"
        w_bias = self.mpk.attach_input(
            torch_tensor=state_dict[bias_key],
            name=f"layer_{layer_idx}_moe_gate_bias",
        )
        # Router is full-replica on every rank; no inter-rank synchronization is
        # needed before the local top-k routing mask is produced.
        # C13 (2026-05-17): the kernel's VPT template (8, ROWS_PER_WARP=1)
        # lives in src/kernel/task_register.cc; block_dim 256 (8 warps) here
        # must stay in parity with it.
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=router_logits,
            bias=w_bias,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
            local_expert_start=self.local_expert_start,
        )

        # Expert W1+W3 (gate + up projection)
        # Check if weights are FP8 (have scale_inv) or BF16 (post-dequant)
        w13_scale_key = f"{prefix}experts.w13.weight_scale_inv"
        use_fp8_experts = w13_scale_key in state_dict
        w_experts_w13 = self._safe_attach(
            state_dict[f"{prefix}experts.w13.weight"],
            f"layer_{layer_idx}_experts_w13")
        mbt = self.max_num_batched_tokens

        # ====================================================================
        # Routed-MoE path — PR-674 fp8_group_gemm (smallm/largem auto-pick):
        # Permute(routing) → group_gemm(W13) → silu → quantize →
        # group_gemm(W2) → unpermute(combine + residual).
        # ====================================================================
        if not use_fp8_experts:
            raise RuntimeError(
                "DeepSeek V3 routed MoE requires FP8 expert weights "
                f"({w13_scale_key} missing from the state_dict).")
        self._setup_new_moe_buffers()  # idempotent
        bm_pad = self._moe_bm_padding
        E_local = self.num_local_experts
        m_total = E_local * bm_pad
        K = self.hidden_size
        K_intermediate = self.routed_moe_intermediate_size
        N_w13 = 2 * K_intermediate
        N_w2 = K
        nk_K = (K + 127) // 128
        nk_int = (K_intermediate + 127) // 128
        K_PACKED_K = (nk_K + 3) // 4
        K_PACKED_INT = (nk_int + 3) // 4

        # Per-layer allocations (NOT shared across MoE layers — sharing
        # tripped MPK's dep tracker on case-3 fork+join).
        new_moe_input_fp8 = self.mpk.new_tensor(
            dims=(mbt, K), dtype=float8_e4m3,
            name=f"layer_{layer_idx}_new_moe_input_fp8",
            io_category="cuda_tensor")
        new_moe_input_scale = self.mpk.new_tensor(
            dims=(mbt, K_PACKED_K), dtype=uint32,
            name=f"layer_{layer_idx}_new_moe_input_scale",
            io_category="cuda_tensor")
        new_moe_permuted_in_fp8 = self.mpk.new_tensor(
            dims=(m_total, K), dtype=float8_e4m3,
            name=f"layer_{layer_idx}_new_moe_permuted_in_fp8",
            io_category="cuda_tensor")
        new_moe_permuted_in_scale = self.mpk.new_tensor(
            dims=(K_PACKED_K, m_total), dtype=uint32,
            name=f"layer_{layer_idx}_new_moe_permuted_in_scale",
            io_category="cuda_tensor")
        # 2D `(2, N)` int32 so the shared `tensor_init` kernel — which
        # zeros `BATCH_SIZE * OUTPUT_SIZE * sizeof(bf16)` bytes — covers
        # the FULL int32 byte range (BATCH_SIZE=2, OUTPUT_SIZE=N int32
        # → 2*N*2 bytes = N*4 bytes = sizeof int32 buffer of length N).
        # L8 instrumentation: attach the meta buffer for layer_idx=8
        # to a host-readable torch tensor (set up by demo.py) so we can
        # dump tok_to_perm / out_weights after the run. Otherwise the
        # buffer lives in cuda_tensor pool and isn't host-accessible.
        new_moe_meta = self.mpk.new_tensor(
            dims=(2, m_total + mbt * NUM_EXPERTS_PER_TOK),
            dtype=int32,
            name=f"layer_{layer_idx}_new_moe_meta",
            io_category="cuda_tensor")
        new_moe_w13_out = self.mpk.new_tensor(
            dims=(m_total, N_w13), dtype=bfloat16,
            name=f"layer_{layer_idx}_new_moe_w13_out",
            io_category="cuda_tensor")
        new_moe_silu_out = self.mpk.new_tensor(
            dims=(m_total, K_intermediate), dtype=bfloat16,
            name=f"layer_{layer_idx}_new_moe_silu_out",
            io_category="cuda_tensor")
        new_moe_silu_fp8 = self.mpk.new_tensor(
            dims=(m_total, K_intermediate), dtype=float8_e4m3,
            name=f"layer_{layer_idx}_new_moe_silu_fp8",
            io_category="cuda_tensor")
        new_moe_silu_scale_Mfirst = self.mpk.new_tensor(
            dims=(m_total, K_PACKED_INT), dtype=uint32,
            name=f"layer_{layer_idx}_new_moe_silu_scale_Mfirst",
            io_category="cuda_tensor")
        new_moe_silu_scale = self.mpk.new_tensor(
            dims=(K_PACKED_INT, m_total), dtype=uint32,
            name=f"layer_{layer_idx}_new_moe_silu_scale",
            io_category="cuda_tensor")
        new_moe_w2_out = self.mpk.new_tensor(
            dims=(m_total, N_w2), dtype=bfloat16,
            name=f"layer_{layer_idx}_new_moe_w2_out",
            io_category="cuda_tensor")

        self._new_moe_dispatch_inline(
            layer_idx, prefix, state_dict, mbt, bm_pad, m_total,
            K, K_intermediate, K_PACKED_K, K_PACKED_INT,
            w13_scale_key, w_experts_w13,
            moe_topk_weights, moe_routing_indices,
            new_moe_input_fp8, new_moe_input_scale,
            new_moe_permuted_in_fp8, new_moe_permuted_in_scale,
            new_moe_meta,
            new_moe_w13_out, new_moe_silu_out,
            new_moe_silu_fp8, new_moe_silu_scale_Mfirst,
            new_moe_silu_scale, new_moe_w2_out)
        self._new_moe_layer_w2_out = new_moe_w2_out
        self._new_moe_layer_meta = new_moe_meta

        # ---- Shared Expert ----
        shared_residual = self._build_shared_expert(
            layer_idx, prefix, state_dict)

        # Final MoE contribution before transformer residual:
        #   routed_experts * topk_weights + shared_expert
        # The model residual is added after the tensor-parallel allreduce in
        # build_layers, otherwise each rank would add the same residual before
        # the reduction and over-count it.
        # NEW path: moe_unpermute does
        #   output[t] = shared_residual[t]
        #             + sum_k(permuted_w2_out[token_to_perm[t,k]-1]
        #                      * permuted_weights[same row])
        # — i.e. the topk-weighted combine AND the shared-residual add
        # in one task. Skips the OLD moe_mul_sum_add entirely.
        # 2026-05-15: MOE_UNPERMUTE sits on the post-W2 critical path
        # (nothing else runs concurrently — see perfetto), so the per-task
        # wallclock is what matters, NOT freeing worker slots. Force
        # rows_per_cta=1 so the kernel fan-out matches the worker pool
        # (mbt=128 → 128 CTAs each doing 1 token of unpermute work). With
        # rows_per_cta=8 (the wrapper default tuned for the OPPOSITE
        # constraint of leaving workers free for concurrent tasks), each
        # of the 16 active CTAs had to chew through 8 row-blocks
        # serially, giving 28-32 μs straggler wallclock per layer.
        # 2026-05-15 stragglers fix: hidden_split=8 partitions the
        # HIDDEN axis across 8 CTAs per token. For decode
        # (active_rows=1) this gives 8 CTAs working on token 0's
        # hidden cols concurrently — the previous 32 μs per-token
        # straggler shrinks by ~8x. For prefill (active_rows=128)
        # the grid balloons to 128*8=1024 CTAs but each CTA does
        # 1/8 the per-token compute, so the longest CTA wallclock
        # also shrinks (overall MOE_UNPERMUTE wave bounded by the
        # slowest CTA, not total work).
        dsv3_tasks.moe_unpermute_sm100_layer(
            self.mpk,
            permuted_output=self._new_moe_layer_w2_out,
            meta=self._new_moe_layer_meta,
            residual=shared_residual,
            output=moe_output,
            # C4 (2026-05-16): rows_per_cta 1→8 collapses grid to
            # (16, 8, 1)=128 CTAs = 1 wave (was 1024 CTAs = 8 waves).
            # Per-CTA work 8x more but wave-transition overhead × 7
            # was dominating cluster wallclock at ~17 μs/call.
            rows_per_cta=8,
            hidden_split=8,
        )
        self.mlp_out = moe_output








    def build_layers(self, state_dict: dict, layer_indices: list = None):
        """Build decoder layers.

        Args:
            layer_indices: If provided, only build these specific layer indices
                          (e.g., [0, 3] for 1 dense + 1 MoE). If None, build all.
        """
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        # Optional per-layer residual dump infrastructure. Demo allocates
        # `mpk.dump_hidden_tensors` as a list of bf16 (mbt, hidden) tensors,
        # one per built layer. We attach them as IO buffers and append a
        # zero-add copy task after each layer's residual update so the
        # snapshot is captured into the torch tensor that the demo can read
        # back after mpk() returns. Activated only when --dump-hidden-dir
        # is set; zero overhead otherwise.
        dump_layer_dts = None
        dump_zero_dt = None
        if getattr(self.mpk, "dump_hidden_tensors", None) is not None:
            assert len(self.mpk.dump_hidden_tensors) == len(layer_indices), (
                f"dump_hidden_tensors length {len(self.mpk.dump_hidden_tensors)} "
                f"!= built layer count {len(layer_indices)}"
            )
            self._dump_zero_pt = torch.zeros(
                (self.max_num_batched_tokens, self.hidden_size),
                dtype=torch.bfloat16, device="cuda",
            )
            dump_zero_dt = self.mpk.attach_input(
                torch_tensor=self._dump_zero_pt,
                name="hidden_dump_zero",
            )
            dump_layer_dts = []
            for slot, t in enumerate(self.mpk.dump_hidden_tensors):
                dump_layer_dts.append(self.mpk.attach_input(
                    torch_tensor=t,
                    name=f"hidden_dump_layer_{slot}",
                ))

        # Optional: dump several layer-0 intra-layer states to localize
        # where MPK first diverges from reference. Slots: 0=input_norm,
        # 1=attn_unabsorbed (raw chunked-prefill output, before o_proj),
        # 2=attn_out (= o_proj+residual), 3=dense_mlp_out.
        layer0_intra = getattr(self.mpk, "dump_layer0_intra_tensors", None)
        layer0_intra_dts = None
        layer0_attn_zero_dt = None
        if layer0_intra is not None:
            assert len(layer0_intra) >= 4, "expected at least 4 intra-layer dump slots"
            self._layer0_intra_zero_pt = torch.zeros(
                (self.max_num_batched_tokens, self.hidden_size),
                dtype=torch.bfloat16, device="cuda",
            )
            layer0_intra_zero_dt = self.mpk.attach_input(
                torch_tensor=self._layer0_intra_zero_pt,
                name="layer0_intra_zero",
            )
            # Slot 1 captures the full qkv_a_out (mbt, 2176); Python comparator
            # slices to the relevant sub-region (q_a / c_latent / k_pe).
            slot1_cols = QKV_A_FUSED_N
            self._layer0_q_a_zero_pt = torch.zeros(
                (self.max_num_batched_tokens, slot1_cols),
                dtype=torch.bfloat16, device="cuda",
            )
            layer0_attn_zero_dt = self.mpk.attach_input(
                torch_tensor=self._layer0_q_a_zero_pt,
                name="layer0_q_a_out_zero",
            )
            layer0_intra_dts = [
                self.mpk.attach_input(
                    torch_tensor=layer0_intra[k],
                    name=f"layer0_intra_dump_{k}",
                )
                for k in range(4)
            ]

        for slot, i in enumerate(layer_indices):
            prefix = f"model.layers.{i}."

            # Input layernorm
            w_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
                name=f"layer_{i}_input_layernorm",
            )
            # B37: the input-layernorm RMSNorm is fused with the downstream
            # qkv_a FP8 quantize. The fused task writes the BF16 normalized
            # output AND the FP8 + scale buffers in one pass, so the qkv_a
            # `_fp8_linear` call can skip its internal quantize via
            # share_quantize_tag.
            self._emit_fused_rmsnorm_qkv_a_quantize(
                input_x=self.x,
                w_norm=w_norm,
                layer_idx=i,
                reduction_size=self.hidden_size,
            )

            if i == 0 and layer0_intra_dts is not None:
                # Dump self.rmsnorm_out (= input-layernormed embed) into slot 0
                self.mpk.elementwise_add_layer(
                    input_a=self.rmsnorm_out, input_b=layer0_intra_zero_dt,
                    output=layer0_intra_dts[0],
                    grid_dim=(self.max_num_batched_tokens, 1, 1),
                    block_dim=(128, 1, 1),
                )

            # MLA attention. The residual is fused into o_proj's with_residual
            # kernel, so attn_proj_out already contains (matmul + residual).
            self._build_mla_attention_layer(i, state_dict)

            # Slot 1 (attn_unabsorbed) intentionally not dumped via
            # elementwise_add — that triggered a long hang at runtime when
            # the dump tensor shape (mbt, num_q_heads*128 = 4096) didn't
            # match the standard (mbt, hidden=7168) elementwise path.
            # We can infer attn_unabsorbed validity from attn_proj_out
            # via the o_proj+residual relation (slot 2 - embed).

            self.x = self.attn_proj_out

            if i == 0 and layer0_intra_dts is not None:
                # Dump self.attn_proj_out (= attn matmul + residual) into slot 2
                self.mpk.elementwise_add_layer(
                    input_a=self.x, input_b=layer0_intra_zero_dt,
                    output=layer0_intra_dts[2],
                    grid_dim=(self.max_num_batched_tokens, 1, 1),
                    block_dim=(128, 1, 1),
                )

            # Post-attention layernorm
            w_post_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
                name=f"layer_{i}_post_attn_layernorm",
            )
            self.mpk.rmsnorm_layer(
                input=self.x, weight=w_post_norm, output=self.rmsnorm_out,
                grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
                block_dim=(128, 1, 1),
            )

            # MLP: dense (layers 0-2) or MoE (layers 3-60)
            if i < FIRST_MOE_LAYER:
                # Dense MLP down_proj already fuses the residual into the
                # projection kernel. That path does not need the explicit MoE
                # allreduce-plus-residual sequence below.
                self._build_dense_mlp(i, state_dict)
                self.x = self.mlp_out
            else:
                self._build_moe_mlp(i, state_dict)
                # MoE: always use explicit residual add (not fused into shared_expert).
                if self.world_size > 1:
                    moe_residual_out = self.mpk.new_tensor(
                        dims=(self.max_num_batched_tokens, self.hidden_size),
                        dtype=bfloat16,
                        name=f"layer_{i}_moe_residual",
                        io_category="cuda_tensor",
                    )
                    # Residual add must happen after allreduce; fusing it into
                    # moe_mul_sum_add would over-count residual on TP ranks.
                    # The NVSHMEM allreduce task fuses the post-reduce add at
                    # its final local store.
                    self.mpk.allreduce_layer(
                        input=self.mlp_out, buffer=self.allreduce_buf,
                        output=moe_residual_out,
                        residual=self.x,
                        grid_dim=_tensor_parallel_allreduce_grid(self.hidden_size),
                        block_dim=(128, 1, 1),
                    )
                    self.x = moe_residual_out
                else:
                    moe_residual_out = self.mpk.new_tensor(
                        dims=(self.max_num_batched_tokens, self.hidden_size),
                        dtype=bfloat16,
                        name=f"layer_{i}_moe_residual",
                        io_category="cuda_tensor",
                    )
                    self.mpk.elementwise_add_layer(
                        input_a=self.x, input_b=self.mlp_out,
                        output=moe_residual_out,
                        grid_dim=(self.max_num_batched_tokens, 1, 1),
                        block_dim=(128, 1, 1),
                    )
                    self.x = moe_residual_out

            # Per-layer residual dump (diagnostic): copy self.x into a
            # torch-backed tensor so the demo can read it back. The copy is
            # `dump = self.x + 0` via elementwise_add. Only fires when
            # dump_hidden_tensors was provided by the demo.
            if dump_layer_dts is not None:
                self.mpk.elementwise_add_layer(
                    input_a=self.x,
                    input_b=dump_zero_dt,
                    output=dump_layer_dts[slot],
                    grid_dim=(self.max_num_batched_tokens, 1, 1),
                    block_dim=(128, 1, 1),
                )

            if i == 0 and layer0_intra_dts is not None:
                # Slot 3 = self.x at end of layer 0 (= dense MLP + residual).
                # Duplicates dump_hidden_tensors[0] but kept for symmetry.
                self.mpk.elementwise_add_layer(
                    input_a=self.x, input_b=layer0_intra_zero_dt,
                    output=layer0_intra_dts[3],
                    grid_dim=(self.max_num_batched_tokens, 1, 1),
                    block_dim=(128, 1, 1),
                )

    def build_from_dict(self, state_dict: dict, with_lm_head: bool,
                        layer_indices: list = None):
        """Build the DeepSeek V3 computation graph.

        Args:
            layer_indices: If provided, only build these layers (for correctness testing).
        """
        padded_vocab_size = 129280  # DeepSeek V3 vocab size (already aligned)

        # Embed layer
        self.x = self.mpk.attach_input(
            torch_tensor=self.input_tokens, name="input_token"
        )
        self.w_embed = self.mpk.attach_input(
            torch_tensor=state_dict["model.embed_tokens.weight"],
            name="embed_tokens",
        )
        w_embed = self.w_embed
        self.y = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16, name="embed_out", io_category="cuda_tensor",
        )
        self.mpk.embed_layer(
            input=self.x, weight=w_embed, output=self.y,
            grid_dim=(self.hidden_size // 128, 1, 1),
            block_dim=(128, 1, 1), input_source=1,
        )
        self.x = self.y

        if getattr(self.mpk, "dump_embed_tensor", None) is not None:
            self._dump_embed_zero_pt = torch.zeros(
                (self.max_num_batched_tokens, self.hidden_size),
                dtype=torch.bfloat16, device="cuda",
            )
            embed_zero_dt = self.mpk.attach_input(
                torch_tensor=self._dump_embed_zero_pt, name="embed_dump_zero")
            embed_dump_dt = self.mpk.attach_input(
                torch_tensor=self.mpk.dump_embed_tensor, name="embed_dump")
            self.mpk.elementwise_add_layer(
                input_a=self.x, input_b=embed_zero_dt, output=embed_dump_dt,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

        # Intermediate tensors
        self._new_intermediate_tensors()
        self._precompute_rope_embeddings()

        # Build all decoder layers
        self.build_layers(state_dict, layer_indices=layer_indices)

        # Final norm + LM head
        w_final_norm = self.mpk.attach_input(
            torch_tensor=state_dict["model.norm.weight"],
            name="model_norm_weight",
        )
        self.mpk.rmsnorm_layer(
            input=self.x, weight=w_final_norm, output=self.rmsnorm_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
        )

        if with_lm_head:
            lm_head_weight = state_dict["lm_head.weight"]
            vocab_parallel_lm_head = bool(
                getattr(self.mpk, "deepseek_vocab_parallel_lm_head", False)
            )
            if vocab_parallel_lm_head:
                lm_head_vocab_size = lm_head_weight.shape[0]
                lm_head_grid = lm_head_vocab_size // 256
                if lm_head_vocab_size % 256 != 0:
                    raise ValueError(
                        "vocab-parallel lm_head expects a 256-aligned local vocab, "
                        f"got {lm_head_vocab_size}"
                    )
                vocab_offset = int(
                    getattr(self.mpk, "deepseek_lm_head_vocab_offset", 0)
                )
                valid_vocab_size = int(
                    getattr(
                        self.mpk,
                        "deepseek_lm_head_valid_vocab_size",
                        lm_head_vocab_size,
                    )
                )
                if not (0 < valid_vocab_size <= lm_head_vocab_size):
                    raise ValueError(
                        "vocab-parallel lm_head requires at least one valid "
                        f"local vocab row, got {valid_vocab_size}"
                    )
            else:
                lm_head_vocab_size = padded_vocab_size
                lm_head_grid = grid_for_rmsnorm_linear_layer(lm_head_vocab_size)
                vocab_offset = 0
                valid_vocab_size = lm_head_vocab_size

            # Keep vocab rows aligned to the argmax/linear task grid. DeepSeek
            # V3's checkpoint vocab is already 129280, so this is a no-op for
            # the normal model and only handles smaller test fixtures.
            if not vocab_parallel_lm_head and lm_head_weight.shape[0] < padded_vocab_size:
                lm_head_weight = torch.cat([
                    lm_head_weight,
                    torch.zeros(padded_vocab_size - lm_head_weight.shape[0],
                                self.hidden_size, device=lm_head_weight.device,
                                dtype=lm_head_weight.dtype),
                ], dim=0)

            # Keep the (possibly padded) weight alive — persistent kernel stores
            # the raw GPU pointer, not a PyTorch tensor reference.
            self._lm_head_weight_buf = lm_head_weight
            self.w_lm_head = self.mpk.attach_input(
                torch_tensor=self._lm_head_weight_buf, name="lm_head",
            )
            w_lm_head = self.w_lm_head
            self.lm_head_out_buf = None
            lm_head_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, lm_head_vocab_size),
                dtype=bfloat16, name="lm_head_out", io_category="cuda_tensor",
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out, weight=w_lm_head, output=lm_head_out,
                grid_dim=(lm_head_grid, 1, 1),
                block_dim=(128, 1, 1),
            )

            # Argmax
            self.argmax_out_dtensor = self.mpk.attach_input(
                torch_tensor=self.output_tokens, name="output_token",
            )
            argmax_out = self.argmax_out_dtensor
            if vocab_parallel_lm_head:
                # Do not tie argmax fan-out to the LM-head GEMM grid.  The
                # GEMM grid can be 127 for TP4 local vocab (32512 / 256), while
                # the downstream NVSHMEM global argmax path is much more stable
                # when the partial-reduction fan-out matches the worker count.
                # The local vocab shard is padded to a 256-row multiple, so it
                # is divisible by the 128-worker argmax grid on the target path.
                lm_head_argmax_grid = self.mpk.num_workers
                if lm_head_vocab_size % lm_head_argmax_grid != 0:
                    lm_head_argmax_grid = lm_head_grid
                local_argmax_part_value = self.mpk.new_tensor(
                    dims=(self.max_num_batched_tokens, lm_head_argmax_grid),
                    dtype=bfloat16,
                    name="lm_head_local_argmax_part_value",
                    io_category="cuda_tensor",
                )
                local_argmax_part_index = self.mpk.new_tensor(
                    dims=(self.max_num_batched_tokens, lm_head_argmax_grid),
                    dtype=int64,
                    name="lm_head_local_argmax_part_index",
                    io_category="cuda_tensor",
                )
                partial_chunk_size = lm_head_vocab_size // lm_head_argmax_grid
                self.mpk.argmax_partial_layer(
                    input=lm_head_out,
                    output=(local_argmax_part_value, local_argmax_part_index),
                    grid_dim=(lm_head_argmax_grid, 1, 1),
                    block_dim=(128, 1, 1),
                )
                global_argmax_value = self.mpk.new_tensor(
                    dims=(self.world_size, self.max_num_batched_tokens),
                    dtype=float32,
                    name="lm_head_global_argmax_value",
                    io_category="nvshmem_tensor",
                )
                global_argmax_index = self.mpk.new_tensor(
                    dims=(self.world_size, self.max_num_batched_tokens),
                    dtype=int64,
                    name="lm_head_global_argmax_index",
                    io_category="nvshmem_tensor",
                )
                dsv3_tasks.nvshmem_global_argmax_layer(
                    self.mpk,
                    partial_value=local_argmax_part_value,
                    partial_index=local_argmax_part_index,
                    scratch_value=global_argmax_value,
                    scratch_index=global_argmax_index,
                    output=argmax_out,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                    vocab_offset=vocab_offset,
                    valid_vocab_size=valid_vocab_size,
                    partial_chunk_size=partial_chunk_size,
                )
            else:
                self.mpk.argmax_partial_layer(
                    input=lm_head_out,
                    output=(self.argmax_part_value, self.argmax_part_index),
                    grid_dim=(self.mpk.num_workers, 1, 1),
                    block_dim=(128, 1, 1),
                )
                # Non-vocab-parallel lm_head is replicated on every tensor
                # parallel rank, so the final logits cover the full vocab on
                # each rank.  A cross-rank argmax is only required for the
                # vocab-parallel path above, where each rank owns a shard.
                self.mpk.argmax_reduce_layer(
                    input=(self.argmax_part_value, self.argmax_part_index),
                    output=argmax_out,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )

