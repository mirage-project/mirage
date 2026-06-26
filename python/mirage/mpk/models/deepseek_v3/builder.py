"""DeepSeek V3 model builder for Mirage MPK.

Architecture: 61 decoder layers with MLA attention and MoE MLP.
- Layers 0-2: dense MLP (gate_up + silu_mul + down).
- Layers 3-60: MoE MLP (256 routed experts, top-8 sigmoid routing, + 1
  shared expert).
- MLA: 128 Q heads, 1 KV head after weight absorption; the KV cache holds
  the compressed per-token entry [c_latent(512) | k_pe(64)] = 576 dims.

Per-layer task chain (registered onto the megakernel graph):
  fused rmsnorm+quantize → fused qkv_a FP8 GEMM → q_a layernorm+quantize
  → decode absorbed q_b_proj FP8 GEMM
  [+ prefill q_b GEMMs] → ROPE q/k → kv_a layernorm → KV gather
  [+ phantom bridge + kv_b_k/v GEMMs + chunked prefill] → MLA decode
  (+ reduce) → decode O BMM chain (quantize / BMM / o_proj+residual)
  [+ prefill o_proj] → post-attn rmsnorm → dense MLP or MoE
  (router → topk_sigmoid → permute → group GEMM W13 → silu → quantize →
  group GEMM W2 → unpermute+shared-expert; AllReduce + residual at TP>1)
  → final norm → lm_head → argmax.

Build-time configuration (fixed at graph-build time):
- world_size / ep_size: tensor-parallel world and expert-parallel group
  count. Routed experts are sharded EP×TP (routed_tp_size = world_size /
  ep_size); shared experts and dense layers are TP over all ranks.
- max_num_batched_tokens (mbt): compile-time M of every per-token task.
  `_use_prefill = mbt > 8` enables dual-dispatch: prefill AND decode
  tasks are both registered, and runtime Q_LEN gates (prefill: Q_LEN > 8,
  decode: Q_LEN <= 8) pick which one does work each iteration.
- Weight forms: demo.py emits absorbed and unabsorbed weights at load
  time (qkv_a fused, q_b_proj absorbed, q_b_nope/q_b_pe split,
  kv_b_k/kv_b_v split, BMM per-head forms, fused o_proj and
  o_proj_original).
"""

import math
import os
import torch
from typing import Optional

from ..utils import grid_for_rmsnorm_linear_layer
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from . import tasks as dsv3_tasks
from ....core import bfloat16, float8_e4m3, float32, uint8, uint32, int32, int64


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
# FULLY-fused FFN mega-task scratch (rmsnorm+router+topk+MoE). MUST equal the
# kernel's kernel::ffn_full_megakernel_sm100::SCRATCH_BYTES (computed there as the
# %16-rounded high-water of: 8B barrier + rmsnorm_out[7168]bf16 + a_fp8[7168] +
# a_scale[56]f32 + logits[256]bf16 + inter[8*512]f32 + y13[8*1024]f32 +
# i_fp8[8*512] + i_scale[8*4]f32 + sg[512]f32 + si_fp8[256] + si_scale[2]f32 +
# out_acc[7168]f32, with EACH section start 16B-aligned). The per-section 16B
# alignment (the misaligned-y13-cp.async.cg-16 fix) is +16 vs the old contiguous
# 106608 (an 8B pad before rmsnorm_out and before out_acc) → 106624. %16 == 0 so
# bytes/2 bf16 element count (53312) is %8 (the tensor_init 16B-vec zero-init
# static_assert).
FFN_FULL_MEGAKERNEL_SCRATCH_BYTES = 106624
# Fused DENSE-MLP mega-task scratch (dsv3_dense_mlp_fused_sm100.cuh). The kernel's
# only GLOBAL cross-block buffer is y13 (W13_N=4608 fp32), laid out AFTER a 64B
# head reserved for the 8B grid barrier (count+gen) + pad (make_scratch: off=64).
# So bytes = 64 + W13_N*4 = 64 + 18432 = 18496. y13 starts 16B-aligned (64) and is
# read/written element-wise (float*), so no further alignment beyond 16B is needed.
# 18496 % 16 == 0 AND (18496/2 == 9248) % 8 == 0 (the tensor_init 16B-vec zero-init
# static_assert on the bf16 element count). Everything else (rmsnorm_out / a_fp8 /
# a_scale / silu / i_fp8 / i_scale) is block-local SMEM (recomputed per block),
# never DMEM — so this scratch is just barrier + y13.
DENSE_MLP_MEGAKERNEL_SCRATCH_BYTES = 18496
# Attention mega-task scratch (TP8 decode, 16 local heads): 2xuint32 barrier +
# all inter-stage activations the ferret kernel kept in __device__ globals
# (g_hdeq/g_hf8/.../g_mla_acc), 16-byte aligned per section. MUST equal the
# kernel's kernel::attn_block_megakernel_sm100::ATTN_SCRATCH_BYTES (verified at
# build time: attn_make_scratch reaches 434464; 434720 carries 256B slack).
# MUST be a multiple of 16: the scratch tensor's element count (bytes/2 bf16) must
# be a multiple of 8 for tensor_init's 16B-vec zero-init static_assert.
ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES = 434864  # +144B: FAST levers' g_head_done[16]+g_head_wuv_ready[16]; MUST == kernel ATTN_SCRATCH_BYTES
FIRST_MOE_LAYER = 3
VOCAB_SIZE = 129280
RMS_NORM_EPS = 1e-6

# Rows each RMSNorm CTA processes via the kernel's batch_idx loop. The
# kernel's threadblock partition asserts (mbt % grid.x == 0); `_rmsnorm_grid`
# snaps to a divisor of mbt accordingly.
_RMSNORM_ROWS_PER_TASK = 1


def _rmsnorm_grid(mbt: int) -> tuple:
    """RMSNorm grid: ~mbt / _RMSNORM_ROWS_PER_TASK CTAs, snapped up to a
    divisor of mbt (the kernel asserts mbt % grid.x == 0)."""
    if mbt <= _RMSNORM_ROWS_PER_TASK:
        # Single CTA covers the whole batch — kernel BATCH_SIZE = mbt.
        return (1, 1, 1)
    target = mbt // _RMSNORM_ROWS_PER_TASK
    g = target
    while g <= mbt and mbt % g != 0:
        g += 1
    if g > mbt:
        g = mbt  # 1-row-per-CTA fallback
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

    All current producers of an allreduce input partition the hidden dim
    into 128-wide column tiles (linear.grid.x = output_size // 128). A
    wider allreduce tile would generate fewer tasks than the producing
    layer (e.g. 7 vs 56 for DSv3 hidden=7168 at 1024-wide), starving the
    persistent runtime of dispatchable work right after the matmul.

    Defaulting the allreduce tile to 128 keeps the partition aligned with
    the producer so each upstream task has a one-to-one downstream
    consumer.
    """
    if output_size % 128 != 0:
        raise ValueError(
            "Tensor-parallel all-reduce expects a 128-aligned output "
            f"dimension, got {output_size}")
    # The AR grid mirrors the producer's 128-wide column tiles: the CTAs are
    # PARALLEL (each reduces the full vector concurrently → fast wall), not
    # redundant, so collapsing the grid would only serialize/slow the reduce.
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

        # Per-expert row padding in the permuted MoE layout. M_TOTAL for the
        # group GEMM = num_local_experts * _moe_bm_padding; must be a
        # multiple of fp8_group_gemm's internal BM tile (128).
        self._moe_bm_padding = 128
        # Experts per CTA for moe_permute_sm100: shrinks the permute launch
        # to (E_LOCAL / EPC, 1, 1). num_local_experts must be divisible by
        # it — asserted at the moe_permute call site.
        self._moe_permute_epc = 4
        # Fan the MLA KV-gather seq_pos loop across this many CTAs (each
        # CTA strides seq_pos by the split count).
        self._kv_gather_splits = 8

    def _decode_q_len(self) -> int:
        """Compile-time q capacity of the decode MLA tasks (grid q-groups +
        the codegen's runtime-q_len clamp `if (q_len_rt_ > Q) q_len_rt_ = Q`).

        Must cover the largest q_len a DECODE iteration can see, NOT just 1:
        the scheduler feeds num_new_tokens = min(prompt_len - step, mbt), so
        decode-only builds (mbt <= 8) consume the prompt in mbt-token chunks
        and prefill-capable builds (mbt > 8) see a <= 8-token prompt TAIL
        (the dual-dispatch decode gate is q_len <= 8). The old hardcoded 1
        silently truncated every multi-token iteration to ONE query row
        (rows 1..q_len-1 got zero attention and, with causal_limit derived
        from Q_LEN=1, row 0 attended the whole chunk including its future) —
        invisible at mbt=1, catastrophic at q_len 2..8.

        mbt=1 still returns 1 (the validated decode build is byte-identical).
        An MTP rebuild would take spec_length into account here too."""
        return min(self.max_num_batched_tokens, 8)

    # ------------------------------------------------------------------
    # Fused decode pipeline capability predicates.
    #
    # The DeepSeek-V3 decode pass is a SINGLE fused pipeline: one
    # attention-block megakernel (`_build_mla_attention_megakernel`) + one
    # FFN-full MoE megakernel (`_build_moe_mlp_ffn_full`). These ARE the
    # default and only decode path — there is no per-task decode chain any
    # more. The chain code that remains in `_build_mla_attention_layer` /
    # `_build_moe_mlp` is the PREFILL (chunked-prefill, dual-dispatch, mbt>8)
    # + unsupported-geometry COMPAT fallback, NOT a decode alternative.
    #
    # Each megakernel hard-codes the bs=1 TP8/EP2/B200 decode geometry, so it
    # may only be selected when the build matches that geometry EXACTLY (the
    # same conjunction the kernels assert internally). Any other config
    # (prefill mbt>8, TP1/2/4 decode smoke tests, non-EP2) falls through to the
    # compat path. Gating on the full predicate — not just mbt==1 — is what
    # keeps the TP1/TP4 decode smoke builds from crashing the megakernel
    # asserts. Reviewer + Codex vetted 2026-06-25.
    @property
    def _use_attn_megakernel(self) -> bool:
        return (self.max_num_batched_tokens == 1
                and self.world_size == 8
                and self.mpk.num_workers == 136
                and self.mpk.max_num_batched_requests == 1)

    @property
    def _use_ffn_full_megakernel(self) -> bool:
        return (self.max_num_batched_tokens == 1
                and self.mpk.num_workers == 136
                and self.num_local_experts == 128
                and self.routed_moe_intermediate_size == 512
                and self.hidden_size == 7168
                and NUM_EXPERTS == 256)

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

    def _fp8_quant_buffers(self, rows: int, reduction_size: int,
                           fp8_name: str, scale_name: str):
        """Cached (FP8 activation buffer, float32 scale) pair for the dense
        FP8 GEMM family.

        Layout contract (fp8_gemm_dense_{smallm,mediumm}_sm100): FP8 input is
        `(rows, K)` fp8_e4m3 row-major; the activation scale is row-major
        float32 `[rows, ceil(K/128)]` (one scale per 128-col group). Buffers
        are allocated once per distinct `fp8_name` and reused by every GEMM
        that quantizes the same logical input.
        """
        num_groups = (reduction_size + 127) // 128
        if not hasattr(self, "_fp8_quant_bufs"):
            self._fp8_quant_bufs = {}
        if fp8_name not in self._fp8_quant_bufs:
            fp8_buf = self.mpk.new_tensor(
                dims=(rows, reduction_size), dtype=float8_e4m3,
                name=fp8_name,
                io_category="cuda_tensor",
            )
            scale_buf = self.mpk.new_tensor(
                dims=(rows, num_groups), dtype=float32,
                name=scale_name,
                io_category="cuda_tensor",
            )
            self._fp8_quant_bufs[fp8_name] = (fp8_buf, scale_buf)
        return self._fp8_quant_bufs[fp8_name]

    def _fp8_mbt_buffers_for_reduction_f32scale(self, reduction_size: int):
        """Shared per-token-batch (mbt rows) FP8 input + float32 scale pair,
        keyed by reduction_size; one allocation serves every dense FP8 GEMM
        with the same K."""
        return self._fp8_quant_buffers(
            self.max_num_batched_tokens, reduction_size,
            f"fp8_input_v2_{reduction_size}_shared",
            f"fp8_scale_v2_{reduction_size}_shared")

    def _fp8_sequence_buffers_for_reduction(
        self, reduction_size: int, tag: str = "shared"
    ):
        """Sequence-rows FP8 input + scale pair for the chunked-prefill
        kv_b projections. Rows = max_num_batched_requests * max_seq_length
        padded up to a multiple of 128 (chunked-prefill TMA box BN_BOX=128;
        unpadded rows OOB-NaN-fill the PV MMA)."""
        _raw_rows = (self.mpk.max_num_batched_requests
                     * self.mpk.max_seq_length)
        rows = ((_raw_rows + 127) // 128) * 128
        return self._fp8_quant_buffers(
            rows, reduction_size,
            f"fp8_seq_input_{reduction_size}_{tag}",
            f"fp8_seq_scale_{reduction_size}_{tag}")

    def _fp8_dense_num_workers(self, output_size=None):
        """Worker count for one fp8_gemm_dense_{smallm,mediumm} call.

        Decode-only builds (`_use_prefill=False`) use the full worker pool.
        Dual-dispatch builds cap at 80 so other tasks (ROPE / rmsnorm / KV
        append) can overlap the dense wave. When `output_size` is given the
        count is further collapsed to the single-wave tile count
        `ceil(M_max/128) * ceil(N/128)` (BN=128 fixed in the kernel; the
        kernel strides output tiles by num_workers and idle CTAs
        early-return, so this is byte-identical). FLOOR=24 because the
        NVSHMEM barrier crashes at low worker counts.

        `_fp8_dense_kv_b_proj` does NOT use this — its runtime_m_mode=1
        large-M path requires the full `num_workers`.
        """
        if not self._use_prefill:
            return self.num_workers
        base = min(80, self.num_workers)
        if output_size is None:
            return base
        # M_max = compile-time mbt (runtime M is capped to active_rows at
        # exec time, never larger).
        FLOOR = 24
        bn = 128
        m_tiles = (self.max_num_batched_tokens + bn - 1) // bn
        n_tiles = (output_size + bn - 1) // bn
        single_wave = m_tiles * n_tiles
        return min(base, max(single_wave, FLOOR))

    def _fp8_linear(self, input_bf16, weight, weight_scale, output,
                    grid_dim, block_dim, residual=None, gate_mode: int = 0,
                    input_row_stride: int = None,
                    input_col_offset: int = 0,
                    share_quantize_tag: str = None,
                    input_fp8_override=None,
                    input_scale_override=None,
                    no_wave_collapse: bool = False):
        """Quantize BF16 input → FP8, then run the dense FP8 GEMM
        (fp8_gemm_dense_smallm/mediumm_sm100), optionally with TP
        allreduce + residual.

        Args
        ----
        input_bf16: (mbt, K) bf16 tensor — pre-quantize input.
        weight: (N, K) fp8_e4m3 raw checkpoint weight. If `weight_scale`
            is None, `weight` must instead be a BF16 weight and the layer
            falls back to the BF16 linear kernels (fixtures /
            pre-dequantized weights only).
        weight_scale: (N/128, K/128) float32 checkpoint scale. Attach the
            weight via `_attach_fp8_weight` / `_attach_raw_fp8_weight` so
            the scale stays in raw float32 layout.
        output: (mbt, N) bf16, 2D or 3D (M, H, D_per_head) — same
            contiguous byte layout; 3D keeps the head dim explicit for
            downstream BMM consumers.
        grid_dim / block_dim: used only by the BF16 fallback. The FP8
            kernel uses a persistent (num_workers, 1, 1) grid internally.
        residual: optional (mbt, N) bf16. With world_size>1 the GEMM goes
            to a partial buffer, then AllReduce + residual add. At
            world_size=1 the residual is added via elementwise_add (the
            kernel has no fused residual epilogue).
        gate_mode: 0=always run, 1=prefill iters only, 2=decode iters only.
            Mirrored into the GEMM's runtime_m_mode (0/2/3) and the
            quantize's active_mode so wrong-phase dual-dispatch tasks
            early-exit.
        input_row_stride / input_col_offset: read a column slice of a
            wider input buffer (QKV-a fused path): parent row stride +
            slice start column. K (= weight.dim(1)) sets how many cols are
            quantized per row. Defaults preserve contiguous reads.
        share_quantize_tag: dedup the input-quantize task across GEMMs
            reading the same input slice. The FIRST call with a given tag
            emits one quantize with active_mode=0 (always run, so both
            phases see fresh data); later calls skip the quantize and
            reuse the cached FP8/scale buffers.
        input_fp8_override / input_scale_override: pre-allocated FP8 +
            scale buffers that bypass the shared per-reduction-size cache.
            Used by the fused rmsnorm+quantize paths to give the fused
            task a unique writer per buffer — sharing across layers makes
            it a cross-layer join-consumer and trips annotated_graph's
            case-3 (fork+join producer) check. Must be both set or both
            None.
        no_wave_collapse: keep the blanket (80/full) num_workers instead
            of the per-call-site `ceil(N/128)` wave-collapse. Required by
            call sites whose output feeds a `linear_fp8_bmm_*` GEMM — the
            BMM is templated on the per-head N-tile shape and the
            producing GEMM's grid must stay at the validated value
            (q_b_nope → q_nope_fp8 → BMM chain).
        """
        if weight_scale is None:
            # BF16 fallback for fixtures or pre-converted weights without
            # FP8 scale metadata.
            if residual is not None:
                if self.world_size > 1:
                    idx = getattr(self, "_tp_residual_linear_idx", 0)
                    self._tp_residual_linear_idx = idx + 1
                    partial = self._new_tp_partial(
                        output, f"tp_bf16_residual_partial_{idx}")
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

        if input_bf16.num_dims != 2:
            raise ValueError("FP8 linear v2 expects 2D input.")
        if output.num_dims not in (2, 3):
            raise ValueError("FP8 linear v2 expects 2D or 3D output.")
        if weight.num_dims != 2 or weight_scale.num_dims != 2:
            raise ValueError("FP8 linear v2 expects 2D weight + scale.")

        dense_nw = (self._fp8_dense_num_workers()
                    if no_wave_collapse
                    else self._fp8_dense_num_workers(weight.dim(0)))
        reduction_size = weight.dim(1)
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
        emit_quantize = True
        if share_quantize_tag is not None:
            already = getattr(self, "_fp8_quantize_emitted", set())
            if share_quantize_tag in already:
                emit_quantize = False
            else:
                already.add(share_quantize_tag)
                self._fp8_quantize_emitted = already

        if emit_quantize:
            # Shared-quantize uses active_mode=0 (always run) so both
            # phases see fresh data; otherwise gate_mode maps to the
            # quantize active_mode (1→2 prefill-only, 2→3 decode-only).
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

        # gate_mode → GEMM runtime_m_mode (1→2 prefill-only, 2→3
        # decode-only) so the wrong-phase dual-dispatch GEMM early-exits.
        gemm_runtime_m_mode = (2 if gate_mode == 1
                               else 3 if gate_mode == 2
                               else 0)

        # finen (fine-N CUDA-core GEMV) is **M=1-ONLY** — it ignores M and writes
        # only C[col] = output row 0 (fp8_gemm_dense_finen_sm100.cuh:180,290). It is
        # 3.1× faster than mediumm at decode M=1 (faithful in-MPK gate 2026-06-18:
        # qkv_a 8.35µs vs mediumm 26.02µs @nw=136, cos=1.0) for the qualifying dense
        # projections (qkv_a N=2176, shared_gate_up N=512, q_b_pe). Default-ON ONLY
        # for decode-only builds (max_num_batched_tokens == 1 ⇒ M is always 1); set
        # MPK_DSV3_DENSE_FINEN=0 to force mediumm. The mbt==1 guard is MANDATORY:
        # enabling finen when mbt>1 (prefill/batched) leaves output rows 1..M-1
        # UNWRITTEN (silent miscompute). e2e TP8 perf verdict pending (8×B200).
        # 6/20: fine-N dense GEMM is the DEFAULT for decode (mbt==1 ⇒ M=1). Eligibility
        # (N≤2304, N%16, K%512) + the MANDATORY mbt==1 guard unchanged (finen at mbt>1
        # leaves rows 1..M-1 unwritten → silent miscompute, so prefill stays mediumm).
        _use_finen = (self.max_num_batched_tokens == 1
                      and gate_mode == 0
                      and weight.dim(0) <= 2304
                      and weight.dim(0) % 16 == 0
                      and weight.dim(1) % 512 == 0)

        if residual is None:
            if _use_finen:
                self.mpk.fp8_gemm_dense_finen_layer(
                    input_fp8=input_fp8, weight_fp8=weight,
                    input_scale=input_scale, weight_scale=weight_scale,
                    output=output, num_workers=dense_nw)
            else:
                dsv3_tasks.fp8_gemm_dense_layer(
                    self.mpk,
                    input_fp8=input_fp8,
                    weight_fp8=weight,
                    input_scale=input_scale,
                    weight_scale=weight_scale,
                    output=output,
                    num_workers=dense_nw,
                    runtime_m_mode=gemm_runtime_m_mode,
                )
            return

        if self.world_size > 1:
            idx = getattr(self, "_tp_residual_linear_idx", 0)
            self._tp_residual_linear_idx = idx + 1
            partial = self._new_tp_partial(output, f"tp_v2_residual_partial_{idx}")
            if _use_finen:
                self.mpk.fp8_gemm_dense_finen_layer(
                    input_fp8=input_fp8, weight_fp8=weight,
                    input_scale=input_scale, weight_scale=weight_scale,
                    output=partial, num_workers=dense_nw)
            else:
                dsv3_tasks.fp8_gemm_dense_layer(
                    self.mpk,
                    input_fp8=input_fp8,
                    weight_fp8=weight,
                    input_scale=input_scale,
                    weight_scale=weight_scale,
                    output=partial,
                    num_workers=dense_nw,
                    runtime_m_mode=gemm_runtime_m_mode,
                )
            self._allreduce_residual(partial, output, residual,
                                     gate_mode=gate_mode)
            return

        # TP=1 path: dense GEMM into a partial buffer, then add residual.
        partial = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, weight.dim(0)),
            dtype=bfloat16, name=f"fp8_v2_partial_{id(weight)}",
            io_category="cuda_tensor",
        )
        if _use_finen:
            self.mpk.fp8_gemm_dense_finen_layer(
                input_fp8=input_fp8, weight_fp8=weight,
                input_scale=input_scale, weight_scale=weight_scale,
                output=partial, num_workers=dense_nw)
        else:
            dsv3_tasks.fp8_gemm_dense_layer(
                self.mpk,
                input_fp8=input_fp8,
                weight_fp8=weight,
                input_scale=input_scale,
                weight_scale=weight_scale,
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

    def _fused_rmsnorm_quantize_qkv_a_tag(self, layer_idx: int) -> str:
        """Deterministic share_quantize_tag for the fused
        rmsnorm + qkv_a-quantize path. Pre-populated in
        `_fp8_quantize_emitted` so the qkv_a `_fp8_linear` call skips its
        internal quantize (the fused task already wrote the FP8 + scale
        buffers)."""
        return f"layer_{layer_idx}_qkv_a_fused_rmsnorm_quantize"

    def _emit_fused_rmsnorm_qkv_a_quantize(self,
                                            input_x: 'DTensor',
                                            w_norm: 'DTensor',
                                            layer_idx: int,
                                            reduction_size: int) -> str:
        """Fused input-layernorm + qkv_a-side FP8 quantize in one task.

        Returns the share_quantize_tag the caller MUST forward to
        `_fp8_linear(share_quantize_tag=...)`.

        Buffer ownership (case-3 constraint): unlike the standalone
        quantize path (cross-layer SHARED buffers via
        `_fp8_mbt_buffers_for_reduction_f32scale`), the fused task
        allocates **per-layer-unique** FP8 + scale buffers. The fused task
        takes its FP8/scale outputs as store_in_dmem inputs in the task
        graph; sharing those buffers across layers gives them multiple
        writers, makes the fused task a join-consumer, and turns its
        input's producer into fork-producer + join-producer at once —
        rejected by annotated_graph as case 3. Per-layer-unique buffers
        keep one writer per buffer.

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
            # Per-layer-unique BF16 rmsnorm output buffer (case-3: the
            # shared self.rmsnorm_out has one writer per layer, which
            # would make the fused task a join-consumer).
            self._fused_rmsnorm_out_per_layer[layer_idx] = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=bfloat16,
                name=f"fused_rmsnorm_out_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
        input_fp8, input_scale = self._fused_qkv_a_bufs[layer_idx]
        rmsnorm_out_bf16 = self._fused_rmsnorm_out_per_layer[layer_idx]
        # Pre-populate the emitted-set so _fp8_linear skips the internal
        # quantize that would redundantly rewrite the fused-task output.
        already = getattr(self, "_fp8_quantize_emitted", set())
        tag = self._fused_rmsnorm_quantize_qkv_a_tag(layer_idx)
        already.add(tag)
        self._fp8_quantize_emitted = already

        # scale_ue8m0=False → row-major (M, K/128) f32 scales (the dense
        # GEMM's layout). emit_bf16=False: nothing downstream reads the
        # bf16 in fused mode (the qkv_a GEMM reads fp8/scale directly).
        dsv3_tasks.fused_rmsnorm_quantize_fp8_layer(
            self.mpk,
            input=input_x,
            weight=w_norm,
            output_bf16=rmsnorm_out_bf16,
            output_fp8=input_fp8,
            output_scale=input_scale,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
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
        """Fused q_a_layernorm + per-token-group FP8 quantize.

        Analogous to `_emit_fused_rmsnorm_qkv_a_quantize` but for the
        INNER q_a layernorm (after the qkv_a GEMM, before the q_b GEMMs):
        collapses the rmsnorm_layer + q_b-input-quantize chain into one
        fused task.

        Returns `(input_fp8, input_scale, tag)` — the caller threads
        input_fp8/scale to q_b `_fp8_linear(..., input_fp8_override=...,
        input_scale_override=...)` calls. The tag is also pre-populated
        in `_fp8_quantize_emitted` so a `share_quantize_tag=tag` arg makes
        downstream callers skip their quantize emission.

        Buffer ownership: per-layer-unique FP8/scale buffers (case-3 —
        same rationale as `_emit_fused_rmsnorm_qkv_a_quantize`).

        emit_bf16=False: no consumer reads the rmsnormed q_a as BF16 —
        the q_b GEMMs read the FP8/scale via the override threading.

        scale_ue8m0=False (float32): the dense GEMM family's expected
        row-major (M, K/128) scale layout.
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
        # Pre-populate the emitted-set so _fp8_linear skips the internal
        # quantize that would redundantly rewrite the fused-task output.
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
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
            process_dim=reduction_size,
            in_offset_elems=in_offset_elems,
            out_offset_elems=out_offset_elems,
            scale_ue8m0=False,
            emit_bf16=False,
        )
        return input_fp8, input_scale, tag

    def _attach_bmm_weight_pair(self, state_dict, key, scale_key, name):
        """Attach a per-head BMM weight + its scale tensor as
        (`name`, `name`_scale)."""
        w = self.mpk.attach_input(
            torch_tensor=state_dict[key], name=name)
        s = self.mpk.attach_input(
            torch_tensor=state_dict[scale_key], name=f"{name}_scale")
        return w, s

    def _bmm_decode_q_path(self, state_dict, attn, layer_idx, qb_slice_kwargs,
                           qb_share_tag=None,
                           qb_input_fp8_ovr=None,
                           qb_input_scale_ovr=None):
        """Decode Q path: replaces the absorbed q_b_proj decode GEMM with a
        per-head BMM chain that loads the unabsorbed weights at runtime:

          fp8_linear(q_a, q_b_pe)         → q_pe   (mbt, H, 64)   bf16
          fp8out GEMM(q_a, q_b_nope)      → q_nope_fp8 + UE8M0 scale
          linear_fp8_bmm(q_nope_fp8, kv_b_k_bmm) → q_nope_abs (mbt, H, 512)
          assemble_q_decode(pe_only)      → q_nope_pe (mbt, H, 576)

        vs the absorbed monolith (single (H*576, q_lora) FP8 GEMM): smaller
        per-task weight loads (less TMA traffic per CTA) and no absorbed
        weight buffer.

        qb_share_tag / qb_input_fp8_ovr / qb_input_scale_ovr: the fused
        q_a layernorm+quantize task already wrote q_a's FP8/scale into the
        per-layer override buffers; all q_b GEMMs here read those and skip
        their own input quantize via the share tag.
        """
        H_local = self.num_local_q_heads
        mbt = self.max_num_batched_tokens
        # Buffers are shared across layers (allocated once).
        if not hasattr(self, "_bmm_decode_buffers"):
            self._bmm_decode_buffers = {}
            # q_b_pe output, 3D so the BMM input partition map can see H.
            self._bmm_decode_buffers["q_pe_3d"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 64), dtype=bfloat16,
                name="q_pe_decode_3d", io_category="cuda_tensor")
            # FP8 q_nope + UE8M0 packed scale for the BMM input. K=128 ≤ 512
            # so packed_K = 1 (one uint32 per row).
            self._bmm_decode_buffers["q_nope_fp8"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 128), dtype=float8_e4m3,
                name="q_nope_decode_fp8", io_category="cuda_tensor")
            self._bmm_decode_buffers["q_nope_scale"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 1), dtype=uint32,
                name="q_nope_decode_scale", io_category="cuda_tensor")
            # BMM output FUSE: q_nope_abs is a slice view
            # q_nope_pe[:, :, :512] of the (mbt, H, 576) torch parent —
            # strides (H*576, 576, 1) put each head's 512 nope cols at the
            # [h*576 : h*576+512) slot, so the BMM TMA writes the per-head
            # [nope|pe] interleaved layout directly (no separate buffer,
            # no full assemble pass).
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
        # 1) q_b_pe FIRST: its _fp8_linear emits the q_a input-side
        # quantize (when not already emitted by the fused task) so the
        # q_b_nope GEMM below can read the same q_a FP8 buffer.
        self._fp8_linear(
            self.q_a_out, w_q_b_pe, s_q_b_pe, q_pe_3d,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_pe.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
            gate_mode=2 if self._use_prefill else 0,
            share_quantize_tag=qb_share_tag,
            input_fp8_override=qb_input_fp8_ovr,
            input_scale_override=qb_input_scale_ovr,
            **qb_slice_kwargs)
        # 2) q_b_nope FP8 dense GEMM with epilogue UE8M0 quantize — emits
        # q_nope_fp8 + q_nope_scale directly (no separate quantize task).
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
            # BMM-feeding GEMM: keep the blanket num_workers (no per-call-
            # site wave-collapse) so the downstream BMM template's
            # validated grid is preserved.
            num_workers=self._fp8_dense_num_workers(),
            runtime_m_mode=gemm_runtime_m_mode,
        )
        # 3) BMM(q_nope_fp8, kv_b_k_bmm) → q_nope_abs (mbt, H, 512).
        # swapAB body (dense=False): UE8M0-packed scales, D_out shardable.
        w_kvk_bmm, s_kvk_bmm = self._attach_bmm_weight_pair(
            state_dict, f"{attn}kv_b_k_bmm.weight",
            f"{attn}kv_b_k_bmm.weight_scale_ue8m0",
            f"layer_{layer_idx}_kv_b_k_bmm")
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
        # 4) Assemble (PE-only): the BMM already wrote nope into
        # q_nope_pe[:, :, :512] via the slice-view fuse; only q_pe goes
        # into the tail [512:576).
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
        """Decode O path: replaces the load-time-absorbed decode o_proj
        (fused with W_UV) with runtime BMM + smaller linear:

          quantize(attn_out)                  → attn_out_fp8 (mbt, H, 512)
          BMM(attn_out_fp8, kv_b_v_bmm_dense) → attn_out_reduced (mbt, H*128)
          fp8_linear+residual(o_proj_original) → attn_proj_out

        The BMM uses the DENSE block-scaled GEMM body (float32 128-K-group
        scales). o_proj_original.weight is the SAME (hidden, H*128) FP8
        weight the prefill path uses.

        On dual-dispatch builds the whole path is decode-only: quantize
        active_mode=3, o_proj gate_mode=2, and the BMM's MMA_N=16
        constraint. Writes attn_proj_out directly.
        """
        H_local = self.num_local_q_heads
        mbt = self.max_num_batched_tokens
        V_HEAD_DIM = 128  # post-attn V un-absorption dim per head
        KV_LORA = 512     # attn_out per-head dim
        nk_o = (KV_LORA + 127) // 128  # 128-K groups per head (f32 scale)
        # Buffers are shared across layers (allocated once).
        if not hasattr(self, "_bmm_decode_o_buffers"):
            self._bmm_decode_o_buffers = {}
            self._bmm_decode_o_buffers["attn_out_fp8"] = self.mpk.new_tensor(
                dims=(mbt, H_local, KV_LORA), dtype=float8_e4m3,
                name="attn_out_bmm_fp8", io_category="cuda_tensor")
            self._bmm_decode_o_buffers["attn_out_scale_f32"] = self.mpk.new_tensor(
                dims=(mbt, H_local, nk_o), dtype=float32,
                name="attn_out_bmm_scale_f32", io_category="cuda_tensor")
            # BMM output (mbt, H, 128), allocated 2D so it feeds
            # _fp8_linear without a reshape (BMM accepts 2D output).
            self._bmm_decode_o_buffers["attn_out_reduced"] = self.mpk.new_tensor(
                dims=(mbt, H_local * V_HEAD_DIM), dtype=bfloat16,
                name="attn_out_reduced_2d", io_category="cuda_tensor")

        attn_out_fp8 = self._bmm_decode_o_buffers["attn_out_fp8"]
        attn_out_scale_f32 = self._bmm_decode_o_buffers["attn_out_scale_f32"]
        attn_out_reduced = self._bmm_decode_o_buffers["attn_out_reduced"]

        active_mode_o = 3 if self._use_prefill else 0  # decode-only gate

        # 1) quantize attn_out BF16 → FP8 + float32 128-group scale.
        # Input self.attn_out is (mbt, H*512) 2D; the 3D FP8 output has the
        # same byte layout, and the row-major [mbt*H, nk] scale views as
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
        # 2) per-head BMM via the DENSE block-scaled body. kv_b_v_bmm_dense
        # is prepared in demo.py: weight (H, 128, 512) FP8 + float32 block
        # scale (H, 1, nk).
        w_kvv_bmm, s_kvv_bmm = self._attach_bmm_weight_pair(
            state_dict, f"{attn}kv_b_v_bmm_dense.weight",
            f"{attn}kv_b_v_bmm_dense.weight_scale_inv",
            f"layer_{layer_idx}_kv_b_v_bmm_dense")
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
        # 3) unabsorbed o_proj linear with residual.
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
        input_stride: int = None,
    ):
        """Chunked-prefill kv_b_k / kv_b_v projection: quantize the latent
        rows to FP8 (sequence-rows buffer), then dense FP8 GEMM with
        runtime_m_mode=1 (prefill-only; runtime M = the contiguous-cache KV
        length step+q_len).

        ckv is a NARROW VIEW of the per-layer contiguous KV buffer
        ([rows, 576] → cols [0, 512)); `input_stride` carries the parent row
        width because quantize_fp8_layer does NOT derive the stride from the
        view (column-slice contract — defaults to the view width).

        shared_quantize_tag: kv_b_k and kv_b_v quantize the SAME latent
        bytes — pass the same tag to both calls in a layer so only the
        FIRST emits the quantize task and the second reuses the buffer.

        The GEMM keeps the full `self.num_workers` (NOT the wave-collapsed
        `_fp8_dense_num_workers`) — the runtime_m_mode=1 large-M path
        crashes at lower worker counts.
        """
        if weight_scale is None:
            raise ValueError("kv_b prefill projection requires FP8 weight scale.")
        buf_tag = shared_quantize_tag if shared_quantize_tag is not None else tag
        input_fp8, input_scale = self._fp8_sequence_buffers_for_reduction(
            self.kv_lora_rank, tag=buf_tag)
        emit_quantize = True
        if shared_quantize_tag is not None:
            already_quantized = getattr(self, "_kv_b_quantized_tags", set())
            if shared_quantize_tag in already_quantized:
                emit_quantize = False
            else:
                already_quantized.add(shared_quantize_tag)
                self._kv_b_quantized_tags = already_quantized
        if emit_quantize:
            self.mpk.quantize_fp8_layer(
                input=ckv,
                output_fp8=input_fp8,
                output_scale=input_scale,
                grid_dim=(input_fp8.dim(0), 1, 1),
                block_dim=(128, 1, 1),
                scale_ue8m0=False,
                active_mode=1,
                hidden_size_override=self.kv_lora_rank,
                input_stride_override=input_stride,
            )
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

    def _silu_mul_fp8_linear(self, silu_input, silu_bf16_output, weight,
                             weight_scale, output, silu_grid_dim,
                             linear_grid_dim, block_dim, residual=None):
        """SiLU-mul on the interleaved gate|up buffer, then `_fp8_linear`
        (down projection, optional residual).
        """
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

        # bs=1 one-shot prefill: with mbt >= prompt_length the scheduler feeds
        # the whole prompt in ONE step (num_new_tokens = min(prompt_len-step,
        # mbt)) — there is no scheduler-visible chunk loop. The prefill tasks
        # (unabsorbed q_b GEMMs / kv_b up-projection / chunked attention /
        # prefill o_proj) dual-register alongside decode and gate on runtime
        # Q_LEN (> 8 = prefill iter, <= 8 = decode iter). All KV flows through
        # the per-layer contiguous buffer: the append task writes the new
        # rows; prefill consumes them via narrow views (no paged cache, no
        # gather).
        self._use_prefill = mbt > 8
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

        # MLA QKV-a fusion: one qkv_a_out (mbt, QKV_A_FUSED_N) buffer; the
        # fused FP8 GEMM writes q_a + c_latent + k_pe in one task and
        # downstream consumers read their slice via (row_stride, offset)
        # kernel params. Layout per row:
        #   cols [0    : 1536) = q_a_out      (q_lora_rank = 1536)
        #   cols [1536 : 2048) = c_latent_out (kv_lora_rank = 512)
        #   cols [2048 : 2112) = k_pe_out real (QK_ROPE_HEAD_DIM = 64)
        #   cols [2112 : 2176) = k_pe_out zero pad (= MMA_M tail)
        qkv_a_total = QKV_A_FUSED_N
        self.qkv_a_out = self.mpk.new_tensor(
            dims=(mbt, qkv_a_total),
            dtype=bfloat16, name="qkv_a_out", io_category="cuda_tensor",
        )
        # Each logical slot is an `mpk.narrow` view of qkv_a_out: the view
        # bakes the slot's byte offset into base_ptr (view_offset) and
        # inherits the parent row stride into view.stride[0]; task_register,
        # the FP8 TMA descriptor builder (tma.cuh) and annotated_graph's 2D
        # bbox overlap check all consume the view metadata. The explicit
        # *_offset / row_stride params still passed at call sites encode
        # 0 offset + parent row stride, matching what the view supplies.
        self._qkv_a_row_stride = qkv_a_total
        self._qkv_a_q_offset = 0
        self._qkv_a_c_latent_offset = 0
        self._qkv_a_k_pe_offset = 0
        self.q_a_out = self.mpk.narrow(
            self.qkv_a_out, dim=1, start=0, length=self.q_lora_rank)
        self.q_a_out_buf = None
        # q_b output (after absorption): [batch, num_local_q_heads * qk_head_dim]
        self.q_nope_pe_buf = None
        # Allocated as a 3D torch tensor so decode writes per-head
        # [nope_512|pe_64] in the exact layout consumed by fused ROPE and
        # MLA. The dormant BMM helper can still attach slice views if it is
        # re-enabled.
        import torch as _torch
        self._q_nope_pe_torch = _torch.zeros(
            mbt, self.num_local_q_heads, self.qk_head_dim,
            dtype=_torch.bfloat16, device="cuda")
        self.q_nope_pe = self.mpk.attach_input(
            self._q_nope_pe_torch, name="q_nope_pe")
        # Decode consumes the absorbed [CKV, KPE] Q (q_nope_pe). Prefill
        # consumes vLLM's original per-head split Q: [nope(128), rope(64)].
        if self._use_prefill:
            self.q_nope = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads * QK_NOPE_HEAD_DIM),
                dtype=bfloat16, name="q_nope", io_category="cuda_tensor",
            )
            self.q_pe = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads * QK_ROPE_HEAD_DIM),
                dtype=bfloat16, name="q_pe", io_category="cuda_tensor",
            )
        else:
            self.q_nope = None
            self.q_pe = None
        # kv_a outputs (c_latent + k_pe) are `mpk.narrow` views of the
        # fused qkv_a_out, mirroring q_a_out above (see the view note at
        # qkv_a_out).
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
        if self._use_prefill:
            # Prefill-side buffers. Rows = max KV length padded to a multiple
            # of 128 (chunked-prefill TMA box BN_BOX=128: unpadded rows OOB
            # NaN-fill V/K SMEM, which propagates through hmma16 0*NaN=NaN).
            # ZERO-INITIALIZED via attach_input(torch.zeros) — the kv_b GEMMs
            # only write rows [0, step+q_len) each prefill iteration, so the
            # tail rows the attention's BN=128 TMA windows can touch must be
            # valid (zero) bf16, not pool garbage (masked scores still NaN-
            # poison the PV product if V holds NaN bit patterns).
            _raw_rows = (self.mpk.max_num_batched_requests
                         * self.mpk.max_seq_length)
            self._prefill_kv_rows = ((_raw_rows + 127) // 128) * 128
            # `kpe_sep_v2` receives the identity-copy "phantom bridge"
            # between the per-layer KV append and the chunked-prefill kernel
            # — not real compute, purely to legalize the task graph (the
            # append is a fork-producer and chunked_prefill a join-consumer;
            # see the identity_layer call in the attention block for the
            # case-3 rationale).
            self._kpe_sep_v2_torch = _torch.zeros(
                self._prefill_kv_rows, QK_ROPE_HEAD_DIM,
                dtype=_torch.bfloat16, device="cuda")
            self.kpe_sep_v2 = self.mpk.attach_input(
                self._kpe_sep_v2_torch, name="kpe_sep_v2")
            # Transient per-head K/V materialized by the kv_b up-projection
            # GEMMs each prefill iteration (consumed by chunked attention,
            # never part of the persistent cache — the cache stays the
            # compressed latent in the per-layer contiguous KV buffer).
            self._prefill_k_nope_torch = _torch.zeros(
                self._prefill_kv_rows,
                self.num_local_q_heads * QK_NOPE_HEAD_DIM,
                dtype=_torch.bfloat16, device="cuda")
            self.prefill_k_nope = self.mpk.attach_input(
                self._prefill_k_nope_torch, name="prefill_k_nope")
            self._prefill_v_torch = _torch.zeros(
                self._prefill_kv_rows,
                self.num_local_q_heads * V_HEAD_DIM,
                dtype=_torch.bfloat16, device="cuda")
            self.prefill_v = self.mpk.attach_input(
                self._prefill_v_torch, name="prefill_v")
        else:
            self._prefill_kv_rows = None
            self.kpe_sep_v2 = None
            self.prefill_k_nope = None
            self.prefill_v = None
        # MLA decode partial outputs (bf16 partials).
        # MLA kernel writes blocks at stride D_V*128 and LSE at stride 128.
        # TP kernels use split-K: each split handles one KV tile.
        # TILE_S=128 by default; TILE_S=32 when MPK_DSV3_MLA_FINESPLIT=1 (TP=8
        # only — the finesplit macro is keyed on num_heads==16).  Buffer must be
        # sized for the LARGEST num_splits that any kernel instance can write:
        #   ceil(max_seq_length / TILE_S)
        # Using TILE_S=128 when finesplit writes TILE_S=32 underestimates by 4×
        # → IMA on the overflowing ranks.
        # 6/20: finesplit (TILE_S=32) is the DEFAULT for the TP8 decode build; the
        # partial-O/LSE buffers must be sized for ceil(seq/32) splits to match.
        _mla_tile_s = (
            32
            if (self.world_size == 8
                and self.max_num_batched_tokens == 1)
            else 128
        )
        mbr = self.mpk.max_num_batched_requests
        if self.world_size > 1:
            max_splits = (self.mpk.max_seq_length + _mla_tile_s - 1) // _mla_tile_s
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
            max_splits = (self.mpk.max_seq_length + _mla_tile_s - 1) // _mla_tile_s
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
        if self._use_prefill:
            # Chunked-prefill attention output: per-head ORIGINAL v dim
            # (128, before absorption). Consumed by the prefill o_proj.
            self.attn_unabsorbed = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads * V_HEAD_DIM),
                dtype=bfloat16,
                name="attn_unabsorbed",
                io_category="cuda_tensor",
            )
        else:
            self.attn_unabsorbed = None
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

    def _attach_fp8_weight(self, state_dict, key, name):
        """Attach FP8 weight + float32 scale_inv (the dense FP8 GEMM
        family's raw block-scale format), or a BF16 weight as fallback
        (returns scale=None → `_fp8_linear` takes its BF16 path). Same
        `(w, s)` contract as `_attach_raw_fp8_weight`, which additionally
        REQUIRES the scale."""
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
        """fp32 → UE8M0 (8-bit exponent only). MUST use CEIL rounding to
        match the kernel-side `encode_ue8m0` in
        per_token_group_quantize_fp8.cuh (`ceilf(log2f(scale))`) — the
        kernel re-encodes SFA at runtime, so the Python weight-pack must
        follow the same rounding convention."""
        pos = torch.where(t > 0, t, torch.full_like(t, 1e-30))
        p2 = torch.pow(2.0, torch.ceil(torch.log2(pos)))
        bits = p2.view(torch.int32)
        ue = ((bits >> 23) & 0xFF).to(torch.uint8)
        ue = torch.where(t > 0, ue, torch.zeros_like(ue))
        return ue

    @staticmethod
    def _pack_moe_scale_ue8m0(scale_per_row: torch.Tensor) -> torch.Tensor:
        """[dim, nk] fp32 → [num_sf_k, dim] uint32 row-major, UE8M0-packed
        (4 UE8M0 scales per uint32 along K).

        The grouped FP8 GEMM's SFA/SFB TMA descriptors expect this
        transposed layout (gd=[dim, num_sf_k] with dim innermost).

        scale_per_row: per-output-row dequant scale (after
        repeat_interleave along the output dim). For W13 this is reshape
        from (E, 2*intermediate, K/128); pass it flattened as
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

    @staticmethod
    def _requantize_moe_fp8_for_pow2(state_dict, wkey, skey):
        """Re-quantize an FP8 MoE expert payload against the CEIL-pow2
        (UE8M0) scales the grouped GEMM actually applies.

        The SM100 grouped GEMM consumes UE8M0 weight scales — i.e.
        2^ceil(log2(scale_inv)) per 128x128 block (`_float_to_ue8m0`,
        matching the kernel-side `encode_ue8m0`) — via the hardware
        block-scaled MMA. Checkpoint FP8 payloads, however, were quantized
        by DeepSeek against the RAW fp32 scale_inv (0% of which are powers
        of two; ceil factor mean 1.42x, max 2x). Packing ceiled scales
        over raw-quantized bytes inflates every block by
        2^ceil(log2 s)/s ∈ [1, 2) — measured 53% rel error / 1.51x norm on
        the production largem GEMM with real layer-3 weights.

        Fix: rescale the payload by s/2^ceil(log2 s) ∈ (0.5, 1] so
        (payload, ceil-packed scale) is self-consistent:
        q_new * 2^ceil(log2 s) ≈ q_old * s. Values only shrink, so no
        overflow; cost is ≤1 original-grid ulp of extra rounding
        (min-magnitude FP8 codes may flush to zero — same convention as
        the test-suite quantizer). Scale tensors stay RAW (the packer
        ceils them, which is now consistent), so the weight cache format
        is unchanged. One-shot per key (sentinel guard) — re-applying
        would compound the shrink.
        """
        sentinel = wkey + "._pow2_requantized"
        if state_dict.get(sentinel) is not None:
            return
        w = state_dict[wkey]
        s = state_dict[skey]
        assert w.dtype == torch.float8_e4m3fn, (wkey, w.dtype)
        E, N, K = w.shape
        r = s.float() / torch.pow(
            2.0, torch.ceil(torch.log2(s.float().clamp(min=1e-30))))
        for e in range(E):  # per-expert to bound transient fp32 memory
            rf = (r[e].repeat_interleave(128, 0)
                      .repeat_interleave(128, 1)[:N, :K]).to(w.device)
            w[e] = (w[e].float() * rf).to(torch.float8_e4m3fn)
        state_dict[sentinel] = torch.tensor(True)

    def _attach_raw_fp8_weight(self, state_dict, key, name):
        """Attach checkpoint-style FP8 weight + float32 block scale
        (required). The dense FP8 GEMM consumes the original block-scale
        layout [output/128, K/128] (no UE8M0 packing)."""
        scale_key = f"{key}_scale_inv"
        if scale_key not in state_dict:
            raise ValueError(f"{key} requires {scale_key} for FP8 dense GEMM.")
        if state_dict[key].dtype != torch.float8_e4m3fn:
            raise TypeError(f"{key} must be torch.float8_e4m3fn.")
        w = self._safe_attach(state_dict[key], name)
        s = self._safe_attach(
            state_dict[scale_key].float().contiguous(), f"{name}_scale")
        return w, s

    def _build_mla_attention_megakernel(self, layer_idx: int,
                                        state_dict: dict):
        """The default fused decode-attention path (selected by
        `_use_attn_megakernel` for the bs=1 TP8/EP2/B200 decode geometry).

        Registers ONE megakernel task (attn_block_megakernel_sm100) in place
        of the ~13-task decode attention chain. The kernel runs the whole
        block (qkv_a -> q_a_ln + kv_a_ln -> q_b -> YaRN rope -> kv_append ->
        MLA decode -> reduce -> W_UV BMM -> o_proj + residual) on a 136-CTA
        cooperative grid synced by the MPK atomic grid_barrier. Decode-only:
        the kernel is hard-wired to the bs=1 M=1 TP8 decode geometry.

        REUSES the SAME weight tensors the prefill/compat chain binds. Writes
        the per-layer attn_proj_out (pre-AR; +residual fused). The post-attn
        rmsnorm + MLP/MoE and the AllReduce stay OUTSIDE this task (unchanged).

        The asserts below restate the `_use_attn_megakernel` predicate the
        caller already checked — defensive, since the kernel's grid_barrier
        participant count + per-rank head count + step[0] sourcing all bake in
        this exact geometry.
        """
        assert self.mpk.num_workers == 136, (
            "attn-block megakernel needs num_workers==136 (B200 148-SM); "
            f"got {self.mpk.num_workers}. The kernel's grid_barrier participant "
            "count (ATTN_NUM_WORKERS) is hard-wired to 136.")
        assert self.max_num_batched_tokens == 1, (
            "attn-block megakernel is a bs=1 / M=1 decode kernel; got "
            f"mbt={self.max_num_batched_tokens}.")
        assert self.world_size == 8, (
            "attn-block megakernel is hard-wired to TP8 (16 local q-heads, "
            f"K_HLOCAL=16); got world_size={self.world_size}.")
        # The kernel hardcodes step = runtime_config.step[0], so it is only
        # correct for the single-active-request decode (row 0). Guard it.
        assert self.mpk.max_num_batched_requests == 1, (
            "attn-block megakernel sources the decode position from "
            "step[0]; it requires max_num_batched_requests==1, got "
            f"{self.mpk.max_num_batched_requests}.")
        prefix = f"model.layers.{layer_idx}."
        attn = f"{prefix}self_attn."

        # --- bf16 RMSNorm of the layer input (input_layernorm) -> `hidden` ---
        # The kernel's S2 quantizes a BF16 RMSNorm'd hidden itself, so it needs
        # the bf16 norm output. The default path's fused rmsnorm+qkv_a-quant
        # task runs with emit_bf16=False (its bf16 out is unused), so we emit a
        # dedicated per-layer bf16 RMSNorm here. (Minor: the caller's fused
        # task still runs but its outputs are unread on this path — the main
        # agent can gate it out later for a small efficiency win.)
        w_input_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
            name=f"layer_{layer_idx}_attnmega_input_layernorm")
        hidden_bf16 = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_attnmega_hidden_bf16",
            io_category="cuda_tensor")
        self.mpk.rmsnorm_layer(
            input=self.x, weight=w_input_ln, output=hidden_bf16,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
            process_dim=self.hidden_size)

        # --- weights (SAME tensors the chain binds) -------------------------
        # qkv_a_proj: fp8 (2176,7168) + f32 per-128-block scale_inv (17,56). The
        # kernel's GEMV reads this fp32 [N/128,K/128] scale as a plain fp32
        # (Wsc[(n>>7)*nk + g]), the SAME format the production
        # fp8_gemm_dense_finen GEMV uses — RESOLVED, no reconcile needed.
        w_qkv_a, s_qkv_a = self._attach_fp8_weight(
            state_dict, f"{attn}qkv_a_proj.weight",
            f"layer_{layer_idx}_attnmega_qkv_a_proj")
        # q_a_layernorm (1536,) + kv_a_layernorm (512,) bf16, CONCATENATED into
        # one ln_weights (2048,) buffer (the kernel reads q_a_ln at [0:1536) and
        # kv_a_ln at [1536:2048)) to stay under MAX_INPUTS_PER_TASK=14.
        ln_weights_pt = torch.cat([
            state_dict[f"{attn}q_a_layernorm.weight"].to(torch.bfloat16),
            state_dict[f"{attn}kv_a_layernorm.weight"].to(torch.bfloat16),
        ], dim=0).contiguous()
        w_ln = self.mpk.attach_input(
            torch_tensor=ln_weights_pt,
            name=f"layer_{layer_idx}_attnmega_ln_weights")
        # q_b_proj ABSORBED decode form: fp8 (9216=16*576,1536) + f32 (72,12).
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight",
            f"layer_{layer_idx}_attnmega_q_b_proj")
        # W_UV per-head BMM: kv_b_v_bmm_dense fp8 (16,128,512) + f32 (16,1,4).
        # The (16,1,4) f32 scale MATCHES the kernel's `const float* kvbv_s`.
        w_kvv, s_kvv = self._attach_bmm_weight_pair(
            state_dict, f"{attn}kv_b_v_bmm_dense.weight",
            f"{attn}kv_b_v_bmm_dense.weight_scale_inv",
            f"layer_{layer_idx}_attnmega_kv_b_v_bmm_dense")
        # o_proj_original: fp8 (7168,2048) + f32 (56,16).
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj_original.weight",
            f"layer_{layer_idx}_attnmega_o_proj_original")

        # --- flat contiguous KV buffer (the kernel both reads history rows
        # [0,step) and writes the current row [step]). Rows padded to a 128
        # multiple (same as the chain's mla_decode_kv). The megakernel is the
        # SOLE writer of this buffer (S3/S5 do the kv_append internally), so no
        # separate mla_kv_append task is registered — avoids a multi-writer
        # case-3 on the same buffer. ---------------------------------------
        _kv_rows_raw = (self.mpk.max_num_batched_requests
                        * self.mpk.max_seq_length)
        _kv_rows_pad = ((_kv_rows_raw + 127) // 128) * 128
        mla_decode_kv = self.mpk.new_tensor(
            dims=(_kv_rows_pad, self.qk_head_dim),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_attnmega_kv_contig",
            io_category="cuda_tensor")

        # --- cos/sin CONCATENATED into one cos_sin (max_seq, 128) buffer:
        # [cos(64) | sin(64)] per row (the kernel reads cos at
        # cos_sin[pos*128 + d], sin at cos_sin[pos*128 + 64 + d]). Layer-
        # independent → built once and cached on self. ---------------------
        if not hasattr(self, "_attnmega_cos_sin"):
            cos_sin_pt = torch.cat(
                [self._rope_cos_buf, self._rope_sin_buf],
                dim=1).contiguous()  # (max_seq, 128) bf16, on cuda
            self._attnmega_cos_sin_pt = cos_sin_pt  # keep alive (raw ptr)
            self._attnmega_cos_sin = self.mpk.attach_input(
                torch_tensor=cos_sin_pt, name="attnmega_cos_sin")

        # --- per-layer output (pre-AR attn proj + residual) -----------------
        self.attn_proj_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_attnmega_attn_proj_fused",
            io_category="cuda_tensor")

        # --- scratch (barrier + all inter-stage activations) ----------------
        # %16 (not just %2): the scratch tensor's element count (bytes/2) must be a
        # multiple of 8 for tensor_init's 16B-vec zero-init (the compile-time assert
        # that bit the first attn-megakernel build).
        assert ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES % 16 == 0
        attn_scratch = self.mpk.new_tensor(
            dims=(1, ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES // 2),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_attn_block_megakernel_scratch",
            io_category="cuda_tensor")
        # Zero the barrier counters (and the rest) before first use, exactly as
        # the FFN mega-task does for its barrier_scratch.
        # MPK_DSV3_SKIP_TENSORINIT_AFTER_STEP0 (default-OFF, byte-identical when
        # unset): make this per-step zero a runtime no-op on decode steps>=1.
        # SAFE here because the attn-block megakernel's sense/generation grid
        # barrier self-maintains (last arriver resets count before flipping gen;
        # gen is change-based, never needs reset) and every scratch activation
        # region is dead / fully-overwritten-before-read / read-only-at-written-
        # indices (g_mla_acc s<nsp, nsp clamped to 8 => ntask<=128<136) /
        # zeroed-inside-the-kernel (g_head_done, g_head_wuv_ready before B2). The
        # only thing this per-step zero protected was the cudaMalloc step-0
        # garbage (step 0 STILL zeroes). PREMISE for the skip staying safe: the
        # kernel keeps EXACTLY 136 CTAs hitting all 3 barriers each step with no
        # early-return, and this scratch stays per-layer-distinct. Verified by
        # first-principles + Codex + ablation-logic-reviewer; gate on a >=3-step
        # token-identity A/B before flipping the default.
        _skip_ti = (os.environ.get(
            "MPK_DSV3_SKIP_TENSORINIT_AFTER_STEP0") == "1")
        # MPK_DSV3_POISON_TENSORINIT_AFTER_STEP0 (default-OFF, DIAGNOSTIC): on
        # decode steps>=1 fill the scratch with a NaN sentinel instead of either
        # zeroing or skipping. Safety-tests the skip lever: if the skip premise
        # (every region overwritten-before-read / barrier self-resets) holds, the
        # poison never reaches the logits -> clean output; if any region is read-
        # before-write, the NaN propagates -> deterministic NaN/garbage (or hang),
        # IMMUNE to the FP-atomicAdd nondeterminism that makes token-identity an
        # inconclusive safety gate on this path. Mutually exclusive with the skip.
        _poison_ti = (os.environ.get(
            "MPK_DSV3_POISON_TENSORINIT_AFTER_STEP0") == "1") and not _skip_ti
        self.mpk.tensor_init_layer(
            target=attn_scratch,
            dummy=hidden_bf16,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            dummy_input_map=(-1, -1, -1),
            target_input_map=(-1, -1, -1),
            skip_after_step0=_skip_ti,
            poison_after_step0=_poison_ti,
        )

        # --- TP RowParallel o_proj combine (THE TP>1 correctness fix) -------
        # o_proj_original.weight is sharded on dim=1 (the contraction dim) =>
        # RowParallel => each rank's o_proj produces a FULL-hidden [7168]
        # PARTIAL from ONLY its local head subset. These per-rank partials MUST
        # be summed (AllReduce) across ranks, then the residual added EXACTLY
        # ONCE. The chain does this inside its o_proj _fp8_linear (the world_size
        # >1 branch). The fused-residual o_proj epilogue in the megakernel is
        # only correct at world_size==1 (single rank, all 64 heads local); at
        # TP>1 it leaves every rank holding (its-heads-only + residual), missing
        # (N-1)/N of the heads -> garbage. So at TP>1: the kernel writes a
        # residual-FREE partial (we bind a persistent ZERO buffer as its
        # `residual`, so out[n] = o_proj_dot + 0), then allreduce_layer sums the
        # partials AND adds the real residual once.
        if self.world_size > 1:
            # one persistent zero [mbt, hidden] buffer (zeroed once, never
            # rewritten) so the megakernel's fused-residual epilogue adds 0 and
            # the partial is residual-free. Shared across all attn-mega layers.
            if not hasattr(self, "_attnmega_zero_resid"):
                self._attnmega_zero_resid = self.mpk.new_tensor(
                    dims=(self.max_num_batched_tokens, self.hidden_size),
                    dtype=bfloat16,
                    name="attnmega_zero_resid",
                    io_category="cuda_tensor")
                # zero it once (the tensor_init task) using a dep-only dummy.
                self.mpk.tensor_init_layer(
                    target=self._attnmega_zero_resid,
                    dummy=hidden_bf16,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                    dummy_input_map=(-1, -1, -1),
                    target_input_map=(-1, -1, -1),
                )
            oproj_partial = self._new_tp_partial(
                self.attn_proj_out,
                f"layer_{layer_idx}_attnmega_oproj_partial")
            mega_out = oproj_partial
            mega_residual = self._attnmega_zero_resid
        else:
            mega_out = self.attn_proj_out
            mega_residual = self.x

        dsv3_tasks.attn_block_megakernel_layer(
            self.mpk,
            hidden=hidden_bf16,
            qkv_a_w=w_qkv_a,
            qkv_a_s=s_qkv_a,
            ln_weights=w_ln,
            q_b_w=w_q_b,
            q_b_s=s_q_b,
            cos_sin=self._attnmega_cos_sin,
            kv_cache=mla_decode_kv,
            kvbv_w=w_kvv,
            kvbv_s=s_kvv,
            oproj_w=w_o,
            oproj_s=s_o,
            residual=mega_residual,
            out=mega_out,
            scratch=attn_scratch,
            grid_dim=(136, 1, 1),
            block_dim=(256, 1, 1),
        )

        if self.world_size > 1:
            # Sum the per-rank residual-free o_proj partials across ranks AND
            # add the real residual (self.x) exactly once -> attn_proj_out.
            # gate_mode=0: the megakernel runs every decode iter (decode-only
            # build, _use_prefill is False at mbt=1).
            self._allreduce_residual(
                oproj_partial, self.attn_proj_out, self.x, gate_mode=0)

    def _build_mla_attention_layer(self, layer_idx: int, state_dict: dict):
        """Build MLA attention for one decoder layer (FP8 weights).

        The fused attention-block megakernel is the DEFAULT decode path: for the
        bs=1 TP8/EP2/B200 decode geometry (`_use_attn_megakernel`) this returns
        after registering the single megakernel task. The per-task chain below
        is the PREFILL (dual-dispatch, mbt>8) + unsupported-geometry COMPAT
        fallback — NOT a decode alternative (decode always takes the megakernel).
        """
        prefix = f"model.layers.{layer_idx}."
        attn = f"{prefix}self_attn."

        if self._use_attn_megakernel:
            self._build_mla_attention_megakernel(layer_idx, state_dict)
            return

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
        # The fused rmsnorm+quantize task pre-populated the
        # share_quantize_tag in _fp8_quantize_emitted at the input_layernorm
        # call site, so _fp8_linear's internal quantize is skipped. Pull
        # the per-layer-unique FP8/scale buffers the fused task wrote
        # (case-3 — see `_emit_fused_rmsnorm_qkv_a_quantize`).
        qkv_a_quantize_tag = self._fused_rmsnorm_quantize_qkv_a_tag(layer_idx)
        qkv_a_fp8_ovr, qkv_a_scale_ovr = self._fused_qkv_a_bufs[layer_idx]
        self._fp8_linear(
            self.rmsnorm_out, w_qkv_a, s_qkv_a, self.qkv_a_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_qkv_a.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
            share_quantize_tag=qkv_a_quantize_tag,
            input_fp8_override=qkv_a_fp8_ovr,
            input_scale_override=qkv_a_scale_ovr)

        # Diagnostic: capture RAW qkv_a_out immediately after the fused
        # GEMM, before any consumer touches it (demo --dump flags only;
        # zero overhead otherwise).
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

        # Step 2: q_a_layernorm (BF16 norm weight) — RMSnorm of the q_a
        # slice [0:q_lora_rank) inside the fused qkv_a_out buffer, fused
        # with the downstream q_b input-quantize.
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
        # The fused q_a layernorm above already emitted q_a's FP8/scale
        # into per-layer buffers; thread its tag + buffers to ALL q_b
        # GEMMs (decode + prefill) so they share one quantize and read
        # the per-layer buffers (NOT the shared cache; case-3).
        qb_share_tag = q_a_fused_tag
        qb_input_fp8_ovr = q_a_fused_fp8_ovr
        qb_input_scale_ovr = q_a_fused_scale_ovr
        # Decode Q path: use the load-time absorbed q_b_proj dense GEMM.
        # The per-head swapAB BMM chain (linear_fp8_bmm_layer dense=False)
        # overflows ptxas register allocation in the full decode megakernel
        # (C7600). The absorbed GEMM is the original math and writes the same
        # fused [nope_512|pe_64] q_nope_pe tensor consumed by ROPE/MLA below.
        w_q_b_proj, s_q_b_proj = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight",
            f"layer_{layer_idx}_q_b_proj_decode")
        self._fp8_linear(
            self.q_a_out,
            w_q_b_proj,
            s_q_b_proj,
            self.q_nope_pe,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_proj.dim(0)),
                      1, 1),
            block_dim=(128, 1, 1),
            gate_mode=2 if self._use_prefill else 0,
            share_quantize_tag=qb_share_tag,
            input_fp8_override=qb_input_fp8_ovr,
            input_scale_override=qb_input_scale_ovr,
            **qb_slice_kwargs)
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
                **qb_slice_kwargs)
        # Step 4: kv_a (c_latent + k_pe) is produced by the fused qkv_a GEMM
        # above; no separate kv_a_proj_with_mqa GEMMs are emitted.

        rope_q_grid = (
            self.mpk.max_num_batched_requests,
            self.num_local_q_heads,
            1,  # TILE_Q==mbt → 1 CTA per (req, head); kernel loop covers all tokens
        )
        # Dual-dispatch: the fused (absorbed) ROPE_Q only matters on decode
        # iters, the split (unabsorbed) ROPE_Q only on prefill iters. Phase
        # gates make the wrong-phase ROPE return immediately instead of
        # rotating stale data.
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
                1,  # TILE_Q==mbt collapses grid.z to 1
            ),
            q_tile_size=self.max_num_batched_tokens,
            k_pe_row_stride=self._qkv_a_row_stride,
            k_pe_offset=self._qkv_a_k_pe_offset,
        )

        # Step 5: kv_a_layernorm on c_latent slice [1536:2048) of qkv_a_out.
        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_a_layernorm.weight"],
            name=f"layer_{layer_idx}_kv_a_layernorm")
        # c_latent_out is an mpk.narrow view (offset baked into the view base
        # pointer), so no explicit in/out offsets are needed — this matches
        # the upstream rmsnorm_layer signature, which dropped the
        # *_offset_elems params in favour of view-carried offsets.
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln,
            output=self.c_latent_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
            process_dim=self.kv_lora_rank)

        # Step 6: MLA attention (KV gather + unified prefill/decode + reduce).
        # When `_use_prefill` is True, register one MLA main task that chooses
        # prefill vs decode from runtime Q_LEN. The decode reduce stays
        # separate and keeps its Q_LEN gate.
        q_len_mla = self.max_num_batched_tokens
        decode_q_len_mla = self._decode_q_len()
        kv_len_max = self.mpk.max_seq_length
        kv_tiles_max = (kv_len_max + self.mpk.page_size - 1) // self.mpk.page_size
        # TP decode's direct-write path is only validated for one 128-token
        # KV tile; multi-tile configs keep the partial+reduce path.
        single_split_mla = kv_tiles_max <= 1
        mla_num_splits_override = 1 if single_split_mla else None
        mla_decode_out = self.attn_out if single_split_mla else self.mla_partial_o
        # bs=1 contiguous KV: one persistent per-layer buffer the append task
        # writes at row = sequence position and the decode kernels read via
        # their contiguous branch (page_indices == nullptr). No paged cache,
        # no gather.
        # Rows padded to a 128 multiple so the prefill views/TMA windows stay
        # in-buffer (chunked TMA box BN=128); decode only reads [0, kv_len).
        _kv_rows_raw = (self.mpk.max_num_batched_requests
                        * self.mpk.max_seq_length)
        _kv_rows_pad = ((_kv_rows_raw + 127) // 128) * 128
        mla_decode_kv = self.mpk.new_tensor(
            dims=(_kv_rows_pad, self.qk_head_dim),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_kv_contig",
            io_category="cuda_tensor",
        )
        # c_latent and k_pe live at offsets 1536 / 2048 of the 2176-wide
        # qkv_a_out row. Row strides communicate the parent width; the slice
        # offsets are carried by the mpk.narrow views themselves (the
        # *_offset_elems params were dropped from the upstream gather API).
        kv_gather_slice_kwargs = dict(
            c_latent_row_stride=self._qkv_a_row_stride,
            k_pe_row_stride=self._qkv_a_row_stride)
        # bs=1 contiguous KV: append the new token's [c_latent|k_pe] at
        # row = step (single sequence => logical position == physical row).
        # Decode reads the same per-layer buffer via its contiguous branch —
        # no page table, no gather copy.
        dsv3_tasks.mla_kv_append_layer(
            self.mpk,
            c_latent_new=self.c_latent_out,
            k_pe_new=self.k_pe_out,
            kv_buf=mla_decode_kv,
            mla_params=(self.qk_head_dim, self.v_head_dim),
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
            **kv_gather_slice_kwargs,
        )
        if self._use_prefill:
            # Prefill-EXCLUSIVE attention work, all fed straight from the
            # per-layer contiguous KV buffer the append just wrote — the
            # SAME compressed-latent cache decode reads. No gather, no
            # second copy of the cache:
            #   ckv_sep = kv_buf[:, 0:512)  (normalized latent, strided view)
            #   kpe_sep = kv_buf[:, 512:576) (rotated k_pe, strided view)
            ckv_sep = self.mpk.narrow(
                mla_decode_kv, dim=1, start=0, length=self.kv_lora_rank)
            kpe_sep = self.mpk.narrow(
                mla_decode_kv, dim=1, start=self.kv_lora_rank,
                length=QK_ROPE_HEAD_DIM)
            # PHANTOM BRIDGE for the chunked-prefill kpe_sep dependency.
            # `chunked_prefill` is a join-consumer (producers: split ROPE_Q,
            # q_b GEMMs, kv_b_k/v GEMMs, append); the append is also a
            # fork-producer (its other consumers: the kv_b quantize and the
            # decode MLA main). A task has exactly ONE trigger_event slot,
            # so being fork-producer AND join-producer at once is rejected
            # by annotated_graph.cc as case-3. The identity copy
            # `kpe_sep → kpe_sep_v2` (chunked_prefill reads kpe_sep_v2)
            # breaks the append→join-consumer direct edge: the identity has
            # a single producer (no join) and a single consumer (no fork),
            # so both tasks become case-3-safe.
            #
            # grid_dim: identity_layer's dim_maps partition the LAST tensor
            # dim across grid.x, which must DIVIDE the inner dim — kpe_sep's
            # inner dim is 64 (rope), so (8,1,1) gives 8 cols per CTA.
            self.mpk.identity_layer(
                input=kpe_sep,
                output=self.kpe_sep_v2,
                grid_dim=(8, 1, 1),
                block_dim=(128, 1, 1),
                # Decode iters (Q_LEN<=8) skip the copy body: kpe_sep_v2
                # keeps stale data, harmless because chunked_prefill
                # doesn't read it on decode (its own Q_LEN gate).
                gate_decode_q_len=True,
            )
            # kv_b_k/v dense GEMMs: up-project the WHOLE latent cache
            # [0, step+q_len) into transient per-head k_nope/v.
            w_kv_b_k, s_kv_b_k = self._attach_raw_fp8_weight(
                state_dict, f"{attn}kv_b_k.weight",
                f"layer_{layer_idx}_kv_b_k")
            w_kv_b_v, s_kv_b_v = self._attach_raw_fp8_weight(
                state_dict, f"{attn}kv_b_v.weight",
                f"layer_{layer_idx}_kv_b_v")
            # Share the FP8 quantize of the latent view between kv_b_k and
            # kv_b_v (same input, same group_size): one quantize, two GEMM
            # consumers. input_stride = the cache row width (576) — the
            # quantize column-slice contract does NOT derive it from the view.
            kv_b_shared_tag = f"layer_{layer_idx}_kv_b_shared"
            self._fp8_dense_kv_b_proj(
                ckv_sep, w_kv_b_k, s_kv_b_k, self.prefill_k_nope,
                tag=f"layer_{layer_idx}_kv_b_k",
                shared_quantize_tag=kv_b_shared_tag,
                input_stride=self.qk_head_dim)
            self._fp8_dense_kv_b_proj(
                ckv_sep, w_kv_b_v, s_kv_b_v, self.prefill_v,
                tag=f"layer_{layer_idx}_kv_b_v",
                shared_quantize_tag=kv_b_shared_tag,
                input_stride=self.qk_head_dim)
            self.mpk.mla_prefill_tp8_chunked_layer(
                q_nope=self.q_nope,
                q_pe=self.q_pe,
                k_nope=self.prefill_k_nope,
                # k_rope comes from `kpe_sep_v2`, the phantom-bridged copy
                # of the kv_buf rope view produced by the identity_layer
                # above (case-3, see comment there).
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
            # NOOP graph-shaping bridge for the DECODE edge (case-3 again):
            # the append already forks to {kv_b quantize, kpe identity}; the
            # decode MLA main is a JOIN-consumer (q_nope_pe + kv_buf), and
            # annotated_graph rejects a task that is fork-producer AND
            # join-producer at once. Routing the decode read through a no-op
            # identity (empty kernel body — the output is a full-range view
            # of the same buffer, no data motion) makes every append
            # consumer fork-only; the noop alone carries the join edge.
            # Registration ORDER matters: the noop "writes" the kv_buf bytes
            # graph-wise, so it must come AFTER the quantize/identity reads
            # (their producer stays the append) and right BEFORE the decode.
            kv_buf_decode = self.mpk.narrow(
                mla_decode_kv, dim=1, start=0, length=self.qk_head_dim)
            self.mpk.identity_layer(
                input=mla_decode_kv,
                output=kv_buf_decode,
                grid_dim=(8, 1, 1),
                block_dim=(128, 1, 1),
                noop=True,
            )
            mla_decode_kv = kv_buf_decode
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

        # Step 7: O projection. Both phases use the unabsorbed
        # o_proj_original [hidden, H*128]: prefill projects the chunked-
        # prefill attn_unabsorbed directly (gate_mode=1); decode goes
        # through the post-attn BMM path (`_bmm_decode_o_path`). Runtime
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
        # Decode o_proj via the post-attn BMM path (decode-gated on
        # dual-dispatch builds via active_mode_o / gate_mode above).
        self._bmm_decode_o_path(state_dict, attn, layer_idx, residual=self.x)

    def _build_dense_mlp_fused(self, layer_idx: int, state_dict: dict):
        """Register the FUSED dense-MLP mega-task in place of the unfused dense
        chain (MPK_DSV3_DENSE_MLP_MEGAKERNEL=1) for dense layers 0-2.

        One task = post-attn RMSNorm + W13(gate+up) GEMV + silu(gate)*up
        (384-chunk interleave) + W2(down) GEMV -> bf16. It reads the PRE-rmsnorm
        residual stream self.x + post_attention_layernorm.weight and rms-norms
        internally (Phase A), so the separate rmsnorm_layer that wrote
        self.rmsnorm_out is dead this layer (its consumers are bypassed; the
        unused write is harmless, same as the FFN-full MoE path).

        The kernel writes the PRE-AllReduce W2 result; the RowParallel down_proj
        AllReduce + residual stay OUTSIDE this task (mirrors how the unfused
        dense chain's down_proj feeds _allreduce_residual at TP>1).
        """
        prefix = f"model.layers.{layer_idx}."

        # --- config guards: the kernel hard-codes the TP8 EP2 per-rank shapes.
        assert self.mpk.num_workers == 136, (
            "MPK_DSV3_DENSE_MLP_MEGAKERNEL needs num_workers==136 (B200); got "
            f"{self.mpk.num_workers}. The 136-CTA<->136-worker bijection is the "
            "grid_barrier participant count; a non-136 count deadlocks.")
        assert self.max_num_batched_tokens == 1, (
            "MPK_DSV3_DENSE_MLP_MEGAKERNEL is decode-only (mbt==1, M=1 GEMV); got "
            f"{self.max_num_batched_tokens}.")
        assert self.hidden_size == 7168, (
            f"dense-MLP kernel hard-codes HIDDEN=7168; got {self.hidden_size}.")
        # 2 * per-rank intermediate = W13_N = 4608; per-rank intermediate =
        # W2_K = 2304 (TP8 shard of INTERMEDIATE_SIZE=18432).
        assert self.intermediate_size == 2304, (
            "dense-MLP kernel hard-codes W2_K=2304 (= per-rank intermediate); got "
            f"intermediate_size={self.intermediate_size} (TP{self.world_size}).")

        # --- W13 (gate_up_proj) + W2 (down_proj): RAW checkpoint fp8 payload +
        # RAW float32 block scale [N/128, K/128]. The kernel reads the f32 scale
        # as-is and UE8M0-rounds only the ACTIVATION scales internally. NO
        # pow2-requantize (that is the MoE-grouped-GEMM convention, NOT the dense
        # path which is scale_ue8m0=False). _attach_fp8_weight returns exactly
        # (fp8 weight, raw f32 scale_inv) — the same tensors the unfused chain's
        # _fp8_linear consumes, so the weight cache is unchanged.
        w13, w13_scale = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.gate_up_proj.weight",
            f"layer_{layer_idx}_gate_up_proj")
        w2, w2_scale = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.down_proj.weight",
            f"layer_{layer_idx}_down_proj")
        if w13_scale is None or w2_scale is None:
            raise RuntimeError(
                "MPK_DSV3_DENSE_MLP_MEGAKERNEL requires FP8 dense weights with "
                "raw float32 scale_inv (gate_up_proj / down_proj); got a BF16 "
                "fallback weight (no scale). Use the unfused dense chain for "
                "BF16 fixtures.")

        # --- rmsnorm weight (post_attention_layernorm.weight, bf16 [HIDDEN]).
        # NOTE: build_layers attaches this SAME tensor directly via
        # self.mpk.attach_input(name="layer_{i}_post_attn_layernorm") — that path
        # BYPASSES _attach_cache, so reusing that name here would emit a SECOND
        # `model_tensors.at("layer_{i}_post_attn_layernorm")` declaration in the
        # generated test.cu -> nvcc "already declared in the current scope". So
        # we attach under a DISTINCT name (mirrors the FFN-full mega path's
        # `layer_{i}_ffn_full_rmsnorm_w`); the kernel only reads input_ptrs[5],
        # the name is codegen-local.
        rmsnorm_weight = self._safe_attach(
            state_dict[f"{prefix}post_attention_layernorm.weight"],
            f"layer_{layer_idx}_dense_mlp_rmsnorm_w")

        # --- barrier + y13-global scratch (zero the 8B barrier head via
        # tensor_init; the kernel zero-inits y13 implicitly by writing every
        # element in Phase 1 before any Phase-2 read).
        assert DENSE_MLP_MEGAKERNEL_SCRATCH_BYTES % 2 == 0
        barrier_scratch = self.mpk.new_tensor(
            dims=(1, DENSE_MLP_MEGAKERNEL_SCRATCH_BYTES // 2),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_dense_mlp_megakernel_scratch",
            io_category="cuda_tensor")
        self.mpk.tensor_init_layer(
            target=barrier_scratch,
            dummy=self.x,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            dummy_input_map=(-1, -1, -1),
            target_input_map=(-1, -1, -1),
        )

        # --- W2 output (PRE-AllReduce, pre-residual). At TP>1 it must live in
        # symmetric memory so the downstream NVSHMEM AllReduce can read it; the
        # _new_tp_partial helper picks nvshmem_tensor when nvshmem is enabled.
        # At TP=1 a plain cuda_tensor partial feeds the elementwise residual add.
        idx = getattr(self, "_tp_residual_linear_idx", 0)
        self._tp_residual_linear_idx = idx + 1
        partial = self._new_tp_partial(
            self.x, f"layer_{layer_idx}_dense_mlp_fused_partial_{idx}")

        dsv3_tasks.dsv3_dense_mlp_fused_layer(
            self.mpk,
            hidden=self.x,                 # PRE-rmsnorm residual stream
            w13=w13,
            w13_scale_fp32=w13_scale,
            w2=w2,
            w2_scale_fp32=w2_scale,
            rmsnorm_weight=rmsnorm_weight,
            scratch=barrier_scratch,
            out=partial,
            grid_dim=(136, 1, 1),
            block_dim=(256, 1, 1),
        )

        # --- RowParallel combine: AllReduce(partial) + residual(self.x) at TP>1,
        # else a plain residual add. Identical to what the unfused dense chain's
        # down_proj does (residual fused into the AllReduce's final local store,
        # so it is NOT double-counted across ranks).
        self.mlp_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_mlp_fused",
            io_category="cuda_tensor",
        )
        if self.world_size > 1:
            self._allreduce_residual(partial, self.mlp_out, self.x)
        else:
            self.mpk.elementwise_add_layer(
                input_a=self.x, input_b=partial,
                output=self.mlp_out,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

    def _build_dense_mlp(self, layer_idx: int, state_dict: dict):
        """Build dense MLP for layers 0-2 (FP8 weights)."""
        # FUSED dense-MLP mega-task (MPK_DSV3_DENSE_MLP_MEGAKERNEL=1): replace the
        # whole unfused chain (rmsnorm + gate_up GEMM + silu_mul + down GEMM) with
        # one task. Early-return mirrors how _build_mla_attention_layer dispatches
        # into the attn megakernel. Default-OFF -> the default build is
        # byte-identical (dense layers keep their chain).
        if os.environ.get("MPK_DSV3_DENSE_MLP_MEGAKERNEL") == "1":
            self._build_dense_mlp_fused(layer_idx, state_dict)
            return

        prefix = f"model.layers.{layer_idx}."

        w_gate_up, s_gate_up = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.gate_up_proj.weight",
            f"layer_{layer_idx}_gate_up_proj")
        # TP=2 workaround: the fp8_gemm_dense_mediumm kernel faults
        # (cudaErrorLaunchFailure at mb_arrive_tx) at the TP=2 gate_up
        # shape (M=8, N=18432, K=7168); other shapes run cleanly. Split
        # the gate_up GEMM into two N=9216 sub-calls (a known-good size):
        # first the local gate half, then the local up half. silu_mul
        # reads them from disjoint output slots, so the layout is
        # preserved.
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
        # NOTE: per-iter buffers (input quantized, meta, permuted_*,
        # w13/silu/w2 outs) are allocated PER LAYER inside `_build_moe_mlp`,
        # NOT shared globally — sharing them across layers gives the same
        # buffer two distinct producer→consumer chains in consecutive
        # layers, which the task-graph dep tracker rejects (case-3
        # fork+join producer). Globally this setup only does the static
        # m_indices attach + the weight-scale-pack cache.
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
                                  moe_mask,
                                  new_moe_input_fp8, new_moe_input_scale,
                                  new_moe_permuted_in_fp8,
                                  new_moe_permuted_in_scale,
                                  new_moe_meta,
                                  new_moe_w13_out, new_moe_silu_out,
                                  new_moe_silu_fp8,
                                  new_moe_silu_scale_Mfirst,
                                  new_moe_silu_scale, new_moe_w2_out):
        """Per-layer NEW MoE task dispatch. Tasks numbered 1..8 below."""
        _routed_nw = self.mpk.num_workers
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
        # 3) Permute + scale-transpose. E_LOCAL
        # (= num_local_experts = m_total // bm_pad) must be divisible by
        # e_per_cta.
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
            grid_dim_y=1,
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
            num_workers=_routed_nw,
            meta=new_moe_meta,
        )
        # 5) SiLU+MUL. With input_map=(0,-1,-1) the runtime partitions
        # dim 0 across grid.x CTAs and the C++ register pulls
        # num_active_tokens from the per-CTA STensor shape
        # (= m_total / grid.x rows per CTA), so the work is fully
        # parallel across SMs.
        # grid.x MUST equal num_local_experts so rows_per_cta == bm_padding
        # (one CTA per expert), else the runtime's per-CTA row offset
        # (bid.x * (m_total // grid.x)) drifts off the 128-row expert blocks
        # and silu_mul reads/writes the WRONG w13_out rows (the wrapper's
        # ctas_per_expert mapping only holds when bm_padding % rows_per_cta
        # == 0 AND grid.x is a multiple of E_local). min(num_workers, m_total)
        # = 136 at TP-EP gave rows_per_cta=120 → null routed-MoE.
        _silu_grid = m_total // bm_pad

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
        # 6) Quantize SiLU → UE8M0 directly into K-outermost layout: the
        # quantize_fp8 kernel writes the packed scale at offset
        # `packed_idx * aligned_batch + batch_idx`, which IS K-outer
        # row-major — declaring the output as (K_PACKED, m_total) matches
        # the write pattern, and the downstream W2 SFA TMA descriptor
        # (which expects K-outer) reads correct bytes directly (no
        # transpose_scale task needed).
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
        # 7+8) Pack W2 weight scale + attach W2 weight + Group GEMM W2.
        w2_scale_key_for_pack = f"{prefix}experts.w2.weight_scale_inv"
        self._requantize_moe_fp8_for_pow2(
            state_dict, f"{prefix}experts.w2.weight", w2_scale_key_for_pack)
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
            num_workers=_routed_nw,
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
            # The megakernel already overlaps the shared-expert with W13 by
            # default (prelaunch-all), so the shared-down GEMM runs at the full
            # packing grid.
            linear_grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
            residual=None,
        )
        return shared_residual

    def _build_moe_mlp_ffn_full(self, layer_idx, prefix, state_dict):
        """The default fused decode MoE-FFN path (selected by
        `_use_ffn_full_megakernel` for the bs=1 TP8/EP2/B200 decode geometry):
        ONE mega-task in place of the whole per-task MoE chain. See
        _build_moe_mlp. The asserts below restate that predicate (defensive —
        the kernel hard-codes the TP8/EP2 per-rank shapes)."""
        from ..utils import shuffle_tensors as _shuffle_tensors

        # --- config guards: the kernel hard-codes the TP8 EP2 per-rank shapes.
        assert self.mpk.num_workers == 136, (
            "FFN-FULL kernel needs num_workers==136 (B200); got "
            f"{self.mpk.num_workers}. The 136-CTA<->136-worker bijection is the "
            "grid_barrier participant count; a non-136 count deadlocks.")
        assert self.max_num_batched_tokens == 1, (
            "FFN-FULL kernel is decode-only (mbt==1); got "
            f"{self.max_num_batched_tokens}.")
        assert self.num_local_experts == 128, (
            "FFN-FULL kernel hard-codes E_LOCAL=128 (TP8 EP2); got "
            f"num_local_experts={self.num_local_experts}.")
        assert self.hidden_size == 7168, (
            f"FFN-FULL kernel hard-codes HIDDEN=7168; got {self.hidden_size}.")
        assert self.routed_moe_intermediate_size == 512, (
            "FFN-FULL kernel hard-codes W2_K=512 (routed inter); got "
            f"{self.routed_moe_intermediate_size}.")
        assert NUM_EXPERTS == 256, (
            f"FFN-FULL kernel hard-codes ROUTER_N=256; got {NUM_EXPERTS}.")

        w13_scale_key = f"{prefix}experts.w13.weight_scale_inv"
        if w13_scale_key not in state_dict:
            raise RuntimeError(
                "DeepSeek V3 routed MoE requires FP8 expert weights "
                f"({w13_scale_key} missing from the state_dict).")

        # --- W13 / W2: requantize the PAYLOAD for the CEIL-pow2 (UE8M0) scale,
        # then bake the pow2 fp32 scale the kernel multiplies by (same as the
        # COLD FFN gate). The kernel applies a plain fp32 scale, NOT the raw
        # scale_inv, so the value must be 2^ceil(log2(scale_inv)).
        self._requantize_moe_fp8_for_pow2(
            state_dict, f"{prefix}experts.w13.weight", w13_scale_key)
        w_experts_w13 = self._safe_attach(
            state_dict[f"{prefix}experts.w13.weight"],
            f"layer_{layer_idx}_experts_w13")
        _w13s = state_dict[w13_scale_key].float().clamp_min(1e-30)
        w13_scale_fp32 = self._safe_attach(
            torch.pow(2.0, torch.ceil(torch.log2(_w13s))).contiguous(),
            f"layer_{layer_idx}_experts_w13_scale_fp32")

        w2_weight_key = f"{prefix}experts.w2.weight"
        w2_scale_key = f"{prefix}experts.w2.weight_scale_inv"
        self._requantize_moe_fp8_for_pow2(state_dict, w2_weight_key, w2_scale_key)
        w_experts_w2 = self._safe_attach(
            state_dict[w2_weight_key], f"layer_{layer_idx}_experts_w2")
        _w2s = state_dict[w2_scale_key].float().clamp_min(1e-30)
        w2_scale_fp32 = self._safe_attach(
            torch.pow(2.0, torch.ceil(torch.log2(_w2s))).contiguous(),
            f"layer_{layer_idx}_experts_w2_scale_fp32")

        # --- shared expert: gate_up shuffled to a plain [gate;up] concat (the
        # kernel assumes gate=[0:256],up=[256:512] contiguous), pow2 scale.
        shared_prefix = f"{prefix}shared_experts."
        wgu_raw = self._safe_attach(
            _shuffle_tensors(
                [state_dict[f"{shared_prefix}gate_proj.weight"],
                 state_dict[f"{shared_prefix}up_proj.weight"]],
                split=1, dim=0).contiguous(),
            f"layer_{layer_idx}_shared_expert_gate_up_raw")
        wgu_scale = self._safe_attach(
            _shuffle_tensors(
                [state_dict[f"{shared_prefix}gate_proj.weight_scale_inv"],
                 state_dict[f"{shared_prefix}up_proj.weight_scale_inv"]],
                split=1, dim=0).to(torch.float32).contiguous(),
            f"layer_{layer_idx}_shared_expert_gate_up_raw_scale_fp32")
        wdn = self._safe_attach(
            state_dict[f"{shared_prefix}down_proj.weight"],
            f"layer_{layer_idx}_shared_expert_down")
        wdn_scale = self._safe_attach(
            state_dict[f"{shared_prefix}down_proj.weight_scale_inv"]
            .to(torch.float32).contiguous(),
            f"layer_{layer_idx}_shared_expert_down_scale_fp32")

        # --- front-stage inputs that the chain consumed as separate tasks:
        #   rmsnorm weight (post_attention_layernorm.weight, bf16) — the kernel
        #     rms-norms self.x internally (the chain's separate rmsnorm_layer in
        #     build_layers still runs but writes an unused self.rmsnorm_out).
        #   router gate weight (gate.weight, bf16 [256,7168]).
        #   e_score_correction_bias (fp32 [256]) — kernel reads `float const*`.
        rmsnorm_weight = self._safe_attach(
            state_dict[f"{prefix.split('mlp.')[0]}post_attention_layernorm.weight"],
            f"layer_{layer_idx}_ffn_full_rmsnorm_w")
        router_gate_weight = self._safe_attach(
            state_dict[f"{prefix}gate.weight"],
            f"layer_{layer_idx}_ffn_full_router_gate_w")
        bias = self._safe_attach(
            state_dict[f"{prefix}gate.e_score_correction_bias"]
            .float().contiguous(),
            f"layer_{layer_idx}_ffn_full_router_bias")

        # --- output (pre-AR MoE output) — same alloc as the chain's moe_output.
        _moe_io = "nvshmem_tensor" if self._use_nvshmem else "cuda_tensor"
        moe_output = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_output",
            io_category=_moe_io)

        # --- barrier + globals scratch (zero head via tensor_init).
        assert FFN_FULL_MEGAKERNEL_SCRATCH_BYTES % 2 == 0
        barrier_scratch = self.mpk.new_tensor(
            dims=(1, FFN_FULL_MEGAKERNEL_SCRATCH_BYTES // 2),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_ffn_full_megakernel_scratch",
            io_category="cuda_tensor")
        self.mpk.tensor_init_layer(
            target=barrier_scratch,
            dummy=self.x,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            dummy_input_map=(-1, -1, -1),
            target_input_map=(-1, -1, -1),
        )

        # routed_scaling_factor: DSv3 default 2.5 (matches the chain's
        # moe_topk_sigmoid_routing_layer default).
        dsv3_tasks.ffn_full_megakernel_layer(
            self.mpk,
            hidden=self.x,             # PRE-rmsnorm residual stream
            w13=w_experts_w13,
            w13_scale_fp32=w13_scale_fp32,
            w2=w_experts_w2,
            w2_scale_fp32=w2_scale_fp32,
            rmsnorm_weight=rmsnorm_weight,
            router_gate_weight=router_gate_weight,
            bias=bias,
            wgu_raw=wgu_raw,
            wgu_scale=wgu_scale,
            wdn=wdn,
            wdn_scale=wdn_scale,
            out=moe_output,
            barrier_scratch=barrier_scratch,
            local_expert_start=self.local_expert_start,
            num_local_experts=self.num_local_experts,
            routed_scaling_factor=2.5,
            grid_dim=(136, 1, 1),
            block_dim=(512, 1, 1),
        )
        self.mlp_out = moe_output

    def _build_moe_mlp(self, layer_idx: int, state_dict: dict):
        """Build MoE MLP for layers 3-60."""
        prefix = f"model.layers.{layer_idx}.mlp."

        # ============================================================
        # FULLY-fused FFN mega-task — the DEFAULT decode MoE path.
        # For the bs=1 TP8/EP2/B200 decode geometry (`_use_ffn_full_megakernel`)
        # this ONE task REPLACES the entire MoE chain: post-attn rmsnorm
        # (computed internally from self.x) + router-gate-GEMV + topk-sigmoid +
        # permute + W13/W2 group-GEMM + silu + the COLD FFN. It binds the
        # PRE-rmsnorm hidden (self.x), the rmsnorm weight, the router gate
        # weight + bias, and the requantized W13/W2/shared weights, and produces
        # the pre-AR MoE output (moe_output); the 2 AllReduces around it are
        # untouched (added in build_layers after this returns).
        #
        # The per-task MoE chain below is the PREFILL (dual-dispatch, mbt>8) +
        # unsupported-geometry COMPAT fallback — NOT a decode alternative.
        # ============================================================
        if self._use_ffn_full_megakernel:
            self._build_moe_mlp_ffn_full(layer_idx, prefix, state_dict)
            return

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
        if self.max_num_batched_tokens == 1:
            # ferret BF16 CUDA-core GEMV replaces the tcgen05 linear_layer for
            # the router gate at bs=1 decode. Default-OFF: env unset ⇒ the
            # linear_layer path below (byte-identical baseline build).
            self.mpk.dsv3_router_gate_gemv_layer(
                input=self.rmsnorm_out,
                weight=w_gate,
                output=router_logits,
                num_workers=self.mpk.num_workers,
            )
        else:
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
        # The kernel's VPT template (8, ROWS_PER_WARP=1) lives in
        # src/kernel/task_register.cc; block_dim 256 (8 warps) here must
        # stay in parity with it.
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
        if use_fp8_experts:
            self._requantize_moe_fp8_for_pow2(
                state_dict, f"{prefix}experts.w13.weight", w13_scale_key)
        w_experts_w13 = self._safe_attach(
            state_dict[f"{prefix}experts.w13.weight"],
            f"layer_{layer_idx}_experts_w13")
        mbt = self.max_num_batched_tokens

        # Routed-MoE path — fp8_group_gemm (smallm/largem auto-pick):
        # Permute(routing) → group_gemm(W13) → silu → quantize →
        # group_gemm(W2) → unpermute(combine + residual).
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
        # K-outer layout [K_PACKED, round4(mbt)]: the UE8M0 quantize writes
        # word (packed_k * aligned_batch + row) with aligned_batch =
        # round4(mbt) (task_register scale_outer_stride), so the allocation
        # must cover the full K_PACKED x round4(mbt) footprint — the old
        # (mbt, K_PACKED) shape under-allocated whenever mbt % 4 != 0
        # (compute-sanitizer: 4B OOB writes 137B past a 56B allocation).
        new_moe_input_scale = self.mpk.new_tensor(
            dims=(K_PACKED_K, ((mbt + 3) // 4) * 4), dtype=uint32,
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
        # meta is 2D `(2, N)` int32 so the shared `tensor_init` kernel —
        # which zeros `BATCH_SIZE * OUTPUT_SIZE * sizeof(bf16)` bytes —
        # covers the FULL int32 byte range (BATCH_SIZE=2, OUTPUT_SIZE=N
        # → 2*N*2 bytes = N int32s).
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
            moe_mask,
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
        # the reduction and over-count it. moe_unpermute does
        #   output[t] = shared_residual[t]
        #             + sum_k(permuted_w2_out[token_to_perm[t,k]-1]
        #                      * permuted_weights[same row])
        # — the topk-weighted combine AND the shared-residual add in one
        # task. rows_per_cta / hidden_split control the launch fan-out
        # only (see moe_unpermute_sm100_layer).
        dsv3_tasks.moe_unpermute_sm100_layer(
            self.mpk,
            permuted_output=self._new_moe_layer_w2_out,
            meta=self._new_moe_layer_meta,
            residual=shared_residual,
            output=moe_output,
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
            # The input-layernorm RMSNorm is fused with the downstream
            # qkv_a FP8 quantize. The fused task writes the BF16 normalized
            # output AND the FP8 + scale buffers in one pass, so the qkv_a
            # `_fp8_linear` call (compat/prefill path) can skip its internal
            # quantize via share_quantize_tag. On the fused decode path the
            # attn megakernel rms-norms the input itself; this task's outputs
            # then go unread (harmless dead write — same pattern as the
            # post-attn rmsnorm below). Emitted unconditionally for build
            # uniformity (a prior env-gated skip was box-measured NULL on perf
            # and broke token-identity, so it was reverted).
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
            # Post-attention RMSNorm, emitted unconditionally. On the fused
            # decode path the FFN-full / dense-MLP megakernels rms-norm self.x
            # INTERNALLY (Phase A, eps 1e-6), so this self.rmsnorm_out write is
            # a dead write for those layers; it stays LIVE for dense layers 0-2
            # (unfused chain), the prefill/compat MoE chain, and (via the shared
            # buffer) the final-norm -> lm_head tail.
            #
            # NOTE (2026-06-25 investigation, scaffolding removed): a direct skip
            # of the dead write was tried and reverted — it measured NULL on perf
            # + token-differ, BUT the token-differ was later traced to BASELINE
            # NONDETERMINISM (ffn_full cross-CTA atomicAdd FP non-assoc, two
            # identical runs diverge from token ~10), so the "broke token-
            # identity" verdict is REFUTED. A per-layer dedicated buffer
            # (to decouple the dead write from the shared producer set) was also
            # tried and proven a graph NO-OP — the post-attn rmsnorm is already
            # terminal on the shared buffer (DAG-dump: skip-off == skip-on). The
            # skip's real graph perturbation is dropping self.x's fork
            # cardinality 2->1. Any re-test must use the poison-fill / distri-
            # butional gate (token-identity is dead on this nondeterministic path).
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
