"""DeepSeek V3 model builder for Mirage MPK with MTP support.

Architecture: 61 decoder layers with MLA attention and MoE MLP.
- Layers 0-2: Dense MLP (DeepseekV2MLP)
- Layers 3-60: MoE MLP (256 experts, top-8, + shared experts)
- MLA: 128 Q heads, 1 KV head after weight absorption, head_dim=576 (512+64)
- Optional MTP: 1 predictor layer for multi-token prediction

Weight absorption: at load time, kv_b_proj is absorbed into q_b_proj so that
runtime only needs compressed KV cache [c_latent(512), k_pe(64)] = 576 dims.
"""

import math
import os
import torch
from typing import Optional

from ..utils import grid_for_rmsnorm_linear_layer
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...persistent_kernel import PersistentKernel
from ...model_registry import register_model_builder
from ....core import bfloat16, float8_e4m3, float32, uint32, int32, int64


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


def _moe_fp8_m_split(output_size: int, preferred: int) -> int:
    """Pick a valid output-dimension split for the FP8 MoE group GEMM."""
    max_y = min(preferred, max(1, output_size // _MOE_FP8_MMA_M))
    for y in range(max_y, 0, -1):
        if output_size % y == 0 and (output_size // y) % _MOE_FP8_MMA_M == 0:
            return y
    return 1


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
# Env override: MPK_DSV3_RMSNORM_ROWS_PER_TASK={1,2,4,8}.
_RMSNORM_ROWS_PER_TASK = int(os.environ.get("MPK_DSV3_RMSNORM_ROWS_PER_TASK", "1"))


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


def _moe_expert_grid_x(max_num_batched_tokens: int,
                       num_experts: int = NUM_EXPERTS,
                       preferred_groups: int | None = None) -> int:
    # The MoE kernels iterate over the compact activated-expert list with a
    # stride equal to grid_dim.x. A batch can activate at most top_k experts per
    # token. The preferred group count raises parallelism for larger batches,
    # but it must remain bounded by the active routing slots for MBT=1 graphs.
    active_slots = max(1, max_num_batched_tokens * NUM_EXPERTS_PER_TOK)
    group_cap = num_experts if preferred_groups is None else min(
        num_experts, preferred_groups)
    return min(group_cap, active_slots)


def _moe_hidden_split(hidden_size: int, preferred: int = 56) -> int:
    """Pick a valid hidden-dimension split for lightweight MoE epilogues.

    Used by `moe_mul_sum_add_layer` whose grid is (mbt, y, 1). For batch=1
    the per-token work splits across `y` workers, so we want `y` as close to
    num_workers as the alignment allows. The 128-multiple constraint comes
    from the underlying kernel's epilogue tile.
    """
    max_y = min(preferred, max(1, hidden_size // 128))
    for y in range(max_y, 0, -1):
        if hidden_size % y == 0 and (hidden_size // y) % 128 == 0:
            return y
    return 1


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
    splitk linear, and `moe_mul_sum_add_layer` — partition the hidden dim
    into 128-wide column tiles (linear.grid.x = output_size // 128, or
    moe_mul_sum_add.grid.y = _moe_hidden_split(...) which lands on the same
    128-wide split). The legacy default of 1024-wide allreduce tiles
    therefore generated ~8x fewer tasks than the producing layer (7 vs 56
    for DSv3 hidden=7168), starving the persistent runtime of dispatchable
    work right after the matmul finished.

    Defaulting the allreduce tile to 128 keeps the partition aligned with
    the producer so each upstream task has a one-to-one downstream
    consumer. `MPK_ALLREDUCE_TILE_SIZE` overrides for ablation (e.g. coarse
    1024-wide tiles for small TP-only configs that prefer fewer collectives).

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
    override = os.environ.get("MPK_ALLREDUCE_TILE_SIZE")
    if override:
        tile_size = int(override)
        if tile_size <= 0 or output_size % tile_size != 0:
            raise ValueError(
                "MPK_ALLREDUCE_TILE_SIZE must be a positive divisor of "
                f"{output_size}, got {tile_size}")
        if tile_size % 128 != 0:
            raise ValueError(
                "MPK_ALLREDUCE_TILE_SIZE must be 128-aligned, "
                f"got {tile_size}")
        return (output_size // tile_size, 1, 1)
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
        # Weight attach cache: avoid re-declaring same C++ variable in MTP draft loop
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

        # Fuse residual into linear kernels (with_residual). Always on.
        self._fuse_residual = True
        # NEW MoE path: route MoE W13/W2 through the PR-674 fp8_group_gemm
        # via two peripheral tasks (moe_permute → fp8_group_gemm →
        # silu_mul → quantize → fp8_group_gemm → moe_unpermute). Default
        # OFF; A/B test under MPK_DSV3_NEW_MOE=1 then flip default after
        # correctness + perf are validated. See scratch/pr674_moe_kernel_wiring_plan.md.
        self._new_moe = os.environ.get("MPK_DSV3_NEW_MOE", "0") == "1"
        # MPK_DSV3_ACTIVE_SKIP=0 disables the per-expert active-mask
        # short-circuit in fp8_group_gemm (commit ecf1f8e5) — useful for
        # A/B correctness checks. Default ON when NEW MoE is on.
        self._new_moe_active_skip = (
            os.environ.get("MPK_DSV3_ACTIVE_SKIP", "1") == "1"
        )
        # Per-layer NEW-MoE skip list — env-gated escape hatch while the
        # L8+ correctness bug is unresolved. Comma-separated layer indices
        # (e.g. "8,9,10,11,12,13") fall back to OLD MoE on those layers,
        # preserving the 2.7x perf win on the remaining layers.
        _skip_str = os.environ.get("MPK_DSV3_NEW_MOE_SKIP_LAYERS", "")
        self._new_moe_skip_layers = set()
        if _skip_str:
            for tok in _skip_str.split(","):
                tok = tok.strip()
                if tok:
                    self._new_moe_skip_layers.add(int(tok))
        # M_TOTAL for the new GEMM = NUM_LOCAL_EXPERTS * BM_PADDING. Both
        # are compile-time-ish (NUM_LOCAL_EXPERTS depends on ep_size at
        # __init__ time; BM_PADDING matches the new GEMM's largest BM tile).
        # Must be a multiple of fp8_group_gemm's internal BM tile (=128).
        # Temporarily raisable for "BM_PADDING saturation" debugging via env.
        self._moe_bm_padding = int(os.environ.get("MPK_DSV3_BM_PADDING", "128"))
        assert self._moe_bm_padding % 128 == 0, "BM_PADDING must be 128-aligned"
        # MPK_DSV3_PERMUTE_EPC (default 1): experts-per-CTA for the NEW-MoE
        # moe_permute_sm100 task. >1 (e.g. 4) shrinks the permute launch from
        # (E_LOCAL,1,1) to (E_LOCAL/EPC,1,1) so the decode permute fits in one
        # SM wave instead of contending with the shared-expert GEMM across
        # ~3 waves (analyzer-found ~40 μs/layer decode "valley"). EPC==1 is
        # byte-identical to the legacy 1-CTA-per-expert path. E_LOCAL
        # (= num_local_experts) must be divisible by EPC — asserted at the
        # moe_permute call site once num_local_experts is known.
        self._moe_permute_epc = int(
            os.environ.get("MPK_DSV3_PERMUTE_EPC", "1"))
        assert self._moe_permute_epc >= 1, "MPK_DSV3_PERMUTE_EPC must be >= 1"
        # MPK_DSV3_BMM=1: switch the decode Q path from the load-time absorbed
        # q_b_proj (single fused (H*576, q_lora) FP8 GEMM) to a per-head BMM
        # chain: rmsnorm_linear(q_b_nope, 128) + rmsnorm_linear(q_b_pe, 64)
        # + quantize_fp8(q_nope) + linear_fp8_bmm(q_nope, kv_b_k_bmm) →
        # q_nope_abs (mbt, H, 512) + assemble_q_decode → q_nope_pe.
        # Win: smaller weight loads per task (per-head (512, 128) vs absorbed
        # (576, q_lora=1536) monolith) → less TMA traffic, room to overlap.
        # 2026-05-17: flipped DEFAULT ON after correctness validated — the
        # BMM=1 run produced BIT-IDENTICAL generated tokens vs the absorbed
        # path. PR-674's FP8 dense GEMM kernels are tuned for the
        # unabsorbed shapes (smaller M, K) and only the unabsorbed path
        # benefits from those optimizations; this also fetches less HBM per
        # decode iter (per-head 128/64 instead of fused 576). Latency is
        # currently ~5% slower vs absorbed due to extra quantize+BMM
        # serialization; the optimization roadmap (quantize+BMM fusion,
        # grid-size tuning, eventual overlap) closes the gap and goes past
        # absorbed once those land. Set MPK_DSV3_BMM=0 to fall back to
        # the absorbed path for regression isolation.
        self._dsv3_bmm = os.environ.get("MPK_DSV3_BMM", "1") == "1"
        # MPK_DSV3_BMM_DENSE=1: route the decode BMM2 (post-attn kv_b_v
        # un-absorption, _bmm_decode_o_path) through the DENSE block-scaled
        # GEMM body (float32 128-K-aligned scales) instead of the swapAB
        # body (UE8M0, 512-K-packed). Same math; the dense float32 scale
        # layout is split-K-friendly (when the kernel team lands dense
        # split-K, BMM2's per-head K=512 can be split), whereas swapAB's
        # 512-K UE8M0 cannot. Default OFF — no immediate perf gain (dense
        # split-K not landed), correctness-equivalent forward-compat path.
        self._dsv3_bmm_dense = os.environ.get("MPK_DSV3_BMM_DENSE", "0") == "1"
        # D1 (2026-05-17): fuse the q_b_nope FP8 GEMM with its downstream
        # per_token_group_quantize_fp8 task. The new
        # fp8_gemm_dense_*_fp8out kernel computes a per-row UE8M0 scale in
        # registers (each consumer thread already holds the full BN=128
        # K-group) and writes FP8 + packed scale directly, eliminating the
        # bf16 HBM round-trip + standalone quantize dispatch wave on the
        # BMM Q-up critical path.
        # Default ON after 5-run validation: text bit-identical to unfused,
        # median per-token 6.692 ms vs 6.735 ms unfused (n=5, max-min 3 μs)
        # = -43 μs/iter saved (~2.5 μs/MoE-layer × 17). Set
        # MPK_DSV3_FUSED_QB_QUANTIZE=0 to revert to the legacy
        # bf16-GEMM-then-standalone-quantize chain for regression isolation.
        self._fused_qb_quantize = (
            os.environ.get("MPK_DSV3_FUSED_QB_QUANTIZE", "1") == "1")
        # B37 (2026-05-15): replace the (input_layernorm RMSNorm + qkv_a
        # quantize) two-task chain with one fused kernel that writes BF16
        # rmsnorm_out and FP8 + scale in one pass. Saves ~30 μs/layer
        # (−5.7% per decode iter on TP=4 EP=2 mbt=128) by eliminating one
        # dispatch wave plus a bf16 HBM round-trip.
        # Default ON (2026-05-15) after the case-3 fix at commit 27dd8771
        # made the task graph annotate correctly. Set
        # MPK_DSV3_FUSED_RMSNORM_QUANTIZE=0 to fall back to the legacy
        # split rmsnorm + standalone quantize chain.
        self._fused_rmsnorm_quantize = (
            os.environ.get("MPK_DSV3_FUSED_RMSNORM_QUANTIZE", "1") == "1")
        # C17 (2026-05-17): extend B37's rmsnorm+quantize fusion to the
        # post-attn RMSNorm + NEW MoE input quantize. Gated separately so
        # the QKV-a path's setting doesn't have to flip together. Default
        # OFF until we verify cosine + e2e (per C13 lesson: per-call ≠
        # per-token in megakernel).
        self._fused_post_attn_rmsnorm_quantize = (
            os.environ.get("MPK_DSV3_FUSED_POST_ATTN_RMSNORM_QUANTIZE",
                           "0") == "1")
        # C18 (2026-05-17): fuse NEW MoE moe_silu_mul + quantize_fp8 into
        # one task. Eliminates BF16 silu_out HBM round-trip + one task
        # launch. Default OFF; flip via MPK_DSV3_FUSED_SILU_QUANTIZE=1.
        self._fused_silu_quantize = (
            os.environ.get("MPK_DSV3_FUSED_SILU_QUANTIZE", "0") == "1")
        # C1 (2026-05-16): fan the MLA KV-gather unified task across multiple
        # CTAs by striding seq_pos. The legacy 1-CTA gather was 121 μs/layer
        # (15% of layer wallclock) with 127 workers idle. With N splits each
        # CTA strides seq_pos by N, so wallclock drops to ~121/N μs/layer.
        # MPK_DSV3_KV_GATHER_SPLITS=N (default 8); set to 1 to disable.
        self._kv_gather_splits = int(
            os.environ.get("MPK_DSV3_KV_GATHER_SPLITS", "8"))
        assert self._kv_gather_splits >= 1 and self._kv_gather_splits <= 128
        # TP decode's direct-write path is only validated for one 128-token
        # KV tile. For two or more tiles, keep the partial+reduce path.
        self._mla_single_split_max_kv_tiles = int(
            os.environ.get("MPK_MLA_SINGLE_SPLIT_MAX_KV_TILES", "1"))
        self._mla_num_splits_override = os.environ.get(
            "MPK_MLA_NUM_SPLITS_OVERRIDE")

        # MTP config
        self.mtp_config = getattr(mpk, 'spec_decode_config', None)

    def _decode_q_len(self) -> int:
        spec_length = 0
        if self.mtp_config is not None:
            spec_length = int(getattr(self.mtp_config, "spec_length", 0))
        return max(1, min(self.max_num_batched_tokens, spec_length + 1, 8))

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
        """Pick grid_x for the FP8 swapAB non-splitk kernel.

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
            # 2026-05-13 DEBUG: optionally attach the FP8 input + scale buffers
            # as torch tensors so we can inspect them post-megakernel from Python.
            if os.environ.get("MPK_DSV3_FP8_BUF_ATTACH", "0") == "1":
                import torch as _torch
                if not hasattr(self.mpk, "_fp8_input_torch"):
                    self.mpk._fp8_input_torch = {}
                    self.mpk._fp8_scale_torch = {}
                fp8_t = _torch.zeros((mbt, reduction_size), dtype=_torch.float8_e4m3fn, device="cuda")
                scale_t = _torch.zeros((mbt, num_groups), dtype=_torch.float32, device="cuda")
                self.mpk._fp8_input_torch[reduction_size] = fp8_t
                self.mpk._fp8_scale_torch[reduction_size] = scale_t
                fp8_buf = self.mpk.attach_input(torch_tensor=fp8_t, name=f"fp8_input_v2_{reduction_size}_shared")
                scale_buf = self.mpk.attach_input(torch_tensor=scale_t, name=f"fp8_scale_v2_{reduction_size}_shared")
            else:
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

    def _fp8_dense_num_workers(self):
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

        Override via env `MPK_FP8_DENSE_NUM_WORKERS` for experiments
        (sweep showed 64 crashes nvshmem barrier, 80/96 both work but
        80 is faster). `_fp8_dense_kv_b_proj` keeps full `num_workers`
        independently — the runtime_m_mode=1 + large M path has tighter
        constraints and crashes at <128.

        Each task strides through output tiles internally, so lowering num_workers
        below the actual tile count just means each task does more iterations.
        For output 1536/128 = 12 tiles, num_workers >= 12 covers all tiles in
        one wave; <12 means each worker handles multiple tiles.
        """
        override = os.environ.get("MPK_FP8_DENSE_NUM_WORKERS")
        if override:
            return int(override)
        if self._use_prefill:
            return min(80, self.num_workers)
        return self.num_workers

    def _fp8_linear_v2(self, input_bf16, weight_fp8_raw, weight_scale_raw,
                       output, residual=None, gate_mode: int = 0,
                       input_row_stride: int = None,
                       input_col_offset: int = 0,
                       share_quantize_tag: str = None,
                       input_fp8_override=None,
                       input_scale_override=None):
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
        """
        if weight_scale_raw is None:
            raise ValueError("FP8 linear v2 requires FP8 weight scale.")
        if input_bf16.num_dims != 2:
            raise ValueError("FP8 linear v2 expects 2D input.")
        # Output may be 2D (M, N) or 3D (M, H, D_per_head). Storage is
        # row-major contiguous either way; the kernel writes M*N bf16. The
        # 3D form is for the MPK_DSV3_BMM=1 path that wants H exposed
        # downstream without an extra reshape task.
        if output.num_dims not in (2, 3):
            raise ValueError("FP8 linear v2 expects 2D or 3D output.")
        if weight_fp8_raw.num_dims != 2 or weight_scale_raw.num_dims != 2:
            raise ValueError("FP8 linear v2 expects 2D weight + scale.")

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

        gemm_layer = (
            self.mpk.fp8_gemm_dense_smallm_layer
            if self.mpk.max_seq_length <= 512
            else self.mpk.fp8_gemm_dense_mediumm_layer
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

        # B36 (2026-05-15): env-gated decode-only SplitK kernel for the
        # decode O_proj. Stock mediumm runs 56 tiles in 1 underutilized
        # 80-worker wave for the M=128, N=7168, K=16384 (TP=4) shape;
        # splitk=4 gives 224 tiles in 3 better-utilized waves with K/4
        # work per tile.
        #
        # Gating: gate_mode=2 (decode-only) AND residual is not None.
        # The residual-not-None check restricts to the O_proj path —
        # otherwise q_b decode (also gate_mode=2 but residual=None,
        # K=1536 N=18432) hits the kernel with a shape the SplitK
        # kernel was never tested against and crashes with "illegal
        # memory access". The kernel is tuned for the M=128 N=7168
        # K=16384 O_proj shape (and similar TP=1/TP=2 variants).
        use_decode_splitk = (
            gate_mode == 2
            and residual is not None
            and os.environ.get("MPK_DSV3_DECODE_OPROJ_SPLITK") == "1"
        )
        decode_split_k = int(
            os.environ.get("MPK_DSV3_DECODE_OPROJ_SPLITK_FACTOR", "4"))
        # Hard-stop if the shape's K can't be split evenly: a static
        # split_k that doesn't divide K/128 corrupts scale indexing.
        if use_decode_splitk:
            K_ = weight_fp8_raw.dim(1)
            if K_ % (128 * decode_split_k) != 0:
                use_decode_splitk = False

        def _emit_decode_splitk(out_tensor):
            self.mpk.fp8_gemm_dense_decode_splitk_layer(
                input_fp8=input_fp8,
                weight_fp8=weight_fp8_raw,
                input_scale=input_scale,
                weight_scale=weight_scale_raw,
                output=out_tensor,
                num_workers=self._fp8_dense_num_workers(),
                split_k=decode_split_k,
            )

        if residual is None:
            if use_decode_splitk:
                _emit_decode_splitk(output)
            else:
                gemm_layer(
                    input_fp8=input_fp8,
                    weight_fp8=weight_fp8_raw,
                    input_scale=input_scale,
                    weight_scale=weight_scale_raw,
                    output=output,
                    num_workers=self._fp8_dense_num_workers(),
                    runtime_m_mode=gemm_runtime_m_mode,
                )
            return

        if self.world_size > 1:
            idx = getattr(self, "_tp_residual_linear_idx", 0)
            self._tp_residual_linear_idx = idx + 1
            partial = self._new_tp_partial(output, f"tp_v2_residual_partial_{idx}")
            if use_decode_splitk:
                _emit_decode_splitk(partial)
            else:
                gemm_layer(
                    input_fp8=input_fp8,
                    weight_fp8=weight_fp8_raw,
                    input_scale=input_scale,
                    weight_scale=weight_scale_raw,
                    output=partial,
                    num_workers=self._fp8_dense_num_workers(),
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
        if use_decode_splitk:
            _emit_decode_splitk(partial)
        else:
            gemm_layer(
                input_fp8=input_fp8,
                weight_fp8=weight_fp8_raw,
                input_scale=input_scale,
                weight_scale=weight_scale_raw,
                output=partial,
                num_workers=self._fp8_dense_num_workers(),
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

    def _fp8_linear_prequantized(self, input_fp8, input_scale, weight,
                                 weight_scale, output, grid_dim, block_dim,
                                 residual=None, gate_mode: int = 0):
        if weight_scale is None:
            raise ValueError("Prequantized FP8 linear requires FP8 weight scale.")
        if input_fp8.num_dims != 2 or output.num_dims != 2:
            raise ValueError("FP8 linear expects 2D input and output tensors.")
        if weight.num_dims != 2:
            raise ValueError("FP8 linear expects a 2D weight tensor.")
        if weight_scale.num_dims != 2:
            raise ValueError("FP8 linear expects a 2D packed UE8M0 scale tensor.")

        grid_dim = self._fp8_linear_grid_dim(weight, grid_dim)
        use_decode_kernel = self._can_use_decode_fp8_linear(
            input_fp8, weight, output, grid_dim)
        linear_layer = (
            self.mpk.linear_fp8_swapAB_layer
            if use_decode_kernel
            else self.mpk.linear_fp8_layer
        )
        linear_with_residual_layer = (
            self.mpk.linear_fp8_swapAB_with_residual_layer
            if use_decode_kernel
            else self.mpk.linear_fp8_with_residual_layer
        )

        if residual is not None:
            if self.world_size > 1:
                idx = getattr(self, "_tp_residual_linear_idx", 0)
                self._tp_residual_linear_idx = idx + 1
                partial = self._new_tp_partial(output, f"tp_fp8_residual_partial_{idx}")
                linear_layer(
                    input_fp8=input_fp8,
                    input_scale=input_scale,
                    weight_fp8=weight,
                    weight_scale=weight_scale,
                    output=partial,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    gate_mode=gate_mode,
                )
                self._allreduce_residual(partial, output, residual,
                                         gate_mode=gate_mode)
            else:
                linear_with_residual_layer(
                    input_fp8=input_fp8,
                    input_scale=input_scale,
                    weight_fp8=weight,
                    weight_scale=weight_scale,
                    residual=residual,
                    output=output,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    gate_mode=gate_mode,
                )
        else:
            linear_layer(
                input_fp8=input_fp8,
                input_scale=input_scale,
                weight_fp8=weight,
                weight_scale=weight_scale,
                output=output,
                grid_dim=grid_dim,
                block_dim=block_dim,
                gate_mode=gate_mode,
            )

    def _fp8_linear(self, input_bf16, weight, weight_scale, output,
                     grid_dim, block_dim, residual=None, gate_mode: int = 0,
                     input_row_stride: int = None,
                     input_col_offset: int = 0,
                     share_quantize_tag: str = None,
                     input_fp8_override=None,
                     input_scale_override=None):
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
        # emit_bf16 is normally False because nothing downstream reads the
        # bf16 in fused mode (qkv_a GEMM reads fp8/scale directly). BUT when
        # builder-side qkv_a split-K is active, the split-K partials need a
        # real bf16 normalized embedding to re-quantize per K-slice, and the
        # split-K identity bridge reads THIS per-layer buffer (not the stale
        # cross-layer-shared self.rmsnorm_out). Emit the bf16 in that case so
        # the data is materialized AND the input-layernorm task stays on the
        # qkv_a->attention->o_proj chain (otherwise the embedding->o_proj
        # residual edges can't be residual-stripped -> case-3 fork+join).
        _emit_bf16 = self._qkva_splitk_active()
        self.mpk.fused_rmsnorm_quantize_fp8_layer(
            input=input_x,
            weight=w_norm,
            output_bf16=rmsnorm_out_bf16,
            output_fp8=input_fp8,
            output_scale=input_scale,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
            scale_ue8m0=False,
            emit_bf16=_emit_bf16,
        )
        return tag

    def _qkva_splitk_active(self) -> bool:
        """True when builder-side qkv_a split-K is env-enabled (MPK_DSV3_QKVA_SPLITK
        >= 2). The actual per-layer guard also checks K divisibility at the call
        site; this helper only governs whether the fused input-layernorm task
        must materialize its bf16 output for the split-K identity bridge."""
        return int(os.environ.get("MPK_DSV3_QKVA_SPLITK", "0")) >= 2

    def _fp8_experts_available(self, state_dict: dict, layer_idx: int) -> bool:
        """C17: check if layer `layer_idx`'s experts.w13.weight_scale_inv
        exists in state_dict — i.e., whether this layer uses FP8 experts.
        Mirrors the inline check at line ~3192 in `_build_moe_mlp`."""
        prefix = f"model.layers.{layer_idx}.mlp."
        return f"{prefix}experts.w13.weight_scale_inv" in state_dict

    def _emit_fused_post_attn_rmsnorm_moe_quantize(self, input_x: 'DTensor',
                                                    w_norm: 'DTensor',
                                                    layer_idx: int,
                                                    reduction_size: int):
        """C17 (2026-05-17): post-attention RMSNorm + NEW-MoE input quantize
        fusion. Mirrors B37's pattern but for the post-attn rmsnorm. Writes:
          * `rmsnorm_out_bf16` (per-layer-unique BF16 buffer) — consumed by
            router linear + shared_expert + downstream BF16 readers.
          * `moe_input_fp8` (per-layer-unique FP8) — consumed by NEW MoE
            permute.
          * `moe_input_scale` (per-layer-unique UE8M0 K-outer) — same path.

        Returns the tuple `(rmsnorm_out_bf16, moe_input_fp8, moe_input_scale)`.

        Buffer-ownership / case-3 rationale (B37 docstring extended):
          The fused task takes its outputs as `store_in_dmem` inputs in the
          task graph; shared cross-layer buffers would make the producer of
          this task's input (post-AR output) BOTH a fork-producer AND a
          join-producer (case-3 violation). With per-layer-unique outputs
          here, the fused task has a unique writer per buffer.

        Scale layout: UE8M0 column-major `[packed_k, aligned_batch]`. The
        kernel writes `output_s[packed_idx * aligned_batch + batch_idx]`,
        which is the SAME byte layout the standalone `quantize_fp8_layer`
        wrote at builder.py:2790 (kernel-side this is "shape lie"). So
        downstream readers (moe_permute) see byte-identical scale data —
        no consumer-side changes needed.
        """
        mbt = self.max_num_batched_tokens
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size
        # K_PACKED packs 4 UE8M0 bytes per uint32, then rounds up.
        k_packed = (num_groups + 3) // 4
        if not hasattr(self, "_fused_post_attn_bufs"):
            self._fused_post_attn_bufs = {}
        if layer_idx not in self._fused_post_attn_bufs:
            rmsnorm_out_bf16 = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=bfloat16,
                name=f"fused_post_attn_rmsnorm_out_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
            moe_input_fp8 = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=float8_e4m3,
                name=f"fused_post_attn_moe_input_fp8_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
            moe_input_scale = self.mpk.new_tensor(
                dims=(mbt, k_packed), dtype=uint32,
                name=f"fused_post_attn_moe_input_scale_layer_{layer_idx}",
                io_category="cuda_tensor",
            )
            self._fused_post_attn_bufs[layer_idx] = (
                rmsnorm_out_bf16, moe_input_fp8, moe_input_scale)
        rmsnorm_out_bf16, moe_input_fp8, moe_input_scale = (
            self._fused_post_attn_bufs[layer_idx])

        # UE8M0 K-outer packed scale (matches existing moe_input_scale layout
        # written by the standalone quantize_fp8 task; downstream
        # moe_permute reads the buffer byte-for-byte identically).
        self.mpk.fused_rmsnorm_quantize_fp8_layer(
            input=input_x,
            weight=w_norm,
            output_bf16=rmsnorm_out_bf16,
            output_fp8=moe_input_fp8,
            output_scale=moe_input_scale,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
            scale_ue8m0=True,
            emit_bf16=True,
        )
        return rmsnorm_out_bf16, moe_input_fp8, moe_input_scale

    def _bmm_decode_q_path(self, state_dict, attn, layer_idx, qb_slice_kwargs):
        """MPK_DSV3_BMM=1: replaces the absorbed q_b_proj decode GEMM with a
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
            # bf16 outputs of q_b_nope/q_b_pe FP8 dense GEMMs (3D so the
            # BMM input partition map can see H as an explicit dim).
            self._bmm_decode_buffers["q_nope_3d"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 128), dtype=bfloat16,
                name="q_nope_decode_3d", io_category="cuda_tensor")
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
        q_nope_3d = self._bmm_decode_buffers["q_nope_3d"]
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
        # consumer). When _fused_qb_quantize is OFF (legacy path), the
        # order between q_b_nope and q_b_pe doesn't matter; when it's ON,
        # we want the input-quantize task already emitted before the
        # fused q_b_nope GEMM since the fused GEMM reads the same q_a
        # FP8 buffer and skips the redundant quantize via the share tag.
        self._fp8_linear(
            self.q_a_out, w_q_b_pe, s_q_b_pe, q_pe_3d,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_pe.dim(0)), 1, 1),
            block_dim=(128, 1, 1),
            gate_mode=2 if self._use_prefill else 0,
            **qb_slice_kwargs)
        if self._fused_qb_quantize:
            # 2 fused) q_b_nope FP8 dense GEMM with epilogue UE8M0 quantize
            # → q_nope_fp8 + q_nope_scale directly. Reads q_a's FP8 / scale
            # from the shared cache (the q_b_pe call above already emitted
            # the quantize). Replaces the (bf16 q_b_nope GEMM →
            # quantize_fp8) two-task chain with one task; saves ~9 μs/layer
            # on the BMM Q-up critical path.
            reduction_size = w_q_b_nope.dim(1)
            input_fp8_buf, input_scale_buf = (
                self._fp8_mbt_buffers_for_reduction_f32scale(reduction_size))
            gemm_fp8out_layer = (
                self.mpk.fp8_gemm_dense_smallm_fp8out_layer
                if self.mpk.max_seq_length <= 512
                else self.mpk.fp8_gemm_dense_mediumm_fp8out_layer
            )
            gemm_runtime_m_mode = 3 if self._use_prefill else 0
            gemm_fp8out_layer(
                input_fp8=input_fp8_buf,
                weight_fp8=w_q_b_nope,
                input_scale=input_scale_buf,
                weight_scale=s_q_b_nope,
                output_fp8=q_nope_fp8,
                output_scale=q_nope_scale,
                num_workers=self._fp8_dense_num_workers(),
                runtime_m_mode=gemm_runtime_m_mode,
            )
        else:
            # 2) q_b_nope FP8 dense GEMM → q_nope_3d (mbt, H, 128) bf16
            self._fp8_linear(
                self.q_a_out, w_q_b_nope, s_q_b_nope, q_nope_3d,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_nope.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
                gate_mode=2 if self._use_prefill else 0,
                **qb_slice_kwargs)
            # 3) Quantize q_nope BF16 → FP8 + UE8M0 packed scale.
            # Each row's K=128, so packed_K=1. row_count = mbt * H rows.
            active_mode_bmm = 3 if self._use_prefill else 0  # decode-only when prefill enabled
            self.mpk.quantize_fp8_layer(
                input=q_nope_3d,
                output_fp8=q_nope_fp8,
                output_scale=q_nope_scale,
                grid_dim=(1, mbt * H_local, 1),
                block_dim=(128, 1, 1),
                scale_ue8m0=True,
                active_mode=active_mode_bmm,
            )
        # 4) BMM(q_nope_fp8, kv_b_k_bmm) → q_nope_abs (mbt, H, 512).
        w_kvk_bmm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_b_k_bmm.weight"],
            name=f"layer_{layer_idx}_kv_b_k_bmm")
        s_kvk_bmm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_b_k_bmm.weight_scale_ue8m0"],
            name=f"layer_{layer_idx}_kv_b_k_bmm_scale")
        self.mpk.linear_fp8_bmm_sm100_layer(
            input_fp8=q_nope_fp8,
            input_scale=q_nope_scale,
            weight_fp8=w_kvk_bmm,
            weight_scale=s_kvk_bmm,
            output=q_nope_abs,
            grid_dim=(512 // 128, H_local, 1),  # (4, H, 1)
            block_dim=(256, 1, 1),
        )
        # 5) Assemble (PE-only): BMM already wrote nope into q_nope_pe[:, :, :512]
        # via the q_nope_abs slice-view fuse, so the assemble step only needs
        # to write q_pe into the tail [512:576]. Half the per-CTA traffic.
        self.mpk.assemble_q_decode_sm100_layer(
            q_nope_abs=q_nope_abs,
            q_pe=q_pe_3d,
            q_nope_pe=self.q_nope_pe,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
            pe_only=True,
        )

    def _bmm_decode_o_path(self, state_dict, attn, layer_idx, residual):
        """C9 (2026-05-16): post-attn BMM path for MPK_DSV3_BMM=1.

        Replaces the load-time-absorbed decode o_proj (fused with W_UV)
        with runtime BMM + smaller linear:
          quantize(attn_out)           → attn_out_fp8 (mbt, H, 512) FP8
          BMM(attn_out_fp8, kv_b_v_bmm) → attn_out_reduced (mbt, H, 128) bf16
          fp8_linear_with_residual(attn_out_reduced, o_proj_original) → attn_proj_out

        The o_proj_original.weight is the SAME weight used by the prefill
        path (hidden × H*128, FP8). After BMM, decode + prefill both use
        the smaller unabsorbed o_proj.

        Gate: this entire path runs only on decode iters (Q_LEN<=8) via
        the FP8 linear's gate_mode=2 + BMM's MMA_N=16 decode constraint.

        Returns: None (writes attn_proj_out directly).
        """
        H_local = self.num_local_q_heads
        mbt = self.max_num_batched_tokens
        V_HEAD_DIM = 128  # post-attn V un-absorption dim per head
        KV_LORA = 512     # current attn_out per-head dim

        # nk = number of 128-K groups in the per-head reduction (= 512/128 = 4),
        # used for the dense (float32) scale buffer.
        nk_o = (KV_LORA + 127) // 128
        if not hasattr(self, "_bmm_decode_o_buffers"):
            self._bmm_decode_o_buffers = {}
            # FP8 of attn_out (shared by both scale encodings). K=512.
            self._bmm_decode_o_buffers["attn_out_fp8"] = self.mpk.new_tensor(
                dims=(mbt, H_local, KV_LORA), dtype=float8_e4m3,
                name="attn_out_bmm_fp8", io_category="cuda_tensor")
            # Scale of attn_out. swapAB path: UE8M0 packed (K=512 → packed_K=1).
            self._bmm_decode_o_buffers["attn_out_scale"] = self.mpk.new_tensor(
                dims=(mbt, H_local, 1), dtype=uint32,
                name="attn_out_bmm_scale", io_category="cuda_tensor")
            # Dense path: float32 1x128-group activation scale [mbt, H, nk].
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
        attn_out_scale = self._bmm_decode_o_buffers["attn_out_scale"]
        attn_out_scale_f32 = self._bmm_decode_o_buffers["attn_out_scale_f32"]
        attn_out_reduced = self._bmm_decode_o_buffers["attn_out_reduced"]

        active_mode_o = 3 if self._use_prefill else 0  # decode-only on dual-dispatch

        if self._dsv3_bmm_dense:
            # Step 1 (dense): quantize attn_out BF16 → FP8 + float32 1x128-group
            # scale [mbt, H, nk]. Input self.attn_out is (mbt, H*KV_LORA) 2D;
            # output FP8 is 3D (mbt, H, KV_LORA), same byte layout; the float32
            # scale is row-major [batch, num_groups] = [mbt*H, nk] which views
            # as [mbt, H, nk].
            self.mpk.quantize_fp8_layer(
                input=self.attn_out,
                output_fp8=attn_out_fp8,
                output_scale=attn_out_scale_f32,
                grid_dim=(1, mbt * H_local, 1),
                block_dim=(128, 1, 1),
                scale_ue8m0=False,
                active_mode=active_mode_o,
            )
            # Step 2 (dense): per-head BMM via the DENSE block-scaled GEMM body.
            # kv_b_v_bmm_dense prepared in demo.py: weight (H, 128, 512) FP8 +
            # float32 block scale (H, 1, nk).
            w_kvv_bmm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{attn}kv_b_v_bmm_dense.weight"],
                name=f"layer_{layer_idx}_kv_b_v_bmm_dense")
            s_kvv_bmm = self.mpk.attach_input(
                torch_tensor=state_dict[
                    f"{attn}kv_b_v_bmm_dense.weight_scale_inv"],
                name=f"layer_{layer_idx}_kv_b_v_bmm_dense_scale")
            self.mpk.linear_fp8_bmm_dense_sm100_layer(
                input_fp8=attn_out_fp8,
                input_scale=attn_out_scale_f32,
                weight_fp8=w_kvv_bmm,
                weight_scale=s_kvv_bmm,
                output=attn_out_reduced,
                grid_dim=(1, H_local, 1),
                block_dim=(256, 1, 1),
            )
        else:
            # Step 1: quantize attn_out BF16 → FP8 + UE8M0 packed scale.
            # Input self.attn_out is (mbt, H*KV_LORA) 2D. Output is 3D
            # (mbt, H, KV_LORA). Same byte layout; the kernel writes row-by-row
            # using global batch_idx.
            self.mpk.quantize_fp8_layer(
                input=self.attn_out,
                output_fp8=attn_out_fp8,
                output_scale=attn_out_scale,
                grid_dim=(1, mbt * H_local, 1),
                block_dim=(128, 1, 1),
                scale_ue8m0=True,
                active_mode=active_mode_o,
            )

            # Step 2: BMM(attn_out_fp8, kv_b_v_bmm) → attn_out_reduced
            # (mbt, H, 128). kv_b_v_bmm prepared in demo.py: per-head
            # (H, 128, 512) FP8.
            w_kvv_bmm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{attn}kv_b_v_bmm.weight"],
                name=f"layer_{layer_idx}_kv_b_v_bmm")
            s_kvv_bmm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{attn}kv_b_v_bmm.weight_scale_ue8m0"],
                name=f"layer_{layer_idx}_kv_b_v_bmm_scale")
            self.mpk.linear_fp8_bmm_sm100_layer(
                input_fp8=attn_out_fp8,
                input_scale=attn_out_scale,
                weight_fp8=w_kvv_bmm,
                weight_scale=s_kvv_bmm,
                output=attn_out_reduced,
                # D_out=128, BMM constraint D_out/grid.x must be multiple of
                # MMA_M=128 → grid.x must be 1.
                grid_dim=(1, H_local, 1),
                block_dim=(256, 1, 1),
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
        gemm_layer = (
            self.mpk.fp8_gemm_dense_smallm_layer
            if self.mpk.max_seq_length <= 512
            else self.mpk.fp8_gemm_dense_mediumm_layer
        )
        # Chunked-prefill kv_b_k/v: keep at full self.num_workers (NOT the
        # env-overridable). The runtime_m_mode=1 + larger M_total path has
        # tighter constraints and crashes at low num_workers (tested 48/64
        # both fail). Decode-only env override (`MPK_FP8_DENSE_NUM_WORKERS`)
        # leaves this call at default 128.
        gemm_layer(
            input_fp8=input_fp8,
            weight_fp8=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            output=output,
            num_workers=self.num_workers,
            runtime_m_mode=1,
        )

    # FP8 splitk replacements verified end-to-end on TP=1 layers 0-8.
    _FP8_SPLITK_ENABLED = True
    # BF16 splitk_linear_layer hangs the MPK runtime in the DSv3 gate
    # configuration (batch=1, N=256, K=7168). The standalone regression
    # matrix at tests/runtime_python/blackwell/sm100_splitk_linear_bf16/
    # times out for every BATCH_SIZE < 16. A "fix" that replaced the
    # `kClampedBN = min(BATCH_SIZE, MMA_N)` clamp with MMA_N would let TMA
    # over-read past the in-bounds gmem rows (out-of-bounds access into
    # whatever follows the buffer), which is why qwen3/DSv3 still hang
    # even though the standalone matrix turns green. The original clamp
    # is the correct semantic; the kernel needs a deeper rework to
    # support batch < MMA_N=16 cleanly. Until then, keep the gate on the
    # original linear_layer kernel.
    _BF16_GATE_SPLITK_ENABLED = True

    @staticmethod
    def _pick_splitk_factor(n_tiles, K, num_workers, k_align):
        """Pick the split_k that maximizes total tasks (= n_tiles * split_k)
        subject to a single-wave-on-the-persistent-runtime cap.

        The splitk linear kernel produces `grid = (n_tiles, split_k, 1)` tasks
        where `n_tiles = output_size // MMA_M=128`. We want as many of those
        tasks as possible without spilling past `num_workers` (otherwise the
        layer pays for a partial second wave, which on B200's 128-worker
        config is up to 2x slower than the single-wave optimum).

        Constraints:
          - K must be divisible by `k_align` (kernel prereq; FP8 needs 512 for
            UE8M0 packing, BF16 only needs the 64-byte TILE_SIZE).
          - K // split_k must remain a multiple of `k_align` (per-task K
            still satisfies the kernel prereq).
          - n_tiles * split_k <= num_workers (single wave).

        Returns the best split_k, or None if K is not k_align-aligned.
        """
        if K % k_align != 0:
            return None
        if n_tiles > num_workers:
            # Even split_k=1 already overflows a single wave; splitk only adds
            # tasks (more grid.y), never removes them. Signal "splitk can't
            # help" so the caller can fall back to a non-splitk path that may
            # use a coarser N-tile (e.g. grid_for_rmsnorm_linear_layer).
            return None
        # split_k must divide quotient = K // k_align so that K/s is still a
        # multiple of k_align. Iterate divisors in ascending s; tasks = n*s
        # is monotonically increasing, so we can stop once it exceeds the cap.
        quotient = K // k_align
        best_s = None
        for s in range(1, quotient + 1):
            if quotient % s != 0:
                continue
            if n_tiles * s > num_workers:
                break
            best_s = s
        return best_s

    def _pick_fp8_splitk_factor(self, weight):
        """FP8 splitk picker for a `weight` tensor of shape [output, K].

        Returns None when splitk is disabled or K isn't 512-aligned, so the
        caller can fall back to the non-splitk path.

        IMPORTANT: `splitk_linear_fp8_swapAB_sm100` is decode-only — the
        kernel asserts `BATCH_SIZE <= 16` at registration. When the builder
        is configured for prefill (`_use_prefill = mbt > 8`, i.e.
        max_num_batched_tokens >= 9), the per-task batch shape can exceed
        16 and the kernel produces wrong (mostly-zero) output. Force the
        non-splitk path in that case so prefill correctness is preserved.
        Decoding-only deployments (mbt <= 8) keep the splitk fast-path.
        """
        if not self._FP8_SPLITK_ENABLED:
            return None
        if self._use_prefill:
            return None
        return self._pick_splitk_factor(
            n_tiles=weight.dim(0) // 128,
            K=weight.dim(1),
            num_workers=self.num_workers,
            k_align=512,
        )

    def _fp8_linear_builder_splitk(self, input_bf16, weight_key, state_dict,
                                   output, split_k, name_prefix):
        """Builder-side split-K for a decode M=1 starved K-bound GEMM.

        Splits K into `split_k` CONTIGUOUS slices, runs split_k separate
        `_fp8_linear` calls (each fills ceil(N/128) CTAs → split_k× parallel
        working CTAs, filling idle SMs at decode M=1), then reduces the bf16
        partials. Uses the WORKING dense kernel — orthogonal to the broken
        decode_splitk/swapAB split-K kernels (which crash / can't compile at
        BATCH=128). Verified facts this enables correctness:
        - weight_scale (`.weight_scale_inv`) layout = [N/128, K/128] row-major
          (the dense GEMM `sb`, indexed sb[(on/128)*nk + ki], nk=K/128), so a
          K-slice = a contiguous slice on dim 1 → row-stride matches the
          slice's own nk=Ks/128. (A narrow VIEW would NOT work: its row-stride
          stays K/128.)
        - `_attach_fp8_weight` is a passthrough (raw fp8 weight + f32 scale),
          so slicing the state_dict tensors directly is valid.
        - `_fp8_linear`'s input_col_offset/input_row_stride slice the bf16
          input in-place (no input copy).
        Precision: bf16 partial-sum vs the kernel's FP32 accumulate — verify
        TP=2 token-match before enabling. Gated by MPK_DSV3_BUILDER_SPLITK.
        """
        import torch
        wt = state_dict[f"{weight_key}.weight"]
        st = state_dict[f"{weight_key}.weight_scale_inv"]
        N, K_full = int(wt.shape[0]), int(wt.shape[1])
        assert K_full % (128 * split_k) == 0, (K_full, split_k)
        Ks = K_full // split_k
        Ksg = Ks // 128
        if not hasattr(self, "_builder_splitk_chunks"):
            self._builder_splitk_chunks = []
        partials = []
        for i in range(split_k):
            wc_t = wt[:, i * Ks:(i + 1) * Ks].contiguous()
            sc_t = st[:, i * Ksg:(i + 1) * Ksg].to(torch.float32).contiguous()
            # keep python refs alive — attach binds by pointer
            self._builder_splitk_chunks += [wc_t, sc_t]
            wc = self._safe_attach(wc_t, f"{name_prefix}_skw{i}")
            sc = self._safe_attach(sc_t, f"{name_prefix}_sks{i}")
            pi = self.mpk.new_tensor(
                dims=(output.dim(0), N), dtype=bfloat16,
                name=f"{name_prefix}_skp{i}", io_category="cuda_tensor")
            self._fp8_linear(
                input_bf16, wc, sc, pi,
                grid_dim=(grid_for_rmsnorm_linear_layer(N), 1, 1),
                block_dim=(128, 1, 1),
                input_col_offset=i * Ks, input_row_stride=K_full)
            partials.append(pi)
        acc = partials[0]
        for i in range(1, split_k):
            out = output if i == split_k - 1 else self.mpk.new_tensor(
                dims=(output.dim(0), N), dtype=bfloat16,
                name=f"{name_prefix}_skacc{i}", io_category="cuda_tensor")
            self.mpk.elementwise_add_layer(
                input_a=acc, input_b=partials[i], output=out,
                grid_dim=(grid_for_rmsnorm_linear_layer(N), 1, 1),
                block_dim=(128, 1, 1))
            acc = out

    def _pick_bf16_splitk_factor(self, weight):
        """BF16 splitk picker for a `weight` tensor of shape [output, K].

        2026-05-14 (P5): the `_use_prefill → return 1` bypass was paranoia
        about the BF16-splitk small-batch hang, but the hang reproduces
        only for *compile-time* BATCH_SIZE < 16 — our prefill-enabled
        config has compile-time BATCH_SIZE = mbt = 128, well above the
        clamp threshold. Drop the bypass so the DSv3 router gate
        dispatches grid=(n_tiles, split_k, 1) instead of (n_tiles, 1, 1)
        (was 2 CTAs → 90 μs on the user-flagged perfetto ID 64861).
        Earlier attempts thought this broke correctness but baseline
        decode-from-step-100 already outputs all-zero tokens regardless
        of this knob (the 19-layer DSv3 model is structurally degenerate
        at that step), so the regression was misdiagnosed.
        """
        return self._pick_splitk_factor(
            n_tiles=weight.dim(0) // 128,
            K=weight.dim(1),
            num_workers=self.num_workers,
            k_align=64,
        ) or 1

    def _fp8_linear_splitk(self, input_bf16, weight, weight_scale, output,
                           split_k, residual=None,
                           input_fp8=None, input_scale=None):
        """Same residual semantics as `_fp8_linear`, via the FP8 splitk swapAB
        kernel. The kernel uses tma_reduce_add_async; both with-residual and
        no-residual paths share that fact.

        residual=None: prepends tensor_init that zeroes `output`, then splitk
            writes the matmul into it (accumulate=False).
        residual + TP=1: caller must alias `output is residual` so the kernel
            can reduce-add the matmul on top of the residual in place
            (accumulate=True, no tensor_init).
        residual + TP>1: produces a partial in a fresh buffer, then runs the
            allreduce + add-residual sequence into `output`.

        `input_fp8`/`input_scale`: skip quantization and use these directly;
        otherwise quantize `input_bf16` first.
        """
        if weight_scale is None:
            raise ValueError("FP8 splitk requires an FP8 weight scale.")
        if input_bf16 is not None and input_bf16.num_dims != 2:
            raise ValueError("FP8 splitk expects 2D bf16 input.")
        if weight.num_dims != 2 or weight_scale.num_dims != 2:
            raise ValueError("FP8 splitk expects 2D weight + scale.")
        if output.num_dims != 2:
            raise ValueError("FP8 splitk expects 2D output.")

        output_size = weight.dim(0)
        if output_size % 128 != 0:
            raise ValueError(
                f"FP8 splitk requires output divisible by 128, got {output_size}")
        grid = (output_size // 128, split_k, 1)
        block = (256, 1, 1)

        if input_fp8 is None:
            reduction_size = weight.dim(1)
            input_fp8, input_scale = self._fp8_buffers_for_reduction(reduction_size)
            self.mpk.quantize_fp8_layer(
                input=input_bf16,
                output_fp8=input_fp8,
                output_scale=input_scale,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

        if residual is None:
            self.mpk.linear_splitk_swapAB_fp8_layer(
                input_fp8=input_fp8, input_scale=input_scale,
                weight_fp8=weight, weight_scale=weight_scale,
                output=output, grid_dim=grid, block_dim=block,
                accumulate=False,
            )
            return

        if self.world_size > 1:
            idx = getattr(self, "_tp_residual_linear_idx", 0)
            self._tp_residual_linear_idx = idx + 1
            partial = self._new_tp_partial(
                output, f"tp_fp8_splitk_residual_partial_{idx}")
            self.mpk.linear_splitk_swapAB_fp8_layer(
                input_fp8=input_fp8, input_scale=input_scale,
                weight_fp8=weight, weight_scale=weight_scale,
                output=partial, grid_dim=grid, block_dim=block,
                accumulate=False,
            )
            self._allreduce_residual(partial, output, residual)
            return

        if output is not residual:
            raise ValueError(
                "FP8 splitk with residual on TP=1 requires `output` to be the "
                "same DTensor as `residual` (alias the residual buffer to the "
                "output before calling).")
        self.mpk.linear_splitk_swapAB_fp8_layer(
            input_fp8=input_fp8, input_scale=input_scale,
            weight_fp8=weight, weight_scale=weight_scale,
            output=output, grid_dim=grid, block_dim=block,
            accumulate=True,
        )

    def _silu_mul_fp8_linear(self, silu_input, silu_bf16_output, weight,
                             weight_scale, output, silu_grid_dim,
                             linear_grid_dim, block_dim, residual=None,
                             use_splitk=False, splitk_split_k=None):
        self.mpk.silu_mul_layer(
            input=silu_input,
            output=silu_bf16_output,
            grid_dim=silu_grid_dim,
            block_dim=(128, 1, 1),
        )
        if use_splitk:
            split_k = (splitk_split_k
                       if splitk_split_k is not None
                       else self._pick_fp8_splitk_factor(weight))
            if split_k is not None:
                self._fp8_linear_splitk(
                    silu_bf16_output, weight, weight_scale, output,
                    split_k=split_k, residual=residual,
                )
                return
            # Fall through to non-splitk if K isn't splitk-able.
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
        disable_direct_paged_decode_kv = (
            os.environ.get("MPK_DISABLE_DIRECT_PAGED_DECODE_KV", "0") == "1"
        )
        self._direct_paged_decode_kv = (
            self.mpk.page_size == 128
            and (direct_paged_tp_decode or direct_paged_tp1_decode)
            and not disable_direct_paged_decode_kv
        )
        if self._use_prefill:
            print(f"  [MLA path] MBT={mbt} → MLA prefill + runtime-gated decode")
        else:
            print(f"  [MLA path] Q_LEN={mbt} → MLA decode / MTP decode")

        # RMSNorm output
        # 2026-05-13 DEBUG: optionally attach rmsnorm_out as a torch tensor
        # (bypassing MPK's buffer pool) AND pre-fill with a sentinel value
        # so we can tell post-megakernel whether rows are
        #   (a) still at sentinel → rmsnorm task never wrote them (skip bug)
        #   (b) zero → something else wrote zero over them (overwrite bug)
        #   (c) normal rmsnorm output → rmsnorm wrote but quantize saw something else
        if os.environ.get("MPK_DSV3_RMSNORM_OUT_ATTACH", "0") == "1":
            import torch as _torch
            sentinel = float(os.environ.get("MPK_DSV3_RMSNORM_SENTINEL", "0.0"))
            self.mpk._rmsnorm_out_torch = _torch.full(
                (mbt, self.hidden_size), sentinel,
                dtype=_torch.bfloat16, device="cuda")
            self.rmsnorm_out = self.mpk.attach_input(
                torch_tensor=self.mpk._rmsnorm_out_torch, name="rmsnorm_out")
        else:
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
        if self._dsv3_bmm:
            # MPK_DSV3_BMM=1: allocate as 3D torch tensor so we can attach
            # slice views (q_nope_pe[:, :, :512] for BMM output, q_nope_pe[:, :, 512:]
            # for q_pe) and have BMM write per-head [nope|pe] interleaved
            # directly without an assemble task.
            import torch as _torch
            self._q_nope_pe_torch = _torch.zeros(
                mbt, self.num_local_q_heads, self.qk_head_dim,
                dtype=_torch.bfloat16, device="cuda")
            self.q_nope_pe = self.mpk.attach_input(
                self._q_nope_pe_torch, name="q_nope_pe")
        else:
            self.q_nope_pe = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads * self.qk_head_dim),
                dtype=bfloat16,
                name="q_nope_pe",
                io_category="cuda_tensor",
            )
        # Decode consumes absorbed [CKV, KPE] Q. Prefill consumes vLLM's
        # original per-head split Q: [nope(128), rope(64)].
        # 2026-05-12 (user #2 FuseTensor): when MPK_DSV3_QB_FUSED=1,
        # prefill uses a single q_b_proj_unabsorbed Linear emitting one
        # (mbt, H*192) fused tensor. q_nope and q_pe both alias into this
        # fused tensor (same DTensor handle); the chunked_prefill kernel
        # reads Qn at the base and Qp at base + 128 via qfused_mode=1.
        # Default OFF for incremental validation; flip to "1" once validated
        # against reference across prefill workloads.
        self._qb_fused = os.environ.get("MPK_DSV3_QB_FUSED", "0") == "1"
        if self._qb_fused:
            # Fused Q for prefill: per-head 192 = 128 (nope) + 64 (pe).
            self.q_b_prefill_fused = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads *
                      (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)),
                dtype=bfloat16, name="q_b_prefill_fused",
                io_category="cuda_tensor",
            )
            # Aliases used by the prefill kernel call sites — both point at
            # the same fused tensor; chunked_prefill's qfused_mode=1 path
            # uses input_ptrs[0] for both and computes the 128-offset
            # internally.
            self.q_nope = self.q_b_prefill_fused
            self.q_pe = self.q_b_prefill_fused
        else:
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
        Uses _attach_cache to avoid re-declaring same C++ variable (needed for
        MTP draft step loop where same weights are used across multiple steps)."""
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
    def _requantize_fp8_for_ue8m0(weight_fp8, scale_inv):
        """Re-quantize FP8 weight so that scales are exact powers of 2 (UE8M0).

        SM100 block-scaled UMMA uses UE8M0 (8-bit exponent-only) scale factors.
        Checkpoint float32 scales are NOT powers of 2, so directly converting
        them to UE8M0 introduces up to 2x error per block.

        Fix (same as SGLang/vLLM): dequant → re-quantize with power-of-2 scales.

        Input:
            weight_fp8: [M, K] float8_e4m3fn — original checkpoint FP8 weight
            scale_inv: [ceil(M/128), ceil(K/128)] float32 — original block scale_inv

        Output:
            new_fp8: [M, K] float8_e4m3fn — re-quantized weight
            packed_ue8m0: [M, padded_scale_k] int32 — packed UE8M0 per-row scale
        """
        M, K = weight_fp8.shape
        group_size = 128
        scale_k = K // group_size
        padded_scale_k = ((scale_k + 3) // 4) * 4

        # Step 1: Dequant to float32
        # Expand block scale_inv [ceil(M/128), ceil(K/128)] to per-element [M, K]
        scale_inv_expanded = scale_inv.float().repeat_interleave(
            group_size, dim=0)[:M].repeat_interleave(
            group_size, dim=1)[:, :K]
        w_float = weight_fp8.float() * scale_inv_expanded

        # Step 2: Compute new UE8M0 scales (per 128-element block)
        # Reshape to blocks, find max per block
        w_blocks = w_float.reshape(M, scale_k, group_size)
        block_amax = w_blocks.abs().amax(dim=2).clamp(min=1e-12)  # [M, scale_k]
        # New scale = ceil_to_ue8m0(amax / 448)
        raw_scale = block_amax / 448.0
        ue8m0_exp = torch.ceil(torch.log2(raw_scale.clamp(min=1e-30)))
        new_scale = torch.pow(2.0, ue8m0_exp)  # exact power of 2
        ue8m0_byte = (ue8m0_exp + 127).clamp(0, 254).to(torch.int32)

        # Step 3: Re-quantize to FP8
        new_scale_expanded = new_scale.unsqueeze(2).expand_as(w_blocks)
        w_rescaled = (w_blocks / new_scale_expanded).clamp(-448, 448)
        new_fp8 = w_rescaled.reshape(M, K).to(torch.float8_e4m3fn)

        # Step 4: Pack 4 consecutive UE8M0 bytes into uint32, column-major
        # stored as transposed row-major [packed_k, aligned_M]
        packed_k = padded_scale_k // 4
        aligned_M = ((M + 3) // 4) * 4
        # Pad ue8m0_byte to padded_scale_k columns if needed
        if padded_scale_k > scale_k:
            padding = torch.zeros(M, padded_scale_k - scale_k,
                                  dtype=torch.int32, device=ue8m0_byte.device)
            ue8m0_byte = torch.cat([ue8m0_byte, padding], dim=1)
        # ue8m0_byte is [M, padded_scale_k] — reshape to [M, packed_k, 4]
        ue8m0_groups = ue8m0_byte.reshape(M, packed_k, 4)
        packed_per_row = (ue8m0_groups[:, :, 0]
                          | (ue8m0_groups[:, :, 1] << 8)
                          | (ue8m0_groups[:, :, 2] << 16)
                          | (ue8m0_groups[:, :, 3] << 24))  # [M, packed_k]
        # Create column-major [M, packed_k] scale: physical layout has M contiguous
        # allocate_packed_ue8m0_scale equivalent: strided (M, packed_k) stride (1, aligned_M)
        packed_colmajor = torch.empty_strided(
            (M, packed_k), (1, aligned_M),
            dtype=torch.int32, device=packed_per_row.device)
        packed_colmajor.copy_(packed_per_row)  # copy [M, packed_k] row-major into col-major storage

        return new_fp8.contiguous(), packed_colmajor.view(torch.uint32)

    @property
    def _weights_are_fp8(self):
        """Check if we're working with FP8 weights (vs BF16 post-dequant)."""
        return hasattr(self, '_is_fp8_mode') and self._is_fp8_mode

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
        # B37: when fused rmsnorm+quantize is active, the share_quantize_tag
        # is pre-populated in _fp8_quantize_emitted by the input_layernorm
        # call site, so _fp8_linear_v2's internal quantize is skipped.
        # Also pull the per-layer-unique FP8/scale buffers the fused task
        # wrote to (case-3 fix — see `_emit_fused_rmsnorm_qkv_a_quantize`
        # docstring).
        qkv_a_quantize_tag = (
            self._fused_rmsnorm_quantize_qkv_a_tag(layer_idx)
            if self._fused_rmsnorm_quantize else None)
        qkv_a_fp8_ovr, qkv_a_scale_ovr = None, None
        if self._fused_rmsnorm_quantize:
            qkv_a_fp8_ovr, qkv_a_scale_ovr = self._fused_qkv_a_bufs[layer_idx]
        # Builder-side split-K for the qkv_a GEMM (K=hidden=7168, N=2176): the
        # biggest on-chain decode compute block (~30μs MEDIUMM, ~17/80 CTAs
        # working → ~63 idle), analyzer-ranked #2 system lever. Splitting K via
        # the WORKING dense kernel + bf16 partial reduce fills the idle SMs.
        # The split_k partials re-quantize a bf16 normalized embedding per
        # K-slice (the B37 fused qkv_a FP8 `qkv_a_fp8_ovr` is unused on this
        # path — a minor wasted write; net win). An identity_layer phantom
        # bridge copies that bf16 so the split-K subtree has a single producer.
        # The bridge MUST read the buffer the input-layernorm task actually
        # writes (`_rms_src` below), not the cross-layer-shared
        # `self.rmsnorm_out`: keeping the input-layernorm on the
        # qkv_a->attention->o_proj chain lets `build_annotated_graph` residual-
        # strip the embedding->o_proj-residual edges, otherwise the embedding
        # is flagged as a case-3 fork+join producer (two dependent_events).
        # Gated MPK_DSV3_QKVA_SPLITK (int >=2); K must be divisible by 128*sk.
        _qkv_sk = int(os.environ.get("MPK_DSV3_QKVA_SPLITK", "0"))
        if _qkv_sk >= 2 and w_qkv_a.dim(1) % (128 * _qkv_sk) == 0:
            # Source the bf16 normalized embedding that the split-K partials
            # re-quantize. CRITICAL: when `_fused_rmsnorm_quantize` is ON (the
            # default), the input-layernorm fused task writes a *per-layer*
            # bf16 buffer (`_fused_rmsnorm_out_per_layer[layer_idx]`) and
            # leaves the cross-layer-shared `self.rmsnorm_out` untouched (it
            # also runs with emit_bf16=False because the non-split-K qkv_a GEMM
            # reads the FP8 override directly). Reading `self.rmsnorm_out` here
            # would (a) feed the split-K stale/garbage data, and (b) orphan the
            # input-layernorm task from the qkv_a->attention->o_proj chain so
            # `build_annotated_graph`'s residual strip can no longer remove the
            # embedding->o_proj-residual edges — making the embedding a
            # fork+join (case 3) producer. Bind to the buffer the fused task
            # actually writes, and force that write on (emit_bf16) below.
            _rms_src = self.rmsnorm_out
            if (self._fused_rmsnorm_quantize
                    and layer_idx in getattr(
                        self, "_fused_rmsnorm_out_per_layer", {})):
                # The fused input-layernorm task wrote its bf16 output into
                # this per-layer buffer (with emit_bf16 forced on because
                # `_qkva_splitk_active()` is true — see
                # `_emit_fused_rmsnorm_qkv_a_quantize`).
                _rms_src = self._fused_rmsnorm_out_per_layer[layer_idx]
            if not hasattr(self, "_qkva_sk_bridge"):
                self._qkva_sk_bridge = {}
            if layer_idx not in self._qkva_sk_bridge:
                self._qkva_sk_bridge[layer_idx] = self.mpk.new_tensor(
                    dims=(_rms_src.dim(0), _rms_src.dim(1)),
                    dtype=bfloat16,
                    name=f"layer_{layer_idx}_rmsnorm_out_qkv_sk",
                    io_category="cuda_tensor")
            _bridge = self._qkva_sk_bridge[layer_idx]
            # identity dim_map splits the LAST (hidden) dim across grid.x;
            # grid.x must divide hidden. 7168 = 56*128.
            _hid = _rms_src.dim(1)
            _gx = 56 if (_hid % 56 == 0) else (8 if (_hid % 8 == 0) else 1)
            self.mpk.identity_layer(
                input=_rms_src, output=_bridge,
                grid_dim=(_gx, 1, 1), block_dim=(128, 1, 1))
            self._fp8_linear_builder_splitk(
                _bridge, f"{attn}qkv_a_proj", state_dict, self.qkv_a_out,
                _qkv_sk, f"layer_{layer_idx}_qkv_a")
        else:
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
        # q_a slice [0:q_lora_rank) inside the fused qkv_a_out buffer.
        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}q_a_layernorm.weight"],
            name=f"layer_{layer_idx}_q_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
            process_dim=self.q_lora_rank,
            in_offset_elems=self._qkv_a_q_offset,
            out_offset_elems=self._qkv_a_q_offset)

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
        # Hoisted above the bmm branch so _dsv3_bmm path can still share
        # quantize with prefill q_b_nope/pe below (fixes UnboundLocalError).
        qb_share_tag = (
            f"layer_{layer_idx}_qb_q_a_shared"
            if self._use_prefill else None)
        if self._dsv3_bmm:
            # MPK_DSV3_BMM=1: replace the load-time absorbed q_b_proj with
            # runtime BMM-based absorption. Five tasks instead of one
            # monolithic FP8 GEMM, but each task loads smaller per-head
            # weights → less TMA traffic, better overlap potential.
            self._bmm_decode_q_path(state_dict, attn, layer_idx, qb_slice_kwargs)
        else:
            # Existing absorbed-Q path (default).
            w_q_b, s_q_b = self._attach_fp8_weight(
                state_dict, f"{attn}q_b_proj.weight",
                f"layer_{layer_idx}_q_b_proj")
            self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                             grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
                             block_dim=(128, 1, 1),
                             gate_mode=2 if self._use_prefill else 0,
                             share_quantize_tag=qb_share_tag,
                             **qb_slice_kwargs)
        if self._use_prefill:
            if self._qb_fused:
                # FuseTensor path (2026-05-12 user #2): single FP8 GEMM
                # emitting q_b_prefill_fused (mbt, H*192). chunked_prefill
                # reads with qfused_mode=1 (kernel splits via offset 128).
                w_q_b_unabs, s_q_b_unabs = self._attach_fp8_weight(
                    state_dict, f"{attn}q_b_proj_unabsorbed.weight",
                    f"layer_{layer_idx}_q_b_proj_unabsorbed")
                self._fp8_linear(
                    self.q_a_out, w_q_b_unabs, s_q_b_unabs,
                    self.q_b_prefill_fused,
                    grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_unabs.dim(0)),
                              1, 1),
                    block_dim=(128, 1, 1),
                    gate_mode=1,
                    share_quantize_tag=qb_share_tag,
                    **qb_slice_kwargs)
            else:
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
        self.mpk.deepseek_mla_rope_q_fused_layer(
            q_nope_pe=self.q_nope_pe,
            cos_pos_embed=self.cos_pos_embed,
            sin_pos_embed=self.sin_pos_embed,
            num_heads=self.num_local_q_heads,
            grid_dim=rope_q_grid,
            q_tile_size=self.max_num_batched_tokens,
            phase_gate=2 if self._use_prefill else 0,
        )
        if self._use_prefill:
            # 2026-05-12 (user #2 FuseTensor row-swap): when q_b_prefill_fused
            # aliases q_pe, the ROPE kernel must use the row-swap addressing
            # (row stride = H*192, pe block at H*128 within each row). The
            # default split variant assumes a standalone (mbt, H*64) buffer.
            self.mpk.deepseek_mla_rope_q_split_layer(
                q_pe=self.q_pe,
                cos_pos_embed=self.cos_pos_embed,
                sin_pos_embed=self.sin_pos_embed,
                num_heads=self.num_local_q_heads,
                grid_dim=rope_q_grid,
                q_tile_size=self.max_num_batched_tokens,
                qfused_mode=1 if self._qb_fused else 0,
                phase_gate=1,
            )
        # k_pe lives at cols [2048:2112) inside the 2176-wide qkv_a_out;
        # pass row stride + offset so the ROPE kernel rotates the right slice.
        self.mpk.deepseek_mla_rope_k_layer(
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
        single_split_mla = kv_tiles_max <= self._mla_single_split_max_kv_tiles
        mla_num_splits_override = (
            int(self._mla_num_splits_override)
            if self._mla_num_splits_override
            else (1 if single_split_mla else None)
        )
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
            self.mpk.mla_kv_gather_unified_layer(
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
            self.mpk.mla_prefill_tp8_chunked_layer(
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
                # FuseTensor (2026-05-12 user #2): q_nope and q_pe are the
                # same fused [mbt, H*192] tensor when MPK_DSV3_QB_FUSED=1.
                qfused_mode=1 if self._qb_fused else 0,
            )
            if self.world_size == 2:
                self.mpk.mla_mtp_decode_tp2_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
            elif self.world_size == 4:
                self.mpk.mla_mtp_decode_tp4_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
            elif self.world_size == 8:
                self.mpk.mla_mtp_decode_tp8_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
            else:
                self.mpk.mla_mtp_decode_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max)
            if not single_split_mla:
                if self.world_size == 2:
                    self.mpk.mla_mtp_decode_tp2_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
                elif self.world_size == 4:
                    self.mpk.mla_mtp_decode_tp4_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
                elif self.world_size == 8:
                    self.mpk.mla_mtp_decode_tp8_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
                else:
                    self.mpk.mla_mtp_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
        else:
            if self.world_size == 2:
                self.mpk.mla_mtp_decode_tp2_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
                if not single_split_mla:
                    self.mpk.mla_mtp_decode_tp2_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
            elif self.world_size == 4:
                self.mpk.mla_mtp_decode_tp4_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
                if not single_split_mla:
                    self.mpk.mla_mtp_decode_tp4_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
            elif self.world_size == 8:
                self.mpk.mla_mtp_decode_tp8_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
                if not single_split_mla:
                    self.mpk.mla_mtp_decode_tp8_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
            else:
                self.mpk.mla_mtp_decode_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max)
                if not single_split_mla:
                    self.mpk.mla_mtp_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)

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
            # C9: BMM post-attn path replaces the fused decode o_proj
            # (absorbed with W_UV) with: quantize(attn_out) → BMM(kv_b_v_bmm)
            # → smaller o_proj_original linear. Eliminates the H*512 fused
            # GEMM in favor of an H*128 unabsorbed GEMM + BMM. Gated by
            # _dsv3_bmm. The BMM is decode-only via gate_mode=2.
            if self._dsv3_bmm:
                self._bmm_decode_o_path(state_dict, attn, layer_idx, residual=self.x)
            else:
                w_o_decode, s_o_decode = self._attach_fp8_weight(
                    state_dict, f"{attn}o_proj.weight",
                    f"layer_{layer_idx}_o_proj")
                self._fp8_linear(
                    self.attn_out,
                    w_o_decode,
                    s_o_decode,
                    self.attn_proj_out,
                    grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                    block_dim=(128, 1, 1),
                    residual=self.x,
                    gate_mode=2,
                )
        else:
            # Pure decode (mbt<=8). When _dsv3_bmm=True, route through the
            # post-attn BMM path; otherwise legacy absorbed o_proj.
            if self._dsv3_bmm:
                self._bmm_decode_o_path(state_dict, attn, layer_idx, residual=self.x)
            else:
                w_o, s_o = self._attach_fp8_weight(
                    state_dict, f"{attn}o_proj.weight", f"layer_{layer_idx}_o_proj")
                o_split_k = self._pick_fp8_splitk_factor(w_o)
                if o_split_k is not None and self.world_size == 1:
                    self.attn_proj_out = self.x
                    self._fp8_linear_splitk(
                        self.attn_out, w_o, s_o, self.attn_proj_out,
                        split_k=o_split_k, residual=self.x)
                elif o_split_k is not None:
                    self._fp8_linear_splitk(
                        self.attn_out, w_o, s_o, self.attn_proj_out,
                        split_k=o_split_k, residual=self.x)
                else:
                    self._fp8_linear(
                        self.attn_out,
                        w_o,
                        s_o,
                        self.attn_proj_out,
                        grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                        block_dim=(128, 1, 1),
                        residual=self.x,
                    )

    def _build_dense_mlp(self, layer_idx: int, state_dict: dict):
        """Build dense MLP for layers 0-2 (FP8 weights)."""
        prefix = f"model.layers.{layer_idx}."

        w_gate_up, s_gate_up = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.gate_up_proj.weight",
            f"layer_{layer_idx}_gate_up_proj")
        gate_up_split_k = self._pick_fp8_splitk_factor(w_gate_up)
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
            os.environ.get("MPK_DSV3_TP2_GATE_UP_SPLIT", "1") == "1"
            and self.world_size == 2
            and gate_up_split_k is None
            and w_gate_up.dim(0) % 2 == 0
        )
        if gate_up_split_k is not None:
            self._fp8_linear_splitk(
                self.rmsnorm_out, w_gate_up, s_gate_up, self.mlp_mid,
                split_k=gate_up_split_k)
        elif split_tp2_gate_up:
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
        down_split_k = self._pick_fp8_splitk_factor(w_down)
        if down_split_k is not None and self.world_size == 1:
            # TP=1 splitk path: alias mlp_out to self.x for in-place residual
            # accumulation via tma_reduce_add.
            self.mlp_out = self.x
        else:
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
            use_splitk=(down_split_k is not None),
            splitk_split_k=down_split_k,
        )

    def _setup_new_moe_buffers(self):
        """Lazy-init the SHARED NEW-MoE buffers + static m_indices buffer
        the first time _build_moe_mlp runs with MPK_DSV3_NEW_MOE=1.

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
        """Per-layer NEW MoE task dispatch. Extracted so the bisect gate
        in `_build_moe_mlp` can skip it without indentation gymnastics.

        Set MPK_DSV3_NEW_MOE_TASKS_UPTO=N (default 99) to stop after the
        N-th task; rest skipped. Tasks numbered 1..8 below.
        """
        upto = int(os.environ.get("MPK_DSV3_NEW_MOE_TASKS_UPTO", "99"))
        if upto < 1: return
        # Pack W13 weight scale (always — needs to be attached for w13).
        s_w13_packed = self._pack_and_attach_moe_weight_scale(
            state_dict, w13_scale_key,
            f"layer_{layer_idx}_experts_w13_scale_ue8m0")
        # 1) Quantize MoE input with UE8M0-packed scale.
        # C17 (2026-05-17): when post-attn rmsnorm was fused with this
        # quantize (via _emit_fused_post_attn_rmsnorm_moe_quantize at
        # build_layers), the FP8 + scale buffers are already filled by
        # the fused task — skip the redundant standalone quantize call.
        if layer_idx not in getattr(self, "_fused_post_attn_bufs", {}):
            self.mpk.quantize_fp8_layer(
                input=self.rmsnorm_out,
                output_fp8=new_moe_input_fp8,
                output_scale=new_moe_input_scale,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                scale_ue8m0=True,
            )
        if upto < 2: return
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
        if upto < 3: return
        # 3) Permute + scale-transpose.
        # E_PER_CTA: collapse the decode permute valley. E_LOCAL
        # (= num_local_experts = m_total // bm_pad) must divide evenly.
        _epc = self._moe_permute_epc
        _e_local = m_total // bm_pad
        assert _e_local % _epc == 0, (
            f"MPK_DSV3_PERMUTE_EPC ({_epc}) must divide num_local_experts "
            f"({_e_local})")
        self.mpk.moe_permute_sm100_layer(
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
        if upto < 4: return
        # 4) Group GEMM W13.
        self.mpk.fp8_group_gemm_layer(
            a_fp8=new_moe_permuted_in_fp8,
            b_fp8=w_experts_w13,
            sfa_packed=new_moe_permuted_in_scale,
            sfb_packed=s_w13_packed,
            m_indices=self.new_moe_m_indices_dt,
            output=new_moe_w13_out,
            num_workers=self.mpk.num_workers,
            meta=new_moe_meta if self._new_moe_active_skip else None,
        )
        if upto < 5: return
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
        # C18 (2026-05-17): when env-gated, fuse silu·mul + quantize_fp8
        # into one task. Skip both standalone steps. Note: the fused
        # kernel doesn't yet implement the B11/B15 active-expert skip
        # (it processes every row); for ACTIVE_SKIP=1 callers, the fused
        # path does more silu·mul work on inactive rows than the
        # standalone path. Future: thread the active-skip through.
        if self._fused_silu_quantize:
            # C18 (2026-05-17): pass meta + bm_padding when active-skip is
            # enabled so the kernel skips inactive-expert rows (matches B11
            # silu_mul + B15 quantize_fp8 behavior). Without active-skip,
            # the fused kernel processes garbage rows and regresses badly.
            _silu_quant_meta = (
                new_moe_meta if self._new_moe_active_skip else None)
            self.mpk.moe_silu_mul_quantize_fp8_sm100_layer(
                input=new_moe_w13_out,
                output_fp8=new_moe_silu_fp8,
                output_scale=new_moe_silu_scale,
                grid_dim=(m_total, 1, 1),
                block_dim=(128, 1, 1),
                rows_per_task=1,
                meta=_silu_quant_meta,
                bm_padding=bm_pad,
            )
            if upto < 7: return
            # Skip the standalone silu_mul + quantize (steps 5+6).
            # Jump straight to step 7 (W2 GEMM).
        else:
            self.mpk.moe_silu_mul_layer(
                input=new_moe_w13_out,
                output=new_moe_silu_out,
                grid_dim=(_silu_grid, 1, 1),
                block_dim=(128, 1, 1),
                meta=new_moe_meta if self._new_moe_active_skip else None,
                # bm_padding = per-expert row count in the permuted buffer.
                # Wrapper combines with rows_per_cta (= input.dim(0)/grid.x)
                # to derive my_expert = bid.x / (bm_padding / rows_per_cta).
                bm_padding=bm_pad,
            )
            if upto < 6: return
            # 6) Quantize SiLU → UE8M0 directly into K-outermost layout.
            # C8 (2026-05-16): quantize_fp8 kernel writes UE8M0 packed scale at
            # offset `packed_idx * aligned_batch + batch_idx` which IS K-outer
            # row-major. The previous (m_total, K_PACKED) declaration was a
            # "shape lie" that required a separate transpose_scale task to
            # reconcile. By declaring the output as (K_PACKED, m_total) the
            # write pattern matches the declared shape, and the downstream W2
            # SFA TMA descriptor (which expects K-outer) reads correct bytes
            # directly — eliminating TRANSPOSE_SCALE (-19 μs/layer).
            if self._new_moe_active_skip:
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
            else:
                # B12 fallback (no active-skip): process_all_rows still
                # required because the row axis is permuted-expert layout,
                # not token-indexed.
                self.mpk.quantize_fp8_layer(
                    input=new_moe_silu_out,
                    output_fp8=new_moe_silu_fp8,
                    output_scale=new_moe_silu_scale,
                    grid_dim=(m_total, 1, 1),
                    block_dim=(128, 1, 1),
                    scale_ue8m0=True,
                    process_all_rows=True,
                )
        # C8: transpose_scale eliminated — quantize_fp8 writes directly
        # into the K-outer-declared new_moe_silu_scale.
        if upto < 8: return
        # 7+8) Pack W2 weight scale + attach W2 weight + Group GEMM W2.
        w2_scale_key_for_pack = f"{prefix}experts.w2.weight_scale_inv"
        w_experts_w2_new = self._safe_attach(
            state_dict[f"{prefix}experts.w2.weight"],
            f"layer_{layer_idx}_experts_w2")
        s_w2_packed = self._pack_and_attach_moe_weight_scale(
            state_dict, w2_scale_key_for_pack,
            f"layer_{layer_idx}_experts_w2_scale_ue8m0")
        self.mpk.fp8_group_gemm_layer(
            a_fp8=new_moe_silu_fp8,
            b_fp8=w_experts_w2_new,
            sfa_packed=new_moe_silu_scale,
            sfb_packed=s_w2_packed,
            m_indices=self.new_moe_m_indices_dt,
            output=new_moe_w2_out,
            num_workers=self.mpk.num_workers,
            meta=new_moe_meta if self._new_moe_active_skip else None,
        )

    def _build_shared_expert(self, layer_idx: int, prefix: str, state_dict: dict):
        """Register shared-expert dense FP8 path. Returns ``shared_residual``
        which the routed-MoE finalize step (moe_unpermute / moe_mul_sum_add)
        adds to the per-token routed contribution.

        Registered ahead of the routed-MoE block (C12, 2026-05-17) so the
        runtime scheduler stamps these tasks onto workers before the W13
        group-GEMM wave. Both paths depend only on ``self.rmsnorm_out``, so
        the reorder is dep-safe and may expose worker-level parallelism via
        EVENT_LAUNCH_MASSIVE_TASKS round-robin (see scheduler topology notes).
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
        shared_gu_split_k = self._pick_fp8_splitk_factor(w_shared_gate_up)
        _bsk = int(os.environ.get("MPK_DSV3_BUILDER_SPLITK", "0"))
        if (_bsk >= 2 and has_shared_scale
                and w_shared_gate_up.dim(1) % (128 * _bsk) == 0):
            # Builder-side split-K (decode 1-CTA-GEMM parallelization, the
            # −45μs system lever; the existing decode_splitk kernel crashes
            # at TP=4 so we split-K via the working dense kernel + reduce).
            self._fp8_linear_builder_splitk(
                self.rmsnorm_out, fused_key, state_dict, shared_mid, _bsk,
                f"layer_{layer_idx}_shared_gate_up")
        elif shared_gu_split_k is not None:
            self._fp8_linear_splitk(
                self.rmsnorm_out, w_shared_gate_up, s_shared_gate_up,
                shared_mid, split_k=shared_gu_split_k)
        else:
            self._fp8_linear(self.rmsnorm_out, w_shared_gate_up, s_shared_gate_up,
                             shared_mid,
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

        # C12 (2026-05-17, NULL RESULT): Register shared_expert BEFORE routed
        # MoE via MPK_DSV3_SHARED_EXPERT_FIRST=1. Hypothesis: same fork-event
        # broadcast lets shared_expert's FP8 GEMMs occupy workers ahead of
        # W13 group_gemm. Tested at TP=4 EP=2 mbt=128 19l: ON 1306 μs/layer
        # vs OFF 1282 μs (within noise, ~1.9%). Overlap analysis: only ~7 μs
        # shared_expert × W13 per layer because W13 is 80% mbarrier-stalled
        # (242 μs span / 42 μs compute union), so its idle slots happen
        # during waits the runtime doesn't schedule into. Default OFF; helper
        # kept for future scheduler-side experiments.
        _shared_first = os.environ.get(
            "MPK_DSV3_SHARED_EXPERT_FIRST", "0") == "1"
        shared_residual = None
        if _shared_first:
            shared_residual = self._build_shared_expert(
                layer_idx, prefix, state_dict)

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
        if self._BF16_GATE_SPLITK_ENABLED:
            # Router gate via BF16 splitk swapAB: output=NUM_EXPERTS, K=hidden.
            # grid.x splits N into 128-wide tiles; grid.y is split_k chosen to
            # pack as many tasks as possible into a single worker wave (~128
            # SMs on B200). accumulate=False so the prepended tensor_init
            # zeroes router_logits before reduce-add.
            gate_split_k = self._pick_bf16_splitk_factor(w_gate)
            self.mpk.splitk_linear_layer(
                input=self.rmsnorm_out,
                weight=w_gate,
                output=router_logits,
                grid_dim=(w_gate.dim(0) // 128, gate_split_k, 1),
                block_dim=(256, 1, 1),
                accumulate=False,
            )
        else:
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
        # C13 (2026-05-17): VPT 8 -> 16 (env MPK_DSV3_TOPK_VPT) makes the
        # kernel template pick ROWS_PER_WARP=2 instead of 1. block_dim stays
        # at 256 (8 warps); only ROWS_PER_CTA changes from 8 to 16. The
        # change actually lives in src/kernel/task_register.cc (template arg);
        # this comment exists so callers know the parity.
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
        # NEW MoE PATH — wires through PR-674 fp8_group_gemm_smallm/largem.
        # Permute(routing) → group_gemm(W13) → silu → quantize → transpose →
        # group_gemm(W2) → unpermute(combine + residual). Gated by env var
        # MPK_DSV3_NEW_MOE=1 so we can A/B against the OLD per-expert kernel.
        # See scratch/pr674_moe_kernel_wiring_plan.md for the full design.
        # ====================================================================
        # DEBUG bisect levels for the NEW MoE correctness bug
        # (layer_02_residual row 127 anomaly):
        #   MPK_DSV3_NEW_MOE_DRY_RUN=B  → setup only (just attach static
        #                                  m_indices); skip per-layer allocs
        #                                  and tasks; fall through to OLD.
        #   MPK_DSV3_NEW_MOE_DRY_RUN=C  → setup + per-layer buffer alloc;
        #                                  skip tasks; fall through to OLD.
        #   MPK_DSV3_NEW_MOE_DRY_RUN=0  → (default) full NEW path.
        new_moe_dry_run = os.environ.get("MPK_DSV3_NEW_MOE_DRY_RUN", "0")
        # NEW MoE branch. The setup runs whenever env is set (so we can
        # isolate whether merely attaching m_indices_static causes drift).
        # The per-layer body skips for bisect level "B".
        if self._new_moe and use_fp8_experts:
            self._setup_new_moe_buffers()  # idempotent
        if (self._new_moe and use_fp8_experts
                and new_moe_dry_run != "B"):
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
            # C17 (2026-05-17): if post-attn rmsnorm+quantize was fused for
            # this layer, the fused task already wrote new_moe_input_fp8 +
            # new_moe_input_scale into a per-layer buffer — reuse those
            # instead of allocating new (otherwise we'd lose the fused
            # output and _new_moe_dispatch_inline would re-quantize anyway).
            _post_attn_bufs = getattr(self, "_fused_post_attn_bufs", {})
            if layer_idx in _post_attn_bufs:
                _, new_moe_input_fp8, new_moe_input_scale = (
                    _post_attn_bufs[layer_idx])
            else:
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

            if new_moe_dry_run == "C":
                # Bisect level C: allocations done, skip all NEW task
                # dispatch + tasks; fall through to OLD path.
                new_moe_skip_old_routed = False
            else:
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
                # Always skip OLD routed path when NEW path takes over.
                # If MPK_DSV3_NEW_MOE_TASKS_UPTO < 8 truncates the dispatch
                # mid-chain, the moe_unpermute downstream will read partly-
                # uninitialized buffers (layer 3 output will be garbage)
                # but the dense layers 0..2 residual dumps stay valid for
                # bisect comparison.
                new_moe_skip_old_routed = True

        elif self._new_moe and use_fp8_experts and new_moe_dry_run == "B":
            # Level B: setup-only (handled above), no per-layer body.
            new_moe_skip_old_routed = False
        else:
            new_moe_skip_old_routed = False

        # (NEW MoE dispatch body extracted into `_new_moe_dispatch_inline`.)

        # OLD routed-experts path (W13 → silu → W2). Skipped when NEW MoE
        # path took over above.
        if not new_moe_skip_old_routed:
            # Group GEMM expects per-row weight_scale (not per-block scale_inv)
            # Checkpoint: scale_inv [num_experts, out/128, K/128]
            # Kernel expects: scale [num_experts*out, K/128] (per-row, float32)
            if use_fp8_experts:
                raw_scale_inv = state_dict[w13_scale_key].float().clamp(min=1e-30)
                # scale_inv IS the dequant scale (weight_float = weight_fp8 * scale_inv)
                # Group GEMM kernel expects this directly, NOT 1/scale_inv
                # Expand per-block to per-row: repeat each block row 128 times
                # Result: [num_experts, out_rows, K/128] — 3D (PR 652 format)
                w13_scale_expanded = raw_scale_inv.repeat_interleave(128, dim=1).contiguous().to(torch.float32)
                s_experts_w13 = self._safe_attach(
                    w13_scale_expanded, f"layer_{layer_idx}_experts_w13_scale")
            else:
                s_experts_w13 = None
            if use_fp8_experts:
                # Quantize input for MoE FP8
                moe_input_fp8 = self.mpk.new_tensor(
                    dims=(mbt, self.hidden_size), dtype=float8_e4m3,
                    name=f"layer_{layer_idx}_moe_input_fp8", io_category="cuda_tensor",
                )
                moe_input_scale = self.mpk.new_tensor(
                    dims=(mbt, self.hidden_size // 128), dtype=float32,
                    name=f"layer_{layer_idx}_moe_input_scale", io_category="cuda_tensor",
                )
                # MoE group GEMM expects float32 scale (does internal UE8M0 conversion)
                self.mpk.quantize_fp8_layer(
                    input=self.rmsnorm_out,
                    output_fp8=moe_input_fp8,
                    output_scale=moe_input_scale,
                    grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                    scale_ue8m0=False,
                )

            moe_mid = self.mpk.new_tensor(
                dims=(mbt, NUM_EXPERTS_PER_TOK, 2 * self.routed_moe_intermediate_size),
                dtype=bfloat16,
                name=f"layer_{layer_idx}_moe_mid",
                io_category="cuda_tensor",
            )

            if use_fp8_experts:
                # 2026-05-12: MoE W13 dominates prefill (88% of layer wallclock,
                # 4076 μs/call mean per perf-analyzer). Env override lets a sweep
                # find a better Y without rebuilding the builder.
                _w13_pref = int(os.environ.get("MPK_MOE_W13_M_SPLIT", "16"))
                w13_m_split = _moe_fp8_m_split(2 * self.routed_moe_intermediate_size,
                                               preferred=_w13_pref)
                w13_expert_grid_x = _moe_expert_grid_x(
                    mbt, self.num_local_experts, preferred_groups=8)
                self.mpk.moe_w13_fp8_layer(
                    input_fp8=moe_input_fp8,
                    input_scale=moe_input_scale,
                    weight_fp8=w_experts_w13,
                    weight_scale=s_experts_w13,
                    moe_routing_indices=moe_routing_indices,
                    moe_mask=moe_mask,
                    output=moe_mid,
                    grid_dim=(w13_expert_grid_x, w13_m_split, 1),
                    block_dim=(128, 1, 1),
                )
            else:
                raise RuntimeError("No bf16 moe experts for now.")

            moe_silu_out = self.mpk.new_tensor(
                dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size),
                dtype=bfloat16,
                name=f"layer_{layer_idx}_moe_silu",
                io_category="cuda_tensor",
            )
            # NOTE: grid=(mbt, topk, 1) kept as-is. silu_mul has a known
            # OOB-write pattern (CTAs > BM write past buffer end since the
            # kernel iterates num_active_tokens=mbt*topk rows from each
            # CTA's offset). For OLD MoE this has been benign in
            # production (writes apparently land in pool padding); FIXING
            # OLD to grid=(1,1,1) introduces new divergence at L15+ vs
            # established baseline. So OLD silu_mul stays as-is. NEW MoE
            # path uses grid=(1,1,1) — see _new_moe_dispatch_inline.
            self.mpk.moe_silu_mul_layer(
                input=moe_mid, output=moe_silu_out,
                grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1),
                block_dim=(128, 1, 1),
            )

            # Expert W2 (down projection)
            w2_scale_key = f"{prefix}experts.w2.weight_scale_inv"
            w_experts_w2 = self._safe_attach(
                state_dict[f"{prefix}experts.w2.weight"],
                f"layer_{layer_idx}_experts_w2")
            if use_fp8_experts:
                raw_scale_inv = state_dict[w2_scale_key].float().clamp(min=1e-30)
                w2_scale_expanded = raw_scale_inv.repeat_interleave(128, dim=1).contiguous().to(torch.float32)
                s_experts_w2 = self._safe_attach(
                    w2_scale_expanded, f"layer_{layer_idx}_experts_w2_scale")
            else:
                s_experts_w2 = None

            if use_fp8_experts:
                moe_silu_fp8 = self.mpk.new_tensor(
                    dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size),
                    dtype=float8_e4m3,
                    name=f"layer_{layer_idx}_moe_silu_fp8",
                    io_category="cuda_tensor",
                )
                moe_silu_scale = self.mpk.new_tensor(
                    dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size // 128),
                    dtype=float32,
                    name=f"layer_{layer_idx}_moe_silu_scale",
                    io_category="cuda_tensor",
                )
                self.mpk.quantize_fp8_layer(
                    input=moe_silu_out,
                    output_fp8=moe_silu_fp8,
                    output_scale=moe_silu_scale,
                    grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1),
                    scale_ue8m0=False,
                    block_dim=(128, 1, 1),
                )

            moe_down_out = self.mpk.new_tensor(
                dims=(mbt, NUM_EXPERTS_PER_TOK, self.hidden_size),
                dtype=bfloat16,
                name=f"layer_{layer_idx}_moe_down",
                io_category="cuda_tensor",
            )
            if use_fp8_experts:
                _w2_pref = int(os.environ.get("MPK_MOE_W2_M_SPLIT", "14"))
                w2_m_split = _moe_fp8_m_split(self.hidden_size, preferred=_w2_pref)
                w2_expert_grid_x = _moe_expert_grid_x(
                    mbt, self.num_local_experts, preferred_groups=10)
                self.mpk.moe_w2_fp8_layer(
                    input_fp8=moe_silu_fp8,
                    input_scale=moe_silu_scale,
                    weight_fp8=w_experts_w2,
                    weight_scale=s_experts_w2,
                    moe_routing_indices=moe_routing_indices,
                    moe_mask=moe_mask,
                    output=moe_down_out,
                    grid_dim=(w2_expert_grid_x, w2_m_split, 1),
                    block_dim=(128, 1, 1),
                )
            else:
                raise RuntimeError("No bf16 moe experts for now.")

        # ---- Shared Expert ----
        # Registered earlier via _build_shared_expert() at the top of
        # _build_moe_mlp when MPK_DSV3_SHARED_EXPERT_FIRST=1 (default, C12).
        # Fall back to legacy registration order otherwise.
        if shared_residual is None:
            shared_residual = self._build_shared_expert(
                layer_idx, prefix, state_dict)

        # Final MoE contribution before transformer residual:
        #   routed_experts * topk_weights + shared_expert
        # The model residual is added after the tensor-parallel allreduce in
        # build_layers, otherwise each rank would add the same residual before
        # the reduction and over-count it.
        if new_moe_skip_old_routed:
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
            self.mpk.moe_unpermute_sm100_layer(
                permuted_output=self._new_moe_layer_w2_out,
                meta=self._new_moe_layer_meta,
                residual=shared_residual,
                output=moe_output,
                # C4 (2026-05-16): rows_per_cta 1→8 collapses grid to
                # (16, 8, 1)=128 CTAs = 1 wave (was 1024 CTAs = 8 waves).
                # Per-CTA work 8x more but wave-transition overhead × 7
                # was dominating cluster wallclock at ~17 μs/call. Env
                # override: MPK_DSV3_UNPERMUTE_ROWS_PER_CTA={1,2,4,8}.
                rows_per_cta=int(os.environ.get(
                    "MPK_DSV3_UNPERMUTE_ROWS_PER_CTA", "8")),
                hidden_split=8,
            )
        else:
            self.mpk.moe_mul_sum_add_layer(
                input=moe_down_out,
                weight=moe_topk_weights,
                residual=shared_residual,
                output=moe_output,
                grid_dim=(self.max_num_batched_tokens, _moe_hidden_split(self.hidden_size), 1),
                block_dim=(128, 1, 1),
            )
        self.mlp_out = moe_output

    def _cached_attach(self, tensor, name, **kwargs):
        """attach_input with caching — avoids duplicate C++ variable declarations
        when MTP draft loop calls _build_mtp_decoder_layer multiple times."""
        safe_name = name.replace('.', '_')
        if safe_name in self._attach_cache:
            return self._attach_cache[safe_name]
        dtensor = self.mpk.attach_input(torch_tensor=tensor, name=safe_name, **kwargs)
        self._attach_cache[safe_name] = dtensor
        return dtensor

    def _cached_new_tensor(self, dims, dtype, name, io_category="cuda_tensor"):
        """new_tensor with caching — reuses tensor from first call with same name.
        Needed for MTP draft loop where intermediate tensors are shared across steps."""
        safe_name = name.replace('.', '_')
        if safe_name in self._attach_cache:
            return self._attach_cache[safe_name]
        dtensor = self.mpk.new_tensor(dims=dims, dtype=dtype, name=safe_name,
                                       io_category=io_category)
        self._attach_cache[safe_name] = dtensor
        return dtensor

    def _build_mtp_decoder_layer(self, state_dict: dict, prefix: str):
        """Build one MTP decoder layer (same structure as main model layer).

        The MTP block is a full DeepseekV2DecoderLayer with its own weights.
        It shares the same architecture: input_layernorm → MLA → post_norm → MLP.
        """
        # Input layernorm
        w_norm = self._cached_attach(
            state_dict[f"{prefix}input_layernorm.weight"],
            "mtp_block_input_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.mtp_x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
        )

        # MLA attention (same structure as main model, own weights)
        # Set self.x = self.mtp_x so that _build_mla_attention_layer_with_prefix
        # uses the correct hidden state for fuse_residual (it reads self.x for
        # the residual in o_proj and down_proj).
        _saved_x = self.x
        self.x = self.mtp_x
        self._build_mla_attention_layer_with_prefix(prefix, state_dict)

        # Residual fused inside o_proj (with_residual kernel). attn_proj_out
        # already includes self.x as the residual (mirrors main layer).
        self.mtp_x = self.attn_proj_out
        # Restore self.x to main model's hidden state
        self.x = _saved_x

        # Post-attention layernorm
        w_post_norm = self._cached_attach(
            state_dict[f"{prefix}post_attention_layernorm.weight"],
            "mtp_block_post_attn_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.mtp_x, weight=w_post_norm, output=self.rmsnorm_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1),
        )

        # MLP: DeepSeek V3 MTP block uses MoE MLP (same as main layers 3-60)
        # Check if MoE weights exist, fallback to dense
        # Set self.x = self.mtp_x so MLP's fuse_residual uses correct hidden state
        self.x = self.mtp_x
        mlp_gate_key = f"{prefix}mlp.gate.weight"
        # Production DeepSeek V3 MTP checkpoints use the MoE path. The dense
        # fallback is kept for reduced fixtures that omit router weights.
        if mlp_gate_key in state_dict:
            self._build_moe_mlp_with_prefix(prefix, state_dict)
        else:
            self._build_dense_mlp_with_prefix(prefix, state_dict)

        # MLP residual: MTP uses MoE MLP (same as main layers 3-60). MoE has no
        # fused-residual linear variant — the shared_expert's down_proj returns
        # partial output and moe_mul_sum_add combines routed+shared but does
        # not add the MLP-input residual. So match main layer's MoE pattern:
        # allreduce first, then add residual. TP>1 fuses the residual add into
        # the allreduce task's final local store.
        mtp_mlp_residual = self._cached_new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name="mtp_mlp_residual", io_category="cuda_tensor")
        if self.world_size > 1:
            self.mpk.allreduce_layer(
                input=self.mlp_out, buffer=self.allreduce_buf,
                output=mtp_mlp_residual,
                residual=self.mtp_x,
                grid_dim=_tensor_parallel_allreduce_grid(self.hidden_size),
                block_dim=(128, 1, 1),
            )
            self.mtp_x = mtp_mlp_residual
        else:
            _mtp_mlp_contrib = self.mlp_out
            self.mpk.elementwise_add_layer(
                input_a=self.mtp_x, input_b=_mtp_mlp_contrib,
                output=mtp_mlp_residual,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mtp_x = mtp_mlp_residual
        # Restore main model's hidden state
        self.x = _saved_x

    def _build_mla_attention_layer_with_prefix(self, prefix: str, state_dict: dict):
        """Build MLA attention using a custom weight prefix (FP8, for MTP reuse)."""
        attn = f"{prefix}self_attn."

        # q_a_proj (FP8)
        w_q_a, s_q_a = self._attach_fp8_weight(
            state_dict, f"{attn}q_a_proj.weight", f"mtp_{attn}q_a_proj")
        q_a_split_k = self._pick_fp8_splitk_factor(w_q_a)
        if q_a_split_k is not None:
            self._fp8_linear_splitk(
                self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out,
                split_k=q_a_split_k)
        else:
            self._fp8_linear(self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out,
                             grid_dim=(grid_for_rmsnorm_linear_layer(w_q_a.dim(0)), 1, 1),
                             block_dim=(128, 1, 1))

        w_q_a_ln = self._cached_attach(
            state_dict[f"{attn}q_a_layernorm.weight"],
            f"mtp_{attn}q_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1))

        # MTP runs both prefill and decode attention so its KV cache and
        # per-position hidden states are populated correctly. Mirrors vLLM's
        # `DeepseekV2DecoderLayer` flow inside the EagleSpeculator's first
        # post-target call (see `vllm/v1/worker/gpu/spec_decode/eagle/
        # speculator.py:374`): MTP's MLA attention runs over every prefill
        # position, producing the contextualised hidden state that the
        # subsequent draft loop reads.
        #
        # Without prefill attention here, MTP's `mla_kv_gather` writes
        # k/v from `kv_a_proj(rmsnorm(eh_proj(target_h, embed(t))))` for
        # every prefill position, but `attn_out` is left undefined because
        # the decode kernel returns early on `Q_LEN > 8`. The resulting
        # MTP layer output (and downstream o_proj/MLP/residual) is garbage
        # during prefill iterations.
        #
        # An older comment claimed Q_LEN>=9 deadlocks the persistent-kernel
        # schedule when both main and MTP register prefill MLA tasks.
        # That came from commit 54de0a31 (2026-04-30) which predates PR 674
        # chunked prefill. Re-tested 2026-05-07 on `dev-v8-rope-prefill-main`:
        # no deadlock observed (see `scripts/test_mtp_deadlock.sh` and the
        # `regression_test.sh` post-fix run). If the deadlock returns,
        # bisect by flipping this flag back to False and capture a perfetto
        # trace; the dual-prefill schedule is the suspect.
        use_mtp_prefill_attention = True

        # q_b_proj (FP8) — produce the fused q_nope_pe for the decode kernel.
        # Split q_nope/q_pe are only needed by prefill MLA, which the predictor
        # intentionally does not use.
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight", f"mtp_{attn}q_b_proj")
        self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        if use_mtp_prefill_attention:
            w_q_b_nope, s_q_b_nope = self._attach_fp8_weight(
                state_dict, f"{attn}q_b_nope.weight", f"mtp_{attn}q_b_nope")
            self._fp8_linear(
                self.q_a_out, w_q_b_nope, s_q_b_nope, self.q_nope,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_nope.dim(0)), 1, 1),
                block_dim=(128, 1, 1))
            w_q_b_pe, s_q_b_pe = self._attach_fp8_weight(
                state_dict, f"{attn}q_b_pe.weight", f"mtp_{attn}q_b_pe")
            self._fp8_linear(
                self.q_a_out, w_q_b_pe, s_q_b_pe, self.q_pe,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_pe.dim(0)), 1, 1),
                block_dim=(128, 1, 1))

        # kv_a_proj split (FP8)
        kv_a_w = state_dict[f"{attn}kv_a_proj_with_mqa.weight"]
        kv_a_s = state_dict[f"{attn}kv_a_proj_with_mqa.weight_scale_inv"]
        scale_rows_total = kv_a_s.shape[0]
        latent_ratio = self.kv_lora_rank / (self.kv_lora_rank + QK_ROPE_HEAD_DIM)
        scale_rows_latent = round(scale_rows_total * latent_ratio)

        # c_latent: raw FP8 + float32 scale (new dense GEMM kernel format)
        latent_fp8 = kv_a_w[:self.kv_lora_rank].contiguous()
        latent_scale = kv_a_s[:scale_rows_latent].to(torch.float32).contiguous()
        w_kv_latent = self._safe_attach(latent_fp8, f"mtp_{attn}kv_a_latent")
        s_kv_latent = self._safe_attach(latent_scale, f"mtp_{attn}kv_a_latent_scale")
        # kv_a_rope: pad raw FP8 [64, H] → [128, H], pad float32 scale.
        rope_fp8_raw = kv_a_w[self.kv_lora_rank:].contiguous()
        rope_scale_raw = kv_a_s[scale_rows_latent:].to(torch.float32).contiguous()
        rope_fp8_padded = torch.zeros(128, rope_fp8_raw.shape[1],
                                      dtype=rope_fp8_raw.dtype, device=rope_fp8_raw.device)
        rope_fp8_padded[:QK_ROPE_HEAD_DIM] = rope_fp8_raw
        rope_scale_padded = torch.zeros(
            (128 + 127) // 128, rope_scale_raw.shape[1],
            dtype=rope_scale_raw.dtype, device=rope_scale_raw.device)
        rope_scale_padded[:rope_scale_raw.shape[0]] = rope_scale_raw
        w_kv_rope = self._safe_attach(rope_fp8_padded, f"mtp_{attn}kv_a_rope")
        s_kv_rope = self._safe_attach(rope_scale_padded, f"mtp_{attn}kv_a_rope_scale")

        self._fp8_linear(self.rmsnorm_out, w_kv_latent, s_kv_latent, self.c_latent_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.kv_lora_rank), 1, 1),
                         block_dim=(128, 1, 1))
        self._fp8_linear(self.rmsnorm_out, w_kv_rope, s_kv_rope, self.k_pe_out,
                         grid_dim=(1, 1, 1),
                         block_dim=(128, 1, 1))

        rope_q_grid = (
            self.mpk.max_num_batched_requests,
            self.num_local_q_heads,
            1,  # B35: TILE_Q==mbt -> 1 CTA per (req, head); kernel inner-loop covers all tokens
        )
        self.mpk.deepseek_mla_rope_q_fused_layer(
            q_nope_pe=self.q_nope_pe,
            cos_pos_embed=self.cos_pos_embed,
            sin_pos_embed=self.sin_pos_embed,
            num_heads=self.num_local_q_heads,
            grid_dim=rope_q_grid,
            q_tile_size=self.max_num_batched_tokens,
        )
        if use_mtp_prefill_attention:
            self.mpk.deepseek_mla_rope_q_split_layer(
                q_pe=self.q_pe,
                cos_pos_embed=self.cos_pos_embed,
                sin_pos_embed=self.sin_pos_embed,
                num_heads=self.num_local_q_heads,
                grid_dim=rope_q_grid,
                q_tile_size=self.max_num_batched_tokens,
            )
        self.mpk.deepseek_mla_rope_k_layer(
            k_pe=self.k_pe_out,
            cos_pos_embed=self.cos_pos_embed,
            sin_pos_embed=self.sin_pos_embed,
            grid_dim=(
                self.mpk.max_num_batched_requests,
                1,
                1,  # B35: TILE_Q==mbt collapses grid.z to 1
            ),
            q_tile_size=self.max_num_batched_tokens,
        )

        w_kv_a_ln = self._cached_attach(
            state_dict[f"{attn}kv_a_layernorm.weight"],
            f"mtp_{attn}kv_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln, output=self.c_latent_out,
            grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
            block_dim=(128, 1, 1))

        # MTP attention uses its own KV cache and decode-only MLA kernels.
        q_len_mla = self.max_num_batched_tokens
        decode_q_len_mla = self._decode_q_len()
        kv_len_max = self.mpk.max_seq_length
        kv_tiles_max = (kv_len_max + self.mpk.page_size - 1) // self.mpk.page_size
        single_split_mla = kv_tiles_max <= self._mla_single_split_max_kv_tiles
        mla_num_splits_override = (
            int(self._mla_num_splits_override)
            if self._mla_num_splits_override
            else (1 if single_split_mla else None)
        )
        mla_decode_out = self.attn_out if single_split_mla else self.mla_partial_o
        mla_decode_kv = (
            self.mtp_ckv_kpe_cache_tensor
            if self._direct_paged_decode_kv
            else self.contiguous_kv
        )
        # MTP's `mla_prefill_absorbed_layer` (below) reads `self.contiguous_kv`
        # via a dense `bi * MPK_MAX_SEQ_LENGTH * D` offset — it has no paged
        # variant. So we always pass `self.contiguous_kv` as the gather's
        # `contiguous_kv` target (rather than `mla_decode_kv`), even when
        # direct-paged is enabled. Otherwise, with direct-paged on,
        # `mla_decode_kv == mtp_ckv_kpe_cache_tensor` and the dense buffer
        # never gets written, making the absorbed-prefill kernel read stale
        # data. Decode kernels still read from `mla_decode_kv` (paged when
        # direct-paged is on, fast path).
        if use_mtp_prefill_attention:
            self.mpk.mla_kv_gather_unified_layer(
                c_latent_new=self.c_latent_out,
                k_pe_new=self.k_pe_out,
                paged_cache=self.mtp_ckv_kpe_cache_tensor,
                contiguous_kv=self.contiguous_kv,
                ckv_sep=self.ckv_sep,
                kpe_sep=self.kpe_sep,
                mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
                grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
            )
        else:
            self.mpk.mla_kv_gather_layer(
                c_latent_new=self.c_latent_out,
                k_pe_new=self.k_pe_out,
                paged_cache=self.mtp_ckv_kpe_cache_tensor,
                contiguous_kv=mla_decode_kv,
                mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
                grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
            )
        if use_mtp_prefill_attention:
            self.mpk.mla_prefill_absorbed_layer(
                self.q_nope_pe, self.contiguous_kv, self.attn_out,
                mla_params=(
                    self.num_local_q_heads,
                    kv_len_max,
                    self.kv_lora_rank,
                    QK_ROPE_HEAD_DIM,
                    self.v_head_dim,
                ),
                grid_dim=(
                    self.num_local_q_heads,
                    (q_len_mla + 63) // 64,
                    self.mpk.max_num_batched_requests,
                ),
                block_dim=(256, 1, 1),
            )
            if self.world_size == 2:
                self.mpk.mla_mtp_decode_tp2_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
            elif self.world_size == 4:
                self.mpk.mla_mtp_decode_tp4_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
            elif self.world_size == 8:
                self.mpk.mla_mtp_decode_tp8_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
            else:
                self.mpk.mla_mtp_decode_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max)
            if not single_split_mla:
                if self.world_size == 2:
                    self.mpk.mla_mtp_decode_tp2_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
                elif self.world_size == 4:
                    self.mpk.mla_mtp_decode_tp4_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
                elif self.world_size == 8:
                    self.mpk.mla_mtp_decode_tp8_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
                else:
                    self.mpk.mla_mtp_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
        else:
            if self.world_size == 2:
                self.mpk.mla_mtp_decode_tp2_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
                if not single_split_mla:
                    self.mpk.mla_mtp_decode_tp2_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
            elif self.world_size == 4:
                self.mpk.mla_mtp_decode_tp4_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
                if not single_split_mla:
                    self.mpk.mla_mtp_decode_tp4_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
            elif self.world_size == 8:
                self.mpk.mla_mtp_decode_tp8_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max,
                    num_splits_override=mla_num_splits_override)
                if not single_split_mla:
                    self.mpk.mla_mtp_decode_tp8_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)
            else:
                self.mpk.mla_mtp_decode_layer(
                    self.q_nope_pe, mla_decode_kv,
                    mla_decode_out, self.mla_partial_lse,
                    decode_q_len_mla, kv_len_max)
                if not single_split_mla:
                    self.mpk.mla_mtp_reduce_layer(
                        self.mla_partial_o, self.mla_partial_lse,
                        self.attn_out, decode_q_len_mla, kv_len_max)

        # o_proj (FP8). Match main layer's pattern: fuse (matmul + residual)
        # via the with_residual kernel, or via splitk + alias on TP=1.
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj.weight", f"mtp_{attn}o_proj")
        o_split_k = self._pick_fp8_splitk_factor(w_o)
        if o_split_k is not None and self.world_size == 1:
            self.attn_proj_out = self.x
            self._fp8_linear_splitk(
                self.attn_out, w_o, s_o, self.attn_proj_out,
                split_k=o_split_k, residual=self.x)
        elif o_split_k is not None:
            self.attn_proj_out = self._cached_new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name=f"mtp_{attn}attn_proj_fused",
                io_category="cuda_tensor",
            )
            self._fp8_linear_splitk(
                self.attn_out, w_o, s_o, self.attn_proj_out,
                split_k=o_split_k, residual=self.x)
        else:
            # Per-call output tensor to avoid aliasing self.mtp_x ↔
            # self.attn_proj_out across MTP draft steps.
            self.attn_proj_out = self._cached_new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name=f"mtp_{attn}attn_proj_fused",
                io_category="cuda_tensor",
            )
            self._fp8_linear(self.attn_out, w_o, s_o, self.attn_proj_out,
                             grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                             block_dim=(128, 1, 1),
                             residual=self.x)

    def _build_dense_mlp_with_prefix(self, prefix: str, state_dict: dict):
        """Build dense MLP using a custom weight prefix (FP8, for MTP reuse)."""
        mlp_prefix = f"{prefix}mlp."

        w_gate_up, s_gate_up = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}gate_up_proj.weight",
            f"mtp_{mlp_prefix}gate_up_proj")
        gate_up_split_k = self._pick_fp8_splitk_factor(w_gate_up)
        if gate_up_split_k is not None:
            self._fp8_linear_splitk(
                self.rmsnorm_out, w_gate_up, s_gate_up, self.mlp_mid,
                split_k=gate_up_split_k)
        else:
            self._fp8_linear(self.rmsnorm_out, w_gate_up, s_gate_up, self.mlp_mid,
                             grid_dim=(grid_for_rmsnorm_linear_layer(w_gate_up.dim(0)), 1, 1),
                             block_dim=(128, 1, 1))
        w_down, s_down = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}down_proj.weight",
            f"mtp_{mlp_prefix}down_proj")
        down_split_k = self._pick_fp8_splitk_factor(w_down)
        self._silu_mul_fp8_linear(
            self.mlp_mid,
            self.silu_mul_out,
            w_down,
            s_down,
            self.mlp_out,
            silu_grid_dim=(self.intermediate_size // 64, 1, 1),
            linear_grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
            block_dim=(128, 1, 1),
            use_splitk=(down_split_k is not None),
            splitk_split_k=down_split_k,
        )

    def _build_moe_mlp_with_prefix(self, prefix: str, state_dict: dict):
        """Build MoE MLP using a custom weight prefix (FP8, for MTP reuse)."""
        mbt = self.max_num_batched_tokens

        mlp_prefix = f"{prefix}mlp."

        # Router (BF16 — gate.weight is BF16)
        w_gate = self._cached_attach(
            state_dict[f"{mlp_prefix}gate.weight"],
            f"mtp_{mlp_prefix}gate")
        moe_topk_weights = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK), dtype=float32,
            name="mtp_moe_topk_weights", io_category="cuda_tensor")
        moe_routing_indices = self._cached_new_tensor(
            dims=(self.num_local_experts, mbt), dtype=int32,
            name="mtp_moe_routing_indices", io_category="cuda_tensor")
        moe_mask = self._cached_new_tensor(
            dims=(self.num_local_experts + 1,), dtype=int32,
            name="mtp_moe_mask", io_category="cuda_tensor")
        router_logits = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS), dtype=bfloat16,
            name="mtp_router_logits", io_category="cuda_tensor")
        if self._BF16_GATE_SPLITK_ENABLED:
            # MTP router gate: BF16 splitk swapAB, same shape as the main router.
            mtp_gate_split_k = self._pick_bf16_splitk_factor(w_gate)
            self.mpk.splitk_linear_layer(
                input=self.rmsnorm_out, weight=w_gate, output=router_logits,
                grid_dim=(w_gate.dim(0) // 128, mtp_gate_split_k, 1),
                block_dim=(256, 1, 1),
                accumulate=False,
            )
        else:
            mtp_router_grid = min(grid_for_rmsnorm_linear_layer(w_gate.dim(0)),
                                  w_gate.dim(0) // 8)
            self.mpk.linear_layer(
                input=self.rmsnorm_out, weight=w_gate, output=router_logits,
                grid_dim=(mtp_router_grid, 1, 1),
                block_dim=(128, 1, 1))

        _mtp_moe_io = "nvshmem_tensor" if self._use_nvshmem else "cuda_tensor"
        moe_output = self._cached_new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_moe_output", io_category=_mtp_moe_io)

        w_gate_bias = self._cached_attach(
            state_dict[f"{mlp_prefix}gate.e_score_correction_bias"],
            f"mtp_{mlp_prefix}gate_bias")
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=router_logits, bias=w_gate_bias,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
            local_expert_start=self.local_expert_start)

        # Expert W13 (FP8) — 3D weight (num_experts, 2*intermediate, hidden).
        # Use _safe_attach + manual scale expansion (same as main MoE path); the
        # _attach_fp8_weight helper assumes 2D and would fail to unpack 3D shape.
        w13_key = f"{mlp_prefix}experts.w13.weight"
        w13_scale_key = f"{mlp_prefix}experts.w13.weight_scale_inv"
        w_w13 = self._safe_attach(state_dict[w13_key],
                                  f"mtp_{mlp_prefix}experts_w13")
        if w13_scale_key in state_dict:
            raw_scale_inv = state_dict[w13_scale_key].float().clamp(min=1e-30)
            w13_scale_expanded = raw_scale_inv.repeat_interleave(128, dim=1).contiguous().to(torch.float32)
            s_w13 = self._safe_attach(w13_scale_expanded,
                                      f"mtp_{mlp_prefix}experts_w13_scale")
        else:
            s_w13 = None
        moe_input_fp8 = self._cached_new_tensor(
            dims=(mbt, self.hidden_size), dtype=float8_e4m3,
            name="mtp_moe_input_fp8", io_category="cuda_tensor")
        moe_input_scale = self._cached_new_tensor(
            dims=(mbt, self.hidden_size // 128), dtype=float32,
            name="mtp_moe_input_scale", io_category="cuda_tensor")
        self.mpk.quantize_fp8_layer(
            input=self.rmsnorm_out, output_fp8=moe_input_fp8,
            output_scale=moe_input_scale,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            scale_ue8m0=False)

        moe_mid = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, 2 * self.routed_moe_intermediate_size),
            dtype=bfloat16, name="mtp_moe_mid", io_category="cuda_tensor")
        _mtp_w13_pref = int(os.environ.get("MPK_MOE_W13_M_SPLIT", "16"))
        mtp_w13_m_split = _moe_fp8_m_split(2 * self.routed_moe_intermediate_size,
                                           preferred=_mtp_w13_pref)
        mtp_w13_expert_grid_x = _moe_expert_grid_x(
            mbt, self.num_local_experts, preferred_groups=8)
        self.mpk.moe_w13_fp8_layer(
            input_fp8=moe_input_fp8, input_scale=moe_input_scale,
            weight_fp8=w_w13, weight_scale=s_w13,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_mid,
            grid_dim=(mtp_w13_expert_grid_x, mtp_w13_m_split, 1),
            block_dim=(128, 1, 1))

        w2_key = f"{mlp_prefix}experts.w2.weight"
        w2_scale_key = f"{mlp_prefix}experts.w2.weight_scale_inv"
        w_w2 = self._safe_attach(state_dict[w2_key],
                                 f"mtp_{mlp_prefix}experts_w2")
        if w2_scale_key in state_dict:
            raw_scale_inv = state_dict[w2_scale_key].float().clamp(min=1e-30)
            w2_scale_expanded = raw_scale_inv.repeat_interleave(128, dim=1).contiguous().to(torch.float32)
            s_w2 = self._safe_attach(w2_scale_expanded,
                                     f"mtp_{mlp_prefix}experts_w2_scale")
        else:
            s_w2 = None
        mtp_silu_fp8 = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size),
            dtype=float8_e4m3, name="mtp_moe_silu_fp8", io_category="cuda_tensor")
        mtp_silu_scale = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size // 128),
            dtype=float32, name="mtp_moe_silu_scale", io_category="cuda_tensor")
        moe_silu_out = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size),
            dtype=bfloat16, name="mtp_moe_silu", io_category="cuda_tensor")
        self.mpk.moe_silu_mul_layer(
            input=moe_mid, output=moe_silu_out,
            grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1), block_dim=(128, 1, 1))
        self.mpk.quantize_fp8_layer(
            input=moe_silu_out, output_fp8=mtp_silu_fp8,
            output_scale=mtp_silu_scale,
            grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1), block_dim=(128, 1, 1),
            scale_ue8m0=False)
        moe_down_out = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.hidden_size),
            dtype=bfloat16, name="mtp_moe_down", io_category="cuda_tensor")
        _mtp_w2_pref = int(os.environ.get("MPK_MOE_W2_M_SPLIT", "14"))
        mtp_w2_m_split = _moe_fp8_m_split(self.hidden_size, preferred=_mtp_w2_pref)
        mtp_w2_expert_grid_x = _moe_expert_grid_x(
            mbt, self.num_local_experts, preferred_groups=10)
        self.mpk.moe_w2_fp8_layer(
            input_fp8=mtp_silu_fp8, input_scale=mtp_silu_scale,
            weight_fp8=w_w2, weight_scale=s_w2,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_down_out,
            grid_dim=(mtp_w2_expert_grid_x, mtp_w2_m_split, 1),
            block_dim=(128, 1, 1))

        # Shared expert (FP8) — same pattern as main MoE shared expert:
        # interleave gate+up, requantize for UE8M0, proper silu_mul grid.
        sp = f"{mlp_prefix}shared_experts."
        shared_gate_w = state_dict[f"{sp}gate_proj.weight"]
        shared_up_w = state_dict[f"{sp}up_proj.weight"]
        gate_scale_key = f"{sp}gate_proj.weight_scale_inv"
        has_shared_scale = gate_scale_key in state_dict
        # Interleave gate/up at split granularity (Bug 2+5 fix for MTP)
        from ..utils import shuffle_tensors as _shuffle_tensors
        out_dim_total = shared_gate_w.shape[0] + shared_up_w.shape[0]
        linear_grid = grid_for_rmsnorm_linear_layer(out_dim_total)
        scale_dim_0 = shared_gate_w.shape[0] // 128
        shared_split = min(linear_grid // 2, scale_dim_0)
        while shared_gate_w.shape[0] % shared_split != 0 or scale_dim_0 % shared_split != 0:
            shared_split -= 1
            if shared_split < 1:
                shared_split = 1; break
        fused_key = f"mtp_{sp}gate_up_fused"
        fused_w = _shuffle_tensors([shared_gate_w, shared_up_w], split=shared_split, dim=0)
        if has_shared_scale:
            fused_s = _shuffle_tensors(
                [state_dict[f"{sp}gate_proj.weight_scale_inv"],
                 state_dict[f"{sp}up_proj.weight_scale_inv"]],
                split=shared_split, dim=0)
            state_dict[f"{fused_key}.weight"] = fused_w
            state_dict[f"{fused_key}.weight_scale_inv"] = fused_s
        else:
            state_dict[f"{fused_key}.weight"] = fused_w
        w_s_gu, s_s_gu = self._attach_fp8_weight(
            state_dict, f"{fused_key}.weight", f"mtp_{sp}gate_up")
        shared_mid = self._cached_new_tensor(
            dims=(mbt, 2 * self.moe_intermediate_size), dtype=bfloat16,
            name="mtp_shared_mid", io_category="cuda_tensor")
        mtp_shared_gu_split_k = self._pick_fp8_splitk_factor(w_s_gu)
        if mtp_shared_gu_split_k is not None:
            self._fp8_linear_splitk(
                self.rmsnorm_out, w_s_gu, s_s_gu, shared_mid,
                split_k=mtp_shared_gu_split_k)
        else:
            gate_up_grid = grid_for_rmsnorm_linear_layer(out_dim_total)
            self._fp8_linear(self.rmsnorm_out, w_s_gu, s_s_gu, shared_mid,
                             grid_dim=(gate_up_grid, 1, 1),
                             block_dim=(128, 1, 1))
        shared_silu = self._cached_new_tensor(
            dims=(mbt, self.moe_intermediate_size), dtype=bfloat16,
            name="mtp_shared_silu", io_category="cuda_tensor")
        w_s_down, s_s_down = self._attach_fp8_weight(
            state_dict, f"{sp}down_proj.weight", f"mtp_{sp}down_proj")
        shared_residual = self._cached_new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_shared_residual", io_category="cuda_tensor")
        # MoE internal residual always OFF (handled by external elementwise_add)
        _mtp_resid = None
        self._silu_mul_fp8_linear(
            shared_mid,
            shared_silu,
            w_s_down,
            s_s_down,
            shared_residual,
            silu_grid_dim=(shared_split, 1, 1),
            linear_grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
            residual=_mtp_resid,
        )

        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out, weight=moe_topk_weights,
            residual=shared_residual, output=moe_output,
            grid_dim=(mbt, _moe_hidden_split(self.hidden_size), 1),
            block_dim=(128, 1, 1))
        self.mlp_out = moe_output

    def _build_mtp_layer(self, state_dict: dict):
        """Build MTP predictor layer.

        Architecture (from vLLM's DeepSeekMultiTokenPredictorLayer):
        1. embed(draft_token) → enorm
        2. hnorm(previous_hidden_states)
        3. eh_proj(cat[enorm_out, hnorm_out]) → via split: W1@e + W2@h
        4. Full decoder layer (MLA attention + dense MLP)
        5. Shared LM head → draft logits → argmax → draft_token_ids[step]

        Draft steps are statically unrolled at compile time.
        MTP layer weights recycle via modulo: step_idx % num_mtp_layers.
        """
        if self.mtp_config is None:
            return

        from ...speculative import LookaheadConfig
        if not isinstance(self.mtp_config, LookaheadConfig):
            return

        num_draft_steps = self.mtp_config.spec_length
        # Checkpoint stores MTP layer at model.layers.{num_hidden_layers}
        # (e.g., model.layers.61 for DeepSeek V3 with 61 main layers)
        mtp_layer_idx = self.num_layers  # 61
        mtp_prefix = f"model.layers.{mtp_layer_idx}."
        # The transformer block weights use the same prefix (no mtp_block sub-prefix)
        mtp_block_prefix = mtp_prefix

        # ---- Shared weights ----
        # embed_tokens and lm_head are shared with main model (already attached)

        # MTP-specific weights: enorm, hnorm, eh_proj
        w_enorm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mtp_prefix}enorm.weight"],
            name="mtp_enorm_weight",
        )
        w_hnorm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mtp_prefix}hnorm.weight"],
            name="mtp_hnorm_weight",
        )

        # eh_proj: [hidden_size, 2*hidden_size] → split into W1 (embed) + W2 (hidden)
        # IMPORTANT: .contiguous() creates new tensors — must keep references alive
        # so the GPU memory is not freed and reused by later allocations (the
        # persistent kernel stores raw data pointers, not PyTorch tensor refs).
        eh_proj_full = state_dict[f"{mtp_prefix}eh_proj.weight"]
        self._mtp_eh_proj_embed_tensor = eh_proj_full[:, :self.hidden_size].contiguous()
        self._mtp_eh_proj_hidden_tensor = eh_proj_full[:, self.hidden_size:].contiguous()
        w_eh_proj_1 = self.mpk.attach_input(
            torch_tensor=self._mtp_eh_proj_embed_tensor,
            name="mtp_eh_proj_embed",
        )
        w_eh_proj_2 = self.mpk.attach_input(
            torch_tensor=self._mtp_eh_proj_hidden_tensor,
            name="mtp_eh_proj_hidden",
        )

        # ---- MTP KV cache (separate from main model) ----
        # IMPORTANT: keep the PyTorch tensor alive on self so GPU memory is not
        # freed — the persistent kernel stores the raw data pointer.
        self._mtp_ckv_kpe_cache_buf = torch.zeros(
            (self.mpk.max_num_pages, self.mpk.page_size, self.qk_head_dim),
            dtype=torch.bfloat16, device="cuda",
        )
        self.mtp_ckv_kpe_cache_tensor = self.mpk.attach_input(
            torch_tensor=self._mtp_ckv_kpe_cache_buf,
            name="mtp_ckv_kpe_cache",
        )

        # ---- Intermediate tensors ----
        mbt = self.max_num_batched_tokens
        mtp_embed_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_embed_out", io_category="cuda_tensor",
        )
        mtp_enorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_enorm_out", io_category="cuda_tensor",
        )
        mtp_hnorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_hnorm_out", io_category="cuda_tensor",
        )
        mtp_proj_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_proj_out", io_category="cuda_tensor",
        )

        # Draft token ID buffers
        draft_token_ids = self.mpk.new_tensor(
            dims=(mbt, 1), dtype=int64,
            name="mtp_draft_token_ids", io_category="cuda_tensor",
        )

        # Collect all draft token IDs for verification
        all_draft_ids = self.mpk.new_tensor(
            dims=(mbt, num_draft_steps), dtype=int64,
            name="mtp_all_draft_ids", io_category="cuda_tensor",
        )

        # vLLM-aligned MTP embedding input buffer (Task #29). At step 0 MTP
        # should embed shifted ground-truth prompt tokens during prefill (not
        # main's argmax which is only accurate for a fully-trained model).
        # `mtp_build_embed_input_layer` populates this per iteration:
        #   mtp_input_tokens[i] = tokens[step+i+1]  for i < mbt-1 (ground truth)
        #                       = output_tokens[i]  for i == mbt-1 (current argmax)
        # Matches vLLM/v1/spec_decode/eagle.py L666-669 behavior.
        mtp_step0_input_tokens = self.mpk.new_tensor(
            dims=(mbt, 1), dtype=int64,
            name="mtp_step0_input_tokens", io_category="cuda_tensor",
        )

        # ---- Shared embed weight reference (saved during build_from_dict) ----
        w_embed = self.w_embed

        # ---- Save main model state ----
        main_hidden_states = self.x  # After all 61 layers + final norm

        # Verification method: needed early (draft loop uses it for prob computation)
        method = getattr(self.mtp_config, 'rejection_sample_method', 'strict')

        # Build the MTP step-0 input tokens buffer ONCE per MPK iteration, before
        # the draft loop. Reads main's argmax (output_tokens) via task_desc input;
        # reads tokens + step from runtime_config internally.
        self.mpk.mtp_build_embed_input_layer(
            output_tokens=self.argmax_out_dtensor,
            mtp_input_tokens=mtp_step0_input_tokens,
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
            batch_size=mbt,
            max_seq_len=self.mpk.max_seq_length,
        )

        # ---- Draft generation loop (statically unrolled) ----
        for step in range(num_draft_steps):
            # 1. Get draft token: step 0 uses the vLLM-aligned shifted tokens
            # (ground-truth prompt during prefill, main argmax during decode
            # via the prep task above). step 1+ uses the previous MTP iter's
            # argmax draft_token_ids (standard autoregressive draft chain).
            draft_input = mtp_step0_input_tokens if step == 0 else draft_token_ids

            # 2. Embed draft token (shared embed_tokens weight)
            self.mpk.embed_layer(
                input=draft_input, weight=w_embed, output=mtp_embed_out,
                grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1,
            )

            # 3. enorm(embed_out)
            self.mpk.rmsnorm_layer(
                input=mtp_embed_out, weight=w_enorm, output=mtp_enorm_out,
                grid_dim=_rmsnorm_grid(mbt), block_dim=(128, 1, 1),
            )

            # 4. hnorm(previous_hidden_states)
            hidden_input = main_hidden_states if step == 0 else self.mtp_x
            self.mpk.rmsnorm_layer(
                input=hidden_input, weight=w_hnorm, output=mtp_hnorm_out,
                grid_dim=_rmsnorm_grid(mbt), block_dim=(128, 1, 1),
            )

            # 5. eh_proj: output = W1 @ enorm_out + W2 @ hnorm_out
            self.mpk.linear_layer(
                input=mtp_enorm_out, weight=w_eh_proj_1, output=mtp_proj_out,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_eh_proj_1.dim(0)), 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.linear_with_residual_layer(
                input=mtp_hnorm_out, weight=w_eh_proj_2,
                residual=mtp_proj_out, output=mtp_proj_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )

            # 6. Full MTP decoder layer (MLA attention + MLP, own weights)
            self.mtp_x = mtp_proj_out
            self._build_mtp_decoder_layer(state_dict, mtp_block_prefix)

            # 7. Final norm → shared lm_head → argmax → draft_token_ids
            # shared_head.norm is the MTP's output norm
            # Checkpoint key: model.layers.61.shared_head.norm.weight
            w_mtp_norm = self.mpk.attach_input(
                torch_tensor=state_dict.get(
                    f"{mtp_prefix}shared_head.norm.weight",
                    state_dict["model.norm.weight"],  # fallback to main model norm
                ),
                name=f"mtp_step{step}_norm",
            )
            self.mpk.rmsnorm_layer(
                input=self.mtp_x, weight=w_mtp_norm, output=self.rmsnorm_out,
                grid_dim=_rmsnorm_grid(mbt), block_dim=(128, 1, 1),
            )

            # Shared lm_head (saved during build_from_dict)
            w_lm_head = self.w_lm_head
            padded_vocab_size = 129280
            lm_head_out = self.mpk.new_tensor(
                dims=(mbt, padded_vocab_size), dtype=bfloat16,
                name=f"mtp_step{step}_logits", io_category="cuda_tensor",
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out, weight=w_lm_head, output=lm_head_out,
                grid_dim=(grid_for_rmsnorm_linear_layer(padded_vocab_size), 1, 1),
                block_dim=(128, 1, 1),
            )

            # Argmax → draft_token_ids
            # Use the same grid size (num_workers) as the main model's argmax
            # to properly fill all entries of the shared argmax_part_value/index
            # buffers. Using grid=(mbt,1,1) only writes 1 of num_workers entries,
            # leaving stale values that argmax_reduce reads and may select.
            _argmax_grid = self.mpk.num_workers
            self.mpk.argmax_partial_layer(
                input=lm_head_out,
                output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=(_argmax_grid, 1, 1), block_dim=(128, 1, 1),
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=draft_token_ids,
                grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            )

            # Probabilistic: compute P_draft(draft_token) from this step's logits
            if method == "probabilistic":
                draft_prob_current = self._cached_new_tensor(
                    dims=(mbt, 1), dtype=float32,
                    name="mtp_draft_prob_current")
                self.mpk.softmax_gather_layer(
                    logits=lm_head_out, token_ids=draft_token_ids,
                    output_probs=draft_prob_current,
                    grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
                if not hasattr(self, '_draft_prob_buffer'):
                    self._draft_prob_buffer = self._cached_new_tensor(
                        dims=(mbt, num_draft_steps), dtype=float32,
                        name="mtp_draft_prob_buffer")
                # Scatter to buffer[batch, step] with compile-time slot index
                self.mpk.mtp_float_scatter_layer(
                    src=draft_prob_current, dst=self._draft_prob_buffer,
                    grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
                    batch_size=mbt, num_slots=num_draft_steps, slot_idx=step)

            # Scatter this step's draft token into the collection buffer
            self.mpk.mtp_token_scatter_layer(
                src=draft_token_ids,
                dst=all_draft_ids,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
                batch_size=mbt,
                num_slots=num_draft_steps,
                slot_idx=step,
            )

        # ---- Prepare verify: write draft tokens to sequence buffer ----
        # This sets up input for the next iteration's verification forward:
        # tokens[request, step+1] = main_token, tokens[request, step+2..K+1] = drafts
        # Note: these meta tensors must be attached as DTensors for the task graph
        tokens_buf_raw = self.mpk.meta_tensors.get("tokens", None)
        step_raw = self.mpk.meta_tensors.get("step", None)
        num_new_raw = self.mpk.meta_tensors.get("num_new_tokens", None)

        if tokens_buf_raw is not None and step_raw is not None:
            d_tokens_buf = self.mpk.attach_input(
                torch_tensor=tokens_buf_raw, name="mtp_tokens_buffer")
            d_step = self.mpk.attach_input(
                torch_tensor=step_raw, name="mtp_step")
            d_num_new = self.mpk.attach_input(
                torch_tensor=num_new_raw, name="mtp_num_new_tokens")
            self.mpk.mtp_prepare_verify_layer(
                main_token=self.argmax_out_dtensor,
                draft_tokens=all_draft_ids,
                tokens_buffer=d_tokens_buf,
                step=d_step,
                num_new_tokens=d_num_new,
                grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=num_draft_steps,
                max_seq_len=self.mpk.max_seq_length,
            )

        # ---- Verification + Accept/Commit ----
        # The "target token IDs" for verifying drafts D_1..D_K placed at
        # positions [step+2..step+K+1] are the main model's argmax at
        # input positions [step+1..step+K+1] (i.e., predicted tokens at
        # positions [step+2..step+K+2]). The main model already computed
        # those argmax values into `self.argmax_out_dtensor`, which is
        # bound to `self.output_tokens` of shape (mbt, 1) int64. The
        # verify kernel reads `target_ids[i]` linearly for i in 0..K, so
        # the first K+1 int64 entries of `output_tokens` are exactly what
        # we need. Aliasing avoids a redundant copy task and the previous
        # dead-buffer bug where `mtp_target_token_ids` was allocated
        # fresh and never written, making the strict-verify kernel
        # compare drafts against zeros (always reject for any non-zero
        # draft). [2026-05-11 fix]
        target_token_ids = self.argmax_out_dtensor
        accepted_count = self.mpk.new_tensor(
            dims=(mbt, 1), dtype=int64,
            name="mtp_accepted_count", io_category="cuda_tensor",
        )
        verified_output_tokens = self.mpk.new_tensor(
            dims=(mbt, num_draft_steps + 1), dtype=int64,
            name="mtp_verified_output", io_category="cuda_tensor",
        )

        # Select verification method (default to strict for lookahead)
        method = getattr(self.mtp_config, 'rejection_sample_method', 'strict')
        if method == "strict":
            self.mpk.mtp_verify_strict_layer(
                draft_token_ids=all_draft_ids,
                target_token_ids=target_token_ids,
                accepted_count=accepted_count,
                output_tokens=verified_output_tokens,
                grid_dim=(mbt, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=num_draft_steps,
            )
        elif method == "probabilistic":
            # Probabilistic rejection sampling: P_target(x) > u * P_draft(x)
            #
            # Target probs: accumulated by softmax_gather + prob_scatter in the main
            # model graph during verify forward pass (inserted in build_from_dict).
            # Each verify iteration writes P_target(input_token) to target_prob_buffer[step].
            #
            # Draft probs: computed here from the per-step draft logits stored during
            # draft generation.

            # Extract draft probs from per-step logits via softmax_gather
            draft_probs = self._cached_new_tensor(
                dims=(mbt, num_draft_steps), dtype=float32,
                name="mtp_draft_probs")
            for step_idx in range(num_draft_steps):
                # Draft logits for this step are in mtp_step{step_idx}_logits
                # The draft token for this step is all_draft_ids[:, step_idx].
                # Per-column gather is not wired here; probabilistic verify
                # consumes self._draft_prob_buffer below.
                pass

            # For now: use target_prob_buffer from main graph + dummy draft_probs
            rng_seed = self._cached_new_tensor(
                dims=(mbt, 1), dtype=uint64,
                name="mtp_rng_seed")

            # Extract target probs from the accumulation buffer
            # target_prob_buffer[batch, pos] has P_target(input_token) at each position.
            # The verify positions are step+1..step+K+1 (set by prepare_verify).
            # We need target_probs[0..K-1] = target_prob_buffer[step+1..step+K].

            # Extract target probs from accumulation buffer at verify positions
            target_probs = self._cached_new_tensor(
                dims=(mbt, num_draft_steps), dtype=float32,
                name="mtp_target_probs_extracted")
            step_tensor = self.mpk.meta_tensors.get("step", None)
            if step_tensor is not None and hasattr(self, '_target_prob_buffer'):
                self.mpk.prob_extract_layer(
                    buffer=self._target_prob_buffer,
                    offset=self._cached_attach(step_tensor, "step_for_prob_extract"),
                    output=target_probs,
                    grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
                    max_positions=self.mpk.max_seq_length,
                    num_extract=num_draft_steps)

            # RNG seed for rejection sampling
            rng_seed = self._cached_new_tensor(
                dims=(mbt, 1), dtype=uint64,
                name="mtp_rng_seed")

            # Probabilistic verify
            self.mpk.mtp_verify_probabilistic_layer(
                draft_token_ids=all_draft_ids,
                target_token_ids=target_token_ids,
                target_probs=target_probs,
                draft_probs=self._draft_prob_buffer,
                seed=rng_seed,
                accepted_count=accepted_count,
                output_tokens=verified_output_tokens,
                grid_dim=(mbt, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=num_draft_steps,
            )

        # Accept/commit: update position and output final tokens
        step_raw = self.mpk.meta_tensors.get("step", None)
        if step_raw is not None:
            current_position = self.mpk.attach_input(
                torch_tensor=step_raw, name="mtp_accept_step")
            new_position = self.mpk.new_tensor(
                dims=(mbt, 1), dtype=int64,
                name="mtp_new_position", io_category="cuda_tensor",
            )
            final_output = self.mpk.new_tensor(
                dims=(mbt, num_draft_steps + 1), dtype=int64,
                name="mtp_final_output", io_category="cuda_tensor",
            )
            # accept_commit writes `accepted_count` (which already
            # includes the bonus) into the runtime-visible meta tensor
            # `meta_tensors["num_new_tokens"]` (attached above as
            # `d_num_new`). This overwrites the optimistic K+1 that
            # prepare_verify wrote earlier in the iteration, so the
            # scheduler's prepare_next_batch advances `step` by only
            # the accepted positions. Previously this wrote to a fresh
            # `mpk.new_tensor` builder-local buffer that nothing
            # downstream reads, making the scheduler always advance
            # K+1 regardless of accept/reject. Implicitly fixes
            # KV-cache rollback: the next iteration's main forward
            # then starts at `step+accepted_count+1` and writes K+1
            # fresh K/V entries, overwriting any stale K/V at rejected
            # positions. [2026-05-11 fix]
            self.mpk.mtp_accept_commit_layer(
                accepted_count=accepted_count,
                output_tokens=verified_output_tokens,
                current_position=current_position,
                new_position=new_position,
                final_output=final_output,
                num_new_tokens=d_num_new,
                grid_dim=(mbt, 1, 1),
                block_dim=(128, 1, 1),
                num_draft_tokens=num_draft_steps,
            )

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
            # B37: when env-gated, fuse the input-layernorm RMSNorm with
            # the downstream qkv_a FP8 quantize. The fused task writes the
            # BF16 normalized output AND the FP8 + scale buffers in one
            # pass, so the qkv_a `_fp8_linear` call can skip its internal
            # quantize via share_quantize_tag.
            if self._fused_rmsnorm_quantize:
                self._emit_fused_rmsnorm_qkv_a_quantize(
                    input_x=self.x,
                    w_norm=w_norm,
                    layer_idx=i,
                    reduction_size=self.hidden_size,
                )
            else:
                self.mpk.rmsnorm_layer(
                    input=self.x, weight=w_norm, output=self.rmsnorm_out,
                    grid_dim=_rmsnorm_grid(self.max_num_batched_tokens),
                    block_dim=(128, 1, 1),
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
            # C17 (2026-05-17): when env-gated, fuse post-attn rmsnorm + NEW
            # MoE input quantize into one task. The fused task writes a
            # per-layer-unique rmsnorm_out + per-layer-unique moe_input_fp8 +
            # moe_input_scale; we rebind self.rmsnorm_out so all downstream
            # consumers in this layer (router linear, shared_expert, fp8
            # MoE permute via _build_moe_mlp) read the per-layer buffer.
            # Gating constraints:
            #   * layer must be MoE (i >= FIRST_MOE_LAYER)
            #   * use_fp8_experts must be true (NEW MoE path)
            #   * NEW_MOE env must be on (otherwise OLD path quantize_fp8
            #     uses float32 scale, not UE8M0 — fused kernel writes UE8M0
            #     only)
            _post_attn_fuse_eligible = (
                self._fused_post_attn_rmsnorm_quantize
                and i >= FIRST_MOE_LAYER
                and self._new_moe
                and self._fp8_experts_available(state_dict, i))
            if _post_attn_fuse_eligible:
                rmsnorm_out_bf16, _, _ = (
                    self._emit_fused_post_attn_rmsnorm_moe_quantize(
                        input_x=self.x, w_norm=w_post_norm,
                        layer_idx=i,
                        reduction_size=self.hidden_size,
                    ))
                self.rmsnorm_out = rmsnorm_out_bf16
            else:
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
            if vocab_parallel_lm_head and self.mtp_config is not None:
                vocab_parallel_lm_head = False

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

            # Probabilistic MTP: insert softmax_gather + prob_scatter before argmax.
            # This accumulates P_target(input_token) at each iteration's position.
            _prob_method = getattr(self, 'mtp_config', None)
            _use_prob_mtp = (_prob_method is not None and
                             getattr(_prob_method, 'rejection_sample_method', 'strict') == 'probabilistic')
            if _use_prob_mtp:
                mbt = self.max_num_batched_tokens
                # Buffer: accumulate probs across iterations [mbt, max_seq]
                self._target_prob_buffer = self.mpk.new_tensor(
                    dims=(mbt, self.mpk.max_seq_length), dtype=float32,
                    name="target_prob_buffer", io_category="cuda_tensor")
                # Per-iteration prob scratch [mbt, 1]
                self._target_prob_current = self.mpk.new_tensor(
                    dims=(mbt, 1), dtype=float32,
                    name="target_prob_current", io_category="cuda_tensor")
                # softmax_gather: lm_head_out + input_tokens → prob_current
                self.mpk.softmax_gather_layer(
                    logits=lm_head_out,
                    token_ids=self.mpk.attach_input(
                        torch_tensor=self.input_tokens, name="input_tokens_for_prob"),
                    output_probs=self._target_prob_current,
                    grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
                # prob_scatter: write prob_current to buffer[step_position]
                step_tensor = self.mpk.meta_tensors.get("step", None)
                if step_tensor is not None:
                    self.mpk.prob_scatter_layer(
                        prob=self._target_prob_current,
                        step_counter=self.mpk.attach_input(
                            torch_tensor=step_tensor, name="step_for_prob_scatter"),
                        buffer=self._target_prob_buffer,
                        grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
                        max_positions=self.mpk.max_seq_length)

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
                self.mpk.nvshmem_global_argmax_layer(
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

        # Optional MTP layer
        self._build_mtp_layer(state_dict)
