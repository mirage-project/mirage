"""DeepSeek V3 model builder for Mirage MPK with MTP support.

Architecture: 61 decoder layers with MLA attention and MoE MLP.
- Layers 0-2: Dense MLP (DeepseekV2MLP)
- Layers 3-60: MoE MLP (256 experts, top-8, + shared experts)
- MLA: 128 Q heads, 1 KV head after weight absorption, head_dim=576 (512+64)
- Optional MTP: 1 predictor layer for multi-token prediction

Weight absorption: at load time, kv_b_proj is absorbed into q_b_proj so that
runtime only needs compressed KV cache [c_latent(512), k_pe(64)] = 576 dims.
"""

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

        # MTP config
        self.mtp_config = getattr(mpk, 'spec_decode_config', None)

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

        self.build_from_dict(
            model_config.state_dict,
            model_config.with_lm_head,
            layer_indices=layer_indices,
        )

    def _fp8_linear(self, input_bf16, weight, weight_scale, output,
                     grid_dim, block_dim, residual=None):
        """Quantize BF16 input → FP8, then run FP8 GEMM."""
        if weight_scale is None:
            # BF16 fallback is kept for fixtures or pre-converted weights that
            # intentionally arrive without FP8 scale metadata.
            if residual is not None:
                self.mpk.linear_with_residual_layer(
                    input=input_bf16, weight=weight, residual=residual,
                    output=output, grid_dim=grid_dim, block_dim=block_dim)
            else:
                self.mpk.linear_layer(
                    input=input_bf16, weight=weight, output=output,
                    grid_dim=grid_dim, block_dim=block_dim)
            return

        if input_bf16.num_dims != 2 or output.num_dims != 2:
            raise ValueError("FP8 linear expects 2D input and output tensors.")
        if weight.num_dims != 2:
            raise ValueError("FP8 linear expects a 2D weight tensor.")
        if weight_scale.num_dims != 2:
            raise ValueError("FP8 linear expects a 2D packed UE8M0 scale tensor.")

        # New FP8 kernel: each CTA processes output_size=128. Grid splits output.
        output_size = weight.dim(0)
        max_grid = output_size // 128
        if max_grid < 1:
            raise ValueError(
                f"FP8 linear: output_size={output_size} < 128 (BLOCK_N). "
                f"Not supported yet.")
        # Each FP8 linear task serializes all of the output tiles in its shard.
        # Large layers were still using the old 64/96-task heuristic, which
        # leaves many workers idle on B200. Keep the caller's grid for small
        # layers, but raise the task count for large layers so we at least fill
        # the available worker pool without oversharding past the kernel's tile
        # granularity (one 128-row output shard per task).
        target_grid_x = min(max_grid, self.num_workers)
        grid_dim = (min(max_grid, max(grid_dim[0], target_grid_x)),
                    grid_dim[1],
                    grid_dim[2])

        mbt = self.max_num_batched_tokens
        reduction_size = weight.dim(1) if weight.num_dims == 2 else weight.dim(-1)
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size

        # Share FP8 quantization buffer by reduction_size across layers.
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
            # physical shape=[packed_k, aligned_batch], dtype=uint32
            packed_k = (num_groups + 3) // 4
            aligned_batch = ((mbt + 3) // 4) * 4
            # The CUTLASS FP8 kernel reads scales by packed-K first and aligned
            # batch second, matching the UE8M0 column-major descriptor layout.
            scale_buf = self.mpk.new_tensor(
                dims=(packed_k, aligned_batch), dtype=uint32,
                name=f"fp8_scale_{reduction_size}_shared",
                io_category="cuda_tensor",
            )
            self._fp8_bufs[cache_key] = (fp8_buf, scale_buf)
        self._fp8_input_buf, self._fp8_scale_buf = self._fp8_bufs[cache_key]

        self.mpk.quantize_fp8_layer(
            input=input_bf16,
            output_fp8=self._fp8_input_buf,
            output_scale=self._fp8_scale_buf,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
        )

        if residual is not None:
            self.mpk.linear_fp8_with_residual_layer(
                input_fp8=self._fp8_input_buf,
                input_scale=self._fp8_scale_buf,
                weight_fp8=weight,
                weight_scale=weight_scale,
                residual=residual,
                output=output,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            self.mpk.linear_fp8_layer(
                input_fp8=self._fp8_input_buf,
                input_scale=self._fp8_scale_buf,
                weight_fp8=weight,
                weight_scale=weight_scale,
                output=output,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

    def _precompute_rope_embeddings(self):
        """Precompute cos/sin RoPE embeddings for DeepSeek V3."""
        rope_dim = QK_ROPE_HEAD_DIM  # 64
        max_seq = self.mpk.max_seq_length
        # DeepSeek V3 uses standard RoPE with theta=10000
        theta = 10000.0
        half = rope_dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        positions = torch.arange(max_seq, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # [max_seq, half]
        # Expand to full rope_dim: [max_seq, rope_dim] = [cos_half, cos_half]
        # Keep PyTorch tensors alive on self — the persistent kernel stores
        # raw GPU pointers, so the tensors must not be garbage-collected.
        self._rope_cos_buf = torch.cat([angles.cos(), angles.cos()], dim=-1).to(
            dtype=torch.bfloat16, device="cuda")
        self._rope_sin_buf = torch.cat([-angles.sin(), angles.sin()], dim=-1).to(
            dtype=torch.bfloat16, device="cuda")
        # Attach as DTensors
        self.cos_pos_embed = self.mpk.attach_input(
            torch_tensor=self._rope_cos_buf, name="rope_cos")
        self.sin_pos_embed = self.mpk.attach_input(
            torch_tensor=self._rope_sin_buf, name="rope_sin")

    def _new_intermediate_tensors(self):
        """Allocate intermediate computation buffers."""
        mbt = self.max_num_batched_tokens

        # Runtime Q_LEN decides the MLA algorithm: Q_LEN <= 8 is decode/MTP
        # verify, Q_LEN >= 9 is prefill. MBT caps the maximum prefill chunk
        # size, so any graph with mbt > 8 must include the prefill path.
        self._use_prefill = mbt > 8
        if self._use_prefill:
            print(f"  [MLA path] Q_LEN={mbt} → mla_prefill_sm100 (chunked prefill)")
        else:
            print(f"  [MLA path] Q_LEN={mbt} → MLA decode / MTP decode")

        # RMSNorm output
        self.rmsnorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )

        # MLA projections
        # q_a output: [batch, q_lora_rank]
        self.q_a_out_buf = None
        self.q_a_out = self.mpk.new_tensor(
            dims=(mbt, self.q_lora_rank),
            dtype=bfloat16,
            name="q_a_out",
            io_category="cuda_tensor",
        )
        # q_b output (after absorption): [batch, num_local_q_heads * qk_head_dim]
        self.q_nope_pe_buf = None
        self.q_nope_pe = self.mpk.new_tensor(
            dims=(mbt, self.num_local_q_heads * self.qk_head_dim),
            dtype=bfloat16,
            name="q_nope_pe",
            io_category="cuda_tensor",
        )
        # Prefill-path Q tensors: split per-head [nope(512) | pe(64)]. For the
        # mla_prefill_sm100 kernel which expects them as two separate dense
        # tensors (shape [S, H, D_CKV] and [S, H, D_KPE]).
        if self._use_prefill:
            self.q_nope = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads * self.kv_lora_rank),
                dtype=bfloat16, name="q_nope", io_category="cuda_tensor",
            )
            self.q_pe = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads * QK_ROPE_HEAD_DIM),
                dtype=bfloat16, name="q_pe", io_category="cuda_tensor",
            )
        else:
            self.q_nope = None
            self.q_pe = None
        # kv_a output split: c_latent [batch, 512] and k_pe [batch, 64]
        # We use two separate linear layers instead of one 576-dim output,
        # so we can apply kv_a_layernorm to c_latent only.
        self.c_latent_out_buf = None
        self.c_latent_out = self.mpk.new_tensor(
            dims=(mbt, self.kv_lora_rank),  # [batch, 512]
            dtype=bfloat16,
            name="c_latent_out",
            io_category="cuda_tensor",
        )
        # Pad K-PE from 64 to 128 rows so the FP8 linear and downstream KV
        # copy use the SM100 128-row tile shape. Real data remains in [0:64].
        self.k_pe_out = self.mpk.new_tensor(
            dims=(mbt, 128),  # [batch, 128] — padded from 64
            dtype=bfloat16,
            name="k_pe_out",
            io_category="cuda_tensor",
        )
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
        # Prefill-path KV: split into CKV [B, S, 512] and KPE [B, S, 64]. The
        # prefill kernel itself is single-request; task registration offsets
        # each request to its own [S, *] window inside this flattened buffer.
        if self._use_prefill:
            self.ckv_sep = self.mpk.new_tensor(
                dims=(self.mpk.max_num_batched_requests * self.mpk.max_seq_length,
                      self.kv_lora_rank),
                dtype=bfloat16, name="ckv_sep", io_category="cuda_tensor",
            )
            self.kpe_sep = self.mpk.new_tensor(
                dims=(self.mpk.max_num_batched_requests * self.mpk.max_seq_length,
                      QK_ROPE_HEAD_DIM),
                dtype=bfloat16, name="kpe_sep", io_category="cuda_tensor",
            )
        else:
            self.ckv_sep = None
            self.kpe_sep = None
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
        """Attach FP8 weight + scale_inv (converted to UE8M0), or BF16 weight."""
        scale_key = f"{key}_scale_inv"
        if scale_key in state_dict:
            # Requantize: dequant with float32 scale, re-quantize with UE8M0 scale
            if state_dict[key].dtype != torch.float8_e4m3fn:
                raise TypeError(f"{key} must be torch.float8_e4m3fn when {scale_key} exists.")
            if state_dict[scale_key].dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError(f"{scale_key} must be a floating scale tensor.")
            new_fp8, packed_ue8m0 = self._requantize_fp8_for_ue8m0(
                state_dict[key], state_dict[scale_key])
            w = self._safe_attach(new_fp8, name)
            s = self._safe_attach(packed_ue8m0, f"{name}_scale")
        else:
            # BF16 fallback is used by reduced fixtures or explicitly
            # pre-converted weights that have no scale tensor.
            if state_dict[key].dtype != torch.bfloat16:
                raise TypeError(f"{key} without scale must be torch.bfloat16.")
            w = self._safe_attach(state_dict[key], name)
            s = None  # weight is already BF16 (post-dequant)
        return w, s

    def _build_mla_attention_layer(self, layer_idx: int, state_dict: dict):
        """Build MLA attention for one decoder layer (FP8 weights)."""
        prefix = f"model.layers.{layer_idx}."
        attn = f"{prefix}self_attn."

        # Step 1: q_a_proj (FP8)
        w_q_a, s_q_a = self._attach_fp8_weight(
            state_dict, f"{attn}q_a_proj.weight", f"layer_{layer_idx}_q_a_proj")
        self._fp8_linear(self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.q_lora_rank), 1, 1),
                         block_dim=(128, 1, 1))

        # Step 2: q_a_layernorm (BF16 norm weight)
        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}q_a_layernorm.weight"],
            name=f"layer_{layer_idx}_q_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # Step 3: q_b_proj absorbed (BF16 — scale deleted after absorption)
        # Dual-dispatch (opt/mla-dual-dispatch):
        #  - _use_prefill = False (decode/MTP): one linear, fused [H*576]
        #    output -> self.q_nope_pe, consumed by the MLA decode kernel.
        #  - _use_prefill = True: BOTH forms produced — the decode kernel
        #    consumes q_nope_pe [H*576] and the prefill kernel consumes
        #    q_nope [H*512] + q_pe [H*64]. The builder's dual-dispatch
        #    registers both attention kernels; at runtime one of them
        #    early-exits based on Q_LEN, but both Q forms must be present.
        #    Split weights (q_b_nope, q_b_pe) are produced at load time in
        #    demo.py's Phase 2 absorption; the fused q_b_proj weight is
        #    retained alongside.
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight",
            f"layer_{layer_idx}_q_b_proj")
        self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        if self._use_prefill:
            w_q_b_nope, s_q_b_nope = self._attach_fp8_weight(
                state_dict, f"{attn}q_b_nope.weight",
                f"layer_{layer_idx}_q_b_nope")
            self._fp8_linear(
                self.q_a_out, w_q_b_nope, s_q_b_nope, self.q_nope,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_nope.dim(0)), 1, 1),
                block_dim=(128, 1, 1))
            w_q_b_pe, s_q_b_pe = self._attach_fp8_weight(
                state_dict, f"{attn}q_b_pe.weight",
                f"layer_{layer_idx}_q_b_pe")
            self._fp8_linear(
                self.q_a_out, w_q_b_pe, s_q_b_pe, self.q_pe,
                grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b_pe.dim(0)), 1, 1),
                block_dim=(128, 1, 1))

        # Step 4: kv_a_proj split — c_latent (FP8) + k_pe (BF16 padded)
        # k_pe output=64 < MMA_M=128, so dequant to BF16 and pad weight to [128, H]
        kv_a_w = state_dict[f"{attn}kv_a_proj_with_mqa.weight"]
        kv_a_s_key = f"{attn}kv_a_proj_with_mqa.weight_scale_inv"
        has_kv_scale = kv_a_s_key in state_dict

        if has_kv_scale:
            kv_a_s = state_dict[kv_a_s_key]
            scale_rows_total = kv_a_s.shape[0]
            latent_ratio = self.kv_lora_rank / (self.kv_lora_rank + QK_ROPE_HEAD_DIM)
            scale_rows_latent = round(scale_rows_total * latent_ratio)
            # c_latent: requantize [512, hidden] FP8
            latent_fp8, latent_ue8m0 = self._requantize_fp8_for_ue8m0(
                kv_a_w[:self.kv_lora_rank].contiguous(),
                kv_a_s[:scale_rows_latent].contiguous())
            w_kv_latent = self._safe_attach(latent_fp8,
                f"layer_{layer_idx}_kv_a_latent")
            s_kv_latent = self._safe_attach(latent_ue8m0,
                f"layer_{layer_idx}_kv_a_latent_scale")
            # k_pe: requantize [64, hidden] → pad to [128, hidden]
            rope_fp8_raw = kv_a_w[self.kv_lora_rank:].contiguous()
            rope_scale_raw = kv_a_s[scale_rows_latent:].contiguous()
            # Pad FP8 weight from [64, H] to [128, H] BEFORE requantize
            rope_fp8_padded = torch.zeros(128, rope_fp8_raw.shape[1],
                                          dtype=rope_fp8_raw.dtype, device=rope_fp8_raw.device)
            rope_fp8_padded[:QK_ROPE_HEAD_DIM] = rope_fp8_raw
            # Pad scale_inv from [64/128_rows, K/128_cols] to [128/128_rows, K/128_cols]
            rope_scale_padded = torch.zeros(
                (128 + 127) // 128, rope_scale_raw.shape[1],
                dtype=rope_scale_raw.dtype, device=rope_scale_raw.device)
            rope_scale_padded[:rope_scale_raw.shape[0]] = rope_scale_raw
            rope_fp8_req, rope_ue8m0_req = self._requantize_fp8_for_ue8m0(
                rope_fp8_padded, rope_scale_padded)
            w_kv_rope = self._safe_attach(rope_fp8_req,
                                          f"layer_{layer_idx}_kv_a_rope")
            s_kv_rope = self._safe_attach(rope_ue8m0_req,
                                          f"layer_{layer_idx}_kv_a_rope_scale")
        else:
            w_kv_latent = self._safe_attach(
                kv_a_w[:self.kv_lora_rank].contiguous(),
                f"layer_{layer_idx}_kv_a_latent")
            s_kv_latent = None
            # Pad FP8 weight to [128, H]
            kv_rope_raw = kv_a_w[self.kv_lora_rank:].contiguous()
            kv_rope_padded = torch.zeros(128, kv_rope_raw.shape[1],
                                         dtype=kv_rope_raw.dtype, device=kv_rope_raw.device)
            kv_rope_padded[:QK_ROPE_HEAD_DIM] = kv_rope_raw
            w_kv_rope = self._safe_attach(kv_rope_padded,
                                          f"layer_{layer_idx}_kv_a_rope")
            s_kv_rope = None

        # Keep the latent and RoPE projections separate because only c_latent
        # goes through kv_a_layernorm; a fused projection would need a fused
        # split-plus-layernorm kernel that is not available here.
        self._fp8_linear(self.rmsnorm_out, w_kv_latent, s_kv_latent,
                         self.c_latent_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.kv_lora_rank), 1, 1),
                         block_dim=(128, 1, 1))
        # FP8 GEMM for k_pe [N, 128] (padded from 64)
        self._fp8_linear(self.rmsnorm_out, w_kv_rope, s_kv_rope,
                         self.k_pe_out,
                         grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

        # Step 5: kv_a_layernorm on c_latent ONLY
        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_a_layernorm.weight"],
            name=f"layer_{layer_idx}_kv_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln, output=self.c_latent_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # Step 6: MLA attention (KV gather + decode + reduce)
        # Dual-dispatch architecture (opt/mla-dual-dispatch, 2026-04-22):
        # When `_use_prefill` is True (mbt >= 32), BOTH the prefill kernel and
        # the decode kernels are registered so that a single compiled task
        # graph handles both regimes at runtime:
        #   - Q_LEN large (prefill chunk): prefill kernel runs, decode kernels
        #     early-exit on Q_LEN>8 gate.
        #   - Q_LEN small (MTP verify, Q_LEN 1..8): decode kernels run, prefill
        #     kernel early-exits on Q_LEN<16 gate.
        # Both write `self.attn_out`. Builder order is prefill -> decode; the
        # MPK event graph serialises the two writes, so whichever kernel really
        # runs produces the final value (the other becomes a no-op).
        layer_cache = self.mpk.attach_input(
            torch_tensor=self.ckv_kpe_cache[layer_idx],
            name=f"layer_{layer_idx}_kv_cache")
        q_len_mla = self.max_num_batched_tokens
        kv_len_max = self.mpk.max_seq_length
        if self._use_prefill:
            # 6a. Prefill path: gather into SPLIT CKV/KPE, then mla_prefill.
            self.mpk.mla_kv_gather_split_layer(
                c_latent_new=self.c_latent_out,
                k_pe_new=self.k_pe_out,
                paged_cache=layer_cache,
                ckv_sep=self.ckv_sep,
                kpe_sep=self.kpe_sep,
                mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
                grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
            )
            num_q_blocks = (q_len_mla + 64 - 1) // 64  # PF_BM=64 in kernel
            self.mpk.mla_prefill_layer(
                q_nope=self.q_nope,
                q_pe=self.q_pe,
                ckv=self.ckv_sep,
                kpe=self.kpe_sep,
                output=self.attn_out,
                mla_params=(self.num_local_q_heads, kv_len_max,
                            self.kv_lora_rank, QK_ROPE_HEAD_DIM,
                            self.v_head_dim),
                grid_dim=(self.num_local_q_heads, num_q_blocks,
                          self.mpk.max_num_batched_requests),
                block_dim=(256, 1, 1),
            )

        # 6b. Decode path: always register (both when _use_prefill and when
        # not). When _use_prefill is True, this runs in addition to prefill;
        # one of the two early-exits on runtime Q_LEN.
        self.mpk.mla_kv_gather_layer(
            c_latent_new=self.c_latent_out,
            k_pe_new=self.k_pe_out,
            paged_cache=layer_cache,
            contiguous_kv=self.contiguous_kv,
            mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
        )
        num_splits = self.mla_max_splits
        if self.world_size == 2:
            self.mpk.mla_mtp_decode_tp2_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_decode_tp2_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)
        elif self.world_size == 4:
            self.mpk.mla_mtp_decode_tp4_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_decode_tp4_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)
        elif self.world_size == 8:
            self.mpk.mla_mtp_decode_tp8_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_decode_tp8_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)
        else:
            self.mpk.mla_mtp_decode_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)

        # Step 7: O projection (V un-absorption fused into o_proj during conversion)
        # o_proj_fused: [7168, H*kv_lora_rank] — directly takes attn_out [N, H*kv_lora_rank]
        # Fuses the residual add into o_proj via the with_residual FP8 kernel.
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj.weight", f"layer_{layer_idx}_o_proj")
        # Allocate per-layer output tensor to avoid aliasing self.x ↔
        # self.attn_proj_out in the next layer's with_residual call.
        self.attn_proj_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_attn_proj_fused",
            io_category="cuda_tensor",
        )
        self._fp8_linear(self.attn_out, w_o, s_o, self.attn_proj_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                         block_dim=(128, 1, 1),
                         residual=self.x)

    def _build_dense_mlp(self, layer_idx: int, state_dict: dict):
        """Build dense MLP for layers 0-2 (FP8 weights)."""
        prefix = f"model.layers.{layer_idx}."

        w_gate_up, s_gate_up = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.gate_up_proj.weight",
            f"layer_{layer_idx}_gate_up_proj")
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
        self.mpk.silu_mul_layer(
            input=self.mlp_mid, output=self.silu_mul_out,
            grid_dim=(silu_mul_grid, 1, 1), block_dim=(128, 1, 1))
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
        self._fp8_linear(self.silu_mul_out, w_down, s_down, self.mlp_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                         block_dim=(128, 1, 1),
                         residual=self.x)

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
        # Router gate: output=NUM_EXPERTS=256, need grid small enough so each
        # block handles ≥8 BF16 elements (16B alignment for TMA descriptor).
        router_grid = min(grid_for_rmsnorm_linear_layer(w_gate.dim(0)),
                          w_gate.dim(0) // 8)  # ≥8 elements per block
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=w_gate,
            output=router_logits,
            grid_dim=(router_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # Initialize MoE output tensor
        # When TP > 1, this feeds into nvshmem allreduce and must be in symmetric memory.
        _moe_io = "nvshmem_tensor" if self._use_nvshmem else "cuda_tensor"
        moe_output = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_output",
            io_category=_moe_io,
        )
        # MoE accumulation only writes active routed slots, so initialize the
        # whole output tensor before routed/shared expert accumulation.
        self.mpk.tensor_init_layer(
            input=moe_output,
            dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
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
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=router_logits,
            bias=w_bias,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),  # 8 warps required by topk kernel
            local_expert_start=self.local_expert_start,
        )

        # Expert W1+W3 (gate + up projection)
        # Check if weights are FP8 (have scale_inv) or BF16 (post-dequant)
        w13_scale_key = f"{prefix}experts.w13.weight_scale_inv"
        use_fp8_experts = w13_scale_key in state_dict
        w_experts_w13 = self._safe_attach(
            state_dict[f"{prefix}experts.w13.weight"],
            f"layer_{layer_idx}_experts_w13")
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
        mbt = self.max_num_batched_tokens
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
        if self.ep_size > 1:
            self.mpk.tensor_init_layer(
                input=moe_mid,
                dummy_input=self.rmsnorm_out,
                dummy_output=self.rmsnorm_out,
                grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1),
                block_dim=(128, 1, 1),
            )

        if use_fp8_experts:
            w13_m_split = _moe_fp8_m_split(2 * self.routed_moe_intermediate_size,
                                           preferred=16)
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
            self.mpk.moe_w13_linear_layer(
                input=self.rmsnorm_out,
                weight=w_experts_w13,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=moe_mid,
                grid_dim=(_moe_expert_grid_x(mbt, self.num_local_experts), 1, 1),
                block_dim=(128, 1, 1),
            )

        # SiLU activation
        moe_silu_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_silu",
            io_category="cuda_tensor",
        )
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
            # This flattening maps each (token, selected_expert) row to one
            # quantization task. Current DeepSeek demo targets small batch sizes.
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
        if self.ep_size > 1:
            self.mpk.tensor_init_layer(
                input=moe_down_out,
                dummy_input=self.rmsnorm_out,
                dummy_output=self.rmsnorm_out,
                grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1),
                block_dim=(128, 1, 1),
            )
        if use_fp8_experts:
            w2_m_split = _moe_fp8_m_split(self.hidden_size, preferred=14)
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
            self.mpk.moe_w2_linear_layer(
                input=moe_silu_out,
                weight=w_experts_w2,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=moe_down_out,
                grid_dim=(_moe_expert_grid_x(mbt, self.num_local_experts), 1, 1),
                block_dim=(128, 1, 1),
            )

        # ---- Shared Expert (1 expert, TP parallel, same as dense MLP) ----
        # Shared expert runs on ALL tokens independently of routing.
        # Its output is added to the residual before the routed expert reduction:
        #   final = sum(routed * weights) + (residual + shared_expert_out)
        shared_prefix = f"{prefix}shared_experts."

        # gate_proj + up_proj fused (FP8) — use _attach_fp8_weight for requantize
        shared_gate_w = state_dict[f"{shared_prefix}gate_proj.weight"]
        shared_up_w = state_dict[f"{shared_prefix}up_proj.weight"]
        gate_scale_key = f"{shared_prefix}gate_proj.weight_scale_inv"
        has_shared_scale = gate_scale_key in state_dict
        # Verify shard was applied: gate_proj.shape[0] should equal moe_intermediate_size
        if shared_gate_w.shape[0] != self.moe_intermediate_size:
            pass  # shard mismatch warning removed
        # Interleave gate/up at split = min(linear_grid//2, scale_dim_0).
        # Scale dim 0 is moe_intermediate_size/128. For small intermediate (2048),
        # scale_dim_0 = 16 which limits the split.
        from ..utils import shuffle_tensors as _shuffle_tensors
        out_dim_total = shared_gate_w.shape[0] + shared_up_w.shape[0]
        linear_grid = grid_for_rmsnorm_linear_layer(out_dim_total)
        scale_dim_0 = shared_gate_w.shape[0] // 128  # rows per gate scale
        shared_split = min(linear_grid // 2, scale_dim_0)
        # Ensure both weight rows and scale rows divide evenly by split
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
        self._fp8_linear(self.rmsnorm_out, w_shared_gate_up, s_shared_gate_up,
                         shared_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(
                             w_shared_gate_up.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        # silu_mul: grid must equal shared_split (matching the interleave granularity).
        shared_silu_out = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_silu",
            io_category="cuda_tensor",
        )
        self.mpk.silu_mul_layer(
            input=shared_mid, output=shared_silu_out,
            grid_dim=(shared_split, 1, 1),
            block_dim=(128, 1, 1))

        # down_proj: shared_residual = shared_down(shared_silu_out) [partial sum for TP>1]
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
        self._fp8_linear(shared_silu_out, w_shared_down, s_shared_down,
                         shared_residual,
                         grid_dim=(self.hidden_size // 64, 1, 1),
                         block_dim=(128, 1, 1),
                         residual=None)

        # Final MoE contribution before transformer residual:
        #   routed_experts * topk_weights + shared_expert
        # The model residual is added after the tensor-parallel allreduce in
        # build_layers, otherwise each rank would add the same residual before
        # the reduction and over-count it.
        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out,
            weight=moe_topk_weights,
            residual=shared_residual,
            output=moe_output,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
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
            grid_dim=(self.max_num_batched_tokens, 1, 1),
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
            grid_dim=(self.max_num_batched_tokens, 1, 1),
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
        # always AllReduce + external elementwise_add.
        if self.world_size > 1:
            self.mpk.allreduce_layer(
                input=self.mlp_out, buffer=self.allreduce_buf,
                output=self.allreduce_out,
                grid_dim=(self.hidden_size // 128, 1, 1),
                block_dim=(128, 1, 1),
            )
            _mtp_mlp_contrib = self.allreduce_out
        else:
            _mtp_mlp_contrib = self.mlp_out
        mtp_mlp_residual = self._cached_new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name="mtp_mlp_residual", io_category="cuda_tensor")
        self.mpk.elementwise_add_layer(
            input_a=self.mtp_x, input_b=_mtp_mlp_contrib, output=mtp_mlp_residual,
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
        self._fp8_linear(self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_a.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        w_q_a_ln = self._cached_attach(
            state_dict[f"{attn}q_a_layernorm.weight"],
            f"mtp_{attn}q_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # q_b_proj (FP8) — dual-dispatch: always produce the fused q_nope_pe
        # for the decode kernel; additionally produce split q_nope/q_pe when
        # _use_prefill so the prefill kernel also has its inputs.
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight", f"mtp_{attn}q_b_proj")
        self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        if self._use_prefill:
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

        # c_latent: requantize
        latent_fp8, latent_ue8m0 = self._requantize_fp8_for_ue8m0(
            kv_a_w[:self.kv_lora_rank].contiguous(),
            kv_a_s[:scale_rows_latent].contiguous())
        w_kv_latent = self._safe_attach(latent_fp8, f"mtp_{attn}kv_a_latent")
        s_kv_latent = self._safe_attach(latent_ue8m0, f"mtp_{attn}kv_a_latent_scale")
        # kv_a_rope: requantize + pad to [128, H]
        # Pad FP8 weight + scale before requantize (128 rows for SM100 MMA_M)
        rope_fp8_raw = kv_a_w[self.kv_lora_rank:].contiguous()
        rope_scale_raw = kv_a_s[scale_rows_latent:].contiguous()
        rope_fp8_padded = torch.zeros(128, rope_fp8_raw.shape[1],
                                      dtype=rope_fp8_raw.dtype, device=rope_fp8_raw.device)
        rope_fp8_padded[:QK_ROPE_HEAD_DIM] = rope_fp8_raw
        rope_scale_padded = torch.zeros(
            (128 + 127) // 128, rope_scale_raw.shape[1],
            dtype=rope_scale_raw.dtype, device=rope_scale_raw.device)
        rope_scale_padded[:rope_scale_raw.shape[0]] = rope_scale_raw
        rope_fp8_req, rope_ue8m0_req = self._requantize_fp8_for_ue8m0(
            rope_fp8_padded, rope_scale_padded)
        w_kv_rope = self._safe_attach(rope_fp8_req, f"mtp_{attn}kv_a_rope")
        s_kv_rope = self._safe_attach(rope_ue8m0_req, f"mtp_{attn}kv_a_rope_scale")

        self._fp8_linear(self.rmsnorm_out, w_kv_latent, s_kv_latent, self.c_latent_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.kv_lora_rank), 1, 1),
                         block_dim=(128, 1, 1))
        self._fp8_linear(self.rmsnorm_out, w_kv_rope, s_kv_rope, self.k_pe_out,
                         grid_dim=(1, 1, 1),
                         block_dim=(128, 1, 1))

        w_kv_a_ln = self._cached_attach(
            state_dict[f"{attn}kv_a_layernorm.weight"],
            f"mtp_{attn}kv_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln, output=self.c_latent_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # MTP attention uses its own KV cache (new MLA flow)
        # Dual-dispatch (opt/mla-dual-dispatch): register BOTH prefill and
        # decode when _use_prefill, then rely on per-kernel runtime Q_LEN
        # early-exits to pick the right one. See main MLA builder above for
        # details.
        q_len_mla = self.max_num_batched_tokens
        kv_len_max = self.mpk.max_seq_length
        if self._use_prefill:
            self.mpk.mla_kv_gather_split_layer(
                c_latent_new=self.c_latent_out,
                k_pe_new=self.k_pe_out,
                paged_cache=self.mtp_ckv_kpe_cache_tensor,
                ckv_sep=self.ckv_sep,
                kpe_sep=self.kpe_sep,
                mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
                grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
            )
            num_q_blocks = (q_len_mla + 64 - 1) // 64
            self.mpk.mla_prefill_layer(
                q_nope=self.q_nope,
                q_pe=self.q_pe,
                ckv=self.ckv_sep,
                kpe=self.kpe_sep,
                output=self.attn_out,
                mla_params=(self.num_local_q_heads, kv_len_max,
                            self.kv_lora_rank, QK_ROPE_HEAD_DIM,
                            self.v_head_dim),
                grid_dim=(self.num_local_q_heads, num_q_blocks, 1),
                block_dim=(256, 1, 1),
            )
            # FALL THROUGH to decode-path registration below.
        self.mpk.mla_kv_gather_layer(
            c_latent_new=self.c_latent_out,
            k_pe_new=self.k_pe_out,
            paged_cache=self.mtp_ckv_kpe_cache_tensor,
            contiguous_kv=self.contiguous_kv,
            mla_params=(self.qk_head_dim, self.v_head_dim, self.mpk.page_size),
            grid_dim=(self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
        )
        num_splits = self.mla_max_splits
        if self.world_size == 2:
            self.mpk.mla_mtp_decode_tp2_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_decode_tp2_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)
        elif self.world_size == 4:
            self.mpk.mla_mtp_decode_tp4_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_decode_tp4_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)
        elif self.world_size == 8:
            self.mpk.mla_mtp_decode_tp8_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_decode_tp8_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)
        else:
            self.mpk.mla_mtp_decode_layer(
                self.q_nope_pe, self.contiguous_kv,
                self.mla_partial_o, self.mla_partial_lse,
                q_len_mla, kv_len_max)
            self.mpk.mla_mtp_reduce_layer(
                self.mla_partial_o, self.mla_partial_lse,
                self.attn_out, q_len_mla, kv_len_max)

        # o_proj (FP8). Match main layer's pattern: use the with_residual kernel
        # to fuse (matmul + residual) in one pass.
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj.weight", f"mtp_{attn}o_proj")
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
        self._fp8_linear(self.rmsnorm_out, w_gate_up, s_gate_up, self.mlp_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_gate_up.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        self.mpk.silu_mul_layer(
            input=self.mlp_mid, output=self.silu_mul_out,
            grid_dim=(self.intermediate_size // 64, 1, 1), block_dim=(128, 1, 1))
        w_down, s_down = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}down_proj.weight",
            f"mtp_{mlp_prefix}down_proj")
        self._fp8_linear(self.silu_mul_out, w_down, s_down, self.mlp_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                         block_dim=(128, 1, 1))

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
        # Clamp grid so each block handles ≥8 BF16 elements (16B TMA alignment).
        # Matches main MoE router at line ~715 that uses min(grid, dim(0) // 8).
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
        self.mpk.tensor_init_layer(
            input=moe_output, dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))

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
        if self.ep_size > 1:
            self.mpk.tensor_init_layer(
                input=moe_mid, dummy_input=self.rmsnorm_out,
                dummy_output=self.rmsnorm_out,
                grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1), block_dim=(128, 1, 1))
        mtp_w13_m_split = _moe_fp8_m_split(2 * self.routed_moe_intermediate_size,
                                           preferred=16)
        mtp_w13_expert_grid_x = _moe_expert_grid_x(
            mbt, self.num_local_experts, preferred_groups=8)
        self.mpk.moe_w13_fp8_layer(
            input_fp8=moe_input_fp8, input_scale=moe_input_scale,
            weight_fp8=w_w13, weight_scale=s_w13,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_mid,
            grid_dim=(mtp_w13_expert_grid_x, mtp_w13_m_split, 1),
            block_dim=(128, 1, 1))

        moe_silu_out = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.routed_moe_intermediate_size),
            dtype=bfloat16, name="mtp_moe_silu", io_category="cuda_tensor")
        self.mpk.moe_silu_mul_layer(
            input=moe_mid, output=moe_silu_out,
            grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1), block_dim=(128, 1, 1))

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
        self.mpk.quantize_fp8_layer(
            input=moe_silu_out, output_fp8=mtp_silu_fp8,
            output_scale=mtp_silu_scale,
            grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1), block_dim=(128, 1, 1),
            scale_ue8m0=False)
        moe_down_out = self._cached_new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.hidden_size),
            dtype=bfloat16, name="mtp_moe_down", io_category="cuda_tensor")
        if self.ep_size > 1:
            self.mpk.tensor_init_layer(
                input=moe_down_out, dummy_input=self.rmsnorm_out,
                dummy_output=self.rmsnorm_out,
                grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1), block_dim=(128, 1, 1))
        mtp_w2_m_split = _moe_fp8_m_split(self.hidden_size, preferred=14)
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
        gate_up_grid = grid_for_rmsnorm_linear_layer(out_dim_total)
        self._fp8_linear(self.rmsnorm_out, w_s_gu, s_s_gu, shared_mid,
                         grid_dim=(gate_up_grid, 1, 1),
                         block_dim=(128, 1, 1))
        shared_silu = self._cached_new_tensor(
            dims=(mbt, self.moe_intermediate_size), dtype=bfloat16,
            name="mtp_shared_silu", io_category="cuda_tensor")
        self.mpk.silu_mul_layer(
            input=shared_mid, output=shared_silu,
            grid_dim=(shared_split, 1, 1), block_dim=(128, 1, 1))
        w_s_down, s_s_down = self._attach_fp8_weight(
            state_dict, f"{sp}down_proj.weight", f"mtp_{sp}down_proj")
        shared_residual = self._cached_new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_shared_residual", io_category="cuda_tensor")
        # MoE internal residual always OFF (handled by external elementwise_add)
        _mtp_resid = None
        self._fp8_linear(shared_silu, w_s_down, s_s_down, shared_residual,
                         grid_dim=(self.hidden_size // 64, 1, 1),
                         block_dim=(128, 1, 1), residual=_mtp_resid)

        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out, weight=moe_topk_weights,
            residual=shared_residual, output=moe_output,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
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
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )

            # 4. hnorm(previous_hidden_states)
            hidden_input = main_hidden_states if step == 0 else self.mtp_x
            self.mpk.rmsnorm_layer(
                input=hidden_input, weight=w_hnorm, output=mtp_hnorm_out,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
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
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
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
        # After target model re-runs on draft tokens (managed by scheduler),
        # the target token IDs are available. Wire up verification here.
        target_token_ids = self.mpk.new_tensor(
            dims=(mbt, num_draft_steps + 1), dtype=int64,
            name="mtp_target_token_ids", io_category="cuda_tensor",
        )
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
            num_new = self.mpk.new_tensor(
                dims=(mbt, 1), dtype=int64,
                name="mtp_accept_num_new", io_category="cuda_tensor",
            )
            self.mpk.mtp_accept_commit_layer(
                accepted_count=accepted_count,
                output_tokens=verified_output_tokens,
                current_position=current_position,
                new_position=new_position,
                final_output=final_output,
                num_new_tokens=num_new,
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
        for i in layer_indices:
            prefix = f"model.layers.{i}."

            # Input layernorm
            w_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
                name=f"layer_{i}_input_layernorm",
            )
            self.mpk.rmsnorm_layer(
                input=self.x, weight=w_norm, output=self.rmsnorm_out,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )

            # MLA attention. The residual is fused into o_proj's with_residual
            # kernel, so attn_proj_out already contains (matmul + residual).
            self._build_mla_attention_layer(i, state_dict)
            self.x = self.attn_proj_out

            # Post-attention layernorm
            w_post_norm = self.mpk.attach_input(
                torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
                name=f"layer_{i}_post_attn_layernorm",
            )
            self.mpk.rmsnorm_layer(
                input=self.x, weight=w_post_norm, output=self.rmsnorm_out,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
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
                    self.mpk.allreduce_layer(
                        input=self.mlp_out, buffer=self.allreduce_buf,
                        output=self.allreduce_out,
                        grid_dim=(self.hidden_size // 128, 1, 1),
                        block_dim=(128, 1, 1),
                    )
                    moe_residual_out = self.mpk.new_tensor(
                        dims=(self.max_num_batched_tokens, self.hidden_size),
                        dtype=bfloat16,
                        name=f"layer_{i}_moe_residual",
                        io_category="cuda_tensor",
                    )
                    # Residual add must happen after allreduce; fusing it into
                    # moe_mul_sum_add would over-count residual on TP ranks.
                    self.mpk.elementwise_add_layer(
                        input_a=self.x, input_b=self.allreduce_out,
                        output=moe_residual_out,
                        grid_dim=(self.max_num_batched_tokens, 1, 1),
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
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1,
        )
        self.x = self.y

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
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        if with_lm_head:
            lm_head_weight = state_dict["lm_head.weight"]
            # Keep vocab rows aligned to the argmax/linear task grid. DeepSeek
            # V3's checkpoint vocab is already 129280, so this is a no-op for
            # the normal model and only handles smaller test fixtures.
            if lm_head_weight.shape[0] < padded_vocab_size:
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
                dims=(self.max_num_batched_tokens, padded_vocab_size),
                dtype=bfloat16, name="lm_head_out", io_category="cuda_tensor",
            )
            self.mpk.linear_layer(
                input=self.rmsnorm_out, weight=w_lm_head, output=lm_head_out,
                grid_dim=(grid_for_rmsnorm_linear_layer(padded_vocab_size), 1, 1),
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
            self.mpk.argmax_partial_layer(
                input=lm_head_out, output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=(self.mpk.num_workers, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=argmax_out,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )

        # Optional MTP layer
        self._build_mtp_layer(state_dict)
