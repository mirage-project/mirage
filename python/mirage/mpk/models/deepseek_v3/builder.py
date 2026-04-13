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


@register_model_builder("deepseek-v3", "DeepSeek-V3", "deepseek-ai/DeepSeek-V3")
class DeepSeekV3Builder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.world_size = mpk.world_size
        self.input_tokens = mpk.meta_tensors["input_tokens"]
        self.output_tokens = mpk.meta_tensors["output_tokens"]
        self.rank = mpk.mpi_rank
        self.max_num_batched_tokens = mpk.max_num_batched_tokens

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
        self.moe_intermediate_size = MOE_INTERMEDIATE_SIZE // self.world_size

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
            # BF16 path (post-dequant weights)
            if residual is not None:
                self.mpk.linear_with_residual_layer(
                    input=input_bf16, weight=weight, residual=residual,
                    output=output, grid_dim=grid_dim, block_dim=block_dim)
            else:
                self.mpk.linear_layer(
                    input=input_bf16, weight=weight, output=output,
                    grid_dim=grid_dim, block_dim=block_dim)
            return

        # New FP8 kernel: each CTA processes output_size=128. Grid splits output.
        output_size = weight.dim(0)
        max_grid = output_size // 128
        if max_grid < 1:
            raise ValueError(
                f"FP8 linear: output_size={output_size} < 128 (BLOCK_N). "
                f"Must use BF16 linear for this dimension.")
        if grid_dim[0] > max_grid:
            grid_dim = (max_grid, grid_dim[1], grid_dim[2])

        mbt = self.max_num_batched_tokens
        reduction_size = weight.dim(1) if weight.num_dims == 2 else weight.dim(-1)
        group_size = 128
        num_groups = (reduction_size + group_size - 1) // group_size

        # ABLATION: MPK_NO_SHARE_FP8_BUF=1 → use a unique FP8 buffer per call
        # (default shares buffer by reduction_size across layers — WAR race risk)
        if not hasattr(self, '_fp8_bufs'):
            self._fp8_bufs = {}
            self._fp8_buf_counter = 0
        no_share = os.environ.get('MPK_NO_SHARE_FP8_BUF', '0') == '1'
        cache_key = (reduction_size, self._fp8_buf_counter) if no_share else reduction_size
        if no_share:
            self._fp8_buf_counter += 1
        if cache_key not in self._fp8_bufs:
            fp8_buf = self.mpk.new_tensor(
                dims=(mbt, reduction_size), dtype=float8_e4m3,
                name=f"fp8_input_{reduction_size}_{self._fp8_buf_counter if no_share else 'shared'}",
                io_category="cuda_tensor",
            )
            # Column-major UE8M0 scale stored as transposed row-major:
            # physical shape=[packed_k, aligned_batch], dtype=uint32
            packed_k = (num_groups + 3) // 4
            aligned_batch = ((mbt + 3) // 4) * 4
            scale_buf = self.mpk.new_tensor(
                dims=(packed_k, aligned_batch), dtype=uint32,
                name=f"fp8_scale_{reduction_size}_{self._fp8_buf_counter if no_share else 'shared'}",
                io_category="cuda_tensor",
            )
            self._fp8_bufs[cache_key] = (fp8_buf, scale_buf)
        self._fp8_input_buf, self._fp8_scale_buf = self._fp8_bufs[cache_key]

        # Quantize kernel now loops over BATCH_SIZE internally, so launch one block
        # per call (was grid=(mbt,1,1) which redundantly quantized all rows mbt times).
        self.mpk.quantize_fp8_layer(
            input=input_bf16,
            output_fp8=self._fp8_input_buf,
            output_scale=self._fp8_scale_buf,
            grid_dim=(1, 1, 1),
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
        cos_embed = torch.cat([angles.cos(), angles.cos()], dim=-1).to(
            dtype=torch.bfloat16, device="cuda")
        sin_embed = torch.cat([-angles.sin(), angles.sin()], dim=-1).to(
            dtype=torch.bfloat16, device="cuda")
        # Attach as DTensors
        self.cos_pos_embed = self.mpk.attach_input(
            torch_tensor=cos_embed, name="rope_cos")
        self.sin_pos_embed = self.mpk.attach_input(
            torch_tensor=sin_embed, name="rope_sin")

    def _new_intermediate_tensors(self):
        """Allocate intermediate computation buffers."""
        mbt = self.max_num_batched_tokens

        # RMSNorm output
        self.rmsnorm_out = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size),
            dtype=bfloat16,
            name="rmsnorm_out",
            io_category="cuda_tensor",
        )

        # MLA projections
        # q_a output: [batch, q_lora_rank]
        self.q_a_out = self.mpk.new_tensor(
            dims=(mbt, self.q_lora_rank),
            dtype=bfloat16,
            name="q_a_out",
            io_category="cuda_tensor",
        )
        # q_b output (after absorption): [batch, num_local_q_heads * qk_head_dim]
        if os.environ.get("MPK_DUMP_QNOPE", "0") == "1":
            self.q_nope_pe_buf = torch.zeros(
                mbt, self.num_local_q_heads * self.qk_head_dim,
                dtype=torch.bfloat16, device="cuda")
            self.q_nope_pe = self.mpk.attach_input(
                torch_tensor=self.q_nope_pe_buf, name="q_nope_pe")
        else:
            self.q_nope_pe_buf = None
            self.q_nope_pe = self.mpk.new_tensor(
                dims=(mbt, self.num_local_q_heads * self.qk_head_dim),
                dtype=bfloat16,
                name="q_nope_pe",
                io_category="cuda_tensor",
            )
        # kv_a output split: c_latent [batch, 512] and k_pe [batch, 64]
        # We use two separate linear layers instead of one 576-dim output,
        # so we can apply kv_a_layernorm to c_latent only.
        self.c_latent_out = self.mpk.new_tensor(
            dims=(mbt, self.kv_lora_rank),  # [batch, 512]
            dtype=bfloat16,
            name="c_latent_out",
            io_category="cuda_tensor",
        )
        # Pad to 128 for SM100 MMA_M alignment (real data is first 64 elements)
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
        # Contiguous KV buffer for new MLA decode (gathered from paged cache)
        self.contiguous_kv = self.mpk.new_tensor(
            dims=(self.mpk.max_seq_length, self.qk_head_dim),
            dtype=bfloat16,
            name="contiguous_kv",
            io_category="cuda_tensor",
        )
        # MLA decode partial outputs (PR 651: bf16 for partials)
        # Sized for [B * Q_LEN * num_splits, H * D_V]; Q_LEN = mbt for prefill batching.
        max_splits = 1  # single split for decode
        mbr = self.mpk.max_num_batched_requests
        self.mla_partial_o = self.mpk.new_tensor(
            dims=(mbr * mbt * max_splits, self.v_head_dim * self.num_local_q_heads),
            dtype=bfloat16,
            name="mla_partial_o",
            io_category="cuda_tensor",
        )
        self.mla_partial_lse = self.mpk.new_tensor(
            dims=(mbr * mbt * max_splits, self.num_local_q_heads),
            dtype=float32,
            name="mla_partial_lse",
            io_category="cuda_tensor",
        )
        self.mla_max_splits = max_splits
        # Attention output: [batch, num_local_q_heads * v_head_dim_absorbed]
        # v_head_dim = 512 (kv_lora_rank, after absorption)
        if os.environ.get("MPK_DUMP_ATTN_OUT", "0") == "1":
            self.attn_out_buf = torch.zeros(mbt, self.num_local_q_heads * self.v_head_dim,
                                            dtype=torch.bfloat16, device="cuda")
            self.attn_out = self.mpk.attach_input(
                torch_tensor=self.attn_out_buf, name="attn_out")
        else:
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
        if os.environ.get("MPK_DUMP_ATTN_PROJ", "0") == "1":
            self.attn_proj_out_buf = torch.zeros(mbt, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.attn_proj_out = self.mpk.attach_input(
                torch_tensor=self.attn_proj_out_buf, name="attn_proj_out")
        else:
            self.attn_proj_out_buf = None
            self.attn_proj_out = self.mpk.new_tensor(
                dims=(mbt, self.hidden_size),
                dtype=bfloat16,
                name="attn_proj_out",
                io_category="cuda_tensor",
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
        # MPK_DUMP_MLP_OUT=1 → attach external tensor so we can read mlp_out post-mpk()
        if os.environ.get("MPK_DUMP_MLP_OUT", "0") == "1":
            self.mlp_out_buf = torch.zeros(mbt, self.hidden_size, dtype=torch.bfloat16, device="cuda")
            self.mlp_out = self.mpk.attach_input(torch_tensor=self.mlp_out_buf, name="mlp_out")
        else:
            self.mlp_out_buf = None
            self.mlp_out = self.mpk.new_tensor(
                dims=(mbt, self.hidden_size),
                dtype=bfloat16,
                name="mlp_out",
                io_category="cuda_tensor",
            )

        # AllReduce buffer
        if self.world_size > 1:
            self.allreduce_buf = self.mpk.new_tensor(
                dims=(self.world_size, mbt, self.hidden_size),
                dtype=bfloat16,
                name="allreduce_buf",
                io_category="nvshmem_tensor",
            )
            self.allreduce_out = self.mpk.new_tensor(
                dims=(mbt, self.hidden_size),
                dtype=bfloat16,
                name="allreduce_out",
                io_category="cuda_tensor",
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
        Sanitizes name for C++ codegen (dots → underscores)."""
        if not hasattr(self, '_attached_tensors'):
            self._attached_tensors = []
        self._attached_tensors.append(tensor)
        safe_name = name.replace('.', '_')
        return self.mpk.attach_input(torch_tensor=tensor, name=safe_name)

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
        if os.environ.get("MPK_PRINT_WEIGHTS", "0") == "1" and 'gate_up_proj' in key and 'shared' not in key:
            w = state_dict[key]
            wf = w.float() if w.dtype != torch.bfloat16 else w.float()
            print(f"[WSTATS] {name} ({key}): shape={tuple(w.shape)} dtype={w.dtype} amax={wf.abs().max().item():.4f} "
                  f"first8={wf.flatten()[:8].tolist()} mid8={wf.flatten()[wf.numel()//2:wf.numel()//2+8].tolist()}", flush=True)
        scale_key = f"{key}_scale_inv"
        if scale_key in state_dict and os.environ.get("MPK_BF16_BYPASS"):
            # ABLATION: dequant FP8→BF16 to bypass FP8 precision issues
            raw_w = state_dict[key]
            raw_s = state_dict[scale_key]
            w_bf16 = (raw_w.float() * raw_s.float().repeat_interleave(
                128, dim=0)[:raw_w.shape[0]].repeat_interleave(
                128, dim=1)[:, :raw_w.shape[1]]).to(torch.bfloat16)
            w = self._safe_attach(w_bf16, name)
            s = None
        elif scale_key in state_dict:
            # Requantize: dequant with float32 scale, re-quantize with UE8M0 scale
            new_fp8, packed_ue8m0 = self._requantize_fp8_for_ue8m0(
                state_dict[key], state_dict[scale_key])
            w = self._safe_attach(new_fp8, name)
            s = self._safe_attach(packed_ue8m0, f"{name}_scale")
        else:
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
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight", f"layer_{layer_idx}_q_b_proj")
        self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        # Step 4: kv_a_proj split — c_latent (FP8) + k_pe (BF16 padded)
        # k_pe output=64 < MMA_M=128, so dequant to BF16 and pad weight to [128, H]
        kv_a_w = state_dict[f"{attn}kv_a_proj_with_mqa.weight"]
        kv_a_s_key = f"{attn}kv_a_proj_with_mqa.weight_scale_inv"
        has_kv_scale = kv_a_s_key in state_dict

        if has_kv_scale and os.environ.get("MPK_BF16_BYPASS"):
            # ABLATION: dequant kv_a to BF16
            kv_a_s = state_dict[kv_a_s_key]
            kv_bf16 = (kv_a_w.float() * kv_a_s.float().repeat_interleave(
                128, dim=0)[:kv_a_w.shape[0]].repeat_interleave(
                128, dim=1)[:, :kv_a_w.shape[1]]).to(torch.bfloat16)
            w_kv_latent = self._safe_attach(
                kv_bf16[:self.kv_lora_rank].contiguous(),
                f"layer_{layer_idx}_kv_a_latent")
            s_kv_latent = None
            kv_rope_bf16 = kv_bf16[self.kv_lora_rank:].contiguous()
            kv_rope_padded = torch.zeros(128, kv_rope_bf16.shape[1],
                                         dtype=torch.bfloat16, device=kv_rope_bf16.device)
            kv_rope_padded[:QK_ROPE_HEAD_DIM] = kv_rope_bf16
            w_kv_rope = self._safe_attach(kv_rope_padded,
                                          f"layer_{layer_idx}_kv_a_rope")
            s_kv_rope = None
        elif has_kv_scale:
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

        # FP8 GEMM for c_latent [N, 512]
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
        layer_cache = self.mpk.attach_input(
            torch_tensor=self.ckv_kpe_cache[layer_idx],
            name=f"layer_{layer_idx}_kv_cache")
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
        # Q_LEN = mbt: process all batched tokens in one MLA decode call.
        # When mbt>1, the kernel does multi-query MLA with causal masking.
        # When mbt=1, behavior matches the original single-query decode.
        q_len_mla = self.max_num_batched_tokens
        # Derive head_groups so blocks×heads cover all queries × NUM_HEADS.
        _hpb = 128 // q_len_mla
        while 128 % _hpb != 0:
            _hpb -= 1
        num_head_groups_mla = self.num_local_q_heads // _hpb
        self.mpk.mla_decode_layer(
            q_input=self.q_nope_pe,
            kv_input=self.contiguous_kv,
            output_partial=self.mla_partial_o,
            output_lse=self.mla_partial_lse,
            mla_params=(self.num_local_q_heads, self.qk_head_dim,
                        self.v_head_dim, num_splits, self.mpk.max_seq_length,
                        q_len_mla),
            grid_dim=(num_splits, num_head_groups_mla, 1),
            block_dim=(128, 1, 1),
        )
        # PR 651 reduce: 256 threads handle 2 V-dims per call (256/128=2 lanes)
        lanes_per_reduce = 256 // 128  # 2
        for d_start in range(0, self.v_head_dim, lanes_per_reduce):
            self.mpk.mla_reduce_layer(
                input_partial=self.mla_partial_o,
                input_lse=self.mla_partial_lse,
                output=self.attn_out,
                mla_params=(self.num_local_q_heads, self.v_head_dim,
                            num_splits, d_start, lanes_per_reduce, q_len_mla),
                grid_dim=(num_head_groups_mla, 1, 1)
                    if q_len_mla > 1 else
                    (self.mpk.max_num_batched_requests, 1, 1),
                block_dim=(128, 1, 1),
            )

        # Step 7: O projection (V un-absorption fused into o_proj during conversion)
        # o_proj_fused: [7168, H*kv_lora_rank] — directly takes attn_out [N, H*kv_lora_rank]
        # MPK_FUSE_RESIDUAL=2 enables o_proj+residual fusion (BF16 with_residual).
        # MPK_FUSE_RESIDUAL=1 enables down_proj-only residual (FP8 with_residual).
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj.weight", f"layer_{layer_idx}_o_proj")
        _fr = os.environ.get("MPK_FUSE_RESIDUAL", "0")
        _o_residual = self.x if _fr in ("2", "3") else None
        self._fp8_linear(self.attn_out, w_o, s_o, self.attn_proj_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                         block_dim=(128, 1, 1),
                         residual=_o_residual)

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
        # input block. Match Qwen3's contract: silu grid = gate_up_grid // 2,
        # which requires gate_up weight to be INTERLEAVED at split=grid//2
        # granularity (handled in demo.py weight prep).
        self.mpk.silu_mul_layer(
            input=self.mlp_mid, output=self.silu_mul_out,
            grid_dim=(gate_up_grid // 2, 1, 1), block_dim=(128, 1, 1))
        # MPK_FUSE_RESIDUAL=1 enables fused (silu_mul_out @ W_down + self.x).
        w_down, s_down = self._attach_fp8_weight(
            state_dict, f"{prefix}mlp.down_proj.weight",
            f"layer_{layer_idx}_down_proj")
        _fr_d = os.environ.get("MPK_FUSE_RESIDUAL", "0")
        _d_residual = self.x if _fr_d in ("1", "3") else None
        self._fp8_linear(self.silu_mul_out, w_down, s_down, self.mlp_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                         block_dim=(128, 1, 1),
                         residual=_d_residual)

    def _build_moe_mlp(self, layer_idx: int, state_dict: dict):
        """Build MoE MLP for layers 3-60."""
        # ABLATION: skip MoE at various stages
        skip_level = int(os.environ.get("MPK_SKIP_MOE_EXPERTS", "0"))
        # 0=no skip, 1=skip at start, 2=skip after routing, 3=skip after shared expert
        if skip_level == 1:
            self.mlp_out = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, self.hidden_size),
                dtype=bfloat16,
                name=f"layer_{layer_idx}_moe_output_zero",
                io_category="cuda_tensor",
            )
            self.mpk.tensor_init_layer(
                input=self.mlp_out,
                dummy_input=self.rmsnorm_out,
                dummy_output=self.rmsnorm_out,
                grid_dim=(self.max_num_batched_tokens, 1, 1),
                block_dim=(128, 1, 1),
            )
            return

        prefix = f"model.layers.{layer_idx}.mlp."

        # Router
        w_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}gate.weight"],
            name=f"layer_{layer_idx}_moe_gate",
        )

        # MoE routing tensors — topk_sigmoid outputs float32 weights and int32 indices/mask
        # MPK_DUMP_MOE=<layer_idx>: attach external buffers to dump routing for a layer
        _dump_moe = os.environ.get("MPK_DUMP_MOE", "")
        if _dump_moe and int(_dump_moe) == layer_idx:
            self.moe_topk_weights_buf = torch.zeros(
                self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK,
                dtype=torch.float32, device="cuda")
            moe_topk_weights = self.mpk.attach_input(
                torch_tensor=self.moe_topk_weights_buf,
                name=f"layer_{layer_idx}_moe_topk_weights")
            self.moe_routing_indices_buf = torch.zeros(
                NUM_EXPERTS, self.max_num_batched_tokens,
                dtype=torch.int32, device="cuda")
            moe_routing_indices = self.mpk.attach_input(
                torch_tensor=self.moe_routing_indices_buf,
                name=f"layer_{layer_idx}_moe_routing_indices")
        else:
            moe_topk_weights = self.mpk.new_tensor(
                dims=(self.max_num_batched_tokens, NUM_EXPERTS_PER_TOK),
                dtype=float32,
                name=f"layer_{layer_idx}_moe_topk_weights",
                io_category="cuda_tensor",
            )
            moe_routing_indices = self.mpk.new_tensor(
                dims=(NUM_EXPERTS, self.max_num_batched_tokens),
                dtype=int32,
                name=f"layer_{layer_idx}_moe_routing_indices",
                io_category="cuda_tensor",
            )
        moe_mask = self.mpk.new_tensor(
            dims=(NUM_EXPERTS + 1,),
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
        moe_output = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_output",
            io_category="cuda_tensor",
        )
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
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=router_logits,
            bias=w_bias,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),  # 8 warps required by topk kernel
        )

        # Expert W1+W3 (gate + up projection)
        # Check if weights are FP8 (have scale_inv) or BF16 (post-dequant)
        w13_scale_key = f"{prefix}experts.w13.weight_scale_inv"
        use_fp8_experts = w13_scale_key in state_dict and not os.environ.get("MPK_BF16_BYPASS")
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
            dims=(mbt, NUM_EXPERTS_PER_TOK, 2 * self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_mid",
            io_category="cuda_tensor",
        )
        if skip_level == 5:
            self.mlp_out = moe_output
            return

        if use_fp8_experts:
            print(f"[DEBUG] before moe_w13_fp8", flush=True)
            self.mpk.moe_w13_fp8_layer(
                input_fp8=moe_input_fp8,
                input_scale=moe_input_scale,
                weight_fp8=w_experts_w13,
                weight_scale=s_experts_w13,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=moe_mid,
                grid_dim=(NUM_EXPERTS, 1, 1),
                block_dim=(128, 1, 1),
            )
        else:
            self.mpk.moe_w13_linear_layer(
                input=self.rmsnorm_out,
                weight=w_experts_w13,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=moe_mid,
                grid_dim=(NUM_EXPERTS, 1, 1),
                block_dim=(128, 1, 1),
            )

        if skip_level == 4:
            self.mlp_out = moe_output
            return

        # SiLU activation
        moe_silu_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_silu",
            io_category="cuda_tensor",
        )
        # moe_silu_mul input_map=(0,1,-1): grid.x→dim0(batch), grid.y→dim1(topk)
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
        # Group GEMM expects per-row weight_scale
        if use_fp8_experts:
            raw_scale_inv = state_dict[w2_scale_key].float().clamp(min=1e-30)
            # scale_inv IS the dequant scale, pass directly (no inversion)
            w2_scale_expanded = raw_scale_inv.repeat_interleave(128, dim=1).contiguous().to(torch.float32)
            s_experts_w2 = self._safe_attach(
                w2_scale_expanded, f"layer_{layer_idx}_experts_w2_scale")
        else:
            s_experts_w2 = None

        if use_fp8_experts:
            moe_silu_fp8 = self.mpk.new_tensor(
                dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
                dtype=float8_e4m3,
                name=f"layer_{layer_idx}_moe_silu_fp8",
                io_category="cuda_tensor",
            )
            moe_silu_scale = self.mpk.new_tensor(
                dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size // 128),
                dtype=float32,
                name=f"layer_{layer_idx}_moe_silu_scale",
                io_category="cuda_tensor",
            )
            self.mpk.quantize_fp8_layer(
                input=moe_silu_out,
                output_fp8=moe_silu_fp8,
                output_scale=moe_silu_scale,
                grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1),
                scale_ue8m0=False,  # MoE group GEMM expects float32 scale
                block_dim=(128, 1, 1),
            )

        moe_down_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_moe_down",
            io_category="cuda_tensor",
        )
        if use_fp8_experts:
            self.mpk.moe_w2_fp8_layer(
                input_fp8=moe_silu_fp8,
                input_scale=moe_silu_scale,
                weight_fp8=w_experts_w2,
                weight_scale=s_experts_w2,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=moe_down_out,
                grid_dim=(NUM_EXPERTS, 1, 1),
                block_dim=(128, 1, 1),
            )
        else:
            self.mpk.moe_w2_linear_layer(
                input=moe_silu_out,
                weight=w_experts_w2,
                moe_routing_indices=moe_routing_indices,
                moe_mask=moe_mask,
                output=moe_down_out,
                grid_dim=(NUM_EXPERTS, 1, 1),
                block_dim=(128, 1, 1),
            )

        if skip_level == 3:
            self.mlp_out = moe_output
            return

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
        print(f"[DEBUG] shared gate_up: w_dim0={w_shared_gate_up.dim(0)}, grid={grid_for_rmsnorm_linear_layer(w_shared_gate_up.dim(0))}", flush=True)
        import sys; sys.stdout.flush()
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

        # down_proj with residual (FP8): shared_residual = self.x + shared_down(shared_silu)
        # NOTE: MPK_NO_RESIDUAL=1 disables this residual to match the reference (which
        # OVERWRITES hidden instead of += because the with_residual op is broken for
        # dense layers). Without this gate, MoE layers add an extra residual not added
        # by dense layers — making MPK output diverge from reference for layers >= 3.
        w_shared_down, s_shared_down = self._attach_fp8_weight(
            state_dict, f"{shared_prefix}down_proj.weight",
            f"layer_{layer_idx}_shared_expert_down")
        shared_residual = self.mpk.new_tensor(
            dims=(self.max_num_batched_tokens, self.hidden_size),
            dtype=bfloat16,
            name=f"layer_{layer_idx}_shared_residual",
            io_category="cuda_tensor",
        )
        _shared_resid_input = None if os.environ.get("MPK_NO_RESIDUAL", "0") == "1" else self.x
        self._fp8_linear(shared_silu_out, w_shared_down, s_shared_down,
                         shared_residual,
                         grid_dim=(self.hidden_size // 64, 1, 1),
                         block_dim=(128, 1, 1),
                         residual=_shared_resid_input)

        # Final: moe_output = sum(routed_experts * weights) + shared_residual
        # where shared_residual = original_hidden + shared_expert_output
        self.mpk.moe_mul_sum_add_layer(
            input=moe_down_out,
            weight=moe_topk_weights,
            residual=shared_residual,
            output=moe_output,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mlp_out = moe_output

    def _build_mtp_decoder_layer(self, state_dict: dict, prefix: str):
        """Build one MTP decoder layer (same structure as main model layer).

        The MTP block is a full DeepseekV2DecoderLayer with its own weights.
        It shares the same architecture: input_layernorm → MLA → post_norm → MLP.
        """
        # Input layernorm
        w_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}input_layernorm.weight"],
            name="mtp_block_input_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.mtp_x, weight=w_norm, output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # MLA attention (same structure as main model, own weights)
        self._build_mla_attention_layer_with_prefix(prefix, state_dict)

        # Residual (attn output is in self.attn_proj_out)
        self.mtp_x = self.attn_proj_out

        # AllReduce after attention
        if self.world_size > 1:
            self.mpk.allreduce_layer(
                input=self.attn_proj_out, buffer=self.allreduce_buf,
                output=self.allreduce_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mtp_x = self.allreduce_out

        # Post-attention layernorm
        w_post_norm = self.mpk.attach_input(
            torch_tensor=state_dict[f"{prefix}post_attention_layernorm.weight"],
            name="mtp_block_post_attn_layernorm",
        )
        self.mpk.rmsnorm_layer(
            input=self.mtp_x, weight=w_post_norm, output=self.rmsnorm_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )

        # MLP: DeepSeek V3 MTP block uses MoE MLP (same as main layers 3-60)
        # Check if MoE weights exist, fallback to dense
        mlp_gate_key = f"{prefix}mlp.gate.weight"
        if mlp_gate_key in state_dict:
            self._build_moe_mlp_with_prefix(prefix, state_dict)
        else:
            self._build_dense_mlp_with_prefix(prefix, state_dict)

        self.mtp_x = self.mlp_out
        if self.world_size > 1:
            self.mpk.allreduce_layer(
                input=self.mlp_out, buffer=self.allreduce_buf,
                output=self.allreduce_out,
                grid_dim=(self.hidden_size // 64, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.mtp_x = self.allreduce_out

    def _build_mla_attention_layer_with_prefix(self, prefix: str, state_dict: dict):
        """Build MLA attention using a custom weight prefix (FP8, for MTP reuse)."""
        attn = f"{prefix}self_attn."

        # q_a_proj (FP8)
        w_q_a, s_q_a = self._attach_fp8_weight(
            state_dict, f"{attn}q_a_proj.weight", f"mtp_{attn}q_a_proj")
        self._fp8_linear(self.rmsnorm_out, w_q_a, s_q_a, self.q_a_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_a.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))

        w_q_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}q_a_layernorm.weight"],
            name=f"mtp_{attn}q_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.q_a_out, weight=w_q_a_ln, output=self.q_a_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # q_b_proj (FP8)
        w_q_b, s_q_b = self._attach_fp8_weight(
            state_dict, f"{attn}q_b_proj.weight", f"mtp_{attn}q_b_proj")
        self._fp8_linear(self.q_a_out, w_q_b, s_q_b, self.q_nope_pe,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_q_b.dim(0)), 1, 1),
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

        w_kv_a_ln = self.mpk.attach_input(
            torch_tensor=state_dict[f"{attn}kv_a_layernorm.weight"],
            name=f"mtp_{attn}kv_a_layernorm")
        self.mpk.rmsnorm_layer(
            input=self.c_latent_out, weight=w_kv_a_ln, output=self.c_latent_out,
            grid_dim=(self.max_num_batched_tokens, 1, 1), block_dim=(128, 1, 1))

        # MTP attention uses its own KV cache (new MLA flow)
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
        # Mirror main builder: Q_LEN=mbt for prefill batching support.
        q_len_mla = self.max_num_batched_tokens
        _hpb = 128 // q_len_mla
        while 128 % _hpb != 0:
            _hpb -= 1
        num_head_groups_mla = self.num_local_q_heads // _hpb
        self.mpk.mla_decode_layer(
            q_input=self.q_nope_pe,
            kv_input=self.contiguous_kv,
            output_partial=self.mla_partial_o,
            output_lse=self.mla_partial_lse,
            mla_params=(self.num_local_q_heads, self.qk_head_dim,
                        self.v_head_dim, num_splits, self.mpk.max_seq_length,
                        q_len_mla),
            grid_dim=(num_splits, num_head_groups_mla, 1),
            block_dim=(128, 1, 1),
        )
        self.mpk.mla_reduce_layer(
            input_partial=self.mla_partial_o,
            input_lse=self.mla_partial_lse,
            output=self.attn_out,
            mla_params=(self.num_local_q_heads, self.v_head_dim,
                        num_splits, 0, self.v_head_dim, q_len_mla),
            grid_dim=(num_head_groups_mla, 1, 1) if q_len_mla > 1 else
                (self.mpk.max_num_batched_requests, 1, 1),
            block_dim=(128, 1, 1),
        )

        # o_proj (FP8)
        w_o, s_o = self._attach_fp8_weight(
            state_dict, f"{attn}o_proj.weight", f"mtp_{attn}o_proj")
        self._fp8_linear(self.attn_out, w_o, s_o, self.attn_proj_out,
                         grid_dim=(grid_for_rmsnorm_linear_layer(self.hidden_size), 1, 1),
                         block_dim=(128, 1, 1))

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

        # Skip MoE if flagged (group GEMM kernel has batch_size issues)
        skip_level = int(os.environ.get("MPK_SKIP_MOE_EXPERTS", "0"))
        if skip_level >= 1:
            self.mlp_out = self.mpk.new_tensor(
                dims=(mbt, self.hidden_size), dtype=bfloat16,
                name=f"mtp_moe_output_zero", io_category="cuda_tensor")
            self.mpk.tensor_init_layer(
                input=self.mlp_out, dummy_input=self.rmsnorm_out,
                dummy_output=self.rmsnorm_out,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
            return

        mlp_prefix = f"{prefix}mlp."

        # Router (BF16 — gate.weight is BF16)
        w_gate = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}gate.weight"],
            name=f"mtp_{mlp_prefix}gate")
        moe_topk_weights = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK), dtype=bfloat16,
            name="mtp_moe_topk_weights", io_category="cuda_tensor")
        moe_routing_indices = self.mpk.new_tensor(
            dims=(NUM_EXPERTS, mbt), dtype=bfloat16,
            name="mtp_moe_routing_indices", io_category="cuda_tensor")
        moe_mask = self.mpk.new_tensor(
            dims=(NUM_EXPERTS + 1,), dtype=int32,
            name="mtp_moe_mask", io_category="cuda_tensor")
        router_logits = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS), dtype=bfloat16,
            name="mtp_router_logits", io_category="cuda_tensor")
        self.mpk.linear_layer(
            input=self.rmsnorm_out, weight=w_gate, output=router_logits,
            grid_dim=(grid_for_rmsnorm_linear_layer(w_gate.dim(0)), 1, 1),
            block_dim=(128, 1, 1))

        moe_output = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_moe_output", io_category="cuda_tensor")
        self.mpk.tensor_init_layer(
            input=moe_output, dummy_input=self.rmsnorm_out,
            dummy_output=self.rmsnorm_out,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))

        w_gate_bias = self.mpk.attach_input(
            torch_tensor=state_dict[f"{mlp_prefix}gate.e_score_correction_bias"],
            name=f"mtp_{mlp_prefix}gate_bias")
        self.mpk.moe_topk_sigmoid_routing_layer(
            input=router_logits, bias=w_gate_bias,
            output=(moe_topk_weights, moe_routing_indices, moe_mask),
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1))

        # Expert W13 (FP8)
        w_w13, s_w13 = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}experts.w13.weight",
            f"mtp_{mlp_prefix}experts_w13")
        moe_input_fp8 = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_moe_input_fp8", io_category="cuda_tensor")
        moe_input_scale = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size // 128), dtype=float32,
            name="mtp_moe_input_scale", io_category="cuda_tensor")
        self.mpk.quantize_fp8_layer(
            input=self.rmsnorm_out, output_fp8=moe_input_fp8,
            output_scale=moe_input_scale,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            scale_ue8m0=False)  # MoE group GEMM expects float32 scale

        moe_mid = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, 2 * self.moe_intermediate_size),
            dtype=bfloat16, name="mtp_moe_mid", io_category="cuda_tensor")
        self.mpk.moe_w13_fp8_layer(
            input_fp8=moe_input_fp8, input_scale=moe_input_scale,
            weight_fp8=w_w13, weight_scale=s_w13,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_mid, grid_dim=(NUM_EXPERTS, 1, 1), block_dim=(128, 1, 1))

        moe_silu_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
            dtype=bfloat16, name="mtp_moe_silu", io_category="cuda_tensor")
        self.mpk.moe_silu_mul_layer(
            input=moe_mid, output=moe_silu_out,
            grid_dim=(mbt, NUM_EXPERTS_PER_TOK, 1), block_dim=(128, 1, 1))

        # Expert W2 (FP8) — quantize 3D silu_out first
        w_w2, s_w2 = self._attach_fp8_weight(
            state_dict, f"{mlp_prefix}experts.w2.weight",
            f"mtp_{mlp_prefix}experts_w2")
        mtp_silu_fp8 = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size),
            dtype=float8_e4m3, name="mtp_moe_silu_fp8", io_category="cuda_tensor")
        mtp_silu_scale = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.moe_intermediate_size // 128),
            dtype=float32, name="mtp_moe_silu_scale", io_category="cuda_tensor")
        self.mpk.quantize_fp8_layer(
            input=moe_silu_out, output_fp8=mtp_silu_fp8,
            output_scale=mtp_silu_scale,
            grid_dim=(mbt * NUM_EXPERTS_PER_TOK, 1, 1), block_dim=(128, 1, 1),
            scale_ue8m0=False)  # MoE group GEMM expects float32 scale
        moe_down_out = self.mpk.new_tensor(
            dims=(mbt, NUM_EXPERTS_PER_TOK, self.hidden_size),
            dtype=bfloat16, name="mtp_moe_down", io_category="cuda_tensor")
        self.mpk.moe_w2_fp8_layer(
            input_fp8=mtp_silu_fp8, input_scale=mtp_silu_scale,
            weight_fp8=w_w2, weight_scale=s_w2,
            moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
            output=moe_down_out, grid_dim=(NUM_EXPERTS, 1, 1), block_dim=(128, 1, 1))

        # Shared expert (FP8)
        sp = f"{mlp_prefix}shared_experts."
        shared_gate_w = state_dict[f"{sp}gate_proj.weight"]
        shared_up_w = state_dict[f"{sp}up_proj.weight"]
        shared_gate_s = state_dict[f"{sp}gate_proj.weight_scale_inv"]
        shared_up_s = state_dict[f"{sp}up_proj.weight_scale_inv"]
        w_s_gu = self._safe_attach(
            torch.cat([shared_gate_w, shared_up_w], dim=0), f"mtp_{sp}gate_up")
        s_s_gu = self._safe_attach(
            torch.cat([shared_gate_s, shared_up_s], dim=0), f"mtp_{sp}gate_up_scale")
        shared_mid = self.mpk.new_tensor(
            dims=(mbt, 2 * self.moe_intermediate_size), dtype=bfloat16,
            name="mtp_shared_mid", io_category="cuda_tensor")
        self._fp8_linear(self.rmsnorm_out, w_s_gu, s_s_gu, shared_mid,
                         grid_dim=(grid_for_rmsnorm_linear_layer(w_s_gu.dim(0)), 1, 1),
                         block_dim=(128, 1, 1))
        shared_silu = self.mpk.new_tensor(
            dims=(mbt, self.moe_intermediate_size), dtype=bfloat16,
            name="mtp_shared_silu", io_category="cuda_tensor")
        self.mpk.silu_mul_layer(
            input=shared_mid, output=shared_silu,
            grid_dim=(self.moe_intermediate_size // 64, 1, 1), block_dim=(128, 1, 1))
        w_s_down, s_s_down = self._attach_fp8_weight(
            state_dict, f"{sp}down_proj.weight", f"mtp_{sp}down_proj")
        shared_residual = self.mpk.new_tensor(
            dims=(mbt, self.hidden_size), dtype=bfloat16,
            name="mtp_shared_residual", io_category="cuda_tensor")
        # Same gate as main MoE: when MPK_NO_RESIDUAL=1, drop the residual add.
        _mtp_resid = None if os.environ.get("MPK_NO_RESIDUAL", "0") == "1" else self.mtp_x
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
        eh_proj_full = state_dict[f"{mtp_prefix}eh_proj.weight"]
        w_eh_proj_1 = self.mpk.attach_input(
            torch_tensor=eh_proj_full[:, :self.hidden_size].contiguous(),
            name="mtp_eh_proj_embed",
        )
        w_eh_proj_2 = self.mpk.attach_input(
            torch_tensor=eh_proj_full[:, self.hidden_size:].contiguous(),
            name="mtp_eh_proj_hidden",
        )

        # ---- MTP KV cache (separate from main model) ----
        mtp_ckv_kpe_cache = torch.zeros(
            (self.mpk.max_num_pages, self.mpk.page_size, self.qk_head_dim),
            dtype=torch.bfloat16, device="cuda",
        )
        self.mtp_ckv_kpe_cache_tensor = self.mpk.attach_input(
            torch_tensor=mtp_ckv_kpe_cache,
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

        # ---- Shared embed weight reference (saved during build_from_dict) ----
        w_embed = self.w_embed

        # ---- Save main model state ----
        main_hidden_states = self.x  # After all 61 layers + final norm

        # ---- Draft generation loop (statically unrolled) ----
        for step in range(num_draft_steps):
            # 1. Get draft token: step 0 from main argmax, step 1+ from prev MTP
            draft_input = self.argmax_out_dtensor if step == 0 else draft_token_ids

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
            if not os.environ.get("MPK_SKIP_MTP_DECODER"):
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
            self.mpk.argmax_partial_layer(
                input=lm_head_out,
                output=(self.argmax_part_value, self.argmax_part_index),
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )
            self.mpk.argmax_reduce_layer(
                input=(self.argmax_part_value, self.argmax_part_index),
                output=draft_token_ids,
                grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
            )

            # Scatter this step's draft token into the collection buffer
            if os.environ.get("MPK_SKIP_MTP_VERIFY"):
                continue  # skip scatter too
            self.mpk.mtp_token_scatter_layer(
                src=draft_token_ids,
                dst=all_draft_ids,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
                batch_size=mbt,
                num_slots=num_draft_steps,
                slot_idx=step,
            )

        # ABLATION: skip verify/accept for now
        if os.environ.get("MPK_SKIP_MTP_VERIFY"):
            return

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
        # TODO: add probabilistic and synthetic verify paths

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
        # Ablation env vars (each may be "1" to skip the corresponding piece)
        skip_layer = os.environ.get("MPK_SKIP_LAYER", "0") == "1"
        skip_attn = os.environ.get("MPK_SKIP_ATTN", "0") == "1"
        skip_mlp = os.environ.get("MPK_SKIP_MLP", "0") == "1"
        for i in layer_indices:
            prefix = f"model.layers.{i}."

            # MPK_SKIP_LAYER=1: bypass entire decoder layer — keeps self.x as-is
            if skip_layer:
                continue

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

            # MLA attention
            if not skip_attn:
                self._build_mla_attention_layer(i, state_dict)
                # Residual connection (NOTE: MPK currently overwrites, no `+`)
                self.x = self.attn_proj_out
            # If skip_attn, self.x stays as previous value (effectively skip attention)

            # AllReduce after attention
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.attn_proj_out, buffer=self.allreduce_buf,
                    output=self.allreduce_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.allreduce_out

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
            if skip_mlp:
                # Bypass MLP entirely — keep self.x from attention output
                pass
            elif os.environ.get("MPK_SKIP_MOE") and i >= FIRST_MOE_LAYER:
                # Skip MoE MLP entirely for debugging
                self.mlp_out = self.x
                self.x = self.mlp_out
            elif i < FIRST_MOE_LAYER:
                self._build_dense_mlp(i, state_dict)
                self.x = self.mlp_out
            else:
                self._build_moe_mlp(i, state_dict)
                self.x = self.mlp_out
            if self.world_size > 1:
                self.mpk.allreduce_layer(
                    input=self.mlp_out, buffer=self.allreduce_buf,
                    output=self.allreduce_out,
                    grid_dim=(self.hidden_size // 64, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.x = self.allreduce_out

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
            if lm_head_weight.shape[0] < padded_vocab_size:
                lm_head_weight = torch.cat([
                    lm_head_weight,
                    torch.zeros(padded_vocab_size - lm_head_weight.shape[0],
                                self.hidden_size, device="cuda"),
                ], dim=0)

            self.w_lm_head = self.mpk.attach_input(
                torch_tensor=lm_head_weight, name="lm_head",
            )
            w_lm_head = self.w_lm_head
            # MPK_DUMP_LOGITS=1 → allocate a torch tensor and attach it so the
            # caller can read the logits (pre-argmax) after mpk() returns.
            if os.environ.get("MPK_DUMP_LOGITS", "0") == "1":
                self.lm_head_out_buf = torch.zeros(
                    self.max_num_batched_tokens, padded_vocab_size,
                    dtype=torch.bfloat16, device="cuda")
                lm_head_out = self.mpk.attach_input(
                    torch_tensor=self.lm_head_out_buf, name="lm_head_out",
                )
            else:
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

            # Argmax
            self.argmax_out_dtensor = self.mpk.attach_input(
                torch_tensor=self.output_tokens, name="output_token",
            )
            argmax_out = self.argmax_out_dtensor
            # Argmax grid: matches Qwen3 pattern.
            # Partial: (num_workers, 1, 1) splits vocab across all workers.
            # Reduce: (1, 1, 1) — single block combines partials, iterates batches.
            # Old (mbt, 1, 1) caused race for mbt>1: blocks wrote overlapping
            # output_tokens slots because the kernel uses NUM_PARTIAL_TASKS for
            # output offset but the buffer's actual stride is num_workers.
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
