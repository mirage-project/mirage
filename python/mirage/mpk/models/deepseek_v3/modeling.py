"""DeepSeek V3 model defined against the new ``mirage.mpk.layers`` catalog.

This is the v1 catalog-based implementation of DeepSeek V3 (companion to the
existing :mod:`mirage.mpk.models.deepseek_v3.builder` which uses the
direct-pk path). It mirrors the structure of
:mod:`mirage.mpk.models.qwen3.modeling`:

  * One :class:`MPKModule` subclass per architectural block (MLA, dense MLP,
    MoE MLP, decoder layer, the model, and the LM head wrapper).
  * Each block implements ``compile()`` which registers MPK tasks via the
    catalog modules from :mod:`mirage.mpk.layers`. The PyTorch ``forward()``
    paths are stubbed (``NotImplementedError``) because the MLA / paged-KV /
    MoE-routing references depend on MPK runtime state that has no eager
    counterpart; the official HF reference at
    ``transformers/models/deepseek_v3/modeling_deepseek_v3.py`` is the
    correctness oracle.
  * HF state_dict loading goes through a custom ``_load_from_state_dict``
    on each block that maps HF keys to the un-fused ``nn.Parameter`` names
    used here. The driver
    (``demo/deepseek_v3/demo_new.py``) is responsible for KV-absorption and
    W_UV→o_proj fusion **before** ``load_state_dict()``.

Scope (deliberately reduced for v1)
-----------------------------------

* **BF16 only.** No FP8 paths — the catalog ``Linear`` /
  ``LinearWithResidual`` / ``MoEW13(bf16)`` / ``MoEW2(bf16)`` modules are
  used. FP8 catalog modules (``LinearFP8`` etc.) are deferred.
* **Single GPU only.** ``world_size=1``, ``ep_size=1``, no NVShmem.
* **Decode-only.** ``max_num_batched_tokens<=8``. Uses ``MLADecode`` +
  ``MLAReduce`` (with ``num_splits=1`` when ``max_seq_length/page_size<=1``).
  No prefill path, no chunked prefill, no MTP.
* **No BMM Q, no qb_fused, no direct_paged_decode_kv.** Always:
  ``MLAKVGather(variant="standard")`` + ``contiguous_kv``.
* **Per-layer intermediates.** Every ``pk.new_tensor`` allocation is
  per-decoder-layer (no sharing across layers).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...context import current_pk
from ...layers import (
    ArgmaxPartial,
    ArgmaxReduce,
    Embed,
    Linear,
    LinearWithResidual,
    MPKModule,
    MLADecode,
    MLAKVGather,
    MLAReduce,
    MLARopeK,
    MLARopeQ,
    MoeMulSumAdd,
    MoESiluMul,
    MoETopkRouting,
    MoEW13,
    MoEW2,
    RMSNorm,
    RotaryEmbedding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_for_linear(size: int) -> int:
    """Mirror ``grid_for_rmsnorm_linear_layer`` from demo/deepseek_v3/demo.py.

    Picks the tile divisor that the kernel's task atom expects. Order of
    preference matches both qwen3 and deepseek demos.
    """
    if size / 96 > 400:
        assert size % 256 == 0, f"linear size not supported: {size}"
        return size // 256
    if size % 96 == 0:
        return 96
    if size % 64 == 0:
        return 64
    raise ValueError(f"linear out-dim {size} not divisible by 96 or 64")


def _moe_hidden_split(hidden_size: int, preferred: int = 56) -> int:
    """Pick a valid hidden-dimension split for the MoE mul_sum_add epilogue.

    Mirrors :func:`_moe_hidden_split` in
    ``python/mirage/mpk/models/deepseek_v3/builder.py``. Must be a divisor
    of ``hidden_size`` AND the per-CTA slab (``hidden_size // y``) must be
    a 128-multiple (the underlying kernel's epilogue tile).
    """
    max_y = min(preferred, max(1, hidden_size // 128))
    for y in range(max_y, 0, -1):
        if hidden_size % y == 0 and (hidden_size // y) % 128 == 0:
            return y
    return 1


# ---------------------------------------------------------------------------
# DeepseekV3MLA
# ---------------------------------------------------------------------------


class DeepseekV3MLA(MPKModule):
    """Multi-head Latent Attention (decode-only, BF16, single GPU).

    Pipeline:
        1. ``q_a_proj``  : Linear ``(hidden -> q_lora_rank)``, BF16.
        2. ``q_a_layernorm`` : RMSNorm over ``q_a_proj`` output.
        3. ``q_b_proj``  : Linear ``(q_lora_rank -> H * (kv_lora_rank +
           qk_rope_head_dim))`` — KV-absorbed; the driver MUST apply
           absorption via ``absorb_kv_into_q`` before ``load_state_dict()``.
        4. ``kv_a_proj_with_mqa`` : Linear ``(hidden -> kv_lora_rank +
           qk_rope_head_dim)``, BF16. Output split into ``c_latent`` and
           ``k_pe``.
        5. ``kv_a_layernorm`` : RMSNorm over the ``c_latent`` half only.
        6. RoPE on Q (fused per-head NoPE-PE layout) and K (PE only).
        7. ``mla_kv_gather`` (standard) : appends to per-layer paged
           ``ckv_kpe_cache`` and materialises a contiguous ``(R * S, D_K)``
           slab for the decode kernel.
        8. ``mla_decode`` + ``mla_reduce`` (split-K = 1 when the per-request
           KV length fits a single 128-token tile; otherwise the catalog
           default split-K applies).
        9. ``o_proj`` : Linear ``(H * kv_lora_rank -> hidden)`` with residual
           — the W_UV absorption into o_proj is performed by the driver at
           load time (the resulting fused weight is ``(hidden,
           H * kv_lora_rank)``, NOT the HF-native ``(hidden, H * v_head_dim)``).
    """

    def __init__(self, config, layer_idx: int, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # MLA per-token KV-latent width after absorption.
        # 576 = kv_lora_rank(512) + qk_rope_head_dim(64) for DeepSeek V3.
        self.qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # v_head_dim ABSORBED == kv_lora_rank (the absorbed attention emits
        # output of width kv_lora_rank).
        self.v_head_dim = self.kv_lora_rank

        # ---- Layernorm scales ----------------------------------------
        self.q_a_layernorm = nn.Parameter(torch.empty(self.q_lora_rank))
        self.kv_a_layernorm = nn.Parameter(torch.empty(self.kv_lora_rank))

        # ---- Linear weights (raw nn.Parameter; the v1 path does NOT
        #     fuse qkv_a at compile time — keeping the wiring minimal).
        # q_a_proj: (q_lora_rank, hidden)
        self.q_a_proj_weight = nn.Parameter(
            torch.empty(self.q_lora_rank, self.hidden_size)
        )
        # kv_a_proj_with_mqa: (kv_lora_rank + qk_rope_head_dim, hidden)
        self.kv_a_proj_with_mqa_weight = nn.Parameter(
            torch.empty(
                self.kv_lora_rank + self.qk_rope_head_dim,
                self.hidden_size,
            )
        )
        # q_b_proj (KV-absorbed): (H * (kv_lora_rank + qk_rope_head_dim),
        # q_lora_rank) — driver applies absorption before load_state_dict.
        self.q_b_proj_weight = nn.Parameter(
            torch.empty(self.num_heads * self.qk_head_dim, self.q_lora_rank)
        )
        # o_proj (W_UV-fused): (hidden, H * kv_lora_rank). HF native is
        # (hidden, H * v_head_dim); the driver fuses W_UV in at load time.
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.num_heads * self.kv_lora_rank)
        )

        # ---- Catalog leaves (no parameters in catalog; just dispatch) ----
        self.rope_q = MLARopeQ(num_heads=self.num_heads, variant="fused")
        self.rope_k = MLARopeK()
        self.kv_gather = MLAKVGather(
            d_k=self.qk_head_dim,
            d_v=self.kv_lora_rank,
            page_size=getattr(config, "page_size", 128),
            variant="standard",
        )
        # decode/reduce concrete params are filled in by compile() (they
        # depend on pk.max_seq_length and pk.page_size at compile time).
        self._decode = None
        self._reduce = None

    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Map HF keys to our parameters. The driver is responsible for
        # producing the absorbed q_b_proj and fused o_proj before calling
        # load_state_dict (see demo_new.py::_load_hf_weights_with_absorption).
        for hf_name, param in [
            ("q_a_proj.weight", self.q_a_proj_weight),
            ("q_b_proj.weight", self.q_b_proj_weight),
            ("kv_a_proj_with_mqa.weight", self.kv_a_proj_with_mqa_weight),
            ("o_proj.weight", self.o_proj_weight),
            ("q_a_layernorm.weight", self.q_a_layernorm),
            ("kv_a_layernorm.weight", self.kv_a_layernorm),
        ]:
            hf_key = prefix + hf_name
            if hf_key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict.pop(hf_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        # The MLA pipeline (paged KV, RoPE on a slice, group-limited routing,
        # split-K decode) is intrinsically tied to MPK runtime state. The
        # official HF reference at transformers/models/deepseek_v3/
        # modeling_deepseek_v3.py is the correctness oracle for forward().
        raise NotImplementedError(
            "DeepseekV3MLA.forward() is not implemented in the MPK catalog. "
            "Use transformers.DeepseekV3ForCausalLM for eager-mode reference."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError(
            "composite module — see child compile()s"
        )

    # ------------------------------------------------------------------
    def compile(self, x_dt, cos_dt, sin_dt, *, residual_dt, output):
        """Build the MLA task graph for one decoder layer.

        Args:
            x_dt: input DTensor of shape ``(mbt, hidden)`` (post-input-RMSnorm).
            cos_dt / sin_dt: RoPE tables, shape ``(max_seq_len, D_PE)``.
            residual_dt: residual DTensor for the o_proj+residual epilogue.
            output: destination DTensor for ``attn_proj_out`` (the output
                of o_proj+residual). Shape ``(mbt, hidden)``.
        """
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16, float32 as _mi_f32

        mbt = pk.max_num_batched_tokens
        mbr = pk.max_num_batched_requests
        H = self.num_heads
        D_K = self.qk_head_dim          # 576
        D_V = self.kv_lora_rank          # 512
        D_PE = self.qk_rope_head_dim     # 64
        D_NOPE = self.kv_lora_rank       # in absorbed path, "NoPE" half == c_latent width
        page_size = pk.page_size
        kv_len_max = pk.max_seq_length

        # ---- Per-layer intermediate tensors --------------------------
        # q_a_out (mbt, q_lora_rank)
        per_layer_q_a_out = pk.new_tensor(
            dims=(mbt, self.q_lora_rank),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_q_a_out",
        )
        # q_nope_pe (mbt, H * (kv_lora_rank + qk_rope_head_dim))
        per_layer_q_nope_pe = pk.new_tensor(
            dims=(mbt, H * D_K),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_q_nope_pe",
        )
        # kv_a_out (mbt, kv_lora_rank + qk_rope_head_dim) — c_latent + k_pe.
        per_layer_kv_a_out = pk.new_tensor(
            dims=(mbt, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_kv_a_out",
        )
        # contiguous_kv: gather destination, shape (R*S_max, D_K).
        per_layer_contig_kv = pk.new_tensor(
            dims=(mbr * kv_len_max, D_K),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_contiguous_kv",
        )
        # MLA decode partial outputs. Use a single split when the per-
        # request KV range fits one 128-token tile (max_seq_len/page_size
        # <= 1 in MLA terms). Otherwise the natural deepseek scheme
        # of num_splits = ceil(kv_len_max / 128) applies.
        max_kv_tiles = (kv_len_max + 127) // 128
        # Force num_splits=1 for v1 (only valid when max_kv_tiles <= 1).
        # Fall back to the deepseek default split count otherwise.
        if max_kv_tiles <= 1:
            num_splits = 1
        else:
            num_splits = max_kv_tiles
        per_layer_partial_o = pk.new_tensor(
            dims=(mbr * 1 * num_splits, H * D_V),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_partial_o",
        )
        per_layer_partial_lse = pk.new_tensor(
            dims=(mbr * 1 * num_splits, H),
            dtype=_mi_f32,
            name=f"{self.prefix}per_layer_partial_lse",
        )
        # attn_out (mbt, H * kv_lora_rank)
        per_layer_attn_out = pk.new_tensor(
            dims=(mbt, H * D_V),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_out",
        )

        # ---- Attach raw weights as DTensors --------------------------
        w_q_a_dt = pk.attach_input(
            self.q_a_proj_weight, name=f"{self.prefix}q_a_proj_weight"
        )
        w_q_b_dt = pk.attach_input(
            self.q_b_proj_weight, name=f"{self.prefix}q_b_proj_weight"
        )
        w_kv_a_dt = pk.attach_input(
            self.kv_a_proj_with_mqa_weight,
            name=f"{self.prefix}kv_a_proj_with_mqa_weight",
        )
        w_q_a_ln_dt = pk.attach_input(
            self.q_a_layernorm, name=f"{self.prefix}q_a_layernorm"
        )
        w_kv_a_ln_dt = pk.attach_input(
            self.kv_a_layernorm, name=f"{self.prefix}kv_a_layernorm"
        )
        w_o_dt = pk.attach_input(
            self.o_proj_weight, name=f"{self.prefix}o_proj_weight"
        )

        # ---- 1. q_a_proj : Linear (BF16) -----------------------------
        q_a_grid = _grid_for_linear(self.q_lora_rank)
        pk.linear_layer(
            input=x_dt,
            weight=w_q_a_dt,
            output=per_layer_q_a_out,
            grid_dim=(q_a_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 2. q_a_layernorm : RMSNorm (in-place) -------------------
        pk.rmsnorm_layer(
            input=per_layer_q_a_out,
            weight=w_q_a_ln_dt,
            output=per_layer_q_a_out,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 3. q_b_proj : Linear ------------------------------------
        q_b_grid = _grid_for_linear(H * D_K)
        pk.linear_layer(
            input=per_layer_q_a_out,
            weight=w_q_b_dt,
            output=per_layer_q_nope_pe,
            grid_dim=(q_b_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 4. kv_a_proj_with_mqa : Linear --------------------------
        kv_a_grid = _grid_for_linear(self.kv_lora_rank + self.qk_rope_head_dim)
        pk.linear_layer(
            input=x_dt,
            weight=w_kv_a_dt,
            output=per_layer_kv_a_out,
            grid_dim=(kv_a_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 5. kv_a_layernorm : RMSNorm over the c_latent slice -----
        # The kv_a_out row layout is [c_latent (kv_lora_rank) | k_pe (D_PE)].
        # rmsnorm operates on the [0:kv_lora_rank) slice in place.
        pk.rmsnorm_layer(
            input=per_layer_kv_a_out,
            weight=w_kv_a_ln_dt,
            output=per_layer_kv_a_out,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
            process_dim=self.kv_lora_rank,
            in_offset_elems=0,
            out_offset_elems=0,
        )

        # ---- 6a. RoPE on Q (fused per-head [NoPE | PE]) -------------
        # The fused Q tensor's per-row layout is
        # [h0_nope (D_NOPE=512) | h0_pe (D_PE=64) | h1_nope | h1_pe | ...].
        # The MLA fused-RoPE kernel applies rotate-half to the PE slice
        # of each head in place.
        self.rope_q.compile(
            q_pe=per_layer_q_nope_pe,
            cos_pos_embed=cos_dt,
            sin_pos_embed=sin_dt,
        )

        # ---- 6b. RoPE on K (in-place on the k_pe slice of kv_a_out) -
        self.rope_k.compile(
            k_pe=per_layer_kv_a_out,
            cos_pos_embed=cos_dt,
            sin_pos_embed=sin_dt,
            # The k_pe slice lives at [kv_lora_rank : kv_lora_rank + D_PE)
            # within each row of the (kv_lora_rank + D_PE)-wide kv_a_out.
            k_pe_row_stride=self.kv_lora_rank + self.qk_rope_head_dim,
            k_pe_offset=self.kv_lora_rank,
        )

        # ---- 7. MLA KV gather (standard variant) --------------------
        # Attaches the per-layer paged KV cache pool, appends the new
        # c_latent / k_pe rows from kv_a_out (the row stride / offset
        # kwargs let the gather kernel read the c_latent slice from the
        # combined kv_a_out buffer), and materialises a contiguous KV
        # slab for the decode kernel.
        k_cache_torch, _ = pk.get_kv_cache(self.layer_idx)
        layer_cache_dt = pk.attach_input(
            k_cache_torch, name=f"{self.prefix}ckv_kpe_cache"
        )
        # c_latent_new and k_pe_new are both views into per_layer_kv_a_out.
        # We pass kv_a_out as BOTH inputs; the kernel uses the (row_stride,
        # offset) kwargs to address the c_latent slice [0:kv_lora_rank)
        # and the k_pe slice [kv_lora_rank:kv_lora_rank+D_PE) inside it.
        self.kv_gather.compile(
            c_latent_new=per_layer_kv_a_out,
            k_pe_new=per_layer_kv_a_out,
            paged_cache=layer_cache_dt,
            contiguous_kv=per_layer_contig_kv,
            c_latent_row_stride=self.kv_lora_rank + self.qk_rope_head_dim,
            c_latent_offset_elems=0,
            k_pe_row_stride=self.kv_lora_rank + self.qk_rope_head_dim,
            k_pe_offset_elems=self.kv_lora_rank,
        )

        # ---- 8. MLA decode + reduce ---------------------------------
        # Instantiate the catalog modules lazily (kv_len / num_splits
        # depend on pk fields that aren't known at __init__ time).
        if self._decode is None:
            self._decode = MLADecode(
                num_heads=H,
                d_k=D_K,
                d_v=D_V,
                num_splits=num_splits,
                kv_len=kv_len_max,
                q_len=1,
                prefix=self.prefix,
            )
        if self._reduce is None:
            self._reduce = MLAReduce(
                num_heads=H,
                d_v=D_V,
                num_splits=num_splits,
                d_start=0,
                d_count=2,
                q_len=1,
                prefix=self.prefix,
            )

        self._decode.compile(
            q_input=per_layer_q_nope_pe,
            kv_input=per_layer_contig_kv,
            output_partial=per_layer_partial_o,
            output_lse=per_layer_partial_lse,
        )
        self._reduce.compile(
            input_partial=per_layer_partial_o,
            input_lse=per_layer_partial_lse,
            output=per_layer_attn_out,
        )

        # ---- 9. o_proj + residual -----------------------------------
        # Output shape: (mbt, hidden_size). The fused W_UV * W_o weight
        # has shape (hidden, H * kv_lora_rank); produced by the driver.
        pk.linear_with_residual_layer(
            input=per_layer_attn_out,
            weight=w_o_dt,
            residual=residual_dt,
            output=output,
            grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        return output


# ---------------------------------------------------------------------------
# DeepseekV3MLP (dense MLP for layers 0..first_k_dense_replace-1)
# ---------------------------------------------------------------------------


class DeepseekV3MLP(MPKModule):
    """Dense gated MLP (qwen3-style): gate+up fused via ``shuffle_tensors``,
    then silu_mul, then down_proj + residual. BF16.

    This mirrors :class:`mirage.mpk.models.qwen3.modeling.Qwen3MLP` exactly,
    using the dense ``intermediate_size`` for layers 0..first_k_dense_replace-1.
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size, self.hidden_size)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size, self.hidden_size)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size)
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        for hf_name, param in [
            ("gate_proj.weight", self.gate_proj_weight),
            ("up_proj.weight", self.up_proj_weight),
            ("down_proj.weight", self.down_proj_weight),
        ]:
            hf_key = prefix + hf_name
            if hf_key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict.pop(hf_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    def forward(self, x, residual):
        gate = F.linear(x, self.gate_proj_weight)
        up = F.linear(x, self.up_proj_weight)
        silu_out = (F.silu(gate.float()) * up.float()).to(x.dtype)
        return F.linear(silu_out, self.down_proj_weight) + residual

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, x_dt, residual_dt, *, output):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        fused_out = 2 * self.intermediate_size
        num_tasks_linear = _grid_for_linear(fused_out)

        w_gate_dt = pk.attach_input(
            self.gate_proj_weight, name=f"{self.prefix}gate_proj_weight"
        )
        w_up_dt = pk.attach_input(
            self.up_proj_weight, name=f"{self.prefix}up_proj_weight"
        )
        w_gateup_dt = pk.shuffle_tensors(
            inputs=[w_gate_dt, w_up_dt],
            shuffled_dim=0,
            num_groups=num_tasks_linear // 2,
            name=f"{self.prefix}gateup_proj",
        )

        per_layer_mlp_mid = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, fused_out),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_mlp_mid",
        )
        pk.linear_layer(
            input=x_dt,
            weight=w_gateup_dt,
            output=per_layer_mlp_mid,
            grid_dim=(num_tasks_linear, 1, 1),
            block_dim=(128, 1, 1),
        )
        per_layer_silu_mul_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, self.intermediate_size),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_silu_mul_out",
        )
        pk.silu_mul_layer(
            input=per_layer_mlp_mid,
            output=per_layer_silu_mul_out,
            grid_dim=(num_tasks_linear // 2, 1, 1),
            block_dim=(128, 1, 1),
        )

        w_down_dt = pk.attach_input(
            self.down_proj_weight, name=f"{self.prefix}down_proj_weight"
        )
        pk.linear_with_residual_layer(
            input=per_layer_silu_mul_out,
            weight=w_down_dt,
            residual=residual_dt,
            output=output,
            grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        return output


# ---------------------------------------------------------------------------
# DeepseekV3MoEMLP (MoE MLP for layers first_k_dense_replace..)
# ---------------------------------------------------------------------------


class DeepseekV3MoEMLP(MPKModule):
    """MoE MLP: sigmoid-routed top-k experts + shared expert + residual.

    Pipeline:
        1. Router: ``Linear(hidden -> num_experts)`` producing logits in
           BF16; sigmoid-based top-k routing with per-expert
           ``e_score_correction_bias``, group-limited gating
           (``num_groups=8``, ``topk_group=4``), and
           ``routed_scaling_factor=2.5``.
        2. Routed experts: per-expert W13 (gate + up fused) → silu_mul →
           W2 down-projection. BF16. Weights are stacked into ``experts.w13``
           / ``experts.w2`` 3D tensors at load time.
        3. Shared expert (one expert in DeepSeek V3): same dense
           gate/up/silu/down pattern as :class:`DeepseekV3MLP` but with
           ``moe_intermediate_size`` instead of dense ``intermediate_size``.
        4. Combine: ``MoeMulSumAdd`` performs
           ``out = shared_out + sum_k(routed_k * topk_weight_k)`` and adds
           the layer residual via the shared-expert path's residual chain.
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_shared_experts = getattr(config, "n_shared_experts", 1)
        self.num_groups = getattr(config, "n_group", 8)
        self.topk_group = getattr(config, "topk_group", 4)
        self.routed_scaling_factor = getattr(
            config, "routed_scaling_factor", 2.5
        )

        # ---- Router parameters ----------------------------------------
        # Note: HF stores ``mlp.gate.weight`` in shape (num_experts, hidden).
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )
        # Catalog MoETopkRouting owns the e_score_correction_bias via its
        # ``bias`` parameter; we don't duplicate it here.
        self.routing = MoETopkRouting(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            variant="sigmoid",
            num_groups=self.num_groups,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            local_num_experts=self.num_experts,
            local_expert_start=0,
            prefix=f"{prefix}routing_",
        )

        # ---- Routed experts (catalog leaves own the 3D weight tensors) ----
        self.w13 = MoEW13(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            dtype="bf16",
            prefix=f"{prefix}experts_w13_",
        )
        self.silu_mul = MoESiluMul(
            intermediate_size=self.moe_intermediate_size,
            prefix=f"{prefix}experts_silu_mul_",
        )
        self.w2 = MoEW2(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            dtype="bf16",
            prefix=f"{prefix}experts_w2_",
        )
        self.combine = MoeMulSumAdd(
            hidden_size=self.hidden_size,
            num_experts_per_tok=self.num_experts_per_tok,
            prefix=f"{prefix}combine_",
        )

        # ---- Shared expert (1 expert in DeepSeek V3) ----------------
        self.shared_gate_proj_weight = nn.Parameter(
            torch.empty(self.moe_intermediate_size, self.hidden_size)
        )
        self.shared_up_proj_weight = nn.Parameter(
            torch.empty(self.moe_intermediate_size, self.hidden_size)
        )
        self.shared_down_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.moe_intermediate_size)
        )

    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Router weight and bias. HF stores:
        #   ``mlp.gate.weight``                   (num_experts, hidden)
        #   ``mlp.gate.e_score_correction_bias``  (num_experts,)
        gate_key = prefix + "gate.weight"
        if gate_key in state_dict:
            with torch.no_grad():
                self.gate_weight.copy_(state_dict.pop(gate_key))
        bias_key = prefix + "gate.e_score_correction_bias"
        if bias_key in state_dict:
            with torch.no_grad():
                # MoETopkRouting stores its bias in fp32.
                self.routing.bias.copy_(
                    state_dict.pop(bias_key).to(torch.float32)
                )

        # Routed experts: caller is responsible for producing the stacked
        # 3D weights ``experts.w13.weight`` (num_experts, 2*moe_inter, hidden)
        # and ``experts.w2.weight`` (num_experts, hidden, moe_inter) before
        # calling load_state_dict. demo_new.py's
        # ``_load_hf_weights_with_absorption`` handles the stacking.
        w13_key = prefix + "experts.w13.weight"
        if w13_key in state_dict:
            with torch.no_grad():
                self.w13.weight.copy_(state_dict.pop(w13_key))
        w2_key = prefix + "experts.w2.weight"
        if w2_key in state_dict:
            with torch.no_grad():
                self.w2.weight.copy_(state_dict.pop(w2_key))

        # Shared expert weights.
        for hf_name, param in [
            ("shared_experts.gate_proj.weight", self.shared_gate_proj_weight),
            ("shared_experts.up_proj.weight", self.shared_up_proj_weight),
            ("shared_experts.down_proj.weight", self.shared_down_proj_weight),
        ]:
            hf_key = prefix + hf_name
            if hf_key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict.pop(hf_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV3MoEMLP.forward() is not implemented in the MPK "
            "catalog. Use transformers.DeepseekV3ForCausalLM as oracle."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    # ------------------------------------------------------------------
    def compile(self, x_dt, *, residual_dt, output):
        """Build the MoE task graph for one decoder layer.

        Args:
            x_dt: input DTensor of shape ``(mbt, hidden)`` (post-RMSnorm).
            residual_dt: residual DTensor for the shared-expert path.
            output: destination DTensor for the MoE block output
                ``(mbt, hidden)``.
        """
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16, float32 as _mi_f32, int32 as _mi_i32

        mbt = pk.max_num_batched_tokens
        H = self.hidden_size
        E = self.num_experts
        K = self.num_experts_per_tok
        I = self.moe_intermediate_size

        # ---- 1. Router : Linear x w_gate -> router_logits ------------
        w_gate_dt = pk.attach_input(
            self.gate_weight, name=f"{self.prefix}moe_gate_weight"
        )
        router_logits = pk.new_tensor(
            dims=(mbt, E),
            dtype=_mi_bf16,
            name=f"{self.prefix}router_logits",
        )
        # Router GEMM is small (E=256, hidden=7168). Pick grid_x = E // 8
        # (the deepseek builder's heuristic for routers) but cap to the
        # demo-friendly _grid_for_linear / 8 pattern.
        router_grid = min(_grid_for_linear(E), E // 8)
        pk.linear_layer(
            input=x_dt,
            weight=w_gate_dt,
            output=router_logits,
            grid_dim=(router_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 2. Top-k sigmoid routing -------------------------------
        moe_topk_weights = pk.new_tensor(
            dims=(mbt, K), dtype=_mi_f32,
            name=f"{self.prefix}moe_topk_weights",
        )
        moe_routing_indices = pk.new_tensor(
            dims=(E, mbt), dtype=_mi_i32,
            name=f"{self.prefix}moe_routing_indices",
        )
        moe_mask = pk.new_tensor(
            dims=(E + 1,), dtype=_mi_i32,
            name=f"{self.prefix}moe_mask",
        )
        self.routing.compile(
            router_logits, moe_topk_weights, moe_routing_indices, moe_mask
        )

        # ---- 3. W13 (gate+up fused) ---------------------------------
        moe_mid = pk.new_tensor(
            dims=(mbt, K, 2 * I),
            dtype=_mi_bf16,
            name=f"{self.prefix}moe_mid",
        )
        self.w13.compile(
            x=x_dt,
            routing_indices=moe_routing_indices,
            mask=moe_mask,
            output=moe_mid,
        )

        # ---- 4. SiLU-Mul on the 3D layout ---------------------------
        moe_silu = pk.new_tensor(
            dims=(mbt, K, I),
            dtype=_mi_bf16,
            name=f"{self.prefix}moe_silu",
        )
        self.silu_mul.compile(
            gateup=moe_mid,
            output=moe_silu,
            # The kernel grid for OLD MoE is (mbt, K, 1).
            grid_dim=(mbt, K, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 5. W2 (down) -------------------------------------------
        moe_down = pk.new_tensor(
            dims=(mbt, K, H),
            dtype=_mi_bf16,
            name=f"{self.prefix}moe_down",
        )
        self.w2.compile(
            x=moe_silu,
            routing_indices=moe_routing_indices,
            mask=moe_mask,
            output=moe_down,
        )

        # ---- 6. Shared expert (dense gate+up fused + silu_mul + down) ----
        # Same fused gate/up pattern as DeepseekV3MLP, but with
        # moe_intermediate_size instead of intermediate_size, and the
        # residual is the layer input (residual_dt) so the shared-expert
        # output already contains (residual + shared_expert(x)).
        w_shared_gate_dt = pk.attach_input(
            self.shared_gate_proj_weight,
            name=f"{self.prefix}shared_gate_proj_weight",
        )
        w_shared_up_dt = pk.attach_input(
            self.shared_up_proj_weight,
            name=f"{self.prefix}shared_up_proj_weight",
        )
        shared_fused_out = 2 * I
        shared_linear_grid = _grid_for_linear(shared_fused_out)
        w_shared_gateup_dt = pk.shuffle_tensors(
            inputs=[w_shared_gate_dt, w_shared_up_dt],
            shuffled_dim=0,
            num_groups=shared_linear_grid // 2,
            name=f"{self.prefix}shared_gateup_proj",
        )
        shared_mid = pk.new_tensor(
            dims=(mbt, shared_fused_out),
            dtype=_mi_bf16,
            name=f"{self.prefix}shared_mid",
        )
        pk.linear_layer(
            input=x_dt,
            weight=w_shared_gateup_dt,
            output=shared_mid,
            grid_dim=(shared_linear_grid, 1, 1),
            block_dim=(128, 1, 1),
        )
        shared_silu = pk.new_tensor(
            dims=(mbt, I),
            dtype=_mi_bf16,
            name=f"{self.prefix}shared_silu",
        )
        pk.silu_mul_layer(
            input=shared_mid,
            output=shared_silu,
            grid_dim=(shared_linear_grid // 2, 1, 1),
            block_dim=(128, 1, 1),
        )
        w_shared_down_dt = pk.attach_input(
            self.shared_down_proj_weight,
            name=f"{self.prefix}shared_down_proj_weight",
        )
        # shared_residual = shared_down(shared_silu) + residual_dt
        shared_residual = pk.new_tensor(
            dims=(mbt, H),
            dtype=_mi_bf16,
            name=f"{self.prefix}shared_residual",
        )
        pk.linear_with_residual_layer(
            input=shared_silu,
            weight=w_shared_down_dt,
            residual=residual_dt,
            output=shared_residual,
            grid_dim=(H // 64, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 7. MoeMulSumAdd : combine routed experts + shared residual ----
        self.combine.compile(
            x=moe_down,
            topk_weights=moe_topk_weights,
            residual=shared_residual,
            output=output,
            grid_dim=(mbt, _moe_hidden_split(H), 1),
            block_dim=(128, 1, 1),
        )
        return output


# ---------------------------------------------------------------------------
# DeepseekV3DecoderLayer
# ---------------------------------------------------------------------------


class DeepseekV3DecoderLayer(MPKModule):
    """One decoder layer = input-RMSnorm → MLA → post-attn-RMSnorm → MLP.

    The MLP is dense (:class:`DeepseekV3MLP`) for ``layer_idx <
    first_k_dense_replace`` and MoE (:class:`DeepseekV3MoEMLP`) thereafter.
    """

    def __init__(self, config, layer_idx: int, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}input_layernorm_",
        )
        self.self_attn = DeepseekV3MLA(
            config, layer_idx, prefix=f"{prefix}self_attn_"
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}post_attention_layernorm_",
        )
        first_moe = getattr(config, "first_k_dense_replace", 3)
        if layer_idx < first_moe:
            self.mlp = DeepseekV3MLP(config, prefix=f"{prefix}mlp_")
            self.is_moe = False
        else:
            self.mlp = DeepseekV3MoEMLP(config, prefix=f"{prefix}mlp_")
            self.is_moe = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV3DecoderLayer.forward() not implemented. "
            "Use transformers reference."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, x_dt, cos_dt, sin_dt):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        hidden = self.input_layernorm.hidden_size

        per_layer_rmsnorm_attn_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_attn_out",
        )
        per_layer_attn_proj_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_proj_out",
        )
        per_layer_rmsnorm_mlp_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_mlp_out",
        )
        per_layer_mlp_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_mlp_out",
        )

        # Input RMSNorm → MLA (with residual fused into o_proj).
        self.input_layernorm.compile(
            x_dt,
            output=per_layer_rmsnorm_attn_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.self_attn.compile(
            per_layer_rmsnorm_attn_out, cos_dt, sin_dt,
            residual_dt=x_dt,
            output=per_layer_attn_proj_out,
        )

        # Post-attention RMSNorm → MLP. Dense MLP fuses residual into
        # down_proj. MoE MLP fuses residual into the shared-expert path
        # and the final mul_sum_add reduces over the routed expert outputs
        # (so per_layer_mlp_out is the post-MLP, post-residual hidden state).
        self.post_attention_layernorm.compile(
            per_layer_attn_proj_out,
            output=per_layer_rmsnorm_mlp_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mlp.compile(
            per_layer_rmsnorm_mlp_out,
            residual_dt=per_layer_attn_proj_out,
            output=per_layer_mlp_out,
        )
        return per_layer_mlp_out


# ---------------------------------------------------------------------------
# DeepseekV3Model
# ---------------------------------------------------------------------------


class DeepseekV3Model(MPKModule):
    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.embed_tokens = Embed(
            config.vocab_size, config.hidden_size,
            prefix=f"{prefix}embed_tokens_",
        )
        # RoPE on the qk_rope_head_dim (=64) channels — NOT head_dim.
        # We use the plain RotaryEmbedding from the catalog; if the model
        # config has rope_scaling/yarn, the proper YARN-aligned cos/sin
        # would go through builder._precompute_rope_embeddings. v1 falls
        # back to plain RoPE so the modeling can stand alone for the
        # smoke-test, with a recorded TODO to plumb YARN later.
        rope_max = min(4096, getattr(config, "max_position_embeddings", 4096))
        rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.qk_rope_head_dim,
            max_position_embeddings=rope_max,
            base=rope_theta,
            prefix=f"{prefix}rotary_emb_",
        )
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(
                config, layer_idx=i, prefix=f"{prefix}layers_{i}_"
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}norm_",
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV3Model.forward() not implemented. Use transformers."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, input_tokens_dt):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        cos_dt, sin_dt = self.rotary_emb.compile()
        hidden = self.config.hidden_size
        embed_out_dt = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}embed_out",
        )
        self.embed_tokens.compile(
            input_tokens_dt,
            input_source=1,
            output=embed_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        h_dt = embed_out_dt
        for layer in self.layers:
            h_dt = layer.compile(h_dt, cos_dt, sin_dt)
        final_rmsnorm_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}final_rmsnorm_out",
        )
        self.norm.compile(
            h_dt,
            output=final_rmsnorm_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        return final_rmsnorm_out


# ---------------------------------------------------------------------------
# DeepseekV3ForCausalLM
# ---------------------------------------------------------------------------


class DeepseekV3ForCausalLM(MPKModule):
    """Full DeepSeek V3 + lm_head + split-reduce argmax (greedy decode).

    Driver responsibilities (see ``demo/deepseek_v3/demo_new.py``):
      * Allocate the per-layer combined CKV/KPE cache pool and pass it via
        ``PersistentKernel(kv_cache=...)``.
      * Pre-pad ``lm_head.weight`` to a 256-multiple vocab.
      * Pass ``output_tokens`` torch tensor through ``model.compile()``.
      * Perform KV absorption + W_UV→o_proj fusion + expert stacking BEFORE
        ``load_state_dict()``.
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.model = DeepseekV3Model(config, prefix=f"{prefix}model_")
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size,
            prefix=f"{prefix}lm_head_",
        )
        # Greedy-decode head — split-reduce so the large vocab fans out.
        self.argmax_partial = ArgmaxPartial(
            vocab_size=config.vocab_size,
            num_partial_tasks=1,  # overwritten in compile()
            prefix=f"{prefix}argmax_partial_",
        )
        self.argmax_reduce = ArgmaxReduce(
            num_partial_tasks=1,
            prefix=f"{prefix}argmax_reduce_",
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV3ForCausalLM.forward() not implemented in MPK catalog."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, input_tokens_dt, *, output_tokens=None,
                lm_head_padded_vocab: Optional[int] = None):
        pk = current_pk()
        h_dt = self.model.compile(input_tokens_dt)

        logits_dt = self.lm_head.compile(
            h_dt,
            grid_dim=(pk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )

        self.argmax_partial.num_partial_tasks = pk.num_workers
        self.argmax_reduce.num_partial_tasks = pk.num_workers
        part_val_dt, part_idx_dt = self.argmax_partial.compile(
            logits_dt,
            grid_dim=(pk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )
        return self.argmax_reduce.compile(
            part_val_dt, part_idx_dt,
            output=output_tokens,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
