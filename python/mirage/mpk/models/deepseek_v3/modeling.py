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
  * HF checkpoint loading goes through a streaming ``load_weights`` on
    :class:`DeepseekV3ForCausalLM` (vLLM-style): it consumes raw
    ``(name, tensor)`` pairs, **dequantizes FP8 weights inline** by pairing
    each weight with its ``<name>_scale_inv`` partner, **filters routed
    experts by EP** (non-local experts are skipped, never stored) and stacks
    local experts via the per-expert ``weight_loader``, **stashes the raw
    MLA ``q_b`` / ``kv_b`` / ``o_proj`` weights** on the owning MLA module
    for the later ``process_weights`` absorption step (KV-absorption +
    W_UV→o_proj fusion), and copies the router ``e_score_correction_bias``
    straight to FP32. All other (directly-mappable) keys are remapped to the
    catalog ``named_parameters()`` path and copied in BF16.

Scope (deliberately reduced for v1)
-----------------------------------

* **FP8 dequantized inline at load.** The catalog ``Linear`` /
  ``LinearWithResidual`` / ``MoEW13(bf16)`` / ``MoEW2(bf16)`` modules hold
  BF16 weights; ``load_weights`` dequantizes any FP8 checkpoint tensors to
  BF16 on the fly. The FP8 catalog modules (``LinearFP8`` etc.) are deferred.
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

import re
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...context import current_pk
from ...layers._base import SKIP_WEIGHT
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
# Streaming weight-load helpers (pure; CPU-testable)
# ---------------------------------------------------------------------------


# Matches a routed-expert weight key, e.g.
# ``model.layers.5.mlp.experts.37.gate_proj.weight`` ->
# (layer_idx=5, expert_id=37, proj="gate").
_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
)


def _parse_expert_key(name: str) -> Optional[Tuple[int, int, str]]:
    """Parse a routed-expert weight key.

    Returns ``(layer_idx, expert_id, proj)`` where ``proj`` is one of
    ``"gate"`` / ``"up"`` / ``"down"``, or ``None`` if ``name`` is not a
    routed-expert weight key.
    """
    m = _EXPERT_KEY_RE.match(name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)


# Matches any per-layer key, e.g. ``model.layers.61.eh_proj.weight`` ->
# layer_idx=61. Used to skip keys for layers the model did NOT build
# (reduced-layer runs via --num-hidden-layers-override) and MTP keys
# (``model.layers.<num_hidden_layers>.*``: eh_proj, enorm, hnorm,
# shared_head.*) that have no catalog counterpart.
_LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _is_out_of_range_layer_key(name: str, built_layers: int) -> bool:
    """Whether ``name`` is a ``model.layers.{i}.*`` key for a layer the model
    did NOT build (``i >= built_layers``).

    Such keys arise from MTP weights (at ``model.layers.<num_hidden_layers>.*``)
    and from reduced-layer runs (``--num-hidden-layers-override`` builds fewer
    layers than the checkpoint contains). They have no catalog parameter and
    must be skipped rather than treated as unrecognized.
    """
    m = _LAYER_KEY_RE.match(name)
    if m is None:
        return False
    return int(m.group(1)) >= built_layers


# FP8 dequant helpers, transcribed verbatim from
# demo/deepseek_v3/models/convert.py (kept inline so the package does not
# depend on the demo directory being importable).


def is_fp8(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def dequantize_fp8(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 (e4m3fn) weight to ``target_dtype`` using per-tensor,
    per-output-channel, or block-wise scale factors. Verbatim copy of
    ``convert.dequantize_fp8``.
    """
    if weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e4m3fnuz:
        weight_f = weight.to(torch.float32)
    else:
        # Already a normal float type, just cast.
        return weight.to(target_dtype)

    if scale is None:
        return weight_f.to(target_dtype)

    scale = scale.to(torch.float32)

    if scale.numel() == 1:
        # Per-tensor scale
        result = weight_f * scale.item()
    elif scale.dim() == 1 and scale.shape[0] == weight.shape[0]:
        # Per-output-channel scale: scale shape [out_features]
        result = weight_f * scale.unsqueeze(1)
    elif scale.dim() == 2:
        # Block-wise scale: scale shape [ceil(out/block_out), ceil(in/block_in)]
        block_size = 128
        out_features, in_features = weight.shape
        expanded = scale.repeat_interleave(block_size, dim=0)[:out_features]
        expanded = expanded.repeat_interleave(block_size, dim=1)[:, :in_features]
        result = weight_f * expanded
    else:
        # Fallback: try broadcasting
        result = weight_f * scale

    return result.to(target_dtype)


def _absorb_kv_into_q(
    q_b_proj_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    q_lora_rank: int,
) -> torch.Tensor:
    """Absorb ``kv_b_proj`` into ``q_b_proj`` for MLA decode.

    Transcribed verbatim (math only) from
    ``demo/deepseek_v3/models/convert.absorb_kv_into_q``, with the model-arch
    dims passed in explicitly (read off the owning ``DeepseekV3MLA`` instance)
    rather than via a ``params`` dict / ``get_model_params`` — so the package
    has no dependency on the demo directory.

    Input shapes (HF-native):
        q_b_proj_weight:  ``(num_heads * (qk_nope_head_dim + qk_rope_head_dim), q_lora_rank)``
        kv_b_proj_weight: ``(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)``

    Output shape:
        ``(num_heads * (kv_lora_rank + qk_rope_head_dim), q_lora_rank)``

    The math (float32 for precision; caller casts back to bf16):
        - reshape q_b per head, split into nope / rope halves;
        - reshape kv_b per head, take the k_nope half;
        - q_absorbed_nope = bmm(k_nope^T, q_nope) -> (H, kv_lora_rank, q_lora_rank);
        - concat the unchanged q_rope back on -> (H, kv_lora_rank+qk_rope_head_dim, q_lora_rank);
        - flatten heads.
    """
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_head_dim = qk_nope_head_dim + v_head_dim

    # q_b_proj_weight: [num_heads * q_head_dim, q_lora_rank]
    q_b = q_b_proj_weight.float().reshape(num_heads, q_head_dim, q_lora_rank)
    q_nope = q_b[:, :qk_nope_head_dim, :]  # [H, qk_nope_head_dim, q_lora_rank]
    q_rope = q_b[:, qk_nope_head_dim:, :]  # [H, qk_rope_head_dim, q_lora_rank]

    # kv_b_proj_weight: [num_heads * kv_head_dim, kv_lora_rank]
    kv_b = kv_b_proj_weight.float().reshape(num_heads, kv_head_dim, kv_lora_rank)
    k_nope = kv_b[:, :qk_nope_head_dim, :]  # [H, qk_nope_head_dim, kv_lora_rank]

    # q_absorbed_nope_h = k_nope_h^T @ q_nope_h -> [H, kv_lora_rank, q_lora_rank]
    q_absorbed_nope = torch.bmm(k_nope.transpose(1, 2), q_nope)

    # Concat absorbed nope with unchanged rope per head.
    q_absorbed = torch.cat([q_absorbed_nope, q_rope], dim=1)

    out_dim = kv_lora_rank + qk_rope_head_dim
    return q_absorbed.reshape(num_heads * out_dim, q_lora_rank)


def _remap_dsv3_key(name: str) -> str:
    """Map an HF DeepSeek V3 state_dict key to its catalog ``named_parameters()`` path.

    Only handles the *directly-mappable* (non-expert, non-MLA-stash, non-router-bias)
    keys; the streaming ``load_weights`` loop intercepts routed-expert keys, the raw
    MLA ``q_b/kv_b/o_proj`` weights, and the router ``e_score_correction_bias`` before
    this function is consulted.

    The catalog stores the MLA projections / layernorms, the MoE router-gate and
    shared-expert projections, AND the dense-MLP projections as raw ``nn.Parameter``
    attributes (NOT child modules), so the trailing ``...module.weight`` HF segment
    collapses to a single ``..._weight`` attribute name. Embedding, final norm,
    lm_head, and the decoder-layer layernorms are child-module ``.weight`` leaves
    (``Embed.weight`` / ``Linear.weight`` / ``RMSNorm.weight``), so those keys pass
    through unchanged.
    """
    # --- MLA layernorms: HF '<attn>.q_a_layernorm.weight' -> catalog '<attn>.q_a_layernorm'
    if name.endswith(".self_attn.q_a_layernorm.weight"):
        return name[: -len(".weight")]
    if name.endswith(".self_attn.kv_a_layernorm.weight"):
        return name[: -len(".weight")]
    # --- MLA linear projections written directly (q_a / kv_a). q_b/kv_b/o are
    #     stashed separately and never reach this helper.
    if name.endswith(".self_attn.q_a_proj.weight"):
        return name[: -len(".self_attn.q_a_proj.weight")] + ".self_attn.q_a_proj_weight"
    if name.endswith(".self_attn.kv_a_proj_with_mqa.weight"):
        return (
            name[: -len(".self_attn.kv_a_proj_with_mqa.weight")]
            + ".self_attn.kv_a_proj_with_mqa_weight"
        )
    # --- MoE router gate matrix: '<mlp>.gate.weight' -> catalog '<mlp>.gate_weight'.
    if name.endswith(".mlp.gate.weight"):
        return name[: -len(".mlp.gate.weight")] + ".mlp.gate_weight"
    # --- MoE shared experts: '<mlp>.shared_experts.{gate,up,down}_proj.weight'
    #     -> catalog '<mlp>.shared_{gate,up,down}_proj_weight'.
    for proj in ("gate", "up", "down"):
        suffix = f".mlp.shared_experts.{proj}_proj.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".mlp.shared_{proj}_proj_weight"
    # --- Dense-MLP projections (layers < first_k_dense_replace):
    #     '<mlp>.{gate,up,down}_proj.weight' -> catalog '<mlp>.{gate,up,down}_proj_weight'.
    #     (DeepseekV3MLP stores these as raw nn.Parameter attributes.)
    for proj in ("gate", "up", "down"):
        suffix = f".mlp.{proj}_proj.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".mlp.{proj}_proj_weight"
    # --- Everything else (embed_tokens, norm, lm_head, the decoder-layer
    #     input/post_attention layernorms) maps 1:1 by named_modules path.
    return name


# ---------------------------------------------------------------------------
# DeepseekV3MLA
# ---------------------------------------------------------------------------


class DeepseekV3MLA(MPKModule):
    """Multi-head Latent Attention (decode-only, BF16, single GPU).

    Pipeline:
        1. ``q_a_proj``  : Linear ``(hidden -> q_lora_rank)``, BF16.
        2. ``q_a_layernorm`` : RMSNorm over ``q_a_proj`` output.
        3. ``q_b_proj``  : Linear ``(q_lora_rank -> H * (kv_lora_rank +
           qk_rope_head_dim))`` — KV-absorbed; the absorption (kv_b -> q_b)
           is applied by :meth:`process_weights` from the raw HF weights
           stashed at load time, not directly during weight loading.
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
           — the W_UV fusion into o_proj is performed by
           :meth:`process_weights` from the raw HF weights stashed at load
           time (the resulting fused weight is ``(hidden, H * kv_lora_rank)``,
           NOT the HF-native ``(hidden, H * v_head_dim)``).
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
        # q_lora_rank) — filled by process_weights from the stashed raw weights.
        self.q_b_proj_weight = nn.Parameter(
            torch.empty(self.num_heads * self.qk_head_dim, self.q_lora_rank)
        )
        # o_proj (W_UV-fused): (hidden, H * kv_lora_rank). HF native is
        # (hidden, H * v_head_dim); process_weights fuses W_UV in from the
        # stashed raw weights.
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

    # ------------------------------------------------------------------
    def process_weights(self) -> None:
        """Perform MLA weight absorption from the raw HF weights stashed by
        ``DeepseekV3ForCausalLM.load_weights`` (Task C2).

        ``load_weights`` stashes the raw bf16 HF MLA weights as
        ``self._raw_q_b`` / ``self._raw_kv_b`` / ``self._raw_o`` and does NOT
        write ``q_b_proj_weight`` / ``o_proj_weight`` directly. This hook:

          1. Absorbs ``kv_b`` into ``q_b`` -> ``q_b_proj_weight`` of shape
             ``(num_heads * (kv_lora_rank + qk_rope_head_dim), q_lora_rank)``.
          2. Fuses ``W_UV`` (the V half of ``kv_b``) into ``o_proj`` ->
             ``o_proj_weight`` of shape ``(hidden, num_heads * kv_lora_rank)``.

        Math (float32 internally, cast back to bf16) is transcribed from
        ``demo/deepseek_v3/demo_new.py`` (the absorption block) and
        ``demo/deepseek_v3/models/convert.absorb_kv_into_q``.

        Requires ``load_weights`` to have run first; if no raw weights are
        stashed (process_weights called without a prior load) this is a no-op
        for the MLA transform (after recursing into children).
        """
        # Recurse into catalog-leaf children (rope_q / rope_k / kv_gather /
        # decode / reduce) first per the base-class contract. They carry no
        # parameters, so this is a harmless no-op, but it keeps the override
        # consistent with MPKModule.process_weights.
        super().process_weights()

        if not hasattr(self, "_raw_q_b"):
            # No prior load_weights (or already processed); nothing to absorb.
            return

        # HF-native dims. NOTE: ``self.v_head_dim`` was overwritten in
        # __init__ to ``kv_lora_rank`` (the ABSORBED output width); the
        # absorption math needs the HF-native v_head_dim, read from config.
        num_heads = self.num_heads
        qk_nope_head_dim = self.qk_nope_head_dim
        qk_rope_head_dim = self.qk_rope_head_dim
        kv_lora_rank = self.kv_lora_rank
        q_lora_rank = self.q_lora_rank
        v_head_dim = self.config.v_head_dim  # HF-native (e.g. 128), NOT self.v_head_dim

        q_w = self._raw_q_b   # bf16, (num_heads * (qk_nope+qk_rope), q_lora_rank)
        kv_w = self._raw_kv_b  # bf16, (num_heads * (qk_nope+v_head_dim), kv_lora_rank)
        o_w = self._raw_o      # bf16, (hidden, num_heads * v_head_dim)

        # ---- 1. Absorb kv_b into q_b. ----
        absorbed = _absorb_kv_into_q(
            q_w,
            kv_w,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
        )
        assert absorbed.shape == self.q_b_proj_weight.shape, (
            f"absorbed q_b shape {tuple(absorbed.shape)} != "
            f"q_b_proj_weight {tuple(self.q_b_proj_weight.shape)}"
        )
        self.q_b_proj_weight.data.copy_(absorbed.to(torch.bfloat16).contiguous())

        # ---- 2. Fuse W_UV (the V half of kv_b) into o_proj. ----
        # hidden comes from o_proj's own dim-0 (matches demo_new.py).
        hidden = o_w.shape[0]
        W_UV = kv_w.reshape(
            num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
        )[:, qk_nope_head_dim:, :]  # (H, v_head_dim, kv_lora_rank)
        o_fused = torch.einsum(
            "dhn,hnk->dhk",
            o_w.reshape(hidden, num_heads, v_head_dim).float(),
            W_UV.float(),
        )  # (hidden, H, kv_lora_rank)
        o_fused_flat = o_fused.reshape(hidden, num_heads * kv_lora_rank)
        assert o_fused_flat.shape == self.o_proj_weight.shape, (
            f"fused o_proj shape {tuple(o_fused_flat.shape)} != "
            f"o_proj_weight {tuple(self.o_proj_weight.shape)}"
        )
        self.o_proj_weight.data.copy_(
            o_fused_flat.to(torch.bfloat16).contiguous()
        )

        # ---- Free the stashes. ----
        del self._raw_q_b, self._raw_kv_b, self._raw_o


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
        # EP topology comes from the active PersistentKernel when the model is
        # built inside a compile scope (e.g. via build_from_config). The
        # single-GPU demo constructs the model BEFORE any compile scope exists,
        # so current_pk() is unavailable there — fall back to ep_size=1 (no
        # expert parallelism), which is the correct single-GPU behavior. Matches
        # the catalog convention that __init__ does not require a compile scope.
        try:
            pc = current_pk().parallel_config
            self.ep_size = pc.ep_size
            self.ep_rank = pc.ep_rank
        except RuntimeError:
            self.ep_size = 1
            self.ep_rank = 0
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"DeepseekV3MoEMLP: n_routed_experts ({self.num_experts}) % "
                f"ep_size ({self.ep_size}) != 0"
            )
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
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
            local_num_experts=self.num_local_experts,
            local_expert_start=self.local_expert_start,
            prefix=f"{prefix}routing_",
        )

        # ---- Routed experts (catalog leaves own the 3D weight tensors) ----
        self.w13 = MoEW13(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            dtype="bf16",
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
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
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
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
      * Stream raw HF weights through :meth:`load_weights`; the KV absorption
        + W_UV→o_proj fusion run in :meth:`DeepseekV3MLA.process_weights`
        (invoked at the end of ``load_weights``) from the stashed raw weights,
        and routed experts are stacked via their per-expert ``weight_loader``.
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

    # ------------------------------------------------------------------
    # Streaming weight loading (vLLM-style; replaces the per-block
    # _load_from_state_dict path + the demo driver's bulk state_dict build).
    # ------------------------------------------------------------------
    def resolve_weight(self, name, params):
        """Directly-mappable keys only: remap HF -> catalog path, then default.

        Routed-expert keys, the raw MLA q_b/kv_b/o weights, and the router
        ``e_score_correction_bias`` are intercepted by :meth:`load_weights`
        and never reach this method.
        """
        return super().resolve_weight(_remap_dsv3_key(name), params)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Set[str]:
        """Stream raw HF safetensors ``(name, tensor)`` into catalog params.

        Stateful single-pass loop that:
          * SKIPS keys for layers the model did NOT build
            (``model.layers.{i}.*`` with ``i >= len(self.model.layers)`` —
            MTP weights + reduced-layer runs); done early so out-of-range fp8
            weights/scales are never left dangling,
          * dequantizes fp8 weights inline by pairing each weight with its
            ``<name>_scale_inv`` partner (either order of arrival),
          * routes routed-expert weights through the EP-aware per-expert
            ``weight_loader`` (non-local experts are skipped, never stored),
          * STASHES the raw ``q_b_proj`` / ``kv_b_proj`` / ``o_proj`` weights on
            the owning MLA module for the later MLA-absorption step
            (``process_weights``; Task C3) — they are NOT written into params
            here,
          * copies the router ``e_score_correction_bias`` straight from its
            raw (native fp32) tensor to ``routing.bias`` (no bf16 round-trip),
          * writes every other (directly-mappable) param via
            :meth:`resolve_weight` + ``param.data.copy_`` in bf16.

        Returns the set of HF keys consumed (including skipped non-local
        expert weights and their scales).
        """
        params = dict(self.named_parameters())
        consumed: Set[str] = set()
        built_layers = len(self.model.layers)

        # fp8 pairing buffers: a weight and its '<name>_scale_inv' partner may
        # arrive in either order. ``fp8_pending`` holds weights awaiting a
        # scale; ``scale_pending`` holds scales awaiting their weight (keyed by
        # the BASE weight name, i.e. with the '_scale_inv' suffix stripped).
        fp8_pending: dict = {}
        scale_pending: dict = {}

        def _finalize_and_route(wname: str, w: torch.Tensor) -> None:
            """Cast/route a fully-dequantized (or already-bf16) tensor."""
            self._route_weight(wname, w, params, consumed)

        for name, tensor in weights:
            # Skip keys for layers the model did NOT build (MTP layer
            # ``model.layers.<num_hidden_layers>.*`` + reduced-layer runs).
            # Done EARLY — before fp8 buffering and expert/MLA routing — so
            # an out-of-range fp8 weight/scale is never left dangling in the
            # pairing buffers. Covers both weight keys and their
            # ``_scale_inv`` companions (same ``model.layers.{i}.`` prefix).
            if _is_out_of_range_layer_key(name, built_layers):
                consumed.add(name)
                continue

            if name.endswith("_scale_inv"):
                base = name[: -len("_scale_inv")]
                # If the scale belongs to a NON-LOCAL routed expert, the weight
                # was never buffered — just consume the scale and move on.
                ek = _parse_expert_key(base)
                if ek is not None and not self._expert_is_local(ek[0], ek[1]):
                    consumed.add(name)
                    continue
                if base in fp8_pending:
                    w_fp8 = fp8_pending.pop(base)
                    w = dequantize_fp8(w_fp8, tensor, target_dtype=torch.bfloat16)
                    _finalize_and_route(base, w)
                    consumed.add(name)
                else:
                    scale_pending[base] = tensor
                    consumed.add(name)
                continue

            # A weight tensor (not a scale).
            # Non-local routed experts: skip entirely (do not buffer fp8, do
            # not store). Its scale (if any) is consumed in the branch above.
            ek = _parse_expert_key(name)
            if ek is not None and not self._expert_is_local(ek[0], ek[1]):
                consumed.add(name)
                continue

            if is_fp8(tensor):
                if name in scale_pending:
                    scale = scale_pending.pop(name)
                    w = dequantize_fp8(tensor, scale, target_dtype=torch.bfloat16)
                    _finalize_and_route(name, w)
                    consumed.add(name)
                else:
                    fp8_pending[name] = tensor
                continue

            # Router e_score_correction_bias: route the RAW (native fp32)
            # tensor to fp32 BEFORE the bf16 cast below, avoiding a lossy
            # F32->bf16->F32 round-trip. The checkpoint stores this bias as
            # F32 and the routing.bias param is fp32; fp8 dequant (above)
            # never applies to it, so ``tensor`` is its native fp32 here.
            if name.endswith(".mlp.gate.e_score_correction_bias"):
                layer_idx = self._layer_idx_from_key(name)
                moe = self._layer_mlp(layer_idx)
                if moe is None:
                    raise ValueError(
                        "DeepseekV3ForCausalLM.load_weights: "
                        "e_score_correction_bias for non-MoE/oob layer in key "
                        f"{name!r}"
                    )
                moe.routing.bias.data.copy_(tensor.to(torch.float32))
                consumed.add(name)
                continue

            # Already a normal float dtype — cast to bf16 and route now.
            _finalize_and_route(name, tensor.to(torch.bfloat16))
            consumed.add(name)

        if fp8_pending or scale_pending:
            raise ValueError(
                "DeepseekV3ForCausalLM.load_weights: unpaired fp8 tensors at "
                f"end of stream. weights awaiting scale: "
                f"{sorted(fp8_pending)}; scales awaiting weight: "
                f"{sorted(scale_pending)}"
            )

        # MLA absorption (q_b/kv_b -> q_b, W_UV -> o_proj) runs in
        # process_weights (Task C3): it consumes the raw weights stashed above
        # (_raw_q_b / _raw_kv_b / _raw_o) and fills q_b_proj_weight /
        # o_proj_weight.
        self.process_weights()
        # q_b_proj_weight and o_proj_weight are filled by process_weights (MLA
        # absorption), not directly here, so the base completeness check
        # (_assert_fully_loaded) is intentionally skipped.
        return consumed

    # ------------------------------------------------------------------
    def _layer_mlp(self, layer_idx: int):
        """Return the (DeepseekV3MoEMLP) mlp for ``layer_idx``, or None if dense/OOB."""
        layers = self.model.layers
        if not (0 <= layer_idx < len(layers)):
            return None
        layer = layers[layer_idx]
        if not getattr(layer, "is_moe", False):
            return None
        return layer.mlp

    def _expert_is_local(self, layer_idx: int, expert_id: int) -> bool:
        """Whether ``expert_id`` is owned by this rank for the given MoE layer.

        A routed-expert key for a dense layer / out-of-range layer should not
        occur, but if it does we treat it as non-local (skip) rather than crash.
        """
        moe = self._layer_mlp(layer_idx)
        if moe is None:
            return False
        local = expert_id - moe.local_expert_start
        return 0 <= local < moe.num_local_experts

    def _route_weight(self, name, w, params, consumed) -> None:
        """Route a finalized bf16 tensor ``w`` for HF key ``name``."""
        # ---- Routed experts -> EP-aware per-expert weight_loader. ----
        ek = _parse_expert_key(name)
        if ek is not None:
            layer_idx, expert_id, proj = ek
            moe = self._layer_mlp(layer_idx)
            # locality was already checked before we got here (non-local were
            # skipped); moe is therefore not None.
            if proj == "down":
                moe.w2.weight_loader(moe.w2.weight, w, expert_id=expert_id)
            else:
                moe.w13.weight_loader(
                    moe.w13.weight, w, expert_id=expert_id, slot=proj
                )
            consumed.add(name)
            return

        # ---- Raw MLA weights stashed for process_weights (C3). ----
        if name.endswith(".self_attn.q_b_proj.weight"):
            self._stash_mla(name, ".self_attn.q_b_proj.weight", "_raw_q_b", w)
            consumed.add(name)
            return
        if name.endswith(".self_attn.kv_b_proj.weight"):
            self._stash_mla(name, ".self_attn.kv_b_proj.weight", "_raw_kv_b", w)
            consumed.add(name)
            return
        if name.endswith(".self_attn.o_proj.weight"):
            self._stash_mla(name, ".self_attn.o_proj.weight", "_raw_o", w)
            consumed.add(name)
            return

        # (Router e_score_correction_bias is routed directly from its raw fp32
        # tensor in load_weights, before the bf16 cast — never reaches here.)

        # ---- Directly-mappable params: remap + default resolve + copy_. ----
        res = self.resolve_weight(name, params)
        if res is None:
            raise ValueError(
                "DeepseekV3ForCausalLM.load_weights: unexpected checkpoint key "
                f"{name!r} (no matching parameter)."
            )
        if res is SKIP_WEIGHT:
            consumed.add(name)
            return
        param, loader, kwargs = res
        call_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        if loader is not None:
            loader(param, w, **call_kwargs)
        else:
            param.data.copy_(w)
        consumed.add(name)

    @staticmethod
    def _layer_idx_from_key(name: str) -> int:
        m = re.match(r"^model\.layers\.(\d+)\.", name)
        if m is None:
            raise ValueError(f"cannot parse layer index from key {name!r}")
        return int(m.group(1))

    def _stash_mla(self, name, suffix, attr, w) -> None:
        """Stash raw MLA weight ``w`` on the owning DeepseekV3MLA module."""
        layer_idx = self._layer_idx_from_key(name)
        layers = self.model.layers
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(
                f"DeepseekV3ForCausalLM.load_weights: MLA key {name!r} for "
                f"out-of-range layer {layer_idx}"
            )
        setattr(layers[layer_idx].self_attn, attr, w)
