"""DeepSeek V3 PyTorch reference modules.

vLLM-aligned plain-PyTorch implementation. Each module's docstring
cites the vLLM source it was aligned to. **Do not edit the math**
without independently re-validating against vLLM.

Numerical notes:

- RMSNorm computes variance in float32 regardless of input dtype, then
  casts back. This matches vLLM's `forward_static` in
  `vllm/model_executor/layers/layernorm.py:202-235` and HuggingFace's
  reference. Skipping the float32 cast loses ~1e-3 cosine on long
  contexts.

- MLA softmax scale is `(1/sqrt(qk_head_dim)) * mscale**2` — note the
  square. See `vllm/model_executor/models/deepseek_v2.py:889,966`.

- MTP `eh_proj` is a single Linear over the concatenation
  `[enorm(embed); hnorm(hidden)]` — NOT two parallel matmuls summed.
  See `vllm/model_executor/models/deepseek_mtp.py:96-118`. (Two parallel
  matmuls is mathematically equivalent — `Linear([x;y])` factorises into
  `x @ W[:H]^T + y @ W[H:]^T` — but the reference uses the literal
  vLLM form so weight loading is straightforward.)

- MoE routing is sigmoid scoring + e_score_correction_bias + grouped
  topk (n_group=8, topk_group=4, num_experts_per_tok=8) with topk-prob
  renormalization and routed_scaling_factor=2.5. See
  `vllm/model_executor/models/deepseek_v2.py:235-398` and DeepSeek V3's
  config.json (scoring_func="sigmoid", topk_method="noaux_tc").
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .config import Config


# =============================================================================
# RMSNorm
# =============================================================================
class RMSNorm(nn.Module):
    """DeepSeek V2/V3 RMSNorm.

    Aligned with vLLM's `forward_static` at
    `vllm/model_executor/layers/layernorm.py:202-235`.

    Forward:
        residual_optional input — fused into the input before norm
        var = mean(x^2, dim=-1)   # float32
        out = x * rsqrt(var + eps)
        out = (out * weight).to(orig_dtype)

    The float32 cast is load-bearing for long contexts. Float16/BF16
    accumulation in `mean(x^2)` accumulates significant error.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Normalise.

        If `residual` is given, the layer behaves as the fused
        residual-add+norm op vLLM uses (the decoder layer's pattern).
        Returns `(normed, new_residual)` in that case so the caller
        can pass `new_residual` to the *next* fused layer norm.
        """
        orig_dtype = hidden_states.dtype
        if residual is not None:
            hidden_states = hidden_states + residual
            residual_out = hidden_states  # to feed into the next fused norm
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        out = (x_fp32 * self.weight.to(torch.float32)).to(orig_dtype)
        if residual is not None:
            return out, residual_out
        return out


# =============================================================================
# RoPE — YaRN-extended for DeepSeek V3
# =============================================================================

def yarn_get_mscale(scale: float, mscale: float) -> float:
    """vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:20-23."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """`yarn_find_correction_dim` from the YaRN paper appendix.

    See `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:36-44`.
    """
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> Tuple[int, int]:
    """`yarn_find_correction_range` — clamps to [0, dim-1]."""
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype) -> Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


class DeepseekYarnRotaryEmbedding(nn.Module):
    """YaRN-extended RoPE for DeepSeek V3.

    Aligned with
    `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:1-153`.

    Inputs:
        positions: [T] int — token absolute positions
        q_pe: [T, H, qk_rope_head_dim] — Q rope-side slice
        k_pe: [T, 1, qk_rope_head_dim] — K rope-side slice (MLA: shared across heads)

    Returns rotated `(q_pe, k_pe)`.

    Only the `qk_rope_head_dim` slice is rotated; nope/v dims pass
    through unchanged elsewhere.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.dim = cfg.qk_rope_head_dim
        self.base = cfg.rope_theta
        self.scaling_factor = float(cfg.rope_scaling["factor"])
        self.mscale_base = float(cfg.rope_scaling["mscale"])
        self.mscale_all_dim = float(cfg.rope_scaling["mscale_all_dim"])
        self.beta_fast = float(cfg.rope_scaling["beta_fast"])
        self.beta_slow = float(cfg.rope_scaling["beta_slow"])
        self.original_max_position_embeddings = int(
            cfg.rope_scaling["original_max_position_embeddings"]
        )
        self.max_position_embeddings = (
            self.original_max_position_embeddings * int(self.scaling_factor)
        )

        # Build inv_freq with the YaRN linear-ramp mask between
        # extrapolation and interpolation. See
        # `deepseek_scaling_rope.py:71-100`.
        base_freqs = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
        )
        inv_freq_extrapolation = 1.0 / base_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * base_freqs)
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        ramp_mask = 1 - yarn_linear_ramp_mask(low, high, self.dim // 2, torch.float32)
        inv_freq = (
            inv_freq_interpolation * (1 - ramp_mask)
            + inv_freq_extrapolation * ramp_mask
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # The mscale factor multiplies cos/sin; in attention scale it
        # appears as `mscale**2` (see deepseek_v2.py:966).
        mscale = float(yarn_get_mscale(self.scaling_factor, self.mscale_base) /
                       yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))
        self.attn_mscale = mscale  # consumer reads this
        # Pre-compute cos/sin cache.
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        # Concat to match (cos | sin) layout used by GPTJ-style rotation
        # (vllm rotary uses real-imag interleave or cat-half; vLLM's
        # deepseek_scaling_rope.py uses cat-half via `_rotate_gptj`).
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _rotate_gptj(x: Tensor) -> Tensor:
        """Mirror of vLLM's `_rotate_gptj` in the cuda kernel.

        Treats consecutive pairs (x_even, x_odd) as a complex number
        and rotates by 90°: (x_even, x_odd) → (-x_odd, x_even).
        """
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1)
        return rotated.flatten(-2)

    def forward(
        self, positions: Tensor, q_pe: Tensor, k_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply YaRN-RoPE to the rope-dim slices of Q and K.

        positions: [T]
        q_pe: [T, H, rope_dim]
        k_pe: [T, 1, rope_dim]
        """
        cache = self.cos_sin_cache.to(q_pe.device)
        # cache layout: [..., 2*half_dim] where first half is cos, second is sin
        half = self.dim // 2
        cos = cache[positions, :half].repeat_interleave(2, dim=-1).to(q_pe.dtype)
        sin = cache[positions, half:].repeat_interleave(2, dim=-1).to(q_pe.dtype)
        cos = cos.unsqueeze(-2)  # [T, 1, rope_dim]
        sin = sin.unsqueeze(-2)
        q_rotated = q_pe * cos + self._rotate_gptj(q_pe) * sin
        k_rotated = k_pe * cos + self._rotate_gptj(k_pe) * sin
        return q_rotated, k_rotated


# =============================================================================
# MLA attention (unabsorbed form, matches vLLM)
# =============================================================================
class DeepseekV2MLAAttention(nn.Module):
    """Multi-Head Latent Attention.

    Aligned with vLLM's `DeepseekV2MLAAttention` in
    `vllm/model_executor/models/deepseek_v2.py:847-1032`.

    Forward (per the cited lines 543-580):

        # 1. Q LoRA
        q_c = q_a_proj(hidden_states)            # [T, q_lora_rank]
        q_c = q_a_layernorm(q_c)
        q   = q_b_proj(q_c)                      # [T, H * qk_head_dim]
        q   = q.view(T, H, qk_head_dim)
        q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], -1)

        # 2. KV LoRA
        kv = kv_a_proj_with_mqa(h)               # [T, kv_lora_rank + qk_rope_head_dim]
        kv_c, k_pe = kv.split([kv_lora_rank, qk_rope_head_dim], -1)
        kv_c = kv_a_layernorm(kv_c)
        kv   = kv_b_proj(kv_c)                   # [T, H * (qk_nope_head_dim + v_head_dim)]
        kv   = kv.view(T, H, qk_nope_head_dim + v_head_dim)
        k_nope, v = kv.split([qk_nope_head_dim, v_head_dim], -1)
        k_pe = k_pe.unsqueeze(1)                 # [T, 1, qk_rope_head_dim]

        # 3. RoPE on the rope-dim slice
        q_pe, k_pe = rope(positions, q_pe, k_pe)

        # 4. Recompose K and pad V
        k = cat([k_nope, k_pe.expand(-1, H, -1)], dim=-1)   # [T, H, qk_head_dim]
        v = pad(v, [0, qk_head_dim - v_head_dim])           # [T, H, qk_head_dim]

        # 5. Attention with softmax_scale = (1/sqrt(qk_head_dim)) * mscale^2
        attn = scaled_dot_product_attention(q, k, v, scale=softmax_scale, is_causal=True)
        attn = attn[..., :v_head_dim]                       # discard padding
        attn = attn.reshape(T, H * v_head_dim)

        # 6. Output projection
        return o_proj(attn)

    The reference always does the full unabsorbed math (no W_UK fold-in).
    The "absorbed" variant MPK uses for decode is a fused-Q form that's
    mathematically equivalent (q_nope_512_abs = W_UK^T @ q_nope_128) and
    produces the same hidden_state output up to floating-point rounding.
    """

    def __init__(self, cfg: Config, rope: DeepseekYarnRotaryEmbedding):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        H = cfg.num_attention_heads

        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, H * cfg.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, eps=cfg.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank,
            H * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(H * cfg.v_head_dim, cfg.hidden_size, bias=False)

        # softmax scale incorporates YaRN's mscale^2.
        self.softmax_scale = (1.0 / math.sqrt(cfg.qk_head_dim)) * (
            rope.attn_mscale * rope.attn_mscale
        )

    def forward(
        self,
        positions: Tensor,         # [T]
        hidden_states: Tensor,     # [T, hidden_size]
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        cfg = self.cfg
        T = hidden_states.shape[0]
        H = cfg.num_attention_heads

        # Q path (LoRA + b_proj + reshape + split nope/pe)
        q_c = self.q_a_proj(hidden_states)
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c).view(T, H, cfg.qk_head_dim)
        q_nope = q[..., : cfg.qk_nope_head_dim]
        q_pe = q[..., cfg.qk_nope_head_dim :]

        # KV path (LoRA + split kv_c/k_pe + b_proj over kv_c)
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_c = kv_a[..., : cfg.kv_lora_rank]
        k_pe = kv_a[..., cfg.kv_lora_rank :].unsqueeze(1)  # [T, 1, qk_rope_head_dim]
        kv_c = self.kv_a_layernorm(kv_c)
        kv = self.kv_b_proj(kv_c).view(
            T, H, cfg.qk_nope_head_dim + cfg.v_head_dim
        )
        k_nope = kv[..., : cfg.qk_nope_head_dim]
        v = kv[..., cfg.qk_nope_head_dim :]  # [T, H, v_head_dim]

        # RoPE on the rope-dim slices only.
        q_pe, k_pe = self.rope(positions, q_pe, k_pe)

        # Recompose K (nope || pe) and broadcast pe across heads.
        k_pe = k_pe.expand(-1, H, -1)
        k = torch.cat([k_nope, k_pe], dim=-1)  # [T, H, qk_head_dim]
        q_full = torch.cat([q_nope, q_pe], dim=-1)  # [T, H, qk_head_dim]

        # Pad V to qk_head_dim so the SDPA kernel can consume Q/K/V with
        # matching head dims (vLLM does the same trick — pad here, slice
        # back to v_head_dim after attention).
        v_padded = F.pad(v, (0, cfg.qk_head_dim - cfg.v_head_dim))

        # Run causal scaled-dot-product attention. Math contract:
        #   attn = softmax(Q K^T * softmax_scale + mask) @ V
        # is_causal=True applies the standard upper-triangular mask. For
        # MTP/decode where T=1, causal mask is a no-op.
        # Permute to [H, T, D] for SDPA's expected (H, T, D) layout.
        q_t = q_full.permute(1, 0, 2)         # [H, T, qk_head_dim]
        k_t = k.permute(1, 0, 2)              # [H, T, qk_head_dim]
        v_t = v_padded.permute(1, 0, 2)       # [H, T, qk_head_dim]
        attn = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=attention_mask,
            is_causal=(attention_mask is None),
            scale=self.softmax_scale,
        )                                      # [H, T, qk_head_dim]
        attn = attn.permute(1, 0, 2)           # [T, H, qk_head_dim]
        attn = attn[..., : cfg.v_head_dim]     # [T, H, v_head_dim]
        attn = attn.reshape(T, H * cfg.v_head_dim)
        return self.o_proj(attn)


# =============================================================================
# Dense MLP — used for layers 0..first_k_dense_replace-1
# =============================================================================
class DeepseekV2DenseMLP(nn.Module):
    """Dense MLP (gate_up + down with silu).

    Aligned with `vllm/model_executor/models/deepseek_v2.py:115-173`
    (the non-MoE branch of DeepseekV2MLP).

    Forward:
        gate_up = gate_up_proj(x)              # [T, 2 * intermediate]
        gate, up = gate_up.chunk(2, dim=-1)
        x = silu(gate) * up
        return down_proj(x)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            cfg.hidden_size, 2 * cfg.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            cfg.intermediate_size, cfg.hidden_size, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# =============================================================================
# MoE — DeepSeek V3's sigmoid + correction-bias + grouped-topk routing
# =============================================================================
class DeepseekV3MoE(nn.Module):
    """Routed experts + shared expert MoE.

    Aligned with `vllm/model_executor/models/deepseek_v2.py:235-398`
    (DeepseekV2MoE) and the FusedMoE select_experts logic in
    `vllm/model_executor/layers/fused_moe/fused_moe.py:select_experts`
    for the `noaux_tc` topk_method + sigmoid scoring.

    Forward (cited lines 348-398):
        router_logits = gate(x)                                  # [T, n_routed]
        scores = sigmoid(router_logits)                          # scoring_func=sigmoid
        scores_with_bias = scores + e_score_correction_bias      # noaux_tc

        # Grouped topk: divide n_routed_experts into n_group groups,
        # pick `topk_group` groups (by sum of top-2 expert scores per
        # group), then within those groups pick top-`num_experts_per_tok`
        # experts overall. Weights come from the un-biased scores.
        group_scores = scores_with_bias.view(T, n_group, n_per_grp)
                       .topk(2, -1).values.sum(-1)               # [T, n_group]
        group_idx = group_scores.topk(topk_group, -1).indices    # [T, topk_group]

        # Mask out experts not in the selected groups.
        mask = group_mask_from_idx(group_idx)                    # [T, n_routed]
        masked = scores_with_bias.masked_fill(~mask, -inf)
        topk_idx = masked.topk(num_experts_per_tok, -1).indices  # [T, topk]
        topk_weights = scores.gather(-1, topk_idx)               # un-biased

        if norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        topk_weights = topk_weights * routed_scaling_factor      # = 2.5

        # Apply experts (each is gate_up_proj + down_proj)
        out = sum_over_topk(weight_e * expert_e(x))

        # Shared expert
        out = out + shared_expert(x)
        return out
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Gate (router scoring)
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False)
        # noaux_tc requires a per-expert correction bias added before topk
        self.gate_e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts)
        )

        # Routed experts (each is a dense MLP with moe_intermediate_size)
        self.experts = nn.ModuleList(
            [
                _MoEExpert(cfg.hidden_size, cfg.moe_intermediate_size)
                for _ in range(cfg.n_routed_experts)
            ]
        )
        # Shared expert (n_shared_experts copies of the same architecture
        # in DeepSeek V3 — but the published config uses 1 shared expert).
        self.shared_experts = _MoEExpert(
            cfg.hidden_size,
            cfg.moe_intermediate_size * cfg.n_shared_experts,
        )

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        T = x.shape[0]

        # 1. Score router logits (sigmoid activation per scoring_func).
        router_logits = self.gate(x.to(self.gate.weight.dtype)).to(torch.float32)
        if cfg.scoring_func == "sigmoid":
            scores = router_logits.sigmoid()
        elif cfg.scoring_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        else:
            raise ValueError(f"Unknown scoring_func={cfg.scoring_func}")

        # 2. Add correction bias before topk (noaux_tc).
        scores_for_topk = scores + self.gate_e_score_correction_bias.to(
            scores.dtype
        )

        # 3. Grouped topk.
        n_per_grp = cfg.n_routed_experts // cfg.n_group
        group_scores = scores_for_topk.view(T, cfg.n_group, n_per_grp)
        # Sum of top-2 per group is the group score (DeepSeek V3 specific).
        group_score = group_scores.topk(2, dim=-1).values.sum(dim=-1)
        # Pick top `topk_group` groups.
        group_idx = group_score.topk(cfg.topk_group, dim=-1).indices  # [T, topk_group]
        # Build a mask covering experts in selected groups.
        group_mask = torch.zeros_like(group_score)                    # [T, n_group]
        group_mask.scatter_(1, group_idx, 1.0)
        expert_mask = (
            group_mask.unsqueeze(-1)                                  # [T, n_group, 1]
            .expand(-1, -1, n_per_grp)
            .reshape(T, cfg.n_routed_experts)
            .bool()
        )
        masked_scores = scores_for_topk.masked_fill(~expert_mask, float("-inf"))

        # 4. topk over remaining experts.
        topk_vals, topk_idx = masked_scores.topk(
            cfg.num_experts_per_tok, dim=-1
        )                                              # [T, topk]
        # Use UN-biased scores as weights (norm before scaling per
        # vLLM's `select_experts` for noaux_tc).
        topk_weights = scores.gather(-1, topk_idx)
        if cfg.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights * cfg.routed_scaling_factor
        topk_weights = topk_weights.to(x.dtype)

        # 5. Apply experts. Naive loop over experts — slow but exact.
        # Acceptable for a reference; production uses grouped GEMM.
        routed_out = torch.zeros_like(x)
        for e in range(cfg.n_routed_experts):
            # Tokens that route to expert e (any of the topk slots).
            mask = (topk_idx == e)                     # [T, topk]
            if not mask.any():
                continue
            # Per-token weight for expert e (sum if a token chose e in
            # multiple slots; topk distinct so at most one).
            w = (mask.float() * topk_weights).sum(dim=-1, keepdim=True)  # [T, 1]
            sel = w.squeeze(-1) > 0
            if not sel.any():
                continue
            x_sel = x[sel]
            y = self.experts[e](x_sel) * w[sel]
            routed_out[sel] = routed_out[sel] + y.to(routed_out.dtype)

        # 6. Shared expert addition.
        shared_out = self.shared_experts(x)
        return routed_out + shared_out


class _MoEExpert(nn.Module):
    """Single expert = gate_up_proj + silu*up + down_proj.

    Same shape as DenseMLP but with `moe_intermediate_size` instead of
    `intermediate_size`. vLLM stores routed experts as a stack
    `experts.w13.weight` `[E, 2*I, H]` and `experts.w2.weight` `[E, H, I]`;
    the reference uses one `_MoEExpert` per-expert (slow but explicit).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# =============================================================================
# Decoder layer — input_layernorm → self_attn → residual → post_attn_layernorm
# → mlp → residual
# =============================================================================
class DeepseekV2DecoderLayer(nn.Module):
    """Single decoder block.

    Aligned with `vllm/model_executor/models/deepseek_v2.py:1119-1166`.

    Forward order (verbatim from cited lines):
        if residual is None:
            residual = hidden_states
            hidden_states = input_layernorm(hidden_states)
        else:
            hidden_states, residual = input_layernorm(hidden_states, residual)
        hidden_states = self_attn(positions, hidden_states)
        hidden_states, residual = post_attention_layernorm(hidden_states, residual)
        hidden_states = mlp(hidden_states)
        return hidden_states, residual

    The fused (norm + residual-add) pattern is critical — both calls to
    the layernorm class consume `residual` as an in-place input. This
    matches what MPK does at the megakernel level.
    """

    def __init__(
        self,
        cfg: Config,
        layer_idx: int,
        rope: DeepseekYarnRotaryEmbedding,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = DeepseekV2MLAAttention(cfg, rope)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if layer_idx < cfg.first_k_dense_replace:
            self.mlp = DeepseekV2DenseMLP(cfg)
        else:
            self.mlp = DeepseekV3MoE(cfg)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        residual: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # Input layernorm (fused with prior residual).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # Self attention.
        hidden_states = self.self_attn(positions, hidden_states)
        # Post-attention layernorm (fused with attn-output + prior residual).
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        # MLP.
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# =============================================================================
# MTP layer — vLLM's DeepSeek MTP head
# =============================================================================
class DeepseekV3MTPLayer(nn.Module):
    """Multi-Token-Prediction head.

    Aligned with `vllm/model_executor/models/deepseek_mtp.py:60-118`.

    Forward (cited lines 96-118):
        # 1. Mask embeds at position 0 (the start-of-sequence sentinel).
        inputs_embeds = where(positions == 0, 0, inputs_embeds)
        # 2. Per-input norms.
        e = enorm(inputs_embeds)                                # [T, H]
        h = hnorm(previous_hidden_states)                       # [T, H]
        # 3. Project the concatenation through eh_proj.
        x = eh_proj(cat([e, h], dim=-1))                        # [T, H]
        # 4. Run a full MTP decoder block (same architecture as a main
        #    DeepSeek V3 layer) starting fresh (residual=None).
        x, residual = mtp_block(positions, x, None)
        # 5. Add residual and feed to shared head.
        x = x + residual
        return x   # caller passes through shared_head.norm + lm_head

    Note on `eh_proj`: the literal vLLM form is a single `Linear` over
    the concatenation (input dim 2*H → output H). MPK in
    `python/mirage/mpk/models/deepseek_v3/builder.py:2814-2825` splits
    the same weight tensor into two halves W1 (rows :H of W^T) and W2
    (rows H:) and computes `W1 @ e + W2 @ h`. The two are
    mathematically equivalent (`Linear([x;y]) = x @ W[:,:H]^T + y @ W[:,H:]^T`)
    so the reference uses the literal vLLM form for clarity — weight
    loading must split (or unsplit) accordingly.
    """

    def __init__(self, cfg: Config, rope: DeepseekYarnRotaryEmbedding):
        super().__init__()
        self.enorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.hnorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        # The MTP decoder block reuses the architecture of a main MoE
        # layer (per the published DeepSeek V3 checkpoint). first_k_dense_replace
        # is bypassed by setting layer_idx >= first_k_dense_replace.
        self.mtp_block = DeepseekV2DecoderLayer(
            cfg, layer_idx=cfg.num_hidden_layers, rope=rope,
        )
        # Shared head's norm — applied to MTP output before the shared
        # lm_head for logits.
        self.shared_head_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        positions: Tensor,            # [T]
        input_ids_embed: Tensor,      # [T, H] — embed(target_token_at_pos+1) per vLLM
        previous_hidden_states: Tensor,  # [T, H] — target's last-layer hidden
    ) -> Tensor:
        # Mask SOS position's embed (vLLM: `where(positions == 0, 0, ...)`).
        # In our reference we pass already-shifted token IDs so positions==0
        # is the actual zero-index sample; mask accordingly.
        mask = (positions != 0).to(input_ids_embed.dtype).unsqueeze(-1)
        input_ids_embed = input_ids_embed * mask

        e = self.enorm(input_ids_embed)
        h = self.hnorm(previous_hidden_states)
        x = self.eh_proj(torch.cat([e, h], dim=-1))

        x, residual = self.mtp_block(positions, x, residual=None)
        # Add residual then feed through shared_head.norm.
        x = x + residual
        x = self.shared_head_norm(x)
        return x


# =============================================================================
# Top-level model
# =============================================================================
class DeepseekV3Model(nn.Module):
    """Top-level DeepSeek V3 (decoder + MTP) for reference.

    Aligned with `DeepseekV2ForCausalLM` in
    `vllm/model_executor/models/deepseek_v2.py` and the MTP wiring in
    `vllm/model_executor/models/deepseek_mtp.py`.

    Args:
        cfg: model config.
        layer_indices: which main-model layer indices to actually
            instantiate. The rest are skipped (the reference runs only
            over these layers, mirroring MPK's `--layers 0-3` mode).
        enable_mtp: if True, also build and run the MTP head.

    Forward returns a `dict` of named tensors (intermediate hidden
    states + final argmax) for downstream comparison with MPK.
    """

    def __init__(
        self,
        cfg: Config,
        layer_indices: Optional[list[int]] = None,
        enable_mtp: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_indices = (
            list(layer_indices) if layer_indices is not None
            else list(range(cfg.num_hidden_layers))
        )
        self.enable_mtp = enable_mtp

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = DeepseekYarnRotaryEmbedding(cfg)

        # Build only the requested layers — keys are the original layer
        # indices, values are the modules. Missing indices mean those
        # layers are skipped at forward (MPK does the same).
        self.layers = nn.ModuleDict(
            {
                str(i): DeepseekV2DecoderLayer(cfg, i, self.rope)
                for i in self.layer_indices
            }
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # Tie embed and lm_head per typical HF convention; weight loader
        # may overwrite if the checkpoint has a separate lm_head.
        self.lm_head.weight = self.embed_tokens.weight

        if enable_mtp:
            # Single MTP layer per DeepSeek V3 (num_nextn_predict_layers=1).
            self.mtp_layer = DeepseekV3MTPLayer(cfg, self.rope)

    def forward(
        self,
        input_ids: Tensor,        # [T]
        positions: Tensor,        # [T]
        prev_mtp_input_ids: Optional[Tensor] = None,
        # ^ token IDs to feed MTP's embed input — typically a shift-by-1
        # of the target token IDs (so MTP at position p sees embed(t_{p+1})).
        # See `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py:374-410`.
    ) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}

        x = self.embed_tokens(input_ids)
        out["embed"] = x.detach().clone()

        residual: Optional[Tensor] = None
        for li in self.layer_indices:
            layer = self.layers[str(li)]
            x, residual = layer(positions, x, residual)
            out[f"layer_{li}_output"] = x.detach().clone()
            out[f"layer_{li}_residual"] = residual.detach().clone()

        # Final norm of main model.
        # Mirror vLLM's pattern: the last layer returns (x, residual)
        # with residual NOT yet added; the final norm fuses the residual.
        x_main, _ = self.norm(x, residual=residual)
        out["final_norm"] = x_main.detach().clone()

        logits = self.lm_head(x_main)
        out["logits"] = logits.detach().clone()
        argmax = logits.argmax(dim=-1)
        out["argmax"] = argmax.detach().clone()

        if self.enable_mtp:
            assert prev_mtp_input_ids is not None, (
                "MTP forward needs prev_mtp_input_ids (shifted target tokens)"
            )
            # MTP's previous_hidden_states is the target's final hidden
            # state (`x_main` after the final norm) per
            # `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py:181`
            # which passes `hidden_states[:num_tokens]` from the target's
            # forward.
            mtp_embed = self.embed_tokens(prev_mtp_input_ids)
            mtp_out = self.mtp_layer(
                positions=positions,
                input_ids_embed=mtp_embed,
                previous_hidden_states=x_main,
            )
            out["mtp_output"] = mtp_out.detach().clone()
            mtp_logits = self.lm_head(mtp_out)
            out["mtp_logits"] = mtp_logits.detach().clone()
            out["mtp_argmax"] = mtp_logits.argmax(dim=-1).detach().clone()

        return out
