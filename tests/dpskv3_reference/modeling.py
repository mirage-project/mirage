"""DeepSeek V3 PyTorch reference modules with TP + EP support.

vLLM-aligned plain-PyTorch implementation. Each module's docstring
cites the vLLM source it was aligned to. **Do not edit the math**
without independently re-validating against vLLM.

TP/EP semantics (when `parallel_config.tp_size > 1`):

  - Attention: heads split across `tp_size`; q_b_proj/kv_b_proj
    Column-parallel; o_proj Row-parallel (all-reduces).
  - Dense MLP: gate_up_proj Column-parallel; down_proj Row-parallel.
  - MoE routed experts: split across EP groups (`ep_size` groups);
    within each EP group, each expert is TP-sharded by
    `routed_tp_size = tp_size / ep_size`. The MoE forward gates over
    ALL experts (router is replicated), but only computes for THIS
    rank's local experts; one global AllReduce combines.
  - MoE shared experts: TP-sharded by full `tp_size` (gate_up Column,
    down Row).
  - Embed / lm_head: replicated (small enough).

Numerical notes:

- RMSNorm computes variance in float32 regardless of input dtype, then
  casts back. Matches `vllm/model_executor/layers/layernorm.py:202-235`.

- MLA softmax scale is `(1/sqrt(qk_head_dim)) * mscale**2` — note the
  square. See `vllm/model_executor/models/deepseek_v2.py:889,966`.

- MTP `eh_proj` is a single Linear over the concatenation
  `[enorm(embed); hnorm(hidden)]` — NOT two parallel matmuls summed.
  See `vllm/model_executor/models/deepseek_mtp.py:96-118`.

- MoE routing is sigmoid scoring + e_score_correction_bias + grouped
  topk (n_group=8, topk_group=4, num_experts_per_tok=8) with topk-prob
  renormalization and routed_scaling_factor=2.5.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .config import Config
from .parallel import (
    ParallelConfig,
    ColumnParallelLinear,
    RowParallelLinear,
    RoutedExpertColumnParallel,
    RoutedExpertRowParallel,
    all_reduce_tp,
)


# =============================================================================
# RMSNorm (parallel-agnostic — operates on hidden_size which is replicated)
# =============================================================================
class RMSNorm(nn.Module):
    """DeepSeek V2/V3 RMSNorm.

    Aligned with vLLM's `forward_static` at
    `vllm/model_executor/layers/layernorm.py:202-235`.

    The float32 cast is load-bearing for long contexts (BF16 accumulation
    of `mean(x^2)` accumulates significant error).
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
        orig_dtype = hidden_states.dtype
        if residual is not None:
            hidden_states = hidden_states + residual
            residual_out = hidden_states
        x_fp32 = hidden_states.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        out = (x_fp32 * self.weight.to(torch.float32)).to(orig_dtype)
        if residual is not None:
            return out, residual_out
        return out


# =============================================================================
# YaRN-extended RoPE (parallel-agnostic)
# =============================================================================

def yarn_get_mscale(scale: float, mscale: float) -> float:
    """vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:20-23."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int,
    base: float, max_position_embeddings: int,
) -> Tuple[int, int]:
    low = math.floor(yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings
    ))
    high = math.ceil(yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings
    ))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


class DeepseekYarnRotaryEmbedding(nn.Module):
    """YaRN-extended RoPE for DeepSeek V3.

    Aligned with `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:1-153`.
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
        base_freqs = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim
        )
        inv_freq_extrapolation = 1.0 / base_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * base_freqs)
        low, high = yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.dim, self.base,
            self.original_max_position_embeddings,
        )
        ramp_mask = 1 - yarn_linear_ramp_mask(low, high, self.dim // 2, torch.float32)
        inv_freq = (
            inv_freq_interpolation * (1 - ramp_mask)
            + inv_freq_extrapolation * ramp_mask
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        mscale = float(yarn_get_mscale(self.scaling_factor, self.mscale_base) /
                       yarn_get_mscale(self.scaling_factor, self.mscale_all_dim))
        self.attn_mscale = mscale
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _rotate_gptj(x: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1)
        return rotated.flatten(-2)

    def forward(
        self, positions: Tensor, q_pe: Tensor, k_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        cache = self.cos_sin_cache.to(q_pe.device)
        half = self.dim // 2
        cos = cache[positions, :half].repeat_interleave(2, dim=-1).to(q_pe.dtype)
        sin = cache[positions, half:].repeat_interleave(2, dim=-1).to(q_pe.dtype)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        q_rotated = q_pe * cos + self._rotate_gptj(q_pe) * sin
        k_rotated = k_pe * cos + self._rotate_gptj(k_pe) * sin
        return q_rotated, k_rotated


# =============================================================================
# MLA attention (TP-sharded by heads)
# =============================================================================
class DeepseekV2MLAAttention(nn.Module):
    """Multi-Head Latent Attention.

    Aligned with `vllm/model_executor/models/deepseek_v2.py:847-1032`.

    TP semantics:
        - q_a_proj: replicated (small q_lora_rank)
        - q_a_layernorm: replicated
        - q_b_proj: ColumnParallel — output `[T, num_heads*qk_head_dim]`
          → `[T, num_local_heads*qk_head_dim]`
        - kv_a_proj_with_mqa: replicated (small)
        - kv_a_layernorm: replicated
        - kv_b_proj: ColumnParallel
        - rope: replicated
        - attention math: per-rank with `num_local_heads` heads
        - o_proj: RowParallel (all-reduces)
    """

    def __init__(
        self, cfg: Config, rope: DeepseekYarnRotaryEmbedding,
        pcfg: ParallelConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        self.pcfg = pcfg
        H = cfg.num_attention_heads
        if H % pcfg.tp_size != 0:
            raise ValueError(
                f"num_attention_heads={H} must be divisible by tp_size={pcfg.tp_size}"
            )
        self.num_local_heads = H // pcfg.tp_size

        # Replicated linears (small, <=q_lora_rank input).
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps)
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, eps=cfg.rms_norm_eps)

        # ColumnParallel linears (split by head).
        self.q_b_proj = ColumnParallelLinear(
            cfg.q_lora_rank, H * cfg.qk_head_dim, pcfg,
        )
        self.kv_b_proj = ColumnParallelLinear(
            cfg.kv_lora_rank,
            H * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            pcfg,
        )

        # RowParallel: input is per-rank heads' output, allreduces.
        self.o_proj = RowParallelLinear(
            H * cfg.v_head_dim, cfg.hidden_size, pcfg,
        )

        # softmax scale incorporates YaRN's full mscale^2, NOT the
        # cos/sin pre-mul ratio `rope.attn_mscale` (which is 1.0 for DSv3
        # since `mscale_base == mscale_all_dim` and the ratio cancels).
        # vLLM uses `yarn_get_mscale(scaling_factor, mscale_all_dim)`
        # directly here (see vllm/.../deepseek_v2.py:963-966). Using the
        # ratio instead made the softmax 1.87× too flat for DSv3, which
        # systematically inflated row-0 attention magnitude (causal mask
        # restricts row 0 to single-token attention, so it never benefits
        # from softmax-peaking the way later rows do). The bias compounds
        # through the residual stream and triggers a row-0 MoE blow-up
        # ~6-11 layers in (2026-05-11).
        attn_factor_mscale = yarn_get_mscale(
            rope.scaling_factor, rope.mscale_all_dim
        )
        self.softmax_scale = (1.0 / math.sqrt(cfg.qk_head_dim)) * (
            attn_factor_mscale * attn_factor_mscale
        )

    def forward(
        self, positions: Tensor, hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        intra_dumps: Optional[dict] = None,
    ) -> Tensor:
        cfg = self.cfg
        T = hidden_states.shape[0]
        H_local = self.num_local_heads

        # Q path (replicated up to q_a_layernorm; ColumnParallel q_b_proj).
        q_c = self.q_a_proj(hidden_states)
        q_c = self.q_a_layernorm(q_c)
        if intra_dumps is not None:
            intra_dumps["q_a_out"] = q_c.detach().clone()
        q = self.q_b_proj(q_c).view(T, H_local, cfg.qk_head_dim)
        if intra_dumps is not None:
            intra_dumps["q_b_full"] = q.detach().clone()
        q_nope = q[..., : cfg.qk_nope_head_dim]
        q_pe = q[..., cfg.qk_nope_head_dim :]

        # KV path.
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        if intra_dumps is not None:
            intra_dumps["kv_a_out"] = kv_a.detach().clone()
        kv_c = kv_a[..., : cfg.kv_lora_rank]
        k_pe = kv_a[..., cfg.kv_lora_rank :].unsqueeze(1)
        if intra_dumps is not None:
            intra_dumps["k_pe_pre_rope"] = k_pe.detach().clone()
        kv_c = self.kv_a_layernorm(kv_c)
        if intra_dumps is not None:
            intra_dumps["kv_c"] = kv_c.detach().clone()
        kv = self.kv_b_proj(kv_c).view(
            T, H_local, cfg.qk_nope_head_dim + cfg.v_head_dim
        )
        k_nope = kv[..., : cfg.qk_nope_head_dim]
        v = kv[..., cfg.qk_nope_head_dim :]

        # RoPE.
        q_pe, k_pe = self.rope(positions, q_pe, k_pe)

        # DEBUG 2026-05-10: dump K_nope, K_pe post-RoPE, V for chunked prefill
        # PyTorch reference validation.
        if intra_dumps is not None:
            intra_dumps["k_nope"] = k_nope.detach().clone()
            intra_dumps["k_pe_post_rope"] = k_pe.detach().clone()
            intra_dumps["v"] = v.detach().clone()

        # Recompose K and pad V.
        k_pe = k_pe.expand(-1, H_local, -1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        if intra_dumps is not None:
            intra_dumps["q_full_post_rope"] = q_full.detach().clone()
        v_padded = F.pad(v, (0, cfg.qk_head_dim - cfg.v_head_dim))

        # Attention with YaRN-mscale^2 softmax scale.
        q_t = q_full.permute(1, 0, 2)
        k_t = k.permute(1, 0, 2)
        v_t = v_padded.permute(1, 0, 2)
        attn = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
            scale=self.softmax_scale,
        )
        attn = attn.permute(1, 0, 2)
        attn = attn[..., : cfg.v_head_dim]
        attn = attn.reshape(T, H_local * cfg.v_head_dim)
        if intra_dumps is not None:
            intra_dumps["attn_unabsorbed"] = attn.detach().clone()

        # RowParallel: each rank's partial output is all-reduced.
        out = self.o_proj(attn)
        if intra_dumps is not None:
            intra_dumps["attn_proj_out"] = out.detach().clone()
        return out


# =============================================================================
# Dense MLP (used for layers 0..first_k_dense_replace-1; TP-sharded)
# =============================================================================
class DeepseekV2DenseMLP(nn.Module):
    """Dense MLP (gate_up Column + down Row, all-reduces in down).

    Aligned with `vllm/model_executor/models/deepseek_v2.py:115-173`.
    """

    def __init__(self, cfg: Config, pcfg: ParallelConfig):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            cfg.hidden_size, 2 * cfg.intermediate_size, pcfg,
        )
        self.down_proj = RowParallelLinear(
            cfg.intermediate_size, cfg.hidden_size, pcfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# =============================================================================
# MoE
# =============================================================================
class _RoutedExpert(nn.Module):
    """One routed expert (TP-sharded internally by `routed_tp_size`).

    Stored as Column-parallel gate_up_proj + Row-parallel down_proj.
    No internal all-reduce; the outer MoE forward does ONE global
    all-reduce across the full TP world after summing all local
    experts' partial contributions.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, pcfg: ParallelConfig
    ):
        super().__init__()
        self.gate_up_proj = RoutedExpertColumnParallel(
            hidden_size, 2 * intermediate_size, pcfg,
        )
        self.down_proj = RoutedExpertRowParallel(
            intermediate_size, hidden_size, pcfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DeepseekV3MoE(nn.Module):
    """Routed experts + shared expert with TP + EP.

    Aligned with `vllm/model_executor/models/deepseek_v2.py:235-398`
    and FusedMoE's `select_experts` routing for DeepSeek V3
    (sigmoid + correction-bias + grouped topk).

    EP semantics:
        - Each rank holds `n_routed_experts / ep_size` experts.
        - Each expert is TP-sharded by `routed_tp_size = tp_size / ep_size`.
        - Router (`gate`) is replicated; topk decisions are the same on
          every rank.
        - Each rank only computes its local experts' contributions.
        - One global AllReduce across the full TP world combines.

    Mathematical equivalence to vLLM's all-to-all dispatch + per-rank
    expert compute + combine: each token's contributions from experts
    in EP group g all live on the ranks in g (TP-sharded by routed_tp_size).
    Their sum across those ranks recovers the full per-expert output.
    Across EP groups, a token's contributions never overlap (different
    experts), so a global sum across the full world correctly combines
    all groups' contributions.

    Why we don't do explicit all-to-all:
        - Same final result.
        - Simpler PyTorch reference (no all_to_all primitive needed).
        - Slower in production (sends more zero-traffic across the
          network), but irrelevant for a correctness reference.
    """

    def __init__(self, cfg: Config, pcfg: ParallelConfig):
        super().__init__()
        self.cfg = cfg
        self.pcfg = pcfg
        # Layer index is set by the parent DecoderLayer for debug printing
        # of MoE routing decisions. Defaults to -1 ("unknown").
        self.layer_idx = -1

        # Gate (replicated).
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False)
        self.gate_e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts)
        )

        # Local experts (only this rank's slice of the EP partition).
        self.num_local_experts = pcfg.num_local_routed_experts(cfg.n_routed_experts)
        self.first_local_expert = pcfg.first_local_routed_expert(cfg.n_routed_experts)
        self.local_experts = nn.ModuleList(
            [
                _RoutedExpert(cfg.hidden_size, cfg.moe_intermediate_size, pcfg)
                for _ in range(self.num_local_experts)
            ]
        )

        # Shared experts: TP-sharded by full tp_size (Column gate_up + Row down).
        # Note: shared_expert intermediate is `moe_intermediate_size * n_shared_experts`
        # in the published checkpoint (DeepSeek V3 uses 1 shared expert).
        shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
        self.shared_gate_up = ColumnParallelLinear(
            cfg.hidden_size, 2 * shared_inter, pcfg,
        )
        self.shared_down = RowParallelLinear(
            shared_inter, cfg.hidden_size, pcfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        pcfg = self.pcfg
        T = x.shape[0]
        import os
        _moe_dbg = os.environ.get("DSV3_REF_MOE_DEBUG", "")
        _dbg_this_layer = (
            _moe_dbg and str(self.layer_idx) in _moe_dbg.split(",")
            and (not hasattr(self, "_dbg_done") or not self._dbg_done)
        )

        # 1. Score router logits (sigmoid scoring per DeepSeek V3 config).
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
        group_score = group_scores.topk(2, dim=-1).values.sum(dim=-1)
        group_idx = group_score.topk(cfg.topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_score)
        group_mask.scatter_(1, group_idx, 1.0)
        expert_mask = (
            group_mask.unsqueeze(-1).expand(-1, -1, n_per_grp)
            .reshape(T, cfg.n_routed_experts).bool()
        )
        masked_scores = scores_for_topk.masked_fill(~expert_mask, float("-inf"))

        # 4. topk over remaining experts (un-biased weights).
        #
        # vLLM keeps the routing weights in float32 and applies the
        # `routed_scaling_factor` AFTER the per-expert sum + global
        # AllReduce (`vllm/model_executor/models/deepseek_v2.py:325,379`).
        # Our earlier code (a) pre-multiplied by routed_scaling_factor=2.5
        # and (b) demoted to bf16 before the per-expert mul. For "diffuse"
        # rows that's fine — accumulation noise cancels across channels.
        # But for row 0 (causal mask → attends only to self → "spiky"
        # hidden state) the bf16 (large_w × large_y) addend can saturate
        # mantissa precision, especially when one expert (e.g. expert 121
        # at L6) fires hard. The error compounds layer-on-layer and
        # blows up the row-0 residual stream.
        # Fix: keep topk_weights in float32 through the per-expert mul,
        # don't pre-multiply by routed_scaling_factor here, apply the
        # factor once after the global AllReduce. Matches vLLM's order
        # exactly. [2026-05-11]
        topk_vals, topk_idx = masked_scores.topk(
            cfg.num_experts_per_tok, dim=-1
        )
        topk_weights = scores.gather(-1, topk_idx)
        if cfg.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(torch.float32)
        if _dbg_this_layer and pcfg.rank == 0:
            # Print row 0 vs row 1 routing decisions for comparison.
            import sys
            for r in (0, 1):
                print(f"[moe-dbg L{self.layer_idx} rank0] row {r}: "
                      f"x_norm={x[r].float().norm().item():.4f}  "
                      f"topk_idx={topk_idx[r].tolist()}  "
                      f"topk_w={[f'{v:.4f}' for v in topk_weights[r].tolist()]}",
                      file=sys.stderr, flush=True)

        # 5. Apply only this rank's local experts.
        # Optimization: iterate only the UNIQUE experts that appear in
        # topk_idx (at most num_experts_per_tok * T distinct experts),
        # filtered to those owned by this rank. Avoids the O(n_routed_experts)
        # Python loop that dominates forward time at long context.
        #
        # Accumulate in FP32 to match vLLM's FusedMoE kernel which uses
        # FP32 register accumulation across experts. BF16 accumulation
        # loses mantissa precision when one expert fires hard for a
        # "spiky" hidden state (e.g. row 0 with causal-mask single-position
        # attention), compounding into a row-specific magnitude blow-up
        # over many layers.
        routed_out = torch.zeros_like(x, dtype=torch.float32)
        # `topk_idx` shape: [T, topk]. Get the unique expert indices used.
        unique_experts = torch.unique(topk_idx).tolist()
        first = self.first_local_expert
        last = first + self.num_local_experts
        for global_e in unique_experts:
            if global_e < first or global_e >= last:
                continue
            local_e_idx = global_e - first
            mask = (topk_idx == global_e)
            if not mask.any():
                continue
            w = (mask.float() * topk_weights).sum(dim=-1, keepdim=True)
            sel = w.squeeze(-1) > 0
            if not sel.any():
                continue
            x_sel = x[sel]
            y_raw = self.local_experts[local_e_idx](x_sel)
            # Apply routing weight in FP32 (matches vLLM's FusedMoE
            # which keeps weights in FP32 register through the mul).
            y = y_raw.float() * w[sel]  # FP32 (routed_out is FP32)
            if _dbg_this_layer and pcfg.rank == 0 and sel[0]:
                # Row 0 routes to this expert. Print expert output magnitude.
                # Index of row 0 in sel-selected rows:
                row0_in_sel = sel[:1].sum().item() - 1 + 1 - 1  # = 0 if sel[0]
                row0_in_sel = 0
                import sys
                print(f"[moe-dbg L{self.layer_idx} rank0] row0 expert {global_e} "
                      f"(local {local_e_idx}): "
                      f"raw_y_norm={y_raw[row0_in_sel].float().norm().item():.4f}  "
                      f"w={w[0,0].item():.4f}  "
                      f"y_norm={y[row0_in_sel].float().norm().item():.4f}",
                      file=sys.stderr, flush=True)
            routed_out[sel] = routed_out[sel] + y  # FP32 accumulation

        # 6. ONE global AllReduce across full TP world.
        if _dbg_this_layer and pcfg.rank == 0:
            import sys
            print(f"[moe-dbg L{self.layer_idx} rank0] pre-allreduce "
                  f"routed_out row0 norm={routed_out[0].float().norm().item():.4f}",
                  file=sys.stderr, flush=True)
        routed_out = all_reduce_tp(routed_out, pcfg)
        # Apply routed_scaling_factor ONCE after the per-expert sum and the
        # global AllReduce (vLLM: `deepseek_v2.py:379`). Pre-multiplying
        # into topk_weights inflated bf16 addends for "spiky" hidden
        # states (row 0) and lost mantissa precision.
        routed_out = (routed_out * cfg.routed_scaling_factor).to(x.dtype)
        if _dbg_this_layer and pcfg.rank == 0:
            import sys
            print(f"[moe-dbg L{self.layer_idx} rank0] post-allreduce*scale "
                  f"routed_out row0 norm={routed_out[0].float().norm().item():.4f}",
                  file=sys.stderr, flush=True)
            self._dbg_done = True

        # 7. Shared expert (TP across full tp_size; the RowParallel
        #    `shared_down` already does its own all-reduce).
        gate_up = self.shared_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        shared_out = self.shared_down(F.silu(gate) * up)

        return routed_out + shared_out


# =============================================================================
# Decoder layer
# =============================================================================
class DeepseekV2DecoderLayer(nn.Module):
    """Single decoder block.

    Aligned with `vllm/model_executor/models/deepseek_v2.py:1119-1166`.
    """

    def __init__(
        self, cfg: Config, layer_idx: int,
        rope: DeepseekYarnRotaryEmbedding, pcfg: ParallelConfig,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = DeepseekV2MLAAttention(cfg, rope, pcfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        if layer_idx < cfg.first_k_dense_replace:
            self.mlp = DeepseekV2DenseMLP(cfg, pcfg)
        else:
            self.mlp = DeepseekV3MoE(cfg, pcfg)
            self.mlp.layer_idx = layer_idx

    def forward(
        self, positions: Tensor, hidden_states: Tensor, residual: Optional[Tensor],
        intra_dumps: Optional[dict] = None,
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if intra_dumps is not None and self.layer_idx == 0:
            intra_dumps["input_norm"] = hidden_states.detach().clone()
        hidden_states = self.self_attn(
            positions, hidden_states,
            intra_dumps=intra_dumps if (intra_dumps is not None and self.layer_idx == 0) else None,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        if intra_dumps is not None and self.layer_idx == 0:
            intra_dumps["attn_residual"] = residual.detach().clone()
        hidden_states = self.mlp(hidden_states)
        if intra_dumps is not None and self.layer_idx == 0:
            intra_dumps["mlp_delta"] = hidden_states.detach().clone()
        return hidden_states, residual


# =============================================================================
# MTP layer
# =============================================================================
class DeepseekV3MTPLayer(nn.Module):
    """Multi-Token-Prediction head.

    Aligned with `vllm/model_executor/models/deepseek_mtp.py:60-118`.

    The `eh_proj` is a single Linear over `cat([enorm(embed), hnorm(h)])`.
    For TP, we keep eh_proj replicated for simplicity (it's a single
    [2*H, H] matrix, ~100 MB at H=7168, replicated across 4 ranks =
    400 MB total — acceptable). vLLM uses ColumnParallelLinear here;
    the math is identical.
    """

    def __init__(
        self, cfg: Config, rope: DeepseekYarnRotaryEmbedding, pcfg: ParallelConfig,
    ):
        super().__init__()
        self.enorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.hnorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # Kept replicated for simplicity. vLLM uses ColumnParallel; the
        # math output is the same after consumers all-reduce.
        self.eh_proj = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.mtp_block = DeepseekV2DecoderLayer(
            cfg, layer_idx=cfg.num_hidden_layers, rope=rope, pcfg=pcfg,
        )
        self.shared_head_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        input_ids_embed: Tensor,
        previous_hidden_states: Tensor,
    ) -> Tensor:
        mask = (positions != 0).to(input_ids_embed.dtype).unsqueeze(-1)
        input_ids_embed = input_ids_embed * mask
        e = self.enorm(input_ids_embed)
        h = self.hnorm(previous_hidden_states)
        x = self.eh_proj(torch.cat([e, h], dim=-1))
        x, residual = self.mtp_block(positions, x, residual=None)
        x = x + residual
        x = self.shared_head_norm(x)
        return x


# =============================================================================
# Top-level model
# =============================================================================
class DeepseekV3Model(nn.Module):
    """DeepSeek V3 (decoder + optional MTP) reference with TP + EP.

    Args:
        cfg: model config.
        layer_indices: which main-model layer indices to actually
            instantiate.
        enable_mtp: if True, also build and run the MTP head.
        parallel_config: TP + EP topology + this rank's index.
    """

    def __init__(
        self,
        cfg: Config,
        layer_indices: Optional[list[int]] = None,
        enable_mtp: bool = False,
        parallel_config: Optional[ParallelConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.pcfg = parallel_config or ParallelConfig()
        self.layer_indices = (
            list(layer_indices) if layer_indices is not None
            else list(range(cfg.num_hidden_layers))
        )
        self.enable_mtp = enable_mtp

        # Embed + lm_head replicated (small enough).
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rope = DeepseekYarnRotaryEmbedding(cfg)

        self.layers = nn.ModuleDict(
            {
                str(i): DeepseekV2DecoderLayer(cfg, i, self.rope, self.pcfg)
                for i in self.layer_indices
            }
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # NOTE: DeepSeek V3 checkpoint has SEPARATE `lm_head.weight` and
        # `model.embed_tokens.weight` (different magnitudes — embed L2 ≈ 3.8,
        # lm_head L2 ≈ 8.1). Tying them and then copying both from the
        # checkpoint causes the second copy to silently overwrite the first
        # (since the tied tensor is a single Parameter), leaving the model
        # using lm_head values for embeddings. Keep the two as separate
        # Parameters so loader._copy_replicated can populate each correctly.

        if enable_mtp:
            self.mtp_layer = DeepseekV3MTPLayer(cfg, self.rope, self.pcfg)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        prev_mtp_input_ids: Optional[Tensor] = None,
        record_hidden: bool = False,
        record_layer0_intra: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass.

        `record_hidden=False` (default) returns only the final tokens
        (`argmax` + `mtp_argmax` if MTP). This is the fast path used
        for end-to-end token-level alignment with MPK; per-layer
        intermediates would consume significant GPU memory at long
        context.

        `record_hidden=True` additionally clones every intermediate
        (embed, per-layer output+residual, final_norm, logits, MTP
        equivalents) into the returned dict. Use for fine-grained
        debugging — but watch the memory.

        `record_layer0_intra=True` additionally saves per-substep
        attention intermediates for layer 0 into out['layer0_intra'].
        Independent of record_hidden; defaults to False so existing
        callers are unaffected.
        """
        out: dict[str, Tensor] = {}
        x = self.embed_tokens(input_ids)
        if record_hidden:
            out["embed"] = x.detach().clone()
        residual: Optional[Tensor] = None
        layer0_intra: Optional[dict] = {} if record_layer0_intra else None
        for li in self.layer_indices:
            layer = self.layers[str(li)]
            x, residual = layer(
                positions, x, residual,
                intra_dumps=layer0_intra if li == 0 else None,
            )
            if record_hidden:
                out[f"layer_{li}_output"] = x.detach().clone()
                out[f"layer_{li}_residual"] = residual.detach().clone()
        if layer0_intra is not None:
            out["layer0_intra"] = layer0_intra
        x_main, _ = self.norm(x, residual=residual)
        if record_hidden:
            out["final_norm"] = x_main.detach().clone()
            logits = self.lm_head(x_main)
            out["logits"] = logits.detach().clone()
            out["argmax"] = logits.argmax(dim=-1).detach().clone()
        else:
            # Compute logits only on the last position when record_hidden
            # is off — for autoregressive decode we only need the next
            # token's prediction. Saves a [T, vocab=129280] tensor on GPU.
            logits_last = self.lm_head(x_main[-1:])
            out["argmax"] = torch.cat([
                # Pad earlier positions with zeros for shape compatibility
                # with the runner's `argmax[-1]` access.
                torch.zeros(x_main.shape[0] - 1,
                            dtype=torch.long, device=x_main.device),
                logits_last.argmax(dim=-1).detach().clone(),
            ])
        # Run MTP only when caller provided prev_mtp_input_ids. The model
        # being built with enable_mtp=True doesn't FORCE running it on
        # every forward — the runner does a "main-only" pass first to
        # get target argmax, then a "main+MTP" pass with the shifted
        # ground-truth input IDs.
        if self.enable_mtp and prev_mtp_input_ids is not None:
            mtp_embed = self.embed_tokens(prev_mtp_input_ids)
            mtp_out = self.mtp_layer(
                positions=positions,
                input_ids_embed=mtp_embed,
                previous_hidden_states=x_main,
            )
            if record_hidden:
                out["mtp_output"] = mtp_out.detach().clone()
                mtp_logits = self.lm_head(mtp_out)
                out["mtp_logits"] = mtp_logits.detach().clone()
                out["mtp_argmax"] = mtp_logits.argmax(dim=-1).detach().clone()
            else:
                # Same compute-on-last-only optimization for MTP.
                mtp_logits_last = self.lm_head(mtp_out[-1:])
                out["mtp_argmax"] = torch.cat([
                    torch.zeros(mtp_out.shape[0] - 1,
                                dtype=torch.long, device=mtp_out.device),
                    mtp_logits_last.argmax(dim=-1).detach().clone(),
                ])
        return out
