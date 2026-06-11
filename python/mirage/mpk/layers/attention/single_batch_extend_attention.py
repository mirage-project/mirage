"""Single-batch extend attention (multi-token decode / spec-verify).

Wraps :meth:`PersistentKernel.single_batch_extend_attention_layer`
(task ``"single_batch_extend_attention"``). Code-gen emits
``kernel::single_batch_extend_kernel`` from
``include/mirage/persistent_kernel/tasks/ampere/single_batch_extend.cuh``;
the Blackwell ``task_header.cuh`` re-includes the same Ampere file, so
there is no separate SM90/SM100 variant.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .._base import BlockDim, GridDim, MPKModule


__all__ = ["SingleBatchExtendAttention"]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_e = cos.unsqueeze(0).unsqueeze(2).to(q.dtype)
    sin_e = sin.unsqueeze(0).unsqueeze(2).to(q.dtype)
    return (q * cos_e) + (_rotate_half(q) * sin_e), (k * cos_e) + (
        _rotate_half(k) * sin_e
    )


def _per_head_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    input_dtype = x.dtype
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(var + eps)
    return (x_normed.to(input_dtype) * weight).to(input_dtype)


class SingleBatchExtendAttention(MPKModule):
    """Multi-token extend variant of single-batch decode attention.

    Identical algebra to :class:`Attention` but processes
    ``T = extend_num + 1`` tokens at once (trailing tokens are candidates
    being verified in speculative decode).  Input ``(T, fused_outdim)``
    with ``[q | k | v]`` per row; ``k_cache``/``v_cache`` are
    ``(B=1, max_seq_len, H_kv, D)`` and get appended at
    ``[seq_len - T, seq_len)``.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int = 0,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads "
                f"({num_kv_heads}) for GQA"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_norm = nn.Parameter(torch.ones(head_dim, dtype=torch.bfloat16))
        self.k_norm = nn.Parameter(torch.ones(head_dim, dtype=torch.bfloat16))

    def forward(
        self,
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        seq_len: int,
    ) -> torch.Tensor:
        """PyTorch reference across ``T = extend_num + 1`` tokens."""
        T = q_proj.shape[0]
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.head_dim
        q = q_proj.view(1, T, H, D)
        k = k_proj.view(1, T, H_kv, D)
        v = v_proj.view(1, T, H_kv, D)
        q = _per_head_rmsnorm(q, self.q_norm)
        k = _per_head_rmsnorm(k, self.k_norm)
        cos_slice = cos[seq_len - T : seq_len]
        sin_slice = sin[seq_len - T : seq_len]
        q, k = _apply_rotary(q, k, cos_slice, sin_slice)
        k_cache[:, seq_len - T : seq_len] = k
        v_cache[:, seq_len - T : seq_len] = v
        K = k_cache[:, :seq_len]
        V = v_cache[:, :seq_len]
        groups = H // H_kv
        K = K.repeat_interleave(groups, dim=2)
        V = V.repeat_interleave(groups, dim=2)
        q_t = q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_t, K_t, V_t, is_causal=False, enable_gqa=False,
        )
        return attn.transpose(1, 2).contiguous().view(T, H * D)

    def auto_grid_dim(self, input_dt: Any) -> GridDim:
        """``(extend_num + 1, num_kv_heads, 1)`` — one CTA per (token, kv-head)."""
        return (input_dt.dim(0), self.num_kv_heads, 1)

    def default_block_dim(self) -> BlockDim:
        """Kernel hard-wires 128 threads (single-batch decode family)."""
        return (128, 1, 1)

    def compile(
        self,
        input: Any,
        k_cache: Any,
        v_cache: Any,
        cos: Any,
        sin: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> Any:
        """Register ``single_batch_extend_attention`` (Ampere/Hopper only — no SM100 variant).

        Tensor contract:
          input:   (T, (NQ + 2*NKV) * D)         bf16, fused QKV; T = extend_num + 1.
          k_cache: (1, max_seq_len, NKV, D)      bf16, contiguous (B=1) KV cache.
          v_cache: (1, max_seq_len, NKV, D)      bf16, contiguous (B=1) KV cache.
          cos:     (max_seq_len, D)              bf16, RoPE table.
          sin:     (max_seq_len, D)              bf16, RoPE table.
          q_norm:  (D,)                          bf16, per-head RMSNorm weight (auto-attached).
          k_norm:  (D,)                          bf16, per-head RMSNorm weight (auto-attached).
          output:  (T, NQ * D)                   bf16, attention output.

        Notes: appends k/v at ``[seq_len - T, seq_len)``; non-causal across the
        extend block. Meta deps: ``step`` for the position window.
        """
        import torch as _torch
        from ...context import current_pk

        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(input)
        if block_dim is None:
            block_dim = self.default_block_dim()

        prefix = self.prefix or "extend_attn."
        T = input.dim(0)
        if output is None:
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.new_tensor(
                dims=(T, self.num_heads * self.head_dim),
                dtype=input.dtype,
                name=out_name,
            )
        elif isinstance(output, _torch.Tensor):
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.attach_input(output, name=out_name)
        else:
            out_dt = output

        q_norm_dt = pk.attach_input(self.q_norm.data, name=f"{prefix}q_norm")
        k_norm_dt = pk.attach_input(self.k_norm.data, name=f"{prefix}k_norm")

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert input.num_dims == 2
        assert out_dt.num_dims == 2
        assert k_cache.num_dims == 4
        assert v_cache.num_dims == 4
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = out_dt.dim(1) // head_dim
        rotary_embed = 0
        output_stride = out_dt.dim(1)

        extend_num = input.dim(0) - 1
        if cos is not None or sin is not None:
            assert cos.num_dims == 2
            assert sin.num_dims == 2
            assert cos.dim(1) == head_dim
            assert sin.dim(1) == head_dim
            rotary_embed = 1
        qk_norm = 0
        if q_norm_dt is not None or k_norm_dt is not None:
            assert q_norm_dt.num_dims == 1
            assert k_norm_dt.num_dims == 1
            qk_norm = 1
            assert q_norm_dt.dim(0) == head_dim
            assert k_norm_dt.dim(0) == head_dim

        # params: [num_q_heads, num_kv_heads, qk_norm, rotary_embed, extend_num, output_stride]
        params = [
            num_q_heads, num_kv_heads, qk_norm, rotary_embed,
            extend_num, output_stride,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input, (0, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (0, 2, -1), 1, True)
        tb_graph.new_input(q_norm_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (0, 1, -1), -1, True)
        pk.kn_graph.customized(
            [input, k_cache, v_cache, q_norm_dt, k_norm_dt, cos, sin, out_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph, "single_batch_extend_attention", params
        )
        return out_dt
