"""Paged-KV-cache attention (prefill + decode unified) — production attention.

Wraps :meth:`PersistentKernel.paged_attention_layer`. Arch dispatch by
``pk.target_cc``: Ampere -> ``"paged_attention"``
(``tasks/ampere/multitoken_paged_attention.cuh``), Hopper (90) ->
``"paged_attention_hopper"`` (``tasks/hopper/multitoken_paged_attention_hopper.cuh``),
Blackwell (100) -> ``"paged_attention_sm100"``
(``tasks/blackwell/attention_sm100.cuh::multitoken_paged_attention_sm100_task_impl``).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from .._base import MPKModule
from ...context import current_pk
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_e = cos.unsqueeze(0).unsqueeze(2).to(q.dtype)
    sin_e = sin.unsqueeze(0).unsqueeze(2).to(q.dtype)
    return (q * cos_e) + (_rotate_half(q) * sin_e), (k * cos_e) + (_rotate_half(k) * sin_e)


def _per_head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = x.dtype
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x.to(torch.float32) * torch.rsqrt(var + eps)
    return (x_normed.to(input_dtype) * weight).to(input_dtype)


class PagedAttention(MPKModule):
    """Paged-KV-cache attention used by qwen3 for both prefill and decode.

    One task per (request, kv-head); per-request token range comes from
    ``qo_indptr_buffer`` and KV pages from ``paged_kv_indptr_buffer`` /
    ``paged_kv_indices_buffer`` / ``paged_kv_last_page_len_buffer``.
    Fused: per-head q/k RMSNorm + RoPE + paged KV-append + causal flash-attn.
    Input ``(T_max, H*D + 2*H_kv*D)``; for the SM100 path the QKV is
    GQA-interleaved per kv-head: ``[g_q (groups*D) | g_k (D) | g_v (D)] * H_kv``.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_idx: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads}) for GQA"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))

    def forward(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch reference using a contiguous KV cache (paged layout is only in compile)."""
        B, T, _ = qkv.shape
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.head_dim
        groups = H // H_kv

        # Input layout: GQA-interleaved per kv-head, matching the SM100 kernel.
        per_group = (groups + 2) * D
        qkv_view = qkv.view(B, T, H_kv, per_group)
        q = qkv_view[..., :groups * D].reshape(B, T, H_kv * groups, D)
        k = qkv_view[..., groups * D:(groups + 1) * D].reshape(B, T, H_kv, D)
        v = qkv_view[..., (groups + 1) * D:].reshape(B, T, H_kv, D)

        q = _per_head_rmsnorm(q, self.q_norm)
        k = _per_head_rmsnorm(k, self.k_norm)

        pos0 = positions[0].to(torch.long)
        cos_slice = cos.index_select(0, pos0).to(q.dtype)
        sin_slice = sin.index_select(0, pos0).to(q.dtype)
        q, k = _apply_rotary(q, k, cos_slice, sin_slice)

        for b in range(B):
            pos_b = positions[b].to(torch.long)
            k_cache[b].index_copy_(0, pos_b, k[b])
            v_cache[b].index_copy_(0, pos_b, v[b])

        seq_len = int(positions[0, -1].item()) + 1
        K = k_cache[:, :seq_len]
        V = v_cache[:, :seq_len]
        K = K.repeat_interleave(groups, dim=2)
        V = V.repeat_interleave(groups, dim=2)

        q_t = q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_t, K_t, V_t, is_causal=(T > 1), enable_gqa=False,
        )
        return attn.transpose(1, 2).contiguous().view(B, T, H * D)

    def auto_grid_dim(self, input_dt: DTensor) -> GridDim:
        """``(max_num_batched_requests, num_kv_heads, 1)`` — saturates for DSv3-style large H."""
        pk = current_pk()
        return (pk.max_num_batched_requests, self.num_kv_heads, 1)

    def compile(
        self,
        input: DTensor,
        k_cache: DTensor,
        v_cache: DTensor,
        cos: DTensor,
        sin: DTensor,
        *,
        output: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> DTensor:
        """Register ``paged_attention[_hopper|_sm100]`` (arch by ``pk.target_cc``).

        Tensor contract:
          input:   (T_max, (NQ + 2*NKV) * D)            bf16, fused QKV.
                   SM100 layout is GQA-interleaved per kv-head:
                   ``[ Q (H*D) | K (H_kv*D) | V (H_kv*D) ]`` stride ``(H + 2*H_kv) * D``.
          k_cache: (max_num_pages, page_size, NKV, D)   bf16, paged KV blocks.
          v_cache: (max_num_pages, page_size, NKV, D)   bf16, paged KV blocks.
          cos:     (max_seq_len, D)                     bf16, RoPE table.
          sin:     (max_seq_len, D)                     bf16, RoPE table.
          q_norm:  (D,)                                 bf16, per-head RMSNorm weight.
          k_norm:  (D,)                                 bf16, per-head RMSNorm weight.
          output:  (T_max, NQ * D)                      bf16, attention output.

        Meta deps: ``qo_indptr_buffer``, ``paged_kv_indptr_buffer``,
        ``paged_kv_indices_buffer``, ``paged_kv_last_page_len_buffer``.
        """
        pk = current_pk()

        if input.num_dims != 2:
            raise ValueError(
                f"PagedAttention.compile expects a 2-D input DTensor "
                f"(fused QKV after the qkv projection); got "
                f"num_dims={input.num_dims}"
            )
        if k_cache.num_dims != 4 or v_cache.num_dims != 4:
            raise ValueError(
                "PagedAttention.compile expects 4-D k_cache and v_cache "
                f"DTensors (max_num_pages, page_size, num_kv_heads, "
                f"head_dim); got num_dims={k_cache.num_dims} / "
                f"{v_cache.num_dims}"
            )

        prefix = self.prefix or "paged_attention."
        num_tokens = input.dim(0)

        if output is None:
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.new_tensor(
                dims=(num_tokens, self.num_heads * self.head_dim),
                dtype=input.dtype,
                name=out_name,
            )
        elif isinstance(output, torch.Tensor):
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.attach_input(output, name=out_name)
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "PagedAttention.compile output must be None, a "
                f"torch.Tensor, or a DTensor; got {type(output).__name__}"
            )

        q_norm_dt = pk.attach_input(self.q_norm.data, name=f"{prefix}q_norm")
        k_norm_dt = pk.attach_input(self.k_norm.data, name=f"{prefix}k_norm")

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(input)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert input.num_dims == 2
        assert out_dt.num_dims == 2
        assert k_cache.num_dims == 4
        assert v_cache.num_dims == 4
        assert k_cache.dim(0) == pk.max_num_pages
        assert v_cache.dim(0) == pk.max_num_pages
        assert k_cache.dim(1) == pk.page_size
        assert v_cache.dim(1) == pk.page_size
        head_dim = k_cache.dim(3)
        num_kv_heads = k_cache.dim(2)
        num_q_heads = out_dt.dim(1) // head_dim
        rotary_embed = 0
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

        # params: [num_q_heads, num_kv_heads, qk_norm, rotary_embed, max_seq_len, page_size]
        params = [
            num_q_heads,
            num_kv_heads,
            qk_norm,
            rotary_embed,
            pk.max_seq_length,
            pk.page_size,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        assert grid_dim[0] == pk.max_num_batched_requests
        assert grid_dim[1] == num_kv_heads
        tb_graph.new_input(input, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(v_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(q_norm_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_norm_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (-1, 1, -1), -1, True)
        pk.kn_graph.customized(
            [input, k_cache, v_cache, q_norm_dt, k_norm_dt, cos, sin, out_dt],
            tb_graph,
        )
        if pk.target_cc == 90:
            pk.kn_graph.register_task(tb_graph, "paged_attention_hopper", params)
        elif pk.target_cc == 100:
            pk.kn_graph.register_task(tb_graph, "paged_attention_sm100", params)
        else:
            pk.kn_graph.register_task(tb_graph, "paged_attention", params)
        return out_dt
