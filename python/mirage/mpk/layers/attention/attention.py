"""Single-batch decode attention (per-head qk-RMSNorm + RoPE + KV append + softmax).

Wraps :meth:`PersistentKernel.attention_layer` (task name ``"attention"``).
Code-gen emits ``kernel::single_batch_decoding_kernel<bfloat16, ...>`` from
``include/mirage/persistent_kernel/tasks/ampere/single_batch_decoding.cuh``
(also used on Hopper). **Not supported on Blackwell** — ``tasks/blackwell/``
ships only the paged form; use :class:`PagedAttention` there.
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


class Attention(MPKModule):
    """Decode-only fused attention (Ampere/Hopper only; not on Blackwell).

    Per-head q/k RMSNorm (eps 1e-6), RoPE on q/k at position ``step[0]``,
    in-place KV-cache append, then ``softmax(q @ K^T / sqrt(D)) @ V`` over
    the cached prefix. Wraps :meth:`PersistentKernel.attention_layer` 1:1.
    Input contract is the fused QKV layout
    ``[ Q (H*D) | K (H_kv*D) | V (H_kv*D) ]`` produced by the upstream
    rmsnorm-linear; ``k_cache`` / ``v_cache`` are 4-D ``(B, max_seq_len, H_kv, D)``.
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
        raise RuntimeError(
            "layers.Attention (plain decode, wraps pk.attention_layer) "
            "has no Blackwell-SM100 kernel variant — attention_sm100.cuh "
            "only ships the paged form (multitoken_paged_attention_sm100). "
            "On Blackwell, use layers.PagedAttention instead."
        )
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
        q_proj: torch.Tensor,
        k_proj: torch.Tensor,
        v_proj: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """PyTorch reference for the fused decode body (norm+RoPE+KV-append+softmax)."""
        B, T, _ = q_proj.shape
        H, H_kv, D = self.num_heads, self.num_kv_heads, self.head_dim
        q = q_proj.view(B, T, H, D)
        k = k_proj.view(B, T, H_kv, D)
        v = v_proj.view(B, T, H_kv, D)
        q = _per_head_rmsnorm(q, self.q_norm)
        k = _per_head_rmsnorm(k, self.k_norm)
        if seq_len is None:
            seq_len = T
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
        return attn.transpose(1, 2).contiguous().view(B, T, H * D)

    def auto_grid_dim(self, input_dt: DTensor) -> GridDim:
        """``(batch_size, num_kv_heads, 1)`` — one task per (request, kv-head)."""
        return (input_dt.dim(0), self.num_kv_heads, 1)

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
        """Register the ``"attention"`` task (Ampere/Hopper; no SM100 variant).

        Tensor contract:
          input:   (T, (NQ + 2*NKV) * D) bf16, fused QKV ``[Q | K | V]`` row-major.
          k_cache: (B, max_seq_len, NKV, D) bf16, contiguous KV cache (NOT paged).
          v_cache: (B, max_seq_len, NKV, D) bf16, contiguous KV cache.
          cos:     (max_seq_len, D)        bf16, RoPE table (HF repeat-half layout).
          sin:     (max_seq_len, D)        bf16, RoPE table (HF repeat-half layout).
          q_norm:  (D,)                    bf16, per-head RMSNorm weight (auto-attached).
          k_norm:  (D,)                    bf16, per-head RMSNorm weight (auto-attached).
          output:  (T, NQ * D)             bf16, attention output (auto-alloc / attach / passthrough).

        Notes: decode-only (one new token); KV-cache append at ``step[0]``; eps=1e-6 hard-coded.
        Meta deps: ``step`` for the RoPE/KV-append position.
        """
        pk = current_pk()

        if input.num_dims != 2:
            raise ValueError(
                f"Attention.compile expects a 2-D input DTensor "
                f"(fused QKV after the qkv projection); got "
                f"num_dims={input.num_dims}"
            )
        if k_cache.num_dims != 4 or v_cache.num_dims != 4:
            raise ValueError(
                "Attention.compile expects 4-D k_cache and v_cache "
                f"DTensors (B, max_seq_len, num_kv_heads, head_dim); "
                f"got num_dims={k_cache.num_dims} / {v_cache.num_dims}"
            )

        prefix = self.prefix or "attention."
        batch_size = input.dim(0)

        if output is None:
            out_name = name if name is not None else f"{prefix}attn_out"
            out_dt = pk.new_tensor(
                dims=(batch_size, self.num_heads * self.head_dim),
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
                "Attention.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
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

        # params: [num_q_heads, num_kv_heads, qk_norm, rotary_embed]
        params = [num_q_heads, num_kv_heads, qk_norm, rotary_embed]

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
        pk.kn_graph.register_task(tb_graph, "attention", params)
        return out_dt
