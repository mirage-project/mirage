"""MLA RoPE catalog modules: 3 Q-side classes + 1 K-side class.

All four wrap ``deepseek_mla_rope_sm100.cuh`` under
``include/mirage/persistent_kernel/tasks/blackwell/``; tasks
``deepseek_mla_rope_q{_,_fused_,_split_,_k_}sm100`` instantiate it with
different template flags. Rotation is GPT-J interleaved on pairs
``(x[2i], x[2i+1])``; cos/sin tables are ``(max_seq_len, D_PE)`` with
``repeat_interleave``'d layout (kernel reads only the even-index entry).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """GPT-J interleaved rotation: pairs ``(x[2i], x[2i+1])`` swap+negate."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class _MLARopeQBase(MPKModule):
    """Shared Q-side RoPE plumbing. Args:
    ``num_heads`` (template ``NUM_HEADS``),
    ``q_tile_size`` (template ``Q_TILE``),
    ``prefix`` (HF state_dict key prefix; no params owned).
    """

    def __init__(
        self,
        num_heads: int,
        *,
        q_tile_size: int = 16,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_heads = num_heads
        self.q_tile_size = q_tile_size

    def forward(
        self,
        q_pe: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """GPT-J interleaved RoPE on the PE half of Q (eager reference)."""
        pos = positions.to(torch.long)
        cos_e = cos.index_select(0, pos).unsqueeze(1).to(q_pe.dtype)
        sin_e = sin.index_select(0, pos).unsqueeze(1).to(q_pe.dtype)
        return (q_pe * cos_e) + (_rotate_interleaved(q_pe) * sin_e)

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(R, H, ceil(T_max / q_tile_size))``. With H=128 (DSv3 single GPU)
        the H axis dominates and already saturates beyond 148 workers."""
        pk = current_pk()
        t_blocks = (pk.max_num_batched_tokens + self.q_tile_size - 1) // self.q_tile_size
        return (pk.max_num_batched_requests, self.num_heads, t_blocks)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)


class MLARopeQSingle(_MLARopeQBase):
    """Q-side RoPE: rotate ``q_pe`` in place + ``q_nope_pe`` barrier dep.

    Task ``deepseek_mla_rope_q_sm100``. ``has_split_q`` flags ``q_pe``
    as its own DTensor. Args: see :class:`_MLARopeQBase`.
    """

    def compile(
        self,
        q_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        *,
        q_nope_pe: DTensor,
        has_split_q: bool = False,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``deepseek_mla_rope_q_sm100``; in-place GPT-J interleaved RoPE.

        Tensor contract:
          q_nope_pe: (T_max, H, D_K=192) bf16, row-major, in-place. PE tail
            (cols D_NoPE..D_K, i.e. [128:192)) of every head is rotated.
          q_pe: (T_max, H, D_PE=64) bf16, row-major, in-place. Rotated only when
            ``has_split_q=True`` (kernel template flag ``HAS_SPLIT_Q``).
          cos_pos_embed / sin_pos_embed: (max_seq_len, D_PE=64) bf16, row-major;
            indexed by runtime ``positions = step[req_id] + token_offset``.
          output: alias of ``q_pe`` (DTensor return). Both q_nope_pe and q_pe
            are emitted as task outputs (deps for downstream MLA).

        Notes: kernel-template params = (NUM_HEADS, TILE_Q, HAS_SPLIT_Q, DO_Q=true,
        DO_K=false). Position read from ``step``+``qo_indptr`` meta-tensors.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_heads, self.q_tile_size, 1 if has_split_q else 0]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_nope_pe, q_pe, cos_pos_embed, sin_pos_embed, q_nope_pe, q_pe],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph, "deepseek_mla_rope_q_sm100", params
        )
        return q_pe


class MLARopeQFused(_MLARopeQBase):
    """Q-side RoPE on a fused per-row ``[..h_nope|h_pe..]`` tensor.

    Task ``deepseek_mla_rope_q_fused_sm100`` (D_NOPE=128, D_PE=64);
    rotates the PE slice of each head in place. Args: see
    :class:`_MLARopeQBase`.
    """

    def compile(
        self,
        q_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``deepseek_mla_rope_q_fused_sm100``; in-place GPT-J RoPE
        on the PE half of a fused per-head ``[nope|pe]`` Q tensor.

        Tensor contract:
          q_pe (fused): (T_max, H, D_K=D_NoPE+D_PE=128+64=192) bf16, row-major,
            in-place. Kernel rotates only cols [D_NoPE..D_K) of each head.
          cos_pos_embed / sin_pos_embed: (max_seq_len, D_PE=64) bf16, row-major;
            indexed at runtime by ``positions = step[req_id] + token_offset``
            (derived from ``qo_indptr_buffer``).
          output: alias of ``q_pe`` (in-place); registered as task output.

        Notes: kernel template (NUM_HEADS, TILE_Q, HAS_SPLIT_Q=false, DO_Q=true,
        DO_K=false). Per-head stride = D_K (FUSED_HEAD_DIM template default 576
        is overridden by params: actual H_FUSED stride is NUM_HEADS*D_K).
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_heads, self.q_tile_size]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_pe, cos_pos_embed, sin_pos_embed, q_pe], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "deepseek_mla_rope_q_fused_sm100", params
        )
        return q_pe


class MLARopeQSplit(_MLARopeQBase):
    """Q-side RoPE on standalone ``q_pe`` (optional row-swap aliasing).

    Task ``deepseek_mla_rope_q_split_sm100``. ``qfused_mode=0`` =
    standalone ``(T_max, H*64)``; ``qfused_mode=1`` = alias of
    ``q_b_prefill_fused`` (row stride H*192, PE at H*128).
    """

    def compile(
        self,
        q_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        *,
        qfused_mode: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``deepseek_mla_rope_q_split_sm100``; in-place GPT-J RoPE
        on standalone (or row-swap aliased) ``q_pe`` with no NoPE half.

        Tensor contract:
          q_pe: bf16, in-place.
            qfused_mode=0: (T_max, H, D_PE=64); per-head stride=64.
            qfused_mode=1: alias of ``q_b_prefill_fused`` (T_max, H*192); PE
              at col offset H*128, per-head stride 64 (template
              ``Q_ROW_STRIDE_OVERRIDE`` addressing).
          cos_pos_embed / sin_pos_embed: (max_seq_len, 64) bf16; positions
            derived from ``step`` + ``qo_indptr_buffer``.
          output: alias of ``q_pe``.

        Notes: template (H, TILE_Q, false, DO_Q=true, DO_K=false, 64, 64, 64).
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_heads, self.q_tile_size]
        if qfused_mode != 0:
            params.append(qfused_mode)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_pe, cos_pos_embed, sin_pos_embed, q_pe], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "deepseek_mla_rope_q_split_sm100", params
        )
        return q_pe


class MLARopeK(MPKModule):
    """K-PE RoPE (in place, no head fan-out).

    Task ``deepseek_mla_rope_k_sm100``. ``k_pe_row_stride`` /
    ``k_pe_offset`` enable slicing a wider parent buffer.
    Args: ``q_tile_size`` (template ``Q_TILE``), ``prefix``.
    """

    def __init__(
        self,
        *,
        q_tile_size: int = 16,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.q_tile_size = q_tile_size

    def forward(
        self,
        k_pe: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """GPT-J interleaved RoPE on K-PE; ``k_pe`` has no head axis."""
        pos = positions.to(torch.long)
        cos_e = cos.index_select(0, pos).to(k_pe.dtype)
        sin_e = sin.index_select(0, pos).to(k_pe.dtype)
        return (k_pe * cos_e) + (_rotate_interleaved(k_pe) * sin_e)

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(R, 1, ceil(T_max / q_tile_size))``; no H axis on K side under MLA."""
        pk = current_pk()
        t_blocks = (pk.max_num_batched_tokens + self.q_tile_size - 1) // self.q_tile_size
        return (pk.max_num_batched_requests, 1, t_blocks)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        k_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        *,
        k_pe_row_stride: Optional[int] = None,
        k_pe_offset: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``deepseek_mla_rope_k_sm100``; in-place GPT-J RoPE on K-PE
        (no head axis under MLA).

        Tensor contract:
          k_pe: bf16, in-place. No head axis.
            Default: (T_max, 128); real PE [0:64), cols [64:128) pad.
            Slice-override: col offset ``k_pe_offset`` inside (T_max,
              ``k_pe_row_stride``) parent (e.g. QKV-a: 2176/2048 → [2048:2112)).
          cos_pos_embed / sin_pos_embed: (max_seq_len, 64) bf16; positions
            from ``step`` + ``qo_indptr_buffer``.
          output: alias of ``k_pe``.

        Notes: template (1, TILE_Q, false, false, DO_K=true, 576, 64,
        K_PE_STRIDE, 0, 0, 0, K_PE_OFFSET).
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.q_tile_size]
        if k_pe_row_stride is not None or k_pe_offset != 0:
            row_stride = k_pe_row_stride if k_pe_row_stride is not None else 128
            params = [self.q_tile_size, row_stride, k_pe_offset]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(k_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_pe, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [k_pe, cos_pos_embed, sin_pos_embed, k_pe], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "deepseek_mla_rope_k_sm100", params)
        return k_pe


# ---------------------------------------------------------------------------
# Legacy variant-kwarg shim — kept for existing model/test/demo call sites.
# New code should use MLARopeQSingle / MLARopeQFused / MLARopeQSplit directly.
# ---------------------------------------------------------------------------
_Q_VARIANT_CLASSES = {
    "single": MLARopeQSingle,
    "fused": MLARopeQFused,
    "split": MLARopeQSplit,
}


def MLARopeQ(
    num_heads: int,
    *,
    variant: str = "fused",
    q_tile_size: int = 16,
    prefix: str = "",
) -> _MLARopeQBase:
    """Legacy dispatcher; returns the variant-specific subclass instance."""
    try:
        cls = _Q_VARIANT_CLASSES[variant]
    except KeyError:
        raise ValueError(
            f"MLARopeQ variant must be one of {sorted(_Q_VARIANT_CLASSES)}; "
            f"got {variant!r}"
        )
    return cls(num_heads=num_heads, q_tile_size=q_tile_size, prefix=prefix)
