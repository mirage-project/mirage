"""MLA RoPE catalog modules — Q-side (3 variants) and K-side.

These are the catalog counterparts to four pk methods on
:class:`PersistentKernel` (see ``python/mirage/mpk/persistent_kernel.py``):

* ``deepseek_mla_rope_q_layer``        (task ``deepseek_mla_rope_q_sm100``)
* ``deepseek_mla_rope_q_fused_layer``  (task ``deepseek_mla_rope_q_fused_sm100``)
* ``deepseek_mla_rope_q_split_layer``  (task ``deepseek_mla_rope_q_split_sm100``)
* ``deepseek_mla_rope_k_layer``        (task ``deepseek_mla_rope_k_sm100``)

DeepSeek-V3 MLA partitions Q and K into a NoPE half (no rotary) and a
PE half (rotary). RoPE is applied IN-PLACE to the PE half only — the
NoPE half is untouched. The kernels operate over the per-token tile
``q_tile_size`` (default 16) along the token axis; the runtime
infers each token's absolute position from
``paged_kv_indptr_buffer`` / ``paged_kv_last_page_len_buffer``.

Why three Q variants
--------------------

The Q-side kernel must address different memory layouts depending on
how the upstream ``q_b`` GEMM stages its output(s):

* ``single``  (``deepseek_mla_rope_q_layer``) — separate NoPE and PE Q
  tensors. The kernel reads ``q_nope_pe`` (the un-rotated copy) only
  for dependency-ordering; it actually writes RoPE results into the
  ``q_pe`` tensor in place. ``has_split_q`` (1 in pk params) tells the
  kernel the PE tensor is its own DTensor and not aliased into a fused
  Q-NoPE-PE buffer.
* ``fused``   (``deepseek_mla_rope_q_fused_layer``) — single
  ``q_nope_pe`` DTensor of shape ``(T_max, H * (D_NOPE + D_PE))``
  laid out per-head NoPE-then-PE in memory. The kernel applies RoPE to
  the PE half of each row in place. Used when the ``q_b`` GEMM writes
  fused Q tiles.
* ``split``   (``deepseek_mla_rope_q_split_layer``) — like ``single``
  but with explicit ``qfused_mode`` for the row-swap addressing used
  by the chunked-prefill path (when ``q_pe`` is aliased to the same
  buffer as ``q_b_prefill_fused``, row stride = ``H * 192`` and the PE
  block lives at ``H * 128`` within each row). ``qfused_mode=1``
  enables that addressing; ``qfused_mode=0`` treats ``q_pe`` as a
  standalone ``(T_max, H * 64)`` tensor.

K-side RoPE
-----------

``deepseek_mla_rope_k_layer`` is conceptually simpler — there is a
single ``k_pe`` tensor of shape ``(T_max, D_KPE)`` (no head fan-out
on the K side under MLA). The same in-place-on-a-slice trick from the
QKV-a fused path is supported via ``k_pe_row_stride`` /
``k_pe_offset`` so the kernel can address ``k_pe`` as a slice of the
2176-wide ``qkv_a_out`` buffer ([2048:2112) for DeepSeek V3).

Tensor contract
---------------

Let ``T_max`` = ``max_num_batched_tokens``, ``H`` = ``num_heads`` (q
side, e.g. 128 for DeepSeek V3 single-GPU, 16 per rank at TP=8),
``D_NOPE`` = ``qk_nope_head_dim`` (128), ``D_PE`` =
``qk_rope_head_dim`` (64), ``D_K`` = ``D_NOPE + D_PE`` (192).

* Q variants
    * ``fused`` ``q_nope_pe``: ``(T_max, H * D_K)`` bf16 — per row
      ``[h0_nope (D_NOPE) | h0_pe (D_PE) | h1_nope | h1_pe | ...]``.
      RoPE applied in-place to the PE slice of each head.
    * ``single`` / ``split``:
        * ``q_nope_pe`` (single only): ``(T_max, H * D_K)`` bf16 —
          read as a barrier-only dependency.
        * ``q_pe``: ``(T_max, H * D_PE)`` bf16 — RoPE applied in
          place. Under ``qfused_mode=1`` this DTensor aliases the
          row-swap ``q_b_prefill_fused`` buffer; see the kernel for
          the addressing.
* K side
    * ``k_pe``: ``(T_max, D_PE)`` bf16. RoPE applied in place. May be
      a slice of a wider parent buffer — see ``k_pe_row_stride`` /
      ``k_pe_offset``.

* ``cos_pos_embed`` / ``sin_pos_embed``: ``(max_seq_len, D_PE)`` bf16
  RoPE tables. The kernel indexes them per absolute position of each
  token in the current step's new-token range.

Meta-tensor dependencies
------------------------

The kernel reads ``qo_indptr_buffer``, ``paged_kv_indptr_buffer``, and
``paged_kv_last_page_len_buffer`` to compute each token's absolute
position. In test mode the driver MUST set non-trivial values for
those buffers if any rotation is to happen.

Parallelism axis
----------------

``grid_dim == (R, H, ceil(T_max / q_tile_size))`` is the canonical
shape used by ``demo/deepseek_v3/builder.py`` for the Q variants
(``rope_q_grid``). ``deepseek_mla_rope_k_layer`` is typically launched
with ``grid_dim == (R, 1, ceil(T_max / q_tile_size))`` because there
is no H axis on the K side. ``q_tile_size`` defaults to 16 — must
match the kernel template.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

RopeQVariant = Literal["single", "fused", "split"]


class MLARopeQ(MPKModule):
    """MLA RoPE on the Q tensor (3 layout variants).

    Wraps :meth:`PersistentKernel.deepseek_mla_rope_q_layer` and its
    ``_fused_layer`` / ``_split_layer`` siblings 1:1. The variant flag
    picks which pk method (and therefore which task) is registered:

    * ``variant="single"`` -> task ``deepseek_mla_rope_q_sm100``
      (two-input form: separate NoPE-PE Q tensor + standalone PE Q).
    * ``variant="fused"``  -> task ``deepseek_mla_rope_q_fused_sm100``
      (one fused per-row [NoPE | PE] Q tensor).
    * ``variant="split"``  -> task ``deepseek_mla_rope_q_split_sm100``
      (standalone PE Q, with optional row-swap addressing via
      ``qfused_mode=1`` for the chunked-prefill row-swap path).

    Args:
        num_heads: Per-rank query head count fed to the kernel as
                template param ``NUM_HEADS``. For DeepSeek V3 this is
                128 on single GPU, 64/32/16 at TP=2/4/8.
        variant:  Which pk method to dispatch to. See above.
        q_tile_size: Per-block token tile (template ``Q_TILE``).
                Default 16 matches all current callers and must match
                the kernel template. ``grid_dim[2]`` must be
                ``ceil(T_max / q_tile_size)``.
        prefix:   HF state_dict key prefix (this module owns no
                parameters; prefix is unused at present).

    Forward
    -------
    Implements the rotate-half RoPE math for the PE half of Q in
    eager PyTorch, used as a correctness oracle in unit tests. The
    NoPE half is passed through unchanged.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        variant: RopeQVariant = "fused",
        q_tile_size: int = 16,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if variant not in ("single", "fused", "split"):
            raise ValueError(
                f"MLARopeQ variant must be 'single', 'fused', or "
                f"'split'; got {variant!r}"
            )
        self.num_heads = num_heads
        self.variant = variant
        self.q_tile_size = q_tile_size

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        """GPT-J interleaved rotation: pairs ``(x[2i], x[2i+1])`` rotate
        as ``(x[2i] -> -x[2i+1], x[2i+1] -> x[2i])``. This matches what
        the SM100 kernel (``deepseek_mla_rope_sm100.cuh``) does, and
        what vLLM/SGLang's MLA RoPE does. Distinct from HF's rotate-half
        convention used by Qwen3-style RoPE (see ``layers/rotary.py``).
        """
        # Split along the last dim into even/odd lanes, swap+negate.
        # x[..., ::2] = even lanes; x[..., 1::2] = odd lanes.
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1)
        return rotated.flatten(-2)

    def forward(
        self,
        q_pe: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """GPT-J interleaved RoPE on the PE half of Q.

        Matches the kernel's rotation convention. The kernel uses
        ``repeat_interleave``'d cos/sin (each base value appears twice
        in adjacent lanes), so we duplicate accordingly here.

        Args:
            q_pe: ``(T, H, D_PE)`` bf16. PE half of Q (NoPE half is not
                touched by RoPE and is therefore not passed here).
            cos:  ``(max_seq_len, D_PE)`` cos table. Either pre-doubled
                (D_PE = 2 * d_half via repeat_interleave) or raw
                (D_PE == d_half — we'll expand here).
            sin:  ``(max_seq_len, D_PE)`` sin table, same convention.
            positions: ``(T,)`` int64/int32 absolute positions per
                token.

        Returns:
            ``(T, H, D_PE)`` rotated PE half.
        """
        pos = positions.to(torch.long)
        cos_e = cos.index_select(0, pos).unsqueeze(1).to(q_pe.dtype)
        sin_e = sin.index_select(0, pos).unsqueeze(1).to(q_pe.dtype)
        return (q_pe * cos_e) + (self._rotate_interleaved(q_pe) * sin_e)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid: ``(R, H, ceil(T_max / q_tile_size))``.

        Matches ``rope_q_grid`` in
        ``python/mirage/mpk/models/deepseek_v3/builder.py`` line
        ~1805 — one task per (request, head, token-tile).
        """
        pk = current_pk()
        t_blocks = (pk.max_num_batched_tokens + self.q_tile_size - 1) // self.q_tile_size
        return (pk.max_num_batched_requests, self.num_heads, t_blocks)

    def default_block_dim(self) -> BlockDim:
        """All three Q-RoPE pk methods hard-default to ``(128, 1, 1)``."""
        return (128, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        q_pe: DTensor,
        cos_pos_embed: DTensor,
        sin_pos_embed: DTensor,
        *,
        q_nope_pe: Optional[DTensor] = None,
        has_split_q: bool = False,
        qfused_mode: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register the chosen MLA-Q-RoPE task on the current PK.

        Required positional args differ by ``variant``:

        * ``variant="single"``: pass both ``q_pe`` (in-place PE target)
          and ``q_nope_pe`` (read-only barrier dependency). The kernel
          name is ``deepseek_mla_rope_q_sm100`` and the layer also
          accepts ``has_split_q`` (1 if PE is standalone, 0 otherwise).
        * ``variant="fused"``: pass ``q_pe`` as the fused ``q_nope_pe``
          DTensor of shape ``(T_max, H * (D_NOPE + D_PE))``. RoPE is
          applied in-place to the PE slice of each row.
        * ``variant="split"``: pass ``q_pe`` (standalone PE target).
          ``qfused_mode`` (0 or 1) selects the standalone vs row-swap
          addressing; see module docstring.

        Args:
            q_pe: Primary in-place PE target (or fused tensor for
                  ``variant="fused"``).
            cos_pos_embed: ``(max_seq_len, D_PE)`` cos table DTensor.
            sin_pos_embed: ``(max_seq_len, D_PE)`` sin table DTensor.
            q_nope_pe: Required for ``variant="single"`` only —
                barrier-dependency input on the NoPE-fused Q.
            has_split_q: ``variant="single"`` only — passes 1 to the
                kernel when ``q_pe`` is standalone (the common case).
            qfused_mode: ``variant="split"`` only — 0 for standalone
                ``q_pe`` (legacy), 1 for the row-swap addressing into
                the ``q_b_prefill_fused`` buffer.
            grid_dim / block_dim: overrides; ``None`` falls back to
                :meth:`auto_grid_dim` / :meth:`default_block_dim`.

        Returns:
            The (in-place-rotated) ``q_pe`` DTensor, so a downstream
            ``compile()`` can chain on it.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # PersistentKernel.deepseek_mla_rope_q[_fused,_split]_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.variant == "single":
            if q_nope_pe is None:
                raise ValueError(
                    "MLARopeQ(variant='single').compile requires "
                    "q_nope_pe (the fused NoPE-PE tensor) as a barrier "
                    "dependency."
                )
            params = [self.num_heads, self.q_tile_size, 1 if has_split_q else 0]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            # Duplicate Q tensors are used as task outputs. This gives
            # downstream MLA tasks a real dependency on the in-place RoPE
            # write without joining the independent K-RoPE dependency chain.
            tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
            tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
            tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [
                    q_nope_pe,
                    q_pe,
                    cos_pos_embed,
                    sin_pos_embed,
                    q_nope_pe,
                    q_pe,
                ],
                tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "deepseek_mla_rope_q_sm100", params
            )
        elif self.variant == "fused":
            if q_nope_pe is not None:
                raise ValueError(
                    "MLARopeQ(variant='fused') takes a single "
                    "fused-NoPE-PE tensor — pass it as q_pe; do not "
                    "also pass q_nope_pe."
                )
            # For the fused variant, q_pe IS the fused q_nope_pe tensor.
            params = [self.num_heads, self.q_tile_size]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
            tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [q_pe, cos_pos_embed, sin_pos_embed, q_pe],
                tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "deepseek_mla_rope_q_fused_sm100", params
            )
        else:  # "split"
            if q_nope_pe is not None:
                raise ValueError(
                    "MLARopeQ(variant='split') takes only q_pe; do "
                    "not pass q_nope_pe."
                )
            # qfused_mode = 0: q_pe is a standalone (mbt, num_heads*64) tensor.
            # qfused_mode = 1: q_pe is the same DTensor as the fused
            # q_b_prefill buffer (mbt, num_heads*192) with row-swap layout.
            # Kernel uses row_stride = num_heads*192 and
            # pe_base_in_row = num_heads*128.
            params = [self.num_heads, self.q_tile_size]
            if qfused_mode != 0:
                params.append(qfused_mode)
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
            tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [q_pe, cos_pos_embed, sin_pos_embed, q_pe],
                tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "deepseek_mla_rope_q_split_sm100", params
            )
        return q_pe


class MLARopeK(MPKModule):
    """MLA RoPE on the K-PE tensor (single variant, in-place).

    Wraps :meth:`PersistentKernel.deepseek_mla_rope_k_layer` 1:1 — task
    ``deepseek_mla_rope_k_sm100``. The K side under MLA has no head
    fan-out (a single shared K-PE vector per token, hence the absence
    of a ``num_heads`` arg). RoPE is applied in place.

    The optional ``k_pe_row_stride`` / ``k_pe_offset`` kwargs let the
    kernel run in-place on a slice of a wider parent buffer (the
    DeepSeek V3 ``qkv_a_out`` 2176-wide row, with K-PE at
    ``[2048:2112)``).

    Args:
        q_tile_size: Per-block token tile (template ``Q_TILE``).
                Default 16, must match the kernel template.
        prefix:    HF state_dict key prefix (unused — no params).

    Forward
    -------
    GPT-J interleaved RoPE math, applied to the PE half of K (no head
    axis). Matches the kernel convention (see :meth:`MLARopeQ.forward`).
    """

    def __init__(
        self,
        *,
        q_tile_size: int = 16,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.q_tile_size = q_tile_size

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        """GPT-J interleaved rotation; see :meth:`MLARopeQ._rotate_interleaved`."""
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1)
        return rotated.flatten(-2)

    def forward(
        self,
        k_pe: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """GPT-J interleaved RoPE on K-PE.

        Args:
            k_pe: ``(T, D_PE)`` bf16. K-PE tensor (no head axis).
            cos:  ``(max_seq_len, D_PE)`` cos table.
            sin:  ``(max_seq_len, D_PE)`` sin table.
            positions: ``(T,)`` absolute positions per token.

        Returns:
            ``(T, D_PE)`` rotated K-PE.
        """
        pos = positions.to(torch.long)
        cos_e = cos.index_select(0, pos).to(k_pe.dtype)
        sin_e = sin.index_select(0, pos).to(k_pe.dtype)
        return (k_pe * cos_e) + (self._rotate_interleaved(k_pe) * sin_e)

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid: ``(R, 1, ceil(T_max / q_tile_size))``.

        Matches ``demo/deepseek_v3/builder.py`` line ~1836 — one task
        per (request, token-tile); no head axis on the K side under
        MLA.
        """
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
        """Register the K-PE RoPE task on the current PK.

        Args:
            k_pe: ``(T_max, D_PE)`` bf16 — in-place rotated. May alias
                a slice of a wider buffer (see ``k_pe_row_stride`` /
                ``k_pe_offset``).
            cos_pos_embed: ``(max_seq_len, D_PE)`` cos table DTensor.
            sin_pos_embed: ``(max_seq_len, D_PE)`` sin table DTensor.
            k_pe_row_stride: optional override for the parent buffer's
                row stride in elements. Default ``None`` -> the kernel
                uses ``128`` (the standalone-K-PE width assumption).
            k_pe_offset: column offset (elements) into the parent
                buffer where the K-PE slice starts. Default 0
                (standalone tensor).
            grid_dim / block_dim: overrides; ``None`` falls back to
                :meth:`auto_grid_dim` / :meth:`default_block_dim`.

        Returns:
            The in-place rotated ``k_pe`` DTensor.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # PersistentKernel.deepseek_mla_rope_k_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        # k_pe_row_stride / k_pe_offset support running the K_PE rotation
        # in-place on a slice of a wider buffer (e.g., qkv_a_out (mbt, 2176)
        # where k_pe lives at cols [2048:2112)). Defaults preserve legacy.
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
            [
                k_pe,
                cos_pos_embed,
                sin_pos_embed,
                k_pe,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "deepseek_mla_rope_k_sm100", params)
        return k_pe
