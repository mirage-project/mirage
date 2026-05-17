"""MLA paged-KV gather (catalog module).

This is the catalog counterpart to three pk methods on
:class:`PersistentKernel` (see ``python/mirage/mpk/persistent_kernel.py``):

* ``mla_kv_gather_layer``        (task ``mla_kv_gather_sm100``)
* ``mla_kv_gather_split_layer``  (task ``mla_kv_gather_split_sm100``)
* ``mla_kv_gather_unified_layer``(task ``mla_kv_gather_unified_sm100``)

All three append the **new** per-token MLA latent vectors (``c_latent``,
the low-rank KV joint, and ``k_pe``, the K rotary-positional component)
to the paged KV cache for the current step AND materialise a contiguous
view of the per-request KV history that downstream MLA-attention kernels
can read with TMA. They differ only in the output layout(s) they emit:

* ``standard``  — appends to paged cache, writes a single contiguous
  ``[max_seq_len_pad, D_K]`` slab (``D_K = D_V + D_KPE``, e.g. 576 for
  DeepSeek V3) with ``c_latent`` and ``k_pe`` concatenated along the
  last dim. Consumed by ``mla_decode_sm100`` /
  ``mla_prefill_absorbed_sm100``.
* ``split``     — appends to paged cache, writes TWO contiguous slabs:
  ``ckv_sep`` of shape ``[max_seq_len_pad, D_V]`` and ``kpe_sep`` of
  shape ``[max_seq_len_pad, D_KPE]``. This is the layout
  ``mla_prefill_sm100`` (non-absorbed prefill) expects.
* ``unified``   — appends to paged cache **once** and materialises BOTH
  outputs (the contiguous concat slab and the split slabs). Used by
  DeepSeek V3's `_use_prefill` path which dispatches prefill vs decode
  from runtime ``Q_LEN`` and so needs both layouts available without
  paying for two paged-cache appends.

Per-request iteration
---------------------

Each task instance handles ONE request (``request_id`` baked into
``task_desc->task_metadata`` by the MPK runtime). The kernel reads:

* ``c_latent_new[first_token_pos : first_token_pos + num_new_tokens, :]``
  and ``k_pe_new[first_token_pos : first_token_pos + num_new_tokens, :]``
  where ``first_token_pos = qo_indptr_buffer[request_id]`` and
  ``num_new_tokens = qo_indptr_buffer[request_id + 1] - first_token_pos``
  (this is what makes the same task graph handle prefill and decode).
* the request's page list
  ``paged_kv_indices_buffer[paged_kv_indptr_buffer[request_id]
   : paged_kv_indptr_buffer[request_id + 1]]`` to know which physical
  pages in the slab belong to this request.

After append, it walks the full per-request sequence (already-cached
pages + the new tokens) and writes the contiguous view(s) at row offset
``request_id * max_seq_len_pad`` in the output buffer.

Tensor contract
---------------

Let ``T_max`` = ``max_num_batched_tokens``, ``R`` =
``max_num_batched_requests``, ``D_V`` = ``kv_lora_rank`` (the
low-rank KV joint dim, 512 for DeepSeek V3), ``D_KPE`` =
``qk_rope_head_dim`` (rotary-position part of K, 64 for DeepSeek V3),
``D_K`` = ``D_V + D_KPE`` (the per-token MLA latent width fed to the
absorbed attention kernel), ``P`` = ``page_size``, ``N`` =
``max_num_pages``.

* ``c_latent_new``: ``(T_max, D_V)`` bf16. NEW per-token c_latent
  vectors, post-``kv_a_layernorm``. May be a slice of a wider parent
  buffer — see ``c_latent_row_stride`` / ``c_latent_offset_elems``.
* ``k_pe_new``:     ``(T_max, D_KPE)`` bf16. NEW per-token k_pe
  vectors, post-RoPE. May likewise be sliced.
* ``paged_cache``:  ``(N, P, D_K)`` bf16. Per-layer paged slab. The
  kernel appends new tokens here AND reads back the request's full
  sequence.
* ``contiguous_kv`` (``standard`` / ``unified``): ``(R * S_pad, D_K)``
  bf16 — destination for the concat layout. ``S_pad`` is the
  per-request stride MLA decode uses (typically rounded up to a TILE_S
  multiple).
* ``ckv_sep``       (``split`` / ``unified``): ``(R * S_pad, D_V)``
  bf16 — destination for the c_latent-only slab consumed by
  ``mla_prefill_sm100`` non-absorbed prefill.
* ``kpe_sep``       (``split`` / ``unified``): ``(R * S_pad, D_KPE)``
  bf16 — destination for the k_pe-only slab consumed by
  ``mla_prefill_sm100`` non-absorbed prefill.

Stride / offset overrides (``standard`` and ``unified`` only)
-------------------------------------------------------------

DeepSeek V3 fuses ``kv_a_proj`` + ``q_a_proj`` into a single ``qkv_a``
GEMM whose output rows have layout::

    row[0 : 1536)     -> q_a_proj output (used by q_b path)
    row[1536 : 2048)  -> c_latent (kv_lora_rank = 512)
    row[2048 : 2112)  -> k_pe       (qk_rope_head_dim = 64)
    row[2112 : 2176)  -> pad / unused

``c_latent_row_stride`` / ``c_latent_offset_elems`` /
``k_pe_row_stride`` / ``k_pe_offset_elems`` let the kernel address the
right slice of that wider buffer without an extra copy. Defaults
preserve legacy "contiguous c_latent / k_pe input" layouts.

The ``split`` variant does NOT accept slice overrides — its input
tensors must be standalone (slicing parent buffers was never wired
through ``mla_kv_gather_split_sm100``).

Meta-tensor dependencies
------------------------

Reads from the runtime ``runtime_config``:

* ``qo_indptr_buffer`` — per-request new-token offsets
* ``paged_kv_indptr_buffer`` / ``paged_kv_indices_buffer`` /
  ``paged_kv_last_page_len_buffer`` — the paged page table

Parallelism axis
----------------

``grid_dim == (max_num_batched_requests, 1, 1)`` — one task per
request. The kernel iterates over both the new-token slice and the
historic page list within a single block.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

KVGatherVariant = Literal["standard", "split", "unified"]


class MLAKVGather(MPKModule):
    """Paged MLA KV gather + contiguous-view materialisation.

    Wraps three pk methods that differ only in the output layout(s) they
    emit. Pick a ``variant``:

    * ``"standard"`` -> :meth:`PersistentKernel.mla_kv_gather_layer`
      -> task ``mla_kv_gather_sm100``. One concat output
      ``contiguous_kv: (R*S, D_K)``.
    * ``"split"``    -> :meth:`PersistentKernel.mla_kv_gather_split_layer`
      -> task ``mla_kv_gather_split_sm100``. Two separate outputs
      ``(ckv_sep, kpe_sep)`` of shapes ``(R*S, D_V)`` and ``(R*S, D_KPE)``.
    * ``"unified"``  -> :meth:`PersistentKernel.mla_kv_gather_unified_layer`
      -> task ``mla_kv_gather_unified_sm100``. Emits BOTH the concat
      output AND the split outputs in a single paged-cache append.

    Args:
        d_k:    Per-token MLA latent width fed to the absorbed
                attention kernel. For DeepSeek V3 this is
                ``kv_lora_rank + qk_rope_head_dim == 512 + 64 == 576``.
        d_v:    Width of the c_latent half. Equals ``kv_lora_rank``
                (512 for DeepSeek V3). Also the width of the K-V "value"
                consumed by attention (the absorbed kernel reuses
                c_latent as V).
        page_size: Page size of the paged KV cache. Must match
                ``pk.page_size``.
        variant: ``"standard"`` | ``"split"`` | ``"unified"``.
        prefix: HF state_dict key prefix. KVGather owns no parameters
                today, so this is currently only used to name the task
                in any debug output.

    Forward
    -------
    ``forward()`` is intentionally NOT IMPLEMENTED. The reference
    semantics for paged gather are intrinsically tied to MPK runtime
    meta-tensors (``qo_indptr_buffer``, ``paged_kv_indptr_buffer``,
    etc.) and to physical page allocation, neither of which a plain
    PyTorch reference can model without recreating the entire runtime.
    Use the test-mode PK driver instead (see
    ``tests/runtime_python/blackwell/sm100_mla/test_mla_kv_gather_testmode.py``).
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        page_size: int,
        *,
        variant: KVGatherVariant = "standard",
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if variant not in ("standard", "split", "unified"):
            raise ValueError(
                f"MLAKVGather variant must be one of "
                f"'standard'/'split'/'unified'; got {variant!r}"
            )
        self.d_k = d_k
        self.d_v = d_v
        self.page_size = page_size
        self.variant = variant

    # ------------------------------------------------------------------
    # PyTorch reference — intrinsically tied to MPK runtime metadata.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MLAKVGather.forward() is not implementable as a plain "
            "PyTorch reference: the gather depends on MPK runtime "
            "meta-tensors (qo_indptr_buffer / paged_kv_indptr_buffer / "
            "paged_kv_indices_buffer / paged_kv_last_page_len_buffer) "
            "and on physical page allocation in the paged slab. Use the "
            "test-mode driver (tests/runtime_python/blackwell/sm100_mla/"
            "test_mla_kv_gather_testmode.py) for end-to-end validation."
        )

    # ------------------------------------------------------------------
    # Grid heuristic — one task per request.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid: ``(max_num_batched_requests, 1, 1)``.

        Matches ``demo/deepseek_v3/builder.py`` callers (~lines 1896,
        1959). The kernel iterates per request internally; the y/z axes
        are unused.
        """
        pk = current_pk()
        return (pk.max_num_batched_requests, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """The gather kernel hard-wires 128 threads per block on SM100.

        Overrides the base ``MPKModule`` default which would otherwise
        return 256 on Hopper/Blackwell. See pk callers (always pass
        ``block_dim=(128, 1, 1)``).
        """
        return (128, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        c_latent_new: DTensor,
        k_pe_new: DTensor,
        paged_cache: DTensor,
        *,
        contiguous_kv: Optional[DTensor] = None,
        ckv_sep: Optional[DTensor] = None,
        kpe_sep: Optional[DTensor] = None,
        c_latent_row_stride: Optional[int] = None,
        c_latent_offset_elems: int = 0,
        k_pe_row_stride: Optional[int] = None,
        k_pe_offset_elems: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> None:
        """Register the chosen ``mla_kv_gather*`` task on the current PK.

        Required output args vary by ``variant``:

        * ``standard``: pass ``contiguous_kv``.
        * ``split``:    pass ``ckv_sep`` and ``kpe_sep``.
        * ``unified``:  pass ``contiguous_kv``, ``ckv_sep``, and
          ``kpe_sep``.

        The caller is responsible for allocating the output DTensors
        (gather doesn't have a single canonical output type — the
        downstream consumer chooses the layout).

        Slice-override kwargs (``c_latent_row_stride`` etc.) are accepted
        only on the ``standard`` and ``unified`` variants; passing them
        to ``split`` raises.

        Returns ``None`` — KVGather is consumed for its side effects on
        the output DTensors which the caller already owns.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        d_k, d_v, page_size = self.d_k, self.d_v, self.page_size

        # Inlined task registration (the body that used to live on
        # PersistentKernel.mla_kv_gather[_split,_unified]_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.variant == "standard":
            if contiguous_kv is None:
                raise ValueError(
                    "MLAKVGather(variant='standard').compile requires "
                    "contiguous_kv (the [R*S, D_K] concat output)."
                )
            if ckv_sep is not None or kpe_sep is not None:
                raise ValueError(
                    "MLAKVGather(variant='standard') emits only "
                    "contiguous_kv; pass variant='unified' for both."
                )
            # Stride/offset overrides let the kernel read c_latent / k_pe
            # from a wider parent buffer (QKV-a fused). Defaults preserve
            # legacy contiguous inputs.
            slice_override = (
                c_latent_row_stride is not None
                or c_latent_offset_elems != 0
                or k_pe_row_stride is not None
                or k_pe_offset_elems != 0
            )
            if slice_override:
                params = [
                    d_k, d_v, page_size,
                    c_latent_row_stride if c_latent_row_stride is not None else d_v,
                    c_latent_offset_elems,
                    k_pe_row_stride if k_pe_row_stride is not None else 128,
                    k_pe_offset_elems,
                ]
            else:
                params = [d_k, d_v, page_size]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(c_latent_new, (-1, 1, -1), -1, True)
            tb_graph.new_input(k_pe_new, (-1, 1, -1), -1, True)
            tb_graph.new_input(paged_cache, (-1, 2, -1), 1, True)
            tb_graph.new_input(contiguous_kv, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [c_latent_new, k_pe_new, paged_cache, contiguous_kv], tb_graph
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_kv_gather_sm100", params
            )
        elif self.variant == "split":
            if ckv_sep is None or kpe_sep is None:
                raise ValueError(
                    "MLAKVGather(variant='split').compile requires both "
                    "ckv_sep and kpe_sep DTensors."
                )
            if contiguous_kv is not None:
                raise ValueError(
                    "MLAKVGather(variant='split') emits only ckv_sep + "
                    "kpe_sep; pass variant='unified' if you also need "
                    "the concat output."
                )
            if (c_latent_row_stride is not None
                    or c_latent_offset_elems != 0
                    or k_pe_row_stride is not None
                    or k_pe_offset_elems != 0):
                raise ValueError(
                    "MLAKVGather(variant='split') does not support "
                    "stride/offset overrides — the split task expects "
                    "standalone c_latent / k_pe inputs."
                )
            params = [d_k, d_v, page_size]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(c_latent_new, (-1, 1, -1), -1, True)
            tb_graph.new_input(k_pe_new, (-1, 1, -1), -1, True)
            tb_graph.new_input(paged_cache, (-1, 2, -1), 1, True)
            tb_graph.new_input(ckv_sep, (-1, -1, -1), -1, True)
            tb_graph.new_input(kpe_sep, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [c_latent_new, k_pe_new, paged_cache, ckv_sep, kpe_sep],
                tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_kv_gather_split_sm100", params
            )
        else:  # "unified"
            if contiguous_kv is None or ckv_sep is None or kpe_sep is None:
                raise ValueError(
                    "MLAKVGather(variant='unified').compile requires "
                    "contiguous_kv, ckv_sep, and kpe_sep DTensors."
                )
            slice_override = (
                c_latent_row_stride is not None
                or c_latent_offset_elems != 0
                or k_pe_row_stride is not None
                or k_pe_offset_elems != 0
            )
            if slice_override:
                params = [
                    d_k, d_v, page_size,
                    c_latent_row_stride if c_latent_row_stride is not None else d_v,
                    c_latent_offset_elems,
                    k_pe_row_stride if k_pe_row_stride is not None else 128,
                    k_pe_offset_elems,
                ]
            else:
                params = [d_k, d_v, page_size]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(c_latent_new, (-1, 1, -1), -1, True)
            tb_graph.new_input(k_pe_new, (-1, 1, -1), -1, True)
            tb_graph.new_input(paged_cache, (-1, 2, -1), 1, True)
            tb_graph.new_input(contiguous_kv, (-1, -1, -1), -1, True)
            tb_graph.new_input(ckv_sep, (-1, -1, -1), -1, True)
            tb_graph.new_input(kpe_sep, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [
                    c_latent_new,
                    k_pe_new,
                    paged_cache,
                    contiguous_kv,
                    ckv_sep,
                    kpe_sep,
                ],
                tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_kv_gather_unified_sm100", params
            )
        return None
