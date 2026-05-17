"""MLA paged-KV append-and-gather (3 classes; one task per request).

Kernel files under ``include/mirage/persistent_kernel/tasks/blackwell/``:
``mla_kv_cache_gather_sm100.cuh`` hosts the ``standard`` (concat slab)
and ``unified`` (concat + split slabs) task impls;
``mla_kv_cache_gather_split_sm100.cuh`` hosts the ``split`` (two
separate ``ckv_sep`` / ``kpe_sep`` slabs) task impl. Each task appends
new ``c_latent``/``k_pe`` then materialises contiguous TMA views.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class _MLAKVGatherBase(MPKModule):
    """Shared paged-KV gather plumbing. Args:
    ``d_k`` (MLA latent width, 576 for DSv3),
    ``d_v`` (c_latent half = ``kv_lora_rank``, 512 for DSv3),
    ``page_size`` (must match ``pk.page_size``), ``prefix``.
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        page_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.d_k = d_k
        self.d_v = d_v
        self.page_size = page_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MLA KV gather has no plain-PyTorch reference: it depends on "
            "MPK runtime meta-tensors and physical page allocation. Use "
            "the test-mode driver (tests/runtime_python/blackwell/sm100_mla/"
            "test_mla_kv_gather_testmode.py)."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(max_num_batched_requests, 1, 1)`` — one task per request;
        the kernel iterates over new tokens + page list internally."""
        pk = current_pk()
        return (pk.max_num_batched_requests, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """The SM100 gather kernels hard-wire 128 threads/block."""
        return (128, 1, 1)


class MLAKVGatherStandard(_MLAKVGatherBase):
    """Append + materialise one concat slab ``(R * S_pad, D_K)``.

    Task ``mla_kv_gather_sm100``. Consumed by ``mla_decode_sm100`` /
    ``mla_prefill_absorbed_sm100``. Slice-override kwargs read
    ``c_latent`` / ``k_pe`` from a wider parent buffer.
    """

    def compile(
        self,
        c_latent_new: DTensor,
        k_pe_new: DTensor,
        paged_cache: DTensor,
        *,
        contiguous_kv: DTensor,
        c_latent_row_stride: Optional[int] = None,
        c_latent_offset_elems: int = 0,
        k_pe_row_stride: Optional[int] = None,
        k_pe_offset_elems: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> None:
        """Register ``mla_kv_gather_sm100``; append KV to paged cache and
        materialise a concat slab for decode TMA.

        Tensor contract:
          c_latent_new: (T_max, D_V=512) bf16; slice-override via
            ``c_latent_row_stride``/``c_latent_offset_elems``.
          k_pe_new: (T_max, 128) bf16; real rope [0:64), rest pad. Override
            via ``k_pe_row_stride``/``k_pe_offset_elems``.
          paged_cache: (max_num_pages, page_size, 576) bf16; [c_latent(512)|
            k_pe(64)] per pos. In-place append.
          contiguous_kv: (R, MPK_MAX_SEQ_LENGTH, 576) bf16 input; gather slab.
            Aliasing paged_cache skips gather.

        Notes: driven by request_id + qo/paged_kv_indptr/indices/last_page_len.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        d_k, d_v, page_size = self.d_k, self.d_v, self.page_size
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
        pk.kn_graph.register_task(tb_graph, "mla_kv_gather_sm100", params)
        return None


class MLAKVGatherSplit(_MLAKVGatherBase):
    """Append + materialise two slabs ``ckv_sep`` + ``kpe_sep``.

    Task ``mla_kv_gather_split_sm100``: ``(R*S_pad, D_V)`` and
    ``(R*S_pad, D_KPE)``; consumed by non-absorbed ``mla_prefill_sm100``.
    Slice-override kwargs are NOT accepted.
    """

    def compile(
        self,
        c_latent_new: DTensor,
        k_pe_new: DTensor,
        paged_cache: DTensor,
        *,
        ckv_sep: DTensor,
        kpe_sep: DTensor,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> None:
        """Register ``mla_kv_gather_split_sm100``; append new KV then gather
        into TWO separate dense slabs (CKV and KPE) for non-absorbed prefill.

        Tensor contract:
          c_latent_new: (T_max, D_V=512) bf16; per-token stride hard-wired
            to D_V (no slice-override).
          k_pe_new: (T_max, 128) bf16; rope cols [0:64), [64:128) zero pad.
          paged_cache: (max_num_pages, page_size, D_K=576) bf16; in-place
            append. Layout [c_latent(512)|k_pe(64)].
          ckv_sep: (R, MPK_MAX_SEQ_LENGTH, 512) bf16 input; CKV slab output.
          kpe_sep: (R, MPK_MAX_SEQ_LENGTH, 64) bf16 input; KPE slab output.

        Notes: slice-override kwargs NOT accepted (k_pe stride hard-wired
        128). Consumed by ``mla_prefill_sm100`` (chunked prefill).
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        d_k, d_v, page_size = self.d_k, self.d_v, self.page_size
        params = [d_k, d_v, page_size]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(c_latent_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(k_pe_new, (-1, 1, -1), -1, True)
        tb_graph.new_input(paged_cache, (-1, 2, -1), 1, True)
        tb_graph.new_input(ckv_sep, (-1, -1, -1), -1, True)
        tb_graph.new_input(kpe_sep, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [c_latent_new, k_pe_new, paged_cache, ckv_sep, kpe_sep], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "mla_kv_gather_split_sm100", params
        )
        return None


class MLAKVGatherUnified(_MLAKVGatherBase):
    """Append once + materialise BOTH concat and split slabs.

    Task ``mla_kv_gather_unified_sm100``: emits ``contiguous_kv`` AND
    ``(ckv_sep, kpe_sep)`` from a single append. Slice-override kwargs
    work as in :class:`MLAKVGatherStandard`.
    """

    def compile(
        self,
        c_latent_new: DTensor,
        k_pe_new: DTensor,
        paged_cache: DTensor,
        *,
        contiguous_kv: DTensor,
        ckv_sep: DTensor,
        kpe_sep: DTensor,
        c_latent_row_stride: Optional[int] = None,
        c_latent_offset_elems: int = 0,
        k_pe_row_stride: Optional[int] = None,
        k_pe_offset_elems: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> None:
        """Register ``mla_kv_gather_unified_sm100``; one append, then emit
        concat (decode) OR split (prefill) per runtime ``prompt_prefill_``.

        Tensor contract:
          c_latent_new: (T_max, D_V=512) bf16; override via
            ``c_latent_row_stride``/``c_latent_offset_elems``.
          k_pe_new: (T_max, 128) bf16; override via
            ``k_pe_row_stride``/``k_pe_offset_elems``.
          paged_cache: (max_num_pages, page_size, 576) bf16; in-place append.
          contiguous_kv: (R, MPK_MAX_SEQ_LENGTH, 576) bf16 input; decode slab.
          ckv_sep/kpe_sep: (R, MPK_MAX_SEQ_LENGTH, 512|64) bf16 task OUTPUTS
            (output_ptrs[0..1]); prefill slabs (Split treats these as inputs).

        Notes: prompt_prefill_ = q_len>8 ∧ step<prompt_length; decode skips
        gather when contiguous_kv aliases paged_cache."""
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        d_k, d_v, page_size = self.d_k, self.d_v, self.page_size
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


# ---------------------------------------------------------------------------
# Legacy variant-kwarg shim — kept for existing model/test/demo call sites.
# New code should use MLAKVGatherStandard / Split / Unified directly.
# ---------------------------------------------------------------------------
_KVG_VARIANT_CLASSES = {
    "standard": MLAKVGatherStandard,
    "split": MLAKVGatherSplit,
    "unified": MLAKVGatherUnified,
}


def MLAKVGather(
    d_k: int,
    d_v: int,
    page_size: int,
    *,
    variant: str = "standard",
    prefix: str = "",
) -> _MLAKVGatherBase:
    """Legacy dispatcher; returns the variant-specific subclass instance."""
    try:
        cls = _KVG_VARIANT_CLASSES[variant]
    except KeyError:
        raise ValueError(
            f"MLAKVGather variant must be one of {sorted(_KVG_VARIANT_CLASSES)}; "
            f"got {variant!r}"
        )
    return cls(d_k=d_k, d_v=d_v, page_size=page_size, prefix=prefix)
