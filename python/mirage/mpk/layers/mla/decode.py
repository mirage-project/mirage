"""MLA decode + reduce catalog modules (split-K flash attention).

This is the catalog counterpart to two pk methods on
:class:`PersistentKernel` (see ``python/mirage/mpk/persistent_kernel.py``):

* ``mla_decode_layer``  (task ``mla_decode_sm100``)
* ``mla_reduce_layer``  (task ``mla_reduce_sm100``)

These two tasks together implement split-K MLA flash attention for the
absorbed-attention path that DeepSeek V3 uses for decode (and short-Q
verify when MTP / spec-decode is enabled). The decode kernel computes
partial output / partial LSE for each of ``num_splits`` KV chunks; the
reduce kernel merges them into the final per-head attention output.

Note: the two layers are normally called consecutively, BUT they have
different tensor shapes (decode output is the partial ``[..., D_V]``
slab; reduce output is the final ``[B*Q_LEN, H, D_V]`` slab) and
different grid heuristics, so per the locked design they live in
separate modules.

Decode (``mla_decode_sm100``)
-----------------------------

Computes ``softmax(Q @ K^T / sqrt(D_K)) @ V`` for each (batch, head
group, KV split) triplet. Q is the per-head latent Q (NoPE-PE
concat); K is the per-token MLA latent (c_latent + k_pe = D_K =
D_V + D_KPE); V reuses c_latent (the "absorbed" trick — the o_proj's
absorption of the V-projection means the kernel emits an output of
width ``D_V``, not the projected width).

Inputs:

* ``q_input`` (TMA-desc attached): ``(B * Q_LEN * H, D_K)`` bf16 —
  per-token-per-head Q tile (NoPE 512 + PE 64 concatenated).
* ``kv_input`` (TMA-desc attached): ``(B * max_seq_len_pad, D_K)`` bf16
  — the MLA latent KV slab. For DeepSeek V3 this is the gather output
  of :class:`MLAKVGather` (``standard`` variant) with width
  ``D_K = 576``.
* ``output_partial``: ``(B * Q_LEN * num_splits, H * D_V)`` float32 (or
  bf16) — split-K partial output buffer. ``D_V`` is the c_latent
  half-width (= ``kv_lora_rank`` = 512 for DeepSeek V3).
* ``output_lse``: ``(B * Q_LEN * num_splits, H)`` float32 — split-K
  partial log-sum-exp buffer for the reduce kernel's correction.

Reduce (``mla_reduce_sm100``)
-----------------------------

Re-normalises and sums the partial outputs from ``num_splits`` KV
splits into a single per-head output. The kernel parallelises over
``(D_V / d_count, num_head_groups, B)``: each block reduces a
``d_count``-element chunk of the V dim for one head group and one
batch.

Inputs / outputs:

* ``input_partial`` — same shape as the decode's ``output_partial``.
* ``input_lse``     — same shape as the decode's ``output_lse``.
* ``output``        — ``(B * Q_LEN, H, D_V)`` bf16 final attention
  output, ready for ``o_proj``.

Parallelism axis
----------------

* Decode: ``(num_splits, num_head_groups, B)`` for ``Q_LEN == 1``;
  for ``Q_LEN > 1`` the grid becomes
  ``(num_splits, num_head_groups, B)`` with **block_linear** indexing
  inside the kernel (``bi * num_head_groups * sk + gi * sk + si``)
  reading the full partial buffer base pointer. The pk layer's
  ``new_input`` map encodes this — see the ``q_len > 1`` branch.
* Reduce: ``((D_V + d_count - 1) / d_count, num_head_groups, B)`` for
  ``Q_LEN == 1``; for ``Q_LEN > 1`` the kernel does its own
  ``block_linear`` offsetting and the layer must NOT auto-partition
  along grid.x.

Per-request iteration
---------------------

Each task instance handles ONE request (``request_id`` baked into
``task_desc->task_metadata`` by the MPK runtime).

Forward (PyTorch reference)
---------------------------

The compiled path is intrinsically tied to MPK runtime meta-tensors
(``paged_kv_indptr_buffer``, ``paged_kv_last_page_len_buffer``,
``qo_indptr_buffer``) AND to the split-K partition strategy. A faithful
reference would have to reconstruct each request's KV range from the
page table and replicate the split-K reduction. We mark
``forward()`` as ``NotImplementedError`` and recommend the test-mode
PK driver for end-to-end validation (see
``tests/runtime_python/blackwell/sm100_mla/``).
"""
from __future__ import annotations

from typing import Optional, Tuple

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MLADecode(MPKModule):
    """MLA split-K decode (partial output + partial LSE).

    Wraps :meth:`PersistentKernel.mla_decode_layer` 1:1 — task
    ``mla_decode_sm100``. Computes attention partials for each
    (batch, head group, KV split) tuple; pair with :class:`MLAReduce`
    to obtain the final per-head output.

    Args:
        num_heads:  Per-rank query head count (= 128 on single GPU
                    DeepSeek V3; 64/32/16 at TP=2/4/8). Baked into the
                    kernel template as ``NUM_HEADS``.
        d_k:        Per-token MLA latent width (=
                    ``kv_lora_rank + qk_rope_head_dim`` = 576 for
                    DeepSeek V3). Baked as ``D_K``.
        d_v:        c_latent half-width (= ``kv_lora_rank`` = 512 for
                    DeepSeek V3). Baked as ``D_V``.
        num_splits: Number of KV split-K chunks. Baked as ``NUM_SPLITS``.
        kv_len:     Maximum KV sequence length the kernel is compiled
                    for (used to compute the per-split tile bounds).
        q_len:      Number of new Q tokens per request the kernel
                    handles in one call (1 for plain decode; >1 for
                    MTP / spec-decode verify).
        prefix:     HF state_dict key prefix (this module owns no
                    parameters).

    Forward
    -------
    ``forward()`` is **not implemented**: the compiled kernel reads
    from MPK runtime meta-tensors (paged KV indices, qo indptr) and
    uses a non-trivial split-K partitioning scheme that has no clean
    eager-PyTorch counterpart. See module docstring.
    """

    def __init__(
        self,
        num_heads: int,
        d_k: int,
        d_v: int,
        num_splits: int,
        kv_len: int,
        *,
        q_len: int = 1,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.num_splits = num_splits
        self.kv_len = kv_len
        self.q_len = q_len

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MLADecode.forward() is not implemented as a plain "
            "PyTorch reference: the compiled kernel depends on MPK "
            "runtime meta-tensors (paged_kv_indptr_buffer, "
            "paged_kv_last_page_len_buffer, qo_indptr_buffer) and on "
            "the split-K partition scheme. Use the test-mode PK "
            "driver for validation (see tests/runtime_python/blackwell/"
            "sm100_mla/test_mla_decode_testmode.py)."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid: ``(num_splits, num_head_groups, R)``.

        Matches ``demo/deepseek_v3/builder.py`` line ~1709 for the
        plain mtp_decode call: ``(num_splits, num_head_groups,
        max_num_batched_requests)``. For decode ``num_head_groups`` is
        whatever the caller picks; this default uses ``num_heads`` as
        a placeholder that callers should override when they need a
        different head-group factoring.

        Note: for ``q_len > 1`` the pk layer uses input_map ``(-1,-1,-1)``
        for the partial tensors (grid.x maps to head_group, NOT a
        batch-partition); the grid shape itself is the same.
        """
        pk = current_pk()
        return (self.num_splits, self.num_heads, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
        """``mla_decode_sm100`` is a Hopper/Blackwell-only kernel that
        uses 128 threads/block. Override the base 256-default."""
        return (128, 1, 1)

    def compile(
        self,
        q_input: DTensor,
        kv_input: DTensor,
        output_partial: DTensor,
        output_lse: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register ``mla_decode_sm100`` on the current PK.

        Args:
            q_input:        ``(B*Q_LEN*H, D_K)`` bf16, TMA-desc attached.
            kv_input:       ``(B*max_seq_len_pad, D_K)`` bf16, TMA-desc.
            output_partial: ``(B*Q_LEN*num_splits, H*D_V)`` partial-O.
            output_lse:     ``(B*Q_LEN*num_splits, H)`` float32 LSE.
            grid_dim / block_dim: overrides; ``None`` -> auto.

        Returns:
            ``(output_partial, output_lse)`` — the same DTensors,
            returned for convenient chaining into :class:`MLAReduce`.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # PersistentKernel.mla_decode_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        q_len = self.q_len
        params = [
            self.num_heads,
            self.d_k,
            self.d_v,
            self.num_splits,
            self.kv_len,
            q_len,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_input, (0, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (0, -1, -1), -1, True)
        # When q_len > 1, grid is (num_splits, num_head_groups, 1). grid.y
        # blocks read the full output buffer and offset internally via
        # block_linear (bi*num_head_groups*sk + gi*sk + si). Don't partition.
        partial_map = (-1, -1, -1) if q_len > 1 else (0, -1, -1)
        tb_graph.new_input(output_partial, partial_map, -1, True)
        tb_graph.new_input(output_lse, partial_map, -1, True)
        pk.kn_graph.customized(
            [q_input, kv_input, output_partial, output_lse], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "mla_decode_sm100", params)
        return output_partial, output_lse


class MLAReduce(MPKModule):
    """MLA split-K reduce — merges per-split partials into final O.

    Wraps :meth:`PersistentKernel.mla_reduce_layer` 1:1 — task
    ``mla_reduce_sm100``. Paired downstream of :class:`MLADecode` to
    produce ``(B*Q_LEN, H, D_V)`` bf16 final attention output.

    Args:
        num_heads:  Per-rank query head count (must match the decode).
        d_v:        c_latent half-width (= ``kv_lora_rank`` = 512).
                    Baked as ``D_V``.
        num_splits: Number of KV split-K chunks (must match decode).
        d_start:    Starting offset along the V dim that THIS reduce
                    kernel covers. The base kernel uses ``0`` and
                    ``d_count == D_V``; the TP-variants split D_V.
        d_count:    Number of V-dim elements per CTA. Baked as
                    ``RD_DV`` template param.
        q_len:      Q tokens per request (1 for plain decode; >1 for
                    MTP / spec-decode).
        prefix:     HF state_dict key prefix.

    Forward
    -------
    Not implemented for the same reason as :class:`MLADecode`.
    """

    def __init__(
        self,
        num_heads: int,
        d_v: int,
        num_splits: int,
        d_start: int,
        d_count: int,
        *,
        q_len: int = 1,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_heads = num_heads
        self.d_v = d_v
        self.num_splits = num_splits
        self.d_start = d_start
        self.d_count = d_count
        self.q_len = q_len

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MLAReduce.forward() is not implemented; the reduce step "
            "is intrinsically tied to the decode's split-K partition. "
            "Use the test-mode PK driver for validation."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid: ``(ceil(D_V / d_count), num_heads, R)``.

        Matches the base ``mla_mtp_reduce_layer`` shape: each block
        reduces a ``d_count`` slice of the V dim for one head group
        and one batch.
        """
        pk = current_pk()
        d_blocks = (self.d_v + self.d_count - 1) // self.d_count
        return (d_blocks, self.num_heads, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
        """``mla_reduce_sm100`` uses 256 threads/block (see all callers)."""
        return (256, 1, 1)

    def compile(
        self,
        input_partial: DTensor,
        input_lse: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_reduce_sm100`` on the current PK.

        Args:
            input_partial: ``(B*Q_LEN*num_splits, H*D_V)`` from decode.
            input_lse:     ``(B*Q_LEN*num_splits, H)`` from decode.
            output:        ``(B*Q_LEN, H, D_V)`` bf16 final output.
            grid_dim / block_dim: overrides; ``None`` -> auto.

        Returns:
            The ``output`` DTensor.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # PersistentKernel.mla_reduce_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        q_len = self.q_len
        params = [
            self.num_heads,
            self.d_v,
            self.num_splits,
            self.d_start,
            self.d_count,
            q_len,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # When q_len > 1, grid.x maps to head_group (not batch). The kernel
        # uses block_linear = bi * num_head_groups * sk + gi * sk to offset
        # into the same shared input buffer, so we must NOT partition
        # input/output along grid.x — every block needs the full base pointer.
        partial_map = (-1, -1, -1) if q_len > 1 else (0, -1, -1)
        tb_graph.new_input(input_partial, partial_map, -1, True)
        tb_graph.new_input(input_lse, partial_map, -1, True)
        tb_graph.new_input(output, partial_map, -1, True)
        pk.kn_graph.customized(
            [input_partial, input_lse, output], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "mla_reduce_sm100", params)
        return output
