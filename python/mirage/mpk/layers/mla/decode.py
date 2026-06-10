"""MLA split-K decode + reduce catalog modules.

Wraps two SM100 tasks (kernels under
``include/mirage/persistent_kernel/tasks/blackwell/``):

* :class:`MLADecode`  -> ``mla_decode_sm100`` (codegen → ``mla_mtp_decode_sm100_task_impl``)
* :class:`MLAReduce`  -> ``mla_reduce_sm100``  (``mla_reduce_sm100.cuh``)

Both kernels bake in ``NUM_HEADS=128``, ``D_K=576``, ``D_V=512`` for
DeepSeek V3. Each task instance handles ONE request (request_id comes
from ``task_desc->task_metadata``); paged KV is consumed via the MPK
runtime's ``paged_kv_indptr_buffer`` / ``paged_kv_last_page_len_buffer``.
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

    Task ``mla_decode_sm100``. Pair with :class:`MLAReduce` to obtain
    the final per-head output. Inputs: ``q_input`` (B*Q_LEN*H, D_K)
    bf16 (TMA-desc) and ``kv_input`` (B*max_seq_len_pad, D_K) bf16
    (TMA-desc). Outputs split-K partial-O ``(B*Q_LEN*num_splits, H*D_V)``
    and partial-LSE ``(B*Q_LEN*num_splits, H)`` float32.
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
        """Not implemented: depends on MPK runtime meta-tensors (paged KV
        indptr, qo indptr) and the split-K partition scheme."""
        raise NotImplementedError("MLADecode.forward(): use test-mode PK driver.")

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(num_splits, num_heads, max_num_batched_requests)``;
        kernel has H=128, KV_H=1 so grid.y collapses to head-groups."""
        pk = current_pk()
        return (self.num_splits, self.num_heads, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
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
        """Register ``mla_decode_sm100`` (codegen routes to ``mla_mtp_decode_sm100_task_impl``).

        Tensor contract:
          q_input:        (R*Q_LEN*NUM_HEADS=128, D_K=576) bf16, row-major, TMA-desc (input_tma_desc_ptrs[0][0]).
          kv_input:       (R*KV_LEN, D_K=576) bf16, contiguous gathered paged-KV slab, TMA-desc (input_tma_desc_ptrs[1][0]).
          output_partial: (R*Q_LEN*NUM_SPLITS, NUM_HEADS*D_V=128*512) bf16, partial-O per split (kernel ``Oa``, output_ptrs[0]).
          output_lse:     (R*Q_LEN*NUM_SPLITS, NUM_HEADS=128) fp32, partial-LSE per split (kernel ``La``, output_ptrs[1]).

        Notes: paged-KV via ``paged_kv_indptr_buffer`` / ``paged_kv_last_page_len_buffer``;
        single-tile-per-split required (sk = ceil(kv_len/128)). For ``q_len > 1`` partial maps are
        ``(-1,-1,-1)`` so every block sees the full base and applies its own ``bi*nhg*sk + gi*sk + si`` offset.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

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
        partial_map = (-1, -1, -1) if q_len > 1 else (0, -1, -1)
        tb_graph.new_input(output_partial, partial_map, -1, True)
        tb_graph.new_input(output_lse, partial_map, -1, True)
        pk.kn_graph.customized(
            [q_input, kv_input, output_partial, output_lse], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "mla_decode_sm100", params)
        return output_partial, output_lse


class MLAReduce(MPKModule):
    """MLA split-K reduce: merges per-split partials into final O.

    Task ``mla_reduce_sm100``. Each block reduces a ``d_count`` slice
    of the V dim (``D_V=512``) for one head and one batch. Inputs:
    ``input_partial`` and ``input_lse`` from the decode; output
    ``(B, NUM_HEADS, D_V)`` bf16 ready for ``o_proj``.
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
        """Not implemented: tied to the upstream decode's split-K layout."""
        raise NotImplementedError("MLAReduce.forward(): use test-mode PK driver.")

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(ceil(D_V / d_count), num_heads, max_num_batched_requests)``."""
        pk = current_pk()
        d_blocks = (self.d_v + self.d_count - 1) // self.d_count
        return (d_blocks, self.num_heads, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
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
        """Register ``mla_reduce_sm100`` (codegen routes to ``mla_mtp_reduce_sm100_task_impl<256>``).

        Tensor contract:
          input_partial: (R*Q_LEN*NUM_SPLITS, NUM_HEADS*D_V=128*512) bf16, partial-O from MLADecode (input_ptrs[0]).
          input_lse:    (R*Q_LEN*NUM_SPLITS, NUM_HEADS=128) fp32, partial-LSE from MLADecode (input_ptrs[1]).
          output:       (B, NUM_HEADS=128, D_V=512) bf16, final attn output ready for ``o_proj`` (output_ptrs[0]).

        Notes: grid partitions output over (D_V/d_count, head_group, batch); each block reduces a ``d_count``
        slice for one head. For ``q_len > 1`` partial maps are ``(-1,-1,-1)`` so kernel applies its own block_linear offset.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

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
        partial_map = (-1, -1, -1) if q_len > 1 else (0, -1, -1)
        tb_graph.new_input(input_partial, partial_map, -1, True)
        tb_graph.new_input(input_lse, partial_map, -1, True)
        tb_graph.new_input(output, partial_map, -1, True)
        pk.kn_graph.customized(
            [input_partial, input_lse, output], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "mla_reduce_sm100", params)
        return output
