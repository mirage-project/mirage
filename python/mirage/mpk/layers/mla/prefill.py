"""MLA prefill catalog modules (7 single-purpose classes).

Each class wraps one SM100 prefill task. Kernels live under
``include/mirage/persistent_kernel/tasks/blackwell/``:

* :class:`MLAPrefillAbsorbed`           -> ``mla_prefill_absorbed_sm100`` (``mla_prefill_sm100.cuh`` absorbed branch)
* :class:`MLAPrefillPlain`              -> ``mla_prefill_sm100``          (``mla_prefill_sm100.cuh``)
* :class:`MLAPrefillUnified`            -> ``mla_unified_sm100``          (``mla_unified_sm100.cuh``)
* :class:`MLAPrefillTP8`                -> ``mla_prefill_tp8_sm100``      (``mla_prefill_tp8_sm100.cuh``)
* :class:`MLAPrefillTP8Chunked`         -> ``mla_prefill_tp8_chunked_sm100`` (``mla_prefill_tp8_chunked_sm100.cuh``)
* :class:`MLAPrefillTP8ChunkedSplitK`   -> ``mla_prefill_tp8_chunked_splitk_sm100`` (``mla_prefill_tp8_chunked_splitk_sm100.cuh``)
* :class:`MLAPrefillTP8ChunkedReduce`   -> ``mla_prefill_tp8_chunked_reduce_sm100``
"""
from __future__ import annotations

from typing import Optional, Tuple

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class _MLAPrefillBase(MPKModule):
    """Shared base for MLA prefill modules. Owns no parameters.

    Common dims: ``H`` = per-rank num query heads, ``D_CKV`` =
    ``kv_lora_rank`` (512), ``D_KPE`` = ``qk_rope_head_dim`` (64),
    ``D_K`` = 576, ``D_V`` = 512 absorbed / 128 TP8-unabsorbed.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_heads = num_heads

    def forward(self, *args, **kwargs):
        """Not implemented: depends on MPK runtime meta-tensors (paged KV
        indices, qo_indptr_buffer) with no clean eager-PyTorch counterpart."""
        raise NotImplementedError(
            f"{type(self).__name__}.forward(): use test-mode PK driver."
        )

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)


def _require(value, name: str, cls: str) -> None:
    if value is None:
        raise ValueError(f"{cls}.compile requires {name} (was None).")


# ---------------------------------------------------------------------------
# Absorbed prefill
# ---------------------------------------------------------------------------
class MLAPrefillAbsorbed(_MLAPrefillBase):
    """DeepSeek V3 absorbed-attention prefill (``mla_prefill_absorbed_sm100``).

    Q is the fused ``[S, H, D_CKV+D_KPE]`` per-head latent Q; KV is
    the contiguous ``[B*max_seq_len, D_CKV+D_KPE]`` slab from
    :class:`MLAKVGather` (``standard``/``unified``). Output width is
    ``D_V = D_CKV`` (o_proj absorbs V). Grid dominates along H.
    """

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d_ckv: int,
        d_kpe: int,
        d_v: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(num_heads, prefix=prefix)
        self.seq_len = seq_len
        self.d_ckv = d_ckv
        self.d_kpe = d_kpe
        self.d_v = d_v

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(H, ceil(seq_len/BM), max_num_batched_requests)`` with BM=64."""
        pk = current_pk()
        return (self.num_heads, (self.seq_len + 63) // 64, pk.max_num_batched_requests)

    def compile(
        self,
        q_nope_pe: DTensor,
        kv: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_prefill_absorbed_sm100`` (codegen aliases q_nope/q_pe to one fused Q and ckv/kpe to one fused KV).

        Tensor contract:
          q_nope_pe: (T_total, H*(D_CKV+D_KPE)=H*576) bf16, fused Q-latent (nope||pe) per-token (input_ptrs[0]).
          kv:        (R*MPK_MAX_SEQ_LENGTH, D_CKV+D_KPE=576) bf16, contiguous gathered KV slab from MLAKVGather (input_ptrs[1]).
          output:    (T_total, H*D_V=H*512) bf16, attn output ready for ``o_proj`` (output_ptrs[0]).

        Notes: row-strides forced to ``(D_CKV+D_KPE)`` (Q and KV share latent dim); each task handles ONE request,
        skipped when ``Q_LEN <= 8`` (handed off to decode). Meta-tensor deps: ``qo_indptr_buffer``,
        ``paged_kv_indptr_buffer``, ``paged_kv_last_page_len_buffer``, ``step``, ``prompt_length``.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_heads, self.seq_len, self.d_ckv, self.d_kpe, self.d_v]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([q_nope_pe, kv, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mla_prefill_absorbed_sm100", params)
        return output


# ---------------------------------------------------------------------------
# Plain (non-absorbed) prefill
# ---------------------------------------------------------------------------
class MLAPrefillPlain(_MLAPrefillBase):
    """Non-absorbed MLA prefill (``mla_prefill_sm100``).

    Q is split into NoPE/PE tensors; KV is the SPLIT layout from
    :class:`MLAKVGather` (``variant='split'``) — two separate ``ckv``
    and ``kpe`` tensors. The kernel computes its own (S, H, D)
    addressing; inputs use ``(-1,-1,-1)`` maps. Grid dominates along H.
    """

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        d_ckv: int,
        d_kpe: int,
        d_v: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(num_heads, prefix=prefix)
        self.seq_len = seq_len
        self.d_ckv = d_ckv
        self.d_kpe = d_kpe
        self.d_v = d_v

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(H, ceil(seq_len/BM), max_num_batched_requests)`` with BM=64."""
        pk = current_pk()
        return (self.num_heads, (self.seq_len + 63) // 64, pk.max_num_batched_requests)

    def compile(
        self,
        q_nope: DTensor,
        q_pe: DTensor,
        ckv: DTensor,
        kpe: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_prefill_sm100`` (non-absorbed split-buffer MLA prefill).

        Tensor contract:
          q_nope: (T_total, H*D_CKV=H*512) bf16, NoPE Q per-token (input_ptrs[0]).
          q_pe:   (T_total, H*D_KPE=H*64)  bf16, RoPE Q per-token (input_ptrs[1]).
          ckv:    (R*MPK_MAX_SEQ_LENGTH, D_CKV=512) bf16, NoPE KV slab from MLAKVGather(split) (input_ptrs[2]).
          kpe:    (R*MPK_MAX_SEQ_LENGTH, D_KPE=64)  bf16, RoPE  KV slab from MLAKVGather(split) (input_ptrs[3]).
          output: (T_total, H*D_V=H*512) bf16, attn output (output_ptrs[0]).

        Notes: kernel computes (S, H, D) addressing itself, so all input_maps are ``(-1,-1,-1)``;
        per-request slice via ``qo_indptr_buffer`` / paged-KV indptrs. Skipped when ``Q_LEN <= 8``.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_heads, self.seq_len, self.d_ckv, self.d_kpe, self.d_v]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(ckv, (-1, -1, -1), -1, True)
        tb_graph.new_input(kpe, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([q_nope, q_pe, ckv, kpe, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mla_prefill_sm100", params)
        return output


# ---------------------------------------------------------------------------
# Unified prefill-and-decode dispatch
# ---------------------------------------------------------------------------
class MLAPrefillUnified(_MLAPrefillBase):
    """Runtime prefill-vs-decode dispatch in one fused task (``mla_unified_sm100``).

    Reads BOTH prefill inputs (NoPE/PE/ckv/kpe) and decode inputs
    (TMA-attached fused Q, contiguous KV, partial-O, partial-LSE);
    branches on runtime Q_LEN. Grid is computed internally — callers
    MUST NOT pass ``grid_dim``.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        q_len: int,
        kv_len: int,
        d_ckv: Optional[int] = None,
        d_kpe: Optional[int] = None,
        d_v: Optional[int] = None,
        tp_size: int = 1,
        decode_q_len: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(num_heads, prefix=prefix)
        self.q_len = q_len
        self.kv_len = kv_len
        self.d_ckv = d_ckv
        self.d_kpe = d_kpe
        self.d_v = d_v
        self.tp_size = tp_size
        self.decode_q_len = decode_q_len

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid is computed inside ``compile()``; calling this raises."""
        raise RuntimeError(
            "MLAPrefillUnified computes grid internally — do not call "
            "auto_grid_dim() and do not pass grid_dim to compile()."
        )

    def compile(
        self,
        q_nope: DTensor,
        q_pe: DTensor,
        ckv: DTensor,
        kpe: DTensor,
        output: DTensor,
        q_input: DTensor,
        kv_input: DTensor,
        output_partial: DTensor,
        output_lse: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_unified_sm100`` — runtime prefill-vs-decode dispatch in one fused task.

        Tensor contract:
          q_nope:         (T_total, H*D_CKV=H*512) bf16, prefill NoPE Q (input_ptrs[0]).
          q_pe:           (T_total, H*D_KPE=H*64)  bf16, prefill RoPE Q (input_ptrs[1]).
          ckv:            (R*MPK_MAX_SEQ_LENGTH, D_CKV=512) bf16, prefill NoPE KV slab (input_ptrs[2]).
          kpe:            (R*MPK_MAX_SEQ_LENGTH, D_KPE=64)  bf16, prefill RoPE KV slab (input_ptrs[3]).
          output:         (T_total, H*D_V=H*512) bf16, prefill attn output (output_ptrs[0]).
          q_input:        (R*decode_q_len*H, D_K=576) bf16, decode fused Q-latent, TMA-desc.
          kv_input:       (R*KV_LEN, D_K=576) bf16, decode gathered KV slab, TMA-desc.
          output_partial: (R*decode_q_len*NUM_SPLITS, H*D_V=H*512) bf16, decode partial-O (output_ptrs[1]).
          output_lse:     (R*decode_q_len*NUM_SPLITS, H) fp32, decode partial-LSE (output_ptrs[2]).

        Notes: branch on runtime ``Q_LEN`` (>8 prefill, else decode-split-K); grid computed internally
        from tp_size/decode_q_len/kv_len — callers MUST NOT pass ``grid_dim``. Meta deps: ``qo_indptr_buffer``,
        ``paged_kv_indptr_buffer``, ``paged_kv_indices_buffer``, ``paged_kv_last_page_len_buffer``, ``step``.
        """
        pk = current_pk()
        if grid_dim is not None:
            raise ValueError(
                "MLAPrefillUnified.compile() does not accept grid_dim — "
                "the task computes it internally."
            )
        if block_dim is None:
            block_dim = (256, 1, 1)

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        num_heads = self.num_heads
        q_len = self.q_len
        kv_len = self.kv_len
        tp_size = self.tp_size
        d_ckv = self.d_ckv if self.d_ckv is not None else 512
        d_kpe = self.d_kpe if self.d_kpe is not None else 64
        d_v = self.d_v if self.d_v is not None else 512

        num_splits = (kv_len + 128 - 1) // 128
        decode_q_len = self.decode_q_len if self.decode_q_len is not None else q_len
        decode_q_len = min(decode_q_len, 8)
        if tp_size == 1:
            hpb = num_heads // decode_q_len
            if hpb < 1:
                hpb = 1
            while num_heads % hpb != 0:
                hpb -= 1
            num_groups = num_heads // hpb
            x_mul = 1
        elif tp_size == 2:
            qpg = min(2, decode_q_len)
            num_groups = (decode_q_len + qpg - 1) // qpg
            x_mul = 1
        elif tp_size == 4:
            qpg = min(4, decode_q_len)
            num_groups = (decode_q_len + qpg - 1) // qpg
            x_mul = 2
        elif tp_size == 8:
            q_len_padded = (decode_q_len + 1) & ~1
            qpg = 2
            num_groups = (q_len_padded + qpg - 1) // qpg
            x_mul = 1
        else:
            raise ValueError(f"Unsupported MLA unified tp_size={tp_size}")

        num_q_blocks = (q_len + 64 - 1) // 64
        decode_blocks_x = num_groups * num_splits * x_mul
        grid_dim_u = (
            max(num_heads, decode_blocks_x),
            max(num_q_blocks, pk.max_num_batched_requests),
            pk.max_num_batched_requests,
        )
        params = [
            num_heads, decode_q_len, kv_len, num_splits,
            tp_size, d_ckv, d_kpe, d_v,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim_u, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(ckv, (-1, -1, -1), -1, True)
        tb_graph.new_input(kpe, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_lse, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_nope, q_pe, ckv, kpe, output, q_input, kv_input,
             output_partial, output_lse],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mla_unified_sm100", params)
        return output


# ---------------------------------------------------------------------------
# TP=8 unabsorbed prefill
# ---------------------------------------------------------------------------
class MLAPrefillTP8(_MLAPrefillBase):
    """TP=8 unabsorbed prefill (``mla_prefill_tp8_sm100``).

    16 heads per rank. Inputs (per-head shards): q_nope ``[B,S,H,128]``,
    q_pe ``[B,S,H,64]``, k ``[B,S,192]``, v ``[B,S,128]``; output
    ``[B,S,H,128]``. Grid dominates along H, BM=64.
    """

    def __init__(
        self,
        num_heads: int,
        seq_len: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(num_heads, prefix=prefix)
        self.seq_len = seq_len

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(H, ceil(seq_len/BM), max_num_batched_requests)`` with BM=64."""
        pk = current_pk()
        return (self.num_heads, (self.seq_len + 63) // 64, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        q_nope: DTensor,
        q_pe: DTensor,
        k: DTensor,
        v: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_prefill_tp8_sm100`` — TP=8 unabsorbed prefill (16 heads/rank).

        Tensor contract:
          q_nope: (B, S, H_rank, 128) bf16, per-head NoPE Q shard (input_ptrs[0]).
          q_pe:   (B, S, H_rank, 64)  bf16, per-head RoPE Q shard (input_ptrs[1]).
          k:      (B, S, 192) bf16, NoPE+RoPE concatenated K, TMA-desc (input_tma_desc_ptrs[2][0]).
          v:      (B, S, 128) bf16, V per token, TMA-desc (input_tma_desc_ptrs[3][0]).
          output: (B, S, H_rank, 128) bf16, attn output (output_ptrs[0]).

        Notes: grid ``(H_rank, ceil(S/BM=64), max_num_batched_requests)``; ``seq_len <= 4096``.
        request_id/kv_idx/merge_task_offset = head/q_block/batch.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.num_heads, self.seq_len]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(k, (-1, -1, -1), -1, True)
        tb_graph.new_input(v, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([q_nope, q_pe, k, v, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mla_prefill_tp8_sm100", params)
        return output


# ---------------------------------------------------------------------------
# TP=8 chunked prefill
# ---------------------------------------------------------------------------
class MLAPrefillTP8Chunked(_MLAPrefillBase):
    """TP=8 chunked unabsorbed prefill (``mla_prefill_tp8_chunked_sm100``).

    Iterates only over ``[q_start, q_start+q_len)`` of the Q axis.
    Inputs (BM=64, BN=128): q_nope/q_pe ``[B,q_len,H,128/64]``,
    k_nope ``[B,kv_len,H,128]``, k_rope ``[B,kv_len,1,64]``, v
    ``[B,kv_len,H,128]``, output ``[B,q_len,H,128]``. ``qfused_mode=1``
    aliases q_nope/q_pe to one width-192 DTensor.
    """

    def __init__(
        self,
        num_heads: int,
        q_len: int,
        kv_len: int,
        *,
        q_start: int = 0,
        qfused_mode: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(num_heads, prefix=prefix)
        self.q_len = q_len
        self.kv_len = kv_len
        self.q_start = q_start
        self.qfused_mode = qfused_mode

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(H, ceil(q_len/BM), max_num_batched_requests)`` with BM=64."""
        pk = current_pk()
        return (self.num_heads, (self.q_len + 63) // 64, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        q_nope: DTensor,
        q_pe: DTensor,
        k_nope: DTensor,
        k_rope: DTensor,
        v: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_prefill_tp8_chunked_sm100`` — TP=8 chunked unabsorbed prefill over ``[q_start, q_start+q_len)``.

        Tensor contract (BM=64, BN=128, H = H_rank):
          q_nope: (T_total, H*128) bf16 NoPE Q; if qfused_mode=1 aliases the (T_total, H*192) fused buffer (input_ptrs[0]).
          q_pe:   (T_total, H*64)  bf16 RoPE Q; if qfused_mode=1 same fused buffer + H*128 elem offset (input_ptrs[1]).
          k_nope: (R*kv_len_pad, H, 128) bf16, paged-KV NoPE TMA-desc (input_tma_desc_ptrs[2][0]).
          k_rope: (R*kv_len_pad, 1, 64)  bf16, paged-KV RoPE TMA-desc (input_tma_desc_ptrs[3][0]).
          v:      (R*kv_len_pad, H, 128) bf16, paged-KV V TMA-desc (input_tma_desc_ptrs[4][0]).
          output: (T_total, H*128) bf16, attn output (output_ptrs[0]).

        Notes: ``qfused_mode=1`` row-stride = H*192 with row-swap layout (nope-then-pe per row).
        Runs only when ``step < prompt_length && Q_LEN > 8``. Meta deps: ``qo_indptr_buffer``,
        ``paged_kv_indptr_buffer``, ``paged_kv_last_page_len_buffer``, ``step``, ``prompt_length``.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [
            self.num_heads, self.q_len, self.kv_len,
            self.q_start, self.qfused_mode,
        ]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_rope, (-1, -1, -1), -1, True)
        tb_graph.new_input(v, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_nope, q_pe, k_nope, k_rope, v, output], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "mla_prefill_tp8_chunked_sm100", params
        )
        return output


# ---------------------------------------------------------------------------
# TP=8 chunked split-K
# ---------------------------------------------------------------------------
class MLAPrefillTP8ChunkedSplitK(_MLAPrefillBase):
    """Split-K variant of TP=8 chunked prefill (``mla_prefill_tp8_chunked_splitk_sm100``).

    Computes per-(KV split, Q tile) partial O+LSE into a
    ``[num_splits, B, nqb, H, BM, D_V+4]`` float32 buffer (BM=64,
    D_V=128). Pair with :class:`MLAPrefillTP8ChunkedReduce` to merge.
    """

    def __init__(
        self,
        num_heads: int,
        q_len: int,
        kv_len: int,
        num_splits: int,
        *,
        q_start: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(num_heads, prefix=prefix)
        self.q_len = q_len
        self.kv_len = kv_len
        self.q_start = q_start
        self.num_splits = num_splits

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(H, nqb*num_splits, max_num_batched_requests)`` with nqb=ceil(q_len/64)."""
        pk = current_pk()
        nqb = (self.q_len + 63) // 64
        return (self.num_heads, nqb * self.num_splits, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        q_nope: DTensor,
        q_pe: DTensor,
        k_nope: DTensor,
        k_rope: DTensor,
        v: DTensor,
        partial: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> None:
        """Register ``mla_prefill_tp8_chunked_splitk_sm100`` — split-K of TP=8 chunked prefill (writes ``partial`` only).

        Tensor contract (BM=64, D_V=128, H = H_rank):
          q_nope:  (T_total, H*128) bf16, NoPE Q (input_ptrs[0]).
          q_pe:    (T_total, H*64)  bf16, RoPE Q (input_ptrs[1]).
          k_nope:  (R*kv_len_pad, H, 128) bf16, paged-KV NoPE TMA-desc (input_tma_desc_ptrs[2][0]).
          k_rope:  (R*kv_len_pad, 1, 64)  bf16, paged-KV RoPE TMA-desc (input_tma_desc_ptrs[3][0]).
          v:       (R*kv_len_pad, H, 128) bf16, paged-KV V    TMA-desc (input_tma_desc_ptrs[4][0]).
          partial: (num_splits, B, nqb, H, BM=64, D_V+4=132) fp32, per-(KV split, Q tile) O+LSE (output_ptrs[0]).

        Notes: nqb = ceil(q_len/64); kv_idx packs (q_block, split_id). Pair with MLAPrefillTP8ChunkedReduce.
        Meta deps mirror MLAPrefillTP8Chunked.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        nqb = (self.q_len + 63) // 64
        params = [
            self.num_heads, self.q_len, self.kv_len,
            self.q_start, self.num_splits, nqb,
        ]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_nope, (-1, -1, -1), -1, True)
        tb_graph.new_input(k_rope, (-1, -1, -1), -1, True)
        tb_graph.new_input(v, (-1, -1, -1), -1, True)
        tb_graph.new_input(partial, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_nope, q_pe, k_nope, k_rope, v, partial], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "mla_prefill_tp8_chunked_splitk_sm100", params
        )
        return None


# ---------------------------------------------------------------------------
# TP=8 chunked reduce
# ---------------------------------------------------------------------------
class MLAPrefillTP8ChunkedReduce(_MLAPrefillBase):
    """Reduce phase for TP=8 chunked split-K (``mla_prefill_tp8_chunked_reduce_sm100``).

    Merges ``num_splits`` partials into the final ``[B, q_len, H, 128]``
    output.
    """

    def __init__(
        self,
        num_heads: int,
        q_len: int,
        num_splits: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(num_heads, prefix=prefix)
        self.q_len = q_len
        self.num_splits = num_splits

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(H, nqb, max_num_batched_requests)`` with nqb=ceil(q_len/64)."""
        pk = current_pk()
        nqb = (self.q_len + 63) // 64
        return (self.num_heads, nqb, pk.max_num_batched_requests)

    def compile(
        self,
        partial: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_prefill_tp8_chunked_reduce_sm100`` — merges ``num_splits`` partials into final O.

        Tensor contract (BM=64, D_V=128, H = H_rank):
          partial: (num_splits, B, nqb, H, BM=64, D_V+4=132) fp32, partials from MLAPrefillTP8ChunkedSplitK (input_ptrs[0]).
          output:  (B, q_len, H, 128) bf16, final attn output (output_ptrs[0]).

        Notes: grid ``(H, nqb=ceil(q_len/64), max_num_batched_requests)``; request_id/kv_idx/merge_task_offset
        = head/q_block/batch. Meta deps inherited from the paired split-K decode.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        nqb = (self.q_len + 63) // 64
        params = [self.num_heads, self.q_len, self.num_splits, nqb]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([partial, output], tb_graph)
        pk.kn_graph.register_task(
            tb_graph, "mla_prefill_tp8_chunked_reduce_sm100", params
        )
        return output
