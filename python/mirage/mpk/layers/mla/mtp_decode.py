"""MLA-MTP decode + reduce TP variants.

Wraps the MLA-MTP decode/reduce SM100 tasks. Kernels live under
``include/mirage/persistent_kernel/tasks/blackwell/`` —
``mla_mtp_decode_sm100.cuh``, ``mla_mtp_decode_tp{2,4,8}_sm100.cuh``
and the matching reduce kernels. ``tp_size`` selects the per-rank
sharding of the 128 Q heads (per-rank H = 128 / tp_size).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _validate_tp_size(tp_size: int) -> None:
    if tp_size not in (1, 2, 4, 8):
        raise ValueError(
            f"MLAMtp*TP tp_size must be 1, 2, 4, or 8; got {tp_size}"
        )


# TP=4 build-flag knobs (mirror persistent_kernel.py helpers so this
# module is self-contained).
def _mla_tp4_v_splits(max_seq_length=None):
    env_value = os.environ.get("MPK_MLA_TP4_V_SPLITS")
    if env_value is not None:
        value = int(env_value)
    else:
        value = 2 if max_seq_length is not None and max_seq_length >= 3072 else 8
    if value not in (1, 2, 4, 8):
        raise ValueError("MPK_MLA_TP4_V_SPLITS must be one of 1, 2, 4, 8")
    return value


def _mla_tp4_head_groups():
    value = int(os.environ.get("MPK_MLA_TP4_HEAD_GROUPS", "1"))
    if value not in (1, 2, 4, 8):
        raise ValueError("MPK_MLA_TP4_HEAD_GROUPS must be one of 1, 2, 4, 8")
    return value


def _mla_tp4_rd_dv():
    value = int(os.environ.get("MPK_MLA_TP4_RD_DV", "2"))
    if value not in (2, 4, 8):
        raise ValueError("MPK_MLA_TP4_RD_DV must be one of 2, 4, 8")
    return value


def _mla_mtp_decode_tp_register(
    pk,
    q_input, kv_input, output_partial, output_lse,
    q_len, kv_len, num_heads,
    task_name, has_v_split=False, q_len_real=None, head_groups=1,
    v_splits=2, num_splits_override=None,
):
    """Inlined task registration for TP=2/4/8 MTP decode."""
    from ....core import CyTBGraph
    from ....kernel import TBGraph

    if num_heads == 64:
        qpg = min(2, q_len)
    elif num_heads == 32:
        qpg = min(4, q_len)
    else:  # TP=8
        qpg = 2
    num_groups = (q_len + qpg - 1) // qpg
    num_splits = (
        num_splits_override
        if num_splits_override is not None
        else (kv_len + 128 - 1) // 128
    )  # TILE_S=128
    # TP=4 packs the V split id into block_x -> multiple tasks per split.
    x_mul = v_splits if has_v_split else 1
    grid_dim = (
        num_groups * num_splits * x_mul * head_groups,
        pk.max_num_batched_requests,
        1,
    )
    block_dim = (128, 1, 1)

    if num_heads == 16:  # TP=8
        params = [
            num_groups, q_len, kv_len, num_splits,
            q_len_real if q_len_real is not None else q_len,
        ]
    else:  # TP=2 and TP=4
        params = [num_groups, q_len, kv_len, num_splits]

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(q_input, (-1, -1, -1), -1, True)
    tb_graph.new_input(kv_input, (-1, -1, -1), -1, True)
    tb_graph.new_input(output_partial, (-1, -1, -1), -1, True)
    tb_graph.new_input(output_lse, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [q_input, kv_input, output_partial, output_lse], tb_graph
    )
    pk.kn_graph.register_task(tb_graph, task_name, params)


def _mla_mtp_reduce_tp_register(
    pk,
    input_partial, input_lse, output,
    q_len, kv_len, num_heads, task_name,
):
    """Inlined task registration for TP=2/4/8 MTP reduce."""
    from ....core import CyTBGraph
    from ....kernel import TBGraph

    if num_heads == 64:
        qpg = min(2, q_len)
    elif num_heads == 32:
        qpg = min(4, q_len)
    else:
        qpg = 2
    num_groups = (q_len + qpg - 1) // qpg
    num_splits = (kv_len + 128 - 1) // 128
    d_v = 512
    rd_dv = (
        _mla_tp4_rd_dv()
        if task_name == "mla_mtp_decode_tp4_reduce_sm100"
        else 2
    )

    params = [num_groups, q_len, num_splits, rd_dv]
    grid_dim = (
        (d_v + rd_dv - 1) // rd_dv,
        num_groups,
        pk.max_num_batched_requests,
    )
    block_dim = (256, 1, 1)

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(input_partial, (-1, -1, -1), -1, True)
    tb_graph.new_input(input_lse, (-1, -1, -1), -1, True)
    tb_graph.new_input(output, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input_partial, input_lse, output], tb_graph
    )
    pk.kn_graph.register_task(tb_graph, task_name, params)


class MLAMtpDecodeTP(MPKModule):
    """MLA-MTP decode dispatcher over TP world sizes 1/2/4/8.

    Task ``mla_mtp_decode_sm100`` (tp=1) or ``mla_mtp_decode_tp{2,4,8}_sm100``
    (sharded). Per-rank head count is ``128 / tp_size``. ``q_input``
    uses TMA-desc; ``kv_input`` is the contiguous MLA KV slab
    ``[B*max_seq_len_pad, D_K=576]``.
    """

    def __init__(self, tp_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        _validate_tp_size(tp_size)
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        """Not implemented: depends on MPK runtime meta-tensors and split-K."""
        raise NotImplementedError(
            f"MLAMtpDecodeTP(tp_size={self.tp_size}).forward(): use test-mode PK driver."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid computed inside compile(); calling this raises."""
        raise RuntimeError(
            "MLAMtpDecodeTP computes grid_dim internally — do not call "
            "auto_grid_dim() and do not pass grid_dim to compile()."
        )

    def compile(
        self,
        q_input: DTensor,
        kv_input: DTensor,
        output_partial: DTensor,
        output_lse: DTensor,
        *,
        q_len: int,
        kv_len: int,
        num_splits_override: Optional[int] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register ``mla_mtp_decode_sm100`` (tp=1) or ``mla_mtp_decode_tp{2,4,8}_sm100``.

        Tensor contract (HEADS_PER_RANK = 128 / tp_size; D_K=576, D_V=512):
          q_input:        (R*Q_LEN_PADDED*HEADS_PER_RANK, D_K=576) bf16, TMA-desc (input_tma_desc_ptrs[0][0]).
                          tp=8 pads Q_LEN to next even; tp=1 uses Q_LEN as-is.
          kv_input:       (R*MPK_MAX_SEQ_LENGTH_PAD, D_K=576) bf16, contiguous gathered paged-KV slab,
                          TMA-desc (input_tma_desc_ptrs[1][0]).
          output_partial: (R*num_groups*num_splits, HEADS_PER_RANK*D_V=HPR*512) bf16, partial-O ``Oa`` (output_ptrs[0]).
          output_lse:     (R*num_groups*num_splits, HEADS_PER_RANK*128) fp32, partial-LSE ``La`` (output_ptrs[1]).

        Notes: grid computed internally (``auto_grid_dim`` raises); callers MUST NOT pass grid_dim/block_dim.
        Meta deps: ``paged_kv_indptr_buffer``, ``paged_kv_indices_buffer`` (tp>=2), ``paged_kv_last_page_len_buffer``,
        ``qo_indptr_buffer``, ``request_ids`` (tp>=2 early-exits when ``Q_LEN>8``). num_splits_override only valid for tp>1.
        """
        pk = current_pk()
        if grid_dim is not None or block_dim is not None:
            raise ValueError(
                "MLAMtpDecodeTP.compile() does not accept grid_dim / "
                "block_dim — the task computes them internally."
            )

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.tp_size == 1:
            if num_splits_override is not None:
                raise ValueError(
                    "MLAMtpDecodeTP(tp_size=1) does not support "
                    "num_splits_override (num_splits = ceil(kv_len / 128))."
                )
            # DeepSeek V3 single-GPU: 128 heads, TILE_S=128.
            hpb = 128 // q_len
            while 128 % hpb != 0:
                hpb -= 1
            num_head_groups = 128 // hpb
            num_splits = (kv_len + 128 - 1) // 128

            params = [num_head_groups, q_len, kv_len, num_splits]
            grid_dim_d = (
                num_splits, num_head_groups, pk.max_num_batched_requests,
            )
            block_dim_d = (128, 1, 1)

            tb_graph = TBGraph(CyTBGraph(grid_dim_d, block_dim_d, 1, 64))
            tb_graph.new_input(q_input, (-1, -1, -1), -1, True)
            tb_graph.new_input(kv_input, (-1, -1, -1), -1, True)
            tb_graph.new_input(output_partial, (-1, -1, -1), -1, True)
            tb_graph.new_input(output_lse, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [q_input, kv_input, output_partial, output_lse], tb_graph
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_mtp_decode_sm100", params
            )
        elif self.tp_size == 2:
            _mla_mtp_decode_tp_register(
                pk,
                q_input, kv_input, output_partial, output_lse,
                q_len, kv_len, num_heads=64,
                task_name="mla_mtp_decode_tp2_sm100",
                head_groups=2,
                num_splits_override=num_splits_override,
            )
        elif self.tp_size == 4:
            _mla_mtp_decode_tp_register(
                pk,
                q_input, kv_input, output_partial, output_lse,
                q_len, kv_len, num_heads=32,
                task_name="mla_mtp_decode_tp4_sm100", has_v_split=True,
                head_groups=_mla_tp4_head_groups(),
                v_splits=_mla_tp4_v_splits(pk.max_seq_length),
                num_splits_override=num_splits_override,
            )
        else:  # tp_size == 8 — kernel pads Q_LEN to even internally.
            q_len_real = q_len
            q_len_padded = (q_len_real + 1) & ~1
            _mla_mtp_decode_tp_register(
                pk,
                q_input, kv_input, output_partial, output_lse,
                q_len_padded, kv_len, num_heads=16,
                task_name="mla_mtp_decode_tp8_sm100",
                q_len_real=q_len_real,
                num_splits_override=num_splits_override,
            )
        return output_partial, output_lse


class MLAMtpReduceTP(MPKModule):
    """MLA-MTP reduce dispatcher over TP world sizes 1/2/4/8.

    Task ``mla_mtp_reduce_sm100`` (tp=1) or
    ``mla_mtp_decode_tp{2,4,8}_reduce_sm100``. Paired downstream of
    :class:`MLAMtpDecodeTP` with the SAME ``tp_size``; merges the
    split-K partials into ``(B, NUM_HEADS_PER_RANK, D_V=512)``.
    """

    def __init__(self, tp_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        _validate_tp_size(tp_size)
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        """Not implemented: tied to upstream decode's split-K partition."""
        raise NotImplementedError(
            f"MLAMtpReduceTP(tp_size={self.tp_size}).forward(): use test-mode PK driver."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid computed inside compile(); calling this raises."""
        raise RuntimeError(
            "MLAMtpReduceTP computes grid_dim internally — do not call "
            "auto_grid_dim() and do not pass grid_dim to compile()."
        )

    def compile(
        self,
        input_partial: DTensor,
        input_lse: DTensor,
        output: DTensor,
        *,
        q_len: int,
        kv_len: int,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_mtp_reduce_sm100`` (tp=1) or ``mla_mtp_decode_tp{2,4,8}_reduce_sm100``.

        Tensor contract (HEADS_PER_RANK = 128 / tp_size; D_V=512):
          input_partial: (R*num_groups*num_splits, HEADS_PER_RANK*D_V=HPR*512) bf16, partial-O from MLAMtpDecodeTP (input_ptrs[0]).
          input_lse:     (R*num_groups*num_splits, HEADS_PER_RANK*128) fp32, partial-LSE from MLAMtpDecodeTP (input_ptrs[1]).
          output:        (B, HEADS_PER_RANK, D_V=512) bf16, final attn output ready for ``o_proj`` (output_ptrs[0]).

        Notes: grid ``(ceil(D_V/rd_dv), num_head_groups, max_num_batched_requests)``; rd_dv=2 except TP=4 reads
        ``MPK_MLA_TP4_RD_DV``. Pass UNPADDED ``q_len`` for tp=8 (reduce pads internally to match decode);
        ``kv_len`` must match paired decode. Early-exits when runtime ``Q_LEN>8``. Meta deps mirror the paired decode.
        """
        pk = current_pk()
        if grid_dim is not None or block_dim is not None:
            raise ValueError(
                "MLAMtpReduceTP.compile() does not accept grid_dim / "
                "block_dim — the task computes them internally."
            )

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.tp_size == 1:
            hpb = 128 // q_len
            while 128 % hpb != 0:
                hpb -= 1
            num_head_groups = 128 // hpb
            num_splits = (kv_len + 128 - 1) // 128
            d_v = 512
            rd_dv = 2

            params = [num_head_groups, q_len, num_splits, rd_dv]
            grid_dim_r = (
                (d_v + rd_dv - 1) // rd_dv,
                num_head_groups,
                pk.max_num_batched_requests,
            )
            block_dim_r = (256, 1, 1)

            tb_graph = TBGraph(CyTBGraph(grid_dim_r, block_dim_r, 1, 64))
            tb_graph.new_input(input_partial, (-1, -1, -1), -1, True)
            tb_graph.new_input(input_lse, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [input_partial, input_lse, output], tb_graph
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_mtp_reduce_sm100", params
            )
        elif self.tp_size == 2:
            _mla_mtp_reduce_tp_register(
                pk,
                input_partial, input_lse, output, q_len, kv_len,
                num_heads=64,
                task_name="mla_mtp_decode_tp2_reduce_sm100",
            )
        elif self.tp_size == 4:
            _mla_mtp_reduce_tp_register(
                pk,
                input_partial, input_lse, output, q_len, kv_len,
                num_heads=32,
                task_name="mla_mtp_decode_tp4_reduce_sm100",
            )
        else:  # tp_size == 8 — pad Q_LEN to even (matches decode).
            q_len_padded = (q_len + 1) & ~1
            _mla_mtp_reduce_tp_register(
                pk,
                input_partial, input_lse, output, q_len_padded, kv_len,
                num_heads=16,
                task_name="mla_mtp_decode_tp8_reduce_sm100",
            )
        return output
