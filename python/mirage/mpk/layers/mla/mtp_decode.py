"""MLA-MTP decode + reduce TP variants.

This is the catalog counterpart to the MLA-MTP decode and reduce pk
methods on :class:`PersistentKernel`. MTP ("multi-token prediction")
is the DeepSeek-V3 spec-decode-style mode that runs attention on
multiple new-Q tokens per request in a single step. The TP variants
shard the 128 Q heads across the world: ``num_heads`` per rank =
128 / TP.

Decode side:

==================== =================================================== ====================
``tp_size``          pk method                                           task name
==================== =================================================== ====================
``1``                ``mla_mtp_decode_layer``                            ``mla_mtp_decode_sm100``
``2``                ``mla_mtp_decode_tp2_layer``                        ``mla_mtp_decode_tp2_sm100``
``4``                ``mla_mtp_decode_tp4_layer``                        ``mla_mtp_decode_tp4_sm100``
``8``                ``mla_mtp_decode_tp8_layer``                        ``mla_mtp_decode_tp8_sm100``
==================== =================================================== ====================

Reduce side (paired 1:1 with the decode variant):

==================== =================================================== ====================
``tp_size``          pk method                                           task name
==================== =================================================== ====================
``1``                ``mla_mtp_reduce_layer``                            ``mla_mtp_reduce_sm100``
``2``                ``mla_mtp_decode_tp2_reduce_layer``                 ``mla_mtp_decode_tp2_reduce_sm100``
``4``                ``mla_mtp_decode_tp4_reduce_layer``                 ``mla_mtp_decode_tp4_reduce_sm100``
``8``                ``mla_mtp_decode_tp8_reduce_layer``                 ``mla_mtp_decode_tp8_reduce_sm100``
==================== =================================================== ====================

Key tensor-layout quirks
------------------------

* ``q_input`` is TMA-desc attached on Hopper/Blackwell. The pk layer
  presents it with input_map ``(-1,-1,-1)`` because the kernel does
  its own (batch, head-group, V-split) addressing — it never reads
  via block_idx and the MPK runtime must NOT auto-partition.
* ``kv_input`` is the MLA contiguous-KV slab from the gather; shape
  ``[B * max_seq_len_pad, D_K]`` (``D_K = 576`` for DeepSeek V3).
* ``output_partial`` / ``output_lse`` carry the split-K partial
  results for the reduce kernel — same as the plain :class:`MLADecode`.
* For ``tp_size=8`` the kernel pads Q_LEN up to the next even
  number internally (because TP=8 packs two Q tokens per group).
  Callers pass ``q_len_real`` (the unpadded count); the layer wraps
  it.
* For ``tp_size=4`` ``has_v_split=True`` packs the V-split id into
  ``block_x`` — there are ``V_SPLITS`` (configurable via
  ``MIRAGE_MLA_TP4_V_SPLITS``) tasks per (Q-group, KV-split) tuple.
  The reduce kernel for TP=4 uses ``RD_DV`` from the build flag
  ``MIRAGE_MLA_TP4_RD_DV``.
* ``num_splits_override`` lets the caller force a non-default KV
  split count (default is ``ceil(kv_len / 128)`` with TILE_S=128).
  Used when the kernel was compiled with a coarser split layout
  than the runtime kv_len would suggest.

Grid / block dim
----------------

Both pk methods compute their own ``grid_dim`` and ``block_dim``
internally (the MTP-TP pk layers do NOT take grid_dim/block_dim
arguments). The catalog modules therefore raise from
``auto_grid_dim()`` and ignore grid_dim/block_dim kwargs passed to
``compile()`` — the pk layer is the source of truth.

Forward (PyTorch reference)
---------------------------

``forward()`` is intentionally NOT IMPLEMENTED. The decode + reduce
pair depend on MPK runtime meta-tensors and split-K partitioning
that have no clean eager-PyTorch counterpart. Validate via the
test-mode PK driver (see ``tests/runtime_python/blackwell/sm100_mla/``).
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


# Mirror the module-level helpers in persistent_kernel.py so the inlined task
# registrations don't depend on private pk helpers. The TP=4 variants read
# these at task-registration time to match the build-flag-set kernel shape.
def _mla_tp4_v_splits(max_seq_length=None):
    env_value = os.environ.get("MPK_MLA_TP4_V_SPLITS")
    if env_value is not None:
        value = int(env_value)
    else:
        # Standalone ablation on B200:
        #   KV <= 2048: 8 V splits is fastest for the current MPK single-split
        #   write-final path.
        #   KV >= 3072: 2 V splits matches the original TP4 MLA PR and avoids
        #   redoing the QK/softmax work eight times.
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
    """Inlined body of PersistentKernel._mla_mtp_decode_tp_layer."""
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
    # TP=4 packs the V split id into block_x → multiple tasks per split.
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
    """Inlined body of PersistentKernel._mla_mtp_reduce_tp_layer."""
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
    # TP4 can be compiled with a different RD_DV for ablation. Keep the
    # grid matched to the compiled coverage; standalone tests currently
    # show RD_DV=2 is fastest for KV=4096.
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

    Wraps :meth:`PersistentKernel.mla_mtp_decode_layer` (``tp_size=1``)
    or :meth:`PersistentKernel.mla_mtp_decode_tp{2,4,8}_layer` (sharded)
    1:1. For TP > 1 the per-rank head count is ``128 / tp_size``.

    Args:
        tp_size:  1, 2, 4, or 8.
        prefix:   HF state_dict key prefix (this module owns no params).

    Forward
    -------
    Not implemented — see module docstring.
    """

    def __init__(self, tp_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        _validate_tp_size(tp_size)
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"MLAMtpDecodeTP(tp_size={self.tp_size}).forward() is not "
            "implemented; depends on MPK runtime meta-tensors and "
            "split-K partitioning. Use the test-mode PK driver."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """grid_dim is computed inside the pk method; never call this."""
        raise RuntimeError(
            "MLAMtpDecodeTP computes grid_dim inside the pk method "
            "(mla_mtp_decode_layer / mla_mtp_decode_tp{2,4,8}_layer "
            "don't accept a grid_dim argument). Do not call "
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
        """Register the chosen MTP decode task on the current PK.

        Args:
            q_input:        ``(B*Q_LEN*H, D_K)`` bf16, TMA-desc.
            kv_input:       ``(B*max_seq_len_pad, D_K)`` bf16, TMA-desc.
            output_partial: split-K partial-O buffer.
            output_lse:     split-K partial-LSE buffer.
            q_len:          For ``tp_size=1`` this is Q_LEN. For
                            ``tp_size=8`` this is ``q_len_real`` —
                            the kernel internally pads it to the next
                            even number.
            kv_len:         KV sequence length the kernel is compiled
                            for.
            num_splits_override: Optional override for the KV split
                            count. ``None`` -> ``ceil(kv_len / 128)``.
            grid_dim / block_dim: ignored; raises if passed (the pk
                            method computes these internally).

        Returns:
            ``(output_partial, output_lse)`` for chaining.
        """
        pk = current_pk()
        if grid_dim is not None or block_dim is not None:
            raise ValueError(
                "MLAMtpDecodeTP.compile() does not accept grid_dim / "
                "block_dim — the pk method computes them internally."
            )

        # Inlined task registration (the bodies that used to live on
        # PersistentKernel.mla_mtp_decode_layer / _mla_mtp_decode_tp_layer
        # and its thin TP2/TP4/TP8 wrappers).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.tp_size == 1:
            # mla_mtp_decode_layer doesn't take num_splits_override.
            if num_splits_override is not None:
                raise ValueError(
                    "MLAMtpDecodeTP(tp_size=1) does not support "
                    "num_splits_override (the base pk method computes "
                    "num_splits from kv_len)."
                )
            # Derive internal params (DeepSeek V3: 128 heads, TILE_S=128).
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
            # mla_mtp_decode_tp2_layer wraps _mla_mtp_decode_tp_layer with
            # num_heads=64, head_groups=2.
            _mla_mtp_decode_tp_register(
                pk,
                q_input, kv_input, output_partial, output_lse,
                q_len, kv_len, num_heads=64,
                task_name="mla_mtp_decode_tp2_sm100",
                head_groups=2,
                num_splits_override=num_splits_override,
            )
        elif self.tp_size == 4:
            # TP=4 V-split: each split writes a disjoint D_V chunk. Keep
            # this configurable for ablation because each split repeats
            # QK/softmax.
            _mla_mtp_decode_tp_register(
                pk,
                q_input, kv_input, output_partial, output_lse,
                q_len, kv_len, num_heads=32,
                task_name="mla_mtp_decode_tp4_sm100", has_v_split=True,
                head_groups=_mla_tp4_head_groups(),
                v_splits=_mla_tp4_v_splits(pk.max_seq_length),
                num_splits_override=num_splits_override,
            )
        else:  # tp_size == 8
            # TP=8 pads Q_LEN to even.
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

    Wraps :meth:`PersistentKernel.mla_mtp_reduce_layer` (``tp_size=1``)
    or :meth:`PersistentKernel.mla_mtp_decode_tp{2,4,8}_reduce_layer`
    1:1. Paired downstream of :class:`MLAMtpDecodeTP` with the SAME
    ``tp_size``.

    Args:
        tp_size:  1, 2, 4, or 8 (must match the decode it's paired with).
        prefix:   HF state_dict key prefix.

    Forward
    -------
    Not implemented — see module docstring.
    """

    def __init__(self, tp_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        _validate_tp_size(tp_size)
        self.tp_size = tp_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"MLAMtpReduceTP(tp_size={self.tp_size}).forward() is not "
            "implemented; depends on the upstream decode's split-K "
            "partition. Use the test-mode PK driver."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        raise RuntimeError(
            "MLAMtpReduceTP computes grid_dim inside the pk method. "
            "Do not call auto_grid_dim() and do not pass grid_dim to "
            "compile()."
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
        """Register the chosen MTP reduce task on the current PK.

        Args:
            input_partial: Partial-O from the paired decode.
            input_lse:     Partial-LSE from the paired decode.
            output:        ``(B*Q_LEN, H, D_V)`` bf16 final output.
            q_len:         For ``tp_size=8`` pass the UNPADDED
                           ``q_len_real``; the layer pads to the next
                           even number internally. For other tp sizes
                           pass Q_LEN directly.
            kv_len:        KV sequence length (must match decode).
            grid_dim / block_dim: ignored; raises if passed.

        Returns:
            ``output`` for chaining.
        """
        pk = current_pk()
        if grid_dim is not None or block_dim is not None:
            raise ValueError(
                "MLAMtpReduceTP.compile() does not accept grid_dim / "
                "block_dim — the pk method computes them internally."
            )

        # Inlined task registration (the bodies that used to live on
        # PersistentKernel.mla_mtp_reduce_layer / _mla_mtp_reduce_tp_layer
        # and its thin TP2/TP4/TP8 wrappers).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.tp_size == 1:
            hpb = 128 // q_len
            while 128 % hpb != 0:
                hpb -= 1
            num_head_groups = 128 // hpb
            num_splits = (kv_len + 128 - 1) // 128
            d_v = 512
            # TODO: rd_dv=2 gives 256-1024 reduce blocks (many small tasks
            # in MPK). Consider rd_dv=4 with loop to halve block count, but
            # benchmarked slower. Revisit after MPK runtime refactor when
            # task dispatch overhead is known.
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
        else:  # tp_size == 8
            # TP=8 pads Q_LEN to even (matches decode).
            q_len_padded = (q_len + 1) & ~1
            _mla_mtp_reduce_tp_register(
                pk,
                input_partial, input_lse, output, q_len_padded, kv_len,
                num_heads=16,
                task_name="mla_mtp_decode_tp8_reduce_sm100",
            )
        return output
