"""MLA prefill catalog module (7 variants).

This is the catalog counterpart to seven pk methods on
:class:`PersistentKernel` covering the full set of MLA prefill
strategies in the codebase. Pick a ``variant``:

==================== ============================================== ====================
``variant``          pk method                                       task name
==================== ============================================== ====================
``"absorbed"``       ``mla_prefill_absorbed_layer``                  ``mla_prefill_absorbed_sm100``
``"plain"``          ``mla_prefill_layer``                           ``mla_prefill_sm100``
``"unified"``        ``mla_unified_layer``                           ``mla_unified_sm100``
``"tp8"``            ``mla_prefill_tp8_layer``                       ``mla_prefill_tp8_sm100``
``"tp8_chunked"``    ``mla_prefill_tp8_chunked_layer``               ``mla_prefill_tp8_chunked_sm100``
``"tp8_chunked_splitk"`` ``mla_prefill_tp8_chunked_splitk_layer``    ``mla_prefill_tp8_chunked_splitk_sm100``
``"tp8_chunked_reduce"`` ``mla_prefill_tp8_chunked_reduce_layer``    ``mla_prefill_tp8_chunked_reduce_sm100``
==================== ============================================== ====================

What each variant does
----------------------

* ``"absorbed"`` — DeepSeek V3 absorbed-attention prefill. Q is the
  fused ``[S, H, D_CKV + D_KPE]`` per-head latent Q; KV is the
  contiguous ``[B*max_seq_len, D_CKV + D_KPE]`` slab produced by the
  KVGather (``standard``/``unified`` variants). Output width is
  ``D_V`` (= ``D_CKV`` = ``kv_lora_rank``, since o_proj absorbs V).
* ``"plain"`` — Non-absorbed prefill. Q is split into NoPE and PE
  tensors; KV is the **split** layout from :class:`MLAKVGather`
  ``variant='split'`` — two separate ``ckv`` and ``kpe`` tensors. The
  kernel implements its own (S, H, D) addressing; the layer presents
  inputs with all-``(-1,-1,-1)`` input_maps to avoid spurious
  auto-partitioning.
* ``"unified"`` — Runtime prefill-vs-decode dispatch in a single
  fused kernel. Registers one task that reads BOTH the prefill
  inputs (NoPE, PE, ckv, kpe) AND the decode inputs (fused Q with
  TMA, contiguous KV with TMA, partial-O, partial-LSE) and chooses
  the branch based on runtime ``Q_LEN``. Grid is computed by the pk
  layer itself (callers MUST NOT pass ``grid_dim``).
* ``"tp8"`` — TP=8 unabsorbed prefill (16 heads per rank). Reads NoPE
  ``[B,S,H,128]``, PE ``[B,S,H,64]``, fused K ``[B,S,192]``, V
  ``[B,S,128]`` (full GPU); writes O ``[B,S,H,128]``. Used by the
  TP=8 sharded DeepSeek V3 inference path.
* ``"tp8_chunked"`` — TP=8 unabsorbed prefill, chunked along Q. Adds a
  ``q_start`` param so the kernel iterates over only ``[q_start,
  q_start + q_len)`` of the Q axis. Supports a fused-Q layout
  (``qfused_mode=1``) where ``q_nope`` and ``q_pe`` are the SAME
  DTensor of width ``H*192`` (row-swap addressing).
* ``"tp8_chunked_splitk"`` — Split-K variant of ``tp8_chunked``.
  Computes per-(KV split, Q tile) partial O+LSE into a
  ``[num_splits, B, nqb, H, BM, D_V+4]`` float32 buffer; pair with
  ``"tp8_chunked_reduce"`` to merge.
* ``"tp8_chunked_reduce"`` — Reduce phase for ``tp8_chunked_splitk``.
  Merges the ``num_splits`` partials into the final
  ``[B, q_len, H, 128]`` output.

Tensor contract (variant-specific)
----------------------------------

Common dims: ``H`` = per-rank num query heads, ``S`` = seq len,
``D_CKV`` = ``kv_lora_rank`` (512), ``D_KPE`` = ``qk_rope_head_dim``
(64), ``D_K`` = ``D_CKV + D_KPE`` (576), ``D_V`` = output width
(= ``D_CKV`` for absorbed; = 128 for unabsorbed-TP8).

absorbed: q_nope_pe ``[S, H, D_CKV+D_KPE]`` flattened, kv
``[B*max_seq_len, D_CKV+D_KPE]``, output ``[S, H, D_V]``.

plain: q_nope ``[S, H, D_CKV]``, q_pe ``[S, H, D_KPE]``, ckv
``[S, D_CKV]``, kpe ``[S, D_KPE]``, output ``[S, H, D_V]``.

unified: same as plain inputs PLUS decode-side TMA-attached q_input
``[S, H*D_K]``, kv_input ``[B*S, D_K]``, output_partial / output_lse.

tp8: q_nope ``[B, S, H, 128]``, q_pe ``[B, S, H, 64]``, k
``[B, S, 192]``, v ``[B, S, 128]``, output ``[B, S, H, 128]``.

tp8_chunked: q_nope ``[B, q_len, H, 128]`` (or ``[B, q_len, H, 192]``
for fused), q_pe same / aliased, k_nope ``[B, kv_len, H, 128]``,
k_rope ``[B, kv_len, 1, 64]``, v ``[B, kv_len, H, 128]``, output
``[B, q_len, H, 128]``. Plus a ``q_start`` int param for chunk offset.

tp8_chunked_splitk: same inputs as tp8_chunked PLUS a ``partial``
float buffer of size ``num_splits*B*nqb*H*BM*(D_V+4)``; no ``output``.

tp8_chunked_reduce: ``partial`` (from splitk) -> ``output`` of shape
``[B, q_len, H, 128]``.

Parallelism axis
----------------

* absorbed / plain: ``(H, ceil(seq_len/BM), B)`` with ``BM=64``.
* unified: computed by the pk layer; callers must NOT pass grid_dim.
* tp8: ``(H, ceil(S/BM), B)``, ``BM=64``.
* tp8_chunked: ``(H, ceil(q_len/64), B)``.
* tp8_chunked_splitk: ``(H, nqb * num_splits, B)`` with
  ``nqb = ceil(q_len / 64)``.
* tp8_chunked_reduce: ``(H, nqb, B)``.

Forward (PyTorch reference)
---------------------------

All prefill variants are flagged as ``NotImplementedError`` —
they depend on MPK runtime meta-tensors (paged KV indices,
qo_indptr_buffer for per-request ranges) and on the absorbed-vs-plain
distinction in how V is recovered from c_latent (controlled at code
generation, not at the Python tensor level). Use the test-mode PK
driver for end-to-end validation (see
``tests/runtime_python/blackwell/sm100_mla/`` for examples).
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

PrefillVariant = Literal[
    "absorbed",
    "plain",
    "unified",
    "tp8",
    "tp8_chunked",
    "tp8_chunked_splitk",
    "tp8_chunked_reduce",
]


class MLAPrefill(MPKModule):
    """MLA prefill (7 strategy variants).

    Wraps the corresponding pk method 1:1; the variant flag picks the
    task name registered.

    Args:
        num_heads:     Per-rank query head count.
        seq_len:       For non-chunked variants: the max seq_len the
                       kernel is compiled for.
        d_ckv:         ``kv_lora_rank`` (512 for DeepSeek V3). Required
                       for ``absorbed`` / ``plain`` / ``unified``.
        d_kpe:         ``qk_rope_head_dim`` (64 for DeepSeek V3).
                       Required for ``absorbed`` / ``plain`` /
                       ``unified``.
        d_v:           Output width (``D_V``). For absorbed this is
                       ``d_ckv``; for TP8 unabsorbed it's 128.
                       Required for ``absorbed`` / ``plain`` /
                       ``unified``.
        variant:       Which pk method to dispatch to.
        q_len:         For chunked TP=8 variants — Q-axis chunk
                       length (also Q axis for plain TP=8).
        kv_len:        For chunked TP=8 variants — KV-axis length.
        q_start:       For chunked TP=8 variants — Q-axis offset.
        num_splits:    For ``tp8_chunked_splitk`` /
                       ``tp8_chunked_reduce`` — number of KV splits.
        qfused_mode:   For ``tp8_chunked`` — 0 = separate q_nope/q_pe,
                       1 = fused single Q DTensor (row-swap).
        tp_size:       For ``unified`` — tensor-parallel world size
                       (1, 2, 4, or 8) used to pick the kernel's
                       per-rank head-group factoring.
        decode_q_len:  For ``unified`` — Q tokens per request in the
                       decode branch (1 normally; >1 for MTP/spec).
                       Clamped to <= 8 internally.
        prefix:        HF state_dict key prefix.

    Forward
    -------
    ``forward()`` is not implemented; see module docstring.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        variant: PrefillVariant = "absorbed",
        seq_len: Optional[int] = None,
        d_ckv: Optional[int] = None,
        d_kpe: Optional[int] = None,
        d_v: Optional[int] = None,
        q_len: Optional[int] = None,
        kv_len: Optional[int] = None,
        q_start: int = 0,
        num_splits: Optional[int] = None,
        qfused_mode: int = 0,
        tp_size: int = 1,
        decode_q_len: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if variant not in (
            "absorbed", "plain", "unified", "tp8",
            "tp8_chunked", "tp8_chunked_splitk", "tp8_chunked_reduce",
        ):
            raise ValueError(f"MLAPrefill unknown variant: {variant!r}")
        self.num_heads = num_heads
        self.variant = variant
        self.seq_len = seq_len
        self.d_ckv = d_ckv
        self.d_kpe = d_kpe
        self.d_v = d_v
        self.q_len = q_len
        self.kv_len = kv_len
        self.q_start = q_start
        self.num_splits = num_splits
        self.qfused_mode = qfused_mode
        self.tp_size = tp_size
        self.decode_q_len = decode_q_len

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MLAPrefill.forward() is not implemented as a plain "
            "PyTorch reference: prefill depends on MPK runtime "
            "meta-tensors and on the absorbed-vs-plain V-recovery "
            "scheme controlled at code generation. Use the test-mode "
            "PK driver (see tests/runtime_python/blackwell/sm100_mla/)."
        )

    # ------------------------------------------------------------------
    # Grid heuristics
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Variant-specific defaults.

        See module docstring for the per-variant grid layout. For
        ``unified`` the grid is computed inside the pk method, so this
        method raises (callers must NOT supply grid_dim for unified).
        """
        pk = current_pk()
        if self.variant in ("absorbed", "plain"):
            if self.seq_len is None:
                raise ValueError(
                    f"MLAPrefill(variant='{self.variant}').auto_grid_dim "
                    "needs seq_len at construction time."
                )
            return (
                self.num_heads,
                (self.seq_len + 63) // 64,
                pk.max_num_batched_requests,
            )
        if self.variant == "tp8":
            if self.seq_len is None:
                raise ValueError(
                    "MLAPrefill(variant='tp8').auto_grid_dim needs "
                    "seq_len at construction time."
                )
            return (
                self.num_heads,
                (self.seq_len + 63) // 64,
                pk.max_num_batched_requests,
            )
        if self.variant == "tp8_chunked":
            if self.q_len is None:
                raise ValueError(
                    "MLAPrefill(variant='tp8_chunked').auto_grid_dim "
                    "needs q_len at construction time."
                )
            return (
                self.num_heads,
                (self.q_len + 63) // 64,
                pk.max_num_batched_requests,
            )
        if self.variant == "tp8_chunked_splitk":
            if self.q_len is None or self.num_splits is None:
                raise ValueError(
                    "MLAPrefill(variant='tp8_chunked_splitk').auto_grid_dim "
                    "needs q_len and num_splits."
                )
            nqb = (self.q_len + 63) // 64
            return (
                self.num_heads,
                nqb * self.num_splits,
                pk.max_num_batched_requests,
            )
        if self.variant == "tp8_chunked_reduce":
            if self.q_len is None:
                raise ValueError(
                    "MLAPrefill(variant='tp8_chunked_reduce').auto_grid_dim "
                    "needs q_len."
                )
            nqb = (self.q_len + 63) // 64
            return (self.num_heads, nqb, pk.max_num_batched_requests)
        # unified
        raise ValueError(
            "MLAPrefill(variant='unified') computes its grid inside "
            "the pk method; do not request auto_grid_dim and do not "
            "pass grid_dim to compile()."
        )

    def default_block_dim(self) -> BlockDim:
        """Per-variant block_dim conventions.

        Most prefill kernels use 256 threads/block. The chunked TP=8
        kernels use 128 threads/block (per pk layer defaults).
        ``tp8_chunked_reduce`` returns to 256.
        """
        if self.variant in ("tp8", "tp8_chunked", "tp8_chunked_splitk"):
            return (128, 1, 1)
        return (256, 1, 1)

    # ------------------------------------------------------------------
    # Compile dispatcher
    # ------------------------------------------------------------------
    def compile(
        self,
        *,
        q_nope: Optional[DTensor] = None,
        q_pe: Optional[DTensor] = None,
        q_nope_pe: Optional[DTensor] = None,
        ckv: Optional[DTensor] = None,
        kpe: Optional[DTensor] = None,
        kv: Optional[DTensor] = None,
        k: Optional[DTensor] = None,
        v: Optional[DTensor] = None,
        k_nope: Optional[DTensor] = None,
        k_rope: Optional[DTensor] = None,
        output: Optional[DTensor] = None,
        partial: Optional[DTensor] = None,
        q_input: Optional[DTensor] = None,
        kv_input: Optional[DTensor] = None,
        output_partial: Optional[DTensor] = None,
        output_lse: Optional[DTensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Optional[DTensor]:
        """Register the chosen prefill task on the current PK.

        The required kwargs vary by ``variant`` — see module docstring
        and per-variant validation below. Returns the output DTensor
        when one exists (``None`` for ``tp8_chunked_splitk`` which
        produces only ``partial``).
        """
        pk = current_pk()
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the bodies that used to live on
        # PersistentKernel.mla_prefill_*_layer / mla_unified_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.variant == "absorbed":
            _require(q_nope_pe, "q_nope_pe", self.variant)
            _require(kv, "kv", self.variant)
            _require(output, "output", self.variant)
            for n, v_ in (("d_ckv", self.d_ckv), ("d_kpe", self.d_kpe),
                          ("d_v", self.d_v), ("seq_len", self.seq_len)):
                if v_ is None:
                    raise ValueError(
                        f"MLAPrefill(variant='absorbed') requires {n} "
                        "at construction time."
                    )
            if grid_dim is None:
                grid_dim = self.auto_grid_dim()
            params = [
                self.num_heads,
                self.seq_len,
                self.d_ckv,
                self.d_kpe,
                self.d_v,
            ]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(kv, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, -1, -1), -1, True)
            pk.kn_graph.customized([q_nope_pe, kv, output], tb_graph)
            pk.kn_graph.register_task(
                tb_graph, "mla_prefill_absorbed_sm100", params
            )
            return output

        if self.variant == "plain":
            _require(q_nope, "q_nope", self.variant)
            _require(q_pe, "q_pe", self.variant)
            _require(ckv, "ckv", self.variant)
            _require(kpe, "kpe", self.variant)
            _require(output, "output", self.variant)
            for n, v_ in (("d_ckv", self.d_ckv), ("d_kpe", self.d_kpe),
                          ("d_v", self.d_v), ("seq_len", self.seq_len)):
                if v_ is None:
                    raise ValueError(
                        f"MLAPrefill(variant='plain') requires {n}."
                    )
            if grid_dim is None:
                grid_dim = self.auto_grid_dim()
            params = [
                self.num_heads,
                self.seq_len,
                self.d_ckv,
                self.d_kpe,
                self.d_v,
            ]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            # Kernel reads based on task_metadata.{request_id=head,
            # kv_idx=q_block} and computes its own (S, H, D) offsets, so MPK
            # must NOT try to auto-partition dim 0 by grid.x (grid.x is H,
            # not S). Use -1 on all dims → full barrier event semantics.
            tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(ckv, (-1, -1, -1), -1, True)
            tb_graph.new_input(kpe, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [q_nope, q_pe, ckv, kpe, output], tb_graph
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_prefill_sm100", params
            )
            return output

        if self.variant == "unified":
            for n, v_ in (("q_nope", q_nope), ("q_pe", q_pe),
                          ("ckv", ckv), ("kpe", kpe),
                          ("output", output), ("q_input", q_input),
                          ("kv_input", kv_input),
                          ("output_partial", output_partial),
                          ("output_lse", output_lse)):
                if v_ is None:
                    raise ValueError(
                        f"MLAPrefill(variant='unified') requires {n}."
                    )
            for n, v_ in (("q_len", self.q_len), ("kv_len", self.kv_len)):
                if v_ is None:
                    raise ValueError(
                        f"MLAPrefill(variant='unified') requires {n} "
                        "at construction time."
                    )
            if grid_dim is not None:
                raise ValueError(
                    "MLAPrefill(variant='unified') computes its grid "
                    "internally — do not pass grid_dim."
                )

            # Mirror the mla_unified_layer grid/param derivation exactly.
            num_heads = self.num_heads
            q_len = self.q_len
            kv_len = self.kv_len
            tp_size = self.tp_size
            d_ckv = self.d_ckv if self.d_ckv is not None else 512
            d_kpe = self.d_kpe if self.d_kpe is not None else 64
            d_v = self.d_v if self.d_v is not None else 512

            num_splits = (kv_len + 128 - 1) // 128
            # q_len is the prompt-prefill chunk budget. Decode width is
            # controlled by generation semantics (one token, or MTP verify
            # width), not MBT.
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
                raise ValueError(
                    f"Unsupported MLA unified tp_size={tp_size}"
                )

            num_q_blocks = (q_len + 64 - 1) // 64
            decode_blocks_x = num_groups * num_splits * x_mul
            grid_dim_u = (
                max(num_heads, decode_blocks_x),
                max(num_q_blocks, pk.max_num_batched_requests),
                pk.max_num_batched_requests,
            )
            block_dim_u = (256, 1, 1)
            params = [
                num_heads,
                decode_q_len,
                kv_len,
                num_splits,
                tp_size,
                d_ckv,
                d_kpe,
                d_v,
            ]

            tb_graph = TBGraph(CyTBGraph(grid_dim_u, block_dim_u, 1, 64))
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
                [
                    q_nope,
                    q_pe,
                    ckv,
                    kpe,
                    output,
                    q_input,
                    kv_input,
                    output_partial,
                    output_lse,
                ],
                tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_unified_sm100", params
            )
            return output

        if self.variant == "tp8":
            _require(q_nope, "q_nope", self.variant)
            _require(q_pe, "q_pe", self.variant)
            _require(k, "k", self.variant)
            _require(v, "v", self.variant)
            _require(output, "output", self.variant)
            if self.seq_len is None:
                raise ValueError(
                    "MLAPrefill(variant='tp8') requires seq_len."
                )
            if grid_dim is None:
                grid_dim = self.auto_grid_dim()
            # MLA Prefill TP=8 (unabsorbed, TMA K/V). NUM_HEADS per rank = 16.
            # Grid: (H, ceil(S/BM), B) where BM=64.
            params = [self.num_heads, self.seq_len]
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            # Kernel does its own per-block slicing (head, q_block, batch
            # come via task metadata). Each input is presented as the full
            # tensor.
            tb_graph.new_input(q_nope, (-1, -1, -1), -1, True)
            tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
            tb_graph.new_input(k, (-1, -1, -1), -1, True)
            tb_graph.new_input(v, (-1, -1, -1), -1, True)
            tb_graph.new_input(output, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [q_nope, q_pe, k, v, output], tb_graph
            )
            pk.kn_graph.register_task(
                tb_graph, "mla_prefill_tp8_sm100", params
            )
            return output

        if self.variant == "tp8_chunked":
            _require(q_nope, "q_nope", self.variant)
            _require(q_pe, "q_pe", self.variant)
            _require(k_nope, "k_nope", self.variant)
            _require(k_rope, "k_rope", self.variant)
            _require(v, "v", self.variant)
            _require(output, "output", self.variant)
            for n, v_ in (("q_len", self.q_len), ("kv_len", self.kv_len)):
                if v_ is None:
                    raise ValueError(
                        f"MLAPrefill(variant='tp8_chunked') requires {n}."
                    )
            if grid_dim is None:
                grid_dim = self.auto_grid_dim()
            params = [
                self.num_heads,
                self.q_len,
                self.kv_len,
                self.q_start,
                self.qfused_mode,
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

        if self.variant == "tp8_chunked_splitk":
            _require(q_nope, "q_nope", self.variant)
            _require(q_pe, "q_pe", self.variant)
            _require(k_nope, "k_nope", self.variant)
            _require(k_rope, "k_rope", self.variant)
            _require(v, "v", self.variant)
            _require(partial, "partial", self.variant)
            for n, v_ in (("q_len", self.q_len), ("kv_len", self.kv_len),
                          ("num_splits", self.num_splits)):
                if v_ is None:
                    raise ValueError(
                        f"MLAPrefill(variant='tp8_chunked_splitk') requires {n}."
                    )
            if grid_dim is None:
                grid_dim = self.auto_grid_dim()
            nqb = (self.q_len + 63) // 64
            params = [
                self.num_heads,
                self.q_len,
                self.kv_len,
                self.q_start,
                self.num_splits,
                nqb,
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
            return None  # only partial is written, returned by caller

        # tp8_chunked_reduce
        _require(partial, "partial", self.variant)
        _require(output, "output", self.variant)
        for n, v_ in (("q_len", self.q_len), ("num_splits", self.num_splits)):
            if v_ is None:
                raise ValueError(
                    f"MLAPrefill(variant='tp8_chunked_reduce') requires {n}."
                )
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
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


def _require(value, name, variant) -> None:
    if value is None:
        raise ValueError(
            f"MLAPrefill(variant='{variant}').compile requires "
            f"{name} (was None)."
        )
