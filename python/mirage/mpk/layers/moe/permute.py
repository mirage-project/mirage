"""MoE permute / unpermute — glue for the PR-674 group-GEMM path.

Two distinct catalog modules wrap the two sides of the NEW MoE pipeline
(DeepSeek V3 ``MPK_DSV3_NEW_MOE=1`` path).

* :class:`MoEPermute` — wraps
  :meth:`PersistentKernel.moe_permute_sm100_layer`
  (task ``moe_permute_sm100``).
* :class:`MoEUnpermute` — wraps
  :meth:`PersistentKernel.moe_unpermute_sm100_layer`
  (task ``moe_unpermute_sm100``).

Both are highly metadata-dependent and have no plain-PyTorch
reference: the permutation depends on the routing output (which lives
in MPK runtime tensors) and on per-expert padding (``bm_padding``)
that exists only because the downstream grouped-GEMM kernel requires
the M dimension to be a multiple of 128. We document the layouts but
raise ``NotImplementedError`` from ``forward()``.

Layout reminder — :class:`MoEPermute`
-------------------------------------

Inputs:

* ``input_fp8``         : ``(mbt, hidden_size)`` E4M3.
* ``input_scale``       : ``(mbt, K_PACKED)`` uint32 — UE8M0-PACKED
  scales (4 group-scales per uint32). The kernel REQUIRES UE8M0
  packing; passing a plain fp32 scale tensor will silently produce
  wrong results.
* ``topk_weights``      : ``(mbt, topk)`` float32 — the renormalized
  top-k weights from routing.
* ``routing_indices``   : ``(E_local, mbt)`` int32 — expert-major
  routing tensor.

Outputs:

* ``permuted_fp8``      : ``(M_total, hidden_size)`` E4M3 — tokens
  re-laid by destination expert. ``M_total = E_local * bm_padding``.
* ``permuted_scale``    : ``(K_PACKED, M_total)`` uint32 (K-outermost
  layout — note the transpose vs ``input_scale``).
* ``meta``              : ``(2, M_total + mbt*topk)`` int32 — packed
  metadata. ``meta[0, : M_total]`` = ``permuted_weights`` (fp32 bits);
  ``meta[0, M_total :]`` = ``token_to_permuted`` (row + 1 — 0 means
  "not routed locally"). The doubled first dim (``BATCH_SIZE=2``) is a
  ``tensor_init`` workaround for byte alignment; both rows hold the
  same logical data after the permute runs.

Grid: ``(E_local, 1, 1)`` — one CTA per local expert. Block 128. The
pk method sets these unconditionally (the caller passes ``bm_padding``
only).

Layout reminder — :class:`MoEUnpermute`
---------------------------------------

Inputs:

* ``permuted_output``   : ``(M_total, hidden_size)`` bf16 — output of
  the W2 group GEMM (NOT bf16-cast yet on disk; the kernel reads bf16
  per row).
* ``meta``              : ``(2, M_total + mbt*topk)`` int32 — same as
  produced by :class:`MoEPermute`.
* ``residual``          : ``(mbt, hidden_size)`` bf16 — the shared-
  expert + transformer residual (DeepSeek V3 folds the residual add
  into the unpermute).
* ``output``            : ``(mbt, hidden_size)`` bf16.

Grid: ``(mbt, 1, 1)`` — one CTA per (output) token. Block 128.
The kernel decodes ``meta`` then does
``output[t] = residual[t] + sum_k(permuted_output[token_to_permuted[t,k]-1]
                                     * permuted_weights[same row])``.

Both modules expose ``__init__`` parameters that match what
:meth:`PersistentKernel.moe_permute_sm100_layer` /
:meth:`...unpermute_sm100_layer` need; ``compile()`` is the only entry
point. Grid/block are baked into the pk methods (the kernel constants
are fixed), so the ``grid_dim``/``block_dim`` overrides are accepted
but rejected if non-default values are passed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoEPermute", "MoEUnpermute"]


class MoEPermute(MPKModule):
    """MoE expand-permute-sort glue task.

    Args:
        num_local_experts: Number of local (per-rank) experts. Equals
            ``moe_permute_sm100_layer``'s ``E_LOCAL`` and the kernel's
            grid.x.
        hidden_size: Reduction (K) axis of the downstream group GEMM
            (i.e. ``input_fp8.dim(1)``). Used only for shape-checking.
        num_experts_per_tok: Top-k width (``topk_weights.dim(1)``).
        bm_padding: Per-expert padding to a multiple-of-128 M length
            for the grouped GEMM. Matches the
            ``self._moe_bm_padding`` value in the DeepSeek V3 builder
            (default 128).
        scale_ue8m0: Must be ``True`` — the permute kernel REQUIRES
            UE8M0-packed input scales. Kept as an explicit kwarg so
            the API matches the FP8 quantize-side modules.
        prefix: Reserved for symmetry. The module owns no parameters.
    """

    def __init__(
        self,
        num_local_experts: int,
        hidden_size: int,
        num_experts_per_tok: int,
        *,
        bm_padding: int = 128,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if not scale_ue8m0:
            raise ValueError(
                "MoEPermute requires UE8M0-packed input scales "
                "(the moe_permute_sm100 kernel does not accept plain fp32 scales)."
            )
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.bm_padding = bm_padding
        self.scale_ue8m0 = scale_ue8m0

    # ------------------------------------------------------------------
    # PyTorch reference — intentionally not implemented.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MoEPermute.forward() is not implemented: the permutation "
            "depends on the routing tensors and on the M-padding scheme "
            "used only inside the MPK NEW-MoE path. Validate via the "
            "test-mode driver (see demo/deepseek_v3/builder.py "
            "_new_moe_dispatch_inline) rather than a torch oracle."
        )

    # ------------------------------------------------------------------
    # Grid heuristic — fixed by the kernel.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(num_local_experts, 1, 1)`` — one CTA per local expert."""
        return (self.num_local_experts, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """``moe_permute_sm100`` is hard-wired to 128 threads."""
        return (128, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        input_fp8: DTensor,
        input_scale: DTensor,
        topk_weights: DTensor,
        routing_indices: DTensor,
        permuted_fp8: DTensor,
        permuted_scale: DTensor,
        meta: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        """Register the ``moe_permute_sm100`` task.

        Grid/block are baked into the pk method (which constructs the
        TBGraph with ``grid_dim=(E_LOCAL, 1, 1)``, ``block_dim=(128, 1, 1)``).
        Passing non-default overrides raises — the kernel constants
        cannot be changed externally.

        Returns ``(permuted_fp8, permuted_scale, meta)`` for caller
        convenience.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is not None and grid_dim != self.auto_grid_dim():
            raise ValueError(
                f"MoEPermute: grid_dim is fixed at {self.auto_grid_dim()} "
                f"(one CTA per local expert); got {grid_dim}"
            )
        if block_dim is not None and block_dim != self.default_block_dim():
            raise ValueError(
                f"MoEPermute: block_dim is fixed at {self.default_block_dim()}; "
                f"got {block_dim}"
            )

        # Inlined task registration (formerly pk.moe_permute_sm100_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert input_fp8.num_dims == 2
        assert input_scale.num_dims == 2
        assert topk_weights.num_dims == 2
        assert routing_indices.num_dims == 2
        assert permuted_fp8.num_dims == 2
        assert permuted_scale.num_dims == 2
        # meta is shaped (2, M_TOTAL + MBT*TOPK) int32.
        assert meta.num_dims == 2
        assert meta.dim(0) == 2

        K = input_fp8.dim(1)
        K_PACKED = input_scale.dim(1)
        MBT = input_fp8.dim(0)
        TOPK = topk_weights.dim(1)
        E_LOCAL = routing_indices.dim(0)
        M_TOTAL = E_LOCAL * self.bm_padding
        assert routing_indices.dim(1) == MBT
        assert topk_weights.dim(0) == MBT
        assert permuted_fp8.dim(0) == M_TOTAL
        assert permuted_fp8.dim(1) == K
        assert permuted_scale.dim(0) == K_PACKED
        assert permuted_scale.dim(1) == M_TOTAL
        assert meta.dim(1) == M_TOTAL + MBT * TOPK, (
            f"meta length must be {M_TOTAL + MBT * TOPK}, got {meta.dim(1)}")

        params = [K, K_PACKED, MBT, TOPK, E_LOCAL, self.bm_padding]
        # Grid/block constants are fixed by the kernel; we still respect any
        # explicit override resolved above (it must equal the auto values).
        kernel_grid_dim = (E_LOCAL, 1, 1)
        kernel_block_dim = (128, 1, 1)
        tb_graph = TBGraph(CyTBGraph(kernel_grid_dim, kernel_block_dim, 1, 64))
        tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weights, (-1, -1, -1), -1, True)
        # routing_indices: (-1, -1, -1) so the kernel sees the FULL (E_LOCAL, MBT)
        # buffer and computes its expert row from task_metadata.expert_offset.
        tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(permuted_fp8, (-1, -1, -1), -1, True)
        tb_graph.new_input(permuted_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(meta, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [input_fp8, input_scale, topk_weights, routing_indices,
             permuted_fp8, permuted_scale, meta], tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "moe_permute_sm100", params)
        return permuted_fp8, permuted_scale, meta


class MoEUnpermute(MPKModule):
    """MoE combine-unpermute task — inverse of :class:`MoEPermute`.

    Args:
        hidden_size: Trailing dim of ``permuted_output`` / ``residual``
            / ``output``. Used for shape-checking.
        prefix: Reserved. No parameters live here.

    The kernel infers ``MBT``, ``M_TOTAL``, and ``TOPK`` from the
    shapes of its DTensor inputs at compile time, so this module
    needs no additional ``__init__`` configuration beyond
    ``hidden_size``.
    """

    def __init__(self, hidden_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size

    # ------------------------------------------------------------------
    # PyTorch reference — intentionally not implemented.
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MoEUnpermute.forward() is not implemented: the inverse "
            "permutation reads MPK meta-tensors (token_to_permuted + "
            "permuted_weights packed into `meta`) that are only "
            "produced by MoEPermute on the device side. Use the "
            "test-mode driver for end-to-end validation."
        )

    # ------------------------------------------------------------------
    # Grid heuristic — read from the residual / output shape.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, residual: DTensor) -> GridDim:
        """``(MBT, 1, 1)`` — one CTA per output token."""
        return (residual.dim(0), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        permuted_output: DTensor,
        meta: DTensor,
        residual: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register the ``moe_unpermute_sm100`` task.

        Grid/block are baked into the pk method (which constructs the
        TBGraph with ``grid_dim=(MBT, 1, 1)``, ``block_dim=(128, 1, 1)``).
        ``grid_dim``/``block_dim`` overrides are accepted but must
        equal the auto values.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        auto = self.auto_grid_dim(residual)
        if grid_dim is not None and grid_dim != auto:
            raise ValueError(
                f"MoEUnpermute: grid_dim is fixed at {auto} "
                f"(one CTA per output token); got {grid_dim}"
            )
        if block_dim is not None and block_dim != self.default_block_dim():
            raise ValueError(
                f"MoEUnpermute: block_dim is fixed at {self.default_block_dim()}; "
                f"got {block_dim}"
            )

        # Inlined task registration (formerly pk.moe_unpermute_sm100_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert permuted_output.num_dims == 2
        # meta is shaped (2, M_TOTAL + MBT*TOPK) int32 — same contract as
        # MoEPermute writes.
        assert meta.num_dims == 2
        assert meta.dim(0) == 2
        assert residual.num_dims == 2
        assert output.num_dims == 2

        MBT = residual.dim(0)
        HIDDEN = permuted_output.dim(1)
        M_TOTAL = permuted_output.dim(0)
        # meta = M_TOTAL (weights) + MBT*TOPK (token_to_permuted) entries.
        meta_len = meta.dim(1)
        TOPK = (meta_len - M_TOTAL) // MBT
        assert M_TOTAL + MBT * TOPK == meta_len
        assert residual.dim(1) == HIDDEN
        assert output.dim(0) == MBT
        assert output.dim(1) == HIDDEN

        params = [MBT, TOPK, HIDDEN, M_TOTAL]
        kernel_grid_dim = (MBT, 1, 1)
        kernel_block_dim = (128, 1, 1)
        tb_graph = TBGraph(CyTBGraph(kernel_grid_dim, kernel_block_dim, 1, 64))
        # All inputs/outputs are (-1, -1, -1) so the kernel sees the FULL
        # tensors and indexes them with task_metadata.request_id (= my_token).
        tb_graph.new_input(permuted_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(meta, (-1, -1, -1), -1, True)
        tb_graph.new_input(residual, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [permuted_output, meta, residual, output], tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "moe_unpermute_sm100", params)
        return output
