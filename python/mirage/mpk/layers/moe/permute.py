"""MoE permute / unpermute — glue for the grouped-GEMM MoE path.

Wraps the two halves of the expand-then-contract MoE flow (kernels:
``include/mirage/persistent_kernel/tasks/blackwell/moe_permute_sm100.cuh``
and ``moe_unpermute_sm100.cuh``).

* :class:`MoEPermute`   -> ``moe_permute_sm100``.
* :class:`MoEUnpermute` -> ``moe_unpermute_sm100``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoEPermute", "MoEUnpermute"]


class MoEPermute(MPKModule):
    """Expand-permute-sort tokens by destination expert.

    Permute writes:

    * ``permuted_fp8`` ``(M_total, hidden)`` E4M3, where ``M_total =
      E_local * bm_padding`` (each expert's rows are padded to a
      multiple of 128 for the grouped GEMM).
    * ``permuted_scale`` ``(K_PACKED, M_total)`` uint32 — transposed
      vs the input layout, UE8M0-packed (4 group-scales per uint32).
      The permute kernel REQUIRES UE8M0; passing plain fp32 silently
      produces wrong results.
    * ``meta`` ``(2, M_total + MBT*TOPK)`` int32 — first row holds the
      ``permuted_weights`` (fp32 bits) for ``[0:M_total)`` followed by
      ``token_to_permuted`` (row + 1, 0 = "not routed locally") for
      ``[M_total:)``; second row is a tensor_init byte-alignment dupe.
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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MoEPermute.forward(): the permutation depends on device-side "
            "routing metadata; validate via the test-mode driver."
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Kernel hardcodes ``(E_LOCAL, 1, 1)`` — one CTA per local expert."""
        return (self.num_local_experts, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """``moe_permute_sm100`` is hard-wired to 128 threads."""
        return (128, 1, 1)

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
        """Register ``moe_permute_sm100`` — expand-permute-sort by destination expert.

        Tensor contract:
          input_fp8: (MBT, K) fp8_e4m3 (uint8 in kernel) — pre-quantized activations.
          input_scale: (MBT, K_PACKED) uint32 — UE8M0 packed (4 group-scales per uint32, REQUIRED).
          topk_weights: (MBT, TOPK) fp32 — routing scores to be permuted into ``meta``.
          routing_indices: (E_LOCAL, MBT) int32, EXPERT-MAJOR (slot+1 or 0).
          permuted_fp8 (out): (M_TOTAL=E_LOCAL*bm_padding, K) fp8 (uint8), per-expert padded.
          permuted_scale (out): (K_PACKED, M_TOTAL) uint32 — TRANSPOSED vs input, UE8M0 packed.
          meta (out): (2, M_TOTAL + MBT*TOPK) int32 — row 0: [permuted_weights fp32-bits |
            token_to_permuted (row+1, 0 = unrouted)]; row 1 is a tensor_init alignment dupe.

        Notes: grid hardcoded to (E_LOCAL, 1, 1), 128 threads; params=(K, K_PACKED, MBT,
        TOPK, E_LOCAL, bm_padding). ``scale_ue8m0=True`` is required (kernel rejects fp32 scales).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if grid_dim is not None and grid_dim != self.auto_grid_dim():
            raise ValueError(
                f"MoEPermute: grid_dim is fixed at {self.auto_grid_dim()}; "
                f"got {grid_dim}"
            )
        if block_dim is not None and block_dim != self.default_block_dim():
            raise ValueError(
                f"MoEPermute: block_dim is fixed at {self.default_block_dim()}; "
                f"got {block_dim}"
            )

        assert input_fp8.num_dims == 2
        assert input_scale.num_dims == 2
        assert topk_weights.num_dims == 2
        assert routing_indices.num_dims == 2
        assert permuted_fp8.num_dims == 2
        assert permuted_scale.num_dims == 2
        # meta is shaped (2, M_TOTAL + MBT*TOPK) int32.
        assert meta.num_dims == 2 and meta.dim(0) == 2

        K = input_fp8.dim(1)
        K_PACKED = input_scale.dim(1)
        MBT = input_fp8.dim(0)
        TOPK = topk_weights.dim(1)
        E_LOCAL = routing_indices.dim(0)
        M_TOTAL = E_LOCAL * self.bm_padding
        assert routing_indices.dim(1) == MBT
        assert topk_weights.dim(0) == MBT
        assert permuted_fp8.dim(0) == M_TOTAL and permuted_fp8.dim(1) == K
        assert permuted_scale.dim(0) == K_PACKED and permuted_scale.dim(1) == M_TOTAL
        assert meta.dim(1) == M_TOTAL + MBT * TOPK

        params = [K, K_PACKED, MBT, TOPK, E_LOCAL, self.bm_padding]
        tb_graph = TBGraph(CyTBGraph((E_LOCAL, 1, 1), (128, 1, 1), 1, 64))
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
    """Combine-unpermute — inverse of :class:`MoEPermute`.

    Reads ``permuted_output (M_total, hidden)`` bf16, the ``meta`` packed
    by MoEPermute (``permuted_weights`` + ``token_to_permuted``), and a
    bf16 residual; writes ``output[t] = residual[t] +
    sum_k(permuted_output[token_to_permuted[t,k]-1] * permuted_weights[..])``.
    Folds the per-expert combine + residual add into one task.
    """

    def __init__(self, hidden_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "MoEUnpermute.forward(): the inverse permutation reads MPK "
            "meta-tensors only produced device-side; validate via test-mode."
        )

    def auto_grid_dim(self, residual: DTensor) -> GridDim:
        """Kernel hardcodes ``(MBT, 1, 1)`` — one CTA per output token."""
        return (residual.dim(0), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

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
        """Register ``moe_unpermute_sm100`` — combine + residual add (inverse of MoEPermute).

        Tensor contract:
          permuted_output: (M_TOTAL, HIDDEN) bf16 — per-expert grouped W2 output.
          meta: (2, M_TOTAL + MBT*TOPK) int32 — same contract as MoEPermute output meta:
            row 0 holds permuted_weights (fp32 bits, reinterpreted) on [0:M_TOTAL) and
            token_to_permuted (int32, 1-indexed; 0 = unrouted) on [M_TOTAL:).
          residual: (MBT, HIDDEN) bf16 — additive (e.g., shared-expert output).
          output (out): (MBT, HIDDEN) bf16 = residual + sum_k(permuted_output[t2p[t,k]-1] *
            permuted_weights[t2p[t,k]-1]).

        Notes: grid hardcoded to (MBT, 1, 1), 128 threads; params=(MBT, TOPK, HIDDEN, M_TOTAL).
        TOPK is derived from ``(meta.dim(1) - M_TOTAL) // MBT``.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        auto = self.auto_grid_dim(residual)
        if grid_dim is not None and grid_dim != auto:
            raise ValueError(
                f"MoEUnpermute: grid_dim is fixed at {auto}; got {grid_dim}"
            )
        if block_dim is not None and block_dim != self.default_block_dim():
            raise ValueError(
                f"MoEUnpermute: block_dim is fixed at {self.default_block_dim()}; "
                f"got {block_dim}"
            )

        assert permuted_output.num_dims == 2
        # meta is (2, M_TOTAL + MBT*TOPK) int32 — same contract as MoEPermute.
        assert meta.num_dims == 2 and meta.dim(0) == 2
        assert residual.num_dims == 2 and output.num_dims == 2

        MBT = residual.dim(0)
        HIDDEN = permuted_output.dim(1)
        M_TOTAL = permuted_output.dim(0)
        meta_len = meta.dim(1)
        TOPK = (meta_len - M_TOTAL) // MBT
        assert M_TOTAL + MBT * TOPK == meta_len
        assert residual.dim(1) == HIDDEN
        assert output.dim(0) == MBT and output.dim(1) == HIDDEN

        params = [MBT, TOPK, HIDDEN, M_TOTAL]
        tb_graph = TBGraph(CyTBGraph((MBT, 1, 1), (128, 1, 1), 1, 64))
        # (-1, -1, -1) on all so the kernel indexes via task_metadata.request_id.
        tb_graph.new_input(permuted_output, (-1, -1, -1), -1, True)
        tb_graph.new_input(meta, (-1, -1, -1), -1, True)
        tb_graph.new_input(residual, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [permuted_output, meta, residual, output], tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "moe_unpermute_sm100", params)
        return output
