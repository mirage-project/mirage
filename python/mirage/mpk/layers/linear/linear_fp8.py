"""FP8 dense / swapAB / BMM / split-K linear layers for SM100.

Dispatches to the FP8 GEMM family under
``include/mirage/persistent_kernel/tasks/blackwell/``:

* ``linear_fp8_sm100.cuh``        — dense + optional residual epilogue
* ``linear_fp8_swapAB_sm100.cuh`` — decode-optimised swapAB (+ optional residual)
* ``linear_fp8_bmm_sm100.cuh``    — per-head batched FP8 matmul
* split-K swapAB is generated from ``linear_fp8_swapAB_sm100.cuh`` with
  TMA reduce-add into the output (no separate ``.cuh``).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule


__all__ = [
    "LinearFP8",
    "LinearFP8WithResidual",
    "LinearFP8SwapAB",
    "LinearFP8SwapABWithResidual",
    "LinearFP8BMM",
    "LinearSplitKFP8SwapAB",
]


# ----------------------------------------------------------------------
# Tensor / scale layout (shared by every variant)
#
# input_fp8    : FP8 E4M3 (raw uint8 storage), row-major, last dim = K.
# input_scale  : uint32 UE8M0-packed scales, 4 scales per uint32 along K
#                (shape (*, K // 128); one logical scale per 128-elem K block).
# weight_fp8   : FP8 E4M3 (uint8), shape (out_features, in_features) for dense,
#                (H, D_out, D_in) for BMM.
# weight_scale : uint32 UE8M0-packed. The dense + with_residual kernels
#                consume SFB via a TMA descriptor that requires M-fastest
#                physical storage — we materialise that at compile() time
#                from a row-major HF-friendly nn.Parameter (see
#                ``_colmajor_weight_scale_for_tma`` below). The swapAB,
#                BMM, and split-K kernels read scales via raw pointers
#                (row-major (M, packed_K)) — no transpose needed.
# ----------------------------------------------------------------------


def _ue8m0_packed_to_fp32(scale_packed: torch.Tensor, k_dim: int) -> torch.Tensor:
    """Decode UE8M0-packed uint32 scales to fp32, expanded along K.

    Each uint32 carries 4 UE8M0 exponent bytes; byte ``s`` decodes to
    ``2 ** (s - 127)``. Used only by the PyTorch reference path.
    """
    leading = scale_packed.shape[:-1]
    packed_k = scale_packed.shape[-1]
    bytes_view = scale_packed.contiguous().view(torch.uint8).reshape(
        *leading, packed_k * 4
    )
    exp_f32 = bytes_view.to(torch.float32) - 127.0
    scales = torch.pow(torch.tensor(2.0), exp_f32)
    scales = scales.repeat_interleave(128, dim=-1)
    return scales[..., :k_dim]


def _dequant_fp8(
    fp8_bytes: torch.Tensor,
    scales_packed: torch.Tensor,
) -> torch.Tensor:
    """Return an fp32 dequant of ``fp8_bytes`` using UE8M0-packed scales."""
    if fp8_bytes.dtype == torch.float8_e4m3fn:
        fp32 = fp8_bytes.float()
    else:
        fp32 = fp8_bytes.view(torch.float8_e4m3fn).float()
    k_dim = fp32.shape[-1]
    scales = _ue8m0_packed_to_fp32(scales_packed, k_dim).to(fp32.device)
    return fp32 * scales


def _colmajor_weight_scale_for_tma(weight_scale: torch.Tensor) -> torch.Tensor:
    """Repack row-major ``(M, packed_K)`` weight_scale into M-fastest storage.

    Required for ``linear_fp8_sm100`` / ``linear_fp8_with_residual_sm100``
    SFB TMA descriptor (stride[0]=1, stride[1]=M). ``.t().contiguous().t()``
    yields the right strided view; caller stashes the result on ``self``
    so the underlying storage outlives the kernel run.
    """
    assert weight_scale.dim() == 2, (
        f"_colmajor_weight_scale_for_tma: expected 2D weight_scale, got "
        f"shape {tuple(weight_scale.shape)}"
    )
    return weight_scale.t().contiguous().t()


# ----------------------------------------------------------------------
# Shared base for the four (residual × swap_ab) dense variants.
# ----------------------------------------------------------------------
class _LinearFP8Base(MPKModule):
    """Shared base for FP8 dense linears.

    Holds weight + UE8M0 weight_scale, the PyTorch reference forward,
    and the grid heuristic. Subclasses override :meth:`compile` to pick
    the right task name and input list.

    Args:
        in_features: K. Multiple of 128.
        out_features: N. Multiple of 128 (MMA-M tile).
        scale_ue8m0: Required True (only UE8M0 packed scales supported).
        prefix: HF state_dict / tensor-name prefix.
    """

    # Subclasses set these:
    _TASK_NAME: str = ""
    _HAS_RESIDUAL: bool = False
    _SWAP_AB: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if not scale_ue8m0:
            raise NotImplementedError(
                f"{type(self).__name__} requires UE8M0-packed scales "
                "(scale_ue8m0=True)."
            )
        if in_features % 128 != 0:
            raise ValueError(
                f"{type(self).__name__}: in_features={in_features} must be "
                "a multiple of 128 (FP8 UE8M0 block size)."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.scale_ue8m0 = scale_ue8m0

        # Raw uint8 / uint32 storage so device pointers match the kernel
        # byte-for-byte (torch.float8_e4m3fn casts are patchy across versions).
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        # weight_scale: row-major (M, packed_K) for HF state_dict compat.
        # TMA-path variants materialise a col-major view in compile().
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, in_features // 128, dtype=torch.uint32),
            requires_grad=False,
        )

    def forward(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``y = dequant(x_fp8) @ dequant(w).T (+ residual)`` in fp32, cast to bf16."""
        if self._HAS_RESIDUAL and residual is None:
            raise ValueError(
                f"{type(self).__name__}.forward requires a residual tensor."
            )
        if not self._HAS_RESIDUAL and residual is not None:
            raise ValueError(
                f"{type(self).__name__}.forward got an unexpected residual."
            )
        x_f32 = _dequant_fp8(x_fp8, x_scale)
        w_f32 = _dequant_fp8(self.weight, self.weight_scale)
        out_f32 = F.linear(x_f32, w_f32)
        if residual is not None:
            out_f32 = out_f32 + residual.float()
        return out_f32.to(torch.bfloat16)

    def auto_grid_dim(self, x_fp8: Any = None) -> GridDim:
        """``(out_features // 128, 1, 1)``, capped at ``pk.num_workers``.

        Per-task output must be a multiple of MMA-M=128 (kernel-side
        constraint for swapAB; layer-enforced for dense for consistency).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        if self.out_features % 128 != 0:
            raise ValueError(
                f"{type(self).__name__}.auto_grid_dim: out_features="
                f"{self.out_features} must be a multiple of 128."
            )
        gx = max(1, min(self.out_features // 128, int(pk.num_workers)))
        return (gx, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """All SM100 FP8 dense kernels use 256 threads."""
        return (256, 1, 1)

    # ------------------------------------------------------------------
    # compile() helpers — subclasses call _compile_dense().
    # ------------------------------------------------------------------
    def _attach_weight_and_scale(self, pk) -> tuple[Any, Any]:
        """Attach weight + (possibly TMA-repacked) weight_scale."""
        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        # TMA-descriptor SFB path requires M-fastest storage; swapAB / BMM /
        # splitK kernels read scales via raw pointers (row-major OK).
        if self._SWAP_AB:
            ws_attached = self.weight_scale
        else:
            ws_attached = _colmajor_weight_scale_for_tma(self.weight_scale)
            self._weight_scale_colmajor = ws_attached
        ws_dt = pk.attach_input(ws_attached, name=f"{self.prefix}weight_scale")
        return w_dt, ws_dt

    def _compile_dense(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        residual: Optional[Any],
        output: Optional[Any],
        grid_dim: Optional[GridDim],
        block_dim: Optional[BlockDim],
        gate_mode: int,
    ) -> Any:
        import torch as _torch
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()

        if self._HAS_RESIDUAL != (residual is not None):
            raise ValueError(
                f"{type(self).__name__}.compile: residual must be "
                f"{'provided' if self._HAS_RESIDUAL else 'None'}."
            )

        w_dt, ws_dt = self._attach_weight_and_scale(pk)

        batch_size = x_fp8.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.out_features),
                dtype=mi.bfloat16,
                name=f"{self.prefix}linear_fp8_out",
            )
        elif isinstance(output, _torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}linear_fp8_out"
            )
        else:
            out_dt = output

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_fp8)
        if block_dim is None:
            block_dim = self.default_block_dim()

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x_fp8,   (-1, -1, -1), -1, True)
        tb_graph.new_input(x_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(w_dt,    (0, -1, -1),  -1, True)
        tb_graph.new_input(ws_dt,   (0, -1, -1),  -1, True)
        if self._HAS_RESIDUAL:
            tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(out_dt,  (1, -1, -1),  -1, True)

        if self._HAS_RESIDUAL:
            pk.kn_graph.customized(
                [x_fp8, x_scale, w_dt, ws_dt, residual, out_dt], tb_graph)
            # task_register expects params[0]=1 for residual-on, optional gate_mode.
            params = [1] if gate_mode == 0 else [1, gate_mode]
        else:
            pk.kn_graph.customized(
                [x_fp8, x_scale, w_dt, ws_dt, out_dt], tb_graph)
            params = [] if gate_mode == 0 else [gate_mode]

        pk.kn_graph.register_task(tb_graph, self._TASK_NAME, params)
        return out_dt


# ----------------------------------------------------------------------
# Four user-facing dense variants.
# ----------------------------------------------------------------------
class LinearFP8(_LinearFP8Base):
    """FP8 dense linear (no residual, no swapAB). Task ``linear_fp8_sm100``."""

    _TASK_NAME = "linear_fp8_sm100"
    _HAS_RESIDUAL = False
    _SWAP_AB = False

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        gate_mode: int = 0,
    ) -> Any:
        """Register ``linear_fp8_sm100`` (dense FP8 GEMM, TMA SFA/SFB).

        Tensor contract:
          x_fp8:        (batch, in_features) fp8_e4m3 viewed as uint8, row-major contiguous. TMA-128B swizzle; callers must align rows to BLOCK_K=128.
          x_scale:      (packed_k, aligned_batch) uint32 UE8M0-packed, row-major (M-fastest view of logical (B, K/128)). Callers MUST pre-pack via ``scale.t().contiguous()`` with batch padded to MMA-M=32.
          weight:       (out_features, in_features) fp8_e4m3 viewed as uint8, row-major contiguous. Built from ``self.weight``; in_features % 128 == 0.
          weight_scale: (out_features, in_features/128) uint32 UE8M0-packed; layer repacks to col-major (M-fastest) via ``_colmajor_weight_scale_for_tma`` for SFB TMA (stride[0]=1, stride[1]=M).
          output:       (batch, out_features) bf16, row-major; allocated if None.
        Notes: out_features % 128 == 0 (MMA-M tile); ``gate_mode`` toggles DSv3 fused-gate path (see task_register params).
        """
        return self._compile_dense(
            x_fp8, x_scale,
            residual=None, output=output,
            grid_dim=grid_dim, block_dim=block_dim, gate_mode=gate_mode,
        )


class LinearFP8WithResidual(_LinearFP8Base):
    """FP8 dense linear with bf16 residual add. Task
    ``linear_fp8_with_residual_sm100``."""

    _TASK_NAME = "linear_fp8_with_residual_sm100"
    _HAS_RESIDUAL = True
    _SWAP_AB = False

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        residual: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        gate_mode: int = 0,
    ) -> Any:
        """Register ``linear_fp8_with_residual_sm100`` (dense FP8 + bf16 residual epilogue).

        Tensor contract:
          x_fp8:        (batch, in_features) fp8_e4m3 viewed as uint8, row-major contiguous. TMA-128B swizzle.
          x_scale:      (packed_k, aligned_batch) uint32 UE8M0-packed, row-major (M-fastest view of logical (B, K/128)). Callers MUST pre-pack with batch padded to MMA-M=32.
          weight:       (out_features, in_features) fp8_e4m3 viewed as uint8, row-major contiguous.
          weight_scale: (out_features, in_features/128) uint32 UE8M0-packed; layer repacks to col-major (M-fastest) for SFB TMA.
          residual:     (batch, out_features) bf16, row-major; TMA-32B swizzle.
          output:       (batch, out_features) bf16, row-major; allocated if None.
        Notes: params=[1] selects residual-on epilogue; out_features % 128 == 0.
        """
        return self._compile_dense(
            x_fp8, x_scale,
            residual=residual, output=output,
            grid_dim=grid_dim, block_dim=block_dim, gate_mode=gate_mode,
        )


class LinearFP8SwapAB(_LinearFP8Base):
    """FP8 swapAB linear (decode fast path, ``batch_size <= 16``).
    Task ``linear_fp8_swapAB_sm100``. Kernel-side constraint: per-task
    output_size % 128 == 0 (MMA_M=128).
    """

    _TASK_NAME = "linear_fp8_swapAB_sm100"
    _HAS_RESIDUAL = False
    _SWAP_AB = True

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        gate_mode: int = 0,
    ) -> Any:
        """Register ``linear_fp8_swapAB_sm100`` (decode swapAB; A=weight, B=activation).

        Tensor contract:
          x_fp8:        (batch, in_features) fp8_e4m3 viewed as uint8, row-major contiguous. TMA rank-5; row stride from gmem stride[0].
          x_scale:      (batch, in_features/128) uint32 UE8M0-packed, row-major contiguous. Read via raw uint32* (no TMA, no transpose).
          weight:       (out_features, in_features) fp8_e4m3 viewed as uint8, row-major contiguous. Routed to kernel TMA_A.
          weight_scale: (out_features, in_features/128) uint32 UE8M0-packed, row-major contiguous. Raw uint32*; ``_SWAP_AB=True`` skips the col-major repack.
          output:       (batch, out_features) bf16, row-major; allocated if None.
        Notes: per-task out_features % 128 (MMA-M); batch <= 16 (MMA-N decode-only); in_features % 128.
        """
        return self._compile_dense(
            x_fp8, x_scale,
            residual=None, output=output,
            grid_dim=grid_dim, block_dim=block_dim, gate_mode=gate_mode,
        )


class LinearFP8SwapABWithResidual(_LinearFP8Base):
    """FP8 swapAB linear with residual add (decode fast path).
    Task ``linear_fp8_swapAB_with_residual_sm100``."""

    _TASK_NAME = "linear_fp8_swapAB_with_residual_sm100"
    _HAS_RESIDUAL = True
    _SWAP_AB = True

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        residual: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        gate_mode: int = 0,
    ) -> Any:
        """Register ``linear_fp8_swapAB_with_residual_sm100`` (decode swapAB + residual epilogue).

        Tensor contract:
          x_fp8:        (batch, in_features) fp8_e4m3 viewed as uint8, row-major contiguous. TMA rank-5, 128B swizzle.
          x_scale:      (batch, in_features/128) uint32 UE8M0-packed, row-major contiguous. Raw uint32* (no TMA).
          weight:       (out_features, in_features) fp8_e4m3 viewed as uint8, row-major contiguous. TMA_A.
          weight_scale: (out_features, in_features/128) uint32 UE8M0-packed, row-major contiguous. Raw uint32* (no transpose).
          residual:     (batch, out_features) bf16, row-major; TMA rank-5, no swizzle.
          output:       (batch, out_features) bf16, row-major; allocated if None.
        Notes: per-task out_features % 128; batch <= 16; params=[1] selects residual-on.
        """
        return self._compile_dense(
            x_fp8, x_scale,
            residual=residual, output=output,
            grid_dim=grid_dim, block_dim=block_dim, gate_mode=gate_mode,
        )


# ----------------------------------------------------------------------
# LinearFP8BMM — per-head batched matmul (decode Q-absorb path).
# ----------------------------------------------------------------------
class LinearFP8BMM(MPKModule):
    """Per-head FP8 batched matmul on SM100.

    Computes ``output[n, h, :] = input[n, h, :] @ weight[h, :, :].T`` with
    FP8 E4M3 operands + UE8M0 scales. Decode-only (``N <= 16``).

    Args:
        num_heads: ``H``.
        in_features_per_head: ``D_in``. Multiple of 128.
        out_features_per_head: ``D_out``. Multiple of 128 (MMA-M).
        scale_ue8m0: Required True.
        prefix: HF state_dict / tensor-name prefix.
    """

    def __init__(
        self,
        num_heads: int,
        in_features_per_head: int,
        out_features_per_head: int,
        *,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if not scale_ue8m0:
            raise NotImplementedError(
                "LinearFP8BMM requires UE8M0-packed scales."
            )
        if in_features_per_head % 128 != 0:
            raise ValueError(
                f"LinearFP8BMM: in_features_per_head={in_features_per_head} "
                "must be a multiple of 128."
            )
        if out_features_per_head % 128 != 0:
            raise ValueError(
                f"LinearFP8BMM: out_features_per_head={out_features_per_head} "
                "must be a multiple of 128 (kernel MMA-M=128)."
            )
        self.num_heads = num_heads
        self.in_features_per_head = in_features_per_head
        self.out_features_per_head = out_features_per_head
        self.scale_ue8m0 = scale_ue8m0

        packed_k = in_features_per_head // 128
        self.weight = nn.Parameter(
            torch.empty(
                num_heads, out_features_per_head, in_features_per_head,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                num_heads, out_features_per_head, packed_k,
                dtype=torch.uint32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        """``out[n,h,o] = sum_i dequant(x)[n,h,i] * dequant(w)[h,o,i]``; cast to bf16."""
        x_f32 = _dequant_fp8(x_fp8, x_scale)
        w_f32 = _dequant_fp8(self.weight, self.weight_scale)
        out_f32 = torch.einsum("nhi,hoi->nho", x_f32, w_f32)
        return out_f32.to(torch.bfloat16)

    def auto_grid_dim(self, x_fp8: Any = None) -> GridDim:
        """``(D_out // 128, H, 1)``, capped so that ``gx*gy <= num_workers``.

        Kernel hardcodes one head per CTA (``grid.y == H``); we add as
        many M-shards as fit in the remaining worker budget while keeping
        ``out_features_per_head % (128 * grid.x) == 0``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        gy = self.num_heads
        max_gx_by_tile = max(1, self.out_features_per_head // 128)
        budget = max(1, int(pk.num_workers) // max(gy, 1))
        gx = 1
        for candidate in range(min(max_gx_by_tile, budget), 0, -1):
            if self.out_features_per_head % (candidate * 128) == 0:
                gx = candidate
                break
        return (gx, gy, 1)

    def default_block_dim(self) -> BlockDim:
        """Kernel role layout fixed at 256 threads (8 warps)."""
        return (256, 1, 1)

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register ``linear_fp8_bmm_sm100`` (per-head FP8 BMM via swapAB body).

        Tensor contract:
          x_fp8:        (N, H, D_in) fp8_e4m3 viewed as uint8, row-major; per-head row stride = H*D_in (gmem stride[0]). 2D (N, H*D_in) folded layout also accepted.
          x_scale:      (N, H, D_in/128) uint32 UE8M0-packed, row-major; per-head row stride = H*packed_K (input_scale_row_stride). Raw uint32* (no TMA).
          weight:       (H, D_out, D_in) fp8_e4m3 viewed as uint8, row-major; within-head row stride = D_in (stride[1]). Routed to TMA_A.
          weight_scale: (H, D_out, D_in/128) uint32 UE8M0-packed, row-major; within-head row stride = packed_K. Raw uint32* (no transpose).
          output:       (N, H, D_out) bf16, row-major; per-head row stride = H*D_out. 2D output requires grid.x == 1.
        Notes: D_out % 128 (MMA-M), D_in % 128 (BLOCK_K), N <= 16 decode-only; one head per CTA (grid.y == H).
        """
        import torch as _torch
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_fp8)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        # SFB read via raw pointer in row-major (D_out, packed_K) — no
        # transpose required (kernel uses weight_scale_row_stride = packed_K).
        ws_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

        N = x_fp8.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(N, self.num_heads, self.out_features_per_head),
                dtype=mi.bfloat16,
                name=f"{self.prefix}linear_fp8_bmm_out",
            )
        elif isinstance(output, _torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}linear_fp8_bmm_out"
            )
        else:
            out_dt = output

        assert w_dt.num_dims == 3
        assert ws_dt.num_dims == 3
        assert x_fp8.num_dims in (2, 3)
        assert x_scale.num_dims in (2, 3)
        assert out_dt.num_dims in (2, 3)

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x_fp8,   (-1, 1, -1), -1, True)
        tb_graph.new_input(x_scale, (-1, 1, -1), -1, True)
        tb_graph.new_input(w_dt,    (1, 0, -1),  -1, True)
        tb_graph.new_input(ws_dt,   (1, 0, -1),  -1, True)
        if out_dt.num_dims == 3:
            tb_graph.new_input(out_dt, (2, 1, -1), -1, True)
        else:
            assert grid_dim[0] == 1, (
                "linear_fp8_bmm with 2D output requires grid.x=1 "
                "(D_out cannot be sharded across CTAs when packed flat)")
            tb_graph.new_input(out_dt, (-1, 1, -1), -1, True)
        pk.kn_graph.customized(
            [x_fp8, x_scale, w_dt, ws_dt, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "linear_fp8_bmm_sm100", [])
        return out_dt


# ----------------------------------------------------------------------
# LinearSplitKFP8SwapAB — split-K decode variant.
# ----------------------------------------------------------------------
class LinearSplitKFP8SwapAB(MPKModule):
    """Split-K FP8 swapAB linear (decode). Task
    ``splitk_linear_fp8_swapAB_sm100``.

    ``grid.y`` CTAs compute partial K-slices and TMA reduce-add into
    ``output``. ``accumulate=True``: caller pre-populates output (added
    on top); ``accumulate=False``: layer prepends a tensor_init to zero
    it. Constraints: per-task N % 128, per-task K % 512 (UE8M0 packs
    4 logical-K per uint32), batch_size <= 16.

    Args:
        in_features: K. Multiple of 128.
        out_features: N. Multiple of 128.
        accumulate: See above.
        scale_ue8m0: Required True.
        prefix: HF state_dict / tensor-name prefix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        accumulate: bool,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if not scale_ue8m0:
            raise NotImplementedError(
                "LinearSplitKFP8SwapAB requires UE8M0-packed scales."
            )
        if in_features % 128 != 0:
            raise ValueError(
                f"LinearSplitKFP8SwapAB: in_features={in_features} must "
                "be a multiple of 128."
            )
        if out_features % 128 != 0:
            raise ValueError(
                f"LinearSplitKFP8SwapAB: out_features={out_features} must "
                "be a multiple of 128."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.accumulate = accumulate
        self.scale_ue8m0 = scale_ue8m0

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, in_features // 128, dtype=torch.uint32),
            requires_grad=False,
        )

    def forward(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``y = dequant(x) @ dequant(w).T (+ output if accumulate)``, cast bf16."""
        x_f32 = _dequant_fp8(x_fp8, x_scale)
        w_f32 = _dequant_fp8(self.weight, self.weight_scale)
        result = F.linear(x_f32, w_f32)
        if self.accumulate:
            if output is None:
                raise ValueError(
                    "LinearSplitKFP8SwapAB(accumulate=True).forward "
                    "requires the prior `output` tensor."
                )
            result = result + output.float()
        return result.to(torch.bfloat16)

    def auto_grid_dim(self, x_fp8: Any = None) -> GridDim:
        """``(N // 128, split_k, 1)``, with ``grid.x*grid.y`` targeting
        ``pk.num_workers``. split_k must divide ``in_features // 512``
        (UE8M0 packs 4 logical-K per uint32, so per-task K must be a
        multiple of 512 — see task_register assertion).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        gx = max(1, min(self.out_features // 128, int(pk.num_workers)))
        # K_per_task = in_features / split_k must be a multiple of 512, i.e.,
        # split_k must divide (in_features // 512).
        max_k_units = max(1, self.in_features // 512)
        budget = max(1, int(pk.num_workers) // gx)
        split_k = 1
        for candidate in range(min(budget, max_k_units), 0, -1):
            if max_k_units % candidate == 0:
                split_k = candidate
                break
        return (gx, split_k, 1)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        output: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register ``splitk_linear_fp8_swapAB_sm100`` (split-K decode swapAB; TMA reduce-add).

        Tensor contract:
          x_fp8:        (batch, in_features) fp8_e4m3 viewed as uint8, row-major contiguous; per-task K-slice via TBGraph stride/offset, gmem row stride = full in_features.
          x_scale:      (batch, in_features/128) uint32 UE8M0-packed, row-major contiguous; raw uint32*, row stride = full packed_K (=in_features/512). No transpose.
          weight:       (out_features, in_features) fp8_e4m3 viewed as uint8, row-major; TMA_A with row stride = full in_features (per-task K-slice via base_ptr offset).
          weight_scale: (out_features, in_features/128) uint32 UE8M0-packed, row-major; raw uint32*, row stride = full packed_K. No transpose.
          output:       (batch, out_features) bf16, row-major; caller-owned reduce-add target (input + output of the layer).
        Notes: per-task out % 128 (MMA-M); per-task K % 512 (UE8M0 packs 4 logical-K / uint32 — split_k must divide in_features/512); batch <= 16. ``accumulate=False`` prepends a ``tensor_init_layer`` to zero ``output`` (``=True`` skips it, treating output as the prior accumulator).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_fp8)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        # SFB read via raw pointer in row-major (M, packed_K) — no transpose.
        ws_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

        if not self.accumulate:
            pk.tensor_init_layer(
                target=output,
                dummy=x_fp8,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dummy_input_map=(-1, 1, -1),
                target_input_map=(1, -1, -1),
            )
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x_fp8,   (-1, 1, -1), 1, True)
        tb_graph.new_input(x_scale, (-1, 1, -1), 1, True)
        tb_graph.new_input(w_dt,    (0, 1, -1),  1, True)
        tb_graph.new_input(ws_dt,   (0, 1, -1),  1, True)
        tb_graph.new_input(output,  (1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [x_fp8, x_scale, w_dt, ws_dt, output], tb_graph)
        pk.kn_graph.register_task(
            tb_graph, "splitk_linear_fp8_swapAB_sm100", [])
        return output
