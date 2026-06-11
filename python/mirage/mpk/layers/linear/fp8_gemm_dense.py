"""Dense FP8 GEMM on SM100 (smallm + mediumm variants).

Per-arch task kernel:
* SM100 Blackwell smallm  : ``tasks/blackwell/fp8_gemm_dense_smallm_sm100.cuh``  (``fp8_gemm_dense_smallm_sm100``, NE=2)
* SM100 Blackwell mediumm : ``tasks/blackwell/fp8_gemm_dense_mediumm_sm100.cuh`` (``fp8_gemm_dense_mediumm_sm100``, NE=4)

Both invoke the common body in ``fp8_gemm_dense_sm100_common.cuh``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule


__all__ = ["FP8GEMMDenseSmallM", "FP8GEMMDenseMediumM"]


def _dequant_fp8_blockscale(
    fp8_bytes: torch.Tensor,
    block_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize FP8 with 128x128 block float32 scale.

    Args:
        fp8_bytes: ``(M, K)`` float8_e4m3fn or uint8 raw bytes.
        block_scale: ``(M // 128, K // 128)`` float32 (weight) or
            ``(M, K // 128)`` float32 (per-row activation).
    Returns:
        ``(M, K)`` float32.
    """
    if fp8_bytes.dtype == torch.float8_e4m3fn:
        fp32 = fp8_bytes.float()
    else:
        fp32 = fp8_bytes.view(torch.float8_e4m3fn).float()
    M, K = fp32.shape
    if block_scale.shape == (M, K // 128):
        scales = block_scale.float().to(fp32.device).repeat_interleave(128, dim=1)
    elif block_scale.shape == ((M + 127) // 128, K // 128):
        scales = (
            block_scale.float().to(fp32.device)
            .repeat_interleave(128, dim=0)[:M]
            .repeat_interleave(128, dim=1)[:, :K]
        )
    else:
        raise ValueError(
            f"_dequant_fp8_blockscale: unexpected block_scale shape "
            f"{tuple(block_scale.shape)} for fp8 shape {(M, K)}")
    return fp32 * scales


class _FP8GEMMDenseBase(MPKModule):
    """Dense FP8 GEMM base. Subclasses set ``_TASK_NAME`` and override ``compile``.

    Args:
        in_features: K (reduction). Multiple of 128.
        out_features: N (output). Multiple of 128.
        scale_ue8m0: Accepted for API parity; scales are stored as fp32
            block scales (the kernel does its own UE8M0 quant at MMA time).
        prefix: state_dict / tensor-name prefix.

    Owned parameters: ``weight`` ``(N, K)`` uint8 (E4M3 bytes) and
    ``weight_scale`` ``(N // 128, K // 128)`` float32.
    """

    _TASK_NAME: str = ""  # set by subclasses

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if in_features % 128 != 0:
            raise ValueError(
                f"{type(self).__name__}: in_features={in_features} must be a multiple of 128."
            )
        if out_features % 128 != 0:
            raise ValueError(
                f"{type(self).__name__}: out_features={out_features} must be a multiple of 128."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.scale_ue8m0 = scale_ue8m0

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        # Block scale: one fp32 per (128-row, 128-col) tile of W.
        self.weight_scale = nn.Parameter(
            torch.empty(
                out_features // 128, in_features // 128, dtype=torch.float32
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequant + ``F.linear`` reference. ``x_scale`` is fp32 per-row
        ``(M, K//128)``; weight scale is 128x128-block ``(N//128, K//128)``.
        3D ``x_fp8`` ``(M, H, K // H)`` is flattened to ``(M, K)``. Output bf16.
        """
        x_view = x_fp8.reshape(x_fp8.shape[0], -1)
        x_scale_view = x_scale.reshape(x_scale.shape[0], -1).float()
        x_f32 = _dequant_fp8_blockscale(x_view, x_scale_view)
        w_f32 = _dequant_fp8_blockscale(self.weight, self.weight_scale)
        out_f32 = F.linear(x_f32, w_f32)
        return out_f32.to(torch.bfloat16)

    def auto_grid_dim(self, x_fp8: Any = None) -> GridDim:
        """Grid fixed at ``(num_workers, 1, 1)``: each task strides over output tiles via ``task_metadata.request_id``."""
        from ... import context as _ctx

        pk = _ctx.current_pk()
        return (int(pk.num_workers), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

    def _register(
        self,
        x_fp8: Any,
        x_scale: Any,
        output: Optional[Any],
        num_workers: Optional[int],
        runtime_m_mode: int,
    ) -> Any:
        """Shared compile body — wires inputs and registers ``self._TASK_NAME``."""
        import torch as _torch
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if num_workers is None:
            num_workers = int(pk.num_workers)

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        ws_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

        M = x_fp8.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(M, self.out_features),
                dtype=mi.bfloat16,
                name=f"{self.prefix}fp8_gemm_dense_out",
            )
        elif isinstance(output, _torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}fp8_gemm_dense_out"
            )
        else:
            out_dt = output

        assert x_fp8.num_dims in (2, 3)
        assert w_dt.num_dims == 2
        assert x_scale.num_dims == 2
        assert ws_dt.num_dims == 2
        assert out_dt.num_dims in (2, 3)
        M_dim = x_fp8.dim(0)
        K_dim = (x_fp8.dim(1) if x_fp8.num_dims == 2
                 else x_fp8.dim(1) * x_fp8.dim(2))
        N_dim = w_dt.dim(0)
        assert w_dt.dim(1) == K_dim
        assert out_dt.dim(0) == M_dim
        out_flat_n = (out_dt.dim(1) if out_dt.num_dims == 2
                      else out_dt.dim(1) * out_dt.dim(2))
        assert out_flat_n == N_dim
        params = [M_dim, N_dim, K_dim, num_workers]
        if runtime_m_mode:
            params.append(runtime_m_mode)
        tb_graph = TBGraph(CyTBGraph((num_workers, 1, 1), (256, 1, 1), 1, 64))
        tb_graph.new_input(x_fp8,    (-1, -1, -1), -1, True)
        tb_graph.new_input(w_dt,     (-1, -1, -1), -1, True)
        tb_graph.new_input(x_scale,  (-1, -1, -1), -1, True)
        tb_graph.new_input(ws_dt,    (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt,   (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [x_fp8, w_dt, x_scale, ws_dt, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, self._TASK_NAME, params)
        return out_dt

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class FP8GEMMDenseSmallM(_FP8GEMMDenseBase):
    """Dense FP8 GEMM — smallm variant (decode-shaped M; kernel NE=2).

    Args: see :class:`_FP8GEMMDenseBase`.
    """

    _TASK_NAME = "fp8_gemm_dense_smallm_sm100"

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        output: Optional[Any] = None,
        num_workers: Optional[int] = None,
        runtime_m_mode: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register ``fp8_gemm_dense_smallm_sm100`` (NE=2, decode-shaped M).

        Tensor contract:
          x_fp8:        (M, K) or (M, H, K//H) fp8_e4m3 viewed as uint8, row-major. A operand.
          x_scale:      (M, K//128) fp32 per-row block-scale, row-major (sa[mi*nk + ki]).
          weight:       (N, K) fp8_e4m3 as uint8, row-major (owned by self). B operand.
          weight_scale: (N//128, K//128) fp32 block-scale, row-major (sb[(on/128)*nk + ki]).
          output:       (M, N) bf16, row-major. None=alloc, Tensor=host-bind, else use as-is.

        Notes: M/N/K mult of 128; TMA-aligned; grid (num_workers,1,1) — tasks stride via task_metadata.request_id.
        params=[M, N, K, num_workers, (runtime_m_mode)].
        """
        return self._register(x_fp8, x_scale, output, num_workers, runtime_m_mode)


class FP8GEMMDenseMediumM(_FP8GEMMDenseBase):
    """Dense FP8 GEMM — mediumm variant (prefill-shaped M; kernel NE=4).

    Args: see :class:`_FP8GEMMDenseBase`.
    """

    _TASK_NAME = "fp8_gemm_dense_mediumm_sm100"

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        output: Optional[Any] = None,
        num_workers: Optional[int] = None,
        runtime_m_mode: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register ``fp8_gemm_dense_mediumm_sm100`` (NE=4, prefill-shaped M).

        Tensor contract:
          x_fp8:        (M, K) or (M, H, K//H) fp8_e4m3 viewed as uint8, row-major. A operand.
          x_scale:      (M, K//128) fp32 per-row block-scale, row-major (sa[mi*nk + ki]).
          weight:       (N, K) fp8_e4m3 as uint8, row-major (owned by self). B operand.
          weight_scale: (N//128, K//128) fp32 block-scale, row-major (sb[(on/128)*nk + ki]).
          output:       (M, N) bf16, row-major. None=alloc, Tensor=host-bind, else use as-is.

        Notes: M/N/K mult of 128; TMA-aligned; grid (num_workers,1,1) — tasks stride via task_metadata.request_id.
        params=[M, N, K, num_workers, (runtime_m_mode)].
        """
        return self._register(x_fp8, x_scale, output, num_workers, runtime_m_mode)
