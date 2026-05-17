"""Dense FP8 GEMM (smallm / mediumm variants) on SM100.

Wraps :meth:`PersistentKernel.fp8_gemm_dense_smallm_layer` (task
``fp8_gemm_dense_smallm_sm100``) and
:meth:`...fp8_gemm_dense_mediumm_layer` (task
``fp8_gemm_dense_mediumm_sm100``) as a single :class:`FP8GEMMDense`
class with a ``variant`` kwarg.

Variant dispatch
----------------

+------------+--------------------------------------+
| ``variant`` | task name                            |
+============+======================================+
| ``"smallm"`` | ``fp8_gemm_dense_smallm_sm100``     |
| ``"mediumm"``| ``fp8_gemm_dense_mediumm_sm100``    |
+------------+--------------------------------------+

Used by the DSv3 builder for the post-attention down/o projections —
``smallm`` for small ``M`` (decode), ``mediumm`` for prefill-shaped M.
The pk method tiles output across ``num_workers`` persistent tasks via
``task_metadata.request_id`` — no grid/block exposed to the caller
(grid is ``(num_workers, 1, 1)`` and block is ``(256, 1, 1)``, both
fixed inside the pk method).

Layout
------

* ``input_fp8``    : ``(M, K)`` or ``(M, H, K // H)`` (3D allowed when
                     the caller wants to keep a head dimension; the
                     GEMM kernel sees the buffer as flat).
* ``input_scale``  : ``(M, K // 128)`` **float32**, row-major.
* ``weight_fp8``   : ``(N, K)`` E4M3.
* ``weight_scale`` : ``(N // 128, K // 128)`` **float32**, row-major
                     — the kernel reads it via raw LDG as
                     ``sb[(on / 128) * num_sf_k + ki]`` (see
                     ``fp8_gemm_dense_sm100_common.cuh``). Same shape
                     and dtype as the HF ``weight_scale_inv`` block
                     scale that DSv3 builder's ``_attach_raw_fp8_weight``
                     attaches.
* ``output``       : ``(M, N)`` or ``(M, H, N // H)`` bf16.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule


__all__ = ["FP8GEMMDense"]


def _dequant_fp8_blockscale(
    fp8_bytes: torch.Tensor,
    block_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize FP8 with a 128x128 block float32 scale.

    Mirrors the kernel's algebraic semantics: for the dense FP8 GEMM,
    ``sb[on/128, ki]`` is multiplied into A^T*B with hardware UE8M0
    quantization of the float32 scale at MMA time. Here we apply the
    raw float32 scale per (128-row, 128-col) block.

    Args:
        fp8_bytes: ``(M, K)`` float8_e4m3fn or uint8 raw bytes.
        block_scale: ``(M // 128, K // 128)`` float32 (or per-row M
            ``(M, K // 128)`` for activations).

    Returns:
        ``(M, K)`` float32.
    """
    if fp8_bytes.dtype == torch.float8_e4m3fn:
        fp32 = fp8_bytes.float()
    else:
        fp32 = fp8_bytes.view(torch.float8_e4m3fn).float()
    M, K = fp32.shape
    if block_scale.shape == (M, K // 128):
        # Per-row activation scale.
        scales = block_scale.float().to(fp32.device).repeat_interleave(128, dim=1)
    elif block_scale.shape == ((M + 127) // 128, K // 128):
        # 128x128 block weight scale.
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


Variant = Literal["smallm", "mediumm"]


class FP8GEMMDense(MPKModule):
    """Dense FP8 GEMM.

    Args:
        in_features: K (reduction). Multiple of 128.
        out_features: N (output). Multiple of 128.
        variant: ``"smallm"`` for decode-shaped M (BN=64, NS=8 inside
            the kernel) or ``"mediumm"`` for prefill-shaped M
            (BN=128, NS=6).
        scale_ue8m0: Required True (UE8M0-packed scales).
        prefix: HF state_dict / tensor-name prefix.

    The weight + weight_scale are owned by this module as
    ``nn.Parameter``s (uint8 + uint32 storage). The input + input_scale
    come in as ``compile()`` arguments; the producing op (typically
    :class:`QuantizeFP8`) attaches them.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        variant: Variant = "smallm",
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if variant not in ("smallm", "mediumm"):
            raise ValueError(
                f"FP8GEMMDense.variant must be 'smallm' or 'mediumm'; "
                f"got {variant!r}"
            )
        # The dense FP8 GEMM kernel (`fp8_gemm_dense_smallm/mediumm_sm100`)
        # takes **raw float32 block scales** (`sa`, `sb` are `float const*`
        # in `fp8_gemm_dense_sm100_common.cuh`). It does its own UE8M0
        # quantization at MMA time; the Python side must NOT pre-pack to
        # UE8M0 uint32 like the small-batch linear_fp8 path. The
        # `scale_ue8m0` flag is accepted for API parity but ignored —
        # the storage layout is fixed to fp32 block scale.
        if in_features % 128 != 0:
            raise ValueError(
                f"FP8GEMMDense: in_features={in_features} must be "
                "a multiple of 128."
            )
        if out_features % 128 != 0:
            raise ValueError(
                f"FP8GEMMDense: out_features={out_features} must be "
                "a multiple of 128."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.variant = variant
        self.scale_ue8m0 = scale_ue8m0

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        # Block scale: one fp32 value per (128-row, 128-col) tile of W.
        # Matches the HF checkpoint's `weight_scale_inv` layout — the
        # builder's `_attach_raw_fp8_weight` attaches this verbatim.
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
        """Dequantize and run a plain ``F.linear`` reference.

        Both ``x_scale`` and ``self.weight_scale`` are **float32 block
        scales** (matches the kernel's ``sa``/``sb`` ``float const*``
        inputs). ``x_scale`` is per-row ``(M, K//128)``; weight is
        128x128-block ``(N//128, K//128)``. For 3D ``x_fp8``
        (``(M, H, K // H)``) we flatten to ``(M, K)``; the kernel sees
        the buffer flat anyway. Output is bf16.
        """
        x_view = x_fp8.reshape(x_fp8.shape[0], -1)
        x_scale_view = x_scale.reshape(x_scale.shape[0], -1).float()
        x_f32 = _dequant_fp8_blockscale(x_view, x_scale_view)
        w_f32 = _dequant_fp8_blockscale(self.weight, self.weight_scale)
        out_f32 = F.linear(x_f32, w_f32)
        return out_f32.to(torch.bfloat16)

    def auto_grid_dim(self, x_fp8: Any = None) -> GridDim:
        """Grid is fixed at ``(num_workers, 1, 1)`` by the pk method."""
        from ... import context as _ctx

        pk = _ctx.current_pk()
        return (int(pk.num_workers), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

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
        """Register the appropriate ``fp8_gemm_dense_*_sm100`` task.

        Args:
            x_fp8:   FP8 activations DTensor ``(M, K)`` or
                     ``(M, H, K // H)``.
            x_scale: UE8M0-packed scale DTensor ``(M, packed_K)``.
            output:  ``None``, ``torch.Tensor``, or ``DTensor`` —
                     same routing convention as the rest of the catalog.
            num_workers: ``grid.x`` width (passes through to the pk
                method). Defaults to ``current_pk().num_workers``.
            runtime_m_mode: Passed through to the pk method (0 = static
                M from shape; non-zero selects the runtime-M variant).
            grid_dim / block_dim: Ignored — the pk method fixes both.
        """
        import torch as _torch
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if num_workers is None:
            num_workers = int(pk.num_workers)

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        ws_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

        # Resolve output. We allocate flat (M, N) by default to match
        # how the kernel addresses the buffer.
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

        # Inlined task registration (was pk.fp8_gemm_dense_{smallm,mediumm}_layer
        # via the shared _fp8_gemm_dense_layer_impl). Both variants share
        # everything but the task name.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

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
        task_name = ("fp8_gemm_dense_smallm_sm100"
                     if self.variant == "smallm"
                     else "fp8_gemm_dense_mediumm_sm100")
        pk.kn_graph.register_task(tb_graph, task_name, params)
        return out_dt
