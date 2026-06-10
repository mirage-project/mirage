"""BF16 → FP8 (E4M3) per-128-element-block quantization (Blackwell).

Backed by ``tasks/blackwell/per_token_group_quantize_fp8.cuh`` —
``per_token_group_quantize_fp8_task_impl<..., SCALE_UE8M0, ...>``. Two
catalog modules dispatch on ``SCALE_UE8M0``:

* :class:`QuantizeFP8UE8M0` → task ``quantize_fp8_sm100``; output scale
  is **uint32** with 4 UE8M0 bytes packed per word along the K-block
  axis. Consumed by the FP8 linear / FP8 group GEMM kernels.
* :class:`QuantizeFP8F32Scale` → task ``quantize_fp8_f32scale_sm100``;
  output scale is **fp32**. Consumed by the MoE W13/W2 FP8 kernels.

Both produce M-outermost scale layout; the FP8 group GEMM kernels want
K-outermost, so a scale transpose used to be required in
between (UE8M0 variant only).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

import mirage as mi

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["QuantizeFP8UE8M0", "QuantizeFP8F32Scale", "QuantizeFP8"]


def _quantize_fp8_reference(
    x: torch.Tensor,
    scale_ue8m0: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-128-element-block max-abs FP8 E4M3 quantization (reference).

    Returns ``(fp8_bytes, scale)``. UE8M0 path encodes
    ``s = round(log2(scale)) + 127`` and packs four bytes per ``uint32``;
    f32 path returns plain fp32 scales of shape ``(*, num_groups)``.
    """
    *leading, K = x.shape
    if K % 128 != 0:
        raise ValueError(
            f"QuantizeFP8: trailing dim must be a multiple of 128; got {K}"
        )
    num_groups = K // 128
    grouped = x.float().reshape(*leading, num_groups, 128)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-12)
    scale = amax / 448.0
    x_scaled = grouped / scale.unsqueeze(-1)
    fp8 = x_scaled.to(torch.float8_e4m3fn).view(torch.uint8).reshape(*leading, K)

    if scale_ue8m0:
        log2_scale = (
            torch.log2(scale).round() + 127.0
        ).clamp(0, 255).to(torch.uint8)
        if num_groups % 4 != 0:
            raise ValueError(
                f"QuantizeFP8(scale_ue8m0=True): num_groups={num_groups} "
                "must be a multiple of 4 for UE8M0 packing."
            )
        packed = log2_scale.reshape(*leading, num_groups // 4, 4).contiguous()
        packed_u32 = packed.view(torch.uint32).reshape(*leading, num_groups // 4)
        return fp8, packed_u32
    return fp8, scale.to(torch.float32)


def _fp8_group_tiles(num_groups: int, scale_ue8m0: bool) -> int:
    """Replicate ``pk._fp8_quantize_group_tiles``."""
    if scale_ue8m0:
        for candidate in range(min(16, num_groups), 1, -1):
            if num_groups % candidate == 0:
                groups_per_tile = num_groups // candidate
                if groups_per_tile % 4 == 0:
                    return candidate
        return 1
    return min(4, max(1, num_groups // 8))


class _QuantizeFP8Base(MPKModule):
    """Shared base: ``__init__`` (hidden_size + prefix) and the compile skeleton.

    Subclasses set ``_scale_ue8m0`` and ``_task_name`` and provide the
    scale dtype via :meth:`_scale_dtype` / :meth:`_scale_dims`.
    """

    _scale_ue8m0: bool = False
    _task_name: str = ""

    def __init__(
        self,
        hidden_size: int,
        *,
        active_mode: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size % 128 != 0:
            raise ValueError(
                f"QuantizeFP8: hidden_size={hidden_size} must be a multiple "
                "of 128 (UE8M0 block size)."
            )
        if self._scale_ue8m0 and (hidden_size // 128) % 4 != 0:
            raise ValueError(
                f"QuantizeFP8(scale_ue8m0=True): hidden_size // 128 "
                f"({hidden_size // 128}) must be a multiple of 4 for "
                "UE8M0 packing."
            )
        self.hidden_size = hidden_size
        self.active_mode = active_mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block-wise max-abs FP8 E4M3 — returns ``(fp8_bytes, scale)``."""
        return _quantize_fp8_reference(x, self._scale_ue8m0)

    def auto_grid_dim(self, x: Any) -> GridDim:
        """Recomputed inside :meth:`compile` from input shape +
        ``hidden_size``; the value returned here is a placeholder.
        """
        return (1, 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    # ------- subclass hooks -------
    def _scale_dtype(self):
        raise NotImplementedError

    def _scale_dims(self, row_count: int, hidden: int) -> Tuple[int, int]:
        raise NotImplementedError

    def compile(
        self,
        x: Any,
        *,
        output_fp8: Optional[Any] = None,
        output_scale: Optional[Any] = None,
        hidden_size_override: Optional[int] = None,
        input_stride_override: Optional[int] = None,
        in_offset_elems: int = 0,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[Any, Any]:
        """Register a ``quantize_fp8_sm100`` / ``quantize_fp8_f32scale_sm100`` task.

        Tensor contract:
          x:            (rows, hidden)        bf16, activations to quantize.
          output_fp8:   (rows, hidden)        uint8, packed FP8 E4M3 bits.
          output_scale: UE8M0 path → (rows, hidden // (128 * 4)) uint32 (4 UE8M0
                        bytes packed per word along K-block axis).
                        F32 path  → (rows, hidden // 128)        float32.

        Notes: ``hidden`` must be a multiple of 128 (block size); UE8M0 path
        additionally requires ``hidden // 128 % 4 == 0``. ``hidden_size_override``
        / ``input_stride_override`` / ``in_offset_elems`` enable column-slice
        quantization (QKV-a fused buffer). Grid is recomputed internally.
        """
        import torch as _torch
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        hidden = (
            hidden_size_override if hidden_size_override is not None else self.hidden_size
        )
        row_count = 1
        for axis in range(x.num_dims - 1):
            row_count *= x.dim(axis)
        if output_fp8 is None:
            out_fp8_dt = pk.new_tensor(
                dims=(row_count, hidden),
                dtype=mi.uint8,
                name=f"{self.prefix}fp8",
            )
        elif isinstance(output_fp8, _torch.Tensor):
            out_fp8_dt = pk.attach_input(output_fp8, name=f"{self.prefix}fp8")
        else:
            out_fp8_dt = output_fp8

        if output_scale is None:
            scale_dt = pk.new_tensor(
                dims=self._scale_dims(row_count, hidden),
                dtype=self._scale_dtype(),
                name=f"{self.prefix}fp8_scale",
            )
        elif isinstance(output_scale, _torch.Tensor):
            scale_dt = pk.attach_input(
                output_scale, name=f"{self.prefix}fp8_scale"
            )
        else:
            scale_dt = output_scale

        # Inline task registration (formerly pk.quantize_fp8_layer). The
        # kernel grid is recomputed from input shape + hidden_size; this
        # mirrors the pk method's internal override.
        from ...core import CyTBGraph
        from ...kernel import TBGraph

        legacy_hidden_size = x.dim(x.num_dims - 1)
        row_count_local = 1
        for axis in range(x.num_dims - 1):
            row_count_local *= x.dim(axis)
        slice_override = (
            hidden_size_override is not None
            or input_stride_override is not None
            or in_offset_elems != 0
        )
        hidden_size_resolved = hidden_size_override or legacy_hidden_size
        in_stride = (
            input_stride_override
            if input_stride_override is not None
            else legacy_hidden_size
        )
        num_groups_q = max(1, hidden_size_resolved // 128)
        group_tiles_q = _fp8_group_tiles(num_groups_q, self._scale_ue8m0)
        grid_y_q = min(row_count_local, max(int(pk.num_workers), 1))
        grid_dim_local = (group_tiles_q, grid_y_q, 1)
        if slice_override:
            params = [
                self.active_mode,
                hidden_size_resolved,
                in_stride,
                in_offset_elems,
            ]
        else:
            params = [] if self.active_mode == 0 else [self.active_mode]
        tb_graph = TBGraph(CyTBGraph(grid_dim_local, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_fp8_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(scale_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([x, out_fp8_dt, scale_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, self._task_name, params)
        return out_fp8_dt, scale_dt


class QuantizeFP8UE8M0(_QuantizeFP8Base):
    """BF16 → FP8 E4M3 with UE8M0-packed uint32 scales.

    Output scale dtype: ``uint32`` of shape ``(rows, hidden_size // 128 // 4)``
    (4 UE8M0 bytes packed per word along K). ``hidden_size`` must be a
    multiple of 128 *and* ``hidden_size // 128`` must be a multiple of 4.
    Task name: ``quantize_fp8_sm100``.
    """

    _scale_ue8m0 = True
    _task_name = "quantize_fp8_sm100"

    def _scale_dtype(self):
        return mi.uint32

    def _scale_dims(self, row_count: int, hidden: int) -> Tuple[int, int]:
        packed_k = hidden // 128 // 4
        return (row_count, max(1, packed_k))


class QuantizeFP8F32Scale(_QuantizeFP8Base):
    """BF16 → FP8 E4M3 with plain fp32 scales.

    Output scale dtype: ``float32`` of shape ``(rows, hidden_size // 128)``.
    ``hidden_size`` must be a multiple of 128. Task name:
    ``quantize_fp8_f32scale_sm100``.
    """

    _scale_ue8m0 = False
    _task_name = "quantize_fp8_f32scale_sm100"

    def _scale_dtype(self):
        return mi.float32

    def _scale_dims(self, row_count: int, hidden: int) -> Tuple[int, int]:
        return (row_count, hidden // 128)


def QuantizeFP8(
    hidden_size: int,
    *,
    scale_ue8m0: bool = True,
    active_mode: int = 0,
    prefix: str = "",
):
    """Back-compat factory dispatching on ``scale_ue8m0``.

    New code should instantiate :class:`QuantizeFP8UE8M0` or
    :class:`QuantizeFP8F32Scale` directly.
    """
    cls = QuantizeFP8UE8M0 if scale_ue8m0 else QuantizeFP8F32Scale
    return cls(hidden_size, active_mode=active_mode, prefix=prefix)
