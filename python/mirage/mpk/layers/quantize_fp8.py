"""BF16 → FP8 (E4M3) quantization with block-wise scales.

Wraps :meth:`PersistentKernel.quantize_fp8_layer`. The task name
varies with ``scale_ue8m0``:

* ``scale_ue8m0=True``  → ``quantize_fp8_sm100`` (UE8M0 uint32 scales,
  consumed by the FP8 linear and FP8 group GEMM kernels).
* ``scale_ue8m0=False`` → ``quantize_fp8_f32scale_sm100`` (plain fp32
  scales, consumed by the MoE W13/W2 FP8 kernels).

The kernel quantizes per row across the trailing axis: each
128-element K-block produces one scale. The output scale buffer is
**M-outermost** (logical ``(rows, packed_K)`` for UE8M0 or
``(rows, num_groups)`` for fp32) — the FP8 group GEMM kernels expect a
K-outermost layout, so a :class:`TransposeScale` insertion is
required between this op and ``fp8_group_gemm_*``.

Forward reference
-----------------

For both modes the reference rounds-to-zero-style maps bf16 → fp8 E4M3
using the per-block max-abs scaling scheme the kernel implements:

    scale[b] = max(|x[b]|) / 448.0    # 448 = max representable in E4M3
    fp8[b]   = round_to_e4m3(x[b] / scale[b])

For ``scale_ue8m0=True`` the scale is encoded as a UE8M0 byte
(``s = round(log2(scale)) + 127``) and four bytes are packed per
``uint32`` along the K-block axis. For ``scale_ue8m0=False`` the scale
is plain fp32. The reference is exact up to the rounding mode; the
test driver compares with a tolerance.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

import mirage as mi

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["QuantizeFP8"]


def _quantize_fp8_reference(
    x: torch.Tensor,
    scale_ue8m0: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-128-element-block max-abs FP8 E4M3 quantization.

    Returns ``(fp8_bytes, scale)``:

    * ``fp8_bytes`` is a ``uint8`` tensor with the same shape as ``x``.
      The bytes encode FP8 E4M3 via ``torch.float8_e4m3fn``'s cast.
    * For ``scale_ue8m0=True`` returns ``uint32`` of shape
      ``(*, packed_K)`` (4 logical scales / uint32 along K).
      For ``scale_ue8m0=False`` returns ``float32`` of shape
      ``(*, num_groups)``.
    """
    *leading, K = x.shape
    if K % 128 != 0:
        raise ValueError(
            f"QuantizeFP8: trailing dim must be a multiple of 128; got {K}"
        )
    num_groups = K // 128
    grouped = x.float().reshape(*leading, num_groups, 128)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-12)  # (*, num_groups)
    scale = amax / 448.0
    # Dequant input back through the chosen scale → fp8 cast.
    x_scaled = grouped / scale.unsqueeze(-1)
    fp8 = x_scaled.to(torch.float8_e4m3fn).view(torch.uint8).reshape(*leading, K)

    if scale_ue8m0:
        # Encode each fp32 scale as UE8M0 byte: s = round(log2(scale)) + 127
        # (IEEE-style 8-bit exponent with bias 127). Mirrors the kernel-side
        # encode_ue8m0 in include/mirage/persistent_kernel/utils/fp8_quant.cuh.
        log2_scale = (torch.log2(scale).round() + 127.0).clamp(0, 255).to(torch.uint8)
        # Pack 4 bytes per uint32 along the K-block axis.
        if num_groups % 4 != 0:
            raise ValueError(
                f"QuantizeFP8(scale_ue8m0=True): num_groups={num_groups} "
                "must be a multiple of 4 for UE8M0 packing."
            )
        packed = log2_scale.reshape(*leading, num_groups // 4, 4).contiguous()
        # Interpret the last-4 axis as a single uint32.
        packed_u32 = packed.view(torch.uint32).reshape(*leading, num_groups // 4)
        return fp8, packed_u32
    return fp8, scale.to(torch.float32)


class QuantizeFP8(MPKModule):
    """BF16 → FP8 E4M3 with per-128-element-block scales.

    Args:
        hidden_size: Trailing dim of the input (the K axis quantized
            in 128-element blocks). Used to size the output scale
            buffer. Must be a multiple of 128 (kernel block size).
        scale_ue8m0: ``True`` (default) for the UE8M0-packed uint32
            scale used by FP8 linear / group GEMM. ``False`` for the
            plain fp32 scale used by MoE W13/W2.
        active_mode: Passes through to the pk method as the first
            param. ``0`` = standard quantize. Non-zero selects
            specialized variants (gate-active masking) used by
            DSv3's MoE-fused decode.
        prefix: Reserved. No parameters live here.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        scale_ue8m0: bool = True,
        active_mode: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size % 128 != 0:
            raise ValueError(
                f"QuantizeFP8: hidden_size={hidden_size} must be a multiple "
                "of 128 (UE8M0 block size)."
            )
        if scale_ue8m0 and (hidden_size // 128) % 4 != 0:
            # UE8M0 packs 4 scales / uint32. The pk method has a
            # group_tiles heuristic that requires groups_per_tile % 4
            # == 0 — checking up front gives a better error message.
            raise ValueError(
                f"QuantizeFP8(scale_ue8m0=True): hidden_size // 128 "
                f"({hidden_size // 128}) must be a multiple of 4 for "
                "UE8M0 packing."
            )
        self.hidden_size = hidden_size
        self.scale_ue8m0 = scale_ue8m0
        self.active_mode = active_mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Block-wise max-abs FP8 E4M3 quantization.

        Returns ``(fp8_bytes, scale)`` — see module docstring.
        """
        return _quantize_fp8_reference(x, self.scale_ue8m0)

    def auto_grid_dim(self, x: Any) -> GridDim:
        """The pk method overrides ``grid_dim`` internally (it computes
        ``(group_tiles, grid_y, 1)`` from input shape + hidden_size).
        We return a placeholder ``(1, 1, 1)`` — the pk method ignores
        whatever we pass and recomputes.
        """
        return (1, 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

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
        """Register a ``quantize_fp8`` task.

        Args:
            x: Input DTensor ``(*, hidden_size)`` bf16.
            output_fp8: Optional output buffer for the FP8 bytes.
                ``None`` allocates via ``pk.new_tensor`` with dtype
                ``mi.uint8``. Caller-routed via the standard
                ``None / torch.Tensor / DTensor`` convention.
            output_scale: Same convention. Dtype is ``mi.uint32`` for
                UE8M0 mode, ``mi.float32`` otherwise.
            hidden_size_override / input_stride_override /
                in_offset_elems: Pass-through to support quantizing a
                column slice of a wider buffer (QKV-a fused path).
                Defaults preserve the legacy whole-row quantize.
            grid_dim: The pk method recomputes grid internally; this
                argument is forwarded for API symmetry but the pk
                method overrides it.
            block_dim: Override; defaults to ``(128, 1, 1)``.

        Returns:
            ``(output_fp8_dt, output_scale_dt)``.
        """
        import torch as _torch
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Resolve output_fp8.
        hidden = hidden_size_override if hidden_size_override is not None else self.hidden_size
        # row_count is the product of all leading dims.
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
            if self.scale_ue8m0:
                packed_k = hidden // 128 // 4  # uint32 columns
                scale_dt = pk.new_tensor(
                    dims=(row_count, max(1, packed_k)),
                    dtype=mi.uint32,
                    name=f"{self.prefix}fp8_scale",
                )
            else:
                num_groups = hidden // 128
                scale_dt = pk.new_tensor(
                    dims=(row_count, num_groups),
                    dtype=mi.float32,
                    name=f"{self.prefix}fp8_scale",
                )
        elif isinstance(output_scale, _torch.Tensor):
            scale_dt = pk.attach_input(
                output_scale, name=f"{self.prefix}fp8_scale"
            )
        else:
            scale_dt = output_scale

        # Inlined task registration (was pk.quantize_fp8_layer). The pk
        # method recomputes grid_dim internally — replicate that here.
        from ...core import CyTBGraph
        from ...kernel import TBGraph

        legacy_hidden_size = x.dim(x.num_dims - 1)
        row_count_local = 1
        for axis in range(x.num_dims - 1):
            row_count_local *= x.dim(axis)
        slice_override = (hidden_size_override is not None or
                          input_stride_override is not None or
                          in_offset_elems != 0)
        hidden_size_resolved = hidden_size_override or legacy_hidden_size
        in_stride = (input_stride_override if input_stride_override is not None
                     else legacy_hidden_size)
        # Mirror pk._fp8_quantize_group_tiles.
        num_groups_q = max(1, hidden_size_resolved // 128)
        if self.scale_ue8m0:
            # Packed UE8M0 stores 4 group scales per uint32; split only at
            # four-group boundaries so each CTA owns whole packed scale words.
            group_tiles_q = 1
            for candidate in range(min(16, num_groups_q), 1, -1):
                if num_groups_q % candidate == 0:
                    groups_per_tile = num_groups_q // candidate
                    if groups_per_tile % 4 == 0:
                        group_tiles_q = candidate
                        break
        else:
            # Float-scale MoE quantization has no packing hazard.
            group_tiles_q = min(4, max(1, num_groups_q // 8))
        # Cap grid.y at num_workers (single wave on persistent runtime).
        grid_y_q = min(row_count_local, max(int(pk.num_workers), 1))
        grid_dim_local = (group_tiles_q, grid_y_q, 1)
        if slice_override:
            params = [self.active_mode, hidden_size_resolved,
                      in_stride, in_offset_elems]
        else:
            params = [] if self.active_mode == 0 else [self.active_mode]
        tb_graph = TBGraph(CyTBGraph(grid_dim_local, block_dim, 1, 64))
        tb_graph.new_input(x,           (-1, -1, -1), -1, True)
        tb_graph.new_input(out_fp8_dt,  (-1, -1, -1), -1, True)
        tb_graph.new_input(scale_dt,    (-1, -1, -1), -1, True)
        pk.kn_graph.customized([x, out_fp8_dt, scale_dt], tb_graph)
        task_name = ("quantize_fp8_sm100" if self.scale_ue8m0
                     else "quantize_fp8_f32scale_sm100")
        pk.kn_graph.register_task(tb_graph, task_name, params)
        return out_fp8_dt, scale_dt
