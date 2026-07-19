"""Pure-PyTorch reference for block-wise BF16 -> FP8 quantization.

Mirrors what the SM100 ``quantize_fp8_sm100`` task computes. Two scale layouts
are supported:

  * ``layout="row_major"``  -> packed UE8M0 scales of shape
    ``(outer_dim, packed_k)`` with row-major strides; matches
    :func:`allocate_packed_ue8m0_scale`.
  * ``layout="deepgemm_col_major"`` -> packed UE8M0 scales of shape
    ``(outer_dim, packed_k)`` but with column-major strides
    (physical layout ``[packed_k, aligned_outer]``); matches
    :func:`allocate_packed_ue8m0_scale_deepgemm_style` and is the layout the
    MPK ``quantize_fp8_layer`` writes to.
"""

import os
import sys

import torch

# The block-wise quantize helpers live in the shared common dir alongside
# the kernel layout utilities. Reuse them rather than duplicate the math.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (  # noqa: E402  (import after sys.path tweak)
    BLOCK_K,
    FP8_MAX,
    ceil_div,
    quantize_to_fp8_deepgemm_style,
    quantize_to_fp8_packed_ue8m0,
)


def _quantize_to_fp8_f32_scale(x_bf16: torch.Tensor):
    """Block-wise BF16->FP8 with a plain float32 per-group scale.

    Mirrors the ``SCALE_UE8M0=false`` branch of
    ``per_token_group_quantize_fp8_task_impl``: per 128-element group,
    ``y_scale = max(|group|, 1e-10) / 448`` (no UE8M0 snapping), and the FP8
    value is ``clamp(orig / y_scale, -448, 448)``. The kernel inits its
    running max to ``eps = 1e-10`` and then floors the reduced max with
    ``1e-10`` again, so the effective floor is ``1e-10`` -- replicated here.
    Scale tensor is ``(outer_dim, num_groups)`` row-major float32, matching
    the kernel's ``output_s[batch * num_groups + group]`` write (the layout
    the MPK MoE-path ``quantize_fp8_layer`` allocates).
    """
    assert x_bf16.dim() == 2
    outer_dim, reduction_size = x_bf16.shape
    assert reduction_size % BLOCK_K == 0
    num_groups = ceil_div(reduction_size, BLOCK_K)

    x_fp32 = x_bf16.float()
    x_q = torch.empty_like(x_fp32, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (outer_dim, num_groups), dtype=torch.float32, device=x_bf16.device
    )
    for outer_idx in range(outer_dim):
        for g in range(num_groups):
            k_start = g * BLOCK_K
            k_end = k_start + BLOCK_K
            block = x_fp32[outer_idx, k_start:k_end]
            group_max = max(block.abs().max().item(), 1e-10)
            y_scale = group_max / FP8_MAX
            x_q[outer_idx, k_start:k_end] = torch.clamp(
                block / y_scale, -FP8_MAX, FP8_MAX
            ).to(torch.float8_e4m3fn)
            scales[outer_idx, g] = y_scale
    return x_q, scales


def quantize_fp8_ref(
    input: torch.Tensor,
    scale_ue8m0: bool = True,
    layout: str = "row_major",
):
    """Reference block-wise BF16->FP8 quantization.

    Args:
        input: ``(outer_dim, reduction_size)`` ``bfloat16`` tensor.
        scale_ue8m0: ``True`` -> packed UE8M0 ``uint32`` scale (FP8 linear /
            dense GEMM path). ``False`` -> plain ``float32`` per-group scale
            of shape ``(outer_dim, reduction_size/128)`` row-major (the MoE
            group-GEMM path). The ``layout`` arg is ignored when
            ``scale_ue8m0=False`` (f32 scale has a single fixed row-major
            layout in the kernel).
        layout: ``"row_major"`` (kernel-test default; matches
            ``allocate_packed_ue8m0_scale``) or ``"deepgemm_col_major"``
            (the layout the MPK ``quantize_fp8_layer`` produces, with
            physical shape ``[packed_k, aligned_outer]``). Only meaningful
            for the UE8M0 path.

    Returns:
        ``(output_fp8, output_scale)`` -- ``output_fp8`` is
        ``float8_e4m3fn`` of the same shape as ``input``; ``output_scale``
        is ``uint32`` packed UE8M0 (UE8M0 path) or ``float32``
        ``(outer_dim, reduction_size/128)`` (f32 path).
    """
    assert input.dtype == torch.bfloat16, "expected bf16 input"

    if not scale_ue8m0:
        return _quantize_to_fp8_f32_scale(input)

    if layout == "row_major":
        return quantize_to_fp8_packed_ue8m0(input)
    if layout == "deepgemm_col_major":
        return quantize_to_fp8_deepgemm_style(input)
    raise ValueError(f"unsupported layout: {layout}")


__all__ = ["quantize_fp8_ref", "BLOCK_K"]
