"""PyTorch reference implementations for the sm100_rmsnorm folder.

Contains:
  * ``rmsnorm_ref``  -- plain RMSNorm (default eps=1e-5, callers pass 1e-6
    for the DeepSeek-V3 shapes; the kernel hard-codes 1e-6f).
  * ``fused_rmsnorm_quantize_fp8_ref`` -- RMSNorm followed by block-wise
    BF16->FP8 quantization, matching the SM100
    ``fused_rmsnorm_quantize_fp8_sm100`` task. Reuses the shared UE8M0
    quantizer from ``blackwell/common`` for the UE8M0 path and a small f32
    block-scale helper (kernel ``SCALE_UE8M0=false`` path) for the f32 path.

The eps default of ``rmsnorm_ref`` is intentionally kept at 1e-5 to preserve
the behavior of the test that originally lived in ``test_mode/``; DSV3 callers
pass eps=1e-6 explicitly.
"""

import os
import sys

import torch

# Reuse the shared block-quantize helpers rather than duplicating the math.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (  # noqa: E402  (import after sys.path tweak)
    BLOCK_K,
    FP8_MAX,
    ceil_div,
    quantize_to_fp8_deepgemm_style,
)


def rmsnorm_ref(x, weight, eps=1e-5):
    """RMSNorm: (x / RMS(x)) * weight."""
    x_f32 = x.to(torch.float32)
    rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * rms * weight.to(torch.float32)).to(x.dtype)


def _quantize_to_fp8_f32_scale(x_bf16: torch.Tensor):
    """Block-wise BF16->FP8 with a plain float32 per-group scale.

    Mirrors the ``SCALE_UE8M0=false`` branch of
    ``fused_rmsnorm_quantize_fp8_impl``: per 128-element group,
    ``y_scale = max(|group|, 1e-10) / 448`` (no UE8M0 snapping), and the FP8
    value is ``clamp(orig / y_scale, -448, 448)``. Scale tensor is
    ``(outer_dim, num_groups)`` row-major float32, matching the kernel's
    ``d_scale[batch * num_groups + group]`` write.
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


def fused_rmsnorm_quantize_fp8_ref(x, weight, scale_ue8m0=True, eps=1e-6):
    """RMSNorm (eps) then block-wise FP8 quantize of the bf16 normalized row.

    Returns ``(out_bf16, out_fp8, out_scale)``:
      * ``out_bf16``  -- the bf16 RMSNorm output (kernel's EMIT_BF16 store).
      * ``out_fp8``   -- ``float8_e4m3fn`` quantized normalized row.
      * ``out_scale`` -- packed UE8M0 ``uint32`` (deepgemm col-major) when
        ``scale_ue8m0=True``, else float32 ``(batch, num_groups)`` row-major.

    The quantizer consumes the *bf16* normalized output (not the f32 one), so
    the reference quantizes ``out_bf16`` to match the kernel, which quantizes
    the same bf16 values it stored to smem.
    """
    out_bf16 = rmsnorm_ref(x, weight, eps=eps)
    if scale_ue8m0:
        out_fp8, out_scale = quantize_to_fp8_deepgemm_style(out_bf16)
    else:
        out_fp8, out_scale = _quantize_to_fp8_f32_scale(out_bf16)
    return out_bf16, out_fp8, out_scale
