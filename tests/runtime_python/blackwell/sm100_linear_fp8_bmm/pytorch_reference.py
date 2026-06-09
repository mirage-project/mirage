"""PyTorch references for `linear_fp8_bmm_sm100_layer`.

The kernel reuses the SM100 swapAB FP8 GEMM body, so the per-row UE8M0
quantization layout is identical to the swapAB tests'. The only BMM-specific
work here is reshaping the (N, H, D) and (H, D_out, D_in) tensors to the
2D flat views the row-wise quantizer expects, and computing the per-head
reference matmul with `torch.einsum`.
"""

import os
import sys
import torch

# Pull in the canonical UE8M0 packing/dequant helpers used by every
# SM100 FP8 test.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
from sm100_fp8_scale_layout import (  # noqa: E402
    quantize_to_fp8_packed_ue8m0,
    dequant_from_packed_ue8m0,
)


def quantize_bmm_input(x_bf16: torch.Tensor):
    """Quantize a `[N, H, D_in]` BF16 tensor to FP8 + UE8M0 packed scales.

    Each (token, head) row gets its own per-128-K block scale, so flatten
    the leading two dims, quantize row-wise, and reshape back. Returns
    `(fp8 [N, H, D_in], packed_scale [N, H, packed_K])`.
    """
    assert x_bf16.dim() == 3
    n, h, d = x_bf16.shape
    flat = x_bf16.reshape(n * h, d).contiguous()
    flat_fp8, flat_scale = quantize_to_fp8_packed_ue8m0(flat)
    packed_k = flat_scale.shape[1]
    return (
        flat_fp8.reshape(n, h, d).contiguous(),
        flat_scale.reshape(n, h, packed_k).contiguous(),
    )


def quantize_bmm_weight(w_bf16: torch.Tensor):
    """Quantize a `[H, D_out, D_in]` BF16 weight to FP8 + UE8M0 packed
    scales. Flatten the (H, D_out) outer dims, quantize row-wise, reshape.
    Returns `(fp8 [H, D_out, D_in], packed_scale [H, D_out, packed_K])`.
    """
    assert w_bf16.dim() == 3
    h, d_out, d_in = w_bf16.shape
    flat = w_bf16.reshape(h * d_out, d_in).contiguous()
    flat_fp8, flat_scale = quantize_to_fp8_packed_ue8m0(flat)
    packed_k = flat_scale.shape[1]
    return (
        flat_fp8.reshape(h, d_out, d_in).contiguous(),
        flat_scale.reshape(h, d_out, packed_k).contiguous(),
    )


def dequant_bmm_input(x_fp8: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize `[N, H, D_in]` FP8 (with `[N, H, packed_K]` scale) back to
    FP32. Same flatten/reshape sandwich as the quantizer."""
    n, h, d = x_fp8.shape
    flat_fp32 = dequant_from_packed_ue8m0(
        x_fp8.reshape(n * h, d).contiguous(),
        x_scale.reshape(n * h, x_scale.shape[-1]).contiguous(),
    )
    return flat_fp32.reshape(n, h, d)


def dequant_bmm_weight(w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    h, d_out, d_in = w_fp8.shape
    flat_fp32 = dequant_from_packed_ue8m0(
        w_fp8.reshape(h * d_out, d_in).contiguous(),
        w_scale.reshape(h * d_out, w_scale.shape[-1]).contiguous(),
    )
    return flat_fp32.reshape(h, d_out, d_in)


def bmm_reference_from_dequant(input_fp32: torch.Tensor,
                               weight_fp32: torch.Tensor) -> torch.Tensor:
    """Per-head matmul reference matching what the kernel actually
    computes once both operands are dequantized:
        output[n, h, m] = sum_k input[n, h, k] * weight[h, m, k]
    Output dtype is BF16 to match the kernel's epilogue.
    """
    out = torch.einsum("nhk,hmk->nhm", input_fp32, weight_fp32)
    return out.to(torch.bfloat16)
