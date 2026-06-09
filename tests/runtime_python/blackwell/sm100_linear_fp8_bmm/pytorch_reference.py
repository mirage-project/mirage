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


# ===========================================================================
# Dense (float32-block-scale) BMM reference for
# `linear_fp8_bmm_dense_sm100_layer`.
#
# The dense kernel uses the same per-head matmul math, but with plain float32
# block scales instead of UE8M0:
#   input_scale  [N, H, nk]            (nk = D_in/128) — 1x128-group activation
#                                        scale; per-head row stride = H*nk
#                                        (sa indexed `sa + n*(H*nk) + h*nk + ki`)
#   weight_scale [H, D_out/128, nk]    — 128x128-block weight scale. With
#                                        D_out=128 the middle dim is 1, so the
#                                        whole 128-row head shares one scale per
#                                        128-K group (`sb + 0*nk + ki`).
# The dequant therefore broadcasts each f32 scale over a 128-element K block.
# Confirmed from fp8_gemm_dense_qout_sm100_common.cuh L301-302 and the BMM-dense
# task_register (sa_row_stride=H*nk, sb per-head base = h*(D_out/128)*nk).
# ===========================================================================

_FP8_MAX = 448.0


def quantize_bmm_input_f32(x_bf16: torch.Tensor):
    """Quantize `[N, H, D_in]` BF16 -> FP8 e4m3 + float32 1x128-group scale.

    Each (token, head, 128-K block) gets one f32 scale = abs_max / 448.
    Returns `(fp8 [N, H, D_in], scale [N, H, nk])` with nk = D_in/128.
    """
    assert x_bf16.dim() == 3
    n, h, d = x_bf16.shape
    assert d % 128 == 0
    nk = d // 128
    x_f32 = x_bf16.float()
    fp8 = torch.empty_like(x_f32, dtype=torch.float8_e4m3fn)
    scale = torch.zeros((n, h, nk), dtype=torch.float32, device=x_bf16.device)
    for ni in range(n):
        for hi in range(h):
            for ki in range(nk):
                block = x_f32[ni, hi, ki * 128:(ki + 1) * 128]
                abs_max = block.abs().max().item()
                s = abs_max / _FP8_MAX if abs_max > 0 else 1.0
                scale[ni, hi, ki] = s
                fp8[ni, hi, ki * 128:(ki + 1) * 128] = (
                    (block / s).clamp(-_FP8_MAX, _FP8_MAX).to(
                        torch.float8_e4m3fn))
    return fp8.contiguous(), scale.contiguous()


def quantize_bmm_weight_f32(w_bf16: torch.Tensor):
    """Quantize `[H, D_out, D_in]` BF16 -> FP8 e4m3 + float32 128x128-block
    scale `[H, D_out/128, nk]`. Each (head, 128-row block, 128-K block) shares
    one scale = abs_max / 448. Returns `(fp8, scale)`.
    """
    assert w_bf16.dim() == 3
    h, d_out, d_in = w_bf16.shape
    assert d_out % 128 == 0 and d_in % 128 == 0
    nb = d_out // 128
    nk = d_in // 128
    w_f32 = w_bf16.float()
    fp8 = torch.empty_like(w_f32, dtype=torch.float8_e4m3fn)
    scale = torch.zeros((h, nb, nk), dtype=torch.float32, device=w_bf16.device)
    for hi in range(h):
        for bi in range(nb):
            for ki in range(nk):
                block = w_f32[hi, bi * 128:(bi + 1) * 128,
                              ki * 128:(ki + 1) * 128]
                abs_max = block.abs().max().item()
                s = abs_max / _FP8_MAX if abs_max > 0 else 1.0
                scale[hi, bi, ki] = s
                fp8[hi, bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                    (block / s).clamp(-_FP8_MAX, _FP8_MAX).to(
                        torch.float8_e4m3fn))
    return fp8.contiguous(), scale.contiguous()


def dequant_bmm_input_f32(x_fp8: torch.Tensor,
                          x_scale: torch.Tensor) -> torch.Tensor:
    """Dequant `[N, H, D_in]` FP8 with f32 scale `[N, H, nk]` to f32."""
    n, h, d = x_fp8.shape
    nk = x_scale.shape[-1]
    x_f = x_fp8.float()
    out = torch.empty(n, h, d, dtype=torch.float32, device=x_fp8.device)
    for ki in range(nk):
        out[:, :, ki * 128:(ki + 1) * 128] = (
            x_f[:, :, ki * 128:(ki + 1) * 128]
            * x_scale[:, :, ki:ki + 1])
    return out


def dequant_bmm_weight_f32(w_fp8: torch.Tensor,
                           w_scale: torch.Tensor) -> torch.Tensor:
    """Dequant `[H, D_out, D_in]` FP8 with f32 block scale `[H, D_out/128, nk]`
    to f32. Each 128-row block + 128-K block shares one scale."""
    h, d_out, d_in = w_fp8.shape
    nb = w_scale.shape[1]
    nk = w_scale.shape[2]
    w_f = w_fp8.float()
    out = torch.empty(h, d_out, d_in, dtype=torch.float32, device=w_fp8.device)
    for bi in range(nb):
        for ki in range(nk):
            out[:, bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128] = (
                w_f[:, bi * 128:(bi + 1) * 128, ki * 128:(ki + 1) * 128]
                * w_scale[:, bi:bi + 1, ki].unsqueeze(-1))
    return out


def bmm_reference_dense_f32(input_fp8: torch.Tensor, input_scale: torch.Tensor,
                            weight_fp8: torch.Tensor,
                            weight_scale: torch.Tensor) -> torch.Tensor:
    """Per-head dense-scale BMM reference:
        output[n, h, m] = sum_k input_dq[n, h, k] * weight_dq[h, m, k]
    matching the dense kernel's float32-block dequant. Output BF16.
    """
    input_dq = dequant_bmm_input_f32(input_fp8, input_scale)
    weight_dq = dequant_bmm_weight_f32(weight_fp8, weight_scale)
    out = torch.einsum("nhk,hmk->nhm", input_dq, weight_dq)
    return out.to(torch.bfloat16)
