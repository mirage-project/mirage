"""PyTorch reference implementations for FP8 linear kernels (SM100).

Lifted from ``test_linear_1d2d_fp8.py`` (kernel-test reference at lines 67-72).
The dequantization uses the UE8M0 packed-scale layout from
``../common/sm100_fp8_scale_layout.py``.
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import dequant_from_packed_ue8m0  # noqa: E402


def linear_fp8_ref(input_fp8, input_scale, weight_fp8, weight_scale):
    """Pure-PyTorch reference for ``linear_fp8_layer``.

    Mirrors the random-input check in ``test_linear_1d2d_fp8.py``:
        x_ref = dequant_from_packed_ue8m0(x_q, x_scale)
        w_ref = dequant_from_packed_ue8m0(w_q, w_scale)
        out = (x_ref @ w_ref.T).to(bfloat16)

    Both scale tensors must use the UE8M0 packed layout (row-major or
    deepgemm column-major; both are accepted by
    ``dequant_from_packed_ue8m0``).
    """
    x_ref = dequant_from_packed_ue8m0(input_fp8, input_scale)
    w_ref = dequant_from_packed_ue8m0(weight_fp8, weight_scale)
    out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1))
    return out.to(torch.bfloat16)


def linear_fp8_with_residual_ref(
    input_fp8, input_scale, weight_fp8, weight_scale, residual
):
    """Pure-PyTorch reference for ``linear_fp8_with_residual_layer``.

    Computes ``dequant(input) @ dequant(weight).T + residual``.
    """
    x_ref = dequant_from_packed_ue8m0(input_fp8, input_scale)
    w_ref = dequant_from_packed_ue8m0(weight_fp8, weight_scale)
    out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1)) + residual.float()
    return out.to(torch.bfloat16)
