"""Compile-only test for FP8GEMMDense (smallm + mediumm variants).

Validates the Python-side wiring of the catalog module after the fix
to use fp32 block scales (M-block-outer layout matching the kernel's
SFA pointer math). The full runtime exercise is validated by
demo/deepseek_v3/demo_new.py.

Both prior root causes are fixed:
1. `_quantize_fp8_reference` UE8M0 encode now adds +127 exponent bias.
2. `FP8GEMMDense.weight_scale` is now `(out_features//128, in_features//128)
   float32` row-major (matches the kernel's `sb[(on/128)*nk + ki]`
   indexing in `fp8_gemm_dense_sm100_common.cuh`).
"""

import os
import sys

import torch

from mirage.mpk.layers.linear.fp8_gemm_dense import (
    FP8GEMMDenseSmallM, FP8GEMMDenseMediumM,
)


def test_fp8_gemm_dense_smallm_compile_only():
    m = FP8GEMMDenseSmallM(
        in_features=512, out_features=256, scale_ue8m0=False, prefix="dgsm_",
    )
    assert m.weight.shape == (256, 512), f"weight: {m.weight.shape}"
    assert m.weight_scale.shape == (256 // 128, 512 // 128), \
        f"weight_scale: {m.weight_scale.shape}"
    assert m.weight_scale.dtype == torch.float32
    print("PASSED: FP8GEMMDenseSmallM Python-side shapes correct")


def test_fp8_gemm_dense_mediumm_compile_only():
    m = FP8GEMMDenseMediumM(
        in_features=512, out_features=256, scale_ue8m0=False, prefix="dgmm_",
    )
    assert m.weight.shape == (256, 512)
    assert m.weight_scale.shape == (2, 4)
    assert m.weight_scale.dtype == torch.float32
    print("PASSED: FP8GEMMDenseMediumM Python-side shapes correct")


if __name__ == "__main__":
    test_fp8_gemm_dense_smallm_compile_only()
    test_fp8_gemm_dense_mediumm_compile_only()
    print("FP8GEMMDense tests completed (compile-only). Full runtime "
          "exercise validated by demo/deepseek_v3/demo_new.py.")
