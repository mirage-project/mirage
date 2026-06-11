"""Compile-only test for FP8GroupGEMM (smallm + largem variants).

Validates the Python-side wiring of the catalog after the weight_scale
shape fix: `(num_sf_k = ceil(packed_K/4), E*N) uint32` matching the
kernel's TMA descriptor (`tma.cuh:1721` param_id=3) for the
K-block-outer UE8M0 layout (4 UE8M0 bytes packed per uint32 along K).

Full runtime exercise validated by demo/deepseek_v3/demo_new.py
(MoE path).
"""

import torch

from mirage.mpk.layers.linear.fp8_group_gemm import (
    FP8GroupGEMMSmallM, FP8GroupGEMMLargeM,
)


def test_fp8_group_gemm_smallm_compile_only():
    num_experts = 4
    in_features = 512
    out_features = 256
    m = FP8GroupGEMMSmallM(
        in_features=in_features, out_features=out_features,
        num_experts=num_experts, scale_ue8m0=True, prefix="ggsm_",
    )
    # weight shape: (E, N, K)
    assert m.weight.shape == (num_experts, out_features, in_features), \
        f"weight: {m.weight.shape}"
    # weight_scale shape: (num_sf_k = ceil(packed_K / 4), E*N) uint32.
    # packed_K = K // 128 = 4, num_sf_k = ceil(4/4) = 1.
    packed_k = in_features // 128
    num_sf_k = (packed_k + 3) // 4
    expected_scale_shape = (num_sf_k, num_experts * out_features)
    assert m.weight_scale.shape == expected_scale_shape, \
        f"weight_scale: {m.weight_scale.shape} (expected {expected_scale_shape})"
    assert m.weight_scale.dtype == torch.uint32
    print("PASSED: FP8GroupGEMM(smallm) Python-side shapes correct")


def test_fp8_group_gemm_largem_compile_only():
    num_experts = 4
    in_features = 512
    out_features = 256
    m = FP8GroupGEMMLargeM(
        in_features=in_features, out_features=out_features,
        num_experts=num_experts, scale_ue8m0=True, prefix="gglm_",
    )
    assert m.weight.shape == (num_experts, out_features, in_features)
    packed_k = in_features // 128
    num_sf_k = (packed_k + 3) // 4
    assert m.weight_scale.shape == (num_sf_k, num_experts * out_features)
    print("PASSED: FP8GroupGEMM(largem) Python-side shapes correct")


if __name__ == "__main__":
    test_fp8_group_gemm_smallm_compile_only()
    test_fp8_group_gemm_largem_compile_only()
    print("FP8GroupGEMM tests completed (compile-only). Full runtime "
          "exercise validated by demo/deepseek_v3/demo_new.py.")
