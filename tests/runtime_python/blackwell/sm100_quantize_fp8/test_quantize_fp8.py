import os
import sys

import torch
import runtime_kernel_blackwell_quantize_fp8 as runtime_kernel_blackwell

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (
    BLOCK_K,
    allocate_packed_ue8m0_scale,
    quantize_to_fp8_deepgemm_style,
)

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

batch_sizes = [1, 2, 4, 8, 16]
hidden_sizes = [int(x) for x in runtime_kernel_blackwell.supported_hidden_sizes()]
group_sizes = [int(x) for x in runtime_kernel_blackwell.supported_group_sizes()]
block_k = BLOCK_K

assert group_sizes == [block_k]


for batch_size in batch_sizes:
    for hidden_size in hidden_sizes:
        print(
            f"\n=== Testing batch_size={batch_size} hidden_size={hidden_size} block_k={block_k} ==="
        )

        x = torch.randn(
            (batch_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=g,
        )
        output = torch.empty(
            (batch_size, hidden_size), device="cuda", dtype=torch.float8_e4m3fn
        )
        scales = allocate_packed_ue8m0_scale(batch_size, hidden_size, x.device)

        runtime_kernel_blackwell.quantize_fp8_sm100(x, output, scales, group_size=block_k)

        quant_ref, scale_ref = quantize_to_fp8_deepgemm_style(x)

        assert scales.shape == scale_ref.shape
        assert scales.stride() == scale_ref.stride()
        torch.testing.assert_close(scales, scale_ref, rtol=0, atol=0)
        torch.testing.assert_close(
            output.float(),
            quant_ref.float(),
            rtol=1e-1,
            atol=16.0,
        )
        print("Random-input test passed!")

        zero_x = torch.zeros_like(x)
        zero_output = torch.empty_like(output)
        zero_scales = allocate_packed_ue8m0_scale(batch_size, hidden_size, x.device)
        runtime_kernel_blackwell.quantize_fp8_sm100(
            zero_x, zero_output, zero_scales, group_size=block_k
        )
        zero_quant_ref, zero_scale_ref = quantize_to_fp8_deepgemm_style(zero_x)

        assert zero_scales.shape == zero_scale_ref.shape
        assert zero_scales.stride() == zero_scale_ref.stride()
        torch.testing.assert_close(zero_scales, zero_scale_ref, rtol=0, atol=0)
        torch.testing.assert_close(
            zero_output.float(),
            zero_quant_ref.float(),
            rtol=1e-1,
            atol=16.0,
        )
        print("Zero-input bring-up passed!")


print("\n=== Negative tests ===")

unsupported_hidden_size = 640
unsupported_output = torch.empty(
    (1, unsupported_hidden_size), device="cuda", dtype=torch.float8_e4m3fn
)
unsupported_scales = allocate_packed_ue8m0_scale(1, unsupported_hidden_size, "cuda")
unsupported_x = torch.randn(
    (1, unsupported_hidden_size), device="cuda", dtype=torch.bfloat16, generator=g
)
try:
    runtime_kernel_blackwell.quantize_fp8_sm100(
        unsupported_x, unsupported_output, unsupported_scales, group_size=block_k
    )
    raise AssertionError("Expected unsupported hidden_size failure")
except RuntimeError as exc:
    assert "Unsupported hidden_size" in str(exc)
    print("Unsupported hidden_size negative test passed!")

bad_group_size = 64
bad_group_output = torch.empty((1, block_k), device="cuda", dtype=torch.float8_e4m3fn)
bad_group_scales = allocate_packed_ue8m0_scale(1, block_k, "cuda")
bad_group_x = torch.randn((1, block_k), device="cuda", dtype=torch.bfloat16, generator=g)
try:
    runtime_kernel_blackwell.quantize_fp8_sm100(
        bad_group_x, bad_group_output, bad_group_scales, group_size=bad_group_size
    )
    raise AssertionError("Expected unsupported group_size failure")
except RuntimeError as exc:
    assert "Unsupported group_size" in str(exc)
    print("Unsupported group_size negative test passed!")
