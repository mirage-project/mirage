import os
import sys

import torch
import runtime_kernel_blackwell_linear_fp8 as runtime_kernel_blackwell

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (
    BLOCK_K,
    allocate_packed_ue8m0_scale,
    dequant_from_packed_ue8m0_deepgemm_style,
    quantize_to_fp8_deepgemm_style,
)

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

supported_shapes = [
    tuple(int(dim) for dim in shape)
    for shape in runtime_kernel_blackwell.supported_dense_gemm_shapes()
]
benchmark_shape = (1, 128, 768)


for batch_size, output_size, reduction_size in supported_shapes:
    for has_residual in (False, True):
        print(
            f"\n=== Testing batch_size={batch_size} output_size={output_size} reduction_size={reduction_size} has_residual={has_residual} ==="
        )

        x_bf16 = torch.randn(
            (batch_size, reduction_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=g,
        )
        w_bf16 = torch.randn(
            (output_size, reduction_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=g,
        )

        x_q, x_scale = quantize_to_fp8_deepgemm_style(x_bf16)
        w_q, w_scale = quantize_to_fp8_deepgemm_style(w_bf16)

        residual = torch.randn(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16, generator=g
        )
        if not has_residual:
            residual = None

        output = torch.empty(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16
        )

        runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
            x_q, x_scale, w_q, w_scale, residual, output
        )

        x_ref = dequant_from_packed_ue8m0_deepgemm_style(x_q, x_scale)
        w_ref = dequant_from_packed_ue8m0_deepgemm_style(w_q, w_scale)
        torch_out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1))
        if has_residual:
            torch_out = torch_out + residual.float()
        torch_out = torch_out.to(torch.bfloat16)

        torch.testing.assert_close(output, torch_out, rtol=1e-2, atol=1e-2)
        print("Random-input test passed!")

        zero_x_bf16 = torch.zeros_like(x_bf16)
        zero_x_q, zero_x_scale = quantize_to_fp8_deepgemm_style(zero_x_bf16)
        zero_output = torch.empty_like(output)

        runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
            zero_x_q, zero_x_scale, w_q, w_scale, residual, zero_output
        )

        zero_x_ref = dequant_from_packed_ue8m0_deepgemm_style(
            zero_x_q, zero_x_scale
        )
        zero_torch_out = torch.matmul(zero_x_ref, torch.transpose(w_ref, 0, 1))
        if has_residual:
            zero_torch_out = zero_torch_out + residual.float()
        zero_torch_out = zero_torch_out.to(torch.bfloat16)

        torch.testing.assert_close(zero_output, zero_torch_out, rtol=1e-2, atol=1e-2)
        print("Zero-input bring-up passed!")

        if (batch_size, output_size, reduction_size) == benchmark_shape and not has_residual:
            for _ in range(16):
                runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
                    x_q, x_scale, w_q, w_scale, residual, output
                )

            torch.cuda.synchronize()
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            repetitions = 1000
            starter.record()
            for _ in range(repetitions):
                runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
                    x_q, x_scale, w_q, w_scale, residual, output
                )
            ender.record()
            torch.cuda.synchronize()
            total_time = starter.elapsed_time(ender)
            avg_time = total_time / repetitions
            print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")


print("\n=== Negative tests ===")

unsupported_k = 640
x_bad_k = torch.randn((1, unsupported_k), device="cuda", dtype=torch.bfloat16, generator=g)
w_bad_k = torch.randn((128, unsupported_k), device="cuda", dtype=torch.bfloat16, generator=g)
x_bad_k_q = torch.empty_like(x_bad_k, dtype=torch.float8_e4m3fn)
w_bad_k_q = torch.empty_like(w_bad_k, dtype=torch.float8_e4m3fn)
x_bad_k_scale = allocate_packed_ue8m0_scale(1, unsupported_k, x_bad_k.device)
w_bad_k_scale = allocate_packed_ue8m0_scale(128, unsupported_k, w_bad_k.device)
bad_k_output = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)
try:
    runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
        x_bad_k_q, x_bad_k_scale, w_bad_k_q, w_bad_k_scale, None, bad_k_output
    )
    raise AssertionError("Expected unsupported K failure")
except RuntimeError as exc:
    assert "Unsupported linear_fp8_1d2d_sm100 shape" in str(exc)
    print("Unsupported K negative test passed!")

unsupported_n = 256
x_bad_n = torch.randn((1, 768), device="cuda", dtype=torch.bfloat16, generator=g)
w_bad_n = torch.randn((unsupported_n, 768), device="cuda", dtype=torch.bfloat16, generator=g)
x_bad_n_q, x_bad_n_scale = quantize_to_fp8_deepgemm_style(x_bad_n)
w_bad_n_q, w_bad_n_scale = quantize_to_fp8_deepgemm_style(w_bad_n)
bad_n_output = torch.empty((1, unsupported_n), device="cuda", dtype=torch.bfloat16)
try:
    runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
        x_bad_n_q, x_bad_n_scale, w_bad_n_q, w_bad_n_scale, None, bad_n_output
    )
    raise AssertionError("Expected unsupported N failure")
except RuntimeError as exc:
    assert "Unsupported linear_fp8_1d2d_sm100 shape" in str(exc)
    print("Unsupported N negative test passed!")

unsupported_b = 3
x_bad_b = torch.randn((unsupported_b, 768), device="cuda", dtype=torch.bfloat16, generator=g)
w_bad_b = torch.randn((128, 768), device="cuda", dtype=torch.bfloat16, generator=g)
x_bad_b_q, x_bad_b_scale = quantize_to_fp8_deepgemm_style(x_bad_b)
w_bad_b_q, w_bad_b_scale = quantize_to_fp8_deepgemm_style(w_bad_b)
bad_b_output = torch.empty((unsupported_b, 128), device="cuda", dtype=torch.bfloat16)
try:
    runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
        x_bad_b_q, x_bad_b_scale, w_bad_b_q, w_bad_b_scale, None, bad_b_output
    )
    raise AssertionError("Expected unsupported B failure")
except RuntimeError as exc:
    assert "Unsupported linear_fp8_1d2d_sm100 shape" in str(exc)
    print("Unsupported B negative test passed!")

unsupported_large_b = 32
x_bad_large_b = torch.randn(
    (unsupported_large_b, 768), device="cuda", dtype=torch.bfloat16, generator=g
)
w_bad_large_b = torch.randn(
    (128, 768), device="cuda", dtype=torch.bfloat16, generator=g
)
x_bad_large_b_q, x_bad_large_b_scale = quantize_to_fp8_deepgemm_style(
    x_bad_large_b
)
w_bad_large_b_q, w_bad_large_b_scale = quantize_to_fp8_deepgemm_style(
    w_bad_large_b
)
bad_large_b_output = torch.empty(
    (unsupported_large_b, 128), device="cuda", dtype=torch.bfloat16
)
try:
    runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
        x_bad_large_b_q,
        x_bad_large_b_scale,
        w_bad_large_b_q,
        w_bad_large_b_scale,
        None,
        bad_large_b_output,
    )
    raise AssertionError("Expected unsupported large B failure")
except RuntimeError as exc:
    assert "Unsupported linear_fp8_1d2d_sm100 shape" in str(exc)
    print("Unsupported large B negative test passed!")

x_valid = torch.randn((1, 768), device="cuda", dtype=torch.bfloat16, generator=g)
w_valid = torch.randn((128, 768), device="cuda", dtype=torch.bfloat16, generator=g)
x_valid_q, x_valid_scale = quantize_to_fp8_deepgemm_style(x_valid)
w_valid_q, w_valid_scale = quantize_to_fp8_deepgemm_style(w_valid)
bad_scale_output = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)
bad_input_scale = x_valid_scale.clone()
bad_input_scale = bad_input_scale.contiguous()
try:
    runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
        x_valid_q,
        bad_input_scale,
        w_valid_q,
        w_valid_scale,
        None,
        bad_scale_output,
    )
    raise AssertionError("Expected input_scale layout mismatch failure")
except RuntimeError as exc:
    assert "input_scale must use DeepGEMM packed UE8M0 layout" in str(exc)
    print("Scale layout mismatch negative test passed!")
