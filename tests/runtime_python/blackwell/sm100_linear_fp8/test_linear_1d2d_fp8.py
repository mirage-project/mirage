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
    dequant_from_packed_ue8m0,
    quantize_to_fp8_deepgemm_style,
    quantize_to_fp8_packed_ue8m0,
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

        x_q, x_scale = quantize_to_fp8_packed_ue8m0(x_bf16)
        w_q, w_scale = quantize_to_fp8_packed_ue8m0(w_bf16)

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

        x_ref = dequant_from_packed_ue8m0(x_q, x_scale)
        w_ref = dequant_from_packed_ue8m0(w_q, w_scale)
        torch_out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1))
        if has_residual:
            torch_out = torch_out + residual.float()
        torch_out = torch_out.to(torch.bfloat16)

        torch.testing.assert_close(output, torch_out, rtol=1e-2, atol=1e-2)
        print("Random-input test passed!")

        zero_x_bf16 = torch.zeros_like(x_bf16)
        zero_x_q, zero_x_scale = quantize_to_fp8_packed_ue8m0(zero_x_bf16)
        zero_output = torch.empty_like(output)

        runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
            zero_x_q, zero_x_scale, w_q, w_scale, residual, zero_output
        )

        zero_x_ref = dequant_from_packed_ue8m0(zero_x_q, zero_x_scale)
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
x_bad_n_q, x_bad_n_scale = quantize_to_fp8_packed_ue8m0(x_bad_n)
w_bad_n_q, w_bad_n_scale = quantize_to_fp8_packed_ue8m0(w_bad_n)
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
x_bad_b_q, x_bad_b_scale = quantize_to_fp8_packed_ue8m0(x_bad_b)
w_bad_b_q, w_bad_b_scale = quantize_to_fp8_packed_ue8m0(w_bad_b)
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
x_bad_large_b_q, x_bad_large_b_scale = quantize_to_fp8_packed_ue8m0(
    x_bad_large_b
)
w_bad_large_b_q, w_bad_large_b_scale = quantize_to_fp8_packed_ue8m0(
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
x_valid_q, x_valid_scale = quantize_to_fp8_packed_ue8m0(x_valid)
w_valid_q, w_valid_scale = quantize_to_fp8_packed_ue8m0(w_valid)
bad_scale_output = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)
bad_input_scale = torch.empty_strided(
    x_valid_scale.shape,
    (x_valid_scale.shape[1] + 1, 1),
    device=x_valid_scale.device,
    dtype=x_valid_scale.dtype,
)
bad_input_scale.copy_(x_valid_scale)
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
    assert "input_scale must use packed UE8M0 layout" in str(exc)
    print("Scale layout mismatch negative test passed!")

print("\n=== Legacy column-major compatibility test ===")

compat_shape = (4, 128, 2048)
compat_batch_size, compat_output_size, compat_reduction_size = compat_shape
compat_x_bf16 = torch.randn(
    (compat_batch_size, compat_reduction_size),
    device="cuda",
    dtype=torch.bfloat16,
    generator=g,
)
compat_w_bf16 = torch.randn(
    (compat_output_size, compat_reduction_size),
    device="cuda",
    dtype=torch.bfloat16,
    generator=g,
)
compat_x_q_row, compat_x_scale_row = quantize_to_fp8_packed_ue8m0(compat_x_bf16)
compat_w_q_row, compat_w_scale_row = quantize_to_fp8_packed_ue8m0(compat_w_bf16)
compat_x_q_col, compat_x_scale_col = quantize_to_fp8_deepgemm_style(compat_x_bf16)
compat_w_q_col, compat_w_scale_col = quantize_to_fp8_deepgemm_style(compat_w_bf16)
compat_output_row = torch.empty(
    compat_batch_size, compat_output_size, device="cuda", dtype=torch.bfloat16
)
compat_output_col = torch.empty_like(compat_output_row)
runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
    compat_x_q_row,
    compat_x_scale_row,
    compat_w_q_row,
    compat_w_scale_row,
    None,
    compat_output_row,
)
runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
    compat_x_q_col,
    compat_x_scale_col,
    compat_w_q_col,
    compat_w_scale_col,
    None,
    compat_output_col,
)
compat_x_ref = dequant_from_packed_ue8m0(compat_x_q_row, compat_x_scale_row)
compat_w_ref = dequant_from_packed_ue8m0(compat_w_q_row, compat_w_scale_row)
compat_ref = torch.matmul(compat_x_ref, torch.transpose(compat_w_ref, 0, 1)).to(
    torch.bfloat16
)
torch.testing.assert_close(compat_output_row, compat_ref, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(compat_output_col, compat_ref, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(
    compat_output_row, compat_output_col, rtol=1e-2, atol=1e-2
)
print("Legacy column-major dense compatibility test passed!")

print("\n=== Split-K override tests ===")

forced_split_shape = (1, 128, 7168)
batch_size, output_size, reduction_size = forced_split_shape
x_split = torch.randn(
    (batch_size, reduction_size),
    device="cuda",
    dtype=torch.bfloat16,
    generator=g,
)
w_split = torch.randn(
    (output_size, reduction_size),
    device="cuda",
    dtype=torch.bfloat16,
    generator=g,
)
x_split_q, x_split_scale = quantize_to_fp8_packed_ue8m0(x_split)
w_split_q, w_split_scale = quantize_to_fp8_packed_ue8m0(w_split)
x_split_ref = dequant_from_packed_ue8m0(x_split_q, x_split_scale)
w_split_ref = dequant_from_packed_ue8m0(w_split_q, w_split_scale)
split_ref = torch.matmul(x_split_ref, torch.transpose(w_split_ref, 0, 1)).to(
    torch.bfloat16
)

old_split_env = os.environ.get("MIRAGE_FORCE_SM100_FP8_SPLIT_K")
try:
    for forced_split_k in ("1", "2", "4", "8"):
        os.environ["MIRAGE_FORCE_SM100_FP8_SPLIT_K"] = forced_split_k
        split_output = torch.empty_like(split_ref)
        runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
            x_split_q, x_split_scale, w_split_q, w_split_scale, None, split_output
        )
        torch.testing.assert_close(split_output, split_ref, rtol=1e-2, atol=1e-2)
        print(f"Forced split-K={forced_split_k} correctness test passed!")

    residual = torch.randn(
        batch_size, output_size, device="cuda", dtype=torch.bfloat16, generator=g
    )
    os.environ["MIRAGE_FORCE_SM100_FP8_SPLIT_K"] = "4"
    residual_output = torch.empty_like(split_ref)
    runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
        x_split_q,
        x_split_scale,
        w_split_q,
        w_split_scale,
        residual,
        residual_output,
    )
    residual_ref = (split_ref.float() + residual.float()).to(torch.bfloat16)
    torch.testing.assert_close(
        residual_output, residual_ref, rtol=1e-2, atol=1e-2
    )
    print("Residual path bypasses split-K override test passed!")
finally:
    if old_split_env is None:
        os.environ.pop("MIRAGE_FORCE_SM100_FP8_SPLIT_K", None)
    else:
        os.environ["MIRAGE_FORCE_SM100_FP8_SPLIT_K"] = old_split_env
