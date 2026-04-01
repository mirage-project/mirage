import torch
import runtime_kernel_blackwell_linear_fp8 as runtime_kernel_blackwell

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

supported_shapes = [
    tuple(int(dim) for dim in shape)
    for shape in runtime_kernel_blackwell.supported_dense_gemm_shapes()
]
block_k = 128
benchmark_shape = (1, 128, 768)


def round_up_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def quantize_to_fp8_with_packed_ue8m0_scale(x_bf16: torch.Tensor, block_k: int):
    assert x_bf16.dim() == 2
    outer_dim, k = x_bf16.shape
    assert k % block_k == 0
    assert block_k == 128

    valid_scale_k = k // block_k
    padded_scale_k = round_up_to_multiple(valid_scale_k, 4)

    x_fp32 = x_bf16.float()
    x_q = torch.empty_like(x_fp32, dtype=torch.float8_e4m3fn)
    packed_scales = torch.zeros(
        (outer_dim, padded_scale_k), device=x_bf16.device, dtype=torch.uint32
    )

    max_fp8 = 448.0

    for outer_idx in range(outer_dim):
        for blk_idx in range(valid_scale_k):
            k_start = blk_idx * block_k
            k_end = k_start + block_k

            block = x_fp32[outer_idx, k_start:k_end].clone()
            packed = 0
            q_block = torch.empty_like(block, dtype=torch.float8_e4m3fn)

            for sub_idx in range(4):
                sub_start = sub_idx * 32
                sub_end = sub_start + 32
                sub = block[sub_start:sub_end]

                abs_max = max(sub.abs().max().item(), 1e-10)
                scale = abs_max / max_fp8
                ue8m0 = int(torch.ceil(torch.log2(torch.tensor(scale))).item()) + 127
                ue8m0 = max(0, min(255, ue8m0))
                packed |= (ue8m0 & 0xFF) << (8 * sub_idx)

                q_sub = torch.clamp(sub / scale, -max_fp8, max_fp8).to(
                    torch.float8_e4m3fn
                )
                q_block[sub_start:sub_end] = q_sub

            x_q[outer_idx, k_start:k_end] = q_block
            packed_scales[outer_idx, blk_idx] = packed

    return x_q, packed_scales


def dequant_from_fp8_and_packed_ue8m0(
    x_q: torch.Tensor, packed_scales: torch.Tensor, block_k: int
):
    assert x_q.dim() == 2
    outer_dim, k = x_q.shape
    assert k % block_k == 0
    assert block_k == 128

    valid_scale_k = k // block_k

    assert packed_scales.dim() == 2
    assert packed_scales.shape[0] == outer_dim
    assert packed_scales.shape[1] >= valid_scale_k

    x_q_fp32 = x_q.float()
    out = torch.empty_like(x_q_fp32, dtype=torch.float32)

    for outer_idx in range(outer_dim):
        for blk_idx in range(valid_scale_k):
            k_start = blk_idx * block_k
            k_end = k_start + block_k

            packed = int(packed_scales[outer_idx, blk_idx].item())
            q_block = x_q_fp32[outer_idx, k_start:k_end]

            deq_block = torch.empty_like(q_block, dtype=torch.float32)

            for sub_idx in range(4):
                ue8m0 = (packed >> (8 * sub_idx)) & 0xFF
                scale = 2.0 ** (ue8m0 - 127)

                sub_start = sub_idx * 32
                sub_end = sub_start + 32
                deq_block[sub_start:sub_end] = q_block[sub_start:sub_end] * scale

            out[outer_idx, k_start:k_end] = deq_block

    return out


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

        x_q, x_scale = quantize_to_fp8_with_packed_ue8m0_scale(x_bf16, block_k)
        w_q, w_scale = quantize_to_fp8_with_packed_ue8m0_scale(w_bf16, block_k)

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

        x_ref = dequant_from_fp8_and_packed_ue8m0(x_q, x_scale, block_k)
        w_ref = dequant_from_fp8_and_packed_ue8m0(w_q, w_scale, block_k)
        torch_out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1))
        if has_residual:
            torch_out = torch_out + residual.float()
        torch_out = torch_out.to(torch.bfloat16)

        torch.testing.assert_close(output, torch_out, rtol=1e-2, atol=1e-2)
        print("Random-input test passed!")

        zero_x_bf16 = torch.zeros_like(x_bf16)
        zero_x_q, zero_x_scale = quantize_to_fp8_with_packed_ue8m0_scale(
            zero_x_bf16, block_k
        )
        zero_output = torch.empty_like(output)

        runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
            zero_x_q, zero_x_scale, w_q, w_scale, residual, zero_output
        )

        zero_x_ref = dequant_from_fp8_and_packed_ue8m0(
            zero_x_q, zero_x_scale, block_k
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
x_bad_k_q, x_bad_k_scale = quantize_to_fp8_with_packed_ue8m0_scale(x_bad_k, block_k)
w_bad_k_q, w_bad_k_scale = quantize_to_fp8_with_packed_ue8m0_scale(w_bad_k, block_k)
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
x_bad_n_q, x_bad_n_scale = quantize_to_fp8_with_packed_ue8m0_scale(x_bad_n, block_k)
w_bad_n_q, w_bad_n_scale = quantize_to_fp8_with_packed_ue8m0_scale(w_bad_n, block_k)
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
x_bad_b_q, x_bad_b_scale = quantize_to_fp8_with_packed_ue8m0_scale(x_bad_b, block_k)
w_bad_b_q, w_bad_b_scale = quantize_to_fp8_with_packed_ue8m0_scale(w_bad_b, block_k)
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
x_bad_large_b_q, x_bad_large_b_scale = quantize_to_fp8_with_packed_ue8m0_scale(
    x_bad_large_b, block_k
)
w_bad_large_b_q, w_bad_large_b_scale = quantize_to_fp8_with_packed_ue8m0_scale(
    w_bad_large_b, block_k
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
x_valid_q, x_valid_scale = quantize_to_fp8_with_packed_ue8m0_scale(x_valid, block_k)
w_valid_q, w_valid_scale = quantize_to_fp8_with_packed_ue8m0_scale(w_valid, block_k)
bad_scale_output = torch.empty((1, 128), device="cuda", dtype=torch.bfloat16)
try:
    runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
        x_valid_q,
        x_valid_scale[:, :-1].contiguous(),
        w_valid_q,
        w_valid_scale,
        None,
        bad_scale_output,
    )
    raise AssertionError("Expected input_scale shape mismatch failure")
except RuntimeError as exc:
    assert "input_scale shape mismatch" in str(exc)
    print("Scale shape mismatch negative test passed!")
