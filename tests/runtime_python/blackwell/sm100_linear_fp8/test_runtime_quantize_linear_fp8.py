import os
import sys
import torch
import runtime_kernel_blackwell_linear_fp8 as linear_kernel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QUANTIZE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../sm100_quantize_fp8"))
if QUANTIZE_DIR not in sys.path:
    sys.path.insert(0, QUANTIZE_DIR)

import runtime_kernel_blackwell_quantize_fp8 as quantize_kernel

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

block_k = 128

supported_shapes = [
    tuple(int(dim) for dim in shape)
    for shape in linear_kernel.supported_dense_gemm_shapes()
]
representative_ks = [128, 768, 2048, 7168]
pipeline_shapes = [
    shape
    for shape in supported_shapes
    if shape[0] == 1 and shape[1] == 128 and shape[2] in representative_ks
]


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


for batch_size, output_size, reduction_size in pipeline_shapes:
    for has_residual in (False, True):
        print(
            f"\n=== Runtime quantize + linear test: batch_size={batch_size} output_size={output_size} reduction_size={reduction_size} has_residual={has_residual} ==="
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

        valid_scale_k = reduction_size // block_k
        padded_scale_k = ((valid_scale_k + 3) // 4) * 4

        x_q = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
        w_q = torch.empty_like(w_bf16, dtype=torch.float8_e4m3fn)
        x_scale = torch.empty(
            (batch_size, padded_scale_k), device="cuda", dtype=torch.uint32
        )
        w_scale = torch.empty(
            (output_size, padded_scale_k), device="cuda", dtype=torch.uint32
        )

        quantize_kernel.quantize_fp8_sm100(x_bf16, x_q, x_scale, group_size=block_k)
        quantize_kernel.quantize_fp8_sm100(w_bf16, w_q, w_scale, group_size=block_k)

        residual = torch.randn(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16, generator=g
        )
        if not has_residual:
            residual = None

        output = torch.empty(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16
        )

        linear_kernel.linear_fp8_1d2d_sm100(
            x_q, x_scale, w_q, w_scale, residual, output
        )

        x_ref = dequant_from_fp8_and_packed_ue8m0(x_q, x_scale, block_k)
        w_ref = dequant_from_fp8_and_packed_ue8m0(w_q, w_scale, block_k)
        torch_out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1))
        if has_residual:
            torch_out = torch_out + residual.float()
        torch_out = torch_out.to(torch.bfloat16)

        torch.testing.assert_close(output, torch_out, rtol=1e-2, atol=1e-2)
        print("Runtime quantize + GEMM test passed!")

        zero_x_bf16 = torch.zeros_like(x_bf16)
        zero_x_q = torch.empty_like(x_q)
        zero_x_scale = torch.empty_like(x_scale)
        zero_output = torch.empty_like(output)

        quantize_kernel.quantize_fp8_sm100(
            zero_x_bf16, zero_x_q, zero_x_scale, group_size=block_k
        )
        linear_kernel.linear_fp8_1d2d_sm100(
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
        print("Zero-input runtime quantize + GEMM bring-up passed!")
