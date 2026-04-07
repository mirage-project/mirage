import os
import sys
import torch
import runtime_kernel_blackwell_linear_fp8 as linear_kernel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
QUANTIZE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../sm100_quantize_fp8"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
if QUANTIZE_DIR not in sys.path:
    sys.path.insert(0, QUANTIZE_DIR)

import runtime_kernel_blackwell_quantize_fp8 as quantize_kernel
from sm100_fp8_scale_layout import (
    BLOCK_K,
    allocate_packed_ue8m0_scale,
    dequant_from_packed_ue8m0_deepgemm_style,
)

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

block_k = BLOCK_K

supported_shapes = [
    tuple(int(dim) for dim in shape)
    for shape in linear_kernel.supported_dense_gemm_shapes()
]
representative_ks = [128, 768, 2048, 7168]
representative_batch_sizes = [1, 2, 4, 8, 16]
pipeline_shapes = [
    shape
    for shape in supported_shapes
    if shape[0] in representative_batch_sizes
    and shape[1] == 128
    and shape[2] in representative_ks
]

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

        x_q = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
        w_q = torch.empty_like(w_bf16, dtype=torch.float8_e4m3fn)
        x_scale = allocate_packed_ue8m0_scale(batch_size, reduction_size, x_bf16.device)
        w_scale = allocate_packed_ue8m0_scale(output_size, reduction_size, w_bf16.device)

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

        x_ref = dequant_from_packed_ue8m0_deepgemm_style(x_q, x_scale)
        w_ref = dequant_from_packed_ue8m0_deepgemm_style(w_q, w_scale)
        torch_out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1))
        if has_residual:
            torch_out = torch_out + residual.float()
        torch_out = torch_out.to(torch.bfloat16)

        torch.testing.assert_close(output, torch_out, rtol=1e-2, atol=1e-2)
        print("Runtime quantize + GEMM test passed!")

        zero_x_bf16 = torch.zeros_like(x_bf16)
        zero_x_q = torch.empty_like(x_q)
        zero_x_scale = allocate_packed_ue8m0_scale(
            batch_size, reduction_size, x_bf16.device
        )
        zero_output = torch.empty_like(output)

        quantize_kernel.quantize_fp8_sm100(
            zero_x_bf16, zero_x_q, zero_x_scale, group_size=block_k
        )
        linear_kernel.linear_fp8_1d2d_sm100(
            zero_x_q, zero_x_scale, w_q, w_scale, residual, zero_output
        )

        zero_x_ref = dequant_from_packed_ue8m0_deepgemm_style(zero_x_q, zero_x_scale)
        zero_torch_out = torch.matmul(zero_x_ref, torch.transpose(w_ref, 0, 1))
        if has_residual:
            zero_torch_out = zero_torch_out + residual.float()
        zero_torch_out = zero_torch_out.to(torch.bfloat16)

        torch.testing.assert_close(zero_output, zero_torch_out, rtol=1e-2, atol=1e-2)
        print("Zero-input runtime quantize + GEMM bring-up passed!")
