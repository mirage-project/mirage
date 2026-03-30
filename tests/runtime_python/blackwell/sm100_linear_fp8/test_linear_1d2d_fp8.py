import torch
import runtime_kernel_blackwell_linear_fp8 as runtime_kernel_blackwell

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

reduction_sizes = [768]
output_sizes = [128]
batch_size = 1
block_k = 128
num_k_blocks = reduction_sizes[0] // block_k

has_residual = False


def round_up_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def quantize_to_fp8_with_packed_ue8m0_scale(x_bf16: torch.Tensor, block_k: int):
    """
    Quantize a BF16 tensor of shape [outer_dim, K] into:
      1. FP8 E4M3FN tensor of the same shape
      2. Packed uint32 scale tensor of shape [outer_dim, padded_scale_k]

    Logical valid scale count is K // 128.
    Physical storage pads the 2nd dimension to a multiple of 4 uint32s (16B).
    """
    assert x_bf16.dim() == 2
    outer_dim, k = x_bf16.shape
    assert k % block_k == 0
    assert block_k == 128

    valid_scale_k = k // block_k
    padded_scale_k = round_up_to_multiple(valid_scale_k, 4)

    x_fp32 = x_bf16.float()
    x_q = torch.empty_like(x_fp32, dtype=torch.float8_e4m3fn)

    # Use zeros so padded tail is well-defined.
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

                abs_max = sub.abs().max().item()
                abs_max = max(abs_max, 1e-10)
                scale = abs_max / max_fp8

                # UE8M0 exponent-only encoding
                ue8m0 = int(torch.ceil(torch.log2(torch.tensor(scale))).item()) + 127
                ue8m0 = max(0, min(255, ue8m0))
                packed |= (ue8m0 & 0xFF) << (8 * sub_idx)

                q_sub = torch.clamp(sub / scale, -max_fp8, max_fp8).to(torch.float8_e4m3fn)
                q_block[sub_start:sub_end] = q_sub

            x_q[outer_idx, k_start:k_end] = q_block
            packed_scales[outer_idx, blk_idx] = packed

    return x_q, packed_scales


def dequant_from_fp8_and_packed_ue8m0(x_q: torch.Tensor, packed_scales: torch.Tensor, block_k: int):
    """
    Dequantize FP8 E4M3FN tensor using packed uint32 UE8M0 scales.

    packed_scales may be physically padded on dim-1; only the first valid_scale_k
    entries are consumed.
    """
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


for reduction_size in reduction_sizes:
    for output_size in output_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} output_size = {output_size} reduction_size = {reduction_size} has_residual = {has_residual} ==="
        )

        assert reduction_size % block_k == 0

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

        print("x_scale shape:", tuple(x_scale.shape), "stride:", x_scale.stride())
        print("w_scale shape:", tuple(w_scale.shape), "stride:", w_scale.stride())

        residual = torch.randn(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16, generator=g
        )
        output = torch.empty(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16
        )

        if not has_residual:
            residual = None

        # Route B:
        # The runtime binding is expected to accept already-quantized inputs:
        #   x_q      : [BATCH_SIZE, REDUCTION_SIZE], fp8_e4m3fn
        #   x_scale  : [BATCH_SIZE, padded_scale_k], uint32 packed UE8M0
        #   w_q      : [OUTPUT_SIZE, REDUCTION_SIZE], fp8_e4m3fn
        #   w_scale  : [OUTPUT_SIZE, padded_scale_k], uint32 packed UE8M0
        # where only the first valid_scale_k columns are logically valid.
        runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
            x_q, x_scale, w_q, w_scale, residual, output
        )

        x_ref = dequant_from_fp8_and_packed_ue8m0(x_q, x_scale, block_k)
        w_ref = dequant_from_fp8_and_packed_ue8m0(w_q, w_scale, block_k)

        torch_out = torch.matmul(x_ref, torch.transpose(w_ref, 0, 1)).to(torch.bfloat16)
        if has_residual:
            torch_out = torch_out + residual

        torch.testing.assert_close(
            output,
            torch_out,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

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
        zero_torch_out = torch.matmul(
            zero_x_ref, torch.transpose(w_ref, 0, 1)
        ).to(torch.bfloat16)
        if has_residual:
            zero_torch_out = zero_torch_out + residual

        torch.testing.assert_close(
            zero_output,
            zero_torch_out,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Zero-input bring-up passed!")

        for _ in range(16):
            runtime_kernel_blackwell.linear_fp8_1d2d_sm100(
                x_q, x_scale, w_q, w_scale, residual, output
            )

        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
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
