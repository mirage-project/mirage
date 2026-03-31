import torch
import runtime_kernel_blackwell_quantize_fp8 as runtime_kernel_blackwell

torch.set_printoptions(sci_mode=False, profile="full")

g = torch.Generator(device="cuda").manual_seed(1234)

batch_sizes = [1, 8]
hidden_sizes = [int(x) for x in runtime_kernel_blackwell.supported_hidden_sizes()]
group_sizes = [int(x) for x in runtime_kernel_blackwell.supported_group_sizes()]
block_k = 128
fp8_max = 448.0

assert group_sizes == [block_k]


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

    for outer_idx in range(outer_dim):
        for blk_idx in range(valid_scale_k):
            k_start = blk_idx * block_k
            k_end = k_start + block_k

            block = x_fp32[outer_idx, k_start:k_end].clone()
            q_block = torch.empty_like(block, dtype=torch.float8_e4m3fn)
            packed = 0

            for sub_idx in range(4):
                sub_start = sub_idx * 32
                sub_end = sub_start + 32
                sub = block[sub_start:sub_end]

                abs_max = max(sub.abs().max().item(), 1e-10)
                scale = abs_max / fp8_max

                ue8m0 = int(torch.ceil(torch.log2(torch.tensor(scale))).item()) + 127
                ue8m0 = max(0, min(255, ue8m0))
                packed |= (ue8m0 & 0xFF) << (8 * sub_idx)

                q_block[sub_start:sub_end] = torch.clamp(
                    sub / scale, -fp8_max, fp8_max
                ).to(torch.float8_e4m3fn)

            x_q[outer_idx, k_start:k_end] = q_block
            packed_scales[outer_idx, blk_idx] = packed

    return x_q, packed_scales


for batch_size in batch_sizes:
    for hidden_size in hidden_sizes:
        valid_scale_k = hidden_size // block_k
        padded_scale_k = round_up_to_multiple(valid_scale_k, 4)

        print(
            f"\n=== Testing batch_size={batch_size} hidden_size={hidden_size} block_k={block_k} padded_scale_k={padded_scale_k} ==="
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
        scales = torch.empty(
            (batch_size, padded_scale_k), device="cuda", dtype=torch.uint32
        )

        runtime_kernel_blackwell.quantize_fp8_sm100(x, output, scales, group_size=block_k)

        quant_ref, scale_ref = quantize_to_fp8_with_packed_ue8m0_scale(x, block_k)

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
        zero_scales = torch.empty_like(scales)
        runtime_kernel_blackwell.quantize_fp8_sm100(
            zero_x, zero_output, zero_scales, group_size=block_k
        )
        zero_quant_ref, zero_scale_ref = quantize_to_fp8_with_packed_ue8m0_scale(
            zero_x, block_k
        )

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
unsupported_scales = torch.empty(
    (1, round_up_to_multiple(unsupported_hidden_size // block_k, 4)),
    device="cuda",
    dtype=torch.uint32,
)
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
bad_group_scales = torch.empty(
    (1, round_up_to_multiple(block_k // bad_group_size, 4)),
    device="cuda",
    dtype=torch.uint32,
)
bad_group_x = torch.randn((1, block_k), device="cuda", dtype=torch.bfloat16, generator=g)
try:
    runtime_kernel_blackwell.quantize_fp8_sm100(
        bad_group_x, bad_group_output, bad_group_scales, group_size=bad_group_size
    )
    raise AssertionError("Expected unsupported group_size failure")
except RuntimeError as exc:
    assert "Unsupported group_size" in str(exc)
    print("Unsupported group_size negative test passed!")
