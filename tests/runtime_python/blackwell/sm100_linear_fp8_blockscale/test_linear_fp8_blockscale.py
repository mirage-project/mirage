"""Kernel-wrapper correctness test for the preserved-block-scale dense FP8 GEMM.

The kernel consumes the checkpoint's float32 128x128 weight block scales directly
(no UE8M0 requantization, no per-row collapse) and the fp32-scale variant of the
per-token activation quantizer, applying `a_scale * b_scale` per 128-element K
tile. This test drives it through the pybind wrapper and checks it against a
PyTorch dequant reference at the shapes the Qwen3.5 dense projections use.

Run:  python test_linear_fp8_blockscale.py
"""

import torch

import runtime_kernel_blackwell_linear_fp8_blockscale as linear_kernel

# The fp32 references below must be true IEEE fp32, not TF32 on tensor cores,
# or the "kernel is within one bf16 rounding of exact" check compares against a
# reference that is itself ~1e-3 off.
torch.backends.cuda.matmul.allow_tf32 = False

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10

torch.set_printoptions(sci_mode=False)


def quantize_activation(x_bf16):
    """vLLM's dynamic per-token, per-128-group fp8 quantization with fp32 scales.

    Mirrors the primitive MPK's quantize_fp8 kernel implements with
    SCALE_UE8M0=false: absmax = max(max|x|, 1e-10), scale = absmax / 448,
    x / scale, clamp to +-448 before the round-to-nearest-even e4m3 cast
    (docs/qwen35/vllm-graph.md 3.4).
    """
    m, k = x_bf16.shape
    xf = x_bf16.float().reshape(m, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = absmax / FP8_MAX
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(m, k).to(torch.float8_e4m3fn), scale.contiguous()


def quantize_weight_blocks(w_bf16):
    """Block-fp8 weight quantization in the checkpoint's own format:
    one float32 scale per 128x128 weight block, stored as [N/128, K/128]."""
    n, k = w_bf16.shape
    wf = w_bf16.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3)).clamp(min=EPS)
    scale = absmax / FP8_MAX
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(n, k).to(torch.float8_e4m3fn), scale.contiguous()


def dequantize_activation(q, scale):
    m, k = q.shape
    return (q.float().reshape(m, k // BLOCK, BLOCK) * scale.unsqueeze(-1)).reshape(m, k)


def dequantize_weight(q, scale):
    n, k = q.shape
    wf = q.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    return (wf * scale[:, None, :, None]).reshape(n, k)


def frob_rel(actual, ref):
    return (actual - ref).norm().item() / max(ref.norm().item(), 1e-30)


def run_case(batch_size, output_size, reduction_size, has_residual, generator):
    x_bf16 = torch.randn(
        (batch_size, reduction_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w_bf16 = torch.randn(
        (output_size, reduction_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    x_q, x_scale = quantize_activation(x_bf16)
    w_q, w_scale = quantize_weight_blocks(w_bf16)
    assert x_scale.dtype == torch.float32 and w_scale.dtype == torch.float32
    assert tuple(w_scale.shape) == (output_size // BLOCK, reduction_size // BLOCK)

    residual = None
    if has_residual:
        residual = torch.randn(
            (batch_size, output_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )

    output = torch.empty(
        (batch_size, output_size), device="cuda", dtype=torch.bfloat16
    )
    linear_kernel.linear_fp8_blockscale_sm100(
        x_q, x_scale, w_q, w_scale, residual, output
    )
    torch.cuda.synchronize()

    ref = dequantize_activation(x_q, x_scale) @ dequantize_weight(w_q, w_scale).t()
    if has_residual:
        ref = ref + residual.float()

    # The kernel's only unavoidable error is the bf16 rounding of its own output;
    # anything materially above that floor means the scales are being applied
    # incorrectly (the whole point of this path).
    floor = frob_rel(ref.to(torch.bfloat16).float(), ref)
    err = frob_rel(output.float(), ref)
    print(
        f"  B={batch_size:<4} N={output_size:<4} K={reduction_size:<5} "
        f"residual={int(has_residual)} frob_rel={err:.3e} "
        f"bf16_output_floor={floor:.3e} ratio={err / max(floor, 1e-30):.2f}"
    )
    assert err <= max(1.6 * floor, 1e-6), (
        f"B={batch_size} N={output_size} K={reduction_size} "
        f"residual={has_residual}: frob_rel {err:.3e} exceeds 1.6x the bf16 "
        f"output-rounding floor {floor:.3e}"
    )
    torch.testing.assert_close(output, ref.to(torch.bfloat16), rtol=2e-2, atol=2e-2)


def run_scale_sensitivity_check(generator):
    """Negative control: the float32 block scales must actually reach the MMA.

    Doubling one weight block's scale (and halving its fp8 payload) leaves the
    dequantized weight unchanged, so a kernel that consumes the scales exactly
    is invariant; a kernel that drops or rounds them is not.
    """
    batch_size, output_size, reduction_size = 4, 256, 2048
    x_bf16 = torch.randn(
        (batch_size, reduction_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w_bf16 = torch.randn(
        (output_size, reduction_size),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    x_q, x_scale = quantize_activation(x_bf16)
    w_q, w_scale = quantize_weight_blocks(w_bf16)

    base = torch.empty((batch_size, output_size), device="cuda", dtype=torch.bfloat16)
    linear_kernel.linear_fp8_blockscale_sm100(x_q, x_scale, w_q, w_scale, None, base)

    # Halve block (1, 3)'s payload exactly (a power of two is exact in e4m3) and
    # double its scale: the represented weights are bit-identical.
    w_q2 = w_q.clone()
    block = w_q2[BLOCK : 2 * BLOCK, 3 * BLOCK : 4 * BLOCK].float() * 0.5
    w_q2[BLOCK : 2 * BLOCK, 3 * BLOCK : 4 * BLOCK] = block.to(torch.float8_e4m3fn)
    w_scale2 = w_scale.clone()
    w_scale2[1, 3] = w_scale2[1, 3] * 2.0

    rescaled = torch.empty_like(base)
    linear_kernel.linear_fp8_blockscale_sm100(x_q, x_scale, w_q2, w_scale2, None, rescaled)

    # And a control that MUST change: scale one block without touching the payload.
    w_scale3 = w_scale.clone()
    w_scale3[1, 3] = w_scale3[1, 3] * 1.37
    perturbed = torch.empty_like(base)
    linear_kernel.linear_fp8_blockscale_sm100(x_q, x_scale, w_q, w_scale3, None, perturbed)
    torch.cuda.synchronize()

    equal_frac = (rescaled == base).float().mean().item()
    # Block (1, 3) covers weight rows 128..255, i.e. output columns 128..255 only.
    touched = (perturbed[:, BLOCK:] != base[:, BLOCK:]).float().mean().item()
    untouched = (perturbed[:, :BLOCK] != base[:, :BLOCK]).float().mean().item()
    print(
        f"  scale-sensitivity: payload/scale swap identical={equal_frac:.4f} "
        f"(want 1.0), 1.37x scale changed {touched:.4f} of the affected "
        f"output block (want ~1.0) and {untouched:.4f} outside it (want 0.0)"
    )
    assert equal_frac == 1.0, (
        "halving one block's fp8 payload while doubling its float32 scale changed "
        "the output: the block scales are not applied exactly"
    )
    # 1.37 is not a power of two: a kernel that rounded the scales to UE8M0 would
    # ignore this perturbation entirely.
    assert touched > 0.9, (
        "scaling one 128x128 block by 1.37x barely changed its own output block: "
        "the float32 block scales are being rounded or dropped"
    )
    assert untouched == 0.0, (
        "scaling one 128x128 block changed output columns outside that block: "
        "the block scales are indexed incorrectly"
    )


def main():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    batch_sizes = [1, 2, 4, 8, 16, 64, 256]
    output_sizes = [128, 256]
    # 2048 and 4096 are the two dense-path K values in Qwen3.5; 512 is the
    # shared-expert down_proj reduction.
    reduction_sizes = [512, 2048, 4096]

    print("=== preserved-block-scale dense FP8 GEMM vs torch dequant reference ===")
    for batch_size in batch_sizes:
        for output_size in output_sizes:
            for reduction_size in reduction_sizes:
                for has_residual in (False, True):
                    run_case(
                        batch_size,
                        output_size,
                        reduction_size,
                        has_residual,
                        generator,
                    )

    print("=== zero-input bring-up ===")
    zero_x = torch.zeros((16, 2048), device="cuda", dtype=torch.bfloat16)
    x_q, x_scale = quantize_activation(zero_x)
    w_bf16 = torch.randn(
        (128, 2048), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    w_q, w_scale = quantize_weight_blocks(w_bf16)
    output = torch.empty((16, 128), device="cuda", dtype=torch.bfloat16)
    linear_kernel.linear_fp8_blockscale_sm100(x_q, x_scale, w_q, w_scale, None, output)
    torch.cuda.synchronize()
    assert torch.all(output == 0), "zero input must produce a zero output"
    print("  zero input -> zero output: OK")

    print("=== scale-consumption negative control ===")
    run_scale_sensitivity_check(generator)

    print("ALL LINEAR_FP8_BLOCKSCALE KERNEL TESTS PASSED")


if __name__ == "__main__":
    main()
