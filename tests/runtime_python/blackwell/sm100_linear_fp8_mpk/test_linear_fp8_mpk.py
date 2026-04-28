"""Direct kernel-wrapper smoke test for the MPK FP8 swap-AB Linear kernel.

Run:
  CUDA_VISIBLE_DEVICES=0 python tests/runtime_python/blackwell/sm100_linear_fp8_mpk/test_linear_fp8_mpk.py

Build first:
  cd tests/runtime_python/blackwell/sm100_linear_fp8_mpk
  python setup.py build_ext --inplace
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import runtime_kernel_blackwell_linear_fp8_mpk as mod  # noqa: E402
from sm100_fp8_scale_layout import quantize_to_fp8_packed_ue8m0  # noqa: E402


def main():
    torch.manual_seed(42)
    device = "cuda"

    # Default smoke shape; any entry from mod.supported_shapes() works.
    BATCH, OUTPUT, K = 4, 128, 128
    print(f"shape: B={BATCH}, OUTPUT={OUTPUT}, K={K}")

    # Controlled inputs: all-ones. With K=128, expected output = 128 everywhere.
    input_bf16 = torch.ones(BATCH, K, dtype=torch.bfloat16, device=device)
    weight_bf16 = torch.ones(OUTPUT, K, dtype=torch.bfloat16, device=device)

    input_fp8, input_scale = quantize_to_fp8_packed_ue8m0(input_bf16)
    weight_fp8, weight_scale = quantize_to_fp8_packed_ue8m0(weight_bf16)
    print(f"input_fp8 shape={tuple(input_fp8.shape)} dtype={input_fp8.dtype}")
    print(f"input_scale shape={tuple(input_scale.shape)} dtype={input_scale.dtype}")
    print(f"weight_fp8 shape={tuple(weight_fp8.shape)} dtype={weight_fp8.dtype}")
    print(f"weight_scale shape={tuple(weight_scale.shape)} dtype={weight_scale.dtype}")

    output = torch.zeros(BATCH, OUTPUT, dtype=torch.bfloat16, device=device)

    # Reference: BF16 matmul. Loose tolerance because FP8+UE8M0 is lossy.
    ref = (input_bf16.float() @ weight_bf16.float().T).to(torch.bfloat16)

    print("Launching kernel...")
    mod.linear_fp8_mpk_sm100(input_fp8, input_scale, weight_fp8, weight_scale, output)
    torch.cuda.synchronize()

    print(f"output[0, :8]:    {output[0, :8]}")
    print(f"reference[0, :8]: {ref[0, :8]}")
    print(f"output finite:    {torch.isfinite(output).all().item()}")
    print(f"output nonzero:   {(output != 0).any().item()}")
    finite_mask = torch.isfinite(output)
    print(f"finite count:     {finite_mask.sum().item()} / {output.numel()}")
    print(f"max abs (finite): {output[finite_mask].abs().max().item():.4f}")
    print(f"all rows match across batch?")
    for r in range(min(4, BATCH)):
        print(f"  row {r}[0:4]: {output[r, :4]}")
    print(f"  col 0 across batch: {output[:, 0]}")

    if finite_mask.all():
        diff = (output.float() - ref.float()).abs()
        print(f"max-abs-error:    {diff.max().item():.4f}")
        print(f"mean-abs-error:   {diff.mean().item():.4f}")


if __name__ == "__main__":
    main()
