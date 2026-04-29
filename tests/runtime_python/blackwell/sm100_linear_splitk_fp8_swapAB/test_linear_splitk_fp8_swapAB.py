"""Direct kernel-wrapper correctness test for the split-K FP8 swap-AB Linear kernel.

The wrapper launches `split_k_factor` CTAs serially (each with its own
per-slice TMA descriptors) and the kernel reduce-adds partials into the
shared output tile. We verify against a dequant FP8 reference.

Run:
  CUDA_VISIBLE_DEVICES=<free-gpu> python tests/runtime_python/blackwell/sm100_linear_splitk_fp8_swapAB/test_linear_splitk_fp8_swapAB.py

Build first:
  cd tests/runtime_python/blackwell/sm100_linear_splitk_fp8_swapAB
  python setup.py build_ext --inplace
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import runtime_kernel_blackwell_linear_splitk_fp8_swapAB as mod  # noqa: E402
from sm100_fp8_scale_layout import (  # noqa: E402
    quantize_to_fp8_packed_ue8m0,
    dequant_from_packed_ue8m0,
)


# (label, BATCH, OUTPUT, K_per_task, split_k_factor)
# Each row's (BATCH, OUTPUT, K_per_task) must be in mod.supported_shapes().
#
# Constraint: K_per_task MUST be a multiple of 512 (= BLOCK_K * 4 packed
# scales per uint32). The split_k_factor we pick has to divide
# `full_K / 512` evenly. Examples:
#   full_K = 4608 → full_K/512 = 9  → valid splits are 1, 3, 9
#   full_K = 7168 → full_K/512 = 14 → valid splits are 1, 2, 7, 14
#   full_K = 16384 → full_K/512 = 32 → valid splits are 1, 2, 4, 8, 16, 32
CASES = [
    # split_k_factor=1: degenerate, equivalent to the non-split kernel —
    # validates the SplitK=true reduce-add path on its own
    ("smoke split_k=1",         16, 128, 4096, 1),
    # 2-way split (q_a-like, full_K=7168)
    ("q_a-like split_k=2",      16, 128, 3584, 2),
    # 3-way split (down-like, full_K=4608) — full_K/512 = 9, split=3 → K_per_task=1536
    ("down-like split_k=3",     16, 128, 1536, 3),
    # 4-way split (o_proj-like, full_K=16384)
    ("o_proj-like split_k=4",   16, 128, 4096, 4),
    # B=1 / B=4 to verify the BATCH < MMA_N path under split-K
    ("smoke B=1 split_k=2",      1, 128, 2048, 2),
    ("smoke B=4 split_k=2",      4, 128, 1024, 2),
]


def _quantize(x_bf16):
    x_q, scales = quantize_to_fp8_packed_ue8m0(x_bf16)
    return x_q.contiguous(), scales.contiguous()


def run_case(label, batch, out, k_per_task, split_k):
    full_k = k_per_task * split_k
    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  B={batch}  OUT={out}  K_per_task={k_per_task}  "
          f"split_k={split_k}  full_K={full_k}")

    if (batch, out, k_per_task) not in set(tuple(s) for s in mod.supported_shapes()):
        print(f"  SKIP (shape not pre-instantiated)")
        return None

    torch.manual_seed(0)
    device = "cuda"
    input_bf16 = (torch.randn(batch, full_k, dtype=torch.bfloat16, device=device)
                  * 0.1).contiguous()
    weight_bf16 = (torch.randn(out, full_k, dtype=torch.bfloat16, device=device)
                   / (full_k ** 0.5)).contiguous()
    input_fp8, input_scale = _quantize(input_bf16)
    weight_fp8, weight_scale = _quantize(weight_bf16)

    # Output MUST be zero-initialized — the kernel reduce-adds.
    output = torch.zeros(batch, out, dtype=torch.bfloat16, device=device)

    # Reference: matches what the kernel actually computes (dequant FP8).
    input_dq = dequant_from_packed_ue8m0(input_fp8, input_scale)
    weight_dq = dequant_from_packed_ue8m0(weight_fp8, weight_scale)
    ref = (input_dq.float() @ weight_dq.float().T).to(torch.bfloat16)

    mod.linear_splitk_fp8_swapAB_sm100(
        input_fp8, input_scale, weight_fp8, weight_scale, output,
        split_k)
    torch.cuda.synchronize()

    finite = torch.isfinite(output).all().item()
    diff = (output.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"  output[0, :8]:    {output[0, :8].tolist()}")
    print(f"  reference[0, :8]: {ref[0, :8].tolist()}")
    print(f"  finite={finite}  max-abs-error={max_abs:.4f}  mean-abs-error={mean_abs:.6f}")
    ok = finite and max_abs < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = [(c[0], run_case(*c)) for c in CASES]
    print(f"\n{'='*72}")
    print("Summary:")
    n_pass = 0
    n_run = 0
    for name, ok in results:
        if ok is None:
            print(f"  SKIP  {name}")
        else:
            n_run += 1
            n_pass += int(ok)
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{n_run} passed (of {len(results)} total)")
    if n_run > 0 and n_pass != n_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
