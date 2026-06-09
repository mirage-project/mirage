"""Test: fp8_gemm_dense_smallm_sm100_task_impl<128, 3>
Shape under investigation: M=64, N=4096, K=512 (DSv3 kv_b_v projection at
prompt_len=64, i.e. M < BM=128).

Runs via direct C++ wrapper (not PersistentKernel test_mode) so we can call
the kernel with an exact M value without going through the MPK scheduler.

Scale layout expected by the kernel (per fp8_gemm_dense_sm100_common.cuh):
    sa: float32 [M, K/128]   row-major  (1x128 group activation scale)
    sb: float32 [N/128, K/128] row-major (128x128 block weight scale)

Run:
    # Build first:
    cd tests/runtime_python/blackwell/sm100_fp8_gemm_dense
    pip install -e . -v
    # Then:
    python test_fp8_gemm_dense_smallm_testmode.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Canonical references live in pytorch_reference.py (skill convention).
from pytorch_reference import (  # noqa: E402
    quantize_a_f32scale as quantize_a,
    quantize_b_f32scale as quantize_b,
    reference_gemm,
    cosine_sim as cosine_similarity_2d,
)


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_case(kernel_mod, M: int, N: int, K: int, seed: int = 42):
    label = f"M={M}, N={N}, K={K}"
    print(f"\n{'='*70}")
    print(f"Case: {label}")
    print(f"{'='*70}")

    device = "cuda"
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16,
                         generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16,
                         generator=g)

    a_fp8, sa = quantize_a(a_bf16)
    b_fp8, sb = quantize_b(b_bf16)

    print(f"a_fp8 shape: {tuple(a_fp8.shape)}")
    print(f"b_fp8 shape: {tuple(b_fp8.shape)}")
    print(f"sa    shape: {tuple(sa.shape)}")
    print(f"sb    shape: {tuple(sb.shape)}")

    ref = reference_gemm(a_fp8, sa, b_fp8, sb)
    print(f"ref  [0, :8]: {ref[0, :8].tolist()}")

    output = torch.zeros((M, N), device=device, dtype=torch.bfloat16)
    kernel_mod.fp8_gemm_dense_smallm_launch(a_fp8, b_fp8, sa, sb, output)
    torch.cuda.synchronize()

    print(f"out  [0, :8]: {output[0, :8].tolist()}")
    max_diff = (output.float() - ref.float()).abs().max().item()
    cos      = cosine_similarity_2d(output, ref)
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Cosine sim:   {cos:.6f}")

    # Row-level zero detection: which rows of output are all-zero?
    zero_rows = (output.float().abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if zero_rows.numel() > 0:
        print(f"  WARNING: zero rows in output: {zero_rows.tolist()[:20]} "
              f"(total {zero_rows.numel()})")
    else:
        print(f"  No zero rows detected.")

    # Ref zero rows sanity check
    ref_zero = (ref.float().abs().sum(dim=1) == 0).nonzero(as_tuple=True)[0]
    if ref_zero.numel() > 0:
        print(f"  WARNING: zero rows in reference: {ref_zero.tolist()[:20]}")

    passed = cos > 0.99
    status = "PASSED" if passed else "FAILED"
    print(f"  Result: {status} (cos={cos:.4f})")
    return passed, cos, max_diff


def main():
    import importlib.util
    import subprocess

    # Build the extension if .so is not present.
    so_name = "runtime_kernel_blackwell_fp8_gemm_dense"
    so_path = os.path.join(THIS_DIR, f"{so_name}.cpython-311-x86_64-linux-gnu.so")
    build_dir = os.path.join(THIS_DIR, "build")
    if not os.path.exists(so_path) and not os.path.exists(build_dir):
        print("Building C++ extension...")
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=THIS_DIR,
        )

    try:
        import runtime_kernel_blackwell_fp8_gemm_dense as kernel_mod
    except ImportError:
        print("Building C++ extension (fallback)...")
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=THIS_DIR,
        )
        import runtime_kernel_blackwell_fp8_gemm_dense as kernel_mod

    results = {}

    # Case 1: M=128 = BM (baseline — kernel should be perfect)
    passed, cos, md = run_case(kernel_mod, M=128, N=4096, K=512)
    results["M=128"] = (passed, cos, md)

    # Case 2: M=64 < BM=128 (the suspected bug: chunked prefill with fewer rows)
    passed, cos, md = run_case(kernel_mod, M=64, N=4096, K=512)
    results["M=64"] = (passed, cos, md)

    # Case 3: M=32 (extreme small — further verify boundary behavior)
    passed, cos, md = run_case(kernel_mod, M=32, N=4096, K=512)
    results["M=32"] = (passed, cos, md)

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    all_passed = True
    for label, (p, cos, md) in results.items():
        status = "PASS" if p else "FAIL"
        print(f"  {label}: {status}  cos={cos:.4f}  max_abs_diff={md:.4f}")
        all_passed = all_passed and p

    if all_passed:
        print("\nAll cases PASSED — kernel is numerically correct for M<BM.")
    else:
        print("\nSome cases FAILED — M<BM case exposes a kernel bug.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
