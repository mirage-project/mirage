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

from sm100_fp8_scale_layout import FP8_MAX  # noqa: E402


# ---------------------------------------------------------------------------
# Quantize helpers for the fp8_gemm_dense scale layout
# (plain float32, NOT packed UE8M0 used by linear_fp8_sm100)
# ---------------------------------------------------------------------------

def quantize_a(a_bf16: torch.Tensor):
    """Quantize A [M, K] to FP8 e4m3 + float32 scale [M, K/128].
    Each scale covers one 1x128 group (per-row, per-128-columns).
    """
    M, K = a_bf16.shape
    assert K % 128 == 0, "K must be multiple of 128"
    nk = K // 128

    a_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    sa = torch.zeros((M, nk), dtype=torch.float32, device=a_bf16.device)

    a_f32 = a_bf16.float()
    for m in range(M):
        for ki in range(nk):
            block = a_f32[m, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            if abs_max == 0.0:
                scale = 1.0
            else:
                scale = abs_max / FP8_MAX
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (block / scale).clamp(
                -FP8_MAX, FP8_MAX
            ).to(torch.float8_e4m3fn)

    return a_fp8, sa


def quantize_b(b_bf16: torch.Tensor):
    """Quantize B [N, K] to FP8 e4m3 + float32 scale [N/128, K/128].
    Each scale covers one 128x128 block.
    """
    N, K = b_bf16.shape
    assert K % 128 == 0 and N % 128 == 0, "N and K must be multiples of 128"
    nb = N // 128
    nk = K // 128

    b_fp8 = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    sb = torch.zeros((nb, nk), dtype=torch.float32, device=b_bf16.device)

    b_f32 = b_bf16.float()
    for bi in range(nb):
        for ki in range(nk):
            block = b_f32[bi * 128:(bi + 1) * 128,
                          ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            if abs_max == 0.0:
                scale = 1.0
            else:
                scale = abs_max / FP8_MAX
            sb[bi, ki] = scale
            b_fp8[bi * 128:(bi + 1) * 128,
                  ki * 128:(ki + 1) * 128] = (block / scale).clamp(
                -FP8_MAX, FP8_MAX
            ).to(torch.float8_e4m3fn)

    return b_fp8, sb


def reference_gemm(a_fp8, sa, b_fp8, sb):
    """Dequant A and B then compute C = A @ B.T in float32, return bf16."""
    M, K = a_fp8.shape
    N    = b_fp8.shape[0]
    nk   = K // 128

    a_f32 = a_fp8.float()
    b_f32 = b_fp8.float()

    # Dequant A: per-row-128-column group
    a_dq = torch.empty(M, K, dtype=torch.float32, device=a_fp8.device)
    for m in range(M):
        for ki in range(nk):
            a_dq[m, ki * 128:(ki + 1) * 128] = (
                a_f32[m, ki * 128:(ki + 1) * 128] * sa[m, ki]
            )

    # Dequant B: per-128x128-block
    b_dq = torch.empty(N, K, dtype=torch.float32, device=b_fp8.device)
    nb   = N // 128
    for bi in range(nb):
        for ki in range(nk):
            b_dq[bi * 128:(bi + 1) * 128,
                 ki * 128:(ki + 1) * 128] = (
                b_f32[bi * 128:(bi + 1) * 128,
                      ki * 128:(ki + 1) * 128] * sb[bi, ki]
            )

    c = torch.matmul(a_dq, b_dq.t())
    return c.to(torch.bfloat16)


def cosine_similarity_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    cos = torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)
    return cos.item()


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
