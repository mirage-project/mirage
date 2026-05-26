"""Standalone correctness check for fp8_gemm_dense_smallm_sm100 at N=2176.

Calls `fp8_gemm_dense_smallm_sm100_task_impl<128, 3>` via a multi-CTA
launch (grid = (num_workers, 1, 1)) that matches the production
persistent-kernel pattern, and compares the output against a PyTorch
FP32 dequant + matmul reference.

Result (2026-05-12, see scratch/fp8_dense_smallm_n2176_bug.md):

    case                             status       cos       mad  zero_rows
    baseline_q_a (nn=12)               PASS    1.0000    1.0000          0
    bug_qkv_a   (nn=17)                PASS    1.0000    1.0000          0
    pad_2304    (nn=18)                PASS    1.0000    1.0000          0

The kernel is correct at all three shapes in isolation (cos=1.000, all 128
output rows written, bit-equal to PyTorch FP32 dequant+matmul reference
up to bf16 rounding). The L0 cos=0.97 regression seen in MPK with QKV-a
fusion (N=2176) is therefore NOT a kernel bug — the cause is elsewhere
(MPK wiring, race with concurrent task, dump artifact, …).

This script is kept as a regression check for the FP8 dense small-M
kernel at the production shapes that appear in DSv3 prefill.

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_gemm_dense
    rm -rf build *.so   # force rebuild after wrapper changes
    python setup.py build_ext --inplace
    python test_fp8_gemm_dense_smallm_n2176_regression.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import FP8_MAX  # noqa: E402


def quantize_a(a_bf16: torch.Tensor):
    """Quantize A [M, K] to FP8 e4m3 + float32 scale [M, K/128].
    Each scale covers one 1x128 group (per-row, per-128-columns).
    """
    M, K = a_bf16.shape
    assert K % 128 == 0
    nk = K // 128
    a_fp8 = torch.empty_like(a_bf16, dtype=torch.float8_e4m3fn)
    sa = torch.zeros((M, nk), dtype=torch.float32, device=a_bf16.device)
    a_f32 = a_bf16.float()
    for m in range(M):
        for ki in range(nk):
            block = a_f32[m, ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = (abs_max / FP8_MAX) if abs_max != 0.0 else 1.0
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX)
                .to(torch.float8_e4m3fn))
    return a_fp8, sa


def quantize_b(b_bf16: torch.Tensor):
    """Quantize B [N, K] to FP8 e4m3 + float32 scale [N/128, K/128].
    Each scale covers one 128x128 block.
    """
    N, K = b_bf16.shape
    assert K % 128 == 0 and N % 128 == 0
    nb, nk = N // 128, K // 128
    b_fp8 = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
    sb = torch.zeros((nb, nk), dtype=torch.float32, device=b_bf16.device)
    b_f32 = b_bf16.float()
    for bi in range(nb):
        for ki in range(nk):
            block = b_f32[bi * 128:(bi + 1) * 128,
                          ki * 128:(ki + 1) * 128]
            abs_max = block.abs().max().item()
            scale = (abs_max / FP8_MAX) if abs_max != 0.0 else 1.0
            sb[bi, ki] = scale
            b_fp8[bi * 128:(bi + 1) * 128,
                  ki * 128:(ki + 1) * 128] = (
                (block / scale).clamp(-FP8_MAX, FP8_MAX)
                .to(torch.float8_e4m3fn))
    return b_fp8, sb


def reference_gemm(a_fp8, sa, b_fp8, sb):
    """Dequant A and B, then C = A @ B.T in float32, return bfloat16."""
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    nk = K // 128
    a_f32 = a_fp8.float()
    b_f32 = b_fp8.float()
    a_dq = torch.empty(M, K, dtype=torch.float32, device=a_fp8.device)
    for m in range(M):
        for ki in range(nk):
            a_dq[m, ki * 128:(ki + 1) * 128] = (
                a_f32[m, ki * 128:(ki + 1) * 128] * sa[m, ki])
    b_dq = torch.empty(N, K, dtype=torch.float32, device=b_fp8.device)
    nb = N // 128
    for bi in range(nb):
        for ki in range(nk):
            b_dq[bi * 128:(bi + 1) * 128,
                 ki * 128:(ki + 1) * 128] = (
                b_f32[bi * 128:(bi + 1) * 128,
                      ki * 128:(ki + 1) * 128] * sb[bi, ki])
    return torch.matmul(a_dq, b_dq.t()).to(torch.bfloat16)


def cosine_similarity_2d(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def run_case(kernel_mod, M: int, N: int, K: int, num_workers: int,
             seed: int = 42, label_override: str = None):
    label = label_override or f"M={M}, N={N}, K={K}, num_workers={num_workers}"
    print(f"\n{'=' * 78}")
    print(f"Case: {label}  (nn={N // 128})")
    print(f"{'=' * 78}")

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g)

    a_fp8, sa = quantize_a(a_bf16)
    b_fp8, sb = quantize_b(b_bf16)

    ref = reference_gemm(a_fp8, sa, b_fp8, sb)
    print(f"ref [row 1, cols 0..7]: {ref[1, :8].tolist()}")

    output = torch.zeros((M, N), device=device, dtype=torch.bfloat16)
    kernel_mod.fp8_gemm_dense_smallm_multi_cta_launch(
        a_fp8, b_fp8, sa, sb, output, num_workers)
    torch.cuda.synchronize()

    print(f"out [row 1, cols 0..7]: {output[1, :8].tolist()}")

    # Per-row zero check: which output rows are all-zero?
    row_norms = output.float().abs().sum(dim=1)
    zero_rows = (row_norms == 0).nonzero(as_tuple=True)[0]
    if zero_rows.numel() > 0:
        zr = zero_rows.tolist()
        # Compress consecutive ranges for readability.
        ranges = []
        i = 0
        while i < len(zr):
            j = i
            while j + 1 < len(zr) and zr[j + 1] == zr[j] + 1:
                j += 1
            ranges.append((zr[i], zr[j]))
            i = j + 1
        print(f"  zero rows (total {zero_rows.numel()}): "
              + ", ".join(f"{a}" if a == b else f"{a}..{b}"
                          for a, b in ranges))
    else:
        print(f"  no zero rows in output")

    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = cosine_similarity_2d(output, ref)
    print(f"max_abs_diff: {max_diff:.4f}")
    print(f"cosine_sim:   {cos:.4f}")

    passed = (cos > 0.99) and (zero_rows.numel() == 0)
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed, cos, max_diff, int(zero_rows.numel())


def main():
    import importlib.util
    import subprocess

    so_name = "runtime_kernel_blackwell_fp8_gemm_dense"
    so_path = os.path.join(THIS_DIR, f"{so_name}.cpython-311-x86_64-linux-gnu.so")
    build_dir = os.path.join(THIS_DIR, "build")
    needs_rebuild = (not os.path.exists(so_path)) or (
        os.path.getmtime(__file__) > os.path.getmtime(so_path))
    # Note: also rebuild if the wrapper .cu was edited.
    wrapper_cu = os.path.join(THIS_DIR, "runtime_kernel_wrapper_sm100.cu")
    if (os.path.exists(so_path) and
            os.path.getmtime(wrapper_cu) > os.path.getmtime(so_path)):
        needs_rebuild = True

    if needs_rebuild:
        print("Building C++ extension (wrapper .cu is newer than .so)...")
        # Clean build cache so cpython picks up new symbols.
        import shutil
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        if os.path.exists(so_path):
            os.remove(so_path)
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=THIS_DIR)

    import runtime_kernel_blackwell_fp8_gemm_dense as kernel_mod

    results = {}

    # === Baseline: known-good shapes from production ===
    # q_a path in baseline DSv3: M=128, N=1536 (nn=12), K=7168. Always works.
    results["baseline_q_a (nn=12)"] = run_case(
        kernel_mod, M=128, N=1536, K=7168, num_workers=128,
        label_override="baseline q_a (nn=12, KNOWN GOOD)")

    # === Bug repro: QKV-a fused N=2176 ===
    # Production shape that triggers the row-1..71-zero pattern.
    results["bug_qkv_a (nn=17)"] = run_case(
        kernel_mod, M=128, N=2176, K=7168, num_workers=128,
        label_override="QKV-a fused (nn=17, BUG REPRO)")

    # === Pad verification: nn=18 also fails ===
    # Padding N to 2304 (next 128-multiple after 2176) reproduces the same
    # pattern, ruling out "nn=17 is prime" as the trigger.
    results["pad_2304 (nn=18)"] = run_case(
        kernel_mod, M=128, N=2304, K=7168, num_workers=128,
        label_override="pad N=2304 (nn=18, ALSO FAILS)")

    print(f"\n{'=' * 78}")
    print("Summary")
    print(f"{'=' * 78}")
    print(f"  {'case':<32s} {'status':>6s}  {'cos':>8s}  {'mad':>8s}  {'zero_rows':>9s}")
    all_passed = True
    for label, (p, cos, md, nz) in results.items():
        status = "PASS" if p else "FAIL"
        print(f"  {label:<32s} {status:>6s}  {cos:>8.4f}  {md:>8.4f}  {nz:>9d}")
        all_passed = all_passed and p

    if all_passed:
        print("\nAll cases passed — kernel is correct for this set of shapes.")
        return 0

    print("\nAt least one case FAILED. The kernel has a correctness bug.")
    print("See scratch/fp8_dense_smallm_n2176_bug.md for the full diagnosis.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
