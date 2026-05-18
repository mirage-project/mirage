"""Standalone correctness check for fp8_gemm_dense_smallm_sm100 at the
gate_up shape that triggers the TP=2 mb_arrive_tx fault inside MPK.

Calls fp8_gemm_dense_smallm_sm100_task_impl<128, 3> directly via the
existing multi_cta_launch wrapper, with M=8 N=18432 K=7168 num_workers=128
— matching the TP=2 DSv3 dense MLP gate_up shape that crashes MPK.

If the standalone kernel PASSES this shape:
   bug is MPK-specific (cooperative-launch, megakernel, scheduler,
   concurrent task interaction).
If the standalone kernel FAILS this shape:
   bug is intrinsic to the kernel at this shape.

Comparison cases:
    qkv_a    M=8  N=2176  K=7168  (TP-replicated, works in MPK at both TP=1 and TP=2)
    o_proj   M=8  N=7168  K=8192  (works in MPK at TP=2)
    gate_up  M=8  N=18432 K=7168  (FAILS in MPK at TP=2 — bug repro)
    gate_up_tp4 M=8 N=9216 K=7168 (TP=4 known-good; expected PASS)
    gate_up_tp1 M=8 N=36864 K=7168 (TP=1 known-good; expected PASS)

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_gemm_dense
    python test_fp8_gemm_dense_gate_up_shape.py
(builds the .so automatically if needed)
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
            scale = (abs_max / FP8_MAX) if abs_max != 0 else 1.0
            sa[m, ki] = scale
            a_fp8[m, ki * 128:(ki + 1) * 128] = (block / scale).clamp(
                -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return a_fp8, sa


def quantize_b(b_bf16: torch.Tensor):
    N, K = b_bf16.shape
    assert N % 128 == 0 and K % 128 == 0
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
            scale = (abs_max / FP8_MAX) if abs_max != 0 else 1.0
            sb[bi, ki] = scale
            b_fp8[bi * 128:(bi + 1) * 128,
                  ki * 128:(ki + 1) * 128] = (block / scale).clamp(
                -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return b_fp8, sb


def reference_gemm(a_fp8, sa, b_fp8, sb):
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


def cosine_similarity_2d(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def run_case(kernel_mod, M, N, K, num_workers, seed=42, label=None):
    label = label or f"M={M}, N={N}, K={K}, num_workers={num_workers}"
    print(f"\n{'=' * 78}\nCase: {label}  (nn={N // 128})\n{'=' * 78}")
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
    b_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g)
    a_fp8, sa = quantize_a(a_bf16)
    b_fp8, sb = quantize_b(b_bf16)
    ref = reference_gemm(a_fp8, sa, b_fp8, sb)
    output = torch.zeros((M, N), device=device, dtype=torch.bfloat16)
    kernel_mod.fp8_gemm_dense_smallm_multi_cta_launch(
        a_fp8, b_fp8, sa, sb, output, num_workers)
    torch.cuda.synchronize()
    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = cosine_similarity_2d(output, ref)
    print(f"max_abs_diff: {max_diff:.4f}, cosine_sim: {cos:.4f}")
    row_norms = output.float().abs().sum(dim=1)
    nz = int((row_norms == 0).sum().item())
    print(f"zero output rows: {nz}/{M}")
    passed = (cos > 0.99) and (nz == 0)
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed, cos, max_diff


def main():
    import subprocess
    so_name = "runtime_kernel_blackwell_fp8_gemm_dense"
    so_path = os.path.join(THIS_DIR, f"{so_name}.cpython-311-x86_64-linux-gnu.so")
    wrapper_cu = os.path.join(THIS_DIR, "runtime_kernel_wrapper_sm100.cu")
    needs_rebuild = (not os.path.exists(so_path)) or (
        os.path.getmtime(wrapper_cu) > os.path.getmtime(so_path))
    if needs_rebuild:
        print("Building C++ extension...")
        import shutil
        bd = os.path.join(THIS_DIR, "build")
        if os.path.exists(bd):
            shutil.rmtree(bd)
        if os.path.exists(so_path):
            os.remove(so_path)
        subprocess.check_call(
            [sys.executable, "setup.py", "build_ext", "--inplace"], cwd=THIS_DIR)

    sys.path.insert(0, THIS_DIR)
    import runtime_kernel_blackwell_fp8_gemm_dense as kernel_mod

    results = {}
    # M=8, K=7168, varying N to test gate_up at different TP sharding.
    # Pick which case to run via env var so a CUDA fault doesn't kill all
    # subsequent cases in one process.
    only = os.environ.get("TEST_GATE_UP_ONLY")  # one of: 2176, 9216, 18432, 36864
    cases = [
        ("qkv_a TP-all (N=2176)", 2176, 128),
        ("gate_up TP=4 (N=9216)", 9216, 128),
        ("gate_up TP=2 (N=18432, BUG REPRO)", 18432, 128),
        ("gate_up TP=1 (N=36864)", 36864, 128),
    ]
    # Allow MPK_FP8_DENSE_NUM_WORKERS sweep for N=18432.
    nw_only = os.environ.get("TEST_NUM_WORKERS")
    if nw_only:
        N = int(only) if only else 18432
        cases = [(f"N={N} nw={nw_only}", N, int(nw_only))]
    K_override = os.environ.get("TEST_K")
    K_value = int(K_override) if K_override else 7168
    # Allow arbitrary N from env (overrides cases).
    n_arbitrary = os.environ.get("TEST_N")
    if n_arbitrary:
        Nv = int(n_arbitrary)
        nw = int(nw_only) if nw_only else 128
        cases = [(f"N={Nv} K={K_value} nw={nw}", Nv, nw)]
    for label, N, nw in cases:
        if only and not n_arbitrary and str(N) != only:
            continue
        results[label] = run_case(
            kernel_mod, M=8, N=N, K=K_value, num_workers=nw)

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    print(f"  {'case':<40s} {'status':>6s}  {'cos':>8s}  {'mad':>8s}")
    all_passed = True
    for label, (p, cos, md) in results.items():
        status = "PASS" if p else "FAIL"
        print(f"  {label:<40s} {status:>6s}  {cos:>8.4f}  {md:>8.4f}")
        all_passed = all_passed and p

    if all_passed:
        print("\nAll PASS — standalone kernel is correct at all shapes.")
        print("Therefore the TP=2 MPK fault is an MPK-context bug, not a kernel bug.")
        return 0
    print("\nSome cases FAILED — kernel has a shape-dependent bug.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
