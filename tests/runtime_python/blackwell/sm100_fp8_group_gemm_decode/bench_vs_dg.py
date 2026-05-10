"""Bench MPK fp8_group_gemm_decode (via wrapper #include the .cuh) vs
DeepGEMM. L2 flush per iter + disable_ue8m0_cast=True (matches the kernel
author's methodology that gave 1.05-1.21x WIN at source bench)."""
import os
import sys
import torch
import deep_gemm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "build", "lib.linux-x86_64-cpython-312"))
import runtime_kernel_fp8_group_gemm_decode as kern  # noqa: E402

sys.path.insert(0, THIS_DIR)
from test_wrapper import make_inputs  # noqa: E402

device = "cuda"


def bench(fn, n_iters=100, warmup=20):
    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    fn()  # JIT compile (DG)
    for _ in range(warmup):
        flush.zero_()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]  # ms


def bench_kernel(MPE, E, K, N):
    A, B, sfa, sfb, D, mi, _, _ = make_inputs(MPE, E, K, N)
    A_u8 = A.view(torch.uint8)
    B_u8 = B.view(torch.uint8)
    fn = lambda: kern.fp8_group_gemm_decode(A_u8, B_u8, sfa, sfb, D, mi)
    return bench(fn)


def bench_dg(MPE, E, K, N):
    M_total = E * MPE
    A_bf16 = torch.randn(M_total, K, dtype=torch.bfloat16, device=device) * 0.5
    B_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.5
    A = A_bf16.to(torch.float8_e4m3fn).contiguous()
    B = B_bf16.to(torch.float8_e4m3fn).contiguous()
    nk = (K + 127) // 128
    sa = torch.ones(M_total, nk, dtype=torch.float32, device=device).contiguous()
    sb = torch.ones(E, (N + 127) // 128, nk, dtype=torch.float32, device=device).contiguous()
    m_indices = torch.arange(M_total, device=device, dtype=torch.int32) // MPE
    D = torch.empty(M_total, N, dtype=torch.bfloat16, device=device)
    fn = lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (A, sa), (B, sb), D, m_indices, disable_ue8m0_cast=True)
    return bench(fn)


def main():
    MPE_sweep = [1, 4, 16, 32, 64, 128, 256, 512, 1024]
    cfgs = []
    for mpe in MPE_sweep:
        cfgs.append((f"gate_up_M{mpe}", mpe, 32, 7168, 4096))
    for mpe in MPE_sweep:
        cfgs.append((f"down_M{mpe}",    mpe, 32, 2048, 7168))
    print(f"{'config':>14} | {'kern us':>9} {'DG us':>8} | {'kern TF':>9} {'DG TF':>9} | {'k/DG':>6}")
    print("-" * 72)
    for name, MPE, E, K, N in cfgs:
        t_k = bench_kernel(MPE, E, K, N)
        t_d = bench_dg(MPE, E, K, N)
        flops = 2.0 * (E * MPE) * N * K
        tf_k = flops / (t_k / 1000.0) / 1e12
        tf_d = flops / (t_d / 1000.0) / 1e12
        ratio = tf_k / tf_d
        print(f"{name:>14} | {t_k*1000:>8.1f} {t_d*1000:>7.1f} | {tf_k:>8.1f} {tf_d:>8.1f} | {ratio:>5.2f}x")


if __name__ == "__main__":
    main()
