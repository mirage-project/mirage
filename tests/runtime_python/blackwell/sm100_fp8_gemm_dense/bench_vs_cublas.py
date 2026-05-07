"""Bench MPK fp8_gemm_dense task body via the standalone wrapper (which
#includes the same .cuh that MPK uses) vs cuBLAS torch._scaled_mm.

Wrapper avoids the megakernel scheduler overhead but exercises the exact
__device__ task_impl that MPK runs. Same call pattern that other MPK
runtime_python wrappers (e.g. mla_prefill_tp8_chunked) use for benching
device functions in isolation.

Shapes (DeepSeek V3 TP=8):
  q_b_proj:  M × [1536, 3072]
  kv_b_proj: M × [512,  4096]
  o_proj:    M × [2048, 7168]

Tests M from 1 (decode) to 8192 (long prefill).
"""
import os
import sys
import time
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "build", "lib.linux-x86_64-cpython-312"))
import runtime_kernel_fp8_gemm_dense as kern  # noqa: E402

torch.manual_seed(0)
device = "cuda"


def make_inputs(M, K, N):
    """Make (M,K) FP8 input + scale, (N,K) FP8 weight + scale, (M,N) BF16 out."""
    g = torch.Generator(device=device).manual_seed(M * 1009 + K * 31 + N)
    A = torch.randint(0, 256, (M, K), dtype=torch.uint8, device=device, generator=g)
    B = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=device, generator=g)
    sa = (0.5 + torch.rand(M, K // 128, dtype=torch.float32, device=device, generator=g) * 0.5)
    sb = (0.5 + torch.rand(N // 128, K // 128, dtype=torch.float32, device=device, generator=g) * 0.5)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    return A, B, sa, sb, C


def bench_kernel(M, K, N, n_iters=100, warmup=30):
    A, B, sa, sb, C = make_inputs(M, K, N)
    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    A_fp8 = A.view(torch.float8_e4m3fn)
    B_fp8 = B.view(torch.float8_e4m3fn)
    fn = lambda: kern.fp8_gemm_dense(A_fp8, B_fp8, sa, sb, C)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iters)
    times.sort()
    return times[2]  # median


def bench_cublas(M, K, N, n_iters=100, warmup=30):
    """cuBLAS FP8 GEMM via torch._scaled_mm (Hopper/Blackwell native)."""
    A, B, _, _, _ = make_inputs(M, K, N)
    A_fp8 = A.view(torch.float8_e4m3fn)
    # torch._scaled_mm wants B in K-major (column-major B^T) so C = A @ B.T.
    B_T_fp8 = B.view(torch.float8_e4m3fn)  # already (N, K), works as B^T input
    # Per-tensor scales (pure bench: just give a scalar; doesn't change perf).
    sa = torch.tensor(1.0, dtype=torch.float32, device=device)
    sb = torch.tensor(1.0, dtype=torch.float32, device=device)
    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    # Try torch._scaled_mm variants
    try:
        fn = lambda: torch._scaled_mm(A_fp8, B_T_fp8.t(), scale_a=sa, scale_b=sb,
                                       out_dtype=torch.bfloat16)
        out = fn()  # warmup
    except Exception as e:
        return None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_iters)
    times.sort()
    return times[2]


def main():
    layers = [
        ("q_b_proj",  1536, 3072),
        ("kv_b_proj", 512,  4096),
        ("o_proj",    2048, 7168),
    ]
    Ms = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for name, K, N in layers:
        print(f"\n=== {name}  K={K}  N={N} ===")
        print(f"{'M':>6} | {'kern us':>9} {'cuBLAS us':>10}  speedup  "
              f"{'kern TF':>9} {'cuBLAS TF':>10}")
        print("-" * 70)
        for M in Ms:
            t_k = bench_kernel(M, K, N)
            t_c = bench_cublas(M, K, N)
            flops = 2.0 * M * N * K
            tf_k = flops / (t_k / 1000.0) / 1e12
            tf_c = (flops / (t_c / 1000.0) / 1e12) if t_c else 0.0
            sp = (t_c / t_k) if t_c else 0.0
            cstr_us = f"{t_c*1000:>9.1f}" if t_c else f"{'n/a':>9}"
            cstr_tf = f"{tf_c:>9.1f}" if t_c else f"{'n/a':>9}"
            sp_str = f"{sp:>5.2f}x" if t_c else "  n/a"
            print(f"{M:>6} | {t_k*1000:>8.1f} {cstr_us}  {sp_str}  "
                  f"{tf_k:>8.1f} {cstr_tf}")


if __name__ == "__main__":
    main()
