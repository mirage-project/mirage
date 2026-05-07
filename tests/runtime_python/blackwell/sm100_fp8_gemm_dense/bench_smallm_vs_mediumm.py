"""Bench MPK fp8_gemm_dense smallm vs mediumm vs cuBLAS at DeepSeek V3
attention-projection shapes (TP=8 per rank).

Both kernels run via the standalone wrapper that #includes the SAME .cuh
MPK uses for codegen. cuBLAS goes through torch._scaled_mm.
"""
import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "build", "lib.linux-x86_64-cpython-312"))
import runtime_kernel_fp8_gemm_dense as kern  # noqa: E402

torch.manual_seed(0)
device = "cuda"


def make_inputs(M, K, N):
    g = torch.Generator(device=device).manual_seed(M * 1009 + K * 31 + N)
    A = torch.randint(0, 256, (M, K), dtype=torch.uint8, device=device, generator=g)
    B = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=device, generator=g)
    sa = (0.5 + torch.rand(M, K // 128, dtype=torch.float32, device=device, generator=g) * 0.5)
    sb = (0.5 + torch.rand(N // 128, K // 128, dtype=torch.float32, device=device, generator=g) * 0.5)
    C = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    return A, B, sa, sb, C


def bench(fn, n_iters=100, warmup=30):
    flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
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


def bench_kernel(variant, M, K, N):
    A, B, sa, sb, C = make_inputs(M, K, N)
    A_fp8 = A.view(torch.float8_e4m3fn)
    B_fp8 = B.view(torch.float8_e4m3fn)
    fn_map = {"smallm": kern.fp8_gemm_dense_smallm,
              "mediumm": kern.fp8_gemm_dense_mediumm}
    fn = lambda: fn_map[variant](A_fp8, B_fp8, sa, sb, C)
    return bench(fn)


def bench_cublas(M, K, N):
    A, B, _, _, _ = make_inputs(M, K, N)
    A_fp8 = A.view(torch.float8_e4m3fn)
    B_fp8 = B.view(torch.float8_e4m3fn)
    sa = torch.tensor(1.0, dtype=torch.float32, device=device)
    sb = torch.tensor(1.0, dtype=torch.float32, device=device)
    try:
        fn = lambda: torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=sa, scale_b=sb,
                                       out_dtype=torch.bfloat16)
        fn()
    except Exception:
        return None
    return bench(fn)


_DEEP_GEMM_AVAILABLE = None


def _try_deep_gemm():
    global _DEEP_GEMM_AVAILABLE
    if _DEEP_GEMM_AVAILABLE is None:
        try:
            import deep_gemm  # noqa
            _DEEP_GEMM_AVAILABLE = deep_gemm
        except Exception:
            _DEEP_GEMM_AVAILABLE = False
    return _DEEP_GEMM_AVAILABLE


def bench_deep_gemm(M, K, N):
    """Block-scaled FP8 GEMM via deep_gemm.fp8_gemm_nt — same scale layout
    as our kernel (1×128 act, 128×128 wt, UE8M0). Apples-to-apples."""
    dg = _try_deep_gemm()
    if not dg:
        return None
    A, B, _, _, _ = make_inputs(M, K, N)
    A_fp8 = A.view(torch.float8_e4m3fn).contiguous()
    B_fp8 = B.view(torch.float8_e4m3fn).contiguous()
    # deep_gemm wants (tensor, scale) tuples; scales are float32 [outer, K/128].
    sa = torch.ones(M, K // 128, dtype=torch.float32, device=device).contiguous()
    sb = torch.ones(N // 128, K // 128, dtype=torch.float32, device=device).contiguous()
    D = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    try:
        fn = lambda: dg.fp8_gemm_nt((A_fp8, sa), (B_fp8, sb), D)
        fn()
    except Exception as e:
        print(f"  [deep_gemm M={M} K={K} N={N} err: {type(e).__name__}]")
        return None
    return bench(fn)


def main():
    layers = [
        ("q_b_proj",  1536, 3072),
        ("kv_b_proj", 512,  4096),
        ("o_proj",    2048, 7168),
    ]
    Ms = [1, 16, 128, 256, 512, 1024, 2048, 4096, 8192]

    for name, K, N in layers:
        print(f"\n=== {name}  K={K}  N={N} ===")
        print(f"{'M':>6} | {'smallm us':>9} {'mediumm us':>10} {'DG us':>8} "
              f"{'cuBLAS us':>9} | {'sm vs DG':>9} {'md vs DG':>9}")
        print("-" * 90)
        for M in Ms:
            t_s = bench_kernel("smallm", M, K, N)
            t_m = bench_kernel("mediumm", M, K, N)
            t_dg = bench_deep_gemm(M, K, N)
            t_c = bench_cublas(M, K, N)
            sp_s_dg = (t_dg / t_s) if t_dg else 0.0
            sp_m_dg = (t_dg / t_m) if t_dg else 0.0
            cstr = f"{t_c*1000:>7.1f}" if t_c else f"{'n/a':>7}"
            dgstr = f"{t_dg*1000:>7.1f}" if t_dg else f"{'n/a':>7}"
            print(f"{M:>6} | {t_s*1000:>8.1f} {t_m*1000:>9.1f} {dgstr} "
                  f"{cstr} | {sp_s_dg:>7.2f}x {sp_m_dg:>7.2f}x")


if __name__ == "__main__":
    main()
