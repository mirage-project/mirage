"""Performance benchmark for the MXFP4 1d2d linear kernel (1SM and 2SM).

Mirage-only: torch._scaled_mm on the current build accepts NVFP4 (e4m3 block
scales) but rejects MXFP4 (e8m0 block scales) with "Invalid scaling
configuration", so there is no native MXFP4 GEMM baseline to compare against.
We therefore report the Mirage kernel time and achieved TFLOPS only, picking the
faster of the 1SM / 2SM paths per shape (matching how the NVFP4 table reports a
single path per row).

Path constraints (swapAB: A=weight[N,K], B=input[M,K]):
  1SM: M % 128 == 0, N % 128 == 0
  2SM: M % 256 == 0, N % 256 == 0
Small M (< 128 or not a multiple) is padded up to 128 and run on the 1SM path;
only the first M rows of the output are used, so the reported time is an upper
bound for those shapes (it computes a full 128-row tile regardless).
"""

import argparse

import _runtime_path  # noqa: F401
import torch
import runtime_kernel_blackwell_linear_mxfp4 as runtime_kernel_blackwell

DEVICE = "cuda"

# Supported K values from the wrapper dispatch table.
SUPPORTED_K = {256, 512, 1024, 2048, 4096, 7168, 8192, 16384}

SQUARE = [1024, 2048, 4096, 8192, 16384]
RECT = [
    (4096, 4096, 16384),
    (8192, 8192, 16384),
    (8192, 4096, 4096),
    (16384, 4096, 4096),
    (4096, 8192, 4096),
    (4096, 16384, 4096),
]
SMALL_M = [1, 2, 4, 8, 16, 32, 64, 96, 127]
SMALL_NK = 4096


def benchmark_us(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def quantize(t):
    # mma_n=0 -> interleaved SF layout the 1d2d kernel consumes.
    q, sf = runtime_kernel_blackwell.quantize_mxfp4_sm100(t, 0)
    return q, sf


def _pad_rows(t, target):
    if t.shape[0] >= target:
        return t
    pad = torch.zeros((target - t.shape[0], *t.shape[1:]),
                      dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0)


def make_runner(M, N, K, use_2sm):
    """Build a pre-quantized GEMM closure, or None if the shape is unsupported
    on this path. Padding (for small M) is done once, outside the timed loop."""
    block_m = 256 if use_2sm else 128  # batch (M) divisibility
    block_n = 256 if use_2sm else 128  # output (N) divisibility
    if N % block_n != 0:
        return None

    m_pad = ((M + block_m - 1) // block_m) * block_m

    x = torch.randn((M, K), device=DEVICE, dtype=torch.float32) * 0.5
    w = torch.randn((N, K), device=DEVICE, dtype=torch.float32) * 0.5

    x_q, x_sf = quantize(x)
    w_q, w_sf = quantize(w)

    if m_pad != M:
        x_q = _pad_rows(x_q, m_pad)
        # SF is the interleaved 5D atom layout [rows/128, K/64, 32, 4, 4];
        # padding rows means padding the leading atom dimension.
        atoms = (m_pad + 127) // 128
        x_sf = _pad_rows(x_sf, atoms) if x_sf.dim() and x_sf.shape[0] < atoms else x_sf

    def run():
        return runtime_kernel_blackwell.linear_mxfp4_sm100_no_quantization(
            x_q, x_sf, w_q, w_sf, None, use_2sm
        )

    # Validate it actually launches at these shapes.
    try:
        run()
        torch.cuda.synchronize()
    except Exception:
        return None
    return run


def tflops(M, N, K, us):
    return (2.0 * M * N * K) / (us * 1e-6) / 1e12


def bench_shape(M, N, K, warmup, reps):
    best = None
    for use_2sm, label in [(False, "1SM"), (True, "2SM")]:
        runner = make_runner(M, N, K, use_2sm)
        if runner is None:
            continue
        us = benchmark_us(runner, warmup, reps)
        if best is None or us < best[1]:
            best = (label, us)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    print(f"linear_mxfp4_sm100 | warmup={args.warmup} reps={args.reps} | units=us")

    print("\n### Square (M = N = K)")
    print("| M=N=K | Path | Mirage us | TFLOPS |")
    print("|---:|:---:|---:|---:|")
    for s in SQUARE:
        if s not in SUPPORTED_K:
            continue
        best = bench_shape(s, s, s, args.warmup, args.reps)
        if best:
            label, us = best
            print(f"| {s} | {label} | {us:.1f} | {tflops(s, s, s, us):.0f} |")

    print("\n### Rectangular")
    print("| M | N | K | Path | Mirage us | TFLOPS |")
    print("|---:|---:|---:|:---:|---:|---:|")
    for M, N, K in RECT:
        if K not in SUPPORTED_K:
            continue
        best = bench_shape(M, N, K, args.warmup, args.reps)
        if best:
            label, us = best
            print(f"| {M} | {N} | {K} | {label} | {us:.1f} | {tflops(M, N, K, us):.0f} |")

    print(f"\n### Small-M sweep (N = K = {SMALL_NK})")
    print("| M | Path | Mirage us | TFLOPS |")
    print("|---:|:---:|---:|---:|")
    for M in SMALL_M:
        best = bench_shape(M, SMALL_NK, SMALL_NK, args.warmup, args.reps)
        if best:
            label, us = best
            print(f"| {M} | {label} | {us:.1f} | {tflops(M, SMALL_NK, SMALL_NK, us):.0f} |")


if __name__ == "__main__":
    main()
