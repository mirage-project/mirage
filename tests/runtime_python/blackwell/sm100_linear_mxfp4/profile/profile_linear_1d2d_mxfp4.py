"""Multi-shape benchmark for the SM100 MXFP4 linear kernel.

Compares against flashinfer mm_fp4 with the cute-dsl backend (the only flashinfer
backend that currently supports MXFP4 quantization on this host).
"""

import argparse
import warnings

import _runtime_path  # noqa: F401
import torch

import runtime_kernel_blackwell_linear_mxfp4 as runtime_kernel_blackwell

try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False
    warnings.warn("flashinfer not available; skipping flashinfer baseline")

DEVICE = "cuda"
DTYPE_OUT = torch.float32
SMALL_M_MAX = 128

SMALL_SHAPES = [
    (1,    128,  768),
    (2,    128,  768),
    (4,    128,  768),
    (8,    128,  768),
    (16,   128,  768),
    (32,   128,  768),
    (64,   128,  768),
    (128,  128,  768),
]

# Shapes (M, N, K) supported by the large-batch MXFP4 dispatch table in the wrapper.
LARGE_SHAPES = [
    (4096,  128,  768),
    (4096,  128, 1024),
    (4096,  128, 2048),
    (4096,  128, 4096),
    (4096,  256, 1024),
    (4096,  256, 2048),
    (4096,  256, 4096),
    (4096,  512, 1024),
    (4096,  512, 2048),
    (4096,  512, 4096),
    (4096, 1024, 1024),
    (4096, 1024, 2048),
    (4096, 1024, 4096),
    (4096, 2048, 2048),
    (4096, 2048, 4096),
    (4096, 4096, 4096),
    (1024, 1024, 1024),
    (1024, 2048, 2048),
    (1024, 4096, 4096),
    (2048, 2048, 2048),
    (2048, 4096, 4096),
    (8192, 2048, 2048),
    (8192, 4096, 4096),
]
DEFAULT_SHAPES = SMALL_SHAPES + LARGE_SHAPES


def select_mma_n(m: int) -> int:
    if m <= 8:
        return 8
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    if m <= 64:
        return 64
    return 128


def benchmark_us(fn, warmup: int, reps: int) -> float:
    """Time `fn` in a tight loop using CUDA events. Includes per-call dispatch overhead."""
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


def benchmark_graph_us(fn, warmup: int, reps: int, replays_per_capture: int = 32) -> float:
    """Time `fn` via CUDA graph capture + replay. Excludes per-call dispatch overhead.

    Captures `replays_per_capture` invocations into a single graph so the per-replay
    framework cost is amortized further. Returns the average GPU time per call.
    """
    # Warm up so any first-call cudaMalloc/lazy init has already run before capture.
    # cudaMalloc cannot be captured into a graph, so the descriptor cache must be
    # fully initialized before we start.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(replays_per_capture):
            fn()

    # Time graph replays.
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    total_calls = reps * replays_per_capture
    return start.elapsed_time(end) * 1000.0 / total_calls


def custom_runner(m, n, k, residual: bool):
    x = torch.randn((m, k), device=DEVICE, dtype=torch.float32)
    w = torch.randn((n, k), device=DEVICE, dtype=torch.float32)
    mma_n = select_mma_n(m) if m <= SMALL_M_MAX else 0
    x_q, x_sf = runtime_kernel_blackwell.quantize_mxfp4_sm100(x, mma_n)
    w_q, w_sf = runtime_kernel_blackwell.quantize_mxfp4_sm100(w, 0)
    output = torch.empty((m, n), device=DEVICE, dtype=DTYPE_OUT)
    res = torch.randn((m, n), device=DEVICE, dtype=DTYPE_OUT) if residual else None

    def run():
        runtime_kernel_blackwell.linear_mxfp4_sm100_no_quantization(
            x_q, x_sf, w_q[:n], w_sf, res, output
        )

    return run


def flashinfer_runner(m, n, k):
    """flashinfer mm_fp4 with cute-dsl backend on MXFP4 (block_size=32, ue8m0 scales).

    Pre-quantizes the activations and weights so that only the GEMM call is timed.
    """
    x = torch.randn((m, k), device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn((n, k), device=DEVICE, dtype=torch.bfloat16)
    xq, xsf = flashinfer.mxfp4_quantize(x)
    wq, wsf = flashinfer.mxfp4_quantize(w)

    def run():
        return flashinfer.mm_fp4(
            xq, wq.T, xsf, wsf.T,
            alpha=None,
            out_dtype=torch.bfloat16,
            block_size=32,
            use_nvfp4=False,
            backend="cute-dsl",
        )

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small-only", action="store_true",
                        help="Benchmark only the small-batch swapAB shapes")
    parser.add_argument("--large-only", action="store_true",
                        help="Benchmark only the large-batch 1d2d shapes")
    parser.add_argument("--residual", action="store_true",
                        help="Add a residual to the custom kernel (flashinfer baseline never has one)")
    parser.add_argument("--no-flashinfer", action="store_true",
                        help="Skip the flashinfer cute-dsl baseline")
    parser.add_argument("--no-graph", action="store_true",
                        help="Use the event-loop timer instead of CUDA graph replay")
    args = parser.parse_args()

    if args.small_only and args.large_only:
        raise ValueError("--small-only and --large-only are mutually exclusive")

    use_flashinfer = _FLASHINFER_AVAILABLE and not args.no_flashinfer
    torch.manual_seed(args.seed)
    shapes = SMALL_SHAPES if args.small_only else LARGE_SHAPES if args.large_only else DEFAULT_SHAPES

    bench = benchmark_us if args.no_graph else benchmark_graph_us
    mode = "event-loop" if args.no_graph else "cuda-graph-replay"
    print(
        f"linear_mxfp4_sm100 multi-shape benchmark | mode={mode} | warmup={args.warmup} | reps={args.reps} | residual={args.residual} | shapes={len(shapes)}\n"
        f"{'M':>5s} {'N':>5s} {'K':>5s} | {'path':>6s} | {'custom_us':>10s} {'cust_TFLOPS':>11s} | {'fi_us':>9s} {'fi_TFLOPS':>10s} | {'speedup':>7s}"
    )

    for m, n, k in shapes:
        flops = 2.0 * m * n * k
        path = "swapAB" if m <= SMALL_M_MAX else "1d2d"

        cust_run = custom_runner(m, n, k, residual=args.residual)
        cust_us = bench(cust_run, args.warmup, args.reps)
        cust_tflops = flops / (cust_us * 1e-6) / 1e12

        fi_us = float("nan")
        fi_tflops = float("nan")
        speedup = float("nan")
        if use_flashinfer:
            try:
                fi_run = flashinfer_runner(m, n, k)
                fi_us = bench(fi_run, args.warmup, args.reps)
                fi_tflops = flops / (fi_us * 1e-6) / 1e12
                speedup = fi_us / cust_us
            except Exception as e:
                warnings.warn(f"flashinfer M={m} N={n} K={k}: {str(e).splitlines()[0][:120]}")

        print(
            f"{m:5d} {n:5d} {k:5d} | {path:>6s} | {cust_us:10.2f} {cust_tflops:11.1f} | "
            f"{fi_us:9.2f} {fi_tflops:10.1f} | {speedup:7.2f}x"
        )


if __name__ == "__main__":
    main()
