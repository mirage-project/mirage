"""Benchmarks for the mHC kernels (K2, K3, K4, K5) and the end-to-end
hc_pre / hc_post pipelines. Mirrors the style of benchmark_sinkhorn.py."""
import os
import sys

import torch
import runtime_kernel_blackwell_mhc as rt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "profile")
if PROFILE_DIR not in sys.path:
    sys.path.insert(0, PROFILE_DIR)

from utils import (
    hc_post_reference,
    hc_pre_reference,
    k1_reference,
    k2_reference,
    k4_reference,
    k5_reference,
    sinkhorn_knopp_torch,
)


def time_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _print_row(label, num_tokens, kernel_ms, torch_ms, extra=""):
    speedup = torch_ms / kernel_ms if kernel_ms > 0 else float("nan")
    print(
        f"{label:14s} tokens={num_tokens:5d} {extra:20s} "
        f"kernel={kernel_ms:.4f} ms  torch={torch_ms:.4f} ms  "
        f"speedup={speedup:5.2f}x  tokens/ms={num_tokens / kernel_ms:.1f}"
    )


def bench_k1(num_tokens, n, c, dtype=torch.bfloat16):
    """K1 = mHC_rmsnorm + mHC_linear (tcgen05+TMA+TMEM, weight padded to 128)."""
    nC = n * c
    mix_hc = n * n + 2 * n
    x = torch.randn(num_tokens, nC, device="cuda", dtype=torch.float32)
    hc_fn_bf16 = (torch.randn(mix_hc, nC, device="cuda", dtype=torch.float32)
                  * 0.02).to(torch.bfloat16)
    w_pad = torch.zeros(128, nC, device="cuda", dtype=torch.bfloat16)
    w_pad[:mix_hc] = hc_fn_bf16

    x_norm_bf16 = torch.empty(num_tokens, nC, device="cuda",
                              dtype=torch.bfloat16)
    out_pad = torch.empty(num_tokens, 128, device="cuda",
                          dtype=torch.bfloat16)

    def kernel_fn():
        rt.mHC_rmsnorm(x, x_norm_bf16, eps=1e-6)
        rt.mHC_linear(x_norm_bf16, w_pad, out_pad)
        return out_pad[:, :mix_hc]

    kernel_ms = time_ms(kernel_fn)
    torch_ms = time_ms(lambda: k1_reference(x, hc_fn_bf16, 1e-6),
                       warmup=5, iters=20)
    _print_row("K1", num_tokens, kernel_ms, torch_ms,
               extra=f"n={n} c={c} mix_hc={mix_hc} (rms+tcgen05 GEMM)")


def bench_k1_linear_only(num_tokens, n, c):
    """Isolated linear-half timing: mHC_linear vs torch bf16 matmul."""
    nC = n * c
    mix_hc = n * n + 2 * n
    x = torch.randn(num_tokens, nC, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(mix_hc, nC, device="cuda", dtype=torch.bfloat16)
    w_pad = torch.zeros(128, nC, device="cuda", dtype=torch.bfloat16)
    w_pad[:mix_hc] = w
    out_pad = torch.empty(num_tokens, 128, device="cuda",
                          dtype=torch.bfloat16)

    kernel_ms = time_ms(lambda: rt.mHC_linear(x, w_pad, out_pad))
    torch_ms = time_ms(lambda: x @ w.T, warmup=5, iters=20)
    _print_row("K1-lin", num_tokens, kernel_ms, torch_ms,
               extra=f"n={n} c={c} mix_hc={mix_hc} (pad to 128)")


def bench_k1_rmsnorm_only(num_tokens, hidden, dtype=torch.bfloat16):
    """Isolated rmsnorm-half timing (kernel vs torch rsqrt path)."""
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)
    y = torch.empty_like(x)

    kernel_ms = time_ms(lambda: rt.mHC_rmsnorm(x, y, eps=1e-6))

    def torch_rmsnorm():
        rsqrt = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return (x.float() * rsqrt).to(dtype)

    torch_ms = time_ms(torch_rmsnorm, warmup=5, iters=20)
    _print_row("K1-rms", num_tokens, kernel_ms, torch_ms,
               extra=f"hidden={hidden}")


def bench_k3(num_tokens, repeat=20):
    res = torch.randn(num_tokens, 4, 4, device="cuda", dtype=torch.float32)
    out = torch.empty_like(res)
    kernel_ms = time_ms(lambda: rt.sinkhorn_sm100(
        res, out, repeat=repeat, eps=1e-9))
    torch_ms = time_ms(lambda: sinkhorn_knopp_torch(res, repeat=repeat, eps=1e-9),
                       warmup=5, iters=20)
    _print_row("K3", num_tokens, kernel_ms, torch_ms,
               extra=f"n=4 iters={repeat}")


def bench_k5(num_tokens, n, c):
    residual = torch.randn(num_tokens, n, c, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(num_tokens, c, device="cuda", dtype=torch.bfloat16)
    comb = torch.rand(num_tokens, n, n, device="cuda", dtype=torch.float32)
    comb = comb / comb.sum(-1, keepdim=True)
    post = torch.rand(num_tokens, n, device="cuda", dtype=torch.float32)
    out = torch.empty(num_tokens, n, c, device="cuda", dtype=torch.bfloat16)

    kernel_ms = time_ms(lambda: rt.mHC_mul_sum_add_with_outer(
        residual, x, comb, post, out, n))
    torch_ms = time_ms(lambda: k5_reference(residual, x, comb, post),
                       warmup=5, iters=20)
    _print_row("K5", num_tokens, kernel_ms, torch_ms, extra=f"n={n} c={c}")




def bench_hc_post(b, s, n, C):
    bs = b * s
    x = torch.randn(b, s, C, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(b, s, n, C, device="cuda", dtype=torch.bfloat16)
    post = torch.rand(b, s, n, device="cuda", dtype=torch.float32)
    comb = torch.rand(b, s, n, n, device="cuda", dtype=torch.float32)
    comb = comb / comb.sum(-1, keepdim=True)

    residual_bs = residual.reshape(bs, n, C).contiguous()
    x_bs = x.reshape(bs, C).contiguous()
    comb_bs = comb.reshape(bs, n, n).contiguous()
    post_bs = post.reshape(bs, n).contiguous()
    out = torch.empty(bs, n, C, device="cuda", dtype=torch.bfloat16)

    kernel_ms = time_ms(lambda: rt.mHC_mul_sum_add_with_outer(
        residual_bs, x_bs, comb_bs, post_bs, out, n))
    torch_ms = time_ms(lambda: hc_post_reference(x, residual, post, comb),
                       warmup=5, iters=20)
    _print_row("hc_post", bs, kernel_ms, torch_ms,
               extra=f"b={b} s={s} n={n} C={C}")


if __name__ == "__main__":
    torch.cuda.init()

    print("--- K1 rmsnorm half (mHC_rmsnorm kernel only) ---")
    for tokens, hidden in [(1024, 1024), (4096, 1024), (4096, 4096),
                           (4096, 16384)]:
        bench_k1_rmsnorm_only(tokens, hidden)

    print("--- K1 linear half (mHC_linear, tcgen05+TMA+TMEM) ---")
    for tokens, n, c in [(1024, 4, 256), (1024, 4, 1024), (4096, 4, 1024),
                         (4096, 4, 4096)]:
        bench_k1_linear_only(tokens, n, c)

    print("--- K1 full (mHC_rmsnorm + mHC_linear) ---")
    for tokens, n, c in [(1024, 4, 256), (4096, 4, 1024), (4096, 4, 4096)]:
        bench_k1(tokens, n, c)

    print("--- K3: sinkhorn (4x4) ---")
    for tokens in [1024, 4096, 16384]:
        bench_k3(tokens)


    print("--- K5: residual mix + post outer ---")
    for tokens, n, c in [(1024, 4, 1024), (4096, 4, 1024), (4096, 4, 4096),
                         (4096, 8, 1024)]:
        bench_k5(tokens, n, c)


    print("--- hc_post pipeline ---")
    for b, s, n, C in [(1, 1024, 4, 1024), (1, 4096, 4, 1024),
                       (2, 4096, 4, 4096)]:
        bench_hc_post(b, s, n, C)
