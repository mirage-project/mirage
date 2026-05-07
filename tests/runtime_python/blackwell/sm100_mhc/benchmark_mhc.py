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


def bench_k2(num_tokens, n, dtype=torch.bfloat16):
    mix_hc = n * n + 2 * n
    mixes = torch.randn(num_tokens, mix_hc, device="cuda", dtype=dtype)
    scale = torch.randn(3, device="cuda", dtype=torch.float32)
    base = torch.randn(mix_hc, device="cuda", dtype=torch.float32)
    h_pre = torch.empty(num_tokens, n, device="cuda", dtype=torch.float32)
    h_post = torch.empty(num_tokens, n, device="cuda", dtype=torch.float32)
    h_res = torch.empty(num_tokens, n * n, device="cuda", dtype=torch.float32)

    kernel_ms = time_ms(lambda: rt.mHC_affine_split_activation(
        mixes, scale, base, h_pre, h_post, h_res, n))
    torch_ms = time_ms(lambda: k2_reference(mixes, scale, base, n),
                       warmup=5, iters=20)
    _print_row("K2", num_tokens, kernel_ms, torch_ms, extra=f"n={n} {dtype}")


def bench_k3(num_tokens, repeat=20):
    res = torch.randn(num_tokens, 4, 4, device="cuda", dtype=torch.float32)
    out = torch.empty_like(res)
    kernel_ms = time_ms(lambda: rt.sinkhorn_sm100(
        res, out, repeat=repeat, eps=1e-9))
    torch_ms = time_ms(lambda: sinkhorn_knopp_torch(res, repeat=repeat, eps=1e-9),
                       warmup=5, iters=20)
    _print_row("K3", num_tokens, kernel_ms, torch_ms,
               extra=f"n=4 iters={repeat}")


def bench_k4(num_tokens, n, c):
    x = torch.randn(num_tokens, n, c, device="cuda", dtype=torch.bfloat16)
    h_pre = torch.rand(num_tokens, n, device="cuda", dtype=torch.float32)
    residual = torch.zeros(num_tokens, c, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(num_tokens, c, device="cuda", dtype=torch.bfloat16)

    kernel_ms = time_ms(lambda: rt.mul_sum_add_sm100(x, h_pre, residual, out, n))
    torch_ms = time_ms(lambda: k4_reference(h_pre, x), warmup=5, iters=20)
    _print_row("K4", num_tokens, kernel_ms, torch_ms, extra=f"n={n} c={c}")


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


def _hc_pre_kernel_call(x, hc_fn_padded_bf16, hc_scale, hc_base, n,
                        sinkhorn_iters, hc_eps, norm_eps,
                        scratch):
    """Run hc_pre using kernels for K1/K2/K3/K4. `scratch` holds preallocated
    buffers including the padded weight + output for K1's tcgen05 GEMM."""
    b, s, n_chk, C = x.shape
    bs = b * s
    nC = n * C
    mix_hc = n * n + 2 * n

    x_flat_fp32 = x.reshape(bs, nC).float().contiguous()
    rt.mHC_rmsnorm(x_flat_fp32, scratch["x_norm_bf16"], eps=norm_eps)
    rt.mHC_linear(scratch["x_norm_bf16"], hc_fn_padded_bf16,
                  scratch["mixes_pad"])
    mixes = scratch["mixes_pad"][:, :mix_hc].contiguous()

    rt.mHC_affine_split_activation(
        mixes, hc_scale, hc_base,
        scratch["h_pre"], scratch["h_post"], scratch["h_res"], n)

    res_mat = scratch["h_res"].reshape(bs, n, n)
    rt.sinkhorn_sm100(res_mat.contiguous(), scratch["comb"],
                      repeat=sinkhorn_iters, eps=hc_eps)

    rt.mul_sum_add_sm100(scratch["x_bs"], scratch["h_pre"],
                         scratch["zero_res"], scratch["f_pre"], n)


def bench_hc_pre(b, s, n, C, sinkhorn_iters=20):
    nC = n * C
    mix_hc = n * n + 2 * n
    bs = b * s

    x = torch.randn(b, s, n, C, device="cuda", dtype=torch.bfloat16)
    hc_fn = torch.randn(mix_hc, nC, device="cuda", dtype=torch.float32) * 0.02
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = torch.randn(mix_hc, device="cuda", dtype=torch.float32) * 0.1
    hc_fn_bf16 = hc_fn.to(torch.bfloat16)
    hc_fn_padded_bf16 = torch.zeros(128, nC, device="cuda",
                                    dtype=torch.bfloat16)
    hc_fn_padded_bf16[:mix_hc] = hc_fn_bf16

    scratch = {
        "h_pre": torch.empty(bs, n, device="cuda", dtype=torch.float32),
        "h_post": torch.empty(bs, n, device="cuda", dtype=torch.float32),
        "h_res": torch.empty(bs, n * n, device="cuda", dtype=torch.float32),
        "comb": torch.empty(bs, n, n, device="cuda", dtype=torch.float32),
        "x_bs": x.reshape(bs, n, C).contiguous(),
        "x_norm_bf16": torch.empty(bs, nC, device="cuda", dtype=torch.bfloat16),
        "mixes_pad": torch.empty(bs, 128, device="cuda", dtype=torch.bfloat16),
        "zero_res": torch.zeros(bs, C, device="cuda", dtype=torch.bfloat16),
        "f_pre": torch.empty(bs, C, device="cuda", dtype=torch.bfloat16),
    }

    kernel_ms = time_ms(lambda: _hc_pre_kernel_call(
        x, hc_fn_padded_bf16, hc_scale, hc_base, n,
        sinkhorn_iters, 1e-9, 1e-6, scratch))
    torch_ms = time_ms(lambda: hc_pre_reference(
        x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters, 1e-9, 1e-6),
        warmup=5, iters=20)
    _print_row("hc_pre", bs, kernel_ms, torch_ms,
               extra=f"b={b} s={s} n={n} C={C}")


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

    print("--- K2: affine + split + activation ---")
    for tokens, n in [(1024, 4), (4096, 4), (4096, 8), (16384, 4)]:
        bench_k2(tokens, n)

    print("--- K3: sinkhorn (4x4) ---")
    for tokens in [1024, 4096, 16384]:
        bench_k3(tokens)

    print("--- K4: weighted sum across n streams ---")
    for tokens, n, c in [(1024, 4, 1024), (4096, 4, 1024), (4096, 4, 4096),
                         (4096, 8, 1024)]:
        bench_k4(tokens, n, c)

    print("--- K5: residual mix + post outer ---")
    for tokens, n, c in [(1024, 4, 1024), (4096, 4, 1024), (4096, 4, 4096),
                         (4096, 8, 1024)]:
        bench_k5(tokens, n, c)

    print("--- hc_pre pipeline ---")
    for b, s, n, C in [(1, 1024, 4, 1024), (1, 4096, 4, 1024),
                       (2, 4096, 4, 4096)]:
        bench_hc_pre(b, s, n, C)

    print("--- hc_post pipeline ---")
    for b, s, n, C in [(1, 1024, 4, 1024), (1, 4096, 4, 1024),
                       (2, 4096, 4, 4096)]:
        bench_hc_post(b, s, n, C)
