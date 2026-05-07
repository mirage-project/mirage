"""5-way benchmark for hc_pre:
  1. torch eager (hc_pre_reference)
  2. torch.compile (compiled hc_pre_reference)
  3. separate mHC kernels (rmsnorm + linear + K2 + K3 + K4)
  4. fused v1: rmsnorm + linear + fused_v1 (serial inlining; lane-0 sinkhorn)
  5. fused v2: rmsnorm + linear + fused_v2 (32-token CTA batches; 32-lane parallel sinkhorn)

Reports kernel time and speedup vs each baseline.
"""
import os
import sys

import torch
import runtime_kernel_blackwell_mhc as rt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "profile")
if PROFILE_DIR not in sys.path:
    sys.path.insert(0, PROFILE_DIR)

from utils import hc_pre_reference


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


def make_inputs(b, s, n, C, seed=0):
    torch.manual_seed(seed)
    nC = n * C
    mix_hc = n * n + 2 * n
    x = torch.randn(b, s, n, C, device="cuda", dtype=torch.bfloat16)
    hc_fn = (torch.randn(mix_hc, nC, device="cuda", dtype=torch.float32) * 0.02)
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = torch.randn(mix_hc, device="cuda", dtype=torch.float32) * 0.1
    return x, hc_fn, hc_scale, hc_base


def bench_torch_eager(x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters=20,
                      hc_eps=1e-9, norm_eps=1e-6):
    return time_ms(lambda: hc_pre_reference(
        x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters, hc_eps, norm_eps))


def bench_torch_compile(x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters=20,
                        hc_eps=1e-9, norm_eps=1e-6):
    fn = torch.compile(
        hc_pre_reference, mode="max-autotune-no-cudagraphs", fullgraph=True
    )
    # Warmup compiles + autotunes.
    for _ in range(5):
        fn(x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters, hc_eps, norm_eps)
    torch.cuda.synchronize()
    return time_ms(lambda: fn(x, hc_fn, hc_scale, hc_base, n,
                              sinkhorn_iters, hc_eps, norm_eps),
                   warmup=10, iters=100)


def bench_separate(x, hc_fn, hc_scale, hc_base, n, sinkhorn_iters=20,
                   hc_eps=1e-9, norm_eps=1e-6):
    """Pipeline using separate kernel launches (current approach)."""
    b, s, n_chk, C = x.shape
    bs = b * s
    nC = n * C
    mix_hc = n * n + 2 * n
    hc_fn_bf16 = hc_fn.to(torch.bfloat16)
    hc_fn_padded = torch.zeros(128, nC, device="cuda", dtype=torch.bfloat16)
    hc_fn_padded[:mix_hc] = hc_fn_bf16

    x_flat_fp32 = x.reshape(bs, nC).float().contiguous()
    x_norm_bf16 = torch.empty(bs, nC, device="cuda", dtype=torch.bfloat16)
    mixes_pad = torch.empty(bs, 128, device="cuda", dtype=torch.bfloat16)
    h_pre = torch.empty(bs, n, device="cuda", dtype=torch.float32)
    h_post = torch.empty(bs, n, device="cuda", dtype=torch.float32)
    h_res = torch.empty(bs, n * n, device="cuda", dtype=torch.float32)
    comb = torch.empty(bs, n, n, device="cuda", dtype=torch.float32)
    x_bs = x.reshape(bs, n, C).contiguous()
    zero_res = torch.zeros(bs, C, device="cuda", dtype=torch.bfloat16)
    f_pre = torch.empty(bs, C, device="cuda", dtype=torch.bfloat16)

    def run():
        rt.mHC_rmsnorm(x_flat_fp32, x_norm_bf16, eps=norm_eps)
        rt.mHC_linear(x_norm_bf16, hc_fn_padded, mixes_pad)
        mixes = mixes_pad[:, :mix_hc].contiguous()
        rt.mHC_affine_split_activation(
            mixes, hc_scale, hc_base, h_pre, h_post, h_res, n)
        res_mat = h_res.reshape(bs, n, n).contiguous()
        rt.sinkhorn_sm100(res_mat, comb, repeat=sinkhorn_iters, eps=hc_eps)
        rt.mul_sum_add_sm100(x_bs, h_pre, zero_res, f_pre, n)
        return f_pre, h_post, comb

    return time_ms(run)


def _bench_fused(x, hc_fn, hc_scale, hc_base, n, fused_fn,
                 sinkhorn_iters=20, hc_eps=1e-9, norm_eps=1e-6):
    """Pipeline: rmsnorm + linear + (fused tail). `fused_fn` selects v1 or v2."""
    b, s, n_chk, C = x.shape
    bs = b * s
    nC = n * C
    mix_hc = n * n + 2 * n
    hc_fn_bf16 = hc_fn.to(torch.bfloat16)
    hc_fn_padded = torch.zeros(128, nC, device="cuda", dtype=torch.bfloat16)
    hc_fn_padded[:mix_hc] = hc_fn_bf16

    x_flat_fp32 = x.reshape(bs, nC).float().contiguous()
    x_norm_bf16 = torch.empty(bs, nC, device="cuda", dtype=torch.bfloat16)
    mixes_pad = torch.empty(bs, 128, device="cuda", dtype=torch.bfloat16)
    x_bs = x.reshape(bs, n, C).contiguous()
    f_pre = torch.empty(bs, C, device="cuda", dtype=torch.bfloat16)
    h_post = torch.empty(bs, n, device="cuda", dtype=torch.float32)
    comb = torch.empty(bs, n, n, device="cuda", dtype=torch.float32)

    def run():
        rt.mHC_rmsnorm(x_flat_fp32, x_norm_bf16, eps=norm_eps)
        rt.mHC_linear(x_norm_bf16, hc_fn_padded, mixes_pad)
        mixes = mixes_pad[:, :mix_hc].contiguous()
        fused_fn(
            mixes, hc_scale, hc_base, x_bs,
            f_pre, h_post, comb, n,
            sinkhorn_repeat=sinkhorn_iters, sinkhorn_eps=hc_eps)
        return f_pre, h_post, comb

    return time_ms(run)


def bench_fused_v1(x, hc_fn, hc_scale, hc_base, n, **kw):
    return _bench_fused(x, hc_fn, hc_scale, hc_base, n,
                        rt.mHC_hc_pre_tail_fused_v1, **kw)


def bench_fused_v2(x, hc_fn, hc_scale, hc_base, n, tokens_per_cta=32, **kw):
    def fn(mixes, scale, base, x_in, f_pre, h_post, comb, n_,
           sinkhorn_repeat, sinkhorn_eps):
        rt.mHC_hc_pre_tail_fused_v2(
            mixes, scale, base, x_in, f_pre, h_post, comb, n_,
            sinkhorn_repeat=sinkhorn_repeat, sinkhorn_eps=sinkhorn_eps,
            tokens_per_cta=tokens_per_cta)
    return _bench_fused(x, hc_fn, hc_scale, hc_base, n, fn, **kw)


def bench_v3(x, hc_fn, hc_scale, hc_base, n, tokens_per_cta=32,
             sinkhorn_iters=20, hc_eps=1e-9, norm_eps=1e-6):
    """v3: single persistent megakernel fusing rmsnorm + linear + tail."""
    b, s, n_chk, C = x.shape
    bs = b * s
    nC = n * C
    mix_hc = n * n + 2 * n
    hc_fn_bf16 = hc_fn.to(torch.bfloat16)
    hc_fn_padded = torch.zeros(128, nC, device="cuda", dtype=torch.bfloat16)
    hc_fn_padded[:mix_hc] = hc_fn_bf16

    x_flat_fp32 = x.reshape(bs, nC).float().contiguous()
    x_norm_scratch = torch.empty(bs, nC, device="cuda", dtype=torch.bfloat16)
    mixes_pad_scratch = torch.empty(bs, 128, device="cuda",
                                    dtype=torch.bfloat16)
    x_bs = x.reshape(bs, n, C).contiguous()
    f_pre = torch.empty(bs, C, device="cuda", dtype=torch.bfloat16)
    h_post = torch.empty(bs, n, device="cuda", dtype=torch.float32)
    comb = torch.empty(bs, n, n, device="cuda", dtype=torch.float32)

    def run():
        rt.mHC_hc_pre_v3(
            x_flat_fp32, x_norm_scratch, hc_fn_padded, mixes_pad_scratch,
            hc_scale, hc_base, x_bs, f_pre, h_post, comb,
            n, C,
            sinkhorn_repeat=sinkhorn_iters, sinkhorn_eps=hc_eps,
            rmsnorm_eps=norm_eps, tokens_per_cta=tokens_per_cta)
        return f_pre, h_post, comb

    return time_ms(run)


def run_case(b, s, n, C, label):
    x, hc_fn, hc_scale, hc_base = make_inputs(b, s, n, C)
    bs = b * s
    eager_ms = bench_torch_eager(x, hc_fn, hc_scale, hc_base, n)
    try:
        compile_ms = bench_torch_compile(x, hc_fn, hc_scale, hc_base, n)
    except Exception as exc:
        compile_ms = float("nan")
        print(f"  torch.compile failed: {exc}")
    sep_ms = bench_separate(x, hc_fn, hc_scale, hc_base, n)
    v1_ms = bench_fused_v1(x, hc_fn, hc_scale, hc_base, n)
    v2_ms_per_tpc = {
        tpc: bench_fused_v2(x, hc_fn, hc_scale, hc_base, n,
                            tokens_per_cta=tpc)
        for tpc in (32, 64, 128)
    }
    v2_best_tpc, v2_best_ms = min(v2_ms_per_tpc.items(), key=lambda kv: kv[1])
    v3_ms_per_tpc = {
        tpc: bench_v3(x, hc_fn, hc_scale, hc_base, n, tokens_per_cta=tpc)
        for tpc in (32, 64, 128)
    }
    v3_best_tpc, v3_best_ms = min(v3_ms_per_tpc.items(), key=lambda kv: kv[1])

    print(
        f"{label}  bs={bs:5d} c={C:5d}\n"
        f"    eager           = {eager_ms:7.4f} ms  (1.00x)\n"
        f"    compile         = {compile_ms:7.4f} ms  ({eager_ms/compile_ms:5.2f}x vs eager)\n"
        f"    separate        = {sep_ms:7.4f} ms  ({eager_ms/sep_ms:5.2f}x vs eager, {compile_ms/sep_ms:5.2f}x vs compile)\n"
        f"    fused v1        = {v1_ms:7.4f} ms\n"
        f"    fused v2 best   = {v2_best_ms:7.4f} ms  (tpc={v2_best_tpc}; "
        f"{eager_ms/v2_best_ms:5.2f}x vs eager, "
        f"{compile_ms/v2_best_ms:5.2f}x vs compile, "
        f"{sep_ms/v2_best_ms:5.2f}x vs separate)\n"
        f"    v3 tpc=32       = {v3_ms_per_tpc[32]:7.4f} ms\n"
        f"    v3 tpc=64       = {v3_ms_per_tpc[64]:7.4f} ms\n"
        f"    v3 tpc=128      = {v3_ms_per_tpc[128]:7.4f} ms\n"
        f"    v3 best         = {v3_best_ms:7.4f} ms  (tpc={v3_best_tpc}; "
        f"{eager_ms/v3_best_ms:5.2f}x vs eager, "
        f"{compile_ms/v3_best_ms:5.2f}x vs compile, "
        f"{sep_ms/v3_best_ms:5.2f}x vs separate, "
        f"{v2_best_ms/v3_best_ms:5.2f}x vs v2)\n"
    )


if __name__ == "__main__":
    torch.cuda.init()
    print("hc_pre 4-way benchmark: torch eager / torch.compile / separate / fused")
    print()
    cases = [
        (1, 1024, 4, 1024),
        (1, 4096, 4, 1024),
        (2, 4096, 4, 4096),
    ]
    for b, s, n, C in cases:
        run_case(b, s, n, C, label=f"hc_pre b={b} s={s} n={n} C={C}")
