"""Unified mHC test: correctness (vs torch) + benchmark (vs vLLM), at the
DeepSeek-V4 shapes. One script replaces the scattered bench/test/sweep files.

Ops (n=4 fixed):
  pre           ours mHC_pre (fused k1+k2)          vs vLLM pre (gemm + big_fuse)
  post          ours mhc_post                        vs vLLM mhc_post_tilelang
  post_pre      ours mHC_post_pre_v2 (fused)          vs vLLM fused (mhc_fused+tail)
  post_then_pre ours mhc_post THEN mHC_pre (separate) vs vLLM fused post_pre

Configs: c=4096 (V4-Flash), c=7168 (V4-Pro);  t in {1,2,4,8,16,32,1024,2k,4k,8k,16k}.

vLLM timings come from a .pt bundle produced by vllm/run_tilelang.py in the
mhc_cmp env. This script auto-(re)generates it if missing/stale.

    python mhc_benchmark.py                 # correctness + benchmark, all ops
    python mhc_benchmark.py --no-vllm       # correctness + ours-only timing
    python mhc_benchmark.py --ops pre post  # subset
"""
import argparse
import os
import subprocess
import sys

import torch
import runtime_kernel_blackwell_mhc as rt

DEV = "cuda"
N = 4
TS = (1, 2, 4, 8, 16, 32, 1024, 2048, 4096, 8192, 16384)
CS = {4096: "V4-Flash", 7168: "V4-Pro"}
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = "/tmp/mhc_v4_sweep.pt"
MHC_CMP_PY = os.environ.get(
    "MHC_TILELANG_PYTHON", "/home/adityar2/miniconda3/envs/mhc_cmp/bin/python")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def time_ms(fn, warmup=20, iters=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def split_k_for(K, nt, max_sk=32):
    target = 1 if nt >= 148 else (148 + nt - 1) // nt
    sk = 1
    for cand in (1, 2, 4, 8, 16, 32):
        if cand > max_sk:
            break
        if K % cand == 0 and cand <= target:
            sk = cand
    return sk


# ---------------------------------------------------------------------------
# torch references (inline; no dependency on vllm/torch_ref)
# ---------------------------------------------------------------------------
def ref_pre(residual_2d, fn2d, scale, base, c, sk=20, he=1e-9, re_=1e-6):
    t = residual_2d.shape[0]
    nC = N * c
    xf = residual_2d.float()
    m = (xf @ fn2d.float().t()) * torch.rsqrt(xf.square().sum(-1, keepdim=True) / nC + re_)
    pre = torch.sigmoid(m[:, :N] * scale[0] + base[:N])
    post = torch.sigmoid(m[:, N:2 * N] * scale[1] + base[N:2 * N]) * 2.0
    cl = m[:, 2 * N:].view(t, N, N) * scale[2] + base[2 * N:].view(1, N, N)
    cm = torch.softmax(cl, -1) + he
    cm = cm / (cm.sum(-2, keepdim=True) + he)
    for _ in range(sk - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + he)
        cm = cm / (cm.sum(-2, keepdim=True) + he)
    f = torch.sum(pre.unsqueeze(-1) * residual_2d.reshape(t, N, c).float(), 1).to(torch.bfloat16)
    return f, post, cm


def ref_post(residual, x, comb, post):
    mixed = torch.einsum("tik,tic->tkc", comb.float(), residual.float())
    return (mixed + post.float().unsqueeze(-1) * x.float().unsqueeze(1)).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# ops: each returns (ours_time_ms, max_relerr) for a given (nt, c)
# ---------------------------------------------------------------------------
def op_pre(nt, c, check):
    K = N * c
    mh = N * N + 2 * N
    torch.manual_seed(0)
    r = (torch.randn(nt, K, device=DEV) * 0.5).to(torch.bfloat16)
    fn = (torch.randn(mh, K, device=DEV) * 0.02).to(torch.bfloat16)
    w = torch.zeros(128, K, device=DEV, dtype=torch.bfloat16); w[:mh] = fn
    sc = torch.tensor([0.7, 0.9, 1.1], device=DEV); ba = torch.randn(mh, device=DEV) * 0.1
    xo = r.reshape(nt, N, c).contiguous()
    mp = torch.empty(nt, 128, device=DEV, dtype=torch.bfloat16); ss = torch.empty(nt, device=DEV)
    f = torch.empty(nt, c, device=DEV, dtype=torch.bfloat16)
    hp = torch.empty(nt, N, device=DEV); cb = torch.empty(nt, N, N, device=DEV)
    call = lambda: rt.mHC_pre(r, w, xo, sc, ba, f, hp, cb, mp, ss, N, c,
                              sinkhorn_repeat=20, sinkhorn_eps=1e-9, rms_eps=1e-6)
    err = float("nan")
    if check:
        call(); torch.cuda.synchronize()
        fr, pr, cr = ref_pre(r, fn, sc, ba, c)
        err = max(relerr(f, fr), relerr(hp, pr), relerr(cb, cr))
    return time_ms(call), err


def op_post(nt, c, check):
    torch.manual_seed(0)
    res = torch.randn(nt, N, c, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(nt, c, device=DEV, dtype=torch.bfloat16)
    p = torch.rand(nt, N, device=DEV); cb = torch.rand(nt, N, N, device=DEV)
    cb = cb / cb.sum(-1, keepdim=True)
    out = torch.empty(nt, N, c, device=DEV, dtype=torch.bfloat16)
    call = lambda: rt.mhc_post(res, x, cb, p, out, N)
    err = float("nan")
    if check:
        call(); torch.cuda.synchronize()
        err = relerr(out, ref_post(res, x, cb, p))
    return time_ms(call), err


def _postpre_inputs(nt, c, sk):
    mh = N * N + 2 * N
    torch.manual_seed(0)
    res = (torch.randn(nt, N, c, device=DEV) * 0.5).to(torch.bfloat16)
    x = (torch.randn(nt, c, device=DEV) * 0.5).to(torch.bfloat16)
    post = torch.rand(nt, N, device=DEV)
    cin = torch.rand(nt, N, N, device=DEV); cin = cin / cin.sum(-1, keepdim=True)
    fn = torch.randn(mh, N, c, device=DEV) * 0.02
    sc = torch.tensor([0.7, 0.9, 1.1], device=DEV); ba = torch.randn(mh, device=DEV) * 0.1
    rn = torch.empty(nt, N, c, device=DEV, dtype=torch.bfloat16)
    op = torch.empty(sk, nt, mh, device=DEV); sp = torch.empty(sk, nt, device=DEV)
    mp = torch.empty(nt, 128, device=DEV, dtype=torch.bfloat16); ss = torch.empty(nt, device=DEV)
    f = torch.empty(nt, c, device=DEV, dtype=torch.bfloat16)
    hp = torch.empty(nt, N, device=DEV); cb = torch.empty(nt, N, N, device=DEV)
    return locals()


def op_post_pre(nt, c, check):
    sk = split_k_for(c, nt, max_sk=16)
    d = _postpre_inputs(nt, c, sk)
    call = lambda: rt.mHC_post_pre_v2(
        d["res"], d["x"], d["cin"], d["post"], d["fn"], d["rn"], d["op"], d["sp"],
        d["mp"], d["ss"], d["sc"], d["ba"], d["f"], d["hp"], d["cb"], N, c,
        split_k=sk, sinkhorn_repeat=20, sinkhorn_eps=1e-9, rms_eps=1e-6)
    err = float("nan")
    if check:
        call(); torch.cuda.synchronize()
        rn_ref = ref_post(d["res"], d["x"], d["cin"], d["post"])
        fr, pr, cr = ref_pre(rn_ref.reshape(nt, N * c), d["fn"].reshape(-1, N * c),
                             d["sc"], d["ba"], c)
        err = max(relerr(d["rn"], rn_ref), relerr(d["f"], fr),
                  relerr(d["hp"], pr), relerr(d["cb"], cr))
    return time_ms(call), err


def op_post_then_pre(nt, c, check):
    # our post (-> next residual) THEN our fused pre, as two separate launches.
    K = N * c; mh = N * N + 2 * N
    torch.manual_seed(0)
    res = torch.randn(nt, N, c, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(nt, c, device=DEV, dtype=torch.bfloat16)
    p = torch.rand(nt, N, device=DEV); cin = torch.rand(nt, N, N, device=DEV)
    cin = cin / cin.sum(-1, keepdim=True)
    fn = (torch.randn(mh, K, device=DEV) * 0.02).to(torch.bfloat16)
    w = torch.zeros(128, K, device=DEV, dtype=torch.bfloat16); w[:mh] = fn
    sc = torch.tensor([0.7, 0.9, 1.1], device=DEV); ba = torch.randn(mh, device=DEV) * 0.1
    rn = torch.empty(nt, N, c, device=DEV, dtype=torch.bfloat16)
    rn_flat = rn.view(nt, K)
    mp = torch.empty(nt, 128, device=DEV, dtype=torch.bfloat16); ss = torch.empty(nt, device=DEV)
    f = torch.empty(nt, c, device=DEV, dtype=torch.bfloat16)
    hp = torch.empty(nt, N, device=DEV); cb = torch.empty(nt, N, N, device=DEV)

    def call():
        rt.mhc_post(res, x, cin, p, rn, N)
        rt.mHC_pre(rn_flat, w, rn, sc, ba, f, hp, cb, mp, ss, N, c,
                   sinkhorn_repeat=20, sinkhorn_eps=1e-9, rms_eps=1e-6)
    err = float("nan")
    if check:
        call(); torch.cuda.synchronize()
        rn_ref = ref_post(res, x, cin, p)
        fr, pr, cr = ref_pre(rn_ref.reshape(nt, K), fn, sc, ba, c)
        err = max(relerr(f, fr), relerr(hp, pr), relerr(cb, cr))
    return time_ms(call), err


OPS = {
    "pre": (op_pre, "pre"),
    "post": (op_post, "post"),
    "post_pre": (op_post_pre, "post_pre"),
    "post_then_pre": (op_post_then_pre, "post_pre"),  # vs vLLM's fused post_pre
}


# ---------------------------------------------------------------------------
# vLLM bundle
# ---------------------------------------------------------------------------
def ensure_bundle():
    if os.path.exists(BUNDLE):
        return load_bundle()
    print(f"vLLM bundle missing; generating via {MHC_CMP_PY} ...", flush=True)
    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "vllm", "_shim") +
               os.pathsep + os.path.join(HERE, "vllm"))
    subprocess.run([MHC_CMP_PY, os.path.join(HERE, "vllm", "run_tilelang.py"),
                    BUNDLE], check=True, env=env)
    return load_bundle()


def load_bundle():
    v = {"pre": {}, "post": {}, "post_pre": {}}
    for r in torch.load(BUNDLE):
        key = (r["num_tokens"], r["H"])
        if r["kind"] == "pre":
            v["pre"][key] = r["t_total_ms"]
        elif r["kind"] == "post":
            v["post"][key] = r["t_ms"]
        elif r["kind"] == "post_pre":
            v["post_pre"][key] = r["t_total_ms"]
    return v


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    ap.add_argument("--no-vllm", action="store_true", help="skip vLLM comparison")
    ap.add_argument("--no-check", action="store_true", help="skip correctness")
    args = ap.parse_args()
    torch.cuda.init()

    vllm = None if args.no_vllm else ensure_bundle()
    any_fail = False

    for c, name in CS.items():
        print(f"\n################  c={c} ({name})  ################")
        for op in args.ops:
            fn, vkey = OPS[op]
            vtag = "fused post_pre" if op == "post_then_pre" else vkey
            hdr = f"{'t':>7s} | {'ours us':>9s}"
            if not args.no_check:
                hdr += f" {'relerr':>9s} {'ok':>3s}"
            if vllm is not None:
                hdr += f" | {'vLLM us':>9s} {'speedup':>8s}"
            print(f"\n--- {op}" + (f"  (vLLM: {vtag})" if vllm is not None else "") + " ---")
            print(hdr)
            for nt in TS:
                t_ms, err = fn(nt, c, not args.no_check)
                line = f"{nt:7d} | {t_ms*1000:9.2f}"
                if not args.no_check:
                    ok = err < 2e-2
                    any_fail |= not ok
                    line += f" {err:9.2e} {'OK' if ok else 'BAD':>3s}"
                if vllm is not None:
                    vl = vllm[vkey].get((nt, c), float("nan")) * 1000
                    sp = vl / (t_ms * 1000) if vl == vl else float("nan")
                    line += f" | {vl:9.2f} {sp:7.2f}x"
                print(line)

    if not args.no_check:
        print("\n", "ALL CORRECTNESS PASSED" if not any_fail else "SOME CORRECTNESS FAILED")
        sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
