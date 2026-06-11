"""mHC correctness (vs torch) + benchmark (vs vLLM) at DeepSeek-V4 shapes.

Modes (n=4 fixed):
  pre    ours mHC_pre                      vs vLLM pre
  post   ours mHC_post                     vs vLLM post
  4way   post->pre four ways: ours_fused (mHC_post_pre_v2) | ours_2k (mHC_post
         then mHC_pre) | vLLM fused (mhc_fused+tail) | vLLM 2k (pre + post)
  all    pre + post + 4way (default)

Configs: c=4096 (V4-Flash), c=7168 (V4-Pro);  t in {1..16384}.

vLLM timings come from a .pt bundle produced by vllm/run_tilelang.py in the
mhc_cmp env; this script auto-(re)generates it if missing.

    python mhc_benchmark.py                 # all modes, correctness + timing
    python mhc_benchmark.py --mode 4way     # just the post->pre comparison
    python mhc_benchmark.py --no-vllm       # ours-only timing + correctness
"""
import argparse
import os
import subprocess
import sys

import torch
import runtime_kernel_blackwell_mhc as rt

DEV = "cuda"
N = 4
TS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
      1024, 2048, 4096, 8192, 16384)
CS = {4096: "V4-Flash", 7168: "V4-Pro"}
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = "/tmp/mhc_v4_sweep.pt"
MHC_CMP_PY = os.environ.get(
    "MHC_TILELANG_PYTHON", "/home/adityar2/miniconda3/envs/mhc_cmp/bin/python")


def time_ms(fn, warmup=500, iters=100):
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


# ---------------------------------------------------------------------------
# mHC_post_pre_v2 launch heuristics: tile_n (outputs/CTA) and split_k (hidden
# splits). Both control how many CTAs the k1 GEMM launches, which on a 148-SM
# B200 is the whole game: total CTAs ~= ceil(t/tpb) * split_k * (24/tile_n).
#
#   - Low t (decode): tokens alone leave the grid nearly empty, so split the
#     work hard -- small tile_n widens the grid by 24/tile_n and split_k adds
#     hidden-dim parallelism -- until ~the SM count is covered several times.
#   - Prefill (t>=512): tokens already fill the grid many times over, so keep
#     each CTA dense (tile_n=24) and use only light split_k.
#
# Co-tuned by joint (tile_n x split_k) sweep on B200 across c=4096/7168, all
# t in 1..16384: this pair lands within 1.10x of the per-shape optimum and is
# ~1.34x faster (geomean) than the previous tile_n=24-only schedule.
# ---------------------------------------------------------------------------
def split_k_for(c, nt):
    if nt <= 2:
        return 16 if c > 4096 else 8     # tiny t: split hard to fill the grid
    if nt <= 8:
        return 8
    if nt <= 32:
        return 4
    if nt <= 64:
        return 1
    if nt <= 256:
        return 2 if c > 4096 else 1
    return 4 if c <= 4096 else 2         # prefill: light split keeps SMs busy


def tile_n_for(c, nt):
    if nt <= 2:
        return 1                         # widest grid for the emptiest case
    if nt <= 256:
        return 6                         # mid band: 4x more CTAs than full width
    if nt <= 1024 and c > 4096:
        return 6                         # wide c still gains a wide grid here
    return 24                            # prefill: dense per-CTA work


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
    call = lambda: rt.mHC_post(res, x, cb, p, out, N)
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
    fn = (torch.randn(mh, N, c, device=DEV) * 0.02).to(torch.bfloat16)
    sc = torch.tensor([0.7, 0.9, 1.1], device=DEV); ba = torch.randn(mh, device=DEV) * 0.1
    rn = torch.empty(nt, N, c, device=DEV, dtype=torch.bfloat16)
    op = torch.empty(sk, nt, mh, device=DEV); sp = torch.empty(sk, nt, device=DEV)
    mp = torch.empty(nt, 128, device=DEV, dtype=torch.bfloat16); ss = torch.empty(nt, device=DEV)
    f = torch.empty(nt, c, device=DEV, dtype=torch.bfloat16)
    hp = torch.empty(nt, N, device=DEV); cb = torch.empty(nt, N, N, device=DEV)
    return locals()


# Configs our mHC_post_pre_v2 instantiates: tile_n in {1,6,24} (divisors of
# mix_hc=24 the kernel compiles), split_k in {1,2,4,8,16,32}. Mirrors vLLM's
# tile_n x split_k sweep in run_tilelang.run_post_pre so both sides report their
# own best config (apples-to-apples), not one tuned vs one fixed.
POSTPRE_TILE_N = (24, 6, 1)
POSTPRE_SPLIT_K = (1, 2, 4, 8, 16, 32)


def _time_post_pre(nt, c, sk, tn, check):
    d = _postpre_inputs(nt, c, sk)
    call = lambda: rt.mHC_post_pre_v2(
        d["res"], d["x"], d["cin"], d["post"], d["fn"], d["rn"], d["op"], d["sp"],
        d["mp"], d["ss"], d["sc"], d["ba"], d["f"], d["hp"], d["cb"], N, c,
        split_k=sk, sinkhorn_repeat=20, sinkhorn_eps=1e-9, rms_eps=1e-6,
        tile_n=tn)
    err = float("nan")
    if check:
        call(); torch.cuda.synchronize()
        rn_ref = ref_post(d["res"], d["x"], d["cin"], d["post"])
        fr, pr, cr = ref_pre(rn_ref.reshape(nt, N * c), d["fn"].reshape(-1, N * c),
                             d["sc"], d["ba"], c)
        err = max(relerr(d["rn"], rn_ref), relerr(d["f"], fr),
                  relerr(d["hp"], pr), relerr(d["cb"], cr))
    return time_ms(call), err


def op_post_pre(nt, c, check, sweep=False):
    if not sweep:
        sk = split_k_for(c, nt)
        tn = tile_n_for(c, nt)
        t_ms, err = _time_post_pre(nt, c, sk, tn, check)
        return t_ms, err, sk, tn
    # Fair sweep: time every instantiated (tile_n, split_k); keep the fastest.
    best = (float("inf"), float("nan"), -1, -1)
    for tn in POSTPRE_TILE_N:
        for sk in POSTPRE_SPLIT_K:
            try:
                t_ms, err = _time_post_pre(nt, c, sk, tn, check)
            except Exception:
                continue
            if t_ms < best[0]:
                best = (t_ms, err, sk, tn)
    return best


def op_post_then_pre(nt, c, check):
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
        rt.mHC_post(res, x, cin, p, rn, N)
        rt.mHC_pre(rn_flat, w, rn, sc, ba, f, hp, cb, mp, ss, N, c,
                   sinkhorn_repeat=20, sinkhorn_eps=1e-9, rms_eps=1e-6)
    err = float("nan")
    if check:
        call(); torch.cuda.synchronize()
        rn_ref = ref_post(res, x, cin, p)
        fr, pr, cr = ref_pre(rn_ref.reshape(nt, K), fn, sc, ba, c)
        err = max(relerr(f, fr), relerr(hp, pr), relerr(cb, cr))
    return time_ms(call), err


# Standalone ops timed against vLLM's same op. (op fn, vLLM bundle key.)
# post_pre / post_then_pre are driven directly by run_4way, not via this table.
OPS = {
    "pre": (op_pre, "pre"),
    "post": (op_post, "post"),
}


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


def run_single_op(op, vllm, check):
    """pre / post on their own vs vLLM's same op. One line per shape:
        t=    1  c=4096  mirage=  15.5us  vllm=  19.7us  speedup=1.27x
    """
    fn, vkey = OPS[op]
    any_fail = False
    for c in CS:
        for nt in TS:
            t_ms, err = fn(nt, c, check)
            mirage_us = t_ms * 1000
            line = f"t={nt:6d}  c={c:5d}  mirage={mirage_us:8.2f}us"
            if vllm is not None:
                vl = vllm[vkey].get((nt, c), float("nan")) * 1000
                sp = vl / mirage_us if vl == vl else float("nan")
                line += f"  vllm={vl:8.2f}us  speedup={sp:.2f}x"
            if check:
                ok = err < 2e-2
                any_fail |= not ok
                line += f"  relerr={err:.2e} {'OK' if ok else 'BAD'}"
            print(line, flush=True)
    return any_fail


def run_4way(vllm, check, sweep=False):
    """post->pre four ways. ours_fused = mHC_post_pre_v2 (single fused path);
    ours_2k = mHC_post then mHC_pre (two kernels); vllm_fused = mhc_fused+tail;
    vllm_2k = vLLM post + vLLM pre run separately (sum of the two timings).
    Speedups are vs ours_fused. One line per shape. With sweep=True, ours_fused
    picks its best (tile_n, split_k) per shape, matching vLLM's own sweep."""
    any_fail = False
    for c in CS:
        for nt in TS:
            tf_ms, ef, sk, tn = op_post_pre(nt, c, check, sweep=sweep)
            t2_ms, e2 = op_post_then_pre(nt, c, check)
            ours_f = tf_ms * 1000
            ours_2 = t2_ms * 1000
            line = (f"t={nt:6d}  c={c:5d}  tn={tn:2d} sk={sk:2d}  "
                    f"ours_fused={ours_f:8.2f}us  ours_2k={ours_2:8.2f}us")
            if vllm is not None:
                vf = vllm["post_pre"].get((nt, c), float("nan")) * 1000
                vp = vllm["pre"].get((nt, c), float("nan")) * 1000
                vpo = vllm["post"].get((nt, c), float("nan")) * 1000
                v2 = vp + vpo
                line += (f"  vllm_fused={vf:8.2f}us  vllm_2k={v2:8.2f}us"
                         f"  | best_ours/vllm_fused={vf/min(ours_f, ours_2):.2f}x")
            if check:
                ok = max(ef, e2) < 2e-2
                any_fail |= not ok
                line += f"  relerr={max(ef, e2):.2e} {'OK' if ok else 'BAD'}"
            print(line, flush=True)
    return any_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "pre", "post", "4way"],
                    help="all = pre + post + 4way post_pre comparison")
    ap.add_argument("--no-vllm", action="store_true", help="skip vLLM comparison")
    ap.add_argument("--no-check", action="store_true", help="skip correctness")
    ap.add_argument("--sweep-4way", action="store_true",
                    help="ours_fused sweeps (tile_n, split_k) per shape and "
                         "keeps the best, matching vLLM's own sweep (fair cmp)")
    args = ap.parse_args()
    torch.cuda.init()

    vllm = None if args.no_vllm else ensure_bundle()
    check = not args.no_check
    any_fail = False

    if args.mode in ("all", "pre"):
        print("\n==================  pre  (ours mHC_pre vs vLLM pre)  ==============")
        any_fail |= run_single_op("pre", vllm, check)
    if args.mode in ("all", "post"):
        print("\n==================  post (ours mHC_post vs vLLM post)  ============")
        any_fail |= run_single_op("post", vllm, check)
    if args.mode in ("all", "4way"):
        print("\n========  post->pre: ours_fused vs ours_2k vs vllm_fused vs "
              "vllm_2k  ========")
        any_fail |= run_4way(vllm, check, sweep=args.sweep_4way)

    if check:
        print("\n", "ALL CORRECTNESS PASSED" if not any_fail else "SOME CORRECTNESS FAILED")
        sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
