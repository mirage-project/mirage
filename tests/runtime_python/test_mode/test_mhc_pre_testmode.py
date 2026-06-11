"""Test-mode validation for mHC_pre_block (mHC prenorm pipeline).

Drives the high-level mHC_pre_block through the full MPK compile -> runtime path
at two token counts that exercise both k1 implementations:
  * decode  (bs < 256): CUDA-core GEMM  (TASK_MHC_PRE_K1_SM100)
  * prefill (bs >= 256): tcgen05 + TMA  (TASK_MHC_PRE_K1_PREFILL_SM100)
Both feed the shared k2 tail. Outputs f_pre / h_post / comb are compared to a
torch reference (mirrors ref_pre in the standalone mhc_benchmark).
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

DEV = "cuda"
N = 4


def ref_pre(residual_2d, fn2d, scale, base, c, sk=20, he=1e-9, re_=1e-6):
    t = residual_2d.shape[0]
    nC = N * c
    xf = residual_2d.float()
    m = (xf @ fn2d.float().t()) * torch.rsqrt(
        xf.square().sum(-1, keepdim=True) / nC + re_)
    pre = torch.sigmoid(m[:, :N] * scale[0] + base[:N])
    post = torch.sigmoid(m[:, N:2 * N] * scale[1] + base[N:2 * N]) * 2.0
    cl = m[:, 2 * N:].view(t, N, N) * scale[2] + base[2 * N:].view(1, N, N)
    cm = torch.softmax(cl, -1) + he
    cm = cm / (cm.sum(-2, keepdim=True) + he)
    for _ in range(sk - 1):
        cm = cm / (cm.sum(-1, keepdim=True) + he)
        cm = cm / (cm.sum(-2, keepdim=True) + he)
    f = torch.sum(pre.unsqueeze(-1) * residual_2d.reshape(t, N, c).float(),
                  1).to(torch.bfloat16)
    return f, post, cm


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def run_one(bs, c, tag):
    K = N * c
    mix_hc = N * N + 2 * N
    torch.manual_seed(0)

    # x is the [bs, n, C] residual stream; x_flat is its [bs, n*C] 2D view
    # (same storage) for the k1 GEMM.
    x = (torch.randn(bs, N, c, device=DEV) * 0.5).to(torch.bfloat16).contiguous()
    x_flat = x.view(bs, K)
    fn = (torch.randn(mix_hc, K, device=DEV) * 0.02).to(torch.bfloat16)
    w = torch.zeros(128, K, device=DEV, dtype=torch.bfloat16)
    w[:mix_hc] = fn
    sc = torch.tensor([0.7, 0.9, 1.1], device=DEV, dtype=torch.float32)
    ba = (torch.randn(mix_hc, device=DEV) * 0.1).to(torch.float32)

    f_pre = torch.zeros(bs, c, device=DEV, dtype=torch.bfloat16)
    h_post = torch.zeros(bs, N, device=DEV, dtype=torch.float32)
    comb = torch.zeros(bs, N, N, device=DEV, dtype=torch.float32)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    sc_dt = pk.attach_input(sc, name="sc")
    ba_dt = pk.attach_input(ba, name="ba")
    f_pre_dt = pk.attach_input(f_pre, name="f_pre")
    h_post_dt = pk.attach_input(h_post, name="h_post")
    comb_dt = pk.attach_input(comb, name="comb")

    pk.mHC_pre_block(
        x=x_dt, hc_fn_padded=w_dt,
        hc_scale=sc_dt, hc_base=ba_dt,
        f_pre=f_pre_dt, h_post=h_post_dt, comb=comb_dt,
        sinkhorn_iters=20, tokens_per_cta=32)

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=os.path.join(folder, f"mhc_pre_out_{tag}"))
    pk.run_test_mode()
    torch.cuda.synchronize()

    fr, pr, cr = ref_pre(x_flat, fn, sc, ba, c)
    err = max(relerr(f_pre, fr), relerr(h_post, pr), relerr(comb, cr))
    pk.finalize()
    print(f"[{tag}] bs={bs} c={c}  relerr={err:.2e}  "
          f"{'PASS' if err < 2e-2 else 'FAIL'}")
    return err < 2e-2


def test_mhc_pre_testmode():
    ok = True
    ok &= run_one(bs=64, c=4096, tag="decode")    # bs<256 -> CUDA-core k1
    ok &= run_one(bs=512, c=4096, tag="prefill")  # bs>=256 -> tcgen05 k1
    if not ok:
        sys.exit(1)
    print("ALL mHC_pre_block test-mode checks PASSED")


if __name__ == "__main__":
    test_mhc_pre_testmode()
