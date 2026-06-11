"""Test-mode validation for mHC_post_block (mHC post mixing).

Drives the high-level mHC_post_block through the full MPK compile -> runtime
path and compares the output y to a torch reference:
    y[k] = post[k] * x + sum_t comb[t, k] * residual[t]
(comb NOT transposed; matches the torch hc_post convention).
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

DEV = "cuda"
N = 4


def ref_post(residual, x, comb, post):
    # y[t,k,c] = post[t,k]*x[t,c] + sum_i comb[t,i,k]*residual[t,i,c]
    mixed = torch.einsum("tik,tic->tkc", comb.float(), residual.float())
    return (mixed + post.float().unsqueeze(-1) * x.float().unsqueeze(1)).to(
        torch.bfloat16)


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def run_one(bs, c, tag):
    torch.manual_seed(0)
    residual = torch.randn(bs, N, c, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(bs, c, device=DEV, dtype=torch.bfloat16)
    post = torch.rand(bs, N, device=DEV, dtype=torch.float32)
    cb = torch.rand(bs, N, N, device=DEV, dtype=torch.float32)
    cb = cb / cb.sum(-1, keepdim=True)
    y = torch.zeros(bs, N, c, device=DEV, dtype=torch.bfloat16)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    res_dt = pk.attach_input(residual, name="residual")
    x_dt = pk.attach_input(x, name="x")
    post_dt = pk.attach_input(post, name="post")
    cb_dt = pk.attach_input(cb, name="comb")
    y_dt = pk.attach_input(y, name="y")

    pk.mHC_post_block(
        x=x_dt, residual=res_dt, post=post_dt, comb=cb_dt, y=y_dt)

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=os.path.join(folder, f"mhc_post_out_{tag}"))
    pk.run_test_mode()
    torch.cuda.synchronize()

    err = relerr(y, ref_post(residual, x, cb, post))
    pk.finalize()
    print(f"[{tag}] bs={bs} c={c}  relerr={err:.2e}  "
          f"{'PASS' if err < 2e-2 else 'FAIL'}")
    return err < 2e-2


def test_mhc_post_testmode():
    ok = True
    ok &= run_one(bs=64, c=4096, tag="post64")
    ok &= run_one(bs=512, c=7168, tag="post512")
    if not ok:
        sys.exit(1)
    print("ALL mHC_post_block test-mode checks PASSED")


if __name__ == "__main__":
    test_mhc_post_testmode()
