"""PA->PB: dflash_attention_layer (K3) via MPK test-mode (full megakernel path).

Validates the 7-file task wiring (enum/register/graph/python) end-to-end. Feeds
pre-normed/roped q,k,v (random) and compares to dflash_attention_core.

Run: CUDA_VISIBLE_DEVICES=2 python test_dflash_attn_testmode.py
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import dflash_attention_core  # noqa: E402

NQ, NKV, D = 64, 8, 128


def run(B, ctx_len, sliding_window):
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    T = ctx_len + B
    q = torch.randn(B, NQ, D, dtype=dtype, device=device)
    k = torch.randn(T, NKV, D, dtype=dtype, device=device)
    v = torch.randn(T, NKV, D, dtype=dtype, device=device)
    ref = dflash_attention_core(q, k, v, sliding_window, NQ, NKV, D)

    q2 = q.reshape(B, NQ * D).contiguous()
    ck2 = k[:ctx_len].reshape(ctx_len, NKV * D).contiguous()
    cv2 = v[:ctx_len].reshape(ctx_len, NKV * D).contiguous()
    bk2 = k[ctx_len:].reshape(B, NKV * D).contiguous()
    bv2 = v[ctx_len:].reshape(B, NKV * D).contiguous()
    out = torch.zeros(B, NQ * D, dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = B
    params["max_num_batched_requests"] = 1
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q2, name="q")
    ck_dt = pk.attach_input(ck2, name="ck"); cv_dt = pk.attach_input(cv2, name="cv")
    bk_dt = pk.attach_input(bk2, name="bk"); bv_dt = pk.attach_input(bv2, name="bv")
    o_dt = pk.attach_input(out, name="o")
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.dflash_attention_layer(q=q_dt, ctx_k=ck_dt, ctx_v=cv_dt, blk_k=bk_dt, blk_v=bv_dt,
                              output=o_dt, grid_dim=(1, 1, 1), block_dim=block_dim,
                              sliding_window=sliding_window, head_dim=D)
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()
    pk.finalize()

    out3 = out.reshape(B, NQ, D)
    err = (out3.float() - ref.float()).abs().max().item()
    rel = err / max(ref.float().abs().max().item(), 1e-6)
    ok = rel < 0.02
    print(f"B={B} ctx_len={ctx_len} sw={sliding_window}: maxerr {err:.4f} "
          f"relmax {rel:.4f} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys as _s
    case = _s.argv[1] if len(_s.argv) > 1 else "full"
    if case == "full":
        ok = run(8, 16, 0)
    elif case == "sw":
        ok = run(8, 2100, 2048)
    print("PASSED" if ok else "FAILED")
    assert ok
