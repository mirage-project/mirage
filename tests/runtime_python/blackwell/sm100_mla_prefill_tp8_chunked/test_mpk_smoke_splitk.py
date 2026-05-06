"""MPK end-to-end smoke for chunked prefill via SPLITK + REDUCE pair (per-head MLA).

Allocates Qn/Qp/K_nope/K_rope/V/O + a partial buffer, registers two MPK
tasks (splitk writes partial, reduce combines into O), runs once,
compares to the per-head reference.
"""
import math
import os
import sys

import torch
import mirage
from mirage.core import float32
from mirage.mpk.persistent_kernel import PersistentKernel

D_QK_NOPE = 128
D_QK_ROPE = 64
D_QK = 192
D_V = 128


def torch_reference(qn, qp, k_nope, k_rope, v, q_start, sm_scale):
    B, q_len, H, _ = qn.shape
    kv_len = k_nope.shape[1]
    q = torch.cat([qn, qp], dim=-1).float()
    kr = k_rope.float().expand(B, kv_len, H, D_QK_ROPE)
    k = torch.cat([k_nope.float(), kr], dim=-1)
    vf = v.float()
    scores = torch.einsum("bihd,bjhd->bhij", q, k) * sm_scale
    j = torch.arange(kv_len, device=q.device)
    i = torch.arange(q_len, device=q.device)
    mask = j[None, :] > (q_start + i[:, None])
    scores.masked_fill_(mask[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", probs, vf)
    return out.to(qn.dtype)


def main():
    H = 16
    B = 1
    q_len = 256
    kv_len = 2048
    q_start = kv_len - q_len
    num_splits = 4
    nqb = (q_len + 63) // 64

    device = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)
    q_nope = torch.randn(B * q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
    q_pe = torch.randn(B * q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
    k_nope = torch.randn(B * kv_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
    k_rope = torch.randn(B * kv_len, 1, D_QK_ROPE, dtype=dt, device=device) * 0.2
    v = torch.randn(B * kv_len, H, D_V, dtype=dt, device=device) * 0.2
    o = torch.zeros(B * q_len, H, D_V, dtype=dt, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_num_batched_tokens=q_len,
        max_num_batched_requests=B,
        max_seq_length=kv_len,
    )
    pk = PersistentKernel(**params)

    q_nope_dt = pk.attach_input(q_nope, name="q_nope")
    q_pe_dt = pk.attach_input(q_pe, name="q_pe")
    k_nope_dt = pk.attach_input(k_nope, name="k_nope")
    k_rope_dt = pk.attach_input(k_rope, name="k_rope")
    v_dt = pk.attach_input(v, name="v")
    o_dt = pk.attach_input(o, name="o")
    # MPK-managed intermediate buffer (cuda_tensor). 2D shape: total floats
    # in (rows=num_splits*B*nqb*H, cols=BM*(D_V+4)).
    partial_dt = pk.new_tensor(
        dims=(num_splits * B * nqb * H, 64 * (D_V + 4)),
        dtype=float32,
        name="partial",
        io_category="cuda_tensor",
    )

    # SPLITK task: produces partial.
    pk.mla_prefill_tp8_chunked_splitk_layer(
        q_nope=q_nope_dt, q_pe=q_pe_dt,
        k_nope=k_nope_dt, k_rope=k_rope_dt, v=v_dt,
        partial=partial_dt,
        mla_params=(H, q_len, kv_len, q_start, num_splits),
        grid_dim=(H, nqb * num_splits, B),
        block_dim=(128, 1, 1),
    )
    # REDUCE task: combines partial → O.
    pk.mla_prefill_tp8_chunked_reduce_layer(
        partial=partial_dt, output=o_dt,
        mla_params=(H, q_len, num_splits),
        grid_dim=(H, nqb, B),
        block_dim=(256, 1, 1),
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    print("compiling...", flush=True)
    pk.compile(output_dir=folder)

    cu_path = os.path.join(folder, "test_rank0.cu")
    if os.path.exists(cu_path):
        with open(cu_path) as f:
            src = f.read()
        for marker in [
            "TASK_MLA_PREFILL_TP8_CHUNKED_SPLITK_SM100",
            "TASK_MLA_PREFILL_TP8_CHUNKED_REDUCE_SM100",
            "mla_prefill_tp8_chunked_splitk_sm100_task_impl",
            "mla_prefill_tp8_chunked_reduce_sm100_task_impl",
        ]:
            ok = marker in src
            print(f"  {'[+]' if ok else '[ ]'} {marker}")

    print("running...", flush=True)
    pk()
    torch.cuda.synchronize()

    sm_scale = 1.0 / math.sqrt(D_QK)
    o_ref = torch_reference(
        q_nope.view(B, q_len, H, D_QK_NOPE),
        q_pe.view(B, q_len, H, D_QK_ROPE),
        k_nope.view(B, kv_len, H, D_QK_NOPE),
        k_rope.view(B, kv_len, 1, D_QK_ROPE),
        v.view(B, kv_len, H, D_V),
        q_start, sm_scale,
    ).view(B * q_len, H, D_V)
    err = (o.float() - o_ref.float()).abs()
    max_err, mean_err = err.max().item(), err.mean().item()
    status = "OK" if max_err < 3e-2 else "FAIL"
    print(f"B={B} q={q_len} kv={kv_len} qs={q_start} splits={num_splits} "
          f"max_err={max_err:.5f} mean_err={mean_err:.5f} [{status}]")
    if max_err >= 3e-2:
        sys.exit(1)


if __name__ == "__main__":
    main()
