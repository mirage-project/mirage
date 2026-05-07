"""E2E smoke: kv_b_proj → chunked attention with vLLM-style strided K_nope/V.

Pipeline:
  1. Allocate compressed `c_latent [B*kv_len, kv_lora_rank=512]` and
     `kv_b_w [H*(qk_nope+v_head)=H*256, kv_lora_rank=512]`.
  2. Compute `kv_combined = c_latent @ kv_b_w.T` and view as
     `[B*kv_len, H, 256]`. K_nope and V are strided views (head stride=256).
  3. Run MPK chunked attention with these views.
  4. Compare to a reference that does the same matmul + attention in fp32.

This validates the vLLM-aligned data path (single fused kv_b_proj + view-split)
without requiring the full FP8 linear inside MPK — the matmul is done in torch
to keep this test focused on the strided attention input.
"""
import math
import os
import sys

import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

D_QK_NOPE = 128
D_QK_ROPE = 64
D_QK = D_QK_NOPE + D_QK_ROPE
D_V = 128
D_FUSED = D_QK_NOPE + D_V  # 256, kv_b_proj per-head output
KV_LORA_RANK = 512


def torch_reference(qn, qp, kv_combined, k_rope, q_start, sm_scale):
    B, q_len, H, _ = qn.shape
    kv_len = kv_combined.shape[1]
    k_nope = kv_combined[..., :D_QK_NOPE].float()
    v = kv_combined[..., D_QK_NOPE:].float()
    q = torch.cat([qn, qp], dim=-1).float()
    kr = k_rope.float().expand(B, kv_len, H, D_QK_ROPE)
    k = torch.cat([k_nope, kr], dim=-1)
    scores = torch.einsum("bihd,bjhd->bhij", q, k) * sm_scale
    j = torch.arange(kv_len, device=q.device)
    i = torch.arange(q_len, device=q.device)
    mask = j[None, :] > (q_start + i[:, None])
    scores.masked_fill_(mask[None, None, :, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", probs, v)
    return out.to(qn.dtype)


def main():
    H = 16
    B = 1
    q_len = 256
    kv_len = 1024
    q_start = kv_len - q_len

    device = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)

    # Step 1: simulate KV cache (compressed c_latent) and kv_b_proj weight.
    c_latent = torch.randn(B * kv_len, KV_LORA_RANK, dtype=dt, device=device) * 0.1
    kv_b_w = torch.randn(H * D_FUSED, KV_LORA_RANK, dtype=dt, device=device) * 0.1

    # Step 2: kv_b_proj GEMM (in torch BF16 here; in production this is FP8).
    kv_combined_2d = (c_latent.float() @ kv_b_w.T.float()).to(dt)  # [B*kv_len, H*256]
    kv_combined = kv_combined_2d.view(B * kv_len, H, D_FUSED)
    k_nope = kv_combined[..., :D_QK_NOPE]   # strided view, head stride=256
    v = kv_combined[..., D_QK_NOPE:]        # strided view, head stride=256

    q_nope = torch.randn(B * q_len, H, D_QK_NOPE, dtype=dt, device=device) * 0.2
    q_pe = torch.randn(B * q_len, H, D_QK_ROPE, dtype=dt, device=device) * 0.2
    k_rope = torch.randn(B * kv_len, 1, D_QK_ROPE, dtype=dt, device=device) * 0.2
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

    pk.mla_prefill_tp8_chunked_layer(
        q_nope=q_nope_dt, q_pe=q_pe_dt,
        k_nope=k_nope_dt, k_rope=k_rope_dt, v=v_dt,
        output=o_dt,
        mla_params=(H, q_len, kv_len, q_start),
        grid_dim=(H, (q_len + 63) // 64, B),
        block_dim=(128, 1, 1),
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    print("compiling...", flush=True)
    pk.compile(output_dir=folder)
    print("running...", flush=True)
    pk.run_test_mode()
    torch.cuda.synchronize()

    sm_scale = 1.0 / math.sqrt(D_QK)
    o_ref = torch_reference(
        q_nope.view(B, q_len, H, D_QK_NOPE),
        q_pe.view(B, q_len, H, D_QK_ROPE),
        kv_combined.view(B, kv_len, H, D_FUSED),
        k_rope.view(B, kv_len, 1, D_QK_ROPE),
        q_start, sm_scale,
    ).view(B * q_len, H, D_V)
    err = (o.float() - o_ref.float()).abs()
    max_err, mean_err = err.max().item(), err.mean().item()
    status = "OK" if max_err < 3e-2 else "FAIL"
    print(f"B={B} q={q_len} kv={kv_len} qs={q_start} H={H} "
          f"max_err={max_err:.5f} mean_err={mean_err:.5f} [{status}]")
    if max_err >= 3e-2:
        sys.exit(1)


if __name__ == "__main__":
    main()
