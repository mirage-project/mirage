"""Smoke test for layers.attention.single_batch_extend_attention.SingleBatchExtendAttention.

The forward() reference takes separate q/k/v projections, but compile()
takes a single fused [Q|K|V] input row buffer. Mapping the two paths
precisely requires careful tensor reshaping; we instead do a smoke test
that the kernel compiles + runs + produces no NaN/Inf.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.attention.single_batch_extend_attention import (
    SingleBatchExtendAttention,
)


def test_single_batch_extend_attention_smoke():
    # Header include for kernel::single_batch_extend_kernel is now in
    # task_header.cuh (added 2026-05-16). Smoke test exercises full
    # compile + run path.
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    extend_num = 1
    T = extend_num + 1  # number of new Q tokens
    max_seq_len = 64
    batch_size = 1

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    fused_outdim = q_size + kv_size + kv_size

    fused_qkv = torch.randn(T, fused_outdim, dtype=dtype, device=device)
    k_cache = torch.zeros(
        batch_size, max_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device,
    )
    v_cache = torch.zeros(
        batch_size, max_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device,
    )
    # RoPE tables.
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    ))
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    out_buf = torch.zeros(T, num_heads * head_dim, dtype=dtype, device=device)

    m = SingleBatchExtendAttention(
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, layer_idx=0, prefix="sbe_",
    ).to(device=device, dtype=dtype)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_seq_length"] = max_seq_len
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    in_dt = pk.attach_input(fused_qkv, name="sbe_in")
    k_dt = pk.attach_input(k_cache, name="sbe_k")
    v_dt = pk.attach_input(v_cache, name="sbe_v")
    cos_dt = pk.attach_input(cos, name="sbe_cos")
    sin_dt = pk.attach_input(sin, name="sbe_sin")

    with pk.compile_scope():
        _ = m.compile(in_dt, k_dt, v_dt, cos_dt, sin_dt, output=out_buf)

    print("Compiling SingleBatchExtendAttention (smoke)...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    if out_buf.isnan().any() or out_buf.isinf().any():
        print("FAILED: out_buf contains NaN/Inf")
        pk.finalize()
        sys.exit(1)
    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print("PASSED: SingleBatchExtendAttention smoke (no crash, no NaN/Inf)")
    pk.finalize()


if __name__ == "__main__":
    test_single_batch_extend_attention_smoke()
