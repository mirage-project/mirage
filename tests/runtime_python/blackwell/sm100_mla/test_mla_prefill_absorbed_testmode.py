"""Absorbed-format MLA prefill correctness through MPK test_mode.

The production DeepSeek path now feeds prefill with the same decode-format
layout as decode:

  Q:  [Q_LEN, H * (CKV_DIM + KPE_DIM)]
  KV: [max_seq_len, CKV_DIM + KPE_DIM]

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \
        tests/runtime_python/blackwell/sm100_mla/test_mla_prefill_absorbed_testmode.py
"""

import math
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_prefill_ref


def _yarn_sm_scale():
    mscale = 0.1 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(192.0)) * mscale * mscale


def test_mla_prefill_absorbed_testmode():
    device = "cuda"
    torch.manual_seed(11)

    batch_size = 1
    seq_len = 128
    num_heads = 4
    d_ckv = 512
    d_kpe = 64
    d_v = 512
    d_total = d_ckv + d_kpe

    q_fused = (
        torch.randn(seq_len, num_heads * d_total,
                    dtype=torch.bfloat16, device=device) * 0.1
    )
    kv = (
        torch.randn(seq_len, d_total, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    out = torch.zeros(seq_len, num_heads * d_v, dtype=torch.bfloat16,
                      device=device)

    q_view = q_fused.view(seq_len, num_heads, d_total)
    q_nope = q_view[:, :, :d_ckv].contiguous()
    q_pe = q_view[:, :, d_ckv:].contiguous()
    ckv = kv[:, :d_ckv].contiguous()
    kpe = kv[:, d_ckv:].contiguous()
    ref = mla_prefill_ref(
        q_nope.unsqueeze(0),
        q_pe.unsqueeze(0),
        ckv.unsqueeze(0),
        kpe.unsqueeze(0),
        _yarn_sm_scale(),
    ).squeeze(0).reshape(seq_len, num_heads * d_v)

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([seq_len], dtype=torch.int32,
                                          device=device)
    prompt_lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.int64, device=device)
    step = torch.zeros(batch_size, dtype=torch.int32, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = seq_len
    params["max_num_batched_requests"] = batch_size
    params["max_seq_length"] = seq_len
    params["max_num_pages"] = 1
    params["page_size"] = seq_len
    params["meta_tensors"] = {
        "tokens": tokens,
        "step": step,
        "prompt_lengths": prompt_lengths,
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_indices_buffer": paged_kv_indices,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
    }
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q_fused, name="q_fused")
    kv_dt = pk.attach_input(kv, name="kv_absorbed")
    out_dt = pk.attach_input(out, name="out")

    pk.mla_prefill_absorbed_layer(
        q_nope_pe=q_dt,
        kv=kv_dt,
        output=out_dt,
        mla_params=(num_heads, seq_len, d_ckv, d_kpe, d_v),
        grid_dim=(num_heads, (seq_len + 63) // 64, batch_size),
        block_dim=(256, 1, 1),
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    max_diff = (out.float() - ref.float()).abs().max().item()
    mean_diff = (out.float() - ref.float()).abs().mean().item()
    nan_count = torch.isnan(out).sum().item()
    if nan_count > 0:
        nan_rows = (
            torch.isnan(out.view(seq_len, num_heads, d_v))
            .any(dim=-1)
            .nonzero()
            .detach()
            .cpu()
            .tolist()
        )
        print(f"nan rows/heads: {nan_rows[:32]}")
    assert nan_count == 0
    print(f"max abs diff: {max_diff:.6f}")
    print(f"mean abs diff: {mean_diff:.6f}")
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
    print("PASSED: mla_prefill_absorbed_sm100 matches split-layout reference")
    pk.finalize()


if __name__ == "__main__":
    test_mla_prefill_absorbed_testmode()
