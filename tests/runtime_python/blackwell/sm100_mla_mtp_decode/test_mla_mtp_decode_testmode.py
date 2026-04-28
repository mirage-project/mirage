"""
Test: mla_mtp_decode_sm100 via PersistentKernel test_mode.

Builds a single-layer PK with mla_mtp_decode_layer, compiles it, runs once,
and compares the per-split partial attention outputs (bf16) and LSE values
(fp32) against the PyTorch reference in pytorch_reference.py.

Constants are hard-coded in the kernel (DeepSeek V3 MLA): NUM_HEADS=128,
D_K=576, D_V=512, TILE_S=128. Test shape: B=1, Q_LEN=4, KV_LEN=256 →
sk=2 splits and num_head_groups=4 (hpb=32).

Run:
    python tests/runtime_python/blackwell/sm100_mla_mtp_decode/test_mla_mtp_decode_testmode.py
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# Make sibling import work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import (
    mla_mtp_decode_ref,
    NUM_HEADS,
    D_K,
    D_V,
)


def test_mla_mtp_decode_testmode():
    device = "cuda"
    torch.manual_seed(42)

    # Test shape — chosen so the kernel's split-K is exercised.
    batch_size = 1
    q_len = 4
    kv_len = 256                         # → kvt = 2 tiles → sk = 2

    # Mirror the layer's internal derivation.
    hpb = 128 // q_len
    while 128 % hpb != 0:
        hpb -= 1
    num_head_groups = 128 // hpb         # = 4
    num_splits = (kv_len + 128 - 1) // 128  # = 2

    print(f"\n{'='*60}")
    print("Test: mla_mtp_decode_sm100 via PersistentKernel test_mode")
    print(f"  B={batch_size}, Q_LEN={q_len}, KV_LEN={kv_len}")
    print(f"  H={NUM_HEADS}, D_K={D_K}, D_V={D_V}")
    print(f"  num_head_groups={num_head_groups}, hpb={hpb}, sk={num_splits}")
    print(f"{'='*60}")

    # Inputs (bf16, contiguous on CUDA).
    q = torch.randn(
        batch_size * q_len * NUM_HEADS, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    kv = torch.randn(
        batch_size * kv_len, D_K,
        device=device, dtype=torch.bfloat16) * 0.1
    q = q.contiguous()
    kv = kv.contiguous()

    # Outputs.
    partial_blocks = batch_size * num_head_groups * num_splits
    output_partial = torch.zeros(
        partial_blocks, D_V * 128, device=device, dtype=torch.bfloat16)
    output_lse = torch.zeros(
        partial_blocks, 128, device=device, dtype=torch.float32)

    # Meta-tensor stubs — the generated task code reads:
    #   q_len_rt = qo_indptr_buffer[bi+1] - qo_indptr_buffer[bi]
    #   kv_len   = (lp - fp - 1) * MPK_PAGE_SIZE + paged_kv_last_page_len_buffer[bi]
    # PersistentKernel defaults to page_size=1 (see __init__ defaults).
    # With PAGE_SIZE=1, encoding kv_len=K requires num_pages=K and last=1 →
    # kv_len_ = (K-1)*1 + 1 = K.
    MPK_PAGE_SIZE = 1
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for bi in range(batch_size):
        qo_indptr[bi + 1] = qo_indptr[bi] + q_len
    num_pages = kv_len  # since PAGE_SIZE==1
    last_page_len = 1
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for bi in range(batch_size):
        paged_kv_indptr[bi + 1] = paged_kv_indptr[bi] + num_pages
    paged_kv_last = torch.full(
        (batch_size,), last_page_len, dtype=torch.int32, device=device)

    # PersistentKernel setup.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(q_len * batch_size, 1)
    params["max_num_batched_requests"] = batch_size
    params["max_seq_length"] = max(kv_len, 256)
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_last_page_len_buffer": paged_kv_last,
    }
    pk = PersistentKernel(**params)

    # Attach inputs and outputs.
    q_dt = pk.attach_input(q, name="q_input")
    kv_dt = pk.attach_input(kv, name="kv_input")
    op_dt = pk.attach_input(output_partial, name="output_partial")
    ol_dt = pk.attach_input(output_lse, name="output_lse")

    # Build layer (block_dim is hard-coded inside the layer).
    pk.mla_mtp_decode_layer(q_dt, kv_dt, op_dt, ol_dt, q_len, kv_len)

    # Compile + run.
    print("Compiling test kernel...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Reference.
    ref_part, ref_lse = mla_mtp_decode_ref(
        q, kv,
        batch_size=batch_size, q_len=q_len, kv_len=kv_len,
        num_head_groups=num_head_groups, num_splits=num_splits)

    # Restrict comparison to "active" slots — kernel writes only
    # tids in [0, hpb*q_len). The remaining tids in [hpb*q_len, 128) are
    # untouched (and irrelevant — reduce reads only those active tids).
    used_tid = hpb * q_len  # = 128 for our config; kernel touches all 128
    # Reshape partials to [blocks, D_V, 128] for slicing.
    out_part_r = output_partial.reshape(partial_blocks, D_V, 128)
    ref_part_r = ref_part.reshape(partial_blocks, D_V, 128)
    out_lse_r = output_lse                # [blocks, 128]
    ref_lse_r = ref_lse

    # Slice to used tids only.
    out_part_used = out_part_r[..., :used_tid].contiguous()
    ref_part_used = ref_part_r[..., :used_tid].contiguous()
    out_lse_used = out_lse_r[..., :used_tid].contiguous()
    ref_lse_used = ref_lse_r[..., :used_tid].contiguous()

    part_diff = (out_part_used.float() - ref_part_used.float()).abs().max().item()
    lse_diff = (out_lse_used - ref_lse_used).abs().max().item()
    print(f"max |partial - ref| = {part_diff}")
    print(f"max |lse - ref|     = {lse_diff}")

    try:
        torch.testing.assert_close(
            out_part_used, ref_part_used, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            out_lse_used, ref_lse_used, rtol=1e-2, atol=1e-2)
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    print("PASSED")
    pk.finalize()


if __name__ == "__main__":
    test_mla_mtp_decode_testmode()
