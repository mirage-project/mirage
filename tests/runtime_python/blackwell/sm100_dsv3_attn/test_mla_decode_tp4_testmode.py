"""
Test: DeepSeek-V3 TP=4 decode-attention CORE (mla_mtp_decode_tp4 + reduce) via
PersistentKernel test_mode, on ONE GPU, with NO AllReduce.

This validates the attention compute that sits *between* the two per-layer
AllReduces (the per-rank MLA decode + LSE reduce). The TP=4 decode kernel has
zero world_size/NVSHMEM dependence — it is driven purely by NUM_HEADS=32 shapes
and the runtime qo/paged-kv meta tensors — so a world_size=1 test exercises the
exact TP=4 kernel without any multi-rank / NVSHMEM machinery.

We compile decode_tp4 -> reduce_tp4 and compare the FINAL reduced attention
output (attn_out, [B, q_len, 32, D_V]) against a direct full-attention reference.
Comparing the final (not the internal per-split partials) is layout-agnostic, so
it is independent of the kernel's v_split / num_split / head_group packing.

Run:
    python tests/runtime_python/blackwell/sm100_dsv3_attn/test_mla_decode_tp4_testmode.py
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.deepseek_v3 import tasks as dsv3_tasks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_decode_full_ref, NUM_HEADS, D_K, D_V


def test_mla_decode_tp4_testmode():
    device = "cuda"
    torch.manual_seed(42)

    # bs=1 pure decode (the locked goal). q_len=1 keeps the TP4 query-grouping
    # trivial (qpg=min(4,1)=1, num_groups=1) and passes the decode gate (q_len<=8).
    batch_size = 1
    q_len = 1                                 # bs=1 pure decode (the locked goal)
    kv_len = 256                              # -> num_splits = ceil(256/128) = 2
    # NOTE: as of 2026-06-04 this + the sibling sm100_mla_mtp_decode test see the
    # kernel reading kv_len≈mbt (not 256) — test-mode prepare_next_batch sets up a
    # prefill from prompt_lengths and the passed paged_kv_indptr decode-state is not
    # surviving into the kernel's runtime kv_len. Shared harness issue (the committed
    # generic test fails identically); not a kernel/reference bug. Needs a test-mode
    # decode-state meta fix (step/prompt_lengths) before this asserts cleanly.
    max_seq_length = max(kv_len, 256)

    # Mirror the builder's TP=4 partial-buffer sizing (builder.py:2954-2971):
    #   _qpg = min(4, mbt); num_groups = ceil(mbt/_qpg)
    #   max_splits = ceil(max_seq_length/128)
    #   partial_blocks = mbr * num_groups * max_splits   (NO v_split/head_group factor)
    mbt = q_len * batch_size
    mbr = batch_size
    qpg = min(4, mbt)
    num_groups = (mbt + qpg - 1) // qpg
    max_splits = (max_seq_length + 127) // 128
    partial_blocks = mbr * num_groups * max_splits

    print(f"\n{'='*64}")
    print("Test: DeepSeek-V3 TP=4 decode-attn core (decode_tp4 + reduce_tp4)")
    print(f"  B={batch_size} q_len={q_len} kv_len={kv_len}  H={NUM_HEADS} D_K={D_K} D_V={D_V}")
    print(f"  num_groups={num_groups} max_splits={max_splits} partial_blocks={partial_blocks}")
    print(f"{'='*64}")

    # Inputs (bf16, contiguous). Small magnitude keeps softmax well-conditioned.
    q = (torch.randn(batch_size * q_len * NUM_HEADS, D_K,
                     device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    kv = (torch.randn(batch_size * kv_len, D_K,
                      device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    # Partial + final buffers, exact builder shapes.
    output_partial = torch.zeros(partial_blocks, D_V * 128,
                                 device=device, dtype=torch.bfloat16)
    output_lse = torch.zeros(partial_blocks, 128,
                             device=device, dtype=torch.float32)
    attn_out = torch.zeros(batch_size * q_len, NUM_HEADS * D_V,
                           device=device, dtype=torch.bfloat16)

    # Meta-tensor stubs (PAGE_SIZE=1 default): kv_len encoded via num_pages=kv_len,
    # last=1; q_len via qo_indptr. Same pattern as sm100_mla_mtp_decode.
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for bi in range(batch_size):
        qo_indptr[bi + 1] = qo_indptr[bi] + q_len
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for bi in range(batch_size):
        paged_kv_indptr[bi + 1] = paged_kv_indptr[bi] + kv_len   # num_pages = kv_len
    paged_kv_last = torch.full((batch_size,), 1, dtype=torch.int32, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1                  # single GPU; tp4 kernel needs no world_size
    params["max_num_batched_tokens"] = max(mbt, 1)
    params["max_num_batched_requests"] = mbr
    params["max_seq_length"] = max_seq_length
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_last_page_len_buffer": paged_kv_last,
    }
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q, name="q_input")
    kv_dt = pk.attach_input(kv, name="kv_input")
    op_dt = pk.attach_input(output_partial, name="output_partial")
    ol_dt = pk.attach_input(output_lse, name="output_lse")
    out_dt = pk.attach_input(attn_out, name="attn_out")

    # TP=4 decode + reduce (the real production kernels). Do NOT override
    # MPK_MLA_TP4_V_SPLITS — the Python grid v_splits must match the kernel's
    # compile-time constexpr (both default to 8 for max_seq_length < 3072).
    dsv3_tasks.mla_mtp_decode_layer(pk, q_dt, kv_dt, op_dt, ol_dt, q_len, kv_len, tp_size=4)
    dsv3_tasks.mla_mtp_reduce_layer(pk, op_dt, ol_dt, out_dt, q_len, kv_len, tp_size=4)

    print("Compiling...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # Diagnostics: which stage produced output? (decode writes output_partial;
    # reduce writes attn_out). attn_out≈0 with output_partial≠0 ⇒ decode ran but
    # the reduce didn't see the partials (missing decode→reduce ordering).
    print(f"[diag] output_partial |max|={output_partial.float().abs().max().item():.5f}  "
          f"nonzero_frac={(output_partial!=0).float().mean().item():.3f}")
    print(f"[diag] output_lse min/max={output_lse.min().item():.3f}/{output_lse.max().item():.3f}")
    print(f"[diag] attn_out |max|={attn_out.float().abs().max().item():.5f}  "
          f"nonzero_frac={(attn_out!=0).float().mean().item():.3f}")

    # Reference: direct full attention (layout-agnostic ground truth).
    ref = mla_decode_full_ref(q, kv, batch_size, q_len, kv_len)   # [B, q_len, H, D_V]
    print(f"[diag] ref |max|={ref.float().abs().max().item():.5f}")
    got = attn_out.reshape(batch_size, q_len, NUM_HEADS, D_V)

    max_diff = (got.float() - ref.float()).abs().max().item()
    # cosine over the flattened attention output
    cos = torch.nn.functional.cosine_similarity(
        got.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
    print(f"max |attn_out - ref| = {max_diff:.5f}   cos = {cos:.6f}")

    try:
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        assert cos > 0.99, f"cosine {cos} below 0.99"
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    print("PASSED")
    pk.finalize()


if __name__ == "__main__":
    test_mla_decode_tp4_testmode()
