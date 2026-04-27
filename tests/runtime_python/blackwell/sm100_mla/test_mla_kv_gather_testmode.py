"""
Test ``PersistentKernel.mla_kv_gather_layer`` end-to-end through the full MPK
compilation pipeline (test_mode), comparing against the canonical PyTorch
reference in ``pytorch_reference.py``.

Semantics (from include/mirage/persistent_kernel/tasks/blackwell/
mla_kv_cache_gather_sm100.cuh):
  1. Append ``num_new_tokens`` rows of c_latent_new + k_pe_new to the paged
     cache at slots [seq_len - num_new_tokens : seq_len].
  2. Gather the full sequence into a contiguous [seq_len, D_K] buffer
     (D_K = D_V + ROPE_DIM).

Run:
    CUDA_VISIBLE_DEVICES=<gpu> conda run -n mirage \
        python tests/runtime_python/blackwell/sm100_mla/test_mla_kv_gather_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_kv_gather_ref


def test_mla_kv_gather_testmode():
    device = "cuda"
    torch.manual_seed(42)

    # DeepSeek V3 dims; small batch / page count for a fast test.
    D_V = 512                      # latent / value dim
    ROPE_DIM = 64
    D_K = D_V + ROPE_DIM           # 576
    K_PE_ROW_STRIDE = 128          # padded layout used by DeepSeek V3 builder

    batch_size = 1
    page_size = 32
    num_pages_total = 4            # cache capacity; single request uses all 4
    seq_len = 100                  # spans 4 pages: 32+32+32+4 = 100
    num_new_tokens = 8             # last 8 tokens are "new" — appended this step

    print(f"\n{'='*60}")
    print(f"Test: mla_kv_gather_layer test_mode")
    print(f"  S={seq_len}, num_new={num_new_tokens}, "
          f"page_size={page_size}, D_K={D_K}, D_V={D_V}")

    # ----- Inputs -----
    # c_latent_new: [num_new_tokens, D_V]; k_pe_new: [num_new_tokens, 128]
    # (real ROPE data in first 64 cols, rest is zero pad — the kernel only
    # reads ROPE_DIM cols, so we leave the padding deterministic at zero.)
    c_latent_new = torch.randn(num_new_tokens, D_V, dtype=torch.bfloat16,
                               device=device) * 0.1
    k_pe_new_full = torch.zeros(num_new_tokens, K_PE_ROW_STRIDE,
                                dtype=torch.bfloat16, device=device)
    k_pe_new_full[:, :ROPE_DIM] = torch.randn(num_new_tokens, ROPE_DIM,
                                              dtype=torch.bfloat16,
                                              device=device) * 0.1

    # Paged cache: pre-fill the slots for the (seq_len - num_new_tokens)
    # already-cached tokens with random data; the new-token slots will be
    # overwritten by both the kernel and the reference.
    paged_cache = torch.zeros(num_pages_total, page_size, D_K,
                              dtype=torch.bfloat16, device=device)
    # Fill the already-resident range so the gather has something meaningful
    # to read for the older tokens.
    page_indices_list = list(range(num_pages_total))  # 1:1 mapping for test
    kv_start_pos = seq_len - num_new_tokens
    for seq_pos in range(kv_start_pos):
        page_idx = page_indices_list[seq_pos // page_size]
        pos_in_page = seq_pos % page_size
        paged_cache[page_idx, pos_in_page, :] = (
            torch.randn(D_K, dtype=torch.bfloat16, device=device) * 0.1
        )

    # Output: contiguous_kv [batch * max_seq, D_K]; for batch=1 with
    # max_seq=seq_len, this is [seq_len, D_K]. The kernel writes only the
    # first seq_len rows for this request.
    contiguous_kv = torch.zeros(seq_len, D_K, dtype=torch.bfloat16,
                                device=device)

    # Reference works on a clone (its append step mutates paged_cache).
    paged_cache_ref = paged_cache.clone()

    # ----- meta tensors -----
    qo_indptr = torch.tensor([0, num_new_tokens], dtype=torch.int32,
                             device=device)
    last_page_len = seq_len - (seq_len // page_size) * page_size
    if last_page_len == 0:
        last_page_len = page_size
    num_pages_in_seq = (seq_len + page_size - 1) // page_size
    paged_kv_indptr = torch.tensor([0, num_pages_in_seq], dtype=torch.int32,
                                   device=device)
    paged_kv_indices = torch.tensor(page_indices_list[:num_pages_in_seq],
                                    dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32,
                                          device=device)

    # ----- Build PersistentKernel -----
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_new_tokens
    params["max_num_batched_requests"] = batch_size
    params["max_seq_length"] = seq_len
    params["max_num_pages"] = num_pages_total
    params["page_size"] = page_size
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_indices_buffer": paged_kv_indices,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
    }
    pk = PersistentKernel(**params)

    c_latent_dt = pk.attach_input(c_latent_new, name="c_latent_new")
    k_pe_dt = pk.attach_input(k_pe_new_full, name="k_pe_new")
    paged_dt = pk.attach_input(paged_cache, name="paged_cache")
    out_dt = pk.attach_input(contiguous_kv, name="contiguous_kv")

    pk.mla_kv_gather_layer(
        c_latent_new=c_latent_dt,
        k_pe_new=k_pe_dt,
        paged_cache=paged_dt,
        contiguous_kv=out_dt,
        mla_params=(D_K, D_V, page_size),
        grid_dim=(batch_size, 1, 1),
        block_dim=(128, 1, 1),
    )

    print("Compiling...")
    folder = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder)
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # ----- PyTorch reference -----
    ref_out = mla_kv_gather_ref(
        c_latent_new=c_latent_new,
        k_pe_new=k_pe_new_full,
        paged_cache=paged_cache_ref,
        page_indices=torch.tensor(page_indices_list[:num_pages_in_seq],
                                  dtype=torch.int32, device=device),
        seq_len=seq_len,
        d_k=D_K, d_v=D_V, page_size=page_size,
    )

    max_diff = (contiguous_kv.float() - ref_out.float()).abs().max().item()
    print(f"  contiguous_kv max abs diff: {max_diff:.6f}")
    torch.testing.assert_close(contiguous_kv, ref_out, rtol=1e-2, atol=2e-3)

    # Also check the in-place append matches.
    cache_diff = (paged_cache.float() - paged_cache_ref.float()).abs().max().item()
    print(f"  paged_cache (after append) max abs diff: {cache_diff:.6f}")
    torch.testing.assert_close(paged_cache, paged_cache_ref,
                               rtol=1e-2, atol=2e-3)
    print("PASSED: mla_kv_gather test_mode matches PyTorch reference")

    pk.finalize()


if __name__ == "__main__":
    test_mla_kv_gather_testmode()
