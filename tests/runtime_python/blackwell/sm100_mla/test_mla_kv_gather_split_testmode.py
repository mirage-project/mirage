"""
Test ``PersistentKernel.mla_kv_gather_split_layer`` end-to-end through the
full MPK compilation pipeline (test_mode), comparing against the canonical
PyTorch reference in ``pytorch_reference.py``.

Same semantics as ``mla_kv_gather_layer`` but the gather emits TWO dense
buffers (ckv_sep, kpe_sep) instead of one concatenated [S, D_K] buffer.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> conda run -n mirage \
        python tests/runtime_python/blackwell/sm100_mla/test_mla_kv_gather_split_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_kv_gather_split_ref


def test_mla_kv_gather_split_testmode():
    device = "cuda"
    torch.manual_seed(42)

    D_V = 512
    ROPE_DIM = 64
    D_K = D_V + ROPE_DIM
    K_PE_ROW_STRIDE = 128

    batch_size = 1
    page_size = 32
    num_pages_total = 4
    seq_len = 100                 # 32+32+32+4
    num_new_tokens = 8

    # The split kernel uses MPK_MAX_SEQ_LENGTH for the per-request stride into
    # the SEPARATE outputs (ckv_sep[bi*MPK_MAX_SEQ_LENGTH*D_V] etc.). Pick
    # max_seq = seq_len so the [seq_len, D_V] / [seq_len, ROPE_DIM] tensors
    # cover the full window.
    max_seq_length = seq_len

    print(f"\n{'='*60}")
    print(f"Test: mla_kv_gather_split_layer test_mode")
    print(f"  S={seq_len}, num_new={num_new_tokens}, "
          f"page_size={page_size}, D_K={D_K}, D_V={D_V}")

    # ----- Inputs -----
    c_latent_new = torch.randn(num_new_tokens, D_V, dtype=torch.bfloat16,
                               device=device) * 0.1
    k_pe_new_full = torch.zeros(num_new_tokens, K_PE_ROW_STRIDE,
                                dtype=torch.bfloat16, device=device)
    k_pe_new_full[:, :ROPE_DIM] = torch.randn(num_new_tokens, ROPE_DIM,
                                              dtype=torch.bfloat16,
                                              device=device) * 0.1

    paged_cache = torch.zeros(num_pages_total, page_size, D_K,
                              dtype=torch.bfloat16, device=device)
    page_indices_list = list(range(num_pages_total))
    kv_start_pos = seq_len - num_new_tokens
    for seq_pos in range(kv_start_pos):
        page_idx = page_indices_list[seq_pos // page_size]
        pos_in_page = seq_pos % page_size
        paged_cache[page_idx, pos_in_page, :] = (
            torch.randn(D_K, dtype=torch.bfloat16, device=device) * 0.1
        )

    # Outputs: per-request slabs of [max_seq, D_V] and [max_seq, ROPE_DIM].
    ckv_sep = torch.zeros(batch_size * max_seq_length, D_V,
                          dtype=torch.bfloat16, device=device)
    kpe_sep = torch.zeros(batch_size * max_seq_length, ROPE_DIM,
                          dtype=torch.bfloat16, device=device)

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
    params["max_seq_length"] = max_seq_length
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
    ckv_sep_dt = pk.attach_input(ckv_sep, name="ckv_sep")
    kpe_sep_dt = pk.attach_input(kpe_sep, name="kpe_sep")

    pk.mla_kv_gather_split_layer(
        c_latent_new=c_latent_dt,
        k_pe_new=k_pe_dt,
        paged_cache=paged_dt,
        ckv_sep=ckv_sep_dt,
        kpe_sep=kpe_sep_dt,
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
    ckv_ref, kpe_ref = mla_kv_gather_split_ref(
        c_latent_new=c_latent_new,
        k_pe_new=k_pe_new_full,
        paged_cache=paged_cache_ref,
        page_indices=torch.tensor(page_indices_list[:num_pages_in_seq],
                                  dtype=torch.int32, device=device),
        seq_len=seq_len,
        d_k=D_K, d_v=D_V, page_size=page_size,
    )

    # The kernel writes the per-request slab starting at row 0 (batch=1).
    out_ckv = ckv_sep[:seq_len]
    out_kpe = kpe_sep[:seq_len]

    ckv_diff = (out_ckv.float() - ckv_ref.float()).abs().max().item()
    kpe_diff = (out_kpe.float() - kpe_ref.float()).abs().max().item()
    print(f"  ckv_sep max abs diff: {ckv_diff:.6f}")
    print(f"  kpe_sep max abs diff: {kpe_diff:.6f}")
    torch.testing.assert_close(out_ckv, ckv_ref, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(out_kpe, kpe_ref, rtol=1e-2, atol=2e-3)

    cache_diff = (paged_cache.float() - paged_cache_ref.float()).abs().max().item()
    print(f"  paged_cache (after append) max abs diff: {cache_diff:.6f}")
    torch.testing.assert_close(paged_cache, paged_cache_ref,
                               rtol=1e-2, atol=2e-3)
    print("PASSED: mla_kv_gather_split test_mode matches PyTorch reference")

    pk.finalize()


if __name__ == "__main__":
    test_mla_kv_gather_split_testmode()
