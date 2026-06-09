"""
Test ``PersistentKernel.mla_kv_gather_split_layer`` end-to-end through the
full MPK compilation pipeline (test_mode), comparing against the canonical
PyTorch reference in ``pytorch_reference.py``.

Same semantics as ``mla_kv_gather_layer`` but the gather emits TWO dense
buffers (ckv_sep [S, D_V=512], kpe_sep [S, ROPE_DIM=64]) instead of one
concatenated [S, D_K] buffer. Per-request slabs are MPK_MAX_SEQ_LENGTH-strided
(kernel offset = bi * MPK_MAX_SEQ_LENGTH * D_V / ROPE_DIM).

As with the non-split test, the scenario is driven by ``prompt_lengths`` — see
that file's header for why the page tables can't be set directly in test_mode.
Each request bi is a fresh prefill of length prompt_lengths[bi].

Sweep: bs = number of requests ∈ {1,2,4,8,16}.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \
        tests/runtime_python/blackwell/sm100_mla/test_mla_kv_gather_split_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_kv_gather_split_ref

D_V = 512
ROPE_DIM = 64
D_K = D_V + ROPE_DIM
K_PE_ROW_STRIDE = 128
PAGE_SIZE = 16

BS_LIST = [1, 2, 4, 8, 16]
SEQ_LEN = 40                  # 16+16+8; q_len > 8 => fresh-prefill gather


def _run_case(bs):
    device = "cuda"
    torch.manual_seed(7 + bs)

    prompt_lengths = [SEQ_LEN] * bs
    mbt = sum(prompt_lengths)
    pages_per = [(L + PAGE_SIZE - 1) // PAGE_SIZE for L in prompt_lengths]
    max_num_pages = sum(pages_per) + 1
    max_seq_length = max(prompt_lengths)

    print(f"\n{'='*60}")
    print(f"Test: mla_kv_gather_split_layer test_mode  bs={bs}")
    print(f"  S={SEQ_LEN}, page_size={PAGE_SIZE}, D_K={D_K}, D_V={D_V}, "
          f"mbt={mbt}, pages={max_num_pages}")

    # ----- Inputs -----
    c_latent_new = torch.randn(mbt, D_V, dtype=torch.bfloat16,
                               device=device) * 0.1
    k_pe_new = torch.zeros(mbt, K_PE_ROW_STRIDE,
                           dtype=torch.bfloat16, device=device)
    k_pe_new[:, :ROPE_DIM] = torch.randn(mbt, ROPE_DIM,
                                         dtype=torch.bfloat16,
                                         device=device) * 0.1

    paged_cache = torch.randn(max_num_pages, PAGE_SIZE, D_K,
                              dtype=torch.bfloat16, device=device) * 0.1

    # Outputs: per-request slabs of [max_seq, D_V] and [max_seq, ROPE_DIM].
    ckv_sep = torch.zeros(bs * max_seq_length, D_V,
                          dtype=torch.bfloat16, device=device)
    kpe_sep = torch.zeros(bs * max_seq_length, ROPE_DIM,
                          dtype=torch.bfloat16, device=device)

    paged_cache_ref = paged_cache.clone()

    tokens = torch.zeros(bs, max_seq_length, dtype=torch.int64, device=device)
    prompt_lengths_t = torch.tensor(prompt_lengths, dtype=torch.int32,
                                    device=device)

    # ----- Build PersistentKernel -----
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = mbt
    params["max_num_batched_requests"] = bs
    params["max_seq_length"] = max_seq_length
    params["max_num_pages"] = max_num_pages
    params["page_size"] = PAGE_SIZE
    params["meta_tensors"] = {
        "tokens": tokens,
        "prompt_lengths": prompt_lengths_t,
    }
    pk = PersistentKernel(**params)

    c_latent_dt = pk.attach_input(c_latent_new, name="c_latent_new")
    k_pe_dt = pk.attach_input(k_pe_new, name="k_pe_new")
    paged_dt = pk.attach_input(paged_cache, name="paged_cache")
    ckv_sep_dt = pk.attach_input(ckv_sep, name="ckv_sep")
    kpe_sep_dt = pk.attach_input(kpe_sep, name="kpe_sep")

    pk.mla_kv_gather_split_layer(
        c_latent_new=c_latent_dt,
        k_pe_new=k_pe_dt,
        paged_cache=paged_dt,
        ckv_sep=ckv_sep_dt,
        kpe_sep=kpe_sep_dt,
        mla_params=(D_K, D_V, PAGE_SIZE),
        grid_dim=(bs, 1, 1),
        block_dim=(128, 1, 1),
    )

    print("Compiling...")
    folder = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # ----- PyTorch reference (per request) -----
    page_start = 0
    tok_start = 0
    max_ckv_diff = 0.0
    max_kpe_diff = 0.0
    for bi in range(bs):
        L = prompt_lengths[bi]
        n_pages = pages_per[bi]
        page_indices = torch.arange(page_start, page_start + n_pages,
                                    dtype=torch.int32, device=device)
        c_bi = c_latent_new[tok_start:tok_start + L]
        k_bi = k_pe_new[tok_start:tok_start + L]

        ckv_ref, kpe_ref = mla_kv_gather_split_ref(
            c_latent_new=c_bi,
            k_pe_new=k_bi,
            paged_cache=paged_cache_ref,
            page_indices=page_indices,
            seq_len=L,
            d_k=D_K, d_v=D_V, page_size=PAGE_SIZE,
        )

        out_ckv = ckv_sep[bi * max_seq_length: bi * max_seq_length + L]
        out_kpe = kpe_sep[bi * max_seq_length: bi * max_seq_length + L]
        ckv_diff = (out_ckv.float() - ckv_ref.float()).abs().max().item()
        kpe_diff = (out_kpe.float() - kpe_ref.float()).abs().max().item()
        max_ckv_diff = max(max_ckv_diff, ckv_diff)
        max_kpe_diff = max(max_kpe_diff, kpe_diff)

        page_start += n_pages
        tok_start += L

    cache_diff = (paged_cache.float() - paged_cache_ref.float()).abs().max().item()

    print(f"  ckv_sep max abs diff: {max_ckv_diff:.6f}")
    print(f"  kpe_sep max abs diff: {max_kpe_diff:.6f}")
    print(f"  paged_cache (after append) max abs diff: {cache_diff:.6f}")

    assert max_ckv_diff == 0.0, f"bs={bs}: ckv_sep mismatch ({max_ckv_diff})"
    assert max_kpe_diff == 0.0, f"bs={bs}: kpe_sep mismatch ({max_kpe_diff})"
    assert cache_diff == 0.0, f"bs={bs}: paged_cache mismatch ({cache_diff})"
    print(f"PASS bs={bs}")

    pk.finalize()
    return max_ckv_diff, max_kpe_diff, cache_diff


def test_mla_kv_gather_split_testmode():
    for bs in BS_LIST:
        _run_case(bs)


if __name__ == "__main__":
    results = []
    for bs in BS_LIST:
        ckv, kpe, cache = _run_case(bs)
        results.append((bs, ckv, kpe, cache))
    print(f"\n{'='*60}")
    print("MLA_KV_GATHER_SPLIT SUMMARY")
    for bs, ckv, kpe, cache in results:
        print(f"  bs={bs:2d}: ckv={ckv:.6f} kpe={kpe:.6f} cache={cache:.6f} PASS")
    print(f"ALL PASS ({len(results)}/{len(results)})")
