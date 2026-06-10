"""
Test ``PersistentKernel.mla_kv_gather_layer`` (legacy DECODE-ONLY variant)
end-to-end through the full MPK compilation pipeline (test_mode), comparing
against the canonical PyTorch reference ``mla_kv_gather_ref`` in
``pytorch_reference.py``.

Semantics (``mla_kv_cache_gather_sm100_task_impl``):
  1. Append ``num_new_tokens`` rows of c_latent_new + k_pe_new to the paged
     cache at slots [seq_len - num_new_tokens : seq_len].
  2. Gather the full sequence into a contiguous [seq_len, D_K] buffer
     (D_K = D_V + ROPE_DIM = 576).

Unlike the unified gather, this layer has no prefill/decode gate: it always
writes the concatenated [S, D_K] ``contiguous_kv`` output (decode layout).

In test_mode the page tables are recomputed from ``prompt_lengths`` by
``prepare_next_batch`` on iter 0.  We use short kv_len values (≤8 tokens)
to stay clearly in the "decode" regime used by the legacy kernel.  Pages are
allocated sequentially from the queue starting at 0, so for request ``bi``
the owned pages are a contiguous range reconstructed in Python below.

Sweep: bs ∈ {1,2,4,8,16} × kv_len ∈ {32,64}  (at most 2 pages each).

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \\
        tests/runtime_python/blackwell/sm100_mla/test_mla_kv_gather_legacy_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_kv_gather_ref

D_V = 512
ROPE_DIM = 64
D_K = D_V + ROPE_DIM          # 576
K_PE_ROW_STRIDE = 128          # padded layout used by DeepSeek V3 builder
PAGE_SIZE = 32                 # legacy default

BS_LIST = [1, 2, 4, 8, 16]
KV_LEN_LIST = [32, 64]         # 1 and 2 full pages respectively


def _run_case(bs, kv_len):
    device = "cuda"
    torch.manual_seed(42 + bs * 7 + kv_len)

    prompt_lengths = [kv_len] * bs
    mbt = sum(prompt_lengths)
    pages_per = [(L + PAGE_SIZE - 1) // PAGE_SIZE for L in prompt_lengths]
    # +1 slack page so the queue never wraps.
    max_num_pages = sum(pages_per) + 1
    max_seq_length = kv_len

    print(f"\n{'='*60}")
    print(f"Test: mla_kv_gather_layer (legacy) bs={bs} kv_len={kv_len}")
    print(f"  page_size={PAGE_SIZE}, D_K={D_K}, D_V={D_V}, "
          f"mbt={mbt}, pages={max_num_pages}")

    # ----- Inputs -----
    # Each request is a fresh prefill: ALL kv_len tokens are "new".
    c_latent_new = torch.randn(mbt, D_V, dtype=torch.bfloat16,
                               device=device) * 0.1
    k_pe_new = torch.zeros(mbt, K_PE_ROW_STRIDE,
                           dtype=torch.bfloat16, device=device)
    k_pe_new[:, :ROPE_DIM] = torch.randn(mbt, ROPE_DIM,
                                         dtype=torch.bfloat16,
                                         device=device) * 0.1

    paged_cache = torch.randn(max_num_pages, PAGE_SIZE, D_K,
                              dtype=torch.bfloat16, device=device) * 0.1

    # Output: per-request slab [S, D_K] at offset bi * max_seq_length.
    contiguous_kv = torch.zeros(bs * max_seq_length, D_K,
                                dtype=torch.bfloat16, device=device)

    # Reference works on a clone (its append step mutates paged_cache).
    paged_cache_ref = paged_cache.clone()

    # meta tensors: prompt_lengths drives page-table allocation.
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
    out_dt = pk.attach_input(contiguous_kv, name="contiguous_kv")

    pk.mla_kv_gather_layer(
        c_latent_new=c_latent_dt,
        k_pe_new=k_pe_dt,
        paged_cache=paged_dt,
        contiguous_kv=out_dt,
        mla_params=(D_K, D_V, PAGE_SIZE),
        grid_dim=(bs, 1, 1),
        block_dim=(128, 1, 1),
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    print("Compiling...")
    pk.compile(output_dir=folder)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # ----- PyTorch reference (per request) -----
    # Pages are allocated sequentially: request bi owns
    # pages [page_start : page_start + pages_per[bi]].
    page_start = 0
    tok_start = 0
    max_ckv_diff = 0.0
    for bi in range(bs):
        L = prompt_lengths[bi]
        n_pages = pages_per[bi]
        page_indices = torch.arange(page_start, page_start + n_pages,
                                    dtype=torch.int32, device=device)
        c_bi = c_latent_new[tok_start:tok_start + L]
        k_bi = k_pe_new[tok_start:tok_start + L]

        ref_out = mla_kv_gather_ref(
            c_latent_new=c_bi,
            k_pe_new=k_bi,
            paged_cache=paged_cache_ref,
            page_indices=page_indices,
            seq_len=L,
            d_k=D_K, d_v=D_V, page_size=PAGE_SIZE,
        )

        out_bi = contiguous_kv[bi * max_seq_length: bi * max_seq_length + L]
        diff = (out_bi.float() - ref_out.float()).abs().max().item()
        max_ckv_diff = max(max_ckv_diff, diff)

        page_start += n_pages
        tok_start += L

    cache_diff = (paged_cache.float() - paged_cache_ref.float()).abs().max().item()

    print(f"  contiguous_kv max abs diff:  {max_ckv_diff:.6f}")
    print(f"  paged_cache append max diff: {cache_diff:.6f}")

    # Gather is a pure memory copy -> bit-exact.
    assert max_ckv_diff == 0.0, (
        f"bs={bs} kv_len={kv_len}: contiguous_kv mismatch (max_diff={max_ckv_diff})")
    assert cache_diff == 0.0, (
        f"bs={bs} kv_len={kv_len}: paged_cache append mismatch ({cache_diff})")
    print(f"PASS bs={bs} kv_len={kv_len}")

    pk.finalize()
    return max_ckv_diff, cache_diff


def test_mla_kv_gather_legacy_testmode():
    for bs in BS_LIST:
        for kv_len in KV_LEN_LIST:
            _run_case(bs, kv_len)


if __name__ == "__main__":
    results = []
    for bs in BS_LIST:
        for kv_len in KV_LEN_LIST:
            ckv, cache = _run_case(bs, kv_len)
            results.append((bs, kv_len, ckv, cache))

    print(f"\n{'='*60}")
    print("MLA_KV_GATHER_LEGACY SUMMARY")
    for bs, kv_len, ckv, cache in results:
        print(f"  bs={bs:2d} kv_len={kv_len:3d}: "
              f"ckv_max_diff={ckv:.6f}  cache_max_diff={cache:.6f}  PASS")
    print(f"ALL PASS ({len(results)}/{len(results)})")
    sys.exit(0)
