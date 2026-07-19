"""
Test ``PersistentKernel.mla_kv_gather_unified_layer`` end-to-end through the
full MPK compilation pipeline (test_mode), comparing against the canonical
PyTorch references in ``pytorch_reference.py``.

The unified gather appends new KV to the paged cache ONCE, then materializes
EITHER the decode view OR the prefill view depending on a runtime gate:

  prompt_prefill = (request_ids[bi] >= 0
                    && step[bi] < prompt_length[bi]
                    && q_len > 8)

  * prompt_prefill == True  -> writes ckv_sep [S, D_V] + kpe_sep [S, ROPE_DIM]
                               (prefill layout); contiguous_kv untouched.
                               Compared vs ``mla_kv_gather_split_ref``.
  * prompt_prefill == False -> writes contiguous_kv [S, D_K] (decode layout);
                               ckv_sep / kpe_sep untouched.
                               Compared vs ``mla_kv_gather_ref``.

In test_mode the kernel runs on iter 0 with the page tables that
``prepare_next_batch`` derives from ``prompt_lengths`` (it clobbers any
user-passed qo_indptr / paged_kv_* — verified empirically). So this test drives
the scenario via ``prompt_lengths``:
  * PROMPT_LEN_PREFILL = 40  > 8  -> prefill branch  (ckv_sep / kpe_sep)
  * PROMPT_LEN_DECODE  = 8  (<=8) -> decode branch   (contiguous_kv)
Each request is a fresh prefill (step 0), so the kernel appends ALL of a
request's tokens (kv_start_pos=0) and gathers them; the pages a request owns
are allocated sequentially from the queue starting at 0.

``num_gather_splits`` fans the append+gather seq_pos loops over grid.y CTAs
(each strides seq_pos by num_gather_splits). grid.y MUST equal num_gather_splits.

Matrix:  bs ∈ {1,2,4,8,16} × num_splits ∈ {1,2,4}, for BOTH branches.
The KV head dim D_K=576 is NOT head-sharded -> no TP axis; bs (#requests) and
num_gather_splits (the fan-out) are the relevant axes.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \
        tests/runtime_python/blackwell/sm100_mla/test_mla_kv_gather_unified_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import mla_kv_gather_ref, mla_kv_gather_split_ref

D_V = 512
ROPE_DIM = 64
D_K = D_V + ROPE_DIM          # 576
K_PE_ROW_STRIDE = 128
PAGE_SIZE = 16

BS_LIST = [1, 2, 4, 8, 16]
SPLITS_LIST = [1, 2, 4]

PROMPT_LEN_PREFILL = 40       # > 8 -> prefill branch (ckv_sep / kpe_sep)
PROMPT_LEN_DECODE = 8         # <=8 -> decode branch (contiguous_kv)


def _run_case(bs, num_splits, decode):
    """One (bs, num_splits, branch) config.

    decode=False -> prefill branch (verify ckv_sep / kpe_sep).
    decode=True  -> decode branch  (verify contiguous_kv).
    """
    device = "cuda"
    torch.manual_seed(123 + bs * 7 + num_splits + (1000 if decode else 0))

    seq_len = PROMPT_LEN_DECODE if decode else PROMPT_LEN_PREFILL
    branch = "decode " if decode else "prefill"
    prompt_lengths = [seq_len] * bs
    mbt = sum(prompt_lengths)
    pages_per = [(L + PAGE_SIZE - 1) // PAGE_SIZE for L in prompt_lengths]
    max_num_pages = sum(pages_per) + 1
    max_seq_length = max(prompt_lengths)

    print(f"\n{'='*60}")
    print(f"Test: mla_kv_gather_unified [{branch}] bs={bs} num_splits={num_splits}")
    print(f"  S={seq_len}, page_size={PAGE_SIZE}, D_K={D_K}, D_V={D_V}, "
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

    # All three outputs are allocated; the kernel writes only the branch's set.
    contiguous_kv = torch.zeros(bs * max_seq_length, D_K,
                                dtype=torch.bfloat16, device=device)
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
    cont_dt = pk.attach_input(contiguous_kv, name="contiguous_kv")
    ckv_sep_dt = pk.attach_input(ckv_sep, name="ckv_sep")
    kpe_sep_dt = pk.attach_input(kpe_sep, name="kpe_sep")

    pk.mla_kv_gather_unified_layer(
        c_latent_new=c_latent_dt,
        k_pe_new=k_pe_dt,
        paged_cache=paged_dt,
        contiguous_kv=cont_dt,
        ckv_sep=ckv_sep_dt,
        kpe_sep=kpe_sep_dt,
        mla_params=(D_K, D_V, PAGE_SIZE),
        grid_dim=(bs, num_splits, 1),
        block_dim=(128, 1, 1),
        num_gather_splits=num_splits,
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
    diffs = {"out": 0.0, "untouched": 0.0}
    for bi in range(bs):
        L = prompt_lengths[bi]
        n_pages = pages_per[bi]
        page_indices = torch.arange(page_start, page_start + n_pages,
                                    dtype=torch.int32, device=device)
        c_bi = c_latent_new[tok_start:tok_start + L]
        k_bi = k_pe_new[tok_start:tok_start + L]
        base = bi * max_seq_length

        if decode:
            ref_out = mla_kv_gather_ref(
                c_latent_new=c_bi, k_pe_new=k_bi,
                paged_cache=paged_cache_ref, page_indices=page_indices,
                seq_len=L, d_k=D_K, d_v=D_V, page_size=PAGE_SIZE)
            out_bi = contiguous_kv[base: base + L]
            diffs["out"] = max(
                diffs["out"],
                (out_bi.float() - ref_out.float()).abs().max().item())
        else:
            ckv_ref, kpe_ref = mla_kv_gather_split_ref(
                c_latent_new=c_bi, k_pe_new=k_bi,
                paged_cache=paged_cache_ref, page_indices=page_indices,
                seq_len=L, d_k=D_K, d_v=D_V, page_size=PAGE_SIZE)
            out_ckv = ckv_sep[base: base + L]
            out_kpe = kpe_sep[base: base + L]
            diffs["out"] = max(
                diffs["out"],
                (out_ckv.float() - ckv_ref.float()).abs().max().item(),
                (out_kpe.float() - kpe_ref.float()).abs().max().item())

        page_start += n_pages
        tok_start += L

    cache_diff = (paged_cache.float() - paged_cache_ref.float()).abs().max().item()

    # The untouched branch's output must stay zero (the kernel writes only one
    # set of views). This guards against a branch-selection regression.
    if decode:
        diffs["untouched"] = max(ckv_sep.abs().max().item(),
                                 kpe_sep.abs().max().item())
    else:
        diffs["untouched"] = contiguous_kv.abs().max().item()

    print(f"  output max abs diff:        {diffs['out']:.6f}")
    print(f"  untouched-branch max value: {diffs['untouched']:.6f}")
    print(f"  paged_cache append diff:    {cache_diff:.6f}")

    # Gather-only op (pure memory copy) -> bit-exact.
    assert diffs["out"] == 0.0, (
        f"[{branch}] bs={bs} splits={num_splits}: output mismatch "
        f"(max_diff={diffs['out']})")
    assert diffs["untouched"] == 0.0, (
        f"[{branch}] bs={bs} splits={num_splits}: untouched branch was "
        f"written (max={diffs['untouched']}) — wrong branch selected")
    assert cache_diff == 0.0, (
        f"[{branch}] bs={bs} splits={num_splits}: paged_cache append "
        f"mismatch ({cache_diff})")
    print(f"PASS [{branch}] bs={bs} splits={num_splits}")

    pk.finalize()
    return diffs["out"], cache_diff


def _matrix():
    cases = []
    for decode in (False, True):
        for bs in BS_LIST:
            for ns in SPLITS_LIST:
                cases.append((bs, ns, decode))
    return cases


def test_mla_kv_gather_unified_testmode():
    for bs, ns, decode in _matrix():
        _run_case(bs, ns, decode)


if __name__ == "__main__":
    results = []
    for bs, ns, decode in _matrix():
        out_diff, cache_diff = _run_case(bs, ns, decode)
        results.append((bs, ns, decode, out_diff, cache_diff))
    print(f"\n{'='*60}")
    print("MLA_KV_GATHER_UNIFIED SUMMARY")
    for bs, ns, decode, out_diff, cache_diff in results:
        branch = "decode " if decode else "prefill"
        print(f"  [{branch}] bs={bs:2d} splits={ns}: out={out_diff:.6f} "
              f"cache={cache_diff:.6f} PASS")
    print(f"ALL PASS ({len(results)}/{len(results)})")
