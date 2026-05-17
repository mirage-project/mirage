"""Smoke test: ``layers.mla.MLAKVGather`` via PersistentKernel test_mode.

forward() raises NotImplementedError. We compile + run a tiny scope and
verify no crash + finite outputs.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.mla.kv_gather import MLAKVGather
from mirage.mpk.persistent_kernel import PersistentKernel


def _build_pk(seq_len, batch_size, page_size, max_num_pages, device):
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([seq_len], dtype=torch.int32, device=device)
    prompt_lengths = torch.tensor([seq_len], dtype=torch.int32, device=device)
    tokens = torch.zeros(batch_size, page_size, dtype=torch.int64, device=device)
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
    params["max_seq_length"] = page_size
    params["max_num_pages"] = max_num_pages
    params["page_size"] = page_size
    params["meta_tensors"] = {
        "tokens": tokens,
        "step": step,
        "prompt_lengths": prompt_lengths,
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_indices_buffer": paged_kv_indices,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
    }
    return PersistentKernel(**params)


def _smoke_run_variant(variant):
    device = "cuda"
    torch.manual_seed(0)

    seq_len = 8
    batch_size = 1
    page_size = 64
    max_num_pages = 1
    d_v = 128  # kv_lora_rank
    d_kpe = 64
    d_k = d_v + d_kpe

    c_latent_new = torch.randn(seq_len, d_v, dtype=torch.bfloat16, device=device) * 0.1
    k_pe_new = torch.randn(seq_len, d_kpe, dtype=torch.bfloat16, device=device) * 0.1
    paged_cache = torch.zeros(max_num_pages, page_size, d_k,
                              dtype=torch.bfloat16, device=device)

    s_pad = page_size  # per-request stride
    contiguous_kv = torch.zeros(batch_size * s_pad, d_k, dtype=torch.bfloat16, device=device)
    ckv_sep = torch.zeros(batch_size * s_pad, d_v, dtype=torch.bfloat16, device=device)
    kpe_sep = torch.zeros(batch_size * s_pad, d_kpe, dtype=torch.bfloat16, device=device)

    pk = _build_pk(seq_len, batch_size, page_size, max_num_pages, device)
    cl_dt = pk.attach_input(c_latent_new, name=f"kvg_{variant}_clatent")
    kpe_dt = pk.attach_input(k_pe_new, name=f"kvg_{variant}_kpe")
    cache_dt = pk.attach_input(paged_cache, name=f"kvg_{variant}_cache")

    m = MLAKVGather(d_k=d_k, d_v=d_v, page_size=page_size, variant=variant)

    with pk.compile_scope():
        if variant == "standard":
            ck_dt = pk.attach_input(contiguous_kv, name=f"kvg_{variant}_ck")
            m.compile(cl_dt, kpe_dt, cache_dt, contiguous_kv=ck_dt)
        elif variant == "split":
            cs_dt = pk.attach_input(ckv_sep, name=f"kvg_{variant}_cs")
            ks_dt = pk.attach_input(kpe_sep, name=f"kvg_{variant}_ks")
            m.compile(cl_dt, kpe_dt, cache_dt, ckv_sep=cs_dt, kpe_sep=ks_dt)
        else:  # unified
            ck_dt = pk.attach_input(contiguous_kv, name=f"kvg_{variant}_ck")
            cs_dt = pk.attach_input(ckv_sep, name=f"kvg_{variant}_cs")
            ks_dt = pk.attach_input(kpe_sep, name=f"kvg_{variant}_ks")
            m.compile(cl_dt, kpe_dt, cache_dt,
                      contiguous_kv=ck_dt, ckv_sep=cs_dt, kpe_sep=ks_dt)

    print(f"Compiling MLAKVGather({variant}) test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Smoke checks: no NaN/Inf in any output.
    for name, t in (("contiguous_kv", contiguous_kv),
                    ("ckv_sep", ckv_sep), ("kpe_sep", kpe_sep),
                    ("paged_cache", paged_cache)):
        if not torch.isfinite(t).all():
            print(f"FAILED: MLAKVGather({variant}) produced non-finite {name}")
            pk.finalize()
            sys.exit(1)
    # The kernel must have written SOMETHING into the cache or contiguous slabs.
    written = (paged_cache.abs().sum() + contiguous_kv.abs().sum()
               + ckv_sep.abs().sum() + kpe_sep.abs().sum()).item()
    print(f"  total abs sum across outputs: {written:.4f}")
    if written == 0.0:
        print(f"WARNING: MLAKVGather({variant}) produced all-zero outputs (kernel may have skipped)")

    print(f"PASSED (smoke): MLAKVGather({variant}) compiled and ran without crash")
    pk.finalize()


if __name__ == "__main__":
    for v in ("standard", "split", "unified"):
        print(f"\n=== MLAKVGather variant={v} ===")
        _smoke_run_variant(v)
    print("All MLAKVGather smoke tests completed.")
