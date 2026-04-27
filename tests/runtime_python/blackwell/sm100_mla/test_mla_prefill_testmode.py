"""
Test ``PersistentKernel.mla_prefill_layer`` end-to-end through the full MPK
compilation pipeline (test_mode), comparing against the canonical PyTorch
reference in ``pytorch_reference.py``.

The kernel uses a hardcoded YARN-style softmax scale
``sm_scale = (1/sqrt(192)) * mscale^2`` where
``mscale = 0.1 * log(40) + 1`` (see register_mla_prefill_sm100_task in
src/kernel/task_register.cc). The PyTorch reference must use that exact
scale to be comparable.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> conda run -n mirage \
        python tests/runtime_python/blackwell/sm100_mla/test_mla_prefill_testmode.py
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
    # Matches register_mla_prefill_sm100_task() in src/kernel/task_register.cc
    mscale = 0.1 * 1.0 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(192.0)) * mscale * mscale


def test_mla_prefill_testmode():
    device = "cuda"
    torch.manual_seed(42)

    # Small but realistic shapes. H=2 keeps compile time low; the kernel
    # template is parameterised on num_heads so any small power-of-two works
    # up to its tile granularity. seq_len=64 is one full PF_BM=64 tile.
    batch_size = 1
    seq_len = 128
    # H=128 matches DeepSeek V3 production config and the existing kernel-
    # wrapper test (test_mla_prefill.py) — keep it here for compile-time
    # consistency.
    num_heads = 128
    D_CKV = 512
    D_KPE = 64
    D_V = 512

    print(f"\n{'='*60}")
    print(f"Test: mla_prefill_layer test_mode")
    print(f"  B={batch_size}, S={seq_len}, H={num_heads}, "
          f"D_CKV={D_CKV}, D_KPE={D_KPE}")

    sm_scale = _yarn_sm_scale()

    # Inputs (small magnitude → stable softmax). Layouts match the layer:
    #   q_nope: [S, H, D_CKV], q_pe: [S, H, D_KPE]
    #   ckv:    [S, D_CKV],    kpe:  [S, D_KPE]
    # (For batch=1 with max_seq_length=seq_len, the per-request slicing in
    #  the registered task collapses to the natural [S, ...] layout.)
    q_nope = torch.randn(seq_len, num_heads, D_CKV, dtype=torch.bfloat16,
                         device=device) * 0.1
    q_pe = torch.randn(seq_len, num_heads, D_KPE, dtype=torch.bfloat16,
                       device=device) * 0.1
    ckv = torch.randn(seq_len, D_CKV, dtype=torch.bfloat16, device=device) * 0.1
    kpe = torch.randn(seq_len, D_KPE, dtype=torch.bfloat16, device=device) * 0.1
    out = torch.zeros(seq_len, num_heads, D_V, dtype=torch.bfloat16,
                      device=device)

    # ----- meta tensor stubs (read by the registered task at runtime) -----
    # qo_indptr_buffer = [0, seq_len]: this batch contributes seq_len Q tokens.
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    # paged_kv_indptr_buffer = [0, num_pages]; with page_size=seq_len there is
    # exactly one page covering the whole sequence.
    page_size = seq_len
    num_pages = 1
    paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.tensor([seq_len], dtype=torch.int32, device=device)

    # ----- Build PersistentKernel -----
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = seq_len
    params["max_num_batched_requests"] = batch_size
    params["max_seq_length"] = seq_len  # MPK_MAX_SEQ_LENGTH used by the task
    params["max_num_pages"] = num_pages
    params["page_size"] = page_size
    params["meta_tensors"] = {
        "qo_indptr_buffer": qo_indptr,
        "paged_kv_indptr_buffer": paged_kv_indptr,
        "paged_kv_indices_buffer": paged_kv_indices,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len,
    }
    pk = PersistentKernel(**params)

    q_nope_dt = pk.attach_input(q_nope, name="q_nope")
    q_pe_dt = pk.attach_input(q_pe, name="q_pe")
    ckv_dt = pk.attach_input(ckv, name="ckv")
    kpe_dt = pk.attach_input(kpe, name="kpe")
    out_dt = pk.attach_input(out, name="mla_out")

    # Grid: (H, num_q_blocks, B); PF_BM=64 in the kernel.
    PF_BM = 64
    num_q_blocks = (seq_len + PF_BM - 1) // PF_BM
    block_dim = (256, 1, 1)  # MLA prefill is SM100-only (Blackwell)

    pk.mla_prefill_layer(
        q_nope=q_nope_dt,
        q_pe=q_pe_dt,
        ckv=ckv_dt,
        kpe=kpe_dt,
        output=out_dt,
        mla_params=(num_heads, seq_len, D_CKV, D_KPE, D_V),
        grid_dim=(num_heads, num_q_blocks, batch_size),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder)
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # PyTorch reference expects [B, S, H, D] / [B, S, D] layouts.
    ref = mla_prefill_ref(
        q_nope.unsqueeze(0), q_pe.unsqueeze(0),
        ckv.unsqueeze(0), kpe.unsqueeze(0),
        sm_scale,
    ).squeeze(0)

    nan_count = torch.isnan(out).sum().item()
    if nan_count > 0:
        # Surface the failure clearly: a real kernel NaN (not a test setup
        # issue) on q_pos == q_start rows. Reported back to the user so
        # mla_prefill_sm100.cuh can be debugged separately.
        nan_rows = (
            torch.isnan(out).any(dim=-1).any(dim=-1).nonzero().flatten().tolist()
        )
        print(f"  FAIL: {nan_count} NaN entries at q_pos rows {nan_rows}.")
        print("  (mla_prefill_sm100 produces NaN on the first row of every "
              "q_block when invoked through MPK test_mode. The standalone "
              "kernel-wrapper test test_mla_prefill.py does not exhibit it.)")

    max_diff = (out.float() - ref.float()).abs().max().item()
    mean_diff = (out.float() - ref.float()).abs().mean().item()
    print(f"  max abs diff:  {max_diff:.6f}")
    print(f"  mean abs diff: {mean_diff:.6f}")

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=2e-3)
    print("PASSED: mla_prefill test_mode matches PyTorch reference")

    pk.finalize()


if __name__ == "__main__":
    test_mla_prefill_testmode()
