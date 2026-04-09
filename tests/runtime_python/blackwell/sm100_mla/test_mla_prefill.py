"""
Test MPK MLA prefilling kernel (mla_prefill_sm100.cuh) correctness and performance.

Build:
    cd tests/runtime_python/blackwell/sm100_mla
    CUDA_HOME=/usr/local/cuda-12.8 python3 setup_mla_prefill.py build_ext --inplace

Run:
    python3 test_mla_prefill.py [--seq-len 1024]
"""

import argparse
import math
import torch

torch.set_printoptions(sci_mode=False)
import runtime_kernel_mla_prefill

D_CKV = 512
D_KPE = 64
D_V = 512
D_QK = 576

try:
    import flashinfer.mla
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


def pytorch_mla_prefill_ref(Q_nope, Q_pe, CKV, KPE, sm_scale):
    B, S, H, _ = Q_nope.shape
    q = torch.cat([Q_nope.float(), Q_pe.float()], dim=-1)  # [B,S,H,576]
    k = torch.cat([CKV.float(), KPE.float()], dim=-1)      # [B,S,576]
    scores = torch.einsum('bshd,btd->bsht', q, k) * sm_scale
    causal = torch.triu(torch.ones(S, S, device=scores.device), diagonal=1).bool()
    scores.masked_fill_(causal.unsqueeze(0).unsqueeze(2), float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    v = CKV.float()
    o = torch.einsum('bsht,btd->bshd', probs, v)
    return o.to(Q_nope.dtype)


def test_correctness(seq_len=1024):
    B, H = 1, 128
    print(f"\n{'='*60}")
    print(f"MPK MLA Prefilling Correctness")
    print(f"  B={B}, S={seq_len}, H={H}")
    print(f"{'='*60}")

    device = "cuda"
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Q_nope = torch.randn(B, seq_len, H, D_CKV, device=device, dtype=torch.bfloat16) * 0.1
    Q_pe = torch.randn(B, seq_len, H, D_KPE, device=device, dtype=torch.bfloat16) * 0.1
    CKV = torch.randn(B, seq_len, D_CKV, device=device, dtype=torch.bfloat16) * 0.1
    KPE = torch.randn(B, seq_len, D_KPE, device=device, dtype=torch.bfloat16) * 0.1
    O = torch.zeros(B, seq_len, H, D_V, device=device, dtype=torch.bfloat16)

    # MPK MLA prefilling kernel
    print("Running MPK MLA prefilling kernel...")
    runtime_kernel_mla_prefill.mla_prefill_test(
        Q_nope.view(B * seq_len, H, D_CKV) if False else Q_nope,
        Q_pe, CKV, KPE, O, sm_scale
    )

    # PyTorch reference
    print("Running PyTorch reference...")
    O_ref = pytorch_mla_prefill_ref(Q_nope, Q_pe, CKV, KPE, sm_scale)

    diff = (O.float() - O_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    torch.testing.assert_close(O, O_ref, rtol=1e-2, atol=2e-3)
    print("  PASSED")


def test_performance(seq_len=1024, warmup=16, repeats=50):
    B, H = 1, 128
    print(f"\n{'='*60}")
    print(f"MLA Prefill Performance (S={seq_len})")
    print(f"{'='*60}")

    device = "cuda"
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D_QK)

    Q_nope = torch.randn(B, seq_len, H, D_CKV, device=device, dtype=torch.bfloat16) * 0.1
    Q_pe = torch.randn(B, seq_len, H, D_KPE, device=device, dtype=torch.bfloat16) * 0.1
    CKV = torch.randn(B, seq_len, D_CKV, device=device, dtype=torch.bfloat16) * 0.1
    KPE = torch.randn(B, seq_len, D_KPE, device=device, dtype=torch.bfloat16) * 0.1
    O = torch.zeros(B, seq_len, H, D_V, device=device, dtype=torch.bfloat16)

    flops = B * H * seq_len * seq_len * (D_CKV + D_KPE + D_CKV)

    # --- MPK MLA prefilling kernel ---
    runtime_kernel_mla_prefill.mla_prefill_init(Q_nope, Q_pe, CKV, KPE, O, sm_scale)
    for _ in range(warmup):
        runtime_kernel_mla_prefill.mla_prefill_run(Q_nope, Q_pe, CKV, KPE, O)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(repeats):
        runtime_kernel_mla_prefill.mla_prefill_run(Q_nope, Q_pe, CKV, KPE, O)
    ender.record()
    torch.cuda.synchronize()
    us = starter.elapsed_time(ender) / repeats * 1000
    tflops = flops / (us * 1e-6) / 1e12
    print(f"  MPK MLA: {us:8.1f} us  {tflops:6.1f} TFLOPS")

    # --- FlashInfer ---
    if HAS_FLASHINFER:
        q_nope_fi = Q_nope.view(B * seq_len, H, D_CKV)
        q_pe_fi = Q_pe.view(B * seq_len, H, D_KPE)
        page_size = 1
        num_pages = B * seq_len
        ckv_paged = CKV.view(num_pages, page_size, D_CKV)
        kpe_paged = KPE.view(num_pages, page_size, D_KPE)
        kv_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
        q_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * seq_len
        kv_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * seq_len
        kv_indices = torch.arange(0, num_pages, device=device, dtype=torch.int32)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend='fa2')
        wrapper.plan(q_indptr, kv_indptr, kv_indices, kv_lens,
                     H, D_CKV, D_KPE, page_size, True, sm_scale,
                     Q_nope.dtype, CKV.dtype)

        for _ in range(warmup):
            wrapper.run(q_nope_fi, q_pe_fi, ckv_paged, kpe_paged)
        torch.cuda.synchronize()

        starter.record()
        for _ in range(repeats):
            wrapper.run(q_nope_fi, q_pe_fi, ckv_paged, kpe_paged)
        ender.record()
        torch.cuda.synchronize()
        fi_us = starter.elapsed_time(ender) / repeats * 1000
        fi_tflops = flops / (fi_us * 1e-6) / 1e12
        print(f"  FlashInfer:  {fi_us:8.1f} us  {fi_tflops:6.1f} TFLOPS")
    else:
        print(f"  FlashInfer:  (not installed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    test_correctness(args.seq_len)

    print(f"\n{'='*60}")
    print(f"Performance Sweep")
    print(f"{'='*60}")
    for s in [512, 1024, 2048, 4096]:
        test_performance(s)
