"""
Test MPK MLA multi-token decode kernel (mla_mtp.cu) correctness and performance.
Supports Q_LEN=1,2,4,8 for MTP verification.

Compares against:
  - PyTorch reference (correctness)
  - FlashInfer fa2 (BatchMLAPagedAttentionWrapper)
  - FlashInfer trtllm-gen (trtllm_batch_decode_with_kv_cache_mla) — production B200 backend

Build:
    cd tests/runtime_python/blackwell/sm100_mla
    CUDA_HOME=/usr/local/cuda-12.8 python3 setup_mla_mtp.py build_ext --inplace

Run:
    python3 test_mla_mtp.py [--kv-len 4096]
"""

import argparse
import math
import torch

torch.set_printoptions(sci_mode=False)
import runtime_kernel_mla_mtp

NUM_HEADS = 128
D_K = 576
D_V = 512
TILE_S = 128

try:
    import flashinfer.mla
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

try:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    HAS_TRTLLM = True
except ImportError:
    HAS_TRTLLM = False


def pytorch_mla_ref(Q, KV, sm_scale, kv_len, q_len):
    """
    Reference MLA multi-token decode in PyTorch.
    Q:  [B, Q_LEN, NUM_HEADS, D_K] bf16
    KV: [B, kv_len, D_K] bf16
    Returns: [B, Q_LEN, NUM_HEADS, D_V] bf16
    """
    B = Q.shape[0]
    outputs = []
    for b in range(B):
        batch_outs = []
        for q in range(q_len):
            qr = Q[b, q].float()
            kv = KV[b, :kv_len].float()
            causal_lim = kv_len - q_len + q + 1 if q_len > 1 else kv_len
            scores = torch.matmul(qr, kv.T) * sm_scale
            if causal_lim < kv_len:
                scores[:, causal_lim:] = float('-inf')
            probs = torch.softmax(scores, dim=-1)
            v = kv[:, :D_V]
            out = torch.matmul(probs, v)
            batch_outs.append(out.to(Q.dtype))
        outputs.append(torch.stack(batch_outs, dim=0))
    return torch.stack(outputs, dim=0)


def test_correctness(kv_len=4096):
    B = 1
    sm_scale = 1.0 / math.sqrt(D_K)
    device = "cuda"

    print("\n  --- MPK MLA vs PyTorch ---")
    for q_len in [1, 2, 3, 4]:
        torch.manual_seed(42)
        Q = torch.randn(B, q_len, NUM_HEADS, D_K, device=device, dtype=torch.bfloat16) * 0.1
        KV = torch.randn(B, kv_len, D_K, device=device, dtype=torch.bfloat16) * 0.1
        O = torch.zeros(B, q_len, NUM_HEADS, D_V, device=device, dtype=torch.bfloat16)
        O_ref = pytorch_mla_ref(Q, KV, sm_scale, kv_len, q_len)

        runtime_kernel_mla_mtp.mla_mtp_test(Q, KV, O, kv_len, sm_scale, q_len)
        diff = (O.float() - O_ref.float()).abs()
        status = "PASSED" if diff.max().item() < 0.01 else "FAILED"
        print(f"  Q_LEN={q_len}: max_diff={diff.max().item():.6f}  {status}")

    if HAS_TRTLLM:
        print("\n  --- trtllm-gen vs PyTorch (causal check) ---")
        page_size = 64
        for q_len in [1, 2, 3, 4]:
            torch.manual_seed(42)
            Q = torch.randn(B, q_len, NUM_HEADS, D_K, device=device, dtype=torch.bfloat16) * 0.1
            KV = torch.randn(B, kv_len, D_K, device=device, dtype=torch.bfloat16) * 0.1
            O_ref = pytorch_mla_ref(Q, KV, sm_scale, kv_len, q_len)

            num_pages = (kv_len + page_size - 1) // page_size
            kv_cache = KV[:, :num_pages * page_size, :].view(num_pages, page_size, D_K)
            block_tables = torch.arange(num_pages, device=device, dtype=torch.int32).view(B, -1)
            seq_lens = torch.full((B,), kv_len, device=device, dtype=torch.int32)
            workspace = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)

            O_trt = trtllm_batch_decode_with_kv_cache_mla(
                Q, kv_cache, workspace,
                qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
                block_tables=block_tables, seq_lens=seq_lens, max_seq_len=kv_len,
                bmm1_scale=sm_scale, bmm2_scale=1.0, backend='trtllm-gen')
            O_trt = O_trt.view(B, q_len, NUM_HEADS, D_V)

            diff_causal = (O_trt.float() - O_ref.float()).abs()
            status = "PASSED" if diff_causal.max().item() < 0.01 else "FAILED"
            print(f"  Q_LEN={q_len}: max_diff={diff_causal.max().item():.6f}  {status}")


def test_performance(kv_len=4096, warmup=16, repeats=200):
    B = 1
    sm_scale = 1.0 / math.sqrt(D_K)
    device = "cuda"
    flops_factor = 2.0

    # trtllm-gen setup
    trt_page_size = 64
    trt_num_pages = (kv_len + trt_page_size - 1) // trt_page_size
    trt_block_tables = torch.arange(trt_num_pages, device=device, dtype=torch.int32).view(B, -1)
    trt_seq_lens = torch.full((B,), kv_len, device=device, dtype=torch.int32)
    trt_workspace = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device)

    # fa2 setup
    if HAS_FLASHINFER:
        fi_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    for q_len in [1, 2, 3, 4]:
        torch.manual_seed(42)
        Q = torch.randn(B, q_len, NUM_HEADS, D_K, device=device, dtype=torch.bfloat16) * 0.1
        KV = torch.randn(B, kv_len, D_K, device=device, dtype=torch.bfloat16) * 0.1
        O = torch.zeros(B, q_len, NUM_HEADS, D_V, device=device, dtype=torch.bfloat16)
        flops = flops_factor * B * NUM_HEADS * q_len * kv_len * (D_K + D_V)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # --- MPK MLA ---
        runtime_kernel_mla_mtp.mla_mtp_init(Q, KV, O, kv_len, sm_scale, q_len)
        for _ in range(warmup):
            runtime_kernel_mla_mtp.mla_mtp_run(O)
        torch.cuda.synchronize()
        starter.record()
        for _ in range(repeats):
            runtime_kernel_mla_mtp.mla_mtp_run(O)
        ender.record()
        torch.cuda.synchronize()
        mpk_us = starter.elapsed_time(ender) / repeats * 1000
        mpk_tflops = flops / (mpk_us * 1e-6) / 1e12

        parts = [f"  Q_LEN={q_len}:  MPK MLA: {mpk_us:7.1f} us {mpk_tflops:6.1f} TFLOPS"]

        # --- trtllm-gen ---
        if HAS_TRTLLM:
            kv_cache = KV[:, :trt_num_pages * trt_page_size, :].view(
                trt_num_pages, trt_page_size, D_K)
            trt_workspace.zero_()
            for _ in range(warmup):
                trtllm_batch_decode_with_kv_cache_mla(
                    Q, kv_cache, trt_workspace,
                    qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
                    block_tables=trt_block_tables, seq_lens=trt_seq_lens, max_seq_len=kv_len,
                    bmm1_scale=sm_scale, bmm2_scale=1.0, backend='trtllm-gen')
            torch.cuda.synchronize()
            starter.record()
            for _ in range(repeats):
                trtllm_batch_decode_with_kv_cache_mla(
                    Q, kv_cache, trt_workspace,
                    qk_nope_head_dim=128, kv_lora_rank=512, qk_rope_head_dim=64,
                    block_tables=trt_block_tables, seq_lens=trt_seq_lens, max_seq_len=kv_len,
                    bmm1_scale=sm_scale, bmm2_scale=1.0, backend='trtllm-gen')
            ender.record()
            torch.cuda.synchronize()
            trt_us = starter.elapsed_time(ender) / repeats * 1000
            trt_tflops = flops / (trt_us * 1e-6) / 1e12
            parts.append(f"  trtllm-gen: {trt_us:7.1f} us {trt_tflops:6.1f} TFLOPS")

        # --- FlashInfer fa2 ---
        if HAS_FLASHINFER:
            q_nope = Q[:, :, :, :D_V].contiguous().view(B * q_len, NUM_HEADS, D_V)
            q_pe = Q[:, :, :, D_V:].contiguous().view(B * q_len, NUM_HEADS, D_K - D_V)
            page_size = 1
            num_pages = B * kv_len
            ckv_paged = KV[:, :kv_len, :D_V].contiguous().view(num_pages, page_size, D_V)
            kpe_paged = KV[:, :kv_len, D_V:].contiguous().view(num_pages, page_size, D_K - D_V)
            kv_lens = torch.full((B,), kv_len, device=device, dtype=torch.int32)
            q_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * q_len
            kv_indptr = torch.arange(0, B + 1, device=device, dtype=torch.int32) * kv_len
            kv_indices = torch.arange(0, num_pages, device=device, dtype=torch.int32)
            causal = (q_len > 1)

            fi_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(fi_workspace, backend='fa2')
            fi_wrapper.plan(q_indptr, kv_indptr, kv_indices, kv_lens,
                            NUM_HEADS, D_V, D_K - D_V, page_size, causal, sm_scale,
                            Q.dtype, KV.dtype)
            for _ in range(warmup):
                fi_wrapper.run(q_nope, q_pe, ckv_paged, kpe_paged)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(repeats):
                fi_wrapper.run(q_nope, q_pe, ckv_paged, kpe_paged)
            ender.record()
            torch.cuda.synchronize()
            fi_us = starter.elapsed_time(ender) / repeats * 1000
            fi_tflops = flops / (fi_us * 1e-6) / 1e12
            parts.append(f"  fa2: {fi_us:7.1f} us {fi_tflops:6.1f} TFLOPS")

        print("".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=4096)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MPK MLA Multi-Token Decode Correctness (kv_len={args.kv_len})")
    print(f"{'='*60}")
    test_correctness(args.kv_len)

    print(f"\n{'='*60}")
    print(f"MPK MLA Multi-Token Decode Performance (kv_len={args.kv_len})")
    print(f"{'='*60}")
    test_performance(args.kv_len)
