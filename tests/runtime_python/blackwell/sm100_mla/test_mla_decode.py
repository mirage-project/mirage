"""
Test MLA decode MPK MLA decoding kernel (mla_decode_sm100.cuh + mla_reduce_sm100.cuh).

Correctness: compare vs PyTorch reference
Performance: compare vs mla_host.cu binary and FlashInfer

Build:
    cd tests/runtime_python/blackwell/sm100_mla
    CUDA_HOME=/usr/local/cuda-12.8 python3 setup_mla_decode.py build_ext --inplace

Run:
    python3 test_mla_decode.py [--kv-len 4096] [--batch 1] [--num-splits 32]
"""

import argparse
import math
import os
import subprocess
import torch

torch.set_printoptions(sci_mode=False)
import runtime_kernel_mla_decode

NUM_HEADS = 128
D_K = 576
D_V = 512

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

try:
    import flashinfer
    import flashinfer.mla
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


def pytorch_mla_reference(Q, KV, softmax_scale, kv_len):
    B = Q.shape[0]
    outputs = []
    for b in range(B):
        q = Q[b].float()
        kv = KV[b, :kv_len].float()
        scores = torch.matmul(q, kv.T) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        v = kv[:, :D_V]
        out = torch.matmul(probs, v)
        outputs.append(out.to(Q.dtype))
    return torch.stack(outputs, dim=0)


def test_correctness(batch_size=1, kv_len=4096, num_splits=32):
    print(f"\n{'='*60}")
    print(f"MLA Decode Correctness Test")
    print(f"  B={batch_size}, kv_len={kv_len}, num_splits={num_splits}")
    print(f"{'='*60}")

    device = "cuda"
    torch.manual_seed(42)

    Q = torch.randn(batch_size, NUM_HEADS, D_K, device=device, dtype=torch.bfloat16)
    KV = torch.randn(batch_size, kv_len, D_K, device=device, dtype=torch.bfloat16)
    O = torch.zeros(batch_size, NUM_HEADS, D_V, device=device, dtype=torch.bfloat16)
    softmax_scale = 1.0 / math.sqrt(D_K)

    runtime_kernel_mla_decode.mla_decode_test(Q, KV, O, num_splits, softmax_scale)

    O_ref = pytorch_mla_reference(Q, KV, softmax_scale, kv_len)

    diff = (O.float() - O_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    torch.testing.assert_close(O, O_ref, rtol=1e-2, atol=1e-2)
    print("  PASSED")


def test_performance(batch_size=1, kv_len=4096, num_splits=32, warmup=16, repeats=1000):
    print(f"\n{'='*60}")
    print(f"MLA Decode Performance Comparison")
    print(f"  B={batch_size}, kv_len={kv_len}, num_splits={num_splits}")
    print(f"{'='*60}")

    device = "cuda"
    torch.manual_seed(42)

    Q = torch.randn(batch_size, NUM_HEADS, D_K, device=device, dtype=torch.bfloat16)
    KV = torch.randn(batch_size, kv_len, D_K, device=device, dtype=torch.bfloat16)
    O = torch.zeros(batch_size, NUM_HEADS, D_V, device=device, dtype=torch.bfloat16)
    softmax_scale = 1.0 / math.sqrt(D_K)

    flops = 2.0 * batch_size * NUM_HEADS * kv_len * (D_K + D_V)

    # --- Our MPK MLA decoding kernel ---
    runtime_kernel_mla_decode.mla_init(Q, KV, O, num_splits, softmax_scale)

    for _ in range(warmup):
        runtime_kernel_mla_decode.mla_run(O)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(repeats):
        runtime_kernel_mla_decode.mla_run(O)
    ender.record()
    torch.cuda.synchronize()
    ours_ms = starter.elapsed_time(ender) / repeats
    ours_us = ours_ms * 1000
    ours_tflops = flops / (ours_us * 1e-6) / 1e12
    print(f"\n  MPK MLA:   {ours_us:.2f} us  ({ours_tflops:.2f} TFLOPS)")

    # --- PyTorch reference ---
    O_ref = pytorch_mla_reference(Q, KV, softmax_scale, kv_len)  # warmup
    torch.cuda.synchronize()
    starter.record()
    for _ in range(100):
        O_ref = pytorch_mla_reference(Q, KV, softmax_scale, kv_len)
    ender.record()
    torch.cuda.synchronize()
    torch_ms = starter.elapsed_time(ender) / 100
    torch_us = torch_ms * 1000
    torch_tflops = flops / (torch_us * 1e-6) / 1e12
    print(f"  PyTorch:       {torch_us:.2f} us  ({torch_tflops:.2f} TFLOPS)")

    # --- FlashInfer ---
    if HAS_FLASHINFER:
        q_nope = Q[:, :, :D_V].contiguous()
        q_pe = Q[:, :, D_V:].contiguous()
        ckv_cache = KV[:, :kv_len, :D_V].contiguous().unsqueeze(0).view(kv_len, 1, D_V).contiguous()
        # FlashInfer needs paged format. Use single page = full seq.
        page_size = kv_len
        num_pages = batch_size
        ckv_paged = KV[:, :kv_len, :D_V].contiguous().view(num_pages, page_size, D_V)
        kpe_paged = KV[:, :kv_len, D_V:].contiguous().view(num_pages, page_size, 64)
        kv_lens = torch.full((batch_size,), kv_len, device=device, dtype=torch.int32)
        page_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(batch_size, 1)

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa2")
        q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        kv_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        kv_indices = page_table.flatten().contiguous()

        mla_wrapper.plan(
            q_indptr, kv_indptr, kv_indices, kv_lens,
            NUM_HEADS, D_V, 64, page_size, False, softmax_scale,
            Q.dtype, ckv_paged.dtype,
        )

        for _ in range(warmup):
            mla_wrapper.run(q_nope, q_pe, ckv_paged, kpe_paged, return_lse=False)
        torch.cuda.synchronize()

        starter.record()
        for _ in range(repeats):
            mla_wrapper.run(q_nope, q_pe, ckv_paged, kpe_paged, return_lse=False)
        ender.record()
        torch.cuda.synchronize()
        fi_ms = starter.elapsed_time(ender) / repeats
        fi_us = fi_ms * 1000
        fi_tflops = flops / (fi_us * 1e-6) / 1e12
        print(f"  FlashInfer:    {fi_us:.2f} us  ({fi_tflops:.2f} TFLOPS)")
    else:
        print(f"  FlashInfer:    (not installed)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--num-splits", type=int, default=32)
    args = parser.parse_args()

    test_correctness(args.batch, args.kv_len, args.num_splits)
    test_performance(args.batch, args.kv_len, args.num_splits)
