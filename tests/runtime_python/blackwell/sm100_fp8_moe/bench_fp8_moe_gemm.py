"""
Benchmark: FP8 MoE Group GEMM — Mirage MPK kernel vs FlashInfer vs PyTorch.

Compares kernel-only latency (excluding launch overhead) of MoE GEMM:
  1. Mirage MPK single-CTA  (1 CTA processing all experts sequentially)
  2. Mirage MPK multi-CTA   (NUM_EXPERTS CTAs, 1 expert per CTA)
  3. FlashInfer SegmentGEMM  (BF16, multi-CTA segment GEMM)
  4. FlashInfer bmm_fp8      (FP8, batched matmul via cuBLAS)
  5. PyTorch baseline        (BF16 per-expert torch.matmul loop)

Timing: Each iteration is timed with its own CUDA event pair (start/end).
The median of all iterations is reported (robust to outliers). The kernel
launch itself is NOT included — events are recorded around the kernel only.

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_moe
    python setup.py build_ext --inplace
    python bench_fp8_moe_gemm.py
"""

import torch
import sys
import os
import random
import statistics

# --------------------------------------------------------------------------
# Import Mirage MPK kernel
# --------------------------------------------------------------------------
try:
    import runtime_kernel_fp8_moe as rk
except ImportError:
    print("ERROR: runtime_kernel_fp8_moe not found.")
    print("Please run: python setup.py build_ext --inplace")
    sys.exit(1)

# --------------------------------------------------------------------------
# Import FlashInfer
# --------------------------------------------------------------------------
try:
    from flashinfer import SegmentGEMMWrapper
    from flashinfer.gemm import (bmm_fp8,
                                  group_gemm_fp8_nt_groupwise,
                                  group_deepgemm_fp8_nt_groupwise,
                                  batch_deepgemm_fp8_nt_groupwise)
    HAS_FLASHINFER = True
except ImportError:
    print("WARNING: FlashInfer not available, skipping FlashInfer benchmarks.")
    HAS_FLASHINFER = False

# --------------------------------------------------------------------------
# Test dimensions (must match the compiled MPK kernel constants)
# --------------------------------------------------------------------------
BATCH_SIZE  = 128   # padded to MMA_N=128
OUTPUT_SIZE = 256   # N
K           = 256   # REDUCTION_SIZE
NUM_EXPERTS = 8
NUM_TOPK    = 2

WARMUP_ITERS = 50
BENCH_ITERS  = 200


def quantize_to_fp8(x: torch.Tensor, block_k: int = 128):
    """Per-128-element block quantization to FP8 E4M3."""
    shape = x.shape
    K_dim = shape[-1]
    assert K_dim % block_k == 0
    num_blocks = K_dim // block_k
    x_blocks = x.reshape(*shape[:-1], num_blocks, block_k)
    amax = x_blocks.abs().amax(dim=-1)
    scale = (amax / 448.0).clamp(min=1e-12)
    x_scaled = x_blocks / scale.unsqueeze(-1)
    x_fp8 = x_scaled.reshape(*shape).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def make_random_routing(batch_size, num_experts, num_topk, device, seed=42):
    """Create random routing for benchmarking."""
    rng = random.Random(seed)
    routing = torch.zeros(num_experts, BATCH_SIZE, dtype=torch.int32, device=device)
    token_to_experts = {}
    for i in range(batch_size):
        experts = rng.sample(range(num_experts), num_topk)
        token_to_experts[i] = experts
        for slot, e in enumerate(experts):
            routing[e, i] = slot + 1

    activated = []
    for e in range(num_experts):
        if routing[e, :batch_size].any():
            activated.append(e)

    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[num_experts] = len(activated)

    return routing, mask, token_to_experts


def bench_kernel(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a CUDA kernel with per-iteration event timing.

    Each iteration gets its own (start, end) event pair. We report the
    median latency across all iterations (robust to outliers from
    scheduling jitter, power state changes, etc.).

    The function `fn` should launch kernel(s) WITHOUT calling synchronize.
    """
    # Warmup — fill caches, stabilize clocks
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Create event pairs for each iteration
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # Timed iterations
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()

    # Collect per-iteration latencies
    latencies = [starts[i].elapsed_time(ends[i]) * 1000.0  # ms -> us
                 for i in range(iters)]

    median_us = statistics.median(latencies)
    min_us = min(latencies)
    p99_us = sorted(latencies)[int(iters * 0.99)]
    return median_us, min_us, p99_us


# =========================================================================
# Benchmark implementations
# =========================================================================

def bench_mirage_single_cta(input_fp8, input_scale, weight_scale,
                             routing, mask, output):
    """Mirage MPK: 1 CTA processes all experts sequentially."""
    def fn():
        rk.fp8_moe_w13_gemm_launch(input_fp8, input_scale, weight_scale,
                                     routing, mask, output, 1)
    return bench_kernel(fn)


def bench_mirage_multi_cta(input_fp8, input_scale, weight_scale,
                            routing, mask, output, num_ctas):
    """Mirage MPK: num_ctas CTAs, each handles 1 expert."""
    def fn():
        rk.fp8_moe_w13_gemm_launch(input_fp8, input_scale, weight_scale,
                                     routing, mask, output, num_ctas)
    return bench_kernel(fn)


def bench_flashinfer_segment_gemm(batch_size, input_bf16, weight_bf16,
                                    token_to_experts):
    """FlashInfer SegmentGEMM (BF16, multi-CTA, optimized for MoE routing)."""
    device = input_bf16.device

    expert_tokens = {e: [] for e in range(NUM_EXPERTS)}
    for tok_idx, experts in token_to_experts.items():
        for e in experts:
            expert_tokens[e].append(tok_idx)

    packed_rows = []
    seg_lens = []
    weight_indices = []
    for e in range(NUM_EXPERTS):
        tokens = expert_tokens[e]
        if len(tokens) == 0:
            continue
        for t in tokens:
            packed_rows.append(input_bf16[t])
        seg_lens.append(len(tokens))
        weight_indices.append(e)

    if len(packed_rows) == 0:
        return 0.0, 0.0, 0.0

    x_packed = torch.stack(packed_rows).to(torch.bfloat16)
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int64, device=device)
    weight_idx_t = torch.tensor(weight_indices, dtype=torch.int64, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    seg_gemm = SegmentGEMMWrapper(workspace)

    def fn():
        seg_gemm.run(x_packed, weight_bf16, len(seg_lens), True,
                     seg_lens=seg_lens_t, weight_indices=weight_idx_t)

    return bench_kernel(fn)


def bench_flashinfer_bmm_fp8(batch_size, weight_bf16, token_to_experts):
    """FlashInfer bmm_fp8 (FP8, batched matmul via cuBLAS per expert)."""
    device = weight_bf16.device

    expert_tokens = {e: [] for e in range(NUM_EXPERTS)}
    for tok_idx, experts in token_to_experts.items():
        for e in experts:
            expert_tokens[e].append(tok_idx)

    input_bf16_rand = torch.randn(batch_size, K, device=device, dtype=torch.bfloat16)
    a_list, b_list = [], []
    a_scale = torch.ones(1, dtype=torch.float32, device=device)
    b_scale = torch.ones(1, dtype=torch.float32, device=device)

    for e in range(NUM_EXPERTS):
        tokens = expert_tokens[e]
        if len(tokens) == 0:
            continue
        a_bf16 = torch.stack([input_bf16_rand[t] for t in tokens]).unsqueeze(0)
        a_fp8 = a_bf16.to(torch.float8_e4m3fn).contiguous()
        b_fp8 = weight_bf16[e].T.unsqueeze(0).to(torch.float8_e4m3fn).contiguous()
        a_list.append(a_fp8)
        b_list.append(b_fp8)

    if not a_list:
        return 0.0, 0.0, 0.0

    def fn():
        for a_fp8, b_fp8 in zip(a_list, b_list):
            bmm_fp8(a_fp8, b_fp8, a_scale, b_scale, dtype=torch.bfloat16)

    return bench_kernel(fn)


def _build_expert_token_map(token_to_experts):
    """Build a map from expert index to list of token indices."""
    expert_tokens = {e: [] for e in range(NUM_EXPERTS)}
    for tok_idx, experts in token_to_experts.items():
        for e in experts:
            expert_tokens[e].append(tok_idx)
    return expert_tokens


def _build_packed_fp8_data(batch_size, input_bf16, weight_bf16, token_to_experts):
    """Build packed FP8 data with block scales for group GEMM APIs.

    Returns:
        a_fp8:    [cum_m, K]           packed FP8 activations
        a_scale:  [cum_m, K//128]      per-token K-major scales
        b_fp8:    [NUM_EXPERTS, N, K]  FP8 weights
        b_scale:  [NUM_EXPERTS, N//128, K//128]  per-block scales
        m_indptr: [NUM_EXPERTS+1]      segment boundaries (padded to multiple of 4)
        m_indices:[cum_m]              per-row group assignment
        expert_tokens: dict            expert -> [token_indices]
    """
    device = input_bf16.device
    expert_tokens = _build_expert_token_map(token_to_experts)
    block_k = 128

    # Build packed A: concatenate tokens per expert, pad each segment to multiple of 4
    packed_a_rows = []
    m_indptr_list = [0]
    m_indices_list = []

    for e in range(NUM_EXPERTS):
        tokens = expert_tokens[e]
        n_tokens = len(tokens)
        # Pad to multiple of 4 (required by group_gemm_fp8_nt_groupwise)
        padded = ((n_tokens + 3) // 4) * 4
        for t in tokens:
            packed_a_rows.append(input_bf16[t])
            m_indices_list.append(e)
        # Pad with zeros
        for _ in range(padded - n_tokens):
            packed_a_rows.append(torch.zeros(K, device=device, dtype=input_bf16.dtype))
            m_indices_list.append(e)
        m_indptr_list.append(m_indptr_list[-1] + padded)

    cum_m = m_indptr_list[-1]
    a_bf16_packed = torch.stack(packed_a_rows)  # [cum_m, K]
    # Per-token FP8 quantization for A
    a_fp8 = a_bf16_packed.to(torch.float8_e4m3fn).contiguous()
    # Per-token scale: [cum_m, K//128]
    num_k_blocks = K // block_k
    a_blocks = a_bf16_packed.float().reshape(cum_m, num_k_blocks, block_k)
    a_amax = a_blocks.abs().amax(dim=-1)  # [cum_m, num_k_blocks]
    a_scale = (a_amax / 448.0).clamp(min=1e-12).float()

    # Per-block FP8 quantization for B: [NUM_EXPERTS, N, K]
    n_blocks = (OUTPUT_SIZE + block_k - 1) // block_k
    b_fp8 = weight_bf16.to(torch.float8_e4m3fn).contiguous()
    # b_scale: [NUM_EXPERTS, N//128, K//128]
    w_blocks = weight_bf16.float().reshape(NUM_EXPERTS, n_blocks, block_k, num_k_blocks, block_k)
    b_amax = w_blocks.abs().amax(dim=(2, 4))  # [NUM_EXPERTS, n_blocks, num_k_blocks]
    b_scale = (b_amax / 448.0).clamp(min=1e-12).float()

    m_indptr = torch.tensor(m_indptr_list, dtype=torch.int32, device=device)
    m_indices = torch.tensor(m_indices_list, dtype=torch.int32, device=device)

    return a_fp8, a_scale, b_fp8, b_scale, m_indptr, m_indices, expert_tokens


def bench_fi_group_gemm_fp8(input_bf16, weight_bf16, token_to_experts):
    """FlashInfer group_gemm_fp8_nt_groupwise (CUTLASS FP8, packed segments)."""
    a_fp8, a_scale, b_fp8, b_scale, m_indptr, _, _ = \
        _build_packed_fp8_data(0, input_bf16, weight_bf16, token_to_experts)

    def fn():
        group_gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale, m_indptr,
            scale_granularity_mnk=(1, 128, 128),
            scale_major_mode='K')

    return bench_kernel(fn)


def bench_fi_group_deepgemm_fp8(input_bf16, weight_bf16, token_to_experts):
    """FlashInfer group_deepgemm_fp8_nt_groupwise (DeepGEMM backend, packed with m_indices)."""
    a_fp8, a_scale, b_fp8, b_scale, _, m_indices, _ = \
        _build_packed_fp8_data(0, input_bf16, weight_bf16, token_to_experts)

    def fn():
        group_deepgemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale, m_indices,
            scale_granularity_mnk=(1, 128, 128))

    return bench_kernel(fn)


def bench_fi_batch_deepgemm_fp8(batch_size, input_bf16, weight_bf16, token_to_experts):
    """FlashInfer batch_deepgemm_fp8_nt_groupwise (DeepGEMM backend, 3D masked)."""
    device = input_bf16.device
    expert_tokens = _build_expert_token_map(token_to_experts)
    block_k = 128

    max_m = max(max(len(v) for v in expert_tokens.values()), 1)
    # Pad max_m to at least 4 for alignment
    max_m = max(max_m, 4)

    # A: [NUM_EXPERTS, max_m, K]
    a_bf16 = torch.zeros(NUM_EXPERTS, max_m, K, device=device, dtype=input_bf16.dtype)
    masked_m = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
    for e in range(NUM_EXPERTS):
        tokens = expert_tokens[e]
        masked_m[e] = len(tokens)
        for local_idx, tok_idx in enumerate(tokens):
            a_bf16[e, local_idx] = input_bf16[tok_idx]

    a_fp8 = a_bf16.to(torch.float8_e4m3fn).contiguous()
    num_k_blocks = K // block_k
    a_blocks = a_bf16.float().reshape(NUM_EXPERTS, max_m, num_k_blocks, block_k)
    a_amax = a_blocks.abs().amax(dim=-1)  # [NUM_EXPERTS, max_m, num_k_blocks]
    a_scale = (a_amax / 448.0).clamp(min=1e-12).float()

    # B: same per-block scales
    n_blocks = (OUTPUT_SIZE + block_k - 1) // block_k
    b_fp8 = weight_bf16.to(torch.float8_e4m3fn).contiguous()
    w_blocks = weight_bf16.float().reshape(NUM_EXPERTS, n_blocks, block_k, num_k_blocks, block_k)
    b_amax = w_blocks.abs().amax(dim=(2, 4))
    b_scale = (b_amax / 448.0).clamp(min=1e-12).float()

    out = torch.empty(NUM_EXPERTS, max_m, OUTPUT_SIZE, device=device, dtype=torch.bfloat16)
    expected_m = max(1, batch_size // NUM_EXPERTS)

    def fn():
        batch_deepgemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale, masked_m, expected_m,
            scale_granularity_mnk=(1, 128, 128), out=out)

    return bench_kernel(fn)


def bench_pytorch_matmul(batch_size, input_bf16, weight_bf16, token_to_experts):
    """PyTorch BF16 per-expert matmul loop (baseline)."""
    device = input_bf16.device
    output = torch.zeros(batch_size, NUM_TOPK, OUTPUT_SIZE,
                         dtype=torch.bfloat16, device=device)

    def fn():
        for i in range(batch_size):
            for slot, e in enumerate(token_to_experts[i]):
                output[i, slot] = (input_bf16[i:i+1] @ weight_bf16[e].T).squeeze(0)

    return bench_kernel(fn)


def fmt_us(median, min_val, p99):
    """Format timing as 'median (min / p99)'."""
    return f"{median:7.2f} ({min_val:6.2f}/{p99:7.2f})"


def main():
    device = torch.device("cuda")
    print("=" * 120)
    print("FP8 MoE Group GEMM Benchmark — Kernel-Only Latency (us)")
    print(f"  N={OUTPUT_SIZE}, K={K}, num_experts={NUM_EXPERTS}, top_k={NUM_TOPK}")
    print(f"  BATCH_SIZE (padded)={BATCH_SIZE}")
    print(f"  Warmup={WARMUP_ITERS}, Iters={BENCH_ITERS}")
    print(f"  Format: median (min / p99) in microseconds")
    print(f"  FlashInfer: {'available' if HAS_FLASHINFER else 'NOT available'}")
    print("=" * 120)

    # ---- Benchmark names and corresponding functions ----
    # Each entry: (short_name, header_width)
    bench_names = ["MPK-1CTA", "MPK-8CTA"]
    if HAS_FLASHINFER:
        bench_names += ["FI-SegGEMM", "FI-bmm_fp8",
                        "FI-grp_fp8", "FI-grpDG_fp8", "FI-batDG_fp8"]
    bench_names += ["PyTorch"]

    col_w = 26
    header = f"{'batch':>5}"
    for name in bench_names:
        header += f"  {(name + ' (us)'):>{col_w}}"
    print()
    print(header)
    print("-" * len(header))

    for batch_size in [1, 2, 4, 8, 16]:
        torch.manual_seed(42 + batch_size)

        # Generate test data
        input_bf16 = torch.randn(BATCH_SIZE, K, device=device, dtype=torch.bfloat16)
        input_fp8, input_scale = quantize_to_fp8(input_bf16.float())
        weight_bf16 = torch.randn(NUM_EXPERTS, OUTPUT_SIZE, K, device=device, dtype=torch.bfloat16)
        weight_fp8_list, weight_scale_list = [], []
        for e in range(NUM_EXPERTS):
            w_fp8, w_scale = quantize_to_fp8(weight_bf16[e].float())
            weight_fp8_list.append(w_fp8)
            weight_scale_list.append(w_scale)
        weight_fp8 = torch.stack(weight_fp8_list, dim=0)
        weight_scale = torch.stack(weight_scale_list, dim=0)

        routing, mask, token_to_experts = make_random_routing(
            batch_size, NUM_EXPERTS, NUM_TOPK, device)

        output = torch.zeros(BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                             dtype=torch.bfloat16, device=device)

        # Setup MPK benchmark mode (pre-create TMA descriptor)
        rk.fp8_moe_w13_gemm_setup(weight_fp8)

        results = {}

        # Mirage single CTA
        results["MPK-1CTA"] = bench_mirage_single_cta(
            input_fp8, input_scale, weight_scale, routing, mask, output)

        # Mirage multi CTA (8 CTAs = 1 per expert)
        results["MPK-8CTA"] = bench_mirage_multi_cta(
            input_fp8, input_scale, weight_scale, routing, mask, output, NUM_EXPERTS)

        if HAS_FLASHINFER:
            active_input = input_bf16[:batch_size]

            # FlashInfer SegmentGEMM (BF16)
            results["FI-SegGEMM"] = bench_flashinfer_segment_gemm(
                batch_size, active_input, weight_bf16, token_to_experts)

            # FlashInfer bmm_fp8 (per-tensor scaled, cuBLAS)
            results["FI-bmm_fp8"] = bench_flashinfer_bmm_fp8(
                batch_size, weight_bf16, token_to_experts)

            # FlashInfer group_gemm_fp8_nt_groupwise (CUTLASS, packed segments)
            try:
                results["FI-grp_fp8"] = bench_fi_group_gemm_fp8(
                    active_input, weight_bf16, token_to_experts)
            except Exception:
                results["FI-grp_fp8"] = None

            # FlashInfer group_deepgemm_fp8_nt_groupwise (DeepGEMM, packed m_indices)
            try:
                results["FI-grpDG_fp8"] = bench_fi_group_deepgemm_fp8(
                    active_input, weight_bf16, token_to_experts)
            except Exception:
                results["FI-grpDG_fp8"] = None

            # FlashInfer batch_deepgemm_fp8_nt_groupwise (DeepGEMM, 3D masked)
            try:
                results["FI-batDG_fp8"] = bench_fi_batch_deepgemm_fp8(
                    batch_size, active_input, weight_bf16, token_to_experts)
            except Exception:
                results["FI-batDG_fp8"] = None

        # PyTorch baseline
        results["PyTorch"] = bench_pytorch_matmul(
            batch_size, input_bf16[:batch_size].bfloat16(), weight_bf16,
            token_to_experts)

        # Print row
        line = f"{batch_size:>5}"
        for name in bench_names:
            r = results.get(name)
            if r:
                line += f"  {fmt_us(*r):>{col_w}}"
            else:
                line += f"  {'N/A':>{col_w}}"
        print(line)

        rk.fp8_moe_w13_gemm_cleanup()

    print()
    print("Legend:")
    print("  MPK-1CTA:      Mirage MPK, single CTA processes all experts sequentially")
    print("  MPK-8CTA:      Mirage MPK, 8 CTAs (1 per expert), processing in parallel")
    print("  FI-SegGEMM:    FlashInfer SegmentGEMM (BF16, multi-CTA)")
    print("  FI-bmm_fp8:    FlashInfer batched FP8 matmul (per-tensor scale, cuBLAS)")
    print("  FI-grp_fp8:    FlashInfer group_gemm_fp8_nt_groupwise (CUTLASS, block-scaled)")
    print("  FI-grpDG_fp8:  FlashInfer group_deepgemm_fp8_nt_groupwise (DeepGEMM, block-scaled)")
    print("  FI-batDG_fp8:  FlashInfer batch_deepgemm_fp8_nt_groupwise (DeepGEMM, 3D masked)")
    print("  PyTorch:       per-expert BF16 torch.matmul loop")


if __name__ == "__main__":
    main()
