"""
Benchmark: FP8 MoE Group GEMM — DeepSeek V3 Configuration.

Compares kernel-only latency AND correctness of MoE GEMM with real dimensions:
  - 256 experts, top-8, hidden_size=7168, intermediate_size*2=4096
  - Batch sizes M = 1, 2, 4, 8, 16

Benchmarks:
  1. Mirage MPK 2D grid (8x16) — 128 CTAs, target config
  2. FlashInfer group_gemm_fp8_nt_groupwise (CUTLASS, packed segments)
  3. FlashInfer group_deepgemm_fp8_nt_groupwise (DeepGEMM, per-row indices)
  4. PyTorch BF16 per-expert matmul (baseline)

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_moe_dsv3
    python setup.py build_ext --inplace
    CUDA_VISIBLE_DEVICES=6 python bench_fp8_moe_gemm.py
"""

import torch
import sys
import random
import statistics
import math

try:
    import runtime_kernel_fp8_moe_dsv3 as rk
except ImportError:
    print("ERROR: runtime_kernel_fp8_moe_dsv3 not found.")
    print("Run: python setup.py build_ext --inplace")
    sys.exit(1)

try:
    import runtime_kernel_bf16_moe_dsv3 as rk_bf16
    HAS_BF16 = True
except ImportError:
    print("WARNING: runtime_kernel_bf16_moe_dsv3 not found, skipping BF16 benchmarks.")
    HAS_BF16 = False

try:
    from flashinfer.gemm import (group_gemm_fp8_nt_groupwise,
                                  group_deepgemm_fp8_nt_groupwise)
    from flashinfer import bmm_bf16
    from flashinfer.testing.utils import quantize_fp8 as fi_quantize_fp8
    HAS_FLASHINFER = True
except ImportError:
    print("WARNING: FlashInfer not available, skipping FlashInfer benchmarks.")
    HAS_FLASHINFER = False

# ================================================================
# DeepSeek V3 dimensions (must match compiled kernel)
# ================================================================
BATCH_SIZE    = 128
OUTPUT_SIZE   = 4096   # N = 2 * intermediate_size
K             = 7168   # hidden_size
NUM_EXPERTS   = 256
NUM_TOPK      = 8
TILE_SIZE     = 128    # FP8 block quantization granularity

WARMUP_ITERS = 20
BENCH_ITERS  = 100


# ================================================================
# Quantization utilities
# ================================================================

def quantize_to_fp8(x: torch.Tensor, block_k: int = 128):
    """Per-128-element block quantization to FP8 E4M3 (for Mirage kernel)."""
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


def dequantize_fp8(x_fp8, scale, block_k=128):
    """Dequantize FP8 with per-block float32 scales."""
    shape = x_fp8.shape
    K_dim = shape[-1]
    num_blocks = K_dim // block_k
    x_blocks = x_fp8.reshape(*shape[:-1], num_blocks, block_k).float()
    return (x_blocks * scale.unsqueeze(-1)).reshape(*shape)


def float32_to_ue8m0_approx(scale):
    """Convert float32 scale to UE8M0-approximated value (power-of-2 floor)."""
    bits = scale.view(torch.int32)
    ue8m0 = (bits >> 23) & 0xFF
    return 2.0 ** (ue8m0.float() - 127.0)


def make_random_routing(batch_size, device, seed=42):
    rng = random.Random(seed)
    routing = torch.zeros(NUM_EXPERTS, BATCH_SIZE, dtype=torch.int32, device=device)
    token_to_experts = {}
    for i in range(batch_size):
        experts = rng.sample(range(NUM_EXPERTS), NUM_TOPK)
        token_to_experts[i] = experts
        for slot, e in enumerate(experts):
            routing[e, i] = slot + 1

    activated = []
    for e in range(NUM_EXPERTS):
        if routing[e, :batch_size].any():
            activated.append(e)

    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[NUM_EXPERTS] = len(activated)
    return routing, mask, token_to_experts


def bench_kernel(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    latencies = [starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(iters)]
    return statistics.median(latencies), min(latencies), sorted(latencies)[int(iters*0.99)]


# ================================================================
# Reference: compute MoE output using dequantized FP8 with UE8M0 scales
# ================================================================

def compute_reference_ue8m0(input_fp8, input_scale, weight_fp8, weight_scale,
                             batch_size, token_to_experts):
    """Reference using UE8M0-approximated scales (matches Mirage kernel)."""
    i_scale = float32_to_ue8m0_approx(input_scale)
    w_scale = float32_to_ue8m0_approx(weight_scale)
    input_deq = dequantize_fp8(input_fp8, i_scale).bfloat16()

    ref = torch.zeros(BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                      dtype=torch.bfloat16, device=input_fp8.device)
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            w_deq = dequantize_fp8(weight_fp8[e], w_scale[e]).bfloat16()
            ref[i, slot] = (input_deq[i:i+1] @ w_deq.T).squeeze(0)
    return ref


def compute_reference_f32(input_bf16, weight_bf16, batch_size, token_to_experts):
    """Reference using original BF16 data (for comparing FlashInfer and PyTorch)."""
    ref = torch.zeros(BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                      dtype=torch.bfloat16, device=input_bf16.device)
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            ref[i, slot] = (input_bf16[i:i+1] @ weight_bf16[e].T).squeeze(0)
    return ref


# ================================================================
# Mirage MPK benchmark
# ================================================================

def bench_mirage_fp8_2d(input_fp8, input_scale, weight_fp8, weight_scale,
                         routing, mask, output, expert_stride, n_splits):
    """Mirage MPK FP8 2D grid: single kernel launch with grid=(es, ns, 1)."""
    rk.fp8_moe_gemm_bench_setup(weight_fp8, expert_stride, n_splits)
    def fn():
        rk.fp8_moe_gemm_bench_launch(input_fp8, input_scale, weight_scale,
                                       routing, mask, output)
    result = bench_kernel(fn)
    rk.fp8_moe_gemm_bench_cleanup()
    return result


def bench_mirage_bf16_2d(input_bf16, weight_bf16, routing, mask, output,
                          expert_stride, n_splits):
    """Mirage MPK BF16 2D grid: single kernel launch with grid=(es, ns, 1)."""
    rk_bf16.bf16_moe_bench_setup(weight_bf16, expert_stride, n_splits)
    def fn():
        rk_bf16.bf16_moe_bench_launch(input_bf16, routing, mask, output)
    result = bench_kernel(fn)
    rk_bf16.bf16_moe_bench_cleanup()
    return result


# ================================================================
# FlashInfer benchmarks
# ================================================================

def build_fi_weight_fp8(weight_bf16):
    """Pre-quantize weights for FlashInfer (done once)."""
    b_fp8, b_scale = fi_quantize_fp8(
        weight_bf16.float(),
        (NUM_EXPERTS, OUTPUT_SIZE // TILE_SIZE, K // TILE_SIZE),
        (1, TILE_SIZE, TILE_SIZE), "K")
    return b_fp8, b_scale


def build_fi_packed_data(batch_size, input_bf16, token_to_experts, device):
    """Build packed FP8 input for group_gemm and group_deepgemm.

    Returns:
        a_fp8:     [cum_m, K]        packed tokens
        a_scale:   [cum_m, K//128]   per-row, per-K-block
        m_indptr:  [NUM_EXPERTS+1]   cumulative segment boundaries (for group_gemm)
        m_indices: [cum_m]           per-row group index (for group_deepgemm)
        expert_order: list of (token_idx, slot, expert_idx) for result gathering
    """
    expert_tokens = {e: [] for e in range(NUM_EXPERTS)}
    for tok_idx in range(batch_size):
        for e in token_to_experts[tok_idx]:
            expert_tokens[e].append(tok_idx)

    packed_rows = []
    m_indptr_list = [0]
    m_indices_list = []
    expert_order = []  # (token_idx, slot_in_packed, expert_idx)

    for e in range(NUM_EXPERTS):
        tokens = expert_tokens[e]
        for t in tokens:
            packed_rows.append(input_bf16[t])
            m_indices_list.append(e)
            # Find which topk slot this token-expert pair corresponds to
            slot = token_to_experts[t].index(e)
            expert_order.append((t, slot, e))
        m_indptr_list.append(m_indptr_list[-1] + len(tokens))

    cum_m = m_indptr_list[-1]
    if cum_m == 0:
        return None

    a_bf16 = torch.stack(packed_rows)
    m_indptr = torch.tensor(m_indptr_list, dtype=torch.int32, device=device)
    m_indices = torch.tensor(m_indices_list, dtype=torch.int32, device=device)

    a_fp8, a_scale = fi_quantize_fp8(
        a_bf16.float(), (cum_m, K // TILE_SIZE), (1, TILE_SIZE), "K")

    return a_fp8, a_scale, m_indptr, m_indices, expert_order


def run_fi_group_gemm(a_fp8, a_scale, b_fp8, b_scale, m_indptr):
    """Run group_gemm_fp8_nt_groupwise and return output [cum_m, N]."""
    return group_gemm_fp8_nt_groupwise(
        a_fp8, b_fp8, a_scale, b_scale, m_indptr,
        scale_major_mode="K", out_dtype=torch.bfloat16)


def run_fi_group_deepgemm(a_fp8, a_scale, b_fp8, b_scale, m_indices):
    """Run group_deepgemm_fp8_nt_groupwise and return output [cum_m, N]."""
    return group_deepgemm_fp8_nt_groupwise(
        a_fp8, b_fp8, a_scale, b_scale, m_indices,
        out_dtype=torch.bfloat16)


def bench_fi_group_gemm(a_fp8, a_scale, b_fp8, b_scale, m_indptr):
    """Benchmark group_gemm_fp8_nt_groupwise."""
    def fn():
        group_gemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale, m_indptr,
            scale_major_mode="K", out_dtype=torch.bfloat16)
    return bench_kernel(fn)


def bench_fi_group_deepgemm(a_fp8, a_scale, b_fp8, b_scale, m_indices):
    """Benchmark group_deepgemm_fp8_nt_groupwise."""
    def fn():
        group_deepgemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale, m_indices,
            out_dtype=torch.bfloat16)
    return bench_kernel(fn)


# ================================================================
# FlashInfer bmm_bf16 benchmark
#
# bmm_bf16(a, b, out=out) computes out = a @ b where:
#   a: [batch, m, k],  b: [batch, k, n]  ->  out: [batch, m, n]
#
# For MoE, we batch per-expert: one "batch element" per active expert.
# Each element: a=[1, K] (one token), b=[K, N] (expert weight transposed).
# We pad all experts to the same m (max tokens per expert) and batch them.
# ================================================================

def build_bmm_bf16_data(batch_size, input_bf16, weight_bf16, token_to_experts):
    """Prepare batched BF16 inputs for bmm_bf16, one batch elem per active expert."""
    device = input_bf16.device

    # Group tokens by expert
    expert_tokens = {e: [] for e in range(NUM_EXPERTS)}
    for tok_idx in range(batch_size):
        for e in token_to_experts[tok_idx]:
            expert_tokens[e].append(tok_idx)

    # Find active experts and max tokens per expert
    active_experts = [e for e in range(NUM_EXPERTS) if len(expert_tokens[e]) > 0]
    if not active_experts:
        return None
    max_m = max(len(expert_tokens[e]) for e in active_experts)
    num_active = len(active_experts)

    # Build batched A: [num_active, max_m, K]
    a_batched = torch.zeros(num_active, max_m, K, device=device, dtype=torch.bfloat16)
    for idx, e in enumerate(active_experts):
        for local_idx, tok_idx in enumerate(expert_tokens[e]):
            a_batched[idx, local_idx] = input_bf16[tok_idx]

    # Build batched B: [num_active, K, N]  (= weight.T per expert)
    b_batched = torch.zeros(num_active, K, OUTPUT_SIZE, device=device, dtype=torch.bfloat16)
    for idx, e in enumerate(active_experts):
        b_batched[idx] = weight_bf16[e].T  # [N, K].T = [K, N]

    out = torch.empty(num_active, max_m, OUTPUT_SIZE, device=device, dtype=torch.bfloat16)
    return a_batched, b_batched, out


def bench_fi_bmm_bf16(batch_size, input_bf16, weight_bf16, token_to_experts):
    """FlashInfer bmm_bf16 or torch.bmm fallback: batched matmul per expert."""
    data = build_bmm_bf16_data(batch_size, input_bf16, weight_bf16, token_to_experts)
    if data is None:
        return None
    a_batched, b_batched, out = data

    # Try FlashInfer bmm_bf16, fall back to torch.bmm if it fails (e.g. cuDNN issues)
    use_flashinfer_bmm = True
    try:
        bmm_bf16(a_batched, b_batched, out=out, out_dtype=torch.bfloat16)
    except Exception:
        use_flashinfer_bmm = False

    if use_flashinfer_bmm:
        def fn():
            bmm_bf16(a_batched, b_batched, out=out, out_dtype=torch.bfloat16)
    else:
        # Fallback: torch.bmm (cuBLAS)
        def fn():
            torch.bmm(a_batched, b_batched, out=out)

    return bench_kernel(fn)


# ================================================================
# PyTorch baseline
# ================================================================

def bench_pytorch(batch_size, input_bf16, weight_bf16, token_to_experts):
    output = torch.zeros(batch_size, NUM_TOPK, OUTPUT_SIZE,
                         dtype=torch.bfloat16, device=input_bf16.device)
    def fn():
        for i in range(batch_size):
            for slot, e in enumerate(token_to_experts[i]):
                output[i, slot] = (input_bf16[i:i+1] @ weight_bf16[e].T).squeeze(0)
    return bench_kernel(fn)


# ================================================================
# Correctness comparison helper
# ================================================================

def compare_fi_vs_ref(fi_out_packed, expert_order, ref, label):
    """Compare FlashInfer packed output against reference [B, topk, N] output.

    fi_out_packed: [cum_m, N] — rows in expert-packed order
    expert_order:  list of (token_idx, topk_slot, expert_idx)
    ref:           [B, topk, N] reference output

    Returns (max_abs_err, max_rel_err, num_compared).
    """
    max_abs = 0.0
    max_rel = 0.0
    n_cmp = 0
    for row_idx, (tok, slot, _) in enumerate(expert_order):
        fi_row = fi_out_packed[row_idx].float()
        ref_row = ref[tok, slot].float()
        diff = (fi_row - ref_row).abs()
        abs_err = diff.max().item()
        denom = ref_row.abs().max().item()
        rel_err = abs_err / max(denom, 1e-6)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        n_cmp += 1
    return max_abs, max_rel, n_cmp


def compare_mpk_vs_ref(mpk_out, ref, batch_size, token_to_experts):
    """Compare Mirage MPK output [B, topk, N] against reference."""
    max_abs = 0.0
    max_rel = 0.0
    n_cmp = 0
    for i in range(batch_size):
        for slot in range(len(token_to_experts[i])):
            out_row = mpk_out[i, slot].float()
            ref_row = ref[i, slot].float()
            diff = (out_row - ref_row).abs()
            abs_err = diff.max().item()
            denom = ref_row.abs().max().item()
            rel_err = abs_err / max(denom, 1e-6)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            n_cmp += 1
    return max_abs, max_rel, n_cmp


def fmt_us(median, min_val, p99):
    return f"{median:8.1f} ({min_val:7.1f}/{p99:8.1f})"


def fmt_err(abs_err, rel_err, n_cmp):
    return f"abs={abs_err:.4f} rel={rel_err:.4f} ({n_cmp})"


def main():
    device = torch.device("cuda")

    # 2D grid configs: (expert_stride, n_splits, label)
    grid_configs = [
        (8, 16, "MPK-8x16"),
    ]

    print("=" * 140)
    print("FP8 MoE Group GEMM Benchmark — DeepSeek V3 Configuration")
    print(f"  N={OUTPUT_SIZE}, K={K}, experts={NUM_EXPERTS}, topk={NUM_TOPK}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Warmup={WARMUP_ITERS}, Iters={BENCH_ITERS}")
    print(f"  FlashInfer: {'available' if HAS_FLASHINFER else 'NOT available'}")
    print("=" * 140)

    # ---- Pre-generate weights (shared across batch sizes) ----
    torch.manual_seed(42)
    weight_bf16 = torch.randn(NUM_EXPERTS, OUTPUT_SIZE, K,
                               device=device, dtype=torch.bfloat16)

    # Mirage-format quantization
    weight_fp8_list, weight_scale_list = [], []
    for e in range(NUM_EXPERTS):
        w_fp8, w_scale = quantize_to_fp8(weight_bf16[e].float())
        weight_fp8_list.append(w_fp8)
        weight_scale_list.append(w_scale)
    weight_fp8 = torch.stack(weight_fp8_list, dim=0)
    weight_scale = torch.stack(weight_scale_list, dim=0)

    # FlashInfer-format quantization
    fi_b_fp8, fi_b_scale = None, None
    if HAS_FLASHINFER:
        print("  Pre-quantizing weights for FlashInfer...")
        fi_b_fp8, fi_b_scale = build_fi_weight_fp8(weight_bf16)
        print(f"  Done. b_fp8={fi_b_fp8.shape}, b_scale={fi_b_scale.shape}")

    # Keep weight_bf16 for BF16 benchmark and PyTorch baseline
    w_bf16 = weight_bf16
    del weight_bf16
    torch.cuda.empty_cache()

    # ---- Benchmark + correctness per batch size ----
    # Build bench_names: FP8 configs, then BF16 config, then FlashInfer, then PyTorch
    bench_names = [cfg[2] for cfg in grid_configs]
    # BF16 kernel with 2D grid matching FP8 (BATCH_SIZE=16 in BF16 wrapper)
    bf16_grid = (8, 16, "BF16-8x16")
    if HAS_BF16:
        bench_names += [bf16_grid[2]]
    if HAS_FLASHINFER:
        bench_names += ["FI-grpGEMM", "FI-grpDG", "FI-bmm16"]
    bench_names += ["PyTorch"]

    # Error column names
    err_names = [grid_configs[0][2]]
    if HAS_BF16:
        err_names += [bf16_grid[2]]
    if HAS_FLASHINFER:
        err_names += ["FI-grpGEMM", "FI-grpDG"]

    # Print header
    col_w = 30
    err_w = 30
    header = f"{'M':>3}"
    for name in bench_names:
        header += f"  {(name + ' (us)'):>{col_w}}"
    for name in err_names:
        header += f"  {(name + ' err'):>{err_w}}"
    print()
    print(header)
    print("-" * len(header))

    for batch_size in [1, 2, 4, 8, 16]:
        torch.manual_seed(100 + batch_size)

        input_bf16 = torch.randn(BATCH_SIZE, K, device=device, dtype=torch.bfloat16)
        input_fp8, input_scale = quantize_to_fp8(input_bf16.float())

        routing, mask, token_to_experts = make_random_routing(batch_size, device)

        output = torch.zeros(BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                             dtype=torch.bfloat16, device=device)

        # ---- Compute references ----
        # UE8M0 reference: matches Mirage kernel's scale approximation
        ref_ue8m0 = compute_reference_ue8m0(input_fp8, input_scale, weight_fp8, weight_scale,
                                             batch_size, token_to_experts)
        # BF16 reference: for FlashInfer/PyTorch (they use their own quantization)
        ref_bf16 = compute_reference_f32(input_bf16[:batch_size], w_bf16,
                                          batch_size, token_to_experts)

        results = {}
        errors = {}

        # ---- Mirage MPK ----
        for es, ns, label in grid_configs:
            output.zero_()
            results[label] = bench_mirage_fp8_2d(
                input_fp8, input_scale, weight_fp8, weight_scale,
                routing, mask, output, es, ns)
            # Run once more for correctness (bench_launch doesn't sync output cleanly)
            output.zero_()
            rk.fp8_moe_gemm_2d(input_fp8, input_scale, weight_fp8, weight_scale,
                                routing, mask, output, es, ns)
            errors[label] = compare_mpk_vs_ref(output, ref_ue8m0, batch_size, token_to_experts)

        # ---- Mirage MPK BF16 ----
        # BF16 kernel compiled with BATCH_SIZE=16 (MMA_N), so we need
        # smaller tensors. Routing/mask use the same token_to_experts.
        BF16_BATCH = 16
        if HAS_BF16:
            es, ns, label = bf16_grid
            bf16_input = input_bf16[:BF16_BATCH].contiguous()
            bf16_routing = routing[:, :BF16_BATCH].contiguous()
            bf16_output = torch.zeros(BF16_BATCH, NUM_TOPK, OUTPUT_SIZE,
                                       dtype=torch.bfloat16, device=device)

            results[label] = bench_mirage_bf16_2d(
                bf16_input, w_bf16, bf16_routing, mask, bf16_output, es, ns)
            # Correctness
            bf16_output.zero_()
            rk_bf16.bf16_moe_bench_setup(w_bf16, es, ns)
            rk_bf16.bf16_moe_bench_launch(bf16_input, bf16_routing, mask, bf16_output)
            torch.cuda.synchronize()
            rk_bf16.bf16_moe_bench_cleanup()
            # Compare against BF16 reference (only for active tokens)
            errors[label] = compare_mpk_vs_ref(bf16_output, ref_bf16[:BF16_BATCH],
                                                batch_size, token_to_experts)

        # ---- FlashInfer ----
        if HAS_FLASHINFER and fi_b_fp8 is not None:
            active_input = input_bf16[:batch_size]
            fi_data = build_fi_packed_data(batch_size, active_input,
                                            token_to_experts, device)
            if fi_data is not None:
                a_fp8, a_scale, m_indptr, m_indices, expert_order = fi_data

                # group_gemm (CUTLASS) — compare against BF16 reference
                try:
                    results["FI-grpGEMM"] = bench_fi_group_gemm(
                        a_fp8, a_scale, fi_b_fp8, fi_b_scale, m_indptr)
                    fi_out = run_fi_group_gemm(
                        a_fp8, a_scale, fi_b_fp8, fi_b_scale, m_indptr)
                    errors["FI-grpGEMM"] = compare_fi_vs_ref(
                        fi_out, expert_order, ref_bf16, "FI-grpGEMM")
                except Exception as ex:
                    print(f"    FI-grpGEMM error: {ex}")

                # group_deepgemm (DeepGEMM) — compare against BF16 reference
                try:
                    results["FI-grpDG"] = bench_fi_group_deepgemm(
                        a_fp8, a_scale, fi_b_fp8, fi_b_scale, m_indices)
                    fi_out_dg = run_fi_group_deepgemm(
                        a_fp8, a_scale, fi_b_fp8, fi_b_scale, m_indices)
                    errors["FI-grpDG"] = compare_fi_vs_ref(
                        fi_out_dg, expert_order, ref_bf16, "FI-grpDG")
                except Exception as ex:
                    print(f"    FI-grpDG error: {ex}")

            # bmm_bf16: batched BF16 matmul (one batch elem per active expert)
            try:
                results["FI-bmm16"] = bench_fi_bmm_bf16(
                    batch_size, active_input, w_bf16, token_to_experts)
            except Exception as ex:
                print(f"    FI-bmm16 error: {ex}")

        # ---- PyTorch baseline ----
        results["PyTorch"] = bench_pytorch(
            batch_size, input_bf16[:batch_size], w_bf16,
            token_to_experts)

        # ---- Print row ----
        line = f"{batch_size:>3}"
        for name in bench_names:
            r = results.get(name)
            if r:
                line += f"  {fmt_us(*r):>{col_w}}"
            else:
                line += f"  {'N/A':>{col_w}}"

        # Errors
        for name in err_names:
            e = errors.get(name)
            line += f"  {fmt_err(*e):>{err_w}}" if e else f"  {'N/A':>{err_w}}"
        print(line)

    print()
    print("Legend:")
    for es, ns, label in grid_configs:
        print(f"  {label:<12}  Mirage MPK FP8, grid=({es},{ns},1), {es*ns} CTAs")
    if HAS_BF16:
        es, ns, label = bf16_grid
        print(f"  {label:<12}  Mirage MPK BF16, grid=({es},{ns},1), {es*ns} CTAs")
    if HAS_FLASHINFER:
        print("  FI-grpGEMM   FlashInfer group_gemm_fp8_nt_groupwise (CUTLASS, packed segments)")
        print("  FI-grpDG     FlashInfer group_deepgemm_fp8_nt_groupwise (DeepGEMM, per-row indices)")
        print("  FI-bmm16     FlashInfer bmm_bf16 (batched BF16 matmul, one batch elem per expert)")
    print("  PyTorch      per-expert BF16 torch.matmul loop")
    print()
    print("Error columns: max absolute error, max relative error, (num elements compared)")
    print("  MPK-FP8 err: vs UE8M0-approximated dequant reference (matches kernel's scale conversion)")
    print("  BF16/FI err: vs BF16 matmul reference (quantization error)")


if __name__ == "__main__":
    main()
