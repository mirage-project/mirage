"""
Benchmark: FP8 MoE Group GEMM — DeepSeek V3 Configuration.

Compares kernel-only latency AND correctness of MoE GEMM with real dimensions:
  - 256 experts, top-8, hidden_size=7168, intermediate_size*2=4096
  - Batch sizes M = 1, 2, 4, 8, 16, 64, 128

Uses FlashInfer's bench_gpu_time for consistent CUPTI-based timing.

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_moe_dsv3
    python setup.py build_ext --inplace
    CUDA_VISIBLE_DEVICES=6 python bench_fp8_moe_gemm.py
"""

import torch
import sys
import random
import numpy as np

try:
    import runtime_kernel_fp8_moe as rk
except ImportError:
    print("ERROR: runtime_kernel_fp8_moe not found.")
    print("Run: python setup.py build_ext --inplace")
    sys.exit(1)

try:
    import runtime_kernel_bf16_moe as rk_bf16
    HAS_BF16 = True
except ImportError:
    print("WARNING: runtime_kernel_bf16_moe not found, skipping BF16 benchmarks.")
    HAS_BF16 = False

try:
    from flashinfer.gemm import (group_gemm_fp8_nt_groupwise,
                                  group_deepgemm_fp8_nt_groupwise)
    from flashinfer import bmm_bf16
    from flashinfer.testing.utils import quantize_fp8 as fi_quantize_fp8
    from flashinfer.testing.utils import bench_gpu_time
    HAS_FLASHINFER = True
except ImportError:
    print("WARNING: FlashInfer not available, skipping FlashInfer benchmarks.")
    HAS_FLASHINFER = False

# ================================================================
# DeepSeek V3 dimensions (must match compiled kernel)
# ================================================================
MPK_BATCH_SIZE = 16     # compiled BATCH_SIZE for MPK kernels (MMA_N=16)
OUTPUT_SIZE    = 4096   # N = 2 * intermediate_size
K              = 7168   # hidden_size
NUM_EXPERTS    = 256
NUM_TOPK       = 8
TILE_SIZE      = 128    # FP8 block quantization granularity


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
    shape = x_fp8.shape
    K_dim = shape[-1]
    num_blocks = K_dim // block_k
    x_blocks = x_fp8.reshape(*shape[:-1], num_blocks, block_k).float()
    return (x_blocks * scale.unsqueeze(-1)).reshape(*shape)


def float32_to_ue8m0_approx(scale):
    bits = scale.view(torch.int32)
    ue8m0 = (bits >> 23) & 0xFF
    return 2.0 ** (ue8m0.float() - 127.0)


def make_random_routing(batch_size, padded_batch, device, seed=42):
    rng = random.Random(seed)
    routing = torch.zeros(NUM_EXPERTS, padded_batch, dtype=torch.int32, device=device)
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


# ================================================================
# Timing helper using FlashInfer's bench_gpu_time
# ================================================================

def bench(fn):
    """Benchmark using FlashInfer's bench_gpu_time. Returns median in microseconds."""
    measurements = bench_gpu_time(fn, dry_run_time_ms=100, repeat_time_ms=1000)
    median_ms = np.median(measurements)
    min_ms = np.min(measurements)
    p99_ms = np.percentile(measurements, 99)
    return median_ms * 1000.0, min_ms * 1000.0, p99_ms * 1000.0  # ms -> us


# ================================================================
# References
# ================================================================

def compute_reference_ue8m0(input_fp8, input_scale, weight_fp8, weight_scale,
                             batch_size, padded_batch, token_to_experts):
    i_scale = float32_to_ue8m0_approx(input_scale)
    w_scale = float32_to_ue8m0_approx(weight_scale)
    input_deq = dequantize_fp8(input_fp8, i_scale).bfloat16()
    ref = torch.zeros(padded_batch, NUM_TOPK, OUTPUT_SIZE,
                      dtype=torch.bfloat16, device=input_fp8.device)
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            w_deq = dequantize_fp8(weight_fp8[e], w_scale[e]).bfloat16()
            ref[i, slot] = (input_deq[i:i+1] @ w_deq.T).squeeze(0)
    return ref


def compute_reference_f32(input_bf16, weight_bf16, batch_size, padded_batch, token_to_experts):
    ref = torch.zeros(padded_batch, NUM_TOPK, OUTPUT_SIZE,
                      dtype=torch.bfloat16, device=input_bf16.device)
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            ref[i, slot] = (input_bf16[i:i+1] @ weight_bf16[e].T).squeeze(0)
    return ref


# ================================================================
# FlashInfer data builders
# ================================================================

def build_fi_weight_fp8(weight_bf16):
    b_fp8, b_scale = fi_quantize_fp8(
        weight_bf16.float(),
        (NUM_EXPERTS, OUTPUT_SIZE // TILE_SIZE, K // TILE_SIZE),
        (1, TILE_SIZE, TILE_SIZE), "K")
    return b_fp8, b_scale


def build_fi_packed_data(batch_size, input_bf16, token_to_experts, device):
    expert_tokens = {e: [] for e in range(NUM_EXPERTS)}
    for tok_idx in range(batch_size):
        for e in token_to_experts[tok_idx]:
            expert_tokens[e].append(tok_idx)

    packed_rows = []
    m_indptr_list = [0]
    m_indices_list = []
    expert_order = []

    for e in range(NUM_EXPERTS):
        tokens = expert_tokens[e]
        for t in tokens:
            packed_rows.append(input_bf16[t])
            m_indices_list.append(e)
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


def build_bmm_bf16_data(batch_size, input_bf16, weight_bf16, token_to_experts):
    device = input_bf16.device
    expert_tokens = {e: [] for e in range(NUM_EXPERTS)}
    for tok_idx in range(batch_size):
        for e in token_to_experts[tok_idx]:
            expert_tokens[e].append(tok_idx)

    active_experts = [e for e in range(NUM_EXPERTS) if len(expert_tokens[e]) > 0]
    if not active_experts:
        return None
    max_m = max(len(expert_tokens[e]) for e in active_experts)
    num_active = len(active_experts)

    a_batched = torch.zeros(num_active, max_m, K, device=device, dtype=torch.bfloat16)
    for idx, e in enumerate(active_experts):
        for local_idx, tok_idx in enumerate(expert_tokens[e]):
            a_batched[idx, local_idx] = input_bf16[tok_idx]

    b_batched = torch.zeros(num_active, K, OUTPUT_SIZE, device=device, dtype=torch.bfloat16)
    for idx, e in enumerate(active_experts):
        b_batched[idx] = weight_bf16[e].T

    out = torch.empty(num_active, max_m, OUTPUT_SIZE, device=device, dtype=torch.bfloat16)
    return a_batched, b_batched, out


# ================================================================
# Correctness helpers
# ================================================================

def compare_fi_vs_ref(fi_out_packed, expert_order, ref, label):
    max_abs, max_rel, n_cmp = 0.0, 0.0, 0
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
    max_abs, max_rel, n_cmp = 0.0, 0.0, 0
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

    # 2D grid configs for FP8: (expert_stride, n_splits, label)
    fp8_grids = [
        (4,  32, "FP8-4x32"),     # 128 CTAs
        (8,  16, "FP8-8x16"),     # 128 CTAs
        (16,  8, "FP8-16x8"),     # 128 CTAs
        (32,  4, "FP8-32x4"),     # 128 CTAs
    ]
    # 2D grid configs for BF16: same set
    bf16_grids = [
        (8,  16, "BF16-8x16"),    # 128 CTAs
        (16,  8, "BF16-16x8"),    # 128 CTAs
        (32,  4, "BF16-32x4"),    # 128 CTAs
    ]

    print("=" * 140)
    print("FP8 MoE Group GEMM Benchmark — DeepSeek V3 Configuration")
    print(f"  N={OUTPUT_SIZE}, K={K}, experts={NUM_EXPERTS}, topk={NUM_TOPK}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Timing: FlashInfer bench_gpu_time (CUPTI or CUDA events)")
    print(f"  FlashInfer: {'available' if HAS_FLASHINFER else 'NOT available'}")
    print("=" * 140)

    # ---- Pre-generate weights ----
    # Generate in bf16, quantize for Mirage FP8, then for FlashInfer.
    # Careful with memory: 256 experts × 4096 × 7168 = ~15GB per copy.
    torch.manual_seed(42)
    weight_bf16 = torch.randn(NUM_EXPERTS, OUTPUT_SIZE, K,
                               device=device, dtype=torch.bfloat16)

    # Mirage FP8 quantization (per-row, per-128-K block)
    print("  Quantizing weights for Mirage FP8...")
    weight_fp8_list, weight_scale_list = [], []
    for e in range(NUM_EXPERTS):
        w_fp8, w_scale = quantize_to_fp8(weight_bf16[e].float())
        weight_fp8_list.append(w_fp8)
        weight_scale_list.append(w_scale)
    weight_fp8 = torch.stack(weight_fp8_list, dim=0)
    weight_scale = torch.stack(weight_scale_list, dim=0)
    del weight_fp8_list, weight_scale_list

    # Keep bf16 weights for BF16 kernel and PyTorch baseline
    w_bf16 = weight_bf16

    # FlashInfer quantization (per-block scales, needs float32 intermediates)
    # Do this AFTER Mirage quantization and free bf16 source to make room.
    fi_b_fp8, fi_b_scale = None, None
    if HAS_FLASHINFER:
        print("  Quantizing weights for FlashInfer...")
        try:
            fi_b_fp8, fi_b_scale = build_fi_weight_fp8(weight_bf16)
            print(f"  Done. b_fp8={fi_b_fp8.shape}, b_scale={fi_b_scale.shape}")
        except torch.OutOfMemoryError:
            print("  WARNING: OOM during FlashInfer weight quantization, skipping FI benchmarks")
            torch.cuda.empty_cache()

    del weight_bf16
    torch.cuda.empty_cache()

    # Collect all results across batch sizes
    all_results = {}  # batch_size -> {label -> (median, min, p99)}
    all_errors = {}   # batch_size -> {label -> (abs, rel, n)}

    for batch_size in [1, 2, 4, 8, 16]:
        torch.manual_seed(100 + batch_size)

        input_bf16 = torch.randn(MPK_BATCH_SIZE, K, device=device, dtype=torch.bfloat16)
        input_fp8, input_scale = quantize_to_fp8(input_bf16.float())

        routing, mask, token_to_experts = make_random_routing(
            batch_size, MPK_BATCH_SIZE, device, seed=100 + batch_size)

        output = torch.zeros(MPK_BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                             dtype=torch.bfloat16, device=device)

        ref_ue8m0 = compute_reference_ue8m0(input_fp8, input_scale, weight_fp8, weight_scale,
                                             batch_size, MPK_BATCH_SIZE, token_to_experts)
        ref_bf16 = compute_reference_f32(input_bf16[:batch_size], w_bf16,
                                          batch_size, MPK_BATCH_SIZE, token_to_experts)

        results = {}
        errors = {}

        # ---- Mirage MPK FP8 ----
        for es, ns, label in fp8_grids:
            rk.fp8_moe_gemm_bench_setup(weight_fp8, es, ns)
            results[label] = bench(
                lambda: rk.fp8_moe_gemm_bench_launch(
                    input_fp8, input_scale, weight_scale,
                    routing, mask, output))
            rk.fp8_moe_gemm_bench_cleanup()
            if label == fp8_grids[0][2]:
                output.zero_()
                rk.fp8_moe_gemm_2d(input_fp8, input_scale, weight_fp8, weight_scale,
                                    routing, mask, output, es, ns)
                errors[label] = compare_mpk_vs_ref(output, ref_ue8m0, batch_size, token_to_experts)

        # ---- Mirage MPK BF16 ----
        if HAS_BF16:
            bf16_input = input_bf16[:MPK_BATCH_SIZE].contiguous()
            bf16_output = torch.zeros(MPK_BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                                       dtype=torch.bfloat16, device=device)
            for es, ns, label in bf16_grids:
                rk_bf16.bf16_moe_bench_setup(w_bf16, es, ns)
                results[label] = bench(
                    lambda: rk_bf16.bf16_moe_bench_launch(
                        bf16_input, routing, mask, bf16_output))
                rk_bf16.bf16_moe_bench_cleanup()
                if label == bf16_grids[0][2]:
                    bf16_output.zero_()
                    rk_bf16.bf16_moe_bench_setup(w_bf16, es, ns)
                    rk_bf16.bf16_moe_bench_launch(bf16_input, routing, mask, bf16_output)
                    torch.cuda.synchronize()
                    rk_bf16.bf16_moe_bench_cleanup()
                    errors[label] = compare_mpk_vs_ref(bf16_output, ref_bf16[:MPK_BATCH_SIZE],
                                                        batch_size, token_to_experts)

        # ---- FlashInfer ----
        if HAS_FLASHINFER and fi_b_fp8 is not None:
            active_input = input_bf16[:batch_size]
            fi_data = build_fi_packed_data(batch_size, active_input, token_to_experts, device)
            if fi_data is not None:
                a_fp8, a_scale, m_indptr, m_indices, expert_order = fi_data
                try:
                    out_grp = [None]
                    def run_grp():
                        out_grp[0] = group_gemm_fp8_nt_groupwise(
                            a_fp8, fi_b_fp8, a_scale, fi_b_scale, m_indptr,
                            scale_major_mode="K", out_dtype=torch.bfloat16)
                    results["FI-grpGEMM"] = bench(run_grp)
                    run_grp()
                    errors["FI-grpGEMM"] = compare_fi_vs_ref(
                        out_grp[0], expert_order, ref_bf16, "FI-grpGEMM")
                except Exception as ex:
                    print(f"    FI-grpGEMM error: {ex}")

                try:
                    out_dg = [None]
                    def run_dg():
                        out_dg[0] = group_deepgemm_fp8_nt_groupwise(
                            a_fp8, fi_b_fp8, a_scale, fi_b_scale, m_indices,
                            out_dtype=torch.bfloat16)
                    results["FI-grpDG"] = bench(run_dg)
                    run_dg()
                    errors["FI-grpDG"] = compare_fi_vs_ref(
                        out_dg[0], expert_order, ref_bf16, "FI-grpDG")
                except Exception as ex:
                    print(f"    FI-grpDG error: {ex}")

            bmm_data = build_bmm_bf16_data(batch_size, active_input, w_bf16, token_to_experts)
            if bmm_data is not None:
                a_bat, b_bat, out_bat = bmm_data
                use_fi_bmm = True
                try:
                    bmm_bf16(a_bat, b_bat, out=out_bat, out_dtype=torch.bfloat16)
                except Exception:
                    use_fi_bmm = False
                if use_fi_bmm:
                    results["FI-bmm16"] = bench(
                        lambda: bmm_bf16(a_bat, b_bat, out=out_bat, out_dtype=torch.bfloat16))
                else:
                    results["FI-bmm16"] = bench(
                        lambda: torch.bmm(a_bat, b_bat, out=out_bat))

        # ---- PyTorch ----
        active_in = input_bf16[:batch_size]
        pt_out = torch.zeros(batch_size, NUM_TOPK, OUTPUT_SIZE,
                              dtype=torch.bfloat16, device=device)
        def run_pytorch():
            for i in range(batch_size):
                for slot, e in enumerate(token_to_experts[i]):
                    pt_out[i, slot] = (active_in[i:i+1] @ w_bf16[e].T).squeeze(0)
        results["PyTorch"] = bench(run_pytorch)

        all_results[batch_size] = results
        all_errors[batch_size] = errors
        print(f"  M={batch_size:>2} done")

    # ================================================================
    # TABLE 1: Main comparison (best MPK configs vs baselines)
    # ================================================================
    print()
    print("=" * 80)
    print("Table 1: Latency Comparison — median (min/p99) in microseconds")
    print("=" * 80)

    main_cols = ["MPK-FP8", "MPK-BF16", "FI-grpGEMM", "FI-bmm16", "PyTorch"]
    # Map display names to result keys
    main_map = {
        "MPK-FP8":    fp8_grids[1][2],     # (8,16) — best FP8 config
        "MPK-BF16":   bf16_grids[0][2],     # (8,16) — best BF16 config
        "FI-grpGEMM": "FI-grpGEMM",
        "FI-bmm16":   "FI-bmm16",
        "PyTorch":    "PyTorch",
    }

    cw = 12  # column width for median
    header = f"{'M':>3}"
    for name in main_cols:
        header += f"  {name:>{cw}}"
    print(header)
    print("-" * len(header))

    for bs in [1, 2, 4, 8, 16]:
        line = f"{bs:>3}"
        for name in main_cols:
            key = main_map[name]
            r = all_results[bs].get(key)
            if r:
                line += f"  {r[0]:>{cw}.1f}"
            else:
                line += f"  {'N/A':>{cw}}"
        print(line)

    # ================================================================
    # TABLE 2: FP8 grid config sweep
    # ================================================================
    print()
    print("=" * 80)
    print("Table 2: FP8 Grid Config Sweep — median latency (us)")
    print("=" * 80)

    cw = 12
    header = f"{'M':>3}"
    for _, _, label in fp8_grids:
        header += f"  {label:>{cw}}"
    print(header)
    print("-" * len(header))

    for bs in [1, 2, 4, 8, 16]:
        line = f"{bs:>3}"
        for _, _, label in fp8_grids:
            r = all_results[bs].get(label)
            line += f"  {r[0]:>{cw}.1f}" if r else f"  {'N/A':>{cw}}"
        print(line)

    # ================================================================
    # TABLE 3: BF16 grid config sweep
    # ================================================================
    if HAS_BF16:
        print()
        print("=" * 80)
        print("Table 3: BF16 Grid Config Sweep — median latency (us)")
        print("=" * 80)

        header = f"{'M':>3}"
        for _, _, label in bf16_grids:
            header += f"  {label:>{cw}}"
        print(header)
        print("-" * len(header))

        for bs in [1, 2, 4, 8, 16]:
            line = f"{bs:>3}"
            for _, _, label in bf16_grids:
                r = all_results[bs].get(label)
                line += f"  {r[0]:>{cw}.1f}" if r else f"  {'N/A':>{cw}}"
            print(line)

    # ================================================================
    # TABLE 4: Correctness (errors)
    # ================================================================
    print()
    print("=" * 80)
    print("Table 4: Correctness — max absolute / relative error")
    print("=" * 80)
    print(f"  MPK-FP8: vs UE8M0 dequant reference")
    print(f"  MPK-BF16 / FI: vs BF16 matmul reference")
    print()

    err_cols = [fp8_grids[0][2], bf16_grids[0][2], "FI-grpGEMM"]
    err_display = ["MPK-FP8", "MPK-BF16", "FI-grpGEMM"]
    ew = 20
    header = f"{'M':>3}"
    for name in err_display:
        header += f"  {name:>{ew}}"
    print(header)
    print("-" * len(header))

    for bs in [1, 2, 4, 8, 16]:
        line = f"{bs:>3}"
        for key in err_cols:
            e = all_errors[bs].get(key)
            if e:
                line += f"  {e[0]:>7.2f} / {e[1]:.4f}"
            else:
                line += f"  {'N/A':>{ew}}"
        print(line)

    # ================================================================
    # Footer
    # ================================================================
    print()
    print("Grid configs (all 128 CTAs):")
    for es, ns, label in fp8_grids:
        print(f"  {label}: grid=({es},{ns},1)")
    if HAS_BF16:
        for es, ns, label in bf16_grids:
            print(f"  {label}: grid=({es},{ns},1)")
    print()
    print("Baselines:")
    if HAS_FLASHINFER:
        print("  FI-grpGEMM: FlashInfer group_gemm_fp8_nt_groupwise (CUTLASS)")
        print("  FI-bmm16:   FlashInfer bmm_bf16 / torch.bmm (batched BF16)")
    print("  PyTorch:    per-expert BF16 torch.matmul loop")
    print()
    print("Timing: FlashInfer bench_gpu_time (CUPTI/CUDA events)")


if __name__ == "__main__":
    main()
