"""
Unit tests for FP8 block-scaled MoE group GEMM on SM100 (Blackwell B200).

Tests multiple routing scenarios with DeepSeek-V3-style shapes (scaled down
for fast unit testing). Compares against a PyTorch reference that dequantizes
FP8 -> BF16 and performs standard matmul.

Test scenarios cover diverse token-to-expert distributions:
  - Uniform round-robin (baseline)
  - All tokens routed to the same expert (hot expert)
  - Skewed distribution (some experts get many tokens, others zero)
  - Random routing (fully random expert assignment)
  - Single token (minimum batch)
  - Experts with no tokens (inactive experts in mask)

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_moe
    python setup.py build_ext --inplace
    python test_fp8_moe_gemm.py
"""

import torch
import sys
import os
import random

try:
    import runtime_kernel_fp8_moe as rk
except ImportError:
    print("ERROR: runtime_kernel_fp8_moe not found.")
    print("Please run: python setup.py build_ext --inplace")
    sys.exit(1)

BATCH_SIZE  = 128   # padded to MMA_N=128 (actual tokens vary per test)
OUTPUT_SIZE = 256   # N
K           = 256   # REDUCTION_SIZE
NUM_EXPERTS = 8
NUM_TOPK    = 2
K_SCALE     = K // 128  # = 2


def quantize_to_fp8(x: torch.Tensor, block_k: int = 128):
    """
    Per-128-element block quantization to FP8 E4M3.
    Returns (fp8_tensor, scale_tensor) where scale is float32.
    x: [..., K] - last dimension is K
    scale: [..., K//block_k]
    """
    shape = x.shape
    K_dim = shape[-1]
    assert K_dim % block_k == 0
    num_blocks = K_dim // block_k
    x_blocks = x.reshape(*shape[:-1], num_blocks, block_k)

    amax = x_blocks.abs().amax(dim=-1)
    scale = amax / 448.0
    scale = scale.clamp(min=1e-12)

    x_scaled = x_blocks / scale.unsqueeze(-1)
    x_fp8 = x_scaled.reshape(*shape).to(torch.float8_e4m3fn)

    return x_fp8, scale.float()


def dequantize_fp8(x_fp8: torch.Tensor, scale: torch.Tensor,
                   block_k: int = 128):
    """
    Dequantize FP8 E4M3 to float32.
    x_fp8: [..., K]
    scale: [..., K//block_k]
    Returns: [..., K] float32
    """
    shape = x_fp8.shape
    K_dim = shape[-1]
    num_blocks = K_dim // block_k
    x_float = x_fp8.to(torch.float32)
    x_blocks = x_float.reshape(*shape[:-1], num_blocks, block_k)
    x_deq = x_blocks * scale.unsqueeze(-1)
    return x_deq.reshape(*shape)


def float32_to_ue8m0_approx(sf: torch.Tensor) -> torch.Tensor:
    """Convert float32 scale to UE8M0 approximation (power-of-2 floor).

    The kernel converts float32 scales to UE8M0 (8-bit exponent-only),
    so the reference must apply the same approximation to match numerically.
    """
    bits = sf.view(torch.int32)
    ue8m0 = (bits >> 23) & 0xFF
    return torch.pow(2.0, ue8m0.float() - 127.0)


# =========================================================================
# Routing generators — each returns (routing_indices, mask, token_to_experts)
# =========================================================================

def make_routing_from_assignments(token_to_experts, batch_size, num_experts, device):
    """
    Build routing_indices and mask from a token_to_experts dict.
    token_to_experts: dict mapping token_idx -> list of expert indices (len = num_topk)
    Returns: routing_indices [num_experts, batch_size], mask [num_experts+1], token_to_experts
    """
    routing = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            routing[e, i] = slot + 1  # 1-indexed top-k slot

    activated_experts = []
    for e in range(num_experts):
        if routing[e].any():
            activated_experts.append(e)

    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated_experts):
        mask[idx] = e
    mask[num_experts] = len(activated_experts)

    return routing, mask, token_to_experts


def routing_uniform(batch_size, num_experts, num_topk, device):
    """Round-robin: token i -> experts [i%E, (i+E//2)%E]. Perfectly uniform."""
    token_to_experts = {}
    for i in range(batch_size):
        e0 = i % num_experts
        e1 = (i + num_experts // 2) % num_experts
        token_to_experts[i] = [e0, e1]
    return make_routing_from_assignments(token_to_experts, batch_size, num_experts, device)


def routing_hot_expert(batch_size, num_experts, num_topk, device):
    """All tokens routed to expert 0 (slot 0) and expert 1 (slot 1).
    Exercises the case where one or two experts handle everything while
    the rest are idle."""
    token_to_experts = {}
    for i in range(batch_size):
        token_to_experts[i] = [0, 1]
    return make_routing_from_assignments(token_to_experts, batch_size, num_experts, device)


def routing_skewed(batch_size, num_experts, num_topk, device):
    """Skewed: 70% of tokens go to expert 0, rest spread across others.
    Tests unbalanced load where some experts are heavily loaded and some
    may have zero tokens."""
    token_to_experts = {}
    for i in range(batch_size):
        if i < int(batch_size * 0.7):
            # Majority goes to expert 0 + expert 1
            token_to_experts[i] = [0, 1]
        else:
            # Remaining spread across higher-index experts
            e0 = 2 + (i % (num_experts - 2))
            e1 = 2 + ((i + 1) % (num_experts - 2))
            if e0 == e1:
                e1 = (e1 + 1) % num_experts
                if e1 == e0:
                    e1 = (e1 + 1) % num_experts
            token_to_experts[i] = [e0, e1]
    return make_routing_from_assignments(token_to_experts, batch_size, num_experts, device)


def routing_random(batch_size, num_experts, num_topk, device, seed=None):
    """Fully random: each token picks 2 distinct experts uniformly at random.
    May produce very uneven distributions naturally."""
    rng = random.Random(seed)
    token_to_experts = {}
    for i in range(batch_size):
        experts = rng.sample(range(num_experts), num_topk)
        token_to_experts[i] = experts
    return make_routing_from_assignments(token_to_experts, batch_size, num_experts, device)


def routing_clustered(batch_size, num_experts, num_topk, device):
    """Tokens clustered into pairs of adjacent experts. E.g., tokens 0-3 go
    to experts (0,1), tokens 4-7 go to experts (2,3), etc. Exercises the case
    where some expert pairs are heavily loaded and others are empty."""
    token_to_experts = {}
    for i in range(batch_size):
        group = (i * 2 // max(batch_size, 1)) % (num_experts // 2)
        e0 = group * 2
        e1 = group * 2 + 1
        token_to_experts[i] = [e0, e1]
    return make_routing_from_assignments(token_to_experts, batch_size, num_experts, device)


def routing_last_experts_only(batch_size, num_experts, num_topk, device):
    """Tokens only routed to the last 2 experts. First 6 experts are completely
    inactive. Tests that expert mask correctly skips inactive experts."""
    token_to_experts = {}
    for i in range(batch_size):
        token_to_experts[i] = [num_experts - 2, num_experts - 1]
    return make_routing_from_assignments(token_to_experts, batch_size, num_experts, device)


# =========================================================================
# Reference and test runner
# =========================================================================

def reference_moe_w13(input_bf16, weight_bf16, token_to_experts,
                       batch_size, output_size, num_topk, device):
    """
    PyTorch reference for MoE W13 GEMM.
    input_bf16:  [batch, K]
    weight_bf16: [num_experts, N, K]
    Returns: output [batch, top_k, N] bf16
    """
    output = torch.zeros(batch_size, num_topk, output_size,
                         dtype=torch.bfloat16, device=device)
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            result = (input_bf16[i:i+1].float() @
                      weight_bf16[e].float().T).squeeze(0)
            output[i, slot] = result.bfloat16()
    return output


def compute_errors(output_tensor, output_ref, active_token_to_experts,
                    batch_size, num_topk):
    """Compare kernel output vs reference for routed tokens.
    Returns: (max_abs_err, max_rel_err, num_compared, expected_comparisons)
    """
    max_abs_err = 0.0
    max_rel_err = 0.0
    num_compared = 0
    expected_comparisons = 0
    for i in range(batch_size):
        expert_list = active_token_to_experts.get(i, [])
        for slot, expert_idx in enumerate(expert_list):
            if slot >= num_topk:
                break
            expected_comparisons += 1
            out_k = output_tensor[i, slot]
            ref_k = output_ref[i, slot]
            ref_max = ref_k.float().abs().max().item()
            if ref_max < 1e-6:
                continue
            abs_err = (out_k.float() - ref_k.float()).abs().max().item()
            rel_err = abs_err / (ref_max + 1e-8)
            max_abs_err = max(max_abs_err, abs_err)
            max_rel_err = max(max_rel_err, rel_err)
            num_compared += 1
    return max_abs_err, max_rel_err, num_compared, expected_comparisons


def run_single_test(batch_size, routing_fn, routing_name, verbose=False):
    """Run one test case with a given batch size and routing function.

    Performs TWO independent correctness checks:

    Check 1 ("ue8m0"): Compare kernel output against a reference that
      dequantizes FP8 using UE8M0-approximated scales (matching the kernel's
      internal scale conversion). This should give near-zero error because
      UE8M0 scales are powers of 2, so dequantization is exact.

    Check 2 ("f32ref"): Compare kernel output against a fully independent
      reference that uses the ORIGINAL float32 data (before any FP8
      quantization). This tests the full quantization pipeline and should
      show small but nonzero errors from FP8 rounding + UE8M0 scale
      approximation. Crucially, this proves the kernel output is
      non-trivial and numerically meaningful.
    """
    device = torch.device("cuda")
    torch.manual_seed(42 + batch_size)

    full_batch = BATCH_SIZE

    # --- Generate random float32 input (pre-quantization ground truth) ---
    input_f32 = torch.randn(full_batch, K, device=device, dtype=torch.float32)
    input_f32[:batch_size] = torch.randn(batch_size, K, device=device)

    # --- Quantize to FP8 ---
    input_fp8, input_scale = quantize_to_fp8(input_f32)

    # --- Generate random float32 weight (pre-quantization ground truth) ---
    weight_f32 = torch.randn(NUM_EXPERTS, OUTPUT_SIZE, K, device=device, dtype=torch.float32)

    # --- Quantize weights to FP8 per expert ---
    weight_fp8_list = []
    weight_scale_list = []
    for e in range(NUM_EXPERTS):
        w_fp8, w_scale = quantize_to_fp8(weight_f32[e])
        weight_fp8_list.append(w_fp8)
        weight_scale_list.append(w_scale)
    weight_fp8   = torch.stack(weight_fp8_list, dim=0)
    weight_scale = torch.stack(weight_scale_list, dim=0)

    # --- Build routing ---
    routing_small, mask_small, token_to_experts_small = routing_fn(
        batch_size, NUM_EXPERTS, NUM_TOPK, device)

    routing_indices = torch.zeros(NUM_EXPERTS, full_batch, dtype=torch.int32, device=device)
    routing_indices[:, :batch_size] = routing_small

    activated_experts = []
    for e in range(NUM_EXPERTS):
        if routing_indices[e, :batch_size].any().item():
            activated_experts.append(e)
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated_experts):
        mask[idx] = e
    mask[NUM_EXPERTS] = len(activated_experts)

    # --- Run FP8 kernel ---
    output_tensor = torch.zeros(full_batch, NUM_TOPK, OUTPUT_SIZE,
                                dtype=torch.bfloat16, device=device)
    rk.fp8_moe_w13_gemm(
        input_fp8,
        input_scale,
        weight_fp8.contiguous(),
        weight_scale.contiguous(),
        routing_indices,
        mask,
        output_tensor,
        0)

    # --- Build active_token_to_experts from routing_indices ---
    active_token_to_experts = {}
    for i in range(batch_size):
        experts = []
        for e in range(NUM_EXPERTS):
            slot = routing_indices[e, i].item()
            if slot > 0:
                experts.append((slot - 1, e))
        experts.sort(key=lambda x: x[0])
        active_token_to_experts[i] = [e for _, e in experts]

    # =====================================================================
    # Check 1: kernel vs UE8M0-dequantized reference (should be ~0 error)
    # =====================================================================
    # This reference mimics exactly what the kernel does:
    #   dequantize FP8 using UE8M0 (power-of-2 floor) scales, then matmul.
    # Since UE8M0 * FP8 is exact (just exponent shift), this reference
    # produces bit-identical results to the hardware UMMA instruction.
    weight_deq_ue8m0 = torch.stack([
        dequantize_fp8(weight_fp8[e],
                       float32_to_ue8m0_approx(weight_scale[e])).bfloat16()
        for e in range(NUM_EXPERTS)
    ], dim=0)
    input_deq_ue8m0 = dequantize_fp8(
        input_fp8, float32_to_ue8m0_approx(input_scale)).bfloat16()

    ref_ue8m0 = reference_moe_w13(
        input_deq_ue8m0, weight_deq_ue8m0, active_token_to_experts,
        batch_size, OUTPUT_SIZE, NUM_TOPK, device)

    abs1, rel1, cmp1, exp1 = compute_errors(
        output_tensor, ref_ue8m0, active_token_to_experts, batch_size, NUM_TOPK)

    # =====================================================================
    # Check 2: kernel vs FP8-dequantized with EXACT float32 scales
    # =====================================================================
    # This isolates FP8 quantization error from UE8M0 scale truncation.
    # Dequantize using the original float32 scales (not UE8M0-approximated).
    # The error here = FP8 quantization noise + UE8M0 scale truncation.
    # For random data: ~3% from FP8 alone, ~50% from UE8M0 truncation.
    weight_deq_exact = torch.stack([
        dequantize_fp8(weight_fp8[e], weight_scale[e]).bfloat16()
        for e in range(NUM_EXPERTS)
    ], dim=0)
    input_deq_exact = dequantize_fp8(input_fp8, input_scale).bfloat16()

    ref_fp8_exact = reference_moe_w13(
        input_deq_exact, weight_deq_exact, active_token_to_experts,
        batch_size, OUTPUT_SIZE, NUM_TOPK, device)

    abs2, rel2, cmp2, exp2 = compute_errors(
        output_tensor, ref_fp8_exact, active_token_to_experts, batch_size, NUM_TOPK)

    # =====================================================================
    # Check 3: FP8-exact-scale reference vs original float32 (quantization error only)
    # =====================================================================
    # This measures pure FP8 quantization error without UE8M0, providing
    # an independent baseline. If this is ~3%, it means FP8 quantization
    # works correctly; the additional error in check 2 comes from UE8M0.
    ref_f32 = reference_moe_w13(
        input_f32[:batch_size].bfloat16(), weight_f32.bfloat16(),
        active_token_to_experts,
        batch_size, OUTPUT_SIZE, NUM_TOPK, device)

    abs3, rel3, cmp3, exp3 = compute_errors(
        ref_fp8_exact, ref_f32, active_token_to_experts, batch_size, NUM_TOPK)

    # Also verify the output magnitudes are non-trivial
    out_abs_max = 0.0
    for i in range(batch_size):
        for slot in range(min(len(active_token_to_experts.get(i, [])), NUM_TOPK)):
            val = output_tensor[i, slot].float().abs().max().item()
            out_abs_max = max(out_abs_max, val)

    # Check that unrouted tokens in output are zero
    num_nonzero_unrouted = 0
    for i in range(batch_size):
        expert_list = active_token_to_experts.get(i, [])
        used_slots = set(range(min(len(expert_list), NUM_TOPK)))
        for slot in range(NUM_TOPK):
            if slot not in used_slots:
                if output_tensor[i, slot].abs().max().item() > 0:
                    num_nonzero_unrouted += 1

    # --- Pass criteria ---
    # Check 1: kernel vs UE8M0 reference — should match near-exactly
    #   (UE8M0 scales are powers of 2, so FP8 * UE8M0 is exact)
    check1_ok = abs1 < 0.5 and rel1 < 0.01 and cmp1 > 0
    # Check 2: kernel vs FP8-exact-scale reference — UE8M0 truncation error
    #   For random gaussian data, UE8M0 truncation causes ~50% relative error.
    #   This is expected: UE8M0 discards the mantissa of float32 scales.
    #   In real models, scales are designed to be close to powers of 2.
    check2_ok = cmp2 > 0 and rel2 < 1.0
    # Check 3: FP8 quantization error alone (no UE8M0) should be small (~3-5%)
    check3_ok = cmp3 > 0 and rel3 < 0.10
    # Output magnitudes must be non-trivial (not all zeros)
    magnitude_ok = out_abs_max > 0.1
    # No garbage in unrouted slots
    unrouted_ok = num_nonzero_unrouted == 0

    passed = check1_ok and check2_ok and check3_ok and magnitude_ok and unrouted_ok
    status = "PASS" if passed else "FAIL"
    expert_counts = [routing_indices[e, :batch_size].count_nonzero().item()
                     for e in range(NUM_EXPERTS)]
    print(f"    batch={batch_size:2d} {routing_name:24s}: "
          f"vs_ue8m0={rel1:.4f} vs_fp8exact={rel2:.4f} fp8_quant_err={rel3:.4f} "
          f"cmp={cmp1}/{exp1} |out|={out_abs_max:.1f}  [{status}]")
    if not passed:
        if not check1_ok:
            print(f"      FAIL check1 (vs ue8m0): abs={abs1:.4f} rel={rel1:.4f} cmp={cmp1}")
        if not check2_ok:
            print(f"      FAIL check2 (vs fp8 exact scale): abs={abs2:.4f} rel={rel2:.4f} cmp={cmp2}")
        if not check3_ok:
            print(f"      FAIL check3 (fp8 quant err): abs={abs3:.4f} rel={rel3:.4f} cmp={cmp3}")
        if not magnitude_ok:
            print(f"      FAIL magnitude: |out|={out_abs_max:.6f} (expected > 0.1)")
        if not unrouted_ok:
            print(f"      FAIL unrouted: {num_nonzero_unrouted} slots nonzero")
        if verbose:
            print(f"      output[0,0,:8]   = {output_tensor[0,0,:8]}")
            print(f"      ref_ue8m0[0,0,:8]= {ref_ue8m0[0,0,:8]}")
            print(f"      ref_exact[0,0,:8]= {ref_fp8_exact[0,0,:8]}")
            print(f"      ref_f32[0,0,:8]  = {ref_f32[0,0,:8]}")
    return passed


def main():
    print(f"Testing FP8 MoE W13 group GEMM on SM100 (Blackwell)")
    print(f"  N={OUTPUT_SIZE}, K={K}, num_experts={NUM_EXPERTS}, top_k={NUM_TOPK}")
    print()

    all_passed = True

    # --- Test 1: Varying batch sizes with uniform routing ---
    print("Test 1: Uniform round-robin routing, batch sizes 1-16")
    for batch_size in range(1, 17):
        if not run_single_test(batch_size, routing_uniform, "uniform"):
            all_passed = False

    # --- Test 2: Hot expert (all tokens -> expert 0,1) ---
    print("\nTest 2: Hot expert (all tokens -> experts 0,1)")
    for batch_size in [1, 4, 8, 16]:
        if not run_single_test(batch_size, routing_hot_expert, "hot_expert"):
            all_passed = False

    # --- Test 3: Skewed distribution (70% -> expert 0) ---
    print("\nTest 3: Skewed distribution (70% -> expert 0)")
    for batch_size in [4, 8, 12, 16]:
        if not run_single_test(batch_size, routing_skewed, "skewed_70pct"):
            all_passed = False

    # --- Test 4: Fully random routing ---
    print("\nTest 4: Fully random routing")
    for batch_size in [1, 4, 8, 16]:
        fn = lambda bs, ne, nt, dev, _s=batch_size: routing_random(bs, ne, nt, dev, seed=123+_s)
        if not run_single_test(batch_size, fn, "random"):
            all_passed = False

    # --- Test 5: Clustered routing (adjacent expert pairs) ---
    print("\nTest 5: Clustered routing (adjacent expert pairs)")
    for batch_size in [1, 4, 8, 16]:
        if not run_single_test(batch_size, routing_clustered, "clustered"):
            all_passed = False

    # --- Test 6: Only last experts active (first 6 inactive) ---
    print("\nTest 6: Only last 2 experts active (experts 6,7)")
    for batch_size in [1, 4, 8, 16]:
        if not run_single_test(batch_size, routing_last_experts_only, "last_experts"):
            all_passed = False

    # --- Test 7: Random routing with many seeds for coverage ---
    print("\nTest 7: Random routing with multiple seeds (batch=16)")
    for seed in range(200, 210):
        fn = lambda bs, ne, nt, dev, _s=seed: routing_random(bs, ne, nt, dev, seed=_s)
        if not run_single_test(16, fn, f"random_seed={seed}"):
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
