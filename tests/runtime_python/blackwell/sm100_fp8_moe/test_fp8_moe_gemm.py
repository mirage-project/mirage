"""
Correctness test: FP8 block-scaled MoE Group GEMM — DeepSeek V3 configuration.

Tests the FP8 MoE W13 kernel with real DeepSeek V3 dimensions:
  - 256 experts, top-8 routing
  - hidden_size=7168 (K), intermediate_size*2=4096 (N)
  - Batch sizes M = 1, 2, 4, 8, 16

Also tests 2D work distribution (expert_stride x n_splits) to verify
that splitting the N dimension across multiple CTAs produces correct results.

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_moe_dsv3
    python setup.py build_ext --inplace
    CUDA_VISIBLE_DEVICES=4 python test_fp8_moe_gemm.py
"""

import torch
import sys
import os
import random
import time

try:
    import runtime_kernel_fp8_moe as rk
except ImportError:
    print("ERROR: runtime_kernel_fp8_moe not found.")
    print("Run: python setup.py build_ext --inplace")
    sys.exit(1)

# ================================================================
# DeepSeek V3 W13 MoE parameters
# ================================================================
BATCH_SIZE    = 16     # matches MMA_N=16 (production config, no padding waste)
OUTPUT_SIZE   = 4096   # 2 * intermediate_size
K             = 7168   # hidden_size
NUM_EXPERTS   = 256
NUM_TOPK      = 8
K_SCALE       = K // 128  # 56 scale blocks along K


# ================================================================
# FP8 quantization utilities
# ================================================================
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


def dequantize_fp8(x_fp8: torch.Tensor, scale: torch.Tensor, block_k: int = 128):
    """Dequantize FP8 with per-block float32 scales."""
    shape = x_fp8.shape
    K_dim = shape[-1]
    num_blocks = K_dim // block_k
    x_blocks = x_fp8.reshape(*shape[:-1], num_blocks, block_k).float()
    return (x_blocks * scale.unsqueeze(-1)).reshape(*shape)


def float32_to_ue8m0_approx(scale: torch.Tensor):
    """Convert float32 scale to UE8M0-approximated value (power-of-2 floor).

    This matches the kernel's conversion: ue8m0 = (__float_as_uint(sf) >> 23) & 0xFF
    """
    bits = scale.view(torch.int32)
    ue8m0 = (bits >> 23) & 0xFF
    approx = (2.0 ** (ue8m0.float() - 127.0))
    return approx


# ================================================================
# Routing generation
# ================================================================
def make_routing(active_tokens, num_experts, num_topk, device, seed=42):
    """Random routing: each of active_tokens tokens picks num_topk distinct experts.

    Args:
        active_tokens: number of real tokens (1-16)
        num_experts: total experts (e.g., 256)
        num_topk: experts per token (e.g., 8)
        device: CUDA device
        seed: random seed

    Returns:
        routing: [num_experts, BATCH_SIZE] int32, padded to compiled BATCH_SIZE
        mask: [num_experts+1] int32, activated expert list
        token_to_experts: dict mapping token_idx -> list of expert indices
    """
    rng = random.Random(seed)
    # Allocate with padded BATCH_SIZE (compiled kernel constant), fill only active_tokens
    routing = torch.zeros(num_experts, BATCH_SIZE, dtype=torch.int32, device=device)
    token_to_experts = {}

    for i in range(active_tokens):
        experts = rng.sample(range(num_experts), num_topk)
        token_to_experts[i] = experts
        for slot, e in enumerate(experts):
            routing[e, i] = slot + 1

    activated = []
    for e in range(num_experts):
        if routing[e, :active_tokens].any():
            activated.append(e)

    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[num_experts] = len(activated)

    return routing, mask, token_to_experts


# ================================================================
# Reference implementation
# ================================================================
def reference_moe_w13(input_fp8, input_scale, weight_fp8, weight_scale,
                       batch_size, token_to_experts, use_ue8m0=True):
    """Pure PyTorch reference using FP8 dequantization."""
    if use_ue8m0:
        i_scale = float32_to_ue8m0_approx(input_scale)
        w_scale = float32_to_ue8m0_approx(weight_scale)
    else:
        i_scale = input_scale
        w_scale = weight_scale

    input_deq = dequantize_fp8(input_fp8, i_scale).bfloat16()
    output = torch.zeros(BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                         dtype=torch.bfloat16, device=input_fp8.device)

    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            w_deq = dequantize_fp8(weight_fp8[e], w_scale[e]).bfloat16()
            output[i, slot] = (input_deq[i:i+1] @ w_deq.T).squeeze(0)

    return output


# ================================================================
# Test runner
# ================================================================
def run_test(batch_size, seed=42, label="", use_2d=False, expert_stride=1, n_splits=1):
    """Run one test case and verify correctness."""
    device = torch.device("cuda")

    # Generate data
    torch.manual_seed(seed)
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

    routing, mask, token_to_experts = make_routing(
        batch_size, NUM_EXPERTS, NUM_TOPK, device, seed=seed)

    output = torch.zeros(BATCH_SIZE, NUM_TOPK, OUTPUT_SIZE,
                         dtype=torch.bfloat16, device=device)

    # Run kernel
    t0 = time.time()
    if use_2d:
        rk.fp8_moe_gemm_2d(
            input_fp8, input_scale, weight_fp8, weight_scale,
            routing, mask, output, expert_stride, n_splits)
    else:
        rk.fp8_moe_gemm_test(
            input_fp8, input_scale, weight_fp8, weight_scale,
            routing, mask, output, 0)
    kernel_ms = (time.time() - t0) * 1000.0

    # Reference with UE8M0-approximated scales (should match kernel closely)
    ref = reference_moe_w13(input_fp8, input_scale, weight_fp8, weight_scale,
                             batch_size, token_to_experts, use_ue8m0=True)

    # Compare only routed tokens
    max_abs = 0.0
    max_rel = 0.0
    num_compared = 0
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            out_row = output[i, slot].float()
            ref_row = ref[i, slot].float()
            diff = (out_row - ref_row).abs()
            abs_err = diff.max().item()
            denom = ref_row.abs().max().item()
            rel_err = abs_err / max(denom, 1e-6)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            num_compared += 1

    # Check unrouted slots are zero
    unrouted_ok = True
    for i in range(batch_size):
        routed_slots = set(range(len(token_to_experts[i])))
        for s in range(NUM_TOPK):
            if s not in routed_slots:
                if output[i, s].abs().max().item() > 0:
                    unrouted_ok = False

    # With K=7168, BF16 accumulation can produce occasional abs errors up to ~1.0
    # due to rounding in the FP32->BF16 conversion. Use a relaxed abs threshold.
    passed = (max_abs < 2.0 and max_rel < 0.05 and num_compared > 0 and unrouted_ok)
    status = "PASS" if passed else "FAIL"

    mode = f"2D({expert_stride}x{n_splits})" if use_2d else "1CTA"
    print(f"  batch={batch_size:>2}  {mode:<12}  abs={max_abs:.4f}  rel={max_rel:.4f}  "
          f"cmp={num_compared:>4}  unrouted={'ok' if unrouted_ok else 'FAIL'}  "
          f"{kernel_ms:.0f}ms  [{status}]  {label}")

    if not passed:
        print(f"    *** FAILED: max_abs={max_abs}, max_rel={max_rel}, "
              f"compared={num_compared}, unrouted={unrouted_ok}")

    return passed


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: N={OUTPUT_SIZE}, K={K}, experts={NUM_EXPERTS}, topk={NUM_TOPK}")
    print(f"BATCH_SIZE (padded)={BATCH_SIZE}")
    print()

    all_passed = True

    # ================================================================
    # Test 1: Single-CTA correctness — batch sizes 1,2,4,8,16
    # ================================================================
    print("=" * 90)
    print("Test 1: Single-CTA correctness (all experts sequential)")
    print("=" * 90)
    for batch_size in [1, 2, 4, 8, 16]:
        ok = run_test(batch_size, seed=42 + batch_size, label="single-CTA")
        all_passed = all_passed and ok

    # ================================================================
    # Test 2: 2D grid — expert_stride=8, n_splits=1 (expert-only split)
    # ================================================================
    print()
    print("=" * 90)
    print("Test 2: 2D grid (8x1) — expert distribution only")
    print("=" * 90)
    for batch_size in [1, 4, 8, 16]:
        ok = run_test(batch_size, seed=100 + batch_size,
                      label="expert-only", use_2d=True,
                      expert_stride=8, n_splits=1)
        all_passed = all_passed and ok

    # ================================================================
    # Test 3: 2D grid — expert_stride=8, n_splits=16 (target config)
    # This is the DeepSeek V3 target: grid_dim=(8, 16, 1)
    # OUTPUT_SIZE/16 = 4096/16 = 256 per CTA
    # ================================================================
    print()
    print("=" * 90)
    print("Test 3: 2D grid (8x16) — DeepSeek V3 target config")
    print("=" * 90)
    for batch_size in [1, 2, 4, 8, 16]:
        ok = run_test(batch_size, seed=200 + batch_size,
                      label="8x16", use_2d=True,
                      expert_stride=8, n_splits=16)
        all_passed = all_passed and ok

    # ================================================================
    # Test 4: 2D grid — various n_splits
    # ================================================================
    print()
    print("=" * 90)
    print("Test 4: 2D grid — varying n_splits (batch=8)")
    print("=" * 90)
    for n_splits in [1, 2, 4, 8, 16, 32]:
        ok = run_test(8, seed=300 + n_splits,
                      label=f"nsplit={n_splits}", use_2d=True,
                      expert_stride=8, n_splits=n_splits)
        all_passed = all_passed and ok

    # ================================================================
    # Test 5: Multiple random seeds (batch=16, 2D)
    # ================================================================
    print()
    print("=" * 90)
    print("Test 5: Random seeds (batch=16, 8x16 grid)")
    print("=" * 90)
    for seed in range(400, 410):
        ok = run_test(16, seed=seed,
                      label=f"seed={seed}", use_2d=True,
                      expert_stride=8, n_splits=16)
        all_passed = all_passed and ok

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 90)
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED!")
    print("=" * 90)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
