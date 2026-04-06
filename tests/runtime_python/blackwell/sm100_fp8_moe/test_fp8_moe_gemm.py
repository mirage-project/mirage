"""
Unit tests for FP8 block-scaled MoE group GEMM on SM100 (Blackwell B200).

Tests batch sizes 1-16 with DeepSeek-V3-style shapes (but scaled down for
fast unit testing). Compares against a PyTorch reference that dequantizes
FP8 → BF16 and performs standard matmul.

Run:
    cd tests/runtime_python/blackwell/sm100_fp8_moe
    python setup.py build_ext --inplace
    python test_fp8_moe_gemm.py
"""

import torch
import numpy as np
import sys
import os

# The test kernel uses small dimensions defined in the wrapper:
#   BATCH_SIZE=128 (padded; actual tokens vary 1-16), OUTPUT_SIZE=256,
#   REDUCTION_SIZE=256, NUM_EXPERTS=8, NUM_TOPK=2, MMA_N=128
# These are compile-time constants aligned to DeepGEMM's BLOCK_N=128.
# We test varying active batch sizes (1-16 real tokens padded to 128).

try:
    import runtime_kernel_fp8_moe as rk
except ImportError:
    print("ERROR: runtime_kernel_fp8_moe not found.")
    print("Please run: python setup.py build_ext --inplace")
    sys.exit(1)

BATCH_SIZE  = 128   # padded to MMA_N=128 (actual tokens 1-16)
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
    x_blocks = x.reshape(*shape[:-1], num_blocks, block_k)  # [..., nb, bk]

    # Per-block max absolute value
    amax = x_blocks.abs().amax(dim=-1)  # [..., nb]
    # FP8 E4M3 max = 448
    scale = amax / 448.0
    scale = scale.clamp(min=1e-12)

    # Quantize
    x_scaled = x_blocks / scale.unsqueeze(-1)  # [..., nb, bk]
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


def make_routing(batch_size, num_experts, num_topk, device):
    """
    Create routing indices and mask for testing.
    Returns:
        routing_indices: [num_experts, batch_size]  int32
            routing_indices[e, i] = top_k slot (1-indexed) if token i is
            routed to expert e, else 0.
        mask: [num_experts+1]  int32
            mask[0..num_experts-1] = expert index (for activated experts)
            mask[num_experts] = num_activated_experts
    """
    # For simplicity: assign tokens round-robin to experts
    # Token i → experts [i % num_experts, (i + num_experts//2) % num_experts]
    routing = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    token_to_experts = {}
    for i in range(batch_size):
        e0 = i % num_experts
        e1 = (i + num_experts // 2) % num_experts
        token_to_experts[i] = [e0, e1]

    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            routing[e, i] = slot + 1  # 1-indexed top-k slot

    # Build mask: which experts are activated (have ≥1 token)
    activated_experts = []
    for e in range(num_experts):
        if routing[e].any():
            activated_experts.append(e)

    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated_experts):
        mask[idx] = e
    mask[num_experts] = len(activated_experts)

    return routing, mask, token_to_experts


def reference_moe_w13(input_bf16, weight_bf16, routing_indices, mask,
                       token_to_experts, batch_size, output_size, num_experts,
                       num_topk, device):
    """
    PyTorch reference for MoE W13 GEMM.
    input_bf16:  [batch, K]
    weight_bf16: [num_experts, N, K]
    routing_indices: [num_experts, batch]
    Returns: output [batch, top_k, N] bf16
    """
    output = torch.zeros(batch_size, num_topk, output_size,
                         dtype=torch.bfloat16, device=device)
    for i in range(batch_size):
        for slot, e in enumerate(token_to_experts[i]):
            # output[i, slot, :] = input[i, :] @ weight[e, :, :].T
            result = (input_bf16[i:i+1].float() @
                      weight_bf16[e].float().T).squeeze(0)
            output[i, slot] = result.bfloat16()
    return output


def test_fp8_moe_w13(batch_size: int, verbose: bool = False):
    device = torch.device("cuda")
    torch.manual_seed(42 + batch_size)

    # Use a subset of BATCH_SIZE tokens (pad rest with zero input)
    full_batch = BATCH_SIZE

    # --- Generate random FP8 input ---
    input_bf16 = torch.randn(full_batch, K, device=device, dtype=torch.float32)
    input_bf16[:batch_size] = torch.randn(batch_size, K, device=device)
    input_fp8, input_scale = quantize_to_fp8(input_bf16)  # [batch, K], [batch, K/128]

    # --- Generate random FP8 weight for each expert ---
    weight_bf16 = torch.randn(NUM_EXPERTS, OUTPUT_SIZE, K, device=device, dtype=torch.float32)
    # Per-expert weight quantization: shape [num_experts, N, K/128]
    weight_fp8_list = []
    weight_scale_list = []
    for e in range(NUM_EXPERTS):
        w_fp8, w_scale = quantize_to_fp8(weight_bf16[e])  # [N, K], [N, K/128]
        weight_fp8_list.append(w_fp8)
        weight_scale_list.append(w_scale)
    weight_fp8   = torch.stack(weight_fp8_list, dim=0)    # [num_experts, N, K]
    weight_scale = torch.stack(weight_scale_list, dim=0)  # [num_experts, N, K/128]

    # --- Build routing ---
    routing_indices, mask, token_to_experts = make_routing(
        full_batch, NUM_EXPERTS, NUM_TOPK, device)

    # Zero out routing for tokens beyond batch_size
    routing_indices[:, batch_size:] = 0
    # Rebuild mask
    for e in range(NUM_EXPERTS):
        has_token = routing_indices[e, :batch_size].any().item()
        if not has_token:
            # Mark expert as inactive (don't change mask, expert might still appear)
            pass

    # --- Run FP8 kernel ---
    output_fp8 = torch.zeros(full_batch, NUM_TOPK, OUTPUT_SIZE,
                             dtype=torch.bfloat16, device=device)
    rk.fp8_moe_w13_gemm(
        input_fp8,
        input_scale,
        weight_fp8.contiguous(),
        weight_scale.contiguous(),
        routing_indices,
        mask,
        output_fp8,
        0)  # expert_offset=0

    # --- Reference computation using UE8M0-approximated scales ---
    # The kernel converts float32 scales to UE8M0 (floor exponent), so the
    # reference must do the same to match numerically.
    def float32_to_ue8m0_approx(sf: torch.Tensor) -> torch.Tensor:
        """Convert float32 scale to UE8M0 approximation (power-of-2 floor)."""
        bits = sf.view(torch.int32)
        ue8m0 = (bits >> 23) & 0xFF  # IEEE 754 exponent = floor log2 + 127
        # Reconstruct: 2^(ue8m0 - 127)
        return torch.pow(2.0, ue8m0.float() - 127.0)

    weight_deq = torch.stack([
        dequantize_fp8(weight_fp8[e],
                       float32_to_ue8m0_approx(weight_scale[e])).bfloat16()
        for e in range(NUM_EXPERTS)
    ], dim=0)  # [num_experts, N, K]

    input_deq = dequantize_fp8(input_fp8,
                               float32_to_ue8m0_approx(input_scale)).bfloat16()

    output_ref = reference_moe_w13(
        input_deq, weight_deq, routing_indices, mask,
        token_to_experts, full_batch, OUTPUT_SIZE, NUM_EXPERTS, NUM_TOPK, device)

    # --- Compare for active tokens ---
    max_abs_err = 0.0
    max_rel_err = 0.0
    for i in range(batch_size):
        for slot in range(NUM_TOPK):
            out_k = output_fp8[i, slot]
            ref_k = output_ref[i, slot]
            if ref_k.abs().max() < 1e-4:
                continue  # skip near-zero (unrouted)
            abs_err = (out_k.float() - ref_k.float()).abs().max().item()
            rel_err = (abs_err / (ref_k.float().abs().max().item() + 1e-8))
            max_abs_err = max(max_abs_err, abs_err)
            max_rel_err = max(max_rel_err, rel_err)

    # BF16 + FP8 rounding: allow up to ~10% relative error
    passed = max_abs_err < 0.5 and max_rel_err < 0.15
    status = "PASS" if passed else "FAIL"
    print(f"  batch={batch_size:2d}: max_abs_err={max_abs_err:.4f}, "
          f"max_rel_err={max_rel_err:.4f}  [{status}]")
    if verbose and not passed:
        print(f"    output_fp8[0,0,:8] = {output_fp8[0,0,:8]}")
        print(f"    output_ref[0,0,:8] = {output_ref[0,0,:8]}")
    return passed


def main():
    print(f"Testing FP8 MoE W13 group GEMM on SM100 (Blackwell)")
    print(f"  N={OUTPUT_SIZE}, K={K}, num_experts={NUM_EXPERTS}, top_k={NUM_TOPK}")
    print()

    all_passed = True
    for batch_size in range(1, 17):
        passed = test_fp8_moe_w13(batch_size)
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
