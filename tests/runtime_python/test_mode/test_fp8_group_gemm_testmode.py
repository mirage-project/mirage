"""
Test: FP8 MoE W13 Group GEMM via PersistentKernel test_mode.

Tests the FP8 block-scaled group GEMM (moe_w13_fp8_layer) with DeepSeek V3
MoE configuration: 256 experts, top-8, hidden_size=7168, intermediate_size=2048.

Run:
    python tests/runtime_python/test_mode/test_fp8_group_gemm_testmode.py
"""

import torch
import sys
import os
import math

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


# ================================================================
# FP8 quantization utilities
# ================================================================

def quantize_fp8_per_token(x):
    """Quantize [M, K] to FP8 E4M3 with per-token per-128-K block scaling."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    M, K = x.shape
    assert K % 128 == 0
    x_blocked = x.reshape(M, K // 128, 128)
    amax = x_blocked.abs().amax(dim=2)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_scaled = x_blocked / scale.unsqueeze(2)
    x_fp8 = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def quantize_fp8_per_row(x_3d):
    """Quantize [E, N, K] to FP8 E4M3 with per-row per-128-K block scaling."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    E, N, K = x_3d.shape
    assert K % 128 == 0
    x_blocked = x_3d.reshape(E, N, K // 128, 128)
    amax = x_blocked.abs().amax(dim=3)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_scaled = x_blocked / scale.unsqueeze(3)
    x_fp8 = x_scaled.reshape(E, N, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def dequantize_fp8_per_token(x_fp8, scale):
    """Dequantize per-token FP8. x_fp8: [M, K], scale: [M, K//128]."""
    M, K = x_fp8.shape
    x = x_fp8.float().reshape(M, K // 128, 128)
    return (x * scale.unsqueeze(2)).reshape(M, K)


def dequantize_fp8_per_row(x_fp8, scale):
    """Dequantize per-row FP8. x_fp8: [E, N, K], scale: [E, N, K//128]."""
    E, N, K = x_fp8.shape
    x = x_fp8.float().reshape(E, N, K // 128, 128)
    return (x * scale.unsqueeze(3)).reshape(E, N, K)


# ================================================================
# Test
# ================================================================

def test_moe_w13_fp8_testmode():
    device = "cuda"
    torch.manual_seed(42)

    # --- DeepSeek V3 MoE configuration ---
    num_experts = 256
    num_experts_per_tok = 8
    hidden_size = 7168            # K
    intermediate_size = 2048
    N = 2 * intermediate_size     # 4096 (gate + up fused)
    batch_size = 16

    print(f"Config: DeepSeek V3 MoE W13")
    print(f"  E={num_experts}, B={batch_size}, K={hidden_size}, N={N}, topk={num_experts_per_tok}")

    # --- Create data ---
    input_val = torch.randn(batch_size, hidden_size, device=device) * 0.1
    weight_val = torch.randn(num_experts, N, hidden_size, device=device) / math.sqrt(hidden_size)

    # Quantize to FP8
    input_fp8, input_scale = quantize_fp8_per_token(input_val)
    weight_fp8, weight_scale = quantize_fp8_per_row(weight_val)

    # --- Routing: round-robin across experts ---
    routing_indices = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    for i in range(batch_size):
        for slot in range(num_experts_per_tok):
            expert_id = (i * num_experts_per_tok + slot) % num_experts
            routing_indices[expert_id, i] = slot + 1

    activated = []
    for e in range(num_experts):
        if routing_indices[e].any():
            activated.append(e)
    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[num_experts] = len(activated)

    # Output
    output = torch.zeros(batch_size, num_experts_per_tok, N,
                          dtype=torch.bfloat16, device=device)

    # --- PyTorch reference ---
    # The kernel converts float32 scales to UE8M0 (power-of-2 floor) internally.
    # The reference must use the same approximation to match.
    def float32_to_ue8m0_approx(scale):
        bits = scale.view(torch.int32)
        ue8m0 = (bits >> 23) & 0xFF
        return 2.0 ** (ue8m0.float() - 127.0)

    input_deq = dequantize_fp8_per_token(input_fp8, float32_to_ue8m0_approx(input_scale))
    weight_deq = dequantize_fp8_per_row(weight_fp8, float32_to_ue8m0_approx(weight_scale))

    ref = torch.zeros_like(output, dtype=torch.float32)
    for i in range(batch_size):
        for slot in range(num_experts_per_tok):
            expert_id = (i * num_experts_per_tok + slot) % num_experts
            ref[i, slot] = input_deq[i] @ weight_deq[expert_id].T
    ref = ref.bfloat16()

    # --- Build PersistentKernel ---
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    # Attach tensors (FP8 natively supported)
    input_fp8_dt = pk.attach_input(input_fp8, name="input_fp8")
    input_scale_dt = pk.attach_input(input_scale, name="input_scale")
    weight_fp8_dt = pk.attach_input(weight_fp8, name="weight_fp8")
    weight_scale_dt = pk.attach_input(weight_scale, name="weight_scale")
    routing_dt = pk.attach_input(routing_indices, name="routing_indices")
    mask_dt = pk.attach_input(mask, name="mask")
    output_dt = pk.attach_input(output, name="output")

    # Build layer: grid_dim=(8, 16, 1) — DeepSeek V3 target config
    pk.moe_w13_fp8_layer(
        input_fp8=input_fp8_dt,
        input_scale=input_scale_dt,
        weight_fp8=weight_fp8_dt,
        weight_scale=weight_scale_dt,
        moe_routing_indices=routing_dt,
        moe_mask=mask_dt,
        output=output_dt,
        grid_dim=(8, 16, 1),
        block_dim=(256, 1, 1),
    )

    # Compile and run
    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # --- Compare ---
    print(f"\nOutput[0, 0, :8]:    {output[0, 0, :8]}")
    print(f"Reference[0, 0, :8]: {ref[0, 0, :8]}")

    max_abs = (output.float() - ref.float()).abs().max().item()
    denom = ref.float().abs().max().item()
    max_rel = max_abs / max(denom, 1e-6)

    print(f"\nMax absolute diff: {max_abs:.6f}")
    print(f"Max relative err:  {max_rel:.6f}")

    if max_rel < 0.1:
        print("\nPASSED: FP8 MoE W13 group GEMM (DSV3 config) produces correct output")
    else:
        print(f"\nFAILED: max relative error {max_rel:.4f} exceeds 10% tolerance")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_w13_fp8_testmode()
