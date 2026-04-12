"""
Test: EP MoE combine kernel (single-GPU, WORLD_SIZE=1).

Tests weighted sum of expert outputs with optional residual addition.
Verifies against a PyTorch reference implementation.

Run:
    cd tests/runtime_python/blackwell/ep_moe/
    python setup.py build_ext --inplace
    python test_moe_combine.py
"""

import torch
import sys

try:
    import runtime_kernel_ep_moe as kernel
except ImportError as e:
    print(f"Error importing kernel: {e}")
    print("Run: python setup.py build_ext --inplace")
    sys.exit(1)


def reference_combine(expert_outputs, routing_weights, residual, add_residual):
    """PyTorch reference: weighted sum + optional residual."""
    # expert_outputs: [B, topk, H]
    # routing_weights: [B, topk]
    # output = sum_k(weight_k * expert_k) [+ residual]
    weights = routing_weights.float().unsqueeze(-1)  # [B, topk, 1]
    experts = expert_outputs.float()  # [B, topk, H]
    output = (weights * experts).sum(dim=1)  # [B, H]
    if add_residual:
        output = output + residual.float()
    return output.to(expert_outputs.dtype)


def test_combine(hidden_dim=64, add_residual=True):
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    topk = 2
    num_experts = 8
    experts_per_rank = num_experts

    label = "with" if add_residual else "without"
    print(f"\n{'='*60}")
    print(f"Test: EP MoE Combine {label} residual "
          f"(B={batch_size}, H={hidden_dim}, topk={topk})")

    expert_outputs = torch.randn(batch_size, topk, hidden_dim,
                                 dtype=dtype, device=device) * 0.1
    routing_weights = torch.softmax(
        torch.randn(batch_size, topk, device=device), dim=1).to(dtype)
    residual = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device) * 0.1
    output = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)

    # Run kernel
    kernel.moe_combine(expert_outputs, routing_weights, residual, output,
                       num_experts, experts_per_rank, 0, add_residual)

    # Reference
    ref = reference_combine(expert_outputs, routing_weights, residual,
                            add_residual)

    max_abs = (output.float() - ref.float()).abs().max().item()
    denom = ref.float().abs().max().item()
    max_rel = max_abs / max(denom, 1e-6)

    print(f"  Output[0, :8]:    {output[0, :8]}")
    print(f"  Reference[0, :8]: {ref[0, :8]}")
    print(f"  Max abs diff: {max_abs:.6f}")
    print(f"  Max rel err:  {max_rel:.6f}")

    assert max_rel < 0.05, f"FAILED: max relative error {max_rel:.4f} exceeds 5%"
    print(f"  PASSED!")

    # Benchmark
    for _ in range(16):
        kernel.moe_combine(expert_outputs, routing_weights, residual, output,
                           num_experts, experts_per_rank, 0, add_residual)

    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    reps = 1000
    starter.record()
    for _ in range(reps):
        kernel.moe_combine(expert_outputs, routing_weights, residual, output,
                           num_experts, experts_per_rank, 0, add_residual)
    ender.record()
    torch.cuda.synchronize()
    print(f"  Avg time: {starter.elapsed_time(ender)/reps:.6f} ms")


def main():
    test_combine(hidden_dim=64, add_residual=True)
    test_combine(hidden_dim=64, add_residual=False)
    test_combine(hidden_dim=256, add_residual=True)
    print(f"\n{'='*60}")
    print("All combine tests PASSED!")


if __name__ == "__main__":
    main()
