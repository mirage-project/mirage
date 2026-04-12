"""
Test: EP MoE distributed routing kernel (single-GPU, WORLD_SIZE=1).

Tests top-k expert selection with softmax normalization and dispatch count
computation.  Verifies against a PyTorch reference implementation.

Run:
    cd tests/runtime_python/blackwell/ep_moe/
    python setup.py build_ext --inplace
    python test_moe_routing.py
"""

import torch
import sys

try:
    import runtime_kernel_ep_moe as kernel
except ImportError as e:
    print(f"Error importing kernel: {e}")
    print("Run: python setup.py build_ext --inplace")
    sys.exit(1)


def reference_routing(logits, topk, num_experts, experts_per_rank):
    """PyTorch reference: top-k selection + softmax normalization."""
    batch_size = logits.size(0)
    logits_f = logits.float()

    # Top-k selection
    topk_values, topk_indices = torch.topk(logits_f, topk, dim=1)

    # Softmax over top-k values
    weights = torch.softmax(topk_values, dim=1)

    # Dispatch counts: count tokens per destination rank
    world_size = 1
    dispatch_counts = torch.zeros(world_size, dtype=torch.int32, device=logits.device)
    for b in range(batch_size):
        for k in range(topk):
            expert_id = topk_indices[b, k].item()
            dest_rank = expert_id // experts_per_rank
            dispatch_counts[dest_rank] += 1

    return topk_indices.int(), weights.to(logits.dtype), dispatch_counts


def test_routing():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    num_experts = 8
    topk = 2
    experts_per_rank = num_experts  # single GPU: all experts on rank 0

    print(f"\n{'='*60}")
    print(f"Test: EP MoE Routing (B={batch_size}, E={num_experts}, topk={topk})")

    logits = torch.randn(batch_size, num_experts, dtype=dtype, device=device)

    # Kernel outputs
    routing_indices = torch.zeros(batch_size, topk, dtype=torch.int32, device=device)
    routing_weights = torch.zeros(batch_size, topk, dtype=dtype, device=device)
    dispatch_counts = torch.zeros(1, dtype=torch.int32, device=device)

    # Run kernel
    kernel.moe_routing(logits, routing_indices, routing_weights,
                       dispatch_counts, num_experts, experts_per_rank, 0)

    # Reference
    ref_indices, ref_weights, ref_counts = reference_routing(
        logits, topk, num_experts, experts_per_rank)

    # Verify dispatch counts
    torch.testing.assert_close(dispatch_counts, ref_counts, rtol=0, atol=0)
    print(f"  Dispatch counts: PASSED (expected {ref_counts.item()}, "
          f"got {dispatch_counts.item()})")

    # Verify routing weights (softmax values)
    # Indices may be in different order within top-k, so sort both
    kern_sorted_w, kern_sort_idx = routing_weights.float().sort(dim=1, descending=True)
    ref_sorted_w, ref_sort_idx = ref_weights.float().sort(dim=1, descending=True)
    torch.testing.assert_close(kern_sorted_w, ref_sorted_w, rtol=1e-2, atol=1e-2)
    print(f"  Routing weights: PASSED")

    # Verify that the same experts are selected
    kern_experts = torch.gather(routing_indices, 1, kern_sort_idx)
    ref_experts = torch.gather(ref_indices, 1, ref_sort_idx)
    torch.testing.assert_close(kern_experts, ref_experts, rtol=0, atol=0)
    print(f"  Expert selection: PASSED")

    # Benchmark
    for _ in range(16):
        kernel.moe_routing(logits, routing_indices, routing_weights,
                           dispatch_counts, num_experts, experts_per_rank, 0)

    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    reps = 1000
    starter.record()
    for _ in range(reps):
        kernel.moe_routing(logits, routing_indices, routing_weights,
                           dispatch_counts, num_experts, experts_per_rank, 0)
    ender.record()
    torch.cuda.synchronize()
    print(f"  Avg time: {starter.elapsed_time(ender)/reps:.6f} ms")

    print(f"\nAll routing tests PASSED!")


if __name__ == "__main__":
    test_routing()
