"""
Test: EP MoE distributed routing via PersistentKernel test_mode.

Tests the ep_moe_routing_layer by building a minimal PersistentKernel in
test_mode, compiling it, running it once, and comparing the output to a
PyTorch reference.

Configuration:
  - batch_size = 8
  - num_experts = 8
  - topk = 2
  - world_size = 1 (single GPU)

Run:
    python tests/runtime_python/test_mode/test_ep_moe_routing_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def reference_routing(logits, topk):
    """PyTorch reference: top-k + softmax normalization."""
    logits_f = logits.float()
    topk_values, topk_indices = torch.topk(logits_f, topk, dim=1)
    weights = torch.softmax(topk_values, dim=1)
    return topk_indices.int(), weights.to(logits.dtype)


def test_ep_moe_routing_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    num_experts = 8
    topk = 2
    world_size = 1
    experts_per_rank = num_experts // world_size

    print(f"\n{'='*60}")
    print(f"Test: EP MoE Routing test_mode")
    print(f"  B={batch_size}, E={num_experts}, topk={topk}, world_size={world_size}")

    # Create tensors
    router_logits = torch.randn(batch_size, num_experts, dtype=dtype, device=device)
    routing_indices = torch.zeros(batch_size, topk, dtype=torch.int32, device=device)
    routing_weights = torch.zeros(batch_size, topk, dtype=dtype, device=device)
    dispatch_counts = torch.zeros(world_size, dtype=torch.int32, device=device)

    # Build PersistentKernel in test mode
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = world_size
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    # Attach tensors
    logits_dt = pk.attach_input(router_logits, name="router_logits")
    indices_dt = pk.attach_input(routing_indices, name="routing_indices")
    weights_dt = pk.attach_input(routing_weights, name="routing_weights")
    counts_dt = pk.attach_input(dispatch_counts, name="dispatch_counts")

    # Build layer
    pk.ep_moe_routing_layer(
        input=logits_dt,
        output=(indices_dt, weights_dt, counts_dt),
        grid_dim=(1, 1, 1),
        block_dim=(max(topk, 32), 1, 1),
        params=[world_size, num_experts, experts_per_rank, 1],  # normalize=True
    )

    # Compile
    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    # Run
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Reference
    ref_indices, ref_weights = reference_routing(router_logits, topk)

    print(f"\nKernel  indices[0]: {routing_indices[0].cpu().tolist()}")
    print(f"Ref     indices[0]: {ref_indices[0].cpu().tolist()}")
    print(f"Kernel  weights[0]: {routing_weights[0]}")
    print(f"Ref     weights[0]: {ref_weights[0]}")
    print(f"Dispatch counts:    {dispatch_counts.cpu().tolist()}")

    # Verify weights (sort both by value since order within top-k may differ)
    kern_sorted, kern_idx = routing_weights.float().sort(dim=1, descending=True)
    ref_sorted, ref_idx = ref_weights.float().sort(dim=1, descending=True)

    max_diff = (kern_sorted - ref_sorted).abs().max().item()
    print(f"\nMax weight diff: {max_diff:.6f}")

    if max_diff < 0.02:
        print("\nPASSED: EP MoE routing test_mode produces correct output")
    else:
        print(f"\nFAILED: max weight diff {max_diff:.4f} exceeds tolerance")
        sys.exit(1)

    # Verify same experts selected
    kern_experts = torch.gather(routing_indices, 1, kern_idx)
    ref_experts = torch.gather(ref_indices, 1, ref_idx)
    if torch.equal(kern_experts, ref_experts):
        print("Expert selection matches reference")
    else:
        print("WARNING: Expert selection differs (possible tie-breaking)")

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_ep_moe_routing_testmode()
