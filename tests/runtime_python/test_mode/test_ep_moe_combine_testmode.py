"""
Test: EP MoE combine via PersistentKernel test_mode.

Tests the ep_moe_combine_layer by building a minimal PersistentKernel in
test_mode, compiling it, running it once, and comparing the output to a
PyTorch reference.

Configuration:
  - batch_size = 8
  - hidden_dim = 64
  - topk = 2
  - world_size = 1 (single GPU)
  - add_residual = True

Run:
    python tests/runtime_python/test_mode/test_ep_moe_combine_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def reference_combine(expert_outputs, routing_weights, residual, add_residual):
    """PyTorch reference: weighted sum + optional residual."""
    weights = routing_weights.float().unsqueeze(-1)  # [B, topk, 1]
    experts = expert_outputs.float()  # [B, topk, H]
    output = (weights * experts).sum(dim=1)  # [B, H]
    if add_residual:
        output = output + residual.float()
    return output.to(expert_outputs.dtype)


def test_ep_moe_combine_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    hidden_dim = 64
    topk = 2
    num_experts = 8
    world_size = 1
    experts_per_rank = num_experts // world_size
    add_residual = True

    print(f"\n{'='*60}")
    print(f"Test: EP MoE Combine test_mode")
    print(f"  B={batch_size}, H={hidden_dim}, topk={topk}, "
          f"residual={'yes' if add_residual else 'no'}")

    # Create tensors
    expert_outputs = torch.randn(batch_size, topk, hidden_dim,
                                 dtype=dtype, device=device) * 0.1
    routing_weights = torch.softmax(
        torch.randn(batch_size, topk, device=device), dim=1).to(dtype)
    residual = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device) * 0.1
    output = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)

    # Reference
    ref = reference_combine(expert_outputs, routing_weights, residual, add_residual)

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
    expert_dt = pk.attach_input(expert_outputs, name="expert_outputs")
    weights_dt = pk.attach_input(routing_weights, name="routing_weights")
    residual_dt = pk.attach_input(residual, name="residual")
    output_dt = pk.attach_input(output, name="output")

    # Build layer
    pk.ep_moe_combine_layer(
        expert_outputs=expert_dt,
        routing_weights=weights_dt,
        residual=residual_dt,
        output=output_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        params=[world_size, num_experts, experts_per_rank,
                1 if add_residual else 0],
    )

    # Compile
    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    # Run
    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Compare
    print(f"\nOutput[0, :8]:    {output[0, :8]}")
    print(f"Reference[0, :8]: {ref[0, :8]}")

    max_abs = (output.float() - ref.float()).abs().max().item()
    denom = ref.float().abs().max().item()
    max_rel = max_abs / max(denom, 1e-6)

    print(f"\nMax absolute diff: {max_abs:.6f}")
    print(f"Max relative err:  {max_rel:.6f}")

    if max_rel < 0.05:
        print("\nPASSED: EP MoE combine test_mode produces correct output")
    else:
        print(f"\nFAILED: max relative error {max_rel:.4f} exceeds 5% tolerance")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_ep_moe_combine_testmode()
