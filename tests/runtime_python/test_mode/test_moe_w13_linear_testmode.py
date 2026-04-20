"""
Test: BF16 MoE W13 linear via PersistentKernel test_mode.

Tests the MoE gate+up fused linear projection (moe_w13_linear_layer) using
the Qwen3-30B-A3B configuration from demo/qwen3/demo_30B_A3B.py:
  - hidden_size = 4096
  - moe_intermediate_size = 2560  (fused output = 5120)
  - num_experts = 128
  - num_experts_per_tok = 8
  - grid_dim = (10, 12, 1)
  - block_dim = (256, 1, 1)

For each token, the kernel selects its top-k experts and computes:
  output[token, slot, :] = input[token, :] @ weight[expert, :, :].T

Run:
    python tests/runtime_python/test_mode/test_moe_w13_linear_testmode.py
"""

import torch
import sys
import os
import math

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def test_moe_w13_linear():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Qwen3-30B-A3B configuration
    batch_size = 1
    hidden_size = 4096
    intermediate_size = 2560  # model.config.moe_intermediate_size
    fused_outdim = 2 * intermediate_size  # 5120 (gate + up fused)
    num_experts = 128
    num_experts_per_tok = 8

    print(f"\n{'='*60}")
    print(f"Test: BF16 MoE W13 linear (Qwen3-30B-A3B config)")
    print(f"  B={batch_size}, K={hidden_size}, N={fused_outdim}, "
          f"E={num_experts}, topk={num_experts_per_tok}")

    # --- Create tensors ---
    input_act = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) * 0.1
    weight = torch.randn(num_experts, fused_outdim, hidden_size,
                         dtype=dtype, device=device) / math.sqrt(hidden_size)
    output = torch.zeros(batch_size, num_experts_per_tok, fused_outdim,
                         dtype=dtype, device=device)

    # --- Build routing data (round-robin assignment) ---
    routing_indices = torch.zeros(num_experts, batch_size,
                                  dtype=torch.int32, device=device)
    for i in range(batch_size):
        for slot in range(num_experts_per_tok):
            expert_id = (i * num_experts_per_tok + slot) % num_experts
            routing_indices[expert_id, i] = slot + 1  # 1-indexed

    activated = []
    for e in range(num_experts):
        if routing_indices[e].any():
            activated.append(e)
    moe_mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        moe_mask[idx] = e
    moe_mask[num_experts] = len(activated)

    print(f"  Activated experts: {len(activated)}")

    # --- PyTorch reference ---
    ref = torch.zeros(batch_size, num_experts_per_tok, fused_outdim,
                      dtype=torch.float32, device=device)
    for i in range(batch_size):
        for slot in range(num_experts_per_tok):
            expert_id = (i * num_experts_per_tok + slot) % num_experts
            ref[i, slot] = input_act[i].float() @ weight[expert_id].float().T
    ref = ref.to(dtype)

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

    input_dt = pk.attach_input(input_act, name="input")
    weight_dt = pk.attach_input(weight, name="weight")
    routing_dt = pk.attach_input(routing_indices, name="moe_routing_indices")
    mask_dt = pk.attach_input(moe_mask, name="moe_mask")
    output_dt = pk.attach_input(output, name="output")

    # grid_dim and block_dim match demo/qwen3/demo_30B_A3B.py line 620-628
    pk.moe_w13_linear_layer(
        input=input_dt,
        weight=weight_dt,
        moe_routing_indices=routing_dt,
        moe_mask=mask_dt,
        output=output_dt,
        grid_dim=(10, 12, 1),
        block_dim=(256, 1, 1),
    )

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

    if max_rel < 0.05:
        print("\nPASSED: BF16 MoE W13 linear produces correct output")
    else:
        print(f"\nFAILED: max relative error {max_rel:.4f} exceeds 5% tolerance")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_w13_linear()
