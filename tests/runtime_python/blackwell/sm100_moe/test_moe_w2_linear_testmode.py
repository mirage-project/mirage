"""
Test: BF16 MoE W2 linear via PersistentKernel test_mode.

Tests the MoE down-projection (moe_w2_linear_layer) end-to-end through the
full MPK compilation pipeline. Round-robin expert assignment, then compare
against the shared PyTorch reference.

Run:
    python tests/runtime_python/blackwell/sm100_moe/test_moe_w2_linear_testmode.py
"""

import torch
import sys
import os
import math

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import moe_w2_linear_ref


def test_moe_w2_linear_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Qwen3-30B-A3B-style configuration (matches demo/qwen3/demo_30B_A3B.py)
    batch_size = 1
    hidden_size = 4096           # output_size of w2 (H)
    intermediate_size = 2560     # input reduction (I)
    num_experts = 128
    num_experts_per_tok = 8

    print(f"\n{'='*60}")
    print(f"Test: BF16 MoE W2 linear (Qwen3-30B-A3B config)")
    print(f"  B={batch_size}, I={intermediate_size}, H={hidden_size}, "
          f"E={num_experts}, topk={num_experts_per_tok}")
    print(f"{'='*60}")

    # --- Create tensors ---
    input_act = torch.randn(
        batch_size, num_experts_per_tok, intermediate_size,
        dtype=dtype, device=device
    ) * 0.1
    weight = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=dtype, device=device,
    ) / math.sqrt(intermediate_size)
    output = torch.zeros(
        batch_size, num_experts_per_tok, hidden_size,
        dtype=dtype, device=device,
    )

    # --- Build routing data (round-robin assignment) ---
    topk_expert_indices = torch.zeros(
        batch_size, num_experts_per_tok, dtype=torch.int64, device=device
    )
    for i in range(batch_size):
        for slot in range(num_experts_per_tok):
            topk_expert_indices[i, slot] = (i * num_experts_per_tok + slot) % num_experts

    routing_indices = torch.zeros(
        num_experts, batch_size, dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        for slot in range(num_experts_per_tok):
            expert_id = topk_expert_indices[i, slot].item()
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
    ref, _ = moe_w2_linear_ref(
        x=input_act,
        w=weight,
        topk_expert_indices=topk_expert_indices,
        num_experts=num_experts,
        num_topk=num_experts_per_tok,
        batch_size=batch_size,
        reduction_size=intermediate_size,
        output_size=hidden_size,
        residual=None,
        expert_offset=0,
        expert_stride=1,
    )

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

    # grid_dim and block_dim match demo/qwen3/demo_30B_A3B.py line 635-643
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.moe_w2_linear_layer(
        input=input_dt,
        weight=weight_dt,
        moe_routing_indices=routing_dt,
        moe_mask=mask_dt,
        output=output_dt,
        grid_dim=(8, 16, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk()
    torch.cuda.synchronize()

    print(f"\nOutput[0, 0, :8]:    {output[0, 0, :8]}")
    print(f"Reference[0, 0, :8]: {ref[0, 0, :8]}")

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("\nPASSED: BF16 MoE W2 linear produces correct output")

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_w2_linear_testmode()
