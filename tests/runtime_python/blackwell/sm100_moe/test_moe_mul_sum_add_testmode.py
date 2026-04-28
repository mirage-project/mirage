"""
Test: BF16 MoE weighted-sum + residual via PersistentKernel test_mode.

Tests the moe_mul_sum_add_layer (final MoE reduction) end-to-end through the
full MPK compilation pipeline:

  output[b, :] = sum_k (input[b, k, :] * weight[b, k]) + residual[b, :]

Run:
    python tests/runtime_python/blackwell/sm100_moe/test_moe_mul_sum_add_testmode.py
"""

import torch
import sys
import os

from torch.nn import functional as F

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import moe_mul_sum_add_ref


def test_moe_mul_sum_add_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    hidden_size = 256          # H — must be a multiple of 256 for grid_dim
    num_experts = 128
    num_experts_per_tok = 8

    print(f"\n{'='*60}")
    print(f"Test: BF16 MoE weighted-sum + residual")
    print(f"  B={batch_size}, K={num_experts_per_tok}, H={hidden_size}")
    print(f"{'='*60}")

    # input: (B, K, H)
    x = torch.randn(
        batch_size, num_experts_per_tok, hidden_size,
        dtype=dtype, device=device,
    )
    # residual: (B, H)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    # weights: (B, K) float32 — softmax over expert scores
    expert_score = torch.randn(batch_size, num_experts, dtype=dtype, device=device)
    topk_expert_score, _ = torch.topk(expert_score, num_experts_per_tok, dim=1)
    topk_weights = F.softmax(topk_expert_score, dim=1, dtype=torch.float)
    output = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

    # --- PyTorch reference ---
    ref = moe_mul_sum_add_ref(x, topk_weights, residual)

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

    input_dt = pk.attach_input(x, name="input")
    weight_dt = pk.attach_input(topk_weights, name="weight")
    residual_dt = pk.attach_input(residual, name="residual")
    output_dt = pk.attach_input(output, name="output")

    # grid: (max_num_batched_tokens, hidden_size//256, 1) per demo line 644-651
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.moe_mul_sum_add_layer(
        input=input_dt,
        weight=weight_dt,
        residual=residual_dt,
        output=output_dt,
        grid_dim=(pk.max_num_batched_tokens, hidden_size // 256, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk()
    torch.cuda.synchronize()

    print(f"\nOutput[0, :8]:    {output[0, :8]}")
    print(f"Reference[0, :8]: {ref[0, :8]}")

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("\nPASSED: BF16 MoE weighted-sum produces correct output")

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_mul_sum_add_testmode()
