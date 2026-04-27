"""
Test: BF16 MoE SiLU-mul via PersistentKernel test_mode.

Tests the per-expert fused SiLU + multiply (moe_silu_mul_layer) end-to-end
through the full MPK compilation pipeline.

  output[b, k, :I] = silu(input[b, k, :I]) * input[b, k, I:]

Run:
    python tests/runtime_python/blackwell/sm100_moe/test_moe_silu_mul_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import moe_silu_mul_ref


def test_moe_silu_mul_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 1
    num_experts_per_tok = 8
    intermediate_size = 768  # I

    print(f"\n{'='*60}")
    print(f"Test: BF16 MoE SiLU-mul")
    print(f"  B={batch_size}, K={num_experts_per_tok}, I={intermediate_size}")
    print(f"{'='*60}")

    # input: (B, K, 2*I), output: (B, K, I)
    input_act = torch.randn(
        batch_size, num_experts_per_tok, intermediate_size * 2,
        dtype=dtype, device=device,
    )
    output = torch.zeros(
        batch_size, num_experts_per_tok, intermediate_size,
        dtype=dtype, device=device,
    )

    # --- PyTorch reference ---
    ref = moe_silu_mul_ref(input_act, intermediate_size)

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
    output_dt = pk.attach_input(output, name="output")

    # grid: (max_num_batched_tokens, num_experts_per_tok, 1) — one TB per (token, slot)
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.moe_silu_mul_layer(
        input=input_dt,
        output=output_dt,
        grid_dim=(pk.max_num_batched_tokens, num_experts_per_tok, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nOutput[0, 0, :8]:    {output[0, 0, :8]}")
    print(f"Reference[0, 0, :8]: {ref[0, 0, :8]}")

    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("\nPASSED: BF16 MoE SiLU-mul produces correct output")

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_silu_mul_testmode()
