"""
Proof-of-concept test for PersistentKernel test_mode.

Tests the rmsnorm_layer by building a minimal PersistentKernel in test_mode,
compiling it, running it once, and comparing the output to a PyTorch reference.
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps=1e-5):
    """Reference RMSNorm implementation in PyTorch."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def test_rmsnorm_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 16
    hidden_dim = 4096  # Must satisfy: HIDDEN_DIM * sizeof(dtype) / NUM_THREADS >= 4

    # Create input tensors
    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
    w = torch.randn(hidden_dim, dtype=dtype, device=device)
    out = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)

    # Build PersistentKernel in test mode
    
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(
      **params
    )

    # Attach tensors to graph
    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    # Build layer
    target_cc = pk.target_cc
    if target_cc >= 90:
        block_dim = (256, 1, 1)
    else:
        block_dim = (128, 1, 1)
    pk.rmsnorm_layer(input=x_dt, weight=w_dt, output=out_dt, 
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)

    # Compile
    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    # Run
    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Compare against reference
    ref = torch_rmsnorm(x, w)
    print(f"Output:\n{out[:2, :8]}")
    print(f"Reference:\n{ref[:2, :8]}")

    max_diff = (out - ref).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    if max_diff < 0.05:
        print("PASSED: rmsnorm test_mode produces correct output")
    else:
        print(f"FAILED: max diff {max_diff} exceeds tolerance 0.05")
        sys.exit(1)

    # Cleanup
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_rmsnorm_testmode()
