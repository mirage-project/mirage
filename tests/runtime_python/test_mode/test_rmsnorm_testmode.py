"""
Proof-of-concept test for PersistentKernel test_mode.

Tests the rmsnorm_layer by building a minimal PersistentKernel in test_mode,
compiling it, running it once, and comparing the output to a PyTorch
reference. Runs once with the default eps (1e-6, no task param emitted) and
once with eps=1e-5 (GLM-4.6's rms_norm_eps, threaded through as float bits
in the task params). The eps=1e-5 case uses small-magnitude inputs
(variance ~ eps) so a kernel that ignored the parameter would be far
outside tolerance.
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps):
    """Reference RMSNorm implementation in PyTorch."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def run_rmsnorm_case(eps, x_scale):
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 16
    hidden_dim = 4096  # Must satisfy: HIDDEN_DIM * sizeof(dtype) / NUM_THREADS >= 4

    # Create input tensors
    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device) * x_scale
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
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim,
                     eps=eps)

    # Compile
    print(f"Compiling test kernel (eps={eps})...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    # Run
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()
    pk.finalize()

    # Compare against reference computed with the SAME eps
    ref = torch_rmsnorm(x, w, eps)
    max_diff = (out - ref).abs().max().item()
    print(f"eps={eps}: max absolute difference vs reference: {max_diff}")
    ok = max_diff < 0.05

    # With x_scale small enough that variance ~ eps, the wrong-eps reference
    # is grossly different; require the kernel to be much closer to the
    # right-eps reference than to the wrong-eps one.
    wrong_eps = 1e-5 if eps == 1e-6 else 1e-6
    wrong_ref = torch_rmsnorm(x, w, wrong_eps)
    wrong_gap = (wrong_ref - ref).abs().max().item()
    if wrong_gap > 0.1:
        wrong_diff = (out - wrong_ref).abs().max().item()
        print(f"eps={eps}: max diff vs wrong-eps ({wrong_eps}) reference: "
              f"{wrong_diff}")
        ok = ok and wrong_diff > 0.1

    if not ok:
        print(f"FAILED: eps={eps} max diff {max_diff} exceeds tolerance 0.05 "
              f"or output matches the wrong eps")
    return ok


def test_rmsnorm_testmode():
    # (eps, x_scale): default eps with unit-scale inputs (backward compat),
    # and GLM-4.6's 1e-5 with inputs small enough that eps dominates.
    cases = [(1e-6, 1.0), (1e-5, 1e-3)]
    results = [(eps, run_rmsnorm_case(eps, x_scale)) for eps, x_scale in cases]

    for eps, ok in results:
        print(f"eps={eps}: {'PASSED' if ok else 'FAILED'}")
    if not all(ok for _, ok in results):
        sys.exit(1)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_rmsnorm_testmode()
