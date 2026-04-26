"""
Fork-point test for the AnnotatedGraph pre-pass and fork emission.

Graph:
      x_input --rmsnorm(A)--> x_a --rmsnorm(B)--> x_b
                                \\--rmsnorm(C)--> x_c

A's output x_a is consumed by TWO distinct downstream layers (B and C), which
is the canonical fork pattern. We verify:
  1. Compilation succeeds (pre-pass detects 1 fork group, 2 outgoing edges).
  2. Output matches a PyTorch reference.
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def test_fork_point_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 16
    hidden_dim = 4096

    torch.manual_seed(0)
    x_input = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
    w_a = torch.randn(hidden_dim, dtype=dtype, device=device)
    w_b = torch.randn(hidden_dim, dtype=dtype, device=device)
    w_c = torch.randn(hidden_dim, dtype=dtype, device=device)

    x_a = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)
    x_b = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)
    x_c = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    x_input_dt = pk.attach_input(x_input, name="x_input")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    w_c_dt = pk.attach_input(w_c, name="w_c")
    x_a_dt = pk.attach_input(x_a, name="x_a")
    x_b_dt = pk.attach_input(x_b, name="x_b")
    x_c_dt = pk.attach_input(x_c, name="x_c")

    target_cc = pk.target_cc
    block_dim = (256, 1, 1) if target_cc >= 90 else (128, 1, 1)

    # Layer A: x_input -> x_a  (fork producer: x_a feeds both B and C)
    pk.rmsnorm_layer(input=x_input_dt, weight=w_a_dt, output=x_a_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)
    # Layer B: x_a -> x_b  (fork branch 1)
    pk.rmsnorm_layer(input=x_a_dt, weight=w_b_dt, output=x_b_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)
    # Layer C: x_a -> x_c  (fork branch 2)
    pk.rmsnorm_layer(input=x_a_dt, weight=w_c_dt, output=x_c_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)

    print("Compiling fork-point test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running fork-point test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref_a = torch_rmsnorm(x_input, w_a)
    ref_b = torch_rmsnorm(ref_a, w_b)
    ref_c = torch_rmsnorm(ref_a, w_c)

    def max_diff(a, b):
        return (a - b).abs().max().item()

    diff_a = max_diff(x_a, ref_a)
    diff_b = max_diff(x_b, ref_b)
    diff_c = max_diff(x_c, ref_c)
    print(f"diff A = {diff_a}, diff B = {diff_b}, diff C = {diff_c}")

    tol = 0.1
    if max(diff_a, diff_b, diff_c) < tol:
        print("PASSED: fork_point test_mode produces correct output on both branches")
    else:
        print(f"FAILED: one or more branch outputs exceed tolerance {tol}")
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_fork_point_testmode()
