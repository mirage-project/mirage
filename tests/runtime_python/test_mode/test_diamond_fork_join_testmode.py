"""
Diamond fork+join test with GEMM branches.

Graph:
        x_input --A(rmsnorm)--> x_a
                                 |
                   +-------------+-------------+
                   |                           |
                   v                           v
            B(linear, w_b) -> x_b      C(linear, w_c) -> x_c
                   |                           |
                   +-------------+-------------+
                                 v
               D(linear_with_residual, input=x_b, residual=x_c) -> x_d

Role classification:
  - A: is_fork_producer (B and C are distinct downstream consumers)
  - D: is_join_consumer (B and C are distinct upstream producers)
  - B: is_fork_consumer (reads x_a from fork producer A)
       AND is_join_producer (writes x_b that feeds join consumer D)
       -> CASE 4 (allowed): trigger_event from fork event, dependent_event
          to join event. Two different slots, no conflict.
  - C: same as B (case 4)

This exercises:
  - Both fork and join in a single compilation.
  - The case-4 role combination on the two middle layers.
  - GEMM (linear) computation on the parallel branches, which uses a 2D
    grid (hidden-tiling x batch) different from rmsnorm's 1D batch grid,
    so the fork/join LCM math runs on non-trivial grid shapes.

Note on event-grid: linear's input_map for its activation input is
(-1, -1, -1) (replicated across all grid axes), which means the fork
event_dim between A and B/C collapses to 1-per-axis after LCM. Similarly
the join event_dim between B and D (via linear_with_residual's `input`
slot, also replicated) collapses. This is the expected outcome: when
any branch reads the producer's output with full replication, the
entire producer must be done before any consumer task starts, so a
single event per boundary is correct.
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


def test_diamond_fork_join_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 8
    hidden = 4096

    torch.manual_seed(3)
    x_input = torch.randn(batch_size, hidden, dtype=dtype, device=device) * 0.1
    w_a = torch.randn(hidden, dtype=dtype, device=device) * 0.1
    # Scale weights down so that linear output stays in a comfortable range
    # for bf16 accumulation.
    w_b = torch.randn(hidden, hidden, dtype=dtype, device=device) * 0.01
    w_c = torch.randn(hidden, hidden, dtype=dtype, device=device) * 0.01
    w_d = torch.randn(hidden, hidden, dtype=dtype, device=device) * 0.01

    x_a = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
    x_b = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
    x_c = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
    x_d = torch.zeros(batch_size, hidden, dtype=dtype, device=device)

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

    x_input_dt = pk.attach_input(x_input, name="x_input")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    w_c_dt = pk.attach_input(w_c, name="w_c")
    w_d_dt = pk.attach_input(w_d, name="w_d")
    x_a_dt = pk.attach_input(x_a, name="x_a")
    x_b_dt = pk.attach_input(x_b, name="x_b")
    x_c_dt = pk.attach_input(x_c, name="x_c")
    x_d_dt = pk.attach_input(x_d, name="x_d")

    target_cc = pk.target_cc
    block_dim = (256, 1, 1) if target_cc >= 90 else (128, 1, 1)
    linear_grid_x = hidden // 96 if hidden % 96 == 0 else hidden // 64

    # A: rmsnorm (fork producer). x_a = rmsnorm(x_input, w_a)
    pk.rmsnorm_layer(input=x_input_dt, weight=w_a_dt, output=x_a_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)
    # B: linear branch 1. x_b = x_a @ w_b.T
    pk.linear_layer(input=x_a_dt, weight=w_b_dt, output=x_b_dt,
                    grid_dim=(linear_grid_x, batch_size, 1), block_dim=block_dim)
    # C: linear branch 2. x_c = x_a @ w_c.T
    pk.linear_layer(input=x_a_dt, weight=w_c_dt, output=x_c_dt,
                    grid_dim=(linear_grid_x, batch_size, 1), block_dim=block_dim)
    # D: join consumer. x_d = x_b @ w_d.T + x_c
    pk.linear_with_residual_layer(
        input=x_b_dt, weight=w_d_dt, residual=x_c_dt, output=x_d_dt,
        grid_dim=(linear_grid_x, batch_size, 1), block_dim=block_dim)

    print("Compiling diamond fork+join test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running diamond fork+join test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref_a = torch_rmsnorm(x_input, w_a)
    ref_b = (ref_a.float() @ w_b.float().T).to(dtype)
    ref_c = (ref_a.float() @ w_c.float().T).to(dtype)
    ref_d = (ref_b.float() @ w_d.float().T).to(dtype) + ref_c

    def max_diff(a, b):
        return (a - b).abs().max().item()

    diff_a = max_diff(x_a, ref_a)
    diff_b = max_diff(x_b, ref_b)
    diff_c = max_diff(x_c, ref_c)
    diff_d = max_diff(x_d, ref_d)
    print(f"diff A = {diff_a}, B = {diff_b}, C = {diff_c}, D = {diff_d}")

    # GEMM accumulation in bf16 has higher noise than rmsnorm; scale tol
    # with number of GEMMs chained.
    if diff_a < 0.1 and diff_b < 0.5 and diff_c < 0.5 and diff_d < 1.0:
        print("PASSED: diamond_fork_join test_mode produces correct output")
    else:
        print(f"FAILED: diffs A={diff_a} B={diff_b} C={diff_c} D={diff_d}")
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_diamond_fork_join_testmode()
