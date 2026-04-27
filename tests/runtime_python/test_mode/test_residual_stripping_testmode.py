"""
Residual-stripping test for the AnnotatedGraph pre-pass.

Graph:
   x_input --A(rmsnorm)--> x_a --B(rmsnorm)--> x_b
                              \\______________________________
                                                              \\
   C: linear_with_residual(input=x_b, weight=w_c, residual=x_a, output=x_c)

The edge A -> C (via x_a as residual) is a direct shortcut that coexists with
the longer path A -> B -> C. Our pre-pass strips that direct edge as a
"false parallel path". After stripping:
  - A has 1 out-edge (to B) -> not a fork
  - C has 1 in-edge (from B)  -> not a join
  - The graph collapses to a simple chain A -> B -> C
  - The residual tensor x_a is still read by C at runtime, but scheduling
    dependency is purely through the B->C event (transitively covers A->B->C).

We verify the graph compiles (no case 2/3 errors despite the multi-input
consumer) and the output matches a PyTorch reference.
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


def test_residual_stripping_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 8
    hidden = 4096

    torch.manual_seed(1)
    x_input = torch.randn(batch_size, hidden, dtype=dtype, device=device) * 0.1
    w_a = torch.randn(hidden, dtype=dtype, device=device) * 0.1
    w_b = torch.randn(hidden, dtype=dtype, device=device) * 0.1
    w_c = torch.randn(hidden, hidden, dtype=dtype, device=device) * 0.01

    x_a = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
    x_b = torch.zeros(batch_size, hidden, dtype=dtype, device=device)
    x_c = torch.zeros(batch_size, hidden, dtype=dtype, device=device)

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
    x_a_dt = pk.attach_input(x_a, name="x_a")
    x_b_dt = pk.attach_input(x_b, name="x_b")
    x_c_dt = pk.attach_input(x_c, name="x_c")

    target_cc = pk.target_cc
    block_dim = (256, 1, 1) if target_cc >= 90 else (128, 1, 1)

    pk.rmsnorm_layer(input=x_input_dt, weight=w_a_dt, output=x_a_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)
    pk.rmsnorm_layer(input=x_a_dt, weight=w_b_dt, output=x_b_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)

    # C: residual reads x_a directly (shortcut), input comes from x_b.
    linear_grid_x = hidden // 96 if hidden % 96 == 0 else hidden // 64
    pk.linear_with_residual_layer(
        input=x_b_dt, weight=w_c_dt, residual=x_a_dt, output=x_c_dt,
        grid_dim=(linear_grid_x, batch_size, 1), block_dim=block_dim)

    print("Compiling residual-stripping test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running residual-stripping test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref_a = torch_rmsnorm(x_input, w_a)
    ref_b = torch_rmsnorm(ref_a, w_b)
    ref_c = (ref_b.float() @ w_c.float().T).to(dtype) + ref_a

    def max_diff(a, b):
        return (a - b).abs().max().item()

    diff_a = max_diff(x_a, ref_a)
    diff_b = max_diff(x_b, ref_b)
    diff_c = max_diff(x_c, ref_c)
    print(f"diff A = {diff_a}, diff B = {diff_b}, diff C = {diff_c}")

    tol_c = 0.5  # linear_with_residual has higher numerical error
    if diff_a < 0.1 and diff_b < 0.1 and diff_c < tol_c:
        print("PASSED: residual_stripping test_mode produces correct output")
    else:
        print(f"FAILED: diffs A={diff_a} B={diff_b} C={diff_c}")
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_residual_stripping_testmode()
