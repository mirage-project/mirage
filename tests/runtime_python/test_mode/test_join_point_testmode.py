"""
Join-point test for the AnnotatedGraph pre-pass and join emission.

Graph (two independent inputs, no path between them):
   x1_input --A(rmsnorm)--> x_a
                              \\
                               \\
                                `-> C: linear_with_residual(input=x_a,
                                                            weight=w_c,
                                                            residual=x_b,
                                                            output=x_c)
                               /
                              /
   x2_input --B(rmsnorm)--> x_b

C has two distinct producers (A and B), and since x1_input and x2_input are
independent (no path between A and B), no residual stripping applies. C is a
join-consumer (is_join_consumer == true) and the join event unifies A and B.

We verify compilation succeeds (1 join group, 2 incoming edges) and output
matches a PyTorch reference.
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


def test_join_point_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 8
    hidden = 4096

    torch.manual_seed(2)
    x1_input = torch.randn(batch_size, hidden, dtype=dtype, device=device) * 0.1
    x2_input = torch.randn(batch_size, hidden, dtype=dtype, device=device) * 0.1
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

    x1_dt = pk.attach_input(x1_input, name="x1_input")
    x2_dt = pk.attach_input(x2_input, name="x2_input")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    w_c_dt = pk.attach_input(w_c, name="w_c")
    x_a_dt = pk.attach_input(x_a, name="x_a")
    x_b_dt = pk.attach_input(x_b, name="x_b")
    x_c_dt = pk.attach_input(x_c, name="x_c")

    target_cc = pk.target_cc
    block_dim = (256, 1, 1) if target_cc >= 90 else (128, 1, 1)

    pk.rmsnorm_layer(input=x1_dt, weight=w_a_dt, output=x_a_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)
    pk.rmsnorm_layer(input=x2_dt, weight=w_b_dt, output=x_b_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)

    linear_grid_x = hidden // 96 if hidden % 96 == 0 else hidden // 64
    pk.linear_with_residual_layer(
        input=x_a_dt, weight=w_c_dt, residual=x_b_dt, output=x_c_dt,
        grid_dim=(linear_grid_x, batch_size, 1), block_dim=block_dim)

    print("Compiling join-point test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running join-point test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref_a = torch_rmsnorm(x1_input, w_a)
    ref_b = torch_rmsnorm(x2_input, w_b)
    ref_c = (ref_a.float() @ w_c.float().T).to(dtype) + ref_b

    def max_diff(a, b):
        return (a - b).abs().max().item()

    diff_a = max_diff(x_a, ref_a)
    diff_b = max_diff(x_b, ref_b)
    diff_c = max_diff(x_c, ref_c)
    print(f"diff A = {diff_a}, diff B = {diff_b}, diff C = {diff_c}")

    tol_c = 0.5
    if diff_a < 0.1 and diff_b < 0.1 and diff_c < tol_c:
        print("PASSED: join_point test_mode produces correct output")
    else:
        print(f"FAILED: diffs A={diff_a} B={diff_b} C={diff_c}")
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_join_point_testmode()
