"""
Grid-dim sweep test for the fork-point emission.

Re-runs the basic fork pattern (A -> {B, C}) under several (batch_size,
hidden) configurations. Varying batch_size directly changes the number of
fork events emitted (one per batch item when grid_dim.x = batch_size), so
this exercises the fork emission / prelaunch conversion / JSON task flush
across meaningfully different task-id layouts.

Parameters covered:
  (batch_size=4,  hidden=4096) -> 4 fork events, 12 tasks total (3 layers x 4)
  (batch_size=8,  hidden=4096) -> 8 fork events, 24 tasks
  (batch_size=16, hidden=4096) -> 16 fork events, 48 tasks
  (batch_size=8,  hidden=2048) -> 8 fork events at a narrower hidden dim

Each config is compiled and executed in its own PersistentKernel so state
from one config does not leak into the next.
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


def _run_one_config(batch_size, hidden, tag):
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42 + batch_size * 100 + hidden)

    x_input = torch.randn(batch_size, hidden, dtype=dtype, device=device)
    w_a = torch.randn(hidden, dtype=dtype, device=device)
    w_b = torch.randn(hidden, dtype=dtype, device=device)
    w_c = torch.randn(hidden, dtype=dtype, device=device)
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
    pk.rmsnorm_layer(input=x_a_dt, weight=w_c_dt, output=x_c_dt,
                     grid_dim=(batch_size, 1, 1), block_dim=block_dim)

    print(f"[{tag}] Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print(f"[{tag}] Running...")
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

    tol = 0.2
    ok = diff_a < tol and diff_b < tol and diff_c < tol
    status = "OK" if ok else "FAIL"
    print(f"[{tag}] diff A={diff_a:.5f}, B={diff_b:.5f}, C={diff_c:.5f} {status}")

    pk.finalize()
    return ok


def test_grid_dim_sweep_testmode():
    configs = [
        (4, 4096, "b=4 h=4096"),
        (8, 4096, "b=8 h=4096"),
        (16, 4096, "b=16 h=4096"),
        (8, 2048, "b=8 h=2048"),
    ]
    results = []
    for batch, hidden, tag in configs:
        print(f"\n=== {tag} ===")
        try:
            ok = _run_one_config(batch, hidden, tag)
        except Exception as e:
            print(f"[{tag}] exception: {e}")
            ok = False
        results.append((tag, ok))

    print("\nSummary:")
    for tag, ok in results:
        print(f"  {tag}: {'PASS' if ok else 'FAIL'}")

    if all(ok for _, ok in results):
        print("PASSED: grid_dim_sweep all configs produce correct output")
    else:
        print("FAILED: at least one config failed")
        sys.exit(1)


if __name__ == "__main__":
    test_grid_dim_sweep_testmode()
