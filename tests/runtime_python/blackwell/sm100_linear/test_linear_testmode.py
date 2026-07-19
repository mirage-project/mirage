"""Test the MPK `linear_layer` end-to-end through the test_mode pipeline.

  out = input @ weight.T

Run:
    python tests/runtime_python/blackwell/sm100_linear/test_linear_testmode.py
"""

import torch
import sys
import os

# Make `pytorch_reference` importable when run from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import linear_ref


def grid_for_linear(output_dim):
    """Compute grid_dim.x for linear layer (matches demo/qwen3/demo.py)."""
    if output_dim % 96 == 0:
        return output_dim // 96
    elif output_dim % 64 == 0:
        return output_dim // 64
    else:
        raise AssertionError(f"Unsupported linear output_dim={output_dim}")


def test_linear_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 16
    in_dim = 4096
    out_dim = 4096

    print(f"\n{'='*60}")
    print(f"Test: linear_layer (out = x @ w.T)")
    print(f"  batch={batch_size}, in_dim={in_dim}, out_dim={out_dim}")

    x = torch.randn(batch_size, in_dim, dtype=dtype, device=device) * 0.1
    w = torch.randn(out_dim, in_dim, dtype=dtype, device=device) * 0.01
    out = torch.zeros(batch_size, out_dim, dtype=dtype, device=device)

    # Build PersistentKernel (test mode)
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

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    num_tasks = grid_for_linear(out_dim)

    pk.linear_layer(
        input=x_dt,
        weight=w_dt,
        output=out_dt,
        grid_dim=(num_tasks, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk()
    torch.cuda.synchronize()

    ref = linear_ref(x, w)
    print(f"\nout[0, :8]: {out[0, :8]}")
    print(f"ref[0, :8]: {ref[0, :8]}")

    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"\nMax absolute diff: {max_diff:.6f}")

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASSED: linear_layer test_mode")
    pk.finalize()


if __name__ == "__main__":
    test_linear_testmode()
