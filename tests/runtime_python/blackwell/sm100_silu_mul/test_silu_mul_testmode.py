"""
Test: SiLU-Mul layer via PersistentKernel test_mode (qwen3-style, non-MoE).

Op:  out = silu(input[..., :I]) * input[..., I:]   with I = intermediate.

Run:
    python tests/runtime_python/blackwell/sm100_silu_mul/test_silu_mul_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import silu_mul_ref


def grid_for_linear(output_dim):
    """Match demo/qwen3/demo.py grid policy used to size the upstream gate+up linear."""
    if output_dim % 96 == 0:
        return output_dim // 96
    elif output_dim % 64 == 0:
        return output_dim // 64
    else:
        raise AssertionError(f"Unsupported linear output_dim={output_dim}")


def test_silu_mul_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 16
    intermediate = 4096
    fused_outdim = 2 * intermediate

    print(f"\n{'='*60}")
    print(f"Test: silu_mul_layer (qwen3-style)")
    print(f"  B={batch_size}, intermediate={intermediate}, fused_in={fused_outdim}")

    # Inputs: gate||up concatenated along last dim.
    x = torch.randn(batch_size, fused_outdim, dtype=dtype, device=device)
    out = torch.zeros(batch_size, intermediate, dtype=dtype, device=device)

    # qo_indptr_buffer feeds num_active_tokens to the silu_mul kernel.
    # Layout: qo_indptr[0..B-1]=0, qo_indptr[MAX_BATCHED_REQUESTS]=B (== batch_size here).
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[batch_size] = batch_size

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    params["meta_tensors"] = {"qo_indptr_buffer": qo_indptr}
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    out_dt = pk.attach_input(out, name="out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)

    # Mirror qwen3 MLP pipeline: silu_mul grid is half of the upstream gate+up linear grid.
    num_tasks_gatedup = grid_for_linear(fused_outdim)
    num_tasks_silu = num_tasks_gatedup // 2
    print(f"  grid_dim=({num_tasks_silu},1,1)  block_dim={block_dim}")

    pk.silu_mul_layer(
        input=x_dt,
        output=out_dt,
        grid_dim=(num_tasks_silu, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk()
    torch.cuda.synchronize()

    ref = silu_mul_ref(x, num_tasks=num_tasks_silu)

    print(f"\nout[0, :8]: {out[0, :8]}")
    print(f"ref[0, :8]: {ref[0, :8]}")

    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"\nMax absolute diff: {max_diff:.6f}")

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("PASSED")

    pk.finalize()


if __name__ == "__main__":
    test_silu_mul_testmode()
