"""Test the ``layers.TensorInit`` catalog module via PersistentKernel test_mode.

TensorInit zero-fills a target buffer with a dependency-chained task.
This is a numerical test: the compiled kernel should leave ``target``
equal to zero. The dummy carrier holds dependency info only.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.tensor_init import TensorInit


def test_tensor_init_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 2
    # tensor_init is designed for splitk-linear outputs: OUTPUT_SIZE per
    # task must be a multiple of 8 (16B vec). With target_input_map=(1,
    # -1, -1), OUTPUT_SIZE = hidden / grid.x; with the default auto-grid
    # (num_workers ~ SM count), we need hidden very large to keep
    # alignment. Override with grid.x=1 so OUTPUT_SIZE = hidden.
    hidden_size = 256  # multiple of 8

    # `target` must be pre-filled with something non-zero so we can
    # tell the kernel actually wrote zeros.
    target = torch.full(
        (batch_size, hidden_size), 7.0, dtype=dtype, device=device
    )
    # `dummy` is a dependency carrier; the kernel never reads/writes it.
    dummy = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    m = TensorInit(prefix="test_")
    ref = m.forward(target)
    # Sanity check the reference.
    assert torch.equal(ref, torch.zeros_like(target))

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

    target_dt = pk.attach_input(target, name="target")
    dummy_dt = pk.attach_input(dummy, name="dummy")

    with pk.compile_scope():
        # grid.x=1 so OUTPUT_SIZE=hidden_size=256 (mult of 8).
        _ = m.compile(target_dt, dummy_dt, grid_dim=(1, 1, 1),
                      block_dim=(128, 1, 1))

    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"target[0, :8]: {target[0, :8]}")
    try:
        torch.testing.assert_close(target, ref, atol=1e-2, rtol=1e-2)
        print("PASSED: TensorInit zeros target")
    except AssertionError as e:
        print(f"FAILED: TensorInit did not zero target\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_tensor_init_testmode()
