import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import tensor_init_ref


def test_tensor_init_testmode():
    """Smoke / correctness test for ``tensor_init_layer``.

    The kernel zero-fills ``linear_output`` via vectorized 16-byte stores.
    ``linear_input`` is a graph-wiring placeholder; the kernel does not
    touch its data, so we just point it at a separate scratch buffer.
    """
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 16
    hidden = 512  # multiple of 8 (16B vec) and small for fast compile

    # Pre-fill ``linear_output`` with non-zero values so we can detect that
    # the kernel actually writes zeros.
    linear_output = torch.randn(
        batch_size, hidden, dtype=torch.bfloat16, device=device
    )
    pre_kernel_snapshot = linear_output.clone()
    linear_input = torch.randn(
        batch_size, hidden, dtype=torch.bfloat16, device=device
    )

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

    in_dt = pk.attach_input(linear_input, name="linear_input")
    out_dt = pk.attach_input(linear_output, name="linear_output")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)

    # Mirror the splitk linear's grid/input_map convention so the test
    # exercises the same path used by linear_splitk_swapAB_fp8_layer:
    # grid.x splits the output (target) dim, grid.y splits the dummy's K dim.
    pk.tensor_init_layer(
        target=out_dt,
        dummy=in_dt,
        grid_dim=(2, 1, 1),
        block_dim=block_dim,
        dummy_input_map=(-1, 1, -1),
        target_input_map=(1, -1, -1),
    )

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    ref = tensor_init_ref(pre_kernel_snapshot, init_val=0.0)
    max_diff = (linear_output - ref).abs().max().item()
    print(f"Max diff (linear_output vs zeros): {max_diff}")

    nonzero = linear_output.abs().sum().item()
    print(f"Sum of |linear_output| after kernel: {nonzero}")

    torch.testing.assert_close(linear_output, ref, rtol=0.0, atol=0.0)
    print("PASSED")
    pk.finalize()


if __name__ == "__main__":
    test_tensor_init_testmode()
