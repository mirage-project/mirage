import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import tensor_init_ref


def test_tensor_init_testmode():
    """Smoke / correctness test for ``tensor_init_layer``.

    The kernel zero-fills its first input tensor (the ``input`` arg of
    ``tensor_init_layer``).  The ``dummy_input`` / ``dummy_output``
    DTensors are graph-wiring placeholders -- the kernel never touches
    their data -- so we just point them at the same scratch buffer.
    """
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 16
    hidden = 512  # small for fast compile

    # Pre-fill ``input`` with non-zero values so we can detect that the
    # kernel actually writes zeros (not just leaves the buffer alone).
    input_tensor = torch.randn(
        batch_size, hidden, dtype=torch.bfloat16, device=device
    )
    pre_kernel_snapshot = input_tensor.clone()
    dummy = torch.randn(
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

    input_dt = pk.attach_input(input_tensor, name="input")
    dummy_dt = pk.attach_input(dummy, name="dummy")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)

    pk.tensor_init_layer(
        input=input_dt,
        dummy_input=dummy_dt,
        dummy_output=dummy_dt,
        grid_dim=(batch_size, 1, 1),
        block_dim=block_dim,
    )

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    ref = tensor_init_ref(pre_kernel_snapshot, init_val=0.0)
    max_diff = (input_tensor - ref).abs().max().item()
    print(f"Max diff (input vs zeros): {max_diff}")

    # Sanity: the kernel really did overwrite the buffer.
    nonzero = input_tensor.abs().sum().item()
    print(f"Sum of |input| after kernel: {nonzero}")

    torch.testing.assert_close(input_tensor, ref, rtol=0.0, atol=0.0)
    print("PASSED")
    pk.finalize()


if __name__ == "__main__":
    test_tensor_init_testmode()
