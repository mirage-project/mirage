"""
Test the layers.Identity catalog module via PersistentKernel test_mode.

Identity is a memory copy: out = x. We check that the new module
- forward(x) matches x.clone()
- compile(x, output=out_torch) under pk.compile_scope() produces a
  kernel that, when run once in test_mode, writes x's values into the
  attached `out` tensor.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_identity_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Tiny shape. Inner dim is divisible by typical grid.x choices.
    batch_size = 8
    hidden_dim = 1024

    # Input and output torch tensors.
    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
    out = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)

    # PyTorch reference: Identity.forward(x) returns x.clone(); the
    # value equals x.
    module = layers.Identity(prefix="test_")
    ref = module.forward(x)
    # Sanity: forward() is a clone, so it equals x value-wise but is a
    # distinct buffer.
    assert ref.data_ptr() != x.data_ptr()
    assert torch.equal(ref, x)

    # Build PersistentKernel in test mode.
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

    # Attach input. compile() will attach `out` itself via the
    # `output=` torch.Tensor path.
    x_dt = pk.attach_input(x, name="x")

    with pk.compile_scope():
        # output=out (torch.Tensor) routes through pk.attach_input so
        # we can inspect `out` after running.
        _ = module.compile(x_dt, output=out)

    # Compile and run once.
    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Identity is a byte-for-byte copy: tolerance is 0.
    max_diff = (out - ref).abs().max().item()
    print(f"out[:1, :8]: {out[:1, :8]}")
    print(f"ref[:1, :8]: {ref[:1, :8]}")
    print(f"Max absolute difference: {max_diff}")

    if max_diff == 0.0:
        print("PASSED: identity test_mode produces exact copy")
    else:
        print(f"FAILED: identity copy disagrees by {max_diff} (expected 0)")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_identity_testmode()
