"""Test the ``layers.RMSNorm`` catalog module via PersistentKernel test_mode.

This file is the Phase-2 test for the RMSNorm catalog migration. It is
a direct port of ``tests/runtime_python/test_mode/test_rmsnorm_testmode.py``
to the new ``MPKModule`` API: the module owns its weight as an
``nn.Parameter``; ``forward`` provides the PyTorch reference; ``compile``
is invoked inside a ``pk.compile_scope()`` and routes the output buffer
through ``pk.attach_input`` so the host can read back from it.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU. The ``mirage`` conda env is required.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_rmsnorm_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 16
    # From test_rmsnorm_testmode.py line 27: hidden_size must satisfy
    # ``HIDDEN_DIM * sizeof(dtype) / NUM_THREADS >= 4``. With dtype=bf16
    # (2 bytes) and NUM_THREADS=256 (Hopper/Blackwell), the smallest
    # legal hidden_size is 512; 4096 is comfortably above the bar.
    hidden_size = 4096
    eps = 1e-6  # Match the kernel's hard-coded ``1e-6f`` (see rmsnorm.py
                # docstring) so forward() agrees with the compiled path.

    # ------------------------------------------------------------------
    # Build module + reference
    # ------------------------------------------------------------------
    m = layers.RMSNorm(hidden_size=hidden_size, eps=eps, prefix="test_")

    # Seed the weight with a randn — matching test_rmsnorm_testmode.py
    # rather than the all-ones default — so we exercise the scale path.
    w = torch.randn(hidden_size, dtype=dtype, device=device)
    m.weight.data = m.weight.data.to(device=device, dtype=dtype)
    m.weight.data.copy_(w)

    # Input and an output buffer the test driver will read back from.
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out_buf = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

    # PyTorch reference. m.forward respects m.weight, so we get the
    # scaled output.
    ref = m.forward(x)

    # ------------------------------------------------------------------
    # Build PersistentKernel in test mode
    # ------------------------------------------------------------------
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

    # Attach the input. ``compile()`` will attach the output torch.Tensor
    # itself via the ``output=`` path so the host can inspect it.
    x_dt = pk.attach_input(x, name="x")

    with pk.compile_scope():
        _ = m.compile(x_dt, output=out_buf)

    # ------------------------------------------------------------------
    # Compile and run once
    # ------------------------------------------------------------------
    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print(f"out_buf[:2, :8]: {out_buf[:2, :8]}")
    print(f"ref[:2, :8]:     {ref[:2, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        # bf16 tolerance, matching the catalog-test convention.
        torch.testing.assert_close(out_buf, ref, atol=0.05, rtol=0.05)
        print("PASSED: layers.RMSNorm compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: layers.RMSNorm compile() disagrees with forward()\n{e}")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_rmsnorm_testmode()
