"""Test for the functional ``layers.add`` op via PersistentKernel test_mode.

Builds a tiny PK with ``test_mode=True``, attaches two bfloat16 inputs and an
output buffer, calls ``layers.add(a_dt, b_dt, output=out)`` inside the PK's
compile scope, runs the kernel once, and verifies against the PyTorch ``+``
operator. Mirrors the shape of
``tests/runtime_python/test_mode/test_rmsnorm_testmode.py`` (the canonical
test-mode harness).

Run:
    python tests/runtime_python/layers/test_add.py

NOTE: ``elementwise_add`` only has a kernel on Blackwell (SM100) today
(the registered task is ``"elementwise_add_sm100"``). On older arches this
test is expected to fail at compile time; that is a kernel-coverage gap, not a
bug in this test.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_add_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Tiny shapes — keep compile + run cheap. The kernel partitions on dim 0
    # (batch), so batch_size > 1 exercises the parallel axis.
    batch_size = 16
    hidden_dim = 4096

    # Inputs / output — all matching shape (batch_size, hidden_dim) bf16.
    a = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
    b = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device)
    out = torch.zeros(batch_size, hidden_dim, dtype=dtype, device=device)

    # PyTorch reference — the catalog docstring for add() documents this
    # equivalence (the "forward()" of a functional layer).
    ref = a + b

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

    # Attach inputs (bf16) and the output buffer (also via attach_input so the
    # host can read back after pk() returns).
    a_dt = pk.attach_input(a, name="add_a")
    b_dt = pk.attach_input(b, name="add_b")

    print("Building add layer inside compile_scope ...")
    with pk.compile_scope():
        out_dt = layers.add(a_dt, b_dt, output=out, name="add_out")
    assert out_dt is not None, "layers.add returned None"

    print("Compiling test kernel ...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel ...")
    pk()
    torch.cuda.synchronize()

    print(f"out[0, :8]: {out[0, :8]}")
    print(f"ref[0, :8]: {ref[0, :8]}")

    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"Max absolute diff: {max_diff:.6f}")

    # bf16 elementwise add is bit-exact in fp32 then rounded once — a tight
    # tolerance is OK. Match the qwen3-MLP-test style.
    try:
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
    except AssertionError as exc:
        print(f"FAILED: torch.testing.assert_close raised: {exc}")
        pk.finalize()
        sys.exit(1)

    print("PASSED: layers.add produces correct output")

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_add_testmode()
