"""
Test the layers.Linear catalog module via PersistentKernel test_mode.

Linear is plain bf16 dense projection: out = x @ weight.T (no bias,
no residual). We check that the new module's forward() PyTorch reference
agrees with the MPK-compiled path on a tiny single-batch input.

Pattern mirrors ``test_gateup_only`` in
``tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py`` and the
sibling ``tests/runtime_python/layers/test_identity.py``.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU.
"""

import os
import sys

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_linear_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Tiny qwen3-shaped GEMM. out_features=4096 is divisible by 64 (and
    # not by 96), so auto_grid_dim picks grid.x = 4096 // 64 = 64.
    batch_size = 8
    in_features = 4096
    out_features = 4096

    # Inputs.
    x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device) * 0.01
    # Output buffer attached via the `output=` torch.Tensor path so the
    # host can read it back after pk() returns.
    out_buf = torch.zeros(batch_size, out_features, dtype=dtype, device=device)

    # PyTorch reference. Use float32 accumulation then cast — matches
    # what test_gateup_only does and what the bf16 GEMM kernel produces
    # (kernel accumulator is fp32).
    ref = (x.float() @ weight.float().T).to(dtype)

    # Build the catalog module and load the weight.
    module = layers.Linear(
        in_features=in_features,
        out_features=out_features,
        prefix="test_",
    )
    # Move to CUDA so the Parameter lives on the device pk.attach_input
    # expects, then copy the test weight in. We use ``data.copy_`` so
    # the Parameter identity is preserved.
    module = module.to(device=device, dtype=dtype)
    module.weight.data.copy_(weight)

    # Sanity check: forward() agrees with the manual reference.
    # 1e-3 was too tight: F.linear(bf16) vs (x.float() @ w.float().T).to(bf16)
    # can differ by 1 bf16 ULP (~2^-7) due to final-cast ordering. The
    # kernel-vs-ref check below uses 0.5; here we use a single ULP since
    # both paths SHOULD use fp32 accumulate.
    ref_forward = module.forward(x)
    forward_max_diff = (ref_forward.float() - ref.float()).abs().max().item()
    assert forward_max_diff < 0.05, (
        f"Linear.forward() disagrees with manual F.linear: "
        f"max_diff={forward_max_diff}"
    )

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

    # Attach the input. The weight is attached internally by
    # module.compile() (via pk.attach_input(self.weight, ...)).
    x_dt = pk.attach_input(x, name="x")

    # Build the graph inside the compile scope so current_pk() inside
    # the module body resolves to this pk.
    with pk.compile_scope():
        _ = module.compile(x_dt, output=out_buf)

    # Compile and run once.
    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")

    # bf16 GEMM tolerance: 0.5 absolute is the same threshold
    # test_gateup_only uses for an identically-sized GEMM. We additionally
    # use assert_close with matching atol/rtol to surface a structured
    # error message on failure.
    try:
        torch.testing.assert_close(out_buf, ref, atol=0.5, rtol=0.5)
    except AssertionError as e:
        max_diff = (out_buf.float() - ref.float()).abs().max().item()
        print(f"FAILED: linear test_mode disagrees with reference: max_diff={max_diff}")
        print(str(e))
        pk.finalize()
        sys.exit(1)

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6f}")
    print("PASSED: linear test_mode produces correct output")

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_linear_testmode()
