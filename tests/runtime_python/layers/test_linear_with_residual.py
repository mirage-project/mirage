"""Test for the ``LinearWithResidual`` catalog module via test_mode.

Builds a tiny PK with ``test_mode=True``, instantiates
``layers.LinearWithResidual``, copies a weight in, attaches the input /
residual / output buffers, calls ``module.compile(x_dt, residual_dt,
output=out_buf)`` inside ``pk.compile_scope()``, runs the kernel once,
and verifies against the PyTorch ``F.linear(x, W) + residual`` reference.

Shape pattern follows the down-projection in
``test_qwen3_mlp_testmode.test_gateup_silu_down``:
input ``(B, intermediate)`` -> linear-out ``(B, hidden)`` then add the
``(B, hidden)`` residual.

Run:
    python tests/runtime_python/layers/test_linear_with_residual.py
"""

import os
import sys

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk.layers.linear.linear_with_residual import LinearWithResidual
from mirage.mpk.persistent_kernel import PersistentKernel


def test_linear_with_residual_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Down-proj-like shape (mirrors test_gateup_silu_down's mlp_out step):
    #   x:        (8, 2048)        -- input from silu_mul
    #   weight:   (4096, 2048)     -- (out_features, in_features)
    #   residual: (8, 4096)        -- carries the residual stream at hidden width
    #   output:   (8, 4096)        -- F.linear(x, W) + residual
    batch_size = 8
    in_features = 2048
    out_features = 4096

    print(f"\n{'=' * 60}")
    print(
        f"Test: LinearWithResidual  B={batch_size}, in={in_features}, "
        f"out={out_features}"
    )

    # Small weight magnitudes so the matmul stays in bf16's dynamic range
    # — matches the convention in test_qwen3_mlp_testmode.
    x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
    weight = torch.randn(out_features, in_features, dtype=dtype, device=device) * 0.01
    residual = torch.randn(batch_size, out_features, dtype=dtype, device=device)
    out_buf = torch.zeros(batch_size, out_features, dtype=dtype, device=device)

    # PyTorch reference: F.linear(x, W) + residual, computed in fp32 then
    # cast back to bf16 — the kernel accumulates in fp32 internally.
    ref = (
        (x.float() @ weight.float().T) + residual.float()
    ).to(dtype)

    # Build module and copy the weight in. We allocate the parameter on
    # the device with the right dtype so attach_input sees a CUDA-resident
    # bf16 tensor (the assert chain in pk.attach_input only checks
    # layout, but the kernel reads bf16).
    module = LinearWithResidual(in_features=in_features, out_features=out_features)
    module = module.to(device=device, dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(weight)

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

    # Attach input + residual as graph inputs. The weight is attached
    # inside module.compile() (via self.weight). The output buffer is
    # attached too so the test driver can read it back after pk().
    x_dt = pk.attach_input(x, name="lwr_input")
    residual_dt = pk.attach_input(residual, name="lwr_residual")

    print("Building LinearWithResidual inside compile_scope ...")
    with pk.compile_scope():
        out_dt = module.compile(x_dt, residual_dt, output=out_buf)
    assert out_dt is not None, "LinearWithResidual.compile returned None"

    print("Compiling test kernel ...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel ...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:    {ref[0, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute diff: {max_diff:.6f}")

    # Looser tolerance for the fused GEMM+add path, matching the
    # convention in test_qwen3_mlp_testmode.test_gateup_silu_down
    # (which uses max-diff < 1.0 for the same shape and dtype).
    try:
        torch.testing.assert_close(out_buf, ref, atol=1.0, rtol=1.0)
    except AssertionError as exc:
        print(f"FAILED: torch.testing.assert_close raised: {exc}")
        pk.finalize()
        sys.exit(1)

    print("PASSED: LinearWithResidual produces correct output")

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_linear_with_residual_testmode()
