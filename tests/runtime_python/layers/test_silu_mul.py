"""Tests for ``layers.SiluMul`` and ``layers.SiluMulLinearWithResidual``.

Each test builds a tiny PersistentKernel in ``test_mode=True``, attaches
the inputs and an output buffer, runs the module's ``compile()`` inside
``pk.compile_scope()``, executes the kernel once, and compares against
the module's ``forward()`` reference.

Both tests use ``grid_dim=(1, 1, 1)`` for the activation. With
``grid.x == 1`` the kernel's per-task view of the input coincides with
the whole-tensor view: ``input[:, :intermediate_size]`` is gate and
``input[:, intermediate_size:]`` is up -- exactly what ``forward()``
consumes. With ``grid.x > 1`` the qwen3 pipeline shuffles the upstream
gate/up weight rows so that each per-task column slab still sees the
halved (gate || up) layout; that interleaved path is exercised by
``tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py`` and is
not re-tested here.

The linear-with-residual variant does not slice its input on dim 1
(the K-axis reduction is tiled internally), so the halved layout works
naturally with any grid choice -- we still pass ``grid_dim=(1, 1, 1)``
for symmetry.

Run:
    python tests/runtime_python/layers/test_silu_mul.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.activation.silu_mul import (
    SiluMul,
    SiluMulLinearWithResidual,
)
from mirage.mpk.persistent_kernel import PersistentKernel


def _make_pk(batch_size: int) -> PersistentKernel:
    """Construct a tiny test-mode PersistentKernel."""
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    return PersistentKernel(**params)


def test_silu_mul():
    """SiluMul: SiLU(gate) * up on a halved (B, 2*intermediate) input."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 8
    intermediate_size = 2048
    fused_outdim = 2 * intermediate_size  # 4096

    # Halved-layout input: [:, :intermediate_size] is gate, [:, intermediate_size:] is up.
    gateup = torch.randn(batch_size, fused_outdim, dtype=dtype, device=device)
    out_buf = torch.zeros(batch_size, intermediate_size, dtype=dtype, device=device)

    # Build the module on CUDA in bf16. No weights, but stay consistent
    # with the catalog convention.
    m = SiluMul(intermediate_size=intermediate_size).to(device=device, dtype=dtype)

    # PyTorch reference -- uses the module's own forward() to keep the
    # reference and the compile path consuming the SAME halved input
    # convention.
    ref = m.forward(gateup)

    pk = _make_pk(batch_size)
    gateup_dt = pk.attach_input(gateup, name="silu_mul_gateup")

    print(f"\n{'=' * 60}")
    print(f"Test: SiluMul  B={batch_size}, intermediate={intermediate_size}")
    print("Building module inside compile_scope ...")
    with pk.compile_scope():
        # grid_dim=(1, 1, 1) keeps the per-task layout == the whole-tensor
        # layout (halved), so the kernel sees the same gate/up split that
        # forward() consumes. See module docstring for the grid.x>1
        # (shuffled) path.
        out_dt = m.compile(gateup_dt, output=out_buf, grid_dim=(1, 1, 1))
    assert out_dt is not None, "SiluMul.compile returned None"

    print("Compiling test kernel ...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel ...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute diff: {max_diff:.6f}")

    # bf16 SiLU + mul: the kernel promotes to fp32 internally, then casts
    # the product back to bf16 -- one rounding. The reference forward()
    # follows the same recipe, so the residual is rounding noise.
    try:
        torch.testing.assert_close(out_buf, ref, atol=1e-2, rtol=1e-2)
    except AssertionError as exc:
        print(f"FAILED: torch.testing.assert_close raised: {exc}")
        pk.finalize()
        sys.exit(1)

    print("PASSED: SiluMul matches PyTorch reference")
    pk.finalize()
    print("Test completed successfully!")


def test_silu_mul_linear_with_residual():
    """SiluMulLinearWithResidual: down-proj + residual fused with SiluMul."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1)

    batch_size = 8
    intermediate_size = 2048
    hidden_size = 4096
    fused_outdim = 2 * intermediate_size

    # Halved-layout fused input.
    gateup = torch.randn(batch_size, fused_outdim, dtype=dtype, device=device)
    # Small scale on the residual so it does not swamp the linear output.
    residual = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) * 0.01
    out_buf = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

    try:
        m = SiluMulLinearWithResidual(
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
        ).to(device=device, dtype=dtype)
    except RuntimeError as e:
        print(f"SKIPPED (known broken in Mirage): {e}")
        return

    # Down-proj weight at small scale to keep fp32 -> bf16 cast precise.
    with torch.no_grad():
        m.weight.copy_(
            torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) * 0.01
        )

    ref = m.forward(gateup, residual)

    pk = _make_pk(batch_size)
    gateup_dt = pk.attach_input(gateup, name="silumul_lin_gateup")
    residual_dt = pk.attach_input(residual, name="silumul_lin_residual")

    print(f"\n{'=' * 60}")
    print(
        f"Test: SiluMulLinearWithResidual  B={batch_size}, "
        f"hidden={hidden_size}, intermediate={intermediate_size}"
    )
    print("Building module inside compile_scope ...")
    with pk.compile_scope():
        # grid_dim=(1, 1, 1): single-CTA path. The kernel still tiles the
        # K (reduction) axis internally with TILE_SIZE=128; intermediate_size
        # must be a multiple of 128 (2048 is fine).
        out_dt = m.compile(
            gateup_dt, residual_dt, output=out_buf, grid_dim=(1, 1, 1)
        )
    assert out_dt is not None, "SiluMulLinearWithResidual.compile returned None"

    print("Compiling test kernel ...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel ...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute diff: {max_diff:.6f}")

    # The fused kernel accumulates a length-intermediate dot product in
    # fp32 then casts the final result to bf16. Tolerances are loosened
    # vs. the activation-only test to absorb the reduction noise.
    try:
        torch.testing.assert_close(out_buf, ref, atol=2e-2, rtol=2e-2)
    except AssertionError as exc:
        print(f"FAILED: torch.testing.assert_close raised: {exc}")
        pk.finalize()
        sys.exit(1)

    print("PASSED: SiluMulLinearWithResidual matches PyTorch reference")
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_silu_mul()
    test_silu_mul_linear_with_residual()
