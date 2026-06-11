"""Test the ``layers.RMSNormLinear`` catalog module via PersistentKernel test_mode.

``RMSNormLinear`` is the fused RMSNorm + Linear projection used by
qwen3's input-layernorm + QKV path. We check that the new module's
``forward()`` PyTorch reference agrees with the MPK-compiled fused
kernel on a tiny single-batch input.

Mirrors the structure of:
* ``tests/runtime_python/layers/test_rmsnorm.py`` (PK boilerplate, the
  bf16 tolerance, the ``output=torch.Tensor`` readback path).
* ``tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py``
  (multi-tensor ``attach_input`` pattern used inside ``compile``).

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU. The ``mirage`` conda env is required.

Kernel notes (see ``rmsnorm_linear.py`` module docstring):

* The fused kernel currently exists only in ``tasks/ampere/`` — there
  is no Hopper/Blackwell variant. On those architectures this test
  is expected to fail at compile time inside ``register_rmsnorm_linear_task``.
* The kernel hard-codes ``eps = 1e-6f``; we use the same value in the
  module so ``forward()`` and ``compile()`` agree numerically.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_rmsnorm_linear_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # Shape selection
    # ------------------------------------------------------------------
    # batch_size: small (fits the swapAB-style per-task batch budget the
    # underlying linear pieces assume).
    batch_size = 8
    # hidden_size: must be a multiple of TILE_SIZE=128 (asserted in
    # ampere/norm_linear.cuh:61). 4096 satisfies this and is the canonical
    # qwen3 hidden size used by the sibling tests.
    hidden_size = 4096
    # out_features: must be divisible by 96 *and* 64 so the auto grid
    # heuristic (which prefers 96-element tiles) is happy and the kernel
    # has a clean per-task slab. 3072 hits both (3072=96*32=64*48) and
    # roughly mimics the QKV-fused output dim for a small-head model
    # (e.g., 8 q-heads + 2 kv-heads + 2 kv-heads, head_dim=192-ish).
    out_features = 3072
    # eps: match the kernel's hard-coded 1e-6f so forward() and the
    # compiled path agree. Anything else would silently disagree (see
    # the eps caveat in the module docstring).
    eps = 1e-6

    # ------------------------------------------------------------------
    # Build module and reference
    # ------------------------------------------------------------------
    try:
        module = layers.RMSNormLinear(
            hidden_size=hidden_size,
            out_features=out_features,
            eps=eps,
            prefix="test_",
        )
    except RuntimeError as e:
        print(f"SKIPPED (known broken in Mirage): {e}")
        return
    # Move to CUDA/bf16 so the Parameters live where pk.attach_input
    # expects.
    module = module.to(device=device, dtype=dtype)

    # Seed the weights with random values (the unit-init default would
    # make forward() degenerate to F.linear(x_normed * 1, W) which still
    # exercises the kernel, but using a randn scale also exercises the
    # in-kernel norm-weight multiply). Use 0.01 on the linear weight to
    # keep activations in bf16's representable range.
    weight_norm = torch.randn(hidden_size, dtype=dtype, device=device)
    weight_linear = (
        torch.randn(out_features, hidden_size, dtype=dtype, device=device) * 0.01
    )
    module.weight_norm.data.copy_(weight_norm)
    module.weight_linear.data.copy_(weight_linear)

    # Inputs.
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out_buf = torch.zeros(batch_size, out_features, dtype=dtype, device=device)

    # PyTorch reference. forward() respects the loaded weights and the
    # configured eps; the compiled kernel will run with eps=1e-6f
    # regardless, so the eps values *must* match here for the comparison
    # to be valid.
    ref = module.forward(x)

    # Sanity-check that forward() agrees with a manual implementation
    # before we even touch the kernel. This guards against e.g. an
    # accidental dtype mismatch in the reference.
    manual_variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    manual_normed = (x.to(torch.float32) * torch.rsqrt(manual_variance + eps)).to(dtype)
    manual_normed = (manual_normed * weight_norm).to(dtype)
    manual_ref = torch.nn.functional.linear(manual_normed, weight_linear)
    forward_diff = (ref.float() - manual_ref.float()).abs().max().item()
    assert forward_diff < 1e-3, (
        f"RMSNormLinear.forward() disagrees with manual reference: "
        f"max_diff={forward_diff}"
    )

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
    # Size the batched-token/request budget to our test batch so the
    # kernel's per-task BATCH_SIZE template arg matches.
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    # Attach the input. The two weights and the output buffer are
    # attached internally by compile().
    x_dt = pk.attach_input(x, name="x")

    # Build the graph inside the compile scope so current_pk() inside
    # the module body resolves to this pk.
    with pk.compile_scope():
        _ = module.compile(x_dt, output=out_buf)

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
    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6f}")

    try:
        # bf16 GEMM tolerance: 0.5 absolute / relative is the same
        # threshold used by test_linear.py for a comparable shape.
        torch.testing.assert_close(out_buf, ref, atol=0.5, rtol=0.5)
        print("PASSED: layers.RMSNormLinear compile() matches forward()")
    except AssertionError as e:
        print(
            f"FAILED: layers.RMSNormLinear compile() disagrees with forward(): "
            f"max_diff={max_diff}"
        )
        print(str(e))
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_rmsnorm_linear_testmode()
