"""Test-mode coverage for ``PersistentKernel.rmsnorm_layer`` at DSV3 shapes.

Runs the SM100 RMSNorm task end-to-end through the MPK compile+run pipeline
and compares against the folder's ``rmsnorm_ref`` (eps=1e-6 to match the
DeepSeek-V3 builder; the kernel hard-codes 1e-6f).

DSV3 facts: HIDDEN=7168 (NOT TP-sharded), dtype bf16 -> this is a bs-only
sweep. The grid mirrors the builder's ``_rmsnorm_grid`` at the default
``MPK_DSV3_RMSNORM_ROWS_PER_TASK=1`` (one row per CTA -> grid.x = bs) and
``block_dim=(128,1,1)``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from pytorch_reference import rmsnorm_ref  # noqa: E402

HIDDEN = 7168
RMS_NORM_EPS = 1e-6
ATOL = 1e-2
RTOL = 1e-2
BS_SWEEP = [1, 2, 4, 8, 16]


def _run_case(bs: int) -> float:
    """Run one rmsnorm config; returns max abs diff vs reference."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1234 + bs)

    x = torch.randn(bs, HIDDEN, dtype=dtype, device=device)
    w = torch.randn(HIDDEN, dtype=dtype, device=device)
    out = torch.zeros(bs, HIDDEN, dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    block_dim = (128, 1, 1)
    pk.rmsnorm_layer(
        input=x_dt,
        weight=w_dt,
        output=out_dt,
        grid_dim=(bs, 1, 1),
        block_dim=block_dim,
    )

    pk.compile(output_dir=THIS_DIR)
    pk()
    torch.cuda.synchronize()

    ref = rmsnorm_ref(x, w, eps=RMS_NORM_EPS)
    out_f = out.float()
    ref_f = ref.float()
    abs_diff = (out_f - ref_f).abs()
    max_diff = abs_diff.max().item()
    # Combined atol+rtol pass criterion (same as torch.testing.assert_close):
    # an element passes if |out-ref| <= atol + rtol*|ref|. This is the
    # decision-log "atol/rtol ~= 1e-2" semantics; a pure-atol check would
    # spuriously flag isolated 1-ULP bf16 rounding differences caused by the
    # kernel's tree-reduction order vs torch's .mean() (these are the two
    # nearest bf16 values to the true result, both correct).
    bad = (abs_diff > (ATOL + RTOL * ref_f.abs()))
    num_bad = int(bad.sum().item())
    pk.finalize()
    return max_diff, num_bad


def test_rmsnorm_testmode():
    assert torch.cuda.is_available(), "CUDA required"
    failures = []
    for bs in BS_SWEEP:
        max_diff, num_bad = _run_case(bs)
        ok = num_bad == 0
        status = "PASS" if ok else "FAIL"
        print(f"[rmsnorm] bs={bs:2d} hidden={HIDDEN} eps={RMS_NORM_EPS} "
              f"max_abs_diff={max_diff:.6e} atol={ATOL} rtol={RTOL} "
              f"num_elems_over_tol={num_bad} -> {status}")
        if not ok:
            failures.append((bs, max_diff, num_bad))

    if failures:
        print(f"FAILED rmsnorm configs: {failures}")
        sys.exit(1)
    print("PASSED: rmsnorm_layer test_mode correct across bs sweep "
          f"{BS_SWEEP}")


if __name__ == "__main__":
    test_rmsnorm_testmode()
