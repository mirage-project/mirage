"""DSV3-shaped test_mode sweep for ``linear_with_residual_layer`` (BF16, sm100).

The real DSV3 use is the BF16 fallback down-projection in ``_fp8_linear``
when a test fixture loads BF16 weights (no FP8 scale).  Shapes (TP=1):
  - N = HIDDEN = 7168  (down-proj output)
  - K = INTERMEDIATE = 18432  (dense MLP intermediate at TP=1)
  - grid_dim = (grid_for_rmsnorm_linear_layer(7168), 1, 1) = (64, 1, 1)
  - block_dim = (128, 1, 1)

There is also a secondary non-MTP use in the MTP eh_proj path, but MTP is
out of scope per DSV3_TESTMODE_DECISIONS.md.

N (hidden) is NOT TP-sharded in the BF16-fallback path (K = intermediate//tp
would be sharded, but in the BF16 fallback the whole K passes through on each
rank, i.e. TP=1 only). → bs-only sweep, 5 configs.

Matrix:  bs ∈ {1, 2, 4, 8, 16}

Run:
    python tests/runtime_python/blackwell/sm100_linear/test_dsv3_linear_with_residual_testmode.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import linear_with_residual_ref

HIDDEN        = 7168
INTERMEDIATE  = 18432   # dense MLP intermediate (TP=1 full size)


def _make_pk(bs: int) -> PersistentKernel:
    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=nw,
        num_local_schedulers=ns,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=bs,
        max_num_batched_requests=bs,
    )
    return PersistentKernel(**params)


def _run_case(bs: int, N: int, K: int, grid_x: int, label: str) -> bool:
    """Run one linear_with_residual_layer config.  Returns True on PASS."""
    device = "cuda"
    torch.manual_seed(42)

    inp      = (torch.randn(bs, K, dtype=torch.bfloat16, device=device) * 0.02).contiguous()
    weight   = (torch.randn(N, K, dtype=torch.bfloat16, device=device) / (K ** 0.5)).contiguous()
    residual = (torch.randn(bs, N, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    output   = torch.zeros(bs, N, dtype=torch.bfloat16, device=device)

    ref = linear_with_residual_ref(inp, weight, residual)

    # linear_with_residual_layer may check qo_indptr_buffer to decide whether
    # the residual addition fires.  Provide a stub that marks all bs rows valid.
    qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    qo_indptr[bs] = bs

    pk = _make_pk(bs)
    pk_params_extra = {"qo_indptr_buffer": qo_indptr}
    # Rebuild pk with the meta_tensors override
    nw, ns = mirage.get_configurations_from_gpu(0)
    params2 = PersistentKernel.get_default_init_parameters()
    params2.update(
        test_mode=True,
        num_workers=nw,
        num_local_schedulers=ns,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=bs,
        max_num_batched_requests=bs,
        meta_tensors={"qo_indptr_buffer": qo_indptr},
    )
    pk = PersistentKernel(**params2)

    in_dt  = pk.attach_input(inp,      name="input")
    w_dt   = pk.attach_input(weight,   name="weight")
    res_dt = pk.attach_input(residual, name="residual")
    out_dt = pk.attach_input(output,   name="output")

    pk.linear_with_residual_layer(
        input=in_dt,
        weight=w_dt,
        residual=res_dt,
        output=out_dt,
        grid_dim=(grid_x, 1, 1),
        block_dim=(128, 1, 1),
    )

    out_dir = os.path.join("/tmp/mpk_test_dsv3_linear_with_residual", label)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Compiling {label} ...", flush=True)
    pk.compile(output_dir=out_dir)
    print(f"  Running   {label} ...", flush=True)
    pk()
    torch.cuda.synchronize()
    pk.finalize()

    err = (output.float() - ref.float()).abs().max().item()
    try:
        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)
        ok = True
    except AssertionError:
        ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  {status} {label}  max_diff={err:.6f}", flush=True)
    return ok


def test_dsv3_linear_with_residual():
    """Down-proj BF16 fallback: N=7168, K=18432, bs ∈ {1,2,4,8,16}."""
    N, K = HIDDEN, INTERMEDIATE
    # grid_for_rmsnorm_linear_layer(7168): 7168/96=74.67 (not >400); 7168%96!=0; 7168%64==0 → 64
    grid_x = 64
    print(f"\n{'='*64}")
    print(f"DSV3 linear_with_residual_layer  N={N}, K={K}, grid_x={grid_x}")

    results = {}
    for bs in (1, 2, 4, 8, 16):
        label = f"down_proj_bs{bs}"
        print(f"\n--- bs={bs} ---")
        results[bs] = _run_case(bs, N, K, grid_x, label)

    failed = [bs for bs, ok in results.items() if not ok]
    if failed:
        raise AssertionError(f"linear_with_residual FAILED for bs={failed}")
    print("\nPASSED: all DSV3 linear_with_residual configs")


if __name__ == "__main__":
    try:
        test_dsv3_linear_with_residual()
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    print("\nAll DSV3 linear_with_residual_layer tests PASSED.")
