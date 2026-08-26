"""DSV3-shaped test_mode sweep for ``linear_layer`` (BF16, sm100).

Two real DSV3 uses of linear_layer:
  1. lm_head:         N=129280, K=7168, grid=(505,1,1), block=(128,1,1)
     (non-vocab-parallel; N is NOT TP-sharded in the default build)
  2. router fallback: N=256,    K=7168, grid=(32,1,1),  block=(128,1,1)
     (_BF16_GATE_SPLITK_ENABLED=True uses splitk, but the fallback linear_layer
      path exists and is what runs when splitk is disabled.  We test the
      linear_layer shape here so the kernel is exercised at DSV3 N/K values.)

Grid formulas mirror the builder:
  grid_for_rmsnorm_linear_layer(129280) = 129280//256 = 505  (size/96>400 path)
  grid_for_rmsnorm_linear_layer(256):
    256/96 = 2.67 (not > 400); 256%96 != 0; 256%64 == 0 → 64
  router_grid = min(64, 256//8) = min(64, 32) = 32

Matrix (union-of-axes for TP×bs, but N is NOT TP-sharded → bs-only sweep):
  bs ∈ {1, 2, 4, 8, 16}  for BOTH shapes.

Run:
    python tests/runtime_python/blackwell/sm100_linear/test_dsv3_linear_testmode.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import linear_ref

HIDDEN = 7168
VOCAB  = 129280   # DSV3 padded vocab (already 129280-aligned)
NUM_EXPERTS = 256  # router gate output dim


def _grid_for_rmsnorm_linear(size: int) -> int:
    """Mirrors builder's grid_for_rmsnorm_linear_layer."""
    if size / 96 > 400:
        assert size % 256 == 0
        return size // 256
    if size % 96 == 0:
        return size // 96
    if size % 64 == 0:
        return size // 64
    raise ValueError(f"Unsupported size {size}")


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
    """Run one linear_layer config.  Returns True on PASS."""
    device = "cuda"
    torch.manual_seed(42)

    inp    = (torch.randn(bs, K, dtype=torch.bfloat16, device=device) * 0.02).contiguous()
    weight = (torch.randn(N, K, dtype=torch.bfloat16, device=device) / (K ** 0.5)).contiguous()
    output = torch.zeros(bs, N, dtype=torch.bfloat16, device=device)

    ref = linear_ref(inp, weight)

    pk = _make_pk(bs)
    in_dt  = pk.attach_input(inp,    name="input")
    w_dt   = pk.attach_input(weight, name="weight")
    out_dt = pk.attach_input(output, name="output")

    pk.linear_layer(
        input=in_dt,
        weight=w_dt,
        output=out_dt,
        grid_dim=(grid_x, 1, 1),
        block_dim=(128, 1, 1),
    )

    out_dir = os.path.join("/tmp/mpk_test_dsv3_linear", label)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Compiling {label} ...", flush=True)
    pk.compile(output_dir=out_dir)
    print(f"  Running   {label} ...", flush=True)
    pk()
    torch.cuda.synchronize()
    pk.finalize()

    err = (output.float() - ref.float()).abs().max().item()
    ok  = err < 0.02  # bf16 atol=rtol=1e-2  (|diff| <= atol + rtol*|ref|)
    # Use torch.testing for proper rtol/atol semantics
    try:
        torch.testing.assert_close(output, ref, atol=1e-2, rtol=1e-2)
        ok = True
    except AssertionError:
        ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  {status} {label}  max_diff={err:.6f}", flush=True)
    return ok


def test_dsv3_linear_lm_head():
    """lm_head: N=129280, K=7168 — bs sweep {1,2,4,8,16}."""
    N, K = VOCAB, HIDDEN
    grid_x = _grid_for_rmsnorm_linear(N)  # 505
    print(f"\n{'='*64}")
    print(f"DSV3 linear_layer — lm_head  N={N}, K={K}, grid_x={grid_x}")

    results = {}
    for bs in (1, 2, 4, 8, 16):
        label = f"lm_head_bs{bs}"
        print(f"\n--- bs={bs} ---")
        results[bs] = _run_case(bs, N, K, grid_x, label)

    failed = [bs for bs, ok in results.items() if not ok]
    if failed:
        raise AssertionError(f"lm_head FAILED for bs={failed}")
    print("\nPASSED: all lm_head configs")


def test_dsv3_linear_router_fallback():
    """Router gate fallback linear: N=256, K=7168 — bs sweep {1,2,4,8,16}."""
    N, K = NUM_EXPERTS, HIDDEN
    # router_grid = min(grid_for_rmsnorm_linear_layer(256), 256//8) = min(64, 32) = 32
    grid_lrl = _grid_for_rmsnorm_linear(N)   # 64  (but capped by builder to 32)
    grid_x = min(grid_lrl, N // 8)            # 32
    print(f"\n{'='*64}")
    print(f"DSV3 linear_layer — router fallback  N={N}, K={K}, grid_x={grid_x}")

    results = {}
    for bs in (1, 2, 4, 8, 16):
        label = f"router_fb_bs{bs}"
        print(f"\n--- bs={bs} ---")
        results[bs] = _run_case(bs, N, K, grid_x, label)

    failed = [bs for bs, ok in results.items() if not ok]
    if failed:
        raise AssertionError(f"router fallback FAILED for bs={failed}")
    print("\nPASSED: all router-fallback configs")


if __name__ == "__main__":
    all_ok = True
    try:
        test_dsv3_linear_lm_head()
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_ok = False

    try:
        test_dsv3_linear_router_fallback()
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_ok = False

    if not all_ok:
        sys.exit(1)
    print("\nAll DSV3 linear_layer tests PASSED.")
