"""DSV3-shaped test_mode sweep for ``splitk_linear_layer`` (BF16, sm100).

Real DSV3 use — router gate (per MoE layer):
  input=(bs, 7168)   weight=(256, 7168)   output=(bs, 256)
  grid=(2, 64, 1)    block=(256, 1, 1)    accumulate=False

Grid derivation (mirrors builder._pick_bf16_splitk_factor on B200, 128 workers):
  n_tiles = 256 // 128 = 2
  k_align = 64  (BF16 splitk requirement)
  quotient = 7168 // 64 = 112
  best split_k = largest s such that quotient%s==0 and 2*s <= 128
               = 56 (2*56=112; 2*64=128) → wait: 112%64=48≠0; 112%56=0 and 2*56=112≤128 ✓
               so split_k = 56? Actually iterate:
               s=1→2, s=2→4, ..., s=56→112≤128 (112%56=0 ✓), s=64→128≤128 (112%64=48≠0 ✗)
               Actually let me be careful: 112%56=0 ✓, 2*56=112≤128 ✓
                                          112%112=0 ✓, 2*112=224>128 ✗ → stop
               So best_s=56? But we also try s between 56 and 112:
               s=56: 2*56=112≤128 ✓ → best_s=56; s=57...: 112%57≠0, ...
               But the builder returned split_k=64 based on the perfetto comment
               ("was 2 CTAs" → "128-CTA / ~1 wave", 2*64=128).
               112%64=48 ≠ 0, so 64 is NOT valid per the code.
               Actually re-read: quotient=112, iterate s=1..112:
                 need quotient%s==0 (so K//s is still k_align-multiple) AND n*s<=workers
               s=1: 112%1=0, 2*1=2≤128 → best=1
               s=2: 112%2=0, 2*2=4≤128 → best=2
               ... s=4,7,8,14,16,28,56 are divisors of 112 ≤ 64 (2*s≤128)
               s=56: 112%56=0, 2*56=112≤128 → best=56
               s=112: 2*112=224>128 → break
               So split_k=56, grid=(2,56,1)
               (The perfetto comment's "64" was approximate; code returns 56.)

NOTE: accumulate=False HANGS for bs < 16 (compile-time BATCH_SIZE < MMA_N=16 bug,
documented in builder comment lines 1526-1536 and test_splitk_linear_bf16_accfalse_testmode.py).
bs=16 is the minimum safe batch for accumulate=False.

accumulate=True does NOT have this bug (no tensor_init prepended).
We sweep acc=True at bs∈{1,2,4,8,16} and acc=False at bs=16 only.
bs<16 with acc=False is marked XFAIL_HANG.

Union-of-axes matrix (GEMM-like):
  acc=True:  bs∈{1,2,4,8,16}  (5 configs)
  acc=False: bs=16             (1 config; bs<16 xfail/skip)

Run:
    python tests/runtime_python/blackwell/sm100_splitk_linear_bf16/test_dsv3_splitk_linear_bf16_testmode.py
"""

import os
import sys
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from pytorch_reference import splitk_linear_ref

HIDDEN      = 7168
NUM_EXPERTS = 256


def _compute_split_k(n_tiles: int, K: int, k_align: int, num_workers: int) -> int:
    """Mirror of builder._pick_splitk_factor (returns 1 as fallback)."""
    if K % k_align != 0:
        return 1
    if n_tiles > num_workers:
        return 1
    quotient = K // k_align
    best_s = 1
    for s in range(1, quotient + 1):
        if quotient % s != 0:
            continue
        if n_tiles * s > num_workers:
            break
        best_s = s
    return best_s


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


def _run_case(bs: int, N: int, K: int, grid_x: int, grid_y: int,
              accumulate: bool, label: str) -> bool:
    """Run one splitk_linear_layer config.  Returns True on PASS."""
    device = "cuda"
    torch.manual_seed(42)

    inp    = (torch.randn(bs, K, dtype=torch.bfloat16, device=device) * 0.02).contiguous()
    weight = (torch.randn(N, K, dtype=torch.bfloat16, device=device) / (K ** 0.5)).contiguous()

    if accumulate:
        pre_output = (torch.randn(bs, N, dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    else:
        pre_output = torch.randn(bs, N, dtype=torch.bfloat16, device=device).contiguous()

    output              = pre_output.clone()
    pre_output_snapshot = pre_output.clone()

    ref = splitk_linear_ref(inp, weight,
                            pre_output=pre_output_snapshot,
                            accumulate=accumulate)

    pk    = _make_pk(bs)
    in_dt = pk.attach_input(inp,    name="input")
    w_dt  = pk.attach_input(weight, name="weight")
    ot_dt = pk.attach_input(output, name="output")

    pk.splitk_linear_layer(
        input=in_dt,
        weight=w_dt,
        output=ot_dt,
        grid_dim=(grid_x, grid_y, 1),
        block_dim=(256, 1, 1),
        accumulate=accumulate,
    )

    out_dir = os.path.join("/tmp/mpk_test_dsv3_splitk_linear_bf16", label)
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
    print(f"  {status} {label}  max_diff={err:.6f}  acc={accumulate}", flush=True)
    return ok


def test_dsv3_splitk_linear_acc_true():
    """Router gate shape, accumulate=True: bs ∈ {1,2,4,8,16}."""
    N, K = NUM_EXPERTS, HIDDEN
    n_tiles = N // 128   # = 2
    nw, _   = mirage.get_configurations_from_gpu(0)
    grid_y  = _compute_split_k(n_tiles, K, k_align=64, num_workers=nw)
    grid_x  = n_tiles

    print(f"\n{'='*64}")
    print(f"DSV3 splitk_linear acc=True  N={N}, K={K}, grid=({grid_x},{grid_y},1)")

    results = {}
    for bs in (1, 2, 4, 8, 16):
        label = f"gate_acc_true_bs{bs}"
        print(f"\n--- bs={bs} ---")
        results[bs] = _run_case(bs, N, K, grid_x, grid_y,
                                accumulate=True, label=label)

    failed = [bs for bs, ok in results.items() if not ok]
    if failed:
        raise AssertionError(f"splitk acc=True FAILED for bs={failed}")
    print("\nPASSED: all acc=True configs")


def test_dsv3_splitk_linear_acc_false_bs16():
    """Router gate shape, accumulate=False: bs=16 only (bs<16 hangs — known bug)."""
    N, K   = NUM_EXPERTS, HIDDEN
    n_tiles = N // 128
    nw, _  = mirage.get_configurations_from_gpu(0)
    grid_y = _compute_split_k(n_tiles, K, k_align=64, num_workers=nw)
    grid_x = n_tiles

    print(f"\n{'='*64}")
    print(f"DSV3 splitk_linear acc=False  N={N}, K={K}, grid=({grid_x},{grid_y},1)")
    print("  NOTE: bs<16 with acc=False HANGS (MMA_N=16 bug) — tested bs=16 only")

    bs = 16
    label = f"gate_acc_false_bs{bs}"
    print(f"\n--- bs={bs} ---")
    ok = _run_case(bs, N, K, grid_x, grid_y, accumulate=False, label=label)
    if not ok:
        raise AssertionError(f"splitk acc=False bs=16 FAILED")
    print("\nPASSED: acc=False bs=16")


def print_xfail_note():
    """Document the known-hang for acc=False bs<16."""
    print("\n--- XFAIL note: acc=False bs<16 ---")
    print("  bs ∈ {1,2,4,8} with accumulate=False would HANG the MPK runtime.")
    print("  Cause: compile-time BATCH_SIZE < MMA_N=16 triggers TMA over-read past")
    print("  in-bounds gmem rows.  Documented in builder line 1526-1536 and")
    print("  test_splitk_linear_bf16_accfalse_testmode.py.  NOT run here.")
    for bs in (1, 2, 4, 8):
        print(f"  XFAIL_HANG gate_acc_false_bs{bs}  (acc=False, bs={bs} < 16)")


if __name__ == "__main__":
    all_ok = True

    try:
        test_dsv3_splitk_linear_acc_true()
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_ok = False

    try:
        test_dsv3_splitk_linear_acc_false_bs16()
    except AssertionError as e:
        print(f"FAILED: {e}")
        all_ok = False

    print_xfail_note()

    if not all_ok:
        sys.exit(1)
    print("\nAll DSV3 splitk_linear_layer tests PASSED.")
