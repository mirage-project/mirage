"""
Test: moe_mul_sum_add_layer residual handling under tensor parallelism.

Under TP the MoE output is row-parallel and followed by an allreduce, so the
correct way to combine the residual depends on what the caller passed:

  add_residual_once=True  (default) - the residual is the same on every rank,
      e.g. a skip connection. Exactly one rank may add it, otherwise the
      allreduce counts it world_size times. This is the PR #722 fix.

  add_residual_once=False - the residual is a per-rank partial that the
      allreduce is meant to sum, e.g. DeepSeek V3's row-parallel shared-expert
      output. Every rank must add its own, otherwise the other ranks'
      contributions are silently dropped.

Two test functions:
  1. test_rank_guard_flags:  which residual-enable flag reaches the task for a
     matrix of (world_size, mpi_rank, add_residual_once). Registration only, so
     it runs on a single GPU without MPI/NVSHMEM.
  2. test_mul_sum_add_numerics: world_size=1 end-to-end compile+run against a
     PyTorch reference. moe_mul_sum_add_layer previously had no test_mode
     coverage at all.

Run:
    python tests/runtime_python/test_mode/test_moe_residual_rank_guard_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


BATCH_SIZE = 1
HIDDEN_SIZE = 4096
NUM_TOPK = 8


def _make_pk(world_size=1, mpi_rank=0):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=mpi_rank,
        world_size=world_size,
        max_num_batched_tokens=BATCH_SIZE,
        max_num_batched_requests=BATCH_SIZE,
    )
    return PersistentKernel(**params)


def _attach_moe_tensors(pk, device="cuda"):
    """Attach the tensors moe_mul_sum_add_layer expects, returning DTensors."""
    x = torch.randn(BATCH_SIZE, NUM_TOPK, HIDDEN_SIZE,
                    dtype=torch.bfloat16, device=device) * 0.1
    w = torch.rand(BATCH_SIZE, NUM_TOPK, dtype=torch.float32, device=device)
    res = torch.randn(BATCH_SIZE, HIDDEN_SIZE,
                      dtype=torch.bfloat16, device=device) * 0.1
    out = torch.zeros(BATCH_SIZE, HIDDEN_SIZE,
                      dtype=torch.bfloat16, device=device)
    return (
        pk.attach_input(x, name="moe_in"),
        pk.attach_input(w, name="moe_topk_weight"),
        pk.attach_input(res, name="moe_residual"),
        pk.attach_input(out, name="moe_out"),
        (x, w, res, out),
    )


def _registered_residual_flag(world_size, mpi_rank, add_residual_once):
    """Register the layer and report the residual-enable flag it passed down.

    The SM100 kernel skips the residual add when params[0] == 0, so this flag
    is exactly what decides whether this rank contributes its residual.
    """
    pk = _make_pk(world_size=world_size, mpi_rank=mpi_rank)
    x_dt, w_dt, res_dt, out_dt, _ = _attach_moe_tensors(pk)

    captured = {}
    original = pk.kn_graph.register_task

    def spy(bgraph, task_type, params=None):
        captured["task_type"] = task_type
        captured["params"] = params
        return original(bgraph, task_type, params)

    pk.kn_graph.register_task = spy
    pk.moe_mul_sum_add_layer(
        input=x_dt, weight=w_dt, residual=res_dt, output=out_dt,
        grid_dim=(BATCH_SIZE, HIDDEN_SIZE // 256, 1),
        block_dim=(256, 1, 1),
        add_residual_once=add_residual_once,
    )

    assert captured["task_type"] == "moe_mul_sum_add_sm100"
    return captured["params"][0]


def test_rank_guard_flags():
    print(f"\n{'='*70}")
    print("Test: residual-enable flag by (world_size, rank, add_residual_once)")

    # (world_size, mpi_rank, add_residual_once, expected_flag, why)
    cases = [
        (1, 0, True,  1, "single GPU always adds the residual"),
        (1, 0, False, 1, "single GPU always adds the residual"),
        (2, 0, True,  1, "replicated residual: rank 0 adds it"),
        (2, 1, True,  0, "replicated residual: other ranks must not (PR #722)"),
        (2, 0, False, 1, "partial residual: every rank adds its own"),
        (2, 1, False, 1, "partial residual: every rank adds its own"),
        (4, 3, True,  0, "replicated residual: only rank 0 adds it"),
        (4, 3, False, 1, "partial residual: every rank adds its own"),
    ]

    ok = True
    for world_size, rank, once, expected, why in cases:
        got = _registered_residual_flag(world_size, rank, once)
        status = "ok " if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"  [{status}] world_size={world_size} rank={rank} "
              f"add_residual_once={str(once):5} -> flag={got} "
              f"(expected {expected}: {why})")

    if not ok:
        print("\nFAILED: residual-enable flag did not match expectations")
        sys.exit(1)
    print("\nPASSED: residual rank guard applies only to replicated residuals")


def test_mul_sum_add_numerics():
    print(f"\n{'='*70}")
    print("Test: moe_mul_sum_add_layer numerics (world_size=1, compile+run)")
    print(f"  B={BATCH_SIZE}, H={HIDDEN_SIZE}, topk={NUM_TOPK}")

    torch.manual_seed(42)
    pk = _make_pk(world_size=1, mpi_rank=0)
    x_dt, w_dt, res_dt, out_dt, (x, w, res, out) = _attach_moe_tensors(pk)

    pk.moe_mul_sum_add_layer(
        input=x_dt, weight=w_dt, residual=res_dt, output=out_dt,
        grid_dim=(BATCH_SIZE, HIDDEN_SIZE // 256, 1),
        block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running...")
    pk()
    torch.cuda.synchronize()

    ref = (x.float() * w.unsqueeze(-1)).sum(dim=1) + res.float()
    ref = ref.to(torch.bfloat16)

    print(f"\nOutput[0, :8]:    {out[0, :8]}")
    print(f"Reference[0, :8]: {ref[0, :8]}")

    max_abs = (out.float() - ref.float()).abs().max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    print(f"\nMax absolute diff: {max_abs:.6f}")
    print(f"Max relative err:  {max_rel:.6f}")

    if max_rel >= 0.05:
        print(f"\nFAILED: max relative error {max_rel:.4f} exceeds 5% tolerance")
        pk.finalize()
        sys.exit(1)

    # The residual must appear exactly once: subtracting it must leave the
    # weighted expert sum on its own.
    weighted_only = (x.float() * w.unsqueeze(-1)).sum(dim=1)
    residual_delta = (out.float() - weighted_only - res.float()).abs().max().item()
    if residual_delta > 0.5:
        print(f"\nFAILED: residual not added exactly once (delta {residual_delta:.4f})")
        pk.finalize()
        sys.exit(1)

    print("\nPASSED: moe_mul_sum_add matches reference, residual added once")
    pk.finalize()


if __name__ == "__main__":
    test_rank_guard_flags()
    test_mul_sum_add_numerics()
    print(f"\n{'='*70}")
    print("All MoE residual rank-guard tests PASSED!")
