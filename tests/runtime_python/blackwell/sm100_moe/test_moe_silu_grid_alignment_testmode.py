"""Regression guard for the 2026-06-13 routed-MoE NULL bug (bug class A:
grid / dispatch mis-map).

The NEW-MoE fused SiLU-mul (``moe_silu_mul_layer`` with ``meta`` + ``bm_padding``)
row-partitions its permuted-expert input across grid.x AND derives the logical
expert ``my_expert = bid.x / ctas_per_expert`` from blockIdx, assuming each CTA
owns exactly ``rows_per_cta = m_total // grid.x`` rows. The builder originally
launched it with ``grid.x = min(num_workers, m_total)`` (= 136 at TP8-EP2),
which makes ``rows_per_cta = 16384 // 136 = 120`` — NOT a divisor of
``bm_padding = 128``. Each CTA then reads the WRONG w13_out rows (mostly the
inactive zero-padding between experts) → ``silu_out = 0`` → a zero W2 GEMM input
→ the entire routed-MoE contribution silently nulls out (every layer, every
token). The fix pins ``grid.x = m_total // bm_padding`` (= 128) so
``rows_per_cta == bm_padding`` and each CTA maps to exactly one expert block.

Why the existing silu test (test_moe_silu_mul_testmode.py) missed it: that test
only exercises the OLD-MoE 3D path ``grid=(bs, TOPK, 1)`` with ``meta=None`` —
the kernel is per-element correct there, so the grid↔logical-index contract was
never asserted. This test instantiates the *production* NEW-MoE signature at the
real TP8-EP2 shape (E_local=128, m_total=16384, bm_padding=128) and pins BOTH
directions of the alignment invariant:

  1. the correct grid (m_total // bm_padding) builds the task graph cleanly;
  2. a clean multiple (ctas_per_expert = 2) also builds;
  3. the historical buggy grid (min(num_workers, m_total) = 136) RAISES the
     grid-alignment AssertionError (so the null-MoE can never silently return).

This is a GRAPH-BUILD test: it constructs the megakernel task graph but never
calls compile()/pk() — no megakernel launch, negligible GPU memory, no D-state
risk. It needs a CUDA context only because PersistentKernel.__init__ probes the
device + allocates tiny meta tensors. Skips cleanly if no GPU is visible.

Run:
    CUDA_VISIBLE_DEVICES=0 \
      python tests/runtime_python/blackwell/sm100_moe/test_moe_silu_grid_alignment_testmode.py
"""

import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# Production DeepSeek-V3 TP8-EP2 routed-MoE dispatch shape.
E_LOCAL = 128          # num_local_experts per rank at EP2
BM_PADDING = 128       # padded rows per expert block
M_TOTAL = E_LOCAL * BM_PADDING   # = 16384 permuted-layout rows
I_PER_RANK = 256       # routed intermediate / routed_tp (only dim(0) matters here)
NUM_WORKERS = 136      # TP8-EP2 worker count → the historical buggy grid.x


def _build_silu_graph(grid_x: int, *, e_local: int = E_LOCAL,
                      bm_padding: int = BM_PADDING):
    """Build (NOT run) a NEW-MoE moe_silu_mul task at grid.x=grid_x.

    Returns None on success; re-raises whatever moe_silu_mul_layer raises
    (the grid-alignment AssertionError for a misaligned grid). No compile(),
    no megakernel launch — pure CPU-side graph construction guarded by a
    CUDA context.
    """
    device = "cuda"
    dtype = torch.bfloat16
    m_total = e_local * bm_padding

    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["world_size"] = 1
    params["mpi_rank"] = 0
    params["num_workers"] = NUM_WORKERS
    params["num_local_schedulers"] = 4
    # Token-batch params are independent of the permuted-expert row count.
    params["max_num_batched_tokens"] = 1
    params["max_num_batched_requests"] = 1
    pk = PersistentKernel(**params)

    # input = gate||up (m_total, 2*I); output = (m_total, I). Only dim(0) drives
    # the alignment assert, but use real attachable (row-major) tensors so the
    # success path exercises the actual graph registration.
    input_act = torch.zeros(m_total, I_PER_RANK * 2, dtype=dtype, device=device)
    output = torch.zeros(m_total, I_PER_RANK, dtype=dtype, device=device)
    # meta row 0 = active_expert_mask, row 1 = actual_count_per_expert;
    # meta.dim(1) == E_local is the active_mask_offset the kernel reads.
    meta = torch.zeros(2, e_local, dtype=torch.int32, device=device)

    input_dt = pk.attach_input(input_act, name="silu_in")
    output_dt = pk.attach_input(output, name="silu_out")
    meta_dt = pk.attach_input(meta, name="silu_meta")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    try:
        pk.moe_silu_mul_layer(
            input=input_dt,
            output=output_dt,
            grid_dim=(grid_x, 1, 1),
            block_dim=block_dim,
            meta=meta_dt,
            bm_padding=bm_padding,
        )
    finally:
        # Release the graph + tiny GPU allocs whether or not the assert fired.
        pk.finalize()


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device visible (graph build needs a CUDA context).")
        return

    print(f"{'='*64}\nmoe_silu_mul grid-alignment guard "
          f"(E_local={E_LOCAL} bm_padding={BM_PADDING} m_total={M_TOTAL})\n{'='*64}")

    # (1) Correct production grid: grid.x = m_total // bm_padding = 128.
    good_grid = M_TOTAL // BM_PADDING
    _build_silu_graph(good_grid)
    print(f"  [PASS] correct grid.x={good_grid} (rows_per_cta=bm_padding) builds")

    # (2) Clean multiple: ctas_per_expert = 2 (rows_per_cta = 64 | bm_padding).
    mult_grid = 2 * (M_TOTAL // BM_PADDING)
    _build_silu_graph(mult_grid)
    print(f"  [PASS] clean-multiple grid.x={mult_grid} (ctas_per_expert=2) builds")

    # (3) Historical buggy grid: grid.x = min(num_workers, m_total) = 136.
    bad_grid = min(NUM_WORKERS, M_TOTAL)
    assert bad_grid == 136 and M_TOTAL % bad_grid != 0, (
        "test premise drifted: 136 must not divide m_total")
    raised = False
    try:
        _build_silu_graph(bad_grid)
    except AssertionError as e:
        raised = True
        assert "misaligns expert blocks" in str(e), (
            f"alignment assert fired but with an unexpected message: {e}")
        print(f"  [PASS] buggy grid.x={bad_grid} (rows_per_cta="
              f"{M_TOTAL // bad_grid}) raises: {str(e).splitlines()[0]}")
    assert raised, (
        f"REGRESSION: moe_silu_mul accepted the misaligned grid.x={bad_grid} "
        f"(rows_per_cta={M_TOTAL // bad_grid} does not divide bm_padding="
        f"{BM_PADDING}) — the routed-MoE NULL bug would be reintroducible.")

    # (4) Generality: a second expert count exercises the same invariant.
    e2 = 64
    m2 = e2 * BM_PADDING
    _build_silu_graph(m2 // BM_PADDING, e_local=e2)
    raised2 = False
    try:
        _build_silu_graph(min(NUM_WORKERS, m2), e_local=e2)  # 136 ∤ 8192
    except AssertionError:
        raised2 = True
    assert raised2, (
        f"REGRESSION: E_local={e2} m_total={m2} accepted misaligned "
        f"grid.x={min(NUM_WORKERS, m2)}.")
    print(f"  [PASS] generality E_local={e2}: correct grid builds, "
          f"misaligned grid.x={min(NUM_WORKERS, m2)} raises")

    print("\nALL PASS: NEW-MoE silu grid↔expert-block alignment invariant locked.")


def test_moe_silu_grid_alignment_testmode():
    main()


if __name__ == "__main__":
    main()
