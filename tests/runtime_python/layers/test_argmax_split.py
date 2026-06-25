"""Tests for the split-reduce argmax catalog modules.

Two tests, both via PersistentKernel test_mode:

1. ``test_argmax_partial`` — exercises :class:`ArgmaxPartial` alone.
   Builds the module, computes the (partial_values, partial_indices)
   reference via ``module.forward(x)``, then runs the same input through
   PK and compares both outputs bit-exactly.

2. ``test_argmax_partial_plus_reduce`` — the chained pipeline that
   qwen3's lm_head actually uses. Compares the final ``(B, 1)`` int64
   token-ids to ``torch.argmax(x, dim=-1, keepdim=True)``.

Both tests mirror the pattern from
``tests/runtime_python/layers/test_argmax.py`` (the single-shot
counterpart). The chained test is the most important — it's the actual
production pattern.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU. The ``mirage`` conda env is required.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.context import compile_scope, current_pk
from mirage.mpk.layers._base import MPKModule
from mirage.mpk.layers.argmax.argmax_partial import ArgmaxPartial
from mirage.mpk.layers.argmax.argmax_reduce import ArgmaxReduce
from mirage.mpk.persistent_kernel import PersistentKernel


def _make_pk(batch_size: int) -> PersistentKernel:
    """Build a tiny test-mode PK sized for ``batch_size`` rows."""
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


def test_argmax_partial():
    """Standalone partial test — both outputs match the reference."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Small vocab so the test is fast. num_partial_tasks=8 divides 1024.
    batch_size = 4
    vocab_size = 1024
    num_partial_tasks = 8
    chunk_size = vocab_size // num_partial_tasks  # 128

    # randn so the per-chunk maxima land at different positions, not
    # always at index 0 — exercises the chunk-local indexing.
    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=device)

    # Output buffers the test driver will read back. partial_values is
    # bf16 (matches kernel T), partial_indices is int64 (kernel writes
    # long long).
    out_values = torch.zeros(
        batch_size, num_partial_tasks, dtype=torch.bfloat16, device=device
    )
    out_indices = torch.full(
        (batch_size, num_partial_tasks), -1, dtype=torch.int64, device=device
    )

    # --------------------------------------------------------------
    # Build module + PyTorch reference
    # --------------------------------------------------------------
    m = ArgmaxPartial(
        vocab_size=vocab_size,
        num_partial_tasks=num_partial_tasks,
        prefix="test_",
    )
    ref_values, ref_indices = m.forward(logits)
    assert ref_values.shape == (batch_size, num_partial_tasks)
    assert ref_indices.shape == (batch_size, num_partial_tasks)
    assert ref_values.dtype == torch.bfloat16
    assert ref_indices.dtype == torch.int64
    # Sanity: indices are chunk-local positions.
    assert int(ref_indices.max().item()) < chunk_size, (
        f"ref_indices should be chunk-local (< {chunk_size}); "
        f"got max {int(ref_indices.max().item())}"
    )

    # --------------------------------------------------------------
    # Build PersistentKernel in test mode
    # --------------------------------------------------------------
    pk = _make_pk(batch_size)
    logits_dt = pk.attach_input(logits, name="logits")

    with pk.compile_scope():
        _ = m.compile(
            logits_dt,
            partial_values=out_values,
            partial_indices=out_indices,
        )

    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # --------------------------------------------------------------
    # Compare. Both outputs should match bit-exactly:
    # - values: bf16 max-reduction has no floating-point reassociation
    #   (it's just element selection), so torch.equal works.
    # - indices: integer chunk-local positions; ties pick the lowest
    #   index in both paths.
    # --------------------------------------------------------------
    print(f"out_values row 0: {out_values[0].tolist()}")
    print(f"ref_values row 0: {ref_values[0].tolist()}")
    print(f"out_indices row 0: {out_indices[0].tolist()}")
    print(f"ref_indices row 0: {ref_indices[0].tolist()}")

    failed = False
    if not torch.equal(out_values, ref_values):
        n = int((out_values != ref_values).sum().item())
        print(f"FAILED: partial_values mismatch on {n} entries")
        failed = True
    if not torch.equal(out_indices, ref_indices):
        n = int((out_indices != ref_indices).sum().item())
        print(f"FAILED: partial_indices mismatch on {n} entries")
        failed = True

    if failed:
        pk.finalize()
        sys.exit(1)

    print("PASSED: ArgmaxPartial compile() matches forward()")
    pk.finalize()


def test_argmax_partial_plus_reduce():
    """Chained partial + reduce — the actual qwen3 lm_head pattern.

    This is the most important test: it builds both modules, wires the
    partial outputs into the reduce inputs inside one compile scope,
    and compares the final token-ids against ``torch.argmax(x, dim=-1)``.
    Argmax is exact integer arithmetic so we use ``torch.equal``.
    """
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 4
    vocab_size = 1024
    num_partial_tasks = 8
    chunk_size = vocab_size // num_partial_tasks  # 128

    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=device)

    # Final output: (B, 1) int64 — matches the existing Argmax module
    # and demo/qwen3/demo.py's ``output_tokens`` layout.
    out_tokens = torch.full(
        (batch_size, 1), -1, dtype=torch.int64, device=device
    )

    # --------------------------------------------------------------
    # PyTorch reference: a plain torch.argmax — the chained MPK pipeline
    # must reproduce this bit-exactly.
    # --------------------------------------------------------------
    ref = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int64)
    assert ref.shape == (batch_size, 1)

    # --------------------------------------------------------------
    # Build modules
    # --------------------------------------------------------------
    ap = ArgmaxPartial(
        vocab_size=vocab_size,
        num_partial_tasks=num_partial_tasks,
        prefix="test_chain_partial_",
    )
    rd = ArgmaxReduce(num_partial_tasks=num_partial_tasks, prefix="test_chain_reduce_")
    # Inform the reduce module of the chunk size so its .forward()
    # reconstructs global indices correctly when used as a pure-PyTorch
    # reference (the compiled path reads it from PK state).
    rd._chunk_size = ap.chunk_size

    # Sanity: the PyTorch chained reference matches plain torch.argmax.
    pv_ref, pi_ref = ap.forward(logits)
    chained_ref = rd.forward(pv_ref, pi_ref)
    assert torch.equal(chained_ref, ref), (
        "Chained PyTorch reference (ArgmaxPartial -> ArgmaxReduce) "
        "must equal torch.argmax — sanity check before touching MPK."
    )

    # --------------------------------------------------------------
    # Build PersistentKernel in test mode and compile the chained
    # pipeline.
    # --------------------------------------------------------------
    pk = _make_pk(batch_size)
    logits_dt = pk.attach_input(logits, name="logits")

    with pk.compile_scope():
        # Partials are intermediate cuda_tensors — let the modules
        # auto-allocate. Only the final output is host-readable.
        values_dt, indices_dt = ap.compile(logits_dt)
        _ = rd.compile(values_dt, indices_dt, output=out_tokens)

    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_tokens[:, 0]: {out_tokens[:, 0].tolist()}")
    print(f"ref[:, 0]:        {ref[:, 0].tolist()}")

    if torch.equal(out_tokens, ref):
        print(
            "PASSED: ArgmaxPartial -> ArgmaxReduce chained compile() "
            "matches torch.argmax"
        )
    else:
        mismatch = (out_tokens != ref).nonzero(as_tuple=False)
        print(
            f"FAILED: chained pipeline disagrees with torch.argmax on "
            f"{mismatch.shape[0]} of {batch_size} rows. First few "
            f"mismatching row indices: {mismatch[:8, 0].tolist()}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_argmax_partial()
    test_argmax_partial_plus_reduce()
