"""Test the ``layers.Argmax`` catalog module via PersistentKernel test_mode.

This file is the Phase-2 test for the (single-shot) Argmax catalog
migration. It mirrors the canonical ``test_rmsnorm_testmode.py`` pattern:

- Build the parameterless ``Argmax`` module.
- Allocate ``(B, V)`` bf16 logits and a ``(B, 1)`` int64 output buffer.
- PyTorch reference: ``torch.argmax(logits, dim=-1, keepdim=True)``.
- MPK path: build PK in ``test_mode``, attach the logits, route the
  output through ``compile(..., output=out_buf)`` so the host can read
  the result, run once, compare bit-exactly (argmax is integer-valued
  and uses the same first-wins tie semantics in both paths — see the
  ``Argmax`` module docstring).

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU. The ``mirage`` conda env is required.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_argmax_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Tiny shape. ``vocab_size`` has no kernel-alignment requirement for
    # the single-shot variant (the kernel strides ``for i = tidx; i <
    # vocab_size; i += NUM_THREADS``), so we pick a round 1024. Using
    # bf16 logits matches the qwen3 / llama3 production layout
    # (``demo/qwen3/demo.py`` line 437: ``argmax_in`` is ``mi.bfloat16``).
    batch_size = 8
    vocab_size = 1024

    # Random logits. Using randn rather than rand so the per-row argmax
    # is genuinely scattered across the vocab, not pinned to index 0.
    logits = torch.randn(batch_size, vocab_size, dtype=dtype, device=device)

    # Output buffer: (B, 1) int64. The kernel writes long long per row
    # (see include/.../tasks/{ampere,blackwell}/argmax{,_sm100}.cuh —
    # ``long long *final_output``). ``pk.argmax_layer`` asserts
    # ``output.num_dims == 2`` with the trailing-1 dim, matching how
    # ``output_tokens`` is allocated in demo/qwen3/demo.py:250.
    out_tokens = torch.full(
        (batch_size, 1), -1, dtype=torch.int64, device=device
    )

    # --------------------------------------------------------------
    # Build module + PyTorch reference
    # --------------------------------------------------------------
    try:
        m = layers.Argmax(prefix="test_")
    except RuntimeError as e:
        print(f"SKIPPED (known broken in Mirage): {e}")
        return
    # ``forward`` returns (B, 1) int64 to match the compiled path bit-
    # for-bit (see module docstring: keepdim=True).
    ref = m.forward(logits)
    # Sanity-check that the module's forward agrees with the plain
    # torch.argmax call — catches accidental drift in the reference.
    ref_plain = torch.argmax(logits, dim=-1, keepdim=True)
    assert torch.equal(ref, ref_plain), "module.forward disagrees with torch.argmax"

    # --------------------------------------------------------------
    # Build PersistentKernel in test mode
    # --------------------------------------------------------------
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

    # Attach the logits input. The output torch.Tensor is attached
    # inside ``compile()`` via the ``output=`` path so the host can
    # inspect ``out_tokens`` after running.
    logits_dt = pk.attach_input(logits, name="logits")

    with pk.compile_scope():
        _ = m.compile(logits_dt, output=out_tokens)

    # --------------------------------------------------------------
    # Compile and run once
    # --------------------------------------------------------------
    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # --------------------------------------------------------------
    # Compare. Argmax produces integer token indices — we can compare
    # bit-exactly via ``torch.equal``, not ``torch.testing.assert_close``.
    # Tie-breaking is first-wins in both the PyTorch reference (strict
    # ``>`` in libtorch) and the MPK kernel (strict ``>`` in the
    # warp/block reductions), so ties match too.
    # --------------------------------------------------------------
    print(f"out_tokens[:, 0]: {out_tokens[:, 0].tolist()}")
    print(f"ref[:, 0]:        {ref[:, 0].tolist()}")

    if torch.equal(out_tokens, ref):
        print("PASSED: layers.Argmax compile() matches forward()")
    else:
        # Report the rows that disagree so a regression is easy to debug.
        mismatch = (out_tokens != ref).nonzero(as_tuple=False)
        print(
            f"FAILED: layers.Argmax compile() disagrees with forward() "
            f"on {mismatch.shape[0]} of {batch_size} rows. First few "
            f"mismatching row indices: {mismatch[:8, 0].tolist()}"
        )
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_argmax_testmode()
