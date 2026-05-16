"""Test the ``layers.Embed`` catalog module via PersistentKernel test_mode.

Embed is a vocab-table lookup: ``out[i] = weight[input_tokens[i]]``.
This test follows the canonical ``test_rmsnorm_testmode.py`` /
``test_identity.py`` pattern:

- Build a tiny ``Embed`` module with a small vocab (128) and small
  hidden dim (256), copy a random ``weight`` into ``module.weight``.
- Reference: ``F.embedding(input_tokens, weight)``.
- MPK path: build PK in test_mode, attach the input tokens, attach the
  output torch tensor via ``compile(..., output=out_buf)``, run once,
  assert close against the reference.

We exercise ``input_source=1`` (the qwen3 builder default): the kernel
reads token IDs from ``task_desc->input_ptrs[0]`` (i.e. the DTensor we
explicitly hand in), not from the persistent runtime's rolling token
buffer.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU.
"""

import os
import sys

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_embed_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Tiny shape. Pick a small power-of-two vocab so the lookup is fast
    # to compile. ``embedding_dim`` must be reasonable for the kernel's
    # threaded copy (256 div by block_dim.x = 128/256 cleanly).
    num_embeddings = 128
    embedding_dim = 256
    batch_size = 4

    # Random embedding table and random tokens within range.
    weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
    input_tokens = torch.randint(
        low=0, high=num_embeddings, size=(batch_size,), dtype=torch.int64, device=device
    )
    out_buf = torch.zeros(batch_size, embedding_dim, dtype=dtype, device=device)

    # PyTorch reference (sanity-check via the module's own ``forward``
    # before we touch MPK).
    module = layers.Embed(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        prefix="test_",
    )
    module = module.to(device=device, dtype=dtype)
    with torch.no_grad():
        module.weight.data.copy_(weight)

    ref = module.forward(input_tokens)
    # Cross-check against the plain functional form too.
    ref_plain = F.embedding(input_tokens, weight)
    assert torch.equal(ref, ref_plain), "module.forward disagrees with F.embedding"

    # Build PersistentKernel in test mode.
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

    # Attach the token-ID buffer as the layer's input. With
    # ``input_source=1`` the kernel reads token IDs from this DTensor
    # rather than from ``runtime_config.tokens``.
    input_dt = pk.attach_input(input_tokens, name="input_tokens")

    with pk.compile_scope():
        # ``output=out_buf`` (torch.Tensor) routes through
        # pk.attach_input so we can inspect ``out_buf`` after running.
        _ = module.compile(
            input_dt,
            input_source=1,
            output=out_buf,
        )

    # Compile and run once.
    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[:2, :8]:\n{out_buf[:2, :8]}")
    print(f"ref[:2, :8]:\n{ref[:2, :8]}")

    # Embedding is a byte-for-byte copy of bfloat16 rows — tolerance is
    # nominally zero, but allow a tiny epsilon to be friendly to any
    # bfloat16 reinterpretation paths the codegen takes.
    try:
        torch.testing.assert_close(out_buf, ref, atol=0.0, rtol=0.0)
        print("PASSED: embed test_mode produces exact lookup")
    except AssertionError as e:
        max_diff = (out_buf.float() - ref.float()).abs().max().item()
        print(f"FAILED: embed lookup disagrees, max diff = {max_diff}")
        print(str(e))
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_embed_testmode()
