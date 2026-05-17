"""Smoke test for layers.sampling.SamplingSM100.

Stochastic Gumbel-Max sampling. The forward() reference and the kernel
use different PRNG schemes (PyTorch RNG vs stateless hash seeded on
(seed, batch, vocab)), so a bit-exact compare is not possible.

We verify:
  1. The kernel produces token IDs in [0, vocab_size).
  2. No NaN/Inf in the output.

Output dtype: int32 (kernel writes int32; the catalog declares the
output as int32 to match — callers needing int64 should cast at use).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.sampling import SamplingSM100


def test_sampling_sm100_smoke():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 2
    vocab_size = 1024

    logits = torch.randn(
        batch_size, vocab_size, dtype=torch.bfloat16, device=device,
    )
    # The kernel writes int32 token ids and the catalog now declares
    # int32 output to match.
    out_tokens = torch.zeros(
        (batch_size, 1), dtype=torch.int32, device=device,
    )

    m = SamplingSM100(seed=42, prefix="samp_")
    # Sanity check that forward returns valid token ids in range.
    ref = m.forward(logits)
    assert ((ref >= 0) & (ref < vocab_size)).all()

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

    logits_dt = pk.attach_input(logits, name="logits_samp")

    with pk.compile_scope():
        _ = m.compile(logits_dt, output=out_tokens)

    print("Compiling SamplingSM100...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    tokens_int32 = out_tokens.long()
    print(f"out_tokens (int32): {out_tokens.flatten().tolist()}")
    if ((tokens_int32 >= 0) & (tokens_int32 < vocab_size)).all():
        print("PASSED: SamplingSM100 smoke (all tokens in [0, vocab))")
    else:
        print("FAILED: out_tokens has values outside [0, vocab)")
        pk.finalize()
        sys.exit(1)
    pk.finalize()


if __name__ == "__main__":
    test_sampling_sm100_smoke()
