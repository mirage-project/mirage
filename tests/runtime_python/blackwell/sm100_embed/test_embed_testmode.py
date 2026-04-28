"""
Test: embed_layer via PersistentKernel test_mode (SM100 / Blackwell).

The embed_layer dispatches to the same template kernel
(`kernel::embedding_kernel<bfloat16, ...>`) on Ampere/Hopper/Blackwell -- it
gathers `weight[input_ids[b]]` into `output[b]` for `b` in `[0, BATCH_SIZE)`.

We test the `input_source=1` path (use task input pointer directly) because
`input_source=0` would read from the runtime `tokens` meta-tensor which is
not populated in test mode.

Embedding lookup is exact (a memory copy of bf16 rows), so the output must
match the reference byte-for-byte.

Run:
    python tests/runtime_python/blackwell/sm100_embed/test_embed_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import embed_ref


def test_embed_testmode():
    device = "cuda"
    torch.manual_seed(42)

    # Tiny config for fast compilation.
    batch_size = 4
    vocab_size = 1024
    hidden_size = 256

    # Inputs: int64 token ids (kernel reinterprets input_ptrs[0] as int64*),
    # bf16 embedding table, bf16 output.
    input_ids = torch.randint(
        0, vocab_size, (batch_size, 1), dtype=torch.int64, device=device
    )
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)
    out = torch.zeros(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

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

    in_dt = pk.attach_input(input_ids, name="input_ids")
    w_dt = pk.attach_input(weight, name="weight")
    out_dt = pk.attach_input(out, name="out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    # The kernel iterates BATCH_SIZE internally, so a single CTA suffices.
    pk.embed_layer(
        input=in_dt,
        weight=w_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=block_dim,
        input_source=1,
    )

    print("Compiling test kernel...")
    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ref = embed_ref(input_ids, weight)
    print(f"out[:2, :8]:\n{out[:2, :8]}")
    print(f"ref[:2, :8]:\n{ref[:2, :8]}")

    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    # Embedding is a pure gather; no FP math => byte-exact.
    if torch.equal(out, ref):
        print("PASSED: embed_layer test_mode produces byte-exact output")
    else:
        print(f"FAILED: outputs differ (max diff {max_diff})")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_embed_testmode()
