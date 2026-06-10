"""
Test ``PersistentKernel.prob_extract_layer`` end-to-end through the full MPK
compilation pipeline (test_mode).

The prob_extract layer reads ``num_extract`` consecutive entries from the
per-request probability buffer, starting at offset+1:

    output[b, i] = buffer[b, offset[b] + 1 + i]   for i in range(num_extract)

In test_mode the persistent runtime forces step=0, so offset=0 and the
kernel reads buffer[:, 1 : num_extract+1].

Reference:
    output_ref[b, i] = buffer[b, 1 + i]

Layer call (from persistent_kernel.py):
    pk.prob_extract_layer(
        buffer=buffer_dt,     # [batch, max_positions] float32
        offset=offset_dt,     # [batch]               int32
        output=output_dt,     # [batch, num_extract]  float32
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        max_positions=max_positions,
        num_extract=num_extract,
    )

Sweep: bs ∈ {1, 2, 4, 8, 16}, num_extract=4, max_positions=64.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \\
        tests/runtime_python/blackwell/sm100_prob_buffer/test_prob_extract_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

MAX_POSITIONS = 64
NUM_EXTRACT = 4
BS_LIST = [1, 2, 4, 8, 16]


def _run_case(bs: int):
    device = "cuda"
    torch.manual_seed(42 + bs)

    # Fill buffer with recognisable random values so we can verify the
    # correct slice is extracted.
    buffer = torch.rand(bs, MAX_POSITIONS, dtype=torch.float32, device=device)
    # offset values are ignored in test_mode (step=0 forces offset=0).
    offset = torch.zeros(bs, dtype=torch.int32, device=device)
    output = torch.zeros(bs, NUM_EXTRACT, dtype=torch.float32, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    params["meta_tensors"] = {
        "prompt_lengths": torch.ones(bs, dtype=torch.int32, device=device),
    }
    pk = PersistentKernel(**params)

    buf_dt = pk.attach_input(buffer, name="buffer")
    off_dt = pk.attach_input(offset, name="offset")
    out_dt = pk.attach_input(output, name="output")

    pk.prob_extract_layer(
        buffer=buf_dt,
        offset=off_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        max_positions=MAX_POSITIONS,
        num_extract=NUM_EXTRACT,
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print(f"Test: prob_extract_layer  bs={bs}  "
          f"num_extract={NUM_EXTRACT}  max_positions={MAX_POSITIONS}")
    print("Compiling...")
    pk.compile(output_dir=folder_path)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # Reference: in test_mode offset=0, so output[:, i] = buffer[:, 1+i].
    output_ref = buffer[:, 1:NUM_EXTRACT + 1].clone()

    torch.testing.assert_close(output, output_ref, rtol=1e-3, atol=1e-3)

    max_diff = (output.float() - output_ref.float()).abs().max().item()
    print(f"  max abs diff: {max_diff:.6f}  PASS")

    pk.finalize()
    return max_diff


def test_prob_extract_testmode():
    for bs in BS_LIST:
        _run_case(bs)


if __name__ == "__main__":
    results = []
    for bs in BS_LIST:
        diff = _run_case(bs)
        results.append((bs, diff))

    print(f"\n{'='*60}")
    print(f"PROB_EXTRACT SUMMARY  num_extract={NUM_EXTRACT}  "
          f"max_positions={MAX_POSITIONS}")
    for bs, diff in results:
        print(f"  bs={bs:2d}: max_diff={diff:.6f}  PASS")
    print(f"ALL PASS ({len(results)}/{len(results)})")
    sys.exit(0)
