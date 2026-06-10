"""
Test ``PersistentKernel.prob_scatter_layer`` end-to-end through the full MPK
compilation pipeline (test_mode).

The prob_scatter layer writes the current per-request probability into a
position-indexed buffer:

    buffer[b, step_counter[b]] = prob[b, 0]

In test_mode the persistent runtime forces step 0, so the kernel always
writes to position 0 regardless of the value passed in step_counter.

Reference:
    buffer_ref[b, 0] = prob[b, 0]  (all other positions remain zero)

Layer call (from persistent_kernel.py):
    pk.prob_scatter_layer(
        prob=prob_dt,          # [batch, 1] float32
        step_counter=step_dt,  # [batch]    int32
        buffer=buffer_dt,      # [batch, max_positions] float32
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
        max_positions=max_positions,
    )

Sweep: bs ∈ {1, 2, 4, 8, 16}, max_positions=64.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \\
        tests/runtime_python/blackwell/sm100_prob_buffer/test_prob_scatter_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

MAX_POSITIONS = 64
BS_LIST = [1, 2, 4, 8, 16]


def _run_case(bs: int):
    device = "cuda"
    torch.manual_seed(42 + bs)

    prob = torch.rand(bs, 1, dtype=torch.float32, device=device)
    # step_counter values are irrelevant in test_mode (runtime forces step=0),
    # but we pass plausible values anyway.
    step_counter = torch.zeros(bs, dtype=torch.int32, device=device)
    buffer = torch.zeros(bs, MAX_POSITIONS, dtype=torch.float32, device=device)

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

    prob_dt = pk.attach_input(prob, name="prob")
    step_dt = pk.attach_input(step_counter, name="step_counter")
    buf_dt = pk.attach_input(buffer, name="buffer")

    pk.prob_scatter_layer(
        prob=prob_dt,
        step_counter=step_dt,
        buffer=buf_dt,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
        max_positions=MAX_POSITIONS,
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print(f"Test: prob_scatter_layer  bs={bs}  max_positions={MAX_POSITIONS}")
    print("Compiling...")
    pk.compile(output_dir=folder_path)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # Reference: in test_mode step=0, so buffer[:, 0] = prob[:, 0].
    buffer_ref = torch.zeros(bs, MAX_POSITIONS, dtype=torch.float32, device=device)
    buffer_ref[:, 0] = prob[:, 0]

    torch.testing.assert_close(buffer, buffer_ref, rtol=1e-3, atol=1e-3)

    max_diff = (buffer.float() - buffer_ref.float()).abs().max().item()
    print(f"  max abs diff: {max_diff:.6f}  PASS")

    pk.finalize()
    return max_diff


def test_prob_scatter_testmode():
    for bs in BS_LIST:
        _run_case(bs)


if __name__ == "__main__":
    results = []
    for bs in BS_LIST:
        diff = _run_case(bs)
        results.append((bs, diff))

    print(f"\n{'='*60}")
    print(f"PROB_SCATTER SUMMARY  max_positions={MAX_POSITIONS}")
    for bs, diff in results:
        print(f"  bs={bs:2d}: max_diff={diff:.6f}  PASS")
    print(f"ALL PASS ({len(results)}/{len(results)})")
    sys.exit(0)
