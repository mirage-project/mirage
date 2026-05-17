"""Test the ``layers.mtp.token_scatter`` catalog modules
(MTPTokenScatter, MTPFloatScatter) via PersistentKernel test_mode.

Numerical test: scatter per-request int64 / float32 values into a chosen
column of a wider buffer.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.mtp.token_scatter import (
    MTPTokenScatter,
    MTPFloatScatter,
)


def _run_test_token_scatter(slot_idx):
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 2
    num_slots = 4

    src = torch.tensor(
        [[42], [101]], dtype=torch.int64, device=device,
    )
    dst = torch.zeros(batch_size, num_slots, dtype=torch.int64, device=device)
    dst_ref = dst.clone()

    m = MTPTokenScatter(
        batch_size=batch_size, num_slots=num_slots, prefix="tok_",
    )
    m.forward(src, dst_ref, slot_idx)

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

    src_dt = pk.attach_input(src, name="src")
    dst_dt = pk.attach_input(dst, name="dst")

    with pk.compile_scope():
        _ = m.compile(src_dt, dst_dt, slot_idx)

    print(f"Compiling MTPTokenScatter (slot={slot_idx})...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    print(f"dst:     {dst.tolist()}")
    print(f"dst_ref: {dst_ref.tolist()}")

    if torch.equal(dst, dst_ref):
        print(f"PASSED: MTPTokenScatter slot={slot_idx}")
    else:
        print(f"FAILED: MTPTokenScatter slot={slot_idx} dst != dst_ref")
        pk.finalize()
        sys.exit(1)
    pk.finalize()


def test_token_scatter():
    _run_test_token_scatter(slot_idx=0)


def test_float_scatter():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 2
    num_slots = 4
    slot_idx = 2

    src = torch.tensor(
        [[0.25], [0.75]], dtype=torch.float32, device=device,
    )
    dst = torch.zeros(batch_size, num_slots, dtype=torch.float32, device=device)
    dst_ref = dst.clone()

    m = MTPFloatScatter(
        batch_size=batch_size, num_slots=num_slots, prefix="flt_",
    )
    m.forward(src, dst_ref, slot_idx)

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

    src_dt = pk.attach_input(src, name="src_f")
    dst_dt = pk.attach_input(dst, name="dst_f")

    with pk.compile_scope():
        _ = m.compile(src_dt, dst_dt, slot_idx)

    print(f"Compiling MTPFloatScatter (slot={slot_idx})...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    print(f"dst:     {dst.tolist()}")
    print(f"dst_ref: {dst_ref.tolist()}")

    try:
        torch.testing.assert_close(dst, dst_ref, atol=1e-6, rtol=1e-6)
        print(f"PASSED: MTPFloatScatter slot={slot_idx}")
    except AssertionError as e:
        print(f"FAILED: MTPFloatScatter slot={slot_idx}\n{e}")
        pk.finalize()
        sys.exit(1)
    pk.finalize()


if __name__ == "__main__":
    test_token_scatter()
    test_float_scatter()
