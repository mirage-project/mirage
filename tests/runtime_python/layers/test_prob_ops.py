"""Test the ``layers.mtp.prob_ops`` catalog modules
(ProbScatter, ProbExtract) via PersistentKernel test_mode.

Numerical test for both halves: scatter writes prob[b, 0] into
buffer[b, step_counter[b]], extract reads buffer[b, offset[b]+1 : ...].
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.mtp.prob_ops import ProbScatter, ProbExtract


def test_prob_scatter_testmode():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 2
    max_positions = 8

    prob = torch.tensor(
        [[0.31], [0.42]], dtype=torch.float32, device=device,
    )
    step_counter = torch.tensor(
        [3, 5], dtype=torch.int32, device=device,
    )
    buffer_in = torch.zeros(
        batch_size, max_positions, dtype=torch.float32, device=device,
    )
    buffer_ref = buffer_in.clone()

    m = ProbScatter(max_positions=max_positions, prefix="ps_")
    m.forward(prob, step_counter, buffer_ref)

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

    prob_dt = pk.attach_input(prob, name="prob")
    step_dt = pk.attach_input(step_counter, name="step")
    buf_dt = pk.attach_input(buffer_in, name="buf")

    with pk.compile_scope():
        _ = m.compile(prob_dt, step_dt, buf_dt)

    print("Compiling ProbScatter...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    print(f"buffer:     {buffer_in.tolist()}")
    print(f"buffer_ref: {buffer_ref.tolist()}")

    try:
        torch.testing.assert_close(
            buffer_in, buffer_ref, atol=1e-6, rtol=1e-6
        )
        print("PASSED: ProbScatter compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: ProbScatter\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


def test_prob_extract_testmode():
    device = "cuda"
    torch.manual_seed(1)

    batch_size = 2
    max_positions = 8
    num_extract = 3

    # Pre-populate buffer with row-distinct sequential values.
    buffer_in = torch.arange(
        batch_size * max_positions, dtype=torch.float32, device=device,
    ).reshape(batch_size, max_positions) * 0.01
    offset = torch.tensor([1, 2], dtype=torch.int32, device=device)
    out = torch.zeros(batch_size, num_extract, dtype=torch.float32, device=device)

    m = ProbExtract(
        max_positions=max_positions, num_extract=num_extract, prefix="pe_",
    )
    ref = m.forward(buffer_in, offset)

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

    buf_dt = pk.attach_input(buffer_in, name="buf2")
    off_dt = pk.attach_input(offset, name="off")
    out_dt = pk.attach_input(out, name="out")

    with pk.compile_scope():
        _ = m.compile(buf_dt, off_dt, out_dt)

    print("Compiling ProbExtract...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    print(f"out: {out.tolist()}")
    print(f"ref: {ref.tolist()}")

    try:
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
        print("PASSED: ProbExtract compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: ProbExtract\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_prob_scatter_testmode()
    test_prob_extract_testmode()
