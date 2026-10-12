"""elementwise_add_layer through the full MPK pipeline on Hopper.

On sm_90 the layer registers elementwise_add_hopper; this builds a test_mode
PersistentKernel with two adds (a vector-path shape split over several tasks
and a scalar-fallback shape), runs it once and checks the outputs bit-exactly
against the PyTorch reference.

    python tests/runtime_python/hopper/elementwise_add_hopper/test_elementwise_add_hopper_testmode.py
"""

import os
import sys
import tempfile

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import elementwise_add_reference  # noqa: E402

BATCH = 16
# (hidden, tasks): 16 rows split over `tasks` tasks along the batch dim
CASES = [(4096, 4), (1000, 2)]


def main():
    assert torch.cuda.get_device_capability() == (9, 0), "needs an sm_90 GPU"
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0,
                  world_size=1, max_num_batched_tokens=BATCH,
                  max_num_batched_requests=BATCH, max_num_pages=BATCH)
    pk = PersistentKernel(**params)

    tensors = []
    for n, (hidden, tasks) in enumerate(CASES):
        a = torch.randn(BATCH, hidden, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(BATCH, hidden, device="cuda", dtype=torch.bfloat16)
        out = torch.full_like(a, float("nan"))
        pk.elementwise_add_layer(
            input_a=pk.attach_input(a, name=f"a{n}"),
            input_b=pk.attach_input(b, name=f"b{n}"),
            output=pk.attach_input(out, name=f"out{n}"),
            grid_dim=(tasks, 1, 1), block_dim=(256, 1, 1))
        tensors.append((a, b, out))

    print("Compiling test kernel...")
    # Generated sources go to a temp dir; *.cu here is the tracked wrapper.
    pk.compile(output_dir=tempfile.mkdtemp(prefix="mpk_hopper_testmode_"))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for (hidden, tasks), (a, b, out) in zip(CASES, tensors):
        exact = torch.equal(out, elementwise_add_reference(a, b))
        ok &= exact
        print(f"[{'PASS' if exact else 'FAIL'}] hidden={hidden} tasks={tasks}")
    pk.finalize()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
