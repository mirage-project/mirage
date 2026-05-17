"""Test the ``layers.TransposeScale`` catalog module via PersistentKernel test_mode.

Numerical test: scale_in (M, K_PACKED) uint32 → scale_out (K_PACKED, M)
uint32 transpose. The kernel is a single-CTA copy that should produce
exactly the same bytes as scale_in.T.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.transpose_scale import TransposeScale


def test_transpose_scale_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes: M=128 (FP8 quantize block-aligned), K_PACKED=4
    # (i.e., 4 packed uint32 columns = 16 logical UE8M0 blocks = 2048 K).
    M = 128
    K_PACKED = 4
    batch_size = 1

    scale_in = torch.randint(
        0, 2**31, (M, K_PACKED), dtype=torch.uint32, device=device
    )
    scale_out = torch.zeros(K_PACKED, M, dtype=torch.uint32, device=device)

    m = TransposeScale(prefix="test_")
    ref = m.forward(scale_in)
    assert ref.shape == (K_PACKED, M)
    assert torch.equal(ref, scale_in.transpose(0, 1).contiguous())

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

    in_dt = pk.attach_input(scale_in, name="scale_in")

    with pk.compile_scope():
        _ = m.compile(in_dt, scale_out=scale_out)

    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"scale_out[0, :8]: {scale_out[0, :8]}")
    print(f"ref[0, :8]:       {ref[0, :8]}")

    if torch.equal(scale_out, ref):
        print("PASSED: TransposeScale compile() matches forward()")
    else:
        diff = (scale_out != ref).sum().item()
        print(f"FAILED: TransposeScale {diff} bytes mismatch")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_transpose_scale_testmode()
