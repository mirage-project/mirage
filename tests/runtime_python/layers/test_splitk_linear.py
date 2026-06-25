"""Test layers.linear.splitk_linear.SplitKLinear via PersistentKernel test_mode.

The split-K kernel reduce-adds the GEMM partials into ``output``. We
use ``accumulate=True`` (output is the residual stream; the test wires
``output`` to a torch.Tensor we pre-populate). A separate known issue
(see tests/runtime_python/blackwell/sm100_splitk_linear_bf16/) deadlocks
``accumulate=False`` — we avoid that branch.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.linear.splitk_linear import SplitKLinear


def test_splitk_linear_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 1
    in_features = 4096
    out_features = 4096

    x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
    weight = torch.randn(
        out_features, in_features, dtype=dtype, device=device,
    ) * 0.01
    # accumulate=True: output is the residual; pre-populate with random
    # values; the kernel adds the GEMM on top.
    output_initial = torch.randn(
        batch_size, out_features, dtype=dtype, device=device,
    )
    output = output_initial.clone()

    m = SplitKLinear(
        in_features=in_features, out_features=out_features,
        accumulate=True, prefix="sk_",
    )
    m = m.to(device=device, dtype=dtype)
    m.weight.data.copy_(weight)

    # Reference: F.linear + residual.
    ref = (x.float() @ weight.float().t() + output_initial.float()).to(dtype)

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

    x_dt = pk.attach_input(x, name="x_sk")
    out_dt = pk.attach_input(output, name="out_sk")

    with pk.compile_scope():
        _ = m.compile(x_dt, out_dt)

    print("Compiling SplitKLinear (accumulate=True)...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    print(f"output[0, :8]: {output[0, :8]}")
    print(f"ref[0, :8]:    {ref[0, :8]}")
    max_diff = (output.float() - ref.float()).abs().max().item()
    print(f"max abs diff: {max_diff}")
    try:
        # bf16 GEMM tolerance — splitk noisy because of partial sum
        # ordering across CTAs.
        torch.testing.assert_close(output, ref, atol=0.5, rtol=0.5)
        print("PASSED: SplitKLinear compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: SplitKLinear\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_splitk_linear_testmode()
