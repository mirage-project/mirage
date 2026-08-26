"""Test: tensor_init_layer via PersistentKernel test_mode (DSV3 shapes).

Op:  zero-fill target (bs, HIDDEN=7168), bf16.
DSV3 use: zero-fills new_moe_meta before moe_permute. Builder uses
    grid_dim=(1,1,1), block_dim=(128,1,1),
    dummy_input_map=(-1,-1,-1), target_input_map=(-1,-1,-1).

DSV3 sweep: bs ∈ {1,2,4,8,16}.

Run:
    python tests/runtime_python/blackwell/sm100_tensor_init/test_tensor_init_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import tensor_init_ref

HIDDEN = 7168
BS_SWEEP = [1, 2, 4, 8, 16]


def _run_case(bs: int):
    device = "cuda"
    torch.manual_seed(bs)

    # Pre-fill with non-zero values to detect the zero-fill.
    linear_output = torch.randn(bs, HIDDEN, dtype=torch.bfloat16, device=device)
    pre_kernel_snapshot = linear_output.clone()
    # dummy is a dependency-edge placeholder; kernel never reads/writes it.
    linear_input = torch.randn(bs, HIDDEN, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs

    pk = PersistentKernel(**params)
    in_dt = pk.attach_input(linear_input, name="linear_input")
    out_dt = pk.attach_input(linear_output, name="linear_output")

    # Mirror DSV3 builder: grid=(1,1,1), maps both (-1,-1,-1).
    pk.tensor_init_layer(
        target=out_dt,
        dummy=in_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        dummy_input_map=(-1, -1, -1),
        target_input_map=(-1, -1, -1),
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    ref = tensor_init_ref(pre_kernel_snapshot, init_val=0.0)
    max_diff = (linear_output.float() - ref.float()).abs().max().item()
    print(f"  bs={bs:2d}  max_diff={max_diff:.6f}", end="")

    torch.testing.assert_close(linear_output, ref, rtol=0.0, atol=0.0)
    print("  PASS")
    pk.finalize()


def test_tensor_init_testmode():
    print(f"\n{'='*60}")
    print(f"tensor_init_layer  HIDDEN={HIDDEN}  bs sweep={BS_SWEEP}")
    for bs in BS_SWEEP:
        _run_case(bs)
    print("ALL PASS")


if __name__ == "__main__":
    test_tensor_init_testmode()
