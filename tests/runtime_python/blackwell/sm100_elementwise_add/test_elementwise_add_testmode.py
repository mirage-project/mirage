"""Test: elementwise_add_layer via PersistentKernel test_mode (DSV3 shapes).

Op:  out = a + b,  a,b in (bs, HIDDEN=7168), bf16.
Builder call: grid_dim=(max_num_batched_tokens, 1, 1), block_dim=(128,1,1).

DSV3 sweep: bs ∈ {1,2,4,8,16}.

Run:
    python tests/runtime_python/blackwell/sm100_elementwise_add/test_elementwise_add_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import elementwise_add_ref

HIDDEN = 7168
BS_SWEEP = [1, 2, 4, 8, 16]


def _run_case(bs: int):
    device = "cuda"
    torch.manual_seed(42 + bs)

    a = torch.randn(bs, HIDDEN, dtype=torch.bfloat16, device=device)
    b = torch.randn(bs, HIDDEN, dtype=torch.bfloat16, device=device)
    out = torch.zeros(bs, HIDDEN, dtype=torch.bfloat16, device=device)

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
    a_dt = pk.attach_input(a, name="a")
    b_dt = pk.attach_input(b, name="b")
    out_dt = pk.attach_input(out, name="out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    # Mirror builder: grid_dim=(max_num_batched_tokens, 1, 1), block_dim=(128,1,1).
    # Builder uses (128,1,1) explicitly; we follow that.
    pk.elementwise_add_layer(
        input_a=a_dt,
        input_b=b_dt,
        output=out_dt,
        grid_dim=(bs, 1, 1),
        block_dim=(128, 1, 1),
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    ref = elementwise_add_ref(a, b)
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  bs={bs:2d}  max_diff={max_diff:.6f}", end="")

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("  PASS")
    pk.finalize()


def test_elementwise_add_testmode():
    print(f"\n{'='*60}")
    print(f"elementwise_add_layer  HIDDEN={HIDDEN}  bs sweep={BS_SWEEP}")
    for bs in BS_SWEEP:
        _run_case(bs)
    print("ALL PASS")


if __name__ == "__main__":
    test_elementwise_add_testmode()
