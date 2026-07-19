"""Test: silu_mul_layer via PersistentKernel test_mode (DSV3 shapes).

Op:  out = silu(gate) * up,  where input has gate||up layout per task chunk.
     Input shape: (bs, 2*I), output shape: (bs, I).

DSV3 intermediate sizes (TP=1):
  - Dense MLP:        I=18432, num_tasks=48   (dense layers 0-2)
  - Shared MoE expert: I=2048,  num_tasks=32  (MoE shared expert path)

Grid/block follow the DSV3 builder (both use block_dim=(128,1,1)):
  - Dense:  silu_mul_grid = grid_for_rmsnorm_linear_layer(2*18432) // 2 = 48
  - Shared: silu_mul_grid = grid_for_rmsnorm_linear_layer(2*2048)  // 2 = 32

Tolerance: bf16 atol/rtol=1e-2.
DSV3 sweep: bs ∈ {1,2,4,8,16}.

Run:
    python tests/runtime_python/blackwell/sm100_silu_mul/test_silu_mul_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import silu_mul_ref

# DSV3 shapes (TP=1): (intermediate, num_tasks)
DSV3_CONFIGS = [
    ("dense_mlp",     18432, 48),  # layers 0-2 dense MLP
    ("shared_expert",  2048, 32),  # routed/shared MoE expert
]
BS_SWEEP = [1, 2, 4, 8, 16]


def _run_case(bs: int, intermediate: int, num_tasks: int, label: str):
    device = "cuda"
    torch.manual_seed(42 + bs + intermediate)

    fused_outdim = 2 * intermediate
    x = torch.randn(bs, fused_outdim, dtype=torch.bfloat16, device=device)
    out = torch.zeros(bs, intermediate, dtype=torch.bfloat16, device=device)

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
    x_dt = pk.attach_input(x, name="x")
    out_dt = pk.attach_input(out, name="out")

    # Mirror DSV3 builder: block_dim=(128,1,1).
    pk.silu_mul_layer(
        input=x_dt,
        output=out_dt,
        grid_dim=(num_tasks, 1, 1),
        block_dim=(128, 1, 1),
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    ref = silu_mul_ref(x, num_tasks=num_tasks)
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"  [{label}] bs={bs:2d}  I={intermediate}  num_tasks={num_tasks}"
          f"  max_diff={max_diff:.6f}", end="")

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
    print("  PASS")
    pk.finalize()


def test_silu_mul_testmode():
    print(f"\n{'='*60}")
    print(f"silu_mul_layer  DSV3 configs  bs sweep={BS_SWEEP}")
    for bs in BS_SWEEP:
        for label, intermediate, num_tasks in DSV3_CONFIGS:
            _run_case(bs, intermediate, num_tasks, label)
    print("ALL PASS")


if __name__ == "__main__":
    test_silu_mul_testmode()
