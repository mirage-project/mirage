"""Unit test for MergedColumnParallelLinear: per-shard_id load into the
local fused buffer at the right offset.
"""

import sys

import torch

from mirage.mpk.context import compile_scope
from mirage.mpk.parallel import ParallelConfig
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers import MergedColumnParallelLinear


def _make_pk(tp_size: int, rank: int) -> PersistentKernel:
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["world_size"] = tp_size
    params["mpi_rank"] = rank
    params["parallel_config"] = ParallelConfig(
        world_size=tp_size, rank=rank, tp_size=tp_size, ep_size=1,
    )
    return PersistentKernel(**params)


def test_merged_column_parallel_gate_up():
    in_features = 256
    gate_out = 128
    up_out = 128
    tp = 2
    gate = torch.arange(gate_out * in_features, dtype=torch.bfloat16).reshape(
        gate_out, in_features,
    )
    up = (torch.arange(up_out * in_features, dtype=torch.bfloat16) + 1e5).reshape(
        up_out, in_features,
    )

    for rank in range(tp):
        pk = _make_pk(tp, rank)
        with compile_scope(pk):
            layer = MergedColumnParallelLinear(
                in_features, [gate_out, up_out], prefix=f"mcp_r{rank}_",
            )
            local_total = (gate_out + up_out) // tp
            if tuple(layer.weight.shape) != (local_total, in_features):
                print(f"FAIL rank={rank}: shape mismatch {layer.weight.shape}")
                sys.exit(1)
            layer.weight.weight_loader(layer.weight, gate, shard_id=0)
            layer.weight.weight_loader(layer.weight, up, shard_id=1)
            local_gate = gate_out // tp
            local_up = up_out // tp
            start = rank * local_gate
            if not torch.equal(layer.weight.data[:local_gate], gate[start:start + local_gate]):
                print(f"FAIL rank={rank}: gate slice")
                sys.exit(1)
            start_u = rank * local_up
            if not torch.equal(layer.weight.data[local_gate:local_gate + local_up],
                               up[start_u:start_u + local_up]):
                print(f"FAIL rank={rank}: up slice")
                sys.exit(1)
    print("PASSED: MergedColumnParallelLinear gate/up shard_id load at tp=2")


if __name__ == "__main__":
    test_merged_column_parallel_gate_up()
