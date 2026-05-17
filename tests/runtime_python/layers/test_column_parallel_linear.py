"""Unit test for ColumnParallelLinear shape + weight_loader narrow math.

No cross-rank communication. Simulates each rank in the same process by
constructing a fresh PK with ``parallel_config=(tp_size=2, rank=r)`` and
verifying that the local weight shape is sharded and that the loader
narrows the full source correctly.
"""

import sys

import torch

from mirage.mpk.context import compile_scope
from mirage.mpk.parallel import ParallelConfig
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers import ColumnParallelLinear


def _make_pk(tp_size: int, rank: int) -> PersistentKernel:
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["world_size"] = tp_size
    params["mpi_rank"] = rank
    params["parallel_config"] = ParallelConfig(
        world_size=tp_size, rank=rank, tp_size=tp_size, ep_size=1,
    )
    return PersistentKernel(**params)


def test_column_parallel_shape_and_narrow():
    in_features = 256
    out_features = 128
    tp = 2

    full = torch.arange(out_features * in_features, dtype=torch.bfloat16).reshape(
        out_features, in_features,
    )

    for rank in range(tp):
        pk = _make_pk(tp, rank)
        with compile_scope(pk):
            layer = ColumnParallelLinear(
                in_features, out_features, prefix=f"cp_r{rank}_",
            )
            expected_shape = (out_features // tp, in_features)
            if tuple(layer.weight.shape) != expected_shape:
                print(f"FAIL rank={rank}: shape {layer.weight.shape} != {expected_shape}")
                sys.exit(1)
            layer.weight.weight_loader(layer.weight, full)
            shard_size = out_features // tp
            start = rank * shard_size
            expected = full[start:start + shard_size]
            if not torch.equal(layer.weight.data, expected):
                print(f"FAIL rank={rank}: narrow != expected")
                sys.exit(1)
    print("PASSED: ColumnParallelLinear shape + weight_loader narrow at tp=2")


def test_column_parallel_via_load_weights_routing():
    """Confirm the loader is invoked when MPKModule.load_weights routes."""
    pk = _make_pk(2, 1)
    with compile_scope(pk):
        layer = ColumnParallelLinear(
            in_features=256, out_features=128, prefix="t_",
        )
        full = torch.arange(128 * 256, dtype=torch.bfloat16).reshape(128, 256)
        consumed = layer.load_weights([("weight", full)])
        assert consumed == {"weight"}, consumed
        assert torch.equal(layer.weight.data, full[64:128])
    print("PASSED: ColumnParallelLinear via MPKModule.load_weights routing")


if __name__ == "__main__":
    test_column_parallel_shape_and_narrow()
    test_column_parallel_via_load_weights_routing()
