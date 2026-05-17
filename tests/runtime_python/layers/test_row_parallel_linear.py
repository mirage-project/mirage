"""Unit test for RowParallelLinear / RowParallelLinearWithResidual shape +
weight_loader narrow.
"""

import sys

import torch

from mirage.mpk.context import compile_scope
from mirage.mpk.parallel import ParallelConfig
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers import RowParallelLinear, RowParallelLinearWithResidual


def _make_pk(tp_size: int, rank: int) -> PersistentKernel:
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["world_size"] = tp_size
    params["mpi_rank"] = rank
    params["parallel_config"] = ParallelConfig(
        world_size=tp_size, rank=rank, tp_size=tp_size, ep_size=1,
    )
    return PersistentKernel(**params)


def test_row_parallel_shape_and_narrow():
    in_features = 256
    out_features = 128
    tp = 2

    full = torch.arange(out_features * in_features, dtype=torch.bfloat16).reshape(
        out_features, in_features,
    )

    for rank in range(tp):
        pk = _make_pk(tp, rank)
        with compile_scope(pk):
            layer = RowParallelLinear(
                in_features, out_features, prefix=f"rp_r{rank}_",
            )
            expected_shape = (out_features, in_features // tp)
            if tuple(layer.weight.shape) != expected_shape:
                print(f"FAIL rank={rank}: shape {layer.weight.shape} != {expected_shape}")
                sys.exit(1)
            layer.weight.weight_loader(layer.weight, full)
            shard_size = in_features // tp
            start = rank * shard_size
            expected = full[:, start:start + shard_size]
            if not torch.equal(layer.weight.data, expected):
                print(f"FAIL rank={rank}: narrow != expected")
                sys.exit(1)
    print("PASSED: RowParallelLinear shape + weight_loader narrow at tp=2")


def test_row_parallel_with_residual_shape_and_narrow():
    in_features = 256
    out_features = 128
    tp = 2

    full = torch.arange(out_features * in_features, dtype=torch.bfloat16).reshape(
        out_features, in_features,
    )

    for rank in range(tp):
        pk = _make_pk(tp, rank)
        with compile_scope(pk):
            layer = RowParallelLinearWithResidual(
                in_features, out_features, prefix=f"rpr_r{rank}_",
            )
            expected_shape = (out_features, in_features // tp)
            if tuple(layer.weight.shape) != expected_shape:
                print(f"FAIL rank={rank}: shape {layer.weight.shape} != {expected_shape}")
                sys.exit(1)
            layer.weight.weight_loader(layer.weight, full)
            shard_size = in_features // tp
            start = rank * shard_size
            expected = full[:, start:start + shard_size]
            if not torch.equal(layer.weight.data, expected):
                print(f"FAIL rank={rank}: narrow != expected")
                sys.exit(1)
    print("PASSED: RowParallelLinearWithResidual shape + weight_loader narrow")


if __name__ == "__main__":
    test_row_parallel_shape_and_narrow()
    test_row_parallel_with_residual_shape_and_narrow()
