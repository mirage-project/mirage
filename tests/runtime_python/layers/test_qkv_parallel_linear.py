"""Unit test for QKVParallelLinear: GQA-aware per-shard_id load.

Uses Qwen3-8B's head numbers (num_q_heads=32, num_kv_heads=8,
head_dim=128, hidden=4096) at tp_size=2 to exercise a realistic shape.
"""

import sys

import torch

from mirage.mpk.context import compile_scope
from mirage.mpk.parallel import ParallelConfig
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers import QKVParallelLinear


def _make_pk(tp_size: int, rank: int) -> PersistentKernel:
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["world_size"] = tp_size
    params["mpi_rank"] = rank
    params["parallel_config"] = ParallelConfig(
        world_size=tp_size, rank=rank, tp_size=tp_size, ep_size=1,
    )
    return PersistentKernel(**params)


def test_qkv_parallel_qwen3_8b_shape():
    in_features = 4096
    head_dim = 128
    total_q = 32
    total_kv = 8
    tp = 2

    q_full = (torch.arange(total_q * head_dim * in_features, dtype=torch.bfloat16)
              .reshape(total_q * head_dim, in_features))
    k_full = (torch.arange(total_kv * head_dim * in_features, dtype=torch.bfloat16) + 1e6).reshape(
        total_kv * head_dim, in_features,
    )
    v_full = (torch.arange(total_kv * head_dim * in_features, dtype=torch.bfloat16) + 2e6).reshape(
        total_kv * head_dim, in_features,
    )

    for rank in range(tp):
        pk = _make_pk(tp, rank)
        with compile_scope(pk):
            qkv = QKVParallelLinear(
                in_features=in_features,
                head_dim=head_dim,
                total_num_heads=total_q,
                total_num_kv_heads=total_kv,
                prefix=f"qkv_r{rank}_",
            )
            num_local_q = total_q // tp
            num_local_kv = total_kv // tp
            q_local = num_local_q * head_dim
            kv_local = num_local_kv * head_dim
            local_total = q_local + 2 * kv_local
            if tuple(qkv.weight.shape) != (local_total, in_features):
                print(f"FAIL rank={rank}: shape {qkv.weight.shape}")
                sys.exit(1)

            qkv.weight.weight_loader(qkv.weight, q_full, shard_id="q")
            qkv.weight.weight_loader(qkv.weight, k_full, shard_id="k")
            qkv.weight.weight_loader(qkv.weight, v_full, shard_id="v")

            q_start_full = rank * q_local
            k_start_full = rank * kv_local
            v_start_full = rank * kv_local
            if not torch.equal(qkv.weight.data[0:q_local],
                               q_full[q_start_full:q_start_full + q_local]):
                print(f"FAIL rank={rank}: q slot")
                sys.exit(1)
            if not torch.equal(qkv.weight.data[q_local:q_local + kv_local],
                               k_full[k_start_full:k_start_full + kv_local]):
                print(f"FAIL rank={rank}: k slot")
                sys.exit(1)
            if not torch.equal(qkv.weight.data[q_local + kv_local:local_total],
                               v_full[v_start_full:v_start_full + kv_local]):
                print(f"FAIL rank={rank}: v slot")
                sys.exit(1)
    print("PASSED: QKVParallelLinear Qwen3-8B-shape at tp=2")


if __name__ == "__main__":
    test_qkv_parallel_qwen3_8b_shape()
