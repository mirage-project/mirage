"""PyTorch reference for the MPK allreduce_layer.

In a real distributed run, allreduce_layer performs an NVSHMEM-based
ld_reduce across `world_size` ranks: output = sum_r input_r.

This test directory drives the layer in single-rank test_mode (world_size=1).
With a single rank, the all-reduce reduces to identity:

    output == input

Multi-rank (world_size > 1) cannot run inside the in-process test_mode harness
because it requires MPI bootstrap + NVSHMEM symmetric heap initialization,
neither of which is set up by `pk()` in single-rank test mode.
"""

import torch


def allreduce_ref(input_tensor: torch.Tensor, world_size: int = 1) -> torch.Tensor:
    """Single-rank all-reduce reference: identity (sum over one rank == input).

    For world_size > 1 the reference would be `world_size * input` if every
    rank submitted the same tensor, but multi-rank execution is out of scope
    for test_mode.
    """
    assert world_size == 1, (
        "allreduce_ref only supports world_size=1 in test_mode; "
        "multi-rank requires MPI/NVSHMEM bootstrap."
    )
    return input_tensor.clone()
