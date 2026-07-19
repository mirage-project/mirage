"""PyTorch reference implementations for argmax_partial / argmax_reduce.

The MPK kernels split the vocab dimension into ``num_partitions`` equal-sized
chunks. Each partial task computes the argmax within its own chunk and stores
the *relative* index inside that chunk (not the global vocab index). The
reduce kernel then picks the partition with the highest value and reconstructs
the global index as ``chunk_idx * chunk_size + relative_idx``.

These references mirror that exact contract so the partial output / reduce
output can be compared element-wise against the kernel's output.
"""

import torch


def argmax_partial_ref(x: torch.Tensor, num_partitions: int):
    """Per-chunk top-1 over the last dim.

    Args:
        x: ``[B, V]`` tensor (any floating dtype). ``V`` must be divisible
            by ``num_partitions``.
        num_partitions: number of equal-sized chunks to split the vocab into.

    Returns:
        (values, indices) with shapes ``[B, num_partitions]``:
            * ``values``  — max value within each chunk (same dtype as ``x``)
            * ``indices`` — relative index *within* the chunk (int64), so
              global index = ``chunk_idx * chunk_size + indices``
    """
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    B, V = x.shape
    assert V % num_partitions == 0, (
        f"vocab {V} must be divisible by num_partitions {num_partitions}"
    )
    chunk_size = V // num_partitions
    # [B, num_partitions, chunk_size]
    x_chunks = x.view(B, num_partitions, chunk_size)
    values, indices = x_chunks.max(dim=-1)
    return values, indices.to(torch.int64)


def argmax_reduce_ref(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Reduce per-partition (value, relative_idx) pairs into the global argmax.

    Args:
        values: ``[B, num_partitions]`` — per-chunk max values.
        indices: ``[B, num_partitions]`` — per-chunk relative indices.

    Returns:
        ``[B, 1]`` int64 tensor of global argmax indices.
    """
    assert values.dim() == 2 and indices.dim() == 2
    assert values.shape == indices.shape
    B, num_partitions = values.shape
    chunk_idx = values.argmax(dim=-1)  # [B]
    relative = indices.gather(1, chunk_idx.unsqueeze(-1)).squeeze(-1)  # [B]
    # Note: chunk_size is *implicit* — the reduce kernel multiplies by the
    # static CHUNK_SIZE template parameter. We don't have that here, so the
    # tests pass it explicitly via the wrapper below.
    return chunk_idx, relative


def argmax_reduce_ref_with_chunk_size(
    values: torch.Tensor, indices: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    """Reduce variant that returns the global vocab index, matching the kernel.

    Composition with ``argmax_partial_ref`` is exact:
        ``argmax_reduce_ref_with_chunk_size(*argmax_partial_ref(x, k), V//k)``
        equals ``x.argmax(dim=-1, keepdim=True)``.
    """
    chunk_idx, relative = argmax_reduce_ref(values, indices)
    global_idx = chunk_idx.to(torch.int64) * chunk_size + relative.to(torch.int64)
    return global_idx.unsqueeze(-1)
