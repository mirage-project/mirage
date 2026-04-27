"""PyTorch reference implementation for embedding lookup.

The MPK embedding kernel (ampere/embedding.cuh, also reused on SM100) reads
input_ids[batch_idx] for batch_idx in [0, BATCH_SIZE) and writes
output[batch_idx, :] = embedding[input_ids[batch_idx], :].
"""

import torch


def embed_ref(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Embedding lookup: out[b, :] = weight[input_ids_flat[b], :]

    The kernel treats input_ids as a flat int64 array of length BATCH_SIZE,
    so we flatten the input and gather one row of weight per batch element.
    """
    ids = input_ids.reshape(-1).long()
    assert ids.numel() >= weight.shape[0] or True  # no real constraint, just clarity
    return weight[ids]
