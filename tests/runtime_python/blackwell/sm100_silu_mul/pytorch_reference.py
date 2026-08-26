"""PyTorch reference for the MPK silu_mul task.

The kernel partitions the input column dimension into ``num_tasks`` chunks,
each of size ``2 * chunk = 2 * I / num_tasks``. Within each chunk the layout
is ``[gate_chunk | up_chunk]`` (gate first, then up). Output column ``j`` of
chunk ``c`` is therefore::

    out[b, c, j] = silu(in[b, c, 0, j]) * in[b, c, 1, j]

If ``num_tasks == 1`` this collapses to the textbook
``silu(x[..., :I]) * x[..., I:]``.
"""

import torch
import torch.nn.functional as F


def silu_mul_ref(x, num_tasks=1):
    """Reference for MPK silu_mul.

    Args:
        x: bf16 tensor of shape ``[B, 2 * I]`` where I = output cols.
        num_tasks: number of grid_dim.x tasks the kernel was launched with.
            Each task processes ``2 * I / num_tasks`` consecutive input cols
            laid out as gate||up.

    Returns:
        bf16 tensor of shape ``[B, I]``.
    """
    in_dtype = x.dtype
    B, two_I = x.shape
    assert two_I % (2 * num_tasks) == 0, (
        f"x last dim {two_I} must be divisible by 2*num_tasks={2*num_tasks}"
    )
    chunk = two_I // (2 * num_tasks)
    # Reshape to [B, num_tasks, 2, chunk] -- [.., 0, :] is gate, [.., 1, :] is up.
    x_v = x.view(B, num_tasks, 2, chunk).to(torch.float32)
    gate = x_v[:, :, 0, :]
    up = x_v[:, :, 1, :]
    out = F.silu(gate) * up                                      # [B, num_tasks, chunk]
    return out.reshape(B, num_tasks * chunk).to(in_dtype)
