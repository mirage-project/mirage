import torch

def grid_for_rmsnorm_linear_layer(size):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size / 96 > 400:
        # TODO: An ad-hoc workaround for linear kernel, both MPK ptx and
        # cutlass version will output unexpected result (not same out put for
        # same prompt) if the OUTPUT_SIZE is too big, try to figure it out.
        assert size % 256 == 0, f"FATAL: Linear layer size not support, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    
# Return the largest factor of m that is less than or equal to n
# This is used to determine the grid size
def max_factor_leq_n(m: int, n: int) -> int:
    max_factor = 1
    i = 1
    while i * i <= m:
        if m % i == 0:
            if i <= n:
                max_factor = max(max_factor, i)
            if m // i <= n:
                max_factor = max(max_factor, m // i)
        i += 1
    return max_factor

def shuffle_tensors(tensors: list[torch.Tensor], split: int, dim: int) -> torch.Tensor:
    """Split each tensor along `dim` into `split` equal chunks and interleave chunks
    by tensor order into a new tensor.

    Example: given [Q, K, V], split=head_num, dim=0, result layout along dim is
    [Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...].

    Args:
        tensors: list of tensors to interleave. Must be same dtype/device and same
                 shape on all non-`dim` dimensions. Each tensor.shape[dim] must be
                 divisible by `split`.
        split: number of equal chunks to split along `dim`.
        dim: dimension index to split/interleave on (supports negative indices).

    Returns:
        A newly allocated tensor with the same rank as inputs. The size on `dim`
        equals sum(t.shape[dim] for t in tensors).
    """
    if not tensors:
        raise ValueError("tensors must be a non-empty list")

    base = tensors[0]
    dtype = base.dtype
    device = base.device
    ndim = base.ndim

    # Normalize dim
    if dim < 0:
        dim = ndim + dim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dim out of range for {ndim}-D tensor: {dim}")

    if split <= 0:
        raise ValueError("split must be a positive integer")

    # Validate shapes, dtype, device
    base_shape = tuple(base.shape)
    for idx, t in enumerate(tensors):
        if t.dtype != dtype:
            raise TypeError(f"All tensors must have same dtype; got {dtype} and {t.dtype} at index {idx}")
        if t.device != device:
            raise TypeError(f"All tensors must be on same device; got {device} and {t.device} at index {idx}")
        if t.ndim != ndim:
            raise ValueError(f"All tensors must have same rank; got {ndim} and {t.ndim} at index {idx}")
        for d in range(ndim):
            if d == dim:
                continue
            if t.shape[d] != base_shape[d]:
                raise ValueError(
                    f"Non-split dimensions must match; dim {d} differs: {base_shape[d]} vs {t.shape[d]} at index {idx}"
                )
        if t.shape[dim] % split != 0:
            raise ValueError(
                f"Tensor at index {idx} has size {t.shape[dim]} on dim {dim}, not divisible by split={split}"
            )

    per_tensor_chunk = [t.shape[dim] // split for t in tensors]
    per_head_size = sum(per_tensor_chunk)
    out_shape = list(base_shape)
    out_shape[dim] = per_head_size * split  # equals sum(t.shape[dim])

    out = torch.empty(out_shape, dtype=dtype, device=device)

    def make_slice(start: int, length: int):
        s = [slice(None)] * ndim
        s[dim] = slice(start, start + length)
        return tuple(s)

    # Interleave by head index
    write_head_base = 0
    for i in range(split):
        write_offset = 0
        for t, chunk in zip(tensors, per_tensor_chunk):
            read_start = i * chunk
            out[make_slice(write_head_base + write_offset, chunk)] = t[make_slice(read_start, chunk)]
            write_offset += chunk
        write_head_base += per_head_size

    return out

def inplace_shuffle_tensors(tensors: list[torch.Tensor], target_tensor: torch.Tensor, split: int, dim: int) -> torch.Tensor:
    """Split each tensor along `dim` into `split` equal chunks and interleave chunks
    by tensor order using a temporary GPU tensor, then copy into `target_tensor`.

    Example: given [Q, K, V], split=head_num, dim=0, result layout along dim is
    [Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...].
    """
    if not tensors:
        raise ValueError("tensors must be a non-empty list")

    device = target_tensor.device
    dtype = target_tensor.dtype

    # Ensure inputs are on the same device/dtype as target; create temporary GPU views if needed
    gpu_tensors = [
        (t if (t.device == device and t.dtype == dtype) else t.to(device=device, dtype=dtype, non_blocking=True))
        for t in tensors
    ]

    gpu_result = shuffle_tensors(gpu_tensors, split, dim)
    assert gpu_result.shape == target_tensor.shape, (
        f"GPU result shape {gpu_result.shape} does not match target tensor shape {target_tensor.shape}"
    )
    if gpu_result.dtype != dtype:
        gpu_result = gpu_result.to(dtype=dtype, non_blocking=True)
    target_tensor.copy_(gpu_result, non_blocking=True)
    # return target_tensor