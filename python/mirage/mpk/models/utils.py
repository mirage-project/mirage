import torch

def grid_for_rmsnorm_linear_layer(size):
    # 96 and 64 are enough to cover all Qwen3 model? Please update the method
    # if you meet any incompatibility.
    if size / 96 > 400:
        # An ad-hoc workaround for the linear kernel. The original note here
        # read: "TODO ... both MPK ptx and cutlass version will output
        # unexpected result (not same out put for same prompt) if the
        # OUTPUT_SIZE is too big, try to figure it out." M3-I11 resolved that
        # PARTLY:
        #  * On Blackwell the "cutlass version" is linear_sm100_mpk.cuh, and one
        #    concrete mechanism for "not the same output for the same prompt"
        #    was found and fixed there: the task-terminal TMA store wait used
        #    cp.async.bulk.wait_group.read (source-read completion) instead of
        #    cp.async.bulk.wait_group (destination-write visibility), so a
        #    consumer task could acquire the trigger event and read this
        #    layer's output before the async-proxy write landed. Exposure grows
        #    with the per-task OUTPUT_SIZE -- more/larger in-flight store atoms
        #    at the terminal wait -- which matches the size dependence this cap
        #    works around. See the comment at that wait for the PTX citations.
        #  * It is NOT the whole story for the "MPK ptx" variant: the ptx-based
        #    Hopper linear (hopper/linear_hopper.cuh:360) has used the correct
        #    write-visibility wait, store_async_wait<0>(), since #459, and the
        #    Ampere linear writes its output with plain generic-proxy stores.
        #    Any residual size-dependent nondeterminism on those two paths has
        #    a different cause and was not re-measured here.
        # The 256 cap stays: relaxing it changes the task graph and the per-task
        # tile shape, so it needs its own perf + bit-exactness run.
        assert size % 256 == 0, f"FATAL: Linear layer size not support, it's {size}."
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64


def prepare_fp8_blockscale_weight(weight_fp8: torch.Tensor,
                                  scale_inv: torch.Tensor,
                                  block: int = 128):
    """Prepare a block-FP8 checkpoint weight for the PRESERVED-scale dense GEMM.

    The checkpoint ships `weight [N, K]` float8_e4m3fn plus
    `weight_scale_inv [N/128, K/128]`, where scale_inv[i, j] is the dequant
    scale of the weight tile W[i*128:(i+1)*128, j*128:(j+1)*128]
    (docs/qwen35/vllm-graph.md 3.4). This is the identity transform on the
    weight; the scale is only widened to float32 (checkpoints store it in BF16 —
    widening cannot recover precision, but it matches what vLLM holds at
    runtime) and made contiguous.

    It is deliberately NOT DeepSeekV3Builder._requantize_fp8_for_ue8m0: no
    dequant/re-quantize round trip, no power-of-two rounding, no collapse to
    per-row scales. Both reference engines disable the UE8M0 scale format for
    this model class on Blackwell (docs/qwen35/v1-architecture.md 6.2).

    Returns (weight_fp8 [N, K] float8_e4m3fn, scale [N/128, K/128] float32).
    """
    if weight_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "preserved-scale FP8 weight must be float8_e4m3fn, got "
            f"{weight_fp8.dtype}")
    if weight_fp8.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError("preserved-scale FP8 weight and scale must both be 2-D")
    n, k = weight_fp8.shape
    if n % block != 0 or k % block != 0:
        raise ValueError(
            f"preserved-scale FP8 weight [{n}, {k}] must be a whole number of "
            f"{block}x{block} scale blocks")
    expected = (n // block, k // block)
    if tuple(scale_inv.shape) != expected:
        raise ValueError(
            f"weight_scale_inv shape {tuple(scale_inv.shape)} does not match "
            f"the weight's block grid {expected}")
    return weight_fp8.contiguous(), scale_inv.float().contiguous()


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