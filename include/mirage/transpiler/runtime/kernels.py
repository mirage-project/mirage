import triton
import triton.language as tl

@triton.jit
def mul_scalar_kernel(out_ptr, in_ptr, scalar, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    val = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, val * scalar, mask=mask)

@triton.jit
def silu_kernel(out_ptr, in_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, x * tl.sigmoid(x), mask=mask)

@triton.jit
def sigmoid_kernel(out_ptr, in_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, tl.sigmoid(x), mask=mask)

@triton.jit
def log_kernel(out_ptr, in_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + idx, tl.log(x), mask=mask)

@triton.jit
def rms_norm_kernel(out_ptr, in_ptr, N, eps, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(in_ptr + idx, mask=mask)

    mean_sq = tl.sum(x * x, axis=0) / N  # Compute mean square
    norm_factor = tl.sqrt(mean_sq + eps) # Normalize

    tl.store(out_ptr + idx, x / norm_factor, mask=mask)

@triton.jit
def concat_kernel(out_ptr, in_ptr1, in_ptr2, N1, N2, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask1 = idx < N1
    mask2 = (idx >= N1) & (idx < (N1 + N2))

    val1 = tl.load(in_ptr1 + idx, mask=mask1, other=0)
    val2 = tl.load(in_ptr2 + (idx - N1), mask=mask2, other=0)
    tl.store(out_ptr + idx, val1 + val2, mask=mask1 | mask2)

@triton.jit
def allreduce_kernel(out_ptr, in_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(in_ptr + idx, mask=mask)

    # Reduce across all threads
    reduced = tl.sum(x, axis=0)

    tl.store(out_ptr + idx, reduced, mask=mask)
