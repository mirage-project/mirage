import triton
import triton.language as tl

@triton.jit
def elementwise_binary_kernel(X_ptr, Y_ptr, Out_ptr, M: tl.constexpr, N: tl.constexpr, OP: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * 128 + tl.arange(0, 128)  #TODO: 128 is hardcoded
    mask = col_idx < N

    X_offset = row_idx * N + col_idx
    Y_offset = row_idx * N + col_idx
    Out_offset = row_idx * N + col_idx

    x = tl.load(X_ptr + X_offset, mask=mask)
    y = tl.load(Y_ptr + Y_offset, mask=mask)

    if OP == 1200:
        out = x + y 
    elif OP == 1201:
        out = x * y 
    elif OP == 1202:
        out = x / y
    else:
        tl.device_print("Unsupported OP code in elementwise_binary_kernel")

    tl.store(Out_ptr + Out_offset, out, mask=mask)

@triton.jit
def elementwise_unary_kernel(X_ptr, Out_ptr, M: tl.constexpr, N: tl.constexpr, OP: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * 128 + tl.arange(0, 128) #TODO: 128 is hardcoded
    mask = col_idx < N

    X_offset = row_idx * N + col_idx
    Out_offset = row_idx * N + col_idx

    x = tl.load(X_ptr + X_offset, mask=mask)

    if OP == 1100:
        out = tl.exp(x)
    elif OP == 1102:
        out = tl.sqrt(x)
    elif OP == 1101:
        out = x * x
    else:
        tl.device_print("Unsupported OP code in elementwise_unary_kernel")

    tl.store(Out_ptr + Out_offset, out, mask=mask)

@triton.jit
def reduce_sum_kernel(X_ptr, Out_ptr, M: tl.constexpr, N: tl.constexpr, dim: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1) * 128 + tl.arange(0, 128) #TODO: 128 is hardcoded
    mask = col_idx < N

    if dim == 1:
        sum_val = tl.zeros([N], dtype=tl.float32)
        for i in range(M):
            sum_val += tl.load(X_ptr + i * N + col_idx, mask=mask)
        if row_idx == 0:
            tl.store(Out_ptr + col_idx, sum_val, mask=mask)

    elif dim == 0:
        sum_val = tl.sum(tl.load(X_ptr + row_idx * N + col_idx, mask=mask))
        if col_idx[0] == 0:
            tl.store(Out_ptr + row_idx, sum_val)

