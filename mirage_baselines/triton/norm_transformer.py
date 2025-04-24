import pytest
import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def mul_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


empty = torch.empty(128, device="cuda")

rms_norm4k = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)

class _norm_transformer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, H, alpha):
        A = torch.empty_like(X)
        B = torch.empty_like(X)
        C = torch.empty_like(X)
        n_elements = X.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        H_norm = rms_norm4k(H)
        add_kernel[grid](H_norm, X, A, n_elements, BLOCK_SIZE=1024)
        mul_kernel[grid](alpha, X, B, n_elements, BLOCK_SIZE=1024)
        add_kernel[grid](X, B, C, n_elements, BLOCK_SIZE=1024)
        O = rms_norm4k(C)
        return O

norm_transformer = _norm_transformer.apply

# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False]:
       configs.append(
           triton.testing.Benchmark(
               x_names=["BATCH"],
               x_vals=[1, 8],
               line_arg="provider",
               line_vals=["triton"],
               line_names=["Triton"],
               styles=[("red", "-"), ("blue", "-")],
               ylabel="ms",
               plot_name="output.plot",
               args={
                   "dtype": torch.float16,
                   "mode": mode,
               },
           ))


@triton.testing.perf_report(configs)
def bench_test(BATCH, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 0
    rep = 100
    X = torch.randn((BATCH, 4096), dtype=dtype, device="cuda", requires_grad=True)
    H = torch.randn((BATCH, 4096), dtype=dtype, device="cuda", requires_grad=True)
    alpha = torch.randn((BATCH, 4096), dtype=dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    fn = lambda: norm_transformer(X, H, alpha)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


# only works on post-Ampere GPUs right now
bench_test.run(save_path=".", print_data=True)
