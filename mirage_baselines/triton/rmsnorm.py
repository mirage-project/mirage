import pytest
import torch

import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    G, # pointer to the weight
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + 1e-9)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        g = tl.load(G + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * g,
        # Write output
        tl.store(Y + cols, y, mask=mask)

empty = torch.empty(128, device="cuda")

rms_norm4k = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)

class _rmsnorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, G, W):
        Y = torch.empty_like(X)
        X_arg = X.reshape(-1, X.shape[-1])
        M, N = X_arg.shape
        mean = torch.empty((M, ), dtype=torch.float16, device=X.device)
        rstd = torch.empty((M, ), dtype=torch.float16, device=X.device)
        MAX_FUSED_SIZE = 65536 // X.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE: raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _rms_norm_fwd_fused[(M, )](
            X_arg, Y, G, Mean=mean, Rstd=rstd,
            stride=X_arg.stride(0), N=N, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        Y = rms_norm4k(X)
        Z = torch.matmul(Y, W)
        return Z

rmsnorm = _rmsnorm.apply

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
    G = torch.randn((BATCH, 4096), dtype=dtype, device="cuda", requires_grad=True)
    W = torch.randn((4096, 4096), dtype=dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    fn = lambda: rmsnorm(X, G, W)
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


# only works on post-Ampere GPUs right now
bench_test.run(save_path=".", print_data=True)
