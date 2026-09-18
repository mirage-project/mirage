import mirage as mi
import torch
import runtime_kernel_hopper

torch.set_printoptions(sci_mode=False)

reduction_size = 4096
output_sizes = [16, 32, 64]
batch_size = 64

rms_norm = torch.nn.RMSNorm(reduction_size, device="cuda:0", dtype=torch.bfloat16)


def torch_rms_norm(X, G, W, eps):
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance + eps)
    X = torch.mul(X, G)
    WT = torch.transpose(W, 0, 1)
    O = torch.matmul(X, WT)
    return O


for output_size in output_sizes:
    print(f"\n=== Testing output_size = {output_size} ===")

    x = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    g = torch.randn((batch_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    eps = 0.8765
    output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

    runtime_kernel_hopper.norm_linear(x, g, w, output, eps)
    torch_out = torch_rms_norm(x, g, w, eps)

    print("Ratio (kernel / torch):")
    print(output / torch_out)

    # Warm-up
    for _ in range(16):
        runtime_kernel_hopper.norm_linear(x, g, w, output, eps)

    torch.cuda.synchronize()
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        runtime_kernel_hopper.norm_linear(x, g, w, output, eps)
    ender.record()
    torch.cuda.synchronize()

    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions

    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")
