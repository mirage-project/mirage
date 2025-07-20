import mirage as mi
import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False)

reduction_size = 4096
output_sizes = [16, 32, 64, 96, 112, 160, 192, 256, 544, 1600]

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

    x = torch.randn((1, reduction_size), device="cuda", dtype=torch.bfloat16)
    g = torch.randn((1, reduction_size), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    eps = 0.8765
    output = torch.empty(1, output_size, device="cuda", dtype=torch.bfloat16)

    runtime_kernel.norm_linear(x, g, w, eps, output)
    torch_out = torch_rms_norm(x, g, w, eps)

    print("Ratio (kernel / torch):")
    print(output / torch_out)

    # Warm-up
    for _ in range(16):
        runtime_kernel.norm_linear(x, g, w, eps, output)

    torch.cuda.synchronize()
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        runtime_kernel.norm_linear(x, g, w, eps, output)
    ender.record()
    torch.cuda.synchronize()

    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions

    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")

    # Compare with Mirage
    if output_size <= 64:
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(1, reduction_size), dtype=mi.bfloat16)
        G = graph.new_input(dims=(1, reduction_size), dtype=mi.bfloat16)
        W = graph.new_input(
            dims=(reduction_size, min(output_size, 64)), dtype=mi.bfloat16
        )
        tb_graph = mi.new_threadblock_graph(
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            forloop_range=reduction_size / 64,
            reduction_dimx=64,
        )
        tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
        tG = tb_graph.new_input(dtensor=G, input_map=(-1, -1, -1), forloop_dim=1)
        tW = tb_graph.new_input(dtensor=W, input_map=(-1, -1, -1), forloop_dim=0)
        tD = tb_graph.mul(tX, tG)
        tM = tb_graph.matmul(tD, tW)
        tAccX = tb_graph.forloop_accum(tX, "rms")
        tAccM = tb_graph.forloop_accum(tM)
        tO = tb_graph.div(tAccM, tAccX)
        tb_graph.new_output(stensor=tO, output_map=(-1, -1, -1))
        O = graph.customized([X, G, W], tb_graph)
        graph.mark_output(O[0])

        wt = w.transpose(0, 1).contiguous()

        input_tensors = [x, g, wt]
        input_strides = [tensor.stride() for tensor in input_tensors]

        for _ in range(16):
            outputs = graph(inputs=input_tensors)

        torch.cuda.synchronize()
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        starter.record()
        for rep in range(repetitions):
            outputs = graph(inputs=input_tensors)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        mean_syn = curr_time / repetitions
        print(f"Mirage average time over {repetitions} runs: {mean_syn:.6f} ms")
    else:
        print(f"Skipping Mirage test for output_size ({output_size}) > 64")
