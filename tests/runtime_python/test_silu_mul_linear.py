import mirage as mi
import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False)

reduction_size = 12288
output_sizes = [16, 32, 64]
batch_size = 6

silu = torch.nn.SiLU()

for output_size in output_sizes:
    print(f"\n=== Testing output_size = {output_size} ===")

    x = torch.randn((batch_size, reduction_size * 2), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((output_size, reduction_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn((batch_size, output_size), device="cuda", dtype=torch.bfloat16)
    output = torch.empty(batch_size, output_size, device="cuda", dtype=torch.bfloat16)

    runtime_kernel.silu_mul_linear(x, w, residual, output)
    torch_out = (
        torch.matmul(
            torch.mul(silu(x[:, :reduction_size]), x[:, reduction_size:]),
            torch.transpose(w, 0, 1),
        )
        + residual
    )

    print("Ratio (kernel / torch):")
    print(output / torch_out)

    # Warm-up
    for _ in range(16):
        runtime_kernel.silu_mul_linear(x, w, residual, output)

    torch.cuda.synchronize()
    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        runtime_kernel.silu_mul_linear(x, w, residual, output)
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions
    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")

    # Compare with Mirage

    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(1, reduction_size), dtype=mi.bfloat16)
    M = graph.new_input(dims=(1, reduction_size), dtype=mi.bfloat16)
    W = graph.new_input(dims=(reduction_size, output_size), dtype=mi.bfloat16)
    b = graph.new_input(dims=(1, output_size), dtype=mi.bfloat16)
    tb_graph = mi.new_threadblock_graph(
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        forloop_range=reduction_size / 64,
        reduction_dimx=64,
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tM = tb_graph.new_input(dtensor=M, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(-1, -1, -1), forloop_dim=0)
    tActX = tb_graph.silu(tX)
    tMul = tb_graph.mul(tActX, tM)
    tMat = tb_graph.matmul(tMul, tW)
    tAccMat = tb_graph.forloop_accum(tMat)
    tb_graph.new_output(stensor=tAccMat, output_map=(-1, -1, -1))
    O = graph.customized([X, M, W], tb_graph)
    O[0] = graph.add(O[0], b)
    graph.mark_output(O[0])

    x1 = x[0, :reduction_size].contiguous().unsqueeze(0)
    x2 = x[0, reduction_size:].contiguous().unsqueeze(0)
    wt = torch.transpose(w, 0, 1).contiguous()

    input_tensors = [x1, x2, wt, residual]

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
