import mirage as mi
import torch
import runtime_kernel

torch.set_printoptions(sci_mode=False)

sequence_size = 3584
output_sizes = [16, 32, 64]

for output_size in output_sizes:
    print(f"\n=== Testing output_dim = {output_size} ===")
    x = torch.randn((1, sequence_size), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((sequence_size, output_size), device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, output_size, device="cuda", dtype=torch.bfloat16)

    runtime_kernel.linear(x, w, output)
    torch_out = torch.matmul(x, w)

    print("Ratio (kernel / torch):")
    print(output / torch_out)

    # Warm-up
    for _ in range(16):
        runtime_kernel.linear(x, w, output)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 100000
    starter.record()
    for rep in range(repetitions):
        runtime_kernel.linear(x, w, output)
    ender.record()
    torch.cuda.synchronize()
    total_time = starter.elapsed_time(ender)
    avg_time = total_time / repetitions
    print(f"Average time over {repetitions} runs: {avg_time:.6f} ms")

    # Compare with Mirage

    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(1, sequence_size), dtype=mi.bfloat16)
    W = graph.new_input(dims=(sequence_size, output_size), dtype=mi.bfloat16)
    tb_graph = mi.new_threadblock_graph(
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1), forloop_range=56, reduction_dimx=64
    )
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(-1, -1, -1), forloop_dim=0)
    tMat = tb_graph.matmul(tX, tW)
    tAccMat = tb_graph.forloop_accum(tMat)
    tb_graph.new_output(stensor=tAccMat, output_map=(-1, -1, -1))
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])

    input_tensors = [x, w]
    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(
        graph.cygraph, target_cc=80, input_strides=input_strides
    )

    for _ in range(16):
        outputs = graph(inputs=input_tensors)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    starter.record()
    for rep in range(repetitions):
        outputs = graph(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / repetitions
    print(f"Mirage average time over {repetitions} runs: {mean_syn:.6f} ms")
