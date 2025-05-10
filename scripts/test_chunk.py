import mirage as mi
import numpy as np
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(8, 14336 * 2), dtype=mi.float16)
    W = graph.new_input(dims=(8, 14336), dtype=mi.float16)
    D1, D2 = graph.chunk(X, 2, 1)
    O = graph.add(D1, W)
    graph.mark_output(O)
    graph.mark_output(D2)

    optimized_graph = graph.superoptimize(config="mlp")

    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(2048, 16, dtype=torch.float16, device='cuda:0'),
    ]

    outputs = optimized_graph(inputs=input_tensors)
    output = outputs[0]
    print(output.shape)
    print(output.stride(0), output.stride(1))

    for _ in range(16):
        optimized_graph(inputs=input_tensors)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(1000):
        optimized_graph(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000

    print("Best muGraph run time (ms): ", mean_syn)
