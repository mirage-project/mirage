import mirage as mi
import numpy as np
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(1, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    A = graph.new_input(dims=(4096, 16), dtype=mi.float16)
    B = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    D = graph.matmul(X, A)
    E = graph.matmul(D, B)
    C = graph.matmul(X, W)
    O = graph.add(C, E)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="lora")

    input_tensors = [
        torch.randn(1, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 16, dtype=torch.float16, device='cuda:0'),
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0')
    ]

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

    print(mean_syn)

