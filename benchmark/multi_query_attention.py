import mirage as mi
import numpy as np
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--file', type=str, default='multi_query_attention.json')
    args = parser.parse_args()
    batch_size = args.bs
    filename = args.file

    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(batch_size, 1024, 64), dtype=mi.float16)
    K = graph.new_input(dims=(batch_size, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(batch_size, 4096, 64), dtype=mi.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="attention", previous_checkpoint=filename)

    input_tensors = [
        torch.randn(batch_size, 1024, 64, dtype=torch.float16, device='cuda:0'),
        torch.randn(batch_size, 64, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(batch_size, 4096, 64, dtype=torch.float16, device='cuda:0')
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

    print("Best muGraph run time (ms): ", mean_syn)

