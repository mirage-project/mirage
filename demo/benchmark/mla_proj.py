import mirage as mi
import numpy as np
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--file', type=str, default='mla_proj.json')
    args = parser.parse_args()
    batch_size = args.bs
    filename = args.file

    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    W_d = graph.new_input(dims=(4096, 512), dtype=mi.float16)
    W_u = graph.new_input(dims=(512, 5120), dtype=mi.float16)
    C = graph.matmul(X, W_d)
    O = graph.matmul(C, W_u)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(previous_checkpoint=filename)

    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 512, dtype=torch.float16, device='cuda:0'),
        torch.randn(512, 5120, dtype=torch.float16, device='cuda:0')
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

