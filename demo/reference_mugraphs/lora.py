import mirage as mi
import argparse
import os
import torch
from mirage import visualizer

@torch.compile(backend="cudagraphs")
def torch_lora(X, W, A, B):
    D = torch.matmul(X, A)
    E = torch.matmul(D, B)
    C = torch.matmul(X, W)
    O = torch.add(C, E)
    return O

def optimize_lora(checkpoint):
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

    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 16, dtype=torch.float16, device='cuda:0'),
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0')
    ]

    #opt_torch_lora = torch.compile(torch_lora)
    opt_torch_lora = torch_lora
    for _ in range(16):
        opt_torch_lora(input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3])
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for rep in range(1000):
        opt_torch_lora(input_tensors[0], input_tensors[1], input_tensors[2], input_tensors[3])
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000
    print(mean_syn)
    graph.visualize("lora")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    optimize_lora(args.checkpoint)
