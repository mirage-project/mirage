import mirage as mi
import numpy as np
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--file', type=str, default='norm_transformer.json')
    args = parser.parse_args()
    batch_size = args.bs
    filename = args.file

    graph = mi.new_kernel_graph()
    H = graph.new_input(dims=(8 * batch_size, 4096), dtype=mi.float16)
    X = graph.new_input(dims=(8 * batch_size, 4096), dtype=mi.float16)
    alpha = graph.new_input(dims=(8 * batch_size, 4096), dtype=mi.float16)
    H_norm = graph.rms_norm(H, normalized_shape=(4096,)) # TODO: replace with standard L2 norm
    A = graph.add(H_norm, X) # TODO: replace with subtract
    B = graph.mul(alpha, A)
    C = graph.add(X, B)
    O = graph.rms_norm(C, normalized_shape=(4096,)) # TODO: replace with standard L2 norm
    graph.mark_output(O)
    
    optimized_graph = graph.superoptimize(previous_checkpoint=filename)

    input_tensors = [
        torch.randn(8 * batch_size, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(8 * batch_size, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(8 * batch_size, 4096, dtype=torch.float16, device='cuda:0')
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

