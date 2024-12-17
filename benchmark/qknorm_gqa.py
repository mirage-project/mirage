import mirage as mi
import numpy as np
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--file', type=str, default='qknorm_gqa.json')
    args = parser.parse_args()
    batch_size = args.bs
    filename = args.file

    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2 * batch_size, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2 * batch_size, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(2 * batch_size, 4096, 64), dtype=mi.float16)
    nQ = graph.rms_norm(Q, normalized_shape=(64,))
    nV = graph.rms_norm(V, normalized_shape=(64,))
    A = graph.matmul(nQ, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, nV)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="attention", previous_checkpoint=filename)

    input_tensors = [
        torch.randn(2 * batch_size, 256, 64, dtype=torch.float16, device='cuda:0'),
        torch.randn(2 * batch_size, 64, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(2 * batch_size, 4096, 64, dtype=torch.float16, device='cuda:0')
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

