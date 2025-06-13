import mirage as mi
import numpy as np
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--file', type=str, default='group_query_attention.json')
    parser.add_argument('--backend', type=str, default='cuda')
    parser.add_argument('--warmup', type=int, default=16)
    parser.add_argument('--profile', type=int, default=1000)
    parser.add_argument('--save_codes', type=bool, default=False)
    
    args = parser.parse_args()
    batch_size = args.bs
    filename = args.file
    backend = args.backend
    warmup_iters = args.warmup
    profile_iters = args.profile
    save_codes = args.save_codes

    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 256), dtype=mi.float16)
    W = graph.new_input(dims=(256, 4096), dtype=mi.float16)
    A = graph.new_input(dims=(256, 16), dtype=mi.float16)
    B = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    D = graph.matmul(X, A)
    E = graph.matmul(D, B)
    C = graph.matmul(X, W)
    O = graph.add(C, E)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="lora", previous_checkpoint=filename, backend=backend, save_codes=save_codes, warmup_iters=warmup_iters, profile_iters=profile_iters)

    input_tensors = [
        torch.randn(16, 256, dtype=torch.float16, device='cuda:0'),
        torch.randn(256, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(256, 16, dtype=torch.float16, device='cuda:0'),
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0')
    ]

    for _ in range(1):
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

