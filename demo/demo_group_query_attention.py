import mirage as mi
import argparse
import os

def optimize_llama_70B(checkpoint):
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    best_graph = graph.superoptimize(config="attention")
    return best_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    gqa = optimize_llama_70B(args.checkpoint)

    input_tensors = [
        torch.randn(2, 256, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(2, 64, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(2, 4096, 64, dtype=torch.float16, device='cuda:0'),
    ]

    for _ in range(16):
        gqa(inputs=input_tensors)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(1000):
        gqa(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000

    print("Best muGraph run time (ms): ", mean_syn)
