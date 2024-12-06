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
    X = graph.new_input(dims=(64, 4096), dtype=mi.float16)
    alpha = graph.new_input(dims=(64, 4096), dtype=mi.float16)
    TC = graph.new_input(dims=(64, 4096), dtype=mi.float16)

    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(256,1,1), forloop_range=1, reduction_dimx=64)

    tX = tb_graph.new_input(dtensor=X, input_map=(0, -1, -1), forloop_dim=-1)
    talpha = tb_graph.new_input(dtensor=alpha, input_map=(0, -1, -1), forloop_dim=-1)
    tC = tb_graph.new_input(dtensor=TC, input_map=(0, -1, -1), forloop_dim=-1)

    X_norm = tb_graph.rms_norm(tX)
    A = tb_graph.add(X_norm, talpha)
    B = tb_graph.mul(A, tC)
    C = tb_graph.add(B, talpha)
    D = tb_graph.forloop_accum(C)
    E = tb_graph.rms_norm(D)
    tb_graph.new_output(stensor=E, output_map=(0, -1, -1))
    O = graph.customized([X, alpha, TC], tb_graph)
    graph.mark_output(O[0])

    input_tensors = [
        torch.randn(64, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(64, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(64, 4096, dtype=torch.float16, device='cuda:0')
    ]

    for _ in range(1000):
        outputs = graph(inputs=input_tensors)
        # torch_rms_norm(input_tensors[0], input_tensors[1])

    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10000
    timings=np.zeros((repetitions,1))
    starter.record()
    for rep in range(repetitions):
        outputs = graph(inputs=input_tensors)
        # torch_rms_norm(input_tensors[0], input_tensors[1])

    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 10000
    #print(timings)
    print(mean_syn)
