import mirage as mi
import numpy as np
import torch

silu = torch.nn.SiLU()
#@torch.compile
def torch_gated_mlp(X, W1, W2):
    D = torch.matmul(X, W1)
    E = torch.matmul(X, W2)
    O = torch.mul(silu(D), E)
    return O

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    W1 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    W2 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=32, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW1 = tb_graph.new_input(dtensor=W1, input_map=(1, -1, -1), forloop_dim=0)
    tW2 = tb_graph.new_input(dtensor=W2, input_map=(1, -1, -1), forloop_dim=0)
    tD1 = tb_graph.matmul(tX, tW1)
    tD2 = tb_graph.matmul(tX, tW2)
    tA1 = tb_graph.forloop_accum(tD1)
    tA2 = tb_graph.forloop_accum(tD2)
    tS = tb_graph.silu(tA1)
    tO = tb_graph.mul(tS, tA2)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W1, W2], tb_graph)
    graph.mark_output(O[0])
    
    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(graph.cygraph, target_cc=86, input_strides=input_strides)
    print(p["code"])
    silu = torch.nn.SiLU()
    # warm up runs
    for _ in range(16):
        outputs = graph(inputs=input_tensors)
        #torch_gated_mlp(input_tensors[0], input_tensors[1], input_tensors[2])

    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    starter.record()
    for rep in range(repetitions):
        outputs = graph(inputs=input_tensors)
        #torch_gated_mlp(input_tensors[0], input_tensors[1], input_tensors[2])
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    #print(timings)
    print(mean_syn)
    graph.visualize("gated_mlp")