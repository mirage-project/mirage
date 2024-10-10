import mirage as mi
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import numpy as np
import torch

rms_norm = torch.nn.RMSNorm(64, device='cuda:0', dtype=torch.float16)
@torch.compile
def torch_chameleon(Q, K, V):
    Q = rms_norm(Q)
    V = rms_norm(V)
    O = flash_attn_func(Q, K, V)
    return O

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(8, 128, 64), dtype=mi.float16)
    K = graph.new_input(dims=(8, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(8, 4096, 64), dtype=mi.float16)
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(8,16,4), block_dim=(128,1,1), forloop_range=4, reduction_dimx=64)
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=1)
    bQ = tbgraph1.rms_norm(bQ)
    bV = tbgraph1.rms_norm(bV)
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    O = graph.customized([Q, K, V], tbgraph1)

    tbgraph2 = mi.new_threadblock_graph(grid_dim=(8,16,1), block_dim=(128,1,1), forloop_range=1, reduction_dimx=64)
    bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized(O, tbgraph2)
    
    input_tensors = [
        torch.randn(8, 128, 64, dtype=torch.float16, device='cuda:0'),
        torch.randn(8, 64, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(8, 4096, 64, dtype=torch.float16, device='cuda:0')
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(graph.cygraph, target_cc=86, input_strides=input_strides)
    print(p["code"])
    # warm up runs
    for _ in range(16):
        outputs = graph(inputs=input_tensors, outputs=O)
        #torch_chameleon(input_tensors[0], input_tensors[1], input_tensors[2])
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    starter.record()
    for rep in range(repetitions):
        outputs = graph(inputs=input_tensors, outputs=O)
        #torch_chameleon(input_tensors[0], input_tensors[1], input_tensors[2])
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    print(mean_syn)
