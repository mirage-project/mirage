import mirage as mi
import numpy as np
import torch

torch.manual_seed(42)

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 64, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 2048), dtype=mi.float16)
    V = graph.new_input(dims=(2, 2048, 64), dtype=mi.float16)
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(2,8,1),block_dim=(256,1,1), forloop_range=4, reduction_dimx=64, cluster_dim=(1, 8, 1))
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, -1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=1)
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    bS = tbgraph1.matmul(bE, bV)
    bV1 = tbgraph1.forloop_accum(bS)
    bEs = tbgraph1.forloop_accum(bE, "sum")
    bEss = tbgraph1.cluster_accum(bEs, "sum")
    bO = tbgraph1.div(bV1, bEss)
    tbgraph1.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized([Q, K, V], tbgraph1)

    # torch.Size([2, 256, 1024])
    # torch.Size([2, 256, 16])
    
    graph.mark_output(O[0])

    input_tensors = [
        torch.full((2, 64, 64), 0.1, dtype=torch.float16, device='cuda:0'),
        torch.full((2, 64, 2048), 0.1, dtype=torch.float16, device='cuda:0'),
        torch.full((2, 2048, 64), 0.1, dtype=torch.float16, device='cuda:0')
    ]
    # input_tensors = [
    #     torch.randn(2, 256, 64, dtype=torch.float16, device='cuda:0'),
    #     torch.randn(2, 64, 2048, dtype=torch.float16, device='cuda:0'),
    #     # torch.randn(2, 2048, 64, dtype=torch.float16, device='cuda:0')
    # ]

    # input_strides = [tensor.stride() for tensor in input_tensors]
    # p = mi.generate_cuda_program(graph.cygraph, target_cc=90, input_strides=input_strides, num_warp_groups = 2, pipeline_stages = 2)
    # print(p["code"])
    outputs = graph(inputs=input_tensors)
    