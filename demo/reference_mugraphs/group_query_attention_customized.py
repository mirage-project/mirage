import mirage as mi
import numpy as np
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 1024), dtype=mi.float16)
    V = graph.new_input(dims=(2, 1024, 64), dtype=mi.float16)
    tbgraph1 = mi.new_threadblock_graph(
        grid_dim=(2, 16, 4), block_dim=(128, 1, 1), forloop_range=4, reduction_dimx=64
    )
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=1)
    bA = tbgraph1.matmul(bQ, bK)
    bAmax, bDiff = tbgraph1.reduction_max(bA, dim=2)
    bAs = tbgraph1.sub(bA, bAmax)
    bEAs = tbgraph1.exp(bAs)
    bEDiff = tbgraph1.exp(bDiff)
    bS = tbgraph1.matmul(bEAs, bV)
    bO1 = tbgraph1.forloop_accum_rescale(bS, bEDiff)
    bO2 = tbgraph1.forloop_accum_rescale(bEAs, bEDiff, "sum")
    bO3 = tbgraph1.forloop_accum_max(bAmax)
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO3, output_map=(0, 2, 1))
    O = graph.customized([Q, K, V], tbgraph1)

    tbgraph2 = mi.new_threadblock_graph(
        grid_dim=(2, 1, 4), block_dim=(128, 1, 1), forloop_range=16, reduction_dimx=64
    )
    bNumerator = tbgraph2.new_input(dtensor=O[0], input_map=(0, -1, 1), forloop_dim=2)
    bDenominator = tbgraph2.new_input(dtensor=O[1], input_map=(0, -1, 1), forloop_dim=2)
    bLocalMax = tbgraph2.new_input(dtensor=O[2], input_map=(0, -1, 1), forloop_dim=2)
    bMaxLocalMax, bDiffLocalMax = tbgraph2.reduction_max(bLocalMax, dim=2)
    bLocalMaxMinusMaxLocalMax = tbgraph2.sub(bLocalMax, bMaxLocalMax)
    bExpDiffMax = tbgraph2.exp(bDiffLocalMax)
    bExpLocalMaxMinusMaxLocalMax = tbgraph2.exp(bLocalMaxMinusMaxLocalMax)
    bNumerator = tbgraph2.mul(bNumerator, bExpLocalMaxMinusMaxLocalMax)
    bDenominator = tbgraph2.mul(bDenominator, bExpLocalMaxMinusMaxLocalMax)
    bN = tbgraph2.forloop_accum_rescale(bNumerator, bExpDiffMax)
    bD = tbgraph2.forloop_accum_rescale(bDenominator, bExpDiffMax, "sum")
    bO = tbgraph2.div(bN, bD)
    tbgraph2.new_output(stensor=bO, output_map=(0, -1, 1))
    Out = graph.customized(O, tbgraph2)

    graph.mark_output(Out[0])

    input_tensors = [
        torch.randn(2, 256, 64, dtype=torch.float16, device="cuda:0"),
        torch.randn(2, 64, 1024, dtype=torch.float16, device="cuda:0"),
        torch.randn(2, 1024, 64, dtype=torch.float16, device="cuda:0"),
    ]

    tQ, tK, tV = input_tensors
    tA = torch.matmul(tQ, tK)
    row_max = torch.max(tA, dim=-1, keepdim=True)[0]
    tA = tA - row_max
    tE = torch.exp(tA)
    tO1 = torch.matmul(tE, tV)
    tO2 = torch.sum(tE, dim=-1, keepdim=True)
    tR = tO1 / tO2

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(
        graph.cygraph,
        target_cc=86,
        input_strides=input_strides,
    )
    print(p["code"])

    # warm up runs
    for _ in range(16):
        outputs = graph(inputs=input_tensors)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 1000
    timings = np.zeros((repetitions, 1))
    starter.record()
    for rep in range(repetitions):
        outputs = graph(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / repetitions
    print(mean_syn)
    print(outputs[0] / tR)
    graph.visualize("group_query_attention_customized")
