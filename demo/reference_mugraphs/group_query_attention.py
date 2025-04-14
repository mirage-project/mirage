import mirage as mi
import numpy as np
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 64, 64), dtype=mi.float32)
    K = graph.new_input(dims=(2, 64, 64), dtype=mi.float32)
    V = graph.new_input(dims=(2, 64, 64), dtype=mi.float32)
    tbgraph1 = mi.new_threadblock_graph(
        grid_dim=(2, 1, 4), block_dim=(128, 1, 1), forloop_range=1, reduction_dimx=64
    )
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=-1)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    O = graph.customized([Q, K, V], tbgraph1)

    tbgraph2 = mi.new_threadblock_graph(
        grid_dim=(2, 1, 1), block_dim=(128, 1, 1), forloop_range=1, reduction_dimx=64
    )
    bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized(O, tbgraph2)

    graph.mark_output(O[0])
    input_tensors = [
        torch.randn(2, 64, 64, dtype=torch.float32, device="cuda:0"),
        torch.randn(2, 64, 64, dtype=torch.float32, device="cuda:0"),
        torch.randn(2, 64, 64, dtype=torch.float32, device="cuda:0"),
    ]

    tQ, tK, tV = input_tensors
    # tQ = torch.ones_like(tQ)
    # tK = torch.ones_like(tK)
    # tV = torch.ones_like(tV)
    # tQ /= 10
    # tK /= 10
    # tV /= 10
    # nQ = torch.bernoulli(0.5 * tQ)
    # nK = torch.bernoulli(0.5 * tK)
    # nV = torch.bernoulli(0.5 * tV)
    # tQ = 0.9 * tQ + 0.2 * nQ
    # tK = 0.9 * tK + 0.2 * nK
    # tV = 0.9 * tV + 0.2 * nV
    input_tensors = [tQ, tK, tV]
    tQ = tQ.float()
    tK = tK.float()
    tV = tV.float()
    tA = torch.matmul(tQ, tK)
    print(tA.shape)
    # tA = torch.nn.functional.softmax(tA, dim=-1)
    row_max = torch.max(tA, dim=-1, keepdim=True)[0]
    tA = tA - row_max
    tE = torch.exp(tA)
    tO1 = torch.matmul(tE, tV)
    tO2 = torch.sum(tE, dim=-1, keepdim=True)
    tS = tO1 / tO2

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(
        graph.cygraph, target_cc=86, input_strides=input_strides
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

    mean_syn = curr_time / 1000
    print(mean_syn)
    print(outputs[0] / tS)
