import mirage as mi
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 1024), dtype=mi.float16)
    V = graph.new_input(dims=(2, 1024, 64), dtype=mi.float16)
    tbgraph1 = mi.new_threadblock_graph(
        grid_dim=(2, 1, 16), block_dim=(128, 1, 1), forloop_range=16, reduction_dimx=64
    )
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, -1, -1), forloop_dim=2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, -1, -1), forloop_dim=1)
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    O = graph.customized([Q, K, V], tbgraph1)

    # tbgraph2 = mi.new_threadblock_graph(
    #     grid_dim=(2, 1, 1), block_dim=(128, 1, 1), forloop_range=1, reduction_dimx=64
    # )
    # bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    # bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    # bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    # bB = tbgraph2.forloop_accum(bB, "sum")
    # bO = tbgraph2.div(bA, bB)
    # tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    # O = graph.customized(O, tbgraph2)

    graph.mark_output(O[0])
    graph.mark_output(O[1])

    input_tensors = [
        torch.randn(2, 256, 64, dtype=torch.float16, device="cuda:0"),
        torch.randn(2, 64, 1024, dtype=torch.float16, device="cuda:0"),
        torch.randn(2, 1024, 64, dtype=torch.float16, device="cuda:0"),
    ]

    # t means torch
    tQ, tK, tV = [tensor.float() for tensor in input_tensors]
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
        enable_online_softmax=True,
    )
    print(p["code"])

    outputs = graph(inputs=input_tensors, enable_online_softmax=True)
    print(outputs[0])
    print(outputs[1])
    print(tO1)
    print(tO2)
    R = outputs[0] / outputs[1]
    print(R / tR)
