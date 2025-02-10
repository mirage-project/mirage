import mirage as mi
import numpy as np
import torch


if __name__ == "__main__":
    # TODO (linsj20)

    graph = mi.new_kernel_graph(gpu_dim=(2, 1, 1))
    X = graph.new_input(dims=(64, 4096), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 4096), gpu_input_map=(0, -1 ,-1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccM = tb_graph.forloop_accum(tM)
    tb_graph.new_output(stensor=tAccM, output_map=(1, -1, -1))
    O = graph.customized([X, W], tb_graph)

    #KN_ALLREDUCE
    OR = graph.allreduce(O[0])
    graph.mark_output(OR)

    input_tensors = [
        torch.randn(64, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(graph.cygraph, target_cc=86, input_strides=input_strides)
    print(p["code"])
    # warm up runs
    for _ in range(16):
        outputs = graph(inputs=input_tensors)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    starter.record()
    for rep in range(repetitions):
        outputs = graph(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    print(mean_syn)
