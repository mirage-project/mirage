import mirage as mi
import numpy as np
import torch

rms_norm = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)
#@torch.compile
def torch_rms_norm(X, W):
    D = rms_norm(X)
    E = torch.matmul(D, W)
    return E

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])
    
    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
    ]

    input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(graph.cygraph, target_cc=80, input_strides=input_strides)
    print(p["code"])
    # warm up runs
    for _ in range(16):
        outputs = graph(inputs=input_tensors)
        #torch_rms_norm(input_tensors[0], input_tensors[1])


    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    starter.record()
    for rep in range(repetitions):
        outputs = graph(inputs=input_tensors)
        #torch_rms_norm(input_tensors[0], input_tensors[1])

    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    #print(timings)
    print(mean_syn)
