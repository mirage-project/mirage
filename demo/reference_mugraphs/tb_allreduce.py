import mirage as mi
import numpy as np
import torch
import os


seed = 42  # Use a fixed seed
torch.manual_seed(seed)

if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    
    torch.cuda.set_device(RANK)
    graph = mi.new_kernel_graph(gpu_dim=(2, 1, 1))
    X = graph.new_input(dims=(64, 4096), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 4096), gpu_input_map=(0, -1 ,-1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccM = tb_graph.forloop_accum(tM)

    #TB_ALLREDUCE_EPILOGUE
    tb_graph.new_output(stensor=tAccM, output_map=(1, -1, -1), epilogue="allreduce")
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])


    input_tensors = [
        torch.randn(64, 4096, dtype=torch.float16, device=torch.cuda.current_device()),
        torch.randn(4096, 4096, dtype=torch.float16, device=torch.cuda.current_device()),
    ]

    outputs = graph(inputs=input_tensors, rank=RANK)
    print(outputs[0])