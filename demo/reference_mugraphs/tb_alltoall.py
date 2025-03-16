import mirage as mi
import numpy as np
import torch
import os
import argparse


seed = 42  # Use a fixed seed
torch.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--save_codes", action="store_true", help="Save the generated codes")
    args = parser.parse_args()
    save_codes = args.save_codes

    RANK = int(os.environ.get("RANK", 0))
    
    torch.cuda.set_device(RANK)
    graph = mi.new_kernel_graph(gpu_dim=(4, 1, 1))
    X = graph.new_input(dims=(512, 128), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
    W = graph.new_input(dims=(128, 256), gpu_input_map=(0, -1 ,-1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(4,8,1), block_dim=(128,1,1), forloop_range=4, reduction_dimx=4)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, 0, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccM = tb_graph.forloop_accum(tM)

    #TB_ALLTOALL_EPILOGUE
    tb_graph.new_output(stensor=tAccM, output_map=(1, 0, -1), epilogue="alltoall") # (0, 1, -1) right?
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])


    print("Current rank: ", RANK)
    print("Current device: ", torch.cuda.current_device())
    input_tensors = [
        torch.randn(512, 256, dtype=torch.float16, device=torch.cuda.current_device()),
        torch.randn(256, 256, dtype=torch.float16, device=torch.cuda.current_device()),
    ]

    outputs = graph(inputs=input_tensors, rank=RANK, save_codes=save_codes)
    print(outputs[0])