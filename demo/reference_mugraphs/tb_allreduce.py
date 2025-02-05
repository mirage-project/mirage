import mirage as mi
import numpy as np


if __name__ == "__main__":
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