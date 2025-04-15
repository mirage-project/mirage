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

    graph = mi.new_kernel_graph(gpu_dim=(4, 1, 1))
    X = graph.new_input(dims=(256, 128), gpu_input_map=(0, -1 ,-1), dtype=mi.float16)
    W = graph.new_input(dims=(128, 256), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(2,8,1), block_dim=(128,1,1), forloop_range=4, reduction_dimx=4)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, 0, -1), forloop_dim=1, prologue="allgather")
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccM = tb_graph.forloop_accum(tM)

    #TB_ALLTOALL_EPILOGUE
    tb_graph.new_output(stensor=tAccM, output_map=(1, 0, -1)) # (0, 1, -1) right?
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])

    print("Graph created")

    from mpi4py import MPI, get_config
    print(get_config())
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()
    print("Current rank: ", rank, "/ Current npes: ", npes)

    torch.cuda.set_device(rank)
    print("Current device: ", torch.cuda.current_device())
    matrix = torch.zeros(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    matrix[:, 0:64] = 1
    matrix[:, 64:128] = 7
    matrix[:, 128:192] = 3
    matrix[:, 192:256] = 4
    input_tensors = [
        torch.ones(256, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        matrix,
        # torch.ones(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
        # torch.randn(256, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        # torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
    ]

    outputs = graph(inputs=input_tensors, rank=rank, save_codes=save_codes)


    # divide input_tensors[1] into 4 chunks along dim 1
    chunks = input_tensors[1].chunk(4, dim=1)
    print("input_tensors[0].shape: ", input_tensors[0].shape)
    print("chunks[", rank, "].shape: ", chunks[rank].shape)
    print("input_tensors[0] * chunks[", rank, "]: ", input_tensors[0] @ chunks[rank])
    

    # chunks = outputs[0].chunk(4)
    # for chunk in chunks[1:]:
    #     assert torch.allclose(chunks[0], chunk)

    # print(f"[{rank}] alltoall demo pass!")
