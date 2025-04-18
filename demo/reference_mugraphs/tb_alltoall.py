import mirage as mi
import numpy as np
import torch
import os
import argparse
import logging

seed = 42  # Use a fixed seed
torch.manual_seed(seed)
torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--save_codes", action="store_true", help="Save the generated codes")
    args = parser.parse_args()
    save_codes = args.save_codes

    graph = mi.new_kernel_graph(gpu_dim=(4, 1, 1))
    X = graph.new_input(dims=(256, 128), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
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

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # file logger
    logging.basicConfig(level=logging.INFO, filename=f"tb_alltoall_{rank}.log")
    logger = logging.getLogger(__name__)

    torch.cuda.set_device(rank)
    # print("Current rank: ", rank)
    # print("Current device: ", torch.cuda.current_device())
    matrix = torch.zeros(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    matrix[0:32, :] = 5
    matrix[32:64, :] = 2
    matrix[64:96, :] = 3
    matrix[96:128, :] = 4
    # print("[", rank, "] matrix: ", matrix)
    input_tensors = [
        # torch.randn(64, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        # torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
        torch.ones(256, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        # torch.ones(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
        matrix,
    ]

    outputs = graph(inputs=input_tensors, rank=rank, save_codes=save_codes)
    # print("[", rank, "] outputs[0].shape: ", outputs[0].shape)
    # save outputs[0] to a txt file as int16 and not scientific notation
    logger.info(f"[{rank}] outputs[0].shape: {outputs[0].shape}")
    logger.info(f"[{rank}] outputs[0]: {outputs[0]}")




    # chunks = outputs[0].chunk(4)
    # idx = 1
    # for chunk in chunks[1:]:
        # print("[", rank, "] chunk ", idx, "th shape: ", chunk.shape)
        # print("[", rank, "] ", idx, "th chunk: ", chunk)
        # idx += 1
        # if not torch.allclose(chunks[0], chunk):
            # print("[", rank, "] chunk ", idx, "th is not equal to the first chunk")
            # print("[", rank, "] chunks[0]: ", chunks[0])
            # print("[", rank, "] chunk: ", chunk)
            # logger.error(f"[{rank}] chunk {idx}th is not equal to the first chunk")
            # logger.error(f"[{rank}] chunks[0]: {chunks[0]}")
            # logger.error(f"[{rank}] chunk: {chunk}")
            # break

    # print(f"[{rank}] alltoall demo pass!")
