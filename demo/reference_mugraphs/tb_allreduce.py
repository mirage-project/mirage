import mirage as mi
import numpy as np
import torch
import os
import argparse
import logging
from utils import analyze_differences

seed = 42  # Use a fixed seed
torch.manual_seed(seed)
#torch.set_printoptions(profile="full")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--save_codes", action="store_true", help="Save the generated codes")
    args = parser.parse_args()
    save_codes = args.save_codes

    graph = mi.new_kernel_graph(gpu_dim=(4, 1, 1))
    X = graph.new_input(dims=(512, 128), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
    W1 = graph.new_input(dims=(128, 256), gpu_input_map=(0, -1 ,-1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(4,8,1), block_dim=(128,1,1), forloop_range=4, reduction_dimx=4)
    tX1 = tb_graph.new_input(dtensor=X, input_map=(-1, 0, -1), forloop_dim=1)
    tW1 = tb_graph.new_input(dtensor=W1, input_map=(1, -1, -1), forloop_dim=0)
    tM1 = tb_graph.matmul(tX1, tW1)
    tAccM1 = tb_graph.forloop_accum(tM1)

    #TB_REDUCESCATTER_EPILOGUE
    tb_graph.new_output(stensor=tAccM1, output_map=(1, 0, -1), epilogue="reduce_scatter") # (0, 1, -1) right?
    O1 = graph.customized([X, W1], tb_graph)

    # graph.mark_output(O1[0])

    W2 = graph.new_input(dims=(256, 512), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(2,8,1), block_dim=(128,1,1), forloop_range=4, reduction_dimx=4)
    #TB_ALLGATHER_PROLOGUE
    tX2 = tb_graph.new_input(dtensor=O1[0], input_map=(-1, 0, -1), forloop_dim=1, prologue="allgather")
    tW2 = tb_graph.new_input(dtensor=W2, input_map=(1, -1, -1), forloop_dim=0)
    tM2 = tb_graph.matmul(tX2, tW2)
    tAccM2 = tb_graph.forloop_accum(tM2)

    tb_graph.new_output(stensor=tAccM2, output_map=(1, 0, -1)) # (0, 1, -1) right?
    O2 = graph.customized([O1[0], W2], tb_graph)
    graph.mark_output(O2[0])

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if os.path.exists(f"tb_allreduce_{rank}.log"):
        os.remove(f"tb_allreduce_{rank}.log")
    # logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(f"tb_allreduce_{rank}.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.info(f"[{rank}] start")

    torch.cuda.set_device(rank)
    print("Current rank: ", rank)
    print("Current device: ", torch.cuda.current_device())
    input_tensors = [
        torch.randn(512, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
        torch.randn(256, 512, dtype=torch.float16, device=f'cuda:{rank}'),
    ]

    outputs = graph(inputs=input_tensors, rank=rank, save_codes=save_codes)
    print(f"[{rank}] outputs.size(): {len(outputs)}")
    logger.info(f"[{rank}] outputs.size(): {len(outputs)}")
    if len(outputs) == 2:
        for output in outputs:
            logger.info(f"[{rank}] output shape: {output.shape}")
            print(f"[{rank}] output shape: {output.shape}")
        logger.info(f"[{rank}] output: {output}")
        mid_output = outputs[0]
        allgather_output = outputs[1]
    else:
        allgather_output = outputs[0]

    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=4,
        rank=rank
    )

    x1_pt = input_tensors[0].chunk(4, dim=1)[rank]
    w1_pt = input_tensors[1].chunk(4, dim=0)[rank]
    result1 = x1_pt @ w1_pt
    mid_result = torch.empty((128, 256), dtype=result1.dtype, device=f'cuda:{rank}')
    dist.reduce_scatter_tensor(mid_result, result1)
    if len(outputs) == 2:
        if not torch.allclose(mid_result, mid_output, rtol=5e-2, atol=1e-1):
            logger.info(f"[{rank}] reduce_scatter demo failed!")
            analyze_differences(mid_result, mid_output, logger)
        else:
            logger.info(f"[{rank}] reduce_scatter demo pass!")

    w2_pt = input_tensors[2].chunk(4, dim=1)[rank]
    result_list = [torch.empty_like(mid_result) for _ in range(4)]
    dist.all_gather(result_list, mid_result)
    x2 = torch.cat(result_list, dim=0)
    result2 = x2 @ w2_pt
    # print(result2)
    # print(outputs[0])
    if not torch.allclose(allgather_output, result2, rtol=5e-2, atol=1e-1):
        logger.info(f"[{rank}] allreduce demo failed!")
        analyze_differences(allgather_output, result2, logger)
    else:
        print(f"[{rank}] allreduce demo pass!")
    dist.destroy_process_group()