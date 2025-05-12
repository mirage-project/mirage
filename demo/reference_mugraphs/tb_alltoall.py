import mirage as mi
import numpy as np
import torch
import os
import argparse
import logging

seed = 42  # Use a fixed seed
torch.manual_seed(seed)
#torch.set_printoptions(profile="full")

import torch
import numpy as np

from utils import analyze_differences


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("--save_codes", action="store_true", help="Save the generated codes")
    args = parser.parse_args()
    save_codes = args.save_codes

    m = 256
    n = 256
    k = 128
    gpu_num = 4
    per_gpu_m = m // gpu_num
    per_gpu_k = k // gpu_num

    graph = mi.new_kernel_graph(gpu_dim=(gpu_num, 1, 1))
    X = graph.new_input(dims=(m, k), gpu_input_map=(1, -1 ,-1), dtype=mi.float16)
    W = graph.new_input(dims=(k, n), gpu_input_map=(0, -1 ,-1), dtype=mi.float16)
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
    # if tb_alltoall_{rank}.log exists, delete it
    if os.path.exists(f"tb_alltoall_{rank}.log"):
        os.remove(f"tb_alltoall_{rank}.log")
    # create a new file logger
    logging.basicConfig(level=logging.INFO, filename=f"tb_alltoall_{rank}.log")
    logger = logging.getLogger(__name__)

    torch.cuda.set_device(rank)
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # torch.set_float32_matmul_precision('highest')
    # print("torch.backends.cuda.matmul.allow_tf32 ", torch.backends.cuda.matmul.allow_tf32)
    # print("torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction ", torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    # print("torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction ", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)

    # print("Current rank: ", rank)
    # print("Current device: ", torch.cuda.current_device())
    # matrix0 = torch.zeros(256, 128, dtype=torch.float16, device=f'cuda:{rank}')
    matrix0 = torch.randn(256, 128, dtype=torch.float16, device=f'cuda:{rank}')
    matrix_0_copy = matrix0.clone()
    # matrix = torch.zeros(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    # matrix[0:32, :] = 5
    # matrix[32:64, :] = 2
    # matrix[64:96, :] = 3
    # matrix[96:128, :] = 4
    matrix = torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    matrix_copy = matrix.clone()

    input_tensors = [
        # torch.randn(64, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        # torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
        torch.randn(m, k, dtype=torch.float16, device=f'cuda:{rank}'),
        torch.randn(k, n, dtype=torch.float16, device=f'cuda:{rank}'),
        #matrix0,
        # torch.ones(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
        #matrix,
    ]

    outputs = graph(inputs=input_tensors, rank=rank, save_codes=save_codes)

    '''
    def correct_mm_alltoall(input1, input2, rank):
        chunks1 = input1.chunk(4, dim=1)
        chunks2 = input2.chunk(4, dim=0)
        output = torch.zeros(m, n, dtype=torch.float16, device=input1.device)
        for i in range(4):
            output[i*per_gpu_m:(i+1)*per_gpu_m, :] = torch.matmul(chunks1[i], chunks2[i])[rank*per_gpu_m:(rank+1)*per_gpu_m, :]
        return output

    print(outputs[0])
    answer = correct_mm_alltoall(input1=matrix_0_copy, input2=matrix_copy, rank=rank)
    print(answer)
    if not torch.allclose(outputs[0], answer):
        # find all not close elements
        # not_close_elements = torch.nonzero(outputs[0] != answer)
        # logger.info(f"[{rank}] result is not equal to the answer")
        # logger.info(f"[{rank}] result: {outputs[0]}")
        # logger.info(f"[{rank}] answer: {answer}")
        # logger.info(f"[{rank}] not close elements: {not_close_elements}")
        analyze_differences(outputs[0], answer, logger)
        print(f"[{rank}] alltoall demo failed!")
    else:
        print(f"[{rank}] alltoall demo pass!")
    '''

    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=4,
        rank=rank
    )

    x_pt = input_tensors[0].chunk(4, dim=1)[rank]
    w_pt = input_tensors[1].chunk(4, dim=0)[rank]
    result = x_pt @ w_pt
    chunks_pt = [chunk.contiguous() for chunk in torch.chunk(result, 4, dim=0)]
    output_list = [torch.empty_like(chunks_pt[0]) for _ in range(4)]
    dist.all_to_all(output_list, chunks_pt)
    final_result = torch.cat(output_list, dim=0)
    assert torch.allclose(outputs[0], final_result, rtol=5e-2, atol=1e-2)
    print(f"[{rank}] alltoall demo pass!")
    dist.destroy_process_group()
