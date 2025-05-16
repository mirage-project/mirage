import mirage as mi
import numpy as np
import torch
import os
import argparse
import sys
import logging
from utils import analyze_differences
# set config to print full tensor
torch.set_printoptions(profile="full")



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
    # print(get_config())
    print("Imported mpi4py")
    # flush
    sys.stdout.flush()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npes = comm.Get_size()
    print("Current rank: ", rank, "/ Current npes: ", npes)

    # 如果日志文件已存在，先清空内容
    log_filename = f"tb_allgather_{rank}.log"
    if os.path.exists(log_filename):
        open(log_filename, 'w').close()  # 清空文件
    logging.basicConfig(level=logging.INFO, filename=log_filename)
    logger = logging.getLogger(__name__)
    logger_groundtruth = logging.getLogger("groundtruth")

    torch.cuda.set_device(rank)
    print("Current device: ", torch.cuda.current_device())
    matrix0 = torch.zeros(256, 128, dtype=torch.float16, device=f'cuda:{rank}')
    # matrix0[0:32, :] = 1
    # matrix0[32:64, :] = 2
    # matrix0[64:96, :] = 3
    # matrix0[96:128, :] = 4
    # matrix0[128:160, :] = 5
    # matrix0[160:192, :] = 6
    # matrix0[192:224, :] = 7
    # matrix0[224:256, :] = 8
    for row in range(256):
        # matrix0[row, :] = row
        matrix0[row, :] = row // 64 + 1
    # for col in range(128):
    #     matrix0[:, col] = col

    logger.info(f"matrix0_{rank} shape: {matrix0.shape}")
    logger.info(f"matrix0_{rank}: {matrix0}")

    # matrix = torch.zeros(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    # matrix[:, 0:32] = 1
    # matrix[:, 32:64] = 2
    # matrix[:, 64:96] = 3
    # matrix[:, 96:128] = 4
    # matrix[:, 128:160] = 5
    # matrix[:, 160:192] = 6
    # matrix[:, 192:224] = 7
    # matrix[:, 224:256] = 8

    # tensor1 = matrix0
    # tensor2 = torch.ones(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    tensor1 = torch.randn(256, 128, dtype=torch.float16, device=f'cuda:{rank}')
    tensor2 = torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    # tensor2 = torch.ones(128, 256, dtype=torch.float16, device=f'cuda:{rank}')
    tensor1_copy = tensor1.clone()
    tensor2_copy = tensor2.clone()
    input_tensors = [
        # torch.ones(256, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        #matrix,
        # torch.ones(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
        tensor1,
        tensor2,
        # torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
    ]

    outputs = graph(inputs=input_tensors, rank=rank, save_codes=save_codes)
    # np.savetxt(f"outputs_{rank}.txt", outputs[0].cpu().numpy().astype(np.int16), fmt="%d")
    logger.info(f"outputs_{rank} shape: {outputs[0].shape}")
    logger.info(f"outputs_{rank}: {outputs[0]}")


    # divide input_tensors[1] into 4 chunks along dim 1
    chunks = input_tensors[1].chunk(4, dim=1)
    # print("output.shape: ", outputs[0].shape)
    # print("output: ", outputs[0])
    # print("input_tensors[0].shape: ", input_tensors[0].shape)
    # print("chunks[", rank, "].shape: ", chunks[rank].shape)
    # print("input_tensors[0] * chunks[", rank, "]: ", input_tensors[0] @ chunks[rank])
    

    # chunks = outputs[0].chunk(4)
    # for chunk in chunks[1:]:
    #     assert torch.allclose(chunks[0], chunk)

    # print(f"[{rank}] alltoall demo pass!")

    import torch.distributed as dist
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=4,
        rank=rank
    )

    # x_pt = input_tensors[0].chunk(4, dim=0)[rank]
    # w_pt = input_tensors[1].chunk(4, dim=1)[rank]
    x_pt = tensor1_copy.chunk(4, dim=0)[rank]
    w_pt = tensor2_copy.chunk(4, dim=1)[rank]
    chunk_list = [torch.empty_like(x_pt) for _ in range(4)]
    dist.all_gather(chunk_list, x_pt)
    x = torch.cat(chunk_list, dim=0)
    # cor_final_result = input_tensors[0] @ w_pt
    # logger.info(f"cor_final_result_{rank} shape: {cor_final_result.shape}")
    # logger.info(f"cor_final_result_{rank}: {cor_final_result}")
    final_result = x @ w_pt

    # logger.info(f"final_result_{rank} shape: {final_result.shape}")
    logger_groundtruth.info(f"final_result_{rank} shape: {final_result.shape}")
    logger_groundtruth.info(f"final_result_{rank}: {final_result}")

    final_chunks = final_result.chunk(4, dim=0)
    # logger.info(f"final_result_{rank} shape: {final_result.shape}")
    # logger.info(f"final_result_{rank}: {final_result}")
    # avg_chunk = torch.zeros_like(final_chunks[0])
    # for chunk in final_chunks:
    #     avg_chunk += chunk
    # avg_chunk /= 4
    # logger.info(f"avg_chunk_{rank} shape: {avg_chunk.shape}")
    # logger.info(f"avg_chunk_{rank}: {avg_chunk}")
    output_chunks = outputs[0].chunk(4, dim=0)
    if not torch.allclose(outputs[0], final_result, rtol=5e-2, atol=1e-2):
        analyze_differences(outputs[0], final_result, logger)
    # assert torch.allclose(output_chunks[0], avg_chunk, rtol=5e-2, atol=1e-2)
    # assert torch.allclose(output_chunks[1], avg_chunk, rtol=5e-2, atol=1e-2)
    # assert torch.allclose(output_chunks[2], avg_chunk, rtol=5e-2, atol=1e-2)
    # assert torch.allclose(output_chunks[3], avg_chunk, rtol=5e-2, atol=1e-2)
    else:
        print(f"[{rank}] allgather demo pass!")
    dist.destroy_process_group()