import mirage as mi
import numpy as np
import torch
import os
import argparse


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
    W = graph.new_input(dims=(128, 256), gpu_input_map=(0, -1 ,-1), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(4,8,1), block_dim=(128,1,1), forloop_range=4, reduction_dimx=4)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, 0, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccM = tb_graph.forloop_accum(tM)

    #TB_ALLTOALL_EPILOGUE
    tb_graph.new_output(stensor=tAccM, output_map=(1, 0, -1), epilogue="reduce_scatter") # (0, 1, -1) right?
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    torch.cuda.set_device(rank)
    print("Current rank: ", rank)
    print("Current device: ", torch.cuda.current_device())
    input_tensors = [
        torch.randn(512, 128, dtype=torch.float16, device=f'cuda:{rank}'),
        torch.randn(128, 256, dtype=torch.float16, device=f'cuda:{rank}'),
    ]

    outputs = graph(inputs=input_tensors, rank=rank, save_codes=save_codes)

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
    print(f'torch[{rank}]: {result[1][0]}')
    final_result = torch.empty_like(outputs[0])
    dist.reduce_scatter_tensor(final_result, result)
    print(outputs[0])
    print(final_result)
    assert torch.allclose(outputs[0], final_result, rtol=5e-2, atol=1e-1)
    print(f"[{rank}] reduce scatter demo pass!")
    dist.destroy_process_group()