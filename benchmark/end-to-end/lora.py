import mirage as mi
import argparse
import os
import torch
import flashinfer

n_local_heads = 32
n_local_kv_heads = 8
head_dim = 128
intermediate_size = 14336
num_tokens = 1
num_kv_tokens = 4096
batch_size = 8

silu = torch.nn.SiLU()
def get_rms_linear():
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim), dtype=mi.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    return graph.superoptimize(config="mlp", previous_checkpoint="llama_rms_linear_bs{batch_size}.json")

def get_lora():
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, intermediate_size * 2), dtype=mi.float16)
    A = graph.new_input(dims=(4096, 16), dtype=mi.float16)
    B = graph.new_input(dims=(16, intermediate_size * 2), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(448,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tA = tb_graph.new_input(dtensor=A, input_map=(-1, -1, -1), forloop_dim=0)
    tB = tb_graph.new_input(dtensor=B, input_map=(1, -1, -1), forloop_dim=-1)
    # tD = tb_graph.matmul(tX, tA)
    # tC = tb_graph.concat(tX, tD, dim=1)
    # tE = tb_graph.concat(tW, tB, dim=0)
    # tO = tb_graph.matmul(tC, tE)
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tC = tb_graph.matmul(tX, tW)
    tD = tb_graph.matmul(tX, tA)
    tE = tb_graph.matmul(tD, tB)
    tM = tb_graph.add(tC, tE)
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W, A, B], tb_graph)
    graph.mark_output(O[0])
    return graph

def get_lora2():
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(batch_size * num_tokens, 14336), dtype=mi.float16)
    W = graph.new_input(dims=(14336, 4096), dtype=mi.float16)
    A = graph.new_input(dims=(14336, 16), dtype=mi.float16)
    B = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(128,1,1), block_dim=(128,1,1), forloop_range=224, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tA = tb_graph.new_input(dtensor=A, input_map=(-1, -1, -1), forloop_dim=0)
    tB = tb_graph.new_input(dtensor=B, input_map=(1, -1, -1), forloop_dim=-1)
    # tD = tb_graph.matmul(tX, tA)
    # tC = tb_graph.concat(tX, tD, dim=1)
    # tE = tb_graph.concat(tW, tB, dim=0)
    # tO = tb_graph.matmul(tC, tE)
    tC = tb_graph.matmul(tX, tW)
    tD = tb_graph.matmul(tX, tA)
    tE = tb_graph.matmul(tD, tB)
    tO = tb_graph.add(tC, tE)
    tAccumO = tb_graph.forloop_accum(tO)
    tb_graph.new_output(stensor=tAccumO, output_map=(1, -1, -1))
    O = graph.customized([X, W, A, B], tb_graph)
    graph.mark_output(O[0])
    return graph

def mirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2, kernels):
    func = kernels[0]
    outputs = func(inputs=[X, Wqkv])
    Xqkv = outputs[0]
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    Xq = Xq.view(Xq.shape[0], n_local_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    #Xq = rms_norm2(Xq)
    #Xk = rms_norm2(Xk)
    output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    output = torch.matmul(output.reshape(output_shape), Wo)
    # RMSNorm + LoRA1
    X = output
    func = kernels[2]
    outputs = func(inputs=[X, W13, A1, B1])
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    # LoRA2
    func = kernels[3]
    outputs = func(inputs=[X1, W2, A2, B2])
    output = outputs[0]
    return output


if __name__ == "__main__":
    X = torch.randn(batch_size * num_tokens, 4096, dtype=torch.float16, device='cuda:0')
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=torch.float16, device='cuda:0')
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device='cuda:0')
    W13 = torch.randn(4096, intermediate_size * 2, dtype=torch.float16, device='cuda:0')
    W2 = torch.rand(14336, 4096, dtype=torch.float16, device='cuda:0')
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    A1 = torch.rand(4096, 16, dtype=torch.float16, device='cuda:0')
    B1 = torch.rand(16, 14336 * 2, dtype=torch.float16, device='cuda:0')
    A2 = torch.rand(14336, 16, dtype=torch.float16, device='cuda:0')
    B2 = torch.rand(16, 4096, dtype=torch.float16, device='cuda:0')

    k1 = get_rms_linear()
    k2 = None
    k3 = get_lora()
    k4 = get_lora2()
    kernels = [k1, k2, k3, k4]

    for _ in range(16):
        mirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2, kernels)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        mirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, A1, B1, A2, B2, kernels)

    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    print(mean_syn)
