import mirage as mi
import argparse
import os
import torch


n_local_heads = 32
n_local_kv_heads = 32
head_dim = 128
intermediate_size = 11008
num_tokens = 4
num_kv_tokens = 4096

silu = torch.nn.SiLU()
def get_rms_linear():
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(num_tokens, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(384,1,1), block_dim=(128,1,1), forloop_range=32, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W], tb_graph)
    return graph, O

def get_rms_linear2():
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(num_tokens, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, intermediate_size * 2), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(344,1,1), block_dim=(128,1,1), forloop_range=32, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W], tb_graph)
    return graph, O

def get_chameleon_attention():
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(n_local_kv_heads, num_tokens, 128), dtype=mi.float16)
    K = graph.new_input(dims=(n_local_kv_heads, 128, num_kv_tokens), dtype=mi.float16)
    V = graph.new_input(dims=(n_local_kv_heads, num_kv_tokens, 128), dtype=mi.float16)
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(n_local_kv_heads,16,1), block_dim=(128,1,1), forloop_range=4, reduction_dimx=128)
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=1)
    bQ = tbgraph1.rms_norm(bQ)
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    O = graph.customized([Q, K, V], tbgraph1)

    tbgraph2 = mi.new_threadblock_graph(grid_dim=(n_local_kv_heads,2,1), block_dim=(128,1,1), forloop_range=1, reduction_dimx=128)
    bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized(O, tbgraph2)
    return graph, O

def mirage_chameleon(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels):
    func, outputs = kernels[0]
    outputs = func(inputs=[X, Wqkv], outputs=outputs)
    Xqkv = outputs[0]
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    Xq = Xq.view(Xq.shape[0], n_local_kv_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    func, outputs = kernels[2]
    outputs = func(inputs=[Xq, Kcache, Vcache], outputs=outputs)
    output = outputs[0]
    #Xq = rms_norm2(Xq)
    #Xk = rms_norm2(Xk)
    #output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    output = torch.matmul(output.reshape(output_shape), Wo)
    # RMSNorm
    X = output
    func, outputs = kernels[1]
    outputs = func(inputs=[X, W13], outputs=outputs)
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    output = torch.matmul(X1, W2)
    return output


if __name__ == "__main__":
    X = torch.randn(num_tokens, 4096, dtype=torch.float16, device='cuda:0')
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=torch.float16, device='cuda:0')
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device='cuda:0')
    W13 = torch.randn(4096, intermediate_size * 2, dtype=torch.float16, device='cuda:0')
    W2 = torch.rand(intermediate_size, 4096, dtype=torch.float16, device='cuda:0')
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')

    k1 = get_rms_linear()
    k2 = get_rms_linear2()
    k3 = get_chameleon_attention()
    kernels = [k1, k2, k3]

    for _ in range(16):
        mirage_chameleon(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        mirage_chameleon(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)

    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    print(mean_syn)
